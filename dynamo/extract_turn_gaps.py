#!/usr/bin/env python3
"""Extract per-turn tool-execution gaps from uni-agent task.log files and
merge them into a traj-mode replay JSONL.

Why: frozen_replay's fixed --think-time is an approximation of the real
arrival process. Real tool calls are heavy-tailed (shell seconds, test
suites tens of seconds); the gap structure decides (a) whether a
trajectory's KV blocks survive in GPU cache until its next turn, (b) the
prefill:decode mixture other trajectories see, (c) how synchronized the
trajectory waves are. Replaying the *recorded* gaps reproduces the real
rollout arrival process while keeping teacher-forced determinism.

Gap definition per turn i: wall time from the model's ACTION being logged
(tool dispatch) to the next STEP banner (next model call) in the same
session's task.log — i.e. the tool-execution + framework overhead the
next request waits behind in live rollout.

Alignment: walks task.log files in the same sorted-glob order as
dump_trajectories.py walks trajectory.npz (session dirs are shared), and
aligns gap[i] with the transition turns[i] -> turns[i+1]. On count
mismatch the median gap pads / the list truncates.

Usage:
  python3 extract_turn_gaps.py \
      --log-glob '/workspace/logs/.../step_*/session-*/task.log' \
      --jsonl /workspace/phase2/kg8_traj.jsonl \
      --out   /workspace/phase2/kg8_traj_gaps.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STEP_RE = re.compile(r"=+ STEP (\d+) =+")
ACTION_RE = re.compile(r"ACTION \(")


def parse_gaps(path: str) -> list[float]:
    """Per-turn gaps: last ACTION timestamp of step i -> STEP i+1 banner."""
    steps: list[datetime] = []   # STEP banner times
    actions: list[datetime] = [] # last ACTION time seen since previous banner
    last_action: datetime | None = None
    with open(path, errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if STEP_RE.search(line):
                steps.append(ts)
                actions.append(last_action)  # action belonging to prior step
                last_action = None
            elif ACTION_RE.search(line):
                last_action = ts
    gaps = []
    # actions[k] is the last ACTION before STEP banner k; the gap feeding
    # STEP k is banner_k - action_{k} (action logged during step k-1).
    for k in range(1, len(steps)):
        a = actions[k]
        if a is not None and steps[k] >= a:
            gaps.append((steps[k] - a).total_seconds())
        else:
            gaps.append(-1.0)  # marker: fill with median later
    med = statistics.median([g for g in gaps if g >= 0]) if any(g >= 0 for g in gaps) else 2.0
    return [g if g >= 0 else med for g in gaps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-glob", required=True, help="glob for session task.log files (same session order as the npz glob used for the JSONL)")
    ap.add_argument("--jsonl", required=True, help="existing traj-mode JSONL to annotate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-scale", type=float, default=1.0,
                    help="scale factor on recorded gaps (1.0 = real time)")
    args = ap.parse_args()

    logs = sorted(glob.glob(args.log_glob))
    trajs = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    # session dir order == npz glob order; map by shared parent dir name
    by_dir = {os.path.basename(os.path.dirname(p)): p for p in logs}
    log_list = [by_dir[k] for k in sorted(by_dir)]
    if len(log_list) < len(trajs):
        raise SystemExit(f"fewer task.logs ({len(log_list)}) than trajectories ({len(trajs)})")

    all_gaps: list[float] = []
    matched = 0
    for i, t in enumerate(trajs):
        gaps = [g * args.gap_scale for g in parse_gaps(log_list[i])]
        need = max(len(t["turns"]) - 1, 0)
        med = statistics.median(gaps) if gaps else 2.0
        if len(gaps) < need:
            gaps = gaps + [med] * (need - len(gaps))
        t["gaps"] = gaps[:need]
        all_gaps.extend(t["gaps"])
        matched += 1

    with open(args.out, "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    if all_gaps:
        s = sorted(all_gaps)
        n = len(s)
        print(f"[gaps] {matched} trajectories, {n} gaps: "
              f"p50={s[n//2]:.1f}s p90={s[int(n*0.9)]:.1f}s p99={s[int(n*0.99)]:.1f}s "
              f"max={s[-1]:.1f}s mean={sum(s)/n:.1f}s (scale={args.gap_scale})")
    print(f"[gaps] wrote {args.out}")


if __name__ == "__main__":
    main()
