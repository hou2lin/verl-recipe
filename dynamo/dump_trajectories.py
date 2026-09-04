#!/usr/bin/env python3
"""Build a fixed frozen-replay request set from an RL run's trajectory.npz dumps.

Uni-agent writes one trajectory.npz per session (keys: trajN_prompt_ids,
trajN_response_ids, ...). For the frozen-replay KV-reuse measurement we want
the *prompt-side* token contexts that recur across turns — the prefixes
FlexKV can cache and re-serve.

Two modes:
  --mode prompt   : one request per trajectory = its full prompt_ids. Simple;
                    reuse comes from replaying the same set across rounds.
  --mode turns    : teacher-forced expansion — for each trajectory emit a
                    growing sequence of requests prompt_ids, prompt_ids+chunk1,
                    prompt_ids+chunk1+chunk2, ... splitting the recorded
                    response into `--turn-chunk` pieces. This mirrors the
                    slide's "prompt frozen per turn, response tokens
                    discarded": each request's context is the frozen prefix of
                    a real multi-turn trajectory, so within a round longer
                    turns reuse shorter turns' prefixes, and across rounds
                    everything reuses. Yields the slide's ~14x request
                    multiplier (64 traj -> ~900 requests).

Output: JSONL, one {"prompt_token_ids": [...]} per line, capped at --limit.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for trajectory.npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["prompt", "turns", "traj"], default="turns")
    ap.add_argument("--turn-chunk", type=int, default=512,
                    help="response tokens per synthetic turn (turns mode)")
    ap.add_argument("--limit", type=int, default=0, help="max requests (0=all)")
    ap.add_argument("--max-context", type=int, default=40000,
                    help="drop requests longer than this many tokens")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob}")

    requests: list[list[int]] = []
    n_traj = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        prompt_keys = [k for k in d.keys() if k.endswith("_prompt_ids")]
        for pk in prompt_keys:
            prefix = pk[: -len("prompt_ids")]
            prompt = d[pk].astype(int).tolist()
            n_traj += 1
            if args.mode == "prompt":
                if len(prompt) <= args.max_context:
                    requests.append(prompt)
                continue
            if args.mode == "traj":
                # one JSONL row per trajectory: ordered teacher-forced turn
                # contexts (turn k+1 strictly extends turn k) — for the paced
                # replay that mirrors the real agent turn sequence.
                resp_key = prefix + "response_ids"
                response = d[resp_key].astype(int).tolist() if resp_key in d else []
                turns = []
                ctx = list(prompt)
                if len(ctx) <= args.max_context:
                    turns.append(list(ctx))
                for i in range(0, len(response), args.turn_chunk):
                    ctx = ctx + response[i : i + args.turn_chunk]
                    if len(ctx) > args.max_context:
                        break
                    turns.append(list(ctx))
                if turns:
                    requests.append({"turns": turns})
                continue
            # turns mode: frozen teacher-forced prefixes
            resp_key = prefix + "response_ids"
            response = d[resp_key].astype(int).tolist() if resp_key in d else []
            ctx = list(prompt)
            # turn 0 = the bare prompt
            if len(ctx) <= args.max_context:
                requests.append(list(ctx))
            for i in range(0, len(response), args.turn_chunk):
                ctx = ctx + response[i : i + args.turn_chunk]
                if len(ctx) > args.max_context:
                    break
                requests.append(list(ctx))

    if args.limit and len(requests) > args.limit:
        # deterministic stride sample to keep the trajectory diversity
        stride = len(requests) / args.limit
        requests = [requests[int(i * stride)] for i in range(args.limit)]

    with open(args.out, "w") as f:
        for r in requests:
            if isinstance(r, dict):
                f.write(json.dumps(r) + "\n")
            else:
                f.write(json.dumps({"prompt_token_ids": r}) + "\n")

    lens = [sum(len(t) for t in r["turns"]) if isinstance(r, dict) else len(r) for r in requests]
    print(f"[dump] {n_traj} trajectories -> {len(requests)} requests "
          f"(mode={args.mode}); ctx len min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)//len(lens)}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
