#!/usr/bin/env python3
"""Frozen-replay harness for the FlexKV steady-state KV-reuse measurement.

Reproduces the old-stack slide methodology (Qwen3-32B, C64, 32G GPU KV/rank,
Uni-Agent SWE): a *fixed* set of trajectories is replayed against the serving
engine at steady state, so the same multi-turn prefixes recur and FlexKV's
CPU tier can be hit (H2D GET) instead of recomputed.

Why a separate harness (not the RL trainer): live single-pass RL rollout has
no KV reuse — each prompt is generated once, the GPU cache never spills, so
GET=0 (measured: arm1 put=297 / get=0). The slide's numbers come from frozen
*replay*, where teacher-forcing (prompt frozen per turn, response tokens
discarded) keeps each trajectory's prefix stable across rounds.

Design (mirrors the slide's "64 fixed trajectories x 8 rollouts = 902
requests, forced shard routing"):
  1. Load a fixed trajectory set (JSONL: one object per turn with the exact
     prompt token context the RL run sent). Built by dump_trajectories.py
     from an RL run's rollout logs.
  2. Warm round: replay every request once so its prefix lands in the GPU
     cache and (on eviction) the FlexKV CPU tier.
  3. Steady-state rounds: replay the same set R times at concurrency C. With
     teacher-forcing the prefixes are identical, so from round 2 on every
     GPU-miss should be a FlexKV CPU hit (H2D GET) rather than a recompute.
  4. Emit per-round wall-clock + throughput; the path breakdown
     (GPU-hit/CPU-fetch/recompute) is read from the engine's FlexKV stats
     (record_get) and vLLM prefix-cache counters after the run.

The two arms (FLEXKV on/off) are the serving side; this harness is identical
across arms — it only drives load. max_tokens=1 per request: we measure
prefill KV reuse (prompt path), not decode, exactly like the slide's
teacher-forced "response tokens discarded".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import aiohttp


async def _one_request(session, url, model, prompt_token_ids, sem, stats, max_tokens=1):
    async with sem:
        t0 = time.monotonic()
        payload = {
            "model": model,
            # Replay the exact token context so prefixes match byte-for-byte
            # across rounds (chat re-templating would perturb them).
            "prompt": prompt_token_ids,
            # Teacher-forced generation: generate the recorded response length
            # (real decode load, real in-flight KV pressure), DISCARD the
            # output; the next turn uses the frozen recorded context. This is
            # the slide's "response tokens discarded" semantics — max_tokens=1
            # (prefill-only) understates rollout time AND removes the decode
            # KV-block pressure that gates fetch landing.
            "max_tokens": max_tokens,
            "ignore_eos": max_tokens > 1,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            async with session.post(url, json=payload) as resp:
                await resp.json()
                stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001 - harness must survive one bad req
            stats["err"] += 1
            stats.setdefault("errors", []).append(str(exc)[:120])
        stats["latency_s"] += time.monotonic() - t0


async def _one_trajectory(session, url, model, turns, think_time, sem, stats):
    """Real agent pattern: turns strictly sequential within a trajectory,
    think-time between them (tool execution gap). Concurrency = trajectories
    in flight, so between one trajectory's turns dozens of others interleave
    — reproducing the turn-gap eviction/refetch dynamics of live rollout."""
    async with sem:
        nturns = len(turns)
        for i, ctx in enumerate(turns):
            # generation length for this turn = the recorded response chunk
            # (next turn context minus this one); last turn generates the
            # trajectory-typical chunk so decode load stays realistic.
            if i < nturns - 1:
                gen_len = max(1, len(turns[i + 1]) - len(ctx))
            elif nturns > 1:
                gen_len = max(1, len(turns[-1]) - len(turns[-2]))
            else:
                gen_len = 512
            await _one_request(session, url, model, ctx, asyncio.Semaphore(1), stats,
                               max_tokens=gen_len)
            if think_time > 0 and i < nturns - 1:
                await asyncio.sleep(think_time)


async def _paced_round(session, url, model, trajectories, concurrency, think_time, stats):
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_one_trajectory(session, url, model, turns, think_time, sem, stats))
        for turns in trajectories
    ]
    await asyncio.gather(*tasks)


async def _round(session, url, model, requests, concurrency, stats):
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_one_request(session, url, model, r, sem, stats))
        for r in requests
    ]
    await asyncio.gather(*tasks)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="http://host:port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trajectories", required=True, help="JSONL of {prompt_token_ids: [...]}")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--warm-rounds", type=int, default=1)
    ap.add_argument("--steady-rounds", type=int, default=3,
                    help="flat mode: fixed steady rounds; paced mode: MAX passes (stops early on convergence)")
    ap.add_argument("--paced", action="store_true",
                    help="trajectory-paced replay: per-trajectory sequential turns with think-time, C = trajectories in flight (real agent pattern)")
    ap.add_argument("--think-time", type=float, default=2.0,
                    help="paced mode: seconds between a trajectory's turns (tool-execution gap)")
    ap.add_argument("--converge-pct", type=float, default=5.0,
                    help="paced mode: stop when round wall improves < this %% vs previous")
    ap.add_argument("--out", default="/workspace/phase2/replay_result.json")
    args = ap.parse_args()

    url = f"{args.endpoint.rstrip('/')}/v1/completions"
    requests: list[list[int]] = []
    trajectories: list[list[list[int]]] = []
    with open(args.trajectories) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "turns" in obj:
                trajectories.append(obj["turns"])
                for t in obj["turns"]:
                    requests.append(t)
            else:
                ids = obj.get("prompt_token_ids") or obj.get("prompt")
                if ids:
                    requests.append(ids)
                    trajectories.append([ids])
    if not requests:
        raise SystemExit(f"no trajectories loaded from {args.trajectories}")

    result = {
        "endpoint": args.endpoint,
        "model": args.model,
        "num_requests_per_round": len(requests),
        "concurrency": args.concurrency,
        "rounds": [],
    }
    result["paced"] = bool(args.paced)
    result["num_trajectories"] = len(trajectories)
    result["think_time_s"] = args.think_time if args.paced else None
    timeout = aiohttp.ClientTimeout(total=None)
    prev_wall = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for phase, nrounds in (("warm", args.warm_rounds), ("steady", args.steady_rounds)):
            for r in range(nrounds):
                stats = {"ok": 0, "err": 0, "latency_s": 0.0}
                t0 = time.monotonic()
                if args.paced:
                    await _paced_round(session, url, args.model, trajectories,
                                       args.concurrency, args.think_time, stats)
                else:
                    await _round(session, url, args.model, requests, args.concurrency, stats)
                wall = time.monotonic() - t0
                row = {
                    "phase": phase,
                    "round": r,
                    "wall_s": round(wall, 3),
                    "ok": stats["ok"],
                    "err": stats["err"],
                    "throughput_req_s": round(stats["ok"] / wall, 3) if wall else 0,
                    "mean_latency_s": round(stats["latency_s"] / max(stats["ok"] + stats["err"], 1), 4),
                }
                result["rounds"].append(row)
                print(f"[replay] {phase} round {r}: {row['ok']} ok / {row['err']} err "
                      f"wall={row['wall_s']}s tput={row['throughput_req_s']} req/s "
                      f"lat={row['mean_latency_s']}s", flush=True)
                if args.paced and phase == "steady" and prev_wall is not None:
                    if prev_wall > 0 and (prev_wall - wall) / prev_wall * 100 < args.converge_pct:
                        print(f"[replay] converged (<{args.converge_pct}% improvement); stopping", flush=True)
                        prev_wall = wall
                        break
                prev_wall = wall if phase == "steady" else None

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[replay] wrote {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
