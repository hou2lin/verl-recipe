# PR draft: FlexKV L2 CPU KV-cache support for the Dynamo rollout backend (recipe-only)

> **STATUS: local draft, NOT submitted. 🔴 Hard rule: no external PR/MR without explicit user review and approval.**
> Goal (user-settled 2026-09-07): make **verl + dynamo + vLLM + FlexKV L2 cache** work, as a **verl-recipe-only** change.
> Base: `verl-project/verl-recipe` `main` (rebase to tip before submitting; `ee3aef1` at time of writing)
> Head: `feat/dynamo-flexkv-l2` (to be cut from `feat/dynamo-dynres`, core + FlexKV files only)
> Supersedes `PR_DRAFT_fully_async.md`. The V1-trainer line stays on its own branch/draft (`feat/dynamo-v1-migration`).
>
> Validation stack (user-specified):
> - verl: `main` lineage (validated at `9c76436`, tree clean throughout)
> - verl-recipe: this PR, based on `main`
> - vLLM: https://github.com/vllm-project/vllm/pull/54484 (validated at its head `4582c0d`)
> - FlexKV: taco-project/FlexKV at/after PR #279 (validated at the #279 merge commit `016c290`, **pure upstream, no local patches**)

---

## Title

[dynamo] FlexKV L2 CPU KV-cache support for the Dynamo rollout backend (recipe-only)

## Body

### What does this PR do?

Adds a **recipe-only** integration that runs verl RL rollout on the full serving chain
**verl + Dynamo (KV-aware router) + vLLM + FlexKV L2 CPU cache**:

1. **Dynamo rollout backend** (carrier): rollout requests are served by
   `dynamo.frontend --router-mode kv` plus `dynamo.vllm` worker shards; weight
   sync (NCCL first hop, CUDA-IPC second hop) and the abort → sync → resume
   choreography reuse verl's stock `CheckpointEngineManager`.
2. **FlexKV L2 wiring** (the core of this PR): per-shard FlexKV server
   lifecycle wired to vLLM's `FlexKVConnectorV1` — KV blocks evicted from GPU
   spill to a CPU (DRAM) tier and are fetched back (H2D) on prefix re-hit
   instead of being recomputed; the CPU tier survives rollout rounds and
   weight updates (reset-after-sync choreography).

**verl itself is untouched.** Everything registers through verl's sanctioned
extension points:

- `VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register` (external-module hook)
- `actor_rollout_ref.rollout.name=dynamo` (rollout registry)
- `engine_kwargs.dynamo.*` backend knobs (including `enable_flexkv`)
- `DynamoReplica.get_ray_class_with_init_args` wiring a subclassed
  `CheckpointEngineWorker` (no monkey-patching)

Acceptance runs assert the verl tree stays `git status`-clean end to end.

### Why

Multi-turn agentic RL rollout has strong prefix recurrence: each trajectory's
context grows turn by turn, and trajectories are replayed across rollout
rounds. GPU KV cache capacity is finite; eviction means recompute. FlexKV
provides a CPU DRAM L2 tier that turns recompute into a PCIe fetch, and
Dynamo's KV-aware router routes same-prefix requests to the worker already
holding the blocks. Together they form the complete KV-reuse stack. This PR
lands that chain inside verl-recipe with zero changes to verl core.

### FlexKV integration surface (core of this PR)

Per-shard isolation (required for multi-shard nodes; `dynamo_async_server.py`):

| Mechanism | Purpose |
|---|---|
| `FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_server_g<cvd>` | one IPC socket per shard (suffix = shard GPU ids); prevents cross-shard registration crosstalk |
| `FLEXKV_INSTANCE_NUM` / `FLEXKV_INSTANCE_ID` | shard identity injection |
| `FLEXKV_PY_METRICS_PORT` / `FLEXKV_CPP_METRICS_PORT` | per-shard metric-port offsets |
| `FLEXKV_SHARED_CPU_CACHE=1` | optional single shared CPU pool across shards (server/client mode) |
| vLLM side | `--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'`; `reset_connector` forwarded on weight sync |

User-facing knobs (env, all defaulted): `DYNAMO_USE_FLEXKV`,
`FLEXKV_CPU_CACHE_GB`, `FLEXKV_CONFIG_PATH`, `FLEXKV_ENABLE_MPS`
(default 0 — a KVManager-started MPS daemon inherits the shard's
CUDA_VISIBLE_DEVICES mask and breaks CUDA visibility for later clients),
`FLEXKV_INIT_READY_TIMEOUT_S`, `FLEXKV_ENABLE_METRICS`,
`VERL_DYNAMO_FE_READY_TIMEOUT` (default 2400 s: first-load + compile of a
large model can exceed the old 600 s window).

### Files

| File | Role |
|---|---|
| `dynamo/register.py` | external-module entry point |
| `dynamo/dynamo_async_server.py` | `DynamoHttpServer` (etcd/nats/frontend/worker lifecycle, abort/resume choreography, watchdogs) + **FlexKV per-shard wiring** + `DynamoCheckpointEngineWorker` + `DynamoReplica` |
| `dynamo/_dynamo_vllm_with_control.py` | dynamo.vllm sidecar: collective_rpc control channel (update_weights_from_ipc / pause / resume / reset) |
| `dynamo/dynamo_worker_extension.py` | vLLM worker extension: CUDA-IPC weight receive |
| `dynamo/dynamo_agent_loop.py`, `dynamo_rollout.py` | agent-loop client / rollout adapter (incl. FlexKV service shutdown ordering) |
| `dynamo/dynamo_thunderagent.py` | ThunderAgent routing start/stop (backend capability, off by default) |
| `dynamo/main_dynamo_fully_async.py`, `dynamo/config/*` | thin entrypoint + hydra config (verl `fully_async_policy`; classic-stack, **no `trainer.use_v1`**) |
| `dynamo/smoke_dynamo_fully_async.sh` | 1+1-GPU smoke; `FLEXKV=1` turns the same script into the FlexKV acceptance run |
| `dynamo/README.md`, `dynamo/REQUIRED_VERL.txt` | install, version pins, knob table, operating guidance |
| `dynamo/tests/`, `dynamo/assets/` | unit tests + fixtures (final selection at review) |

**Excluded** (stay on the research branch or go to their own upstreams):
V1-trainer files (`feat/dynamo-v1-migration`), all replay/benchmark harnesses
(`frozen_replay.py`, `dump_trajectories.py`, `extract_turn_gaps.py`,
`run_kg8_*`, `serve_dynamo_kvrouter_*`, task configs, launchers), `patches/`
(FlexKV/dynamo upstream patches), local design docs.

### Dependencies (documented in README, not shipped here)

- verl `main` (validated at `9c76436`; tree stays clean — see REQUIRED_VERL.txt)
- vLLM from https://github.com/vllm-project/vllm/pull/54484 (validated at `4582c0d`) — provides `FlexKVConnectorV1`, `reset_connector`, `--kv-cache-memory-bytes`
- FlexKV: taco-project/FlexKV main at/after PR #279 (`016c290`) — validated **unpatched**
- ai-dynamo ≥ 1.4.2 (kv-events over ZMQ, overlap-blocks routing)

### Acceptance evidence (validation stack above)

- **smokeA** — Dynamo backend, 1+1 GPU (trainer / standalone rollout), 2 RL
  steps with per-step weight sync choreography, verl tree clean:
  `<PENDING — fill from smokeA_016c290.log>`
- **smokeB** — same smoke with `FLEXKV=1`: FlexKV server per shard, D2H PUT
  observed, connector reset survives weight sync:
  `<PENDING — fill from smokeB_016c290.log>`
- Full-chain serving validation (8×H100, 4×TP2, kv-router): router
  `overlap_blocks` up to 2459; FlexKV H2D fetches land with 0 cancels in the
  healthy regime; GPU-hit + external-hit joint coverage 98%+.

### Known issues / operating guidance (disclosed honestly)

1. **High-miss operating points can livelock the current FlexKV
   implementation** (allocator starvation family: put-pin defeats vLLM
   preemption; transfer storms; lookup-retry scheduler starvation). Observed
   and archived across 7 reproductions on the pre-#279 code; re-validation on
   ≥ #279 pending. Safe-zone rule of thumb:
   `C_per_engine × mean_context_tokens ≤ ~0.7 × gate_tokens`, where
   `gate_tokens = kv-cache-memory-bytes / kv_bytes_per_token`. The README
   ships this formula plus per-GPU suggested parameters. Upstream issues to
   FlexKV will be filed separately with py-spy + log evidence.
2. **Tied-embedding models (e.g. Qwen ≤ 4B, `tie_word_embeddings=true`) fail
   bucketed IPC weight updates** on the PR-54484 loader: the tied-alias guard
   (`_check_skipped_aliases`) raises when `lm_head.weight` arrives in a
   bucket without `model.embed_tokens.weight`. Use untied models (Qwen3-8B+)
   or co-locate tied pairs in one bucket. Worth reporting on the vLLM PR
   thread.
3. **Benefit is hardware-dependent**: on compute-rich GPUs (H100) recompute
   is cheap and the L2 tier's end-to-end gain can be neutral-to-negative;
   compute-constrained parts (H20 class) are the win scenario. This PR
   delivers capability and compatibility, not a universal speedup claim.

### Test plan

```bash
# backend smoke (1+1 GPU)
MODEL_PATH=<untied model, e.g. Qwen3-8B> TRAIN_FILE=... TEST_FILE=... \
  bash dynamo/smoke_dynamo_fully_async.sh \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True

# FlexKV L2 acceptance (same layout)
FLEXKV=1 MODEL_PATH=... TRAIN_FILE=... TEST_FILE=... \
  bash dynamo/smoke_dynamo_fully_async.sh \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True
```

Both must print `PASS: Dynamo fully_async smoke completed` and leave the verl
tree clean.

---

## Pre-submission checklist (internal; delete before posting)

- [ ] smokeA + smokeB PASS on the validation stack (in progress)
- [ ] Cut `feat/dynamo-flexkv-l2` from latest `upstream/main`; copy the file
      manifest above from `feat/dynamo-dynres`; commit as hou2lin
- [ ] Rewrite `dynamo/README.md` in English: install (stack pins), knob
      table, safe-zone formula, known issues
- [ ] Update `REQUIRED_VERL.txt` to the verl `main` pin actually validated
      (`9c76436`), removing the stale V1-era pin
- [ ] tests/assets final selection; sanity-import `recipe.dynamo.register`
      from the new branch
- [ ] User review → explicit approval → submit (hard rule)
