# ThunderAgent × FlexKV L2 — Progress Tracker

Companion to `flexkv_l2_design.md` (the plan) — this file records where we
actually are. Update on every gate decision or item status change.

**Current position (2026-08-16): Phase 2 mechanism complete; Gate A closed
with a structural finding; Gate B (long-trajectory probe) launched 08-16
evening — two arms running.**

## Phase status at a glance

| Phase | Status | Gate verdict |
|---|---|---|
| Phase 1 — victim pinning minimal loop | ✅ **CLOSED** (2026-08-15) | Mechanism pass (1,697/1,697 pins), direction pass (+50% catch), wall not significant → redirected to §3.6 |
| Phase 2 — active-set pinning + quota | 🔶 **Mechanism DONE, Gate A closed with structural finding** (2026-08-16) | All mechanism metrics improve monotonically; wall is insensitive because recompute is off the critical path in the starvation regime — see "Gate A verdict" below |
| Gate B — long-trajectory probe | 🚀 **LAUNCHED 2026-08-16** (decisive) | max_model_len 131072 (4k prompt + 124k response), max_turns 100, p32×r2 = 64 traj, C32, mem-util 0.68, shared FlexKV 336 GB; arms: `k8gbo` (no pin) → `k8gbp` (active pin, quota 0.9); pod-side sequential chain (survives local cert expiry) |
| Phase 3 — CPU-aware victim / prefetch / DMA pacing | ⏳ not started | — |
| R2 — private-mode / multi-node channel | 💤 deferred (user decision 2026-08-15) | — |

## Item-level status

### Phase 1 items

| item | status | evidence |
|---|---|---|
| R1 publish FlexKV capacity → MDC | ✅ done | 8 unit tests; live log `host_total_tokens=1048576 kv_bytes_per_token=98304` (matches capacity audit) twice on-machine |
| D1 router ledger credits host tokens | ✅ staged, **never activated** → 🔥 **activation unblocked, promoted** | patch in `patches/`; gated by `DYN_THUNDERAGENT_FLEXKV_L2` (kept off — activation is bound to a working pin story per design §2). **2026-08-16: precondition met** — Gate A proved the pin story works. D1 is the missing link for the 0.25 starvation regime: admission accounts GPU-only capacity → running ≈ 5; crediting host tokens lifts running, converting saved recompute from idle FLOPs into throughput. Plan: **Gate A′ admission-relaxation arm** (0.25 + `DYN_THUNDERAGENT_FLEXKV_L2=1` + active pin, expanded pool) right after Gate B. Risk to measure: full credit widens admission ~9× — if L2 catch can't absorb the extra eviction storm, wall may regress; may need a partial-credit ratio knob |
| F1 Pin/Unpin messages + KVServer handlers | ✅ done (shared mode) | pinsmk8: 341/341 pinned, unpin balanced |
| D2/D3 router hooks (victim mode) | ✅ done | Phase-1 gate arms |
| Phase-1 gate (0.25 shared ±pin) | ✅ closed | catch 3.2→4.8%, wall 6h46m→6h36m (±variance); pause victims are only ~8% of context loss → §3.6 redirect |

### Phase 2 items (redirected: active-set pinning)

| item | status | evidence |
|---|---|---|
| Pin trigger → each-turn after_request (rolling re-pin) | ✅ done | aspsmk4: 2,210 pins, rolling-update semantics observed (same session 211→243 blocks) |
| F3 quota ledger + deepest-context-first selector | ✅ done | displaced/rejected logged; narrow quota: 61% rejected; expanded quota: 1,454 rejected (edge only) |
| Gate A run 1 (quota 0.5 ≈ 1M tok) | ✅ | GPU hit 43.0%, reprocess ×3.9, catch 4.5% — quota-starved |
| Gate A run 2 (shm 384Gi, pool 336GB ≈ 3.5M tok, quota 0.9) | ✅ | GPU hit **44.6%**, reprocess **×3.6**, FlexKV hit **17.81M**, catch **7.2%** — wall 6h50m (unchanged) |
| D4 dual ledger (router-side) | ⏳ not started | server-side quota currently carries the selector role |
| F2 put_sync forced spill | ⏳ deferred | revival candidate (Phase-1 analysis: PUT-lands-late gap), pending Gate B |
| F4 invalidation merged with rl_kv_clear | 💤 deferred until training | not needed on the current gate line: all gate experiments are rollout-only (`PHASE1_STEPS=0`), weights never update, so `rl_kv_clear` never fires and pinned KV never goes stale. Becomes a **correctness requirement** (stale-weight KV hits = wrong outputs) the moment multi-step RL runs — mandatory before any PR. Interface review folds into PR prep (proposal §5 is the agenda) |

### Gate A verdict (2026-08-16) — the structural finding

Four-arm progression at pool ratio 0.25 (shared, C256, r16):

| | baseline | victim pin | active (narrow quota) | active (expanded) |
|---|---|---|---|---|
| wall | 6h46m | 6h36m | 6h41m | 6h50m |
| GPU hit | 35.8% | 37.0% | 43.0% | **44.6%** |
| reprocess | ×5.2 | ×4.7 | ×3.9 | **×3.6** |
| FlexKV hit | 11.40M | 13.33M | 11.26M | **17.81M** |
| needs-L2 catch | 3.2% | 4.8% | 4.5% | **7.2%** |

Every mechanism metric improves monotonically; wall does not move. Cause is
structural, and mirrors the Phase-H sticky finding: **at 0.25 the bottleneck
is parallelism starvation (running ≈ 5), not recompute** — TA's admission
already pays recompute out of idle FLOPs, so saving recompute saves idle
FLOPs. L2's wall-clock benefit only exists where recompute sits on the
critical path: **long-trajectory regimes (restore cost ∝ c², Gate B) or
admission-less schedulers (sticky+FlexKV −19%, already measured).**

## Operational hardening picked up along the way (permanent)

- launcher FlexKV overlay: `server/request.py` added to cythonize+cleanup list
- pod bootstrap installs `liburing2` (flexkv.c_ext dlopen dependency)
- preflight kills stale FlexKV KVServers + removes `/tmp/flexkv_server*`
- canonical-tree discipline: cross-tree deploys use `patch` only, never `cat`
  (lineage mismatch burned one arm)
- private per-shard mode has **no KVServer process** — direct-IPC pin path is
  shared-mode only; private needs R2

## Timeline

| date | milestone |
|---|---|
| 08-13 | design doc + branch; R1 done+verified; D1 staged; F1/D2/D3 staged |
| 08-14 | 8 smoke iterations → end-to-end pin confirmed (shared mode); Phase-1 gate arms launched |
| 08-15 | Phase-1 gate closed; §3.6 redirect; active-set + quota implemented and smoked |
| 08-16 | Gate A both runs closed; structural finding recorded; Gate B is next |
| 08-16 (evening) | Gate B launched: `k8gbo`/`k8gbp` two-arm pod-side chain (131k ctx, 100 turns, 64 traj, C32) |
