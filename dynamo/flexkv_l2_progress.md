# ThunderAgent × FlexKV L2 — Progress Tracker

Companion to `flexkv_l2_design.md` (the plan) — this file records where we
actually are. Update on every gate decision or item status change.

**Current position (2026-08-17 morning): Gate A′ closed (full-credit fail, GET landing starvation); Gate B arms running on the chain. Prior context: three-arm chain on its third
launch (18:37Z) — Gate A′ (`k8ad1`, D1 admission-relaxation) first, then Gate
B both arms (`k8gbo`/`k8gbp`). Launch 1 died on launcher OOM (batched-tokens
coupling); launch 2 died on an upstream FlexKV KVServer pre-start race (fixed
in the F1 patch). Launch-2 side effect: both Gate B arms completed full
rollouts with FlexKV silently no-op — preserved as bare-131k reference data.**

## Phase status at a glance

| Phase | Status | Gate verdict |
|---|---|---|
| Phase 1 — victim pinning minimal loop | ✅ **CLOSED** (2026-08-15) | Mechanism pass (1,697/1,697 pins), direction pass (+50% catch), wall not significant → redirected to §3.6 |
| Phase 2 — active-set pinning + quota | 🔶 **Mechanism DONE, Gate A closed with structural finding** (2026-08-16) | All mechanism metrics improve monotonically; wall is insensitive because recompute is off the critical path in the starvation regime — see "Gate A verdict" below |
| Gate A′ — D1 admission-relaxation | ❌ **CLOSED 2026-08-17: full credit fails — GET landing starvation** | `k8ad1` (0.25/C256/r16 + pin q0.9 + full host-token credit): admission widened as designed, but the saturated GPU pool starves FlexKV GETs of destination blocks — matched data gets **re-computed anyway** ("recomputing N matched tokens" after 3 alloc cancels). FlexKV hit 0.00–0.07%, GPU hit 4–9% (vs 44.6% in Gate A); shard0 hung in the cancel loop at +2.8h; arm killed at 12.8h. Verdict: partial-credit ratio + engine-side GET landing reservation (Phase-3 prefetch/pacing) are prerequisites; cancel-loop hang is an upstream FlexKV bug. Re-probe (credit≈0.3) after Gate B |
| Gate B — long-trajectory probe | 🚀 queued on same chain (after A′) | max_model_len 131072 (4k prompt + 124k response), max_turns 100, p32×r2 = 64 traj, C32, mem-util 0.68, shared FlexKV 336 GB; arms: `k8gbo` (no pin) → `k8gbp` (active pin, quota 0.9). **False start 08-16**: launcher bound `max_num_batched_tokens` to tok=131k → engine +4 GiB → weight-transfer all-gather OOM (2 GiB ask, 1.74 free); fixed via `PHASE1_MAX_NUM_BATCHED_TOKENS=32768` |
| Phase 3 — CPU-aware victim / prefetch / DMA pacing | ⏳ not started | — |
| R2 — private-mode / multi-node channel | 💤 deferred (user decision 2026-08-15) | — |

## Item-level status

### Phase 1 items

| item | status | evidence |
|---|---|---|
| R1 publish FlexKV capacity → MDC | ✅ done | 8 unit tests; live log `host_total_tokens=1048576 kv_bytes_per_token=98304` (matches capacity audit) twice on-machine |
| D1 router ledger credits host tokens | ✅ **ACTIVATED 2026-08-16** (`k8ad1`, Gate A′) | patch in `patches/`; gate `DYN_THUNDERAGENT_FLEXKV_L2` was kept off until the pin story worked (design §2); Gate A met that precondition. D1 is the missing link for the 0.25 starvation regime: admission accounts GPU-only capacity → running ≈ 5; crediting host tokens (~+1.8M/shard) widens admission ~5×, converting saved recompute from idle FLOPs into throughput. Verdict (Gate A′ row above): full credit fails via GET landing starvation — next probe is partial credit ≈ 0.3 |
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

### Upstream lineage check (2026-08-17)

Host-capacity credit entered upstream in **#11185** (07-06, hicache key,
unconditional); our checkout `59d6146` (#11245, 07-07) includes it. **#11321**
(07-07, not in our checkout) generalized the contract to
`native_offloading_capacity.total_tokens` with write-policy translation
(write_through → `G+(H−G)`, no double counting). Actions recorded in the
design doc: partial credit is the upstream-correct semantic for FlexKV's
copy-based write-behind; R1 should migrate to the neutral key on next rebase.
Admission is also being productized in Rust (#11434 API merged; #11616
Session-Aware policy, device-only) — our L2 work is the prototype line.

## Operational hardening picked up along the way (permanent)

- launcher FlexKV overlay: `server/request.py` added to cythonize+cleanup list
- pod bootstrap installs `liburing2` (flexkv.c_ext dlopen dependency)
- preflight kills stale FlexKV KVServers + removes `/tmp/flexkv_server*`
- canonical-tree discipline: cross-tree deploys use `patch` only, never `cat`
  (lineage mismatch burned one arm)
- private per-shard mode has **no KVServer process** — direct-IPC pin path is
  shared-mode only; private needs R2
- launcher previously bound `max_num_batched_tokens` to `tok` — at 131k this
  grows the engine ~4 GiB and OOMs the colocated weight transfer; now
  env-decoupled (`PHASE1_MAX_NUM_BATCHED_TOKENS`), long-context arms pin 32768
- preflight kill is now a retry loop (6×5s) — kill -9 → GPU memory release
  takes seconds; the old single-shot check false-FATALed the next arm
- **upstream KVServer pre-start race** (latent, bites shared mode): strict
  first-message state machine crashes on early `IsReadyRequest` polls; ~10
  prior arms won the coin flip, then two launches in a row lost it. Fixed in
  the F1 patch (pre-start polls answered `is_ready=False`). Two failure
  shapes from one bug: engine shards hang → 1800s frontend timeout (k8ad1),
  or connector fail-opens → whole arm runs with FlexKV silently disabled
  (k8gbo/k8gbp). The launcher's join-rate≥0 validation (exit 8) is what
  caught the silent shape — keep it

## Timeline

| date | milestone |
|---|---|
| 08-13 | design doc + branch; R1 done+verified; D1 staged; F1/D2/D3 staged |
| 08-14 | 8 smoke iterations → end-to-end pin confirmed (shared mode); Phase-1 gate arms launched |
| 08-15 | Phase-1 gate closed; §3.6 redirect; active-set + quota implemented and smoked |
| 08-16 | Gate A both runs closed; structural finding recorded; Gate B is next |
| 08-16 (evening) | Gate B launched: `k8gbo`/`k8gbp` two-arm pod-side chain (131k ctx, 100 turns, 64 traj, C32) |
| 08-16 (evening) | Gate B false start: `max_num_batched_tokens`=tok OOM at weight transfer; launcher env-decoupled, preflight kill-loop hardened |
| 08-16 (late) | Re-prioritized per user: **Gate A′ first** — 3-arm chain `k8ad1` → `k8gbo` → `k8gbp` launched 15:41Z; D1 activated for the first time (`DYN_THUNDERAGENT_FLEXKV_L2=1`) |
| 08-16 (night) | Chain launch 2 all-failed on the upstream KVServer pre-start race (k8ad1: shards hung 1800s; k8gbo/k8gbp: FlexKV silent no-op, full bare rollouts preserved as 131k reference — 2,744 turns in ~50 min). Race fixed in F1 patch; chain launch 3 at 18:37Z |
| 08-17 (morning) | **Gate A′ closed: full credit fails via GET landing starvation** (FlexKV hit ~0%, GPU hit 4–9%, shard0 hung +2.8h; killed at 12.8h). Chain advanced to Gate B (`k8gbo` from 01:26Z) |
