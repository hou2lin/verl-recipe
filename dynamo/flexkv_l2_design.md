# ThunderAgent × FlexKV L2: Victim Pinning Design

Status: **Phase-2 mechanism complete; Gate A closed 2026-08-16 (structural finding) — Gate A′ (D1 admission-relaxation) running, Gate B queued behind it on the same chain.**
Progress tracker: `flexkv_l2_progress.md`
Branch: `feat/thunderagent-flexkv-l2`
Owners: hou2lin
Last updated: 2026-08-13

## 1. Problem statement

ThunderAgent (TA) admission control keeps the GPU KV pool coherent as long as the
admitted working set fits the pool. When the pool shrinks below the working set
(deep-context agents, small `gpu_memory_utilization`, large models), TA's second
lever engages: **pause** — evict a victim program at a tool boundary, resume it
later. Resume requires the victim's KV back on the GPU; today that means a full
context recompute (cost ∝ c²).

FlexKV should be the natural L2 here: victim KV spills to CPU on pause, DMA-loads
back on resume. **Measured on 8×H100 / Qwen3-Coder-30B-A3B / tp4×2 (Phase G/H,
2026-08-09..13), the current integration cannot deliver this:**

| evidence (mem-util sweep, C256, 512 traj/arm) | value |
|---|---|
| TA collapse point (pause first engages) | pool ratio ~0.30 (268K tok/shard) |
| TA recompute storm at 0.25 (106K tok/shard) | ×5.3 logical reprocessing, hit 35% |
| FlexKV recovery at 0.25 (`+FlexKV` arm) | 3.99M tok = **0.9% of the 443M recompute volume** |
| CPU copy survival at resume (external hit rate) | **16%** |
| Wall-clock delta ±FlexKV, every point 0.68→0.25 | net zero (within ±8% variance) |
| sticky (no admission) reference at 0.25 | ×93.9 reprocessing — what "no protection" costs |

Root cause: FlexKV copies are **write-behind + globally LRU'd**. The CPU pool
turns over in minutes; a paused victim waits tens of minutes. By resume time the
copy is gone. The recovery that does happen is accidental.

**Thesis: victim pinning — a deliberate spill-and-pin protocol driven by TA's
pause/resume events — is the necessary condition for FlexKV to act as TA's L2.**

## 2. Reference architecture: what TA×SGLang-HiCache already does

The TA router already understands a host tier — for SGLang HiCache only
(shipped in ai-dynamo/dynamo#11185):

- **Publish** (`dynamo/sglang/register.py`): worker reads
  `hicache_host_total_tokens` from scheduler info and injects
  `runtime_config.set_engine_specific("sglang_hicache_capacity",
  {"host_total_tokens": N})` into the MDC.
- **Consume** (`thunderagent_router/capacity.py`): the retention budget becomes
  `device_pool + host_total_tokens`, i.e. TA pauses later because spilled KV is
  not lost.
- **Why that is enough for SGLang**: HiCache is engine-native. Eviction
  automatically spills to host, the scheduler knows what the host holds, restore
  is deterministic engine behavior. TA only needs the bigger number.

### Gaps for vLLM+FlexKV (connector, not engine-native)

| # | gap | consequence |
|---|---|---|
| G1 | vLLM worker publishes no host capacity | TA ledger is blind to the CPU pool (confirmed: ta+fkv arms admit identically to bare arms) |
| G2 | eviction does not spill; CPU pool LRU is decoupled from GPU lifecycle | victim copies die before resume (16% survival) |
| G3 | restore is an opportunistic GET, not a deterministic recall | resume = recompute in practice |

⚠️ **Bridging G1 alone is worse than nothing**: TA would admit past the GPU pool
("pause later") while the copies that justify it don't reliably exist —
degenerating toward the sticky ×93.9 regime. G1 and G2/G3 ship together.

## 3. Design

### 3.1 Components and control flow

```
             ┌────────────────────────────┐
             │ thunderagent_router (P1/P2)│
             │  _pause_acting_locked ──┐  │
             │  _resume_program ────┐  │  │
             └──────────┬──────────┼──┼──┘
        MDC (capacity)  │   dynamo endpoint RPC (pin protocol)
             ┌──────────┴──────────▼──▼──────┐
             │ vLLM worker + worker_extension │  (P0 publish / P1 executor)
             │   pin_session / unpin_session  │
             └──────────────┬────────────────┘
                            │ FlexKV client IPC
             ┌──────────────▼────────────────┐
             │ FlexKV KVServer                │  (new capability)
             │  put_sync(session)             │
             │  pin(session) / unpin(session) │
             │  pinned quota + metrics        │
             └───────────────────────────────┘
```

Pause path: victim chosen → `put_sync(victim)` (idempotent by block hash) →
`pin(victim)` → victim's request gate closes. Resume path: worker assigned →
optional prefetch → request flows, GET now hits pinned copies → `unpin(victim)`.

### 3.2 Ledger semantics (dual accounting)

- `Program.resident_tier ∈ {GPU, CPU}`. ACTIVE programs charge the **GPU
  ledger** (unchanged). PAUSED+pinned programs charge the **CPU ledger**.
- Admission (`_select_worker_for_new_program_locked`) checks the GPU ledger, as
  today. Pause decisions additionally check CPU-ledger headroom: **if the pinned
  quota is exhausted, pausing stops buying anything** — the scheduler must know
  that and fall back (queue harder / refuse admission), not thrash.
- Resume (`_greedy_resume`) releases the CPU ledger and re-charges the GPU
  ledger.

### 3.3 CPU pool partitioning

`pinned_quota_ratio` (default 0.5) splits the CPU pool: pinned region (ledgered,
no LRU) vs flow region (write-behind + LRU as today). Guards the write window
that kvrouter-style consumers rely on.

### 3.4 Invalidation

Weight update invalidates **all** tiers including pinned blocks. This must land
on top of `feat/rl_kv_clear_multishard` (`KVManager.reset`) — single reset
semantics, no forked invalidation paths. `end_program` additionally drops that
session's CPU blocks (pinned or not) to free the pool early.

### 3.5 Victim selection, CPU-aware (later phase)

`_smallest_candidates` cost model gains a term: a program whose context is
already fully resident in the pinned region restores at DMA cost
(~0.2 s / 13k tok over PCIe) instead of c² recompute — prefer pausing those.
Batch-resume needs DMA pacing (the sticky+fkv arms showed GET bubbles and
per-shard asymmetry when transfers pile up).

### 3.6 Active-set pinning (Phase-2 redirect, from the Phase-1 gate data)

The Phase-1 gate arms falsified the design's implicit assumption. Formal-metric
comparison at pool ratio 0.25 (shared topology, C256, r16):

| "needs-L2" requests (prompt ≥4k, GPU match < 50%) | +pin | no-pin |
|---|---|---|
| count | 21,115 | 21,594 |
| caught by CPU tier | 1,006 (**4.8%**) | 551 (2.6%) |

Victim pinning works (relative catch rate +85%, big hits +65%) but **pause
victims are only ~8% of context loss** (1,697 pins vs 21k reconstruction
events): the dominant loss channel at deep pool starvation is ordinary
inter-turn eviction of ACTIVE trajectories, whose write-behind copies die in
the CPU pool's minute-scale LRU turnover just like victims did.

**Redirect: protect the whole active set, not just pause victims.** The pin
trigger moves from `_pause_acting_locked` to admission/each-turn PUT (pin the
trajectory's path on every completed turn; unpin unchanged at `end_program`).
The active set's context (~3.3M tokens at 0.25/C256) exceeds the CPU pool
(~2M node-wide), so the quota ledger (F3/D4) is promoted from safety feature
to core mechanism: it decides *who* gets protection (candidate policy:
deepest-context first — the c² recompute gradient).

## 4. Work breakdown

### FlexKV (prerequisite capabilities)

| id | item | notes |
|---|---|---|
| F1 | `pin(session)/unpin(session)` API + IPC messages | needs block→session ownership index; tag on the existing PUT path. Patch also fixes an **upstream KVServer pre-start race**: readiness polls (`IsReadyRequest`) arriving before the `StartRequest` crashed the server (strict first-message state machine) — shared multi-client mode makes this a coin flip per launch; now answered with `is_ready=False` |
| F2 | `put_sync(session)` forced spill | idempotent via block-hash dedup |
| F3 | pinned quota partition + `pinned_bytes/sessions` metrics | see §3.3 |
| F4 | reset covers pinned region | **coordinate with `feat/rl_kv_clear_multishard` first** |

### ai-dynamo (thunderagent_router; follow-up PR to #11185)

| id | item | notes |
|---|---|---|
| D1 | `capacity.py`: consume `flexkv_capacity.host_total_tokens` | ~10 lines, parallel to the sglang key |
| D2 | pause hook → pin protocol RPC | `_pause_acting_locked` |
| D3 | resume hook → unpin (+prefetch) | `_resume_program` |
| D4 | dual ledger (§3.2) | `_worker_used` / `_greedy_resume` / `Program.resident_tier` |
| D5 | CPU-aware victim cost model (§3.5) | `_smallest_candidates` |

### verl-recipe (this repo; all via existing conventions)

| id | item | notes |
|---|---|---|
| R1 | publish host capacity into MDC runtime_data | new patch point in `_dynamo_vllm_with_control.py` (established monkey-patch precedent; zero dynamo changes) |
| R2 | `pin_session/unpin_session` RPC executor → FlexKV client IPC | `dynamo_worker_extension.py` |
| R3 | config surface `thunderagent.flexkv_l2.{enabled,pinned_quota_ratio,prefetch}` | `config/dynamo_trainer.yaml` + launcher passthrough |
| R4 | observability: pinned bytes, **pin hit rate** (copy survival at resume), restore DMA latency histogram | metrics sidecar + summarizer; without these the validation runs are unreadable |
| R5 | unit tests: pin lifecycle / invalidation / quota | mirror `tests/test_dynamo_thunderagent.py` conventions |

## 5. Phasing and validation gates

```
Phase 1 (minimal loop, DONE 2026-08-14 mechanism-level):
  R1 + D1(staged) + F1 + D2/D3 via direct IPC — shared mode only
  (private mode has no KVServer process; see proposal §2)
  GATE (running): 0.25 shared ±pin (k8pin25sp/k8pin25s, C256, r16)
        pass = pin hit rate 16% → ~100%
               AND wall clearly below the no-pin shared baseline

Phase 2 (next, redirected per §3.6):  active-set pinning (pin trigger at
  each-turn PUT) + F3 quota as the protection selector + D4 dual ledger +
  F4 invalidation (merged with rl_kv_clear; interface review is the ticket)
  GATE A: rerun 0.25 shared ±active-pin — needs-L2 catch rate 4.8% → >50%
          AND wall clearly below the no-pin shared baseline
          [closed 2026-08-16: every mechanism metric improves monotonically
           (GPU hit 35.8→44.6%, reprocess ×5.2→×3.6, catch 3.2→7.2%) but wall
           is flat — structural: at 0.25 admission caps running ≈ 5, recompute
           is paid from idle FLOPs. See the progress tracker for the verdict.]
  GATE A′ (added 2026-08-16 from the Gate A finding): D1 admission-relaxation —
          same 0.25/C256/r16 shared arm + active pin (quota 0.9) +
          DYN_THUNDERAGENT_FLEXKV_L2=1: the capacity ledger credits FlexKV
          host tokens (~+1.8M/shard vs ~0.45M GPU ⇒ admission widens ~5×),
          lifting running off ~5 so saved recompute becomes throughput.
          pass = wall clearly below the 0.25 6h4x band while completion stays
          clean. fail-mode to watch: eviction storm outruns L2 catch and wall
          regresses → add a partial-credit ratio knob and re-run.
          [CLOSED 2026-08-17, k8ad1: **full credit fails, and the mechanism is
           one layer deeper than the predicted eviction storm — GET landing
           starvation.** Widened admission keeps the GPU pool saturated, so
           FlexKV GETs that *did match* can't allocate destination blocks:
           3 cancels → allocation fallback → "recomputing N matched tokens".
           Measured: FlexKV hit 0.00–0.07%, GPU hit 4–9% (vs 44.6% Gate A),
           and shard0 hung inside the cancel loop at +2.8h (no crash log);
           arm killed at 12.8h. Conclusions: (1) D1 needs a partial-credit
           ratio (leave GPU headroom for GET landing); (2) engine-side GET
           landing reservation — i.e. Phase-3 resume-prefetch/DMA-pacing — is
           a *prerequisite* for aggressive credit, not a later optimization;
           (3) the FlexKV cancel-loop hang is an upstream bug to fix. Re-run
           deferred until after Gate B; credit_ratio ≈ 0.3 (~2.2× admission)
           is the candidate next probe.
           Why HiCache doesn't hit this: load-back is part of the scheduler's
           unified decision (host-hit load counts against the request's
           prefill budget; can't allocate → request WAITS, never degrades),
           and SGLang never credits host capacity into admission at all —
           its host tier only cheapens post-eviction restore. D1 is where we
           go *beyond* the reference; the fix ladder is ours to build:
           (a) router-only: partial credit + util-coupled dynamic credit
               (credit→0 as util crosses the 0.80 soft ceiling; router
               already subscribes to FPM/util) — Gate A′ re-probe config;
           (b) connector: bounded-wait GET retry instead of fallback, and
               GET-priority eviction (a certain 27k-token hit outranks a
               cold maybe-reused prefix) — new F items;
           (c) scheduler fusion (the HiCache shape): loads enter vLLM
               allocate_slots budgeting, unallocatable → waiting — Phase 3,
               with DMA pacing.]
  GATE B: long-trajectory probe (max_model_len 131072, max_turns 100,
          p32×r2 = 64 trajectories, C32, ±pin) — pause's native regime.
          [false start 2026-08-16: launcher bound max_num_batched_tokens to
           tok=131k → engine profile grew ~4 GiB → colocated weight-transfer
           all-gather OOM. Fixed: PHASE1_MAX_NUM_BATCHED_TOKENS env, pinned
           32768 (chunked prefill splits 131k prompts). Re-queued behind A′.]

Phase 3:  D5 CPU-aware victim + resume prefetch + DMA pacing
  GATE: batch-resume stress (many victims resumed same tick), no GET bubbles /
        per-shard asymmetry regression

Deferred (graduation, de-prioritized 2026-08-14): R2 worker-extension RPC —
  required only to bring pinning to private per-shard topology / multi-node.
  All Phase 2/3 development and validation proceed on shared-mode arms.
```

## 6. Risks / open questions

1. **Pin-time race**: pause is decided in `after_request` while the victim may
   still have an in-flight request; `put_sync` must complete before the victim's
   waiting event can be re-armed, or an early resume misses the copy.
2. **G1-only footgun**: never ship the capacity bridge without pinning (§2).
3. **Quota starvation**: pinned region full ⇒ pause stops helping; the dual
   ledger must make this visible to the scheduler (D4), else 0.25-style thrash
   returns with extra steps.
4. **Reset semantics**: single source of truth with `rl_kv_clear_multishard`;
   agree on the interface before F4 lands.
5. **Control-channel choice**: pause frequency is tick-scale (~seconds), so a
   dynamo endpoint RPC is sufficient; no new bus.

## 7. Measured baselines to beat (Phase G/H, for regression comparison)

| arm | wall | GPU hit | preempt | reprocess | FlexKV recovery |
|---|---|---|---|---|---|
| 0.30 TA bare      | 3h10m | 90.2% | 0 | ×1.00 | — |
| 0.30 TA+FlexKV    | 3h19m | 90.3% | 0 | ×1.00 | 2.01M |
| 0.30 sticky       | 3h51m | 7.6% | 175 | ×19.8 | — |
| 0.25 TA bare      | 6h33m | 35.2% | 12 | ×5.3 | — |
| 0.25 TA+FlexKV    | 6h55m | 38.1% | 3 | ×4.9 | 3.99M (0.9%) |
| 0.25 sticky       | 14h03m | 6.1% | 263 | ×93.9 | — |

Phase-1 gate arms (0.25 shared, C256, r16; measured 2026-08-14..15):

| arm | wall | GPU hit | reprocess | needs-L2 catch | FlexKV hit (trace) |
|---|---|---|---|---|---|
| k8pin25sr2 (shared, no pin) | 6h46m | 35.8% | ×5.2 | 3.2% | 11.40M |
| k8pin25sp (shared, **pin**) | 6h36m | 37.0% | ×4.7 | **4.8%** | 13.33M |

Gate verdict: mechanism **pass** (1,697/1,697 pins locked, lifecycle exact),
direction **pass** (+50% relative catch, +65% big hits), wall **not significant**
(−2.5%, within ±8% variance) — consistent with the §3.6 finding that pause
victims are ~8% of context loss. Phase 2 (active-set pinning, Gate A
catch > 50%) is the continuation.

Full data: `results/phase1/` bundles (k8mu*/k8smu*/k8pin*), PHASE_B_REPORT.md
Phase G/H/I sections, and the interactive summary
(`phase1_visual_summary.html`, §🔧).
