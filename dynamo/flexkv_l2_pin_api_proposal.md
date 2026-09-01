# FlexKV pin protocol — interface proposal (F1/F2, D2/D3, R2)

Status: **proposal / Phase-1 implementation guide** · companion to
`flexkv_l2_design.md` · to be reviewed against `feat/rl_kv_clear_multishard`
before the FlexKV changes graduate to their home repo.

## 1. FlexKV server API (F1, F2)

New request types on the existing KVServer IPC channel (same envelope as
GET/PUT):

```
PIN_SESSIONS   {session_ids: [str], deadline_ms?: int}
  -> {pinned: {session_id: n_blocks}, rejected: [session_id]}
UNPIN_SESSIONS {session_ids: [str]}
  -> {unpinned: {session_id: n_blocks}}            # idempotent, unknown ids ok
PUT_SYNC       {session_id: str, token_ids: [...], kv_ref: ...}   # Phase 2
  -> {stored_blocks: int}                          # idempotent by block hash
```

Semantics:

- **Ownership index**: PUT paths tag blocks with the requesting session
  (`agent_context.session_id` already rides the request envelope end-to-end).
  A block referenced by several sessions carries a refcount; pinned means
  "refcount from pinned sessions > 0".
- **Pinning** removes blocks from LRU consideration; it never copies.
  `rejected` is returned when the pinned-region quota would be exceeded —
  partial success is allowed and the caller must treat `rejected` victims as
  unprotected (their resume falls back to recompute, as today).
- **Quota**: `FLEXKV_PINNED_QUOTA_RATIO` (default 0.5) caps the pinned share of
  the CPU pool, guarding the write-behind flow region.
- **Idempotency**: PIN of an already-pinned session and UNPIN of an unknown
  session are no-ops with success responses — the router retries freely.
- **Invalidation**: `KVManager.reset` (rl_kv_clear) drops pinned blocks like
  any others and clears the pin table; a post-update resume then recomputes,
  which is correct (the KV is stale). `end_program` triggers UNPIN + eager
  drop of that session's blocks.
- **Phase 1 simplification**: PUT_SYNC is deferred. A pause victim's last
  completed turn was already written by write-behind, so pinning alone
  protects a complete copy of everything except tokens generated after the
  last PUT (none: pause lands on a turn boundary, after `after_request`).

## 2. Control channel (D2/D3 → F1)

Two paths; Phase 1 uses (a):

- **(a) Single-node pragmatic (Phase 1)**: the ThunderAgent router process and
  the FlexKV KVServer(s) share a container in this deployment. The router
  broadcasts PIN/UNPIN over the servers' IPC sockets. Zero new transport.
  **Measured constraint (pinsmk1-8, 2026-08-14): this path only exists in
  shared mode** (`FLEXKV_INSTANCE_NUM>1`) — in private per-shard mode
  KVManager runs embedded in the EngineCore process with **no KVServer
  process and no IPC channel at all** (only the gpu_register socket exists).
  Private mode therefore requires path (b); (a) validates the mechanism on
  shared-mode arms.
- **(b) Multi-node proper (graduation)**: the worker extension
  (`dynamo_worker_extension.py`) exposes `pin_sessions/unpin_sessions`
  RPCs (reachable via each shard's control ZMQ, whose endpoint the recipe
  already knows how to publish through MDC runtime_data — same seam as R1);
  the router resolves the victim's pinned worker and calls through. This is
  R2 in the design doc — **required for private per-shard mode** (see the
  measured constraint above), not merely for multi-node.

## 3. Router hooks (D2/D3)

- `_pause_acting_locked(victim)`: after the victim's gate closes, fire-and-log
  `PIN_SESSIONS([victim])` toward the victim's assigned worker socket. A
  `rejected` response downgrades the victim to unprotected (metric below).
- `_resume_program(victim, worker)`: `UNPIN_SESSIONS([victim])` after the
  resumed request is issued (unpin-after-first-GET keeps the copy alive
  through the restore read).
- Failure policy: pin/unpin errors are logged, never block pause/resume — the
  scheduler must function with FlexKV down (degrades to today's behavior).

## 4. Observability (R4, minimum for the validation gate)

- Router log lines: `pin.request/pin.ok/pin.rejected/unpin.ok` with session id
  and block counts (grep-friendly; sidecar aggregation later).
- FlexKV metrics: `flexkv_pinned_blocks`, `flexkv_pinned_sessions`,
  `flexkv_pin_rejects_total`.
- **Pin hit rate** (the gate metric): fraction of resumed victims whose first
  post-resume request served its GET from CPU (join router resume events with
  the flexkv request trace on session id). Baseline today: 16%.

## 5. Open questions for the rl_kv_clear review

1. Block↔session index placement: server-side table vs tags in the existing
   block metadata (`_put_match` already carries request identity).
2. Refcount vs last-writer-wins when rollouts share prefix blocks
   (32-rollout t0 prefixes are shared; pinning one session must not pin the
   world — proposal: pin only blocks whose *unique* owner is the session,
   shared prefix blocks stay LRU-managed since they are hot anyway).
3. Whether UNPIN should support `drop=true` (unpin + immediate eviction) for
   `end_program`, saving a second call.
