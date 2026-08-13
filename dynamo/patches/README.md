# Staged patches for out-of-repo components

Patches here implement flexkv_l2_design.md items whose home is **not** this
repository (ai-dynamo wheel sources, FlexKV). They are staged as patch files so
the whole L2 line of work lives on one branch; each graduates to a PR against
its home repository once the Phase-1 validation gate passes.

Apply against the dynamo checkout root (the directory containing
`components/`):

```bash
cd <dynamo-src>   # e.g. /workspace/verl/bench/dynamo
patch -p1 < d1_thunderagent_capacity_flexkv.patch
```

| patch | design item | home repo | activation |
|---|---|---|---|
| `d1_thunderagent_capacity_flexkv.patch` | D1 — router ledger credits `flexkv_capacity.host_total_tokens` | ai-dynamo (`thunderagent_router/capacity.py`) | `DYN_THUNDERAGENT_FLEXKV_L2=1` (default **off**: crediting capacity without victim pinning degrades admission — design doc §2) |
| `f1_flexkv_pin_requests.patch` | F1 — PinRequest/UnpinRequest messages + KVServer handlers (locks the deepest matched CPU radix node; eviction is leaf-only so the whole path survives) | FlexKV (`flexkv/server/{request,server}.py`; apply at FlexKV source root, the launcher overlay ships these files) | inert unless pin messages arrive |
| `d2d3_thunderagent_pin_hooks.patch` | D2/D3 — router caches each program's latest prompt tokens, pins the victim's path on pause, unpins at end_program (deferred past resume to avoid the unlock-before-GET race); endpoints discovered by globbing the FlexKV IPC sockets, fire-and-forget PUSH | ai-dynamo (`thunderagent_router/{__main__,router}.py`; apply at dynamo checkout root) | `THUNDERAGENT_FLEXKV_PIN=1` (default off) |
