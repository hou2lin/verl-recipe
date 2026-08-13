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
