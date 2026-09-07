# PR 草稿:FlexKV L2 CPU KV cache support for the Dynamo rollout backend(recipe-only)

> **状态:本地草稿,未提交。🔴 红线:任何对外 PR/MR 须经用户 review 并明确批准后才执行。**
> **目标(用户 2026-09-07 定稿):在 verl+dynamo 链路上兼容 vllm+FlexKV 的 L2 cache,只涉及 verl-recipe。**
> base:`verl-project/verl-recipe` `main`(fetch 时点 `ee3aef1`,提交前 rebase 到最新)
> head:待从 `feat/dynamo-dynres` 摘取 core+flexkv 文件切出干净分支 `feat/dynamo-flexkv-l2`
> 本草稿取代 `PR_DRAFT_fully_async.md` 的定位(该稿把 FlexKV 划为"后续 PR",与定稿目标不符)。
> `PR_DRAFT_v1_migration.md` / `feat/dynamo-v1-migration` 保持独立,不与本 PR 混合。

## PR Title

[dynamo] FlexKV L2 CPU KV-cache support for the Dynamo rollout backend (recipe-only)

## PR Body(草稿)

### What does this PR do?

Adds a **recipe-only** integration that runs verl RL rollout on the full
serving chain **verl + dynamo(kv-aware router) + vLLM + FlexKV L2 CPU
cache**:

1. **Dynamo rollout backend**(承载层):rollout requests served by
   `dynamo.frontend --router-mode kv` + `dynamo.vllm` worker shards;
   weight sync(NCCL first hop + CUDA-IPC second hop)and
   abort→sync→resume reuse verl's stock CheckpointEngineManager.
2. **FlexKV L2 wiring**(本 PR 主体):per-shard FlexKV server lifecycle
   与 vLLM `FlexKVConnectorV1` 接线——KV blocks evicted from GPU spill to
   a CPU (DRAM) tier and are fetched back (H2D) on prefix re-hit instead
   of being recomputed;跨 rollout 轮次与权重更新存活(reset-after-sync
   choreography)。

**verl itself is untouched** —— 全部通过 verl 的扩展点注册:
- `VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register`
- `actor_rollout_ref.rollout.name=dynamo`(rollout registry)
- `engine_kwargs.dynamo.*` 后端旋钮
- `DynamoReplica.get_ray_class_with_init_args` 挂接子类化的
  CheckpointEngineWorker(无 monkey-patch)

验收证据:全程 verl 树 `git status` clean(verl main pinned commit 见
`dynamo/REQUIRED_VERL.txt`)。

### Why

RL rollout 的多轮 agent 负载有强前缀重现性(同一轨迹逐轮增长、跨
rollout 轮重放)。GPU KV cache 容量有限,驱逐即重算;FlexKV 提供 CPU
DRAM L2 层,把"重算"变成"PCIe 取回"。dynamo 的 kv-aware router 又把
同前缀请求路由到持块 worker,两者组合构成完整的 KV 复用栈。本 PR 把这
条链在 verl-recipe 内打通,verl 主仓零改动。

### FlexKV integration surface(本 PR 核心)

Per-shard 隔离(多 TP shard 同机共存的关键,`dynamo_async_server.py`):

| 机制 | 说明 |
|---|---|
| `FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_server_g<cvd>` | 每 shard 独立 IPC socket(按 CUDA_VISIBLE_DEVICES 后缀) |
| `FLEXKV_INSTANCE_NUM` / `FLEXKV_INSTANCE_ID` | shard 编号注入 |
| `FLEXKV_PY_METRICS_PORT` / `FLEXKV_CPP_METRICS_PORT` | 每 shard 指标端口错位 |
| `FLEXKV_SHARED_CPU_CACHE` | 单池多 shard 共享 CPU cache 模式 |
| vLLM 侧 | `--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'` + reset-after-sync |

用户可见旋钮(env,均有默认值):`DYNAMO_USE_FLEXKV`、
`FLEXKV_CPU_CACHE_GB`、`FLEXKV_CONFIG_PATH`、`FLEXKV_ENABLE_MPS`
(默认 0,MPS 与 RL 权重更新流程冲突)、`FLEXKV_INIT_READY_TIMEOUT_S`、
`FLEXKV_ENABLE_METRICS`、`FLEXKV_ZMQ_IMMEDIATE` / 
`FLEXKV_TRANSLATE_PHYSICAL_DEVICE`(条件门,默认 off,见 Known issues)。

### Files(拟含清单,从 feat/dynamo-dynres 摘取)

| file | role |
|---|---|
| `dynamo/register.py` | external-module 入口 |
| `dynamo/dynamo_async_server.py` | DynamoHttpServer(etcd/nats/frontend/worker 生命周期 + abort/resume 编排)+ **FlexKV per-shard wiring** + DynamoCheckpointEngineWorker + DynamoReplica |
| `dynamo/_dynamo_vllm_with_control.py` | dynamo.vllm sidecar(collective_rpc 控制通道:update_weights_from_ipc / pause / resume / reset) |
| `dynamo/dynamo_worker_extension.py` | vLLM worker extension(CUDA-IPC 权重接收) |
| `dynamo/dynamo_agent_loop.py` / `dynamo_rollout.py` | agent-loop 客户端 / rollout 适配(含 FlexKV service 关停时序) |
| `dynamo/dynamo_thunderagent.py` | TA 路由启停(后端能力,默认关闭) |
| `dynamo/main_dynamo_fully_async.py` + `dynamo/config/*` | thin entrypoint + hydra 配置(fully_async_policy,v0 系;**无 use_v1**) |
| `dynamo/smoke_dynamo_fully_async.sh` | 1+1 GPU 冒烟;`FLEXKV=1` 变体即 FlexKV 验收(D2H PUT + reset-after-sync 断言) |
| `dynamo/README.md` + `dynamo/REQUIRED_VERL.txt` | 版本钉 + 依赖说明 |
| `dynamo/tests/` + `dynamo/assets/` | 单测与素材(review 时定夺取舍) |

**排除**(留在实验台分支/各自去向):v1 线(独立分支
`feat/dynamo-v1-migration`)、replay/kg8 全部实验工装
(frozen_replay / dump_trajectories / extract_turn_gaps / run_kg8_* /
serve_dynamo_kvrouter_* / task_config_* / source_kg8_launch)、
`patches/`(FlexKV/dynamo 上游补丁,走上游仓库)、flexkv_l2_design /
progress / pin_api_proposal(本地设计文档,正文引用要点)。

### Dependencies(README 明示,不在本 PR 内)

- verl main pinned(`REQUIRED_VERL.txt`)
- vLLM fork(0.28.x,含原生 `FlexKVConnectorV1` / `reset_connector` /
  `--kv-cache-memory-bytes`)
- FlexKV(taco `17bec07`+;初始化就绪超时 / device-sleep 容错 / ZMQ
  稳健性修复已作为 patch 提交上游,合入前可用 `patches/` 本地应用)
- dynamo ≥ 1.4.2(kv-events ZMQ / overlap_blocks 路由)

### Acceptance evidence

- smokeA:dynamo 后端 1+1 GPU 冒烟 rc=0,verl 树 clean;
- smokeB(FlexKV on):D2H PUT 实证 + 权重同步后 reset-after-sync 存活;
- smokeC:TA 路由与 FlexKV 同开存活;
- 完整链路验证(8×H100,4×TP2,kv-router):路由 overlap_blocks 峰值
  2459、FlexKV H2D 落地(0 cancel 健康区)、GPU hit + external hit
  联合覆盖 98%+;端到端训练(colocate 两臂)rc=0。

### Known issues / operating guidance(诚实披露)

当前 FlexKV 实现存在高未命中区的 allocator 饿死类缺陷(put-pin 阻断
vLLM 抢占 / 传输风暴 / lookup-retry 调度环饥饿;已在 6 组复现中留档,
py-spy + 日志证据将随上游 issue 提交 FlexKV 仓库)。**安全运行域**:
活跃在飞 tokens + 在飞传输 pin < 单引擎 KV 容量(经验值 ≤70%),即
`C_per_engine × mean_ctx ≤ 0.7 × gate_tokens`。README 将附此公式与
per-GPU 建议参数表。收益侧披露:H100 上重算成本低,L2 净收益依赖
硬件比值(H20 类算力受限硬件为正收益场景,见 slide 数据);本 PR 交付
的是**能力与兼容性**,不承诺特定硬件上的加速数字。

### Test plan

1. `bash dynamo/smoke_dynamo_fully_async.sh`(1+1 GPU)rc=0;
2. `FLEXKV=1 bash dynamo/smoke_dynamo_fully_async.sh` rc=0 且日志含
   D2H PUT + reset-after-sync 断言;
3. verl 树 `git status` clean 断言(冒烟内置)。

---

## 提交前 checklist(内部,提交时删除)

- [ ] 从 `feat/dynamo-dynres` cherry-pick/摘文件切出 `feat/dynamo-flexkv-l2`(基于最新 upstream/main)
- [ ] README 重写:安装(vllm fork / FlexKV / dynamo 版本钉)、旋钮表、安全运行域公式、Known issues 链接上游 issue 编号
- [ ] tests/assets 取舍;smoke FlexKV 变体入口固化(FLEXKV=1)
- [ ] 复跑 smokeA/B 于新分支尖端
- [ ] 用户 review → 批准 → 提交(红线)
