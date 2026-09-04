#!/bin/bash
set -u
source /workspace/venvs/dynres-main/bin/activate
export PATH=/workspace/bin:$PATH
export PYTHONPATH=/workspace/dynres/pkgroot:${PYTHONPATH:-}
export DOCKER_HOST=tcp://localhost:2375
export VERL_DYNAMO_CE_UPDATE_TIMEOUT_S=600
# self-cleaning
pkill -9 -f "main_pp[o]" 2>/dev/null; pkill -9 -f "_dynamo_vllm_with_contro[l]" 2>/dev/null
pkill -9 -f "dynamo.fronten[d]" 2>/dev/null; pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null
ray stop --force >/dev/null 2>&1; sleep 2
pkill -9 -f "ray:[:]" 2>/dev/null; pkill -9 -f "multiprocessing.spaw[n]" 2>/dev/null
pkill -9 -f "multiprocessing.resource_tracke[r]" 2>/dev/null
pkill -9 -f "cuda-mps-contro[l]" 2>/dev/null; pkill -9 -f "cuda-mps-serve[r]" 2>/dev/null
rm -rf /tmp/nvidia-mps; rm -f /tmp/flexkv_server*
pkill -9 -f "verl_dynamo_etc[d]" 2>/dev/null; pkill -9 -f "etcd --data-di[r]" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
sleep 1; rm -rf /tmp/etcd-smoke-data
setsid /workspace/bin/etcd --data-dir /tmp/etcd-smoke-data >/workspace/phase2/etcd.log 2>&1 &
setsid /workspace/bin/nats-server -js >/workspace/phase2/nats.log 2>&1 &
sleep 3
cd /workspace/phase2
echo "[srcKG8] $(date -u +%FT%TZ) START source-kg8-mt20" >> /workspace/phase2/srcKG8.status
FLEXKV=0 \
EXP_NAME=src-kg8-mt20 \
MODEL_PATH=/models/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
TRAIN_FILE=/workspace/data/uni_agent/swe_rebench_known_good_8.parquet \
TASK_CONFIG=/workspace/phase2/task_config_swe_kg8.yaml \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=36864 \
TRAIN_BATCH_SIZE=8 N_RESP_PER_PROMPT=8 MINI_BATCH_SIZE=8 \
TOTAL_TRAINING_STEPS=1 CONCURRENCY=64 GATEWAY_COUNT=8 \
GEN_TP=2 NGPUS_PER_NODE=2 \
KV_CACHE_MEMORY_BYTES=25769803776 \
bash /workspace/dynres/verl-recipe/dynamo/run_r2c64_flexkv_pair.sh \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  > /workspace/phase2/srcKG8.log 2>&1
rc=$?
echo "[srcKG8] $(date -u +%FT%TZ) EXIT rc=$rc" >> /workspace/phase2/srcKG8.status
