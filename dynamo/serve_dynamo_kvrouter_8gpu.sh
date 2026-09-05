#!/bin/bash
# Serving-only verl-dynamo(kv-router)+vllm+flexkv stack for frozen replay.
# FLEXKV_ARM=1|0. 2 x TP2 workers (kv-router has 2 routing targets), frontend
# --router-mode kv. No control ZMQ = pure serving (no weight-update/sleep).
set -u
FLEXKV_ARM=${FLEXKV_ARM:-1}
source /workspace/venvs/dynres-main/bin/activate
export PATH=/workspace/bin:$PATH
export PYTHONPATH=/workspace/dynres/pkgroot:${PYTHONPATH:-}
MODEL=/models/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
SERVED=qwen3-32b
NS=r2replay
FRONT_PORT=8500
IP=$(hostname -i | awk '{print $1}')
GATE=${GATE:-8589934592}   # 8 GiB/rank (proven eviction point)

# cleanup
pkill -9 -f "_dynamo_vllm_with_contro[l]" 2>/dev/null
pkill -9 -f "dynamo.fronten[d]" 2>/dev/null
pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null
pkill -9 -f "multiprocessing.spaw[n]" 2>/dev/null
pkill -9 -f "etcd --data-di[r]" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
rm -f /tmp/flexkv_server* ; rm -rf /tmp/etcd-r2 /tmp/nvidia-mps
sleep 3

# infra
setsid etcd --data-dir /tmp/etcd-r2 --listen-client-urls http://0.0.0.0:2579 \
  --advertise-client-urls http://$IP:2579 --listen-peer-urls http://0.0.0.0:2580 \
  --initial-advertise-peer-urls http://$IP:2580 --initial-cluster default=http://$IP:2580 \
  >/workspace/phase2/g8_r2_etcd.log 2>&1 &
setsid nats-server -js -p 4522 >/workspace/phase2/g8_r2_nats.log 2>&1 &
sleep 4
export ETCD_ENDPOINTS="http://$IP:2579" NATS_SERVER="nats://$IP:4522"
export DYN_NAMESPACE=$NS DYN_DISCOVERY_BACKEND=etcd DYN_ENABLE_RL=true
export DYN_SDK_DISABLE_ANSI_LOGGING=1

flexkv_cfg=()
if [ "$FLEXKV_ARM" = "1" ]; then
  export DYNAMO_USE_FLEXKV=1 FLEXKV_ENABLE_MPS=0 FLEXKV_CPU_CACHE_GB=96 FLEXKV_INIT_READY_TIMEOUT_S=30 FLEXKV_ENABLE_METRICS=1 FLEXKV_NUM_LOG_INTERVAL_REQUESTS=100
  flexkv_cfg=(--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}')
fi

start_worker() {
  local idx=$1 cvd=$2 vport=$3 kvport=$4
  local wenv=()
  CUDA_VISIBLE_DEVICES=$cvd VLLM_PORT=$vport VLLM_HOST_IP=$IP MASTER_ADDR=$IP MASTER_PORT=$vport \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_SKIP_P2P_CHECK=1 PYTHONHASHSEED=0 \
  FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_g${cvd//,/-}" \
  setsid python3 -m recipe.dynamo._dynamo_vllm_with_control \
    --model $MODEL --served-model-name $SERVED --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 --max-model-len 40960 --max-num-batched-tokens 32768 \
    --max-num-seqs 64 --dtype bfloat16 --enable-chunked-prefill --enable-prefix-caching \
    --distributed-executor-backend mp \
    --kv-events-config "{\"publisher\": \"zmq\", \"topic\": \"kv-events\", \"endpoint\": \"tcp://*:$kvport\", \"enable_kv_cache_events\": true}" \
    "${flexkv_cfg[@]}" \
    --kv-cache-memory-bytes $GATE \
    >/workspace/phase2/g8_r2_worker${idx}.log 2>&1 &
}

echo "[serve] ARM=$FLEXKV_ARM starting 2x TP2 workers + kv-router frontend on :$FRONT_PORT" >> /workspace/phase2/g8_serve.status
start_worker 0 "0,1" 20100 40100
start_worker 1 "2,3" 20200 40200
start_worker 2 "4,5" 20300 40300
start_worker 3 "6,7" 20400 40400
sleep 5
# frontend kv router
setsid python3 -m dynamo.frontend --http-port $FRONT_PORT --http-host 0.0.0.0 \
  --router-mode kv --discovery-backend etcd --namespace-prefix $NS \
  >/workspace/phase2/g8_r2_frontend.log 2>&1 &

# wait for both workers registered + frontend serving
for i in $(seq 1 480); do
  if curl -sf http://localhost:$FRONT_PORT/v1/models 2>/dev/null | grep -q "$SERVED"; then
    echo "[serve] ARM=$FLEXKV_ARM frontend serving after ~$((i*5))s" >> /workspace/phase2/g8_serve.status
    break
  fi
  sleep 5
done
echo "SERVE_READY ARM=$FLEXKV_ARM" 
