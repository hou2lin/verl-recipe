#!/bin/bash
# kg8 long-trajectory two-arm paced frozen replay @19GiB gate.
# Prereq: /workspace/phase2/kg8_traj.jsonl (dump_trajectories traj mode).
set -u
source /workspace/venvs/dynres-main/bin/activate
export PATH=/workspace/bin:$PATH
export PYTHONPATH=/workspace/dynres/pkgroot:${PYTHONPATH:-}
JSONL=${JSONL:-/workspace/phase2/kg8_traj.jsonl}
GATE_VAL=${GATE_VAL:-34359738368}   # 19 GiB/rank
ST=/workspace/phase2/kg8c64.status
EV=/workspace/phase2/kvrouter_evidence
mkdir -p $EV
echo "[kg8c64] $(date -u +%FT%TZ) START jsonl=$JSONL gate=$GATE_VAL" >> $ST
for ARM in 1 0; do
  echo "[kg8c64] $(date -u +%FT%TZ) serving arm$ARM" >> $ST
  FLEXKV_ARM=$ARM GATE=$GATE_VAL bash /workspace/phase2/serve_dynamo_kvrouter.sh \
    > /workspace/phase2/serve_kg8_arm$ARM.out 2>&1
  if ! curl -sf http://localhost:8500/v1/models | grep -q qwen3-32b; then
    echo "[kg8c64] $(date -u +%FT%TZ) ARM$ARM SERVE_FAILED" >> $ST
    exit 1
  fi
  echo "[kg8c64] $(date -u +%FT%TZ) arm$ARM ready, paced replay" >> $ST
  python3 /workspace/dynres/pkgroot/recipe/dynamo/frozen_replay.py \
    --endpoint http://localhost:8500 --model qwen3-32b \
    --trajectories $JSONL --concurrency 64 --warm-rounds 1 --steady-rounds 4 \
    --paced --think-time 2.0 --converge-pct 5 \
    --out /workspace/phase2/kg8_arm${ARM}_c64_32g.json \
    > /workspace/phase2/kg8_replay_arm$ARM.log 2>&1
  rc=$?
  {
    echo "== arm$ARM path stats =="
    echo -n "GET_cancelled="; grep -a "operation=get" /workspace/phase2/r2_worker*.log 2>/dev/null | grep -ac "cancel" || echo 0
    echo -n "H2D_transfers="; grep -a "direction=H2D" /workspace/phase2/r2_worker*.log 2>/dev/null | grep -ac "complete" || echo 0
    echo -n "router_kv_decisions="; grep -ac "router_mode=kv" /workspace/phase2/r2_frontend.log 2>/dev/null || echo 0
    grep -aoE "overlap_blocks=[0-9]+" /workspace/phase2/r2_frontend.log 2>/dev/null | sort -t= -k2 -n | tail -1
    grep -aE "Prefix cache|prefix_cache" /workspace/phase2/r2_worker0.log 2>/dev/null | tail -2
  } > /workspace/phase2/kg8_arm${ARM}_stats.txt 2>&1
  for f in r2_worker0 r2_worker1 r2_frontend; do
    cp /workspace/phase2/$f.log $EV/kg8_arm${ARM}_$f.log 2>/dev/null
  done
  echo "[kg8c64] $(date -u +%FT%TZ) arm$ARM replay rc=$rc" >> $ST
done
# teardown
pkill -9 -f "_dynamo_vllm_with_contro[l]" 2>/dev/null; pkill -9 -f "dynamo.fronten[d]" 2>/dev/null
pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null; pkill -9 -f "multiprocessing.spaw[n]" 2>/dev/null
pkill -9 -f "etcd --data-di[r]" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
rm -f /tmp/flexkv_server*
echo "[kg8c64] $(date -u +%FT%TZ) EXIT_OK" >> $ST
