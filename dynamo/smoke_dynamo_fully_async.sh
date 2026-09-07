#!/usr/bin/env bash
set -xeuo pipefail

# Two-GPU fully_async smoke for the Dynamo rollout path (A档: standalone-only).
#
# Split placement on one node: trainer = 1 GPU (FullyAsyncTrainer, no hybrid
# pool), standalone rollout = 1 GPU (CheckpointEngineWorker + dynamo stack).
# Exercises per parameter sync (every trigger_parameter_sync_step):
#   abort_all_requests (pause + partial-rollout aborted-empty replies)
#   -> release_kv -> nccl first hop -> CUDA-IPC second hop
#   -> resume_kv -> resume_generation (client retries continue trajectories)
# plus staleness accounting via the TokenOutput global_steps tag, the
# probe_logprob_channel startup gate, and manager.shutdown() teardown.
#
# Requires: verl >= REQUIRED_VERL.txt pin, cupy-cuda12x (the nccl
# checkpoint-engine backend registers only when cupy imports), recipe
# mounted as recipe/dynamo under the verl repo root.

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-fully-async-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-1}
# total_rollout_steps counts SAMPLES; train steps come out to
# total_rollout_steps / (ppo_mini_batch_size * require_batches * sync_step).
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-4}
TRIGGER_SYNC_STEP=${TRIGGER_SYNC_STEP:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
# Per-POOL concurrency (the LB sees one dynamo server per pool); 16 per
# engine shard — the smoke pool has a single TP=1 shard.
CONCURRENT_SAMPLES=${CONCURRENT_SAMPLES:-16}
# round-robin | kv | thunderagent (TA = capacity-credit admission routing)
ROUTER_MODE=${ROUTER_MODE:-round-robin}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# FLEXKV=1 turns this into the FlexKV L2 acceptance smoke (same 1+1 layout).
FLEXKV=${FLEXKV:-0}
flexkv_args=()
if [ "$FLEXKV" = "1" ]; then
  export FLEXKV_CPU_CACHE_GB=${FLEXKV_CPU_CACHE_GB:-16} FLEXKV_ENABLE_MPS=${FLEXKV_ENABLE_MPS:-0}
  flexkv_args=(++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_flexkv=True)
fi
# cold-start (first model read + compile) can exceed the 600s default window
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-2400}
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

# bypass_mode=true (yaml default): rollout logprobs feed training directly,
# the upstream fully_async recommendation. fsdp2 matches the upstream
# fully_async example scripts.
#
# NOTE: the strategy MUST be set at the role level (actor.strategy=fsdp2).
# FSDPActorConfig.__post_init__ copies actor.strategy over engine.strategy,
# so a bare actor.fsdp_config.strategy=fsdp2 is silently clobbered back to
# fsdp1 — where every offload knob (offload_policy/param_offload/
# optimizer_offload) is a no-op for training and large models OOM in
# update_actor.
python3 -m recipe.dynamo.main_dynamo_fully_async \
    algorithm.adv_estimator=grpo \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    +critic.model.override_config.attn_implementation=sdpa \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.val_batch_size=1 \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.n=2 \
    rollout.nnodes="${ROLLOUT_NNODES}" \
    rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.concurrent_samples_per_replica="${CONCURRENT_SAMPLES}" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode="${ROUTER_MODE}" \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.total_epochs=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "${flexkv_args[@]}" \
    "$@"

echo "PASS: Dynamo fully_async smoke completed"
