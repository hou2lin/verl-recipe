#!/usr/bin/env bash
set -xeuo pipefail

# Gate R2-C64: live-rollout FlexKV on/off pair on the new stack.
# Reproduces the strongest arm of the old-stack results (C64, 32G GPU KV/rank;
# live path split 89.5/5.33/5.17 GPU-hit/CPU-fetch/recompute) with ONE
# variable: FLEXKV=0|1. Router held equal across arms (round-robin, matching
# the old C64 arm).
#
# Stack: verl main (zero changes) v1 colocate_async trainer + FSDP2 +
# recipe dynamo rollout backend + uni-agent agent framework (new task-config
# schema via the sanctioned agent_loop_manager_class injection point).
# Workload: uni-agent ReAct on SWE-Bench Verified, docker sandboxes against
# the pod's dind sidecar.
#
# Prereqs on the pod:
#   - uni-agent checked out at ${UNIAGENT_ROOT} (PYTHONPATH injection)
#   - docker CLI in PATH + DOCKER_HOST=tcp://localhost:2375 (dind sidecar)
#   - swe_bench_verified.parquet prepared (uni_agent.tasks.swe_bench.preprocess)
#   - smokeF (colocate sleep/wake x FlexKV) green before 32B arms

FLEXKV=${FLEXKV:-1}
project_name=${PROJECT_NAME:-verl-r2c64}
exp_name=${EXP_NAME:-r2c64-flexkv${FLEXKV}}

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-36864}   # 4096+36864 = 40960 (slide max_model_len)
train_prompt_bsz=${TRAIN_BATCH_SIZE:-8}             # 8 prompts x 8 rollouts = 64 trajectories/step
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${MINI_BATCH_SIZE:-8}
total_training_steps=${TOTAL_TRAINING_STEPS:-3}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}

# C{n} of the old slide = max in-flight rollout sessions (runner cap).
CONCURRENCY=${CONCURRENCY:-64}
GATEWAY_COUNT=${GATEWAY_COUNT:-8}
TOOL_PARSER=${TOOL_PARSER:-hermes}
# 32 GiB GPU KV per rank (slide C64-arm capacity gate).
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-34359738368}
GEN_TP=${GEN_TP:-2}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:?set to the Qwen3-32B(-FP8) snapshot path}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
TRAIN_FILE=${TRAIN_FILE:-/workspace/data/uni_agent/swe_bench_verified.parquet}
TEST_FILE=${TEST_FILE:-${TRAIN_FILE}}
TASK_CONFIG=${TASK_CONFIG:?set to task_config_swe_c64.yaml path}
UNIAGENT_ROOT=${UNIAGENT_ROOT:-/workspace/dynres/uni-agent}
CKPTS_DIR=${CKPTS_DIR:-/workspace/ckpts/${project_name}/${exp_name}}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-/workspace/logs/${project_name}/${exp_name}}

export PYTHONPATH="${UNIAGENT_ROOT}:${PYTHONPATH:-}"
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

flexkv_args=()
if [[ "${FLEXKV}" == "1" ]]; then
    flexkv_args=(++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_flexkv=True)
    export FLEXKV_CPU_CACHE_GB=${FLEXKV_CPU_CACHE_GB:-96}   # per shard (private pools)
    export FLEXKV_INIT_READY_TIMEOUT_S=${FLEXKV_INIT_READY_TIMEOUT_S:-30}
fi

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches="${num_warmup_batches}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=round-robin \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=[--kv-cache-memory-bytes,${KV_CACHE_MEMORY_BYTES}]" \
    "${flexkv_args[@]}" \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
