# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamo fully_async training entry point.

Thin wrapper around ``verl.experimental.fully_async_policy.fully_async_main``
that only swaps the hydra config name and, before Ray starts, verifies the
DynamoLLMServerManager rebase (design doc §4 A2): fully_async hard-codes
FullyAsyncLLMServerManager, so the recipe injects itself as its base class via
the VERL_USE_EXTERNAL_MODULES patch — an import-order property this entry
asserts rather than assumes. No trainer/rollouter code is duplicated; the
body mirrors upstream ``fully_async_main.main`` line by line.
"""

import os

import hydra

from recipe.dynamo.register import ensure_fully_async_manager_rebase
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="dynamo_trainer_fully_async", version_base=None)
def main(config):
    # The FullyAsyncRollouter actor builds FullyAsyncLLMServerManager inside
    # its OWN process, where only VERL_USE_EXTERNAL_MODULES re-applies the
    # dynamo patches. Without it the run fails deep inside Ray with an
    # unknown-rollout error; check up front instead.
    external_modules = os.environ.get("VERL_USE_EXTERNAL_MODULES", "")
    if "recipe.dynamo.register" not in external_modules:
        raise RuntimeError(
            "fully_async dynamo runs require VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register "
            "to be exported (Ray worker processes must re-run the recipe registration); got "
            f"VERL_USE_EXTERNAL_MODULES={external_modules!r}."
        )
    ensure_fully_async_manager_rebase()

    # Import AFTER the rebase check: pulling in fully_async_main imports
    # fully_async_rollouter, which is exactly the module the check patches.
    from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
    from verl.trainer.main_ppo import run_ppo

    # Below mirrors verl.experimental.fully_async_policy.fully_async_main.main.
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    auto_set_device(config)
    # fully_async keeps the rollout pool topology under top-level rollout.*;
    # upstream copies it into actor_rollout_ref before launching.
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)


if __name__ == "__main__":
    main()
