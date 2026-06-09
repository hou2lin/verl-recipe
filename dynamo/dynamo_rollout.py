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
"""ServerAdapter for the dynamo backend.

Inherits the vLLM ServerAdapter (HTTP path is identical: trainer rank reads
``replica.server_address`` and POSTs chat completions to it) and only
overrides the Ray actor name prefix used for sleep/wake/update_weights RPC,
so it lands on ``dynamo_server_*`` (created by DynamoReplica.launch_servers)
rather than ``vllm_server_*``.
"""

import logging
import time
from typing import Any, Optional

import ray

_logger = logging.getLogger(__name__)

from verl.workers.rollout.vllm_rollout.vllm_rollout import (
    ServerAdapter as _VllmServerAdapter,
)


class ServerAdapter(_VllmServerAdapter):
    """Per-rank dynamo client.

    All HTTP-based generation goes through the frontend URL stored in
    ``RolloutReplica.server_address``; weight-update / wake-up / sleep
    requests go to the per-replica Ray actor named ``dynamo_server_{r}_{n}``.
    """

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"

    def _get_control_actor_name(self) -> str:
        """Return the shared Dynamo server actor name for control RPCs."""
        dynamo_cfg = (self.config.engine_kwargs or {}).get("dynamo", {}) or {}
        shared_replica_rank = int(dynamo_cfg.get("shared_pool_replica_rank", 0))
        return f"{self._get_server_name_prefix()}server_{shared_replica_rank}_{self.node_rank}"

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute control RPCs against the shared Dynamo pool actor.

        Native vLLM has one named server actor per rollout replica. All logical rollout
        replicas on a node share ``dynamo_server_0_<node_rank>``.

        v4a-8 (Iter 7.9): gate on GLOBAL rank (rollout_rank==0 AND
        replica_rank==0), not just rollout_rank==0. The shared actor's
        collective_rpc broadcasts to all 4 sidecars internally, so if
        each replica's rank-0 fires, we get 4×4 duplicate RPCs and
        engine workers hang on 3 spurious update_weights_from_ipc
        invocations (no paired sender). Only the global rank-0 fires.
        Each replica's rank-0 still fires its own BucketedWeightSender
        (paired 1:1 with its engine workers), but only one RPC triggers
        engine.collective_rpc("update_weights_from_ipc") across all
        replicas via the parallel sidecar dispatch.
        """
        if self.rollout_rank != 0 or self.replica_rank != 0:
            return None

        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())

        future = self.server_handle.collective_rpc.remote(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        # Was a hard-coded no-op; that's only valid when skip_refit=True. With
        # skip_refit=False the trainer puts Dynamo to sleep at startup
        # (ray_trainer.sleep_replicas) and never explicitly wakes it, so the
        # first generate request hangs on a sleeping engine. DynamoHttpServer
        # .wake_up already guards skip_refit and falls through to
        # _engine_method_all("wake_up", ...) via the control sidecar.
        if not self.config.free_cache_engine or self.rollout_rank != 0:
            return None
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())
        t0 = time.perf_counter()
        await self.server_handle.wake_up.remote(tags=tags)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _logger.info(
            "[dynamo_rollout] resume tags=%s elapsed_ms=%.2f", tags, elapsed_ms
        )

    async def release(self):
        if not self.config.free_cache_engine or self.rollout_rank != 0:
            return None
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())
        t0 = time.perf_counter()
        await self.server_handle.sleep.remote(level=self.sleep_level)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _logger.info(
            "[dynamo_rollout] release level=%s elapsed_ms=%.2f",
            self.sleep_level,
            elapsed_ms,
        )

    # v4a-3: sleep_level=1 keeps weights resident on GPU through every
    # sleep call (only KV cache is freed). This bypasses the
    # vLLM-tag-tracking bug that made wake_up('weights') a silent no-op
    # in Iter 7.0/7.1: with weights never sleeping, update_weights_from_ipc
    # can write directly into the live weight tensors without needing
    # any wake_up RPC through the Dynamo sidecar.
    #
    # History:
    # - v1 (Sophia original): no-op drain - off-policy false positive (B v4)
    # - v2 (4becf27): IPC path, OOM at Iter 3
    # - v3 (bead1d1): self-built PyTorch NCCL group, blocked by NCCL
    #   "Duplicate GPU detected" same-GPU 2-rank (Iter 4-5, spike 6.x)
    # - v4 (design only): subprocess as direct receiver, blocked by
    #   ray.util.collective non-actor restriction (P1.0 mini-test)
    # - v4a (Iter 7.0): bare inherit, NCCL watchdog hang due to
    #   implicit-wake silent no-op
    # - v4a-2 (Iter 7.1): explicit wake_up before super(), 120s timeout
    #   in Dynamo sidecar (engine_method ZMQ recv)
    # - v4a-3 (Iter 7.2, current): sleep_level=1 to avoid wake entirely.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force sleep_level=1 regardless of VLLM_SLEEP_LEVEL env default.
        # weights stay on GPU through every sleep; only KV is freed.
        self.sleep_level = 1
        _logger.info(
            "[dynamo_rollout v4a-3] forcing sleep_level=1 "
            "(weights resident, avoid wake_up tag-tracking bug)"
        )


__all__ = ["ServerAdapter"]
