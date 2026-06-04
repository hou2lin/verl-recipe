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
from collections.abc import Generator
from typing import Any, Optional

import ray
import torch

_logger = logging.getLogger(__name__)

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightSender,
)
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
        """
        if self.rollout_rank != 0:
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

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs,
    ):
        """Push trainer-side weights through the control sidecar into each
        dynamo.vllm subprocess's AsyncLLM, then via ZMQ IPC into each TP
        worker's BucketedWeightReceiver.

        Mirrors ``vLLMServerAdapter.update_weights`` but routes the
        "start receiving" RPC through Dynamo's control sidecar
        (``DynamoHttpServer.collective_rpc`` → ZMQ REQ → control listener
        → ``engine.collective_rpc("update_weights_from_ipc", ...)``)
        instead of a direct Ray actor call. The bucket transfer itself
        (``BucketedWeightSender.async_send_weights``) is identical to the
        vLLM path and uses ``self.zmq_handle`` inherited from the parent
        ServerAdapter, whose IPC socket path matches what
        ``vLLMDynamoColocateWorkerExtension._get_zmq_handle`` listens on.

        Replaces the v1 defensive no-op which was a known silent-failure
        bug under skip_refit=False (NCCL broadcast happened but weights
        were discarded — engine kept step:0 weights forever). The new path
        is gated by skip_refit at the CheckpointEngineManager level so
        skip_refit=True still bypasses this method entirely.

        Pre-requisites verified at boot by
        ``DynamoHttpServer._self_test_refit_path``.
        """
        start_time = time.perf_counter()

        # 1. Tell each AsyncLLM to start its BucketedWeightReceiver. Routes
        #    through DynamoHttpServer.collective_rpc → ZMQ control sidecar
        #    → engine.collective_rpc → each TP worker's
        #    vLLMDynamoColocateWorkerExtension.update_weights_from_ipc.
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        # 2. Push bucketed weights over ZMQ IPC. self.zmq_handle is the
        #    socket path computed in the parent vLLMServerAdapter.__init__
        #    (ipc:///tmp/rl-colocate-zmq-replica-{R}-rank-{local}.sock).
        #    The receiver-side path is computed in
        #    dynamo_worker_extension._get_zmq_handle and produces the same
        #    string via VERL_DYNAMO_RANK_OFFSET.
        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(weights)

        # 3. Wait for the engine receivers (all TP workers) to finish
        #    model.load_weights on the buckets we just sent.
        if future is not None:
            await future

        # 4. Reset prefix cache — refit invalidates all KV blocks. Only
        #    rollout_rank 0 needs to issue this (collective_rpc broadcasts).
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            elapsed = time.perf_counter() - start_time
            _logger.info(
                "[dynamo_rollout] update_weights elapsed=%.2fs global_steps=%s",
                elapsed,
                global_steps,
            )


__all__ = ["ServerAdapter"]
