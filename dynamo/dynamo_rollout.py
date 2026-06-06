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
        """v3 — per-tensor NCCL P2P refit (replaces v2 CUDA-IPC path).

        Motivation: v2 used ``update_weights_from_ipc`` on the engine side,
        which implicitly wakes the 'weights' tag and allocates the full
        ~63 GiB weight buffer. Concurrent with the trainer's FSDP
        ``get_per_tensor_param`` all_gather (also ~63 GiB), this pushed
        single-GPU memory over the edge and NCCL hung (see Iter 3 in
        reports/verl_dynamo_refit_iter_log_zh.md).

        v3 mirrors the miles team's design and the vLLM 0.7.0 RLHF
        example: rollout rank-0 acts as broadcaster, dynamo.vllm worker
        subprocesses join an NCCL group via ``init_weight_update_group``,
        and each parameter is broadcast individually via
        ``update_weight(name, dtype, shape)``. Peak engine-side memory is
        ~one tensor (MiB–GiB scale), not the full ~63 GiB.

        Non-rank-0 rollout actors drain the incoming weights generator
        (delivered to them by the verl framework's NCCL backend) and
        return — only rank 0 broadcasts to the engine subprocesses.
        """
        # Non-rank-0 just drain — only rank 0 orchestrates the refit.
        # `weights` is a sync generator (Generator[(str, Tensor), None, None])
        # delivered by the verl framework's checkpoint_engine backend, not an
        # async one; use sync `for` not `async for`.
        if self.rollout_rank != 0:
            for _ in weights:
                pass
            return

        import socket

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        start_time = time.perf_counter()

        # Ensure server_handle is cached for collective_rpc dispatch.
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())

        # 0. Wake up the 'weights' tag BEFORE NCCL init.
        #    diag B v3/v4 confirmed: PyNcclCommunicator init fails with
        #    "NCCL invalid usage" on a fully-sleeping engine; this is
        #    also why vLLM's upstream RLHF example (0.7.0 / 0.8.4)
        #    always calls wake_up before init_weight_update_group.
        #    Diag B v5 confirmed load_weights works under wake_up(['weights']).
        #    Trade-off: this re-allocates ~weight_size on engine GPU,
        #    concurrent with trainer FSDP all_gather → revisits Iter 3
        #    memory pressure. Per-tensor refit (vs v2 bulk IPC) keeps
        #    engine buffer peak small (~MiB-GiB per param).
        await self.server_handle.wake_up.remote(tags=["weights"])

        # 1. Pick rendezvous master for the refit NCCL group.
        master_address = socket.gethostbyname(socket.gethostname())
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        master_port = s.getsockname()[1]
        s.close()

        # 2. Compute world_size: 1 broadcaster (us) + N engine workers
        #    (sum of TP workers across all dynamo.vllm shards on this node).
        n_engine_workers = await self.server_handle.get_num_engine_workers.remote()
        world_size = 1 + int(n_engine_workers)

        # 3. Tell engine workers to join the group (rank_offset=1 leaves
        #    rank 0 for us). Non-blocking — we'll await after we also join.
        init_future = await self._execute_method(
            "init_weight_update_group",
            non_block=True,
            kwargs={
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
            },
        )

        # 4. We (broadcaster) join the same group on rank 0. Pick GPU 0
        #    on this node — DynamoRollout's Ray actor process has access
        #    to all node GPUs but only needs one for broadcast.
        #    vLLM 0.18 split stateless_init_process_group into
        #    StatelessProcessGroup (metadata TCP store) + PyNcclCommunicator
        #    (data-plane NCCL); combine them ourselves.
        device = torch.device(f"cuda:{self.rollout_rank % 8}")
        broadcaster_pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size,
        )
        trainer_group = PyNcclCommunicator(broadcaster_pg, device=device)

        if init_future is not None:
            await init_future

        # 5. Per-tensor refit: signal engine workers, then broadcast.
        #    The incoming `weights` generator yields (name, tensor) tuples
        #    delivered by the verl framework's checkpoint_engine backend
        #    (typically NCCL between trainer ranks 0-7 and rollout-side
        #    actors 0-7). We just re-broadcast each tensor on OUR group
        #    to the engine subprocesses.
        n_tensors = 0
        try:
            for name, tensor in weights:
                # Move to GPU if needed (broadcast requires CUDA tensor).
                if tensor.device.type != "cuda":
                    tensor = tensor.to(device, non_blocking=True)
                elif tensor.device != device:
                    tensor = tensor.to(device, non_blocking=True)

                # Signal each engine worker to allocate buffer and recv
                # via NCCL broadcast. Non-blocking so we can broadcast
                # in parallel; we'll await the engine-side load_weights
                # below.
                signal_future = await self._execute_method(
                    "update_weight",
                    non_block=True,
                    kwargs={
                        "name": name,
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                        "shape": list(tensor.shape),
                    },
                )

                # Broadcast on our group. Engine workers recv into their
                # newly-allocated buffer, then call model.load_weights.
                trainer_group.broadcast(tensor, src=0)

                if signal_future is not None:
                    await signal_future

                n_tensors += 1
        finally:
            # 6. Cleanup — best-effort; failure here doesn't void the refit.
            try:
                await self._execute_method("destroy_weight_update_group")
            except Exception as e:
                _logger.warning(
                    "[dynamo_rollout v3] destroy_weight_update_group failed: %s", e
                )

        # 7. Reset prefix cache — refit invalidates all KV blocks.
        await self.server_handle.clear_kv_cache.remote()
        if global_steps is not None:
            await self.server_handle.set_global_steps.remote(global_steps)

        elapsed = time.perf_counter() - start_time
        _logger.info(
            "[dynamo_rollout v3 NCCL] update_weights n_tensors=%d elapsed=%.2fs "
            "global_steps=%s world_size=%d",
            n_tensors,
            elapsed,
            global_steps,
            world_size,
        )


__all__ = ["ServerAdapter"]
