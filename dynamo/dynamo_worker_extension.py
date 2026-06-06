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
"""vLLM worker_extension_cls for the dynamo backend.

The base ``vLLMColocateWorkerExtension._get_zmq_handle`` (verl/workers/rollout
/vllm_rollout/utils.py:266-273) uses ``self.local_rank``, which is the rank of
the worker *within its TP group*. In the dynamo Route-B topology each DP shard
is a separate ``dynamo.vllm`` subprocess, so two DP shards' TP rank 0 would
both compute ``self.local_rank == 0`` and connect to the same IPC socket file.

Fix: read ``VERL_DYNAMO_RANK_OFFSET`` from env (set by DynamoHttpServer when
spawning each DP shard) and add it to ``self.local_rank`` so the IPC handle
encodes a node-global rank that matches what the trainer side computes
(``rollout_rank % local_world_size`` in vllm_rollout.py:104).

See dynamo_design_0507.md §11.3 for the full rank mapping table.
"""

from __future__ import annotations

import os

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

_RANK_OFFSET_ENV = "VERL_DYNAMO_RANK_OFFSET"


class vLLMDynamoColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker mixin for verl × dynamo.

    Adds two responsibilities on top of ``vLLMColocateWorkerExtension``:
    1. ``_get_zmq_handle`` uses node-local-global rank (was per-shard TP-local).
    2. NCCL-based per-tensor weight refit methods (``init_weight_update_group``,
       ``update_weight``, ``destroy_weight_update_group``) — see v3 refit design
       in reports/verl_dynamo_refit_iter_log_zh.md "Iter 4". These mirror the
       vLLM 0.7.0 RLHF example pattern and avoid the CUDA-IPC wake_up issue
       that hung Iter 3.
    """

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        offset = int(os.environ.get(_RANK_OFFSET_ENV, "0"))
        global_rank = self.local_rank + offset
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{global_rank}.sock"

    # --------------------------------------------------------------------- #
    # v3 refit: per-tensor NCCL broadcast (replacing v2 CUDA-IPC path)
    # --------------------------------------------------------------------- #

    def init_weight_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        """Join the refit NCCL group via StatelessProcessGroup + PyNcclCommunicator.

        Called via ``engine.collective_rpc("init_weight_update_group", ...)``
        from the broadcaster (DynamoRollout.update_weights on rank-0 rollout
        actor). Each TP worker computes its own group rank from its node-
        global rank (local_rank + VERL_DYNAMO_RANK_OFFSET) plus rank_offset.

        vLLM 0.18 split the old ``stateless_init_process_group`` helper into
        two primitives: ``StatelessProcessGroup`` (metadata-only TCP store)
        and ``PyNcclCommunicator`` (data-plane NCCL). Both must be combined.
        See https://docs.vllm.ai/en/v0.8.4/getting_started/examples/rlhf_utils.html

        Args:
            master_address: TCP host of the rendezvous master (rank 0).
            master_port: TCP port of the rendezvous master.
            rank_offset: Offset into the group (typically 1, leaving rank 0
                for the broadcaster).
            world_size: Total members in the group (1 + n_engine_workers).
        """
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from vllm.distributed.utils import StatelessProcessGroup

        offset = int(os.environ.get(_RANK_OFFSET_ENV, "0"))
        global_rank = self.local_rank + offset
        my_group_rank = rank_offset + global_rank
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=my_group_rank,
            world_size=world_size,
        )
        self._model_update_group = PyNcclCommunicator(pg, device=self.device)

    def update_weight(self, name: str, dtype, shape) -> None:
        """Receive one weight tensor via NCCL broadcast and load into model.

        Engine-side handler for a per-tensor refit step. Allocates a fresh
        buffer of (dtype, shape), receives the broadcast from rank 0, calls
        the model's standard load_weights to write into the live parameter,
        then frees the receive buffer. Peak extra memory ≈ one tensor.

        Args:
            name: Parameter name (matches model.named_parameters()).
            dtype: torch.dtype or its string form (e.g. "bfloat16").
            shape: Iterable of int — tensor shape.
        """
        import torch

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        weight = torch.empty(tuple(shape), dtype=dtype, device=self.device)
        self._model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def destroy_weight_update_group(self) -> None:
        """Best-effort teardown of the refit NCCL group.

        StatelessProcessGroup has no explicit destroy API; we just drop our
        reference and let garbage collection / process exit handle the rest.
        Idempotent — safe to call even if the group was never initialized.
        """
        if hasattr(self, "_model_update_group"):
            try:
                del self._model_update_group
            except Exception:
                pass


__all__ = ["vLLMDynamoColocateWorkerExtension"]
