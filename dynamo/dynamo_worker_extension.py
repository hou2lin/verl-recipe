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

    # v4a-5 (Iter 7.4) diagnostic: confirm update_weights_from_ipc actually
    # fires on engine workers. V-1.c grep has shown 0 lines for the past 4
    # iters, suggesting this method never runs. This override adds a print
    # before delegating to the inherited implementation.
    def update_weights_from_ipc(self, *args, **kwargs):
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        offset = int(os.environ.get(_RANK_OFFSET_ENV, "0"))
        try:
            global_rank = self.local_rank + offset
        except Exception:
            global_rank = "?"
        print(
            f"[v4a-5][worker.update_weights_from_ipc] FIRING "
            f"replica={replica_rank} local_rank={self.local_rank} "
            f"offset={offset} global_rank={global_rank} "
            f"zmq={self._get_zmq_handle()}",
            flush=True,
        )
        return super().update_weights_from_ipc(*args, **kwargs)

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

    # --------------------------------------------------------------------- #
    # SPIKE (v4 path): join external NCCL group via torch.distributed.
    # Validates the hypothesis that PyTorch's ProcessGroupNCCL handles
    # same-GPU 2-rank correctly (the case that broke v3's raw
    # PyNcclCommunicator). If this works, the next step is to skip our
    # rollout-side actor entirely and let trainer broadcast directly into
    # the engine subprocess workers — see
    # reports/verl_dynamo_refit_findings_zh.md §6 path J/K.
    # --------------------------------------------------------------------- #

    def init_external_process_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        rank: int | None = None,
        rank_offset: int | None = None,
        group_name: str = "refit_external",
        timeout_seconds: int = 300,
    ) -> dict:
        """SPIKE: join an external NCCL group, separate from vLLM TP group.

        Uses ``torch.distributed.ProcessGroupNCCL`` (PyTorch's NCCL wrapper)
        with a TCPStore-based rendezvous. Crucially this does NOT touch the
        default global process group that vLLM TP already initialized — we
        construct a fresh ProcessGroup object.

        Why this should work where v3 failed:
            v3 used ``vllm.distributed.PyNcclCommunicator`` directly, which
            wraps raw NCCL with minimal coordination. On the colocated
            8-GPU H200 layout, the broadcaster (rollout Ray actor on cuda:0)
            and the dynamo engine worker (also on cuda:0) ended up as two
            ranks in one NCCL communicator on the same GPU — NCCL reports
            "invalid usage" in that case (confirmed in diag B v3, see
            reports/verl_dynamo_refit_iter_log_zh.md).

            verl framework's NCCL backend, by contrast, uses
            ``ray.util.collective`` (cupy-based wrapper with stream
            coordination) and B v4 confirmed it works in the same colocated
            same-GPU layout. ``torch.distributed.ProcessGroupNCCL`` provides
            the same kind of PyTorch-level coordination layer. So this spike
            tests whether the PyTorch wrapper, not the raw NCCL primitive,
            is what enables same-GPU 2-rank to work.

        Args:
            master_address: TCP host of the rendezvous master (rank 0).
            master_port: TCP port of the rendezvous master.
            world_size: Total members in the group.
            rank: Explicit rank for this worker. Mutually exclusive with
                rank_offset. Use when caller knows the exact rank.
            rank_offset: If provided, this worker's rank in the new group
                is computed as ``rank_offset + self.rank`` (vLLM TP rank).
                Required when calling via ``engine.collective_rpc`` because
                that broadcasts the same kwargs to every TP worker — each
                worker must derive its own rank from its TP rank.
            group_name: Label for logging only.
            timeout_seconds: Rendezvous + collective op timeout.

        Returns:
            dict with ``ok`` plus diagnostic fields (or ``error`` + traceback
            on failure). We catch all exceptions so the worker doesn't die —
            we want clean signal back, not a crashed worker.
        """
        import datetime
        import time as _time
        import traceback

        import torch
        import torch.distributed as dist

        t0 = _time.perf_counter()
        try:
            # Resolve rank: explicit rank wins; otherwise rank_offset+self.rank.
            if rank is None:
                if rank_offset is None:
                    return {
                        "ok": False,
                        "error": "ValueError: pass either rank= or rank_offset=",
                    }
                rank = rank_offset + self.rank

            # Pin CUDA device before NCCL init — vLLM workers typically have
            # this set already, but being explicit avoids surprises if this
            # method is called on a "fresh" RPC stack.
            torch.cuda.set_device(self.device)

            timeout = datetime.timedelta(seconds=timeout_seconds)
            is_master = rank == 0

            # TCPStore is the rendezvous primitive. is_master means this
            # process binds the listening socket; others connect.
            store = dist.TCPStore(
                host_name=master_address,
                port=int(master_port),
                world_size=world_size,
                is_master=is_master,
                timeout=timeout,
            )

            # Construct ProcessGroupNCCL directly. This does NOT touch the
            # default global group used by vLLM TP — we hold our own
            # ProcessGroup object, used via group= kwarg in collectives.
            opts = dist.ProcessGroupNCCL.Options()
            opts._timeout = timeout
            pg = dist.ProcessGroupNCCL(store, rank, world_size, opts)

            # Keep references so they don't get GC'd between RPCs.
            self._refit_pg = pg
            self._refit_store = store
            self._refit_rank = rank
            self._refit_world_size = world_size

            elapsed_ms = (_time.perf_counter() - t0) * 1000
            return {
                "ok": True,
                "device": str(self.device),
                "rank": rank,
                "tp_rank": self.rank,
                "world_size": world_size,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "elapsed_ms": (_time.perf_counter() - t0) * 1000,
            }

    def receive_and_load_weights_test(
        self,
        name: str,
        dtype_str: str,
        shape,
        do_load_weights: bool = True,
    ) -> dict:
        """SPIKE: receive a broadcast tensor on the refit group and load it.

        Must be called after ``init_external_process_group``. Allocates a
        buffer of (dtype, shape), receives a broadcast from rank 0 of the
        refit group, then optionally calls ``model.load_weights`` to commit
        the received tensor into the live parameter.

        Args:
            name: Parameter name (matches ``model.named_parameters()``).
            dtype_str: Tensor dtype, e.g. "bfloat16" or "torch.bfloat16".
            shape: Iterable of int — tensor shape.
            do_load_weights: If False, skip the model.load_weights step and
                just measure recv timing. Useful for isolating "did the
                broadcast work" from "did load_weights work" — diag B v5
                already proved load_weights works under sleep+wake, so the
                spike's first concern is the NCCL broadcast step.

        Returns:
            dict with ``ok`` + timing fields, or ``error`` + traceback.
        """
        import time as _time
        import traceback

        import torch
        import torch.distributed as dist

        t0 = _time.perf_counter()
        try:
            if not hasattr(self, "_refit_pg"):
                return {
                    "ok": False,
                    "error": "RuntimeError: external process group not initialized — call init_external_process_group first",
                }

            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device)

            # PyTorch 2.10 `dist.broadcast(tensor, src=0, group=pg)` requires
            # `pg` to be registered via `dist.new_group()`. But we hold a raw
            # ProcessGroupNCCL (constructed directly) that's NOT registered —
            # registering would require touching the default global group,
            # which vLLM TP already owns. So call the ProcessGroup primitive
            # directly, bypassing the dist module's registration check.
            opts = dist.BroadcastOptions()
            opts.rootRank = 0
            work = self._refit_pg.broadcast([tensor], opts)
            work.wait()
            torch.cuda.synchronize()

            recv_elapsed_ms = (_time.perf_counter() - t0) * 1000

            load_elapsed_ms = None
            if do_load_weights:
                t1 = _time.perf_counter()
                self.model_runner.model.load_weights(weights=[(name, tensor)])
                load_elapsed_ms = (_time.perf_counter() - t1) * 1000

            del tensor

            return {
                "ok": True,
                "recv_elapsed_ms": recv_elapsed_ms,
                "load_elapsed_ms": load_elapsed_ms,
                "tensor_bytes": int(
                    torch.tensor(shape).prod().item()
                ) * (torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 8),
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "elapsed_ms": (_time.perf_counter() - t0) * 1000,
            }

    def destroy_external_process_group(self) -> dict:
        """SPIKE: best-effort teardown of the refit external NCCL group.

        Tears down both the ProcessGroupNCCL and the TCPStore so a follow-up
        call to init_external_process_group can rebuild from scratch.
        Idempotent — safe to call without prior init.
        """
        import traceback

        result = {"ok": True, "freed": []}
        if hasattr(self, "_refit_pg"):
            try:
                # ProcessGroupNCCL has _shutdown / shutdown depending on
                # version; try both, swallow attribute errors.
                pg = self._refit_pg
                if hasattr(pg, "shutdown"):
                    pg.shutdown()
                elif hasattr(pg, "_shutdown"):
                    pg._shutdown()
                del self._refit_pg
                result["freed"].append("pg")
            except Exception as e:
                result["pg_error"] = f"{type(e).__name__}: {e}"
                result["pg_traceback"] = traceback.format_exc()

        if hasattr(self, "_refit_store"):
            try:
                del self._refit_store
                result["freed"].append("store")
            except Exception as e:
                result["store_error"] = f"{type(e).__name__}: {e}"

        for attr in ("_refit_rank", "_refit_world_size"):
            if hasattr(self, attr):
                delattr(self, attr)

        return result

    # --------------------------------------------------------------------- #
    # diag-only: test write-to-model-params under various engine states
    # --------------------------------------------------------------------- #

    def diag_test_load_weights_local(self, name: str, dtype_str: str, shape) -> dict:
        """DIAG: bypass NCCL entirely, test if model.load_weights() works.

        Allocate a tensor locally (no broadcast needed), then call the
        standard model.load_weights path. Isolates the question "does
        load_weights fault on a sleeping engine" from the question "does
        NCCL init work on a sleeping engine".

        Returns a dict with keys ok/error/elapsed_ms for the caller to
        report. Intentionally doesn't propagate exceptions — we want clean
        signal back, not a crashed worker.
        """
        import time as _time

        import torch

        t0 = _time.perf_counter()
        try:
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            weight = torch.zeros(
                tuple(shape), dtype=dtype, device=self.device
            )
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight
            return {
                "ok": True,
                "elapsed_ms": (_time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            import traceback

            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "elapsed_ms": (_time.perf_counter() - t0) * 1000,
            }


__all__ = ["vLLMDynamoColocateWorkerExtension"]
