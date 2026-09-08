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
"""Dynamo-specific AgentLoopManager.

Dynamo exposes one logical rollout endpoint through a master dynamo.frontend.
The worker-level routing happens inside Dynamo's KV router, not in verl's
GlobalRequestLoadBalancer. This module keeps the AgentLoop execution model but
replaces the generic server manager with a direct Dynamo server manager.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import update_prometheus_config

from .thunderagent import current_program
from .thunderagent import program_scope as bind_program

logger = logging.getLogger(__name__)


class DynamoServerManager:
    """Direct manager for the shared Dynamo frontend actor.

    Unlike AsyncLLMServerManager, this class intentionally does not acquire a
    server from GlobalRequestLoadBalancer. Dynamo owns routing behind its
    frontend, so verl should only call the single shared Dynamo actor.
    """

    def __init__(
        self,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        *,
        thunderagent_enabled: bool = False,
    ):
        if len(servers) != 1:
            raise ValueError(f"DynamoServerManager expects exactly one shared server, got {len(servers)}")
        self.server_address, self.server = servers[0]
        self.thunderagent_enabled = thunderagent_enabled

    @asynccontextmanager
    async def program_scope(self):
        """Bind all turns in one agent-loop run to one Dynamo program."""
        if not self.thunderagent_enabled:
            yield
            return
        async with bind_program(uuid4().hex, self._finalize_program):
            yield

    async def _finalize_program(self, session_id: str) -> None:
        await self.server.finalize_program.remote(session_id=session_id)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if audio_data is not None or mm_processor_kwargs:
            raise RuntimeError("Dynamo frontend generate does not support audio inputs or processor options")

        generate_kwargs = dict(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )
        if not self.thunderagent_enabled:
            output = await self.server.generate.remote(**generate_kwargs)
            return self._tag_weight_versions(output)

        scope = current_program()
        if scope is None:
            raise RuntimeError("Dynamo generation requires an active ThunderAgent program")
        async with scope.request():
            output = await self.server.generate.remote(
                **generate_kwargs,
                thunderagent_session_id=scope.session_id,
            )
            return self._tag_weight_versions(output)

    @staticmethod
    def _tag_weight_versions(output: TokenOutput) -> TokenOutput:
        """Match LLMServerClient.generate's min/max_global_steps contract.

        V1 sync-mode consumers read these keys unconditionally; leaving them
        unset crashes trainer staleness accounting downstream. Pass through
        outputs without extra_fields untouched (duck-typed test doubles).
        """
        extra_fields = getattr(output, "extra_fields", None)
        if extra_fields is None:
            return output
        global_steps = extra_fields.get("global_steps")
        output.extra_fields.setdefault("min_global_steps", global_steps)
        output.extra_fields.setdefault("max_global_steps", global_steps)
        return output


class DynamoFullyAsyncLLMServerClient(FullyAsyncLLMServerClient):
    """FullyAsyncLLMServerClient with ThunderAgent program affinity.

    ThunderAgent pins all turns of one trajectory to one worker via the
    ``thunderagent_session_id`` kwarg (the server turns it into an
    X-Dynamo-Session-ID routing header). Callers with a stable per-trajectory
    request_id — uni-agent uses its gateway session_id — get affinity with no
    caller-side changes: when the kwarg is absent we key the program by
    request_id.

    Program lifecycle: the router's ProgramTable only frees an entry on an
    explicit session-final request — there is NO passive expiry, so an
    unfinalized program leaks router capacity for the whole training run and
    eventually pauses admission. With ``auto_finalize`` (default) every
    generate call finalizes its program on the way out: correct for
    single-turn callers, at the cost of cross-turn affinity. Multi-turn
    callers should set engine_kwargs.dynamo.thunderagent.auto_finalize=false
    AND call :meth:`finalize_program` from their trajectory-end hook
    (e.g. uni-agent gateway finalize/abort).
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        dynamo_server_handles=None,
        auto_finalize=True,
        finalize_leak_threshold=50,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._dynamo_server_handles = list(dynamo_server_handles or [])
        self._auto_finalize = bool(auto_finalize)
        self._finalize_leak_threshold = int(finalize_leak_threshold)
        self._consecutive_finalize_failures = 0

    async def generate(self, request_id, **kwargs):
        session_id = kwargs.setdefault("thunderagent_session_id", str(request_id))
        try:
            return await super().generate(request_id, **kwargs)
        finally:
            if self._auto_finalize:
                # The LB sticky cache is keyed by the OUTER request_id (the
                # routing key every generate attempt used), which may differ
                # from an explicitly passed session_id — route the finalize
                # with the same key the requests used.
                await self._finalize_with_recovery(session_id, routing_key=str(request_id))

    async def finalize_program(self, session_id: str, routing_key: str = None) -> None:
        """Release the ThunderAgent program for one trajectory.

        Routes the finalize to the frontend that actually served this session:
        the load balancer's sticky cache maps ``routing_key`` (the request_id
        used for generation; defaults to ``session_id``, which is identical on
        the uni-agent path) to that server — covering BOTH pools in
        separate_async, where the hybrid frontend is registered into this
        manager's LB. Falls back to broadcasting to this pool's static handles
        when no LB is wired.
        """
        session_id = str(session_id)
        key = str(routing_key) if routing_key is not None else session_id
        if self._load_balancer is not None:
            server_id, handle = await self._load_balancer.acquire_server.remote(request_id=key)
            try:
                await handle.finalize_program.remote(session_id=session_id)
            finally:
                self._load_balancer.release_server.remote(server_id=server_id)
            return
        await asyncio.gather(
            *[handle.finalize_program.remote(session_id=session_id) for handle in self._dynamo_server_handles]
        )

    async def _finalize_with_recovery(self, session_id: str, routing_key: str) -> None:
        """Bounded retry → static-handle broadcast → leak-threshold escalation.

        The router has no passive program expiry, so a swallowed finalize
        failure leaks capacity until admission pauses. A single failure must
        not destroy an already-successful trajectory, but sustained failures
        mean the run is drifting toward a hang — fail fast past the threshold
        (engine_kwargs.dynamo.thunderagent.finalize_leak_threshold).
        """
        delay = 0.2
        for attempt in range(3):
            try:
                await self.finalize_program(session_id, routing_key=routing_key)
                self._consecutive_finalize_failures = 0
                return
            except Exception:
                logger.warning(
                    "finalize_program attempt %d/3 failed for session %s",
                    attempt + 1,
                    session_id,
                    exc_info=(attempt == 2),
                )
                await asyncio.sleep(delay)
                delay *= 2
        try:
            await asyncio.gather(
                *[h.finalize_program.remote(session_id=str(session_id)) for h in self._dynamo_server_handles]
            )
            self._consecutive_finalize_failures = 0
            return
        except Exception:
            logger.error(
                "finalize_program broadcast fallback failed for session %s; router entry leaks",
                session_id,
                exc_info=True,
            )
        self._consecutive_finalize_failures += 1
        if self._consecutive_finalize_failures >= self._finalize_leak_threshold:
            raise RuntimeError(
                f"{self._consecutive_finalize_failures} consecutive ThunderAgent finalize failures — "
                "the router ProgramTable is leaking toward admission pause. Investigate frontend/"
                "router health (threshold: engine_kwargs.dynamo.thunderagent.finalize_leak_threshold)."
            )


class DynamoLLMServerManager(LLMServerManager):
    """LLM server manager that launches Dynamo through its shared worker pool."""

    def _thunderagent_config(self) -> dict:
        dynamo_config = (self.rollout_config.engine_kwargs or {}).get("dynamo", {}) or {}
        return dynamo_config.get("thunderagent", {}) or {}

    def _thunderagent_enabled(self) -> bool:
        return bool(self._thunderagent_config().get("enabled", False))

    async def _initialize_llm_servers(self, start_rank: int = None):
        if start_rank is None:
            start_rank = self.start_rank

        from recipe.dynamo.dynamo_thunderagent import DynamoThunderAgentReplica

        replica = DynamoThunderAgentReplica(
            replica_rank=start_rank,
            config=self.rollout_config,
            model_config=self.model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
        )
        if self.worker_group is None:
            # separate_async standalone pool: own resource pool + one
            # CheckpointEngineWorker per rollout GPU (nccl first hop),
            # dynamo stack launched on the pool nodes.
            await replica.init_standalone_pool()
        else:
            await replica.init_hybrid_worker_pool(self.worker_group)

        self.rollout_replicas = [replica]
        self.server_handles = [replica._server_handle]
        self.server_addresses = [replica._server_address]
        print(f"DynamoLLMServerManager: {self.server_addresses}")

        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    def get_client(self, client_cls=None, **kwargs):
        """Return an LLM client for the shared Dynamo frontend.

        V1 trainers pass an explicit ``client_cls`` (FullyAsyncLLMServerClient
        for colocate_async / separate_async): delegate to the base manager so
        the client gets the GlobalRequestLoadBalancer (degenerate single-server
        pass-through — Dynamo's KV router still does the real routing), the
        abort/resume retry loop, and min/max_global_steps aggregation. With
        ThunderAgent enabled the client is upgraded to the affinity-aware
        subclass (the server hard-requires a session id per request).

        Legacy V0 callers (ray_trainer / DynamoAgentLoopManager) pass no
        ``client_cls`` and keep the direct DynamoServerManager, which carries
        the ThunderAgent program-affinity path via DynamoAgentLoopWorker.
        """
        if client_cls is not None:
            if self._thunderagent_enabled() and issubclass(DynamoFullyAsyncLLMServerClient, client_cls):
                # separate_async is covered too: finalize_program routes via
                # the LB sticky cache, reaching whichever pool's frontend
                # served the session (hybrid or standalone).
                return super().get_client(
                    client_cls=DynamoFullyAsyncLLMServerClient,
                    dynamo_server_handles=self.server_handles,
                    auto_finalize=bool(self._thunderagent_config().get("auto_finalize", True)),
                    finalize_leak_threshold=int(self._thunderagent_config().get("finalize_leak_threshold", 50)),
                    **kwargs,
                )
            return super().get_client(client_cls=client_cls, **kwargs)

        if self._thunderagent_enabled() and bool(self.config.trainer.get("use_v1", False)):
            # V1 sync mode reaches here (base get_client() passes no client_cls).
            # Its AgentLoopWorkerTQ never establishes a ProgramScope, so the
            # legacy DynamoServerManager below would fail on every generate.
            raise ValueError(
                "engine_kwargs.dynamo.thunderagent.enabled=true is only supported under V1 for "
                "trainer_mode colocate_async/separate_async (their clients are upgraded to "
                "DynamoFullyAsyncLLMServerClient). For trainer_mode=sync disable thunderagent, "
                "or use the legacy path with trainer.use_v1=false."
            )

        dynamo_config = (self.rollout_config.engine_kwargs or {}).get("dynamo", {}) or {}
        thunderagent_config = dynamo_config.get("thunderagent", {}) or {}
        servers = list(zip(self.server_addresses, self.server_handles, strict=True))
        return DynamoServerManager(
            servers,
            thunderagent_enabled=bool(thunderagent_config.get("enabled", False)),
        )


class DynamoAgentLoopWorker(AgentLoopWorker):
    """Bind each trajectory to one ThunderAgent program."""

    async def _run_agent_loop(self, *args, **kwargs):
        async with self.llm_client.program_scope():
            return await super()._run_agent_loop(*args, **kwargs)


class DynamoAgentLoopManager(AgentLoopManager):
    """AgentLoopManager compatible with the current verl LLMServerClient API."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DynamoAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        await super()._init_agent_loop_workers()


__all__ = [
    "DynamoAgentLoopManager",
    "DynamoAgentLoopWorker",
    "DynamoFullyAsyncLLMServerClient",
    "DynamoLLMServerManager",
    "DynamoServerManager",
]
