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
"""R1 (flexkv_l2_design.md): publish FlexKV CPU capacity into MDC runtime_data."""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from recipe.dynamo import _dynamo_vllm_with_control as ctrl


def _qwen30b_vllm_config(cache_dtype: str = "auto") -> SimpleNamespace:
    """Shape-compatible stand-in for Qwen3-Coder-30B-A3B (48L x 4KV x 128d)."""
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=48),
        hf_config=SimpleNamespace(num_hidden_layers=48),
        get_total_num_kv_heads=lambda: 4,
        get_head_size=lambda: 128,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


class _RecordingRuntimeConfig:
    def __init__(self):
        self.engine_specific: dict[str, str] = {}

    def set_engine_specific(self, key: str, value: str):
        self.engine_specific[key] = value


@pytest.fixture
def fake_dyn_main(monkeypatch):
    """Install a fake ``dynamo.vllm.main`` exposing only register_model."""
    calls: list[dict] = []

    async def register_model(*args, **kwargs):
        calls.append(kwargs)
        return "registered"

    fake_main = ModuleType("dynamo.vllm.main")
    fake_main.register_model = register_model
    fake_vllm = ModuleType("dynamo.vllm")
    fake_vllm.main = fake_main
    fake_dynamo = ModuleType("dynamo")
    fake_dynamo.vllm = fake_vllm
    monkeypatch.setitem(sys.modules, "dynamo", fake_dynamo)
    monkeypatch.setitem(sys.modules, "dynamo.vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "dynamo.vllm.main", fake_main)
    return fake_main, calls


def test_kv_bytes_per_token_matches_qwen30b_full_model():
    # 48 layers x 2 (K,V) x 4 KV heads x 128 dim x 2 B = 98,304 B/token,
    # the model-total figure validated in the Phase-1 capacity audit.
    assert ctrl._flexkv_kv_bytes_per_token(_qwen30b_vllm_config()) == 98_304


def test_kv_bytes_per_token_halves_for_fp8_cache():
    assert ctrl._flexkv_kv_bytes_per_token(_qwen30b_vllm_config("fp8")) == 49_152


def test_kv_bytes_per_token_env_override(monkeypatch):
    monkeypatch.setenv("FLEXKV_KV_BYTES_PER_TOKEN", "12345")
    assert ctrl._flexkv_kv_bytes_per_token(None) == 12_345


def test_kv_bytes_per_token_unreadable_config_returns_none():
    assert ctrl._flexkv_kv_bytes_per_token(SimpleNamespace()) is None


def test_publisher_noop_when_flexkv_disabled(fake_dyn_main, monkeypatch):
    fake_main, _ = fake_dyn_main
    monkeypatch.delenv("DYNAMO_USE_FLEXKV", raising=False)
    original = fake_main.register_model
    ctrl._install_flexkv_capacity_publisher()
    assert fake_main.register_model is original


def test_publisher_noop_without_cache_budget(fake_dyn_main, monkeypatch):
    fake_main, _ = fake_dyn_main
    monkeypatch.setenv("DYNAMO_USE_FLEXKV", "1")
    monkeypatch.delenv("FLEXKV_CPU_CACHE_GB", raising=False)
    original = fake_main.register_model
    ctrl._install_flexkv_capacity_publisher()
    assert fake_main.register_model is original


def test_publisher_injects_capacity_record(fake_dyn_main, monkeypatch):
    fake_main, calls = fake_dyn_main
    monkeypatch.setenv("DYNAMO_USE_FLEXKV", "1")
    monkeypatch.setenv("FLEXKV_CPU_CACHE_GB", "96")
    monkeypatch.setitem(ctrl._engine_holder, "vllm_config", _qwen30b_vllm_config())

    ctrl._install_flexkv_capacity_publisher()
    runtime_config = _RecordingRuntimeConfig()
    result = asyncio.run(fake_main.register_model(runtime_config=runtime_config))

    assert result == "registered"
    assert calls and calls[0]["runtime_config"] is runtime_config
    payload = json.loads(runtime_config.engine_specific[ctrl.FLEXKV_CAPACITY_RUNTIME_KEY])
    # 96 GiB / 98,304 B per token = 1,048,576 tokens — the 0.98M/shard figure
    # from the capacity audit (B0/B3).
    assert payload["host_total_tokens"] == 1_048_576
    assert payload["cpu_cache_gb"] == 96.0
    assert payload["kv_bytes_per_token"] == 98_304


def test_publisher_survives_missing_vllm_config(fake_dyn_main, monkeypatch):
    """No vllm_config and no override: register still goes through, no record."""
    fake_main, calls = fake_dyn_main
    monkeypatch.setenv("DYNAMO_USE_FLEXKV", "1")
    monkeypatch.setenv("FLEXKV_CPU_CACHE_GB", "96")
    monkeypatch.delenv("FLEXKV_KV_BYTES_PER_TOKEN", raising=False)
    monkeypatch.setitem(ctrl._engine_holder, "vllm_config", None)

    ctrl._install_flexkv_capacity_publisher()
    runtime_config = _RecordingRuntimeConfig()
    result = asyncio.run(fake_main.register_model(runtime_config=runtime_config))

    assert result == "registered"
    assert calls
    assert ctrl.FLEXKV_CAPACITY_RUNTIME_KEY not in runtime_config.engine_specific
