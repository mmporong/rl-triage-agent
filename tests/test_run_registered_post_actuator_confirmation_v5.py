from __future__ import annotations

from copy import deepcopy

import pytest

import evals.run_registered_post_actuator_confirmation_v5 as OP


def test_retry_scope_uses_proxy_without_mutating_shared_module(monkeypatch) -> None:
    original_gpu = OP.CORE.GPU
    original_shared_query = OP.RETRY.SHARED.query_registered_desktop_gpu_shared
    accepted = {
        "busy": False,
        "admission_policy": "registered_desktop_shared_retry_v1",
        "retry_base_admission_policy": "registered_desktop_shared_baseline_v1",
        "retry_summary": {"max_queries": 12, "max_wait_s": 60, "interval_s": 1},
    }
    calls: list[str] = []
    monkeypatch.setattr(
        OP.RETRY,
        "query_registered_desktop_gpu_retry",
        lambda: calls.append("retry") or deepcopy(accepted),
    )
    with OP.retry_observation_scope():
        assert OP.CORE.GPU.query_registered_desktop_gpu_shared() == accepted
        assert (
            OP.RETRY.SHARED.query_registered_desktop_gpu_shared is original_shared_query
        )
    assert calls == ["retry"]
    assert OP.CORE.GPU is original_gpu
    assert OP.RETRY.SHARED.query_registered_desktop_gpu_shared is original_shared_query


def test_real_retry_function_reaches_base_shared_query(monkeypatch) -> None:
    calls: list[str] = []

    def base_query():
        calls.append("base")
        return {"busy": False, "admission_policy": OP.RETRY.SHARED.PROFILE_ID}

    monkeypatch.setattr(
        OP.RETRY.SHARED, "query_registered_desktop_gpu_shared", base_query
    )
    with OP.retry_observation_scope():
        observed = OP.CORE.GPU.query_registered_desktop_gpu_shared()
    assert calls == ["base"]
    assert observed["busy"] is False
    assert observed["admission_policy"] == OP.RETRY.PROFILE_ID
    assert observed["retry_base_admission_policy"] == OP.RETRY.SHARED.PROFILE_ID


def test_nested_and_exception_scope_restore_exact_namespaces() -> None:
    original_gpu = OP.CORE.GPU
    original_relations = OP.CORE.SOURCE_RELATIONS
    shared_query = OP.RETRY.SHARED.query_registered_desktop_gpu_shared
    with OP.retry_observation_scope():
        outer_gpu = OP.CORE.GPU
        with OP.retry_observation_scope():
            assert OP.CORE.GPU is not outer_gpu
            assert OP.RETRY.SHARED.query_registered_desktop_gpu_shared is shared_query
        assert OP.CORE.GPU is outer_gpu
    assert OP.CORE.GPU is original_gpu
    assert OP.CORE.SOURCE_RELATIONS is original_relations
    with pytest.raises(RuntimeError, match="fixture"), OP.retry_observation_scope():
        raise RuntimeError("fixture")
    assert OP.CORE.GPU is original_gpu
    assert OP.CORE.SOURCE_RELATIONS is original_relations


def test_source_closure_contains_retry_policy_and_core() -> None:
    assert OP.CORE_REL in OP.SOURCE_RELATIONS
    assert set(OP.RETRY_RELATIONS) <= set(OP.SOURCE_RELATIONS)
    assert OP.RUNNER_REL in OP.SOURCE_RELATIONS
