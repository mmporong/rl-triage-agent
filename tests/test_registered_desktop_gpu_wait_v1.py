from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evals.registered_desktop_gpu_wait_v1 as W

UTILIZATION_REASON = "gpu_utilization_above_registered_limit"


class FakeClock:
    def __init__(self) -> None:
        self.wall = 0.0
        self.cpu = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.wall

    def process(self) -> float:
        return self.cpu

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.wall += delay


def receipt(*, busy: bool, reasons: list[str] | None = None, ordinal: int = 1) -> dict:
    values = list(reasons or [])
    return {
        "busy": busy,
        "busy_reasons": values,
        "decision_reasons": list(values),
        "decision": "rejected" if busy else "accepted",
        "admission_policy": "registered_desktop_shared_v2",
        "utilization_percent": 2 if values == [UTILIZATION_REASON] else 0,
        "memory_used_mib": 1457,
        "compute_process_count": 0,
        "active_engines": [],
        "ordinal": ordinal,
    }


def run(sequence: list[dict], clock: FakeClock | None = None) -> tuple[dict, FakeClock, int]:
    timer = clock or FakeClock()
    calls = 0

    def query() -> dict:
        nonlocal calls
        value = sequence[min(calls, len(sequence) - 1)]
        calls += 1
        return deepcopy(value)

    result = W._wait_for_registered_desktop_gpu(
        query=query,
        monotonic=timer.monotonic,
        sleep=timer.sleep,
        process_clock=timer.process,
    )
    return result, timer, calls


def test_immediate_accept_preserves_snapshot_without_sleep() -> None:
    result, clock, calls = run([receipt(busy=False)])
    assert calls == 1
    assert clock.sleeps == []
    assert result["busy"] is False
    assert result["admission_policy"] == "registered_desktop_shared_wait_v1"
    assert result["base_admission_policy"] == "registered_desktop_shared_v2"
    assert result["wait_summary"] == {
        "termination": "accepted",
        "query_count": 1,
        "max_queries": 12,
        "interval_s": 2,
        "sleep_total_s": 0.0,
        "phase_wall_s": 0.0,
        "phase_cpu_s": 0.0,
        "max_wait_s": 60,
        "gpu_owned_s": 0,
        "threshold_relaxation": False,
    }
    assert result["wait_snapshots"][0]["snapshot"]["ordinal"] == 1
    assert result["fixture_callbacks_used"] is True


def test_utilization_only_retries_then_accepts_with_wait_in_wall() -> None:
    sequence = [
        receipt(busy=True, reasons=[UTILIZATION_REASON], ordinal=1),
        receipt(busy=True, reasons=[UTILIZATION_REASON], ordinal=2),
        receipt(busy=False, ordinal=3),
    ]
    result, clock, calls = run(sequence)
    assert calls == 3
    assert clock.sleeps == [2.0, 2.0]
    assert result["busy"] is False
    assert result["wait_summary"]["phase_wall_s"] == 4.0
    assert result["wait_summary"]["sleep_total_s"] == 4.0
    assert [row["elapsed_s"] for row in result["wait_snapshots"]] == [0.0, 2.0, 4.0]
    assert [row["snapshot"]["ordinal"] for row in result["wait_snapshots"]] == [1, 2, 3]


def test_non_retry_reason_and_ambiguous_reason_contract_stop_immediately() -> None:
    result, clock, calls = run([receipt(busy=True, reasons=["active_gpu_engine_present"])])
    assert calls == 1
    assert clock.sleeps == []
    assert result["decision"] == "rejected"
    assert result["busy_reasons"] == ["active_gpu_engine_present"]

    ambiguous = receipt(busy=True, reasons=[UTILIZATION_REASON])
    ambiguous["decision_reasons"] = [UTILIZATION_REASON, "other"]
    result, clock, calls = run([ambiguous])
    assert calls == 1
    assert clock.sleeps == []
    assert result["wait_summary"]["termination"] == "non_retryable_rejection"


def test_query_or_lease_failure_propagates_without_retry() -> None:
    clock = FakeClock()

    def failure() -> dict:
        raise RuntimeError("lease failed")

    with pytest.raises(RuntimeError, match="lease failed"):
        W._wait_for_registered_desktop_gpu(
            query=failure,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            process_clock=clock.process,
        )
    assert clock.sleeps == []


def test_query_limit_keeps_busy_and_never_relaxes_threshold() -> None:
    busy = receipt(busy=True, reasons=[UTILIZATION_REASON])
    result, clock, calls = run([busy])
    assert calls == 12
    assert clock.sleeps == [2.0] * 11
    assert result["busy"] is True
    assert result["busy_reasons"] == [UTILIZATION_REASON]
    assert result["decision"] == "timeout"
    assert result["wait_summary"]["termination"] == "query_limit"
    assert result["wait_summary"]["phase_wall_s"] == 22.0
    assert result["wait_summary"]["threshold_relaxation"] is False


def test_query_finishing_after_hard_timeout_cannot_be_accepted() -> None:
    clock = FakeClock()

    def slow_accepted() -> dict:
        clock.wall = 61.0
        clock.cpu = 0.25
        return receipt(busy=False)

    result = W._wait_for_registered_desktop_gpu(
        query=slow_accepted,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        process_clock=clock.process,
    )
    assert result["busy"] is True
    assert result["decision"] == "timeout"
    assert result["busy_reasons"] == ["registered_desktop_gpu_wait_timeout"]
    assert result["wait_summary"]["phase_wall_s"] == 61.0
    assert result["wait_summary"]["phase_cpu_s"] == 0.25


def test_production_query_reuses_zero_arg_base_query_each_attempt(monkeypatch) -> None:
    clock = FakeClock()
    values = [receipt(busy=True, reasons=[UTILIZATION_REASON]), receipt(busy=False)]
    calls = 0

    def base_query() -> dict:
        nonlocal calls
        value = values[calls]
        calls += 1
        return deepcopy(value)

    monkeypatch.setattr(W.GPU1, "query_registered_desktop_gpu", base_query)
    monkeypatch.setattr(W.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(W.time, "sleep", clock.sleep)
    monkeypatch.setattr(W.time, "process_time", clock.process)
    result = W.query_registered_desktop_gpu_wait()
    assert calls == 2
    assert result["busy"] is False
    assert result["fixture_callbacks_used"] is False


def test_scope_wraps_gpu1_alias_scope_and_restores_all_aliases(tmp_path: Path, monkeypatch) -> None:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    v2 = importlib.import_module("evals.r1_repair_execution_v2")
    originals = (package._query_gpu, top_level._query_gpu)
    accepted = receipt(busy=False)
    accepted["admission_policy"] = "registered_desktop_shared_wait_v1"
    monkeypatch.setattr(W, "query_registered_desktop_gpu_wait", lambda: deepcopy(accepted))

    with W.snapshot_scope():
        outer = (package._query_gpu, top_level._query_gpu)
        assert outer[0] is not originals[0]
        assert outer[1] is not originals[1]
        assert v2.V1._query_gpu is top_level._query_gpu
        with W.snapshot_scope():
            snapshot = v2.V1._snapshot(tmp_path, "wait")
        assert snapshot["busy"] is False
        assert snapshot["admission_policy"] == "registered_desktop_shared_wait_v1"
        assert (package._query_gpu, top_level._query_gpu) == outer

    assert (package._query_gpu, top_level._query_gpu) == originals
    assert v2.V1._query_gpu is originals[1]


def test_scope_has_no_callback_injection_surface(tmp_path: Path) -> None:
    scope: Any = W.snapshot_scope
    with pytest.raises(TypeError), scope(query=lambda: receipt(busy=False)):
        importlib.import_module("r1_repair_execution")._snapshot(tmp_path, "forged")
    assert not (tmp_path / "gpu_forged.json").exists()
