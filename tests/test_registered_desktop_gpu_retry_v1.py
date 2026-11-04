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

import evals.registered_desktop_gpu_retry_v1 as R


class FakeClock:
    def __init__(self) -> None:
        self.wall = 100.0
        self.cpu = 10.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.wall

    def process(self) -> float:
        return self.cpu

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.wall += seconds
        self.cpu += 0.01


def test_nonzero_clock_origin_rejects_elapsed_time_regression() -> None:
    assert R._elapsed(lambda: 102.0, 100.0, 0.0) == 2.0
    with pytest.raises(R.RetryAdmissionError, match="monotonic clock"):
        R._elapsed(lambda: 101.0, 100.0, 2.0)


def shared_receipt(*, busy: bool = False, reason: str = "", ordinal: int = 1) -> dict[str, Any]:
    reasons = [reason] if reason else []
    return {
        "busy": busy,
        "busy_reasons": reasons,
        "decision_reasons": reasons,
        "decision": "rejected" if busy else "accepted",
        "admission_policy": R.SHARED.PROFILE_ID,
        "base_admission_policy": R.GPU1.PROFILE_ID,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "nonexclusive_cost_observation": True,
        "threshold_relaxation": True,
        "raw_snapshot": {"ordinal": ordinal},
        "raw_snapshot_sha256": f"{ordinal:064x}",
        "fixture_callbacks_used": True,
        "actual_gpu_admission_evidence": False,
        "admission_evidence_requires_external_execution_receipt": True,
    }


def missing_identity(pid: int = 17060) -> R.GPU1.AdmissionError:
    missing = KeyError(pid)
    identity = R.GPU1.AdmissionError("GPU identity PID 결합이 잘못됐다")
    identity.__cause__ = missing
    outer = R.GPU1.AdmissionError("기반 GPU snapshot을 읽지 못했다")
    outer.__cause__ = identity
    return outer


def engine_churn(pid: int = 31232) -> Exception:
    return R.SHARED.SharedAdmissionError(f"active engine PID가 GPU process에 없다: {pid}")


def run(sequence: list[object]) -> tuple[dict[str, Any], FakeClock, int]:
    clock = FakeClock()
    calls = 0

    def query() -> dict[str, Any]:
        nonlocal calls
        value = sequence[min(calls, len(sequence) - 1)]
        calls += 1
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, dict)
        return deepcopy(value)

    result = R._retry_for_registered_desktop_gpu(
        query=query,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        process_clock=clock.process,
    )
    return result, clock, calls


def test_policy_closure_and_immediate_accept_preserve_shared_predicate() -> None:
    policy, policy_hash = R._load_policy()
    result, clock, calls = run([shared_receipt()])
    assert policy["final_predicate"] == "registered_desktop_shared_baseline_v1_unchanged"
    assert result["retry_policy_sha256"] == policy_hash
    assert calls == 1
    assert clock.sleeps == []
    assert result["busy"] is False
    assert result["admission_policy"] == R.PROFILE_ID
    assert result["retry_base_admission_policy"] == R.SHARED.PROFILE_ID
    assert result["retry_summary"] == {
        "termination": "accepted",
        "query_count": 1,
        "max_queries": 12,
        "interval_s": 1,
        "sleep_total_s": 0.0,
        "phase_wall_s": 0.0,
        "phase_cpu_s": 0.0,
        "max_wait_s": 60,
        "gpu_owned_s": 0,
        "threshold_relaxation": False,
        "final_shared_predicate_accepted": True,
    }
    assert result["retry_attempts"][0]["snapshot"]["admission_policy"] == R.SHARED.PROFILE_ID


def test_two_registered_churn_shapes_retry_then_accept_and_preserve_every_attempt() -> None:
    result, clock, calls = run([missing_identity(), engine_churn(), shared_receipt(ordinal=3)])
    assert calls == 3
    assert clock.sleeps == [1.0, 1.0]
    assert result["busy"] is False
    assert [row["outcome"] for row in result["retry_attempts"]] == [
        "retryable_observation_churn",
        "retryable_observation_churn",
        "shared_receipt",
    ]
    assert [row.get("retry_condition", {}).get("pid") for row in result["retry_attempts"]] == [
        17060,
        31232,
        None,
    ]
    assert result["retry_summary"]["phase_wall_s"] == 2.0


@pytest.mark.parametrize(
    "error",
    [
        R.GPU1.AdmissionError("lease failed"),
        R.SHARED.SharedAdmissionError("GPU process 이름이 shared 등록 범위 밖이다: 4"),
        R.SHARED.SharedAdmissionError("active engine type이 허용되지 않았다: 4"),
        R.RetryAdmissionError("기반 shared GPU receipt profile이 다르다"),
    ],
)
def test_policy_lease_unknown_name_and_non_churn_fail_without_retry(error: Exception) -> None:
    clock = FakeClock()
    calls = 0

    def query() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(R.RetryAdmissionError, match="GPU retry hard failure") as caught:
        R._retry_for_registered_desktop_gpu(
            query=query,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            process_clock=clock.process,
        )
    assert calls == 1
    assert clock.sleeps == []
    assert caught.value.__cause__ is error
    receipt = caught.value.receipt
    assert isinstance(receipt, dict)
    assert receipt["busy"] is True
    assert receipt["decision"] == "rejected"
    assert receipt["busy_reasons"] == ["registered_desktop_gpu_retry_hard_failure"]
    assert receipt["retry_summary"]["termination"] == "hard_failure"
    assert receipt["original_error"] == {
        "exception_chain": [{"error_type": type(error).__name__, "error_message": str(error)}]
    }
    assert receipt["retry_attempts"] == [
        {
            "query_index": 1,
            "elapsed_s": 0.0,
            "outcome": "hard_failure",
            "exception_chain": [{"error_type": type(error).__name__, "error_message": str(error)}],
        }
    ]
    assert '"termination":"hard_failure"' in str(caught.value)


def test_churn_then_hard_failure_preserves_both_attempts_and_original_cause() -> None:
    clock = FakeClock()
    hard = R.GPU1.AdmissionError("lease changed")
    sequence: list[object] = [engine_churn(31232), hard]
    calls = 0

    def query() -> dict[str, Any]:
        nonlocal calls
        value = sequence[calls]
        calls += 1
        if isinstance(value, BaseException):
            raise value
        raise AssertionError("unexpected receipt")

    with pytest.raises(R.RetryAdmissionError, match="GPU retry hard failure") as caught:
        R._retry_for_registered_desktop_gpu(
            query=query,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            process_clock=clock.process,
        )
    assert calls == 2
    assert clock.sleeps == [1.0]
    assert caught.value.__cause__ is hard
    receipt = caught.value.receipt
    assert isinstance(receipt, dict)
    assert [row["outcome"] for row in receipt["retry_attempts"]] == [
        "retryable_observation_churn",
        "hard_failure",
    ]
    assert receipt["retry_attempts"][0]["retry_condition"] == {
        "kind": "active_engine_pid_disappeared_from_same_snapshot_process_inventory",
        "pid": 31232,
    }
    assert receipt["retry_attempts"][1]["exception_chain"] == [
        {"error_type": "AdmissionError", "error_message": "lease changed"}
    ]
    assert receipt["original_error"] == {
        "exception_chain": [{"error_type": "AdmissionError", "error_message": "lease changed"}]
    }
    assert receipt["retry_summary"]["query_count"] == 2
    assert receipt["retry_summary"]["phase_wall_s"] == 1.0


def test_shared_rejection_is_final_and_not_retried() -> None:
    result, clock, calls = run(
        [shared_receipt(busy=True, reason="unapproved_base_reason:unknown", ordinal=1)]
    )
    assert calls == 1
    assert clock.sleeps == []
    assert result["busy"] is True
    assert result["decision"] == "rejected"
    assert result["busy_reasons"] == ["unapproved_base_reason:unknown"]
    assert result["retry_summary"]["termination"] == "rejected"
    assert result["retry_summary"]["final_shared_predicate_accepted"] is False


def test_query_limit_returns_fail_closed_receipt_with_all_exceptions() -> None:
    result, clock, calls = run([engine_churn(9001)])
    assert calls == 12
    assert clock.sleeps == [1.0] * 11
    assert result["busy"] is True
    assert result["decision"] == "timeout"
    assert result["busy_reasons"] == ["registered_desktop_gpu_retry_exhausted"]
    assert result["retry_summary"]["termination"] == "query_limit"
    assert result["retry_summary"]["query_count"] == 12
    assert len(result["retry_attempts"]) == 12
    assert all(row["retry_condition"]["pid"] == 9001 for row in result["retry_attempts"])


def test_result_after_deadline_cannot_be_accepted() -> None:
    clock = FakeClock()

    def slow_success() -> dict[str, Any]:
        clock.wall += 61.0
        clock.cpu += 0.25
        return shared_receipt()

    result = R._retry_for_registered_desktop_gpu(
        query=slow_success,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        process_clock=clock.process,
    )
    assert result["busy"] is True
    assert result["decision"] == "timeout"
    assert result["retry_summary"]["termination"] == "wait_timeout"
    assert result["retry_summary"]["phase_wall_s"] == 61.0
    assert result["retry_summary"]["final_shared_predicate_accepted"] is False
    assert result["retry_attempts"][0]["snapshot"]["decision"] == "accepted"


@pytest.mark.parametrize(("clock_name", "value"), [("wall", float("nan")), ("cpu", float("inf"))])
def test_nonfinite_start_clocks_are_rejected(clock_name: str, value: float) -> None:
    clock = FakeClock()
    setattr(clock, clock_name, value)
    with pytest.raises(R.RetryAdmissionError, match="clock 시작값"):
        R._retry_for_registered_desktop_gpu(
            query=lambda: shared_receipt(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            process_clock=clock.process,
        )


def test_production_requeries_zero_arg_shared_query_and_marks_nonfixture(monkeypatch) -> None:
    clock = FakeClock()
    values: list[object] = [missing_identity(22), shared_receipt(ordinal=2)]
    calls = 0

    def query() -> dict[str, Any]:
        nonlocal calls
        value = values[calls]
        calls += 1
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, dict)
        return deepcopy(value)

    monkeypatch.setattr(R.SHARED, "query_registered_desktop_gpu_shared", query)
    monkeypatch.setattr(R.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(R.time, "sleep", clock.sleep)
    monkeypatch.setattr(R.time, "process_time", clock.process)
    result = R.query_registered_desktop_gpu_retry()
    assert calls == 2
    assert result["busy"] is False
    assert result["fixture_callbacks_used"] is False


def test_scope_nests_and_restores_all_aliases(tmp_path: Path, monkeypatch) -> None:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    v2 = importlib.import_module("evals.r1_repair_execution_v2")
    originals = (package._query_gpu, top_level._query_gpu)
    accepted = shared_receipt()
    accepted["admission_policy"] = R.PROFILE_ID
    monkeypatch.setattr(R, "query_registered_desktop_gpu_retry", lambda: deepcopy(accepted))

    with R.snapshot_scope():
        outer = (package._query_gpu, top_level._query_gpu)
        assert outer[0] is not originals[0]
        assert outer[1] is not originals[1]
        assert v2.V1._query_gpu is top_level._query_gpu
        with R.snapshot_scope():
            snapshot = v2.V1._snapshot(tmp_path, "retry")
        assert snapshot["busy"] is False
        assert snapshot["admission_policy"] == R.PROFILE_ID
        assert (package._query_gpu, top_level._query_gpu) == outer

    assert (package._query_gpu, top_level._query_gpu) == originals
    assert v2.V1._query_gpu is originals[1]


def test_scope_has_no_callback_injection_surface(tmp_path: Path) -> None:
    scope: Any = R.snapshot_scope
    with pytest.raises(TypeError), scope(query=lambda: shared_receipt()):
        importlib.import_module("r1_repair_execution")._snapshot(tmp_path, "forged")
    assert not (tmp_path / "gpu_forged.json").exists()
