"""동일 shared 판정에 필요한 순간 관측 불일치만 제한 재조회한다."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "bench/protocols/registered_desktop_gpu_retry_v1.json"
PROFILE_ID = "registered_desktop_shared_retry_v1"
_ACTIVE_ENGINE_CHURN = re.compile(r"active engine PID가 GPU process에 없다: ([1-9][0-9]*)\Z")

SHARED = importlib.import_module("evals.registered_desktop_gpu_shared_v1")
GPU1 = SHARED.GPU1


class RetryAdmissionError(RuntimeError):
    """재관측 정책 또는 clock 계약이 잘못됐다."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = deepcopy(dict(receipt)) if receipt is not None else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256(data)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RetryAdmissionError("GPU retry receipt를 결정적으로 직렬화할 수 없다") from error


def _load_policy() -> tuple[dict[str, Any], str]:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RetryAdmissionError("등록 GPU retry 정책을 읽을 수 없다") from error
    expected = {
        "schema": "registered_desktop_gpu_retry_v1",
        "profile_id": PROFILE_ID,
        "base_profile_id": SHARED.PROFILE_ID,
        "max_wait_s": 60,
        "max_queries": 12,
        "query_interval_s": 1,
        "retry_conditions": [
            "base_identity_pid_disappeared_between_xml_and_cim",
            "active_engine_pid_disappeared_from_same_snapshot_process_inventory",
        ],
        "final_predicate": "registered_desktop_shared_baseline_v1_unchanged",
        "threshold_relaxation": False,
        "gpu_owned_s": 0,
        "unclassified_activity": "block",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise RetryAdmissionError(f"등록 GPU retry 정책 필드가 다르다: {key}")
    base = policy.get("base_policy_reference")
    if not isinstance(base, dict) or base != {
        "path": "bench/protocols/registered_desktop_gpu_shared_v1.json",
        "sha256_lf": _sha256_lf(SHARED.POLICY_PATH),
    }:
        raise RetryAdmissionError("기반 shared GPU 정책 reference가 다르다")
    sources = policy.get("source_closure_sha256_lf")
    if not isinstance(sources, dict) or not sources:
        raise RetryAdmissionError("GPU retry 정책 source closure가 없다")
    for relative, expected_hash in sources.items():
        path = Path(relative)
        if (
            type(relative) is not str
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None
        ):
            raise RetryAdmissionError("GPU retry 정책 source closure 형식이 잘못됐다")
        absolute = ROOT / path
        if not absolute.is_file() or _sha256_lf(absolute) != expected_hash:
            raise RetryAdmissionError(f"GPU retry 정책 source closure가 다르다: {relative}")
    return policy, _sha256(raw)


def _elapsed(clock: Callable[[], float], start: float, previous: float) -> float:
    current = clock()
    if (
        type(current) not in {int, float}
        or not all(math.isfinite(float(value)) for value in (current, start, previous))
        or current < start
        or current - start < previous
    ):
        raise RetryAdmissionError("monotonic clock이 뒤로 이동했거나 숫자가 아니다")
    return float(current - start)


def _retryable_exception(error: BaseException) -> dict[str, Any] | None:
    """등록된 두 가지 process churn 원문만 구조적으로 인식한다."""
    if isinstance(error, SHARED.SharedAdmissionError):
        match = _ACTIVE_ENGINE_CHURN.fullmatch(str(error))
        if match is not None:
            return {
                "kind": "active_engine_pid_disappeared_from_same_snapshot_process_inventory",
                "pid": int(match.group(1)),
            }
        return None
    if not isinstance(error, GPU1.AdmissionError) or str(error) != "기반 GPU snapshot을 읽지 못했다":
        return None
    identity = error.__cause__
    if (
        not isinstance(identity, GPU1.AdmissionError)
        or str(identity) != "GPU identity PID 결합이 잘못됐다"
    ):
        return None
    missing = cast(BaseException, identity).__cause__
    if type(missing) is not KeyError or len(missing.args) != 1:
        return None
    pid = missing.args[0]
    if type(pid) is not int or pid <= 0:
        return None
    return {"kind": "base_identity_pid_disappeared_between_xml_and_cim", "pid": pid}


def _exception_row(error: BaseException, retry: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    return {
        "query_index": 0,
        "elapsed_s": elapsed_s,
        "outcome": "retryable_observation_churn",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "retry_condition": deepcopy(dict(retry)),
    }


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None and len(rows) < 8:
        rows.append({"error_type": type(current).__name__, "error_message": str(current)})
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else None
    return rows


def _timeout_receipt(
    *, policy: Mapping[str, Any], policy_hash: str, attempts: list[dict[str, Any]],
    termination: str, wall_s: float, cpu_s: float, slept_s: float,
    final_shared: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = deepcopy(dict(final_shared)) if final_shared is not None else {}
    result.update(
        {
            "busy": True,
            "busy_reasons": ["registered_desktop_gpu_retry_exhausted"],
            "decision_reasons": ["registered_desktop_gpu_retry_exhausted"],
            "decision": "timeout",
            "admission_policy": PROFILE_ID,
            "retry_base_admission_policy": SHARED.PROFILE_ID,
            "retry_policy_sha256": policy_hash,
            "retry_policy_reference": deepcopy(policy["base_policy_reference"]),
            "retry_attempts": deepcopy(attempts),
            "retry_summary": {
                "termination": termination,
                "query_count": len(attempts),
                "max_queries": policy["max_queries"],
                "interval_s": policy["query_interval_s"],
                "sleep_total_s": slept_s,
                "phase_wall_s": wall_s,
                "phase_cpu_s": cpu_s,
                "max_wait_s": policy["max_wait_s"],
                "gpu_owned_s": policy["gpu_owned_s"],
                "threshold_relaxation": False,
                "final_shared_predicate_accepted": False,
            },
            "gpu_exclusive": False,
            "performance_measurement": False,
            "nonexclusive_cost_observation": True,
            "fixture_callbacks_used": True,
            "actual_gpu_admission_evidence": False,
            "admission_evidence_requires_external_execution_receipt": True,
        }
    )
    return result


def _hard_failure_receipt(
    *, policy: Mapping[str, Any], policy_hash: str, attempts: list[dict[str, Any]],
    wall_s: float, cpu_s: float, slept_s: float, original_error: BaseException,
) -> dict[str, Any]:
    result = _timeout_receipt(
        policy=policy,
        policy_hash=policy_hash,
        attempts=attempts,
        termination="hard_failure",
        wall_s=wall_s,
        cpu_s=cpu_s,
        slept_s=slept_s,
        final_shared=None,
    )
    result["busy_reasons"] = ["registered_desktop_gpu_retry_hard_failure"]
    result["decision_reasons"] = ["registered_desktop_gpu_retry_hard_failure"]
    result["decision"] = "rejected"
    result["original_error"] = {"exception_chain": _exception_chain(original_error)}
    return result


def _decorate_final(
    shared_receipt: Mapping[str, Any], *, policy: Mapping[str, Any], policy_hash: str,
    attempts: list[dict[str, Any]], termination: str, wall_s: float, cpu_s: float, slept_s: float,
) -> dict[str, Any]:
    if shared_receipt.get("admission_policy") != SHARED.PROFILE_ID:
        raise RetryAdmissionError("기반 shared GPU receipt profile이 다르다")
    result = deepcopy(dict(shared_receipt))
    result.update(
        {
            "admission_policy": PROFILE_ID,
            "retry_base_admission_policy": SHARED.PROFILE_ID,
            "retry_policy_sha256": policy_hash,
            "retry_policy_reference": deepcopy(policy["base_policy_reference"]),
            "retry_attempts": deepcopy(attempts),
            "retry_summary": {
                "termination": termination,
                "query_count": len(attempts),
                "max_queries": policy["max_queries"],
                "interval_s": policy["query_interval_s"],
                "sleep_total_s": slept_s,
                "phase_wall_s": wall_s,
                "phase_cpu_s": cpu_s,
                "max_wait_s": policy["max_wait_s"],
                "gpu_owned_s": policy["gpu_owned_s"],
                "threshold_relaxation": False,
                "final_shared_predicate_accepted": shared_receipt.get("busy") is False,
            },
            "fixture_callbacks_used": True,
            "actual_gpu_admission_evidence": False,
        }
    )
    return result


def _retry_for_registered_desktop_gpu(
    *, query: Callable[[], dict[str, Any]], monotonic: Callable[[], float],
    sleep: Callable[[float], None], process_clock: Callable[[], float],
) -> dict[str, Any]:
    """등록 shared 판정은 그대로 두고 두 churn 관측만 제한 재시도한다."""
    policy, policy_hash = _load_policy()
    wall_start, cpu_start = monotonic(), process_clock()
    if (
        type(wall_start) not in {int, float}
        or type(cpu_start) not in {int, float}
        or not math.isfinite(float(wall_start))
        or not math.isfinite(float(cpu_start))
    ):
        raise RetryAdmissionError("clock 시작값이 숫자가 아니다")
    previous_elapsed = 0.0
    slept_s = 0.0
    attempts: list[dict[str, Any]] = []
    final_shared: dict[str, Any] | None = None
    termination = "not_started"

    while len(attempts) < policy["max_queries"]:
        if attempts and previous_elapsed >= policy["max_wait_s"]:
            termination = "wait_timeout"
            break
        try:
            shared_receipt = query()
            if not isinstance(shared_receipt, dict):
                raise RetryAdmissionError("기반 shared GPU query receipt가 object가 아니다")
            final_shared = deepcopy(shared_receipt)
        except Exception as error:
            retry = _retryable_exception(error)
            if retry is None:
                elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
                previous_elapsed = elapsed
                attempts.append(
                    {
                        "query_index": len(attempts) + 1,
                        "elapsed_s": elapsed,
                        "outcome": "hard_failure",
                        "exception_chain": _exception_chain(error),
                    }
                )
                cpu_end = process_clock()
                if (
                    type(cpu_end) not in {int, float}
                    or not math.isfinite(float(cpu_end))
                    or cpu_end < cpu_start
                ):
                    raise RetryAdmissionError("process clock이 뒤로 이동했거나 숫자가 아니다") from error
                receipt = _hard_failure_receipt(
                    policy=policy,
                    policy_hash=policy_hash,
                    attempts=attempts,
                    wall_s=elapsed,
                    cpu_s=float(cpu_end - cpu_start),
                    slept_s=slept_s,
                    original_error=error,
                )
                serialized = _canonical(receipt).decode("utf-8")
                raise RetryAdmissionError(
                    f"GPU retry hard failure receipt={serialized}", receipt=receipt
                ) from error
            elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
            previous_elapsed = elapsed
            row = _exception_row(error, retry, elapsed)
            row["query_index"] = len(attempts) + 1
            attempts.append(row)
            if elapsed > policy["max_wait_s"]:
                termination = "wait_timeout"
                break
        else:
            elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
            previous_elapsed = elapsed
            attempts.append(
                {
                    "query_index": len(attempts) + 1,
                    "elapsed_s": elapsed,
                    "outcome": "shared_receipt",
                    "snapshot": deepcopy(shared_receipt),
                }
            )
            if elapsed > policy["max_wait_s"]:
                termination = "wait_timeout"
                break
            termination = "accepted" if shared_receipt.get("busy") is False else "rejected"
            cpu_end = process_clock()
            if (
                type(cpu_end) not in {int, float}
                or not math.isfinite(float(cpu_end))
                or cpu_end < cpu_start
            ):
                raise RetryAdmissionError("process clock이 뒤로 이동했거나 숫자가 아니다")
            return _decorate_final(
                shared_receipt, policy=policy, policy_hash=policy_hash, attempts=attempts,
                termination=termination, wall_s=elapsed, cpu_s=float(cpu_end - cpu_start),
                slept_s=slept_s,
            )

        if len(attempts) >= policy["max_queries"]:
            termination = "query_limit"
            break
        remaining = policy["max_wait_s"] - previous_elapsed
        if remaining <= 0:
            termination = "wait_timeout"
            break
        delay = min(float(policy["query_interval_s"]), float(remaining))
        sleep(delay)
        slept_s += delay

    wall_elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
    cpu_end = process_clock()
    if (
        type(cpu_end) not in {int, float}
        or not math.isfinite(float(cpu_end))
        or cpu_end < cpu_start
    ):
        raise RetryAdmissionError("process clock이 뒤로 이동했거나 숫자가 아니다")
    return _timeout_receipt(
        policy=policy, policy_hash=policy_hash, attempts=attempts, termination=termination,
        wall_s=wall_elapsed, cpu_s=float(cpu_end - cpu_start), slept_s=slept_s,
        final_shared=final_shared,
    )


def query_registered_desktop_gpu_retry() -> dict[str, Any]:
    """production lease와 fresh XML+CIM을 매 attempt 다시 검증한다."""
    result = _retry_for_registered_desktop_gpu(
        query=SHARED.query_registered_desktop_gpu_shared,
        monotonic=time.monotonic,
        sleep=time.sleep,
        process_clock=time.process_time,
    )
    result["fixture_callbacks_used"] = False
    return result


@contextmanager
def snapshot_scope() -> Iterator[None]:
    """shared alias scope 안에서 R1 query만 retry 판으로 교체한다."""
    with SHARED.snapshot_scope():
        modules = GPU1._r1_modules()
        originals = [(module, module._query_gpu) for module in modules]

        def replacement() -> dict[str, Any]:
            return query_registered_desktop_gpu_retry()

        try:
            for module, _original in originals:
                module._query_gpu = replacement
            yield
        finally:
            for module, original in originals:
                module._query_gpu = original
