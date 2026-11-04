"""등록 GPU admission의 일시적인 utilization 단독 거부만 제한 재조회한다."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "bench/protocols/registered_desktop_gpu_wait_v1.json"
PROFILE_ID = "registered_desktop_shared_wait_v1"
RETRY_REASONS = ["gpu_utilization_above_registered_limit"]

GPU1 = importlib.import_module("evals.registered_desktop_gpu_admission_v1")


class WaitAdmissionError(RuntimeError):
    """wait 정책 또는 clock 계약이 잘못됐다."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256(data)


def _load_policy() -> tuple[dict[str, Any], str]:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise WaitAdmissionError("등록 wait 정책을 읽을 수 없다") from error
    expected = {
        "schema": "registered_desktop_gpu_wait_v1",
        "profile_id": PROFILE_ID,
        "base_profile_id": GPU1.PROFILE_ID,
        "max_wait_s": 60,
        "max_queries": 12,
        "query_interval_s": 2,
        "retry_busy_reasons": RETRY_REASONS,
        "threshold_relaxation": False,
        "gpu_owned_s": 0,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise WaitAdmissionError(f"등록 wait 정책 필드가 다르다: {key}")
    base = policy.get("base_policy_reference")
    if not isinstance(base, dict) or base != {
        "path": "bench/protocols/registered_desktop_gpu_admission_v1.json",
        "sha256_lf": _sha256_lf(GPU1.POLICY_PATH),
    }:
        raise WaitAdmissionError("기반 GPU 정책 reference가 다르다")
    sources = policy.get("source_closure_sha256_lf")
    if not isinstance(sources, dict) or not sources:
        raise WaitAdmissionError("wait 정책 source closure가 없다")
    for relative, expected_hash in sources.items():
        path = Path(relative)
        if (
            type(relative) is not str
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None
        ):
            raise WaitAdmissionError("wait 정책 source closure 형식이 잘못됐다")
        absolute = ROOT / path
        if not absolute.is_file() or _sha256_lf(absolute) != expected_hash:
            raise WaitAdmissionError(f"wait 정책 source closure가 다르다: {relative}")
    return policy, _sha256(raw)


def _elapsed(clock: Callable[[], float], start: float, previous: float) -> float:
    current = clock()
    if type(current) not in {int, float} or current < start or current < previous:
        raise WaitAdmissionError("monotonic clock이 뒤로 이동했거나 숫자가 아니다")
    return float(current - start)


def _is_retry_only(snapshot: Mapping[str, Any]) -> bool:
    return (
        snapshot.get("busy") is True
        and snapshot.get("busy_reasons") == RETRY_REASONS
        and snapshot.get("decision_reasons") == RETRY_REASONS
    )


def _wait_for_registered_desktop_gpu(
    *,
    query: Callable[[], dict[str, Any]],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    process_clock: Callable[[], float],
) -> dict[str, Any]:
    """주입 clock으로 경계를 검증하는 순수 wait controller."""
    policy, policy_hash = _load_policy()
    wall_start = monotonic()
    cpu_start = process_clock()
    if type(wall_start) not in {int, float} or type(cpu_start) not in {int, float}:
        raise WaitAdmissionError("clock 시작값이 숫자가 아니다")
    previous_elapsed = 0.0
    slept_s = 0.0
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    termination = "not_started"

    while len(attempts) < policy["max_queries"]:
        if attempts and previous_elapsed >= policy["max_wait_s"]:
            termination = "wait_timeout"
            break
        snapshot = query()
        if not isinstance(snapshot, dict):
            raise WaitAdmissionError("기반 GPU query receipt 형식이 잘못됐다")
        elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
        previous_elapsed = elapsed
        final = deepcopy(snapshot)
        attempts.append(
            {"query_index": len(attempts) + 1, "elapsed_s": elapsed, "snapshot": deepcopy(snapshot)}
        )
        if elapsed > policy["max_wait_s"]:
            termination = "wait_timeout"
            break
        if snapshot.get("busy") is False:
            termination = "accepted"
            break
        if not _is_retry_only(snapshot):
            termination = "non_retryable_rejection"
            break
        if len(attempts) >= policy["max_queries"]:
            termination = "query_limit"
            break
        remaining = policy["max_wait_s"] - elapsed
        if remaining <= 0:
            termination = "wait_timeout"
            break
        delay = min(float(policy["query_interval_s"]), float(remaining))
        sleep(delay)
        slept_s += delay

    if final is None:
        raise WaitAdmissionError("GPU wait가 snapshot 없이 끝났다")
    wall_elapsed = _elapsed(monotonic, float(wall_start), previous_elapsed)
    cpu_end = process_clock()
    if type(cpu_end) not in {int, float} or cpu_end < cpu_start:
        raise WaitAdmissionError("process clock이 뒤로 이동했거나 숫자가 아니다")
    timed_out = termination in {"wait_timeout", "query_limit"}
    result = deepcopy(final)
    base_profile = result.get("admission_policy")
    result["base_admission_policy"] = base_profile
    result["admission_policy"] = PROFILE_ID
    result["base_policy_reference"] = deepcopy(policy["base_policy_reference"])
    result["wait_policy_sha256"] = policy_hash
    result["wait_snapshots"] = attempts
    result["wait_summary"] = {
        "termination": termination,
        "query_count": len(attempts),
        "max_queries": policy["max_queries"],
        "interval_s": policy["query_interval_s"],
        "sleep_total_s": slept_s,
        "phase_wall_s": wall_elapsed,
        "phase_cpu_s": float(cpu_end - cpu_start),
        "max_wait_s": policy["max_wait_s"],
        "gpu_owned_s": 0,
        "threshold_relaxation": False,
    }
    if timed_out:
        result["busy"] = True
        if final.get("busy") is True and final.get("busy_reasons") == RETRY_REASONS:
            result["busy_reasons"] = deepcopy(RETRY_REASONS)
            result["decision_reasons"] = deepcopy(RETRY_REASONS)
        else:
            result["busy_reasons"] = ["registered_desktop_gpu_wait_timeout"]
            result["decision_reasons"] = ["registered_desktop_gpu_wait_timeout"]
        result["decision"] = "timeout"
    elif termination == "accepted":
        result["decision"] = "accepted"
    else:
        result["decision"] = "rejected"
    result["fixture_callbacks_used"] = True
    result["actual_gpu_admission_evidence"] = False
    return result


def query_registered_desktop_gpu_wait() -> dict[str, Any]:
    """production lease와 fresh identity query를 매 attempt마다 다시 검증한다."""
    result = _wait_for_registered_desktop_gpu(
        query=GPU1.query_registered_desktop_gpu,
        monotonic=time.monotonic,
        sleep=time.sleep,
        process_clock=time.process_time,
    )
    result["fixture_callbacks_used"] = False
    return result


@contextmanager
def snapshot_scope() -> Iterator[None]:
    """기반 alias scope 안에서 R1 query만 wait 판으로 교체한다."""
    with GPU1.snapshot_scope():
        modules = GPU1._r1_modules()
        originals = [(module, module._query_gpu) for module in modules]

        def replacement() -> dict[str, Any]:
            return query_registered_desktop_gpu_wait()

        try:
            for module, _original in originals:
                module._query_gpu = replacement
            yield
        finally:
            for module, original in originals:
                module._query_gpu = original
