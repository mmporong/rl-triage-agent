"""기존 GPU 입장 기준을 유지하며 연속된 clear 관측까지 제한 시간 동안 기다린다."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v1 as B

SCHEMA = "lineage_gpu_wait_result_v1"
MAX_WAIT_S = 600
CONSECUTIVE_CLEAR = 2


class GpuWaitError(RuntimeError):
    """입장 확인이 완료되지 않아 SDK 자식을 시작할 수 없다."""


def policy() -> dict:
    return {"admission": GPU.policy(), "max_wait_s": MAX_WAIT_S,
            "interval_s": 2, "consecutive_clear": CONSECUTIVE_CLEAR,
            "training_retry": False, "terminate_other_processes": False}


def _finite(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _write_new(path: Path, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _reference(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _lease(check: Callable, expected: str | None) -> str:
    value = check()
    identifier = value.get("leaseId") if isinstance(value, dict) else None
    if (not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f-]{36}", identifier) is None
            or value.get("owner") != B.OWNER or value.get("state") != "running"
            or (expected is not None and identifier != expected)):
        raise GpuWaitError("GPU 대기 중 공용 lease 소유자·ID·상태가 달라졌다")
    return identifier


def _observation(value: object) -> dict:
    if (not isinstance(value, dict) or type(value.get("busy")) is not bool
            or value.get("admission_policy") != GPU.GPU_PROFILE
            or not isinstance(value.get("busy_reasons"), list)
            or any(not isinstance(item, str) or not item for item in value["busy_reasons"])
            or type(value.get("compute_process_count")) is not int
            or value["compute_process_count"] < 0):
        raise GpuWaitError("GPU 관측의 입장 profile 또는 필수 필드가 다르다")
    if not value["busy"] and (value["busy_reasons"] or value["compute_process_count"]):
        raise GpuWaitError("GPU clear 관측과 차단 사유·compute 수가 모순된다")
    return value


def await_clear(stage: Path, *, max_wait_s: float = 600, interval_s: float = 2,
                consecutive_clear: int = 2, lease_check: Callable = B.lease_context,
                query: Callable = GPU.query_gpu, clock: Callable = time.monotonic,
                sleep: Callable = time.sleep) -> dict:
    """현재 lease 아래 새 기록 폴더를 만들고 동일 기준 clear 두 번만 수용한다."""
    if (not _finite(max_wait_s) or not 0 < max_wait_s <= MAX_WAIT_S
            or not _finite(interval_s) or not 0 < interval_s <= min(30, max_wait_s)
            or type(consecutive_clear) is not int or consecutive_clear != CONSECUTIVE_CLEAR):
        raise ValueError("GPU 대기 한도·간격·연속 clear 조건이 등록 경계와 다르다")
    stage = Path(stage).resolve()
    if not stage.is_dir():
        raise FileNotFoundError("GPU 대기 stage가 없다")
    start = clock()
    if not _finite(start):
        raise ValueError("GPU 대기 clock이 유한 수치가 아니다")
    identifier = _lease(lease_check, None)
    root = stage / "gpu-wait"
    root.mkdir(exist_ok=False)
    samples_dir = root / "samples"
    samples_dir.mkdir()
    parameters = {"max_wait_s": max_wait_s, "interval_s": interval_s,
                  "consecutive_clear": consecutive_clear}
    _write_new(root / "intent.json", {"schema": SCHEMA + "_intent", "parameters": parameters,
                                     "admission": GPU.policy(), "lease_id": identifier,
                                     "gpu_jobs_started": 0, "training_retry": False})
    references: list[dict] = []
    streak = 0

    def finish(status: str, reason: str | None = None) -> dict:
        elapsed = clock() - start
        if not _finite(elapsed) or elapsed < 0:
            if status != "error":
                raise GpuWaitError("GPU 대기 clock이 뒤로 움직였거나 유한하지 않다")
            elapsed = None
        value = {"schema": SCHEMA, "status": status, "parameters": parameters,
                 "admission_policy": GPU.GPU_PROFILE, "lease_id": identifier,
                 "elapsed_s": elapsed, "clear_streak": streak, "samples": references,
                 "final_snapshot": references[-1] if references else None,
                 "reason": reason, "gpu_jobs_started": 0, "training_retry": False}
        path = root / "result.json"
        _write_new(path, value)
        return {**value, "receipt_path": str(path),
                "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    try:
        while True:
            now = clock()
            if not _finite(now) or now < start:
                raise GpuWaitError("GPU 대기 clock이 뒤로 움직였거나 유한하지 않다")
            if now - start >= max_wait_s:
                finish("timeout", "admission_wait_budget_exhausted")
                raise GpuWaitError("GPU 입장 대기 상한이 끝났다; 학습을 시작하지 않는다")
            _lease(lease_check, identifier)
            raw = query()
            _lease(lease_check, identifier)
            path = samples_dir / f"{len(references) + 1:04d}.json"
            if not isinstance(raw, dict):
                raise GpuWaitError("GPU 관측은 JSON 객체여야 한다")
            _write_new(path, raw)
            references.append(_reference(path, root))
            value = _observation(raw)
            streak = 0 if value["busy"] else streak + 1
            now = clock()
            if not _finite(now) or now < start:
                raise GpuWaitError("GPU 대기 clock이 뒤로 움직였거나 유한하지 않다")
            if now - start >= max_wait_s:
                finish("timeout", "admission_wait_budget_exhausted")
                raise GpuWaitError("GPU 입장 관측 중 대기 상한이 끝났다")
            if streak == consecutive_clear:
                return finish("clear")
            sleep(min(interval_s, max_wait_s - (now - start)))
    except BaseException as error:
        if not (root / "result.json").exists():
            finish("error", type(error).__name__)
        raise
