"""PPO ratio backend v2에 한정된 CSRSS 3D 계수기 재표본을 더한 판.

동결 분류기의 첫 판정은 그대로 보존한다. 첫 표본이 정확히 같은 CSRSS 프로세스의 3D 1%만 보고하고
NVIDIA XML은 계산 프로세스를 보고하지 않은 경우에만 2초 간격으로 두 번 더 읽는다. 두 후속 표본이 모두
동결 분류기에서 clear여야 실행한다. PID·생성 시각·세션 신원이 바뀌거나 다른 busy 근거, 2% 이상 엔진,
메모리 증가, XML 프로세스 등 하나라도 나타나면 즉시 차단한다. 모든 원시 표본과 신원은 stage의 기존
gpu_before/gpu_after receipt 안에 보존한다. 독립 query_gpu 호출은 재표본 없이 한 번만 판정한다.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import ppo_ratio_repair_execution as V1
import ppo_ratio_repair_execution_v2 as V2
import r1_repair_execution as FROZEN

TASK, FIXED_EVAL_PROTOCOL = V1.TASK, V1.FIXED_EVAL_PROTOCOL
sha, write_new, preflight = V1.sha, V1.write_new, V1.preflight
GPU_PROFILE, GPU_ADMISSION = V2.GPU_PROFILE, V2.GPU_ADMISSION
BASE_PROFILE, EXTRA_DISPLAY_PROCESS_NAMES = V2.BASE_PROFILE, V2.EXTRA_DISPLAY_PROCESS_NAMES
RESAMPLE_INTERVAL_S = 2
MAX_SAMPLES = 3


def query_gpu() -> dict[str, Any]:
    """독립 조회와 후속 표본이 공유하는 strict single-sample 판정."""
    return V2.query_gpu()


def process_identity(pid: int) -> dict[str, Any]:
    """CIM의 PID·생성 시각·세션 ID를 한 묶음으로 읽는다."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("GPU engine PID가 올바르지 않다")
    script = (
        "$ErrorActionPreference='Stop'; "
        f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
        "if($null -eq $p){throw 'process missing'}; "
        "$v=[ordered]@{ProcessId=[int]$p.ProcessId;"
        "CreationDate=$p.CreationDate.ToUniversalTime().ToString('o');SessionId=[int]$p.SessionId}; "
        "ConvertTo-Json -InputObject $v -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        value = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("GPU engine process 신원을 읽지 못했다") from error
    if (not isinstance(value, dict) or value.get("ProcessId") != pid
            or not isinstance(value.get("CreationDate"), str) or not value["CreationDate"]
            or type(value.get("SessionId")) is not int):
        raise ValueError("GPU engine process 신원 필드가 불완전하다")
    return {"ProcessId": pid, "CreationDate": value["CreationDate"], "SessionId": value["SessionId"]}


def _candidate_pid(snapshot: Mapping[str, Any]) -> int | None:
    reasons = snapshot.get("busy_reasons")
    engines = snapshot.get("active_engines")
    if (snapshot.get("busy") is not True or snapshot.get("utilization_percent") != 0
            or snapshot.get("compute_process_count") != 0 or not isinstance(reasons, list)
            or not isinstance(engines, list) or len(engines) != 1):
        return None
    engine = engines[0]
    if (not isinstance(engine, dict) or engine.get("utilization_percent") != 1
            or str(engine.get("process_name", "")).casefold() != "csrss.exe"):
        return None
    match = re.fullmatch(r"pid_(\d+)_.*_phys_0_eng_\d+_engtype_3D", str(engine.get("name", "")))
    if match is None:
        return None
    pid = int(match[1])
    expected = {f"unapproved_active_engine:{pid}:3D", f"unverified_active_engine_process:{pid}"}
    if len(reasons) != 2 or set(reasons) != expected:
        return None
    processes = snapshot.get("processes")
    if not isinstance(processes, list) or any(isinstance(row, dict) and row.get("pid") == pid for row in processes):
        return None
    return pid


def _safe_memory(snapshot: Mapping[str, Any], maximum: int) -> bool:
    memory = snapshot.get("memory_used_mib")
    return type(memory) is int and 0 <= memory <= maximum


def _clear_followup(snapshot: Mapping[str, Any], pid: int, maximum_memory: int) -> bool:
    if snapshot.get("busy") is not False or snapshot.get("busy_reasons") != []:
        return False
    if snapshot.get("utilization_percent") != 0 or snapshot.get("compute_process_count") != 0:
        return False
    if not _safe_memory(snapshot, maximum_memory):
        return False
    processes = snapshot.get("processes")
    engines = snapshot.get("active_engines")
    if not isinstance(processes, list) or not isinstance(engines, list):
        return False
    if any(isinstance(row, dict) and row.get("pid") == pid for row in processes):
        return False
    return not any(
        not isinstance(row, dict) or type(row.get("utilization_percent")) is not int
        or row["utilization_percent"] >= 2 or "compute" in str(row.get("name", "")).casefold()
        or "cuda" in str(row.get("name", "")).casefold()
        for row in engines
    )


def _receipt(samples: list[dict[str, Any]], identities: list[dict[str, Any]],
             *, accepted: bool, reason: str) -> dict[str, Any]:
    current = dict(samples[-1])
    current["bounded_resample"] = {
        "attempted": len(samples) > 1,
        "accepted": accepted,
        "reason": reason,
        "interval_s": RESAMPLE_INTERVAL_S,
        "max_samples": MAX_SAMPLES,
        "sample_count": len(samples),
        "required_consecutive_clear": 2,
        "identities": identities,
    }
    current["samples"] = samples
    return current


def snapshot(stage: Path, name: str) -> dict[str, Any]:
    """첫 strict 판정과, 유일하게 등록된 CSRSS 순간 계수기 모양만 제한 재표본한다."""
    samples = [query_gpu()]
    first = samples[0]
    if first.get("busy") is False:
        receipt = _receipt(samples, [], accepted=True, reason="initial_clear")
        write_new(stage / f"gpu_{name}.json", receipt)
        return receipt
    pid = _candidate_pid(first)
    identities: list[dict[str, Any]] = []
    maximum_memory = first.get("memory_used_mib")
    if pid is None or type(maximum_memory) is not int:
        receipt = _receipt(samples, identities, accepted=False, reason="strict_busy_not_registered_transient")
        write_new(stage / f"gpu_{name}.json", receipt)
        raise RuntimeError(f"GPU가 사용 중이다: {first.get('busy_reasons')}")
    try:
        before = process_identity(pid)
        identities.append(before)
        for _ in range(MAX_SAMPLES - 1):
            time.sleep(RESAMPLE_INTERVAL_S)
            current = query_gpu()
            samples.append(current)
            identity = process_identity(pid)
            identities.append(identity)
            if identity != before or not _clear_followup(current, pid, maximum_memory):
                raise RuntimeError("CSRSS 신원 또는 후속 strict GPU 표본이 등록 조건과 다르다")
    except Exception as error:
        receipt = _receipt(samples, identities, accepted=False, reason=type(error).__name__)
        write_new(stage / f"gpu_{name}.json", receipt)
        raise RuntimeError("등록된 CSRSS 순간 계수기 재표본을 통과하지 못했다") from error
    receipt = _receipt(samples, identities, accepted=True, reason="two_consecutive_strict_clear")
    write_new(stage / f"gpu_{name}.json", receipt)
    return receipt


@contextmanager
def gpu_admission() -> Iterator[None]:
    """동결 backend의 두 hook을 이 호출 동안만 바꾸고 예외 경로에서도 복원한다."""
    original_query, original_snapshot = FROZEN._query_gpu, FROZEN._snapshot
    FROZEN._query_gpu, FROZEN._snapshot = query_gpu, snapshot
    try:
        yield
    finally:
        FROZEN._query_gpu, FROZEN._snapshot = original_query, original_snapshot


def train(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.train(*args, **kwargs)


def inspect(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.inspect(*args, **kwargs)


def fixed_evaluate(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.fixed_evaluate(*args, **kwargs)


def legacy_probe(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.legacy_probe(*args, **kwargs)
