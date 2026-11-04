"""등록된 데스크톱 C+G 프로세스만 좁게 허용하는 GPU 입장 adapter."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

RETRY = importlib.import_module("lease_read_retry_v1")
LEASE = importlib.import_module("lineage_source_batch_v1")
GPU = importlib.import_module("record_evidence_media")

POLICY_PATH = ROOT / "bench/protocols/registered_desktop_gpu_admission_v1.json"
PROFILE_ID = "registered_desktop_shared_v2"
BASE_PROFILE = "desktop_shared_recording_v1"
_SCOPE_LOCK = threading.RLock()
_SCOPE_LOCAL = threading.local()
_REASON = re.compile(r"unapproved_gpu_process:(\d+)")


class AdmissionError(RuntimeError):
    """정책, lease 또는 GPU snapshot이 등록 계약과 다르다."""


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
        raise AdmissionError("GPU snapshot을 결정적으로 직렬화할 수 없다") from error


def _load_policy() -> tuple[dict[str, Any], str]:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise AdmissionError("등록 GPU 정책을 읽을 수 없다") from error
    expected = {
        "schema": "registered_desktop_gpu_admission_v1",
        "profile_id": PROFILE_ID,
        "base_profile": BASE_PROFILE,
        "gpu_index": 0,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "memory_used_limit_mib": 4096,
        "gpu_utilization_percent_max": 1,
        "active_engine_min_percent": 1,
        "allowed_process_types": ["C+G", "G"],
        "registered_process_type": "C+G",
        "unclassified_activity": "block",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise AdmissionError(f"등록 GPU 정책 필드가 다르다: {key}")
    names = policy.get("registered_desktop_process_names")
    if not isinstance(names, list) or names != sorted(set(names)) or any(
        type(name) is not str or name != name.casefold() for name in names
    ):
        raise AdmissionError("등록 데스크톱 프로세스 이름 집합이 잘못됐다")
    sources = policy.get("source_closure_sha256_lf")
    if not isinstance(sources, dict) or not sources:
        raise AdmissionError("등록 GPU 정책 source closure가 없다")
    for relative, expected_hash in sources.items():
        path = Path(relative)
        if (
            type(relative) is not str
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None
        ):
            raise AdmissionError("등록 GPU 정책 source closure 형식이 잘못됐다")
        absolute = ROOT / path
        if not absolute.is_file() or _sha256_lf(absolute) != expected_hash:
            raise AdmissionError(f"등록 GPU 정책 source closure가 다르다: {relative}")
    return policy, _sha256(raw)


def _validate_lease(lease_reader: Callable[[], dict[str, Any]] | None) -> dict[str, Any]:
    reader = lease_reader or LEASE.lease_context
    try:
        lease = RETRY.read_live_lease(reader)
    except Exception as error:
        raise AdmissionError("현재 caller의 공용 GPU lease를 검증하지 못했다") from error
    if not isinstance(lease, dict):
        raise AdmissionError("공용 GPU lease receipt 형식이 잘못됐다")
    return deepcopy(lease)


def _utc(clock: Callable[[], Any] | None) -> str:
    value = (clock or (lambda: datetime.now(UTC)))()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AdmissionError("clock은 timezone-aware datetime을 반환해야 한다")
    return value.astimezone(UTC).isoformat()


def _identity_evidence(xml_text: str, windows: Mapping[str, Any]) -> dict[str, Any]:
    """같은 fresh 관측의 NVIDIA XML과 CIM identity 비교 결과를 만든다."""
    try:
        devices = ET.fromstring(xml_text).findall("gpu")
        if len(devices) != 1:
            raise ValueError("expected one GPU")
        rows = windows["processes"]
        if not isinstance(rows, list):
            raise TypeError("processes must be a list")
        cim: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("process row must be a mapping")
            pid = row.get("ProcessId")
            if type(pid) is not int or pid <= 0 or pid in cim:
                raise ValueError("duplicate or invalid CIM PID")
            cim[pid] = row
    except (ET.ParseError, KeyError, TypeError, ValueError) as error:
        raise AdmissionError("GPU identity 관측 형식이 잘못됐다") from error

    observation_hash = _sha256(_canonical({"xml": xml_text, "windows": windows}))
    evidence: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in devices[0].findall("processes/process_info"):
        try:
            pid = int(item.findtext("pid", ""))
            if pid <= 0 or pid in seen:
                raise ValueError("duplicate or invalid XML PID")
            seen.add(pid)
            row = cim[pid]
            cim_name = str(row.get("Name", "")).casefold()
            command = row.get("CommandLine")
            xml_path = (item.findtext("process_name") or "").replace("\\", "/").casefold()
            cim_path = str(row.get("ExecutablePath") or "").replace("\\", "/").casefold()
            xml_name = PureWindowsPath(xml_path).name.casefold()
            cim_path_name = PureWindowsPath(cim_path).name.casefold()
        except (KeyError, TypeError, ValueError) as error:
            raise AdmissionError("GPU identity PID 결합이 잘못됐다") from error
        paths_equal = bool(xml_path and cim_path and xml_path == cim_path)
        names_equal = bool(cim_name and cim_name == xml_name == cim_path_name)
        command_present = isinstance(command, str) and bool(command)
        evidence.append(
            {
                "pid": pid,
                "cim_name": cim_name,
                "xml_path_basename": xml_name,
                "cim_path_basename": cim_path_name,
                "xml_path_sha256": _sha256(xml_path.encode("utf-8")),
                "cim_path_sha256": _sha256(cim_path.encode("utf-8")),
                "paths_equal": paths_equal,
                "names_equal": names_equal,
                "command_present": command_present,
                "verified": paths_equal and names_equal and command_present,
                "observation_sha256": observation_hash,
            }
        )
    return {"observation_sha256": observation_hash, "processes": evidence}


def _query_gpu_with_identity() -> dict[str, Any]:
    """한 번의 XML+CIM 묶음으로 기반 분류와 identity proof를 함께 만든다."""
    xml_text = GPU._gpu_command(["nvidia-smi", "-q", "-x"])
    try:
        pids = sorted(
            {
                int(item.text or "")
                for item in ET.fromstring(xml_text).findall("gpu/processes/process_info/pid")
            }
        )
    except (ET.ParseError, ValueError) as error:
        raise AdmissionError("GPU process inventory가 잘못됐다") from error
    pid_literals = ",".join(str(pid) for pid in pids)
    script = (
        "$ErrorActionPreference='Stop'; "
        "$engines=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | "
        "Where-Object {$_.UtilizationPercentage -ge 1} | Select-Object Name,UtilizationPercentage); "
        f"$taskGpuPids=@({pid_literals}); "
        "$engines | ForEach-Object {if($_.Name -match '^pid_(\\d+)_') {$taskGpuPids += [int]$Matches[1]}}; "
        "$processes=@(Get-CimInstance Win32_Process | Where-Object {$_.ProcessId -in $taskGpuPids} | "
        "Select-Object ProcessId,Name,ExecutablePath,CommandLine); "
        "ConvertTo-Json -InputObject @{processes=$processes; engines=$engines} -Depth 4 -Compress"
    )
    try:
        windows = json.loads(
            GPU._gpu_command(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
            )
        )
        raw = GPU.classify_gpu(xml_text, windows, BASE_PROFILE)
    except Exception as error:
        raise AdmissionError("기반 GPU snapshot과 identity 관측을 만들지 못했다") from error
    return {"snapshot": raw, "identity_evidence": _identity_evidence(xml_text, windows)}


def _evaluate(
    raw: Mapping[str, Any], policy: Mapping[str, Any], identity: Any
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required_scalars = {
        "admission_policy": BASE_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "gpu_index": policy["gpu_index"],
        "compute_process_count": 0,
    }
    for key, expected in required_scalars.items():
        if raw.get(key) != expected or type(raw.get(key)) is not type(expected):
            reasons.append(f"invalid_{key}")

    busy = raw.get("busy")
    raw_reasons = raw.get("busy_reasons")
    if type(busy) is not bool or not isinstance(raw_reasons, list) or any(
        type(reason) is not str for reason in raw_reasons
    ):
        reasons.append("invalid_busy_contract")
        raw_reasons = []
    elif busy is not bool(raw_reasons):
        reasons.append("inconsistent_busy_contract")
    if len(raw_reasons) != len(set(raw_reasons)):
        reasons.append("duplicate_busy_reason")

    utilization = raw.get("utilization_percent")
    if type(utilization) is not int or not 0 <= utilization <= policy["gpu_utilization_percent_max"]:
        reasons.append("gpu_utilization_above_registered_limit")
    memory = raw.get("memory_used_mib")
    if type(memory) is not int or not 0 <= memory < policy["memory_used_limit_mib"]:
        reasons.append("memory_headroom_insufficient")

    active = raw.get("active_engines")
    if not isinstance(active, list) or active:
        reasons.append("active_gpu_engine_present")
    processes = raw.get("processes")
    by_pid: dict[int, Mapping[str, Any]] = {}
    if not isinstance(processes, list):
        reasons.append("invalid_process_inventory")
        processes = []
    for row in processes:
        if not isinstance(row, Mapping):
            reasons.append("invalid_process_row")
            continue
        pid = row.get("pid")
        name = row.get("name")
        process_type = row.get("type")
        known = row.get("known_workload")
        if type(pid) is not int or pid <= 0 or pid in by_pid:
            reasons.append("duplicate_or_invalid_process_pid")
            continue
        by_pid[pid] = row
        if type(name) is not str or not name or name != name.casefold():
            reasons.append(f"invalid_process_name:{pid}")
        if process_type not in policy["allowed_process_types"]:
            reasons.append(f"unapproved_process_type:{pid}")
        if type(known) is not bool or known:
            reasons.append(f"known_or_unclassified_workload:{pid}")

    registered = frozenset(policy["registered_desktop_process_names"])
    identity_by_pid: dict[int, Mapping[str, Any]] = {}
    identity_observation = None
    if isinstance(identity, Mapping):
        identity_observation = identity.get("observation_sha256")
        identity_rows = identity.get("processes")
    else:
        identity_rows = None
    if (
        not isinstance(identity_observation, str)
        or re.fullmatch(r"[a-f0-9]{64}", identity_observation) is None
        or not isinstance(identity_rows, list)
    ):
        reasons.append("process_identity_evidence_missing")
        identity_rows = []
    for identity_row in identity_rows:
        if not isinstance(identity_row, Mapping):
            reasons.append("invalid_process_identity_row")
            continue
        identity_pid = identity_row.get("pid")
        if type(identity_pid) is not int or identity_pid <= 0 or identity_pid in identity_by_pid:
            reasons.append("duplicate_or_invalid_identity_pid")
            continue
        identity_by_pid[identity_pid] = identity_row
    for reason in raw_reasons:
        match = _REASON.fullmatch(reason)
        if match is None:
            reasons.append(f"unapproved_base_reason:{reason}")
            continue
        pid = int(match.group(1))
        row = by_pid.get(pid)
        if row is None:
            reasons.append(f"missing_process_for_reason:{pid}")
            continue
        if (
            row.get("name") not in registered
            or row.get("type") != policy["registered_process_type"]
            or row.get("known_workload") is not False
        ):
            reasons.append(f"registered_exception_not_applicable:{pid}")
            continue
        proof = identity_by_pid.get(pid)
        if proof is None:
            reasons.append(f"process_identity_evidence_missing:{pid}")
            continue
        if (
            proof.get("observation_sha256") != identity_observation
            or proof.get("cim_name") != row.get("name")
            or proof.get("xml_path_basename") != row.get("name")
            or proof.get("cim_path_basename") != row.get("name")
            or proof.get("paths_equal") is not True
            or proof.get("names_equal") is not True
            or proof.get("command_present") is not True
            or proof.get("verified") is not True
        ):
            reasons.append(f"process_identity_not_verified:{pid}")

    if raw.get("c_plus_g_contexts_present") is not any(
        row.get("type") == "C+G" for row in by_pid.values()
    ):
        reasons.append("invalid_c_plus_g_summary")
    unique = list(dict.fromkeys(reasons))
    return not unique, unique


def query_registered_desktop_gpu(
    *,
    lease_reader: Callable[[], dict[str, Any]] | None = None,
    gpu_reader: Callable[[], dict[str, Any]] | None = None,
    clock: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """공용 lease와 새 기반 snapshot을 매번 검증해 등록 판정 receipt를 만든다."""
    policy, policy_hash = _load_policy()
    lease = _validate_lease(lease_reader)
    reader = gpu_reader or _query_gpu_with_identity
    try:
        observed = reader()
    except Exception as error:
        raise AdmissionError("기반 GPU snapshot을 읽지 못했다") from error
    if not isinstance(observed, dict):
        raise AdmissionError("기반 GPU snapshot 묶음 형식이 잘못됐다")
    if "snapshot" in observed or "identity_evidence" in observed:
        raw_value = observed.get("snapshot")
        identity = deepcopy(observed.get("identity_evidence"))
    else:
        raw_value = observed
        identity = None
    if not isinstance(raw_value, dict):
        raise AdmissionError("기반 GPU snapshot 형식이 잘못됐다")
    raw = deepcopy(raw_value)
    raw_hash = _sha256(_canonical(raw))
    accepted, reasons = _evaluate(raw, policy, identity)
    lease_digest = _sha256(_canonical(lease))
    return {
        "busy": not accepted,
        "busy_reasons": reasons,
        "admission_policy": PROFILE_ID,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "observed_at_utc": _utc(clock),
        "gpu_index": policy["gpu_index"],
        "utilization_percent": raw.get("utilization_percent"),
        "memory_used_mib": raw.get("memory_used_mib"),
        "compute_process_count": raw.get("compute_process_count"),
        "processes": deepcopy(raw.get("processes")),
        "active_engines": deepcopy(raw.get("active_engines")),
        "c_plus_g_contexts_present": raw.get("c_plus_g_contexts_present"),
        "decision": "accepted" if accepted else "rejected",
        "decision_reasons": reasons,
        "scope": "fixed_and_training_snapshot_only",
        "raw_snapshot": raw,
        "raw_snapshot_sha256": raw_hash,
        "process_identity_evidence": identity,
        "process_identity_evidence_sha256": (
            _sha256(_canonical(identity)) if isinstance(identity, Mapping) else None
        ),
        "policy_sha256": policy_hash,
        "lease_receipt_sha256": lease_digest,
        "caller_pid": os.getpid(),
        "fixture_callbacks_used": lease_reader is not None or gpu_reader is not None,
        "actual_gpu_admission_evidence": False,
        "admission_evidence_requires_external_execution_receipt": True,
    }


def _canonical_file(module: Any) -> Path:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        raise AdmissionError("R1 module 파일 경로가 없다")
    return Path(path).resolve()


def _r1_modules() -> list[Any]:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    v2 = importlib.import_module("evals.r1_repair_execution_v2")
    canonical = _canonical_file(package)
    modules: list[Any] = []
    for module in (package, top_level, v2.V1):
        if _canonical_file(module) != canonical:
            raise AdmissionError("R1 alias가 같은 정본 파일을 가리키지 않는다")
        if all(module is not current for current in modules):
            modules.append(module)
    return modules


@contextmanager
def snapshot_scope() -> Iterator[None]:
    """R1 snapshot query alias만 한정된 수명 동안 등록 query로 바꾼다."""
    with _SCOPE_LOCK:
        depth = getattr(_SCOPE_LOCAL, "depth", 0)
        if depth:
            _SCOPE_LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _SCOPE_LOCAL.depth -= 1
            return

        modules = _r1_modules()
        originals = [(module, module._query_gpu) for module in modules]

        def replacement() -> dict[str, Any]:
            return query_registered_desktop_gpu()

        _SCOPE_LOCAL.depth = 1
        try:
            for module, _original in originals:
                module._query_gpu = replacement
            yield
        finally:
            for module, original in originals:
                module._query_gpu = original
            _SCOPE_LOCAL.depth = 0
