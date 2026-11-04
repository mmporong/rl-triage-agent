"""기존 desktop-shared 기준을 보존하는 opt-in GPU admission profile."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "bench/protocols/registered_desktop_gpu_shared_v1.json"
PROFILE_ID = "registered_desktop_shared_baseline_v1"
QUIET_ONLY_REASONS = {
    "gpu_utilization_above_registered_limit",
    "active_gpu_engine_present",
}
_ENGINE = re.compile(r"pid_(\d+)_.*_phys_0_eng_\d+_engtype_(.+)\Z")

GPU1 = importlib.import_module("evals.registered_desktop_gpu_admission_v1")


class SharedAdmissionError(RuntimeError):
    """공유 GPU 정책 또는 기존 admission receipt 결합이 잘못됐다."""


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
        raise SharedAdmissionError("GPU receipt를 결정적으로 직렬화할 수 없다") from error


def _load_policy() -> tuple[dict[str, Any], str]:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SharedAdmissionError("등록 shared GPU 정책을 읽을 수 없다") from error
    expected = {
        "schema": "registered_desktop_gpu_shared_v1",
        "profile_id": PROFILE_ID,
        "base_profile_id": GPU1.PROFILE_ID,
        "base_recording_profile": GPU1.BASE_PROFILE,
        "gpu_index": 0,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "nonexclusive_cost_observation": True,
        "threshold_relaxation": True,
        "memory_used_limit_mib": 4096,
        "allowed_process_types": ["C+G", "G"],
        "allowed_active_engine_types": ["3D", "Copy", "VideoDecode", "VideoEncode"],
        "excluded_quiet_reasons": sorted(QUIET_ONLY_REASONS),
        "unclassified_activity": "block",
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise SharedAdmissionError(f"등록 shared GPU 정책 필드가 다르다: {key}")
    base = policy.get("base_policy_reference")
    if not isinstance(base, dict) or base != {
        "path": "bench/protocols/registered_desktop_gpu_admission_v1.json",
        "sha256_lf": _sha256_lf(GPU1.POLICY_PATH),
    }:
        raise SharedAdmissionError("기반 GPU 정책 reference가 다르다")
    sources = policy.get("source_closure_sha256_lf")
    if not isinstance(sources, dict) or not sources:
        raise SharedAdmissionError("shared GPU 정책 source closure가 없다")
    for relative, expected_hash in sources.items():
        path = Path(relative)
        if (
            type(relative) is not str
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None
        ):
            raise SharedAdmissionError("shared GPU 정책 source closure 형식이 잘못됐다")
        absolute = ROOT / path
        if not absolute.is_file() or _sha256_lf(absolute) != expected_hash:
            raise SharedAdmissionError(f"shared GPU 정책 source closure가 다르다: {relative}")
    return policy, _sha256(raw)


def _base_allowed_names(policy: Mapping[str, Any]) -> frozenset[str]:
    base = GPU1.GPU.EXPECTED_GPU_ADMISSION
    base_names = base.get("allowed_display_process_names")
    registered = policy.get("registered_desktop_process_names")
    if not isinstance(base_names, list) or not isinstance(registered, list):
        raise SharedAdmissionError("기반 desktop process 이름 계약이 없다")
    return frozenset([*base_names, *registered])


def _validate_raw_shared_activity(
    raw: Mapping[str, Any],
    identity: Any,
    base_policy: Mapping[str, Any],
    shared_policy: Mapping[str, Any],
) -> tuple[set[str], list[dict[str, Any]]]:
    required = {
        "admission_policy": GPU1.BASE_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "gpu_index": 0,
        "compute_process_count": 0,
    }
    for key, expected in required.items():
        if raw.get(key) != expected or type(raw.get(key)) is not type(expected):
            raise SharedAdmissionError(f"기반 raw scalar가 다르다: {key}")
    utilization = raw.get("utilization_percent")
    memory = raw.get("memory_used_mib")
    if type(utilization) is not int or not 0 <= utilization <= 100:
        raise SharedAdmissionError("GPU utilization scalar가 0~100 정수가 아니다")
    if type(memory) is not int or not 0 <= memory < shared_policy["memory_used_limit_mib"]:
        raise SharedAdmissionError("GPU memory headroom이 shared 정책 범위 밖이다")

    processes = raw.get("processes")
    if not isinstance(processes, list):
        raise SharedAdmissionError("GPU process inventory가 list가 아니다")
    allowed_names = _base_allowed_names(base_policy)
    by_pid: dict[int, Mapping[str, Any]] = {}
    for row in processes:
        if not isinstance(row, Mapping):
            raise SharedAdmissionError("GPU process 행이 object가 아니다")
        pid, name = row.get("pid"), row.get("name")
        if type(pid) is not int or pid <= 0 or pid in by_pid:
            raise SharedAdmissionError("GPU process PID가 없거나 중복됐다")
        if type(name) is not str or not name or name != name.casefold():
            raise SharedAdmissionError(f"GPU process 이름 형식이 다르다: {pid}")
        if name not in allowed_names:
            raise SharedAdmissionError(f"GPU process 이름이 shared 등록 범위 밖이다: {pid}")
        if row.get("type") not in shared_policy["allowed_process_types"]:
            raise SharedAdmissionError(f"GPU process type이 허용되지 않았다: {pid}")
        if row.get("known_workload") is not False:
            raise SharedAdmissionError(f"GPU process가 known 또는 미분류 workload다: {pid}")
        by_pid[pid] = row
    if raw.get("c_plus_g_contexts_present") is not any(
        row.get("type") == "C+G" for row in by_pid.values()
    ):
        raise SharedAdmissionError("C+G summary가 process inventory와 다르다")

    engines = raw.get("active_engines")
    if not isinstance(engines, list):
        raise SharedAdmissionError("active engine inventory가 list가 아니다")
    engine_by_pid_and_type: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in engines:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "utilization_percent",
            "process_name",
        }:
            raise SharedAdmissionError("active engine 행 형식이 다르다")
        raw_name = row.get("name")
        utilization_percent = row.get("utilization_percent")
        process_name = row.get("process_name")
        if not isinstance(raw_name, str):
            raise SharedAdmissionError("active engine name이 문자열이 아니다")
        match = _ENGINE.fullmatch(raw_name)
        if match is None:
            raise SharedAdmissionError("active engine name pattern이 다르다")
        pid, engine_type = int(match.group(1)), match.group(2)
        process = by_pid.get(pid)
        if process is None:
            raise SharedAdmissionError(f"active engine PID가 GPU process에 없다: {pid}")
        if engine_type not in shared_policy["allowed_active_engine_types"]:
            raise SharedAdmissionError(f"active engine type이 허용되지 않았다: {pid}")
        if (
            type(utilization_percent) is not int
            or not 1 <= utilization_percent <= 100
            or process_name != process.get("name")
            or process.get("known_workload") is not False
        ):
            raise SharedAdmissionError(f"active engine process 결합이 다르다: {pid}")
        key = (pid, engine_type)
        if key in engine_by_pid_and_type:
            raise SharedAdmissionError(f"active engine PID/type이 중복됐다: {pid}:{engine_type}")
        engine_by_pid_and_type[key] = row

    identity_observation = identity.get("observation_sha256") if isinstance(identity, Mapping) else None
    identity_rows = identity.get("processes") if isinstance(identity, Mapping) else None
    identity_by_pid: dict[int, Mapping[str, Any]] = {}
    if isinstance(identity_rows, list):
        for row in identity_rows:
            if not isinstance(row, Mapping):
                continue
            pid = row.get("pid")
            if type(pid) is int and pid > 0 and pid not in identity_by_pid:
                identity_by_pid[pid] = row
    registered_names = frozenset(base_policy["registered_desktop_process_names"])
    exceptions: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for reason in raw.get("busy_reasons", []):
        if not isinstance(reason, str):
            continue
        match = re.fullmatch(r"unapproved_active_engine:(\d+):([^:]+)", reason)
        if match is None:
            continue
        pid, engine_type = int(match.group(1)), match.group(2)
        process = by_pid.get(pid)
        proof = identity_by_pid.get(pid)
        engine = engine_by_pid_and_type.get((pid, engine_type))
        if (
            process is None
            or engine is None
            or process.get("name") not in registered_names
            or process.get("type") != "C+G"
            or process.get("known_workload") is not False
            or engine_type not in shared_policy["allowed_active_engine_types"]
            or proof is None
            or proof.get("observation_sha256") != identity_observation
            or proof.get("cim_name") != process.get("name")
            or proof.get("xml_path_basename") != process.get("name")
            or proof.get("cim_path_basename") != process.get("name")
            or proof.get("paths_equal") is not True
            or proof.get("names_equal") is not True
            or proof.get("command_present") is not True
            or proof.get("verified") is not True
        ):
            continue
        wrapped = f"unapproved_base_reason:{reason}"
        exceptions.add(wrapped)
        evidence.append(
            {
                "reason": wrapped,
                "pid": pid,
                "process_name": process.get("name"),
                "process_type": process.get("type"),
                "engine_type": engine_type,
                "engine_name": engine.get("name"),
                "engine_utilization_percent": engine.get("utilization_percent"),
                "identity_observation_sha256": identity_observation,
                "identity_verified": True,
            }
        )
    return exceptions, evidence


def _apply_shared_profile(base_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """검증된 GPU1 receipt에 shared baseline 판정을 적용한다."""
    shared_policy, shared_policy_hash = _load_policy()
    base_policy, base_policy_hash = GPU1._load_policy()
    if not isinstance(base_receipt, Mapping):
        raise SharedAdmissionError("기반 GPU receipt가 object가 아니다")
    raw = base_receipt.get("raw_snapshot")
    identity = base_receipt.get("process_identity_evidence")
    if not isinstance(raw, Mapping):
        raise SharedAdmissionError("기반 GPU receipt에 raw snapshot이 없다")
    raw_hash = _sha256(_canonical(raw))
    identity_hash = _sha256(_canonical(identity)) if isinstance(identity, Mapping) else None
    if (
        base_receipt.get("admission_policy") != GPU1.PROFILE_ID
        or base_receipt.get("base_policy_reference") is not None
        or base_receipt.get("policy_sha256") != base_policy_hash
        or base_receipt.get("raw_snapshot_sha256") != raw_hash
        or base_receipt.get("process_identity_evidence_sha256") != identity_hash
        or re.fullmatch(r"[a-f0-9]{64}", str(base_receipt.get("lease_receipt_sha256"))) is None
        or type(base_receipt.get("fixture_callbacks_used")) is not bool
        or base_receipt.get("actual_gpu_admission_evidence") is not False
    ):
        raise SharedAdmissionError("기반 GPU receipt 결합이 다르다")
    for key in (
        "gpu_index",
        "utilization_percent",
        "memory_used_mib",
        "compute_process_count",
        "processes",
        "active_engines",
        "c_plus_g_contexts_present",
    ):
        if base_receipt.get(key) != raw.get(key):
            raise SharedAdmissionError(f"기반 GPU receipt와 raw snapshot이 다르다: {key}")
    recomputed_accepted, recomputed_reasons = GPU1._evaluate(raw, base_policy, identity)
    recorded_reasons = base_receipt.get("decision_reasons")
    if (
        not isinstance(recorded_reasons, list)
        or recorded_reasons != recomputed_reasons
        or base_receipt.get("busy_reasons") != recomputed_reasons
        or base_receipt.get("busy") is not (not recomputed_accepted)
        or base_receipt.get("decision") != ("accepted" if recomputed_accepted else "rejected")
    ):
        raise SharedAdmissionError("기반 GPU decision을 재현할 수 없다")
    registered_engine_exceptions, registered_engine_evidence = _validate_raw_shared_activity(
        raw, identity, base_policy, shared_policy
    )
    allowed_exclusions = QUIET_ONLY_REASONS | registered_engine_exceptions
    excluded = [reason for reason in recomputed_reasons if reason in allowed_exclusions]
    retained = [reason for reason in recomputed_reasons if reason not in allowed_exclusions]
    result = deepcopy(dict(base_receipt))
    result["base_admission_policy"] = GPU1.PROFILE_ID
    result["admission_policy"] = PROFILE_ID
    result["base_policy_reference"] = deepcopy(shared_policy["base_policy_reference"])
    result["shared_policy_sha256"] = shared_policy_hash
    result["base_decision"] = base_receipt.get("decision")
    result["base_decision_reasons"] = deepcopy(recomputed_reasons)
    result["shared_reason_exclusions"] = excluded
    result["registered_gui_active_engine_exceptions"] = registered_engine_evidence
    result["busy"] = bool(retained)
    result["busy_reasons"] = retained
    result["decision_reasons"] = deepcopy(retained)
    result["decision"] = "rejected" if retained else "accepted"
    result["shared_mode_basis"] = GPU1.BASE_PROFILE
    result["gpu_exclusive"] = False
    result["performance_measurement"] = False
    result["nonexclusive_cost_observation"] = True
    result["threshold_relaxation"] = True
    result["quiet_thresholds_are_performance_evidence"] = False
    result["raw_snapshot"] = deepcopy(dict(raw))
    result["raw_snapshot_sha256"] = raw_hash
    result["actual_gpu_admission_evidence"] = False
    result["admission_evidence_requires_external_execution_receipt"] = True
    return result


def query_registered_desktop_gpu_shared() -> dict[str, Any]:
    """공용 lease와 fresh XML+CIM을 검증한 뒤 shared profile을 적용한다."""
    return _apply_shared_profile(GPU1.query_registered_desktop_gpu())


@contextmanager
def snapshot_scope() -> Iterator[None]:
    """GPU1 alias scope 안에서 R1 query만 shared 판으로 교체한다."""
    with GPU1.snapshot_scope():
        modules = GPU1._r1_modules()
        originals = [(module, module._query_gpu) for module in modules]

        def replacement() -> dict[str, Any]:
            return query_registered_desktop_gpu_shared()

        try:
            for module, _original in originals:
                module._query_gpu = replacement
            yield
        finally:
            for module, original in originals:
                module._query_gpu = original
