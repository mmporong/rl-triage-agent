# pyright: reportMissingImports=false
"""D4의 processed GPU receipt를 동결 classifier 규칙으로 다시 확인한다.

원시 NVIDIA XML·CIM 행은 receipt에 없으므로 이 검증은 저장된 processes·active_engines와
classifier가 만든 known_workload 필드 사이의 모순을 fail-closed로 찾는다.
"""
from __future__ import annotations

import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import export_ppo_ratio_repair_v3 as FROZEN_EXPORT  # noqa: E402
import ppo_ratio_repair_execution_v3 as EXECUTION_V3  # noqa: E402
import record_evidence_media as CLASSIFIER  # noqa: E402


def _processed_clear(sample: dict[str, Any]) -> None:
    admission = EXECUTION_V3.V2.GPU_ADMISSION
    allowed_names = frozenset(admission["allowed_display_process_names"])
    allowed_types = frozenset(admission["allowed_process_types"])
    allowed_engines = frozenset(admission["allowed_active_engine_types"])
    memory = sample.get("memory_used_mib")
    utilization = sample.get("utilization_percent")
    processes = sample.get("processes")
    engines = sample.get("active_engines")
    if (
        sample.get("admission_policy") != EXECUTION_V3.GPU_PROFILE
        or type(memory) is not int
        or not 0 <= memory < admission["memory_used_limit_mib"]
        or type(utilization) is not int
        or not 0 <= utilization <= 100
        or not isinstance(processes, list)
        or not isinstance(engines, list)
    ):
        raise ValueError("processed GPU classifier 기본 필드가 등록 profile과 다르다")
    observed: dict[int, dict[str, Any]] = {}
    known_count = 0
    reasons: list[str] = []
    for row in processes:
        if not isinstance(row, dict) or set(row) != {"pid", "name", "type", "known_workload"}:
            raise ValueError("processed GPU process 행 형식이 다르다")
        pid, name, process_type, known = row["pid"], row["name"], row["type"], row["known_workload"]
        if (
            type(pid) is not int
            or pid <= 0
            or pid in observed
            or not isinstance(name, str)
            or name != name.casefold()
            or not isinstance(process_type, str)
            or type(known) is not bool
        ):
            raise ValueError("processed GPU process identity가 잘못됐다")
        name_workload = bool(CLASSIFIER.GPU_WORKLOAD_RE.search(name))
        if known is not name_workload and (known or name_workload):
            raise ValueError("processed known_workload와 process 이름이 모순된다")
        known_count += int(known)
        if known or name_workload or name not in allowed_names:
            reasons.append(f"unapproved_gpu_process:{pid}")
        if process_type not in allowed_types:
            reasons.append(f"unapproved_process_type:{pid}:{process_type}")
        observed[pid] = row
    if sample.get("compute_process_count") != known_count or known_count != 0:
        raise ValueError("processed compute process 집계가 0과 다르다")
    for row in engines:
        if not isinstance(row, dict) or set(row) != {"name", "utilization_percent", "process_name"}:
            raise ValueError("processed GPU engine 행 형식이 다르다")
        raw_name, active, process_name = row["name"], row["utilization_percent"], row["process_name"]
        if not isinstance(raw_name, str) or type(active) is not int or not 1 <= active <= 100:
            raise ValueError("processed GPU engine 값이 잘못됐다")
        match = re.fullmatch(r"pid_(\d+)_.*_phys_0_eng_\d+_engtype_(.+)", raw_name)
        if match is None:
            raise ValueError("processed GPU engine 이름을 분류할 수 없다")
        pid, engine_type = int(match[1]), match[2]
        process = observed.get(pid)
        if (
            not isinstance(process_name, str)
            or process_name != process_name.casefold()
            or process is None
            or process_name != process["name"]
        ):
            raise ValueError("processed GPU engine PID와 process 행이 대응하지 않는다")
        if engine_type not in allowed_engines or process_name not in allowed_names:
            reasons.append(f"unapproved_active_engine:{pid}:{engine_type}")
        if process["known_workload"] or CLASSIFIER.GPU_WORKLOAD_RE.search(process_name):
            reasons.append(f"unverified_active_engine_process:{pid}")
    if sample.get("busy") is not False or sample.get("busy_reasons") != [] or reasons:
        raise ValueError("processed evidence를 다시 계산한 clear 판정과 receipt flag가 다르다")


def verify_gpu_receipt(value: dict[str, Any]) -> str:
    """bounded 구조 검증 뒤 processed classifier evidence를 다시 계산한다."""
    status = FROZEN_EXPORT._validate_gpu_receipt(value)
    samples = value["samples"]
    if status == "initial_clear":
        _processed_clear(samples[0])
        return status
    if status != "resampled_clear":
        raise ValueError("GPU bounded admission이 stage를 허용하지 않았다")
    first = samples[0]
    pid = EXECUTION_V3._candidate_pid(first)
    if pid is None or CLASSIFIER.GPU_WORKLOAD_RE.search(
        " ".join(str(row.get("process_name", "")) for row in first.get("active_engines", []) if isinstance(row, dict))
    ):
        raise ValueError("첫 CSRSS transient 표본이 등록 모양과 다르다")
    # 등록 예외는 해당 CSRSS engine 행 하나에만 적용한다. 첫 표본의 다른
    # process·engine은 최종 clear 표본과 같은 processed classifier 규칙으로 재계산한다.
    remaining = deepcopy(first)
    remaining["active_engines"] = [
        row for row in first["active_engines"]
        if not (isinstance(row, dict) and re.match(rf"pid_{pid}_", str(row.get("name", ""))))
    ]
    remaining["busy"] = False
    remaining["busy_reasons"] = []
    _processed_clear(remaining)
    for sample in samples[1:]:
        _processed_clear(sample)
        if any(
            isinstance(row, dict) and re.match(rf"pid_{pid}_", str(row.get("name", "")))
            for row in sample["active_engines"]
        ):
            raise ValueError("후속 clear 표본에 transient PID engine이 남았다")
    return status
