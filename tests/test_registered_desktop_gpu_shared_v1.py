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

import evals.registered_desktop_gpu_shared_v1 as S


def raw_snapshot(
    *,
    utilization: Any = 2,
    memory: Any = 1400,
    compute: Any = 0,
    pid: int = 4100,
    name: str = "chrome.exe",
    process_type: str = "C+G",
    known: Any = False,
    engine_type: str = "VideoDecode",
    engine_pid: int | None = None,
    engine_process_name: str | None = None,
    engine_utilization: Any = 5,
    with_engine: bool = True,
) -> dict[str, Any]:
    engines = []
    if with_engine:
        target_pid = pid if engine_pid is None else engine_pid
        engines.append(
            {
                "name": f"pid_{target_pid}_luid_0x1_phys_0_eng_0_engtype_{engine_type}",
                "utilization_percent": engine_utilization,
                "process_name": name if engine_process_name is None else engine_process_name,
            }
        )
    return {
        "busy": False,
        "busy_reasons": [],
        "admission_policy": S.GPU1.BASE_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "observed_at_utc": "2026-11-04T00:00:00+00:00",
        "gpu_index": 0,
        "utilization_percent": utilization,
        "memory_used_mib": memory,
        "compute_process_count": compute,
        "processes": [
            {
                "pid": pid,
                "name": name,
                "type": process_type,
                "known_workload": known,
            }
        ],
        "active_engines": engines,
        "c_plus_g_contexts_present": process_type == "C+G",
    }


def identity(*, pid: int | None = None, name: str | None = None) -> dict[str, Any]:
    observation = "a" * 64
    rows = []
    if pid is not None and name is not None:
        rows.append(
            {
                "pid": pid,
                "cim_name": name,
                "xml_path_basename": name,
                "cim_path_basename": name,
                "xml_path_sha256": "b" * 64,
                "cim_path_sha256": "b" * 64,
                "paths_equal": True,
                "names_equal": True,
                "command_present": True,
                "verified": True,
                "observation_sha256": observation,
            }
        )
    return {"observation_sha256": observation, "processes": rows}


def base_receipt(raw: dict[str, Any], proof: dict[str, Any] | None = None) -> dict[str, Any]:
    base_policy, policy_hash = S.GPU1._load_policy()
    evidence = proof or identity()
    accepted, reasons = S.GPU1._evaluate(raw, base_policy, evidence)
    return {
        "busy": not accepted,
        "busy_reasons": list(reasons),
        "decision_reasons": list(reasons),
        "decision": "accepted" if accepted else "rejected",
        "admission_policy": S.GPU1.PROFILE_ID,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "gpu_index": raw.get("gpu_index"),
        "utilization_percent": raw.get("utilization_percent"),
        "memory_used_mib": raw.get("memory_used_mib"),
        "compute_process_count": raw.get("compute_process_count"),
        "processes": deepcopy(raw.get("processes")),
        "active_engines": deepcopy(raw.get("active_engines")),
        "c_plus_g_contexts_present": raw.get("c_plus_g_contexts_present"),
        "raw_snapshot": deepcopy(raw),
        "raw_snapshot_sha256": S._sha256(S._canonical(raw)),
        "process_identity_evidence": deepcopy(evidence),
        "process_identity_evidence_sha256": S._sha256(S._canonical(evidence)),
        "policy_sha256": policy_hash,
        "lease_receipt_sha256": "c" * 64,
        "fixture_callbacks_used": True,
        "actual_gpu_admission_evidence": False,
        "admission_evidence_requires_external_execution_receipt": True,
    }


def test_shared_profile_accepts_only_verified_quiet_reasons_and_preserves_raw() -> None:
    raw = raw_snapshot()
    base = base_receipt(raw)
    expected_quiet = ["gpu_utilization_above_registered_limit", "active_gpu_engine_present"]
    assert base["decision_reasons"] == expected_quiet
    result = S._apply_shared_profile(base)
    assert result["busy"] is False
    assert result["decision"] == "accepted"
    assert result["admission_policy"] == S.PROFILE_ID
    assert result["base_admission_policy"] == S.GPU1.PROFILE_ID
    assert result["base_decision"] == "rejected"
    assert result["base_decision_reasons"] == expected_quiet
    assert result["shared_reason_exclusions"] == expected_quiet
    assert result["threshold_relaxation"] is True
    assert result["nonexclusive_cost_observation"] is True
    assert result["performance_measurement"] is False
    assert result["raw_snapshot"] == raw
    assert result["raw_snapshot_sha256"] == base["raw_snapshot_sha256"]
    assert result["actual_gpu_admission_evidence"] is False


def test_registered_gui_exception_still_requires_same_observation_identity() -> None:
    pid, name = 4200, "applicationframehost.exe"
    raw = raw_snapshot(pid=pid, name=name, engine_type="3D")
    raw["busy"] = True
    raw["busy_reasons"] = [
        f"unapproved_gpu_process:{pid}",
        f"unapproved_active_engine:{pid}:3D",
    ]
    accepted = S._apply_shared_profile(base_receipt(raw, identity(pid=pid, name=name)))
    assert accepted["busy"] is False
    engine_reason = (
        f"unapproved_base_reason:unapproved_active_engine:{pid}:3D"
    )
    assert engine_reason in accepted["shared_reason_exclusions"]
    assert accepted["registered_gui_active_engine_exceptions"] == [
        {
            "reason": engine_reason,
            "pid": pid,
            "process_name": name,
            "process_type": "C+G",
            "engine_type": "3D",
            "engine_name": f"pid_{pid}_luid_0x1_phys_0_eng_0_engtype_3D",
            "engine_utilization_percent": 5,
            "identity_observation_sha256": "a" * 64,
            "identity_verified": True,
        }
    ]

    bad = identity(pid=pid, name=name)
    bad["processes"][0]["cim_path_basename"] = "other.exe"
    rejected = base_receipt(raw, bad)
    result = S._apply_shared_profile(rejected)
    assert result["busy"] is True
    assert result["decision"] == "rejected"
    assert result["busy_reasons"] == [
        f"process_identity_not_verified:{pid}",
        engine_reason,
    ]
    assert result["registered_gui_active_engine_exceptions"] == []


def test_unverified_registered_gui_engine_reason_is_never_excluded() -> None:
    pid, name = 4200, "applicationframehost.exe"
    raw = raw_snapshot(pid=pid, name=name, engine_type="3D")
    raw["busy"] = True
    raw["busy_reasons"] = [
        f"unapproved_gpu_process:{pid}",
        f"unapproved_active_engine:{pid}:3D",
        f"unverified_active_engine_process:{pid}",
    ]
    result = S._apply_shared_profile(base_receipt(raw, identity(pid=pid, name=name)))
    assert result["busy"] is True
    assert result["decision_reasons"] == [
        f"unapproved_base_reason:unverified_active_engine_process:{pid}"
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(utilization_percent="2"), "utilization scalar"),
        (lambda raw: raw.update(compute_process_count=1), "raw scalar"),
        (lambda raw: raw.update(memory_used_mib=4096), "memory headroom"),
        (lambda raw: raw["processes"][0].update(known_workload=True), "known 또는"),
        (lambda raw: raw["processes"][0].update(type="C"), "process type"),
        (lambda raw: raw["active_engines"][0].update(process_name="other.exe"), "process 결합"),
        (lambda raw: raw["active_engines"][0].update(utilization_percent=0), "process 결합"),
        (
            lambda raw: raw["active_engines"][0].update(
                name="pid_4100_luid_0x1_phys_1_eng_0_engtype_3D"
            ),
            "name pattern",
        ),
        (
            lambda raw: raw["active_engines"][0].update(
                name="pid_4100_luid_0x1_phys_0_eng_0_engtype_Compute"
            ),
            "engine type",
        ),
        (
            lambda raw: raw["active_engines"][0].update(
                name="pid_9999_luid_0x1_phys_0_eng_0_engtype_3D"
            ),
            "PID가 GPU process에 없다",
        ),
        (lambda raw: raw["processes"].append(deepcopy(raw["processes"][0])), "PID가 없거나 중복"),
    ],
)
def test_invalid_scalars_processes_and_engines_fail_closed(change, message: str) -> None:
    raw = raw_snapshot()
    change(raw)
    with pytest.raises(S.SharedAdmissionError, match=message):
        S._apply_shared_profile(base_receipt(raw))


def test_python_known_workload_and_unknown_base_reason_are_not_excluded() -> None:
    python = raw_snapshot(name="python.exe", known=True, with_engine=False, utilization=0)
    python["busy"] = True
    python["busy_reasons"] = ["unapproved_gpu_process:4100"]
    with pytest.raises(S.SharedAdmissionError, match="process 이름"):
        S._apply_shared_profile(base_receipt(python))

    raw = raw_snapshot()
    receipt = base_receipt(raw)
    receipt["decision_reasons"].append("unknown_reason")
    receipt["busy_reasons"].append("unknown_reason")
    with pytest.raises(S.SharedAdmissionError, match="재현할 수 없다"):
        S._apply_shared_profile(receipt)


def test_top_level_receipt_cannot_disagree_with_bound_raw_snapshot() -> None:
    receipt = base_receipt(raw_snapshot())
    receipt["active_engines"] = []
    with pytest.raises(S.SharedAdmissionError, match="raw snapshot이 다르다"):
        S._apply_shared_profile(receipt)


def test_production_query_is_zero_arg_and_propagates_lease_or_query_failure(monkeypatch) -> None:
    calls = 0
    accepted = base_receipt(raw_snapshot())

    def query() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return deepcopy(accepted)

    monkeypatch.setattr(S.GPU1, "query_registered_desktop_gpu", query)
    assert S.query_registered_desktop_gpu_shared()["busy"] is False
    assert calls == 1

    def failure() -> dict[str, Any]:
        raise S.GPU1.AdmissionError("lease failed")

    monkeypatch.setattr(S.GPU1, "query_registered_desktop_gpu", failure)
    with pytest.raises(S.GPU1.AdmissionError, match="lease failed"):
        S.query_registered_desktop_gpu_shared()


def test_scope_reuses_gpu1_alias_lock_and_restores_aliases(tmp_path: Path, monkeypatch) -> None:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    v2 = importlib.import_module("evals.r1_repair_execution_v2")
    originals = (package._query_gpu, top_level._query_gpu)
    accepted = S._apply_shared_profile(base_receipt(raw_snapshot()))
    monkeypatch.setattr(S, "query_registered_desktop_gpu_shared", lambda: deepcopy(accepted))
    with S.snapshot_scope():
        outer = (package._query_gpu, top_level._query_gpu)
        assert outer[0] is not originals[0]
        assert outer[1] is not originals[1]
        assert v2.V1._query_gpu is top_level._query_gpu
        with S.snapshot_scope():
            snapshot = v2.V1._snapshot(tmp_path, "shared")
        assert snapshot["busy"] is False
        assert snapshot["admission_policy"] == S.PROFILE_ID
        assert (package._query_gpu, top_level._query_gpu) == outer
    assert (package._query_gpu, top_level._query_gpu) == originals
    assert v2.V1._query_gpu is originals[1]


def test_scope_has_no_callback_injection_surface(tmp_path: Path) -> None:
    scope: Any = S.snapshot_scope
    with pytest.raises(TypeError), scope(query=lambda: base_receipt(raw_snapshot())):
        importlib.import_module("r1_repair_execution")._snapshot(tmp_path, "forged")
    assert not (tmp_path / "gpu_forged.json").exists()
