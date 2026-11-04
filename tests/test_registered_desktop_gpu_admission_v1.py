from __future__ import annotations

import importlib
import json
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evals.registered_desktop_gpu_admission_v1 as A

NOW = datetime(2026, 11, 4, 7, 30, tzinfo=UTC)
REGISTERED = [
    (19616, "applicationframehost.exe"),
    (86308, "msedge.exe"),
    (131576, "codex-computer-use.exe"),
    (159200, "systemsettings.exe"),
]


def raw_snapshot() -> dict:
    processes = [
        {"pid": 4, "name": "system", "type": "C+G", "known_workload": False},
        *[
            {"pid": pid, "name": name, "type": "C+G", "known_workload": False}
            for pid, name in REGISTERED
        ],
    ]
    return {
        "busy": True,
        "busy_reasons": [f"unapproved_gpu_process:{pid}" for pid, _name in REGISTERED],
        "admission_policy": "desktop_shared_recording_v1",
        "gpu_exclusive": False,
        "performance_measurement": False,
        "observed_at_utc": "2026-11-04T06:59:33.163794+00:00",
        "gpu_index": 0,
        "utilization_percent": 1,
        "memory_used_mib": 1457,
        "compute_process_count": 0,
        "processes": processes,
        "active_engines": [],
        "c_plus_g_contexts_present": True,
    }


def identity_evidence(*, executable_name: str | None = None, xml_name: str | None = None) -> dict:
    observation = "a" * 64
    rows = []
    for pid, name in REGISTERED:
        cim_basename = executable_name if pid == 19616 and executable_name else name
        xml_basename = xml_name if pid == 19616 and xml_name else name
        verified = cim_basename == name == xml_basename
        rows.append(
            {
                "pid": pid,
                "cim_name": name,
                "xml_path_basename": xml_basename,
                "cim_path_basename": cim_basename,
                "xml_path_sha256": "b" * 64,
                "cim_path_sha256": "b" * 64 if verified else "c" * 64,
                "paths_equal": verified,
                "names_equal": verified,
                "command_present": True,
                "verified": verified,
                "observation_sha256": observation,
            }
        )
    return {"observation_sha256": observation, "processes": rows}


def gpu_bundle(raw: dict | None = None, identity: dict | None = None) -> dict:
    return {
        "snapshot": raw if raw is not None else raw_snapshot(),
        "identity_evidence": identity if identity is not None else identity_evidence(),
    }


def query(raw: dict) -> dict:
    return A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture", "callerPid": os.getpid()},
        gpu_reader=lambda: gpu_bundle(raw),
        clock=lambda: NOW,
    )


def test_registered_fixture_is_accepted_without_mutating_raw() -> None:
    raw = raw_snapshot()
    before = deepcopy(raw)
    receipt = query(raw)
    assert raw == before
    assert receipt["busy"] is False
    assert receipt["decision"] == "accepted"
    assert receipt["decision_reasons"] == []
    assert receipt["raw_snapshot"] == before
    assert receipt["raw_snapshot"] is not raw
    assert receipt["fixture_callbacks_used"] is True
    assert receipt["actual_gpu_admission_evidence"] is False
    assert len(receipt["raw_snapshot_sha256"]) == 64
    assert len(receipt["policy_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update(compute_process_count=1), "invalid_compute_process_count"),
        (lambda value: value.update(utilization_percent=2), "gpu_utilization_above_registered_limit"),
        (lambda value: value.update(memory_used_mib=4096), "memory_headroom_insufficient"),
        (
            lambda value: value.update(active_engines=[{"name": "3D", "utilization_percent": 1}]),
            "active_gpu_engine_present",
        ),
        (lambda value: value.update(gpu_exclusive=True), "invalid_gpu_exclusive"),
        (lambda value: value.update(performance_measurement=True), "invalid_performance_measurement"),
        (lambda value: value.update(gpu_index=1), "invalid_gpu_index"),
        (
            lambda value: value["processes"][1].update(known_workload=True),
            "known_or_unclassified_workload:19616",
        ),
        (
            lambda value: value["processes"][1].update(type="G"),
            "registered_exception_not_applicable:19616",
        ),
        (
            lambda value: value["processes"][1].update(name="python.exe"),
            "registered_exception_not_applicable:19616",
        ),
        (
            lambda value: value["busy_reasons"].append("unapproved_active_engine:19616:Compute"),
            "unapproved_base_reason:unapproved_active_engine:19616:Compute",
        ),
    ],
)
def test_fail_closed_mutations(mutate, reason: str) -> None:
    raw = raw_snapshot()
    mutate(raw)
    receipt = query(raw)
    assert receipt["busy"] is True
    assert receipt["decision"] == "rejected"
    assert reason in receipt["decision_reasons"]


def test_missing_duplicate_and_unapproved_pid_are_rejected() -> None:
    missing = raw_snapshot()
    missing["processes"] = [row for row in missing["processes"] if row["pid"] != 19616]
    assert "missing_process_for_reason:19616" in query(missing)["decision_reasons"]

    duplicate = raw_snapshot()
    duplicate["processes"].append(deepcopy(duplicate["processes"][1]))
    assert "duplicate_or_invalid_process_pid" in query(duplicate)["decision_reasons"]

    unregistered = raw_snapshot()
    unregistered["processes"][1]["name"] = "notepad.exe"
    assert "registered_exception_not_applicable:19616" in query(unregistered)["decision_reasons"]


def test_identity_mismatch_reviewer_reproductions_are_rejected() -> None:
    raw = raw_snapshot()
    same_wrong_basename = A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture"},
        gpu_reader=lambda: gpu_bundle(
            raw, identity_evidence(executable_name="other.exe", xml_name="other.exe")
        ),
        clock=lambda: NOW,
    )
    assert "process_identity_not_verified:19616" in same_wrong_basename["decision_reasons"]

    different_paths = A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture"},
        gpu_reader=lambda: gpu_bundle(raw, identity_evidence(xml_name="other.exe")),
        clock=lambda: NOW,
    )
    assert "process_identity_not_verified:19616" in different_paths["decision_reasons"]


def test_reviewer_identity_mismatch_through_real_base_classifier_is_rejected() -> None:
    xml = """<nvidia_smi_log><gpu><utilization><gpu_util>1 %</gpu_util></utilization>
    <fb_memory_usage><used>100 MiB</used></fb_memory_usage><processes><process_info>
    <pid>19616</pid><type>C+G</type><process_name>C:\\other.exe</process_name>
    </process_info></processes></gpu></nvidia_smi_log>"""
    windows = {
        "processes": [
            {
                "ProcessId": 19616,
                "Name": "applicationframehost.exe",
                "ExecutablePath": "C:\\other.exe",
                "CommandLine": "C:\\other.exe --fixture",
            }
        ],
        "engines": [],
    }
    raw = A.GPU.classify_gpu(xml, windows, "desktop_shared_recording_v1")
    assert raw["busy_reasons"] == ["unapproved_gpu_process:19616"]
    evidence = A._identity_evidence(xml, windows)
    assert evidence["processes"][0]["verified"] is False
    receipt = A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture"},
        gpu_reader=lambda: gpu_bundle(raw, evidence),
        clock=lambda: NOW,
    )
    assert receipt["busy"] is True
    assert "process_identity_not_verified:19616" in receipt["decision_reasons"]

    windows["processes"][0]["ExecutablePath"] = "C:\\applicationframehost.exe"
    raw = A.GPU.classify_gpu(xml, windows, "desktop_shared_recording_v1")
    assert "unverified_process_identity:19616" in raw["busy_reasons"]
    receipt = A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture"},
        gpu_reader=lambda: gpu_bundle(raw, A._identity_evidence(xml, windows)),
        clock=lambda: NOW,
    )
    assert receipt["busy"] is True
    assert "unapproved_base_reason:unverified_process_identity:19616" in receipt["decision_reasons"]


def test_snapshot_without_same_observation_identity_is_rejected() -> None:
    receipt = A.query_registered_desktop_gpu(
        lease_reader=lambda: {"leaseId": "fixture"},
        gpu_reader=raw_snapshot,
        clock=lambda: NOW,
    )
    assert receipt["busy"] is True
    assert "process_identity_evidence_missing" in receipt["decision_reasons"]


def test_duplicate_and_inconsistent_busy_reasons_are_rejected() -> None:
    duplicate = raw_snapshot()
    duplicate["busy_reasons"].append(duplicate["busy_reasons"][0])
    assert "duplicate_busy_reason" in query(duplicate)["decision_reasons"]

    inconsistent = raw_snapshot()
    inconsistent["busy"] = False
    assert "inconsistent_busy_contract" in query(inconsistent)["decision_reasons"]


def test_each_query_rereads_lease_and_gpu() -> None:
    calls = {"lease": 0, "gpu": 0}

    def lease() -> dict:
        calls["lease"] += 1
        return {"leaseId": f"fixture-{calls['lease']}"}

    def gpu() -> dict:
        calls["gpu"] += 1
        return gpu_bundle()

    for _ in range(2):
        receipt = A.query_registered_desktop_gpu(
            lease_reader=lease, gpu_reader=gpu, clock=lambda: NOW
        )
        assert receipt["busy"] is False
    assert calls == {"lease": 2, "gpu": 2}


def test_default_readers_use_retry_lease_context_and_fresh_identity_query(monkeypatch) -> None:
    calls: list[object] = []

    def lease_context() -> dict:
        calls.append(("lease_context", os.getpid()))
        return {"leaseId": "live"}

    def retry(reader) -> dict:
        calls.append("retry")
        return reader()

    def gpu() -> dict:
        calls.append("gpu_with_identity")
        return gpu_bundle()

    monkeypatch.setattr(A.LEASE, "lease_context", lease_context)
    monkeypatch.setattr(A.RETRY, "read_live_lease", retry)
    monkeypatch.setattr(A, "_query_gpu_with_identity", gpu)
    receipt = A.query_registered_desktop_gpu(clock=lambda: NOW)
    assert receipt["busy"] is False
    assert receipt["actual_gpu_admission_evidence"] is False
    assert receipt["admission_evidence_requires_external_execution_receipt"] is True
    assert calls == ["retry", ("lease_context", os.getpid()), "gpu_with_identity"]


def test_scope_patches_both_aliases_and_v2_snapshot_path(tmp_path: Path, monkeypatch) -> None:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    v2 = importlib.import_module("evals.r1_repair_execution_v2")
    package_original = package._query_gpu
    top_original = top_level._query_gpu
    assert isinstance(package.__file__, str)
    assert isinstance(top_level.__file__, str)
    assert Path(package.__file__).resolve() == Path(top_level.__file__).resolve()
    assert v2.V1 is top_level

    operational_receipt = query(raw_snapshot())
    monkeypatch.setattr(A, "query_registered_desktop_gpu", lambda: deepcopy(operational_receipt))
    with A.snapshot_scope():
        assert package._query_gpu is not package_original
        assert top_level._query_gpu is not top_original
        assert v2.V1._query_gpu is top_level._query_gpu
        with A.snapshot_scope():
            snapshot = v2.V1._snapshot(tmp_path, "before")
        assert snapshot["busy"] is False
        written = json.loads((tmp_path / "gpu_before.json").read_text(encoding="utf-8"))
        assert written["admission_policy"] == "registered_desktop_shared_v2"

    assert package._query_gpu is package_original
    assert top_level._query_gpu is top_original
    assert v2.V1._query_gpu is top_original


def test_scope_restores_aliases_after_error() -> None:
    package = importlib.import_module("evals.r1_repair_execution")
    top_level = importlib.import_module("r1_repair_execution")
    originals = (package._query_gpu, top_level._query_gpu)
    with pytest.raises(RuntimeError, match="boom"), A.snapshot_scope():
        raise RuntimeError("boom")
    assert (package._query_gpu, top_level._query_gpu) == originals


def test_scope_rejects_forged_callbacks_before_r1_snapshot(tmp_path: Path) -> None:
    scope: Any = A.snapshot_scope
    with pytest.raises(TypeError), scope(
        lease_reader=lambda: {"leaseId": "forged"}, gpu_reader=lambda: gpu_bundle()
    ):
        importlib.import_module("r1_repair_execution")._snapshot(tmp_path, "forged")
    assert not (tmp_path / "gpu_forged.json").exists()
