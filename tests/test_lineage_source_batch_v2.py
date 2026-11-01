# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import lineage_source_batch_v2 as B


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _value() -> dict:
    cases, training, smoke = B.jobs(list(range(2550, 2560)), [3550, 3551], 4550)
    return {"schema": B.SCHEMA, "phase": "source_training_and_fixed26_only",
            "cases": cases, "training_jobs": training, "smoke_job": smoke,
            "seed_plan": {"case": list(range(2550, 2560)),
                          "normal_reference": [3550, 3551], "smoke": 4550,
                          "replacement_seeds": 0},
            "source_sha256_lf": {name: "a" * 64 for name in B.source_files()},
            "runtime_source_sha256_lf": {name: "b" * 64 for name in B.V1.RUNTIME_FILES},
            "lease_helper_sha256_lf": "c" * 64,
            "budget": {"smoke_training": 1, "source_training": 20, "fixed26": 20,
                       "maximum_owned_sdk_jobs": 41, "model_api": 0, "repairs": 0,
                       "automatic_retries": 0, "per_job_parent_wall_s": 1800,
                       "total_parent_wall_s": 73800},
            "runtime_facts": {"required_for_source_completion": False,
                              "fabricated_when_unavailable": False,
                              "v1_child_compatible": False},
            "outcomes": {"source_failure_absent": "보존하고 대체 seed를 뽑지 않는다.",
                         "not_applicable": "보존하고 대체 seed를 뽑지 않는다."},
            "claims": {"heldout": False, "new_fault_generalization": False,
                       "diagnosis": False, "recovery": False, "model_advantage": False}}


def test_jobs_are_twenty_with_distinct_normal_references() -> None:
    cases, training, smoke = B.jobs(list(range(2550, 2560)), [3550, 3551], 4550)
    assert len(cases) == 10 and len(training) == 20 and smoke["seed"] == 4550
    by_id = {row["job_id"]: row for row in training}
    assert len(by_id) == 20
    for case in cases[:2]:
        assert by_id[case["source_job"]]["seed"] != by_id[case["reference_job"]]["seed"]
        assert case["reference_relation"] == "different_seed_normal_reference"
    for case in cases[2:]:
        assert by_id[case["source_job"]]["seed"] == by_id[case["reference_job"]]["seed"]


def test_freshness_uses_existing_core_pruning_guard() -> None:
    assert B.SEEDS.__name__ == "core_seed_guard_v1"
    assert "evals/core_seed_guard_v1.py" in B.source_files()
    assert "tests/test_core_seed_guard_v1.py" in B.source_files()


def test_protocol_rejects_budget_and_duplicate_seed() -> None:
    value = _value()
    assert len(B.validate_protocol(value)["training_jobs"]) == 20
    changed = json.loads(json.dumps(value))
    changed["budget"]["fixed26"] = 19
    with pytest.raises(ValueError, match="예산"):
        B.validate_protocol(changed)
    with pytest.raises(ValueError, match="서로 다른"):
        B.jobs([2550] * 10, [3550, 3551], 4550)


def test_run_all_requires_smoke_before_creating_output(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    with pytest.raises(ValueError, match="smoke"):
        B.run_sequence(root, tmp_path / "protocol.json", _value(), "a" * 40,
                       tmp_path, tmp_path / "python", mode="all", runner=lambda *a: {},
                       fixed_runner=lambda *a, **k: ({}, {}))
    assert not root.exists()


def test_completed_accepts_v2_manifest_fingerprint(tmp_path: Path) -> None:
    stage, output, logs = tmp_path / "stage", tmp_path / "stage/training", tmp_path / "logs"
    output.mkdir(parents=True)
    logs.mkdir()
    row = _value()["smoke_job"]
    fixed = {"schema": B.CHILD.SCHEMA, "purpose": "smoke",
             **{key: row[key] for key in B.CHILD.JOB_FIELDS},
             "source_commit": "a" * 40, "protocol_relative": "protocol.json",
             "protocol_sha256_lf": "b" * 64,
             "source_sha256_lf": {name: "c" * 64 for name in B.CHILD.SOURCE_FILES},
             "runtime_source_sha256_lf": {"runtime.py": "d" * 64}}
    checkpoint = logs / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    evidence = {"manifest_sha256": B.CHILD.manifest_fingerprint(fixed), "updates": 1,
                "update_count": 1, "arm": "NONE", "seed": row["seed"],
                "learn_returned": True, "learning_completed": True,
                "recorded_before_app_close": True, "log_dir": str(logs),
                "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint),
                               "log_dir": str(logs)}}
    (output / B.GUARD.RECORDS).write_text("{}\n", encoding="utf-8")
    B._write_new(output / B.GUARD.EVIDENCE, evidence)
    B._write_new(output / B.GUARD.CLOSE_READY, {
        "schema": B.GUARD.CLOSE_SCHEMA, "manifest_sha256": evidence["manifest_sha256"],
        "updates": 1, "evidence_sha256_lf": _sha(output / B.GUARD.EVIDENCE, lf=True),
        "records_sha256_lf": _sha(output / B.GUARD.RECORDS, lf=True)})
    execution = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                 "job_assigned_before_release": True, "timed_out": False,
                 "error_type": None}
    assert B.completed(stage, fixed, execution)["checkpoint"]["sha256"] == _sha(checkpoint)


def test_all_runs_twenty_training_then_twenty_fixed_without_runtime_fact_fabrication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = _value()
    monkeypatch.setattr(B, "ROOT", tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(B.CHILD, "verify_binding", lambda *args, **kwargs: None)
    smoke_ref = {"path": str(tmp_path / "smoke"), "result_sha256": "a" * 64,
                 "evidence_sha256": "b" * 64}
    monkeypatch.setattr(B, "verify_smoke", lambda *args, **kwargs: dict(smoke_ref))
    monkeypatch.setattr(B, "completed", lambda stage, fixed, execution: {
        "learning_completed": True, "recorded_before_app_close": True,
        "checkpoint": {"path": str(stage / "model.pt"), "sha256": "c" * 64,
                       "log_dir": str(stage)}})
    fixed_calls: list[str] = []

    def runner(stage, argv, lab, timeout):
        del argv, lab, timeout
        training = stage / "training"
        training.mkdir()
        (training / B.GUARD.EVIDENCE).write_text("{}\n", encoding="utf-8")
        (training / B.GUARD.RECORDS).write_text("{}\n", encoding="utf-8")
        return {"wall_s": 0.1}

    def fixed(stage, evidence, *, name, **kwargs):
        del stage, evidence, kwargs
        fixed_calls.append(name)
        return {"conditions": 26}, {"exit_code": 0}

    def lease():
        return {"owner": B.OWNER, "leaseId": "lease-a"}
    result = B.run_sequence(
        tmp_path / "all", protocol_path, value, "a" * 40, tmp_path, tmp_path / "python",
        mode="all", runner=runner, fixed_runner=fixed, check_lease=lease,
        smoke_reference=smoke_ref, wait=lambda *args, **kwargs: None)
    assert result["source_jobs_completed"] == result["fixed26_completed"] == 20
    assert len(fixed_calls) == 20
    assert all(row["runtime_facts"] is None for row in result["records"])
