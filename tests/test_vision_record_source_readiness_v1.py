from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "bench"))
import vision_record_source_readiness_v1 as M  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return M.MATERIAL._ref(repo, path)


def _case(repo: Path) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    base = repo / "evidence"
    checkpoint = _write(base / "checkpoint.pt", b"checkpoint")
    trace = _write(base / "trace.json", {"records": []})
    video = _write(base / "same_play.mp4", b"video")
    m1 = _write(base / "measurement-1.json", {"window": 1})
    m2 = _write(base / "measurement-2.json", {"window": 2})
    s1 = _write(base / "spec-1.json", {"window": 1})
    s2 = _write(base / "spec-2.json", {"window": 2})
    adapter = _write(
        base / "record-adapter.json",
        {
            "schema": M.MATERIAL.TECHNICAL_SCHEMA,
            "record_completed": True,
            "model_comparison_ready": False,
            "labels_inferred_from_recording": False,
            "record": {
                "video": _ref(repo, video),
                "trace": _ref(repo, trace),
                "measurement_1": _ref(repo, m1),
                "measurement_2": _ref(repo, m2),
                "spec_1": _ref(repo, s1),
                "spec_2": _ref(repo, s2),
            },
        },
    )
    source_artifact = _write(
        base / "source-training.json",
        {
            "seed": 5101,
            "arm": "NONE",
            "checkpoint": _ref(repo, checkpoint),
            "telemetry": {"sha256": "c" * 64},
        },
    )
    telemetry = _write(base / "telemetry.json", {"seed": 5101})
    training = _write(
        base / "training-binding.json",
        {
            "schema": M.MATERIAL.TRAINING_SCHEMA,
            "source_seed": 5101,
            "arm": "NONE",
            "checkpoint": _ref(repo, checkpoint),
            "telemetry": _ref(repo, telemetry),
            "source_training_artifact": _ref(repo, source_artifact),
        },
    )
    fixed_artifact = _write(base / "fixed.json", {"label": "healthy"})
    fixed = _write(
        base / "fixed-binding.json",
        {
            "schema": M.MATERIAL.FIXED_EVAL_SCHEMA,
            "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
            "label": "healthy",
            "fixed_eval_artifact": _ref(repo, fixed_artifact),
        },
    )
    source_protocol = _write(
        base / "source-protocol.json",
        {
            "schema": "vision_brev_source_protocol_v1",
            "training_plan": [{"seed": 5101, "arm": "NONE"}],
        },
    )
    source_protocol_sha = M._sha_lf(source_protocol)
    case_binding = {
        "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
        "telemetry_sha256": "c" * 64,
    }
    packet = M.VIEW.S.build_packet(
        epoch=1,
        applied_corrections=[],
        prior_receipt=None,
        facts={
            "telemetry:case:Metrics/base_velocity/error_vel_xy": 1.2,
            "telemetry:reference:Metrics/base_velocity/error_vel_xy": 0.15,
            "config:action_scale": 0.25,
        },
        input_bindings={"case": case_binding, "reference": case_binding},
        budget_left=7,
        execution_budget_s=100,
    )
    snapshot = _write(
        base / "snapshot.json",
        M.VIEW.build_snapshot(
            packet,
            "readiness_fixture",
            "model_adaptive",
            M.VIEW.file_digest(M.VIEW.PROTOCOL),
            base / "probes.ledger.jsonl",
        ),
    )
    identity = {
        "resource_id": "brev-one",
        "gpu_uuid": "GPU-one",
        "lease_id": "lease-one",
        "compute_started_at_utc": "2026-10-26T00:00:00+00:00",
        "protocol_sha256": "d" * 64,
    }
    resources: list[dict[str, dict[str, str]]] = []
    for name, issued in (
        ("record", "2026-10-26T00:01:00+00:00"),
        ("smoke", "2026-10-26T00:02:00+00:00"),
    ):
        raw_paths = {
            key: _write(base / f"{name}-{key}.json", {"kind": key})
            for key in M.RESOURCE_SHA_FIELDS
        }
        issuance = _write(base / f"{name}-issuance.json", {"issued_at_utc": issued})
        start = _write(
            base / f"{name}-start.json",
            {"started_at_utc": identity["compute_started_at_utc"]},
        )
        proof = _write(
            base / f"{name}-resource.json",
            {
                "schema": "brev_resource_proof_v1",
                "resource_kind": "remote_gpu",
                "status": "granted",
                **identity,
                "protocol_sha256": source_protocol_sha,
                "issued_at_utc": issued,
                "expires_at_utc": "2026-10-26T01:00:00+00:00",
                **{
                    field: M.MATERIAL._sha(raw_paths[key])
                    for key, field in M.RESOURCE_SHA_FIELDS.items()
                },
            },
        )
        resources.append(
            {
                "proof": _ref(repo, proof),
                "issuance": _ref(repo, issuance),
                "start": _ref(repo, start),
                **{key: _ref(repo, path) for key, path in raw_paths.items()},
            }
        )
    smoke_result = _write(base / "smoke/output/result.json", {"completed": True})
    smoke_execution = _write(base / "smoke/execution.json", {"completed": True})
    binding = _write(
        base / "binding.json",
        {
            "schema": M.BINDING_SCHEMA,
            "seed_group": "source-01",
            "source_role": "normal",
            "source_seed": 5101,
            "record_adapter": _ref(repo, adapter),
            "smoke_result": _ref(repo, smoke_result),
            "smoke_execution": _ref(repo, smoke_execution),
            "training_binding": _ref(repo, training),
            "fixed_eval_binding": _ref(repo, fixed),
            "snapshot": _ref(repo, snapshot),
            "record_resource": resources[0],
            "smoke_resource": resources[1],
            "source_protocol": _ref(repo, source_protocol),
        },
    )
    record_job = {
        "checkpoint": {
            "path": "/remote/model.pt",
            "sha256": M.MATERIAL._sha(checkpoint),
            "logical_id": "normal",
        },
        "profile": {"fixed": True},
        "source": {"commit": "a" * 40},
        "resource_proof": {"sha256": resources[0]["proof"]["sha256"]},
    }
    smoke_job = {
        **record_job,
        "resource_proof": {"sha256": resources[1]["proof"]["sha256"]},
    }
    _write(base / "record-job.json", record_job)
    record_result = {"checkpoint_sha256": M.MATERIAL._sha(checkpoint)}
    return binding, record_job, smoke_job, record_result, smoke_execution


def _patch_terminals(
    monkeypatch: pytest.MonkeyPatch,
    record_job: dict[str, Any],
    smoke_job: dict[str, Any],
    record_result: dict[str, Any],
    smoke_execution: Path,
) -> None:
    monkeypatch.setattr(
        M.MATERIAL,
        "_verify_record_terminal",
        lambda _repo, _adapter: (
            record_job,
            record_result,
            smoke_execution.parent.parent / "record-job.json",
            Path("execution"),
        ),
    )
    monkeypatch.setattr(
        M,
        "_smoke",
        lambda _repo, _result, _execution: (
            smoke_job,
            {"checkpoint_sha256": record_result["checkpoint_sha256"]},
            smoke_execution,
        ),
    )


def _source_protocol(repo: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    value = M.MATERIAL._read(repo / readiness["source_protocol"]["path"])
    value["_verified_sha256_lf"] = readiness["source_protocol_sha256_lf"]
    return value


def test_fixed_eval_and_checkpoint_smoke_create_first_class_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    output = repo / "evidence/readiness.json"
    readiness = M.build_readiness(repo, binding, output)
    assert readiness["model_use_ready"] is True
    assert readiness["labels_inferred_from_recording"] is False
    assert readiness["checkpoint_smoke"]["policy_environment_steps"] == 1
    assert readiness["checkpoint_smoke"]["capture_phase_physics_steps"] == 0
    assert readiness["resource_binding"]["raw_proofs_independently_hashed"] is True
    assert (
        M.verify_model_readiness(repo, output, _source_protocol(repo, readiness))
        == readiness["proof"]
    )


def test_reissued_resource_must_keep_same_vm_gpu_lease_start_and_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding_path, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    binding = M.MATERIAL._read(binding_path)
    smoke_resource_path = repo / binding["smoke_resource"]["proof"]["path"]
    smoke_resource = M.MATERIAL._read(smoke_resource_path)
    smoke_resource["gpu_uuid"] = "GPU-other"
    smoke_resource_path.write_text(json.dumps(smoke_resource), encoding="utf-8")
    binding["smoke_resource"]["proof"]["sha256"] = M.MATERIAL._sha(smoke_resource_path)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        M.build_readiness(repo, binding_path, repo / "evidence/out.json")


def test_recording_cannot_supply_behavior_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding_path, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    binding = M.MATERIAL._read(binding_path)
    fixed_path = repo / binding["fixed_eval_binding"]["path"]
    fixed = M.MATERIAL._read(fixed_path)
    fixed["label"] = "unhealthy"
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    binding["fixed_eval_binding"]["sha256"] = M.MATERIAL._sha(fixed_path)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="고정 평가"):
        M.build_readiness(repo, binding_path, repo / "evidence/out.json")


def test_ready_flag_or_proof_cannot_replace_recomputed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    output = repo / "evidence/readiness.json"
    readiness = M.build_readiness(repo, binding, output)
    forged = M.MATERIAL._read(output)
    forged["model_use_ready"] = True
    forged["proof"]["video_sha256"] = "f" * 64
    output.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="재계산"):
        M.verify_model_readiness(repo, output, _source_protocol(repo, readiness))


def test_bound_evidence_file_mutation_is_rejected_on_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    output = repo / "evidence/readiness.json"
    readiness = M.build_readiness(repo, binding, output)
    measurement = repo / readiness["record"]["measurement_1"]["path"]
    measurement.write_text('{"window":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        M.verify_model_readiness(repo, output, _source_protocol(repo, readiness))


def test_checked_snapshot_case_binding_must_match_training_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding_path, job, smoke_job, result, execution = _case(repo)
    _patch_terminals(monkeypatch, job, smoke_job, result, execution)
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    binding = M.MATERIAL._read(binding_path)
    snapshot_path = repo / binding["snapshot"]["path"]
    snapshot = M.MATERIAL._read(snapshot_path)
    inputs = snapshot["inputs"]
    inputs["packet"]["input_bindings"]["case"]["telemetry_sha256"] = "e" * 64
    snapshot_path.write_text(
        json.dumps(M.VIEW.build_snapshot(**inputs)), encoding="utf-8"
    )
    binding["snapshot"]["sha256"] = M.MATERIAL._sha(snapshot_path)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="case binding"):
        M.build_readiness(repo, binding_path, repo / "evidence/out.json")
