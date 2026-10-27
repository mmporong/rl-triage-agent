from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import vision_corrected_record_service_v1 as S


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(path: Path) -> dict[str, object]:
    return {"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size}


def native_worker_fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict[str, Path]]:
    stage = tmp_path / "stage"
    output = stage / "output"
    output.mkdir(parents=True)
    artifacts: dict[str, Path] = {}
    for key in sorted(S.ARTIFACT_KEYS):
        suffix = ".mp4" if key == "video" else ".json"
        path = output / f"{key}{suffix}"
        path.write_bytes(f"native-{key}".encode())
        artifacts[key] = path
    manifest = {"schema": "vision_pull_record_native_capture_manifest_v2", "frames": []}
    manifest_path = write(output / S.WORKER.MANIFEST_NAME, manifest)
    validated = {"output_directory": output, "job_path": stage / "job.json",
                 "job_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}
    worker = {
        "schema": "vision_pull_record_worker_v3", "completed": True,
        "technical_recording_complete": True, "frame_count": 625, "fps": 50,
        "writer_calls": {"attach": 0, "callback": 0, "create": 0, "detach": 0},
        "atomicity_claimed": False, "checkpoint_sha256": "2" * 64,
        "job_sha256": "1" * 64, "capture_approved": False,
        "model_ready": False, "used_for_decision": False,
        "artifacts": {key: descriptor(path) for key, path in artifacts.items()},
        "native_capture_manifest": descriptor(manifest_path),
    }
    return stage, validated, worker, artifacts


def test_native_eight_artifact_consumer_accepts_exact_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, validated, worker, artifacts = native_worker_fixture(tmp_path)
    result = {"worker": worker}
    monkeypatch.setattr(S.WORKER.PREVIOUS.RECORD_VERIFY, "_record_artifacts", lambda v, a: True)
    monkeypatch.setattr(S.WORKER, "_manifest", lambda v, a: S._read(stage / "output" / S.WORKER.MANIFEST_NAME))
    monkeypatch.setattr(S, "_verify_raw_frames", lambda *a: None)
    checked, manifest = S._verify_worker(stage, validated, result)
    assert checked == artifacts
    assert manifest == stage / "output" / S.WORKER.MANIFEST_NAME
    artifacts["video"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="원문/SHA"):
        S._verify_worker(stage, validated, result)


def test_execution_requires_cleanup_aaba_then_record_and_false_flags(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    result_path = write(stage / "output" / "result.json", {
        "schema": S.RUNNER.RESULT_SCHEMA, "completed": True,
        "technical_recording_complete": True, "checkpoint_logical_id": "corrected-s2220",
        "checkpoint_sha256": "2" * 64, "job_sha256": "1" * 64,
        "training_actions": 0, "model_api_requests": 0,
        "capture_approved": False, "model_ready": False, "used_for_decision": False,
    })
    execution = {
        "schema": "vision_pull_record_execution_v3", "completed": True,
        "technical_recording_complete": True, "cleanup_verified": True,
        "exit_code": 0, "timed_out": False, "gpu_after": [],
        "job_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
        "result_sha256": sha(result_path), "capture_approved": False,
        "model_ready": False, "used_for_decision": False,
        "phases": [{"name": "aaba", "cleanup_verified": True, "exit_code": 0},
                   {"name": "record", "cleanup_verified": True, "exit_code": 0}],
    }
    write(stage / "execution.json", execution)
    validated = {"job_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}
    assert S._verify_execution(stage, validated, result_path)[1] == execution
    execution["phases"].reverse()
    write(stage / "execution.json", execution)
    with pytest.raises(ValueError, match="execution"):
        S._verify_execution(stage, validated, result_path)


def test_aaba_raw_npz_receipts_are_recomputed(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    aaba_dir = stage / "aaba-output"
    rows = []
    for ordinal in range(4):
        array = np.asarray([ordinal, ordinal + 1], dtype=np.int32)
        npz = aaba_dir / f"job.annotator-pull-{ordinal:06d}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, rgb=array)
        npz_descriptor = descriptor(npz)
        receipt = write(aaba_dir / f"job.annotator-pull-{ordinal:06d}.json", {
            "schema": "vision_annotator_pull_raw_receipt_v1", "ordinal": ordinal,
            "complete_capture": True, "npz": npz_descriptor, "atomicity_claimed": False,
            "physics_before": {"sha256": "a" * 64}, "physics_after": {"sha256": "a" * 64},
            "arrays": {"rgb": {"shape": [2], "dtype": "int32",
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}},
        })
        rows.append({"ordinal": ordinal, "complete_capture": True,
                     "npz": npz_descriptor, "receipt": descriptor(receipt)})
    aaba = write(aaba_dir / "native-aaba-preflight.report.json", {
        "diagnostic_complete": True, "approved": False, "atomicity_claimed": False,
        "capture_approved": False, "used_for_decision": False, "raw_artifacts": rows,
    })
    validated = {"checkpoint_sha256": "2" * 64}
    execution = {"aaba_receipt": {"diagnostic_complete": True,
        "checkpoint_sha256": "2" * 64, "report": descriptor(aaba),
        "capture_approved": False, "used_for_decision": False}}
    assert len(S._verify_aaba(stage, validated, execution)[1]) == 4
    with np.load(aaba_dir / "job.annotator-pull-000000.npz") as archive:
        first = archive["rgb"].copy()
    np.savez(aaba_dir / "job.annotator-pull-000000.npz", rgb=first + 1)
    with pytest.raises(ValueError, match="descriptor"):
        S._verify_aaba(stage, validated, execution)


def test_normal_reference_is_same_seed_same_profile_and_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "none.pt"
    receipt = tmp_path / "none-receipt.json"
    checkpoint.write_bytes(b"none")
    receipt.write_bytes(b"{}")
    profile = {"fps": 50, "frame_count": 625}
    value = {"seed": 2220, "role": "NONE", "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
             "fixed_eval": "healthy", "record_profile": profile,
             "record_receipt": {"path": str(receipt), "sha256": sha(receipt)}}
    assert S._normal_reference(value, 2220, profile)["role"] == "NONE"
    value["seed"] = 2221
    with pytest.raises(ValueError, match="동일 seed"):
        S._normal_reference(value, 2220, profile)


def test_service_returns_hold_without_writing_receipt_when_record_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    service = S.NativeCorrectedRecordService(protocol)
    job = tmp_path / "stage" / "job.json"
    monkeypatch.setattr(S.RUNNER, "create_job", lambda *a, **k: job)
    output = tmp_path / "receipt"
    result = service.record(resource_proof=tmp_path / "proof", resource_sha256="0" * 64,
        isaaclab=tmp_path / "lab", python=tmp_path / "python", stage=tmp_path / "stage",
        logical_id="corrected-s2220", correction_result=tmp_path / "correction",
        approval_receipt=tmp_path / "approval", receipt_output=output,
        run=lambda _: {"completed": False})
    assert result == {"status": "HOLD", "reason": "native_record_incomplete", "job": str(job)}
    assert not output.exists()


def test_verify_native_result_writes_distinct_native_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, validated, worker, artifacts = native_worker_fixture(tmp_path)
    protocol_path = write(tmp_path / "protocol.json", {"schema": S.RUNNER.PROTOCOL_SCHEMA})
    logical = "corrected-s2220"
    correction = write(tmp_path / "correction.json", {"status": "complete"})
    approval = write(tmp_path / "approval.json", {"approved": True})
    binding = {"logical_id": logical, **S.RUNNER.CATALOG[logical],
               "checkpoint": {"path": str(tmp_path / "corrected.pt"), "sha256": "2" * 64},
               "correction_result": {"path": str(correction), "sha256": sha(correction)},
               "approval_receipt": {"path": str(approval), "sha256": sha(approval)},
               "request_id": "request-2220"}
    validated.update({"protocol": {"checkpoints": [binding],
        "record_profile": S.RUNNER.RECORD.fixed_profile()},
        "protocol_path": protocol_path, "checkpoint_sha256": "2" * 64})
    write(stage / "job.json", {"schema": S.RUNNER.JOB_SCHEMA})
    write(stage / "output" / "result.json", {
        "checkpoint_logical_id": logical, "checkpoint_sha256": "2" * 64, "worker": worker,
    })
    execution_path = write(stage / "execution.json", {"completed": True})
    aaba_path = write(stage / "aaba-output" / "native-aaba-preflight.report.json", {"ok": True})
    manifest_path = stage / "output" / S.WORKER.MANIFEST_NAME
    monkeypatch.setattr(S.RUNNER, "validate_job", lambda *a, **k: validated)
    monkeypatch.setattr(S, "_verify_execution", lambda *a, **k: (execution_path, {"completed": True}))
    monkeypatch.setattr(S, "_verify_aaba", lambda *a, **k: (aaba_path, []))
    monkeypatch.setattr(S, "_verify_worker", lambda *a, **k: (artifacts, manifest_path))
    monkeypatch.setattr(S.RUNNER, "_verified_binding", lambda *a, **k: binding)
    monkeypatch.setattr(S, "_v01_contracts", lambda *a, **k: [
        {"window": 1, "contract": {"path": "one", "sha256": "a" * 64, "bytes": 1}, "verified": True},
        {"window": 2, "contract": {"path": "two", "sha256": "b" * 64, "bytes": 1}, "verified": True},
    ])
    output = tmp_path / "native-receipt"
    checked = S.verify_native_result(protocol_path, stage / "job.json",
                                     correction, approval, output)
    receipt = S._read(checked["receipt"])
    assert receipt["schema"] == S.SCHEMA
    assert receipt["schema"] != S.RUNNER.RESULT_SCHEMA
    assert receipt["post_observer"] == {"used_for_decision": False}
    assert receipt["claims"] == {"new_recovery_claim": False, "behavior_ground_truth": None}


def test_receipt_schema_is_native_and_not_legacy_top_level() -> None:
    assert S.SCHEMA == "vision_corrected_native_record_receipt_v1"
    assert S.SCHEMA != S.RUNNER.RESULT_SCHEMA
    assert S.FALSE_FLAGS == {"capture_approved": False, "model_ready": False,
                             "used_for_decision": False}
