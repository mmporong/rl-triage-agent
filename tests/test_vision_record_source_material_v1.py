from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_record_source_material_v1 as M  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(repo).as_posix(), "sha256": M._sha(path)}


def _case(repo: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    design = (
        ("source-01", "normal", 5101, "NONE", "healthy"),
        ("source-01", "registered-f1", 5101, "F1", "unhealthy"),
        ("source-02", "normal", 5102, "NONE", "healthy"),
        ("source-02", "registered-f1", 5102, "F1", "unhealthy"),
    )
    for index, (group, role, seed, arm, label) in enumerate(design, 1):
        base = repo / f"evidence/policy-{index}"
        checkpoint = _write(base / "checkpoint.pt", f"checkpoint-{index}".encode())
        telemetry = _write(base / "telemetry.json", {"seed": seed})
        training_artifact = _write(base / "training.json", {"seed": seed, "arm": arm})
        fixed_artifact = _write(base / "fixed-eval.json", {"label": label})
        snapshot = _write(base / "snapshot.json", {"snapshot": index})
        training = _write(
            base / "training-binding.json",
            {
                "schema": M.TRAINING_SCHEMA,
                "source_seed": seed,
                "arm": arm,
                "checkpoint": _ref(repo, checkpoint),
                "telemetry": _ref(repo, telemetry),
                "source_training_artifact": _ref(repo, training_artifact),
            },
        )
        fixed = _write(
            base / "fixed-binding.json",
            {
                "schema": M.FIXED_EVAL_SCHEMA,
                "checkpoint_sha256": M._sha(checkpoint),
                "label": label,
                "fixed_eval_artifact": _ref(repo, fixed_artifact),
            },
        )
        stage = base / "record"
        output = stage / "output"
        artifact_names = {
            "video": "same_play.mp4",
            "raw_trace": "trace.json",
            "encoder_receipt": "encoder.json",
            "timing_receipt": "timing.json",
            "measurement_1": "measurement-1.json",
            "spec_1": "spec-1.json",
            "measurement_2": "measurement-2.json",
            "spec_2": "spec-2.json",
        }
        artifact_paths = {
            key: _write(output / name, f"{key}-{index}".encode())
            for key, name in artifact_names.items()
        }
        descriptors = {
            key: {
                "file": path.name,
                "sha256": M._sha(path),
                "bytes": path.stat().st_size,
            }
            for key, path in artifact_paths.items()
        }
        job = _write(
            stage / "job.json",
            {
                "job_id": f"record-{group}-{role}",
                "checkpoint": {
                    "path": f"/remote/checkpoint-{index}.pt",
                    "sha256": M._sha(checkpoint),
                    "logical_id": f"policy-{index}",
                },
                "source": {"commit": "a" * 40, "files_lf": {}},
                "resource_proof": {"path": "/remote/resource.json", "sha256": "b" * 64},
                "profile": M.RECORD.fixed_profile(),
            },
        )
        result = _write(
            output / "result.json",
            {
                "schema": M.RUN_V1.JOB.RESULT_SCHEMA,
                "completed": True,
                "record_completed": True,
                "checkpoint_sha256": M._sha(checkpoint),
                "job_sha256": M._sha(job),
                "sdk": {"artifacts": descriptors},
            },
        )
        execution = _write(
            stage / "execution.json",
            {
                "schema": M.RUN_V2.EXECUTION_SCHEMA,
                "completed": True,
                "cleanup_verified": True,
                "gpu_after": [],
                "result_sha256": M._sha(result),
                "failure_atom": None,
            },
        )
        adapter = _write(
            base / "record-adapter.json",
            {
                "schema": M.TECHNICAL_SCHEMA,
                "status": "technical_only",
                "record_completed": True,
                "model_comparison_ready": False,
                "policy_row": None,
                "labels_inferred_from_recording": False,
                "record_result": _ref(repo, result),
                "record_execution": _ref(repo, execution),
                "record": {
                    "video": _ref(repo, artifact_paths["video"]),
                    "trace": _ref(repo, artifact_paths["raw_trace"]),
                    "encoder_receipt": _ref(repo, artifact_paths["encoder_receipt"]),
                    "timing_receipt": _ref(repo, artifact_paths["timing_receipt"]),
                    "measurement_1": _ref(repo, artifact_paths["measurement_1"]),
                    "spec_1": _ref(repo, artifact_paths["spec_1"]),
                    "measurement_2": _ref(repo, artifact_paths["measurement_2"]),
                    "spec_2": _ref(repo, artifact_paths["spec_2"]),
                },
            },
        )
        rows.append(
            {
                "seed_group": group,
                "source_role": role,
                "source_seed": seed,
                "record_adapter": _ref(repo, adapter),
                "training_binding": _ref(repo, training),
                "fixed_eval_binding": _ref(repo, fixed),
                "snapshot": _ref(repo, snapshot),
            }
        )
    registration = _write(
        repo / "evidence/registration.json",
        {"schema": M.REGISTRATION_SCHEMA, "policies": rows},
    )
    return registration, rows


def _accept_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        M.RUN_V2,
        "verify_terminal",
        lambda validated: M._read(Path(validated["output_directory"]) / "result.json"),
    )


def test_four_records_emit_four_checkpoint_smokes_and_remain_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    registration, _rows = _case(repo)
    _accept_v2(monkeypatch)
    output = repo / "evidence/smoke-plan.json"
    result = M.build_plan(repo, registration, output)

    assert result["status"] == "checkpoint_smoke_pending"
    assert result["model_comparison_ready"] is False
    assert result["technical_prepare_smoke_reusable"] is False
    assert result["pending_checkpoint_smoke_count"] == 4
    assert len({row["checkpoint_sha256"] for row in result["policies"]}) == 4
    assert all(row["smoke"]["status"] == "pending" for row in result["policies"])
    assert all(
        row["smoke"]["required_artifacts"] == list(M.SMOKE_ARTIFACTS)
        for row in result["policies"]
    )
    assert all(
        row["smoke"]["required_readiness_receipt_fields"]
        == list(M.READINESS_RECEIPT_FIELDS)
        for row in result["policies"]
    )
    assert result["labels_inferred_from_recording"] is False


def test_role_label_must_come_from_fixed_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    registration, rows = _case(repo)
    _accept_v2(monkeypatch)
    fixed_path = repo / rows[0]["fixed_eval_binding"]["path"]
    fixed = M._read(fixed_path)
    fixed["label"] = "unhealthy"
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    registration_body = M._read(registration)
    registration_body["policies"][0]["fixed_eval_binding"]["sha256"] = M._sha(
        fixed_path
    )
    registration.write_text(json.dumps(registration_body), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed evaluation"):
        M.build_plan(repo, registration, repo / "evidence/out.json")


def test_record_failure_atom_or_reused_checkpoint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    registration, rows = _case(repo)
    _accept_v2(monkeypatch)
    adapter = M._read(repo / rows[1]["record_adapter"]["path"])
    execution_path = repo / adapter["record_execution"]["path"]
    execution = M._read(execution_path)
    execution["failure_atom"] = {"status": "failed"}
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    adapter["record_execution"]["sha256"] = M._sha(execution_path)
    adapter_path = repo / rows[1]["record_adapter"]["path"]
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    body = M._read(registration)
    body["policies"][1]["record_adapter"]["sha256"] = M._sha(adapter_path)
    registration.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match="실패 atom"):
        M.build_plan(repo, registration, repo / "evidence/out.json")


def test_adapter_cannot_replace_terminal_video_with_another_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    registration, rows = _case(repo)
    _accept_v2(monkeypatch)
    replacement = _write(repo / "evidence/replacement.mp4", b"different-video")
    adapter_path = repo / rows[0]["record_adapter"]["path"]
    adapter = M._read(adapter_path)
    adapter["record"]["video"] = _ref(repo, replacement)
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    body = M._read(registration)
    body["policies"][0]["record_adapter"]["sha256"] = M._sha(adapter_path)
    registration.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal artifact"):
        M.build_plan(repo, registration, repo / "evidence/out.json")
