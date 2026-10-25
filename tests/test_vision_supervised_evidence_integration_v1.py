from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

A = importlib.import_module("vision_render_attestation_v1")
C = importlib.import_module("vision_render_context_v1")
E = importlib.import_module("vision_execution_evidence_v1")
R = importlib.import_module("record_vision_evidence_v1")
SHUTDOWN = importlib.import_module("vision_shutdown_policy_v1")
B2 = importlib.import_module("vision_brev_render_backend_v2")
B4 = importlib.import_module("vision_brev_render_backend_v4")
AF = importlib.import_module("test_vision_render_attestation_v1")
CF = importlib.import_module("test_vision_render_context_v1")
EF = importlib.import_module("test_vision_execution_evidence_v1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _closure(job: dict[str, Any], snapshot: dict[str, Any]) -> None:
    repo = Path(job["source"]["repo_root"])
    for relative in sorted(A.SUPERVISED_SOURCE_FILES):
        path = repo / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        job["source"]["files_lf"][relative] = digest
        snapshot["source_sha256_lf"][relative] = digest


def _supervised_capture(tmp_path: Path) -> dict[str, Any]:
    job_path, snapshot, python, job = AF._case(tmp_path / "case")
    template, _template_job, _template_result = EF._stage(tmp_path / "template")
    stage = (tmp_path / "record-stage").resolve()
    output = stage / "output"
    job["mode"] = R.RENDER_SYNC_SMOKE_MODE
    job["output_directory"] = str(output)
    _closure(job, snapshot)
    protocol = Path(job["protocol"]["path"])
    _write(protocol, {"source_commit": snapshot["source_anchor"],
                      "source_files_lf": job["source"]["files_lf"]})
    job["protocol"]["sha256"] = _sha(protocol)
    _write(job_path, job)

    def execute() -> dict[str, dict[str, str]]:
        stage.mkdir(parents=True)
        for source in template.glob("job.capture-*"):
            shutil.copy2(source, stage / source.name)
        shutil.copy2(template / "job.capture-readiness.json", stage / "job.capture-readiness.json")
        output.mkdir()
        report = output / "render_sync_smoke.json"
        shutil.copy2(template / "output/render_sync_smoke.json", report)
        stage_job = stage / "job.json"
        _write(stage_job, job)
        result = output / "result.json"
        _write(result, {
            "schema": R.RESULT_SCHEMA,
            "completed": True,
            "job_sha256": _sha(stage_job),
            "artifacts": {"render_sync_smoke": {
                "file": report.name, "sha256": _sha(report), "bytes": report.stat().st_size,
            }},
        })
        SHUTDOWN.apply(
            {"job_path": stage_job, "job_sha256": _sha(stage_job),
             "output_directory": output},
            EF._Settings(), startup_fast_shutdown=False,
        )
        manifest = E.write_manifest(stage_job, result, sdk_child_exit_code=0)
        execution = stage / "execution.json"
        _write(execution, {
            "schema": A.SUPERVISED_EXECUTION_SCHEMA,
            "completed": True,
            "job_sha256": _sha(stage_job),
            "exit_code": 0,
            "timed_out": False,
            "cleanup_verified": True,
            "gpu_before": [],
            "gpu_after": [],
            "gpu_before_observation": {},
            "gpu_after_observation": {},
            "source_before": snapshot,
            "source_after": snapshot,
            "result_sha256": _sha(result),
            "launch": {"pid": 1},
            "wall_s": 1.0,
            "auxiliary": {"file": manifest.name, "sha256": _sha(manifest)},
        })
        return {
            "execution": {"path": str(execution), "sha256": _sha(execution)},
            "result": {"path": str(result), "sha256": _sha(result)},
        }

    snapshots = iter([snapshot, snapshot])
    attestation = A.attest_capture(
        job_path,
        lambda: next(snapshots),
        python,
        (tmp_path / "attestation").resolve(),
        execute,
        clock=AF.Clock(),
        context_capture=lambda *args: C.capture_context(*args, runner=CF.Runner()),
    )
    return {
        "attestation": attestation,
        "context": attestation.parent / "context/context.receipt.json",
        "execution": stage / "execution.json",
        "result": output / "result.json",
        "manifest": stage / E.MANIFEST_NAME,
        "job_path": job_path,
        "stage_job": stage / "job.json",
        "protocol": protocol,
        "snapshot": snapshot,
        "job": job,
    }


def _verify(case: dict[str, Any]) -> dict[str, Any]:
    return A.verify_attestation(
        case["attestation"], case["job"], _sha(case["job_path"]), case["snapshot"],
        case["context"], case["execution"], case["result"],
        protocol_path=case["protocol"],
    )


def test_v1_attestation_remains_valid_without_auxiliary(tmp_path: Path) -> None:
    attestation, refs, job_path, snapshot, _python, job, _events = AF._capture(tmp_path)
    execution = Path(refs["execution"]["path"])
    result = Path(refs["result"]["path"])
    base_backend = object.__new__(B2.AttestedVisionBrevBackend)
    assert base_backend._execution_auxiliary(result, job) is None
    assert A.execution_evidence_path(execution, json.loads(execution.read_text()), job, result) is None
    verified = A.verify_attestation(
        attestation, job, _sha(job_path), snapshot,
        attestation.parent / "context/context.receipt.json", execution, result,
        protocol_path=Path(job["protocol"]["path"]),
    )
    assert verified["completed"] is True


def test_v2_relocated_manifest_is_bound_through_execution_and_attestation(tmp_path: Path) -> None:
    case = _supervised_capture(tmp_path)
    assert E.verify_manifest(case["manifest"], case["stage_job"], case["result"])[
        "supervisor_verification_complete"
    ] is True
    assert _verify(case)["completed"] is True
    execution = json.loads(case["execution"].read_text())
    supervised_backend = object.__new__(B4.SupervisedVisionBrevBackend)
    assert supervised_backend._execution_auxiliary(case["result"], case["job"]) == {
        "file": case["manifest"].name,
        "sha256": _sha(case["manifest"]),
    }
    assert A.execution_evidence_path(
        case["execution"], execution, case["job"], case["result"]
    ) == case["manifest"]


@pytest.mark.parametrize("mutation", ["missing", "tamper"])
def test_v2_missing_or_mutated_auxiliary_is_rejected(tmp_path: Path, mutation: str) -> None:
    case = _supervised_capture(tmp_path)
    if mutation == "missing":
        case["manifest"].unlink()
    else:
        body = json.loads(case["manifest"].read_text())
        body["capture_count"] = 3
        _write(case["manifest"], body)
    execution = json.loads(case["execution"].read_text())
    with pytest.raises((ValueError, FileNotFoundError)):
        A.execution_evidence_path(case["execution"], execution, case["job"], case["result"])


def test_manifest_mutation_after_attestation_is_rejected(tmp_path: Path) -> None:
    case = _supervised_capture(tmp_path)
    body = json.loads(case["manifest"].read_text())
    body["capture_count"] = 3
    _write(case["manifest"], body)
    with pytest.raises(ValueError):
        _verify(case)


def test_auxiliary_path_escape_and_top_reference_mismatch_are_rejected(tmp_path: Path) -> None:
    case = _supervised_capture(tmp_path)
    execution = json.loads(case["execution"].read_text())
    execution["auxiliary"]["file"] = "../job.execution-evidence.json"
    with pytest.raises(ValueError, match="단일 파일명"):
        A.execution_evidence_path(case["execution"], execution, case["job"], case["result"])

    execution = json.loads(case["execution"].read_text())
    execution["auxiliary"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        A.execution_evidence_path(case["execution"], execution, case["job"], case["result"])


def test_execution_schema_and_supervisor_source_must_move_together(tmp_path: Path) -> None:
    case = _supervised_capture(tmp_path)
    execution = json.loads(case["execution"].read_text())
    execution["schema"] = "vision_brev_record_execution_v1"
    execution.pop("auxiliary")
    with pytest.raises(ValueError, match="supervisor source"):
        A.execution_evidence_path(case["execution"], execution, case["job"], case["result"])

    execution = json.loads(case["execution"].read_text())
    no_supervisor = json.loads(json.dumps(case["job"]))
    no_supervisor["source"]["files_lf"].pop("bench/vision_play_worker_v3.py")
    with pytest.raises(ValueError, match="schema"):
        A.execution_evidence_path(case["execution"], execution, no_supervisor, case["result"])

    incomplete = json.loads(json.dumps(case["job"]))
    incomplete["source"]["files_lf"].pop("bench/vision_shutdown_policy_v1.py")
    with pytest.raises(ValueError, match="schema"):
        A.execution_evidence_path(case["execution"], execution, incomplete, case["result"])
