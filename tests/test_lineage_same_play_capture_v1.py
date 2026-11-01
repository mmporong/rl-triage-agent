# pyright: reportMissingImports=false
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bench import lineage_same_play_capture_v1 as C


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str = "F2"
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    lab = tmp_path / "lab"
    runtime_names = {
        *C.V1.RUNTIME_PINS,
        *C.PLAYBACK_RUNTIME_FILES,
        "runtime/training.py",
    }
    for name in runtime_names:
        path = lab / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime {name}\n", encoding="utf-8")
    index = tmp_path / "collected.json"
    write(index, {"index": True})
    source_root = tmp_path / "source-batch"
    job_id = f"{arm.lower()}-s2554"
    stage = source_root / job_id
    checkpoint = stage / "training" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "job_id": job_id,
        "arm": arm,
        "runtime_source_sha256_lf": {
            "runtime/training.py": C.V1._sha(lab / "runtime/training.py", lf=True)
        },
    }
    write(stage / "manifest.json", manifest)
    write(
        stage / "training" / "training_evidence.json", {"checkpoint": str(checkpoint)}
    )
    (stage / "training" / "records.jsonl").write_text(
        '{"update":300}\n', encoding="utf-8"
    )
    write(stage / "fixed-result.json", {"conditions": 26})
    binding = {
        "checkpoint_sha256": C.V1._sha(checkpoint),
        "telemetry_sha256": "1" * 64,
        "manifest_sha256": C.V1._sha(stage / "manifest.json"),
        "evidence_sha256": C.V1._sha(stage / "training" / "training_evidence.json"),
        "records_sha256": C.V1._sha(stage / "training" / "records.jsonl"),
        "fixed_result_sha256": C.V1._sha(stage / "fixed-result.json"),
    }
    row = {
        "job_id": job_id,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": binding["checkpoint_sha256"],
            "iteration": 299,
            "size_bytes": checkpoint.stat().st_size,
        },
        "evidence_sha256": binding["evidence_sha256"],
        "records_sha256": binding["records_sha256"],
        "fixed26": {"conditions": 26, "result_sha256": binding["fixed_result_sha256"]},
    }
    write(stage / "accepted.json", row)
    result = {
        "status": "complete",
        "source_jobs_completed": 20,
        "fixed26_completed": 20,
        "records": [row, *({"job_id": f"other-{i}"} for i in range(19))],
    }
    write(source_root / "result.json", result)
    public = {"sources": {"source-05": binding}}
    metadata = {
        "sources": {
            job_id: {"public_id": "source-05", "binding": binding, "manifest": manifest}
        },
        "source_result": C.V1._ref(source_root / "result.json"),
    }
    monkeypatch.setattr(C.INPUTS, "checked", lambda _path: (public, metadata))
    return index, lab, public, metadata


def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str = "F2"
) -> tuple[Path, Path, Path, dict[str, Any]]:
    index, lab, _, _ = source_fixture(tmp_path, monkeypatch, arm)
    protocol = tmp_path / "protocol.json"
    value = C.prepare_protocol(
        index, "source-05", lab, protocol, source_anchor="a" * 40
    )
    return index, lab, protocol, value


def test_prepare_reconsumes_actual_source_signatures_and_full_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, value = prepared(tmp_path, monkeypatch)
    assert value["status"] == "registered_before_local_gpu"
    assert value["source_input"]["source_id"] == "source-05"
    assert value["private"] == {
        "job_id": "f2-s2554",
        "arm": "F2",
        "fault_metadata_model_visible": False,
    }
    assert value["playback"]["kind"] == "class_pre_env"
    assert value["playback"]["all_go2_joints_required"] == 12
    assert value["source_input"]["index"] == {
        "logical_id": C.INDEX_LOGICAL_ID,
        "sha256": C.V1._sha(index),
    }
    assert value["source_result"] == {
        "logical_id": C.SOURCE_RESULT_LOGICAL_ID,
        "sha256": C.V1._sha(tmp_path / "source-batch" / "result.json"),
    }
    assert set(value["checkpoint"]) == {"logical_id", "sha256", "iteration"}
    serialized = json.dumps(value)
    assert str(tmp_path) not in serialized and "C:\\Users\\" not in serialized
    assert len(value["source_files_lf"]) >= 235
    checked = C.validate_protocol(protocol, index, lab, verify_anchor=False)
    assert checked == value


def test_validate_rejects_changed_fixed20_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, _value = prepared(tmp_path, monkeypatch)
    fixed = tmp_path / "source-batch" / "f2-s2554" / "fixed-result.json"
    write(fixed, {"conditions": 25})
    with pytest.raises(ValueError, match="서명"):
        C.validate_protocol(protocol, index, lab, verify_anchor=False)


def test_f5_is_honest_hold_and_cannot_build_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, value = prepared(tmp_path, monkeypatch, "F5")
    assert value["status"] == "hold_unsupported"
    assert value["playback"] == {
        "kind": "environment_post_create",
        "status": "hold_unsupported",
        "reason": "current recorder child has no verified post-create material restore lifecycle",
        "friction_scale": 0.25,
        "replacement_arm_allowed": False,
    }
    with pytest.raises(ValueError, match="HOLD"):
        C.build_job(
            protocol,
            index,
            lab,
            tmp_path / "stage",
            mode="full",
            execution_anchor="b" * 40,
            verify_anchor=False,
        )


def test_f2_build_is_full_625_at_50fps_and_anonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, _ = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="full mode"):
        C.build_job(
            protocol,
            index,
            lab,
            tmp_path / "smoke",
            mode="smoke",
            execution_anchor="b" * 40,
            verify_anchor=False,
        )
    job = C.build_job(
        protocol,
        index,
        lab,
        tmp_path / "full",
        mode="full",
        execution_anchor="b" * 40,
        verify_anchor=False,
    )
    assert (job["frames"], job["fps"], job["checkpoint_logical_id"]) == (
        625,
        50,
        "source-05",
    )
    assert job["source_id"] == "lineage-source-05-full-v1"
    assert "f2" not in job["source_id"].lower()


def test_f2_worker_routes_existing_playback_callback_without_global_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = object()
    seen: dict[str, Any] = {}

    class Worker:
        @staticmethod
        def run_policy_record(
            validated: Mapping[str, Any], app: Any, **kwargs: Any
        ) -> dict[str, bool]:
            seen["run_real"] = kwargs["run_real"]
            return {"ok": True}

    class Render:
        @staticmethod
        def render_sync_capture_factory(factory: Any) -> str:
            return "wrapped-factory"

    class Sidecar:
        ABORT = "abort_recording"

        @staticmethod
        def with_sidecar_worker(*args: Any, **kwargs: Any) -> Any:
            seen["factory"] = kwargs["capture_factory_maker"]
            return kwargs["run_worker"]

        @staticmethod
        def _default_reader_factory(*args: Any) -> None:
            return None

    monkeypatch.setattr(C.PLAYBACK, "make_run_real", lambda registration: marker)
    validated = {
        "protocol": {
            "private": {"arm": "F2"},
            "playback": {"registration": {"fault_id": "F2"}},
        },
        "sidecar_directory": tmp_path / "sidecar",
        "source_id": "lineage-source-05-full-v1",
        "checkpoint_sha256": "a" * 64,
        "frames": 625,
    }
    modules = {
        "worker": Worker,
        "render": Render,
        "capture": SimpleNamespace(make_capture_factory=object()),
        "sidecar": Sidecar,
    }
    result = C._full_worker(validated, object(), modules, tmp_path / "report.json")
    assert result == {"ok": True}
    assert seen == {"run_real": marker, "factory": "wrapped-factory"}


def test_f2_playback_proof_requires_actual_method_all_twelve_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    registration = {"fault_id": "F2"}
    receipt = {
        "status": "completed",
        "registration": registration,
        "restore_completed": True,
        "installation": {
            "fault_id": "F2",
            "class_method_changed": True,
            "after": {
                "module": "hidden_faults",
                "qualname": "_before_env.<locals>._clip_effort",
            },
        },
    }
    write(stage / "playback-receipt.json", receipt)
    joints = [f"joint-{i}" for i in range(12)]
    manifest = {
        "joint_names": joints,
        "actuators": {
            "base_legs": {
                "clip_effort_function": "hidden_faults._before_env.<locals>._clip_effort",
                "joint_names": joints,
            }
        },
    }
    sidecar = SimpleNamespace(load_sidecar=lambda _path: (manifest, []))
    original = C.importlib.import_module
    monkeypatch.setattr(
        C.importlib,
        "import_module",
        lambda name: (
            sidecar if name == "record_agentic_state_sidecar_v1" else original(name)
        ),
    )
    validated = {
        "protocol": {
            "private": {"arm": "F2"},
            "playback": {"registration": registration},
        },
        "sidecar_directory": tmp_path / "sidecar",
    }
    proof = C._playback_proof(stage, validated)
    assert proof["joint_count"] == 12
    assert proof["all_go2_joints_verified"] is True
    assert proof["restore_completed"] is True


def test_production_validation_checks_public_protocol_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, _ = prepared(tmp_path, monkeypatch)
    seen: list[tuple[Path, Path, str, str]] = []
    monkeypatch.setattr(C, "_verify_source_anchor", lambda *args: None)
    monkeypatch.setattr(
        C.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        C.V1,
        "_verify_protocol_blob",
        lambda path, repo, source, execution: seen.append(
            (path, repo, source, execution)
        ),
    )
    C.validate_protocol(
        protocol, index, lab, execution_anchor="b" * 40, verify_anchor=True
    )
    assert seen == [(protocol, C.ROOT, "a" * 40, "b" * 40)]


def test_export_artifact_reports_f5_hold_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, _ = prepared(tmp_path, monkeypatch, "F5")
    artifact = C.export_artifact(
        protocol, index, lab, None, execution_anchor="b" * 40, verify_anchor=False
    )
    assert artifact["schema"] == C.ARTIFACT_SCHEMA
    assert artifact["status"] == "hold_unsupported"
    assert artifact["frames"] is None and artifact["fps"] == 50
    assert artifact["proofs"] is None and artifact["v01_specs"] == []
    assert artifact["replacement_capture"] is False
    assert artifact["quality"]["status"] == "unavailable_hold"
    assert "arm" not in artifact and "fault" not in artifact


def test_export_artifact_binds_full_proofs_v01_and_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, value = prepared(tmp_path, monkeypatch, "NONE")
    stage = tmp_path / "capture"
    C.build_job(
        protocol,
        index,
        lab,
        stage,
        mode="full",
        execution_anchor="b" * 40,
        verify_anchor=False,
    )
    evidence = stage / "evidence"
    evidence.mkdir(parents=True)
    specs = []
    for number in (1, 2):
        spec = evidence / f"window_{number}_v01_spec.json"
        write(spec, {"window": number})
        specs.append(spec)
    raw = evidence / "raw_trace.json"
    write(raw, {"frames": 625})
    worker = stage / "worker-report.json"
    write(
        worker,
        {
            "artifacts": {
                "spec_1": {"sha256": C.V1._sha(specs[0])},
                "spec_2": {"sha256": C.V1._sha(specs[1])},
            }
        },
    )
    proof_paths = {"worker_report": worker, "raw_trace": raw}
    for name in (
        "native_manifest",
        "sidecar_manifest",
        "sidecar_records",
        "render_sync",
        "same_pts",
        "shutdown",
        "app_close_entered",
    ):
        path = stage / f"{name}.json"
        write(path, {"proof": name})
        proof_paths[name] = path
    proofs = {name: C.V1._ref(path) for name, path in proof_paths.items()}
    checkpoint = value["checkpoint"]
    write(
        stage / "execution-result-lineage-v1.json",
        {
            "schema": C.RESULT_SCHEMA,
            "status": "technical_complete",
            "mode": "full",
            "protocol": C.V1._ref(protocol),
            "source_id": "source-05",
            "checkpoint_sha256": checkpoint["sha256"],
        },
    )
    write(
        stage / "parent-finalized-lineage-result-v1.json",
        {
            "schema": C.FINAL_SCHEMA,
            "status": "technical_complete",
            "mode": "full",
            "frames": 625,
            "fps": 50,
            "source_id": "source-05",
            "checkpoint_sha256": checkpoint["sha256"],
            "base_v3": {
                "schema": C.V1.RESULT_SCHEMA,
                "status": "technical_complete",
                "frames": 625,
                "fps": 50,
                "same_pts": True,
                "proofs": proofs,
            },
            "playback": {
                "kind": "none",
                "receipt": None,
                "all_go2_joints_verified": None,
            },
            "capture_approved": False,
            "model_ready": False,
            "used_for_decision": False,
        },
    )
    artifact = C.export_artifact(
        protocol, index, lab, stage, execution_anchor="b" * 40, verify_anchor=False
    )
    assert artifact["status"] == "available"
    assert (artifact["frames"], artifact["fps"]) == (625, 50)
    assert len(artifact["v01_specs"]) == 2
    assert artifact["proofs"]["raw_trace"] == C.V1._ref(raw)
    assert artifact["quality"] == {
        "status": "technical_complete_not_approved",
        "receipt": C.V1._ref(stage / "parent-finalized-lineage-result-v1.json"),
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
    }
    before = {
        path.relative_to(stage): (path.stat().st_size, C.V1._sha(path))
        for path in stage.rglob("*")
        if path.is_file()
    }
    assert (
        C.export_artifact(
            protocol,
            index,
            lab,
            stage,
            execution_anchor="b" * 40,
            verify_anchor=False,
        )
        == artifact
    )
    after = {
        path.relative_to(stage): (path.stat().st_size, C.V1._sha(path))
        for path in stage.rglob("*")
        if path.is_file()
    }
    assert after == before
    write(stage / "playback-receipt.json", {"unexpected": True})
    with pytest.raises(ValueError, match="no-injection"):
        C.export_artifact(
            protocol,
            index,
            lab,
            stage,
            execution_anchor="b" * 40,
            verify_anchor=False,
        )


def test_verify_artifact_recomputes_exact_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {"schema": C.ARTIFACT_SCHEMA, "status": "hold_unsupported"}
    calls: list[tuple[Any, ...]] = []

    def recompute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((*args, kwargs))
        return expected

    monkeypatch.setattr(C, "export_artifact", recompute)
    assert (
        C.verify_artifact(
            expected,
            tmp_path / "protocol",
            tmp_path / "index",
            tmp_path / "lab",
            execution_anchor="a" * 40,
        )
        == expected
    )
    assert calls[0][3] is None
    with pytest.raises(ValueError, match="재계산"):
        C.verify_artifact(
            {**expected, "changed": True},
            tmp_path / "protocol",
            tmp_path / "index",
            tmp_path / "lab",
            execution_anchor="a" * 40,
        )


def test_export_recomputes_f2_playback_receipt_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, lab, protocol, value = prepared(tmp_path, monkeypatch, "F2")
    stage = tmp_path / "capture-f2"
    C.build_job(
        protocol,
        index,
        lab,
        stage,
        mode="full",
        execution_anchor="b" * 40,
        verify_anchor=False,
    )
    evidence = stage / "evidence"
    evidence.mkdir()
    specs = []
    for number in (1, 2):
        spec = evidence / f"window_{number}_v01_spec.json"
        write(spec, {"window": number})
        specs.append(spec)
    raw = evidence / "raw_trace.json"
    write(raw, {"frames": 625})
    worker = stage / "worker-report.json"
    write(
        worker,
        {
            "artifacts": {
                "spec_1": {"sha256": C.V1._sha(specs[0])},
                "spec_2": {"sha256": C.V1._sha(specs[1])},
            }
        },
    )
    proof_paths = {"worker_report": worker, "raw_trace": raw}
    for name in (
        "native_manifest",
        "sidecar_manifest",
        "sidecar_records",
        "render_sync",
        "same_pts",
        "shutdown",
        "app_close_entered",
    ):
        path = stage / f"{name}.json"
        write(path, {"proof": name})
        proof_paths[name] = path
    proofs = {name: C.V1._ref(path) for name, path in proof_paths.items()}
    checkpoint = value["checkpoint"]
    write(
        stage / "execution-result-lineage-v1.json",
        {
            "schema": C.RESULT_SCHEMA,
            "status": "technical_complete",
            "mode": "full",
            "protocol": C.V1._ref(protocol),
            "source_id": "source-05",
            "checkpoint_sha256": checkpoint["sha256"],
        },
    )
    registration = value["playback"]["registration"]
    receipt = {
        "status": "completed",
        "registration": registration,
        "restore_completed": True,
        "installation": {
            "fault_id": "F2",
            "class_method_changed": True,
            "after": {
                "module": "hidden_faults",
                "qualname": "_before_env.<locals>._clip_effort",
            },
        },
    }
    receipt_path = stage / "playback-receipt.json"
    write(receipt_path, receipt)
    joints = [f"joint-{number}" for number in range(12)]
    manifest = {
        "joint_names": joints,
        "actuators": {
            "base_legs": {
                "clip_effort_function": "hidden_faults._before_env.<locals>._clip_effort",
                "joint_names": joints,
            }
        },
    }
    sidecar = SimpleNamespace(load_sidecar=lambda _path: (manifest, []))
    original = C.importlib.import_module
    monkeypatch.setattr(
        C.importlib,
        "import_module",
        lambda name: (
            sidecar if name == "record_agentic_state_sidecar_v1" else original(name)
        ),
    )
    playback = C._playback_proof(
        stage, C._validated_job(stage / "job.json"), seal=False
    )
    write(
        stage / "parent-finalized-lineage-result-v1.json",
        {
            "schema": C.FINAL_SCHEMA,
            "status": "technical_complete",
            "mode": "full",
            "frames": 625,
            "fps": 50,
            "source_id": "source-05",
            "checkpoint_sha256": checkpoint["sha256"],
            "base_v3": {
                "schema": C.V1.RESULT_SCHEMA,
                "status": "technical_complete",
                "frames": 625,
                "fps": 50,
                "same_pts": True,
                "proofs": proofs,
            },
            "playback": playback,
            "capture_approved": False,
            "model_ready": False,
            "used_for_decision": False,
        },
    )
    artifact = C.export_artifact(
        protocol,
        index,
        lab,
        stage,
        execution_anchor="b" * 40,
        verify_anchor=False,
    )
    before = {
        path.relative_to(stage): (path.stat().st_size, C.V1._sha(path))
        for path in stage.rglob("*")
        if path.is_file()
    }
    assert (
        C.export_artifact(
            protocol,
            index,
            lab,
            stage,
            execution_anchor="b" * 40,
            verify_anchor=False,
        )
        == artifact
    )
    assert before == {
        path.relative_to(stage): (path.stat().st_size, C.V1._sha(path))
        for path in stage.rglob("*")
        if path.is_file()
    }

    receipt_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        C.export_artifact(
            protocol,
            index,
            lab,
            stage,
            execution_anchor="b" * 40,
            verify_anchor=False,
        )
    write(receipt_path, receipt)
    manifest["actuators"]["base_legs"]["clip_effort_function"] = "changed.method"
    with pytest.raises(ValueError, match="actual clip method"):
        C.export_artifact(
            protocol,
            index,
            lab,
            stage,
            execution_anchor="b" * 40,
            verify_anchor=False,
        )


def test_sdk_child_cli_keeps_exact_kit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(C, "sdk_child", lambda path: seen.append(path) or 0)
    assert C.main(["sdk-child", "job.json", *C.V1.KIT_ARGS]) == 0
    assert seen == [Path("job.json")]
