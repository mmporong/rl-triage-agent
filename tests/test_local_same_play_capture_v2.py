from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench import local_same_play_capture_v2 as C


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def inspection(tmp_path: Path) -> tuple[Path, Path, str]:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    path = tmp_path / "inspection.json"
    write(path, {"checkpoint": {"path": str(checkpoint), "sha256": C.V1._sha(checkpoint)}})
    return path, checkpoint, C.V1._sha(checkpoint)


def registered_protocol(checkpoint_sha: str) -> dict:
    return {"schema": C.SCHEMA, "status": "registered_before_local_gpu",
            "source_anchor": "a" * 40, "checkpoint": {"sha256": checkpoint_sha,
            "logical_id": "s2430-none"}, "source_files_lf": {
                "bench/record_agentic_state_sidecar_v1.py": C.V1._sha(
                    C.ROOT / "bench/record_agentic_state_sidecar_v1.py", lf=True)}}


def test_build_job_places_sidecar_in_unique_sibling_and_actual_guard_accepts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path, _, checkpoint_sha = inspection(tmp_path)
    protocol = registered_protocol(checkpoint_sha)
    monkeypatch.setattr(C, "validate_protocol", lambda *args, **kwargs: protocol)
    stage = tmp_path / "run-full-v2"
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, {})
    job = C.build_job(protocol_path, input_path, tmp_path / "lab", stage,
                      mode="full", execution_anchor="b" * 40, verify_anchor=False)
    assert Path(job["sidecar_directory"]) == tmp_path / "run-full-v2.sidecar-v2"
    validated = C.V1._validated_job(stage / "job.json")
    sidecar = C.importlib.import_module("record_agentic_state_sidecar_v1")
    assert sidecar.check_destination(
        Path(validated["sidecar_directory"]), Path(validated["output_directory"]),
        Path(validated["job_path"]).parent) == Path(job["sidecar_directory"])
    assert not Path(job["sidecar_directory"]).exists()
    assert validated["job"]["source"] == job["source"]


def test_old_stage_local_sidecar_is_still_rejected(tmp_path: Path) -> None:
    sidecar = C.importlib.import_module("record_agentic_state_sidecar_v1")
    stage = tmp_path / "run"
    with pytest.raises(ValueError, match="겹친다"):
        sidecar.check_destination(stage / "sidecar", stage / "capture", stage)


def test_sdk_child_cli_forwards_exact_worker_job_and_kit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[Path] = []
    monkeypatch.setattr(C, "sdk_child", lambda path: received.append(path) or 0)
    assert C.main(["sdk-child", "job.json", *C.V1.KIT_ARGS]) == 0
    assert received == [Path("job.json")]


def test_execute_command_uses_v2_child_and_surfaces_sealed_child_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isaac = tmp_path / "python.bat"
    isaac.write_text("@echo off\n", encoding="utf-8")
    stage = tmp_path / "stage"
    input_path, checkpoint, checkpoint_sha = inspection(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, {})
    commands: list[list[str]] = []

    def build(*args, **kwargs):
        stage.mkdir()
        output = stage / "capture"
        output.mkdir()
        write(output / "failure.json", {"error_type": "ValueError",
                                         "error_message": "sidecar destination overlap"})
        job = {"output_directory": str(output), "checkpoint_sha256": checkpoint_sha,
               "source_anchor": "a" * 40, "sidecar_directory": str(tmp_path / "stage.sidecar-v2"),
               "source_id": "local-s2430-none-full-v2"}
        write(stage / "job.json", {**job, "checkpoint_path": str(checkpoint)})
        return job

    def runner(folder, command, cwd, timeout):
        commands.append(command)
        return {"exit_code": 1, "cleanup_verified": True}

    leases = lambda evidence: {"owner": C.V1.OWNER, "state": "running", "leaseId": "lease-1"}
    monkeypatch.setattr(C, "build_job", build)
    with pytest.raises(RuntimeError, match="sidecar destination overlap"):
        C.execute(protocol_path, input_path, "b" * 40, tmp_path, stage, isaac,
                  mode="full", runner=runner, lease_reader=leases)
    assert commands[0][2] == "sdk-child" and commands[0][-2:] == C.V1.KIT_ARGS


def test_prepare_extends_v1_closure_without_changing_base(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: Path) -> None:
    base_path, input_path = tmp_path / "base.json", tmp_path / "inspection.json"
    write(base_path, {})
    write(input_path, {})
    base = {"source_files_lf": {"bench/local_same_play_capture_v1.py": "a" * 64},
            "runtime_pins": {}, "training_runtime_pins": {}, "source_batch": {},
            "checkpoint": {}, "platform": {}, "gpu": {}, "play": {}, "shutdown": {}}
    monkeypatch.setattr(C.V1, "validate_protocol", lambda *args, **kwargs: base)
    monkeypatch.setattr(C.V1, "_sha", lambda path, lf=False: "b" * 64)
    out = tmp_path / "v2.json"
    value = C.prepare_protocol(base_path, input_path, tmp_path / "lab", out, repo=tmp_path)
    assert value["layout"] == C.LAYOUT and value["status"] == "candidate_not_registered"
    assert set(value["source_files_lf"]) == {
        "bench/local_same_play_capture_v1.py", *C.SOURCE_FILES}
    assert base["source_files_lf"] == {"bench/local_same_play_capture_v1.py": "a" * 64}


def test_v2_import_has_no_sdk_or_torch_global() -> None:
    assert "isaaclab" not in C.__dict__ and "torch" not in C.__dict__
    assert C.V1.compose_full_worker is not None and C.V1.verify_result is not None


def test_v2_source_files_are_bound_to_current_bytes_and_source_anchor(tmp_path: Path) -> None:
    originals = {C.SOURCE_FILES[0]: "source-v1\n", C.SOURCE_FILES[1]: "test-v1\n"}
    for relative, content in originals.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ri333@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "RI333 Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *C.SOURCE_FILES], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "source anchor"], cwd=tmp_path, check=True,
                   capture_output=True)
    source_anchor = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True,
        text=True).stdout.strip()
    source_map = {name: C.V1._sha(tmp_path / name, lf=True) for name in C.SOURCE_FILES}
    C._verify_v2_source_blobs(tmp_path, source_anchor, source_map)

    (tmp_path / C.SOURCE_FILES[0]).write_text("source-modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current source changed"):
        C._verify_v2_source_blobs(tmp_path, source_anchor, source_map)
    (tmp_path / C.SOURCE_FILES[0]).write_text(originals[C.SOURCE_FILES[0]], encoding="utf-8")

    (tmp_path / C.SOURCE_FILES[1]).write_text("test-anchor-two\n", encoding="utf-8")
    subprocess.run(["git", "add", C.SOURCE_FILES[1]], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "wrong anchor"], cwd=tmp_path, check=True,
                   capture_output=True)
    wrong_anchor = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True,
        text=True).stdout.strip()
    (tmp_path / C.SOURCE_FILES[1]).write_text(originals[C.SOURCE_FILES[1]], encoding="utf-8")
    with pytest.raises(ValueError, match="source anchor blob changed"):
        C._verify_v2_source_blobs(tmp_path, wrong_anchor, source_map)
