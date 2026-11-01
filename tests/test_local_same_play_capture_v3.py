from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench import local_same_play_capture_v3 as C


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def make_job(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    job_path = tmp_path / "run" / "job.json"
    job = {"schema": C.V1.JOB_SCHEMA, "mode": "full", "frames": C.V1.FULL_FRAMES,
           "fps": C.V1.FPS, "protocol": {"path": "bench/protocols/local-v3.json",
                                           "sha256": "a" * 64},
           "source_anchor": "b" * 40, "execution_anchor": "c" * 40,
           "checkpoint_path": str(checkpoint), "checkpoint_sha256": C.V1._sha(checkpoint),
           "checkpoint_logical_id": "s2430-none",
           "source": {"commit": "b" * 40, "files_lf": {"bench/vision_fxaa_policy_v1.py": "d" * 64}},
           "output_directory": str(tmp_path / "run" / "capture"),
           "sidecar_directory": str(tmp_path / "run.sidecar-v2"),
           "job_path": str(job_path), "source_id": "local-s2430-none-full-v3"}
    write(job_path, job)
    return job_path


def test_validated_job_matches_production_record_contract_without_cloud_claims(tmp_path: Path) -> None:
    value = C._validated_job(make_job(tmp_path))
    record = C._record()
    assert value["job"]["mode"] == record.RECORD_MODE == "record"
    assert value["job"]["job_id"] == "local-s2430-none-full-v3"
    assert value["job"]["profile"] == record.fixed_profile()
    assert set(value["job"]) == set(C.JOB_KEYS)
    assert all(key in value for key in C.WORKER_KEYS)
    assert "resource" not in value and "remote_gpu_admission" not in value["job"]


def test_actual_capture_factory_accepts_mode_profile_until_first_sdk_boundary(tmp_path: Path) -> None:
    validated = C._validated_job(make_job(tmp_path))
    capture = C.importlib.import_module("vision_pull_record_capture_v2")

    class Boundary(RuntimeError):
        """Isaac USD context 호출 지점을 표시한다."""

    def boundary() -> None:
        raise Boundary("usd_context")

    factory = capture.make_capture_factory(validated, object(), dependencies={"usd_context": boundary})
    with pytest.raises(Boundary, match="usd_context"):
        factory(object(), object(), object(), C._record().fixed_profile()["camera"])
    wrong = dict(C._record().fixed_profile()["camera"])
    wrong["fps"] = 49
    with pytest.raises(ValueError, match="fixed profile"):
        factory(object(), object(), object(), wrong)


def test_old_v2_mapping_is_rejected_by_actual_capture_factory(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    old = C.V1._validated_job(job)
    capture = C.importlib.import_module("vision_pull_record_capture_v2")
    with pytest.raises(ValueError, match="record mode"):
        capture.make_capture_factory(old, object())


def test_build_job_keeps_sidecar_sibling_and_v3_child_mapping(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    inspection = tmp_path / "inspection.json"
    write(inspection, {"checkpoint": {"path": str(checkpoint), "sha256": C.V1._sha(checkpoint)}})
    protocol = {"source_anchor": "a" * 40,
                "checkpoint": {"sha256": C.V1._sha(checkpoint), "logical_id": "s2430-none"},
                "source_files_lf": {"bench/vision_fxaa_policy_v1.py": "b" * 64}}
    monkeypatch.setattr(C, "validate_protocol", lambda *args, **kwargs: protocol)
    path = tmp_path / "protocol.json"
    write(path, {})
    stage = tmp_path / "run-v3"
    job = C.build_job(path, inspection, tmp_path / "lab", stage, mode="full",
                      execution_anchor="c" * 40, verify_anchor=False)
    assert Path(job["sidecar_directory"]) == tmp_path / "run-v3.sidecar-v2"
    assert C._validated_job(stage / "job.json")["job"]["mode"] == "record"


def test_sdk_child_cli_forwards_job_and_exact_kit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(C, "sdk_child", lambda path: seen.append(path) or 0)
    assert C.main(["sdk-child", "job.json", *C.V1.KIT_ARGS]) == 0
    assert seen == [Path("job.json")]


def test_v3_source_files_are_bound_to_anchor_blobs(tmp_path: Path) -> None:
    originals = {C.SOURCE_FILES[0]: "source-v3\n", C.SOURCE_FILES[1]: "test-v3\n"}
    for relative, content in originals.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ri335@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "RI335 Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *C.SOURCE_FILES], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "anchor"], cwd=tmp_path, check=True, capture_output=True)
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                            capture_output=True, text=True).stdout.strip()
    source_map = {name: C.V1._sha(tmp_path / name, lf=True) for name in C.SOURCE_FILES}
    C._verify_source_blobs(tmp_path, anchor, source_map)
    (tmp_path / C.SOURCE_FILES[0]).write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current source changed"):
        C._verify_source_blobs(tmp_path, anchor, source_map)
