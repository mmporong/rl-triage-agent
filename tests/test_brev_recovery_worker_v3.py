from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

W = importlib.import_module("brev_recovery_worker_v3")


def _job(tmp_path: Path, run_name: str) -> dict:
    repo, lab, stage = tmp_path / "repo", tmp_path / "lab", tmp_path / "stage"
    repo.mkdir(parents=True)
    lab.mkdir()
    stage.mkdir()
    protocol = repo / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    hashes = {name: "a" * 64 for name in (
        "bench/residual_faults_v1.py",
        "bench/residual_training_v1.py",
        "bench/residual_probe_v1.py",
        "bench/r1_repair_training.py",
        "bench/hidden_faults.py",
        "bench/optimizer_faults.py",
        "evals/optimizer_observability.py",
    )}
    return {
        "schema": "brev_recovery_worker_job_v1",
        "mode": "train",
        "stage": str(stage),
        "lab": str(lab),
        "repo": str(repo),
        "protocol_path": str(protocol),
        "protocol_sha256_lf": "b" * 64,
        "source_commit_sha": "c" * 40,
        "source_sha256_lf": hashes,
        "runtime_source_sha256_lf": hashes,
        "python_package_source_sha256_lf": hashes,
        "arm": "NONE",
        "seed": 2220,
        "run_name": run_name,
        "iterations": 300,
        "manifest_path": str(stage / "training.manifest.json"),
        "records_path": str(stage / "optimizer.records.jsonl"),
        "readback_path": str(stage / "training.readback.json"),
        "telemetry_path": str(stage / "telemetry.json"),
        "source_checkpoint_sha256": None,
    }


def test_preflight_rejects_the_actual_hyphenated_run_name_before_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="영문·숫자·밑줄"):
        W.validate_job(_job(tmp_path, "vision-s2220-none"))


def test_preflight_accepts_the_new_underscore_run_name(tmp_path: Path) -> None:
    assert W.validate_job(_job(tmp_path, "vision_s2220_none"))["run_name"] == "vision_s2220_none"


def test_main_preserves_failure_message_and_traceback(tmp_path: Path) -> None:
    job_path, output = tmp_path / "job.json", tmp_path / "worker.json"
    job_path.write_text(json.dumps(_job(tmp_path / "case", "vision-s2220-none")), encoding="utf-8")

    assert W.main(["--job", str(job_path), "--output", str(output)]) == 1
    failure = json.loads(output.read_text(encoding="utf-8"))
    sidecar = json.loads(output.with_name("worker.failure.json").read_text(encoding="utf-8"))
    assert failure == sidecar
    assert failure["completed"] is False
    assert failure["error_type"] == "ValueError"
    assert failure["error_message"] == "run_name은 영문·숫자·밑줄만 쓴다"
    assert "residual_training_v1.py" in failure["traceback"]
