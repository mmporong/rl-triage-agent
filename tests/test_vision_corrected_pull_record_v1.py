from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import run_vision_corrected_pull_record_v1 as R


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_files(tmp_path: Path, logical: str = "corrected-s2220") -> tuple[Path, Path, Path]:
    row = R.CATALOG[logical]
    checkpoint = tmp_path / f"{logical}.pt"
    checkpoint.write_bytes(b"verified corrected checkpoint")
    correction = write(tmp_path / f"{logical}.correction.json", {
        "schema": "vision_selection_correction_result_v1",
        "status": "complete",
        "group": f"source-{1 if row['seed'] == 2220 else 2:02d}",
        "job_id": row["correction_job_id"],
        "checkpoint_sha256": sha(checkpoint),
        "training": {"arm": "CORRECTED", "seed": row["seed"],
                     "source_checkpoint_sha256": row["source_f1_checkpoint_sha256"],
                     "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}},
        "training_evidence": {}, "evaluation": {},
        "fixed_eval": {"none": "healthy", "corrected": "healthy"},
        "gpu_s": 1.0, "compute_usd": 0.1, "exit_code": 0,
    })
    approval = write(tmp_path / f"{logical}.approval.json", {
        "schema": "vision_corrected_record_approval_v1", "approved": True,
        "request_id": f"request-{row['seed']}", "corrected_job_id": row["correction_job_id"],
        "seed": row["seed"], "correction_id": "restore_body_frame_reward",
        "recovery_mode": "restart", "protocol_case_arm": {"case": "case_01", "arm": "corrected"},
        "correction_result_sha256": sha(correction), "receipt_verdict": "recovered",
    })
    return checkpoint, correction, approval


def test_template_preserves_profile_and_corrected_catalog() -> None:
    path = Path("bench/protocols/vision_corrected_pull_record_template_v1.json")
    value = R.validate_template(path)
    assert [row["logical_id"] for row in value["groups"]] == list(R.CHECKPOINT_IDS)
    assert all("source-s" not in row["logical_id"] for row in value["groups"])
    assert value["record_profile"]["frame_count"] == 625
    assert value["record_profile"]["fps"] == 50
    assert value["record_profile"]["resolution"] == {"width": 640, "height": 480}
    assert value["fixed_flags"] == R.FIXED_FLAGS


def test_verified_late_binding_accepts_dynamic_sha_and_rejects_cross_case(tmp_path: Path) -> None:
    checkpoint, correction, approval = binding_files(tmp_path)
    row = R._verified_binding("corrected-s2220", correction, approval)
    assert row["checkpoint"] == {"path": str(checkpoint.resolve()), "sha256": sha(checkpoint)}
    assert row["correction_job_id"] == "vision-s2220-corrected"
    changed = json.loads(approval.read_text())
    changed["corrected_job_id"] = "vision-s2221-corrected"
    approval.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="approval"):
        R._verified_binding("corrected-s2220", correction, approval)


def test_real_validate_job_shape_uses_corrected_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_path = write(repo / "protocol.json", {"bound": True})
    proof_path = write(repo / "proof.json", {"fresh": True})
    checkpoint = repo / "corrected.pt"
    checkpoint.write_bytes(b"corrected")
    lab = repo / "IsaacLab"
    lab.mkdir()
    python = repo / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    stage = repo / "stage"
    stage.mkdir()
    source_map = {"bench/run_vision_corrected_pull_record_v1.py": "0" * 64}
    protocol = {
        "protocol_id": "corrected-test", "source_commit": "a" * 40,
        "source_sha256_lf": source_map, "record_profile": R.RECORD.fixed_profile(),
        "checkpoints": [{"logical_id": "corrected-s2220",
                         "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}}],
        "_source": {},
    }
    monkeypatch.setattr(R, "validate_protocol", lambda *a, **k: protocol)
    monkeypatch.setattr(R, "validate_resource_proof", lambda *a, **k: {"fresh": True})
    monkeypatch.setattr(R.COMMON, "_directory", lambda path, anchor: Path(path))
    monkeypatch.setattr(R.COMMON, "_python_executable", lambda path: Path(path))
    monkeypatch.setattr(R.COMMON, "_verify_runtime_sources", lambda *a, **k: None)
    job_path = write(stage / "job.json", {
        "schema": R.JOB_SCHEMA, "mode": R.MODE,
        "job_id": "corrected-test-corrected-s2220",
        "source": {"repo_root": str(repo.resolve()), "commit": "a" * 40, "files_lf": source_map},
        "protocol": {"path": str(protocol_path.resolve()), "sha256_lf": R._sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(proof_path.resolve()), "sha256": sha(proof_path)},
        "isaaclab": str(lab.resolve()), "python": str(python.resolve()),
        "output_directory": str(stage / "output"), "profile": R.RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint),
                       "logical_id": "corrected-s2220"},
    })
    checked = R.validate_job(job_path, now=datetime.now(timezone.utc), require_fresh=False)
    assert checked["checkpoint_sha256"] == sha(checkpoint)
    bad = json.loads(job_path.read_text())
    bad["checkpoint"]["logical_id"] = "source-s2220-f1"
    job_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="late binding"):
        R.validate_job(job_path, require_fresh=False)


def test_child_contract_rebinds_run3_job_and_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def child(job: Path, **kwargs: object) -> int:
        observed["job_module"] = R.RUN3.JOB
        def fake_popen(command: list[str], **popen_kwargs: object) -> object:
            observed["command"] = command
            return object()
        R.RUN3._child_launcher(fake_popen)(
            [sys.executable, str(Path(R.RUN3.PREVIOUS.__file__).resolve()), "--sdk-child", "--job", str(job)]
        )
        return 7

    monkeypatch.setattr(R.RUN3, "sdk_child", child)
    assert R.sdk_child(tmp_path / "job.json") == 7
    assert observed["job_module"] is R
    command = observed["command"]
    assert isinstance(command, list) and command[1] == str(Path(R.__file__).resolve())
    assert R.RUN3.JOB is not R


def test_cli_child_uses_corrected_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    seen: list[Path] = []
    monkeypatch.setattr(R, "sdk_child", lambda path: seen.append(path) or 0)
    monkeypatch.setattr(R, "aaba_child", lambda path: seen.append(path) or 0)
    assert R.main(["--sdk-child", "--job", str(job)]) == 0
    assert R.main(["--aaba-child", "--job", str(job)]) == 0
    assert seen == [job, job]
    with pytest.raises(SystemExit):
        R.main(["--protocol", "p", "--resource-proof", "r", "--resource-sha256", "0" * 64,
                "--isaaclab", "i", "--python", "x", "--stage", "s",
                "--checkpoint-logical-id", "source-s2220-f1"])
