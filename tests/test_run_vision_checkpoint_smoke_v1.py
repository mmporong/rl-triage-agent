from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_checkpoint_smoke_v1 as R  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = _write(tmp_path / "job.json", {"mode": R.JOB.MODE})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    return {
        "job": {"mode": R.JOB.MODE},
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": tmp_path / "output",
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def test_policy_probe_requires_actual_environment_step(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    probe = {
        "schema": "vision_checkpoint_policy_probe_v1",
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "command": [1.0, 0.0, 0.0],
        "observation": {"shape": [1, 48], "dtype": "torch.float32", "sha256": "a" * 64},
        "action": {"shape": [1, 12], "dtype": "torch.float32", "sha256": "b" * 64},
        "environment_steps": 1,
        "step_counter_before": 4,
        "step_counter_after": 5,
        "reset_required_before_capture": True,
        "actual_environment_observation_used": True,
        "pure_checkpoint_read_only": False,
    }
    assert R._probe(probe, validated["checkpoint_sha256"])
    probe["pure_checkpoint_read_only"] = True
    assert not R._probe(probe, validated["checkpoint_sha256"])


def test_sdk_child_preserves_first_failure_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    log: list[str] = []

    class App:
        def close(self) -> None:
            log.append("close")

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        log.append("run")
        raise FileNotFoundError("checkpoint dependency missing")

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: validated)
    code = R.sdk_child(
        validated["job_path"], app_factory=lambda _value: App(), run_sdk=fail
    )
    assert code == 1
    assert log == ["run", "close"]
    failure = R.BASE._read(validated["output_directory"] / "failure.json")
    first = R.BASE._read(validated["output_directory"] / R.PRIMARY_ERROR_NAME)
    assert failure["completed"] is False
    assert first["error_type"].endswith("FileNotFoundError")
    assert first["error_message"] == "checkpoint dependency missing"
    assert "FileNotFoundError" in first["traceback"]


def test_valid_failure_is_not_accepted_as_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    output = validated["output_directory"]
    output.mkdir()
    first_path = _write(
        output / R.PRIMARY_ERROR_NAME,
        {
            "schema": "vision_checkpoint_smoke_primary_error_v1",
            "error_type": "builtins.RuntimeError",
            "error_message": "boom",
            "traceback": "RuntimeError: boom\n",
        },
    )
    _write(
        output / "failure.json",
        {
            "schema": R.JOB.FAILURE_SCHEMA,
            "completed": False,
            "mode": R.JOB.MODE,
            "job_sha256": validated["job_sha256"],
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "error_type": "RuntimeError",
            "error_message": "boom",
            "first_cause": R._descriptor(first_path),
            "partial_report": None,
            "sdk": None,
        },
    )
    with pytest.raises(R.ChildFailure, match="boom"):
        R.verify_terminal(validated)
