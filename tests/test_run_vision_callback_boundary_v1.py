from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_callback_boundary_v1 as R  # pyright: ignore[reportMissingImports]


def write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def validated(tmp_path: Path, variant: str = "custom_event") -> dict[str, Any]:
    job = write(tmp_path / "job.json", {"mode": R.JOB.MODE, "callback_variant": variant})
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"model")
    return {"job": {"mode": R.JOB.MODE, "callback_variant": variant}, "job_path": job, "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(), "output_directory": tmp_path / "output", "checkpoint_path": checkpoint, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def envelope(variant: str) -> dict[str, Any]:
    return {
        "schema": R.SDK.SCHEMA, "variant": variant, "underlying": {"schema": "inner", "approved": False},
        "callbacks": [{"ordinal": 0}],
        "operations": {"timeline": [{"name": "orchestrator_step"}], "counts": {}},
        "capture_on_play": {"before": False, "after_startup": False, "restored": False},
        "primary_error": {"error_type": "builtins.RuntimeError", "error_message": "strict cardinality"},
        "diagnostic_complete": True, "used_for_decision": False, "approved": False,
    }


def test_diagnostic_requires_non_approving_complete_terminal(tmp_path: Path) -> None:
    value = envelope("custom_event")
    assert R._diagnostic(validated(tmp_path), value)
    value["approved"] = True
    assert not R._diagnostic(validated(tmp_path), value)


def test_sdk_child_preserves_expected_strict_failure_as_valid_diagnostic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = validated(tmp_path)
    calls: list[str] = []

    class App:
        def close(self) -> None: calls.append("close")

    def run(_validated: Any, _app: Any, *, variant: str, seal_report: Any) -> dict[str, Any]:
        calls.append(variant)
        value = envelope(variant)
        seal_report(value)
        return value

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=run) == 0
    assert calls == ["custom_event", "close"]
    result = R.verify_result(item)
    assert result["diagnostic_complete"] is True
    assert result["capture_approved"] is False
    assert result["sdk"]["primary_error"]["error_message"] == "strict cardinality"


def test_sdk_child_preserves_instrumentation_failure_before_app_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = validated(tmp_path, "manual_schedule")
    calls: list[str] = []

    class App:
        def close(self) -> None: calls.append("close")

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("run")
        raise ValueError("missing boundary evidence")

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=fail) == 1
    assert calls == ["run", "close"]
    failure = R.PARENT.BASE._read(item["output_directory"] / "failure.json")
    first = R.PARENT.BASE._read(item["output_directory"] / R.PRIMARY_ERROR_NAME)
    assert failure["completed"] is False
    assert first["error_message"] == "missing boundary evidence"
    with pytest.raises(R.ChildFailure, match="missing boundary evidence"):
        R.verify_terminal(item)
