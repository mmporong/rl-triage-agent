from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_prepared_record_v2 as W  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = _write(tmp_path / "job.json", {"job": True})
    return {
        "job_path": job,
        "job_sha256": W.V1._sha(job),
        "output_directory": tmp_path / "output",
        "checkpoint_sha256": "c" * 64,
        "job": {
            "job_id": "prepared-record-v2-fixture",
            "mode": W.JOB.MODE,
            "python": sys.executable,
            "isaaclab": str(tmp_path),
        },
        "resource": {"resource_id": "owned", "gpu_uuid": "GPU-owned"},
    }


def _failed_sdk_report(*, sealed: bool) -> dict[str, Any]:
    return {
        "schema": "vision_prepared_record_sdk_v1",
        "prepared_record": None,
        "artifacts": {},
        "lifecycle": {
            "run_real_started": True,
            "run_real_returned": False,
            "writer_attach_attempted": False,
            "writer_attached": False,
            "arms_completed": 0,
            "environment_closed": False,
            "envelope_sealed": sealed,
            "errors": [
                {"phase": "run_real", "error_type": "builtins.FileNotFoundError"}
            ],
        },
        "renderer": {},
        "setup_evidence": {
            "phase_order": ["replicator_startup", "run_real"],
            "raw_callback_count": 0,
        },
        "record_result": None,
        "approved": False,
    }


def _write_sdk_sidecars(value: dict[str, Any]) -> tuple[Path, Path]:
    stage = Path(value["job_path"]).parent
    preflight = _write(
        stage / W.RUNTIME_PREFLIGHT_NAME,
        {
            "schema": "vision_prepared_record_runtime_preflight_v2",
            "status": "complete",
            "tools": [
                {"name": "ffmpeg", "status": "ready"},
                {"name": "ffprobe", "status": "ready"},
            ],
        },
    )
    primary = _write(
        stage / W.PRIMARY_ERROR_NAME,
        {
            "schema": W.PRIMARY_ERROR_SCHEMA,
            "phase": "run_real",
            "exception": {
                "type": "builtins.FileNotFoundError",
                "message": "policy checkpoint leaf is missing",
                "traceback_frames": [
                    {
                        "file": "fixture.py",
                        "line": 10,
                        "function": "run_real",
                        "source": "raise FileNotFoundError(...)",
                    }
                ],
                "cause": None,
                "context": None,
            },
            "preflight": {"path": preflight.name, "sha256": W.V1._sha(preflight)},
            "output_absent": False,
        },
    )
    return preflight, primary


class _SDKFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: Exception) -> None:
        super().__init__("prepared record SDK failed: builtins.FileNotFoundError")
        self.report = report
        self.envelope = report
        self.primary = primary


def test_sdk_child_preserves_run_real_first_cause_and_trace_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    value["output_directory"].mkdir()
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    observed: list[str] = []

    class App:
        def close(self) -> None:
            observed.append("close")
            assert (value["output_directory"] / "failure.json").is_file()
            assert (Path(value["job_path"]).parent / W.PRIMARY_ERROR_NAME).is_file()

    def run_sdk(_value: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        observed.append("run")
        _write_sdk_sidecars(value)
        sealed = _failed_sdk_report(sealed=False)
        seal_report(sealed)
        try:
            raise FileNotFoundError("policy checkpoint leaf is missing")
        except FileNotFoundError as primary:
            raise _SDKFailure(_failed_sdk_report(sealed=True), primary) from primary

    assert (
        W.sdk_child(value["job_path"], app_factory=lambda _: App(), run_sdk=run_sdk)
        == 1
    )
    assert observed == ["run", "close"]
    failure = W.V1._read(value["output_directory"] / "failure.json")
    primary_path = Path(value["job_path"]).parent / W.PRIMARY_ERROR_NAME
    primary = W.V1._read(primary_path)
    assert failure["first_cause"] == W.V1._descriptor(primary_path)
    assert primary["exception"]["type"] == "builtins.FileNotFoundError"
    assert primary["exception"]["message"] == "policy checkpoint leaf is missing"
    assert primary["exception"]["traceback_frames"][0]["function"] == "run_real"


def _write_valid_failure(
    value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = value["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    preflight_path, primary_path = _write_sdk_sidecars(value)
    report = _write(output / W.REPORT_NAME, _failed_sdk_report(sealed=False))
    failure = {
        "schema": W.JOB.FAILURE_SCHEMA,
        "completed": False,
        "mode": W.JOB.MODE,
        "job_sha256": value["job_sha256"],
        "checkpoint_sha256": value["checkpoint_sha256"],
        "error_type": "PreparedRecordSDKFailure",
        "error_message": "prepared record SDK failed: builtins.FileNotFoundError",
        "first_cause": W.V1._descriptor(primary_path),
        "runtime_preflight": W.V1._descriptor(preflight_path),
        "partial_report": W.V1._descriptor(report),
        "sdk": _failed_sdk_report(sealed=True),
    }
    _write(output / "failure.json", failure)
    return failure, {
        "error_type": "builtins.FileNotFoundError",
        "error_message": "policy checkpoint leaf is missing",
        "traceback_frames": W.V1._read(primary_path)["exception"]["traceback_frames"],
        "phase": "run_real",
    }


def test_failure_and_partial_report_are_one_valid_failure_terminal(
    tmp_path: Path,
) -> None:
    value = _validated(tmp_path)
    failure, primary = _write_valid_failure(value)
    with pytest.raises(W.ChildTerminalFailure) as caught:
        W.verify_terminal(value)
    assert caught.value.failure == failure
    assert caught.value.first_cause == primary
    assert "성공과 실패" not in str(caught.value)


def test_child_exit_zero_with_failure_is_rejected_and_first_cause_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    _write_valid_failure(value)
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(W.V1.BASE, "_gpu_reference", lambda *_args: {"path": "gpu"})

    class Reader:
        def __init__(self) -> None:
            self.last_receipt = {"path": "gpu"}

        def __call__(self) -> list[int]:
            return []

    class Gate(io.BytesIO):
        def close(self) -> None:
            self.accepted = self.getvalue()
            super().close()

    class Process:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdin = Gate()

        def wait(self, *, timeout: float) -> int:
            assert timeout <= 400 and self.stdin.accepted == b"GO\n"
            self.returncode = 0
            return 0

    receipt = W.run_job(
        value["job_path"],
        popen=lambda *_args, **_kwargs: Process(),
        gpu_reader=Reader(),
        identify=lambda pid: {"pid": pid, "start_ticks": 1},
        cleanup=lambda *_args: True,
    )
    assert receipt["completed"] is False
    assert receipt["exit_code"] == 0
    assert receipt["failure_atom"]["status"] == "failed"
    assert receipt["failure_atom"]["child_exit_accepted"] is False
    assert receipt["failure_atom"]["first_cause_type"] == "builtins.FileNotFoundError"
    assert receipt["error_message"] == "policy checkpoint leaf is missing"


def test_success_terminal_still_uses_frozen_625_frame_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    value["output_directory"].mkdir()
    _write(value["output_directory"] / "result.json", {"completed": True})
    expected = {"completed": True, "frame_count": 625, "fps": 50}
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        W.V1,
        "verify_result",
        lambda validated: calls.append(validated) or copy.deepcopy(expected),
    )
    assert W.verify_terminal(value) == expected
    assert calls == [value]
