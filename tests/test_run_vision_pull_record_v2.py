from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_pull_record_v2 as RUN


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text('{"immutable":true}', encoding="utf-8")
    return {
        "job_path": job,
        "job_sha256": RUN.JOB._sha(job),
        "output_directory": tmp_path / "output",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_path": tmp_path / "model.pt",
        "job": {
            "mode": RUN.JOB.MODE,
            "output_directory": str(tmp_path / "output"),
            "resource_proof": {"path": str(tmp_path / "proof.json"), "sha256": "b" * 64},
            "checkpoint": {"logical_id": "source-s2220-none"},
            "profile": {},
        },
        "resource": {"resource_id": "resource", "gpu_uuid": "GPU-1"},
    }


class _App:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def test_aaba_child_uses_separate_output_and_keeps_job_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    before = validated["job_path"].read_bytes()
    app = _App()

    monkeypatch.setattr(RUN.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(RUN.AABA, "_diagnostic", lambda _validated, value: value["diagnostic_complete"])

    def diagnostic(adapted: dict[str, Any], _app: object, *, seal_report: Any) -> dict[str, Any]:
        assert adapted["output_directory"] == tmp_path / "aaba-output"
        Path(adapted["output_directory"]).mkdir()
        report = {
            "diagnostic_complete": True,
            "worker": {"policy_probe": {"checkpoint_sha256": "a" * 64, "environment_steps": 1}},
        }
        seal_report(report)
        return report

    assert RUN.aaba_child(
        validated["job_path"], app_factory=lambda _value: app, run_diagnostic=diagnostic
    ) == 0
    assert app.closed == 1
    assert validated["job_path"].read_bytes() == before
    assert not (tmp_path / "output").exists()
    assert (tmp_path / "aaba-output" / RUN.AABA_REPORT_NAME).is_file()


def test_aaba_failure_preserves_first_error_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda _path: validated)

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise FileNotFoundError("raw-output-missing")

    assert RUN.aaba_child(
        validated["job_path"],
        app_factory=lambda _value: _App(RuntimeError("close-secondary")),
        run_diagnostic=fail,
    ) == 1
    failure = RUN._aaba_failure(tmp_path)
    assert failure.first["error_type"] == "builtins.FileNotFoundError"
    assert failure.first["error_message"] == "raw-output-missing"
    assert "FileNotFoundError" in failure.first["traceback"]
    shutdown = json.loads((tmp_path / "aaba-output" / "aaba-shutdown-failure.json").read_text())
    assert shutdown["error_message"] == "close-secondary"
    assert shutdown["first_error_already_sealed"] is True


def test_sdk_child_accepts_stale_resource_only_with_bound_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    output = tmp_path / "aaba-output"
    output.mkdir()
    report_path = output / RUN.AABA_REPORT_NAME
    _write(report_path, {"diagnostic_complete": True})
    observation = {"path": str(tmp_path / "gpu.json"), "sha256": "c" * 64}
    _write(tmp_path / RUN.AABA_CONTINUATION_NAME, RUN._continuation_body(validated, {"diagnostic_complete": True}, observation))
    calls: list[bool] = []

    def validate(_path: Path, *, require_fresh: bool = True) -> dict[str, Any]:
        calls.append(require_fresh)
        return validated

    monkeypatch.setattr(RUN.JOB, "validate_job", validate)
    monkeypatch.setattr(RUN.REMOTE, "verify_observation_reference", lambda value, rid, gpu: None)
    monkeypatch.setattr(
        RUN.WORKER,
        "run_policy_record",
        lambda value, app, seal_report: {
            "technical_recording_complete": True
        },
    )
    monkeypatch.setattr(RUN.JOB, "write_new", lambda _path, _value: None)
    monkeypatch.setattr(RUN.V1, "_result_body", lambda *_args: {})

    assert RUN.sdk_child(validated["job_path"], app_factory=lambda _value: _App()) == 0
    assert calls == [False]


def test_aaba_and_record_outputs_are_distinct(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    adapted = RUN._aaba_validated(validated)
    assert adapted["output_directory"] == tmp_path / "aaba-output"
    assert Path(validated["output_directory"]) == tmp_path / "output"
    assert adapted["job"]["output_directory"] != validated["job"]["output_directory"]


def test_actual_job_v2_requires_parent_created_record_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    python = tmp_path / "python"
    python.write_bytes(b"python")
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    proof = tmp_path / "proof.json"
    proof.write_text("{}", encoding="utf-8")
    job = {
        "schema": RUN.JOB.JOB_SCHEMA,
        "mode": RUN.JOB.MODE,
        "job_id": f"{RUN.JOB.PROTOCOL_ID}-source-s2220-none",
        "source": {"repo_root": str(tmp_path), "commit": "d" * 40, "files_lf": {}},
        "protocol": {"path": str(tmp_path / "protocol.json"), "sha256_lf": "e" * 64},
        "resource_proof": {"path": str(proof), "sha256": "b" * 64},
        "isaaclab": str(isaaclab),
        "python": str(python),
        "output_directory": str(tmp_path / "output"),
        "profile": RUN.JOB.RECORD.fixed_profile(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": "a" * 64,
            "logical_id": "source-s2220-none",
        },
    }
    job_path = tmp_path / "job-v2.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    protocol = {
        "source_commit": "d" * 40,
        "source_sha256_lf": {},
        "source_protocol": {"sha256_lf": "f" * 64},
        "checkpoints": [],
        "_source": {},
    }
    monkeypatch.setattr(RUN.JOB, "validate_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(RUN.JOB, "_sha", lambda path, lf=False: "e" * 64 if lf else "a" * 64)
    monkeypatch.setattr(
        RUN.JOB.RECORD_JOB,
        "_checkpoint",
        lambda *_args: ({"sha256": "a" * 64, "logical_id": "source-s2220-none"}, checkpoint),
    )
    monkeypatch.setattr(RUN.JOB.COMMON, "_python_executable", lambda _path: None)
    monkeypatch.setattr(RUN.JOB.COMMON, "_verify_runtime_sources", lambda *_args: None)
    monkeypatch.setattr(
        RUN.JOB.COMMON,
        "_resource",
        lambda *_args: {"resource_id": "resource", "gpu_uuid": "GPU-1"},
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        RUN.JOB.validate_job(job_path, require_fresh=False)
    (tmp_path / "output").mkdir()
    checked = RUN.JOB.validate_job(job_path, require_fresh=False)
    assert checked["output_directory"] == tmp_path / "output"
    assert validated["output_directory"] == tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        RUN.sdk_child(job_path, app_factory=lambda _value: _App())

    aaba_output = tmp_path / "aaba-output"
    aaba_output.mkdir()
    aaba_report = {"diagnostic_complete": True}
    _write(aaba_output / RUN.AABA_REPORT_NAME, aaba_report)
    observation = {"path": str(tmp_path / "gpu.json"), "sha256": "c" * 64}
    _write(
        tmp_path / RUN.AABA_CONTINUATION_NAME,
        RUN._continuation_body(checked, aaba_report, observation),
    )
    monkeypatch.setattr(RUN.REMOTE, "verify_observation_reference", lambda *_args: None)

    def record_worker(
        value: dict[str, Any], _app: object, *, seal_report: Any
    ) -> dict[str, Any]:
        report = {"technical_recording_complete": True}
        seal_report(report)
        return report

    assert RUN.sdk_child(
        job_path, app_factory=lambda _value: _App(), run_worker=record_worker
    ) == 0
    assert (tmp_path / "output" / "result.json").is_file()


def test_sdk_child_rejects_missing_continuation_before_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    monkeypatch.setattr(
        RUN.JOB, "validate_job", lambda _path, require_fresh=False: validated
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        RUN.sdk_child(validated["job_path"])


def test_timeout_prefers_sealed_aaba_first_cause(tmp_path: Path) -> None:
    _validated(tmp_path)
    output = tmp_path / "aaba-output"
    output.mkdir()
    first = {
        "schema": RUN.PRIMARY_ERROR_SCHEMA,
        "error_type": "builtins.FileNotFoundError",
        "error_message": "first-cause",
        "traceback": "trace",
        "serialization_error": None,
    }
    _write(output / RUN.AABA_PRIMARY_NAME, first)
    _write(
        output / RUN.AABA_FAILURE_NAME,
        {
            "schema": "vision_pull_record_aaba_failure_v2",
            "completed": False,
            "first_cause": RUN._descriptor(output / RUN.AABA_PRIMARY_NAME),
        },
    )
    failure = RUN._aaba_failure_if_present(tmp_path)
    assert failure is not None
    assert failure.first["error_message"] == "first-cause"


def test_actual_subprocess_failure_atom_is_seen_before_long_timeout(
    tmp_path: Path,
) -> None:
    _validated(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def write_failure() -> None:
        time.sleep(0.15)
        output = tmp_path / "aaba-output"
        output.mkdir()
        first = {
            "schema": RUN.PRIMARY_ERROR_SCHEMA,
            "error_type": "builtins.FileNotFoundError",
            "error_message": "first-cause",
            "traceback": "trace",
            "serialization_error": None,
        }
        _write(output / RUN.AABA_PRIMARY_NAME, first)
        _write(
            output / RUN.AABA_FAILURE_NAME,
            {
                "schema": "vision_pull_record_aaba_failure_v2",
                "completed": False,
                "first_cause": RUN._descriptor(output / RUN.AABA_PRIMARY_NAME),
            },
        )

    writer = threading.Thread(target=write_failure)
    writer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RUN.ChildFailure, match="first-cause"):
            RUN._wait_for_child(
                process,
                timeout_s=5.0,
                phase_name="aaba",
                stage=tmp_path,
                poll_s=0.05,
            )
    finally:
        process.terminate()
        process.wait(timeout=5)
        writer.join(timeout=5)
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("outcome", ["success", "aaba_failure", "record_failure"])
def test_parent_receipt_keeps_terminal_sha_and_phase_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    validated = _validated(tmp_path)
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(RUN.AABA, "_diagnostic", lambda *_args: True)
    monkeypatch.setattr(RUN.BASE, "_gpu_reference", lambda *_args: {"path": "gpu", "sha256": "c" * 64})
    monkeypatch.setattr(RUN.V1, "verify_terminal", lambda *_args: {"completed": True, "technical_recording_complete": True})
    launched: list[str] = []
    cleaned: list[str] = []

    class Process:
        pid = 123

        def __init__(self, phase: str) -> None:
            self.phase = phase
            self.stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            del timeout
            output = tmp_path / ("aaba-output" if self.phase == "aaba" else "output")
            output.mkdir(exist_ok=True)
            if outcome == f"{self.phase}_failure":
                is_aaba = self.phase == "aaba"
                first_path = output / (RUN.AABA_PRIMARY_NAME if is_aaba else RUN.V1.PRIMARY_ERROR_NAME)
                _write(first_path, {
                    "schema": RUN.PRIMARY_ERROR_SCHEMA if is_aaba else RUN.V1.PRIMARY_ERROR_SCHEMA,
                    "error_type": "builtins.FileNotFoundError", "error_message": "first-cause", "traceback": "trace",
                })
                _write(output / (RUN.AABA_FAILURE_NAME if is_aaba else "failure.json"), {
                    "schema": "vision_pull_record_aaba_failure_v2" if is_aaba else RUN.JOB.FAILURE_SCHEMA,
                    "completed": False, "technical_recording_complete": False,
                    "job_sha256": validated["job_sha256"], "checkpoint_sha256": validated["checkpoint_sha256"],
                    "first_cause": RUN._descriptor(first_path),
                })
                raise subprocess.TimeoutExpired("child-close", 0.25)
            if self.phase == "aaba":
                _write(output / RUN.AABA_REPORT_NAME, {
                    "diagnostic_complete": True,
                    "worker": {"policy_probe": {"checkpoint_sha256": validated["checkpoint_sha256"], "environment_steps": 1}},
                })
            else:
                _write(output / "result.json", {"completed": True, "technical_recording_complete": True})
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        phase = "aaba" if "--aaba-child" in command else "record"
        launched.append(phase)
        return Process(phase)

    def cleanup(_launch: Any, process: Process) -> bool:
        cleaned.append(process.phase)
        return True

    receipt = RUN.run_job(
        validated["job_path"], popen=popen, gpu_reader=list,
        identify=lambda _pid: {"pid": 123, "start_ticks": 1}, cleanup=cleanup,
    )
    expected = ["aaba"] if outcome == "aaba_failure" else ["aaba", "record"]
    assert launched == cleaned == expected
    assert [row["name"] for row in receipt["phases"]] == expected
    assert all(row["cleanup_verified"] is True and row["gpu_after"] == [] for row in receipt["phases"])
    assert receipt["cleanup_verified"] is True
    if outcome == "success":
        assert receipt["result_sha256"] == RUN.JOB._sha(tmp_path / "output" / "result.json")
        assert receipt["completed"] is True and receipt["full_recordings"] == 1
    else:
        assert receipt["completed"] is False and receipt["full_recordings"] == 0
        assert receipt["error_type"] == "builtins.FileNotFoundError"
        assert receipt["failure_atom"]["first_cause_message"] == "first-cause"
        assert receipt["timed_out"] is False
