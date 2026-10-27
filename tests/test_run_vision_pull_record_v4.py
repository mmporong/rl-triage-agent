from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", Path(__file__).parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_pull_record_v4 as RUN  # pyright: ignore[reportMissingImports]
import test_run_vision_pull_record_v2 as V2_FIXTURE  # pyright: ignore[reportMissingImports]
import test_vision_annotator_pull_checkpoint_v2 as AABA_FIXTURE  # pyright: ignore[reportMissingImports]


class Settings:
    def __init__(self) -> None:
        self.value = False

    def get_as_bool(self, _name: str) -> bool:
        return self.value

    def set_bool(self, _name: str, value: bool) -> None:
        self.value = value


class RawApp:
    def __init__(self) -> None:
        self.close_calls: list[dict[str, Any]] = []

    def update(self) -> None:
        return None

    def close(self, **kwargs: Any) -> None:
        self.close_calls.append(dict(kwargs))


def _emit_aaba(adapted: dict[str, Any], app: object, *, seal_report: Any) -> dict[str, Any]:
    worker = AABA_FIXTURE._worker()
    worker["policy_probe"]["checkpoint_sha256"] = adapted["checkpoint_sha256"]
    worker["lifecycle"].update(
        {"render_product_destroyed": True, "environment_closed": True}
    )
    return RUN.AABA_CHECKPOINT.run_annotator_pull_diagnostic(
        adapted,
        app,
        run_worker=lambda *_args, **_kwargs: worker,
        seal_report=seal_report,
    )


def _validated(tmp_path: Path) -> dict[str, Any]:
    value = V2_FIXTURE._validated(tmp_path)
    value["job"]["profile"] = RECORD.fixed_profile()
    return value


def test_aaba_child_uses_phase_shutdown_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    raw = RawApp()
    settings = Settings()
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(
        RUN.AABA_SHUTDOWN,
        "sdk_app_factory",
        lambda adapted: RUN.AABA_SHUTDOWN.AppProxy(raw, adapted, settings),
    )

    assert RUN.aaba_child(validated["job_path"], run_diagnostic=_emit_aaba) == 0

    assert raw.close_calls == [{"wait_for_replicator": False}]
    assert settings.value is True
    assert RUN._verify_aaba_terminal(validated)["diagnostic_complete"] is True
    assert (tmp_path / RUN.AABA_SHUTDOWN.POLICY_NAME).is_file()
    assert not (tmp_path / "shutdown-policy.json").exists()


def test_parent_does_not_launch_record_without_aaba_shutdown_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda *_args, **_kwargs: validated)
    launched: list[str] = []

    class Process:
        pid = 123
        stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            adapted = RUN._aaba_validated(validated)
            _emit_aaba(adapted, object(), seal_report=lambda value: RUN.JOB.write_new(
                Path(adapted["output_directory"]) / RUN.AABA_REPORT_NAME, value
            ))
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        phase = "aaba" if "--aaba-child" in command else "record"
        launched.append(phase)
        return Process()

    receipt = RUN.run_job(
        validated["job_path"],
        popen=popen,
        gpu_reader=list,
        identify=lambda _pid: {"pid": 123, "start_ticks": 1},
        cleanup=lambda *_args: True,
    )

    assert launched == ["aaba"]
    assert receipt["completed"] is False
    assert receipt["error_type"] in {"FileNotFoundError", "ValueError"}
    assert not (tmp_path / "record-launch.json").exists()


def test_parent_requires_exit_cleanup_gpu_and_policy_before_record(
    tmp_path: Path, monkeypatch
) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(
        RUN.BASE,
        "_gpu_reference",
        lambda *_args: {"path": "gpu", "sha256": "c" * 64},
    )
    monkeypatch.setattr(RUN.REMOTE, "verify_observation_reference", lambda *_args: None)
    monkeypatch.setattr(
        RUN.V1,
        "verify_terminal",
        lambda *_args: {"completed": True, "technical_recording_complete": True},
    )
    launched: list[str] = []
    cleaned: list[str] = []

    class Process:
        pid = 123

        def __init__(self, phase: str) -> None:
            self.phase = phase
            self.stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            if self.phase == "aaba":
                adapted = RUN._aaba_validated(validated)
                raw = RawApp()
                proxy = RUN.AABA_SHUTDOWN.AppProxy(raw, adapted, Settings())
                _emit_aaba(
                    adapted,
                    proxy,
                    seal_report=lambda value: RUN.JOB.write_new(
                        Path(adapted["output_directory"]) / RUN.AABA_REPORT_NAME, value
                    ),
                )
                proxy.close()
            else:
                RUN.JOB.write_new(
                    Path(validated["output_directory"]) / "result.json",
                    {"completed": True, "technical_recording_complete": True},
                )
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        phase = "aaba" if "--aaba-child" in command else "record"
        launched.append(phase)
        return Process(phase)

    def cleanup(_launch: Any, process: Process) -> bool:
        cleaned.append(process.phase)
        return True

    receipt = RUN.run_job(
        validated["job_path"],
        popen=popen,
        gpu_reader=list,
        identify=lambda _pid: {"pid": 123, "start_ticks": 1},
        cleanup=cleanup,
    )

    assert launched == cleaned == ["aaba", "record"]
    assert receipt["completed"] is True
    assert receipt["aaba_receipt"]["shutdown_policy"]["file"] == RUN.AABA_SHUTDOWN.POLICY_NAME
    assert receipt["aaba_receipt"]["close_returned"]["file"] == RUN.AABA_SHUTDOWN.CLOSE_RETURNED_NAME
    assert receipt["phases"][0]["exit_code"] == 0
    assert receipt["phases"][0]["cleanup_verified"] is True
    assert receipt["phases"][0]["gpu_after"] == []


def test_runner_imports_job4_without_fallback() -> None:
    assert RUN.__file__ is not None
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert "import vision_pull_record_job_v4 as JOB" in source
    assert "ImportError" not in source
    assert "vision_pull_record_job_v2 as JOB" not in source
    assert "vision_pull_record_job_v1 as JOB" not in source
    assert RUN.EXECUTION_SCHEMA == "vision_pull_record_execution_v3"
    assert RUN.JOB.LIMITS["max_wall_s_per_policy"] == 300
    assert RUN.POSTFLIGHT_RESERVE_S == 200.0
    assert (
        RUN.JOB.LIMITS["max_wall_s_per_policy"] - RUN.POSTFLIGHT_RESERVE_S
        == RUN.JOB.LIMITS["max_child_s_per_policy"]
    )


def test_policy_failure_poll_stops_hung_close_and_worker_error_keeps_priority(
    tmp_path: Path,
) -> None:
    validated = _validated(tmp_path)
    adapted = RUN._aaba_validated(validated)
    failure = {
        "schema": "vision_aaba_close_policy_failure_v1",
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "error_type": "ValueError",
        "error_message": "terminal-mismatch",
    }
    RUN.JOB.write_new(tmp_path / RUN.AABA_SHUTDOWN.POLICY_FAILURE_NAME, failure)

    class Hung:
        def __init__(self) -> None:
            self.args = ["aaba"]

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired(self.args, timeout)

    with pytest.raises(RUN.ShutdownPolicyFailure, match="terminal-mismatch"):
        RUN._wait_for_child(
            Hung(),
            timeout_s=2,
            phase_name="aaba",
            stage=tmp_path,
            poll_s=0.01,
            validated=validated,
        )

    output = Path(adapted["output_directory"])
    output.mkdir()
    first = output / RUN.AABA_PRIMARY_NAME
    RUN.JOB.write_new(
        first,
        {
            "schema": RUN.PRIMARY_ERROR_SCHEMA,
            "error_type": "builtins.FileNotFoundError",
            "error_message": "worker-first",
            "traceback": "trace",
        },
    )
    RUN.JOB.write_new(
        output / RUN.AABA_FAILURE_NAME,
        {
            "schema": "vision_pull_record_aaba_failure_v2",
            "completed": False,
            "first_cause": RUN._descriptor(first),
        },
    )
    with pytest.raises(RUN.ChildFailure, match="worker-first"):
        RUN._wait_for_child(
            Hung(),
            timeout_s=2,
            phase_name="aaba",
            stage=tmp_path,
            poll_s=0.01,
            validated=validated,
        )
