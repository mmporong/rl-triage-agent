from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "tests")]

import run_vision_corrected_pull_record_v2 as R
import test_run_vision_pull_record_v5 as BASE


def test_run5_job_helpers_are_explicit_and_forwarded() -> None:
    assert R._sha is R.V1._sha
    assert R._regular is R.JOB3._regular
    assert R.write_new is R.JOB3.write_new
    assert callable(R.validate_job) and callable(R.create_job)
    missing = "helper_that_does_not_exist"
    with pytest.raises(AttributeError):
        getattr(R, missing)


def test_contract_installs_v2_job_and_rewrites_both_child_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def previous(job_path: Path, **kwargs: Any) -> dict[str, bool]:
        assert R.RUN3.JOB is R
        old = str(Path(R.RUN3.PREVIOUS.__file__).resolve())
        launch = R.RUN3._child_launcher(kwargs["popen"])
        launch([sys.executable, old, "--aaba-child", "--job", str(job_path)])
        launch([sys.executable, old, "--sdk-child", "--job", str(job_path)])
        return {"completed": True}

    monkeypatch.setattr(R.RUN3, "run_job", previous)
    result = R.run_job(tmp_path / "job.json", popen=lambda command, **_kwargs: observed.append(command))
    assert result == {"completed": True}
    assert len(observed) == 2
    assert all(command[1] == str(Path(R.__file__).resolve()) for command in observed)
    assert R.RUN3.JOB is not R


def test_actual_run5_parent_consumes_helpers_and_uses_v2_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run5 = R.RUN3.PREVIOUS
    validated = BASE._validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(R, "validate_job", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(run5.BASE, "_gpu_reference", lambda *_args: {"path": "gpu", "sha256": "c" * 64})
    monkeypatch.setattr(run5.REMOTE, "verify_observation_reference", lambda *_args: None)
    monkeypatch.setattr(run5.V1, "verify_terminal",
                        lambda *_args: {"completed": True, "technical_recording_complete": True})
    commands: list[list[str]] = []

    class Process:
        pid = 123

        def __init__(self, phase: str) -> None:
            self.phase = phase
            self.stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            if self.phase == "aaba":
                adapted = run5._aaba_validated(validated)
                raw = BASE.RawApp()
                proxy = run5.AABA_SHUTDOWN.AppProxy(raw, adapted, BASE.Settings())
                BASE._emit_aaba(adapted, proxy, seal_report=lambda value: R.write_new(
                    Path(adapted["output_directory"]) / run5.AABA_REPORT_NAME, value
                ))
                proxy.close()
            else:
                output = Path(validated["output_directory"])
                output.mkdir(parents=True, exist_ok=False)
                R.write_new(output / "result.json",
                            {"completed": True, "technical_recording_complete": True})
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        commands.append(command)
        return Process("aaba" if "--aaba-child" in command else "record")

    receipt = R.run_job(validated["job_path"], popen=popen, gpu_reader=list,
                        identify=lambda _pid: {"pid": 123, "start_ticks": 1},
                        cleanup=lambda *_args: True)
    assert receipt["completed"] is True
    assert len(commands) == 2
    assert all(str(Path(R.__file__).resolve()) in command for command in commands)
    assert (tmp_path / "aaba-launch.json").is_file()
    assert (tmp_path / "record-launch.json").is_file()
