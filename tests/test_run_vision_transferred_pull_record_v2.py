from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import run_vision_transferred_pull_record_v2 as RUN  # pyright: ignore[reportMissingImports]


def test_job_contract_swaps_worker_in_run5_and_terminal_verifier_then_restores() -> None:
    previous = (
        RUN.PREVIOUS.JOB,
        RUN.PREVIOUS.WORKER,
        RUN.PREVIOUS.V1.WORKER,
        RUN.PREVIOUS.REPORT_NAME,
        RUN.PREVIOUS.V1.REPORT_NAME,
    )
    with RUN._job_contract():
        assert RUN.PREVIOUS.JOB is RUN.JOB
        assert RUN.PREVIOUS.WORKER is RUN.WORKER
        assert RUN.PREVIOUS.V1.WORKER is RUN.WORKER
        assert RUN.PREVIOUS.REPORT_NAME == RUN.WORKER.REPORT_NAME
        assert RUN.PREVIOUS.V1.REPORT_NAME == RUN.WORKER.REPORT_NAME
    assert (
        RUN.PREVIOUS.JOB,
        RUN.PREVIOUS.WORKER,
        RUN.PREVIOUS.V1.WORKER,
        RUN.PREVIOUS.REPORT_NAME,
        RUN.PREVIOUS.V1.REPORT_NAME,
    ) == previous


def test_actual_run_job_child_commands_use_v2_entry_and_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    old = str(Path(RUN.PREVIOUS.__file__).resolve())
    current = str(Path(RUN.__file__).resolve())

    def outer_popen(command: list[str], **_kwargs: Any) -> object:
        commands.append(command)
        return object()

    def previous(job_path: Path, **kwargs: Any) -> dict[str, Any]:
        assert RUN.PREVIOUS.WORKER is RUN.WORKER
        assert RUN.PREVIOUS.V1.WORKER is RUN.WORKER
        kwargs["popen"](["python", old, "--aaba-child", "--job", str(job_path)])
        kwargs["popen"](["python", old, "--sdk-child", "--job", str(job_path)])
        return {"completed": True}

    monkeypatch.setattr(RUN.PREVIOUS, "run_job", previous)
    assert RUN.run_job(tmp_path / "job.json", popen=outer_popen) == {"completed": True}
    assert commands == [
        ["python", current, "--aaba-child", "--job", str(tmp_path / "job.json")],
        ["python", current, "--sdk-child", "--job", str(tmp_path / "job.json")],
    ]


def test_child_launcher_rejects_command_without_run5_entrypoint() -> None:
    launch = RUN._child_launcher(lambda *_args, **_kwargs: object())
    with pytest.raises(ValueError, match="RUN5"):
        launch(["python", "other.py", "--job", "job.json"])
