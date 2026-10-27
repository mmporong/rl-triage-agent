from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import run_vision_transferred_pull_record_v3 as RUN  # pyright: ignore[reportMissingImports]


def test_job_contract_installs_v3_worker_in_record_and_terminal_paths() -> None:
    previous = (
        RUN.PREVIOUS.JOB,
        RUN.PREVIOUS.WORKER,
        RUN.PREVIOUS.V1.WORKER,
    )
    with RUN._job_contract():
        assert RUN.PREVIOUS.JOB is RUN.JOB
        assert RUN.PREVIOUS.WORKER is RUN.WORKER
        assert RUN.PREVIOUS.V1.WORKER is RUN.WORKER
    assert (
        RUN.PREVIOUS.JOB,
        RUN.PREVIOUS.WORKER,
        RUN.PREVIOUS.V1.WORKER,
    ) == previous


def test_actual_child_commands_use_v3_entry_and_worker(
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
