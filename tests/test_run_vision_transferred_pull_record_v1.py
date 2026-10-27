from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))

import run_vision_transferred_pull_record_v1 as RUN  # pyright: ignore[reportMissingImports]


def test_job_contract_swaps_and_restores_run5_dependency() -> None:
    previous = RUN.PREVIOUS.JOB
    reserve = RUN.PREVIOUS.POSTFLIGHT_RESERVE_S
    with RUN._job_contract():
        assert RUN.PREVIOUS.JOB is RUN.JOB
        assert RUN.PREVIOUS.POSTFLIGHT_RESERVE_S == 200
    assert RUN.PREVIOUS.JOB is previous
    assert RUN.PREVIOUS.POSTFLIGHT_RESERVE_S == reserve


def test_child_launcher_rewrites_run5_entrypoint_for_fresh_aaba_and_record() -> None:
    calls: list[list[str]] = []

    def popen(command: list[str], **_kwargs: Any) -> object:
        calls.append(command)
        return object()

    launch = RUN._child_launcher(popen)
    assert RUN.PREVIOUS.__file__ is not None
    assert RUN.__file__ is not None
    old = str(Path(RUN.PREVIOUS.__file__).resolve())
    current = str(Path(RUN.__file__).resolve())
    launch(["python", old, "--aaba-child", "--job", "job.json"])
    launch(["python", old, "--sdk-child", "--job", "job.json"])
    assert calls == [
        ["python", current, "--aaba-child", "--job", "job.json"],
        ["python", current, "--sdk-child", "--job", "job.json"],
    ]


def test_child_launcher_rejects_command_without_run5_entrypoint() -> None:
    launch = RUN._child_launcher(lambda *_args, **_kwargs: object())
    with pytest.raises(ValueError, match="RUN5"):
        launch(["python", "other.py", "--job", "job.json"])


def test_run_job_delegates_all_safety_callbacks_under_new_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}

    def previous(job_path: Path, **kwargs: Any) -> dict[str, Any]:
        observed.update(job_path=job_path, job=RUN.PREVIOUS.JOB,
                        popen=kwargs["popen"], gpu_reader=kwargs["gpu_reader"],
                        identify=kwargs["identify"], cleanup=kwargs["cleanup"])
        return {"completed": True}

    monkeypatch.setattr(RUN.PREVIOUS, "run_job", previous)
    popen = lambda *_args, **_kwargs: object()
    gpu = list
    identify = lambda _pid: {"pid": 1}
    cleanup = lambda *_args: True
    result = RUN.run_job(tmp_path / "job.json", popen=popen, gpu_reader=gpu,
                         identify=identify, cleanup=cleanup)
    assert result == {"completed": True}
    assert observed["job"] is RUN.JOB
    assert observed["gpu_reader"] is gpu
    assert observed["identify"] is identify
    assert observed["cleanup"] is cleanup
