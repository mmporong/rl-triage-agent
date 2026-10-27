from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import run_vision_pull_record_v6 as R  # pyright: ignore[reportMissingImports]


@pytest.mark.parametrize("entry", ["aaba_child", "sdk_child"])
def test_both_child_entries_use_v6_job_authorization_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []

    def validate(path: Path, **_kwargs: object) -> dict[str, object]:
        calls.append(path)
        return {}

    def child(path: Path, **_kwargs: object) -> int:
        assert R.PREVIOUS.JOB is R.JOB
        R.PREVIOUS.JOB.validate_job(path)
        return 0

    monkeypatch.setattr(R.JOB, "validate_job", validate)
    monkeypatch.setattr(R.PREVIOUS, entry, child)
    assert getattr(R, entry)(job) == 0
    assert calls == [job]
    assert R.PREVIOUS.JOB is not R.JOB


def test_parent_records_and_launches_fixed_v6_child_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    launched: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> object:
        launched.append(command)
        return object()

    def previous_run(path: Path, *, popen, **_kwargs: object) -> dict[str, object]:
        assert path == job
        assert R.PREVIOUS.JOB is R.JOB
        runner = str(Path(R.PREVIOUS.__file__).resolve())
        popen(["python", "-B", runner, "--aaba-child", "--job", str(path)])
        popen(["python", "-B", runner, "--sdk-child", "--job", str(path)])
        return {"completed": True}

    monkeypatch.setattr(R.PREVIOUS, "run_job", previous_run)
    result = R.run_job(job, popen=fake_popen)
    assert result == {"completed": True}
    expected = str((ROOT / "bench/run_vision_pull_record_v6.py").resolve())
    assert len(launched) == 2
    assert all(command[2] == expected for command in launched)
    assert launched[0][3] == "--aaba-child"
    assert launched[1][3] == "--sdk-child"
    assert R.PREVIOUS.JOB is not R.JOB
    assert Path(R.PREVIOUS.__file__).name == "run_vision_pull_record_v5.py"


def test_parent_restores_previous_contract_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = R.PREVIOUS.JOB

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert R.PREVIOUS.JOB is R.JOB
        raise RuntimeError("fixture")

    monkeypatch.setattr(R.PREVIOUS, "run_job", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        R.run_job(tmp_path / "job.json", popen=lambda *_args, **_kwargs: None)
    assert R.PREVIOUS.JOB is previous
    assert Path(R.PREVIOUS.__file__).name == "run_vision_pull_record_v5.py"


def test_main_passes_authorization_and_issuance_into_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / name for name in (
        "protocol.json", "authorization.json", "issuance.json", "resource.json",
        "isaaclab", "python", "stage",
    )]
    created = tmp_path / "created/job.json"
    captured: list[tuple[object, ...]] = []

    def create(*args: object) -> Path:
        captured.append(args)
        return created

    monkeypatch.setattr(R.JOB, "create_job", create)
    monkeypatch.setattr(R, "run_job", lambda path: {"completed": path == created})
    args = [
        "--protocol", str(paths[0]),
        "--resource-authorization", str(paths[1]),
        "--resource-issuance", str(paths[2]),
        "--resource-proof", str(paths[3]),
        "--resource-sha256", "a" * 64,
        "--isaaclab", str(paths[4]),
        "--python", str(paths[5]),
        "--stage", str(paths[6]),
        "--checkpoint-logical-id", R.JOB.CHECKPOINT_IDS[0],
    ]
    assert R.main(args) == 0
    assert captured == [(
        paths[0], paths[1], paths[2], paths[3], "a" * 64,
        paths[4], paths[5], paths[6], R.JOB.CHECKPOINT_IDS[0],
    )]


def test_windows_default_subprocess_is_not_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R.os, "name", "nt")
    with pytest.raises(RuntimeError, match="Linux"):
        R.run_job(tmp_path / "job.json", popen=subprocess.Popen)
