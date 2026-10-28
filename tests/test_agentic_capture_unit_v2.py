"""capture unit v2: GPU 입장 전 lease root 관측과 scope 안의 실제 UNIT1 lease 경로를 CPU에서 검증한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v1 as UNIT1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v2 as UNIT2  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v2 as T2  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

JOB1 = JOB2.JOB1


class Stop(Exception):
    """GPU 입장(admission) 자리에서 흐름을 멈추는 fixture 신호."""


def _globals() -> tuple[Any, Any, Any]:
    return JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher


def _args(stage: Path, protocol: Path | None = None, logical_id: str = FIX.NONE_ID) -> argparse.Namespace:
    return argparse.Namespace(mode="state-smoke", protocol=protocol or Path("unused.json"),
                              provider_inventory=Path("provider.json"),
                              provider_observed_at="2026-10-28T00:00:00+00:00", isaaclab=Path("isaaclab"),
                              python=Path(sys.executable), stage=stage, logical_id=logical_id)


def test_windows_entry_is_rejected_before_scope_or_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(UNIT1, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(UNIT1, "execute", lambda args: pytest.fail("UNIT1이 실행되면 안 된다"))
    stage = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="Linux"):
        UNIT2.execute(_args(stage))
    assert not UNIT2.observation_path(stage).exists() and _globals() == T2.ORIGINAL


def test_observation_is_written_before_unit1_inside_scope_then_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = T2.linked_home(tmp_path, monkeypatch)
    monkeypatch.setattr(UNIT1, "os", SimpleNamespace(name="posix"))
    stage = tmp_path / "runs" / "stage"
    seen: list[Any] = []

    def spy(args: argparse.Namespace) -> dict[str, Any]:
        seen.append((UNIT2.observation_path(args.stage).is_file(), JOB2.scope_active_root(), JOB1.LEASE_ROOT,
                     JOB1.REQUIRED_SOURCE_FILES is JOB2.REQUIRED_SOURCE_FILES, RUN1._child_launcher is T2.ORIGINAL[2]))
        return {"completed": True}

    monkeypatch.setattr(UNIT1, "execute", spy)
    assert UNIT2.execute(_args(stage)) == {"completed": True}
    assert seen == [(True, layout.canonical, layout.canonical, True, False)]
    assert _globals() == T2.ORIGINAL
    path = UNIT2.observation_path(stage)
    assert path == tmp_path / "runs" / "stage-lease-root.json" and not stage.exists()
    observation = json.loads(path.read_text(encoding="utf-8"))
    assert observation["schema"] == JOB2.OBSERVATION_SCHEMA
    assert observation["alias"] == str(layout.alias) and observation["canonical"] == str(layout.canonical)
    assert observation["alias_is_canonical"] is False and observation["canonical_ancestors_link_free"] is True


def test_failure_in_unit1_restores_scope_and_observation_blocks_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    T2.linked_home(tmp_path, monkeypatch)
    monkeypatch.setattr(UNIT1, "os", SimpleNamespace(name="posix"))
    calls: list[Path] = []

    def fail(args: argparse.Namespace) -> dict[str, Any]:
        calls.append(args.stage)
        raise RuntimeError("unit failed")

    monkeypatch.setattr(UNIT1, "execute", fail)
    stage = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="unit failed"):
        UNIT2.execute(_args(stage))
    assert _globals() == T2.ORIGINAL and UNIT2.observation_path(stage).is_file()
    with pytest.raises(FileExistsError):  # 같은 stage 재실행은 관측 기록 단계에서 막힌다
        UNIT2.execute(_args(stage))
    assert calls == [stage] and _globals() == T2.ORIGINAL


def test_real_unit1_lease_inside_v2_scope_takes_canonical_lock_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, layout = T2.build_world_v2(tmp_path, monkeypatch, lease=False, hold=False)
    original = JOB1.validate_protocol

    def validate(path: Path, *, repo: Path = JOB1.ROOT, **kwargs: Any) -> Any:
        return original(path, repo=world.repo, **kwargs)  # UNIT1의 실제 checkout(ROOT) 대신 fixture repo

    monkeypatch.setattr(JOB1, "validate_protocol", validate)
    monkeypatch.setattr(UNIT1, "os", SimpleNamespace(name="posix"))
    seen: list[Any] = []

    def admission(protocol: Path, folder: Path, provider: Path, observed: str, lease: dict[str, str]) -> Path:
        snapshot = json.loads(Path(lease["path"]).read_text(encoding="utf-8"))
        protocol_value = original(protocol, repo=world.repo)
        info = JOB1.verify_lease(lease, protocol=protocol_value, protocol_sha=JOB1._sha(protocol, lf=True))
        seen.append((UNIT2.observation_path(stage).is_file(), snapshot["lock_path"], info["resource_id"]))
        raise Stop

    monkeypatch.setattr(UNIT1, "admission", admission)
    stage = tmp_path / "stage"
    monkeypatch.setattr(JOB1, "LEASE_ROOT", layout.alias)  # v4와 같은 v1 alias root
    with pytest.raises(ValueError, match="lease root 조상에 symlink/reparse"):
        UNIT1.execute(_args(tmp_path / "v1-stage", world.protocol_path))
    assert seen == []
    with pytest.raises(Stop):
        UNIT2.execute(_args(stage, world.protocol_path))
    lock = layout.canonical / f"{FIX.TARGET['resource_id']}.lock"
    assert seen == [(True, str(lock), FIX.TARGET["resource_id"])]
    assert JOB1.LEASE_ROOT == layout.alias and JOB2.scope_active_root() is None
    heartbeat = json.loads((layout.canonical / f"{FIX.TARGET['resource_id']}.heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["active"] is False  # lease가 정상 종료됐다
    assert not stage.exists() and not (tmp_path / "v1-stage").exists()


def test_main_uses_unit1_arguments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(UNIT2, "execute", lambda args: seen.append(args) or {"completed": True})
    argv = ["unit", "--mode", "record", "--protocol", "p.json", "--provider-inventory", "i.json",
            "--provider-observed-at", "2026-10-28T00:00:00+00:00", "--isaaclab", "lab", "--python", "py",
            "--stage", "stage", "--logical-id", FIX.F2_ID]
    monkeypatch.setattr(sys, "argv", argv)
    UNIT2.main()
    assert json.loads(capsys.readouterr().out) == {"completed": True}
    assert seen[0].mode == "record" and seen[0].stage == Path("stage") and seen[0].logical_id == FIX.F2_ID
    monkeypatch.setattr(sys, "argv", ["unit", "--mode", "state-smoke"])
    with pytest.raises(SystemExit):
        UNIT2.main()
