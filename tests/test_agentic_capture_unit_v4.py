"""새 상태 child entry·소유 PID·봉인된 SDK 종료 증거를 부모가 요구하는지 확인한다."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("bench", "evals", "tests", "src")]
import run_agentic_capture_unit_v4 as U
from test_agentic_capture_unit_v1 import fixture as old_smoke_fixture


def smoke(tmp_path, monkeypatch, *, child_pid=123456, rejected=False):
    validated, events = old_smoke_fixture(tmp_path, monkeypatch)

    def verify(path, **expected):
        assert path == tmp_path / "state-smoke"
        assert expected == {"checkpoint_sha256": validated["checkpoint_sha256"],
                            "source_id": "cpu-smoke-fixture-state-smoke", "frames": 1}
        if rejected:
            raise ValueError("fixture shutdown missing")
        return {"child_pid": child_pid, "app_close_returned": False}

    monkeypatch.setitem(sys.modules, "record_agentic_state_sidecar_v2", SimpleNamespace(
        verify_shutdown_evidence=verify, shutdown_markers=lambda _: {"fixture_marker": True}))
    return validated, events


def test_real_unit1_flow_retargets_child_and_accepts_bound_shutdown(tmp_path, monkeypatch):
    validated, events = smoke(tmp_path, monkeypatch)
    result = U._state_smoke(validated, tmp_path, list)
    assert result["completed"] is True and result["shutdown_evidence"]["child_pid"] == 123456
    command, options = events[0]
    assert str(U.NEW_ENTRY) in command and str(U.OLD_ENTRY) not in command
    assert options["start_new_session"] is True
    assert U.V1.subprocess is U._BASE_SUBPROCESS


@pytest.mark.parametrize("problem", ["pid", "missing_shutdown"])
def test_bad_shutdown_prevents_success_and_preserves_markers(tmp_path, monkeypatch, problem):
    validated, _ = smoke(tmp_path, monkeypatch, child_pid=999 if problem == "pid" else 123456,
                         rejected=problem == "missing_shutdown")
    with pytest.raises(ValueError, match="PID|shutdown"):
        U._state_smoke(validated, tmp_path, list)
    rejection = U.V3.J._read(tmp_path / "state-shutdown-rejection.json")
    assert rejection["shutdown_markers"] == {"fixture_marker": True}
    assert U.V1.subprocess is U._BASE_SUBPROCESS


def test_nonzero_exit_is_not_accepted_even_with_shutdown_fixture(tmp_path, monkeypatch):
    validated, events = smoke(tmp_path, monkeypatch)
    original = U._BASE_SUBPROCESS.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        process.returncode = -11
        return process

    monkeypatch.setattr(U._BASE_SUBPROCESS, "Popen", launch)
    with pytest.raises(RuntimeError, match="증거"):
        U._state_smoke(validated, tmp_path, list)
    assert len(events) == 1 and (tmp_path / "state-shutdown-rejection.json").is_file()


def test_new_source_closure_is_required_before_execute(monkeypatch):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(U.V3, "_protocol", lambda _: {"source_sha256_lf": dict.fromkeys(U.V3.SOURCE_FILES, "a" * 64)})
    with pytest.raises(ValueError, match="source closure"):
        U.execute(argparse.Namespace(protocol=Path("fixture.json")))


@pytest.mark.parametrize("missing", ["bench/vision_shutdown_policy_v1.py", "tests/test_vision_shutdown_policy_v1.py"])
def test_shutdown_helper_and_test_are_mandatory(monkeypatch, missing):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="posix"))
    sources = dict.fromkeys(U.SOURCE_FILES, "a" * 64)
    assert missing in sources
    sources.pop(missing)
    monkeypatch.setattr(U.V3, "_protocol", lambda _: {"source_sha256_lf": sources})
    with pytest.raises(ValueError, match="source closure"):
        U.execute(argparse.Namespace(protocol=Path("fixture.json")))


def test_bound_shutdown_helper_mutation_is_rejected_before_dispatch(tmp_path, monkeypatch):
    import test_agentic_play_job_v2 as F

    monkeypatch.setattr(U.V3.J, "REQUIRED_SOURCE_FILES", U.SOURCE_FILES)
    world, _ = F.build_world_v2(tmp_path, monkeypatch, lease=False, hold=False)
    monkeypatch.setattr(U.V3, "ROOT", world.repo)
    calls = []
    monkeypatch.setattr(U.V3, "execute", lambda _: calls.append("cpu_checked") or {"cpu_checked": True})
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="posix"))
    assert U.execute(argparse.Namespace(protocol=world.protocol_path)) == {"cpu_checked": True}
    copied = world.repo / "bench/vision_shutdown_policy_v1.py"
    copied.write_bytes(copied.read_bytes() + b"\n# isolated CPU source mutation\n")
    with pytest.raises(ValueError):
        U.execute(argparse.Namespace(protocol=world.protocol_path))
    assert calls == ["cpu_checked"]


def test_scope_restores_if_unit3_fails(monkeypatch):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(U.V3, "_protocol", lambda _: {"source_sha256_lf": dict.fromkeys(U.SOURCE_FILES, "a" * 64)})

    def fail(_):
        assert U.V1._state_smoke is U._state_smoke
        raise RuntimeError("fixture dispatch failed")

    monkeypatch.setattr(U.V3, "execute", fail)
    with pytest.raises(RuntimeError, match="dispatch"):
        U.execute(argparse.Namespace(protocol=Path("fixture.json")))
    assert U.V1._state_smoke is U._BASE_SMOKE


def test_windows_execute_is_rejected_before_source_or_launch(monkeypatch):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="nt"))
    with pytest.raises(RuntimeError, match="Linux"):
        U.execute(argparse.Namespace())


@pytest.mark.parametrize("exit_code", [0, -11])
def test_actual_sidecar2_producer_and_unit1_parent_are_joined(tmp_path, monkeypatch, exit_code):
    import record_agentic_state_sidecar_v2 as S2
    from test_agentic_state_sidecar_v2 import App, Session, Settings

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"cpu-integrated-policy")
    cp_sha = U.V3.J._sha(checkpoint)
    validated = {"job": {"python": sys.executable, "job_id": "cpu-integrated"},
                 "checkpoint_path": checkpoint, "checkpoint_sha256": cp_sha, "repo": ROOT}
    out = tmp_path / "state-smoke"
    events = []

    class Process:
        pid = os.getpid()
        returncode = None

        def wait(self, **_):
            code = S2.run_state_dry_run(
                checkpoint_path=checkpoint, checkpoint_sha256=cp_sha,
                source_id="cpu-integrated-state-smoke", out_dir=out, frames=1,
                launch_app=lambda: App(events, out), settings_factory=lambda: Settings(events),
                session_factory=lambda _: Session(events, out))
            assert code == 0
            self.returncode = exit_code
            return exit_code

        def poll(self):
            return self.returncode

    def popen(command, **_):
        assert str(U.NEW_ENTRY) in command
        return Process()

    monkeypatch.setattr(U._BASE_SUBPROCESS, "Popen", popen)
    monkeypatch.setattr(U.V1.JOB.OWNED, "_process_identity", lambda pid: {"pid": pid, "cpu_fixture": True})
    monkeypatch.setattr(U.V1.JOB.OWNED, "_safe_cleanup", lambda *_: True)
    if exit_code:
        with pytest.raises(RuntimeError, match="증거"):
            U._state_smoke(validated, tmp_path, list)
        assert (tmp_path / "state-shutdown-rejection.json").is_file()
    else:
        value = U._state_smoke(validated, tmp_path, list)
        assert value["shutdown_evidence"]["child_pid"] == os.getpid()
        assert value["shutdown_evidence"]["app_close_returned"] is True
