"""상태 smoke의 exit0 함정·원문 결합·소유 child 종료를 CPU에서 검증한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_agentic_capture_unit_v1 as U


def fixture(tmp_path, monkeypatch, *, failure=False, wrong_checkpoint=False, cleanup=True):
    cp_sha = "a" * 64
    validated = {"job": {"python": sys.executable, "job_id": "cpu-smoke-fixture"},
                 "checkpoint_path": tmp_path / "fixture.pt", "checkpoint_sha256": cp_sha, "repo": ROOT}
    events = []

    class Process:
        pid = 123456
        returncode = 0

        def wait(self, **_):
            out = tmp_path / "state-smoke"
            out.mkdir()
            U.JOB.write_new(out / "dry_run_result.json", {"completed": True, "frames": 1})
            if failure:
                U.JOB.write_new(out / "dry_run_failure.json", {"completed": False})
            return 0

        def poll(self):
            return self.returncode

    def launch(args, **kwargs):
        events.append((args, kwargs))
        return Process()

    monkeypatch.setattr(U.subprocess, "Popen", launch)
    monkeypatch.setattr(U.JOB.OWNED, "_process_identity", lambda pid: {"pid": pid, "cpu_fixture": True})
    monkeypatch.setattr(U.JOB.OWNED, "_safe_cleanup", lambda *_: cleanup)
    monkeypatch.setattr(U.SIDECAR, "load_sidecar", lambda _: (
        {"source": {"checkpoint_sha256": "b" * 64 if wrong_checkpoint else cp_sha}}, [{}]))
    return validated, events


def test_exit_zero_with_failure_evidence_is_not_success(tmp_path, monkeypatch):
    validated, events = fixture(tmp_path, monkeypatch, failure=True)
    with pytest.raises(RuntimeError, match="증거"):
        U._state_smoke(validated, tmp_path, list)
    receipt = json.loads((tmp_path / "state-smoke-execution.json").read_text(encoding="utf-8"))
    assert receipt["exit_code"] == 0 and receipt["cleanup_verified"] is True and len(events) == 1


def test_smoke_preserves_callback_args_and_ownership(tmp_path, monkeypatch):
    validated, events = fixture(tmp_path, monkeypatch)
    result = U._state_smoke(validated, tmp_path, list)
    assert result["completed"] is True and result["frames"] == 1
    command, options = events[0]
    assert command[-2:] == ["--frames", "1"] and options["start_new_session"] is True
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert result["behavior_label_created"] is False


@pytest.mark.parametrize("problem", ["checkpoint", "cleanup"])
def test_bad_checkpoint_or_cleanup_is_rejected(tmp_path, monkeypatch, problem):
    validated, _ = fixture(tmp_path, monkeypatch, wrong_checkpoint=problem == "checkpoint",
                           cleanup=problem != "cleanup")
    with pytest.raises(RuntimeError, match="결합|증거"):
        U._state_smoke(validated, tmp_path, list)


def test_busy_gpu_does_not_launch_child(tmp_path, monkeypatch):
    validated, events = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="유휴"):
        U._state_smoke(validated, tmp_path, lambda: [999])
    assert events == []


def test_windows_entry_is_rejected_without_a_lease_or_child(monkeypatch):
    monkeypatch.setattr(U, "os", SimpleNamespace(name="nt"))
    with pytest.raises(RuntimeError, match="Linux"):
        U.execute(SimpleNamespace())
