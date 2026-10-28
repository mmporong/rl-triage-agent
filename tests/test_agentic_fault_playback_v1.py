"""등록된 callback의 env 생성 전 설치·실패 뒤 복원·결합 거부 CPU 회귀."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import agentic_fault_playback_v1 as F


def inputs(tmp_path, fault="F2"):
    value = F.registration("motor_case_01", "a" * 64, fault)
    sources = {name: F.R.sha256_file(ROOT / name, lf=True) for name in F.SOURCE_FILES}
    validated = {"repo": ROOT, "job_path": tmp_path / "job.json", "job_sha256": "b" * 64,
        "checkpoint_sha256": "a" * 64,
        "job": {"checkpoint": {"logical_id": value["logical_id"], "sha256": "a" * 64},
                "source": {"files_lf": sources}, "profile": F.R.fixed_profile()},
        "protocol": {"playback_registrations": {value["logical_id"]: value}, "source_sha256_lf": sources}}
    return value, validated


def test_registered_fault_is_installed_before_runner_and_restored(tmp_path):
    value, validated = inputs(tmp_path)
    events = []

    def installer(reg):
        events.append(reg["fault_id"])
        return {"class_method_changed": True}, lambda: events.append("restored")

    def runner(*_, **kwargs):
        assert events == ["F2"] and kwargs == {"camera_capture_factory": "same-callback"}
        events.append("env_and_record")
        return ("trace", "video")

    result = F.make_run_real(value, runner=runner, installer=installer)(
        validated, object(), camera_capture_factory="same-callback")
    assert result == ("trace", "video") and events == ["F2", "env_and_record", "restored"]
    receipt = json.loads((tmp_path / "playback-receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed" and receipt["restore_completed"] is True
    assert receipt["used_for_recovery_decision"] is False


def test_runner_failure_is_recorded_and_restored_before_reraise(tmp_path):
    value, validated = inputs(tmp_path)
    events = []

    def runner(*_):
        raise RuntimeError("CPU capture failure")

    with pytest.raises(RuntimeError, match="CPU capture"):
        F.make_run_real(value, runner=runner, installer=lambda _: ({}, lambda: events.append("restored")))(
            validated, object())
    receipt = json.loads((tmp_path / "playback-receipt.json").read_text(encoding="utf-8"))
    assert events == ["restored"] and receipt["primary_error"] == "RuntimeError"
    assert receipt["status"] == "failed" and receipt["restore_completed"] is True


@pytest.mark.parametrize("field", ["checkpoint", "catalog", "source", "task"])
def test_mismatched_binding_never_installs_fault(tmp_path, field):
    value, validated = inputs(tmp_path)
    if field == "checkpoint":
        validated["checkpoint_sha256"] = "c" * 64
    elif field == "catalog":
        validated["protocol"]["playback_registrations"] = {}
    elif field == "source":
        validated["job"]["source"]["files_lf"] = {}
    else:
        validated["job"]["profile"]["task"] = "unsupported-task"
    with pytest.raises(ValueError, match="결합|closure"):
        F.make_run_real(value, installer=lambda _: pytest.fail("결합 실패 뒤 주입"))(validated, object())
    assert not (tmp_path / "playback-intent.json").exists()


def test_none_default_installer_does_not_import_isaac(tmp_path):
    value, validated = inputs(tmp_path, "NONE")
    assert F.make_run_real(value, runner=lambda *_: "native")(validated, object()) == "native"
    receipt = json.loads((tmp_path / "playback-receipt.json").read_text(encoding="utf-8"))
    assert receipt["installation"] == {"fault_id": "NONE", "class_method_changed": False}
    with pytest.raises(FileExistsError):
        F.make_run_real(value, installer=lambda _: pytest.fail("중복 주입"))(validated, object())


def test_f2_default_installer_uses_frozen_fault_and_restores_class(tmp_path, monkeypatch):
    value, validated = inputs(tmp_path)

    class Motor:
        def _clip_effort(self, effort):
            return effort

    original = Motor._clip_effort
    original.__module__ = "isaaclab.actuators.actuator_pd"
    module = types.ModuleType("isaaclab.actuators.actuator_pd")
    module.DCMotor = Motor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    def runner(*_):
        assert Motor._clip_effort is not original
        assert Motor._clip_effort.__module__ == "hidden_faults"
        return "native-cpu-fixture"

    assert F.make_run_real(value, runner=runner)(validated, object()) == "native-cpu-fixture"
    assert Motor._clip_effort is original
