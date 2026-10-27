from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import run_vision_pull_record_v1 as R  # pyright: ignore[reportMissingImports]


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "job_path": job, "job_sha256": R.JOB._sha(job),
        "output_directory": tmp_path / "output",
        "checkpoint_path": checkpoint, "checkpoint_sha256": R.JOB._sha(checkpoint),
        "job": {
            "mode": R.JOB.MODE, "python": sys.executable, "isaaclab": str(tmp_path),
            "resource_proof": {"sha256": "r" * 64},
            "checkpoint": {"logical_id": "source-s2220-none"},
            "profile": R.JOB.RECORD.fixed_profile(),
        },
        "resource": {"resource_id": "owned", "gpu_uuid": "GPU-owned"},
    }


def test_sdk_child_seals_success_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(R.JOB, "validate_job", lambda *_args, **_kwargs: value)
    report = {"technical_recording_complete": True}

    class App:
        def close(self) -> None:
            assert (value["output_directory"] / "result.json").is_file()

    def worker(_validated: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        value["output_directory"].mkdir()
        seal_report(report)
        return report

    assert R.sdk_child(value["job_path"], app_factory=lambda _: App(), run_worker=worker) == 0
    result = json.loads((value["output_directory"] / "result.json").read_text())
    assert result["technical_recording_complete"] is True
    assert result["capture_approved"] is False
    assert result["model_ready"] is False


def test_aaba_child_uses_same_checkpoint_in_separate_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(R.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(R.AABA, "_diagnostic", lambda adapted, report: report["checkpoint"] == adapted["checkpoint_sha256"])
    closed: list[bool] = []

    def diagnostic(adapted: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        assert adapted["checkpoint_path"] == value["checkpoint_path"]
        assert adapted["job"]["mode"] == R.AABA_CHECKPOINT.MODE
        assert adapted["job"]["callback_variant"] == R.AABA_CHECKPOINT.VARIANT
        report = {"checkpoint": adapted["checkpoint_sha256"]}
        seal_report(report)
        return report

    class App:
        def close(self) -> None:
            closed.append(True)

    assert R.aaba_child(value["job_path"], app_factory=lambda _: App(), run_diagnostic=diagnostic) == 0
    assert closed == [True]
    assert value["job_path"].read_text(encoding="utf-8") == "{}"


class _Gate(io.BytesIO):
    def close(self) -> None:
        self.accepted = self.getvalue()
        super().close()


class _Process:
    _pid = 100

    def __init__(self, action: Any, exit_code: int = 0) -> None:
        type(self)._pid += 1
        self.pid = type(self)._pid
        self.stdin = _Gate()
        self.action = action
        self.exit_code = exit_code

    def wait(self, *, timeout: float) -> int:
        assert timeout > 0 and self.stdin.accepted == b"GO\n"
        self.action()
        return self.exit_code


class _Reader:
    def __init__(self) -> None:
        self.last_receipt = {"path": "gpu"}

    def __call__(self) -> list[int]:
        return []


@pytest.mark.parametrize("aaba_valid", [True, False])
def test_parent_requires_valid_aaba_before_record_and_rejects_exit_zero_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aaba_valid: bool
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(R.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(R.BASE, "_gpu_reference", lambda *_args: {"path": "gpu"})
    launched: list[str] = []

    def popen(argv: list[str], **_kwargs: Any) -> _Process:
        phase = "aaba" if "--aaba-child" in argv else "record"
        launched.append(phase)
        return _Process(lambda: None)

    if aaba_valid:
        monkeypatch.setattr(R, "verify_aaba", lambda _value: {"diagnostic_complete": True, "worker": {"policy_probe": {"checkpoint_sha256": value["checkpoint_sha256"]}}})
        (tmp_path / R.AABA_REPORT_NAME).write_text("{}", encoding="utf-8")
        value["output_directory"].mkdir()
        (value["output_directory"] / "result.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(R, "verify_terminal", lambda _value: {"completed": True, "technical_recording_complete": True})
    else:
        monkeypatch.setattr(R, "verify_aaba", lambda _value: (_ for _ in ()).throw(ValueError("invalid AABA")))

    receipt = R.run_job(
        value["job_path"], popen=popen, gpu_reader=_Reader(),
        identify=lambda pid: {"pid": pid, "start_ticks": 1}, cleanup=lambda *_args: True,
    )
    if aaba_valid:
        assert launched == ["aaba", "record"]
        assert receipt["completed"] is True
        assert receipt["technical_recording_complete"] is True
        assert receipt["full_recordings"] == 1
        assert receipt["aaba_receipt"]["checkpoint_sha256"] == value["checkpoint_sha256"]
    else:
        assert launched == ["aaba"]
        assert receipt["completed"] is False
        assert receipt["full_recordings"] == 0
        assert receipt["error_message"] == "invalid AABA"


def _aaba_report(checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "schema": R.AABA_CHECKPOINT.SCHEMA,
        "variant": R.AABA_CHECKPOINT.VARIANT,
        "worker": {
            "collection": {"status": "collected_unvalidated", "capture_count": 4, "atomicity_claimed": False},
            "validation": {"approved": True, "camera_sequence": ["A", "A", "B", "A"], "marker_codes": [0, 1, 2, 3], "atomicity_claimed": False},
            "policy_probe": {"checkpoint_sha256": checkpoint_sha256, "environment_steps": 1, "annotators_detached_before_policy_step": True},
            "lifecycle": {"annotators_attached": True, "annotators_detached": True, "environment_closed": True},
            "setup": {"app_update_counter": {"source": "omni.kit.app.IApp.get_update_number", "method": "app.get_update_number()", "initial_value": 14, "after_extension_update": 15, "fail_closed": True}},
            "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
            "diagnostic_complete": True,
        },
        "raw_artifacts": [{"complete_capture": True} for _ in range(4)],
        "primary_error": None,
        "diagnostic_complete": True,
        "used_for_decision": False,
        "capture_approved": False,
        "approved": False,
        "atomicity_claimed": False,
    }


@pytest.mark.parametrize("same_checkpoint", [True, False])
def test_embedded_aaba_checkpoint_controls_record_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_checkpoint: bool,
) -> None:
    value = _validated(tmp_path)
    report = _aaba_report(value["checkpoint_sha256"])
    assert R.AABA._diagnostic(R._aaba_validated(value), report)
    if not same_checkpoint:
        report["worker"]["policy_probe"]["checkpoint_sha256"] = "0" * 64
    # 기존 구조 검사는 두 영수증을 허용하므로 새 CP 입장 검사가 둘을 구분해야 한다.
    assert R.AABA._diagnostic(R._aaba_validated(value), report)
    (tmp_path / R.AABA_REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(R.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(R.BASE, "_gpu_reference", lambda *_args: {"path": "gpu"})
    launched: list[str] = []

    def popen(argv: list[str], **_kwargs: Any) -> _Process:
        phase = "aaba" if "--aaba-child" in argv else "record"
        launched.append(phase)
        return _Process(lambda: None)

    if same_checkpoint:
        value["output_directory"].mkdir()
        (value["output_directory"] / "result.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(R, "verify_terminal", lambda _value: {"completed": True, "technical_recording_complete": True})
    receipt = R.run_job(
        value["job_path"], popen=popen, gpu_reader=_Reader(),
        identify=lambda pid: {"pid": pid, "start_ticks": 1}, cleanup=lambda *_args: True,
    )
    assert launched == (["aaba", "record"] if same_checkpoint else ["aaba"])
    assert receipt["completed"] is same_checkpoint
    if same_checkpoint:
        assert receipt["aaba_receipt"]["checkpoint_sha256"] == report["worker"]["policy_probe"]["checkpoint_sha256"]
    else:
        assert receipt["full_recordings"] == 0
        assert "AABA receipt" in receipt["error_message"]
