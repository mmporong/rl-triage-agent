# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import recovery_same_play_capture_v1 as C


def design() -> dict:
    return {
        "targets": {
            "case-03:logs_only": {"case_id": "case-03", "policy": "logs_only", "source_arm": "F1",
                                   "source_pair": {"reference": "source-05", "case": "source-06"},
                                   "first_seed": 4553, "second_seed": 5553}},
    }


def target(checkpoint: Path) -> dict:
    return {"key": "case-03:logs_only", "case_id": "case-03", "policy": "logs_only",
            "source_arm": "F1", "status": "registered_recovered", "final_epoch": "first",
            "registered_seed": 4553, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": C.V1._sha(checkpoint), "original_checkpoint_sha256": "f" * 64,
            "input_pair": {"reference": "source-05", "case": "source-06"}}


def test_build_job_uses_final_checkpoint_without_injection_or_actuator_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"final-r4")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(C, "validate_design", lambda *_args, **_kwargs: design())
    monkeypatch.setattr(C, "validate_source_anchor", lambda *_args: None)
    job = C.build_job(protocol, target(checkpoint), tmp_path / "stage",
                      mode="full", source_anchor="a" * 40, execution_anchor="b" * 40)
    assert job["checkpoint_sha256"] == C.V1._sha(checkpoint)
    assert job["frames"] == 625 and job["fps"] == 50
    assert job["fault_injection"] is False and job["actuator_change"] is False
    assert job["recovery_binding"]["original_checkpoint_sha256"] != job["checkpoint_sha256"]


@pytest.mark.parametrize("field,value,match", [
    ("registered_seed", 4552, "final seed"),
    ("checkpoint_sha256", "0" * 64, "final checkpoint"),
    ("original_checkpoint_sha256", None, "final checkpoint"),
])
def test_build_job_rejects_wrong_seed_or_checkpoint_confusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object, match: str
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"final-r4")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    row = target(checkpoint)
    row[field] = row["checkpoint_sha256"] if field == "original_checkpoint_sha256" else value
    monkeypatch.setattr(C, "validate_design", lambda *_args, **_kwargs: design())
    monkeypatch.setattr(C, "validate_source_anchor", lambda *_args: None)
    with pytest.raises(ValueError, match=match):
        C.build_job(protocol, row, tmp_path / "stage", mode="full",
                    source_anchor="a" * 40, execution_anchor="b" * 40)


def test_execute_rejects_failed_owned_exit_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"final-r4")
    launcher = tmp_path / "python.bat"
    launcher.write_text("fixture", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(C, "validate_design", lambda *_args, **_kwargs: design())
    monkeypatch.setattr(C, "validate_source_anchor", lambda *_args: None)
    lease = lambda *_: {"owner": C.V1.OWNER, "leaseId": "lease-1"}
    runner = lambda *_args, **_kwargs: {"exit_code": 1, "timed_out": False}
    with pytest.raises(ValueError):
        C.execute(protocol, target(checkpoint), "a" * 40, "b" * 40,
                  tmp_path / "stage", launcher, mode="full", runner=runner, lease_reader=lease)
    assert not (tmp_path / "stage" / "execution-result-recovery-v1.json").exists()


def test_sdk_child_rejects_runtime_injection_before_v3(monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    C.V1._write_new(job, {"fault_injection": True, "actuator_change": False})
    monkeypatch.setattr(C.V3, "sdk_child", lambda *_: pytest.fail("V3 must not run"))
    with pytest.raises(ValueError, match="변경 금지"):
        C.sdk_child(job)


def test_sdk_child_rejects_tampered_source_closure_before_v3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    job = tmp_path / "job.json"
    C.V1._write_new(job, {"fault_injection": False, "actuator_change": False,
                          "protocol": C.V1._ref(protocol), "source_anchor": "a" * 40,
                          "source": {"commit": "a" * 40, "files_lf": {"changed": "0" * 64}}})
    monkeypatch.setattr(C, "validate_design", lambda *_: {})
    monkeypatch.setattr(C.V3, "sdk_child", lambda *_: pytest.fail("V3 must not run"))
    with pytest.raises(ValueError, match="source closure"):
        C.sdk_child(job)


def test_registered_design_rejects_source_closure_change(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = ROOT / "bench/protocols/five_type_recovery_media_conditional_20261101_v1.json"
    monkeypatch.setattr(C, "_source_digest", lambda: "0" * 64)
    with pytest.raises(ValueError, match="source closure"):
        C.validate_design(protocol)
