from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]

import lineage_post_correction_inputs_v1 as P


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    recovery_path = _write(tmp_path / "recovery.json", {"stub": "recovery"})
    reference_path = _write(tmp_path / "reference.json", {"stub": "reference"})
    telemetry_path = _write(tmp_path / "telemetry.json", {"summary": {
        field: {"mean_last_20pct": float(i + 1), "n": 300}
        for i, field in enumerate(P.INPUT.V.TELEMETRY_FIELDS)}})
    reference_telemetry_path = _write(tmp_path / "reference-telemetry.json", {"summary": {
        field: {"mean_last_20pct": float(i), "n": 300}
        for i, field in enumerate(P.INPUT.V.TELEMETRY_FIELDS)}})
    lab = tmp_path / "IsaacLab"
    lab.mkdir()
    correction = "restore_body_frame_reward"
    recovery = {
        "source_binding": {"checkpoint_sha256": P._sha(checkpoint),
                           "telemetry_sha256": P._sha(telemetry_path)},
        "private_source_map": {"lab": str(lab), "readback": str(tmp_path / "readback.json")},
        "training": {"seed": 2307, "original_faults": ["F1", "R1"],
                     "remaining_faults": ["R1"], "applied_corrections": [correction]},
    }
    reference = {
        "source_binding": {"checkpoint_sha256": "a" * 64,
                           "telemetry_sha256": P._sha(reference_telemetry_path)},
        "private_source_map": {"seed": 2307, "arm": "NONE"},
    }
    monkeypatch.setattr(
        P.RECOVERY, "checked_recovery",
        lambda _path: (deepcopy(recovery), json.loads(telemetry_path.read_text())))
    monkeypatch.setattr(
        P.INPUT, "_checked_collected",
        lambda _path: (deepcopy(reference), json.loads(reference_telemetry_path.read_text())))
    monkeypatch.setattr(P, "_validate_fixed_reports", lambda _receipt: {"outcome": "not_recovered"})
    ledger_receipt = {"event": "receipt", "request_id": "req", "verdict": "not_recovered",
                      "result_sha": "b" * 64, "exit_code": 0}
    ledger = [
        {"event": "proposed", "request_id": "req", "correction_id": correction},
        {"event": "approved", "request_id": "req"},
        {"event": "consumed", "request_id": "req"}, ledger_receipt,
    ]
    ledger_path = tmp_path / "correction.jsonl"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8")
    receipt = {
        "schema": P.EXEC.RECEIPT_SCHEMA, "status": "not_recovered",
        "next_step": "new_protocol_required", "second_epoch_registered": False,
        "same_seed_pairing": True, "training_mode": "fresh_restart", "seed": 2307,
        "component": "reward", "correction": correction,
        "training": {"updates": 300, "update_count": 300, "fresh_initialization": True,
                     "checkpoint_loaded": False, "resume_requested": False,
                     "applied_corrections": [correction]},
        "collected_recovery": deepcopy(recovery),
        "new_checkpoint": {"path": str(checkpoint), "sha256": P._sha(checkpoint)},
        "reference_checkpoint_sha256": "a" * 64, "ledger_receipt": ledger_receipt,
    }
    receipt_path = _write(tmp_path / "receipt.json", receipt)
    return locals()


def test_prepare_and_packet_use_new_telemetry_without_host_answer_leak(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    value = P.prepare_observation(f["recovery_path"], f["receipt_path"], f["ledger_path"],
                                  f["reference_path"], out=tmp_path / "out", lab=f["lab"])
    checked, _ = P.checked_observation(tmp_path / "out/observation.json")
    packet = P.make_selection_packet(tmp_path / "out/observation.json", f["reference_path"])
    assert checked == value
    assert packet["packet"]["epoch"] == 2
    assert (packet["packet"]["input_bindings"]["case"]["checkpoint_sha256"]
            == P._sha(f["checkpoint"]))
    assert "reward" not in packet["packet"]["remaining_candidates"]
    assert "optimizer_update_count" in packet["packet"]["remaining_candidates"]
    assert packet["private_metadata"]["authority"] == {
        "approval": False, "execution_allowed": False}
    public = json.dumps(value["model_view"], sort_keys=True)
    assert not any(key in public for key in P.FORBIDDEN_PUBLIC_KEYS)


@pytest.mark.parametrize("status", ["recovered", "unknown"])
def test_non_not_recovered_receipt_is_rejected(tmp_path, monkeypatch, status):
    f = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(f["receipt_path"].read_text())
    receipt["status"] = status
    _write(tmp_path / "other.json", receipt)
    with pytest.raises(ValueError, match="not_recovered"):
        P.prepare_observation(f["recovery_path"], tmp_path / "other.json", f["ledger_path"],
                              f["reference_path"], out=tmp_path / "out", lab=f["lab"])


def test_checkpoint_or_telemetry_swap_is_rejected(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(f["receipt_path"].read_text())
    receipt["new_checkpoint"]["sha256"] = "c" * 64
    swapped = _write(tmp_path / "swapped.json", receipt)
    with pytest.raises(ValueError, match="checkpoint"):
        P.prepare_observation(f["recovery_path"], swapped, f["ledger_path"], f["reference_path"],
                              out=tmp_path / "out", lab=f["lab"])
    f["recovery"]["source_binding"]["telemetry_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="recovery collected"):
        P.prepare_observation(f["recovery_path"], f["receipt_path"], f["ledger_path"],
                              f["reference_path"], out=tmp_path / "out2", lab=f["lab"])


def test_ledger_not_consumed_or_receipt_tamper_is_rejected(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    rows = f["ledger"]
    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:2] + rows[3:]), encoding="utf-8")
    with pytest.raises(ValueError, match="네 행"):
        P.prepare_observation(f["recovery_path"], f["receipt_path"], missing, f["reference_path"],
                              out=tmp_path / "out", lab=f["lab"])
    altered = deepcopy(rows)
    altered[-1]["result_sha"] = "e" * 64
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger receipt"):
        P.prepare_observation(f["recovery_path"], f["receipt_path"], bad, f["reference_path"],
                              out=tmp_path / "out2", lab=f["lab"])


def test_none_seed_mismatch_is_rejected(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    f["reference"]["private_source_map"]["seed"] = 9999
    with pytest.raises(ValueError, match="same-seed"):
        P.prepare_observation(f["recovery_path"], f["receipt_path"], f["ledger_path"],
                              f["reference_path"], out=tmp_path / "out", lab=f["lab"])
