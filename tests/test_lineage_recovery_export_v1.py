from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
X = importlib.import_module("lineage_recovery_export_v1")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixed(name: str, checkpoint: str) -> dict:
    return {
        "protocol": "fixed_eval_v1", "protocol_sha256_lf": "a" * 64,
        "eval_seed": 2026, "num_envs": 1040, "horizon_steps": 1000,
        "action_scale": 0.25, "name": name, "checkpoint": {"file": "model.pt", "sha256": checkpoint},
        "versions": {"runtime": "same"},
        "metrics": {"mean_survival_s": 20.0, "fall_rate": 0.0,
                    "lin_vel_rmse_mps": 0.2, "yaw_rate_rmse_radps": 0.2},
        "by_condition": [{"command": [index, 0, 0], "num_envs": 40} for index in range(26)],
    }


def fixture(tmp_path: Path):
    result_root = tmp_path / "execution"
    candidate_path = save(tmp_path / "candidate.json", {
        "component": "reward", "correction": "restore_body_frame_reward",
        "inspection_evidence": {}, "status": "validated_candidate",
    })
    epoch_path = save(tmp_path / "epoch.json", {
        "status": "complete", "end_state": "confirmed", "policy": "model_adaptive",
        "confirmed_subtype": "reward",
    })
    rows = []
    for case, policy, status in (
        ("case-03", "registry_next", "HOLD"), ("case-03", "model_adaptive", "candidate"),
        ("case-07", "registry_next", "HOLD"), ("case-07", "model_adaptive", "HOLD"),
    ):
        rows.append({
            "case_id": case, "policy": policy, "status": status,
            "hold_reason": None if status == "candidate" else "probe_outcome_unknown",
            "arm": "F1" if case == "case-03" else "COMPOUND", "seed": 2303 if case == "case-03" else 2307,
            "candidate": ref(candidate_path) if status == "candidate" else None,
            "epoch_result": ref(epoch_path), "source_checkpoint_sha256": "b" * 64,
            "reference_checkpoint_sha256": "c" * 64,
            "reference_report": {"path": "unused", "sha256": "d" * 64},
        })
    protocol_path = save(tmp_path / "protocol.json", {})
    manifest_path = save(tmp_path / "manifest.json", {
        "rows": rows, "lab": str(tmp_path / "lab"), "anchor": "e" * 40,
        "protocol": ref(protocol_path),
    })
    receipts = []
    for row in rows:
        if row["status"] == "HOLD":
            receipt = {"schema": X.V1.RECEIPT_SCHEMA, "status": "HOLD",
                       "scope": "first_registered_correction_only", "case_id": row["case_id"],
                       "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
                       "approval": False, "model_calls": 0, "overall_goal_complete": False,
                       "second_epoch_registered": False}
        else:
            stage = result_root / "runs/case-03_model_adaptive"
            corrected_path = save(stage / "evaluation/corrected.json", fixed("corrected", "f" * 64))
            reference_path = save(tmp_path / "reference.json", fixed("reference", "c" * 64))
            receipt = {
                "schema": X.V1.RECEIPT_SCHEMA, "status": "recovered",
                "scope": "first_registered_correction_only", "case_id": row["case_id"],
                "policy": row["policy"], "seed": row["seed"], "same_seed_pairing": True,
                "candidate": row["candidate"],
                "new_independent_seed": False, "training_mode": "fresh_restart",
                "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                "reference_checkpoint_sha256": row["reference_checkpoint_sha256"],
                "component": "reward", "correction": "restore_body_frame_reward",
                "second_epoch_registered": False, "model_calls": 0,
                "new_checkpoint": {"path": str(tmp_path / "model.pt"), "sha256": "f" * 64, "iteration": 299},
                "training": {"updates": 300}, "collected_recovery": {"ok": True},
                "corrected_report": ref(corrected_path), "reference_report": ref(reference_path),
                "recovery": {"outcome": "recovered", "verdict": {"label": "healthy"}},
                "evaluation_execution": {"ok": True},
                "ledger_receipt": {"event": "receipt"}, "smoke": {"path": str(tmp_path / "smoke")},
            }
        path = save(result_root / "receipts" / f"{row['case_id']}_{row['policy']}.json", receipt)
        receipts.append({**receipt, **ref(path)})
    result_path = save(result_root / "result.json", {
        "schema": X.V2.SCHEMA + "_result_v1", "status": "complete",
        "scope": "first_registered_correction_only", "overall_goal_complete": False,
        "second_epoch_registered": False, "model_calls": 0, "automatic_retry": False,
        "receipts": receipts,
    })
    return manifest_path, result_path, rows


def patched_validation(manifest_path: Path, result_path: Path, rows: list[dict], *, outcome="recovered"):
    candidate = {"component": "reward", "correction": "restore_body_frame_reward"}
    evidence = {"updates": 300, "update_count": 300, "fresh_initialization": True,
                "checkpoint_loaded": False, "resume_requested": False,
                "checkpoint": {"path": str(result_path.parent / "model.pt"),
                               "sha256": "f" * 64, "iteration": 299}}
    (result_path.parent / "model.pt").write_bytes(b"model")
    evidence["checkpoint"]["sha256"] = hashlib.sha256(b"model").hexdigest()
    receipt_path = result_path.parent / "receipts/case-03_model_adaptive.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["new_checkpoint"] = evidence["checkpoint"]
    receipt["training"] = evidence
    receipt["recovery"] = {"outcome": outcome, "verdict": {"label": "healthy" if outcome == "recovered" else "undetermined"}}
    receipt["status"] = "recovered" if outcome == "recovered" else "HOLD"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = json.loads(result_path.read_text())
    index = next(i for i, row in enumerate(result["receipts"])
                 if row["case_id"] == "case-03" and row["policy"] == "model_adaptive")
    result["receipts"][index] = {**receipt, **ref(receipt_path)}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"] = rows
    corrected_path = Path(receipt["corrected_report"]["path"])
    corrected = json.loads(corrected_path.read_text())
    corrected["checkpoint"]["sha256"] = evidence["checkpoint"]["sha256"]
    corrected_path.write_text(json.dumps(corrected), encoding="utf-8")
    receipt["corrected_report"] = ref(corrected_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = json.loads(result_path.read_text())
    index = next(i for i, row in enumerate(result["receipts"])
                 if row["case_id"] == "case-03" and row["policy"] == "model_adaptive")
    result["receipts"][index] = {**receipt, **ref(receipt_path)}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    candidate_row = next(row for row in manifest["rows"] if row["status"] == "candidate")
    candidate_row["reference_report"] = {**receipt["reference_report"],
                                         "checkpoint_sha256": candidate_row["reference_checkpoint_sha256"]}
    protocol = {}
    stage = result_path.parent / "runs/case-03_model_adaptive"
    save(stage / "train/manifest.json", {})
    save(stage / "train/training/training_evidence.json", evidence)
    save(stage / "collected/collected.json", {"ok": True})
    ledger_events = [{"event": name} for name in ("proposed", "approved", "consumed")]
    ledger_events.append(receipt["ledger_receipt"])

    class Ledger:
        def __init__(self, *_args, **_kwargs):
            pass

        def events(self):
            return ledger_events

    stack = [
        patch.object(X.V1, "_load_manifest", return_value=(manifest, protocol)),
        patch.object(X.V1, "_candidate", return_value=candidate),
        patch.object(X.V1.TRAIN, "verify_binding", return_value=None),
        patch.object(X.V1.GUARD, "validate_completion", return_value=None),
        patch.object(X.V1.TRAIN, "manifest_fingerprint", return_value="fp"),
        patch.object(X.V1.RECOVERY_INPUTS, "checked_recovery", return_value=({"ok": True}, {})),
        patch.object(X.V1.C.D1, "recovery", return_value=receipt["recovery"]),
        patch.object(X.V1, "_owned_ok", return_value=True),
        patch.object(X.V1.C.D1, "CorrectionLedger", Ledger),
        patch.object(X.V1, "verify_smoke", return_value={"ok": True}),
    ]
    return stack, evidence, receipt_path


def enter_all(stack):
    for item in stack:
        item.start()


def exit_all(stack):
    for item in reversed(stack):
        item.stop()


def test_export_writes_whitelist_counts_exact_reports_and_verifies(tmp_path):
    manifest, result, rows = fixture(tmp_path)
    stack, _evidence, _receipt = patched_validation(manifest, result, rows)
    enter_all(stack)
    try:
        out = ROOT / "evals/results" / f"fixture-recovery-export-{tmp_path.name}"
        exported = X.export_completed(manifest, result, out=out)
        assert exported["summary"]["planned_rows"] == 4
        assert exported["summary"]["hold_rows"] == 3
        assert exported["summary"]["training_updates"] == 300
        assert len(list((out / "reports").glob("*.json"))) == 2
        assert X.verify_export(out)["manifest_self_sha256_recorded"] is False
    finally:
        exit_all(stack)
        if out.exists():
            for path in sorted(out.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            out.rmdir()


def test_receipt_tampering_is_rejected(tmp_path):
    manifest, result, rows = fixture(tmp_path)
    stack, _evidence, receipt_path = patched_validation(manifest, result, rows)
    enter_all(stack)
    try:
        value = json.loads(receipt_path.read_text()); value["case_id"] = "case-X"
        receipt_path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="receipt 원문/SHA"):
            X._validated(manifest, result)
        value["case_id"] = "case-03"; receipt_path.write_text(json.dumps(value), encoding="utf-8")
        result_value = json.loads(result.read_text())
        target = next(row for row in result_value["receipts"] if row["policy"] == "model_adaptive")
        target.update(ref(receipt_path)); target["new_checkpoint"]["sha256"] = "0" * 64
        result.write_text(json.dumps(result_value), encoding="utf-8")
        with pytest.raises(ValueError, match="embedded"):
            X._validated(manifest, result)
    finally:
        exit_all(stack)


def test_fixed_report_and_checkpoint_tampering_are_rejected(tmp_path):
    manifest, result, rows = fixture(tmp_path / "report")
    stack, _evidence, receipt_path = patched_validation(manifest, result, rows)
    enter_all(stack)
    try:
        receipt = json.loads(receipt_path.read_text())
        corrected_path = Path(receipt["corrected_report"]["path"])
        report = json.loads(corrected_path.read_text()); report["by_condition"].pop()
        corrected_path.write_text(json.dumps(report), encoding="utf-8")
        receipt["corrected_report"] = ref(corrected_path)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        value = json.loads(result.read_text())
        target = next(row for row in value["receipts"] if row["policy"] == "model_adaptive")
        target.clear(); target.update({**receipt, **ref(receipt_path)})
        result.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="26조건"):
            X._validated(manifest, result)
    finally:
        exit_all(stack)

    manifest2, result2, rows2 = fixture(tmp_path / "checkpoint")
    stack2, _evidence2, _receipt2 = patched_validation(manifest2, result2, rows2)
    enter_all(stack2)
    try:
        (result2.parent / "model.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="checkpoint"):
            X._validated(manifest2, result2)
    finally:
        exit_all(stack2)


def test_missing_hold_and_unknown_promoted_to_recovered_are_rejected(tmp_path):
    manifest, result, rows = fixture(tmp_path)
    stack, _evidence, _receipt_path = patched_validation(manifest, result, rows, outcome="unknown")
    enter_all(stack)
    try:
        result_value = json.loads(result.read_text())
        result_value["receipts"] = [row for row in result_value["receipts"]
                                    if not (row["case_id"] == "case-07" and row["policy"] == "registry_next")]
        result.write_text(json.dumps(result_value), encoding="utf-8")
        with pytest.raises(ValueError, match="4행"):
            X._validated(manifest, result)
    finally:
        exit_all(stack)

    manifest2, result2, rows2 = fixture(tmp_path / "unknown")
    stack2, _evidence2, receipt2 = patched_validation(manifest2, result2, rows2, outcome="unknown")
    enter_all(stack2)
    try:
        raw = json.loads(receipt2.read_text()); raw["status"] = "recovered"
        receipt2.write_text(json.dumps(raw), encoding="utf-8")
        value = json.loads(result2.read_text())
        target = next(row for row in value["receipts"] if row["policy"] == "model_adaptive")
        target.clear(); target.update({**raw, **ref(receipt2)})
        result2.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="승격"):
            X._validated(manifest2, result2)
    finally:
        exit_all(stack2)
