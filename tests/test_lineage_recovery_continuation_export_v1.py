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
X = importlib.import_module("lineage_recovery_continuation_export_v1")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixed(name: str, checkpoint: str) -> dict:
    return {"protocol": "fixed_eval_v1", "eval_seed": 2026, "num_envs": 1040,
            "horizon_steps": 1000, "action_scale": 0.25, "name": name,
            "checkpoint": {"file": "model.pt", "sha256": checkpoint}, "versions": {},
            "by_condition": [{"command": [index, 0, 0]} for index in range(26)]}


def fixture(tmp_path: Path, *, outcome="recovered"):
    result_root = tmp_path / "execution-v4"
    old_root = tmp_path / "execution-v2"
    candidate_path = save(tmp_path / "candidate.json", {"correction": "restore_body_frame_reward"})
    epoch_path = save(tmp_path / "epoch.json", {"status": "complete"})
    reference_path = save(tmp_path / "reference.json", fixed("reference", "c" * 64))
    candidate = {"case_id": "case-03", "policy": "model_adaptive", "status": "candidate",
                 "hold_reason": None, "seed": 2303, "arm": "F1", "candidate": ref(candidate_path),
                 "epoch_result": ref(epoch_path), "reference_checkpoint_sha256": "c" * 64,
                 "reference_report": ref(reference_path)}
    holds = [{**candidate, "case_id": case, "policy": policy, "status": "HOLD",
              "hold_reason": "probe_outcome_unknown", "candidate": None}
             for case, policy in (("case-03", "registry_next"), ("case-07", "registry_next"),
                                  ("case-07", "model_adaptive"))]
    parent = {"anchor": "a" * 40, "rows": [holds[0], candidate, holds[1], holds[2]],
              "batch_manifest": {"path": str(tmp_path / "batch-manifest.json")}}
    evidence = {"updates": 300, "checkpoint": {"sha256": "b" * 64}}
    old_train_manifest = save(old_root / "runs/case-03_model_adaptive/train/manifest.json", {})
    old_ledger = old_root / "ledgers/case-03_model_adaptive.jsonl"; old_ledger.parent.mkdir(parents=True)
    events = [{"event": name, "request_id": "rid"} for name in ("proposed", "approved", "consumed")]
    old_ledger.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    prior = {"row": candidate, "candidate": {"correction": "restore_body_frame_reward"},
             "evidence": evidence, "collected": {"ok": True}, "stage": old_train_manifest.parents[1],
             "ledger_path": old_ledger, "ledger_events": events, "parent_manifest": parent}
    continuation = {"anchor": "d" * 40,
                    "failed_execution": {"failure": {"sha256": "e" * 64}},
                    "training": {"evidence": {"sha256": "f" * 64}},
                    "ledger": {"source": {"sha256": hashlib.sha256(old_ledger.read_bytes()).hexdigest()}}}
    corrected = save(result_root / "evaluation/corrected.json", fixed("corrected", "b" * 64))
    new_ledger = result_root / "ledger/correction.jsonl"; new_ledger.parent.mkdir(parents=True)
    last = {"event": "receipt", "request_id": "rid"}
    new_ledger.write_bytes(old_ledger.read_bytes() + (json.dumps(last) + "\n").encode())
    save(result_root / "ledger/prefix-provenance.json", {
        "source": ref(old_ledger), "copied_prefix_sha256": hashlib.sha256(old_ledger.read_bytes()).hexdigest(),
        "copied_prefix_size": len(old_ledger.read_bytes()), "events": 3})
    status = "HOLD" if outcome == "unknown" else outcome
    candidate_receipt = {
        "schema": X.V1.RECEIPT_SCHEMA, "case_id": "case-03", "policy": "model_adaptive",
        "status": status, "continuation": True, "training_reused": True, "new_training_jobs": 0,
        "model_calls": 0, "candidate": candidate["candidate"], "new_checkpoint": evidence["checkpoint"],
        "training": evidence, "collected_recovery": {"ok": True},
        "prior_failed_execution_sha256": "e" * 64, "prior_training_stage": ref(old_train_manifest),
        "corrected_report": ref(corrected), "reference_report": ref(reference_path),
        "recovery": {"outcome": outcome, "verdict": {}},
        "evaluation_execution": {"ok": True}, "ledger_receipt": last,
    }
    receipts = []
    for row in parent["rows"]:
        if row["status"] == "HOLD":
            receipt = {"schema": X.V1.RECEIPT_SCHEMA, "status": "HOLD",
                       "scope": "first_registered_correction_only", "case_id": row["case_id"],
                       "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
                       "approval": False, "model_calls": 0, "overall_goal_complete": False,
                       "second_epoch_registered": False}
        else:
            receipt = candidate_receipt
        path = save(result_root / "receipts" / f"{row['case_id']}_{row['policy']}.json", receipt)
        receipts.append({**receipt, **ref(path)})
    result = {"schema": X.V4.RESULT_SCHEMA, "status": "complete",
              "scope": "first_registered_correction_evaluation_continuation",
              "planned_rows": 4, "hold_rows": 3, "evaluated_rows": 1,
              "reused_training_rows": 1, "new_training_jobs": 0, "model_calls": 0,
              "new_correction_approvals": 0, "evaluation_conditions": 26,
              "same_seed_pairing": True, "overall_goal_complete": False, "receipts": receipts}
    manifest_path = save(tmp_path / "inputs-v4/manifest.json", {})
    result_path = save(result_root / "result.json", result)
    return manifest_path, result_path, continuation, parent, prior, candidate_receipt


def validation_patches(continuation, parent, prior, candidate_receipt):
    class Ledger:
        def __init__(self, path, *_args): self.path = Path(path)
        def events(self): return [json.loads(line) for line in self.path.read_text().splitlines()]
    return [
        patch.object(X.V3, "_load_continuation", return_value=(continuation, parent, {}, {}, prior)),
        patch.object(X, "_model_origin", return_value={
            "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "transport_protocol_sha256": "1" * 64, "diagnosis_epoch_result_sha256": "2" * 64}),
        patch.object(X.V1.C.D1, "recovery", return_value=candidate_receipt["recovery"]),
        patch.object(X.V1, "_owned_ok", return_value=True),
        patch.object(X.V1.C.D1, "CorrectionLedger", Ledger),
    ]


def start(items):
    for item in items: item.start()


def stop(items):
    for item in reversed(items): item.stop()


def test_export_whitelist_reports_counts_and_verifies(tmp_path):
    manifest, result, continuation, parent, prior, receipt = fixture(tmp_path)
    patches = validation_patches(continuation, parent, prior, receipt); start(patches)
    out = ROOT / "evals/results" / f"fixture-continuation-export-{tmp_path.name}"
    try:
        value = X.export_completed(manifest, result, out=out)
        assert value["summary"]["reused_training_updates"] == 300
        assert value["summary"]["new_training_jobs"] == 0
        assert len(list((out / "reports").glob("*.json"))) == 2
        assert X.verify_export(out)["manifest_self_sha256_recorded"] is False
    finally:
        stop(patches)
        if out.exists():
            for path in sorted(out.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            out.rmdir()


@pytest.mark.parametrize("mutation,match", [
    ("missing_hold", "4행"), ("ledger_prefix", "ledger prefix"),
    ("receipt", "receipt 원문/SHA"), ("unknown_promoted", "승격"),
])
def test_tamper_hold_ledger_receipt_and_unknown_promotion_rejected(tmp_path, mutation, match):
    outcome = "unknown" if mutation == "unknown_promoted" else "recovered"
    manifest, result, continuation, parent, prior, receipt = fixture(tmp_path, outcome=outcome)
    patches = validation_patches(continuation, parent, prior, receipt); start(patches)
    try:
        value = json.loads(result.read_text())
        if mutation == "missing_hold":
            value["receipts"].pop(0); result.write_text(json.dumps(value))
        elif mutation == "ledger_prefix":
            ledger = result.parent / "ledger/correction.jsonl"
            ledger.write_bytes(b"changed\n" + ledger.read_bytes())
        else:
            target = next(row for row in value["receipts"] if row["policy"] == "model_adaptive")
            path = Path(target["path"])
            raw = json.loads(path.read_text())
            if mutation == "receipt":
                raw["case_id"] = "case-X"; path.write_text(json.dumps(raw))
            else:
                raw["status"] = "recovered"; path.write_text(json.dumps(raw))
                target.clear(); target.update({**raw, **ref(path)}); result.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=match):
            X._validated(manifest, result)
    finally:
        stop(patches)
