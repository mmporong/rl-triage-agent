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
M = importlib.import_module("lineage_recovery_continuation_v3")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prior(tmp_path: Path):
    root = tmp_path / "execution-v2"
    row = {"case_id": "case-03", "policy": "model_adaptive", "status": "candidate",
           "hold_reason": None, "seed": 2303, "arm": "F1",
           "candidate": {"path": str(tmp_path / "candidate.json"), "sha256": "a" * 64},
           "source_checkpoint_sha256": "b" * 64, "reference_checkpoint_sha256": "c" * 64,
           "reference_report": ref(save(tmp_path / "reference.json", {"versions": {}, "checkpoint": {"sha256": "c" * 64}}))}
    holds = [{**row, "case_id": case, "policy": policy, "status": "HOLD",
              "hold_reason": "probe_outcome_unknown", "candidate": None}
             for case, policy in (("case-03", "registry_next"),
                                  ("case-07", "registry_next"),
                                  ("case-07", "model_adaptive"))]
    manifest = {"rows": [holds[0], row, holds[1], holds[2]], "anchor": "d" * 40,
                "lab": str(tmp_path / "lab"), "protocol": ref(save(tmp_path / "parent.json", {}))}
    stage = root / "runs/case-03_model_adaptive"
    evidence = {"updates": 300, "update_count": 300, "fresh_initialization": True,
                "checkpoint_loaded": False, "resume_requested": False,
                "checkpoint": {"path": str(tmp_path / "model.pt"), "sha256": "e" * 64,
                               "iteration": 299}}
    save(stage / "train/manifest.json", {})
    save(stage / "train/training/training_evidence.json", evidence)
    save(stage / "train/owned-result.json", {"ok": True})
    save(stage / "collected/collected.json", {"ok": True})
    ledger = root / "ledgers/case-03_model_adaptive.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps({"event": event, "request_id": "rid"})
                                 for event in ("proposed", "approved", "consumed")) + "\n")
    evaluation = stage / "evaluation"
    save(evaluation / "gpu-wait/result.json", {"status": "error", "reason": "PermissionError",
                                                "gpu_jobs_started": 0})
    hold = {"schema": M.V1.RECEIPT_SCHEMA, "status": "HOLD", "case_id": "case-03",
            "policy": "registry_next", "sdk_jobs": 0}
    hold_path = save(root / "receipts/case-03_registry_next.json", hold)
    failure = save(root / "failure.json", {"status": "invalid", "error_type": "PermissionError",
        "error": "[Errno 13] Permission denied: 'C:\\Users\\user\\CodexSignals\\gpu-lease.json'",
        "automatic_retry": False, "completed_receipts": [{**hold, **ref(hold_path)}]})
    return root, manifest, row, evidence, ledger, failure


def test_failure_boundary_requires_exact_permission_and_no_evaluation_child(tmp_path):
    root, _manifest, _row, _evidence, _ledger, _failure = prior(tmp_path)
    assert M._failure_boundary(root)[0]["error_type"] == "PermissionError"
    (root / "runs/case-03_model_adaptive/evaluation/owned").mkdir()
    with pytest.raises(ValueError, match="이미 시작"):
        M._failure_boundary(root)


def test_completed_training_reuses_guarded_300_and_rejects_checkpoint_change(tmp_path):
    root, manifest, _row, evidence, _ledger, _failure = prior(tmp_path)
    Path(evidence["checkpoint"]["path"]).write_bytes(b"model")
    evidence["checkpoint"]["sha256"] = hashlib.sha256(b"model").hexdigest()
    save(root / "runs/case-03_model_adaptive/train/training/training_evidence.json", evidence)
    with patch.object(M.V1, "_candidate", return_value={"component": "reward"}), \
            patch.object(M.V1, "_training_manifest", return_value={}), \
            patch.object(M.V1.TRAIN, "verify_binding"), \
            patch.object(M.V1.GUARD, "validate_completion"), \
            patch.object(M.V1.TRAIN, "manifest_fingerprint", return_value="fp"), \
            patch.object(M.V1, "_owned_ok", return_value=True), \
            patch.object(M.V1.RECOVERY_INPUTS, "checked_recovery", return_value=({"ok": True}, {})), \
            patch.object(M.V1.C.D1, "CorrectionLedger") as ledger_type:
        ledger_type.return_value.events.return_value = [
            {"event": event, "request_id": "rid"} for event in ("proposed", "approved", "consumed")]
        value = M._completed_training(manifest, {}, root)
        assert value["evidence"]["updates"] == 300 and value["request_id"] == "rid"
        Path(evidence["checkpoint"]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="checkpoint"):
            M._completed_training(manifest, {}, root)
        Path(evidence["checkpoint"]["path"]).write_bytes(b"model")
        with patch.object(M.V1, "_owned_ok", return_value=False), \
                pytest.raises(ValueError, match="학습이 완결"):
            M._completed_training(manifest, {}, root)


def test_execute_evaluates_once_without_training_and_preserves_ledger_prefix(tmp_path):
    root, parent_manifest, row, evidence, old_ledger, failure = prior(tmp_path)
    checkpoint = Path(evidence["checkpoint"]["path"]); checkpoint.write_bytes(b"model")
    evidence["checkpoint"]["sha256"] = hashlib.sha256(b"model").hexdigest()
    candidate = {"component": "reward", "correction": "restore_body_frame_reward"}
    prior_value = {"row": row, "candidate": candidate, "evidence": evidence,
                   "collected": {"ok": True}, "ledger_path": old_ledger,
                   "request_id": "rid", "stage": root / "runs/case-03_model_adaptive",
                   "owned": {"wall_s": 488.5},
                   "ledger_events": [{"evidence": {"context": {"case_id": "case-03"}}}]}
    continuation = {"failed_execution": {"failure": ref(failure)}}
    evaluation_calls = []

    def evaluate(stage, *_args, **_kwargs):
        evaluation_calls.append(stage)
        report = {"versions": {}, "checkpoint": {"sha256": evidence["checkpoint"]["sha256"]}}
        save(stage / "corrected.json", report)
        return report, {"wall_s": 1.0}

    class Ledger:
        def __init__(self, path, *_args):
            self.path = Path(path)

        def receipt(self, request_id, *_args, **_kwargs):
            event = {"event": "receipt", "request_id": request_id}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            return event

    out = tmp_path / "continued"
    original = old_ledger.read_bytes()
    with patch.object(M, "_load_continuation", return_value=(
            continuation, parent_manifest, {}, {"path": "smoke", "result_sha256": "f" * 64}, prior_value)), \
            patch.object(M.V1, "_fixed_evaluate", side_effect=evaluate), \
            patch.object(M.V1.C.D1, "recovery", return_value={"outcome": "recovered", "verdict": {}}), \
            patch.object(M.V1.C.D1, "CorrectionLedger", Ledger), \
            patch.object(M.V1, "_recovery_receipt", return_value={
                "schema": M.V1.RECEIPT_SCHEMA, "status": "recovered",
                "case_id": "case-03", "policy": "model_adaptive"}), \
            patch.object(M.V1, "_run_training", side_effect=AssertionError("training called")):
        result = M.execute(tmp_path / "manifest.json", out, runner=lambda *_args: {})
    assert result["new_training_jobs"] == 0 and len(evaluation_calls) == 1
    assert old_ledger.read_bytes() == original
    copied = out / "ledger/correction.jsonl"
    assert copied.read_bytes().startswith(original)
    assert json.loads(copied.read_text().splitlines()[-1])["event"] == "receipt"


def test_prepare_protocol_extends_parent_without_changing_registered_values(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    parent = save(repo / "bench/protocols/parent.json", {"schema": M.V2.SCHEMA})
    for name in M.SOURCE_FILES:
        save(repo / name, {"source": name})
    with patch.object(M, "ROOT", repo):
        value = M.prepare_protocol(parent)
    assert value["reuse"] == {"training_updates": 300, "fresh_initialization": True,
                               "new_training_jobs": 0, "new_correction_approvals": 0,
                               "new_model_calls": 0, "evaluation_conditions": 26,
                               "same_seed_reference": True, "first_correction_only": True}
    assert set(value["source_sha256_lf"]) == set(M.SOURCE_FILES)
