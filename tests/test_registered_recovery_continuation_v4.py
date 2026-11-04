from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import registered_recovery_backend_v1 as backend
from evals import registered_recovery_continuation_v4 as continuation
from rl_triage import investigation_recovery


def _write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        rows = value if isinstance(value, list) else []
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": continuation.V3._sha(path)}


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    proof_status: str = "completed_effect",
    resource_request_id: str = "request-1",
) -> tuple[dict[str, str], dict]:
    case_id, method, request_id = "case-05", "static_rule", "request-1"
    recovery = (tmp_path / "recovery").resolve()
    source_anchor, registration_anchor = "a" * 40, "b" * 40
    prior_plan = _write(tmp_path / "prior-plan.json", {"schema": "plan"})
    diagnostic_source = _write(tmp_path / "diagnostic-source.json", {"schema": "source"})
    diagnostic_manifest = _write(
        tmp_path / "diagnostic-manifest.json", {"source": diagnostic_source}
    )
    approval = _write(tmp_path / "approval.json", {"approved": True})
    fixed_report = _write(tmp_path / "fixed-report.json", {"healthy": False})
    checkpoint = _write(tmp_path / "checkpoint.json", {"iteration": 598})
    protocol = _write(tmp_path / "protocol.json", {
        "source_anchor": source_anchor,
        "source_closure": {},
        "diagnostic": {
            "case_id": case_id,
            "session_manifest_sha256": diagnostic_manifest["sha256"],
            "checkpoint_sha256": "c" * 64,
        },
        "operation": {
            "training_seed": 2026,
            "correction_id": "restore_motor_torque_speed",
        },
    })
    backend_receipt = _write(tmp_path / "backend-receipt.json", {
        "request_id": request_id,
        "exit_code": 0,
        "training_reuse": {"status": "verified"},
        "new_checkpoint": checkpoint,
        "fixed_report": fixed_report,
    })
    intent = _write(recovery / "intents" / f"{request_id}.json", {"request_id": request_id})
    resource = _write(recovery / "resources" / f"{request_id}.json", {
        "status": "complete",
        "request_id": resource_request_id,
        "simulator_sdk_launches": 1,
    })
    ledger_receipt = _write(recovery / "receipts" / f"{request_id}.json", {
        "request_id": request_id,
        "verdict": "not_recovered",
        "receipt_ref": backend_receipt,
    })
    lifecycle = _write(recovery / "recovery.jsonl", [
        {"request_id": request_id, "event": "proposed"},
        {"request_id": request_id, "event": "approved"},
        {"request_id": request_id, "event": "consumed"},
        {"request_id": request_id, "event": "receipt", "verdict": "not_recovered"},
    ])
    recovery_manifest = _write(recovery / "manifest.json", {"status": "prepared"})
    recovery_anchor = _write(recovery / "anchor.json", {"status": "sealed"})
    admission = _write(recovery / "admissions" / f"{request_id}.json", {"admitted": True})
    arm_result = _write(tmp_path / "arm-result.json", {
        "case_id": case_id,
        "method": method,
        "recovery_started": True,
        "outcome": "not_recovered",
        "new_checkpoint": checkpoint,
        "fixed_report": fixed_report,
        "costs": {"training_s": 10.0, "fixed_s": 2.0},
    })
    controller_resource = _write(tmp_path / "controller-resource.json", {
        "status": "complete", "phase_wall_s": 12.0,
    })
    prepared_manifest = _write(tmp_path / "prepared-manifest.json", {
        "plan": prior_plan,
        "arms": [{
            "case_id": case_id,
            "method": method,
            "diagnostic_manifest": diagnostic_manifest,
            "recovery_folder": str(recovery),
        }],
    })
    prepared_anchor = _write(tmp_path / "prepared-anchor.json", {"status": "sealed"})
    refs = {
        "prepared_manifest": prepared_manifest,
        "prepared_anchor": prepared_anchor,
        "prior_plan": prior_plan,
        "arm_result": arm_result,
        "controller_resource": controller_resource,
        "diagnostic_manifest": diagnostic_manifest,
        "diagnostic_source": diagnostic_source,
        "arm_approval": approval,
        "recovery_manifest": recovery_manifest,
        "recovery_anchor": recovery_anchor,
        "recovery_ledger": lifecycle,
        "approval": approval,
        "admission": admission,
        "intent": intent,
        "resource": resource,
        "ledger_receipt": ledger_receipt,
        "protocol": protocol,
        "backend_receipt": backend_receipt,
        "fixed_report": fixed_report,
        "new_checkpoint": checkpoint,
    }
    identity = {
        "prior_plan_sha256": prior_plan["sha256"],
        "diagnostic_manifest_sha256": diagnostic_manifest["sha256"],
        "source_checkpoint_sha256": "c" * 64,
        "final_checkpoint_sha256": checkpoint["sha256"],
        "seed": 2026,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": source_anchor,
        "registration_anchor": registration_anchor,
    }
    proof = {
        "schema": continuation.EFFECT_SCHEMA,
        "status": proof_status,
        "case_id": case_id,
        "method": method,
        "source_anchor": source_anchor,
        "registration_anchor": registration_anchor,
        "request_id": request_id,
        "recovery_folder": str(recovery),
        "identity": identity,
        "refs": refs,
        "claims": {
            "new_dispatch_started": False,
            "ledger_reconsumed": False,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
        },
    }
    proof_ref = _write(tmp_path / "completed-effect.json", proof)
    expected = {
        "artifact_id": "prior_static_effect",
        "kind": "completed_effect",
        "case_id": case_id,
        "method": method,
        "proof_sha256": proof_ref["sha256"],
        "prior_plan_sha256": prior_plan["sha256"],
        "diagnostic_manifest_sha256": diagnostic_manifest["sha256"],
        "source_checkpoint_sha256": "c" * 64,
        "completed_checkpoint_sha256": checkpoint["sha256"],
        "seed": 2026,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": source_anchor,
        "registration_anchor": registration_anchor,
    }
    monkeypatch.setattr(
        continuation,
        "_historical_closure",
        lambda *_args, **_kwargs: {
            "status": "verified", "current_worktree_authoritative": False,
        },
    )
    monkeypatch.setattr(backend, "verify_receipt", lambda *_args, **_kwargs: {"status": "verified"})
    monkeypatch.setattr(
        investigation_recovery,
        "verified_post_observation",
        lambda *_args, **_kwargs: {
            "new_checkpoint": checkpoint, "fixed_report": fixed_report,
            "healthy_declared": False, "behavior_outcome": "failed",
        },
    )
    return proof_ref, expected


def test_v4_verifies_synthetic_completed_effect_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_ref, expected = _fixture(tmp_path, monkeypatch)
    verified = continuation.verify_completed_effect(
        proof_ref, expected_import=expected, repo=tmp_path
    )
    assert verified["execution_origin"] == "verified_prior_effect"
    assert verified["historical_costs"]["simulator_sdk_launches"] == 2
    assert verified["new_dispatch_costs"] == {"sdk_launches": 0, "owned_gpu_s": None}


def test_v4_rejects_non_completed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_ref, expected = _fixture(tmp_path, monkeypatch, proof_status="running")
    with pytest.raises(continuation.ContinuationProofError, match="not closed"):
        continuation.verify_completed_effect(proof_ref, expected_import=expected, repo=tmp_path)


def test_v4_rejects_partial_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_ref, expected = _fixture(tmp_path, monkeypatch)
    expected["completed_checkpoint_sha256"] = "0" * 64
    with pytest.raises(continuation.ContinuationProofError, match="identity differs"):
        continuation.verify_completed_effect(proof_ref, expected_import=expected, repo=tmp_path)


def test_v4_rejects_different_resource_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_ref, expected = _fixture(
        tmp_path, monkeypatch, resource_request_id="other-request"
    )
    with pytest.raises(continuation.ContinuationProofError, match="result/receipt differs"):
        continuation.verify_completed_effect(proof_ref, expected_import=expected, repo=tmp_path)


def test_v4_historical_closure_uses_forward_anchor_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        continuation, "_git_ancestor", lambda _repo, left, right: calls.append((left, right))
    )
    monkeypatch.setattr(
        continuation.V3, "_historical_closure",
        lambda *_args, **_kwargs: {"status": "verified"},
    )
    result = continuation._historical_closure(tmp_path, "source", "registration", {})
    assert calls == [("source", "registration"), ("registration", "HEAD")]
    assert result["current_worktree_authoritative"] is False


def test_v4_rejects_forbidden_ref_before_metadata_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_metadata_access(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("forbidden ref reached metadata access")

    monkeypatch.setattr(continuation.V3, "_validate_ref", forbidden_metadata_access)
    value = {"path": r"C:\safe\ri_codex2_tokenfactory_live_160\secret.json",
             "sha256": "a" * 64}
    with pytest.raises(continuation.ContinuationProofError, match="forbidden subtree"):
        continuation._only_ref(value, label="fixture")
    assert called is False


def test_v4_rejects_private_history_closure_before_git_or_history_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    called = False

    def forbidden_history_access(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("private closure reached history access")

    monkeypatch.setattr(continuation.V3, "_historical_closure", forbidden_history_access)
    monkeypatch.setattr(continuation, "_git_ancestor", forbidden_history_access)
    with pytest.raises(continuation.ContinuationProofError, match="private subtree"):
        continuation._historical_closure(
            tmp_path, "a" * 40, "b" * 40,
            {"bench/private/ri_codex2_tokenfactory_live_160/secret.py": "c" * 64})
    assert called is False
