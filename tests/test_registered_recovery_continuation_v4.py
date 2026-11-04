from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import registered_recovery_continuation_v4 as continuation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / (
    "bench/private/claude_reward_track_20261013/codex_main_20261025/vision_goal/"
    "ri_upgrade_451_20261104"
)
def _module():
    return continuation


def _expected(proof_ref: dict[str, str], proof: dict) -> dict:
    identity = proof["identity"]
    return {
        "artifact_id": "prior_static_effect",
        "kind": "completed_effect",
        "case_id": proof["case_id"],
        "method": proof["method"],
        "proof_sha256": proof_ref["sha256"],
        "prior_plan_sha256": identity["prior_plan_sha256"],
        "diagnostic_manifest_sha256": identity["diagnostic_manifest_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "completed_checkpoint_sha256": identity["final_checkpoint_sha256"],
        "seed": identity["seed"],
        "task": identity["task"],
        "correction_id": identity["correction_id"],
        "source_anchor": identity["source_anchor"],
        "registration_anchor": identity["registration_anchor"],
    }


def test_v4_verifies_actual_v7_static_completed_effect(tmp_path: Path) -> None:
    module = _module()
    actual = BASE / "S4b_runs/actual-v7"
    source_result = actual / "results/case-05/static_rule.json"
    before = module.V3._sha(source_result)
    proof_path = tmp_path / "completed-static-effect.json"
    proof_ref = module.build_completed_effect_proof(
        actual, "case-05", "static_rule", proof_path
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    verified = module.verify_completed_effect(
        proof_ref, expected_import=_expected(proof_ref, proof), repo=ROOT
    )
    assert verified["execution_origin"] == "verified_prior_effect"
    assert verified["historical_costs"]["simulator_sdk_launches"] == 2
    assert verified["historical_source_validation"]["current_worktree_authoritative"] is False
    assert module.V3._sha(source_result) == before


def test_v4_rejects_partial_training_without_completion_receipt(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.ContinuationProofError, match="request files are not unique"):
        module.build_completed_effect_proof(
            BASE / "S4b_runs/actual-v7",
            "case-05",
            "one_shot_llm",
            tmp_path / "must-not-exist.json",
        )


def test_v4_rejects_stage_import_identity_change(tmp_path: Path) -> None:
    module = _module()
    proof_path = tmp_path / "completed-static-effect.json"
    proof_ref = module.build_completed_effect_proof(
        BASE / "S4b_runs/actual-v7", "case-05", "static_rule", proof_path
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    expected = _expected(proof_ref, proof)
    expected["diagnostic_manifest_sha256"] = "0" * 64
    with pytest.raises(module.ContinuationProofError, match="identity differs"):
        module.verify_completed_effect(proof_ref, expected_import=expected, repo=ROOT)


def test_v4_rejects_non_completed_status_before_receipt_reuse(tmp_path: Path) -> None:
    module = _module()
    source = BASE / "generic_completed_stages_500/proofs/completed-static-effect-v2.json"
    proof = json.loads(source.read_text(encoding="utf-8"))
    proof["status"] = "running"
    changed = tmp_path / "running.json"
    changed.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")
    changed_ref = {"path": str(changed), "sha256": module.V3._sha(changed)}
    expected = _expected(changed_ref, proof)
    with pytest.raises(module.ContinuationProofError, match="not closed"):
        module.verify_completed_effect(changed_ref, expected_import=expected, repo=ROOT)


def test_v4_rejects_forbidden_ref_before_metadata_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    called = False

    def forbidden_metadata_access(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("forbidden ref reached metadata access")

    monkeypatch.setattr(module.V3, "_validate_ref", forbidden_metadata_access)
    value = {"path": r"C:\safe\ri_codex2_tokenfactory_live_160\secret.json",
             "sha256": "a" * 64}
    with pytest.raises(module.ContinuationProofError, match="forbidden subtree"):
        module._only_ref(value, label="fixture")
    assert called is False


def test_v4_rejects_private_history_closure_before_git_or_history_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = _module()
    called = False

    def forbidden_history_access(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("private closure reached history access")

    monkeypatch.setattr(module.V3, "_historical_closure", forbidden_history_access)
    monkeypatch.setattr(module, "_git_ancestor", forbidden_history_access)
    with pytest.raises(module.ContinuationProofError, match="private subtree"):
        module._historical_closure(
            tmp_path, "a" * 40, "b" * 40,
            {"bench/private/ri_codex2_tokenfactory_live_160/secret.py": "c" * 64})
    assert called is False

