import inspect
from copy import deepcopy
from pathlib import Path

import pytest

from evals import registered_recovery_continuation_v5 as continuation


def _identity() -> dict:
    return {
        "prior_plan_sha256": "1" * 64,
        "diagnostic_manifest_sha256": "2" * 64,
        "source_checkpoint_sha256": "3" * 64,
        "final_checkpoint_sha256": "4" * 64,
        "seed": 2665,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": "5" * 40,
        "registration_anchor": "6" * 40,
    }


def _stage_import() -> dict:
    identity = _identity()
    return {
        "artifact_id": "prior_training",
        "kind": "completed_training",
        "case_id": "case-06",
        "method": "fixed_sequence",
        "proof_sha256": "7" * 64,
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


def test_training_proof_builder_and_loader_keep_the_registered_signatures() -> None:
    assert list(inspect.signature(continuation.build_completed_training_proof).parameters) == [
        "old_prepared", "case_id", "method", "output",
    ]
    parameters = inspect.signature(continuation.load_completed_training).parameters
    assert list(parameters) == [
        "proof_ref", "repo", "isaaclab", "expected_import",
        "expected_source_checkpoint_sha256", "expected_seed", "expected_correction_id",
        "expected_task", "expected_diagnostic_manifest_sha256", "expected_binding",
        "observed_anchor",
    ]


def test_completed_effect_verifier_delegates_to_v4_without_changing_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_ref = {"path": "proof.json", "sha256": "1" * 64}
    expected_import = {"kind": "completed_effect", "proof_sha256": "1" * 64}
    repo = Path("repo-root")
    calls: list[tuple[object, object, object]] = []
    sentinel = {"status": "completed_effect"}

    def fake_verify(
        received_proof: object,
        *,
        expected_import: object,
        repo: object,
    ) -> dict:
        calls.append((received_proof, expected_import, repo))
        return sentinel

    monkeypatch.setattr(continuation.V4, "verify_completed_effect", fake_verify)
    assert continuation.verify_completed_effect(
        proof_ref,
        expected_import=expected_import,
        repo=repo,
    ) is sentinel
    assert calls == [(proof_ref, expected_import, repo)]


def test_stage_import_requires_exact_identity_and_arm() -> None:
    proof = {"identity": _identity(), "case_id": "case-06", "method": "fixed_sequence"}
    assert continuation._expected_identity(
        proof,
        expected_import=_stage_import(),
        expected_source_checkpoint_sha256=None,
        expected_seed=None,
        expected_correction_id=None,
        expected_task=None,
        expected_diagnostic_manifest_sha256=None,
    ) == _identity()
    changed = _stage_import()
    changed["completed_checkpoint_sha256"] = "8" * 64
    with pytest.raises(continuation.ContinuationProofError, match="caller expectation"):
        continuation._expected_identity(
            proof,
            expected_import=changed,
            expected_source_checkpoint_sha256=None,
            expected_seed=None,
            expected_correction_id=None,
            expected_task=None,
            expected_diagnostic_manifest_sha256=None,
        )


def test_backend_expectations_cannot_replace_the_proof_anchors() -> None:
    proof = {"identity": _identity(), "case_id": "case-06", "method": "fixed_sequence"}
    actual = continuation._expected_identity(
        proof,
        expected_import=None,
        expected_source_checkpoint_sha256="3" * 64,
        expected_seed=2665,
        expected_correction_id="restore_motor_torque_speed",
        expected_task="Isaac-Velocity-Flat-Unitree-Go2-v0",
        expected_diagnostic_manifest_sha256="2" * 64,
    )
    assert actual["prior_plan_sha256"] == "1" * 64
    changed = deepcopy(proof)
    changed["identity"]["source_checkpoint_sha256"] = "9" * 64
    with pytest.raises(continuation.ContinuationProofError, match="caller expectation"):
        continuation._expected_identity(
            changed,
            expected_import=None,
            expected_source_checkpoint_sha256="3" * 64,
            expected_seed=2665,
            expected_correction_id="restore_motor_torque_speed",
            expected_task="Isaac-Velocity-Flat-Unitree-Go2-v0",
            expected_diagnostic_manifest_sha256="2" * 64,
        )


def test_empty_fixed_folder_is_allowed_but_execution_artifacts_are_not(tmp_path: Path) -> None:
    training = tmp_path / "execution" / "training"
    fixed = training.parent / "fixed"
    training.mkdir(parents=True)
    fixed.mkdir()
    continuation._validate_fixed_not_started(training)
    (fixed / "execution.json").write_text("{}", encoding="utf-8")
    with pytest.raises(continuation.ContinuationProofError, match="execution evidence"):
        continuation._validate_fixed_not_started(training)


def test_claims_do_not_relabel_the_unknown_request() -> None:
    assert continuation._CLAIMS == {
        "new_training_launched": False,
        "fixed_evaluation_executed": False,
        "recovery_label_available": False,
        "request_reconsumed": False,
        "full_environment_state_restored": False,
        "rng_state_restored_from_checkpoint": False,
        "policy_optimizer_resumed": True,
    }
