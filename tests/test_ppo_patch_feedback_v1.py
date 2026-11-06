from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

from evals import ppo_patch_contract_v1 as CONTRACT
from evals import ppo_patch_feedback_v1 as FEEDBACK

ROOT = Path(__file__).resolve().parents[1]

ROOT_SOURCE = """def build_algorithm_kwargs(config):
    options = dict(config)
    options[\"num_learning_epochs\"] = 1
    return options
"""
BAD_REVISION = """def build_algorithm_kwargs(config):
    options = dict(config)
    options[\"num_learning_epochs\"] = config.get(\"num_learning_epochs\", 5)
    return options
"""
GOOD_REVISION = """def build_algorithm_kwargs(config):
    options = dict(config)
    return options
"""


def _inspection(config: dict, *, updates: int = 3) -> dict:
    epochs = config.get("num_learning_epochs", 1)
    batches = config["num_mini_batches"]
    work = epochs * batches
    return {
        "schema": "rsl_ppo_patch_cpu_inspection_v1",
        "status": "passed",
        "input": {"config": copy.deepcopy(config), "updates": updates},
        "resolved": {
            "num_learning_epochs": epochs,
            "num_mini_batches": batches,
            "num_learning_epochs_source": (
                "config"
                if "num_learning_epochs" in config
                else "installed_ppo_signature_default"
            ),
            "installed_ppo_signature_default": 1,
        },
        "fixture": {"device": "cpu"},
        "measurement": {
            "updates": updates,
            "minibatches": work * updates,
            "optimizer_steps": work * updates,
            "expected_per_update": work,
            "expected_total": work * updates,
            "counts_match": True,
            "per_update": [
                {
                    "index": index,
                    "minibatches": work,
                    "optimizer_steps": work,
                    "return_values_finite": True,
                }
                for index in range(1, updates + 1)
            ],
            "policy_sha256_before": "a" * 64,
            "policy_sha256_after": "b" * 64,
            "changed_parameter_tensors": 2,
            "policy_parameters_finite": True,
        },
        "isolation": {
            "cuda_initialized_before": False,
            "cuda_initialized_after": False,
            "simulation_modules_before": [],
            "simulation_modules_after": [],
        },
    }


def _context() -> dict:
    return {
        "source": ROOT_SOURCE,
        "source_sha256": CONTRACT.source_sha256(ROOT_SOURCE),
        "config": {"num_learning_epochs": 5, "num_mini_batches": 4},
        "facts": {"configured.epochs": 5, "observed.optimizer_steps": 4},
        "observation_scope": {"kind": "cpu_update_trace"},
    }


def _proposal(context: dict, replacement: str) -> dict:
    return {
        "action": "patch",
        "source_sha256": context["source_sha256"],
        "replacement_source": replacement,
        "evidence_ids": [next(iter(context["facts"]))],
        "rationale": "Preserve configured PPO arguments for measured validation.",
    }


def test_import_does_not_load_torch_or_simulation_sdk():
    watched = {"torch", "rsl_rl", "isaac", "carb", "omni"}
    before = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    sys.modules.pop("evals.ppo_patch_feedback_v1", None)
    imported = importlib.import_module("evals.ppo_patch_feedback_v1")
    after = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    assert imported.SCHEMA == "ppo_patch_cpu_validation_v1"
    assert after == before


def test_validation_plan_covers_explicit_and_installed_default_cases():
    plan = FEEDBACK.validation_plan()
    assert [(row.get("num_learning_epochs"), row["num_mini_batches"]) for row in plan] == [
        (4, 3),
        (2, 4),
        (7, 2),
        (None, 4),
    ]
    assert plan[2]["gamma"] == 0.97
    assert plan[2]["clip_param"] == 0.17
    assert all(row["learning_rate"] == 0.001 for row in plan)
    assert all(row["schedule"] == "fixed" for row in plan)


def test_good_adapter_preserves_every_configuration_and_passes():
    result = FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=_inspection)
    assert result["all_passed"] is True
    assert all(record["passed"] for record in result["records"])
    assert all(
        record["algorithm_kwargs"] == record["configuration"]
        for record in result["records"]
    )


def test_default_edge_exposes_the_observed_model_candidate():
    raw = json.loads(
        (
            ROOT
            / "evals/results/ppo_generated_patch_cpu_20261106_v1/model/late_override.response-01.json"
        ).read_text(encoding="utf-8")
    )
    source = raw["proposal"]["replacement_source"]
    assert source == BAD_REVISION.rstrip("\n")
    result = FEEDBACK.validate_on_cpu(source, inspector=_inspection)
    assert [record["passed"] for record in result["records"]] == [
        True,
        True,
        True,
        False,
    ]
    assert "adapter.output_equals_configuration" in result["records"][3][
        "failed_checks"
    ]
    assert "inspection.resolved_epochs" in result["records"][3]["failed_checks"]


def test_measurement_checks_resolved_values_even_when_total_work_matches():
    def wrong_resolution(config: dict, *, updates: int = 3) -> dict:
        result = _inspection(config, updates=updates)
        result["resolved"]["num_learning_epochs"] = 2
        result["resolved"]["num_mini_batches"] = 6
        return result

    result = FEEDBACK.validate_on_cpu(
        GOOD_REVISION,
        configurations=[FEEDBACK.validation_plan()[0]],
        inspector=wrong_resolution,
    )
    assert result["all_passed"] is False
    assert set(result["records"][0]["failed_checks"]) == {
        "inspection.resolved_batches",
        "inspection.resolved_epochs",
    }


@pytest.mark.parametrize(
    "mutation",
    ["cuda", "device", "unchanged_weights", "nonfinite", "per_update", "simulation"],
)
def test_runtime_mutations_are_rejected(mutation: str):
    def altered(config: dict, *, updates: int = 3) -> dict:
        result = _inspection(config, updates=updates)
        if mutation == "cuda":
            result["isolation"]["cuda_initialized_after"] = True
        elif mutation == "device":
            result["fixture"]["device"] = "cuda:0"
        elif mutation == "unchanged_weights":
            result["measurement"]["policy_sha256_after"] = "a" * 64
        elif mutation == "nonfinite":
            result["measurement"]["policy_parameters_finite"] = False
        elif mutation == "per_update":
            result["measurement"]["per_update"][0]["optimizer_steps"] += 1
        else:
            result["isolation"]["simulation_modules_after"] = ["omni"]
        return result

    result = FEEDBACK.validate_on_cpu(
        GOOD_REVISION,
        configurations=[FEEDBACK.validation_plan()[0]],
        inspector=altered,
    )
    assert result["all_passed"] is False


def test_feedback_context_contains_failures_without_a_correct_source():
    root = _context()
    first = _proposal(root, BAD_REVISION)
    validation = FEEDBACK.validate_on_cpu(BAD_REVISION, inspector=_inspection)
    context = FEEDBACK.build_feedback_context(root, first, validation)
    assert context["source"] == BAD_REVISION
    assert context["config"] == root["config"]
    assert context["revision_index"] == context["max_revisions"] == 1
    assert "adapter.output_equals_configuration" in context["facts"][
        "feedback.record3.failed_checks"
    ]
    assert "inspection.resolved_epochs" in context["facts"][
        "feedback.record3.failed_checks"
    ]
    assert context["feedback"]["first_validation"] == validation
    serialized = json.dumps(context, sort_keys=True)
    assert GOOD_REVISION not in serialized


def test_passing_validation_cannot_request_feedback():
    root = _context()
    proposal = _proposal(root, GOOD_REVISION)
    validation = FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=_inspection)
    with pytest.raises(FEEDBACK.PatchFeedbackError, match="must not request"):
        FEEDBACK.build_feedback_context(root, proposal, validation)


@pytest.mark.parametrize("field", ["passed", "all_passed", "measurement"])
def test_stored_validation_verdict_is_recomputed(field: str):
    validation = FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=_inspection)
    tampered = copy.deepcopy(validation)
    if field == "passed":
        tampered["records"][0]["passed"] = False
    elif field == "all_passed":
        tampered["all_passed"] = False
    else:
        tampered["records"][0]["measurement"]["measurement"]["optimizer_steps"] += 1
    with pytest.raises(FEEDBACK.PatchFeedbackError):
        FEEDBACK.verify_validation(GOOD_REVISION, tampered)


def test_error_record_cannot_be_promoted_to_success():
    def failed_inspector(config: dict, *, updates: int = 3) -> dict:
        raise RuntimeError(f"inspection unavailable for {updates} updates: {len(config)}")

    validation = FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=failed_inspector)
    assert FEEDBACK.verify_validation(GOOD_REVISION, validation)["all_passed"] is False
    promoted = copy.deepcopy(validation)
    promoted["records"][0]["passed"] = True
    promoted["records"][0]["failed_checks"] = []
    with pytest.raises(FEEDBACK.PatchFeedbackError, match="error record"):
        FEEDBACK.verify_validation(GOOD_REVISION, promoted)


def test_one_revision_transition_preserves_native_proposals_and_separates_results():
    root = _context()
    first = _proposal(root, BAD_REVISION)
    first_validation = FEEDBACK.validate_on_cpu(BAD_REVISION, inspector=_inspection)
    revision_context = FEEDBACK.build_feedback_context(root, first, first_validation)
    revised = _proposal(revision_context, GOOD_REVISION)
    revised["evidence_ids"] = ["feedback.record3.failed_checks"]
    revised_validation = FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=_inspection)
    transition = FEEDBACK.validate_transition(
        root,
        first,
        first_validation,
        revision_context,
        revised,
        revised_validation,
    )
    assert transition["status"] == "passed_after_feedback"
    assert transition["first_validation_passed"] is False
    assert transition["after_feedback_validation_passed"] is True
    assert transition["first_proposal"] == first
    assert transition["revision_proposal"] == revised
    assert transition["final_source"] == GOOD_REVISION
    assert transition["approval_granted"] is False
    assert transition["execution_authorized"] is False


def test_transition_rejects_feedback_context_tampering():
    root = _context()
    first = _proposal(root, BAD_REVISION)
    first_validation = FEEDBACK.validate_on_cpu(BAD_REVISION, inspector=_inspection)
    revision_context = FEEDBACK.build_feedback_context(root, first, first_validation)
    revised = _proposal(revision_context, GOOD_REVISION)
    revised["evidence_ids"] = ["feedback.record3.failed_checks"]
    tampered = copy.deepcopy(revision_context)
    tampered["facts"]["feedback.record3.failed_checks"] = []
    with pytest.raises(FEEDBACK.PatchFeedbackError, match="does not match"):
        FEEDBACK.validate_transition(
            root,
            first,
            first_validation,
            tampered,
            revised,
            FEEDBACK.validate_on_cpu(GOOD_REVISION, inspector=_inspection),
        )
