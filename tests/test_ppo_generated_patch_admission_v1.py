from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import ppo_generated_patch_admission_v1 as A
import ppo_patch_contract_v1 as CONTRACT

HEALTHY = '''def build_algorithm_kwargs(config):
    options = dict(config)
    return options
'''
BROKEN = '''def build_algorithm_kwargs(config):
    options = dict(config)
    options["num_learning_epochs"] = 1
    return options
'''
PATCH = '''def build_algorithm_kwargs(config):
    options = dict(config)
    options["num_learning_epochs"] = config["num_learning_epochs"]
    return options
'''
DEFAULT_INSERTING_PATCH = '''def build_algorithm_kwargs(config):
    options = dict(config)
    options["num_learning_epochs"] = config.get("num_learning_epochs", 5)
    return options
'''


CONFIGS = [
    {"num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 0.001},
    {"num_learning_epochs": 3, "num_mini_batches": 2, "learning_rate": 0.002},
    {"num_learning_epochs": 2, "num_mini_batches": 3, "learning_rate": 0.003},
]


def _proposal(original: str, replacement: str | None = PATCH, action: str = "patch") -> dict:
    return {
        "action": action,
        "source_sha256": CONTRACT.source_sha256(original),
        "replacement_source": replacement,
        "evidence_ids": ["configured_epochs", "observed_steps"],
        "rationale": "관측된 갱신 횟수와 설정값의 전달을 비교한다.",
    }


def _row(
    source: str,
    configuration: dict,
    *,
    force_counts: int | None = None,
    cuda: bool = False,
) -> dict:
    kwargs = CONTRACT.apply_adapter(source, dict(configuration))
    configured_epochs = configuration.get("num_learning_epochs", 1)
    configured_expected = configured_epochs * configuration["num_mini_batches"]
    resolved_epochs = kwargs.get("num_learning_epochs", 1)
    resolved_mini_batches = kwargs["num_mini_batches"]
    runtime_expected = resolved_epochs * resolved_mini_batches
    observed = force_counts if force_counts is not None else runtime_expected
    updates = 2
    per_update = [
        {
            "index": index,
            "minibatches": observed,
            "optimizer_steps": observed,
            "return_values_finite": True,
        }
        for index in range(1, updates + 1)
    ]
    runtime_counts_match = observed == runtime_expected
    return {
        "variant": "fixture",
        "configuration": copy.deepcopy(configuration),
        "algorithm_kwargs": copy.deepcopy(kwargs),
        "source_sha256": CONTRACT.source_sha256(source),
        "measurement": {
            "status": "passed" if runtime_counts_match and not cuda else "failed",
            "input": {"config": copy.deepcopy(kwargs)},
            "resolved": {
                "num_learning_epochs": resolved_epochs,
                "num_mini_batches": resolved_mini_batches,
                "num_learning_epochs_source": (
                    "config"
                    if "num_learning_epochs" in kwargs
                    else "installed_ppo_signature_default"
                ),
                "installed_ppo_signature_default": 1,
            },
            "measurement": {
                "updates": updates,
                "minibatches": observed * updates,
                "optimizer_steps": observed * updates,
                "expected_per_update": runtime_expected,
                "expected_total": runtime_expected * updates,
                "counts_match": runtime_counts_match,
                "per_update": per_update,
                "policy_parameters_finite": True,
            },
            "isolation": {
                "cuda_initialized_before": False,
                "cuda_initialized_after": cuda,
            },
        },
        "configured_steps": configured_expected,
        "other_parameters_preserved": all(
            kwargs.get(key) == value
            for key, value in configuration.items()
            if key != "num_learning_epochs"
        ),
    }


def _rows(source: str, configs: list[dict] = CONFIGS) -> list[dict]:
    return [_row(source, config) for config in configs]


def test_defective_baseline_and_general_patch_are_admitted() -> None:
    result = A.assess_candidate(
        original_source=BROKEN,
        proposal=_proposal(BROKEN),
        baseline_records=_rows(BROKEN),
        candidate_records=_rows(PATCH),
    )
    assert result["admitted"] is True
    assert result["reason"] == "candidate_conforms"
    assert result["promoted_source"] == PATCH
    assert result["promoted_source_sha256"] == CONTRACT.source_sha256(PATCH)
    assert result["model_proposal_success"] is True


def test_healthy_baseline_blocks_unnecessary_model_patch() -> None:
    result = A.assess_candidate(
        original_source=HEALTHY,
        proposal=_proposal(HEALTHY),
        baseline_records=_rows(HEALTHY),
        candidate_records=_rows(PATCH),
    )
    assert result["admitted"] is False
    assert result["reason"] == "baseline_conforms"
    assert result["promoted_source"] is None
    assert result["model_proposal_success"] is False


def test_missing_default_key_cannot_be_inserted_by_candidate() -> None:
    configs = [{"num_mini_batches": 4, "learning_rate": 0.001}]
    result = A.assess_candidate(
        original_source=BROKEN,
        proposal=_proposal(BROKEN, DEFAULT_INSERTING_PATCH),
        baseline_records=_rows(BROKEN, configs),
        candidate_records=_rows(DEFAULT_INSERTING_PATCH, configs),
    )
    assert result["admitted"] is False
    assert result["reason"] == "candidate_failed_admission"
    assert result["candidate_reasons"] == ["configuration_not_preserved"]


def test_hold_never_creates_execution_authority() -> None:
    result = A.assess_candidate(
        original_source=HEALTHY,
        proposal=_proposal(HEALTHY, None, "hold"),
        baseline_records=_rows(HEALTHY),
        candidate_records=[],
    )
    assert result["admitted"] is False
    assert result["reason"] == "model_hold"
    assert result["promoted_source"] is None


def test_candidate_count_or_cuda_failure_is_not_admitted() -> None:
    bad_counts = [_row(PATCH, config, force_counts=1) for config in CONFIGS]
    result = A.assess_candidate(
        original_source=BROKEN,
        proposal=_proposal(BROKEN),
        baseline_records=_rows(BROKEN),
        candidate_records=bad_counts,
    )
    assert result["admitted"] is False
    assert "configured_counts_not_observed" in result["candidate_reasons"]

    cuda_rows = _rows(PATCH)
    cuda_rows[0] = _row(PATCH, CONFIGS[0], cuda=True)
    result = A.assess_candidate(
        original_source=BROKEN,
        proposal=_proposal(BROKEN),
        baseline_records=_rows(BROKEN),
        candidate_records=cuda_rows,
    )
    assert result["admitted"] is False
    assert "cuda_was_initialized" in result["candidate_reasons"]


def test_source_and_adapter_output_bindings_are_enforced() -> None:
    rows = _rows(BROKEN)
    rows[0]["source_sha256"] = "0" * 64
    with pytest.raises(A.AdmissionError, match="source SHA"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=rows,
            candidate_records=_rows(PATCH),
        )

    rows = _rows(BROKEN)
    rows[0]["algorithm_kwargs"]["num_learning_epochs"] = 9
    with pytest.raises(A.AdmissionError, match="adapter output"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=rows,
            candidate_records=_rows(PATCH),
        )


def test_measurement_input_comparison_is_type_and_float_exact() -> None:
    rows = _rows(PATCH)
    rows[0]["measurement"]["input"]["config"]["learning_rate"] = 0.0010000000000000002
    with pytest.raises(A.AdmissionError, match="exactly match"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=rows,
        )

    rows = _rows(PATCH)
    rows[0]["measurement"]["input"]["config"]["num_learning_epochs"] = 5.0
    with pytest.raises(A.AdmissionError, match="exactly match"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=rows,
        )


def test_configuration_sets_must_match_without_duplicates() -> None:
    with pytest.raises(A.AdmissionError, match="configuration sets differ"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=_rows(PATCH)[:2],
        )

    duplicate = _rows(PATCH)
    duplicate[1] = copy.deepcopy(duplicate[0])
    with pytest.raises(A.AdmissionError, match="duplicates"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=duplicate,
        )


def test_candidate_variant_and_configured_step_binding_are_enforced() -> None:
    rows = _rows(PATCH)
    rows[0]["variant"] = "different"
    with pytest.raises(A.AdmissionError, match="variants differ"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=rows,
        )

    rows = _rows(PATCH)
    rows[0]["configured_steps"] = True
    with pytest.raises(A.AdmissionError, match="positive integer"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=rows,
        )


def test_missing_measurement_metric_is_rejected_not_inferred() -> None:
    rows = _rows(PATCH)
    del rows[0]["measurement"]["measurement"]["policy_parameters_finite"]
    with pytest.raises(A.AdmissionError, match="finiteness"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=_proposal(BROKEN),
            baseline_records=_rows(BROKEN),
            candidate_records=rows,
        )


def test_proposal_and_original_source_are_bound() -> None:
    proposal = _proposal(BROKEN)
    proposal["source_sha256"] = "f" * 64
    with pytest.raises(A.AdmissionError, match="original source"):
        A.assess_candidate(
            original_source=BROKEN,
            proposal=proposal,
            baseline_records=_rows(BROKEN),
            candidate_records=_rows(PATCH),
        )
