"""Host admission gate for model-generated PPO adapter patches.

The gate binds immutable source and CPU measurement records.  It grants no API,
GPU, approval, or execution authority.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Final

import ppo_patch_contract_v1 as CONTRACT

SCHEMA: Final = "ppo_generated_patch_admission_v1"
DEFAULT_EPOCHS: Final = 1


class AdmissionError(ValueError):
    """Raised when source bindings or required measurement evidence are invalid."""


@dataclass(frozen=True)
class _Assessment:
    variant: str
    configuration_key: tuple
    conforms: bool
    reason: str


def _plain_dict(value: object, label: str) -> dict:
    if type(value) is not dict:
        raise AdmissionError(f"{label} must be a plain dictionary")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AdmissionError(f"{label} must be a positive integer")
    return value


def _exact_key(value: object) -> tuple:
    """Build a type-sensitive, deterministic key for JSON-like records."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdmissionError("record contains a non-finite float")
        return ("float", struct.pack("!d", value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return ("list", tuple(_exact_key(item) for item in value))
    if type(value) is dict:
        if any(not isinstance(key, str) for key in value):
            raise AdmissionError("record dictionary keys must be strings")
        return (
            "dict",
            tuple(sorted((key, _exact_key(item)) for key, item in value.items())),
        )
    raise AdmissionError(f"record contains unsupported value type: {type(value).__name__}")


def _exact_equal(left: object, right: object) -> bool:
    return _exact_key(left) == _exact_key(right)


def _required(mapping: dict, key: str, label: str) -> object:
    if key not in mapping:
        raise AdmissionError(f"{label}.{key} is required")
    return mapping[key]


def _expected_counts(configuration: dict, record: dict) -> tuple[int, bool]:
    mini_batches = _positive_int(
        _required(configuration, "num_mini_batches", "configuration"),
        "configuration.num_mini_batches",
    )
    resolved = _plain_dict(_required(record, "resolved", "measurement"), "measurement.resolved")
    installed_default = _positive_int(
        _required(resolved, "installed_ppo_signature_default", "measurement.resolved"),
        "measurement.resolved.installed_ppo_signature_default",
    )
    if installed_default != DEFAULT_EPOCHS:
        raise AdmissionError("installed PPO signature default differs from the registered default")

    if "num_learning_epochs" in configuration:
        epochs = _positive_int(
            configuration["num_learning_epochs"],
            "configuration.num_learning_epochs",
        )
        expected_source = "config"
    else:
        epochs = installed_default
        expected_source = "installed_ppo_signature_default"

    resolved_epochs = _positive_int(
        _required(resolved, "num_learning_epochs", "measurement.resolved"),
        "measurement.resolved.num_learning_epochs",
    )
    resolved_minibatches = _positive_int(
        _required(resolved, "num_mini_batches", "measurement.resolved"),
        "measurement.resolved.num_mini_batches",
    )
    resolved_source = _required(
        resolved, "num_learning_epochs_source", "measurement.resolved"
    )
    if resolved_source not in {"config", "installed_ppo_signature_default"}:
        raise AdmissionError("measurement.resolved epoch source is invalid")
    resolved_matches = (
        resolved_epochs == epochs
        and resolved_minibatches == mini_batches
        and resolved_source == expected_source
    )
    return epochs * mini_batches, resolved_matches


def _assess_row(*, row: dict, source: str, role: str) -> _Assessment:
    row = _plain_dict(row, f"{role} row")
    variant = _required(row, "variant", f"{role} row")
    if not isinstance(variant, str) or not variant:
        raise AdmissionError(f"{role} row.variant must be a non-empty string")
    configuration = _plain_dict(
        _required(row, "configuration", f"{role} row"),
        f"{role} row.configuration",
    )
    algorithm_kwargs = _plain_dict(
        _required(row, "algorithm_kwargs", f"{role} row"),
        f"{role} row.algorithm_kwargs",
    )
    if row.get("source_sha256") != CONTRACT.source_sha256(source):
        raise AdmissionError(f"{role} row source SHA does not match bound source")

    try:
        expected_kwargs = CONTRACT.apply_adapter(source, dict(configuration))
    except CONTRACT.PatchContractError as exc:
        raise AdmissionError(f"{role} source does not satisfy the adapter contract") from exc
    if not _exact_equal(algorithm_kwargs, expected_kwargs):
        raise AdmissionError(f"{role} algorithm kwargs do not match adapter output")

    record = _plain_dict(
        _required(row, "measurement", f"{role} row"),
        f"{role} row.measurement",
    )
    input_record = _plain_dict(
        _required(record, "input", "measurement"), "measurement.input"
    )
    measured_config = _plain_dict(
        _required(input_record, "config", "measurement.input"),
        "measurement.input.config",
    )
    if not _exact_equal(measured_config, algorithm_kwargs):
        raise AdmissionError("measurement input config does not exactly match algorithm kwargs")

    expected_per_update, resolved_matches = _expected_counts(configuration, record)
    measured = _plain_dict(
        _required(record, "measurement", "measurement"),
        "measurement.measurement",
    )
    updates = _positive_int(
        _required(measured, "updates", "measurement.measurement"),
        "measurement.measurement.updates",
    )
    per_update = _required(measured, "per_update", "measurement.measurement")
    if not isinstance(per_update, list) or len(per_update) != updates:
        raise AdmissionError("per-update evidence must cover every measured update")
    expected_total = updates * expected_per_update

    counts_match = resolved_matches
    returns_finite = True
    for expected_index, item in enumerate(per_update, start=1):
        item = _plain_dict(item, "measurement.measurement.per_update item")
        if item.get("index") != expected_index:
            raise AdmissionError("per-update indexes must be contiguous and one-based")
        if not isinstance(item.get("minibatches"), int) or isinstance(
            item.get("minibatches"), bool
        ):
            raise AdmissionError("per-update minibatches must be an integer")
        if not isinstance(item.get("optimizer_steps"), int) or isinstance(
            item.get("optimizer_steps"), bool
        ):
            raise AdmissionError("per-update optimizer steps must be an integer")
        if type(item.get("return_values_finite")) is not bool:
            raise AdmissionError("per-update return finiteness is required")
        counts_match = counts_match and item["minibatches"] == expected_per_update
        counts_match = counts_match and item["optimizer_steps"] == expected_per_update
        returns_finite = returns_finite and item["return_values_finite"]

    for key, expected in {
        "minibatches": expected_total,
        "optimizer_steps": expected_total,
        "expected_per_update": expected_per_update,
        "expected_total": expected_total,
    }.items():
        if not isinstance(measured.get(key), int) or isinstance(measured.get(key), bool):
            raise AdmissionError(f"measurement.measurement.{key} must be an integer")
        counts_match = counts_match and measured[key] == expected
    if type(measured.get("counts_match")) is not bool:
        raise AdmissionError("aggregate count verdict is required")
    counts_match = counts_match and measured["counts_match"]
    if type(measured.get("policy_parameters_finite")) is not bool:
        raise AdmissionError("policy parameter finiteness is required")
    parameters_finite = measured["policy_parameters_finite"]

    isolation = _plain_dict(
        _required(record, "isolation", "measurement"), "measurement.isolation"
    )
    for key in ("cuda_initialized_before", "cuda_initialized_after"):
        if type(isolation.get(key)) is not bool:
            raise AdmissionError(f"measurement.isolation.{key} is required")
    cuda_free = not isolation["cuda_initialized_before"] and not isolation[
        "cuda_initialized_after"
    ]

    configured_steps = _positive_int(
        _required(row, "configured_steps", f"{role} row"),
        f"{role} row.configured_steps",
    )
    configured_steps_match = configured_steps == expected_per_update
    if type(row.get("other_parameters_preserved")) is not bool:
        raise AdmissionError(f"{role} row.other_parameters_preserved must be a boolean")
    status = record.get("status")
    if status not in {"passed", "failed"}:
        raise AdmissionError("measurement status must be passed or failed")

    forwarded_all = _exact_equal(algorithm_kwargs, configuration)
    conforms = (
        forwarded_all
        and configured_steps_match
        and row["other_parameters_preserved"]
        and status == "passed"
        and counts_match
        and returns_finite
        and parameters_finite
        and cuda_free
    )
    if not forwarded_all:
        reason = "configuration_not_preserved"
    elif not counts_match or not configured_steps_match:
        reason = "configured_counts_not_observed"
    elif not returns_finite or not parameters_finite:
        reason = "non_finite_measurement"
    elif not cuda_free:
        reason = "cuda_was_initialized"
    elif not row["other_parameters_preserved"]:
        reason = "other_parameters_not_preserved"
    elif status != "passed":
        reason = "measurement_failed"
    else:
        reason = "conforms"
    return _Assessment(variant, _exact_key(configuration), conforms, reason)


def _proposal_source(original_source: str, proposal: dict) -> str | None:
    proposal = _plain_dict(proposal, "proposal")
    if set(proposal) != CONTRACT.PROPOSAL_KEYS:
        raise AdmissionError("proposal keys do not match the patch contract")
    if proposal.get("source_sha256") != CONTRACT.source_sha256(original_source):
        raise AdmissionError("proposal is not bound to original source")
    evidence_ids = proposal.get("evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or not item for item in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise AdmissionError("proposal evidence IDs are invalid")
    rationale = proposal.get("rationale")
    if not isinstance(rationale, str) or not rationale or len(rationale) > 1_000:
        raise AdmissionError("proposal rationale is invalid")
    action = proposal.get("action")
    replacement = proposal.get("replacement_source")
    if action == "hold":
        if replacement is not None:
            raise AdmissionError("hold proposal must not include replacement source")
        return None
    if action != "patch" or not isinstance(replacement, str):
        raise AdmissionError("patch proposal must include replacement source")
    if CONTRACT.source_sha256(replacement) == CONTRACT.source_sha256(original_source):
        raise AdmissionError("patch proposal does not change source")
    try:
        CONTRACT.validate_source(replacement)
    except CONTRACT.PatchContractError as exc:
        raise AdmissionError("replacement source violates the patch contract") from exc
    return replacement


def assess_candidate(
    *,
    original_source: str,
    proposal: dict,
    baseline_records: list[dict],
    candidate_records: list[dict],
) -> dict:
    """Assess a patch using source-bound CPU records without authorizing execution."""

    CONTRACT.validate_source(original_source)
    candidate_source = _proposal_source(original_source, proposal)
    if not isinstance(baseline_records, list) or not baseline_records:
        raise AdmissionError("baseline records must be a non-empty list")
    if candidate_source is None:
        if candidate_records:
            raise AdmissionError("hold proposal must not have candidate records")
        return {
            "schema": SCHEMA,
            "admitted": False,
            "reason": "model_hold",
            "promoted_source": None,
            "promoted_source_sha256": None,
            "baseline_conforms": None,
            "candidate_conforms": None,
            "model_proposal_success": False,
        }
    if not isinstance(candidate_records, list) or not candidate_records:
        raise AdmissionError("patch proposal requires candidate records")

    baseline = [
        _assess_row(row=row, source=original_source, role="baseline")
        for row in baseline_records
    ]
    candidate = [
        _assess_row(row=row, source=candidate_source, role="candidate")
        for row in candidate_records
    ]
    baseline_keys = [item.configuration_key for item in baseline]
    candidate_keys = [item.configuration_key for item in candidate]
    baseline_variants = {item.variant for item in baseline}
    candidate_variants = {item.variant for item in candidate}
    if len(baseline_variants) != 1 or candidate_variants != baseline_variants:
        raise AdmissionError("baseline and candidate variants differ")
    if len(set(baseline_keys)) != len(baseline_keys):
        raise AdmissionError("baseline configuration set contains duplicates")
    if len(set(candidate_keys)) != len(candidate_keys):
        raise AdmissionError("candidate configuration set contains duplicates")
    if set(baseline_keys) != set(candidate_keys):
        raise AdmissionError("baseline and candidate configuration sets differ")

    baseline_conforms = all(item.conforms for item in baseline)
    candidate_conforms = all(item.conforms for item in candidate)
    if baseline_conforms:
        reason = "baseline_conforms"
        admitted = False
    elif candidate_conforms:
        reason = "candidate_conforms"
        admitted = True
    else:
        reason = "candidate_failed_admission"
        admitted = False
    return {
        "schema": SCHEMA,
        "admitted": admitted,
        "reason": reason,
        "promoted_source": candidate_source if admitted else None,
        "promoted_source_sha256": (
            CONTRACT.source_sha256(candidate_source) if admitted else None
        ),
        "baseline_conforms": baseline_conforms,
        "candidate_conforms": candidate_conforms,
        "baseline_reasons": [item.reason for item in baseline],
        "candidate_reasons": [item.reason for item in candidate],
        "model_proposal_success": admitted,
    }

