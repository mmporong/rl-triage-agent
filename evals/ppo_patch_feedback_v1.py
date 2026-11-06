"""Validate PPO patch candidates on CPU and permit one evidence-bound revision."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

from evals import ppo_patch_contract_v1 as CONTRACT

SCHEMA = "ppo_patch_cpu_validation_v1"
FEEDBACK_SCHEMA = "ppo_patch_feedback_context_v1"
TRANSITION_SCHEMA = "ppo_patch_feedback_transition_v1"


class PatchFeedbackError(ValueError):
    """Raised when CPU evidence or a one-revision transition is invalid."""


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PatchFeedbackError("value is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def validation_plan() -> list[dict[str, Any]]:
    """Return the fixed configurations used to expose hard-coded PPO values."""

    return [
        {
            "num_learning_epochs": 4,
            "num_mini_batches": 3,
            "learning_rate": 0.001,
            "schedule": "fixed",
        },
        {
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "learning_rate": 0.001,
            "schedule": "fixed",
        },
        {
            "num_learning_epochs": 7,
            "num_mini_batches": 2,
            "learning_rate": 0.001,
            "schedule": "fixed",
            "gamma": 0.97,
            "clip_param": 0.17,
        },
        {
            "num_mini_batches": 4,
            "learning_rate": 0.001,
            "schedule": "fixed",
        },
    ]


def _default_inspector(config: dict[str, Any], *, updates: int) -> dict[str, Any]:
    from bench.inspect_ppo_patch_cpu_v1 import inspect as inspect_cpu

    return inspect_cpu(config, updates=updates)


def _check_measurement(
    result: object,
    *,
    configuration: dict[str, Any],
    algorithm_kwargs: dict[str, Any],
    updates: int,
) -> list[str]:
    failed: list[str] = []
    if not isinstance(result, dict):
        return ["inspection.result_object"]

    runtime_input = result.get("input")
    resolved = result.get("resolved")
    fixture = result.get("fixture")
    measurement = result.get("measurement")
    isolation = result.get("isolation")
    if result.get("status") != "passed":
        failed.append("inspection.status")
    if not isinstance(runtime_input, dict) or (
        runtime_input.get("config") != algorithm_kwargs
        or runtime_input.get("updates") != updates
    ):
        failed.append("inspection.input_binding")

    if not isinstance(resolved, dict):
        failed.extend(
            [
                "inspection.resolved_epochs",
                "inspection.resolved_batches",
                "inspection.epoch_source",
            ]
        )
        expected_epochs = None
        expected_batches = configuration.get("num_mini_batches")
    else:
        default_epochs = resolved.get("installed_ppo_signature_default")
        expected_epochs = configuration.get("num_learning_epochs", default_epochs)
        expected_batches = configuration.get("num_mini_batches")
        if resolved.get("num_learning_epochs") != expected_epochs:
            failed.append("inspection.resolved_epochs")
        if resolved.get("num_mini_batches") != expected_batches:
            failed.append("inspection.resolved_batches")
        expected_source = (
            "config"
            if "num_learning_epochs" in configuration
            else "installed_ppo_signature_default"
        )
        if resolved.get("num_learning_epochs_source") != expected_source:
            failed.append("inspection.epoch_source")
        if "num_learning_epochs" not in configuration and (
            isinstance(default_epochs, bool)
            or not isinstance(default_epochs, int)
            or default_epochs < 1
        ):
            failed.append("inspection.installed_epoch_default")

    if (
        isinstance(expected_epochs, bool)
        or not isinstance(expected_epochs, int)
        or isinstance(expected_batches, bool)
        or not isinstance(expected_batches, int)
    ):
        expected_per_update = None
    else:
        expected_per_update = expected_epochs * expected_batches
    expected_total = (
        expected_per_update * updates if expected_per_update is not None else None
    )

    if not isinstance(measurement, dict):
        failed.extend(
            [
                "inspection.counts",
                "inspection.per_update",
                "inspection.policy_changed",
                "inspection.finite",
            ]
        )
    else:
        if (
            measurement.get("updates") != updates
            or measurement.get("expected_per_update") != expected_per_update
            or measurement.get("expected_total") != expected_total
            or measurement.get("minibatches") != expected_total
            or measurement.get("optimizer_steps") != expected_total
            or measurement.get("counts_match") is not True
        ):
            failed.append("inspection.counts")
        rows = measurement.get("per_update")
        if not isinstance(rows, list) or len(rows) != updates or any(
            not isinstance(row, dict)
            or row.get("index") != index
            or row.get("minibatches") != expected_per_update
            or row.get("optimizer_steps") != expected_per_update
            or row.get("return_values_finite") is not True
            for index, row in enumerate(rows, start=1)
        ):
            failed.append("inspection.per_update")
        if (
            measurement.get("policy_sha256_before")
            == measurement.get("policy_sha256_after")
            or not isinstance(measurement.get("changed_parameter_tensors"), int)
            or measurement.get("changed_parameter_tensors", 0) <= 0
        ):
            failed.append("inspection.policy_changed")
        if measurement.get("policy_parameters_finite") is not True:
            failed.append("inspection.finite")

    if not isinstance(isolation, dict) or (
        isolation.get("cuda_initialized_before") is not False
        or isolation.get("cuda_initialized_after") is not False
    ):
        failed.append("inspection.cuda_isolation")
    if not isinstance(fixture, dict) or fixture.get("device") != "cpu":
        failed.append("inspection.cpu_device")
    if not isinstance(isolation, dict) or (
        isolation.get("simulation_modules_before") != []
        or isolation.get("simulation_modules_after") != []
    ):
        failed.append("inspection.simulation_isolation")
    return sorted(set(failed))


def _validate_envelope(validation: object) -> dict[str, Any]:
    if not isinstance(validation, dict) or validation.get("schema") != SCHEMA:
        raise PatchFeedbackError("validation schema is invalid")
    if not isinstance(validation.get("source_sha256"), str):
        raise PatchFeedbackError("validation source SHA is missing")
    records = validation.get("records")
    if not isinstance(records, list) or not records:
        raise PatchFeedbackError("validation records are missing")
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("passed"), bool)
        or not isinstance(record.get("failed_checks"), list)
        for record in records
    ):
        raise PatchFeedbackError("validation record shape is invalid")
    expected = all(record["passed"] for record in records)
    if validation.get("all_passed") is not expected:
        raise PatchFeedbackError("validation summary does not match records")
    return validation


def validate_on_cpu(
    source: str,
    *,
    configurations: list[dict[str, Any]] | None = None,
    updates: int = 3,
    inspector: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply an adapter and validate its measured PPO work without GPU use."""

    CONTRACT.validate_source(source)
    if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= 3:
        raise PatchFeedbackError("updates must be an integer in [1, 3]")
    plan = validation_plan() if configurations is None else copy.deepcopy(configurations)
    if not isinstance(plan, list) or not plan:
        raise PatchFeedbackError("configurations must be a non-empty list")
    if any(type(item) is not dict for item in plan):
        raise PatchFeedbackError("each configuration must be a plain dictionary")
    inspect_cpu = _default_inspector if inspector is None else inspector
    if not callable(inspect_cpu):
        raise PatchFeedbackError("inspector must be callable")

    records: list[dict[str, Any]] = []
    for configuration in plan:
        original = copy.deepcopy(configuration)
        record: dict[str, Any] = {
            "configuration": original,
            "algorithm_kwargs": None,
            "passed": False,
            "failed_checks": [],
        }
        try:
            algorithm_kwargs = CONTRACT.apply_adapter(source, configuration)
            record["algorithm_kwargs"] = copy.deepcopy(algorithm_kwargs)
            failures: list[str] = []
            if configuration != original:
                failures.append("adapter.input_unchanged")
            if algorithm_kwargs != original:
                failures.append("adapter.output_equals_configuration")
            measurement = inspect_cpu(algorithm_kwargs, updates=updates)
            record["measurement"] = copy.deepcopy(measurement)
            failures.extend(
                _check_measurement(
                    measurement,
                    configuration=original,
                    algorithm_kwargs=algorithm_kwargs,
                    updates=updates,
                )
            )
            record["failed_checks"] = sorted(set(failures))
            record["passed"] = not record["failed_checks"]
        except Exception as exc:  # noqa: BLE001 - inspector containment boundary
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            record["failed_checks"] = ["inspection.exception"]
        records.append(record)

    return {
        "schema": SCHEMA,
        "source_sha256": CONTRACT.source_sha256(source),
        "updates": updates,
        "records": records,
        "all_passed": all(record["passed"] for record in records),
    }


def verify_validation(
    source: str,
    validation: dict[str, Any],
    *,
    configurations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute every stored verdict from its immutable CPU measurement."""

    CONTRACT.validate_source(source)
    checked = _validate_envelope(validation)
    if checked["source_sha256"] != CONTRACT.source_sha256(source):
        raise PatchFeedbackError("validation source SHA does not match source")
    if checked.get("updates") != 3:
        raise PatchFeedbackError("stored validation must contain three updates")
    plan = validation_plan() if configurations is None else copy.deepcopy(configurations)
    if not isinstance(plan, list) or len(plan) != 4 or any(
        type(item) is not dict for item in plan
    ):
        raise PatchFeedbackError("stored validation requires exactly four configurations")
    records = checked["records"]
    if len(records) != len(plan):
        raise PatchFeedbackError("validation record count does not match configurations")

    derived_passes: list[bool] = []
    for index, (configuration, record) in enumerate(zip(plan, records, strict=True)):
        if record.get("configuration") != configuration:
            raise PatchFeedbackError(f"validation record {index} configuration changed")
        algorithm_kwargs = CONTRACT.apply_adapter(source, copy.deepcopy(configuration))
        if record.get("algorithm_kwargs") != algorithm_kwargs:
            raise PatchFeedbackError(f"validation record {index} adapter output changed")

        if "error" in record:
            if set(record) != {
                "configuration",
                "algorithm_kwargs",
                "error",
                "passed",
                "failed_checks",
            }:
                raise PatchFeedbackError(f"validation error record {index} shape changed")
            error = record["error"]
            if (
                not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)
                or record["failed_checks"] != ["inspection.exception"]
                or record["passed"] is not False
            ):
                raise PatchFeedbackError(f"validation error record {index} is invalid")
            derived_passes.append(False)
            continue

        if set(record) != {
            "configuration",
            "algorithm_kwargs",
            "measurement",
            "passed",
            "failed_checks",
        }:
            raise PatchFeedbackError(f"validation measurement record {index} shape changed")
        measurement = record["measurement"]
        failures: list[str] = []
        if algorithm_kwargs != configuration:
            failures.append("adapter.output_equals_configuration")
        failures.extend(
            _check_measurement(
                measurement,
                configuration=configuration,
                algorithm_kwargs=algorithm_kwargs,
                updates=3,
            )
        )
        derived_checks = sorted(set(failures))
        derived_passed = not derived_checks
        if record["failed_checks"] != derived_checks or record["passed"] is not derived_passed:
            raise PatchFeedbackError(f"validation record {index} verdict was not derived")
        derived_passes.append(derived_passed)

    derived_all_passed = all(derived_passes)
    if checked["all_passed"] is not derived_all_passed:
        raise PatchFeedbackError("validation all_passed was not derived")
    return copy.deepcopy(checked)


def _feedback_facts(validation: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {"feedback.validation_all_passed": False}
    for index, record in enumerate(validation["records"]):
        if record["passed"]:
            continue
        prefix = f"feedback.record{index}"
        facts[f"{prefix}.configuration"] = copy.deepcopy(record["configuration"])
        facts[f"{prefix}.failed_checks"] = copy.deepcopy(record["failed_checks"])
        if record.get("algorithm_kwargs") is not None:
            facts[f"{prefix}.algorithm_kwargs"] = copy.deepcopy(
                record["algorithm_kwargs"]
            )
        measurement = record.get("measurement")
        if isinstance(measurement, dict):
            facts[f"{prefix}.resolved"] = copy.deepcopy(measurement.get("resolved"))
            facts[f"{prefix}.measurement"] = copy.deepcopy(
                measurement.get("measurement")
            )
        if "error" in record:
            facts[f"{prefix}.error"] = copy.deepcopy(record["error"])
    return facts


def build_feedback_context(
    context: dict[str, Any], proposal: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    """Build the sole revision context from observed CPU validation failures."""

    CONTRACT.validate_proposal(proposal, context)
    replacement = proposal.get("replacement_source")
    if not isinstance(replacement, str):
        raise PatchFeedbackError("feedback requires a patch proposal")
    checked = verify_validation(replacement, validation)
    if proposal.get("action") != "patch":
        raise PatchFeedbackError("feedback requires a patch proposal")
    replacement_sha = CONTRACT.source_sha256(replacement)
    if checked["source_sha256"] != replacement_sha:
        raise PatchFeedbackError("validation is not bound to the proposed source")
    if checked["all_passed"]:
        raise PatchFeedbackError("passing validation must not request a revision")

    inherited_facts = context.get("facts")
    if type(inherited_facts) is not dict:
        raise PatchFeedbackError("context facts must be a dictionary")
    facts = copy.deepcopy(inherited_facts)
    facts.update(_feedback_facts(checked))
    revision: dict[str, Any] = {
        "schema": FEEDBACK_SCHEMA,
        "source": replacement,
        "source_sha256": replacement_sha,
        "config": copy.deepcopy(context.get("config")),
        "facts": facts,
        "revision_index": 1,
        "max_revisions": 1,
        "feedback": {
            "parent_context_sha256": _canonical_sha256(context),
            "parent_source_sha256": context["source_sha256"],
            "first_proposal_sha256": _canonical_sha256(proposal),
            "first_validation_sha256": _canonical_sha256(checked),
            "first_validation": copy.deepcopy(checked),
        },
    }
    if "observation_scope" in context:
        revision["observation_scope"] = copy.deepcopy(context["observation_scope"])
    return revision


def validate_transition(
    root_context: dict[str, Any],
    first_proposal: dict[str, Any],
    first_validation: dict[str, Any],
    revision_context: dict[str, Any],
    revision_proposal: dict[str, Any],
    revision_validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate one failed proposal, one evidence revision, and its final result."""

    CONTRACT.validate_proposal(first_proposal, root_context)
    expected_context = build_feedback_context(
        root_context, first_proposal, first_validation
    )
    if revision_context != expected_context:
        raise PatchFeedbackError("revision context does not match CPU feedback")
    if revision_context.get("revision_index") != 1 or revision_context.get(
        "max_revisions"
    ) != 1:
        raise PatchFeedbackError("only revision 1 of 1 is allowed")
    CONTRACT.validate_proposal(revision_proposal, revision_context)

    action = revision_proposal["action"]
    if action == "patch":
        final_source = revision_proposal["replacement_source"]
        assert isinstance(final_source, str)
    else:
        final_source = revision_context["source"]
    final_source_sha = CONTRACT.source_sha256(final_source)
    checked_revision = verify_validation(final_source, revision_validation)
    if checked_revision["source_sha256"] != final_source_sha:
        raise PatchFeedbackError("revision validation is not bound to final source")

    first_source = first_proposal["replacement_source"]
    assert isinstance(first_source, str)
    first_checked = verify_validation(first_source, first_validation)
    return {
        "schema": TRANSITION_SCHEMA,
        "status": (
            "passed_after_feedback"
            if checked_revision["all_passed"]
            else "failed_after_feedback"
        ),
        "revision_index": 1,
        "max_revisions": 1,
        "first_validation_passed": first_checked["all_passed"],
        "after_feedback_validation_passed": checked_revision["all_passed"],
        "root_context_sha256": _canonical_sha256(root_context),
        "first_proposal": copy.deepcopy(first_proposal),
        "first_validation": copy.deepcopy(first_checked),
        "revision_context_sha256": _canonical_sha256(revision_context),
        "revision_proposal": copy.deepcopy(revision_proposal),
        "revision_validation": copy.deepcopy(checked_revision),
        "final_source": final_source,
        "final_source_sha256": final_source_sha,
        "approval_granted": False,
        "execution_authorized": False,
    }
