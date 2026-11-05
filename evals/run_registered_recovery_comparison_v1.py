"""Resume-safe controller for preregistered diagnostic-to-recovery comparison arms."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

_ENTRY_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ENTRY_ROOT), *(str(_ENTRY_ROOT / name) for name in ("src", "bench", "evals"))]

from evals import registered_epoch_namespace_failure_v1 as namespace_failure
from evals import registered_epoch_shutdown_failure_v1 as epoch_failure
from evals import registered_recovery_resources_v1 as recovery_resources
from evals import registered_recovery_timing_v1 as recovery_timing
from evals.diagnostic_terminal_status_v1 import terminal_aware_status
from rl_triage import investigate, registered_diagnostic_backend
from rl_triage.correction_registry_v2 import REGISTRY, registry_digest
from rl_triage.investigation_session import (
    InvestigationSession,
    file_sha,
    reference,
    write_new,
)

PLAN_SCHEMA = "registered_recovery_comparison_plan_v1"
PLAN_SCHEMA_V2 = "registered_recovery_comparison_plan_v2"
PLAN_SCHEMA_V3 = "registered_recovery_comparison_plan_v3"
PLAN_SCHEMA_V4 = "registered_recovery_comparison_plan_v4"
PLAN_SCHEMA_V5 = "registered_recovery_comparison_plan_v5"
PLAN_SCHEMA_V6 = "registered_recovery_comparison_plan_v6"
PLAN_SCHEMA_V7 = "registered_recovery_comparison_plan_v7"
PLAN_SCHEMA_V8 = "registered_recovery_comparison_plan_v8"
PLAN_SCHEMA_V9 = "registered_recovery_comparison_plan_v9"
PLAN_SCHEMA_V10 = "registered_recovery_comparison_plan_v10"
PLAN_SCHEMA_V11 = "registered_recovery_comparison_plan_v11"
CONSISTENT_OBSERVATION_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261104_v9.md"
CONSISTENT_OBSERVATION_SOURCES = {
    "evals/registered_desktop_gpu_retry_v1.py", "tests/test_registered_desktop_gpu_retry_v1.py",
    "bench/protocols/registered_desktop_gpu_retry_v1.json", CONSISTENT_OBSERVATION_NOTE,
}
BALANCED_ADMISSION_SOURCES = {
    "evals/registered_desktop_gpu_admission_v1.py",
    "tests/test_registered_desktop_gpu_admission_v1.py",
    "bench/protocols/registered_desktop_gpu_admission_v1.json",
    "evals/registered_desktop_gpu_shared_v1.py",
    "tests/test_registered_desktop_gpu_shared_v1.py",
    "bench/protocols/registered_desktop_gpu_shared_v1.json",
}
COMPLETED_EFFECT_REGISTRY_RELATIVE = (
    "bench/protocols/registered_completed_motor_effects_v1.json"
)
COMPLETED_EFFECT_REGISTRY_SHA256_LF = (
    "1c8a2fdd9e2e44f5af4c22f7b671e36f1604d1c268aefc885079ae3c7d3cc4a8"
)
V10_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261104_v10.md"
V12_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v12.md"
V13_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v13.md"
V14_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v14.md"
V15_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v15.md"
V16_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v16.md"
V17_NOTE = "bench/protocols/diagnostic_recovery_comparison_20261105_v17.md"
V16_PREFIX = "bench/protocols/registered_recovery_v16_prefix_20261105_v1.json"
V13_PUBLIC_SUMMARY = (
    "evals/results/recovery_comparison_s2664_s2667_20261105_v13/summary.json"
)
V13_PUBLIC_SUMMARY_SHA256 = (
    "1b28d0e0b98788508d65bbd1edfe3d8763ac5c6d5905715512ff403106f6c011"
)
RESOURCE_REMEASUREMENT_PROFILE = "balanced_resource_remeasurement_v1"
RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE = "balanced_resource_remeasurement_namespaced_v1"
RESOURCE_MANIFEST_RETRY_PROFILE = "balanced_resource_manifest_retry_v1"
RESOURCE_CONTINUATION_PROFILE = "balanced_resource_continuation_v1"
V15_FAILURE_SUMMARY = (
    "evals/results/balanced_recovery_v15_20261105_manifest_failure_lf_v2/summary.json"
)
V15_FAILURE_SUMMARY_SHA256 = (
    "60992cb88d95ab15ab65e275cd35e9f2157122cd1e43ff4f646a754df31412f0"
)
BALANCED_RESOURCE_SUMMARY_SOURCES = {
    "evals/summarize_balanced_recovery_resources_v1.py",
    "tests/test_summarize_balanced_recovery_resources_v1.py",
}
BALANCED_NAMESPACE_SOURCES = {
    "bench/registered_recovery_training_v1.py",
    "tests/test_registered_recovery_training_v1.py",
    V15_NOTE,
}
BALANCED_CONTINUATION_SOURCES = {
    "evals/registered_recovery_continuation_v5.py",
    "tests/test_registered_recovery_continuation_v5.py",
    "evals/registered_recovery_backend_v1.py",
    "tests/test_registered_recovery_backend_v1.py",
    "evals/registered_gpu_admission_journal_v1.py",
    "tests/test_registered_gpu_admission_journal_v1.py",
    "evals/summarize_balanced_recovery_continuation_v1.py",
    "tests/test_summarize_balanced_recovery_continuation_v1.py",
    V17_NOTE, V16_PREFIX,
}
EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE = (
    epoch_failure.POLICY_RELATIVE
)
EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF = (
    epoch_failure.POLICY_SHA256_LF
)
EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256 = (
    epoch_failure.FAILURE_PROOF_SHA256
)
JOB_SCHEMA = "registered_recovery_comparison_job_v1"
RESULT_SCHEMA = "registered_recovery_comparison_result_v1"
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
SUPPORTED_CASES = ("case-05", "case-06", "case-07", "case-08")
BACKEND = investigate.REGISTERED_BACKEND
REVIEW_POLICY_RELATIVE = "bench/protocols/registered_recovery_review_host_policy_v1.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_HIDDEN = {"scoring", "original_faults", "remaining_faults", "hidden_label", "hidden_labels",
           "answer_key", "correct_any_family"}
_TERMINAL_WITHOUT_RECOVERY = {
    "host_no_intervention": "normal_or_host_gate",
    "hold_observation_unqualified": "hold",
    "none_supported": "unsupported",
    "unidentifiable": "hold",
    "hold_unknown": "hold",
    "hold_budget_exhausted": "hold",
    "hold_no_registered_probe": "hold",
    "model_hold": "hold",
}
V17_EFFECT_KEYS = tuple(
    (case_id, method)
    for case_id, methods in (
        ("case-05", METHODS),
        ("case-06", ("static_rule", "one_shot_llm", "iterative_agent")),
    )
    for method in methods
)
V17_TRAINING_KEY = ("case-06", "fixed_sequence")
_IMPORT_FIELDS = {
    "artifact_id", "kind", "case_id", "method", "proof_sha256",
    "prior_plan_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
    "seed", "task", "correction_id", "diagnostic_manifest_sha256", "source_anchor",
    "registration_anchor",
}


class ComparisonContractError(ValueError):
    """A preregistration, immutable artifact, or state transition is invalid."""


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _reject_hidden(value: object) -> None:
    if isinstance(value, dict):
        if _HIDDEN & {str(key).casefold() for key in value}:
            raise ComparisonContractError("hidden/scoring authority is forbidden")
        for child in value.values():
            _reject_hidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_hidden(child)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ComparisonContractError("JSON object required")
    canonical(value)
    _reject_hidden(value)
    return value


def _ref(value: object, label: str, *, require_now: bool = True) -> tuple[Path, dict | None]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ComparisonContractError(f"{label} reference fields differ")
    if not isinstance(value["sha256"], str) or _SHA.fullmatch(value["sha256"]) is None:
        raise ComparisonContractError(f"{label} reference SHA differs")
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink():
        raise ComparisonContractError(f"{label} reference path differs")
    if not require_now and not path.exists():
        return path.resolve(), None
    path = path.resolve(strict=True)
    if file_sha(path) != value["sha256"]:
        raise ComparisonContractError(f"{label} reference changed")
    return path, _read(path)


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ComparisonContractError(f"{label} must be POSIX relative")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ComparisonContractError(f"{label} escapes repo")
    return path


def _completed_effect_registry() -> dict:
    path = _ENTRY_ROOT / COMPLETED_EFFECT_REGISTRY_RELATIVE
    if _lf_sha(path) != COMPLETED_EFFECT_REGISTRY_SHA256_LF:
        raise ComparisonContractError("completed effect registry changed")
    registry = _read(path)
    registry_fields = {
        "schema", "profile_id", "completed_cases", "methods", "entries",
        "prior_consumed_gpu_sdk_launches", "expected_new_gpu_sdk_launches",
        "new_gpu_sdk_launch_cap", "cumulative_gpu_sdk_launch_cap",
        "new_recovery_arms", "v9_failed_epoch_sdk_launches", "outcome_selection",
    }
    entry_fields = {
        "artifact_id", "kind", "case_id", "method", "proof_sha256",
        "prior_plan_sha256", "diagnostic_manifest_sha256",
        "source_checkpoint_sha256", "completed_checkpoint_sha256", "seed", "task",
        "correction_id", "source_anchor", "registration_anchor",
    }
    if (
        set(registry) != registry_fields
        or registry.get("schema") != "registered_completed_effect_whitelist_v1"
        or registry.get("profile_id") != "all_completed_motor_effects_before_epoch_v1"
        or registry.get("completed_cases") != ["case-05", "case-06"]
        or registry.get("methods") != list(METHODS)
        or registry.get("prior_consumed_gpu_sdk_launches") != 17
        or registry.get("expected_new_gpu_sdk_launches") != 14
        or registry.get("new_gpu_sdk_launch_cap") != 15
        or registry.get("cumulative_gpu_sdk_launch_cap") != 32
        or registry.get("new_recovery_arms") != 7
        or registry.get("v9_failed_epoch_sdk_launches") != 0
        or registry.get("outcome_selection")
        != "all_completed_effects_of_first_two_cases_regardless_of_outcome"
        or not isinstance(registry.get("entries"), dict)
    ):
        raise ComparisonContractError("completed effect registry contract differs")
    entries = registry["entries"]
    matrix = {(case, method) for case in ("case-05", "case-06") for method in METHODS}
    if len(entries) != 8 or {
        (row.get("case_id"), row.get("method"))
        for row in entries.values() if isinstance(row, dict)
    } != matrix:
        raise ComparisonContractError("completed effect registry matrix differs")
    for artifact_id, row in entries.items():
        if (not isinstance(row, dict) or set(row) != entry_fields
                or row.get("artifact_id") != artifact_id):
            raise ComparisonContractError("completed effect registry artifact differs")
    return registry


def _epoch_shutdown_retry_policy() -> dict:
    try:
        return epoch_failure.load_policy(repo=_ENTRY_ROOT)
    except epoch_failure.ShutdownFailureError as exc:
        raise ComparisonContractError(str(exc)) from exc


def _validate_plan(plan: dict) -> None:
    fields = {"schema", "source_anchor", "source_closure", "methods", "cases", "limits", "inputs",
              "backend", "authorization", "recovery_policy", "template_commitments"}
    schema = plan.get("schema")
    if schema in {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                  PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V11}:
        fields.add("continuation")
    if schema in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10}:
        fields.add("resource_remeasurement")
    if set(plan) != fields or schema not in {PLAN_SCHEMA, PLAN_SCHEMA_V2, PLAN_SCHEMA_V3,
                                             PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                                             PLAN_SCHEMA_V6, PLAN_SCHEMA_V7,
                                             PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10,
                                             PLAN_SCHEMA_V11}:
        raise ComparisonContractError("plan fields differ")
    if _COMMIT.fullmatch(str(plan["source_anchor"])) is None:
        raise ComparisonContractError("source anchor differs")
    closure = plan["source_closure"]
    if not isinstance(closure, dict) or not closure:
        raise ComparisonContractError("source closure missing")
    for path, sha in closure.items():
        _relative(path, "source closure")
        if not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
            raise ComparisonContractError("source closure SHA differs")
    if plan["methods"] != list(METHODS) or plan["backend"] != BACKEND:
        raise ComparisonContractError("method/backend differs")
    cases = plan["cases"]
    case_ids = {row.get("case_id") for row in cases if isinstance(row, dict)} if isinstance(cases, list) else set()
    if not case_ids or not case_ids <= set(SUPPORTED_CASES) or len(case_ids) != len(cases):
        raise ComparisonContractError("explicit supported case list required")
    for row in cases:
        if (set(row) != {"case_id", "rotation"} or not isinstance(row["rotation"], list)
                or len(row["rotation"]) != 4 or set(row["rotation"]) != set(METHODS)):
            raise ComparisonContractError("four-arm case rotation differs")
    limits = plan["limits"]
    expected_sdk_cap = {
        PLAN_SCHEMA: len(case_ids) * 8,
        PLAN_SCHEMA_V2: len(case_ids) * 8 - 1,
        PLAN_SCHEMA_V3: len(case_ids) * 8 - 3,
        PLAN_SCHEMA_V4: len(case_ids) * 8 - 5,
        PLAN_SCHEMA_V5: 15,
        PLAN_SCHEMA_V6: 14,
        PLAN_SCHEMA_V7: 14,
        PLAN_SCHEMA_V8: 32,
        PLAN_SCHEMA_V9: 32,
        PLAN_SCHEMA_V10: 31,
        PLAN_SCHEMA_V11: 16,
    }[schema]
    expected_total_cap = {
        PLAN_SCHEMA: len(case_ids) * 16,
        PLAN_SCHEMA_V2: len(case_ids) * 16 - 1,
        PLAN_SCHEMA_V3: len(case_ids) * 16 - 3,
        PLAN_SCHEMA_V4: len(case_ids) * 16 - 5,
        PLAN_SCHEMA_V5: 29,
        PLAN_SCHEMA_V6: 28,
        PLAN_SCHEMA_V7: 28,
        PLAN_SCHEMA_V8: 64,
        PLAN_SCHEMA_V9: 64,
        PLAN_SCHEMA_V10: 63,
        PLAN_SCHEMA_V11: 32,
    }[schema]
    expected_cpu_cap = (
        7 if schema in {PLAN_SCHEMA_V5, PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}
        else 8 if schema == PLAN_SCHEMA_V11
        else len(case_ids) * 4
    )
    if limits != {"max_recoveries": len(case_ids) * 4,
                  "gpu_sdk_launch_cap": expected_sdk_cap,
                  "cpu_checkpoint_launch_cap": expected_cpu_cap,
                  "cpu_telemetry_launch_cap": expected_cpu_cap,
                  "total_owned_launch_cap": expected_total_cap,
                  "max_parent_wall_s_per_recovery": 5000, "api_cap": 0}:
        raise ComparisonContractError("recovery/GPU/API cap differs")
    if schema == PLAN_SCHEMA_V2:
        continuation = plan["continuation"]
        required = {
            "gpu_admission_profile", "prior_training", "prior_consumed_gpu_sdk_launches",
            "prior_training_gpu_s", "expected_new_gpu_sdk_launches", "continuation_tag",
            "prior_failed_attempt_preserved",
        }
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["gpu_admission_profile"] != "registered_desktop_shared_v2"
            or continuation["prior_consumed_gpu_sdk_launches"] != 1
            or continuation["prior_training_gpu_s"] != 515.3279999999795
            or continuation["expected_new_gpu_sdk_launches"] != 29
            or continuation["continuation_tag"] != "v5_fresh"
            or continuation["prior_failed_attempt_preserved"] is not True
            or case_ids != set(SUPPORTED_CASES)
        ):
            raise ComparisonContractError("continuation accounting/profile differs")
        prior = continuation["prior_training"]
        if (
            not isinstance(prior, dict)
            or set(prior) != {
                "logical_id", "proof_sha256", "source_checkpoint_sha256",
                "completed_checkpoint_sha256", "seed", "task", "correction_id",
                "diagnostic_manifest_sha256",
            }
            or prior["logical_id"] != "case-05/fixed_sequence/case_02/CORRECTED"
            or any(
                not isinstance(prior[key], str) or _SHA.fullmatch(prior[key]) is None
                for key in (
                    "proof_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
                    "diagnostic_manifest_sha256",
                )
            )
            or type(prior["seed"]) is not int
            or prior["correction_id"] != "restore_motor_torque_speed"
            or not isinstance(prior["task"], str)
            or not prior["task"]
        ):
            raise ComparisonContractError("prior training logical identity differs")
    if schema == PLAN_SCHEMA_V3:
        continuation = plan["continuation"]
        profile_tag = {
            "registered_desktop_shared_wait_v1": "v6_fresh",
            "registered_desktop_shared_baseline_v1": "v7_shared_desktop",
        }
        required = {
            "gpu_admission_profile", "stage_imports", "prior_consumed_gpu_sdk_launches",
            "expected_new_gpu_sdk_launches", "continuation_tag",
            "prior_v5_registration_anchor", "prior_incomplete_preserved",
        }
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["gpu_admission_profile"] not in profile_tag
            or continuation["prior_consumed_gpu_sdk_launches"] != 3
            or continuation["expected_new_gpu_sdk_launches"] != 27
            or continuation["continuation_tag"]
            != profile_tag.get(continuation["gpu_admission_profile"])
            or _COMMIT.fullmatch(str(continuation["prior_v5_registration_anchor"])) is None
            or continuation["prior_incomplete_preserved"] is not True
            or case_ids != set(SUPPORTED_CASES)
        ):
            raise ComparisonContractError("v3 continuation accounting/profile differs")
        imports = continuation["stage_imports"]
        if not isinstance(imports, list) or len(imports) != 2:
            raise ComparisonContractError("v3 stage import coverage differs")
        expected_imports = {
            ("case-05", "fixed_sequence"): ("completed_effect", "prior_effect"),
            ("case-05", "static_rule"): ("completed_training", "prior_training"),
        }
        seen_imports: set[tuple[str, str]] = set()
        import_fields = {
            "artifact_id", "kind", "case_id", "method", "proof_sha256",
            "prior_plan_sha256",
            "source_checkpoint_sha256", "completed_checkpoint_sha256", "seed", "task",
            "correction_id", "diagnostic_manifest_sha256", "source_anchor",
            "registration_anchor",
        }
        for row in imports:
            if not isinstance(row, dict):
                raise ComparisonContractError("v3 stage import identity differs")
            case_id, method = row.get("case_id"), row.get("method")
            if not isinstance(case_id, str) or not isinstance(method, str):
                raise ComparisonContractError("v3 stage import identity differs")
            key = (case_id, method)
            expected = expected_imports.get(key)
            if (
                set(row) != import_fields
                or expected is None
                or key in seen_imports
                or (row["kind"], row["artifact_id"]) != expected
                or any(
                    not isinstance(row[field], str) or _SHA.fullmatch(row[field]) is None
                    for field in (
                        "proof_sha256", "prior_plan_sha256", "source_checkpoint_sha256",
                        "completed_checkpoint_sha256", "diagnostic_manifest_sha256",
                    )
                )
                or _COMMIT.fullmatch(str(row["source_anchor"])) is None
                or _COMMIT.fullmatch(str(row["registration_anchor"])) is None
                or row["registration_anchor"] != continuation["prior_v5_registration_anchor"]
                or type(row["seed"]) is not int
                or row["correction_id"] != "restore_motor_torque_speed"
                or not isinstance(row["task"], str)
                or not row["task"]
            ):
                raise ComparisonContractError("v3 stage import identity differs")
            seen_imports.add(key)
        if seen_imports != set(expected_imports):
            raise ComparisonContractError("v3 stage import matrix differs")
    if schema == PLAN_SCHEMA_V4:
        continuation = plan["continuation"]
        profile_tags = {
            "registered_desktop_shared_baseline_v1": "v8_resumable",
            "registered_desktop_shared_retry_v1": "v9_consistent_observation",
        }
        required = {
            "gpu_admission_profile", "stage_imports", "prior_consumed_gpu_sdk_launches",
            "expected_new_gpu_sdk_launches", "cumulative_gpu_sdk_launches",
            "continuation_tag", "prior_v7_registration_anchor", "prior_incomplete_preserved",
        }
        if (not isinstance(continuation, dict) or set(continuation) != required
                or continuation["gpu_admission_profile"] not in profile_tags
                or continuation["prior_consumed_gpu_sdk_launches"] != 5
                or continuation["expected_new_gpu_sdk_launches"] != 26
                or continuation["cumulative_gpu_sdk_launches"] != 32
                or continuation["continuation_tag"] != profile_tags.get(continuation["gpu_admission_profile"])
                or _COMMIT.fullmatch(str(continuation["prior_v7_registration_anchor"])) is None
                or continuation["prior_incomplete_preserved"] is not True
                or case_ids != set(SUPPORTED_CASES)):
            raise ComparisonContractError("v4 continuation accounting/profile differs")
        imports = continuation["stage_imports"]
        expected_imports = {
            ("case-05", "fixed_sequence"): ("completed_effect", "prior_effect"),
            ("case-05", "static_rule"): ("completed_effect", "prior_static_effect"),
        }
        import_fields = {
            "artifact_id", "kind", "case_id", "method", "proof_sha256",
            "prior_plan_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
            "seed", "task", "correction_id", "diagnostic_manifest_sha256", "source_anchor",
            "registration_anchor",
        }
        if not isinstance(imports, list) or len(imports) != 2:
            raise ComparisonContractError("v4 stage import coverage differs")
        seen_imports: set[tuple[str, str]] = set()
        for row in imports:
            if not isinstance(row, dict):
                raise ComparisonContractError("v4 stage import identity differs")
            case_id, method = row.get("case_id"), row.get("method")
            if not isinstance(case_id, str) or not isinstance(method, str):
                raise ComparisonContractError("v4 stage import identity differs")
            key = (case_id, method)
            expected = expected_imports.get(key)
            if (set(row) != import_fields or expected is None or key in seen_imports
                    or (row["kind"], row["artifact_id"]) != expected
                    or any(not isinstance(row[field], str) or _SHA.fullmatch(row[field]) is None
                           for field in ("proof_sha256", "prior_plan_sha256",
                                         "source_checkpoint_sha256", "completed_checkpoint_sha256",
                                         "diagnostic_manifest_sha256"))
                    or _COMMIT.fullmatch(str(row["source_anchor"])) is None
                    or _COMMIT.fullmatch(str(row["registration_anchor"])) is None
                    or type(row["seed"]) is not int
                    or row["correction_id"] != "restore_motor_torque_speed"
                    or not isinstance(row["task"], str) or not row["task"]):
                raise ComparisonContractError("v4 stage import identity differs")
            seen_imports.add(key)
        if seen_imports != set(expected_imports):
            raise ComparisonContractError("v4 stage import matrix differs")
    if schema == PLAN_SCHEMA_V5:
        continuation = plan["continuation"]
        required = {
            "gpu_admission_profile", "completed_import_profile",
            "completed_effect_registry_sha256_lf", "stage_imports",
            "prior_consumed_gpu_sdk_launches", "expected_new_gpu_sdk_launches",
            "new_gpu_sdk_launch_cap", "cumulative_gpu_sdk_launches",
            "continuation_tag", "prior_v9_registration_anchor",
            "prior_incomplete_preserved", "v9_failed_epoch_sdk_launches",
        }
        registry = _completed_effect_registry()
        entries = registry["entries"]
        imports = continuation.get("stage_imports") if isinstance(continuation, dict) else None
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["gpu_admission_profile"] != "registered_desktop_shared_retry_v1"
            or continuation["completed_import_profile"] != "motor_completed_v1"
            or continuation["completed_effect_registry_sha256_lf"]
            != COMPLETED_EFFECT_REGISTRY_SHA256_LF
            or continuation["prior_consumed_gpu_sdk_launches"] != 17
            or continuation["expected_new_gpu_sdk_launches"] != 14
            or continuation["new_gpu_sdk_launch_cap"] != 15
            or continuation["cumulative_gpu_sdk_launches"] != 32
            or continuation["continuation_tag"] != "v10_epoch_repair"
            or continuation["prior_v9_registration_anchor"]
            != "fbd0bdd929e68933fdd2d8d354f7d5447625b101"
            or continuation["prior_incomplete_preserved"] is not True
            or continuation["v9_failed_epoch_sdk_launches"] != 0
            or case_ids != set(SUPPORTED_CASES)
            or not isinstance(imports, list)
            or len(imports) != 8
        ):
            raise ComparisonContractError("v5 continuation accounting/profile differs")
        by_artifact = {
            row.get("artifact_id"): row for row in imports if isinstance(row, dict)
        }
        if len(by_artifact) != 8 or by_artifact != entries:
            raise ComparisonContractError("v5 completed effect registry binding differs")
    if schema in {PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}:
        continuation = plan["continuation"]
        required = {
            "gpu_admission_profile", "completed_import_profile",
            "completed_effect_registry_sha256_lf",
            "epoch_shutdown_retry_policy_sha256_lf", "failure_proof_sha256",
            "stage_imports", "prior_consumed_gpu_sdk_launches",
            "expected_new_gpu_sdk_launches", "new_gpu_sdk_launch_cap",
            "cumulative_gpu_sdk_launches", "continuation_tag",
            "prior_v11_registration_anchor", "prior_incomplete_preserved",
            "v11_failed_epoch_sdk_launches",
        }
        if schema == PLAN_SCHEMA_V7:
            required.add("namespace_failure_context_sha256")
        registry = _completed_effect_registry()
        policy = _epoch_shutdown_retry_policy()
        entries = registry["entries"]
        imports = continuation.get("stage_imports") if isinstance(continuation, dict) else None
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["gpu_admission_profile"]
            != "registered_desktop_shared_retry_v1"
            or continuation["completed_import_profile"]
            != ("motor_completed_epoch_namespace_retry_v1" if schema == PLAN_SCHEMA_V7
                else "motor_completed_epoch_close_retry_v1")
            or continuation["completed_effect_registry_sha256_lf"]
            != COMPLETED_EFFECT_REGISTRY_SHA256_LF
            or continuation["epoch_shutdown_retry_policy_sha256_lf"]
            != EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF
            or continuation["failure_proof_sha256"]
            != EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256
            or continuation["prior_consumed_gpu_sdk_launches"] != 18
            or continuation["expected_new_gpu_sdk_launches"] != 14
            or continuation["new_gpu_sdk_launch_cap"] != 14
            or continuation["cumulative_gpu_sdk_launches"] != 32
            or continuation["continuation_tag"] != (
                "v13_epoch_namespace_retry" if schema == PLAN_SCHEMA_V7
                else "v12_epoch_close_retry")
            or (schema == PLAN_SCHEMA_V7 and continuation[
                "namespace_failure_context_sha256"] != namespace_failure.CONTEXT_SHA256)
            or continuation["prior_v11_registration_anchor"]
            != policy["v11_failure"]["identity"]["registration_anchor"]
            or continuation["prior_incomplete_preserved"] is not True
            or continuation["v11_failed_epoch_sdk_launches"] != 1
            or case_ids != set(SUPPORTED_CASES)
            or not isinstance(imports, list)
            or len(imports) != 8
        ):
            raise ComparisonContractError("v6 continuation accounting/profile differs")
        by_artifact = {
            row.get("artifact_id"): row for row in imports if isinstance(row, dict)
        }
        if len(by_artifact) != 8 or by_artifact != entries:
            raise ComparisonContractError("v6 completed effect registry binding differs")
    if schema == PLAN_SCHEMA_V11:
        continuation = plan["continuation"]
        required = {
            "profile_id", "gpu_admission_profile", "stage_imports",
            "prior_plan_sha256", "prior_registration_anchor", "prior_source_anchor",
            "prior_completed_effects", "prior_completed_training",
            "prior_consumed_gpu_sdk_launches", "expected_new_gpu_sdk_launches",
            "new_gpu_sdk_launch_cap", "prior_primary_sdk_launches",
            "prior_post_sdk_launches", "prior_v15_failed_sdk_launches",
            "prior_v16_partial_sdk_launches", "prior_whole_study_sdk_launches",
            "expected_whole_study_sdk_launches", "cumulative_gpu_sdk_launch_cap",
            "prior_incomplete_preserved", "no_model_requery",
        }
        imports = continuation.get("stage_imports") if isinstance(continuation, dict) else None
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["profile_id"] != RESOURCE_CONTINUATION_PROFILE
            or continuation["gpu_admission_profile"] != "registered_desktop_shared_retry_v1"
            or _SHA.fullmatch(str(continuation["prior_plan_sha256"])) is None
            or _COMMIT.fullmatch(str(continuation["prior_registration_anchor"])) is None
            or _COMMIT.fullmatch(str(continuation["prior_source_anchor"])) is None
            or continuation["prior_completed_effects"] != 7
            or continuation["prior_completed_training"] != 1
            or continuation["prior_consumed_gpu_sdk_launches"] != 15
            or continuation["expected_new_gpu_sdk_launches"] != 15
            or continuation["new_gpu_sdk_launch_cap"] != 16
            or continuation["prior_primary_sdk_launches"] != 32
            or continuation["prior_post_sdk_launches"] != 1
            or continuation["prior_v15_failed_sdk_launches"] != 1
            or continuation["prior_v16_partial_sdk_launches"] != 15
            or continuation["prior_whole_study_sdk_launches"] != 49
            or continuation["expected_whole_study_sdk_launches"] != 64
            or continuation["cumulative_gpu_sdk_launch_cap"] != 65
            or continuation["prior_incomplete_preserved"] is not True
            or continuation["no_model_requery"] is not True
            or case_ids != set(SUPPORTED_CASES)
            or not isinstance(imports, list)
            or len(imports) != 8
        ):
            raise ComparisonContractError("v11 continuation accounting/profile differs")
        expected_kinds = {key: "completed_effect" for key in V17_EFFECT_KEYS}
        expected_kinds[V17_TRAINING_KEY] = "completed_training"
        seen_imports: set[tuple[str, str]] = set()
        for row in imports:
            if not isinstance(row, dict):
                raise ComparisonContractError("v11 stage import identity differs")
            case_id = row.get("case_id")
            method = row.get("method")
            if not isinstance(case_id, str) or not isinstance(method, str):
                raise ComparisonContractError("v11 stage import identity differs")
            key = (case_id, method)
            expected_kind = expected_kinds.get(key)
            expected_artifact = (
                "prior_training" if key == V17_TRAINING_KEY
                else f"prior_effect_{str(key[0]).replace('-', '_')}_{key[1]}"
                if key in expected_kinds else None
            )
            if (
                set(row) != _IMPORT_FIELDS
                or expected_kind is None
                or key in seen_imports
                or row.get("kind") != expected_kind
                or row.get("artifact_id") != expected_artifact
                or row.get("prior_plan_sha256") != continuation["prior_plan_sha256"]
                or row.get("source_anchor") != continuation["prior_source_anchor"]
                or row.get("registration_anchor") != continuation["prior_registration_anchor"]
                or any(_SHA.fullmatch(str(row.get(field))) is None for field in (
                    "proof_sha256", "prior_plan_sha256", "diagnostic_manifest_sha256",
                    "source_checkpoint_sha256", "completed_checkpoint_sha256",
                ))
                or type(row.get("seed")) is not int
                or not isinstance(row.get("task"), str) or not row["task"]
                or row.get("correction_id") != "restore_motor_torque_speed"
            ):
                raise ComparisonContractError("v11 stage import identity differs")
            seen_imports.add(key)
        if seen_imports != set(expected_kinds):
            raise ComparisonContractError("v11 stage import matrix differs")
    if schema in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10}:
        remeasurement = plan["resource_remeasurement"]
        expected_remeasurement = {
            "profile_id": ({PLAN_SCHEMA_V8: RESOURCE_REMEASUREMENT_PROFILE,
                            PLAN_SCHEMA_V9: RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE,
                            PLAN_SCHEMA_V10: RESOURCE_MANIFEST_RETRY_PROFILE}[schema]),
            "scope": "already_seen_development_resource_check",
            "imports_allowed": False,
            "known_primary_outcomes": {
                "path": V13_PUBLIC_SUMMARY,
                "sha256": V13_PUBLIC_SUMMARY_SHA256,
            },
            "expected": {
                "planned_arms": 16,
                "fresh_effects": 15,
                "diagnostic_hold": 1,
            },
            "accounting": {
                "prior_primary_sdk_launches": 32,
                "prior_post_sdk_launches": 1,
                "prior_whole_study_sdk_launches": 33,
                "new_gpu_sdk_launch_cap": 32,
                "cumulative_gpu_sdk_launch_cap": 65,
            },
            "no_model_requery": True,
        }
        if schema == PLAN_SCHEMA_V10:
            expected_remeasurement["known_failed_remeasurement"] = {
                "path": V15_FAILURE_SUMMARY, "sha256": V15_FAILURE_SUMMARY_SHA256,
            }
            expected_remeasurement["accounting"] = {
                "prior_primary_sdk_launches": 32,
                "prior_post_sdk_launches": 1,
                "prior_failed_remeasurement_sdk_launches": 1,
                "prior_whole_study_sdk_launches": 34,
                "new_gpu_sdk_launch_cap": 31,
                "cumulative_gpu_sdk_launch_cap": 65,
            }
        if remeasurement != expected_remeasurement:
            raise ComparisonContractError("v8 resource remeasurement contract differs")
        if case_ids != set(SUPPORTED_CASES):
            raise ComparisonContractError("v8 complete case matrix required")
    policy = plan["recovery_policy"]
    if (not isinstance(policy, dict)
            or set(policy) != {"training_iterations", "seeds", "same_case_same_template", "source_cost_policy"}
            or policy["training_iterations"] != 300
            or set(policy["seeds"]) != case_ids
            or any(not isinstance(seeds, dict) or set(seeds) != {"resume", "restart"}
                   or any(type(seed) is not int or seed < 0 for seed in seeds.values())
                   for seeds in policy["seeds"].values())
            or policy["same_case_same_template"] is not True
            or policy["source_cost_policy"] != "phase_a_shared_count_once"):
        raise ComparisonContractError("recovery seed/template/source-cost policy differs")
    if schema in {PLAN_SCHEMA_V6, PLAN_SCHEMA_V7} and policy["seeds"] != {
        "case-05": {"resume": 2664, "restart": 5664},
        "case-06": {"resume": 2665, "restart": 5665},
        "case-07": {"resume": 2666, "restart": 6266 if schema == PLAN_SCHEMA_V7 else 6066},
        "case-08": {"resume": 2667, "restart": 6267 if schema == PLAN_SCHEMA_V7 else 6067},
    }:
        raise ComparisonContractError("v6 registered restart seed map differs")
    if schema in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9} and policy["seeds"] != {
        "case-05": {"resume": 2664, "restart": 5664},
        "case-06": {"resume": 2665, "restart": 5665},
        "case-07": {"resume": 2666, "restart": 6466},
        "case-08": {"resume": 2667, "restart": 6467},
    }:
        raise ComparisonContractError("v8 registered restart seed map differs")
    if schema in {PLAN_SCHEMA_V10, PLAN_SCHEMA_V11} and policy["seeds"] != {
        "case-05": {"resume": 2664, "restart": 5664},
        "case-06": {"resume": 2665, "restart": 5665},
        "case-07": {"resume": 2666, "restart": 6666},
        "case-08": {"resume": 2667, "restart": 6667},
    }:
        raise ComparisonContractError("v10 registered restart seed map differs")
    commitments = plan["template_commitments"]
    mechanisms = {"actuator", "optimizer_update_count", "optimizer_surrogate"}
    if not isinstance(commitments, dict) or set(commitments) != case_ids:
        raise ComparisonContractError("template commitment cases differ")
    for case_rows in commitments.values():
        if not isinstance(case_rows, dict) or set(case_rows) != mechanisms:
            raise ComparisonContractError("template commitment mechanisms differ")
        for row in case_rows.values():
            if (not isinstance(row, dict)
                    or set(row) != {"training_binding_sha256", "fixed_binding_sha256",
                                    "source_checkpoint_sha256"}
                    or any(not isinstance(value, str) or _SHA.fullmatch(value) is None
                           for value in row.values())):
                raise ComparisonContractError("template commitment SHA differs")
    authorization = plan["authorization"]
    if (not isinstance(authorization, dict)
            or set(authorization) != {"actor", "ambiguous_consumed_policy"}
            or not isinstance(authorization["actor"], str) or not authorization["actor"]
            or authorization["ambiguous_consumed_policy"] != "settle_unknown_without_reexecution"):
        raise ComparisonContractError("authorization policy differs")
    inputs = plan["inputs"]
    roles = set()
    if not isinstance(inputs, list):
        raise ComparisonContractError("plan inputs missing")
    for row in inputs:
        if (not isinstance(row, dict) or set(row) != {"role", "path", "sha256"}
                or row["role"] in roles or not isinstance(row["role"], str)
                or _SHA.fullmatch(str(row["sha256"])) is None):
            raise ComparisonContractError("plan input differs")
        _relative(row["path"], "plan input")
        roles.add(row["role"])
    if not {"diagnostic_plan", "recovery_review_policy",
            *(f"diagnostic_protocol:{case}" for case in case_ids)} <= roles:
        raise ComparisonContractError("diagnostic preregistration inputs missing")
    if (schema == PLAN_SCHEMA_V4
            and plan["continuation"]["gpu_admission_profile"] == "registered_desktop_shared_retry_v1"):
        note = next((row for row in inputs
                     if row["role"] == "consistent_gpu_observation_preregistration"), None)
        if (not CONSISTENT_OBSERVATION_SOURCES <= set(closure) or note is None
                or note["path"] != CONSISTENT_OBSERVATION_NOTE
                or note["sha256"] != closure[CONSISTENT_OBSERVATION_NOTE]):
            raise ComparisonContractError("consistent GPU observation preregistration/closure missing")
    if schema == PLAN_SCHEMA_V5:
        required_sources = CONSISTENT_OBSERVATION_SOURCES | {
            V10_NOTE, COMPLETED_EFFECT_REGISTRY_RELATIVE,
        }
        notes = {row["role"]: row for row in inputs}
        if (
            not required_sources <= set(closure)
            or closure.get(COMPLETED_EFFECT_REGISTRY_RELATIVE)
            != COMPLETED_EFFECT_REGISTRY_SHA256_LF
            or notes.get("completed_motor_effects_preregistration")
            != {"role": "completed_motor_effects_preregistration", "path": V10_NOTE,
                "sha256": closure.get(V10_NOTE)}
            or notes.get("completed_motor_effect_registry")
            != {"role": "completed_motor_effect_registry",
                "path": COMPLETED_EFFECT_REGISTRY_RELATIVE,
                "sha256": COMPLETED_EFFECT_REGISTRY_SHA256_LF}
        ):
            raise ComparisonContractError("v5 preregistration/registry closure missing")
    if schema in {PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}:
        required_sources = CONSISTENT_OBSERVATION_SOURCES | {
            V10_NOTE, COMPLETED_EFFECT_REGISTRY_RELATIVE, V12_NOTE,
            EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE,
            "evals/registered_epoch_shutdown_failure_v1.py",
            "tests/test_registered_epoch_shutdown_failure_v1.py",
        }
        if schema == PLAN_SCHEMA_V7:
            required_sources |= {
                V13_NOTE,
                "evals/registered_epoch_namespace_failure_v1.py",
                "tests/test_registered_epoch_namespace_failure_v1.py",
            }
        notes = {row["role"]: row for row in inputs}
        if (
            not required_sources <= set(closure)
            or closure.get(COMPLETED_EFFECT_REGISTRY_RELATIVE)
            != COMPLETED_EFFECT_REGISTRY_SHA256_LF
            or closure.get(EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE)
            != EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF
            or notes.get("completed_motor_effects_preregistration")
            != {"role": "completed_motor_effects_preregistration", "path": V10_NOTE,
                "sha256": closure.get(V10_NOTE)}
            or notes.get("completed_motor_effect_registry")
            != {"role": "completed_motor_effect_registry",
                "path": COMPLETED_EFFECT_REGISTRY_RELATIVE,
                "sha256": COMPLETED_EFFECT_REGISTRY_SHA256_LF}
            or notes.get("epoch_close_retry_preregistration")
            != {"role": "epoch_close_retry_preregistration", "path": V12_NOTE,
                "sha256": closure.get(V12_NOTE)}
            or notes.get("epoch_shutdown_retry_policy")
            != {"role": "epoch_shutdown_retry_policy",
                "path": EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE,
                "sha256": EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF}
            or (schema == PLAN_SCHEMA_V7 and notes.get(
                "epoch_namespace_retry_preregistration") != {
                    "role": "epoch_namespace_retry_preregistration", "path": V13_NOTE,
                    "sha256": closure.get(V13_NOTE)})
        ):
            raise ComparisonContractError("v6 retry preregistration/policy closure missing")
    if schema in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10}:
        notes = {row["role"]: row for row in inputs}
        if (
            not (CONSISTENT_OBSERVATION_SOURCES | BALANCED_ADMISSION_SOURCES
                 | BALANCED_RESOURCE_SUMMARY_SOURCES)
            <= set(closure)
            or V14_NOTE not in closure
            or V13_PUBLIC_SUMMARY not in closure
            or closure[V13_PUBLIC_SUMMARY] != V13_PUBLIC_SUMMARY_SHA256
            or notes.get("balanced_resource_remeasurement_preregistration")
            != {
                "role": "balanced_resource_remeasurement_preregistration",
                "path": V14_NOTE,
                "sha256": closure.get(V14_NOTE),
            }
            or notes.get("known_primary_outcomes")
            != {
                "role": "known_primary_outcomes",
                "path": V13_PUBLIC_SUMMARY,
                "sha256": V13_PUBLIC_SUMMARY_SHA256,
            }
        ):
            raise ComparisonContractError("v8 resource remeasurement preregistration/closure missing")
        if schema == PLAN_SCHEMA_V9 and (
            not BALANCED_NAMESPACE_SOURCES <= set(closure)
            or notes.get("balanced_resource_namespace_preregistration") != {
                "role": "balanced_resource_namespace_preregistration",
                "path": V15_NOTE,
                "sha256": closure.get(V15_NOTE),
            }
        ):
            raise ComparisonContractError("v9 actuator namespace preregistration/closure missing")
        if schema == PLAN_SCHEMA_V10 and (
            not (BALANCED_NAMESPACE_SOURCES | {V16_NOTE, V15_FAILURE_SUMMARY}) <= set(closure)
            or closure.get(V15_FAILURE_SUMMARY) != V15_FAILURE_SUMMARY_SHA256
            or notes.get("balanced_resource_namespace_preregistration") != {
                "role": "balanced_resource_namespace_preregistration",
                "path": V15_NOTE, "sha256": closure.get(V15_NOTE),
            }
            or notes.get("balanced_resource_manifest_retry_preregistration") != {
                "role": "balanced_resource_manifest_retry_preregistration",
                "path": V16_NOTE, "sha256": closure.get(V16_NOTE),
            }
            or notes.get("failed_resource_remeasurement") != {
                "role": "failed_resource_remeasurement",
                "path": V15_FAILURE_SUMMARY, "sha256": V15_FAILURE_SUMMARY_SHA256,
            }
        ):
            raise ComparisonContractError("v10 manifest retry preregistration/closure missing")
    if schema == PLAN_SCHEMA_V11:
        notes = {row["role"]: row for row in inputs}
        required_sources = (
            CONSISTENT_OBSERVATION_SOURCES
            | BALANCED_ADMISSION_SOURCES
            | BALANCED_RESOURCE_SUMMARY_SOURCES
            | BALANCED_NAMESPACE_SOURCES
            | BALANCED_CONTINUATION_SOURCES
            | {V14_NOTE, V13_PUBLIC_SUMMARY, V16_NOTE, V15_FAILURE_SUMMARY}
        )
        if (
            not required_sources <= set(closure)
            or closure.get(V13_PUBLIC_SUMMARY) != V13_PUBLIC_SUMMARY_SHA256
            or closure.get(V15_FAILURE_SUMMARY) != V15_FAILURE_SUMMARY_SHA256
            or notes.get("balanced_resource_remeasurement_preregistration") != {
                "role": "balanced_resource_remeasurement_preregistration",
                "path": V14_NOTE,
                "sha256": closure.get(V14_NOTE),
            }
            or notes.get("known_primary_outcomes") != {
                "role": "known_primary_outcomes",
                "path": V13_PUBLIC_SUMMARY,
                "sha256": V13_PUBLIC_SUMMARY_SHA256,
            }
            or notes.get("balanced_resource_namespace_preregistration") != {
                "role": "balanced_resource_namespace_preregistration",
                "path": V15_NOTE,
                "sha256": closure.get(V15_NOTE),
            }
            or notes.get("balanced_resource_manifest_retry_preregistration") != {
                "role": "balanced_resource_manifest_retry_preregistration",
                "path": V16_NOTE,
                "sha256": closure.get(V16_NOTE),
            }
            or notes.get("failed_resource_remeasurement") != {
                "role": "failed_resource_remeasurement",
                "path": V15_FAILURE_SUMMARY,
                "sha256": V15_FAILURE_SUMMARY_SHA256,
            }
            or notes.get("balanced_resource_continuation_preregistration") != {
                "role": "balanced_resource_continuation_preregistration",
                "path": V17_NOTE,
                "sha256": closure.get(V17_NOTE),
            }
            or notes.get("prior_recovery_timing_prefix") != {
                "role": "prior_recovery_timing_prefix",
                "path": V16_PREFIX,
                "sha256": closure.get(V16_PREFIX),
            }
        ):
            raise ComparisonContractError("v11 continuation preregistration/closure missing")


def load_plan(plan_path: Path, *, repo_root: Path, registration_anchor: str,
              require_committed: bool = True) -> dict:
    repo = Path(repo_root).resolve(strict=True)
    path = Path(plan_path).resolve(strict=True)
    plan = _read(path)
    _validate_plan(plan)
    for row in plan["inputs"]:
        candidate = (repo / _relative(row["path"], "plan input")).resolve(strict=True)
        if _lf_sha(candidate) != row["sha256"]:
            raise ComparisonContractError(f"plan input changed: {row['role']}")
    for relative, sha in plan["source_closure"].items():
        candidate = (repo / _relative(relative, "source closure")).resolve(strict=True)
        if _lf_sha(candidate) != sha:
            raise ComparisonContractError("source closure changed")
    if require_committed:
        if _COMMIT.fullmatch(registration_anchor) is None:
            raise ComparisonContractError("registration anchor differs")
        for ancestor, descendant in ((plan["source_anchor"], registration_anchor),
                                     (registration_anchor, "HEAD")):
            result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                     ancestor, descendant], check=False, capture_output=True)
            if result.returncode != 0:
                raise ComparisonContractError("source/registration ancestry differs")
        registered = [path, *((repo / _relative(row["path"], "input")).resolve() for row in plan["inputs"])]
        for item in registered:
            relative = item.relative_to(repo).as_posix()
            blob = subprocess.run(["git", "-C", str(repo), "show", f"{registration_anchor}:{relative}"],
                                  check=True, capture_output=True).stdout
            if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != _lf_sha(item):
                raise ComparisonContractError("registration blob differs")
        for relative, sha in plan["source_closure"].items():
            blob = subprocess.run(["git", "-C", str(repo), "show", f"{plan['source_anchor']}:{relative}"],
                                  check=True, capture_output=True).stdout
            if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != sha:
                raise ComparisonContractError("source closure blob differs")
    return plan


def _validate_job(job: dict, plan_ref: dict, plan: dict, job_path: Path) -> list[dict]:
    fields = {"schema", "plan", "arms"}
    if plan["schema"] in {PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}:
        fields.add("epoch_shutdown_failure_proof")
    if plan["schema"] == PLAN_SCHEMA_V7:
        fields.add("epoch_namespace_failure_context")
    if set(job) != fields or job["schema"] != JOB_SCHEMA or job["plan"] != plan_ref:
        raise ComparisonContractError("job fields/plan binding differ")
    if plan["schema"] in {PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}:
        try:
            verified_failure = epoch_failure.verify_failure_proof(
                job["epoch_shutdown_failure_proof"], repo=_ENTRY_ROOT)
        except epoch_failure.ShutdownFailureError as exc:
            raise ComparisonContractError(str(exc)) from exc
        if (verified_failure["proof_ref"]["sha256"]
                != plan["continuation"]["failure_proof_sha256"]):
            raise ComparisonContractError("job epoch shutdown failure proof differs from plan")
    if plan["schema"] == PLAN_SCHEMA_V7:
        try:
            verified_namespace = namespace_failure.verify_failure_context(
                job["epoch_namespace_failure_context"])
        except namespace_failure.NamespaceFailureError as exc:
            raise ComparisonContractError(str(exc)) from exc
        if (verified_namespace["context_ref"]["sha256"]
                != plan["continuation"]["namespace_failure_context_sha256"]):
            raise ComparisonContractError("job namespace failure context differs from plan")
    rows = job["arms"]
    case_ids = tuple(row["case_id"] for row in plan["cases"])
    if not isinstance(rows, list) or len(rows) != len(case_ids) * 4:
        raise ComparisonContractError("complete recovery arm matrix required")
    if plan["schema"] in {PLAN_SCHEMA_V5, PLAN_SCHEMA_V6, PLAN_SCHEMA_V7,
                           PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10,
                           PLAN_SCHEMA_V11}:
        expected_order = [
            (case["case_id"], method)
            for case in plan["cases"]
            for method in case["rotation"]
        ]
        actual_order = [
            (row.get("case_id"), row.get("method"))
            if isinstance(row, dict) else (None, None)
            for row in rows
        ]
        if actual_order != expected_order:
            raise ComparisonContractError("completed-import recovery arm rotation order differs")
    expected = {(case, method) for case in case_ids for method in METHODS}
    seen = set()
    normalized = []
    seeds = plan["recovery_policy"]["seeds"]
    stage_imports = {
        (row["case_id"], row["method"]): row
        for row in plan.get("continuation", {}).get("stage_imports", [])
    }
    for row in rows:
        fields = {"case_id", "method", "diagnostic_manifest", "recovery_folder",
                  "recovery_payload", "approval_ref"}
        if plan["schema"] in {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                              PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V11}:
            fields.add("continuation_import")
        if not isinstance(row, dict) or set(row) != fields:
            raise ComparisonContractError("job arm fields differ")
        key = (row["case_id"], row["method"])
        if key not in expected or key in seen:
            raise ComparisonContractError("job arm identity differs")
        seen.add(key)
        manifest_path, manifest = _ref(row["diagnostic_manifest"], "diagnostic manifest")
        if not isinstance(manifest, dict) or manifest.get("method") != row["method"]:
            raise ComparisonContractError("diagnostic method binding differs")
        verification = manifest.get("source_verification", {})
        _, source = _ref(manifest.get("source"), "diagnostic source")
        if (not isinstance(source, dict) or source.get("case_id") != row["case_id"]
                or verification.get("case_id") != row["case_id"]):
            raise ComparisonContractError("diagnostic source case binding differs")
        source_checkpoint = verification.get("checkpoint_sha256")
        if not isinstance(source_checkpoint, str) or _SHA.fullmatch(source_checkpoint) is None:
            raise ComparisonContractError("diagnostic source checkpoint binding missing")
        session = manifest_path.parent
        continuation_import = row.get("continuation_import")
        expected_import = stage_imports.get(key)
        if plan["schema"] in {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                              PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V11}:
            if expected_import is None:
                if continuation_import is not None:
                    raise ComparisonContractError("unexpected continuation import")
            elif (
                not isinstance(continuation_import, dict)
                or set(continuation_import) != {"kind", "artifact_id", "proof"}
                or continuation_import["kind"] != expected_import["kind"]
                or continuation_import["artifact_id"] != expected_import["artifact_id"]
            ):
                raise ComparisonContractError("continuation import binding differs")
            else:
                proof_path, _proof = _ref(continuation_import["proof"], "continuation proof")
                if file_sha(proof_path) != expected_import["proof_sha256"]:
                    raise ComparisonContractError("continuation proof differs from plan")
        optional = (row["recovery_folder"], row["recovery_payload"], row["approval_ref"])
        if any(value is None for value in optional):
            if not all(value is None for value in optional):
                raise ComparisonContractError("recovery folder/payload/approval must be jointly null")
            if expected_import is not None and expected_import["kind"] != "completed_effect":
                raise ComparisonContractError("completed training import requires fresh recovery authority")
            normalized.append({**deepcopy(row), "session_path": str(session)})
            continue
        if expected_import is not None and expected_import["kind"] == "completed_effect":
            raise ComparisonContractError("completed effect import must not carry recovery authority")
        folder = Path(row["recovery_folder"])
        expected_folder = (job_path.parent / "prepared-recoveries" / row["case_id"]
                           / row["method"]).resolve()
        if not folder.is_absolute() or folder != expected_folder:
            raise ComparisonContractError("recovery folder layout differs")
        payload = row["recovery_payload"]
        if not isinstance(payload, dict) or set(payload) != {
                "protocol", "artifact_resolver", "isaaclab_root", "output_root"}:
            raise ComparisonContractError("recovery payload fields differ")
        _, protocol = _ref(payload["protocol"], "recovery protocol")
        diagnostic = protocol.get("diagnostic", {}) if isinstance(protocol, dict) else {}
        operation = protocol.get("operation", {}) if isinstance(protocol, dict) else {}
        mechanism = diagnostic.get("mechanism")
        entry = REGISTRY.get(mechanism) if isinstance(mechanism, str) else None
        mode = operation.get("recovery_mode")
        expected_seed = seeds[row["case_id"]].get(mode) if mode in {"resume", "restart"} else None
        commitment = (plan["template_commitments"].get(row["case_id"], {}).get(mechanism)
                      if isinstance(mechanism, str) else None)
        operation_v2_fields = {
            "correction_id", "recovery_mode", "runner_id", "training_seed",
            "training_iterations", "gpu_budget_s", "source_checkpoint_consumed",
            "training_binding_artifact_id", "fixed_binding_artifact_id",
            "gpu_admission_profile", "prior_training_artifact_id",
        }
        continuation_schema = plan["schema"] in {
            PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
            PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10,
            PLAN_SCHEMA_V11,
        }
        expected_protocol_schema = (
            "registered_recovery_protocol_v4"
            if (plan["schema"] == PLAN_SCHEMA_V11
                and expected_import is not None
                and expected_import["kind"] == "completed_training")
            else "registered_recovery_protocol_v3"
            if plan["schema"] in {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                                  PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V8,
                                  PLAN_SCHEMA_V9, PLAN_SCHEMA_V10, PLAN_SCHEMA_V11}
            else "registered_recovery_protocol_v2"
        )
        if (not isinstance(protocol, dict)
                or (continuation_schema
                    and protocol.get("schema") != expected_protocol_schema)
                or (continuation_schema and set(operation) != operation_v2_fields)
                or diagnostic.get("case_id") != row["case_id"]
                or entry is None or operation.get("correction_id") != entry["correction_id"]
                or mode not in entry["recovery_modes"]
                or operation.get("training_seed") != expected_seed
                or operation.get("training_iterations") != 300
                or protocol.get("actors") != [plan["authorization"]["actor"]]
                or protocol.get("source_anchor") != plan["source_anchor"]
                or protocol.get("source_closure") != plan["source_closure"]
                or protocol.get("registry_sha256") != registry_digest()
                or diagnostic.get("checkpoint_sha256") != source_checkpoint
                or not isinstance(commitment, dict)
                or commitment.get("source_checkpoint_sha256") != source_checkpoint):
            raise ComparisonContractError("per-arm recovery protocol differs from parent policy")
        prior_id = operation.get("prior_training_artifact_id")
        if plan["schema"] == PLAN_SCHEMA_V2:
            continuation = plan["continuation"]
            prior = continuation["prior_training"]
            expected_prior_id = (
                "prior_training"
                if row["case_id"] == "case-05"
                and row["method"] == "fixed_sequence"
                and mechanism == "actuator"
                else None
            )
            if (
                operation.get("gpu_admission_profile") != continuation["gpu_admission_profile"]
                or prior_id != expected_prior_id
                or (
                    expected_prior_id is not None
                    and (
                        diagnostic.get("session_manifest_sha256")
                        != prior["diagnostic_manifest_sha256"]
                        or source_checkpoint != prior["source_checkpoint_sha256"]
                        or operation.get("training_seed") != prior["seed"]
                        or operation.get("correction_id") != prior["correction_id"]
                    )
                )
            ):
                raise ComparisonContractError("prior training arm/profile binding differs")
        if plan["schema"] == PLAN_SCHEMA_V3:
            expected_prior_id = (
                "prior_training"
                if expected_import is not None and expected_import["kind"] == "completed_training"
                else None
            )
            if (
                operation.get("gpu_admission_profile")
                != plan["continuation"]["gpu_admission_profile"]
                or prior_id != expected_prior_id
                or (
                    expected_import is not None
                    and (
                        diagnostic.get("session_manifest_sha256")
                        != expected_import["diagnostic_manifest_sha256"]
                        or source_checkpoint != expected_import["source_checkpoint_sha256"]
                        or operation.get("training_seed") != expected_import["seed"]
                        or operation.get("correction_id") != expected_import["correction_id"]
                    )
                )
            ):
                raise ComparisonContractError("v3 prior arm/profile binding differs")
        if (plan["schema"] == PLAN_SCHEMA_V4
                and (operation.get("gpu_admission_profile")
                     != plan["continuation"]["gpu_admission_profile"]
                     or prior_id is not None
                     or (expected_import is not None and (
                         diagnostic.get("session_manifest_sha256")
                         != expected_import["diagnostic_manifest_sha256"]
                         or source_checkpoint != expected_import["source_checkpoint_sha256"]
                         or operation.get("training_seed") != expected_import["seed"]
                         or operation.get("correction_id") != expected_import["correction_id"]
                     )))):
            raise ComparisonContractError("v4 prior arm/profile binding differs")
        if (plan["schema"] in {PLAN_SCHEMA_V5, PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}
                and (operation.get("gpu_admission_profile")
                     != plan["continuation"]["gpu_admission_profile"]
                     or prior_id is not None
                     or (expected_import is not None and (
                         diagnostic.get("session_manifest_sha256")
                         != expected_import["diagnostic_manifest_sha256"]
                         or source_checkpoint != expected_import["source_checkpoint_sha256"]
                         or operation.get("training_seed") != expected_import["seed"]
                         or operation.get("correction_id") != expected_import["correction_id"]
                     )))):
            raise ComparisonContractError("completed-import prior arm/profile binding differs")
        if (plan["schema"] in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10}
                and (operation.get("gpu_admission_profile")
                     != "registered_desktop_shared_retry_v1"
                     or prior_id is not None
                     or expected_import is not None)):
            raise ComparisonContractError("v8 fresh arm/profile binding differs")
        if plan["schema"] == PLAN_SCHEMA_V11:
            expected_prior_id = (
                "prior_training"
                if expected_import is not None and expected_import["kind"] == "completed_training"
                else None
            )
            if (
                operation.get("gpu_admission_profile")
                != plan["continuation"]["gpu_admission_profile"]
                or prior_id != expected_prior_id
                or (
                    expected_import is not None
                    and (
                        diagnostic.get("session_manifest_sha256")
                        != expected_import["diagnostic_manifest_sha256"]
                        or source_checkpoint != expected_import["source_checkpoint_sha256"]
                        or operation.get("training_seed") != expected_import["seed"]
                        or operation.get("correction_id") != expected_import["correction_id"]
                    )
                )
            ):
                raise ComparisonContractError("v11 continuation arm/profile binding differs")
        _, resolver = _ref(payload["artifact_resolver"], "artifact resolver")
        training_id = operation.get("training_binding_artifact_id")
        fixed_id = operation.get("fixed_binding_artifact_id")
        artifacts = resolver.get("artifacts") if isinstance(resolver, dict) else None
        if (not isinstance(artifacts, dict) or training_id not in artifacts or fixed_id not in artifacts
                or artifacts[training_id].get("sha256") != commitment["training_binding_sha256"]
                or artifacts[fixed_id].get("sha256") != commitment["fixed_binding_sha256"]):
            raise ComparisonContractError("artifact resolver differs from parent template commitment")
        _, training_binding = _ref(artifacts[training_id], "training binding")
        _ref(artifacts[fixed_id], "fixed binding")
        if (plan["schema"] == PLAN_SCHEMA_V9 and mechanism == "actuator"
                and (not isinstance(training_binding, dict)
                     or training_binding.get("schema")
                     != "registered_actuator_recovery_binding_v2"
                     or training_binding.get("namespace_profile")
                     != RESOURCE_REMEASUREMENT_PROFILE)):
            raise ComparisonContractError("v9 actuator namespace binding differs")
        if (plan["schema"] in {PLAN_SCHEMA_V10, PLAN_SCHEMA_V11} and mechanism == "actuator"
                and (not isinstance(training_binding, dict)
                     or training_binding.get("schema")
                     != "registered_actuator_recovery_binding_v3"
                     or training_binding.get("namespace_profile")
                     != RESOURCE_REMEASUREMENT_PROFILE)):
            raise ComparisonContractError("v10 actuator namespace binding differs")
        if prior_id is not None:
            if set(artifacts) != {training_id, fixed_id, prior_id}:
                raise ComparisonContractError("prior training resolver artifacts differ")
            prior_path, _prior_proof = _ref(artifacts[prior_id], "prior training proof")
            expected_proof_sha = (
                plan["continuation"]["prior_training"]["proof_sha256"]
                if plan["schema"] == PLAN_SCHEMA_V2
                else expected_import["proof_sha256"] if expected_import is not None else None
            )
            if expected_proof_sha is None or file_sha(prior_path) != expected_proof_sha:
                raise ComparisonContractError("prior training proof differs from plan")
        elif set(artifacts) != {training_id, fixed_id}:
            raise ComparisonContractError("unexpected resolver artifact")
        approval_path, _ = _ref(row["approval_ref"], "approval")
        normalized.append({**deepcopy(row), "session_path": str(session),
                           "approval_ref": {"path": str(approval_path), "sha256": row["approval_ref"]["sha256"]}})
    if seen != expected:
        raise ComparisonContractError("job arm matrix incomplete")
    if plan["schema"] in {PLAN_SCHEMA_V8, PLAN_SCHEMA_V9, PLAN_SCHEMA_V10, PLAN_SCHEMA_V11}:
        holds = [
            (row["case_id"], row["method"])
            for row in normalized
            if row["recovery_payload"] is None
        ]
        expected_holds = (
            [*V17_EFFECT_KEYS, ("case-07", "iterative_agent")]
            if plan["schema"] == PLAN_SCHEMA_V11
            else [("case-07", "iterative_agent")]
        )
        if holds != expected_holds:
            raise ComparisonContractError("v8 expected fresh/hold matrix differs")
    return normalized


def prepare(plan_path: Path, job_path: Path, output: Path, *, repo_root: Path,
            registration_anchor: str, require_committed: bool = True) -> dict:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("comparison output must be fresh")
    plan = load_plan(plan_path, repo_root=repo_root, registration_anchor=registration_anchor,
                     require_committed=require_committed)
    resolved_job = Path(job_path).resolve(strict=True)
    job = _read(resolved_job)
    plan_ref = {"path": str(Path(plan_path).resolve()), "sha256": file_sha(Path(plan_path))}
    job_ref = {"path": str(resolved_job), "sha256": file_sha(resolved_job)}
    arms = _validate_job(job, plan_ref, plan, resolved_job)
    target.mkdir(parents=True)
    manifest = {"schema": RESULT_SCHEMA + "_manifest", "plan": plan_ref, "job": job_ref,
                "registration_anchor": registration_anchor, "arms": arms,
                "limits": deepcopy(plan["limits"]), "source_cost_policy": "phase_a_shared_count_once"}
    write_new(target / "manifest.json", manifest)
    write_new(target / "anchor.json", {"schema": RESULT_SCHEMA + "_anchor",
                                        "manifest_sha256": file_sha(target / "manifest.json")})
    if plan["schema"] in {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, PLAN_SCHEMA_V5,
                          PLAN_SCHEMA_V6, PLAN_SCHEMA_V7, PLAN_SCHEMA_V11}:
        imported = [
            row for row in arms
            if isinstance(row.get("continuation_import"), dict)
            and row["continuation_import"].get("kind") == "completed_effect"
        ]
        expected_count = (
            7 if plan["schema"] == PLAN_SCHEMA_V11
            else 8 if plan["schema"] in {PLAN_SCHEMA_V5, PLAN_SCHEMA_V6, PLAN_SCHEMA_V7}
            else 2 if plan["schema"] == PLAN_SCHEMA_V4 else 1
        )
        if len(imported) != expected_count:
            raise ComparisonContractError("completed effect import coverage differs")
        for arm in imported:
            _materialize_completed_effect(target, manifest, arm, plan, Path(repo_root))
    return status(target)


def verify_prepared_manifest(folder: Path, *, repo_root: Path,
                             registration_anchor: str, require_committed: bool = True) -> dict:
    """Rebuild prepared authority from the registered plan and normalized job; execute nothing."""
    manifest = _load(folder)
    plan_path, _ = _ref(manifest["plan"], "plan")
    job_path, job = _ref(manifest["job"], "job")
    if not isinstance(job, dict):
        raise ComparisonContractError("prepared job body differs")
    plan = load_plan(plan_path, repo_root=repo_root, registration_anchor=registration_anchor,
                     require_committed=require_committed)
    arms = _validate_job(job, manifest["plan"], plan, job_path)
    expected = {"schema": RESULT_SCHEMA + "_manifest", "plan": manifest["plan"],
                "job": manifest["job"], "registration_anchor": registration_anchor,
                "arms": arms, "limits": deepcopy(plan["limits"]),
                "source_cost_policy": "phase_a_shared_count_once"}
    if manifest != expected:
        raise ComparisonContractError("prepared manifest authority differs from registered job")
    return manifest


def _load(folder: Path) -> dict:
    root = Path(folder).resolve(strict=True)
    manifest = _read(root / "manifest.json")
    anchor = _read(root / "anchor.json")
    if anchor != {"schema": RESULT_SCHEMA + "_anchor", "manifest_sha256": file_sha(root / "manifest.json")}:
        raise ComparisonContractError("comparison anchor changed")
    _ref(manifest["plan"], "plan")
    _ref(manifest["job"], "job")
    return manifest


def _arm_result_path(root: Path, arm: dict) -> Path:
    return root / "results" / arm["case_id"] / f"{arm['method']}.json"


def _record(root: Path, arm: dict, value: dict) -> None:
    value = {"schema": RESULT_SCHEMA + "_arm", "case_id": arm["case_id"],
             "method": arm["method"], "source_cost_accounting": "phase_a_shared_not_repeated", **value}
    write_new(_arm_result_path(root, arm), value)


def _stage_import(plan: dict, arm: dict) -> dict:
    rows = [
        row for row in plan.get("continuation", {}).get("stage_imports", [])
        if row["case_id"] == arm["case_id"] and row["method"] == arm["method"]
    ]
    if len(rows) != 1:
        raise ComparisonContractError("continuation stage import lookup differs")
    return rows[0]


def _continuation_module(plan: dict):
    try:
        name = ("evals.registered_recovery_continuation_v5"
                if plan.get("schema") == PLAN_SCHEMA_V11
                else "evals.registered_recovery_continuation_v4"
                if plan.get("schema") in {PLAN_SCHEMA_V4, PLAN_SCHEMA_V5, PLAN_SCHEMA_V6,
                                           PLAN_SCHEMA_V7}
                else "evals.registered_recovery_continuation_v3")
        return importlib.import_module(name)
    except (ImportError, AttributeError) as exc:
        raise ComparisonContractError("continuation proof verifier unavailable") from exc


def _materialize_completed_effect(root: Path, manifest: dict, arm: dict, plan: dict,
                                  repo_root: Path) -> None:
    binding = _stage_import(plan, arm)
    proof_ref = arm["continuation_import"]["proof"]
    module = _continuation_module(plan)
    if plan["schema"] in {PLAN_SCHEMA_V4, PLAN_SCHEMA_V5, PLAN_SCHEMA_V6, PLAN_SCHEMA_V7,
                          PLAN_SCHEMA_V11}:
        verified = module.verify_completed_effect(
            proof_ref, expected_import=binding, repo=Path(repo_root).resolve(strict=True))
    else:
        verified = module.verify_completed_effect(
            proof_ref,
            expected_case_id=arm["case_id"],
            expected_method=arm["method"],
            expected_prior_plan_sha256=binding["prior_plan_sha256"],
            expected_diagnostic_manifest_sha256=binding["diagnostic_manifest_sha256"],
            expected_source_checkpoint_sha256=binding["source_checkpoint_sha256"],
            repo=Path(repo_root).resolve(strict=True),
        )
    required = {
        "execution_origin", "result", "prior_resource_ref", "prior_admission_ref",
        "recovery_folder", "post_observation", "proof_ref", "historical_costs",
        "new_dispatch_costs",
    }
    if not isinstance(verified, dict) or not required <= set(verified):
        raise ComparisonContractError("completed effect proof result fields differ")
    if verified["execution_origin"] != "verified_prior_effect" or verified["proof_ref"] != proof_ref:
        raise ComparisonContractError("completed effect proof identity differs")
    historical = verified["historical_costs"]
    current = verified["new_dispatch_costs"]
    current_sdk = current.get("simulator_sdk_launches", current.get("sdk_launches")) \
        if isinstance(current, dict) else None
    if (
        not isinstance(historical, dict)
        or historical.get("simulator_sdk_launches") != 2
        or not isinstance(current, dict)
        or current_sdk != 0
    ):
        raise ComparisonContractError("completed effect resource accounting differs")
    current_costs = {
        "simulator_sdk_launches": 0,
        "cpu_checkpoint_launches": 0,
        "cpu_telemetry_launches": 0,
        "total_owned_children": 0,
        "phase_wall_s": None,
        "owned_gpu_s": current.get("owned_gpu_s"),
    }
    resource = {
        "schema": "registered_recovery_imported_resource_v1",
        "case_id": arm["case_id"],
        "method": arm["method"],
        "plan_sha256": manifest["plan"]["sha256"],
        "status": "complete",
        "execution_origin": "verified_prior_effect",
        "prior_effect_proof_ref": deepcopy(proof_ref),
        "prior_resource_ref": deepcopy(verified["prior_resource_ref"]),
        "prior_admission_ref": deepcopy(verified["prior_admission_ref"]),
        "historical_recovery_folder": verified["recovery_folder"],
        "new_dispatch_costs": current_costs,
        "historical_effect_costs": deepcopy(historical),
    }
    resource_path = root / "resources" / arm["case_id"] / f"{arm['method']}.json"
    write_new(resource_path, resource)
    old_result = verified["result"]
    if not isinstance(old_result, dict):
        raise ComparisonContractError("completed effect result missing")
    preserved = {
        key: deepcopy(value)
        for key, value in old_result.items()
        if key not in {"schema", "case_id", "method", "source_cost_accounting", "resource_cost"}
    }
    _record(root, arm, {
        **preserved,
        "execution_origin": "verified_prior_effect",
        "recovery_started": True,
        "completed_effect_proof": deepcopy(proof_ref),
        "source_arm": {
            "historical_recovery_folder": verified["recovery_folder"],
            "prior_resource_ref": deepcopy(verified["prior_resource_ref"]),
            "prior_admission_ref": deepcopy(verified["prior_admission_ref"]),
            "post_observation": deepcopy(verified["post_observation"]),
        },
        "resource_cost_scope": {
            "new_dispatch": deepcopy(current_costs),
            "historical_effect": deepcopy(historical),
            "whole_effect": deepcopy(historical),
        },
    })


def _request_id(state: dict) -> str | None:
    requests = state.get("requests")
    if not isinstance(requests, dict) or not requests:
        return None
    live = [row for row in requests.values() if row.get("state") in {"pending", "approved", "running"}]
    done = [row for row in requests.values() if row.get("state") == "done"]
    rows = live or done
    return rows[-1].get("request_id") if rows else None


def _verified_diagnostic_status(session_path: Path) -> dict:
    session = InvestigationSession(session_path)
    session._verify_operation(registered_diagnostic_backend.verify_source)
    if session.manifest.get("execution_class") != "actual":
        raise ComparisonContractError("phase-A session is not an actual registered execution")
    return terminal_aware_status(session)


def _ambiguous_costs(recovery_folder: Path, request_id: str) -> dict:
    manifest = _read(recovery_folder / "manifest.json")
    output = Path(manifest["recovery_context"]["output_root"])
    result = {"prepare_s": manifest.get("prepare_s"), "authority_s": None, "queue_s": None,
              "training_s": None, "fixed_s": None, "postcheck_s": None, "api_calls": 0,
              "status": "unknown_partial_cost", "refs": []}
    authority = recovery_folder / "authority_costs" / f"{request_id}.json"
    if authority.is_file():
        row = _read(authority)
        result["authority_s"] = row.get("authority_s")
        result["refs"].append({"path": str(authority.resolve()), "sha256": file_sha(authority)})
    for field, path in (("training_s", output / "training/execution.json"),
                        ("fixed_s", output / "fixed/execution.json")):
        if path.is_file():
            row = _read(path)
            wall = row.get("wall_s")
            if isinstance(wall, (int, float)) and not isinstance(wall, bool) \
                    and math.isfinite(float(wall)) and float(wall) >= 0:
                result[field] = float(wall)
            result["refs"].append({"path": str(path.resolve()), "sha256": file_sha(path)})
    return result


def _resource_cost(root: Path, arm: dict, manifest: dict) -> dict | None:
    path = root / "resources" / arm["case_id"] / f"{arm['method']}.json"
    if not path.exists():
        return None
    row = _read(path)
    if row.get("schema") == "registered_recovery_imported_resource_v1":
        fields = {
            "schema", "case_id", "method", "plan_sha256", "status", "execution_origin",
            "prior_effect_proof_ref", "prior_resource_ref", "prior_admission_ref",
            "historical_recovery_folder", "new_dispatch_costs", "historical_effect_costs",
        }
        current = row.get("new_dispatch_costs")
        historical = row.get("historical_effect_costs")
        if (
            set(row) != fields
            or row["case_id"] != arm["case_id"]
            or row["method"] != arm["method"]
            or row["plan_sha256"] != manifest["plan"]["sha256"]
            or row["status"] != "complete"
            or row["execution_origin"] != "verified_prior_effect"
            or not isinstance(current, dict)
            or current.get("simulator_sdk_launches") != 0
            or not isinstance(historical, dict)
            or historical.get("simulator_sdk_launches") != 2
        ):
            raise ComparisonContractError("imported resource receipt binding differs")
        for label in ("prior_effect_proof_ref", "prior_resource_ref", "prior_admission_ref"):
            _ref(row[label], label)
        return {**row, "receipt_ref": {"path": str(path.resolve()), "sha256": file_sha(path)}}
    fields = {"schema", "case_id", "method", "plan_sha256", "status", "phase_wall_s",
              "lease_await_s", "locked_wall_s", "owned_gpu_s", "owned_gpu_s_complete",
              "simulator_sdk_launches", "cpu_checkpoint_launches", "cpu_telemetry_launches", "total_owned_children",
              "api_calls", "sampler_ref", "admission_ref", "recovery_resource_ref",
              "execution_refs", "error_type"}
    if (set(row) != fields or row["schema"] != "registered_recovery_resource_receipt_v1"
            or row["case_id"] != arm["case_id"] or row["method"] != arm["method"]
            or row["plan_sha256"] != manifest["plan"]["sha256"]
            or row["status"] not in {"complete", "failed", "unknown"}
            or type(row["simulator_sdk_launches"]) is not int
            or not 0 <= row["simulator_sdk_launches"] <= 2
            or type(row["cpu_checkpoint_launches"]) is not int
            or not 0 <= row["cpu_checkpoint_launches"] <= 1
            or type(row["cpu_telemetry_launches"]) is not int
            or not 0 <= row["cpu_telemetry_launches"] <= 1
            or type(row["total_owned_children"]) is not int
            or not 0 <= row["total_owned_children"] <= 4
            or row["total_owned_children"] != (row["simulator_sdk_launches"]
                                                + row["cpu_checkpoint_launches"]
                                                + row["cpu_telemetry_launches"])
            or row["api_calls"] != 0):
        raise ComparisonContractError("resource receipt binding differs")
    for field in ("phase_wall_s", "lease_await_s", "locked_wall_s", "owned_gpu_s"):
        value = row[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ComparisonContractError("resource cost differs")
    if row["phase_wall_s"] is not None and row["phase_wall_s"] > manifest["limits"]["max_parent_wall_s_per_recovery"]:
        raise ComparisonContractError("recovery parent wall cap exceeded")
    if row["sampler_ref"] is not None:
        _ref(row["sampler_ref"], "sampler")
    _ref(row["admission_ref"], "lease admission")
    admission = _read(Path(row["admission_ref"]["path"]))
    measured = recovery_resources.verify(row["recovery_resource_ref"],
                                         request_id=admission["request_id"],
                                         effect_key=admission["effect_key"])
    projected = {"status": measured["status"], "phase_wall_s": measured["phase_wall_s"],
                 "lease_await_s": measured["lease_await_s"], "locked_wall_s": measured["locked_wall_s"],
                 "owned_gpu_s": measured["owned_gpu_s"],
                 "owned_gpu_s_complete": measured["owned_gpu_s_complete"],
                 "simulator_sdk_launches": measured["simulator_sdk_launches"],
                 "cpu_checkpoint_launches": measured["cpu_checkpoint_launches"],
                 "cpu_telemetry_launches": measured["cpu_telemetry_launches"],
                 "total_owned_children": measured["total_owned_children"],
                 "sampler_ref": measured["sampler_ref"], "admission_ref": measured["admission_ref"],
                 "execution_refs": measured["execution_refs"],
                 "error_type": measured["error_type"] or measured["sampler_error_type"]}
    if any(row[key] != value for key, value in projected.items()):
        raise ComparisonContractError("projected resource proof differs from recovery evidence")
    return {**row, "receipt_ref": {"path": str(path.resolve()), "sha256": file_sha(path)}}


def _seal_resource(root: Path, manifest: dict, arm: dict, request: dict) -> dict:
    destination = root / "resources" / arm["case_id"] / f"{arm['method']}.json"
    if destination.exists():
        cost = _resource_cost(root, arm, manifest)
        if cost is None:
            raise ComparisonContractError("sealed resource receipt missing")
        return cost
    source = Path(arm["recovery_folder"]) / "resources" / f"{request['request_id']}.json"
    if not source.is_file():
        raise ComparisonContractError("dispatch resource evidence missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    recovery_resources.project({"path": str(source.resolve()), "sha256": file_sha(source)}, destination,
                               case_id=arm["case_id"], method=arm["method"],
                               plan_sha256=manifest["plan"]["sha256"],
                               request_id=request["request_id"], effect_key=request["effect_key"])
    cost = _resource_cost(root, arm, manifest)
    if cost is None:
        raise ComparisonContractError("sealed resource receipt missing")
    return cost


def _halt(root: Path, arm: dict, reason: str, *, error_type: str | None = None,
          detail: str | None = None) -> None:
    path = root / "halt.json"
    value = {"schema": RESULT_SCHEMA + "_halt", "case_id": arm["case_id"], "method": arm["method"],
             "reason": reason, "error_type": error_type, "detail": detail, "automatic_retry": False}
    if path.exists():
        if _read(path) != value:
            raise ComparisonContractError("halt record changed")
    else:
        write_new(path, value)


def _advance_once(folder: Path) -> dict:
    """Advance at most one arm transition; consumed operations are never dispatched again."""
    root = Path(folder).resolve(strict=True)
    manifest = _load(root)
    actor = _read(Path(manifest["plan"]["path"]))["authorization"]["actor"]
    for arm in manifest["arms"]:
        result_path = _arm_result_path(root, arm)
        if result_path.exists():
            continue
        session = Path(arm["session_path"])
        diagnostic = _verified_diagnostic_status(session)
        diagnostic_status = diagnostic.get("status")
        recovery_prepared = arm["recovery_payload"] is not None
        if diagnostic_status != "confirmed":
            if recovery_prepared:
                raise ComparisonContractError(
                    "unconfirmed/terminal diagnosis must not carry recovery authority")
            outcome = _TERMINAL_WITHOUT_RECOVERY.get(diagnostic_status) if isinstance(diagnostic_status, str) else None
            if outcome is None:
                return {**status(root), "resume": {"required": True, "reason": "diagnostic_not_terminal",
                                                   "case_id": arm["case_id"], "method": arm["method"]}}
            _record(root, arm, {"outcome": outcome, "diagnostic_status": diagnostic_status,
                                "recovery_started": False, "costs": None})
            return status(root)
        mechanism = diagnostic.get("confirmed_mechanism")
        if mechanism not in REGISTRY or mechanism not in {"optimizer_update_count", "optimizer_surrogate", "actuator"}:
            if recovery_prepared:
                raise ComparisonContractError(
                    "unsupported diagnosis must not carry recovery authority")
            _record(root, arm, {"outcome": "unsupported", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": False, "costs": None})
            return status(root)
        if not recovery_prepared:
            return {**status(root), "resume": {"required": True,
                                               "reason": "recovery_artifacts_not_prepared",
                                               "case_id": arm["case_id"],
                                               "method": arm["method"],
                                               "confirmed_mechanism": mechanism}}
        recovery_folder = Path(arm["recovery_folder"])
        if not recovery_folder.exists():
            result = investigate.run("recovery-prepare", session,
                                     {"recovery_payload": arm["recovery_payload"],
                                      "folder": str(recovery_folder)}, backend_name=BACKEND, fixture=False)
            if result.get("mechanism") != mechanism:
                raise ComparisonContractError("recovery prepare mechanism differs from verified diagnosis")
            return status(root)
        recovery_state = investigate.run("recovery-status", recovery_folder, {},
                                         backend_name=BACKEND, fixture=False)
        request_id = _request_id(recovery_state)
        if request_id is None:
            investigate.run("recovery-propose", recovery_folder, {}, backend_name=BACKEND, fixture=False)
            return status(root)
        request = recovery_state["requests"][request_id]
        if request["state"] == "pending":
            investigate.run("recovery-approve", recovery_folder,
                            {"request_id": request_id, "actor": actor, "approval_ref": arm["approval_ref"]},
                            backend_name=BACKEND, fixture=False)
            return status(root)
        if request["state"] == "approved":
            investigate.run("recovery-dispatch", recovery_folder, {"request_id": request_id},
                            backend_name=BACKEND, fixture=False)
            refreshed = investigate.run("recovery-status", recovery_folder, {},
                                        backend_name=BACKEND, fixture=False)
            refreshed_request = refreshed["requests"][request_id]
            resource_path = recovery_folder / "resources" / f"{request_id}.json"
            if resource_path.is_file():
                _seal_resource(root, manifest, arm, refreshed_request)
            elif refreshed_request["state"] != "running":
                raise ComparisonContractError("dispatch returned without resource evidence")
            return status(root)
        if request["state"] == "running":
            # The fixed plan chooses unknown closure, not a second launch, for ambiguous consumption.
            resource_path = recovery_folder / "resources" / f"{request_id}.json"
            if resource_path.is_file():
                _seal_resource(root, manifest, arm, request)
            investigate.run("recovery-settle", recovery_folder, {"request_id": request_id, "receipt": None},
                            backend_name=BACKEND, fixture=False)
            _record(root, arm, {"outcome": "unknown", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": True,
                                "failure_reason": "ambiguous_consumed_side_effect",
                                "costs": _ambiguous_costs(recovery_folder, request_id)})
            _halt(root, arm, "ambiguous_consumed_side_effect")
            return status(root)
        ledger_receipt = request.get("receipt", {})
        resource = _seal_resource(root, manifest, arm, request)
        if resource["status"] != "complete":
            _halt(root, arm, "infrastructure_or_integrity_failure",
                  error_type=resource.get("error_type") or "ResourceMeasurementFailure",
                  detail="registered recovery resource measurement did not complete")
            return status(root)
        if ledger_receipt.get("verdict") == "unknown" and not ledger_receipt.get("result_sha"):
            _record(root, arm, {"outcome": "unknown", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": True,
                                "failure_reason": ledger_receipt.get("reason"),
                                "costs": _ambiguous_costs(recovery_folder, request_id)})
            return status(root)
        post = investigate.run("recovery-post", recovery_folder, {"request_id": request_id},
                               backend_name=BACKEND, fixture=False)
        plan = _read(Path(manifest["plan"]["path"]))
        policy = next(row for row in plan["inputs"] if row["role"] == "recovery_review_policy")
        policy_path = Path(__file__).resolve().parents[1] / REVIEW_POLICY_RELATIVE
        if policy["path"] != REVIEW_POLICY_RELATIVE or _lf_sha(policy_path) != policy["sha256"]:
            raise ComparisonContractError("post review host policy changed")
        review = investigate.run("recovery-review", recovery_folder,
            {"request_id": request_id, "host_gate_sources": {
                "schema": "intervention_host_gate_source_v1", "receipt": reference(policy_path)}},
            backend_name=BACKEND, fixture=False)
        outcome = ledger_receipt.get("verdict", "unknown")
        _record(root, arm, {"outcome": outcome, "diagnostic_status": diagnostic_status,
                            "confirmed_mechanism": mechanism, "recovery_started": True,
                            "new_checkpoint": post.get("new_checkpoint"), "fixed_report": post.get("fixed_report"),
                            "behavior_outcome": post.get("behavior_outcome"), "costs": post.get("costs"),
                            "post_review": review})
        return status(root)
    return status(root)


def advance(folder: Path) -> dict:
    """Advance once; integrity/infrastructure failures halt the remaining batch without retry."""
    root = Path(folder).resolve(strict=True)
    current = status(root)
    if current.get("halted"):
        return current
    manifest = _load(root)
    arm = next((row for row in manifest["arms"] if not _arm_result_path(root, row).exists()), None)
    if arm is None:
        return current
    try:
        plan = _read(Path(manifest["plan"]["path"]))
        if plan.get("schema") in {PLAN_SCHEMA_V7, PLAN_SCHEMA_V8, PLAN_SCHEMA_V9,
                                   PLAN_SCHEMA_V10, PLAN_SCHEMA_V11}:
            verify_prepared_manifest(
                root,
                repo_root=_ENTRY_ROOT,
                registration_anchor=manifest["registration_anchor"],
            )
        with recovery_timing.transition(root, case_id=arm["case_id"], method=arm["method"]):
            return _advance_once(root)
    except Exception as exc:  # noqa: BLE001 - preserve the first infrastructure/integrity failure and stop.
        _halt(root, arm, "infrastructure_or_integrity_failure",
              error_type=type(exc).__name__, detail=str(exc))
        return status(root)


def status(folder: Path) -> dict:
    root = Path(folder).resolve(strict=True)
    manifest = _load(root)
    rows = []
    for arm in manifest["arms"]:
        path = _arm_result_path(root, arm)
        row = (_read(path) if path.exists() else {"case_id": arm["case_id"],
                                                  "method": arm["method"], "outcome": "not_started"})
        row["resource_cost"] = _resource_cost(root, arm, manifest)
        rows.append(row)
    completed = sum(row["outcome"] != "not_started"
                    and (row.get("recovery_started") is not True or row["resource_cost"] is not None)
                    for row in rows)
    total = len(manifest["arms"])
    halt_path = root / "halt.json"
    halt = _read(halt_path) if halt_path.exists() else None
    return {"schema": RESULT_SCHEMA, "completed": completed == total, "completed_arms": completed,
            "total_arms": total, "arms": rows,
            "halted": halt is not None, "halt": halt,
            "resume": {"required": completed != total and halt is None,
                       "remaining_arms": total - completed},
            "limits": deepcopy(manifest["limits"]), "api_calls": 0,
            "source_cost_accounting": "phase_a_shared_count_once"}


def run_to_boundary(folder: Path, *, max_transitions: int = 128) -> dict:
    if type(max_transitions) is not int or not 1 <= max_transitions <= 256:
        raise ValueError("transition bound differs")
    for _ in range(max_transitions):
        current = status(folder)
        if current["completed"] or current["halted"]:
            return current
        after = advance(folder)
        if after.get("resume", {}).get("reason") == "diagnostic_not_terminal":
            return after
    return {**status(folder), "resume": {"required": True, "reason": "transition_limit"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--plan", required=True, type=Path)
    prepare_parser.add_argument("--job", required=True, type=Path)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    prepare_parser.add_argument("--registration-anchor", required=True)
    for command in ("advance", "run", "status"):
        child = sub.add_parser(command)
        child.add_argument("--prepared", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.plan, args.job, args.out, repo_root=args.repo_root,
                             registration_anchor=args.registration_anchor)
        elif args.command == "advance":
            result = advance(args.prepared)
        elif args.command == "run":
            result = run_to_boundary(args.prepared)
        else:
            result = status(args.prepared)
    except Exception as exc:  # noqa: BLE001 - controller boundary records fail-closed JSON.
        result = {"schema": RESULT_SCHEMA, "completed": False, "status": "rejected",
                  "error_type": type(exc).__name__, "reason": str(exc),
                  "resume": {"required": True, "reason": "controller_error"}}
    print(canonical(result).decode())
    return 0 if result.get("status") != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
