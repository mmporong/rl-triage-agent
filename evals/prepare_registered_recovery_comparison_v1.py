"""Prepare immutable recovery templates, then bind them to verified phase-A sessions."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import registered_epoch_recovery_training_v1 as EPOCH_CHILD
from bench import registered_recovery_training_v1 as ACTUATOR
from evals import core_seed_guard_v1 as SEED_GUARD
from evals import ppo_ratio_repair_demo_v2 as RATIO_DEMO
from evals import r1_repair_demo_v2 as R1_DEMO
from evals import registered_epoch_namespace_failure_v1 as EPOCH_NAMESPACE_FAILURE
from evals import registered_epoch_shutdown_failure_v1 as EPOCH_FAILURE
from evals import registered_prior_training_v1 as PRIOR
from evals import registered_recovery_backend_v1 as RECOVERY_BACKEND
from evals import run_diagnostic_comparison_v1 as PHASE_A
from evals import run_registered_recovery_comparison_v1 as COMPARISON
from evals import run_upstream_r1_smoke as UPSTREAM
from evals.diagnostic_terminal_status_v1 import terminal_aware_status
from rl_triage import investigate
from rl_triage import investigation_recovery as RECOVERY
from rl_triage.correction_registry_v2 import REGISTRY, registry_digest
from rl_triage.investigation_session import InvestigationSession, write_new

TEMPLATE_SCHEMA = "registered_recovery_comparison_templates_v1"
JOB_SCHEMA = COMPARISON.JOB_SCHEMA
CASES = ("case-05", "case-06", "case-07", "case-08")
SOURCE_SEEDS = {case: 2664 + index for index, case in enumerate(CASES)}
METHODS = COMPARISON.METHODS
MECHANISMS = ("actuator", "optimizer_update_count", "optimizer_surrogate")
RECIPE = ROOT / "bench/protocols/registered_recovery_training_recipe_v1.json"
SDK_CONTRACT_SHA256 = "6e038778c9f7d9a7c03ef22d9eb71b2510cbe89b98b0efaff77b94423c30f8fa"
RECIPE_CODE_CONTRACT_SHA256 = "15320cd87313b28b6f4b873a3c6c6db63db32a7182499e6c4df46bfdebcb96a9"
OPTIMIZER_TRAINING_FIELDS = frozenset({
    "task", "num_envs", "max_iterations", "configured_learning_epochs",
    "configured_mini_batches", "runtime_updates_to_observe", "source_commit_sha",
    "protocol_path", "protocol_sha256_lf", "source_sha256_lf",
    "runtime_source_sha256_lf", "source_checkpoint_sha256",
})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_FILES = {
    "evals/prepare_registered_recovery_comparison_v1.py",
    "evals/diagnostic_terminal_status_v1.py",
    "tests/test_diagnostic_terminal_status_v1.py",
    "tests/test_investigation_session.py",
    "evals/core_seed_guard_v1.py",
    "tests/test_core_seed_guard_v1.py",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v3.md",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v4.md",
    "evals/run_registered_recovery_comparison_v1.py",
    "evals/registered_recovery_backend_v1.py",
    "evals/registered_recovery_resources_v1.py",
    "evals/lease_read_retry_v1.py",
    "evals/registered_recovery_timing_v1.py",
    "tests/test_registered_recovery_timing_v1.py",
    "evals/registered_recovery_confirmation_v1.py",
    "evals/summarize_registered_recovery_comparison_v1.py",
    "tests/test_registered_recovery_confirmation_v1.py",
    "tests/test_summarize_registered_recovery_comparison_v1.py",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v2.md",
    "bench/protocols/registered_recovery_review_host_policy_v1.json",
    "bench/registered_recovery_training_v1.py",
    "bench/registered_epoch_recovery_training_v1.py",
    "evals/registered_epoch_recovery_execution_v1.py",
    "tests/test_registered_epoch_recovery_training_v1.py",
    "tests/test_registered_epoch_recovery_execution_v1.py",
    "bench/train_with_correction.py",
    "bench/correction_adapter.py",
    "evals/run_training_correction.py",
    "src/rl_triage/correction_registry_v2.py",
    "src/rl_triage/investigation_recovery.py",
    "src/rl_triage/investigation_session.py",
    "src/rl_triage/intervention_governor.py",
    "src/rl_triage/intervention_evidence.py",
    "src/rl_triage/registered_diagnostic_backend.py",
    "src/rl_triage/mechanism_confirmation.py",
    "src/rl_triage/mechanism_registry.py",
    "src/rl_triage/correction_ledger.py",
    "src/rl_triage/correction_registry.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
    "evals/model_decision_journal_v1.py",
    "evals/lineage_source_batch_v1.py",
    "src/rl_triage/gpu_resource_sampler.py",
    "src/rl_triage/observable_behavior_v1.py",
    "bench/protocols/observable_behavior_qualification_v1.json",
    "tests/test_registered_recovery_resources_v1.py",
    "tests/test_lease_read_retry_v1.py",
    "tests/test_observable_behavior_v1.py",
    "src/rl_triage/investigate.py",
    "bench/protocols/registered_recovery_training_recipe_v1.json",
} | set(R1_DEMO.CODE) | set(RATIO_DEMO.CODE)
_CONTINUATION_SOURCE_FILES = {
    "evals/registered_desktop_gpu_admission_v1.py",
    "tests/test_registered_desktop_gpu_admission_v1.py",
    "bench/protocols/registered_desktop_gpu_admission_v1.json",
    "evals/registered_prior_training_v1.py",
    "tests/test_registered_prior_training_v1.py",
    "tests/test_registered_recovery_continuation_v2.py",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v5.md",
}
_CONTINUATION_V3_SOURCE_FILES = _CONTINUATION_SOURCE_FILES | {
    "evals/registered_desktop_gpu_wait_v1.py",
    "tests/test_registered_desktop_gpu_wait_v1.py",
    "bench/protocols/registered_desktop_gpu_wait_v1.json",
    "evals/registered_recovery_continuation_v3.py",
    "tests/test_registered_recovery_continuation_v3.py",
    "tests/test_registered_recovery_continuation_v2.py",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v6.md",
    "evals/summarize_mechanism_resource_comparison_v1.py",
    "tests/test_mechanism_resource_comparison_v1.py",
    "evals/aggregate_recovery_attempt_timing_v1.py",
    "tests/test_aggregate_recovery_attempt_timing_v1.py",
}
GPU_ADMISSION_PROFILE = "registered_desktop_shared_v2"
GPU_ADMISSION_PROFILE_V3 = "registered_desktop_shared_wait_v1"
GPU_ADMISSION_PROFILE_V3_SHARED = "registered_desktop_shared_baseline_v1"
GPU_ADMISSION_PROFILE_RETRY = "registered_desktop_shared_retry_v1"
_V4_PROFILE_TAGS = {
    GPU_ADMISSION_PROFILE_V3_SHARED: "v8_resumable",
    GPU_ADMISSION_PROFILE_RETRY: "v9_consistent_observation",
}
_V3_PROFILE_TAGS = {
    GPU_ADMISSION_PROFILE_V3: "v6_fresh",
    GPU_ADMISSION_PROFILE_V3_SHARED: "v7_shared_desktop",
}
_CONTINUATION_V7_SOURCE_FILES = {
    "evals/registered_desktop_gpu_shared_v1.py",
    "tests/test_registered_desktop_gpu_shared_v1.py",
    "bench/protocols/registered_desktop_gpu_shared_v1.json",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v7.md",
}
_CONTINUATION_V4_SOURCE_FILES = _CONTINUATION_V7_SOURCE_FILES | {
    "evals/registered_recovery_continuation_v4.py",
    "tests/test_registered_recovery_continuation_v4.py",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v8.md",
}
_CONTINUATION_RETRY_SOURCE_FILES = {
    "evals/registered_desktop_gpu_retry_v1.py",
    "tests/test_registered_desktop_gpu_retry_v1.py",
    "bench/protocols/registered_desktop_gpu_retry_v1.json",
    "bench/protocols/diagnostic_recovery_comparison_20261104_v9.md",
}
COMPLETED_IMPORT_PROFILE_MOTOR = "motor_completed_v1"
COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY = "motor_completed_epoch_close_retry_v1"
COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY = "motor_completed_epoch_namespace_retry_v1"
COMPLETED_EFFECT_REGISTRY_RELATIVE = (
    "bench/protocols/registered_completed_motor_effects_v1.json"
)
COMPLETED_EFFECT_REGISTRY_SHA256_LF = (
    "1c8a2fdd9e2e44f5af4c22f7b671e36f1604d1c268aefc885079ae3c7d3cc4a8"
)
_CONTINUATION_V5_SOURCE_FILES = {
    "bench/protocols/diagnostic_recovery_comparison_20261104_v10.md",
    COMPLETED_EFFECT_REGISTRY_RELATIVE,
}
RESTART_SEED_PREREGISTRATION_V11 = (
    "bench/protocols/diagnostic_recovery_comparison_20261105_v11.md"
)
REGISTERED_RESTART_SEED_OVERRIDES_V11 = {"case-07": 5866, "case-08": 5867}
_CONTINUATION_V11_SOURCE_FILES = {RESTART_SEED_PREREGISTRATION_V11}
EPOCH_CLOSE_RETRY_PREREGISTRATION_V12 = (
    "bench/protocols/diagnostic_recovery_comparison_20261105_v12.md"
)
EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE = (
    EPOCH_FAILURE.POLICY_RELATIVE
)
EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF = (
    EPOCH_FAILURE.POLICY_SHA256_LF
)
EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256 = (
    EPOCH_FAILURE.FAILURE_PROOF_SHA256
)
REGISTERED_RESTART_SEED_OVERRIDES_V12 = {"case-07": 6066, "case-08": 6067}
_CONTINUATION_V12_SOURCE_FILES = {
    "evals/registered_epoch_shutdown_failure_v1.py",
    "tests/test_registered_epoch_shutdown_failure_v1.py",
    EPOCH_CLOSE_RETRY_PREREGISTRATION_V12,
    EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE,
}
EPOCH_NAMESPACE_RETRY_PREREGISTRATION_V13 = (
    "bench/protocols/diagnostic_recovery_comparison_20261105_v13.md"
)
REGISTERED_RESTART_SEED_OVERRIDES_V13 = {"case-07": 6266, "case-08": 6267}
_CONTINUATION_V13_SOURCE_FILES = {
    "evals/registered_epoch_namespace_failure_v1.py",
    "tests/test_registered_epoch_namespace_failure_v1.py",
    EPOCH_NAMESPACE_RETRY_PREREGISTRATION_V13,
}
RESOURCE_REMEASUREMENT_PROFILE = COMPARISON.RESOURCE_REMEASUREMENT_PROFILE
RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE = (
    COMPARISON.RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE
)
RESOURCE_MANIFEST_RETRY_PROFILE = COMPARISON.RESOURCE_MANIFEST_RETRY_PROFILE
RESOURCE_CONTINUATION_PROFILE = COMPARISON.RESOURCE_CONTINUATION_PROFILE
REGISTERED_RESTART_SEED_OVERRIDES_V14 = {"case-07": 6466, "case-08": 6467}
REGISTERED_RESTART_SEED_OVERRIDES_V16 = {"case-07": 6666, "case-08": 6667}
RESOURCE_REMEASUREMENT_PREREGISTRATION_V14 = COMPARISON.V14_NOTE
_RESOURCE_REMEASUREMENT_SOURCE_FILES = {
    RESOURCE_REMEASUREMENT_PREREGISTRATION_V14,
    COMPARISON.V13_PUBLIC_SUMMARY,
    "evals/summarize_balanced_recovery_resources_v1.py",
    "tests/test_summarize_balanced_recovery_resources_v1.py",
} | COMPARISON.BALANCED_ADMISSION_SOURCES
_RESOURCE_REMEASUREMENT_NAMESPACE_SOURCE_FILES = COMPARISON.BALANCED_NAMESPACE_SOURCES
_RESOURCE_MANIFEST_RETRY_SOURCE_FILES = {
    COMPARISON.V16_NOTE, COMPARISON.V15_FAILURE_SUMMARY,
}
_RESOURCE_CONTINUATION_SOURCE_FILES = COMPARISON.BALANCED_CONTINUATION_SOURCES
_RESOURCE_CONTINUATION_R2_SOURCE_FILES = COMPARISON.BALANCED_CONTINUATION_R2_SOURCES
_RESOURCE_CONTINUATION_R3_SOURCE_FILES = COMPARISON.BALANCED_CONTINUATION_R3_SOURCES
CPU_NONEXECUTION_PROOF_SCHEMA = "registered_epoch_actual_request_cpu_preflight_v2"
CPU_NONEXECUTION_HELPER_SHA256 = (
    "128fbe1e9478f2da2a152c123e47a840f797f216c88146e2bb69665bd9e40d36"
)
V17_EFFECT_KEYS = COMPARISON.V17_EFFECT_KEYS
V17_TRAINING_KEY = COMPARISON.V17_TRAINING_KEY
PRIOR_V5_SOURCE_ANCHOR = "f9d5fea93cce406a325f578bc6f4597cc0bc35c5"
PRIOR_V5_REGISTRATION_ANCHOR = "bf1632ee6770e681e9e8c4c5b1df890ebaed12a6"
PRIOR_V7_REGISTRATION_ANCHOR = "47fec9fc6107e6efde09950d52e64f5465751252"
V4_ARTIFACTS = ("prior_effect", "prior_static_effect")
V4_EXPECTED_IMPORTS = {
    "prior_effect": {
        "artifact_id": "prior_effect", "kind": "completed_effect",
        "case_id": "case-05", "method": "fixed_sequence",
        "proof_sha256": "66fe289e5e8cd7e2057247adfce385a8e2ff37f47b7ba2f77a5e9e4681c1a961",
        "prior_plan_sha256": "a9546936c929dcb03e37aa0cdda8144079c85916cab4d95dfffef4a7ac936dad",
        "diagnostic_manifest_sha256": "4c2045d49162d7ee54f79b4e719a5888fdce520c80c76d1d0d2a3a0277fc4ce4",
        "source_checkpoint_sha256": "4242a4ae3228ca416c28211e6b713e540d5e44079c431774a9516632d95d75dc",
        "completed_checkpoint_sha256": "3aaeabbc6a00f4fcb34fe13dff12fdadb557a031fc7ff581d6436b4452f5a3e7",
        "seed": 2664, "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": "f9d5fea93cce406a325f578bc6f4597cc0bc35c5",
        "registration_anchor": "bf1632ee6770e681e9e8c4c5b1df890ebaed12a6",
    },
    "prior_static_effect": {
        "artifact_id": "prior_static_effect", "kind": "completed_effect",
        "case_id": "case-05", "method": "static_rule",
        "proof_sha256": "8f7dd18c64d2c6f9f6c61cfcc2143ad4efe4c6055d50309d606e2956ca1bbca7",
        "prior_plan_sha256": "6cb8ae70c2d64b4290e6e99d80e8e16d7c15c729cefacffe611a99807e13e44a",
        "diagnostic_manifest_sha256": "9d1060753255b69372884d3ba33506f2f350e596b0bf9f794f508582219cd5fa",
        "source_checkpoint_sha256": "4242a4ae3228ca416c28211e6b713e540d5e44079c431774a9516632d95d75dc",
        "completed_checkpoint_sha256": "3aaeabbc6a00f4fcb34fe13dff12fdadb557a031fc7ff581d6436b4452f5a3e7",
        "seed": 2664, "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": "af6d9f785d8b72f76b91aeb225775ee6f900889a",
        "registration_anchor": "47fec9fc6107e6efde09950d52e64f5465751252",
    },
}


class PreparationError(ValueError):
    """An immutable input or recovery-template binding is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise PreparationError("JSON object required")
    _canonical(value)
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    if path.is_symlink():
        raise PreparationError("symlink input forbidden")
    return {"path": str(path), "sha256": _sha(path)}


def _copy_json_exact(source: Path, output: Path) -> None:
    source = Path(source).resolve(strict=True)
    _read(source)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(source.read_bytes())


def _git_blob(anchor: str, relative: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), "show", f"{anchor}:{relative}"],
                            check=True, capture_output=True)
    return result.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_closure(anchor: str, *, include_continuation: bool = False,
                    include_continuation_v3: bool = False,
                    include_continuation_v7: bool = False,
                    include_continuation_v4: bool = False,
                    include_observation_retry: bool = False,
                    include_completed_v5: bool = False,
                    include_restart_seed_v11: bool = False,
                    include_epoch_close_retry_v12: bool = False,
                    include_epoch_namespace_retry_v13: bool = False,
                    include_resource_remeasurement_v14: bool = False,
                    include_resource_namespace_v15: bool = False,
                    include_resource_manifest_retry_v16: bool = False,
                    include_resource_continuation_v17: bool = False,
                    include_resource_continuation_v17_r2: bool = False,
                    include_resource_continuation_v17_r3: bool = False) -> dict[str, str]:
    if _HEX40.fullmatch(anchor) is None:
        raise PreparationError("source anchor format")
    closure = {}
    source_files = set(_SOURCE_FILES)
    if include_completed_v5:
        if not include_continuation_v4 or not include_observation_retry:
            raise PreparationError("v5 completed imports require retry continuation sources")
        source_files |= _CONTINUATION_V5_SOURCE_FILES
    if include_restart_seed_v11:
        if not include_completed_v5:
            raise PreparationError("v11 restart seeds require v5 completed imports")
        source_files |= _CONTINUATION_V11_SOURCE_FILES
    if include_epoch_close_retry_v12:
        if not include_completed_v5:
            raise PreparationError("v12 epoch-close retry requires v5 completed imports")
        source_files |= _CONTINUATION_V12_SOURCE_FILES
    if include_epoch_namespace_retry_v13:
        if not include_epoch_close_retry_v12:
            raise PreparationError("v13 namespace retry requires v12 failure sources")
        source_files |= _CONTINUATION_V13_SOURCE_FILES
    if include_resource_remeasurement_v14:
        source_files |= _RESOURCE_REMEASUREMENT_SOURCE_FILES | _CONTINUATION_RETRY_SOURCE_FILES
    if include_resource_namespace_v15:
        if not include_resource_remeasurement_v14:
            raise PreparationError("v15 namespace requires v14 resource remeasurement sources")
        source_files |= _RESOURCE_REMEASUREMENT_NAMESPACE_SOURCE_FILES
    if include_resource_manifest_retry_v16:
        if not include_resource_namespace_v15:
            raise PreparationError("v16 manifest retry requires v15 namespace sources")
        source_files |= _RESOURCE_MANIFEST_RETRY_SOURCE_FILES
    if include_resource_continuation_v17:
        if not include_resource_manifest_retry_v16:
            raise PreparationError("v17 continuation requires v16 manifest retry sources")
        source_files |= _RESOURCE_CONTINUATION_SOURCE_FILES
    if include_resource_continuation_v17_r2:
        if not include_resource_continuation_v17:
            raise PreparationError("v17 r2 seed proof requires v17 continuation sources")
        source_files |= _RESOURCE_CONTINUATION_R2_SOURCE_FILES
    if include_resource_continuation_v17_r3:
        if not include_resource_continuation_v17_r2:
            raise PreparationError("v17 r3 backend preflight requires v17 r2 sources")
        source_files |= _RESOURCE_CONTINUATION_R3_SOURCE_FILES
    if include_observation_retry:
        if not include_continuation_v4:
            raise PreparationError("observation retry requires v4 completed-effect imports")
        source_files |= _CONTINUATION_RETRY_SOURCE_FILES
    if include_continuation_v4:
        source_files |= _CONTINUATION_V3_SOURCE_FILES | _CONTINUATION_V4_SOURCE_FILES
    elif include_continuation_v3:
        source_files |= _CONTINUATION_V3_SOURCE_FILES
        if include_continuation_v7:
            source_files |= _CONTINUATION_V7_SOURCE_FILES
    elif include_continuation:
        source_files |= _CONTINUATION_SOURCE_FILES
    for relative in sorted(source_files):
        current = ROOT / relative
        if not current.is_file() or current.is_symlink():
            raise PreparationError(f"source closure missing: {relative}")
        digest = _sha(current, lf=True)
        if hashlib.sha256(_git_blob(anchor, relative)).hexdigest() != digest:
            raise PreparationError(f"source closure differs from anchor: {relative}")
        closure[relative] = digest
    return closure


def _validate_registered_backend_extensions(closure: Mapping[str, str]) -> dict[str, int]:
    """Exercise the production backend extension boundary for every pending V17 arm."""
    protocols = [{
        "schema": RECOVERY_BACKEND.SCHEMA_V4,
        "source_closure": closure,
        "operation": {
            "gpu_admission_profile": GPU_ADMISSION_PROFILE_RETRY,
            "prior_training_artifact_id": "prior_training",
            "runner_id": "actuator_resume_v1",
            "training_binding_artifact_id": "training-case-06-fixed_sequence",
            "fixed_binding_artifact_id": "fixed-case-06",
        },
    }]
    protocols.extend({
        "schema": RECOVERY_BACKEND.SCHEMA_V3,
        "source_closure": closure,
        "operation": {
            "gpu_admission_profile": GPU_ADMISSION_PROFILE_RETRY,
            "prior_training_artifact_id": None,
            "runner_id": "r1_epochs_preserving_environment_v1",
            "training_binding_artifact_id": f"training-{case_id}-{method}",
            "fixed_binding_artifact_id": f"fixed-{case_id}",
        },
    } for case_id, methods in (
        ("case-07", METHODS[:-1]),
        ("case-08", METHODS),
    ) for method in methods)
    for protocol in protocols:
        RECOVERY_BACKEND._extensions(protocol)
    return {"prior_v4": 1, "fresh_v3": len(protocols) - 1}


def _continuation_module():
    try:
        return importlib.import_module("evals.registered_recovery_continuation_v3")
    except (ImportError, AttributeError) as exc:
        raise PreparationError("continuation proof verifier unavailable") from exc


def _continuation_module_v4():
    try:
        return importlib.import_module("evals.registered_recovery_continuation_v4")
    except (ImportError, AttributeError) as exc:
        raise PreparationError("v4 continuation proof verifier unavailable") from exc


def _continuation_module_v5():
    try:
        return importlib.import_module("evals.registered_recovery_continuation_v5")
    except (ImportError, AttributeError) as exc:
        raise PreparationError("v5 continuation proof verifier unavailable") from exc


def _seed_guard_v3():
    try:
        return importlib.import_module("evals.core_seed_guard_v3")
    except (ImportError, AttributeError) as exc:
        raise PreparationError("v17 r2 seed guard unavailable") from exc


def _nonexecution_seed_proof_refs(
    values: Mapping[str, Path],
) -> dict[str, dict[str, str]]:
    if set(values) != {"prior_epoch_cpu_preflight"}:
        raise PreparationError("v17 r2 requires the registered CPU preflight proof")
    registry = _read(ROOT / COMPARISON.V17_SEED_NONEXECUTION_REGISTRY)
    entries = registry.get("entries")
    if (
        registry.get("schema") != "registered_cpu_seed_nonexecution_registry_v1"
        or not isinstance(entries, dict)
        or set(entries) != {"prior_epoch_cpu_preflight"}
    ):
        raise PreparationError("v17 r2 seed nonexecution registry differs")
    entry = entries["prior_epoch_cpu_preflight"]
    if (
        not isinstance(entry, dict)
        or entry
        != {
            "receipt_sha256": "6f765bfe25e05374d2de05a7ae4ca95e2f25bf0e3b9e30f73295b774f4e55e91",
            "seed": 6666,
            "helper_source_sha256": CPU_NONEXECUTION_HELPER_SHA256,
            "source_schema": CPU_NONEXECUTION_PROOF_SCHEMA,
        }
    ):
        raise PreparationError("v17 r2 seed nonexecution entry differs")
    candidate = Path(values["prior_epoch_cpu_preflight"])
    raw_reference = {
        "path": str(candidate.absolute()),
        "sha256": entry["receipt_sha256"],
    }
    try:
        verified = _seed_guard_v3().verify_nonexecution_proofs(
            {"prior_epoch_cpu_preflight": raw_reference}, ROOT
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PreparationError("v17 r2 CPU proof path is not allowed") from exc
    if (
        not isinstance(verified, list)
        or len(verified) != 1
        or verified[0].get("logical_id") != "prior_epoch_cpu_preflight"
        or verified[0].get("proof") != raw_reference
        or verified[0].get("training_seed") != 6666
        or verified[0].get("helper_sha256") != CPU_NONEXECUTION_HELPER_SHA256
    ):
        raise PreparationError("v17 r2 CPU preflight proof differs from registry")
    return {"prior_epoch_cpu_preflight": raw_reference}


def _v4_imports(completed_effect_proofs: Mapping[str, Path]) -> tuple[list[dict], dict[str, dict]]:
    if set(completed_effect_proofs) != set(V4_ARTIFACTS):
        raise PreparationError("v4 requires exactly two completed effect proofs")
    module = _continuation_module_v4()
    rows: list[dict] = []
    refs: dict[str, dict] = {}
    for artifact_id in V4_ARTIFACTS:
        proof_ref = _ref(Path(completed_effect_proofs[artifact_id]))
        proof = _read(Path(proof_ref["path"]))
        identity = proof.get("identity")
        expected = V4_EXPECTED_IMPORTS[artifact_id]
        expected_identity = {
            "prior_plan_sha256": expected["prior_plan_sha256"],
            "diagnostic_manifest_sha256": expected["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": expected["source_checkpoint_sha256"],
            "final_checkpoint_sha256": expected["completed_checkpoint_sha256"],
            "seed": expected["seed"], "task": expected["task"],
            "correction_id": expected["correction_id"],
            "source_anchor": expected["source_anchor"],
            "registration_anchor": expected["registration_anchor"],
        }
        if (not isinstance(identity, dict) or proof.get("status") != "completed_effect"
                or proof.get("case_id") != expected["case_id"]
                or proof.get("method") != expected["method"]
                or proof_ref["sha256"] != expected["proof_sha256"]
                or identity != expected_identity):
            raise PreparationError("v4 completed effect differs from registered basis")
        row = deepcopy(expected)
        verified = module.verify_completed_effect(proof_ref, expected_import=row, repo=ROOT)
        if verified.get("execution_origin") != "verified_prior_effect":
            raise PreparationError("v4 completed effect verification origin differs")
        rows.append(row)
        refs[artifact_id] = proof_ref
    return rows, refs


def _v5_registry() -> tuple[dict, dict[str, dict]]:
    path = ROOT / COMPLETED_EFFECT_REGISTRY_RELATIVE
    if _sha(path, lf=True) != COMPLETED_EFFECT_REGISTRY_SHA256_LF:
        raise PreparationError("v5 completed effect registry changed")
    registry = _read(path)
    expected_fields = {
        "schema", "profile_id", "completed_cases", "methods", "entries",
        "prior_consumed_gpu_sdk_launches", "expected_new_gpu_sdk_launches",
        "new_gpu_sdk_launch_cap", "cumulative_gpu_sdk_launch_cap",
        "new_recovery_arms", "v9_failed_epoch_sdk_launches", "outcome_selection",
    }
    if (
        set(registry) != expected_fields
        or registry["schema"] != "registered_completed_effect_whitelist_v1"
        or registry["profile_id"] != "all_completed_motor_effects_before_epoch_v1"
        or registry["completed_cases"] != ["case-05", "case-06"]
        or registry["methods"] != list(METHODS)
        or registry["prior_consumed_gpu_sdk_launches"] != 17
        or registry["expected_new_gpu_sdk_launches"] != 14
        or registry["new_gpu_sdk_launch_cap"] != 15
        or registry["cumulative_gpu_sdk_launch_cap"] != 32
        or registry["new_recovery_arms"] != 7
        or registry["v9_failed_epoch_sdk_launches"] != 0
        or registry["outcome_selection"]
        != "all_completed_effects_of_first_two_cases_regardless_of_outcome"
        or not isinstance(registry["entries"], dict)
    ):
        raise PreparationError("v5 completed effect registry contract differs")
    entries = registry["entries"]
    expected_matrix = {
        (case, method) for case in ("case-05", "case-06") for method in METHODS
    }
    if len(entries) != 8 or {
        (row.get("case_id"), row.get("method"))
        for row in entries.values() if isinstance(row, dict)
    } != expected_matrix:
        raise PreparationError("v5 completed effect registry matrix differs")
    for artifact_id, row in entries.items():
        if not isinstance(row, dict) or row.get("artifact_id") != artifact_id:
            raise PreparationError("v5 completed effect registry artifact differs")
    return registry, entries


def _epoch_shutdown_retry_policy() -> dict:
    try:
        return EPOCH_FAILURE.load_policy(repo=ROOT)
    except EPOCH_FAILURE.ShutdownFailureError as exc:
        raise PreparationError(str(exc)) from exc


def _v12_failure_proof(path: Path) -> tuple[dict, dict]:
    reference = _ref(Path(path))
    try:
        verified = EPOCH_FAILURE.verify_failure_proof(reference, repo=ROOT)
    except EPOCH_FAILURE.ShutdownFailureError as exc:
        raise PreparationError(str(exc)) from exc
    return verified["proof_ref"], verified


def _v5_imports(completed_effect_proofs: Mapping[str, Path]) -> tuple[list[dict], dict[str, dict]]:
    _registry, expected = _v5_registry()
    if set(completed_effect_proofs) != set(expected):
        raise PreparationError("v5 requires exactly eight registered completed effect proofs")
    module = _continuation_module_v4()
    rows: list[dict] = []
    refs: dict[str, dict] = {}
    for artifact_id in sorted(expected):
        proof_ref = _ref(Path(completed_effect_proofs[artifact_id]))
        row = deepcopy(expected[artifact_id])
        if proof_ref["sha256"] != row["proof_sha256"]:
            raise PreparationError("v5 completed effect proof differs from registry")
        verified = module.verify_completed_effect(proof_ref, expected_import=row, repo=ROOT)
        if verified.get("execution_origin") != "verified_prior_effect":
            raise PreparationError("v5 completed effect verification origin differs")
        rows.append(row)
        refs[artifact_id] = proof_ref
    return rows, refs


def _v17_artifact_id(case_id: str, method: str, kind: str) -> str:
    if kind == "completed_training":
        return "prior_training"
    return f"prior_effect_{case_id.replace('-', '_')}_{method}"


def _v17_imports(
    completed_effect_proofs: Mapping[str, Path],
    completed_training_proof: Path,
    isaaclab_root: Path,
) -> tuple[list[dict], dict[str, dict]]:
    expected_effect_ids = {
        _v17_artifact_id(case_id, method, "completed_effect")
        for case_id, method in V17_EFFECT_KEYS
    }
    if set(completed_effect_proofs) != expected_effect_ids:
        raise PreparationError("v17 requires exactly seven completed effect proofs")
    module = _continuation_module_v5()
    supplied = {
        **{artifact_id: Path(path) for artifact_id, path in completed_effect_proofs.items()},
        "prior_training": Path(completed_training_proof),
    }
    expected_by_artifact = {
        **{
            _v17_artifact_id(case_id, method, "completed_effect"):
                (case_id, method, "completed_effect")
            for case_id, method in V17_EFFECT_KEYS
        },
        "prior_training": (*V17_TRAINING_KEY, "completed_training"),
    }
    rows: list[dict] = []
    refs: dict[str, dict] = {}
    common_identity: tuple[str, str, str] | None = None
    for artifact_id, (case_id, method, kind) in expected_by_artifact.items():
        proof_ref = _ref(supplied[artifact_id])
        proof = _read(Path(proof_ref["path"]))
        identity = proof.get("identity")
        identity_fields = {
            "prior_plan_sha256", "diagnostic_manifest_sha256",
            "source_checkpoint_sha256", "final_checkpoint_sha256", "seed", "task",
            "correction_id", "source_anchor", "registration_anchor",
        }
        if (
            proof.get("status") != (
                "completed_training_only" if kind == "completed_training" else "completed_effect"
            )
            or proof.get("case_id") != case_id
            or proof.get("method") != method
            or not isinstance(identity, dict)
            or set(identity) != identity_fields
            or any(_HEX64.fullmatch(str(identity[field])) is None for field in (
                "prior_plan_sha256", "diagnostic_manifest_sha256",
                "source_checkpoint_sha256", "final_checkpoint_sha256",
            ))
            or _HEX40.fullmatch(str(identity["source_anchor"])) is None
            or _HEX40.fullmatch(str(identity["registration_anchor"])) is None
            or type(identity["seed"]) is not int
            or not isinstance(identity["task"], str) or not identity["task"]
            or identity["correction_id"] != "restore_motor_torque_speed"
        ):
            raise PreparationError("v17 continuation proof logical identity differs")
        lineage = (
            identity["prior_plan_sha256"], identity["source_anchor"],
            identity["registration_anchor"],
        )
        if common_identity is None:
            common_identity = lineage
        elif lineage != common_identity:
            raise PreparationError("v17 continuation proofs cross prior lineage")
        row = {
            "artifact_id": artifact_id, "kind": kind, "case_id": case_id,
            "method": method, "proof_sha256": proof_ref["sha256"],
            "prior_plan_sha256": identity["prior_plan_sha256"],
            "diagnostic_manifest_sha256": identity["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "completed_checkpoint_sha256": identity["final_checkpoint_sha256"],
            "seed": identity["seed"], "task": identity["task"],
            "correction_id": identity["correction_id"],
            "source_anchor": identity["source_anchor"],
            "registration_anchor": identity["registration_anchor"],
        }
        if kind == "completed_effect":
            verified = module.verify_completed_effect(proof_ref, expected_import=row, repo=ROOT)
            if verified.get("execution_origin") != "verified_prior_effect":
                raise PreparationError("v17 completed effect verification origin differs")
        else:
            verified = module.load_completed_training(
                proof_ref, expected_import=row, repo=ROOT,
                isaaclab=Path(isaaclab_root).resolve(strict=True),
            )
            if verified.get("new_training_launched") is not False:
                raise PreparationError("v17 completed training verification launch state differs")
        rows.append(row)
        refs[artifact_id] = proof_ref
    return rows, refs


def _proof_identity(proof_ref: dict, *, kind: str, method: str,
                    prior_plan_sha256: str | None = None) -> dict:
    proof = _read(Path(proof_ref["path"]))
    identity = proof.get("identity")
    effect_fields = {
        "prior_plan_sha256", "diagnostic_manifest_sha256", "source_checkpoint_sha256",
        "final_checkpoint_sha256", "seed", "task", "correction_id", "source_anchor",
        "registration_anchor",
    }
    training_fields = {
        "diagnostic_manifest_sha256", "source_checkpoint_sha256", "final_checkpoint_sha256",
        "seed", "task", "correction_id", "source_anchor", "case_id", "method",
    }
    expected_fields = effect_fields if kind == "completed effect" else training_fields
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_fields
        or (kind == "completed training" and identity["case_id"] != "case-05")
        or (kind == "completed training" and identity["method"] != method)
        or any(
            _HEX64.fullmatch(str(identity[field])) is None
            for field in (
                *(("prior_plan_sha256",) if kind == "completed effect" else ()),
                "diagnostic_manifest_sha256", "source_checkpoint_sha256",
                "final_checkpoint_sha256",
            )
        )
        or type(identity["seed"]) is not int
        or not isinstance(identity["task"], str)
        or not identity["task"]
        or identity["correction_id"] != "restore_motor_torque_speed"
        or identity["source_anchor"] != PRIOR_V5_SOURCE_ANCHOR
        or (kind == "completed effect"
            and identity["registration_anchor"] != PRIOR_V5_REGISTRATION_ANCHOR)
    ):
        raise PreparationError(f"{kind} proof logical identity differs")
    if kind == "completed training" and _HEX64.fullmatch(str(prior_plan_sha256)) is None:
        raise PreparationError("completed training prior plan binding missing")
    return {
        "case_id": "case-05",
        "method": method,
        "prior_plan_sha256": (
            identity["prior_plan_sha256"] if kind == "completed effect" else prior_plan_sha256
        ),
        "diagnostic_manifest_sha256": identity["diagnostic_manifest_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "completed_checkpoint_sha256": identity["final_checkpoint_sha256"],
        "seed": identity["seed"],
        "task": identity["task"],
        "correction_id": identity["correction_id"],
        "source_anchor": identity["source_anchor"],
        "registration_anchor": (
            identity["registration_anchor"]
            if kind == "completed effect"
            else PRIOR_V5_REGISTRATION_ANCHOR
        ),
    }


def _v3_imports(completed_effect_proof: Path, static_training_proof: Path,
                isaaclab_root: Path) -> tuple[list[dict], dict[str, dict]]:
    effect_ref = _ref(completed_effect_proof)
    training_ref = _ref(static_training_proof)
    effect_identity = _proof_identity(effect_ref, kind="completed effect", method="fixed_sequence")
    training_identity = _proof_identity(
        training_ref,
        kind="completed training",
        method="static_rule",
        prior_plan_sha256=effect_identity["prior_plan_sha256"],
    )
    module = _continuation_module()
    effect = module.verify_completed_effect(
        effect_ref,
        expected_case_id="case-05",
        expected_method="fixed_sequence",
        expected_prior_plan_sha256=effect_identity["prior_plan_sha256"],
        expected_diagnostic_manifest_sha256=effect_identity["diagnostic_manifest_sha256"],
        expected_source_checkpoint_sha256=effect_identity["source_checkpoint_sha256"],
        repo=ROOT,
    )
    training = module.load_completed_training(
        training_ref,
        expected_source_checkpoint_sha256=training_identity["source_checkpoint_sha256"],
        expected_seed=training_identity["seed"],
        expected_correction_id=training_identity["correction_id"],
        expected_task=training_identity["task"],
        expected_diagnostic_manifest_sha256=training_identity["diagnostic_manifest_sha256"],
        repo=ROOT,
        isaaclab=Path(isaaclab_root).resolve(strict=True),
    )
    if effect.get("execution_origin") != "verified_prior_effect":
        raise PreparationError("completed effect verification origin differs")
    if training.get("new_training_launched") is not False:
        raise PreparationError("completed training verification launch state differs")
    rows = []
    for artifact_id, kind, proof_ref, identity in (
        ("prior_effect", "completed_effect", effect_ref, effect_identity),
        ("prior_training", "completed_training", training_ref, training_identity),
    ):
        rows.append({
            "artifact_id": artifact_id,
            "kind": kind,
            "proof_sha256": proof_ref["sha256"],
            **identity,
        })
    return rows, {"prior_effect": effect_ref, "prior_training": training_ref}


def _prior_training(proof_path: Path, isaaclab_root: Path) -> tuple[dict, dict]:
    proof_ref = _ref(proof_path)
    proof = _read(proof_path)
    stage = Path(str(proof.get("training_stage"))).resolve(strict=True)
    binding = _read(stage / "binding.json")
    recovery = Path(str(proof.get("recovery_folder"))).resolve(strict=True)
    prepared = _read(recovery / "manifest.json")
    context = prepared.get("recovery_context")
    if not isinstance(context, dict):
        raise PreparationError("prior recovery context missing")
    protocol_ref = context.get("protocol")
    if not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256"}:
        raise PreparationError("prior recovery protocol reference differs")
    protocol_path = Path(str(protocol_ref["path"])).resolve(strict=True)
    if _sha(protocol_path) != protocol_ref["sha256"]:
        raise PreparationError("prior recovery protocol changed")
    protocol = _read(protocol_path)
    diagnostic = protocol.get("diagnostic")
    if not isinstance(diagnostic, dict):
        raise PreparationError("prior diagnostic binding missing")
    loaded = PRIOR.load_completed_actuator_training(
        proof_ref,
        expected_source_checkpoint_sha256=binding["resume_checkpoint"]["sha256"],
        expected_seed=binding["seed"],
        expected_correction_id=protocol["operation"]["correction_id"],
        expected_task=binding["task"],
        expected_diagnostic_manifest_sha256=diagnostic["session_manifest_sha256"],
        repo=ROOT,
        isaaclab=isaaclab_root,
    )
    recovery_validation = loaded["validation"]["recovery"]
    if (
        loaded["new_training_launched"] is not False
        or recovery_validation["case_id"] != "case-05"
        or recovery_validation["method"] != "fixed_sequence"
        or recovery_validation["protocol_case_arm"] != {"arm": "corrected", "case": "case_02"}
        or recovery_validation["prior_training_gpu_s"] != 515.3279999999795
    ):
        raise PreparationError("prior training logical/accounting identity differs")
    logical = {
        "logical_id": "case-05/fixed_sequence/case_02/CORRECTED",
        "proof_sha256": proof_ref["sha256"],
        "source_checkpoint_sha256": binding["resume_checkpoint"]["sha256"],
        "completed_checkpoint_sha256": loaded["training_result"]["checkpoint"]["sha256"],
        "seed": binding["seed"],
        "task": binding["task"],
        "correction_id": protocol["operation"]["correction_id"],
        "diagnostic_manifest_sha256": diagnostic["session_manifest_sha256"],
    }
    return proof_ref, logical


def _validate_source_index(index_path: Path, metadata_path: Path) -> tuple[dict, dict]:
    index, metadata = _read(index_path), _read(metadata_path)
    if index.get("schema") != "lineage_source_inputs_v2" or metadata.get("schema") != "lineage_source_inputs_metadata_v2":
        raise PreparationError("lineage source input schema")
    ref = index.get("metadata_ref")
    if not isinstance(ref, dict) or set(ref) != {"file", "sha256"} or ref["file"] != Path(metadata_path).name:
        raise PreparationError("source index metadata reference")
    if _sha(metadata_path) != ref["sha256"]:
        raise PreparationError("source metadata changed")
    sources = metadata.get("sources")
    if not isinstance(sources, dict) or not isinstance(index.get("sources"), dict):
        raise PreparationError("source maps missing")
    by_public = {row.get("public_id"): row for row in sources.values() if isinstance(row, dict)}
    if set(by_public) != set(index["sources"]):
        raise PreparationError("source index coverage differs")
    for public_id, recorded in index["sources"].items():
        binding = by_public[public_id].get("binding")
        if not isinstance(binding, dict) or any(binding.get(key) != value for key, value in recorded.items()):
            raise PreparationError(f"source index binding differs: {public_id}")
    return index, metadata


def _case_material(metadata: dict, case: str) -> tuple[dict, dict, int]:
    old = case
    rows = [row for row in metadata["cases"] if isinstance(row, dict) and row.get("case_id") == old]
    if len(rows) != 1:
        raise PreparationError(f"case material missing: {case}")
    row = rows[0]
    # These two private names select original/reference material only; eligibility and labels are ignored.
    source = metadata["sources"].get(row.get("source_job"))
    reference = metadata["sources"].get(row.get("reference_job"))
    if not isinstance(source, dict) or not isinstance(reference, dict):
        raise PreparationError(f"case source/reference material missing: {case}")
    seed = row.get("seed")
    expected = SOURCE_SEEDS[case]
    if seed not in {expected, expected - 110}:
        raise PreparationError(f"case seed is neither registered nor development predecessor: {case}")
    return source, reference, int(seed)


def _source_artifacts(source: dict) -> tuple[Path, dict[str, dict]]:
    event = source.get("material", {}).get("event")
    if not isinstance(event, dict) or set(event) != {"path", "sha256"}:
        raise PreparationError("source event reference")
    event_path = Path(event["path"]).resolve(strict=True)
    if _sha(event_path) != event["sha256"]:
        raise PreparationError("source event changed")
    log_dir = event_path.parent
    paths = {
        "checkpoint": log_dir / "model_299.pt",
        "env": log_dir / "params/env.yaml",
        "agent": log_dir / "params/agent.yaml",
    }
    refs = {name: _ref(path) for name, path in paths.items()}
    if refs["checkpoint"]["sha256"] != source.get("binding", {}).get("checkpoint_sha256"):
        raise PreparationError("source checkpoint lineage differs")
    return log_dir, refs


def _source_setup(source: dict) -> tuple[dict, dict]:
    stage = Path(source.get("material", {}).get("stage", "")).resolve(strict=True)
    manifest_path = stage / "manifest.json"
    binding = source.get("binding")
    recorded = source.get("manifest")
    if (not manifest_path.is_file() or not isinstance(binding, dict) or not isinstance(recorded, dict)
            or binding.get("manifest_sha256") != _sha(manifest_path)):
        raise PreparationError("source setup manifest reference differs")
    body = _read(manifest_path)
    if body != recorded:
        raise PreparationError("source setup manifest body differs")
    return body, _ref(manifest_path)


def _reference_report(reference: dict, output: Path) -> tuple[dict, dict]:
    stage = Path(reference.get("material", {}).get("stage", "")).resolve(strict=True)
    fixed = stage / "fixed-result.json"
    if _sha(fixed) != reference.get("binding", {}).get("fixed_result_sha256"):
        raise PreparationError("reference fixed result changed")
    report = _read(fixed).get("report")
    if not isinstance(report, dict):
        raise PreparationError("reference report missing")
    write_new(output, report)
    return report, _ref(output)


def _runtime_sources(source: dict) -> dict:
    value = source.get("manifest", {}).get("runtime_source_sha256_lf")
    if not isinstance(value, dict) or not value or any(_HEX64.fullmatch(str(v)) is None for v in value.values()):
        raise PreparationError("runtime source binding missing")
    return deepcopy(value)


def _optimizer_binding(anchor: str, checkpoint_sha: str, runtime: dict, mechanism: str,
                       closure: dict, environment_preservation: dict | None = None) -> dict:
    code = set(R1_DEMO.CODE if mechanism == "optimizer_update_count" else RATIO_DEMO.CODE)
    code |= {"evals/prepare_registered_recovery_comparison_v1.py",
             "evals/registered_recovery_backend_v1.py", "evals/run_registered_recovery_comparison_v1.py"}
    if mechanism == "optimizer_update_count":
        code |= {"bench/registered_epoch_recovery_training_v1.py",
                 "evals/registered_epoch_recovery_execution_v1.py"}
    binding = {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "num_envs": 4096,
        "max_iterations": 300, "configured_learning_epochs": 5,
        "configured_mini_batches": 4, "runtime_updates_to_observe": 3,
        "source_commit_sha": anchor,
        "protocol_path": RECIPE.relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": _sha(RECIPE, lf=True),
        "source_sha256_lf": {path: closure[path] for path in sorted(code)},
        "runtime_source_sha256_lf": runtime,
        "source_checkpoint_sha256": checkpoint_sha,
    }
    expected_fields = set(OPTIMIZER_TRAINING_FIELDS)
    if mechanism == "optimizer_update_count":
        if environment_preservation is None:
            raise PreparationError("count correction environment preservation missing")
        binding["environment_preservation"] = deepcopy(environment_preservation)
        expected_fields.add("environment_preservation")
    elif environment_preservation is not None:
        raise PreparationError("surrogate binding must not carry environment preservation")
    if set(binding) != expected_fields:
        raise AssertionError("optimizer training fields drifted")
    return binding


def _environment_preservation(*, source: dict, source_seed: int, training_seed: int,
                              refs: dict[str, dict], code_contract_path: Path,
                              code_contract: dict, output: Path) -> dict:
    setup, setup_ref = _source_setup(source)
    arm = setup.get("arm")
    mode = "world_frame_reward" if arm in {"F1", "COMPOUND"} else "nominal_body_frame"
    if arm not in {"NONE", "F1", "F2", "COMPOUND"}:
        raise PreparationError("source environment arm is unsupported")
    motor_key = "preserve_source_envelope" if arm == "F2" else "nominal"
    target = deepcopy(code_contract.get("reward_targets", {}).get(mode))
    secondary = deepcopy(code_contract.get("motor_targets", {}).get(motor_key))
    if not isinstance(target, dict) or not isinstance(secondary, dict):
        raise PreparationError("environment code contract target missing")
    code_ref = _ref(code_contract_path)
    recipe = {
        "schema": EPOCH_CHILD.EXPECTED_SCHEMA,
        "provenance_kind": EPOCH_CHILD.PROVENANCE_KIND,
        "task": setup.get("task"), "source_arm": arm, "source_seed": source_seed,
        "training_seed": training_seed, "source_checkpoint_sha256": refs["checkpoint"]["sha256"],
        "source_setup_sha256": setup_ref["sha256"],
        "source_env_config_sha256": refs["env"]["sha256"],
        "code_contract_sha256": code_ref["sha256"], "mode": mode,
        "target": target, "secondary_target": secondary,
    }
    recipe_path = output / "environment-recipe.json"
    write_new(recipe_path, recipe)
    binding = {
        "schema": EPOCH_CHILD.SCHEMA, "mode": mode, "source_setup": setup_ref,
        "source_env_config": refs["env"], "code_contract": code_ref,
        "expected_reward": _ref(recipe_path), "target": target, "secondary_target": secondary,
    }
    try:
        EPOCH_CHILD.validate_environment_binding(binding)
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise PreparationError(f"environment preservation binding rejected: {error}") from error
    return binding


def _actuator_binding(anchor: str, seed: int, refs: dict, report: dict, sdk_contract: dict,
                      isaaclab_commit: str, namespace_profile: str | None = None,
                      *, schema: str | None = None) -> dict:
    if _HEX40.fullmatch(isaaclab_commit) is None:
        raise PreparationError("Isaac Lab commit format")
    agent = ACTUATOR._read_yaml(Path(refs["agent"]["path"]))
    num_steps = agent.get("num_steps_per_env")
    if type(num_steps) is not int or num_steps <= 0:
        raise PreparationError("source num_steps_per_env missing")
    if namespace_profile not in {None, ACTUATOR.NAMESPACE_PROFILE}:
        raise PreparationError("actuator recovery namespace profile differs")
    expected_schema = (ACTUATOR.SCHEMA if namespace_profile is None else ACTUATOR.SCHEMA_V2)
    selected_schema = schema or expected_schema
    if selected_schema not in {expected_schema, ACTUATOR.SCHEMA_V3} \
            or (selected_schema == ACTUATOR.SCHEMA_V3 and namespace_profile is None):
        raise PreparationError("actuator recovery binding schema differs")
    binding = {
        "schema": selected_schema,
        "source_commit_sha": anchor,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "seed": seed,
        "num_envs": 4096, "max_iterations": 300, "resume_iteration": 299,
        "resume_checkpoint": refs["checkpoint"], "resume_env_config": refs["env"],
        "resume_agent_config": refs["agent"], "correction_source_contract": sdk_contract,
        "action_scale": report["action_scale"],
        "rsl_rl_version": report["versions"]["rsl-rl-lib"],
        "isaaclab_commit": isaaclab_commit,
        "num_steps_per_env": num_steps,
    }
    if namespace_profile is not None:
        binding["namespace_profile"] = namespace_profile
    return binding


def _fixed_binding(report_ref: dict, report: dict, bands: dict, runtime: dict) -> dict:
    return {"task": report["task"], "runtime_source_sha256_lf": deepcopy(runtime),
            "action_scale": report["action_scale"],
            "protocol_sha256_lf": report["protocol_sha256_lf"],
            "reference_report": report_ref, "bands": deepcopy(bands)}


def _rotation(index: int) -> list[str]:
    methods: list[str] = list(METHODS)
    return methods[index:] + methods[:index]


def _restart_seed_map(
    completed_import_profile: str | None,
    restart_seed_overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    seeds = {case: SOURCE_SEEDS[case] + 3000 for case in CASES}
    if restart_seed_overrides is None:
        return seeds
    registered = {
        COMPLETED_IMPORT_PROFILE_MOTOR: REGISTERED_RESTART_SEED_OVERRIDES_V11,
        COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY: REGISTERED_RESTART_SEED_OVERRIDES_V12,
        COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY: REGISTERED_RESTART_SEED_OVERRIDES_V13,
        RESOURCE_REMEASUREMENT_PROFILE: REGISTERED_RESTART_SEED_OVERRIDES_V14,
        RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE: REGISTERED_RESTART_SEED_OVERRIDES_V14,
        RESOURCE_MANIFEST_RETRY_PROFILE: REGISTERED_RESTART_SEED_OVERRIDES_V16,
        RESOURCE_CONTINUATION_PROFILE: REGISTERED_RESTART_SEED_OVERRIDES_V16,
    }
    if completed_import_profile not in registered:
        raise PreparationError("restart seed overrides require motor completed imports")
    if set(restart_seed_overrides) != {"case-07", "case-08"}:
        raise PreparationError("restart seed overrides must cover case-07 and case-08")
    values = list(restart_seed_overrides.values())
    if any(type(value) is not int or value < 0 for value in values):
        raise PreparationError("restart seed overrides must be nonnegative integers")
    if len(set(values)) != len(values):
        raise PreparationError("restart seed overrides must be distinct")
    if set(values) & set(SOURCE_SEEDS.values()):
        raise PreparationError("restart seed overrides collide with resume seeds")
    if dict(restart_seed_overrides) != registered[completed_import_profile]:
        raise PreparationError("restart seed overrides differ from profile registration")
    seeds.update(restart_seed_overrides)
    return seeds


def prepare_templates(*, source_index: Path, metadata: Path, diagnostic_plan: Path,
                      diagnostic_registration_anchor: str, source_anchor: str,
                      sdk_contract: Path, recipe_code_contract: Path, output: Path, actor: str,
                      isaaclab_commit: str,
                      prior_training_proof: Path | None = None,
                      prior_completed_effect_proof: Path | None = None,
                      prior_static_training_proof: Path | None = None,
                      completed_effect_proofs: Mapping[str, Path] | None = None,
                       completed_training_proof: Path | None = None,
                      completed_import_profile: str | None = None,
                      resource_remeasurement_profile: str | None = None,
                      nonexecution_seed_proofs: Mapping[str, Path] | None = None,
                      restart_seed_overrides: Mapping[str, int] | None = None,
                      epoch_shutdown_failure_proof: Path | None = None,
                      epoch_namespace_failure_context: Path | None = None,
                      isaaclab_root: Path | None = None,
                      gpu_admission_profile: str | None = None,
                      require_committed: bool = True) -> dict:
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("template output must be fresh")
    _index, source_meta = _validate_source_index(source_index, metadata)
    plan_rel = Path(diagnostic_plan).resolve(strict=True).relative_to(ROOT).as_posix()
    phase_ref = {"path": plan_rel, "sha256": _sha(diagnostic_plan, lf=True)}
    phase_plan = PHASE_A.load_plan(phase_ref, repo_root=ROOT,
                                   registration_anchor=diagnostic_registration_anchor,
                                   require_committed=require_committed)
    phase_cases = {row["case_id"]: row for row in phase_plan["cases"]}
    if not {f"case-{int(case[-2:]):02d}" for case in CASES} <= set(phase_cases):
        raise PreparationError("phase-A selected cases missing")
    v8_requested = resource_remeasurement_profile == RESOURCE_REMEASUREMENT_PROFILE
    v9_requested = resource_remeasurement_profile == RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE
    v10_requested = resource_remeasurement_profile == RESOURCE_MANIFEST_RETRY_PROFILE
    v11_requested = resource_remeasurement_profile == RESOURCE_CONTINUATION_PROFILE
    v11_r2_requested = v11_requested and nonexecution_seed_proofs is not None
    v11_r3_requested = v11_r2_requested
    resource_requested = v8_requested or v9_requested or v10_requested or v11_requested
    if resource_remeasurement_profile not in {
        None, RESOURCE_REMEASUREMENT_PROFILE, RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE,
        RESOURCE_MANIFEST_RETRY_PROFILE, RESOURCE_CONTINUATION_PROFILE,
    }:
        raise PreparationError("unsupported resource remeasurement profile")
    v7_requested = completed_import_profile == COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY
    v6_requested = completed_import_profile in {
        COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY,
        COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY,
    }
    v5_requested = completed_import_profile in {
        COMPLETED_IMPORT_PROFILE_MOTOR,
        COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY,
        COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY,
    }
    if resource_requested and completed_import_profile is not None:
        raise PreparationError("resource remeasurement forbids completed imports")
    restart_seeds = _restart_seed_map(
        resource_remeasurement_profile if resource_requested else completed_import_profile,
        restart_seed_overrides,
    )
    v4_requested = completed_effect_proofs is not None and not v5_requested and not v11_requested
    v3_requested = prior_completed_effect_proof is not None or prior_static_training_proof is not None
    if gpu_admission_profile is not None and not (
        v3_requested or v4_requested or v5_requested or resource_requested
    ):
        raise PreparationError("explicit GPU admission profile requires v3 continuation proofs")
    selected_v3_profile = gpu_admission_profile or GPU_ADMISSION_PROFILE_V3
    selected_v4_profile = gpu_admission_profile or (
        GPU_ADMISSION_PROFILE_RETRY if v5_requested else GPU_ADMISSION_PROFILE_V3_SHARED
    )
    if not (v4_requested or v5_requested or resource_requested) \
            and selected_v3_profile not in _V3_PROFILE_TAGS:
        raise PreparationError("unsupported v3 GPU admission profile")
    if v3_requested and (
        prior_completed_effect_proof is None
        or prior_static_training_proof is None
        or prior_training_proof is not None
        or isaaclab_root is None
    ):
        raise PreparationError("v3 proofs and Isaac Lab root must be supplied together")
    if v4_requested and (
        v3_requested or prior_training_proof is not None or isaaclab_root is None
        or selected_v4_profile not in _V4_PROFILE_TAGS
    ):
        raise PreparationError("v4 proofs and Isaac Lab root must be supplied alone")
    if v5_requested and (
        completed_effect_proofs is None
        or v3_requested
        or prior_training_proof is not None
        or isaaclab_root is None
        or selected_v4_profile != GPU_ADMISSION_PROFILE_RETRY
    ):
        raise PreparationError("v5 completed import profile inputs differ")
    if resource_requested and (
        v3_requested or v4_requested or v5_requested
        or prior_training_proof is not None
        or (completed_effect_proofs is not None and not v11_requested)
        or (completed_effect_proofs is None and v11_requested)
        or (v11_requested and completed_training_proof is None)
        or isaaclab_root is None
        or gpu_admission_profile != GPU_ADMISSION_PROFILE_RETRY
        or restart_seed_overrides != (REGISTERED_RESTART_SEED_OVERRIDES_V16
                                      if v10_requested or v11_requested
                                      else REGISTERED_RESTART_SEED_OVERRIDES_V14)
        or epoch_shutdown_failure_proof is not None
        or epoch_namespace_failure_context is not None
    ):
        raise PreparationError("v8 resource remeasurement inputs differ")
    if not v11_requested and completed_training_proof is not None:
        raise PreparationError("completed training proof requires v17 continuation profile")
    if not v11_requested and nonexecution_seed_proofs is not None:
        raise PreparationError("nonexecution seed proofs require v17 continuation profile")
    if v6_requested:
        expected_seeds = (REGISTERED_RESTART_SEED_OVERRIDES_V13 if v7_requested
                          else REGISTERED_RESTART_SEED_OVERRIDES_V12)
        if (restart_seed_overrides != expected_seeds
                or epoch_shutdown_failure_proof is None):
            raise PreparationError("registered restart seeds are required by retry profile")
        failure_proof = _v12_failure_proof(epoch_shutdown_failure_proof)
        if v7_requested:
            if epoch_namespace_failure_context is None:
                raise PreparationError("v13 namespace failure context is required")
            namespace_failure = EPOCH_NAMESPACE_FAILURE.verify_failure_context({
                "path": str(Path(epoch_namespace_failure_context).resolve(strict=True)),
                "sha256": EPOCH_NAMESPACE_FAILURE.CONTEXT_SHA256,
            })
        else:
            if epoch_namespace_failure_context is not None:
                raise PreparationError("namespace failure context requires v13 profile")
            namespace_failure = None
    else:
        failure_proof = None
        namespace_failure = None
        if epoch_shutdown_failure_proof is not None:
            raise PreparationError("epoch shutdown failure proof requires v12 profile")
        if epoch_namespace_failure_context is not None:
            raise PreparationError("namespace failure context requires v13 profile")
    if not (v3_requested or v4_requested or v5_requested or resource_requested) and (
        prior_training_proof is None
    ) != (isaaclab_root is None):
        raise PreparationError("prior proof and Isaac Lab root must be supplied together")
    prior = (
        _prior_training(Path(prior_training_proof), Path(isaaclab_root).resolve(strict=True))
        if prior_training_proof is not None and isaaclab_root is not None
        else None
    )
    if v3_requested:
        if (prior_completed_effect_proof is None or prior_static_training_proof is None
                or isaaclab_root is None):
            raise PreparationError("v3 proof inputs missing after validation")
        v3 = _v3_imports(
            prior_completed_effect_proof,
            prior_static_training_proof,
            isaaclab_root,
        )
    else:
        v3 = None
    v4 = _v4_imports(completed_effect_proofs) if v4_requested and completed_effect_proofs else None
    v5 = _v5_imports(completed_effect_proofs) if v5_requested and completed_effect_proofs else None
    v11 = (
        _v17_imports(completed_effect_proofs, completed_training_proof, isaaclab_root)
        if v11_requested and completed_effect_proofs is not None
        and completed_training_proof is not None and isaaclab_root is not None
        else None
    )
    seed_proof_refs = (
        _nonexecution_seed_proof_refs(nonexecution_seed_proofs)
        if v11_r2_requested and nonexecution_seed_proofs is not None
        else None
    )
    v11_actuator_binding: Path | None = None
    if v11_requested and completed_training_proof is not None:
        training_proof = _read(completed_training_proof)
        stage_files = training_proof.get("stage_files")
        binding_ref = stage_files.get("binding.json") if isinstance(stage_files, dict) else None
        if not isinstance(binding_ref, dict) or set(binding_ref) != {"path", "sha256"}:
            raise PreparationError("v17 completed training binding reference missing")
        v11_actuator_binding = Path(str(binding_ref["path"])).resolve(strict=True)
        if _sha(v11_actuator_binding) != binding_ref["sha256"]:
            raise PreparationError("v17 completed training binding changed")
    continuation_sources = prior is not None
    closure = _source_closure(
        source_anchor,
        include_continuation=continuation_sources,
        include_continuation_v3=v3 is not None,
        include_continuation_v7=(
            v3 is not None and selected_v3_profile == GPU_ADMISSION_PROFILE_V3_SHARED
        ),
        include_continuation_v4=v4 is not None or v5 is not None,
        include_observation_retry=(
            (v4 is not None and selected_v4_profile == GPU_ADMISSION_PROFILE_RETRY)
            or v5 is not None
        ),
        include_completed_v5=v5 is not None,
        include_restart_seed_v11=(
            restart_seed_overrides is not None and not v6_requested and not resource_requested
        ),
        include_epoch_close_retry_v12=v6_requested,
        include_epoch_namespace_retry_v13=v7_requested,
        include_resource_remeasurement_v14=resource_requested,
        include_resource_namespace_v15=v9_requested or v10_requested or v11_requested,
        include_resource_manifest_retry_v16=v10_requested or v11_requested,
        include_resource_continuation_v17=v11_requested,
        include_resource_continuation_v17_r2=v11_r2_requested,
        include_resource_continuation_v17_r3=v11_r3_requested,
    ) if require_committed else {
        path: _sha(ROOT / path, lf=True)
        for path in sorted(
            _SOURCE_FILES
            | (_CONTINUATION_SOURCE_FILES if continuation_sources else set())
            | (_CONTINUATION_V3_SOURCE_FILES if v3 is not None else set())
            | (_CONTINUATION_V7_SOURCE_FILES
               if v3 is not None and selected_v3_profile == GPU_ADMISSION_PROFILE_V3_SHARED
               else set())
            | (_CONTINUATION_V3_SOURCE_FILES | _CONTINUATION_V4_SOURCE_FILES
               if v4 is not None or v5 is not None else set())
            | (_CONTINUATION_RETRY_SOURCE_FILES
               if (v4 is not None and selected_v4_profile == GPU_ADMISSION_PROFILE_RETRY)
               or v5 is not None else set())
            | (_CONTINUATION_V5_SOURCE_FILES if v5 is not None else set())
             | (_CONTINUATION_V11_SOURCE_FILES
                if restart_seed_overrides is not None and not v6_requested
                and not resource_requested
                else set())
            | (_CONTINUATION_V12_SOURCE_FILES if v6_requested else set())
            | (_CONTINUATION_V13_SOURCE_FILES if v7_requested else set())
            | (_RESOURCE_REMEASUREMENT_SOURCE_FILES | _CONTINUATION_RETRY_SOURCE_FILES
               if resource_requested else set())
            | (_RESOURCE_REMEASUREMENT_NAMESPACE_SOURCE_FILES
               if v9_requested or v10_requested or v11_requested else set())
            | (_RESOURCE_MANIFEST_RETRY_SOURCE_FILES if v10_requested else set())
            | (_RESOURCE_MANIFEST_RETRY_SOURCE_FILES | _RESOURCE_CONTINUATION_SOURCE_FILES
               if v11_requested else set())
            | (_RESOURCE_CONTINUATION_R2_SOURCE_FILES if v11_r2_requested else set())
            | (_RESOURCE_CONTINUATION_R3_SOURCE_FILES if v11_r3_requested else set())
        )
    }
    if v11_r3_requested:
        _validate_registered_backend_extensions(closure)
    sdk = _read(sdk_contract)
    if _sha(sdk_contract) != SDK_CONTRACT_SHA256:
        raise PreparationError("SDK source contract changed")
    recipe_code = _read(recipe_code_contract)
    if _sha(recipe_code_contract) != RECIPE_CODE_CONTRACT_SHA256:
        raise PreparationError("environment recipe code contract changed")
    bands = _read(UPSTREAM.P0C_SUMMARY).get("bands")
    if not isinstance(bands, dict):
        raise PreparationError("p0c recovery bands missing")
    out.mkdir(parents=True)
    commitments: dict[str, dict] = {}
    development_sources: dict[str, int] = {}
    for case_index, case in enumerate(CASES):
        source, reference, material_seed = _case_material(source_meta, case)
        development_sources[case] = material_seed
        _log, refs = _source_artifacts(source)
        case_dir = out / "templates" / case
        report, report_ref = _reference_report(reference, case_dir / "reference-report.json")
        runtime = _runtime_sources(source)
        fixed = _fixed_binding(report_ref, report, bands, runtime)
        fixed_path = case_dir / "fixed-binding.json"
        write_new(fixed_path, fixed)
        environment = _environment_preservation(
            source=source, source_seed=material_seed, training_seed=restart_seeds[case],
            refs=refs, code_contract_path=recipe_code_contract, code_contract=recipe_code,
            output=case_dir / "optimizer_update_count")
        training = {
            "actuator": _actuator_binding(
                source_anchor, SOURCE_SEEDS[case], refs, report, sdk, isaaclab_commit,
                ACTUATOR.NAMESPACE_PROFILE if v9_requested or v10_requested or v11_requested else None,
                schema=ACTUATOR.SCHEMA_V3 if v10_requested or v11_requested else None,
            ),
            "optimizer_update_count": _optimizer_binding(source_anchor, refs["checkpoint"]["sha256"],
                                                          runtime, "optimizer_update_count", closure,
                                                          environment),
            "optimizer_surrogate": _optimizer_binding(source_anchor, refs["checkpoint"]["sha256"],
                                                       runtime, "optimizer_surrogate", closure),
        }
        commitments[case] = {}
        for mechanism, body in training.items():
            path = case_dir / mechanism / "training-binding.json"
            if v11_requested and case == "case-06" and mechanism == "actuator":
                if v11_actuator_binding is None:
                    raise PreparationError("v17 actuator binding source missing")
                _copy_json_exact(v11_actuator_binding, path)
                body = _read(path)
            else:
                write_new(path, body)
            if mechanism == "actuator":
                ACTUATOR.validate_prepared_binding(body)
            commitments[case][mechanism] = {
                "training_binding_sha256": _sha(path), "fixed_binding_sha256": _sha(fixed_path),
                "source_checkpoint_sha256": refs["checkpoint"]["sha256"],
            }
    phase_inputs = {row["role"]: row for row in phase_plan["inputs"]}
    inputs = [
        {"role": "recovery_review_policy", "path": COMPARISON.REVIEW_POLICY_RELATIVE,
         "sha256": _sha(ROOT / COMPARISON.REVIEW_POLICY_RELATIVE, lf=True)},
        {"role": "measurement_preregistration",
         "path": "bench/protocols/diagnostic_recovery_comparison_20261104_v2.md",
         "sha256": _sha(ROOT / "bench/protocols/diagnostic_recovery_comparison_20261104_v2.md", lf=True)},
        {"role": "diagnostic_plan", "path": plan_rel, "sha256": phase_ref["sha256"]},
        {"role": "recovery_bands_summary", "path": UPSTREAM.P0C_SUMMARY.relative_to(ROOT).as_posix(),
         "sha256": _sha(UPSTREAM.P0C_SUMMARY, lf=True)},
    ]
    inputs.append({"role": "terminal_hold_integration_preregistration",
                   "path": "bench/protocols/diagnostic_recovery_comparison_20261104_v3.md",
                   "sha256": _sha(ROOT / "bench/protocols/diagnostic_recovery_comparison_20261104_v3.md", lf=True)})
    inputs.append({"role": "owned_stage_infrastructure_preregistration",
                   "path": "bench/protocols/diagnostic_recovery_comparison_20261104_v4.md",
                   "sha256": _sha(ROOT / "bench/protocols/diagnostic_recovery_comparison_20261104_v4.md", lf=True)})
    if v4 is not None or v5 is not None:
        inputs.append({"role": "completed_effect_continuation_preregistration",
                       "path": "bench/protocols/diagnostic_recovery_comparison_20261104_v8.md",
                       "sha256": _sha(ROOT / "bench/protocols/diagnostic_recovery_comparison_20261104_v8.md", lf=True)})
        if selected_v4_profile == GPU_ADMISSION_PROFILE_RETRY:
            note = "bench/protocols/diagnostic_recovery_comparison_20261104_v9.md"
            inputs.append({"role": "consistent_gpu_observation_preregistration", "path": note,
                           "sha256": _sha(ROOT / note, lf=True)})
    if v5 is not None:
        note = "bench/protocols/diagnostic_recovery_comparison_20261104_v10.md"
        inputs.append({"role": "completed_motor_effects_preregistration", "path": note,
                       "sha256": _sha(ROOT / note, lf=True)})
        inputs.append({"role": "completed_motor_effect_registry",
                       "path": COMPLETED_EFFECT_REGISTRY_RELATIVE,
                       "sha256": COMPLETED_EFFECT_REGISTRY_SHA256_LF})
    if restart_seed_overrides is not None and not v6_requested and not resource_requested:
        inputs.append({
            "role": "restart_seed_preregistration",
            "path": RESTART_SEED_PREREGISTRATION_V11,
            "sha256": closure[RESTART_SEED_PREREGISTRATION_V11],
        })
    if v6_requested:
        inputs.extend([
            {
                "role": "epoch_close_retry_preregistration",
                "path": EPOCH_CLOSE_RETRY_PREREGISTRATION_V12,
                "sha256": closure[EPOCH_CLOSE_RETRY_PREREGISTRATION_V12],
            },
            {
                "role": "epoch_shutdown_retry_policy",
                "path": EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE,
                "sha256": EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF,
            },
        ])
    if v7_requested:
        inputs.append({
            "role": "epoch_namespace_retry_preregistration",
            "path": EPOCH_NAMESPACE_RETRY_PREREGISTRATION_V13,
            "sha256": closure[EPOCH_NAMESPACE_RETRY_PREREGISTRATION_V13],
        })
    if resource_requested:
        inputs.extend([
            {
                "role": "balanced_resource_remeasurement_preregistration",
                "path": RESOURCE_REMEASUREMENT_PREREGISTRATION_V14,
                "sha256": closure[RESOURCE_REMEASUREMENT_PREREGISTRATION_V14],
            },
            {
                "role": "known_primary_outcomes",
                "path": COMPARISON.V13_PUBLIC_SUMMARY,
                "sha256": COMPARISON.V13_PUBLIC_SUMMARY_SHA256,
            },
        ])
    if v9_requested or v10_requested or v11_requested:
        inputs.append({
            "role": "balanced_resource_namespace_preregistration",
            "path": COMPARISON.V15_NOTE,
            "sha256": closure[COMPARISON.V15_NOTE],
        })
    if v10_requested or v11_requested:
        inputs.extend([
            {"role": "balanced_resource_manifest_retry_preregistration",
             "path": COMPARISON.V16_NOTE, "sha256": closure[COMPARISON.V16_NOTE]},
            {"role": "failed_resource_remeasurement",
             "path": COMPARISON.V15_FAILURE_SUMMARY,
             "sha256": COMPARISON.V15_FAILURE_SUMMARY_SHA256},
        ])
    if v11_requested:
        inputs.extend([
            {
                "role": "balanced_resource_continuation_preregistration",
                "path": COMPARISON.V17_NOTE,
                "sha256": closure[COMPARISON.V17_NOTE],
            },
            {
                "role": "prior_recovery_timing_prefix",
                "path": COMPARISON.V16_PREFIX,
                "sha256": closure[COMPARISON.V16_PREFIX],
            },
        ])
        if v11_r2_requested:
            inputs.extend([
                {
                    "role": "balanced_resource_continuation_r2_preregistration",
                    "path": COMPARISON.V17_R2_NOTE,
                    "sha256": closure[COMPARISON.V17_R2_NOTE],
                },
                {
                    "role": "restart_seed_nonexecution_registry",
                    "path": COMPARISON.V17_SEED_NONEXECUTION_REGISTRY,
                    "sha256": closure[COMPARISON.V17_SEED_NONEXECUTION_REGISTRY],
                },
            ])
        if v11_r3_requested:
            inputs.append({
                "role": "balanced_resource_continuation_r3_preregistration",
                "path": COMPARISON.V17_R3_NOTE,
                "sha256": closure[COMPARISON.V17_R3_NOTE],
            })
    for case in CASES:
        old = case
        row = phase_inputs.get(f"protocol:{old}")
        if row is None:
            raise PreparationError(f"phase-A protocol input missing: {old}")
        inputs.append({"role": f"diagnostic_protocol:{case}", "path": row["path"],
                       "sha256": row["sha256"]})
    plan = {
        "schema": (
            COMPARISON.PLAN_SCHEMA_V11
            if v11_requested
            else COMPARISON.PLAN_SCHEMA_V10
            if v10_requested
            else COMPARISON.PLAN_SCHEMA_V9
            if v9_requested
            else COMPARISON.PLAN_SCHEMA_V8
            if v8_requested
            else COMPARISON.PLAN_SCHEMA_V7
            if v7_requested
            else COMPARISON.PLAN_SCHEMA_V6 if v6_requested
            else COMPARISON.PLAN_SCHEMA_V5 if v5 is not None
            else COMPARISON.PLAN_SCHEMA_V4 if v4 is not None
            else COMPARISON.PLAN_SCHEMA_V3
            if v3 is not None
            else COMPARISON.PLAN_SCHEMA_V2 if prior is not None else COMPARISON.PLAN_SCHEMA
        ),
        "source_anchor": source_anchor,
        "source_closure": closure, "methods": list(METHODS),
        "cases": [{"case_id": case, "rotation": _rotation(i)} for i, case in enumerate(CASES)],
        "limits": {"max_recoveries": 16,
                   "gpu_sdk_launch_cap": 16 if v11_requested else 31 if v10_requested else 14 if v6_requested else 15 if v5 is not None else 27 if v4 is not None else 29 if v3 is not None else 31 if prior is not None else 32,
                   "cpu_checkpoint_launch_cap": 8 if v11_requested else 7 if v5 is not None else 16,
                   "cpu_telemetry_launch_cap": 8 if v11_requested else 7 if v5 is not None else 16,
                   "total_owned_launch_cap": 32 if v11_requested else 63 if v10_requested else 28 if v6_requested else 29 if v5 is not None else 59 if v4 is not None else 61 if v3 is not None else 63 if prior is not None else 64,
                   "max_parent_wall_s_per_recovery": 5000, "api_cap": 0},
        "inputs": inputs, "backend": investigate.REGISTERED_BACKEND,
        "authorization": {"actor": actor,
                          "ambiguous_consumed_policy": "settle_unknown_without_reexecution"},
        "recovery_policy": {"training_iterations": 300,
                            "seeds": {case: {"resume": SOURCE_SEEDS[case],
                                             "restart": restart_seeds[case]} for case in CASES},
                            "same_case_same_template": True,
                            "source_cost_policy": "phase_a_shared_count_once"},
        "template_commitments": commitments,
    }
    if resource_requested:
        plan["resource_remeasurement"] = {
            "profile_id": resource_remeasurement_profile,
            "scope": "already_seen_development_resource_check",
            "imports_allowed": False,
            "known_primary_outcomes": {
                "path": COMPARISON.V13_PUBLIC_SUMMARY,
                "sha256": COMPARISON.V13_PUBLIC_SUMMARY_SHA256,
            },
            "expected": {"planned_arms": 16, "fresh_effects": 15,
                         "diagnostic_hold": 1},
            "accounting": {
                "prior_primary_sdk_launches": 32,
                "prior_post_sdk_launches": 1,
                "prior_whole_study_sdk_launches": 33,
                "new_gpu_sdk_launch_cap": 32,
                "cumulative_gpu_sdk_launch_cap": 65,
            },
            "no_model_requery": True,
        }
        if v10_requested:
            plan["resource_remeasurement"]["known_failed_remeasurement"] = {
                "path": COMPARISON.V15_FAILURE_SUMMARY,
                "sha256": COMPARISON.V15_FAILURE_SUMMARY_SHA256,
            }
            plan["resource_remeasurement"]["accounting"] = {
                "prior_primary_sdk_launches": 32,
                "prior_post_sdk_launches": 1,
                "prior_failed_remeasurement_sdk_launches": 1,
                "prior_whole_study_sdk_launches": 34,
                "new_gpu_sdk_launch_cap": 31,
                "cumulative_gpu_sdk_launch_cap": 65,
            }
    if v11_requested:
        if v11 is None:
            raise PreparationError("v17 continuation proofs missing")
        imports, _proof_refs = v11
        prior_plan_sha = imports[0]["prior_plan_sha256"]
        prior_source_anchor = imports[0]["source_anchor"]
        prior_registration_anchor = imports[0]["registration_anchor"]
        plan.pop("resource_remeasurement", None)
        plan["continuation"] = {
            "profile_id": RESOURCE_CONTINUATION_PROFILE,
            "gpu_admission_profile": GPU_ADMISSION_PROFILE_RETRY,
            "stage_imports": imports,
            "prior_plan_sha256": prior_plan_sha,
            "prior_registration_anchor": prior_registration_anchor,
            "prior_source_anchor": prior_source_anchor,
            "prior_completed_effects": 7,
            "prior_completed_training": 1,
            "prior_consumed_gpu_sdk_launches": 15,
            "expected_new_gpu_sdk_launches": 15,
            "new_gpu_sdk_launch_cap": 16,
            "prior_primary_sdk_launches": 32,
            "prior_post_sdk_launches": 1,
            "prior_v15_failed_sdk_launches": 1,
            "prior_v16_partial_sdk_launches": 15,
            "prior_whole_study_sdk_launches": 49,
            "expected_whole_study_sdk_launches": 64,
            "cumulative_gpu_sdk_launch_cap": 65,
            "prior_incomplete_preserved": True,
            "no_model_requery": True,
        }
    if prior is not None:
        _proof_ref, logical = prior
        plan["continuation"] = {
            "gpu_admission_profile": GPU_ADMISSION_PROFILE,
            "prior_training": logical,
            "prior_consumed_gpu_sdk_launches": 1,
            "prior_training_gpu_s": 515.3279999999795,
            "expected_new_gpu_sdk_launches": 29,
            "continuation_tag": "v5_fresh",
            "prior_failed_attempt_preserved": True,
        }
    if v3 is not None:
        imports, _proof_refs = v3
        plan["continuation"] = {
            "gpu_admission_profile": selected_v3_profile,
            "stage_imports": imports,
            "prior_consumed_gpu_sdk_launches": 3,
            "expected_new_gpu_sdk_launches": 27,
            "continuation_tag": _V3_PROFILE_TAGS[selected_v3_profile],
            "prior_v5_registration_anchor": PRIOR_V5_REGISTRATION_ANCHOR,
            "prior_incomplete_preserved": True,
        }
    if v4 is not None:
        imports, _proof_refs = v4
        plan["continuation"] = {
            "gpu_admission_profile": selected_v4_profile,
            "stage_imports": imports,
            "prior_consumed_gpu_sdk_launches": 5,
            "expected_new_gpu_sdk_launches": 26,
            "cumulative_gpu_sdk_launches": 32,
            "continuation_tag": _V4_PROFILE_TAGS[selected_v4_profile],
            "prior_v7_registration_anchor": PRIOR_V7_REGISTRATION_ANCHOR,
            "prior_incomplete_preserved": True,
        }
    if v5 is not None:
        imports, _proof_refs = v5
        _registry, _entries = _v5_registry()
        plan["continuation"] = {
            "gpu_admission_profile": GPU_ADMISSION_PROFILE_RETRY,
            "completed_import_profile": COMPLETED_IMPORT_PROFILE_MOTOR,
            "completed_effect_registry_sha256_lf": COMPLETED_EFFECT_REGISTRY_SHA256_LF,
            "stage_imports": imports,
            "prior_consumed_gpu_sdk_launches": 17,
            "expected_new_gpu_sdk_launches": 14,
            "new_gpu_sdk_launch_cap": 15,
            "cumulative_gpu_sdk_launches": 32,
            "continuation_tag": "v10_epoch_repair",
            "prior_v9_registration_anchor": "fbd0bdd929e68933fdd2d8d354f7d5447625b101",
            "prior_incomplete_preserved": True,
            "v9_failed_epoch_sdk_launches": 0,
        }
    if v6_requested:
        if v5 is None:
            raise PreparationError("v12 completed imports missing")
        imports, _proof_refs = v5
        policy = _epoch_shutdown_retry_policy()
        plan["continuation"] = {
            "gpu_admission_profile": GPU_ADMISSION_PROFILE_RETRY,
            "completed_import_profile": COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY,
            "completed_effect_registry_sha256_lf": COMPLETED_EFFECT_REGISTRY_SHA256_LF,
            "epoch_shutdown_retry_policy_sha256_lf": EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF,
            "failure_proof_sha256": policy["failure_proof_sha256"],
            "stage_imports": imports,
            "prior_consumed_gpu_sdk_launches": 18,
            "expected_new_gpu_sdk_launches": 14,
            "new_gpu_sdk_launch_cap": 14,
            "cumulative_gpu_sdk_launches": 32,
            "continuation_tag": "v12_epoch_close_retry",
            "prior_v11_registration_anchor": policy["v11_failure"]["identity"][
                "registration_anchor"
            ],
            "prior_incomplete_preserved": True,
            "v11_failed_epoch_sdk_launches": 1,
        }
    if v7_requested:
        plan["continuation"].update({
            "completed_import_profile": COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY,
            "namespace_failure_context_sha256": EPOCH_NAMESPACE_FAILURE.CONTEXT_SHA256,
            "continuation_tag": "v13_epoch_namespace_retry",
        })
    plan_path = out / "registered-plan.json"
    write_new(plan_path, plan)
    COMPARISON.load_plan(plan_path, repo_root=ROOT, registration_anchor=source_anchor,
                         require_committed=False)
    manifest = {"schema": TEMPLATE_SCHEMA, "plan": _ref(plan_path),
                "diagnostic_plan": phase_ref, "source_index": _ref(source_index),
                "metadata": _ref(metadata), "sdk_contract": _ref(sdk_contract),
                "recipe_code_contract": _ref(recipe_code_contract),
                "development_source_seeds": development_sources,
                "registered_source_seeds": SOURCE_SEEDS,
                "actual_training_started": False, "api_calls": 0}
    if prior is not None:
        manifest["prior_training_proof"] = prior[0]
    if v3 is not None:
        manifest["continuation_proofs"] = v3[1]
    if v4 is not None:
        manifest["continuation_proofs"] = v4[1]
    if v5 is not None:
        manifest["continuation_proofs"] = v5[1]
    if v11 is not None:
        manifest["continuation_proofs"] = v11[1]
    if seed_proof_refs is not None:
        manifest["nonexecution_seed_proofs"] = seed_proof_refs
    if failure_proof is not None:
        manifest["epoch_shutdown_failure_proof"] = failure_proof[0]
    if namespace_failure is not None:
        manifest["epoch_namespace_failure_context"] = namespace_failure["context_ref"]
    write_new(out / "manifest.json", manifest)
    return manifest


def _diagnostic_status(session: Path) -> dict:
    current = InvestigationSession(session)
    current._verify_operation(__import__("rl_triage.registered_diagnostic_backend", fromlist=["verify_source"]).verify_source)
    if current.manifest.get("execution_class") != "actual":
        raise PreparationError("diagnostic session is not actual")
    return terminal_aware_status(current)


def _bound_diagnostic_sessions(diagnostic_root: Path, plan: dict) -> dict[tuple[str, str], Path]:
    """Resolve the exact phase-A bound session layout and recheck its immutable binding."""
    root_input = Path(diagnostic_root)
    root = root_input.resolve(strict=True)
    if root_input.is_symlink() or not root.is_dir():
        raise PreparationError("diagnostic comparison root is not a real directory")
    bound = root / "bound"
    if not bound.is_dir() or bound.is_symlink() or bound.resolve(strict=True) != bound:
        raise PreparationError("diagnostic bound root is missing or aliased")
    sessions: dict[tuple[str, str], Path] = {}
    resolved_paths: set[Path] = set()
    for case in CASES:
        commitments = plan["template_commitments"][case]
        checkpoints = {row["source_checkpoint_sha256"] for row in commitments.values()}
        if len(checkpoints) != 1:
            raise PreparationError("case template commitments disagree on source checkpoint")
        checkpoint = checkpoints.pop()
        for method in METHODS:
            legacy_manifest = root / case / method / "manifest.json"
            if legacy_manifest.exists():
                raise PreparationError("legacy diagnostic session layout is forbidden")
            base = bound / case / method
            source_path = base / "source.json"
            session = base / "session"
            manifest_path = session / "manifest.json"
            for path, kind in ((base, "directory"), (source_path, "file"),
                               (session, "directory"), (manifest_path, "file")):
                if path.is_symlink():
                    raise PreparationError("diagnostic arm path is aliased")
                if kind == "directory" and not path.is_dir():
                    raise PreparationError(f"diagnostic session missing: {case}/{method}")
                if kind == "file" and not path.is_file():
                    raise PreparationError(f"diagnostic manifest or source missing: {case}/{method}")
                resolved = path.resolve(strict=True)
                if resolved != path or resolved in resolved_paths:
                    raise PreparationError("diagnostic arm paths overlap or are aliased")
                resolved_paths.add(resolved)
            source = _read(source_path)
            if source.get("case_id") != case or source.get("checkpoint_sha256") != checkpoint:
                raise PreparationError("diagnostic source case or checkpoint differs")
            try:
                current = InvestigationSession(session)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise PreparationError("diagnostic session binding differs from phase-A arm") from exc
            source_ref = current.manifest.get("source")
            verification = current.manifest.get("source_verification")
            if (current.folder != session
                    or current.manifest.get("method") != method
                    or current.manifest.get("execution_class") != "actual"
                    or not isinstance(source_ref, dict)
                    or set(source_ref) != {"path", "sha256"}
                    or Path(source_ref["path"]).resolve(strict=True) != source_path
                    or source_ref["sha256"] != _sha(source_path)
                    or not isinstance(verification, dict)
                    or verification.get("case_id") != case
                    or verification.get("checkpoint_sha256") != checkpoint
                    or current.source.get("case_id") != case
                    or current.source.get("checkpoint_sha256") != checkpoint):
                raise PreparationError("diagnostic session binding differs from phase-A arm")
            sessions[(case, method)] = session
    return sessions


def _approval(protocol_ref: dict, protocol: dict, proposal: dict, actor: str) -> dict:
    request = proposal.get("status", {}).get("pending")
    if not isinstance(request, dict):
        raise PreparationError("recovery proposal missing pending request")
    diagnostic, operation = protocol["diagnostic"], protocol["operation"]
    return {"schema": "registered_recovery_approval_v1", "actor": actor,
            "request_id": request["request_id"], "effect_key": request["effect_key"],
            "protocol_sha256": protocol_ref["sha256"], "registry_sha256": registry_digest(),
            "source_sha256": diagnostic["source_sha256"],
            "checkpoint_sha256": diagnostic["checkpoint_sha256"],
            "mechanism": diagnostic["mechanism"], "correction_id": operation["correction_id"],
            "recovery_mode": operation["recovery_mode"], "gpu_budget_s": operation["gpu_budget_s"]}


def prepare_jobs(*, templates: Path, registered_plan: Path, registration_anchor: str,
                 diagnostic_root: Path, isaaclab_root: Path, output: Path,
                 require_committed: bool = True) -> dict:
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("job output must be fresh")
    template_root = Path(templates).resolve(strict=True)
    template_manifest = _read(template_root / "manifest.json")
    if template_manifest.get("schema") != TEMPLATE_SCHEMA:
        raise PreparationError("template manifest schema")
    plan = COMPARISON.load_plan(registered_plan, repo_root=ROOT,
                                registration_anchor=registration_anchor,
                                require_committed=require_committed)
    if _sha(registered_plan) != template_manifest["plan"]["sha256"]:
        raise PreparationError("registered plan differs from prepared template")
    continuation = (
        plan.get("continuation")
        if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V2, COMPARISON.PLAN_SCHEMA_V3,
                              COMPARISON.PLAN_SCHEMA_V4, COMPARISON.PLAN_SCHEMA_V5,
                              COMPARISON.PLAN_SCHEMA_V6, COMPARISON.PLAN_SCHEMA_V7,
                              COMPARISON.PLAN_SCHEMA_V11}
        else None
    )
    prior_ref = template_manifest.get("prior_training_proof")
    if plan["schema"] == COMPARISON.PLAN_SCHEMA_V2:
        if not isinstance(prior_ref, dict):
            raise PreparationError("v2 template prior training proof missing")
        verified_ref, logical = _prior_training(
            Path(str(prior_ref["path"])), Path(isaaclab_root).resolve(strict=True)
        )
        if verified_ref != prior_ref or not isinstance(continuation, dict) or logical != continuation.get(
            "prior_training"
        ):
            raise PreparationError("v2 prior training proof differs from public plan")
    elif prior_ref is not None:
        raise PreparationError("v1 template must not carry prior training proof")
    v3_proofs = template_manifest.get("continuation_proofs")
    v3_proof_refs: dict[str, dict] = {}
    if plan["schema"] == COMPARISON.PLAN_SCHEMA_V3:
        if not isinstance(v3_proofs, dict) or set(v3_proofs) != {"prior_effect", "prior_training"}:
            raise PreparationError("v3 template continuation proofs missing")
        v3_proof_refs = v3_proofs
        imports, verified_refs = _v3_imports(
            Path(v3_proof_refs["prior_effect"]["path"]),
            Path(v3_proof_refs["prior_training"]["path"]),
            Path(isaaclab_root),
        )
        if (
            not isinstance(continuation, dict)
            or imports != continuation.get("stage_imports")
            or verified_refs != v3_proof_refs
        ):
            raise PreparationError("v3 continuation proofs differ from public plan")
    elif plan["schema"] == COMPARISON.PLAN_SCHEMA_V4:
        if not isinstance(v3_proofs, dict) or set(v3_proofs) != set(V4_ARTIFACTS):
            raise PreparationError("v4 template continuation proofs missing")
        v3_proof_refs = v3_proofs
        imports, verified_refs = _v4_imports({
            artifact_id: Path(v3_proof_refs[artifact_id]["path"])
            for artifact_id in V4_ARTIFACTS
        })
        if (not isinstance(continuation, dict)
                or imports != continuation.get("stage_imports")
                or verified_refs != v3_proof_refs):
            raise PreparationError("v4 continuation proofs differ from public plan")
    elif plan["schema"] in {COMPARISON.PLAN_SCHEMA_V5, COMPARISON.PLAN_SCHEMA_V6,
                             COMPARISON.PLAN_SCHEMA_V7}:
        if not isinstance(v3_proofs, dict):
            raise PreparationError("completed-import template continuation proofs missing")
        v3_proof_refs = v3_proofs
        imports, verified_refs = _v5_imports({
            artifact_id: Path(reference["path"])
            for artifact_id, reference in v3_proof_refs.items()
        })
        if (not isinstance(continuation, dict)
                or imports != continuation.get("stage_imports")
                or verified_refs != v3_proof_refs):
            raise PreparationError("completed-import proofs differ from public plan")
    elif plan["schema"] == COMPARISON.PLAN_SCHEMA_V11:
        expected_artifacts = {
            *(_v17_artifact_id(case_id, method, "completed_effect")
              for case_id, method in V17_EFFECT_KEYS),
            "prior_training",
        }
        if not isinstance(v3_proofs, dict) or set(v3_proofs) != expected_artifacts:
            raise PreparationError("v17 template continuation proofs missing")
        v3_proof_refs = v3_proofs
        effect_paths = {
            artifact_id: Path(reference["path"])
            for artifact_id, reference in v3_proof_refs.items()
            if artifact_id != "prior_training"
        }
        imports, verified_refs = _v17_imports(
            effect_paths, Path(v3_proof_refs["prior_training"]["path"]), Path(isaaclab_root),
        )
        if (not isinstance(continuation, dict)
                or imports != continuation.get("stage_imports")
                or verified_refs != v3_proof_refs):
            raise PreparationError("v17 continuation proofs differ from public plan")
    elif v3_proofs is not None:
        raise PreparationError("non-v3 template must not carry continuation proofs")
    failure_ref = template_manifest.get("epoch_shutdown_failure_proof")
    if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V6, COMPARISON.PLAN_SCHEMA_V7}:
        if not isinstance(failure_ref, dict):
            raise PreparationError("v6 epoch shutdown failure proof missing")
        verified_failure_ref, _proof = _v12_failure_proof(Path(str(failure_ref.get("path"))))
        if verified_failure_ref != failure_ref:
            raise PreparationError("v6 epoch shutdown failure proof reference differs")
    elif failure_ref is not None:
        raise PreparationError("non-v6 template must not carry epoch shutdown failure proof")
    namespace_ref = template_manifest.get("epoch_namespace_failure_context")
    if plan["schema"] == COMPARISON.PLAN_SCHEMA_V7:
        if not isinstance(namespace_ref, dict):
            raise PreparationError("v7 epoch namespace failure context missing")
        verified_namespace = EPOCH_NAMESPACE_FAILURE.verify_failure_context(namespace_ref)
        if verified_namespace["context_ref"] != namespace_ref:
            raise PreparationError("v7 namespace failure context reference differs")
    elif namespace_ref is not None:
        raise PreparationError("non-v7 template must not carry namespace failure context")
    freshness_cases = (
        ("case-07", "case-08")
        if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V5, COMPARISON.PLAN_SCHEMA_V6,
                              COMPARISON.PLAN_SCHEMA_V7, COMPARISON.PLAN_SCHEMA_V8,
                              COMPARISON.PLAN_SCHEMA_V9, COMPARISON.PLAN_SCHEMA_V10,
                              COMPARISON.PLAN_SCHEMA_V11}
        else CASES
    )
    seed_proof_inputs = template_manifest.get("nonexecution_seed_proofs")
    r2_requested = any(
        row.get("role") == "restart_seed_nonexecution_registry" for row in plan["inputs"]
    )
    if r2_requested:
        if not isinstance(seed_proof_inputs, dict):
            raise PreparationError("v17 r2 template seed proof is missing")
        verified_seed_refs = _nonexecution_seed_proof_refs({
            logical_id: Path(str(reference.get("path")))
            for logical_id, reference in seed_proof_inputs.items()
            if isinstance(reference, dict)
        })
        if verified_seed_refs != seed_proof_inputs:
            raise PreparationError("v17 r2 template seed proof reference differs")
        freshness = _seed_guard_v3().assert_fresh_seeds(
            [plan["recovery_policy"]["seeds"][case]["restart"] for case in freshness_cases],
            Path(isaaclab_root), repo=ROOT, nonexecution_proofs=seed_proof_inputs,
        )
    else:
        if seed_proof_inputs is not None:
            raise PreparationError("non-r2 template carries a seed proof")
        freshness = SEED_GUARD.assert_fresh_seeds(
            [plan["recovery_policy"]["seeds"][case]["restart"] for case in freshness_cases],
            Path(isaaclab_root), repo=ROOT,
        )
    out.mkdir(parents=True)
    write_new(out / "restart-seed-freshness.json", freshness)
    diagnostic_sessions = _bound_diagnostic_sessions(diagnostic_root, plan)
    arms = []
    case_rows = {row["case_id"]: row for row in plan["cases"]}
    for case in CASES:
        ordered_methods = (
            tuple(case_rows[case]["rotation"])
            if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V5, COMPARISON.PLAN_SCHEMA_V6,
                                  COMPARISON.PLAN_SCHEMA_V7, COMPARISON.PLAN_SCHEMA_V8,
                                  COMPARISON.PLAN_SCHEMA_V9, COMPARISON.PLAN_SCHEMA_V10,
                                  COMPARISON.PLAN_SCHEMA_V11}
            else METHODS
        )
        for method in ordered_methods:
            session = diagnostic_sessions[(case, method)]
            manifest_path = session / "manifest.json"
            status = _diagnostic_status(session)
            expected_session_case = case
            if status.get("case_id") != expected_session_case:
                raise PreparationError("diagnostic case mapping differs")
            arm = {"case_id": case, "method": method, "diagnostic_manifest": _ref(manifest_path),
                   "recovery_folder": None, "recovery_payload": None, "approval_ref": None}
            if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V3, COMPARISON.PLAN_SCHEMA_V4,
                                  COMPARISON.PLAN_SCHEMA_V5, COMPARISON.PLAN_SCHEMA_V6,
                                  COMPARISON.PLAN_SCHEMA_V7, COMPARISON.PLAN_SCHEMA_V11}:
                arm["continuation_import"] = None
            mechanism = status.get("confirmed_mechanism") if status.get("status") == "confirmed" else None
            if mechanism in MECHANISMS:
                import_row = next((
                    row for row in continuation["stage_imports"]
                    if row["case_id"] == case and row["method"] == method
                ), None) if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V3,
                                                COMPARISON.PLAN_SCHEMA_V4,
                                                COMPARISON.PLAN_SCHEMA_V5,
                                                COMPARISON.PLAN_SCHEMA_V6,
                                                COMPARISON.PLAN_SCHEMA_V7,
                                                COMPARISON.PLAN_SCHEMA_V11} \
                    and isinstance(continuation, dict) else None
                commitment = plan["template_commitments"][case][mechanism]
                case_dir = template_root / "templates" / case
                training = case_dir / mechanism / "training-binding.json"
                fixed = case_dir / "fixed-binding.json"
                if _sha(training) != commitment["training_binding_sha256"] or _sha(fixed) != commitment["fixed_binding_sha256"]:
                    raise PreparationError("template commitment changed")
                source_session = InvestigationSession(session)
                if import_row is not None and (
                    source_session._manifest_sha256 != import_row["diagnostic_manifest_sha256"]
                    or status["source_verification"]["checkpoint_sha256"]
                    != import_row["source_checkpoint_sha256"]
                    or mechanism != "actuator"
                ):
                    raise PreparationError("continuation import diagnostic/source binding differs")
                if import_row is not None:
                    arm["continuation_import"] = {
                        "kind": import_row["kind"],
                        "artifact_id": import_row["artifact_id"],
                        "proof": deepcopy(v3_proof_refs[import_row["artifact_id"]]),
                    }
                    if import_row["kind"] == "completed_effect":
                        arms.append(arm)
                        continue
                request, _intent, diagnostic = RECOVERY._diagnostic(source_session, mechanism)
                mode = REGISTRY[mechanism]["recovery_modes"][0]
                operation = {"correction_id": REGISTRY[mechanism]["correction_id"],
                             "recovery_mode": mode,
                             "runner_id": {"actuator": "actuator_resume_v1",
                                           "optimizer_update_count": "r1_epochs_preserving_environment_v1",
                                           "optimizer_surrogate": "ppo_ratio_v1"}[mechanism],
                             "training_seed": plan["recovery_policy"]["seeds"][case][mode],
                             "training_iterations": 300, "gpu_budget_s": 3600,
                             "source_checkpoint_consumed": mechanism == "actuator",
                             "training_binding_artifact_id": "training",
                             "fixed_binding_artifact_id": "fixed"}
                use_prior = (
                    plan["schema"] == COMPARISON.PLAN_SCHEMA_V2
                    and case == "case-05"
                    and method == "fixed_sequence"
                    and mechanism == "actuator"
                )
                use_prior_v3 = import_row is not None and import_row["kind"] == "completed_training"
                if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V2, COMPARISON.PLAN_SCHEMA_V3,
                                      COMPARISON.PLAN_SCHEMA_V4, COMPARISON.PLAN_SCHEMA_V5,
                                      COMPARISON.PLAN_SCHEMA_V6, COMPARISON.PLAN_SCHEMA_V7,
                                      COMPARISON.PLAN_SCHEMA_V8, COMPARISON.PLAN_SCHEMA_V9,
                                      COMPARISON.PLAN_SCHEMA_V10, COMPARISON.PLAN_SCHEMA_V11}:
                    operation.update({
                        "gpu_admission_profile": (
                            GPU_ADMISSION_PROFILE_RETRY
                            if plan["schema"] in {
                                COMPARISON.PLAN_SCHEMA_V8, COMPARISON.PLAN_SCHEMA_V9,
                                COMPARISON.PLAN_SCHEMA_V10}
                            else plan["continuation"]["gpu_admission_profile"]
                            if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V3,
                                                  COMPARISON.PLAN_SCHEMA_V4,
                                                  COMPARISON.PLAN_SCHEMA_V5,
                                                  COMPARISON.PLAN_SCHEMA_V6,
                                                  COMPARISON.PLAN_SCHEMA_V7,
                                                  COMPARISON.PLAN_SCHEMA_V11}
                            else GPU_ADMISSION_PROFILE
                        ),
                        "prior_training_artifact_id": (
                            "prior_training" if use_prior or use_prior_v3 else None
                        ),
                    })
                if use_prior:
                    if not isinstance(continuation, dict) or not isinstance(prior_ref, dict):
                        raise PreparationError("v2 prior training state missing")
                    prior_logical = continuation["prior_training"]
                    if (
                        source_session._manifest_sha256
                        != prior_logical["diagnostic_manifest_sha256"]
                        or status["source_verification"]["checkpoint_sha256"]
                        != prior_logical["source_checkpoint_sha256"]
                    ):
                        raise PreparationError("prior training diagnostic/source binding differs")
                protocol = {"schema": (
                            "registered_recovery_protocol_v4"
                                if (plan["schema"] == COMPARISON.PLAN_SCHEMA_V11
                                    and use_prior_v3)
                                else "registered_recovery_protocol_v3"
                                if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V3,
                                                      COMPARISON.PLAN_SCHEMA_V4,
                                                      COMPARISON.PLAN_SCHEMA_V5,
                                                      COMPARISON.PLAN_SCHEMA_V6,
                                                      COMPARISON.PLAN_SCHEMA_V7,
                                                      COMPARISON.PLAN_SCHEMA_V8,
                                                      COMPARISON.PLAN_SCHEMA_V9,
                                                      COMPARISON.PLAN_SCHEMA_V10,
                                                      COMPARISON.PLAN_SCHEMA_V11}
                                else "registered_recovery_protocol_v2"
                                if plan["schema"] == COMPARISON.PLAN_SCHEMA_V2
                                else "registered_recovery_protocol_v1"
                            ),
                            "backend_id": "registered_recovery_v1",
                            "source_anchor": plan["source_anchor"], "source_closure": plan["source_closure"],
                            "diagnostic": {"session_manifest_sha256": source_session._manifest_sha256,
                                           "source_sha256": source_session.manifest["source"]["sha256"],
                                           "checkpoint_sha256": status["source_verification"]["checkpoint_sha256"],
                                           "case_id": expected_session_case, "mechanism": mechanism,
                                           "probe": request["probe"],
                                           "diagnostic_receipt_sha256": diagnostic["receipt_ref"]["sha256"]},
                            "operation": operation, "registry_sha256": registry_digest(),
                            "actors": [plan["authorization"]["actor"]],
                            "limits": {"training_watchdog_s": 2700.0, "fixed_watchdog_s": 900.0}}
                arm_dir = out / "artifacts" / case / method
                protocol_path = arm_dir / "recovery-protocol.json"
                resolver_path = arm_dir / "resolver.json"
                write_new(protocol_path, protocol)
                artifacts = {"training": _ref(training), "fixed": _ref(fixed)}
                if use_prior:
                    if not isinstance(prior_ref, dict):
                        raise PreparationError("v2 prior training proof missing")
                    artifacts["prior_training"] = deepcopy(prior_ref)
                if use_prior_v3:
                    artifacts["prior_training"] = deepcopy(v3_proof_refs["prior_training"])
                write_new(resolver_path, {"schema": "registered_recovery_resolver_v1",
                                          "artifacts": artifacts})
                payload = {"protocol": _ref(protocol_path), "artifact_resolver": _ref(resolver_path),
                           "isaaclab_root": str(Path(isaaclab_root).resolve(strict=True)),
                           "output_root": str((arm_dir / "execution").resolve())}
                recovery_folder = out / "prepared-recoveries" / case / method
                prepared = investigate.run("recovery-prepare", session,
                                           {"recovery_payload": payload, "folder": str(recovery_folder)},
                                           backend_name=investigate.REGISTERED_BACKEND, fixture=False)
                if prepared.get("mechanism") != mechanism:
                    raise PreparationError("recovery prepare mechanism mismatch")
                proposed = investigate.run("recovery-propose", recovery_folder, {},
                                           backend_name=investigate.REGISTERED_BACKEND, fixture=False)
                approval_path = arm_dir / "approval.json"
                write_new(approval_path, _approval(payload["protocol"], protocol, proposed,
                                                   plan["authorization"]["actor"]))
                arm.update({"recovery_folder": str(recovery_folder.resolve()),
                            "recovery_payload": payload, "approval_ref": _ref(approval_path)})
            arms.append(arm)
    job = {"schema": JOB_SCHEMA, "plan": _ref(registered_plan), "arms": arms}
    if plan["schema"] in {COMPARISON.PLAN_SCHEMA_V6, COMPARISON.PLAN_SCHEMA_V7}:
        job["epoch_shutdown_failure_proof"] = failure_ref
    if plan["schema"] == COMPARISON.PLAN_SCHEMA_V7:
        job["epoch_namespace_failure_context"] = namespace_ref
    job_path = out / "job.json"
    write_new(job_path, job)
    # Validate the producer output, including every protocol, resolver, template, and approval.
    COMPARISON._validate_job(job, job["plan"], plan, job_path)
    return {"schema": "registered_recovery_comparison_job_preparation_v1", "job": _ref(job_path),
            "arms": len(arms), "recovery_prepared": sum(row["recovery_payload"] is not None for row in arms),
            "approval_consumed": False, "dispatch_started": False, "api_calls": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("templates")
    t.add_argument("--source-index", required=True, type=Path)
    t.add_argument("--metadata", required=True, type=Path)
    t.add_argument("--diagnostic-plan", required=True, type=Path)
    t.add_argument("--diagnostic-registration-anchor", required=True)
    t.add_argument("--source-anchor", required=True)
    t.add_argument("--sdk-contract", required=True, type=Path)
    t.add_argument("--recipe-code-contract", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--actor", required=True)
    t.add_argument("--isaaclab-commit", required=True)
    t.add_argument("--prior-training-proof", type=Path)
    t.add_argument("--prior-completed-effect-proof", type=Path)
    t.add_argument("--prior-static-completed-effect-proof", type=Path)
    t.add_argument("--prior-static-training-proof", type=Path)
    t.add_argument("--isaaclab-root", type=Path)
    t.add_argument("--gpu-admission-profile", choices=tuple(dict.fromkeys((*_V3_PROFILE_TAGS, *_V4_PROFILE_TAGS))))
    t.add_argument("--completed-import-profile", choices=(
        COMPLETED_IMPORT_PROFILE_MOTOR,
        COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY,
        COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY,
    ))
    t.add_argument("--resource-remeasurement-profile", choices=(
        RESOURCE_REMEASUREMENT_PROFILE,
        RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE,
        RESOURCE_MANIFEST_RETRY_PROFILE,
        RESOURCE_CONTINUATION_PROFILE,
    ))
    t.add_argument("--completed-effect-proof", action="append", default=[],
                   metavar="ARTIFACT_ID=PATH")
    t.add_argument("--completed-training-proof", type=Path)
    t.add_argument("--epoch-shutdown-failure-proof", type=Path)
    t.add_argument("--epoch-namespace-failure-context", type=Path)
    t.add_argument("--allow-uncommitted-fixture", action="store_true")
    j = sub.add_parser("jobs")
    j.add_argument("--templates", required=True, type=Path)
    j.add_argument("--registered-plan", required=True, type=Path)
    j.add_argument("--registration-anchor", required=True)
    j.add_argument("--diagnostic-root", required=True, type=Path)
    j.add_argument("--isaaclab-root", required=True, type=Path)
    j.add_argument("--out", required=True, type=Path)
    j.add_argument("--allow-uncommitted-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "templates":
            profile_proofs: dict[str, Path] = {}
            for value in args.completed_effect_proof:
                artifact_id, separator, raw_path = value.partition("=")
                if not separator or not artifact_id or artifact_id in profile_proofs or not raw_path:
                    raise PreparationError("completed effect proof argument differs")
                profile_proofs[artifact_id] = Path(raw_path)
            if (profile_proofs and args.completed_import_profile is None
                    and args.resource_remeasurement_profile != RESOURCE_CONTINUATION_PROFILE):
                raise PreparationError("completed effect proof profile missing")
            if ((args.prior_static_completed_effect_proof is None)
                    != (args.prior_completed_effect_proof is None)
                    and args.prior_static_training_proof is None):
                raise PreparationError("v4 completed effect proofs must be supplied together")
            completed_effect_proofs = (
                profile_proofs
                if args.completed_import_profile is not None
                or args.resource_remeasurement_profile == RESOURCE_CONTINUATION_PROFILE
                else
                {"prior_effect": args.prior_completed_effect_proof,
                 "prior_static_effect": args.prior_static_completed_effect_proof}
                if args.prior_static_completed_effect_proof is not None
                and args.prior_completed_effect_proof is not None
                else None
            )
            result = prepare_templates(source_index=args.source_index, metadata=args.metadata,
                diagnostic_plan=args.diagnostic_plan,
                diagnostic_registration_anchor=args.diagnostic_registration_anchor,
                source_anchor=args.source_anchor, sdk_contract=args.sdk_contract, output=args.out,
                recipe_code_contract=args.recipe_code_contract,
                actor=args.actor, isaaclab_commit=args.isaaclab_commit,
                prior_training_proof=args.prior_training_proof,
                prior_completed_effect_proof=(
                    None if completed_effect_proofs is not None
                    else args.prior_completed_effect_proof
                ),
                prior_static_training_proof=args.prior_static_training_proof,
                completed_effect_proofs=completed_effect_proofs,
                completed_training_proof=args.completed_training_proof,
                completed_import_profile=args.completed_import_profile,
                restart_seed_overrides=(
                    REGISTERED_RESTART_SEED_OVERRIDES_V16
                    if args.resource_remeasurement_profile in {
                        RESOURCE_MANIFEST_RETRY_PROFILE, RESOURCE_CONTINUATION_PROFILE,
                    }
                    else REGISTERED_RESTART_SEED_OVERRIDES_V14
                    if args.resource_remeasurement_profile in {
                        RESOURCE_REMEASUREMENT_PROFILE,
                        RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE,
                    }
                    else REGISTERED_RESTART_SEED_OVERRIDES_V13
                    if args.completed_import_profile
                    == COMPLETED_IMPORT_PROFILE_EPOCH_NAMESPACE_RETRY
                    else REGISTERED_RESTART_SEED_OVERRIDES_V12
                    if args.completed_import_profile
                    == COMPLETED_IMPORT_PROFILE_EPOCH_CLOSE_RETRY
                    else None
                ),
                epoch_shutdown_failure_proof=args.epoch_shutdown_failure_proof,
                epoch_namespace_failure_context=args.epoch_namespace_failure_context,
                resource_remeasurement_profile=args.resource_remeasurement_profile,
                isaaclab_root=args.isaaclab_root,
                gpu_admission_profile=args.gpu_admission_profile,
                require_committed=not args.allow_uncommitted_fixture)
        else:
            result = prepare_jobs(templates=args.templates, registered_plan=args.registered_plan,
                registration_anchor=args.registration_anchor, diagnostic_root=args.diagnostic_root,
                isaaclab_root=args.isaaclab_root, output=args.out,
                require_committed=not args.allow_uncommitted_fixture)
    except Exception as error:  # noqa: BLE001 - CLI emits a bounded failure record.
        result = {"status": "rejected", "error_type": type(error).__name__, "reason": str(error)}
        print(_canonical(result).decode())
        return 2
    print(_canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
