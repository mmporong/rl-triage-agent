"""Import one completed V16 actuator training stage for a fixed-only continuation.

The proof keeps the failed V16 request immutable.  It validates the completed
training and its failed pre-evaluation admission, then returns the checkpoint to a
new registered request without launching training or consuming the old ledger.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never

from bench import registered_recovery_training_v1 as TRAIN
from evals import registered_prior_training_v1 as PRIOR
from evals import registered_recovery_continuation_v3 as V3
from evals import registered_recovery_continuation_v4 as V4

TRAINING_SCHEMA = "registered_completed_training_proof_v2"
COMPLETED_SCHEMA = "completed_registered_actuator_training_v2"
_IMPORT_FIELDS = {
    "artifact_id", "kind", "case_id", "method", "proof_sha256", "prior_plan_sha256",
    "diagnostic_manifest_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
    "seed", "task", "correction_id", "source_anchor", "registration_anchor",
}
_PROOF_FIELDS = {
    "schema", "status", "case_id", "method", "source_anchor", "registration_anchor",
    "request_id", "training_stage", "recovery_folder", "identity", "refs",
    "stage_files", "recovery_files", "external_files", "claims",
}
_IDENTITY_FIELDS = {
    "prior_plan_sha256", "diagnostic_manifest_sha256", "source_checkpoint_sha256",
    "final_checkpoint_sha256", "seed", "task", "correction_id", "source_anchor",
    "registration_anchor",
}
_CLAIMS = {
    "new_training_launched": False,
    "fixed_evaluation_executed": False,
    "recovery_label_available": False,
    "request_reconsumed": False,
    "full_environment_state_restored": False,
    "rng_state_restored_from_checkpoint": False,
    "policy_optimizer_resumed": True,
}
_STAGE_FILES = (*PRIOR.REQUIRED_STAGE_FILES, "run-namespace.json")
_RECOVERY_FILES = PRIOR.REQUIRED_RECOVERY_FILES
_REQUEST_FILES = (
    "approvals/{request_id}.json",
    "admissions/{request_id}.json",
    "intents/{request_id}.json",
    "resources/{request_id}.json",
    "dispatch_errors/{request_id}.json",
    "authority_costs/{request_id}.json",
)


class ContinuationProofError(V4.ContinuationProofError):
    """Raised when the prior training or failed fixed admission is not immutable."""


def verify_completed_effect(
    proof_ref: Mapping[str, Any],
    *,
    expected_import: Mapping[str, Any],
    repo: Path,
) -> dict[str, Any]:
    """Validate completed-effect imports with the established V4 contract."""
    return V4.verify_completed_effect(
        proof_ref,
        expected_import=expected_import,
        repo=repo,
    )


def _fail(message: str) -> Never:
    raise ContinuationProofError(message)


def _ref(path: Path) -> dict[str, str]:
    return V3._ref(Path(path))


def _load_ref(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        return V3._load_ref(value, label=label)
    except (OSError, ValueError, TypeError) as exc:
        raise ContinuationProofError(str(exc)) from exc


def _arm(manifest: Mapping[str, Any], case_id: str, method: str) -> Mapping[str, Any]:
    try:
        return V4._arm(manifest, case_id, method)
    except ValueError as exc:
        raise ContinuationProofError(str(exc)) from exc


def _request_refs(recovery: Path, request_id: str) -> dict[str, dict[str, str]]:
    refs = {name: _ref(recovery / name) for name in _RECOVERY_FILES}
    for pattern in _REQUEST_FILES:
        relative = pattern.format(request_id=request_id)
        refs[relative] = _ref(recovery / relative)
    resource = V3._json(recovery / f"resources/{request_id}.json")
    sampler = resource.get("sampler_ref")
    if not isinstance(sampler, Mapping):
        _fail("prior device sampler reference missing")
    refs["device_sampler_stop"] = {
        "path": str(V3._validate_ref(sampler, label="prior device sampler")),
        "sha256": str(sampler["sha256"]),
    }
    return refs


def _external_refs(binding: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    try:
        return {
            "source_checkpoint": _ref(Path(str(binding["resume_checkpoint"]["path"]))),
            "source_environment_config": _ref(Path(str(binding["resume_env_config"]["path"]))),
            "source_agent_config": _ref(Path(str(binding["resume_agent_config"]["path"]))),
            "checkpoint": _ref(Path(str(receipt["checkpoint"]["path"]))),
            "environment_config": _ref(Path(str(receipt["configs"]["environment"]["path"]))),
            "agent_config": _ref(Path(str(receipt["configs"]["agent"]["path"]))),
        }
    except (KeyError, TypeError, OSError) as exc:
        raise ContinuationProofError("completed training external references missing") from exc


def build_completed_training_proof(
    old_prepared: Path,
    case_id: str,
    method: str,
    output: Path,
) -> dict[str, str]:
    """Seal a completed training stage whose fixed evaluation never started."""
    root = Path(old_prepared).resolve(strict=True)
    prepared_manifest_path = root / "manifest.json"
    prepared_manifest = V3._json(prepared_manifest_path)
    arm = _arm(prepared_manifest, case_id, method)
    recovery = Path(str(arm.get("recovery_folder"))).resolve(strict=True)
    recovery_manifest = V3._json(recovery / "manifest.json")
    context = recovery_manifest.get("recovery_context")
    if not isinstance(context, Mapping):
        _fail("prior recovery context missing")
    protocol_path, protocol = _load_ref(context.get("protocol"), label="prior recovery protocol")
    diagnostic = protocol.get("diagnostic")
    operation = protocol.get("operation")
    if not isinstance(diagnostic, Mapping) or not isinstance(operation, Mapping):
        _fail("prior registered operation missing")
    output_root = Path(str(context.get("output_root"))).resolve(strict=True)
    stage = (output_root / "training").resolve(strict=True)
    fixed = output_root / "fixed"
    if not fixed.is_dir() or fixed.is_symlink() or any(fixed.iterdir()):
        _fail("prior fixed evaluation is not an empty pre-dispatch folder")
    stage_files = {name: _ref(stage / name) for name in _STAGE_FILES}
    binding = V3._json(stage / "binding.json")
    manifest = V3._json(stage / "manifest.json")
    namespace = V3._json(stage / "run-namespace.json")
    receipt = V3._json(stage / "receipt.json")
    request_id = namespace.get("original_run_name")
    if not isinstance(request_id, str) or not request_id or manifest.get("run_name") != namespace.get(
        "effective_run_name"
    ):
        _fail("prior training namespace identity missing")
    plan_ref = prepared_manifest.get("plan")
    if not isinstance(plan_ref, Mapping):
        _fail("prior registered plan reference missing")
    plan_path, plan = _load_ref(plan_ref, label="prior registered plan")
    source_anchor = plan.get("source_anchor")
    registration_anchor = prepared_manifest.get("registration_anchor")
    if not isinstance(source_anchor, str) or not isinstance(registration_anchor, str):
        _fail("prior source/registration anchor missing")
    diagnostic_ref = arm.get("diagnostic_manifest")
    if not isinstance(diagnostic_ref, Mapping):
        _fail("prior diagnostic manifest reference missing")
    V3._validate_ref(diagnostic_ref, label="prior diagnostic manifest")
    protocol_ref = context.get("protocol")
    if not isinstance(protocol_ref, Mapping):
        _fail("prior recovery protocol reference missing")
    proof = {
        "schema": TRAINING_SCHEMA,
        "status": "completed_training_only",
        "case_id": case_id,
        "method": method,
        "source_anchor": source_anchor,
        "registration_anchor": registration_anchor,
        "request_id": request_id,
        "training_stage": str(stage),
        "recovery_folder": str(recovery),
        "identity": {
            "prior_plan_sha256": str(plan_ref["sha256"]),
            "diagnostic_manifest_sha256": str(diagnostic["session_manifest_sha256"]),
            "source_checkpoint_sha256": str(binding["resume_checkpoint"]["sha256"]),
            "final_checkpoint_sha256": str(receipt["checkpoint"]["sha256"]),
            "seed": binding["seed"],
            "task": binding["task"],
            "correction_id": manifest["correction_id"],
            "source_anchor": source_anchor,
            "registration_anchor": registration_anchor,
        },
        "refs": {
            "prepared_manifest": _ref(prepared_manifest_path),
            "prior_plan": {"path": str(plan_path), "sha256": str(plan_ref["sha256"])},
            "protocol": {"path": str(protocol_path), "sha256": str(protocol_ref["sha256"])},
            "diagnostic_manifest": dict(diagnostic_ref),
        },
        "stage_files": stage_files,
        "recovery_files": _request_refs(recovery, request_id),
        "external_files": _external_refs(binding, receipt),
        "claims": dict(_CLAIMS),
    }
    return V3._write_new(Path(output), proof)


def _expected_identity(
    proof: Mapping[str, Any],
    *,
    expected_import: Mapping[str, Any] | None,
    expected_source_checkpoint_sha256: str | None,
    expected_seed: int | None,
    expected_correction_id: str | None,
    expected_task: str | None,
    expected_diagnostic_manifest_sha256: str | None,
) -> dict[str, Any]:
    identity = proof.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        _fail("completed training identity fields differ")
    if expected_import is not None:
        if set(expected_import) != _IMPORT_FIELDS or expected_import.get("kind") != "completed_training":
            _fail("completed training stage import whitelist differs")
        expected = {
            "prior_plan_sha256": expected_import["prior_plan_sha256"],
            "diagnostic_manifest_sha256": expected_import["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": expected_import["source_checkpoint_sha256"],
            "final_checkpoint_sha256": expected_import["completed_checkpoint_sha256"],
            "seed": expected_import["seed"],
            "task": expected_import["task"],
            "correction_id": expected_import["correction_id"],
            "source_anchor": expected_import["source_anchor"],
            "registration_anchor": expected_import["registration_anchor"],
        }
        if proof.get("case_id") != expected_import["case_id"] or proof.get("method") != expected_import["method"]:
            _fail("completed training stage import arm differs")
    else:
        required = {
            "source_checkpoint_sha256": expected_source_checkpoint_sha256,
            "seed": expected_seed,
            "correction_id": expected_correction_id,
            "task": expected_task,
            "diagnostic_manifest_sha256": expected_diagnostic_manifest_sha256,
        }
        if any(value is None for value in required.values()):
            _fail("completed training caller expectation missing")
        expected = dict(identity)
        expected.update(required)
    if dict(identity) != expected:
        _fail("completed training identity differs from caller expectation")
    return expected


def _load_groups(proof: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for group in ("refs", "stage_files", "recovery_files", "external_files"):
        refs = proof.get(group)
        if not isinstance(refs, Mapping):
            _fail(f"{group} missing")
        for name, value in refs.items():
            path = V3._validate_ref(value, label=f"{group}.{name}")
            if path.suffix == ".json":
                loaded[f"{group}.{name}"] = V3._json(path)
            elif path.suffix == ".jsonl":
                loaded[f"{group}.{name}"] = {"rows": V3._jsonl(path)}
    return loaded


def _validate_fixed_not_started(training_stage: Path) -> None:
    fixed = training_stage.parent / "fixed"
    if not fixed.is_dir() or fixed.is_symlink():
        _fail("prior fixed pre-dispatch folder missing")
    entries = list(fixed.iterdir())
    if entries:
        names = {entry.name for entry in entries}
        forbidden = {"jobs", "report", "owned", "execution.json", "gpu_before.json"}
        if names & forbidden:
            _fail("prior fixed evaluation contains execution evidence")
        _fail("prior fixed pre-dispatch folder is not empty")


def _validate_completed_training(
    proof: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo: Path,
    isaaclab: Path,
    expected_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(proof) != _PROOF_FIELDS or proof.get("schema") != TRAINING_SCHEMA:
        _fail("completed training proof fields differ")
    if proof.get("status") != "completed_training_only" or proof.get("claims") != _CLAIMS:
        _fail("completed training proof is not closed")
    if proof.get("source_anchor") != expected["source_anchor"] or proof.get(
        "registration_anchor"
    ) != expected["registration_anchor"]:
        _fail("completed training proof anchors differ")
    loaded = _load_groups(proof)
    stage = Path(str(proof["training_stage"])).resolve(strict=True)
    recovery = Path(str(proof["recovery_folder"])).resolve(strict=True)
    _validate_fixed_not_started(stage)
    if Path(proof["stage_files"]["binding.json"]["path"]).parent != stage:
        _fail("completed training stage path differs")
    if Path(proof["recovery_files"]["manifest.json"]["path"]).parent != recovery:
        _fail("completed recovery folder differs")

    binding = loaded["stage_files.binding.json"]
    manifest = loaded["stage_files.manifest.json"]
    namespace = loaded["stage_files.run-namespace.json"]
    initial = loaded["stage_files.initial.json"]
    correction = loaded["stage_files.correction.json"]
    receipt = loaded["stage_files.receipt.json"]
    execution = loaded["stage_files.execution.json"]
    launch = loaded["stage_files.owned/launch.json"]
    if binding.get("schema") != TRAIN.SCHEMA_V3 or binding.get("source_commit_sha") != expected["source_anchor"]:
        _fail("completed training historical binding differs")
    TRAIN.validate_host_binding(binding, isaaclab)
    host = TRAIN._Host(binding, stage / "manifest.json")
    if host.validate_namespace_receipt() != namespace:
        _fail("completed training namespace receipt differs")
    if namespace.get("original_run_name") != proof["request_id"] or namespace.get(
        "effective_run_name"
    ) != TRAIN.training_run_name(str(proof["request_id"]), int(expected["seed"]), binding):
        _fail("completed training effective namespace differs")
    expected_manifest = TRAIN.manifest_for(
        stage, binding, isaaclab, str(namespace["effective_run_name"]),
        original_run_name=str(namespace["original_run_name"]),
    )
    expected_manifest["protocol_sha256_lf"] = manifest.get("protocol_sha256_lf")
    if len(manifest) != 23 or manifest != expected_manifest:
        _fail("completed training frozen 23-field manifest differs")
    binding_validation = PRIOR._validate_binding_expectation(binding, expected_binding)

    if (
        receipt.get("completed") is not True
        or receipt.get("arm") != "corrected"
        or receipt.get("fault") != "F2"
        or receipt.get("correction_id") != expected["correction_id"]
        or receipt.get("checkpoint") != proof["external_files"]["checkpoint"]
        or receipt.get("budget", {}).get("max_iterations") != 300
        or receipt.get("resume_iteration") != 299
        or receipt.get("final_iteration") != 598
        or receipt.get("actual_control_steps") != 7200
        or receipt.get("aggregate_env_steps") != 29_491_200
        or receipt.get("final_checkpoint_state", {}).get("iteration") != 598
    ):
        _fail("completed training receipt differs")
    if (
        initial.get("resume_checkpoint", {}).get("sha256") != expected["source_checkpoint_sha256"]
        or initial.get("resume_checkpoint", {}).get("load_optimizer") is not True
        or initial.get("full_environment_state_restored") is not False
        or initial.get("rng_state_restored_from_checkpoint") is not False
    ):
        _fail("completed training initial state differs")
    TRAIN.LEGACY.validate_readback(correction, "F2", str(expected["correction_id"]))
    if correction.get("changed_targets") != [expected["correction_id"]] or correction.get(
        "restored_targets"
    ) != [expected["correction_id"]]:
        _fail("completed training correction scope differs")
    wall = execution.get("wall_s")
    if (
        execution.get("exit_code") != 0
        or execution.get("natural_exit") is not True
        or execution.get("cleanup_verified") is not True
        or execution.get("timed_out") is not False
        or execution.get("job_assigned_before_release") is not True
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(wall)
        or wall <= 0
    ):
        _fail("completed training execution differs")
    expected_launch = [
        str(isaaclab / "_isaac_sim/python.bat"), "-B",
        str(repo / "bench/registered_recovery_training_v1.py"),
        "--manifest", str((stage / "manifest.json").resolve()),
        "--binding", str((stage / "binding.json").resolve()),
    ]
    if launch.get("args") != expected_launch or Path(str(launch.get("cwd"))).resolve() != isaaclab:
        _fail("completed training launch differs")
    host.validate_training_configuration(Path(str(initial["log_dir"])), manifest)

    prepared = loaded["recovery_files.manifest.json"]
    context = prepared.get("recovery_context")
    if not isinstance(context, Mapping):
        _fail("completed recovery context missing")
    protocol_path, protocol = _load_ref(context.get("protocol"), label="completed recovery protocol")
    if proof["refs"]["protocol"] != _ref(protocol_path):
        _fail("completed recovery protocol reference differs")
    operation = protocol.get("operation")
    diagnostic = protocol.get("diagnostic")
    if (
        not isinstance(operation, Mapping)
        or not isinstance(diagnostic, Mapping)
        or operation.get("prior_training_artifact_id") is not None
        or operation.get("correction_id") != expected["correction_id"]
        or operation.get("training_seed") != expected["seed"]
        or diagnostic.get("case_id") != proof["case_id"]
        or diagnostic.get("session_manifest_sha256") != expected["diagnostic_manifest_sha256"]
        or diagnostic.get("checkpoint_sha256") != expected["source_checkpoint_sha256"]
        or protocol.get("source_anchor") != expected["source_anchor"]
    ):
        _fail("completed recovery protocol identity differs")
    closure = protocol.get("source_closure")
    if not isinstance(closure, Mapping) or closure.get(
        "bench/registered_recovery_training_v1.py"
    ) != manifest.get("protocol_sha256_lf"):
        _fail("completed historical training source differs")
    source_validation = V4._historical_closure(
        repo, str(expected["source_anchor"]), str(expected["registration_anchor"]), closure,
    )

    prepared_manifest = loaded["refs.prepared_manifest"]
    arm = _arm(prepared_manifest, str(proof["case_id"]), str(proof["method"]))
    if prepared_manifest.get("plan") != proof["refs"]["prior_plan"]:
        _fail("completed prior plan reference differs")
    if arm.get("recovery_folder") != proof["recovery_folder"] or arm.get(
        "diagnostic_manifest"
    ) != proof["refs"]["diagnostic_manifest"]:
        _fail("completed arm references differ")
    if proof["refs"]["prior_plan"]["sha256"] != expected["prior_plan_sha256"]:
        _fail("completed prior plan SHA differs")

    request_id = str(proof["request_id"])
    rows = [
        row for row in loaded["recovery_files.recovery.jsonl"]["rows"]
        if row.get("request_id") == request_id
    ]
    if (
        [row.get("event") for row in rows] != ["proposed", "approved", "consumed", "receipt"]
        or rows[-1].get("verdict") != "unknown"
        or rows[-1].get("reason") != "orphan_without_artifact"
        or rows[-1].get("result_sha") is not None
    ):
        _fail("completed recovery lifecycle differs")
    resource = loaded[f"recovery_files.resources/{request_id}.json"]
    intent = loaded[f"recovery_files.intents/{request_id}.json"]
    admission = loaded[f"recovery_files.admissions/{request_id}.json"]
    dispatch = loaded[f"recovery_files.dispatch_errors/{request_id}.json"]
    authority = loaded[f"recovery_files.authority_costs/{request_id}.json"]
    if (
        resource.get("status") != "failed"
        or resource.get("error_type") != "RetryAdmissionError"
        or resource.get("simulator_sdk_launches") != 1
        or resource.get("total_owned_children") != 1
        or resource.get("owned_gpu_s") != wall
        or resource.get("execution_refs", {}).get("training") != proof["stage_files"]["execution.json"]
        or resource.get("execution_refs", {}).get("fixed") is not None
        or resource.get("admission_ref") != proof["recovery_files"][f"admissions/{request_id}.json"]
        or resource.get("sampler_ref") != proof["recovery_files"]["device_sampler_stop"]
        or admission.get("request_id") != request_id
        or intent.get("request", {}).get("protocol_case_arm") != {"arm": "corrected", "case": "case_02"}
        or dispatch.get("schema") != "investigation_recovery_v1_dispatch_error"
        or dispatch.get("error_type") != "RetryAdmissionError"
        or dispatch.get("reexecute_allowed") is not False
        or "registered_desktop_gpu_retry_hard_failure" not in str(dispatch.get("error_message"))
        or isinstance(authority.get("authority_s"), bool)
        or not isinstance(authority.get("authority_s"), (int, float))
        or not math.isfinite(float(authority["authority_s"]))
    ):
        _fail("completed recovery admission failure evidence differs")
    return {
        "binding": binding,
        "manifest": manifest,
        "namespace": namespace,
        "initial": initial,
        "correction": correction,
        "receipt": receipt,
        "execution": execution,
        "binding_validation": binding_validation,
        "historical_source_validation": source_validation,
        "recovery_validation": {
            "request_id": request_id,
            "case_id": proof["case_id"],
            "method": proof["method"],
            "closed_verdict": "unknown",
            "closed_reason": "orphan_without_artifact",
            "dispatch_error_type": "RetryAdmissionError",
            "fixed_evaluation_executed": False,
            "diagnostic_manifest_sha256": expected["diagnostic_manifest_sha256"],
            "prior_dispatch_resource_ref": proof["recovery_files"][f"resources/{request_id}.json"],
            "prior_training_gpu_s": resource["owned_gpu_s"],
            "prior_phase_wall_s": resource["phase_wall_s"],
            "request_reconsumed": False,
        },
    }


def load_completed_training(
    proof_ref: Mapping[str, Any],
    *,
    repo: Path,
    isaaclab: Path,
    expected_import: Mapping[str, Any] | None = None,
    expected_source_checkpoint_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_correction_id: str | None = None,
    expected_task: str | None = None,
    expected_diagnostic_manifest_sha256: str | None = None,
    expected_binding: Mapping[str, Any] | None = None,
    observed_anchor: str | None = None,
) -> dict[str, Any]:
    """Return verified prior training while keeping the failed request immutable."""
    proof_path, proof = _load_ref(proof_ref, label="completed training proof")
    expected = _expected_identity(
        proof,
        expected_import=expected_import,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_seed=expected_seed,
        expected_correction_id=expected_correction_id,
        expected_task=expected_task,
        expected_diagnostic_manifest_sha256=expected_diagnostic_manifest_sha256,
    )
    if expected_import is not None and proof_ref.get("sha256") != expected_import["proof_sha256"]:
        _fail("completed training proof SHA differs from whitelist")
    if observed_anchor is not None and observed_anchor != expected["source_anchor"]:
        _fail("completed training observed anchor differs")
    validated = _validate_completed_training(
        proof,
        expected=expected,
        repo=Path(repo).resolve(strict=True),
        isaaclab=Path(isaaclab).resolve(strict=True),
        expected_binding=expected_binding,
    )
    receipt = validated["receipt"]
    execution = validated["execution"]
    return {
        "schema": COMPLETED_SCHEMA,
        "training_result": {
            "arm": "CORRECTED",
            "checkpoint": receipt["checkpoint"],
            "readback": validated["correction"],
            "execution": execution,
            "receipt": receipt,
            "configuration": {
                "source_checkpoint_consumed": True,
                "optimizer_state_resumed": True,
                "original_run_name": validated["namespace"]["original_run_name"],
                "effective_run_name": validated["namespace"]["effective_run_name"],
            },
        },
        "training_stage": proof["training_stage"],
        "proof_ref": {"path": str(proof_path), "sha256": V3._sha(proof_path)},
        "prior_owned_execution_ref": proof["stage_files"]["execution.json"],
        "prior_dispatch_resource_ref": validated["recovery_validation"]["prior_dispatch_resource_ref"],
        "new_training_launched": False,
        "validation": {
            "completed_training_only": True,
            "fixed_evaluation_executed": False,
            "recovery_label_available": False,
            "policy_optimizer_resumed": True,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
            "binding": validated["binding_validation"],
            "historical_source": validated["historical_source_validation"],
            "recovery": validated["recovery_validation"],
        },
    }
