"""Import immutable completed recovery evidence without launching new work."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bench import registered_recovery_training_v1 as TRAIN
from evals import registered_prior_training_v1 as PRIOR

EFFECT_SCHEMA = "registered_completed_effect_proof_v1"
TRAINING_SCHEMA = "registered_completed_training_proof_v1"
CORRECTION = "restore_motor_torque_speed"
SOURCE_ANCHOR = "f9d5fea93cce406a325f578bc6f4597cc0bc35c5"
CASE_ID = "case-05"
EFFECT_METHOD = "fixed_sequence"
TRAINING_METHOD = "static_rule"
_STAGE_FILES = PRIOR.REQUIRED_STAGE_FILES
_RECOVERY_FILES = PRIOR.REQUIRED_RECOVERY_FILES


class ContinuationProofError(ValueError):
    """Raised when saved continuation evidence is incomplete or changed."""


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationProofError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationProofError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationProofError(f"invalid JSONL: {path}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ContinuationProofError("JSONL object rows required")
    return values


def _ref(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ContinuationProofError(f"reference is not a file: {resolved}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _validate_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ContinuationProofError(f"{label} reference fields")
    path = Path(str(value["path"])).resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise ContinuationProofError(f"{label} reference changed")
    return path


def _load_ref(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _validate_ref(value, label=label)
    return path, _json(path)


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return _ref(path)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True)
    if result.returncode != 0:
        raise ContinuationProofError(f"git validation failed: {' '.join(args)}")
    return result.stdout


def _historical_closure(repo: Path, anchor: str, closure: Mapping[str, Any]) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    _git(repo, "merge-base", "--is-ancestor", anchor, "HEAD")
    current_differences: list[str] = []
    for relative, expected in sorted(closure.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ContinuationProofError("source closure fields")
        historical = _git(repo, "show", f"{anchor}:{relative}")
        actual = hashlib.sha256(historical.replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ContinuationProofError(f"historical source differs: {relative}")
        current = repo / relative
        if not current.is_file() or _sha(current, lf=True) != expected:
            current_differences.append(relative)
    return {
        "source_anchor": anchor,
        "historical_files_verified": len(closure),
        "current_difference_paths": current_differences,
        "historical_source_authoritative": True,
    }


def _arm(manifest: Mapping[str, Any], case_id: str, method: str) -> Mapping[str, Any]:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise ContinuationProofError("prepared arms missing")
    matches = [arm for arm in arms if isinstance(arm, Mapping)
               and arm.get("case_id") == case_id and arm.get("method") == method]
    if len(matches) != 1:
        raise ContinuationProofError("completed arm identity is not unique")
    return matches[0]


def build_completed_effect_proof(
    old_prepared: Path,
    case_id: str,
    method: str,
    output: Path,
) -> dict[str, str]:
    """Seal one completed comparison arm and all evidence needed to revalidate it."""
    root = Path(old_prepared).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    arm = _arm(manifest, case_id, method)
    recovery = Path(str(arm["recovery_folder"])).resolve(strict=True)
    prepared = _json(recovery / "manifest.json")
    context = prepared.get("recovery_context")
    if not isinstance(context, Mapping):
        raise ContinuationProofError("recovery context missing")
    protocol_ref = context.get("protocol")
    protocol_path = _validate_ref(protocol_ref, label="recovery protocol")
    request_id = str(_json(recovery / "manifest.json")["recovery_context"]["protocol_body"]
                     ["operation"].get("prior_training_artifact_id") or "")
    del request_id
    resources = sorted((recovery / "resources").glob("*.json"))
    receipts = sorted((recovery / "receipts").glob("*.json"))
    if len(resources) != 1 or len(receipts) != 1 or resources[0].stem != receipts[0].stem:
        raise ContinuationProofError("completed effect request files are not unique")
    request_id = resources[0].stem
    result_path = root / "results" / case_id / f"{method}.json"
    controller_resource_path = root / "resources" / case_id / f"{method}.json"
    backend_receipt_path = Path(str(context["output_root"])) / "receipt.json"
    refs = {
        "prepared_manifest": _ref(manifest_path),
        "prepared_anchor": _ref(root / "anchor.json"),
        "prior_plan": dict(manifest["plan"]),
        "arm_result": _ref(result_path),
        "controller_resource": _ref(controller_resource_path),
        "diagnostic_manifest": dict(arm["diagnostic_manifest"]),
        "arm_approval": dict(arm["approval_ref"]),
        "recovery_manifest": _ref(recovery / "manifest.json"),
        "recovery_anchor": _ref(recovery / "anchor.json"),
        "recovery_ledger": _ref(recovery / "recovery.jsonl"),
        "approval": _ref(recovery / f"approvals/{request_id}.json"),
        "admission": _ref(recovery / f"admissions/{request_id}.json"),
        "intent": _ref(recovery / f"intents/{request_id}.json"),
        "resource": _ref(resources[0]),
        "ledger_receipt": _ref(receipts[0]),
        "protocol": _ref(protocol_path),
        "backend_receipt": _ref(backend_receipt_path),
    }
    result = _json(result_path)
    refs["fixed_report"] = dict(result["fixed_report"])
    refs["new_checkpoint"] = dict(result["new_checkpoint"])
    protocol = _json(protocol_path)
    diagnostic = protocol["diagnostic"]
    operation = protocol["operation"]
    training_binding = context["artifacts"][operation["training_binding_artifact_id"]]["body"]
    proof = {
        "schema": EFFECT_SCHEMA,
        "status": "completed_effect",
        "case_id": case_id,
        "method": method,
        "registration_anchor": manifest.get("registration_anchor"),
        "source_anchor": _json(protocol_path).get("source_anchor"),
        "request_id": request_id,
        "recovery_folder": str(recovery),
        "identity": {
            "prior_plan_sha256": manifest["plan"]["sha256"],
            "diagnostic_manifest_sha256": arm["diagnostic_manifest"]["sha256"],
            "source_checkpoint_sha256": diagnostic["checkpoint_sha256"],
            "final_checkpoint_sha256": result["new_checkpoint"]["sha256"],
            "seed": operation["training_seed"],
            "task": training_binding["task"],
            "correction_id": operation["correction_id"],
            "source_anchor": protocol["source_anchor"],
            "registration_anchor": manifest["registration_anchor"],
        },
        "refs": refs,
        "claims": {"new_dispatch_started": False, "ledger_reconsumed": False},
    }
    return _write_new(Path(output), proof)


def verify_completed_effect(
    proof_ref: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_method: str,
    expected_prior_plan_sha256: str,
    expected_diagnostic_manifest_sha256: str,
    expected_source_checkpoint_sha256: str,
    repo: Path,
) -> dict[str, Any]:
    """Verify a completed arm without dispatching or consuming its ledger again."""
    proof_path, proof = _load_ref(proof_ref, label="completed effect proof")
    if (proof.get("schema") != EFFECT_SCHEMA or proof.get("status") != "completed_effect"
            or proof.get("case_id") != expected_case_id or proof.get("method") != expected_method
            or expected_case_id != CASE_ID or expected_method != EFFECT_METHOD
            or proof.get("claims") != {"new_dispatch_started": False, "ledger_reconsumed": False}):
        raise ContinuationProofError("completed effect proof identity differs")
    refs = proof.get("refs")
    if not isinstance(refs, Mapping):
        raise ContinuationProofError("completed effect refs missing")
    loaded: dict[str, dict[str, Any]] = {}
    for name, value in refs.items():
        path = _validate_ref(value, label=f"refs.{name}")
        if path.suffix in {".json", ".jsonl"} and name != "new_checkpoint":
            loaded[name] = _json(path) if path.suffix == ".json" else {"rows": _jsonl(path)}
    manifest = loaded["prepared_manifest"]
    arm = _arm(manifest, expected_case_id, expected_method)
    if (refs["prior_plan"].get("sha256") != expected_prior_plan_sha256
            or manifest.get("plan") != refs["prior_plan"]
            or refs["diagnostic_manifest"].get("sha256") != expected_diagnostic_manifest_sha256
            or arm.get("diagnostic_manifest") != refs["diagnostic_manifest"]
            or arm.get("recovery_folder") != proof["recovery_folder"]):
        raise ContinuationProofError("prepared plan/session binding differs")
    protocol = loaded["protocol"]
    diagnostic = protocol.get("diagnostic")
    operation = protocol.get("operation")
    if (not isinstance(diagnostic, Mapping) or not isinstance(operation, Mapping)
            or diagnostic.get("case_id") != expected_case_id
            or diagnostic.get("session_manifest_sha256") != expected_diagnostic_manifest_sha256
            or diagnostic.get("checkpoint_sha256") != expected_source_checkpoint_sha256
            or operation.get("correction_id") != CORRECTION
            or operation.get("prior_training_artifact_id") != "prior_training"):
        raise ContinuationProofError("completed effect protocol differs")
    expected_identity = {
        "prior_plan_sha256": expected_prior_plan_sha256,
        "diagnostic_manifest_sha256": expected_diagnostic_manifest_sha256,
        "source_checkpoint_sha256": expected_source_checkpoint_sha256,
        "final_checkpoint_sha256": refs["new_checkpoint"]["sha256"],
        "seed": operation["training_seed"],
        "task": loaded["intent"]["context"]["artifacts"]
            [operation["training_binding_artifact_id"]]["body"]["task"],
        "correction_id": operation["correction_id"],
        "source_anchor": protocol["source_anchor"],
        "registration_anchor": manifest["registration_anchor"],
    }
    if proof.get("identity") != expected_identity:
        raise ContinuationProofError("completed effect identity differs")
    closure = protocol.get("source_closure")
    if protocol.get("source_anchor") != SOURCE_ANCHOR or not isinstance(closure, Mapping):
        raise ContinuationProofError("completed effect source binding differs")
    source_validation = _historical_closure(Path(repo), SOURCE_ANCHOR, closure)
    result = loaded["arm_result"]
    if (result.get("case_id") != expected_case_id or result.get("method") != expected_method
            or result.get("outcome") != "not_recovered"
            or result.get("fixed_report") != refs["fixed_report"]
            or result.get("new_checkpoint") != refs["new_checkpoint"]):
        raise ContinuationProofError("completed arm result differs")
    resource = loaded["resource"]
    controller_resource = loaded["controller_resource"]
    admission_ref = resource.get("admission_ref")
    if (resource.get("status") != "complete" or resource.get("simulator_sdk_launches") != 1
            or controller_resource.get("recovery_resource_ref") != refs["resource"]
            or controller_resource.get("admission_ref") != admission_ref
            or refs["admission"] != admission_ref):
        raise ContinuationProofError("completed resource evidence differs")
    request_id = str(proof["request_id"])
    rows = [row for row in loaded["recovery_ledger"]["rows"] if row.get("request_id") == request_id]
    if [row.get("event") for row in rows] != ["proposed", "approved", "consumed", "receipt"]:
        raise ContinuationProofError("completed recovery lifecycle differs")
    if rows[-1].get("verdict") != "not_recovered":
        raise ContinuationProofError("completed recovery verdict differs")
    from evals import registered_recovery_backend_v1 as backend
    from rl_triage import investigation_recovery

    backend_receipt = loaded["backend_receipt"]
    backend_verified = backend.verify_receipt(
        {**backend_receipt, "receipt_ref": refs["backend_receipt"]}, loaded["intent"]
    )
    post = investigation_recovery.verified_post_observation(Path(proof["recovery_folder"]), request_id)
    if post.get("healthy_declared") is not False or post.get("behavior_outcome") != "failed":
        raise ContinuationProofError("post observation differs")
    costs = result.get("costs")
    if not isinstance(costs, Mapping):
        raise ContinuationProofError("completed effect costs missing")
    training_value, fixed_value = costs.get("training_s"), costs.get("fixed_s")
    if (not isinstance(training_value, (int, float)) or not math.isfinite(training_value)
            or not isinstance(fixed_value, (int, float)) or not math.isfinite(fixed_value)):
        raise ContinuationProofError("completed effect scientific cost differs")
    training_s, fixed_s = float(training_value), float(fixed_value)
    if (abs(training_s - 515.3279999999795) > 1e-9
            or abs(fixed_s - 47.07799999997951) > 1e-9):
        raise ContinuationProofError("completed effect scientific cost differs")
    historical_costs = {
        "simulator_sdk_launches": 2,
        "owned_gpu_s": training_s + fixed_s,
        "training_wall_s": training_s,
        "fixed_wall_s": fixed_s,
        "phase_wall_s": controller_resource.get("phase_wall_s"),
        "phase_measurement_ref": refs["controller_resource"],
    }
    return {
        "execution_origin": "verified_prior_effect",
        "result": result,
        "prior_resource_ref": refs["resource"],
        "prior_admission_ref": refs["admission"],
        "recovery_folder": proof["recovery_folder"],
        "post_observation": post,
        "proof_ref": {"path": str(proof_path), "sha256": _sha(proof_path)},
        "historical_costs": historical_costs,
        "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
        "historical_source_validation": source_validation,
        "backend_receipt_validation": backend_verified,
        "old_refs": dict(refs),
    }


def _training_recovery_refs(recovery: Path, request_id: str) -> dict[str, dict[str, str]]:
    refs = {name: _ref(recovery / name) for name in _RECOVERY_FILES}
    for rel in (f"approvals/{request_id}.json", f"admissions/{request_id}.json",
                f"intents/{request_id}.json", f"resources/{request_id}.json",
                f"dispatch_errors/{request_id}.json"):
        refs[rel] = _ref(recovery / rel)
    return refs


def build_completed_training_proof(
    training_stage: Path,
    recovery_folder: Path,
    output: Path,
) -> dict[str, str]:
    """Seal the static-rule completed training stage for a later fixed evaluation."""
    stage = Path(training_stage).resolve(strict=True)
    recovery = Path(recovery_folder).resolve(strict=True)
    stage_refs = {name: _ref(stage / name) for name in _STAGE_FILES}
    binding = _json(stage / "binding.json")
    manifest = _json(stage / "manifest.json")
    request_id = manifest.get("run_name")
    if not isinstance(request_id, str) or not request_id:
        raise ContinuationProofError("training request identity missing")
    receipt = _json(stage / "receipt.json")
    external = {
        "source_checkpoint": _ref(Path(str(binding["resume_checkpoint"]["path"]))),
        "source_environment_config": _ref(Path(str(binding["resume_env_config"]["path"]))),
        "source_agent_config": _ref(Path(str(binding["resume_agent_config"]["path"]))),
        "checkpoint": _ref(Path(str(receipt["checkpoint"]["path"]))),
        "environment_config": _ref(Path(str(receipt["configs"]["environment"]["path"]))),
        "agent_config": _ref(Path(str(receipt["configs"]["agent"]["path"]))),
    }
    proof = {
        "schema": TRAINING_SCHEMA,
        "status": "completed_training_only",
        "case_id": CASE_ID,
        "method": TRAINING_METHOD,
        "training_stage": str(stage),
        "recovery_folder": str(recovery),
        "request_id": request_id,
        "source_anchor": binding.get("source_commit_sha"),
        "identity": {
            "diagnostic_manifest_sha256": _json(recovery / "manifest.json")["recovery_context"]
                ["protocol_body"]["diagnostic"]["session_manifest_sha256"],
            "source_checkpoint_sha256": binding["resume_checkpoint"]["sha256"],
            "final_checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "seed": binding["seed"],
            "task": binding["task"],
            "correction_id": manifest["correction_id"],
            "source_anchor": binding["source_commit_sha"],
            "case_id": CASE_ID,
            "method": TRAINING_METHOD,
        },
        "stage_files": stage_refs,
        "recovery_files": _training_recovery_refs(recovery, request_id),
        "external_files": external,
        "claims": {
            "new_training_launched": False,
            "fixed_evaluation_executed": False,
            "recovery_label_available": False,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
            "policy_optimizer_resumed": True,
        },
    }
    return _write_new(Path(output), proof)


def _validate_completed_training(
    proof: Mapping[str, Any], *, repo: Path, isaaclab: Path,
    expected_source_checkpoint_sha256: str, expected_seed: int,
    expected_correction_id: str, expected_task: str,
    expected_diagnostic_manifest_sha256: str,
    expected_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (proof.get("schema") != TRAINING_SCHEMA or proof.get("status") != "completed_training_only"
            or proof.get("case_id") != CASE_ID or proof.get("method") != TRAINING_METHOD
            or proof.get("source_anchor") != SOURCE_ANCHOR):
        raise ContinuationProofError("completed training proof identity differs")
    for group in ("stage_files", "recovery_files", "external_files"):
        refs = proof.get(group)
        if not isinstance(refs, Mapping):
            raise ContinuationProofError(f"{group} missing")
        for name, value in refs.items():
            _validate_ref(value, label=f"{group}.{name}")
    stage = Path(str(proof["training_stage"])).resolve(strict=True)
    recovery = Path(str(proof["recovery_folder"])).resolve(strict=True)
    binding = _json(stage / "binding.json")
    manifest = _json(stage / "manifest.json")
    initial = _json(stage / "initial.json")
    correction = _json(stage / "correction.json")
    receipt = _json(stage / "receipt.json")
    execution = _json(stage / "execution.json")
    launch = _json(stage / "owned/launch.json")
    expected_identity = {
        "diagnostic_manifest_sha256": expected_diagnostic_manifest_sha256,
        "source_checkpoint_sha256": expected_source_checkpoint_sha256,
        "final_checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "seed": expected_seed,
        "task": expected_task,
        "correction_id": expected_correction_id,
        "source_anchor": SOURCE_ANCHOR,
        "case_id": CASE_ID,
        "method": TRAINING_METHOD,
    }
    if proof.get("identity") != expected_identity:
        raise ContinuationProofError("completed training identity differs")
    if (binding.get("source_commit_sha") != SOURCE_ANCHOR
            or binding.get("resume_checkpoint", {}).get("sha256") != expected_source_checkpoint_sha256
            or binding.get("seed") != expected_seed or binding.get("task") != expected_task
            or expected_correction_id != CORRECTION or manifest.get("correction_id") != CORRECTION):
        raise ContinuationProofError("completed training source/identity differs")
    TRAIN.validate_host_binding(binding, Path(isaaclab))
    if TRAIN.manifest_for(stage, binding, Path(isaaclab), str(proof["request_id"])) != manifest:
        raise ContinuationProofError("completed training manifest differs")
    binding_validation = PRIOR._validate_binding_expectation(binding, expected_binding)
    if (receipt.get("completed") is not True or receipt.get("arm") != "corrected"
            or receipt.get("fault") != "F2" or receipt.get("correction_id") != CORRECTION
            or receipt.get("checkpoint") != proof["external_files"]["checkpoint"]
            or receipt.get("budget", {}).get("max_iterations") != 300
            or receipt.get("resume_iteration") != 299 or receipt.get("final_iteration") != 598
            or receipt.get("actual_control_steps") != 7200
            or receipt.get("aggregate_env_steps") != 29_491_200
            or receipt.get("final_checkpoint_state", {}).get("iteration") != 598):
        raise ContinuationProofError("completed training receipt differs")
    if (initial.get("resume_checkpoint", {}).get("sha256") != expected_source_checkpoint_sha256
            or initial.get("resume_checkpoint", {}).get("load_optimizer") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False):
        raise ContinuationProofError("completed training initial state differs")
    TRAIN.LEGACY.validate_readback(correction, "F2", CORRECTION)
    if correction.get("changed_targets") != [CORRECTION] or correction.get("restored_targets") != [CORRECTION]:
        raise ContinuationProofError("completed training correction scope differs")
    if (execution.get("exit_code") != 0 or execution.get("natural_exit") is not True
            or execution.get("cleanup_verified") is not True or execution.get("timed_out") is not False
            or execution.get("job_assigned_before_release") is not True
            or not isinstance(execution.get("wall_s"), (int, float))
            or abs(float(execution["wall_s"]) - 518.5160000000615) > 1e-9):
        raise ContinuationProofError("completed training execution differs")
    expected_launch = [str(Path(isaaclab).resolve() / "_isaac_sim/python.bat"), "-B",
                       str(Path(repo).resolve() / "bench/registered_recovery_training_v1.py"),
                       "--manifest", str((stage / "manifest.json").resolve()),
                       "--binding", str((stage / "binding.json").resolve())]
    if launch.get("args") != expected_launch or Path(str(launch.get("cwd"))).resolve() != Path(isaaclab).resolve():
        raise ContinuationProofError("completed training launch differs")
    TRAIN._Host(binding, stage / "manifest.json").validate_training_configuration(
        Path(str(initial["log_dir"])), manifest
    )
    prepared = _json(recovery / "manifest.json")
    context = prepared.get("recovery_context")
    if not isinstance(context, Mapping):
        raise ContinuationProofError("static recovery context missing")
    protocol_ref = context.get("protocol")
    _, protocol = _load_ref(protocol_ref, label="static recovery protocol")
    operation = protocol.get("operation")
    diagnostic = protocol.get("diagnostic")
    if (not isinstance(operation, Mapping) or not isinstance(diagnostic, Mapping)
            or operation.get("prior_training_artifact_id") is not None
            or operation.get("correction_id") != CORRECTION
            or operation.get("training_seed") != expected_seed
            or diagnostic.get("case_id") != CASE_ID
            or diagnostic.get("session_manifest_sha256") != expected_diagnostic_manifest_sha256
            or diagnostic.get("checkpoint_sha256") != expected_source_checkpoint_sha256
            or protocol.get("source_anchor") != SOURCE_ANCHOR):
        raise ContinuationProofError("static recovery protocol differs")
    closure = protocol.get("source_closure")
    if (not isinstance(closure, Mapping)
            or closure.get("bench/registered_recovery_training_v1.py") != manifest.get("protocol_sha256_lf")):
        raise ContinuationProofError("static recovery closure differs")
    history = _historical_closure(Path(repo), SOURCE_ANCHOR, closure)
    request_id = str(proof["request_id"])
    rows = [row for row in _jsonl(recovery / "recovery.jsonl") if row.get("request_id") == request_id]
    if ([row.get("event") for row in rows] != ["proposed", "approved", "consumed", "receipt"]
            or rows[-1].get("verdict") != "unknown"
            or rows[-1].get("reason") != "orphan_without_artifact"):
        raise ContinuationProofError("static recovery lifecycle differs")
    resource = _json(recovery / f"resources/{request_id}.json")
    intent = _json(recovery / f"intents/{request_id}.json")
    dispatch = _json(recovery / f"dispatch_errors/{request_id}.json")
    if (resource.get("status") != "failed" or resource.get("simulator_sdk_launches") != 1
            or resource.get("owned_gpu_s") != execution["wall_s"]
            or resource.get("execution_refs", {}).get("training") != proof["stage_files"]["execution.json"]
            or resource.get("execution_refs", {}).get("fixed") is not None
            or intent.get("request", {}).get("protocol_case_arm") != {"arm": "corrected", "case": "case_02"}
            or dispatch != {"error_type": "RuntimeError", "reexecute_allowed": False,
                            "schema": "investigation_recovery_v1_dispatch_error"}):
        raise ContinuationProofError("static recovery evidence differs")
    return {"binding": binding, "manifest": manifest, "initial": initial, "correction": correction,
            "receipt": receipt, "execution": execution, "launch": launch,
            "binding_validation": binding_validation, "historical_source_validation": history,
            "recovery_validation": {"request_id": request_id, "case_id": CASE_ID,
                "method": TRAINING_METHOD, "closed_verdict": "unknown",
                "closed_reason": "orphan_without_artifact",
                "diagnostic_manifest_sha256": expected_diagnostic_manifest_sha256,
                "prior_dispatch_resource_ref": proof["recovery_files"][f"resources/{request_id}.json"],
                "prior_training_gpu_s": resource["owned_gpu_s"],
                "prior_phase_wall_s": resource["phase_wall_s"], "request_reconsumed": False}}


def load_completed_training(
    proof_ref: Mapping[str, Any], *, expected_source_checkpoint_sha256: str,
    expected_seed: int, expected_correction_id: str, expected_task: str,
    expected_diagnostic_manifest_sha256: str, repo: Path, isaaclab: Path,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the verified static-rule training result without launching a process."""
    proof_path, proof = _load_ref(proof_ref, label="completed training proof")
    validated = _validate_completed_training(
        proof, repo=Path(repo), isaaclab=Path(isaaclab),
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_seed=expected_seed, expected_correction_id=expected_correction_id,
        expected_task=expected_task,
        expected_diagnostic_manifest_sha256=expected_diagnostic_manifest_sha256,
        expected_binding=expected_binding,
    )
    receipt, execution = validated["receipt"], validated["execution"]
    return {
        "schema": "completed_registered_actuator_training_v1",
        "training_result": {"arm": "CORRECTED", "checkpoint": receipt["checkpoint"],
            "readback": validated["correction"], "execution": execution, "receipt": receipt,
            "configuration": {"source_checkpoint_consumed": True, "optimizer_state_resumed": True}},
        "training_stage": proof["training_stage"],
        "proof_ref": {"path": str(proof_path), "sha256": _sha(proof_path)},
        "prior_owned_execution_ref": proof["stage_files"]["execution.json"],
        "prior_dispatch_resource_ref": validated["recovery_validation"]["prior_dispatch_resource_ref"],
        "new_training_launched": False,
        "validation": {"completed_training_only": True, "fixed_evaluation_executed": False,
            "recovery_label_available": False, "policy_optimizer_resumed": True,
            "full_environment_state_restored": False, "rng_state_restored_from_checkpoint": False,
            "binding": validated["binding_validation"],
            "historical_source": validated["historical_source_validation"],
            "recovery": validated["recovery_validation"]},
    }
