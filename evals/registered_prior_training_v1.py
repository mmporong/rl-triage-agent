"""Validate and import a completed registered actuator training stage.

This module is intentionally read-only except for the write-once proof file.  It
does not launch training, evaluate a policy, or mutate the recovery ledger.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bench import registered_recovery_training_v1 as TRAIN

SCHEMA = "registered_prior_training_proof_v1"
CORRECTION = "restore_motor_torque_speed"
SOURCE_ANCHOR = "8d911922df5eb79485f02f3cde25dc8d446fa805"
REQUIRED_STAGE_FILES = (
    "binding.json",
    "manifest.json",
    "initial.json",
    "correction.json",
    "receipt.json",
    "execution.json",
    "owned/launch.json",
)
REQUIRED_RECOVERY_FILES = (
    "anchor.json",
    "manifest.json",
    "recovery.jsonl",
)


class PriorTrainingError(ValueError):
    """Raised when a prior training artifact cannot be trusted."""


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriorTrainingError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PriorTrainingError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PriorTrainingError("recovery ledger row must be an object")
        rows.append(value)
    return rows


def _ref(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise PriorTrainingError(f"reference is not a file: {resolved}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _load_ref(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _validate_ref(value, label=label)
    return path, _json(path)


def _validate_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PriorTrainingError(f"{label} reference fields")
    path = Path(str(value["path"])).resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise PriorTrainingError(f"{label} reference changed")
    return path


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return _ref(path)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise PriorTrainingError(f"git validation failed: {' '.join(args)}")
    return result.stdout


def _validate_historical_source(repo: Path, source_anchor: str, protocol_sha: str) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    _git(repo, "merge-base", "--is-ancestor", source_anchor, "HEAD")
    historical = _git(repo, "show", f"{source_anchor}:bench/registered_recovery_training_v1.py")
    historical_lf = hashlib.sha256(historical.replace(b"\r\n", b"\n")).hexdigest()
    current = repo / "bench/registered_recovery_training_v1.py"
    current_lf = _sha(current, lf=True)
    if historical_lf != protocol_sha or current_lf != protocol_sha:
        raise PriorTrainingError("historical/current training wrapper differs from executed protocol")
    return {
        "source_anchor": source_anchor,
        "historical_protocol_sha256_lf": historical_lf,
        "current_protocol_sha256_lf": current_lf,
        "historical_equals_current": True,
    }


def _recovery_refs(recovery_folder: Path, request_id: str) -> dict[str, dict[str, str]]:
    refs = {name: _ref(recovery_folder / name) for name in REQUIRED_RECOVERY_FILES}
    for rel in (
        f"approvals/{request_id}.json",
        f"admissions/{request_id}.json",
        f"intents/{request_id}.json",
        f"resources/{request_id}.json",
        f"dispatch_errors/{request_id}.json",
    ):
        refs[rel] = _ref(recovery_folder / rel)
    return refs


def build_prior_training_proof(
    training_stage: Path,
    *,
    recovery_folder: Path,
    output: Path,
) -> dict[str, str]:
    """Bind immutable prior-training files without launching or relabelling it."""
    stage = Path(training_stage).resolve(strict=True)
    recovery = Path(recovery_folder).resolve(strict=True)
    if not stage.is_dir() or not recovery.is_dir():
        raise PriorTrainingError("training stage and recovery folder must be directories")
    stage_refs = {name: _ref(stage / name) for name in REQUIRED_STAGE_FILES}
    manifest = _json(stage / "manifest.json")
    binding = _json(stage / "binding.json")
    request_id = manifest.get("run_name")
    if not isinstance(request_id, str) or not request_id:
        raise PriorTrainingError("training request identity missing")
    recovery_refs = _recovery_refs(recovery, request_id)
    receipt = _json(stage / "receipt.json")
    checkpoint = receipt.get("checkpoint")
    configs = receipt.get("configs")
    if not isinstance(checkpoint, Mapping) or not isinstance(configs, Mapping):
        raise PriorTrainingError("receipt output references missing")
    external_refs = {
        "source_checkpoint": _ref(Path(str(binding["resume_checkpoint"]["path"]))),
        "source_environment_config": _ref(Path(str(binding["resume_env_config"]["path"]))),
        "source_agent_config": _ref(Path(str(binding["resume_agent_config"]["path"]))),
        "checkpoint": _ref(Path(str(checkpoint["path"]))),
        "environment_config": _ref(Path(str(configs["environment"]["path"]))),
        "agent_config": _ref(Path(str(configs["agent"]["path"]))),
    }
    proof = {
        "schema": SCHEMA,
        "status": "completed_training_only",
        "training_stage": str(stage),
        "recovery_folder": str(recovery),
        "request_id": request_id,
        "source_anchor": binding.get("source_commit_sha"),
        "stage_files": stage_refs,
        "recovery_files": recovery_refs,
        "external_files": external_refs,
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


def _validate_binding_expectation(actual: Mapping[str, Any], expected: Mapping[str, Any] | None) -> dict[str, Any]:
    if expected is None:
        return {"expected_binding_checked": False, "provenance_only_source_commit_difference": False}
    actual_compare = dict(actual)
    expected_compare = dict(expected)
    actual_source = actual_compare.pop("source_commit_sha", None)
    expected_source = expected_compare.pop("source_commit_sha", None)
    if actual_compare != expected_compare:
        raise PriorTrainingError("expected binding changes operational training identity")
    return {
        "expected_binding_checked": True,
        "provenance_only_source_commit_difference": actual_source != expected_source,
        "executed_source_commit_sha": actual_source,
        "caller_source_commit_sha": expected_source,
    }


def _validate_recovery(
    proof: Mapping[str, Any],
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_diagnostic_manifest_sha256: str,
) -> dict[str, Any]:
    recovery = Path(str(proof["recovery_folder"])).resolve(strict=True)
    request_id = str(proof["request_id"])
    prepared = _json(recovery / "manifest.json")
    recovery_context = prepared.get("recovery_context")
    if not isinstance(recovery_context, Mapping):
        raise PriorTrainingError("prepared recovery context missing")
    protocol_ref = recovery_context.get("protocol")
    protocol_path, protocol = _load_ref(protocol_ref, label="recovery protocol")
    del protocol_path
    operation = protocol.get("operation")
    if not isinstance(operation, Mapping):
        raise PriorTrainingError("recovery operation missing")
    expected_operation = {
        "correction_id": CORRECTION,
        "recovery_mode": "resume",
        "training_iterations": 300,
        "training_seed": binding["seed"],
        "source_checkpoint_consumed": True,
    }
    if any(operation.get(key) != value for key, value in expected_operation.items()):
        raise PriorTrainingError("recovery protocol operation differs")
    if protocol.get("source_anchor") != binding["source_commit_sha"]:
        raise PriorTrainingError("recovery protocol source anchor differs")
    diagnostic = protocol.get("diagnostic")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("case_id") != "case-05"
        or diagnostic.get("session_manifest_sha256") != expected_diagnostic_manifest_sha256
        or diagnostic.get("checkpoint_sha256") != binding["resume_checkpoint"]["sha256"]
    ):
        raise PriorTrainingError("diagnostic session/case binding differs")
    closure = protocol.get("source_closure")
    if not isinstance(closure, Mapping) or closure.get("bench/registered_recovery_training_v1.py") != manifest["protocol_sha256_lf"]:
        raise PriorTrainingError("recovery protocol closure differs")

    ledger = _jsonl(recovery / "recovery.jsonl")
    rows = [row for row in ledger if row.get("request_id") == request_id]
    if [row.get("event") for row in rows] != ["proposed", "approved", "consumed", "receipt"]:
        raise PriorTrainingError("recovery request lifecycle differs")
    proposed, _, consumed, closed = rows
    if (
        proposed.get("correction_id") != CORRECTION
        or proposed.get("recovery_mode") != "resume"
        or proposed.get("loop_state_digest") != prepared.get("loop_state_digest")
        or consumed.get("request_id") != request_id
        or closed.get("verdict") != "unknown"
        or closed.get("reason") != "orphan_without_artifact"
    ):
        raise PriorTrainingError("recovery request provenance differs")
    approval = _json(recovery / f"approvals/{request_id}.json")
    admission = _json(recovery / f"admissions/{request_id}.json")
    intent = _json(recovery / f"intents/{request_id}.json")
    resources = _json(recovery / f"resources/{request_id}.json")
    dispatch = _json(recovery / f"dispatch_errors/{request_id}.json")
    effect = proposed.get("effect_key")
    protocol_sha = protocol_ref.get("sha256") if isinstance(protocol_ref, Mapping) else None
    if (
        approval.get("proof", {}).get("request_id") != request_id
        or approval.get("proof", {}).get("effect_key") != effect
        or admission.get("request_id") != request_id
        or admission.get("effect_key") != effect
        or admission.get("protocol_sha256") != protocol_sha
        or intent.get("consumption", {}).get("event") != "consumed"
        or intent.get("consumption", {}).get("request_id") != request_id
        or intent.get("context", {}).get("protocol_body", {}).get("diagnostic", {}).get(
            "session_manifest_sha256"
        ) != expected_diagnostic_manifest_sha256
        or intent.get("request", {}).get("protocol_case_arm")
        != {"arm": "corrected", "case": "case_02"}
        or resources.get("request_id") != request_id
        or resources.get("execution_refs", {}).get("training") != proof["stage_files"]["execution.json"]
        or dispatch != {"error_type": "RuntimeError", "reexecute_allowed": False,
                        "schema": "investigation_recovery_v1_dispatch_error"}
    ):
        raise PriorTrainingError("recovery approval/admission/consumption evidence differs")
    return {
        "request_id": request_id,
        "loop_state_digest": prepared["loop_state_digest"],
        "protocol_sha256": protocol_sha,
        "closed_verdict": "unknown",
        "closed_reason": "orphan_without_artifact",
        "diagnostic_manifest_sha256": expected_diagnostic_manifest_sha256,
        "case_id": "case-05",
        "method": "fixed_sequence",
        "protocol_case_arm": {"arm": "corrected", "case": "case_02"},
        "prior_dispatch_resource_ref": proof["recovery_files"][f"resources/{request_id}.json"],
        "prior_training_gpu_s": resources.get("owned_gpu_s"),
        "prior_phase_wall_s": resources.get("phase_wall_s"),
        "request_reconsumed": False,
    }


def _validate_stage(
    proof: Mapping[str, Any],
    *,
    repo: Path,
    isaaclab: Path,
    expected_source_checkpoint_sha256: str,
    expected_seed: int,
    expected_correction_id: str,
    expected_task: str,
    expected_diagnostic_manifest_sha256: str,
    expected_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if proof.get("schema") != SCHEMA or proof.get("status") != "completed_training_only":
        raise PriorTrainingError("prior training proof schema/status")
    claims = proof.get("claims")
    expected_claims = {
        "new_training_launched": False,
        "fixed_evaluation_executed": False,
        "recovery_label_available": False,
        "full_environment_state_restored": False,
        "rng_state_restored_from_checkpoint": False,
        "policy_optimizer_resumed": True,
    }
    if claims != expected_claims:
        raise PriorTrainingError("prior training proof claims differ")
    for group in ("stage_files", "recovery_files", "external_files"):
        refs = proof.get(group)
        if not isinstance(refs, Mapping):
            raise PriorTrainingError(f"{group} missing")
        for name, ref in refs.items():
            _validate_ref(ref, label=f"{group}.{name}")

    stage = Path(str(proof["training_stage"])).resolve(strict=True)
    binding = _json(stage / "binding.json")
    manifest = _json(stage / "manifest.json")
    initial = _json(stage / "initial.json")
    correction = _json(stage / "correction.json")
    receipt = _json(stage / "receipt.json")
    execution = _json(stage / "execution.json")
    launch = _json(stage / "owned/launch.json")
    if proof["source_anchor"] != SOURCE_ANCHOR or binding.get("source_commit_sha") != SOURCE_ANCHOR:
        raise PriorTrainingError("executed source anchor differs")
    if (
        binding.get("resume_checkpoint", {}).get("sha256") != expected_source_checkpoint_sha256
        or binding.get("seed") != expected_seed
        or binding.get("task") != expected_task
        or manifest.get("correction_id") != expected_correction_id
        or expected_correction_id != CORRECTION
    ):
        raise PriorTrainingError("expected source/seed/task/correction differs")
    TRAIN.validate_host_binding(binding, Path(isaaclab))
    expected_manifest = TRAIN.manifest_for(stage, binding, Path(isaaclab), str(proof["request_id"]))
    if manifest != expected_manifest:
        raise PriorTrainingError("training manifest differs from frozen wrapper")
    binding_meta = _validate_binding_expectation(binding, expected_binding)

    if (
        receipt.get("schema") != "training_correction_receipt_v1"
        or receipt.get("completed") is not True
        or receipt.get("arm") != "corrected"
        or receipt.get("fault") != "F2"
        or receipt.get("correction_id") != CORRECTION
        or receipt.get("seed", expected_seed) != expected_seed
        or receipt.get("task") != expected_task
        or receipt.get("manifest_sha256") != _sha(stage / "manifest.json")
        or receipt.get("initial_sha256") != _sha(stage / "initial.json")
        or receipt.get("correction_sha256") != _sha(stage / "correction.json")
    ):
        raise PriorTrainingError("training receipt identity differs")
    expected_external = {
        "source_checkpoint": _ref(Path(str(binding["resume_checkpoint"]["path"]))),
        "source_environment_config": _ref(Path(str(binding["resume_env_config"]["path"]))),
        "source_agent_config": _ref(Path(str(binding["resume_agent_config"]["path"]))),
        "checkpoint": _ref(Path(str(receipt["checkpoint"]["path"]))),
        "environment_config": _ref(Path(str(receipt["configs"]["environment"]["path"]))),
        "agent_config": _ref(Path(str(receipt["configs"]["agent"]["path"]))),
    }
    if proof["external_files"] != expected_external:
        raise PriorTrainingError("training external file identity differs")
    if (
        initial.get("resume_checkpoint", {}).get("sha256") != expected_source_checkpoint_sha256
        or initial.get("resume_iteration") != 299
        or initial.get("source_checkpoint_state", {}).get("iteration") != 299
        or initial.get("resume_checkpoint", {}).get("load_optimizer") is not True
        or initial.get("full_environment_state_restored") is not False
        or initial.get("rng_state_restored_from_checkpoint") is not False
    ):
        raise PriorTrainingError("training initial state differs")
    resume_scope = receipt.get("resume_scope")
    if resume_scope != {
        "full_environment_state_restored": False,
        "policy_optimizer_restored": True,
        "rng_state_restored_from_checkpoint": False,
    }:
        raise PriorTrainingError("training resume scope differs")
    budget = receipt.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("max_iterations") != 300
        or receipt.get("resume_iteration") != 299
        or receipt.get("final_iteration") != 598
        or receipt.get("actual_control_steps") != 300 * 24
        or receipt.get("aggregate_env_steps") != 300 * 24 * 4096
        or receipt.get("final_checkpoint_state", {}).get("iteration") != 598
    ):
        raise PriorTrainingError("training iteration/budget differs")
    TRAIN.LEGACY.validate_readback(correction, "F2", CORRECTION)
    if (
        correction.get("changed_targets") != [CORRECTION]
        or correction.get("restored_targets") != [CORRECTION]
        or correction.get("before_fault", {}).get("restore_body_frame_reward")
        != correction.get("after_correction", {}).get("restore_body_frame_reward")
        or correction.get("before_fault", {}).get("restore_configured_timeout")
        != correction.get("after_correction", {}).get("restore_configured_timeout")
    ):
        raise PriorTrainingError("correction scope differs")
    if execution != {
        **execution,
        "cleanup_verified": True,
        "exit_code": 0,
        "job_assigned_before_release": True,
        "natural_exit": True,
        "timed_out": False,
    }:
        raise PriorTrainingError("owned execution did not finish cleanly")
    if not isinstance(execution.get("wall_s"), (int, float)) or execution["wall_s"] <= 0:
        raise PriorTrainingError("owned execution wall time missing")
    expected_launch = [
        str(Path(isaaclab).resolve() / "_isaac_sim/python.bat"),
        "-B",
        str(Path(repo).resolve() / "bench/registered_recovery_training_v1.py"),
        "--manifest",
        str((stage / "manifest.json").resolve()),
        "--binding",
        str((stage / "binding.json").resolve()),
    ]
    if launch.get("args") != expected_launch or Path(str(launch.get("cwd"))).resolve() != Path(isaaclab).resolve():
        raise PriorTrainingError("owned launch evidence differs")
    host = TRAIN._Host(binding, stage / "manifest.json")
    host.validate_training_configuration(Path(str(initial["log_dir"])), manifest)
    history = _validate_historical_source(Path(repo), SOURCE_ANCHOR, str(manifest["protocol_sha256_lf"]))
    recovery_meta = _validate_recovery(
        proof,
        binding,
        manifest,
        expected_diagnostic_manifest_sha256=expected_diagnostic_manifest_sha256,
    )
    return {
        "binding": binding,
        "manifest": manifest,
        "initial": initial,
        "correction": correction,
        "receipt": receipt,
        "execution": execution,
        "launch": launch,
        "binding_validation": binding_meta,
        "historical_source_validation": history,
        "recovery_validation": recovery_meta,
    }


def load_completed_actuator_training(
    proof_ref: dict,
    *,
    expected_source_checkpoint_sha256: str,
    expected_seed: int,
    expected_correction_id: str,
    expected_task: str,
    expected_diagnostic_manifest_sha256: str,
    repo: Path,
    isaaclab: Path,
    expected_binding: dict | None = None,
) -> dict[str, Any]:
    """Return a verified prior ``train`` result without launching a process."""
    proof_path, proof = _load_ref(proof_ref, label="prior training proof")
    validated = _validate_stage(
        proof,
        repo=Path(repo),
        isaaclab=Path(isaaclab),
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_seed=expected_seed,
        expected_correction_id=expected_correction_id,
        expected_task=expected_task,
        expected_diagnostic_manifest_sha256=expected_diagnostic_manifest_sha256,
        expected_binding=expected_binding,
    )
    receipt = validated["receipt"]
    execution = validated["execution"]
    training_result = {
        "arm": "CORRECTED",
        "checkpoint": receipt["checkpoint"],
        "readback": validated["correction"],
        "execution": execution,
        "receipt": receipt,
        "configuration": {
            "source_checkpoint_consumed": True,
            "optimizer_state_resumed": True,
        },
    }
    return {
        "schema": "completed_registered_actuator_training_v1",
        "training_result": training_result,
        "training_stage": proof["training_stage"],
        "proof_ref": {"path": str(proof_path), "sha256": _sha(proof_path)},
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

