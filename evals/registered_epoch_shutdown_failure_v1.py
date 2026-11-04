"""Verify the registered V11 shutdown failure before V12 authority is accepted."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_RELATIVE = "bench/protocols/registered_epoch_shutdown_retry_v1.json"
POLICY_SHA256_LF = "4aa8c3f58374255a85dc1e26c6a34a35d03189f4c05755005d1c1a77d144b618"
FAILURE_PROOF_SHA256 = "91aaef7cc486b66d2ce01a1ca13fa27767466033a229eba1692ae5ebe44c5073"
RAW_ROLES = {
    "parent_plan", "controller_result", "controller_resource", "supervisor_exit",
    "recovery_manifest", "recovery_anchor", "recovery_ledger", "recovery_intent",
    "dispatch_error", "recovery_resource", "training_manifest", "training_execution",
    "checkpoint_inspection", "final_checkpoint", "environment_initial_readback",
    "environment_saved_config_check", "training_records", "training_telemetry",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ShutdownFailureError(ValueError):
    """A policy, failure proof, or immutable raw reference differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ShutdownFailureError("JSON object required")
    _canonical(value)
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShutdownFailureError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ShutdownFailureError(f"{name} must be finite and nonnegative")
    return result


def _reference(value: object, label: str) -> tuple[dict[str, str], Path]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str)
            or not isinstance(value["sha256"], str)
            or _SHA.fullmatch(value["sha256"]) is None):
        raise ShutdownFailureError(f"{label} reference differs")
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink():
        raise ShutdownFailureError(f"{label} path differs")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ShutdownFailureError(f"{label} path missing") from exc
    if _sha(path) != value["sha256"]:
        raise ShutdownFailureError(f"{label} SHA differs")
    return {"path": str(path), "sha256": value["sha256"]}, path


def load_policy(*, repo: Path = ROOT) -> dict[str, Any]:
    path = Path(repo).resolve(strict=True) / POLICY_RELATIVE
    if not path.is_file() or path.is_symlink() or _sha(path, lf=True) != POLICY_SHA256_LF:
        raise ShutdownFailureError("epoch shutdown retry policy changed")
    policy = _read(path)
    failure = policy.get("v11_failure")
    if (
        set(policy) != {
            "schema", "profile_id", "prior_completed_effect_profile",
            "prior_consumed_gpu_sdk_launches", "expected_new_gpu_sdk_launches",
            "new_gpu_sdk_launch_cap", "cumulative_gpu_sdk_launch_cap",
            "new_recovery_arms", "api_cap", "failure_proof_sha256", "v11_failure",
        }
        or policy.get("schema") != "registered_epoch_shutdown_retry_policy_v1"
        or policy.get("profile_id") != "motor_completed_epoch_close_retry_v1"
        or policy.get("prior_completed_effect_profile") != "motor_completed_v1"
        or policy.get("prior_consumed_gpu_sdk_launches") != 18
        or policy.get("expected_new_gpu_sdk_launches") != 14
        or policy.get("new_gpu_sdk_launch_cap") != 14
        or policy.get("cumulative_gpu_sdk_launch_cap") != 32
        or policy.get("new_recovery_arms") != 7
        or policy.get("api_cap") != 0
        or policy.get("failure_proof_sha256") != FAILURE_PROOF_SHA256
        or not isinstance(failure, dict)
    ):
        raise ShutdownFailureError("epoch shutdown retry policy contract differs")
    identity = failure.get("identity")
    measurements = failure.get("measurements")
    raw_sha256 = failure.get("raw_sha256")
    if (
        failure.get("status") != "learning_complete_terminal_environment_unobserved"
        or not isinstance(identity, dict)
        or identity != {
            "source_anchor": "99dcbe6d8128c050f46b69c82928a9626f1d3b56",
            "registration_anchor": "389b3ae0c85257b5548a24926584917abbc12ccb",
            "parent_plan_sha256": "0c23a2e07205db48787a11d7471d45dfedfb462db99c3599f7a1065701ec95ee",
            "case_id": "case-07", "method": "one_shot_llm",
            "request_id": "785427445a0a", "training_seed": 5866,
            "checkpoint_iteration": 299,
            "checkpoint_sha256": "c1e1720bd4b50e3ac01fc7b627665f942aea368c18ee68085eadb426900dc3a4",
        }
        or not isinstance(measurements, dict)
        or set(measurements) != {
            "simulator_sdk_launches", "cpu_checkpoint_launches",
            "cpu_telemetry_launches", "total_owned_children", "api_calls",
            "training_owned_wall_s", "phase_wall_s", "outer_owned_wall_s",
            "supervisor_call_wall_s",
        }
        or measurements.get("simulator_sdk_launches") != 1
        or measurements.get("cpu_checkpoint_launches") != 1
        or measurements.get("cpu_telemetry_launches") != 1
        or measurements.get("total_owned_children") != 3
        or measurements.get("api_calls") != 0
        or any(_finite_nonnegative(measurements[name], name) <= 0 for name in (
            "training_owned_wall_s", "phase_wall_s", "outer_owned_wall_s",
            "supervisor_call_wall_s"))
        or failure.get("scientific_outcome") != "unknown"
        or failure.get("fixed_evaluation_executed") is not False
        or failure.get("final_environment_runtime_observed") is not False
        or failure.get("original_request_reexecution_allowed") is not False
        or not isinstance(raw_sha256, dict)
        or set(raw_sha256) != RAW_ROLES
        or any(not isinstance(value, str) or _SHA.fullmatch(value) is None
               for value in raw_sha256.values())
        or raw_sha256.get("parent_plan") != identity["parent_plan_sha256"]
        or raw_sha256.get("final_checkpoint") != identity["checkpoint_sha256"]
    ):
        raise ShutdownFailureError("epoch shutdown failure identity differs")
    return policy


def verify_failure_proof(reference: object, *, repo: Path = ROOT) -> dict[str, Any]:
    proof_ref, proof_path = _reference(reference, "epoch shutdown failure proof")
    if proof_ref["sha256"] != FAILURE_PROOF_SHA256:
        raise ShutdownFailureError("epoch shutdown failure proof SHA differs")
    proof = _read(proof_path)
    policy = load_policy(repo=repo)
    failure = policy["v11_failure"]
    if (
        set(proof) != {
            "schema", "status", "identity", "measurements", "scientific_outcome",
            "fixed_evaluation_executed", "final_environment_runtime_observed",
            "original_request_reexecution_allowed", "refs",
        }
        or proof.get("schema") != "registered_epoch_shutdown_failure_proof_v1"
        or proof.get("status") != failure["status"]
        or proof.get("identity") != failure["identity"]
        or proof.get("measurements") != failure["measurements"]
        or proof.get("scientific_outcome") != "unknown"
        or proof.get("fixed_evaluation_executed") is not False
        or proof.get("final_environment_runtime_observed") is not False
        or proof.get("original_request_reexecution_allowed") is not False
        or not isinstance(proof.get("refs"), dict)
        or set(proof["refs"]) != RAW_ROLES
    ):
        raise ShutdownFailureError("epoch shutdown failure proof contract differs")
    raw_refs: dict[str, dict[str, str]] = {}
    for role in sorted(RAW_ROLES):
        raw_ref, _path = _reference(proof["refs"][role], f"epoch shutdown raw {role}")
        if raw_ref["sha256"] != failure["raw_sha256"][role]:
            raise ShutdownFailureError("epoch shutdown failure raw SHA differs")
        raw_refs[role] = raw_ref
    return {
        "proof_ref": proof_ref,
        "identity": proof["identity"],
        "measurements": proof["measurements"],
        "raw_refs": raw_refs,
    }
