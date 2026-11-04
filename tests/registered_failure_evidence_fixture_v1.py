from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals import registered_epoch_namespace_failure_v1 as namespace_failure
from evals import registered_epoch_shutdown_failure_v1 as shutdown_failure


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def namespace_fixture(tmp_path: Path, monkeypatch: Any) -> dict[str, str]:
    raw: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    for role in namespace_failure.REFERENCE_SHA256:
        path = _write(tmp_path / "namespace-raw" / f"{role}.json", {"role": role})
        hashes[role] = _sha(path)
        raw[role] = {"path": str(path.resolve()), "sha256": hashes[role]}
    monkeypatch.setattr(namespace_failure, "REFERENCE_SHA256", hashes)
    monkeypatch.setattr(namespace_failure, "PLAN_SHA256", hashes["plan"])
    body = {
        "schema": "registered_run_name_collision_evidence_v1",
        "status": "pre_sdk_infrastructure_failure",
        "source_anchor": namespace_failure.SOURCE_ANCHOR,
        "registration_anchor": namespace_failure.REGISTRATION_ANCHOR,
        "case_id": "case-07",
        "method": "one_shot_llm",
        "request_id": namespace_failure.REQUEST_ID,
        "registered_training_seed": 6066,
        "planned_next_restart_seeds_not_executed": [6266, 6267],
        "training_executed": False,
        "fixed_evaluation_executed": False,
        "same_consumed_request_reexecute_allowed": False,
        "prior_record_preserved": True,
        "SDK": 0,
        "API": 0,
        "CPU_helper_children": 0,
        "observed_error": {"error_type": "FileExistsError", "reexecute_allowed": False},
        "references": raw,
    }
    path = _write(tmp_path / "namespace-context.json", body)
    digest = _sha(path)
    monkeypatch.setattr(namespace_failure, "CONTEXT_SHA256", digest)
    return {"path": str(path.resolve()), "sha256": digest}


def shutdown_fixture(tmp_path: Path, monkeypatch: Any) -> dict[str, str]:
    raw_refs: dict[str, dict[str, str]] = {}
    raw_sha256: dict[str, str] = {}
    for role in sorted(shutdown_failure.RAW_ROLES):
        path = _write(tmp_path / "shutdown-raw" / f"{role}.json", {"role": role})
        raw_sha256[role] = _sha(path)
        raw_refs[role] = {"path": str(path.resolve()), "sha256": raw_sha256[role]}
    identity = {
        "source_anchor": "99dcbe6d8128c050f46b69c82928a9626f1d3b56",
        "registration_anchor": "389b3ae0c85257b5548a24926584917abbc12ccb",
        "parent_plan_sha256": "0c23a2e07205db48787a11d7471d45dfedfb462db99c3599f7a1065701ec95ee",
        "case_id": "case-07",
        "method": "one_shot_llm",
        "request_id": "785427445a0a",
        "training_seed": 5866,
        "checkpoint_iteration": 299,
        "checkpoint_sha256": "c1e1720bd4b50e3ac01fc7b627665f942aea368c18ee68085eadb426900dc3a4",
    }
    measurements = {
        "simulator_sdk_launches": 1,
        "cpu_checkpoint_launches": 1,
        "cpu_telemetry_launches": 1,
        "total_owned_children": 3,
        "api_calls": 0,
        "training_owned_wall_s": 1.0,
        "phase_wall_s": 2.0,
        "outer_owned_wall_s": 3.0,
        "supervisor_call_wall_s": 4.0,
    }
    failure = {
        "status": "learning_complete_terminal_environment_unobserved",
        "identity": identity,
        "measurements": measurements,
        "scientific_outcome": "unknown",
        "fixed_evaluation_executed": False,
        "final_environment_runtime_observed": False,
        "original_request_reexecution_allowed": False,
        "raw_sha256": raw_sha256,
    }
    proof = {
        "schema": "registered_epoch_shutdown_failure_proof_v1",
        **{key: failure[key] for key in (
            "status", "identity", "measurements", "scientific_outcome",
            "fixed_evaluation_executed", "final_environment_runtime_observed",
            "original_request_reexecution_allowed",
        )},
        "refs": raw_refs,
    }
    proof_path = _write(tmp_path / "shutdown-proof.json", proof)
    proof_digest = _sha(proof_path)
    policy = {
        "schema": "registered_epoch_shutdown_retry_policy_v1",
        "profile_id": "motor_completed_epoch_close_retry_v1",
        "prior_completed_effect_profile": "motor_completed_v1",
        "prior_consumed_gpu_sdk_launches": 18,
        "expected_new_gpu_sdk_launches": 14,
        "new_gpu_sdk_launch_cap": 14,
        "cumulative_gpu_sdk_launch_cap": 32,
        "new_recovery_arms": 7,
        "api_cap": 0,
        "failure_proof_sha256": proof_digest,
        "v11_failure": failure,
    }
    monkeypatch.setattr(shutdown_failure, "FAILURE_PROOF_SHA256", proof_digest)
    monkeypatch.setattr(shutdown_failure, "load_policy", lambda **_kwargs: policy)
    return {"path": str(proof_path.resolve()), "sha256": proof_digest}
