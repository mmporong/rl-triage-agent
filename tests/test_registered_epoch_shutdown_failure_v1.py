import hashlib
import json
from pathlib import Path

import pytest

from evals import registered_epoch_shutdown_failure_v1 as failure


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    raw_paths = {
        role: _write(tmp_path / "raw" / f"{role}.json", {"role": role})
        for role in failure.RAW_ROLES
    }
    raw_refs = {
        role: {"path": str(path.resolve()), "sha256": _sha(path)}
        for role, path in raw_paths.items()
    }
    identity = {
        "source_anchor": "1" * 40, "registration_anchor": "2" * 40,
        "parent_plan_sha256": raw_refs["parent_plan"]["sha256"],
        "case_id": "case-07", "method": "one_shot_llm", "request_id": "request",
        "training_seed": 5866, "checkpoint_iteration": 299,
        "checkpoint_sha256": raw_refs["final_checkpoint"]["sha256"],
    }
    measurements = {
        "simulator_sdk_launches": 1, "cpu_checkpoint_launches": 1,
        "cpu_telemetry_launches": 1, "total_owned_children": 3, "api_calls": 0,
        "training_owned_wall_s": 1.0, "phase_wall_s": 2.0,
        "outer_owned_wall_s": 3.0, "supervisor_call_wall_s": 4.0,
    }
    proof = {
        "schema": "registered_epoch_shutdown_failure_proof_v1",
        "status": "learning_complete_terminal_environment_unobserved",
        "identity": identity, "measurements": measurements,
        "scientific_outcome": "unknown", "fixed_evaluation_executed": False,
        "final_environment_runtime_observed": False,
        "original_request_reexecution_allowed": False, "refs": raw_refs,
    }
    proof_path = _write(tmp_path / "failure-proof.json", proof)
    policy = {
        "v11_failure": {
            "status": proof["status"], "identity": identity,
            "measurements": measurements, "scientific_outcome": "unknown",
            "fixed_evaluation_executed": False,
            "final_environment_runtime_observed": False,
            "original_request_reexecution_allowed": False,
            "raw_sha256": {role: ref["sha256"] for role, ref in raw_refs.items()},
        }
    }
    return proof_path, policy, raw_paths


def _install_fixture(monkeypatch: pytest.MonkeyPatch, proof_path: Path, policy: dict) -> None:
    monkeypatch.setattr(failure, "FAILURE_PROOF_SHA256", _sha(proof_path))
    monkeypatch.setattr(failure, "load_policy", lambda **_kwargs: policy)


def test_registered_public_policy_loads() -> None:
    policy = failure.load_policy()
    assert policy["failure_proof_sha256"] == failure.FAILURE_PROOF_SHA256
    assert policy["v11_failure"]["scientific_outcome"] == "unknown"


def test_failure_proof_verifies_all_raw_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_path, policy, _raw_paths = _fixture(tmp_path)
    _install_fixture(monkeypatch, proof_path, policy)
    reference = {"path": str(proof_path.resolve()), "sha256": _sha(proof_path)}
    result = failure.verify_failure_proof(reference, repo=tmp_path)
    assert result["proof_ref"] == reference
    assert set(result["raw_refs"]) == failure.RAW_ROLES


def test_failure_proof_rejects_missing_raw_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_path, policy, _raw_paths = _fixture(tmp_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["refs"].pop("training_execution")
    _write(proof_path, proof)
    _install_fixture(monkeypatch, proof_path, policy)
    with pytest.raises(failure.ShutdownFailureError, match="proof contract"):
        failure.verify_failure_proof(
            {"path": str(proof_path.resolve()), "sha256": _sha(proof_path)}, repo=tmp_path)


def test_failure_proof_rejects_changed_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_path, policy, raw_paths = _fixture(tmp_path)
    _install_fixture(monkeypatch, proof_path, policy)
    raw_paths["training_execution"].write_text("changed", encoding="utf-8")
    with pytest.raises(failure.ShutdownFailureError, match="SHA differs"):
        failure.verify_failure_proof(
            {"path": str(proof_path.resolve()), "sha256": _sha(proof_path)}, repo=tmp_path)


def test_failure_proof_rejects_unregistered_proof_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_path, policy, _raw_paths = _fixture(tmp_path)
    monkeypatch.setattr(failure, "load_policy", lambda **_kwargs: policy)
    monkeypatch.setattr(failure, "FAILURE_PROOF_SHA256", "0" * 64)
    with pytest.raises(failure.ShutdownFailureError, match="proof SHA"):
        failure.verify_failure_proof(
            {"path": str(proof_path.resolve()), "sha256": _sha(proof_path)}, repo=tmp_path)
