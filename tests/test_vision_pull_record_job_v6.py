from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "tests")]

import test_vision_pull_record_job_v5 as V5  # pyright: ignore[reportMissingImports]
import vision_pull_record_job_v6 as J  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _result(protocol_ref: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "vision_pull_record_infrastructure_failure_v1",
        "protocol_id": J.PREVIOUS.PROTOCOL_ID,
        "status": "HOLD",
        "protocol": protocol_ref,
        "failure": {
            "stage": "resource_registration",
            "resource_proof_created": False,
        },
        "cleanup": {"vm_status": "STOPPED"},
        "actual_recordings": 0,
        "new_training_actions": 0,
        "model_api_requests": 0,
        "capture_approved": False,
        "used_for_decision": False,
        "cost": {
            "failed_lifecycle_estimate_usd": "0.272854",
            "read_only_recovery_estimate_usd": "0.064157",
            "cumulative_program_estimate_usd": "20.087924",
            "remaining_program_estimate_usd": "4.912076",
            "program_cap_usd": "25.000000",
        },
    }


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    previous_path = V5._fixture(tmp_path, monkeypatch)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_ref = {"path": "protocol.json", "sha256_lf": J._sha(previous_path, lf=True)}
    result_path = _write(tmp_path / "an-summary.json", _result(previous_ref))
    result_ref = {"path": "an-summary.json", "sha256": J._sha(result_path)}
    sources = dict(previous["source_sha256_lf"])
    for name in J.REQUIRED_SOURCE_FILES:
        sources.setdefault(name, "e" * 64)
    protocol = deepcopy(previous)
    protocol.update(
        schema=J.PROTOCOL_SCHEMA,
        protocol_id=J.PROTOCOL_ID,
        source_sha256_lf=sources,
        predecessor={
            "diagnostic": previous["predecessor"]["diagnostic"],
            "appended_stops": [*previous["predecessor"]["appended_stops"], result_ref],
        },
        limits=J.LIMITS,
    )
    protocol["resources"].update(
        prior_compute_estimate_usd="20.087924",
        fresh_lifecycle_budget_usd="4.912076",
        prior_stop_receipts=[*previous["resources"]["prior_stop_receipts"], result_ref],
    )
    path = _write(tmp_path / "protocol-v6.json", protocol)
    monkeypatch.setattr(J, "EXPECTED_PREVIOUS_PROTOCOL", previous_ref)
    monkeypatch.setattr(J, "EXPECTED_PREVIOUS_RESULT", result_ref)
    return path


def test_protocol_preserves_source_checkpoint_and_recording_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture(tmp_path, monkeypatch)
    checked = J.validate_protocol(path, repo=tmp_path)
    assert checked["protocol_id"] == "native-policy-record-20261027ao"
    assert checked["limits"]["frames_per_policy"] == 625
    assert checked["limits"]["fps"] == 50
    assert checked["checkpoints"] == checked["_previous"]["checkpoints"]
    assert checked["source_protocol"] == checked["_previous"]["source_protocol"]
    assert checked["resources"]["prior_compute_estimate_usd"] == "20.087924"
    assert checked["resources"]["fresh_lifecycle_budget_usd"] == "4.912076"


@pytest.mark.parametrize("mutation", ["result_sha", "result_status", "result_cost", "checkpoint", "budget"])
def test_protocol_rejects_changed_an_evidence_lineage_or_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = fixture(tmp_path, monkeypatch)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    result_path = tmp_path / "an-summary.json"
    if mutation == "result_sha":
        protocol["predecessor"]["appended_stops"][-1]["sha256"] = "0" * 64
    elif mutation == "result_status":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["status"] = "complete"
        _write(result_path, result)
        protocol["predecessor"]["appended_stops"][-1]["sha256"] = J._sha(result_path)
        protocol["resources"]["prior_stop_receipts"][-1]["sha256"] = J._sha(result_path)
        monkeypatch.setattr(J, "EXPECTED_PREVIOUS_RESULT", protocol["predecessor"]["appended_stops"][-1])
    elif mutation == "result_cost":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["cost"]["read_only_recovery_estimate_usd"] = "0.064158"
        _write(result_path, result)
        protocol["predecessor"]["appended_stops"][-1]["sha256"] = J._sha(result_path)
        protocol["resources"]["prior_stop_receipts"][-1]["sha256"] = J._sha(result_path)
        monkeypatch.setattr(J, "EXPECTED_PREVIOUS_RESULT", protocol["predecessor"]["appended_stops"][-1])
    elif mutation == "checkpoint":
        protocol["checkpoints"][0]["sha256"] = "f" * 64
    else:
        protocol["resources"]["fresh_lifecycle_budget_usd"] = "4.912075"
    _write(path, protocol)
    with pytest.raises(ValueError):
        J.validate_protocol(path, repo=tmp_path)


def test_validate_job_calls_authorization_gate_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = fixture(tmp_path, monkeypatch)
    protocol = J.validate_protocol(protocol_path, repo=tmp_path)
    lab = tmp_path / "lab"
    lab.mkdir()
    row = protocol["checkpoints"][0]
    checkpoint_path = lab / row["isaaclab_relative"]
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    resource = _write(tmp_path / "resource.json", {
        "protocol_sha256": protocol["source_protocol"]["sha256_lf"],
        "expires_at_utc": "2026-10-28T09:00:00Z",
        "lease_id": f"v04-{'a' * 20}",
    })
    authorization = _write(tmp_path / "authorization.json", {"authorization": True})
    scope = {
        "operation": "existing_checkpoint_recording_only",
        "training_actions": 0,
        "model_api_requests": 0,
        "max_full_recordings": 2,
        "max_wall_s_per_recording": 600,
        "max_compute_usd_batch": "0.580000",
    }
    authorization_value = {
        "source_protocol": protocol["source_protocol"],
        "execution_protocol": {"path": "protocol-v6.json", "sha256_lf": J._sha(protocol_path, lf=True)},
        "historical_source_deadline_utc": "2026-10-27T06:00:00Z",
        "authorization_deadline_utc": "2026-10-28T09:00:00Z",
        "resource_id": "3655a80oo",
        "scope": scope,
        "checkpoint_catalog": protocol["checkpoints"],
        "_execution_path": protocol_path.resolve(),
    }
    issuance = _write(tmp_path / "issuance.json", {
        "schema": J.ISSUER.ISSUANCE_SCHEMA,
        "authorization_path": str(authorization.resolve()),
        "authorization_sha256_lf": J.ISSUER.AUTH._sha(authorization, lf=True),
        "execution_protocol_sha256_lf": J._sha(protocol_path, lf=True),
        "historical_source_protocol_sha256_lf": protocol["source_protocol"]["sha256_lf"],
        "historical_source_deadline_utc": "2026-10-27T06:00:00Z",
        "authorization_deadline_utc": "2026-10-28T09:00:00Z",
        "resource_id": "3655a80oo",
        "resource_proof_sha256": J._sha(resource),
        "start_observation_sha256": "a" * 64,
        "scope": scope,
        "source_experiment_extended": False,
        "gpu_query_performed_by_issuer": False,
        "vm_mutation_performed_by_issuer": False,
    })
    stage = tmp_path / "stage"
    stage.mkdir()
    job = {
        "schema": J.JOB_SCHEMA,
        "mode": J.MODE,
        "job_id": f"{J.PROTOCOL_ID}-{row['logical_id']}",
        "source": {"repo_root": str(tmp_path.resolve()), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path.resolve()), "sha256_lf": J._sha(protocol_path, lf=True)},
        "resource_authorization": {"path": str(authorization.resolve()), "sha256_lf": J._sha(authorization, lf=True)},
        "resource_issuance": {"path": str(issuance.resolve()), "sha256": J._sha(issuance)},
        "resource_proof": {"path": str(resource.resolve()), "sha256": J._sha(resource)},
        "isaaclab": str(lab.resolve()),
        "python": str(tmp_path / "python"),
        "output_directory": str((stage / "output").resolve()),
        "profile": J.RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": row["sha256"],
                       "logical_id": row["logical_id"]},
    }
    job_path = _write(stage / "job.json", job)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(J.RECORD_JOB, "_checkpoint", lambda *_args: (row, checkpoint_path.resolve()))
    monkeypatch.setattr(J.ISSUER.AUTH, "validate", lambda *args, **kwargs: authorization_value)
    original_gate = J.ISSUER.verify_job_binding

    def gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(args)
        return original_gate(*args, **kwargs)

    monkeypatch.setattr(J.ISSUER, "verify_job_binding", gate)
    monkeypatch.setattr(J.COMMON, "_resource", lambda *args: {"resource_id": "3655a80oo"})
    monkeypatch.setattr(J.COMMON, "_python_executable", lambda *args: None)
    monkeypatch.setattr(J.COMMON, "_verify_runtime_sources", lambda *args: None)
    checked = J.validate_job(job_path)
    assert checked["job"]["resource_authorization"]["sha256_lf"] == J._sha(authorization, lf=True)
    assert len(calls) == 1
    assert calls[0][0:4] == (authorization, issuance, resource, job_path)
