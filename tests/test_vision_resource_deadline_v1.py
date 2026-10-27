from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as GPU  # pyright: ignore[reportMissingImports]
import vision_resource_authorization_v1 as AUTH  # pyright: ignore[reportMissingImports]
import vision_resource_issuer_v2 as ISSUER  # pyright: ignore[reportMissingImports]

UTC = timezone.utc  # noqa: UP017 - Python 3.10
NOW = datetime(2026, 10, 27, 7, 0, tzinfo=UTC)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8", newline="")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _source() -> dict:
    value = json.loads((ROOT / "bench/protocols/vision_brev_source_20261026_v8.json").read_text(encoding="utf-8"))
    value["resources"]["deadline_utc"] = "2026-10-27T06:00:00Z"
    return value


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_path = _write(tmp_path / "source.json", _source())
    source_ref = {"path": "source.json", "sha256_lf": AUTH._sha(source_path, lf=True)}
    catalog = [
        {"logical_id": "source-s2220-none", "isaaclab_relative": "logs/none/model_299.pt", "sha256": "1" * 64},
        {"logical_id": "source-s2220-f1", "isaaclab_relative": "logs/f1/model_299.pt", "sha256": "2" * 64},
    ]
    execution = {
        "schema": "vision_pull_record_protocol_fixture",
        "protocol_id": "native-policy-record-deadline-fixture",
        "registered_before_measurement": True,
        "source_commit": "a" * 40,
        "source_protocol": source_ref,
        "resources": {
            "owned_instance_id": "3655a80oo",
            "owned_instance_name": "rl-triage-e2e-20261024-c9f4",
            "max_owned_instances": 1,
        },
        "limits": {
            "training_actions": 0,
            "model_api_requests": 0,
            "max_policies": 2,
            "max_wall_s_per_policy": 600,
            "max_wall_compute_usd_batch": "0.580000",
        },
        "checkpoints": catalog,
    }
    execution_path = _write(tmp_path / "execution.json", execution)
    authorization = AUTH.build(
        source_path,
        execution_path,
        authorization_id="native-policy-record-ao-resource",
        registered_at_utc="2026-10-27T06:30:00Z",
        authorization_deadline_utc="2026-10-27T09:00:00Z",
        repo=tmp_path,
    )
    authorization_path = _write(tmp_path / "authorization.json", authorization)
    return source_path, execution_path, authorization_path


def _captures(tmp_path: Path, authorization_path: Path) -> tuple[Path, Path, Path, Path]:
    authorization = AUTH.validate(authorization_path, repo=tmp_path, now=NOW)
    row: dict[str, Any] = {key: None for key in GPU.PROVIDER_ROW_KEYS}
    row.update({"id": authorization["resource_id"], "name": authorization["resource_name"], "status": "STOPPED"})
    before = _write(tmp_path / "before.json", {"workspaces": [row]})
    row["status"] = "RUNNING"
    provider = _write(tmp_path / "provider.json", {"workspaces": [row]})
    gpu = _write(tmp_path / "gpu.csv", "GPU-fixture\n")
    processes = _write(tmp_path / "processes.csv", "")
    stdout = _write(tmp_path / "start.stdout", "started\n")
    stderr = _write(tmp_path / "start.stderr", "")
    start = _write(tmp_path / "start.json", {
        "schema": "v04_start_observation_v1",
        "resource_id": authorization["resource_id"],
        "resource_name": authorization["resource_name"],
        "before_inventory_path": str(before.resolve()),
        "before_inventory_sha256": SOURCE.sha256(before),
        "command": ["brev", "start", authorization["resource_name"], "--detached"],
        "request_before_utc": "2026-10-27T06:45:00Z",
        "response_after_utc": "2026-10-27T06:45:05Z",
        "exit_code": 0,
        "stdout_path": str(stdout.resolve()),
        "stdout_sha256": SOURCE.sha256(stdout),
        "stderr_path": str(stderr.resolve()),
        "stderr_sha256": SOURCE.sha256(stderr),
    })
    return start, provider, gpu, processes


def test_new_authorization_keeps_expired_source_deadline_as_history(tmp_path: Path) -> None:
    source_path, execution_path, authorization_path = _fixture(tmp_path)
    checked = AUTH.validate(authorization_path, repo=tmp_path, now=NOW)
    assert SOURCE._stamp(checked["historical_source_deadline_utc"]) < NOW
    assert checked["authorization_deadline_utc"] == "2026-10-27T09:00:00Z"
    assert checked["source_protocol"]["sha256_lf"] == AUTH._sha(source_path, lf=True)
    assert checked["execution_protocol"]["sha256_lf"] == AUTH._sha(execution_path, lf=True)
    assert checked["scope"]["training_actions"] == 0


@pytest.mark.parametrize("mutation", ["source_deadline", "short_deadline", "catalog", "training"])
def test_authorization_rejects_implicit_extension_or_scope_change(tmp_path: Path, mutation: str) -> None:
    _source_path, execution_path, authorization_path = _fixture(tmp_path)
    value = json.loads(authorization_path.read_text(encoding="utf-8"))
    if mutation == "source_deadline":
        value["historical_source_deadline_utc"] = "2026-10-28T06:00:00Z"
    elif mutation == "short_deadline":
        value["authorization_deadline_utc"] = (NOW + timedelta(seconds=1799)).isoformat()
    elif mutation == "catalog":
        value["checkpoint_catalog"][0]["sha256"] = "f" * 64
    else:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["limits"]["training_actions"] = 1
        _write(execution_path, execution)
        value["execution_protocol"]["sha256_lf"] = AUTH._sha(execution_path, lf=True)
    _write(authorization_path, value)
    with pytest.raises(ValueError):
        AUTH.validate(authorization_path, repo=tmp_path, now=NOW)


def test_issuer_binds_new_deadline_and_old_source_sha_without_rewriting_either(tmp_path: Path) -> None:
    source_path, _execution_path, authorization_path = _fixture(tmp_path)
    source_before = source_path.read_bytes()
    authorization_before = authorization_path.read_bytes()
    start, provider, gpu, processes = _captures(tmp_path, authorization_path)
    output = tmp_path / "issued"
    proof = ISSUER.issue(
        authorization_path, start, provider, gpu, processes, output, now=NOW, repo=tmp_path
    )
    issuance = ISSUER.verify_issuance(
        authorization_path, output / "issuance.json", output / "resource.json",
        now=NOW, repo=tmp_path,
    )
    assert proof["protocol_sha256"] == AUTH._sha(source_path, lf=True)
    assert proof["lease_id"] == f"v04-{SOURCE.sha256(start)[:20]}"
    assert proof["expires_at_utc"] == "2026-10-27T09:00:00+00:00"
    assert issuance["historical_source_deadline_utc"] == "2026-10-27T06:00:00Z"
    assert issuance["source_experiment_extended"] is False
    assert source_path.read_bytes() == source_before
    assert authorization_path.read_bytes() == authorization_before


def test_issuance_rejects_different_start_for_same_authorization(tmp_path: Path) -> None:
    _source_path, _execution_path, authorization_path = _fixture(tmp_path)
    start, provider, gpu, processes = _captures(tmp_path, authorization_path)
    output = tmp_path / "issued"
    ISSUER.issue(authorization_path, start, provider, gpu, processes, output, now=NOW, repo=tmp_path)
    issuance_path = output / "issuance.json"
    issuance = json.loads(issuance_path.read_text(encoding="utf-8"))
    issuance["start_observation_sha256"] = "f" * 64
    _write(issuance_path, issuance)
    with pytest.raises(ValueError, match="resource"):
        ISSUER.verify_issuance(authorization_path, issuance_path, output / "resource.json", now=NOW, repo=tmp_path)


def test_issuance_rejects_changed_authorization_deadline(tmp_path: Path) -> None:
    _source_path, _execution_path, authorization_path = _fixture(tmp_path)
    start, provider, gpu, processes = _captures(tmp_path, authorization_path)
    output = tmp_path / "issued"
    ISSUER.issue(
        authorization_path, start, provider, gpu, processes, output, now=NOW, repo=tmp_path
    )
    changed = deepcopy(json.loads(authorization_path.read_text(encoding="utf-8")))
    changed["authorization_deadline_utc"] = "2026-10-27T10:00:00Z"
    _write(authorization_path, changed)
    with pytest.raises(ValueError, match="resource"):
        ISSUER.verify_issuance(
            authorization_path, output / "issuance.json", output / "resource.json",
            now=NOW, repo=tmp_path,
        )


def test_execution_gate_requires_authorized_protocol_resource_and_checkpoint(tmp_path: Path) -> None:
    _source_path, execution_path, authorization_path = _fixture(tmp_path)
    start, provider, gpu, processes = _captures(tmp_path, authorization_path)
    output = tmp_path / "issued"
    ISSUER.issue(
        authorization_path, start, provider, gpu, processes, output, now=NOW, repo=tmp_path
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    checkpoint = execution["checkpoints"][0]
    job = {
        "protocol": {"path": str(execution_path.resolve()), "sha256_lf": AUTH._sha(execution_path, lf=True)},
        "resource_proof": {"path": str((output / "resource.json").resolve()), "sha256": SOURCE.sha256(output / "resource.json")},
        "checkpoint": {
            "logical_id": checkpoint["logical_id"],
            "sha256": checkpoint["sha256"],
            "path": str((tmp_path / checkpoint["isaaclab_relative"]).resolve()),
        },
    }
    job_path = _write(tmp_path / "job.json", job)
    checked = ISSUER.verify_job_binding(
        authorization_path, output / "issuance.json", output / "resource.json", job_path,
        now=NOW, repo=tmp_path,
    )
    assert checked["job"] == job
    job["checkpoint"]["sha256"] = "f" * 64
    _write(job_path, job)
    with pytest.raises(ValueError, match="checkpoint"):
        ISSUER.verify_job_binding(
            authorization_path, output / "issuance.json", output / "resource.json", job_path,
            now=NOW, repo=tmp_path,
        )
