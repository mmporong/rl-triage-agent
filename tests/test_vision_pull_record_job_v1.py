from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_job_v1 as J  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_ref = {"path": "source.json", "sha256_lf": ""}
    source = {
        "schema": "vision_brev_source_protocol_v1",
        "protocol_id": "vision-source-20261026x",
        "record_profile_sha256": J.RECORD.digest(J.RECORD.fixed_profile()),
    }
    source_path = _write(tmp_path / source_ref["path"], source)
    source_ref["sha256_lf"] = J._sha(source_path, lf=True)
    rows = [
        {"logical_id": identity, "isaaclab_relative": f"logs/rsl_rl/{identity}.pt", "sha256": str(index + 1) * 64}
        for index, identity in enumerate(J.CHECKPOINT_IDS)
    ]
    base = {
        "schema": J.RECORD_JOB.PROTOCOL_SCHEMA,
        "protocol_id": "prepared-policy-record-20261026z",
        "profile_sha256": J.RECORD.digest(J.RECORD.fixed_profile()),
        "parent_source_protocol": source_ref,
        "checkpoints": rows,
    }
    base_path = _write(tmp_path / "base.json", base)
    base_ref = {"path": "base.json", "sha256_lf": J._sha(base_path, lf=True)}
    old_sources = {f"bench/source-{index}.py": "a" * 64 for index in range(351)}
    previous_resources = {
        "owned_instance_id": "3655a80oo",
        "owned_instance_name": "rl-triage-e2e-20261024-c9f4",
        "max_owned_instances": 1,
        "compute_quote_usd_per_h": "1.74", "total_program_budget_usd": "20.000000",
        "prior_compute_estimate_usd": "18.330565", "fresh_lifecycle_budget_usd": "1.669435",
        "prior_stop_receipts": [{"path": f"old-{i}.json", "sha256": "b" * 64} for i in range(21)],
    }
    diagnostic = {
        "schema": J.DIAGNOSTIC.PROTOCOL_SCHEMA, "protocol_id": "annotator-pull-20261027ai",
        "source_protocol": source_ref, "base_record_protocol": base_ref,
        "source_sha256_lf": old_sources, "resources": previous_resources,
    }
    diagnostic_path = _write(tmp_path / "diagnostic.json", diagnostic)
    diagnostic_ref = {"path": "diagnostic.json", "sha256_lf": J._sha(diagnostic_path, lf=True)}
    stop = {
        "schema": "v04_owned_vm_stop_receipt_v1", "status": "STOPPED",
        "execution_identity": {"run_id": "annotator-pull-20261027ai", "protocol_sha256": source_ref["sha256_lf"]},
        "cost_estimate": {"estimated_compute_cost_usd": "0.150112"},
    }
    stop_path = _write(tmp_path / "stop.json", stop)
    stop_ref = {"path": "stop.json", "sha256": J._sha(stop_path)}
    sources = dict(old_sources)
    for name in J.REQUIRED_SOURCE_FILES:
        sources[name] = "c" * 64
    resources = deepcopy(previous_resources)
    resources.update({
        "prior_compute_estimate_usd": "18.480677", "fresh_lifecycle_budget_usd": "1.519323",
        "prior_stop_receipts": [*previous_resources["prior_stop_receipts"], stop_ref],
    })
    protocol = {
        "schema": J.PROTOCOL_SCHEMA, "protocol_id": J.PROTOCOL_ID, "mode": J.MODE,
        "registered_before_measurement": True, "source_commit": "d" * 40,
        "source_sha256_lf": sources,
        "predecessor": {"diagnostic": diagnostic_ref, "appended_stop": stop_ref},
        "base_record_protocol": base_ref, "source_protocol": source_ref,
        "checkpoints": rows, "resources": resources, "limits": J.LIMITS,
        "preregistration": {
            "prediction": "두 checkpoint를 별도 AABA 뒤 clean pull로 녹화한다.",
            "outcomes": {"recording_complete": "complete", "capture_hold": "hold", "infrastructure_failure": "infra"},
            "fixed_flags": J.FIXED_FLAGS,
        },
    }
    protocol_path = _write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(J, "EXPECTED_DIAGNOSTIC", diagnostic_ref)
    monkeypatch.setattr(J, "EXPECTED_BASE", base_ref)
    monkeypatch.setattr(J, "EXPECTED_SOURCE", source_ref)
    monkeypatch.setattr(J, "EXPECTED_STOP", stop_ref)
    monkeypatch.setattr(J.COMMON, "_verify_head_protocol", lambda *_args: None)
    monkeypatch.setattr(J.COMMON, "_verify_source_blobs", lambda *_args: None)
    monkeypatch.setattr(J.COMMON, "_git", lambda *_args: SimpleNamespace(returncode=0))
    return protocol_path


def test_contract_requires_per_checkpoint_aaba_and_two_registered_checkpoints() -> None:
    assert J.CHECKPOINT_IDS == ("source-s2220-none", "source-s2220-f1")
    assert J.FIXED_FLAGS["per_checkpoint_aaba_required"] is True
    assert J.LIMITS["max_policies"] == 2
    assert J.LIMITS["max_wall_s_per_policy"] == 600
    assert J.EXPECTED_PRIOR + J.EXPECTED_FRESH == J.EXPECTED_TOTAL


@pytest.mark.parametrize("mutation", [None, "predecessor", "source", "stop", "closure"])
def test_bounded_predecessor_and_current_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    path = _fixture(tmp_path, monkeypatch)
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "predecessor":
        value["predecessor"]["diagnostic"]["sha256_lf"] = "0" * 64
    elif mutation == "source":
        diagnostic = json.loads((tmp_path / "diagnostic.json").read_text(encoding="utf-8"))
        diagnostic["source_sha256_lf"].pop(next(iter(diagnostic["source_sha256_lf"])))
        _write(tmp_path / "diagnostic.json", diagnostic)
    elif mutation == "stop":
        value["predecessor"]["appended_stop"]["sha256"] = "0" * 64
    elif mutation == "closure":
        value["source_sha256_lf"].pop("bench/vision_pull_record_worker_v1.py")
    path.write_text(json.dumps(value), encoding="utf-8")
    if mutation is None:
        checked = J.validate_protocol(path, repo=tmp_path)
        assert [row["logical_id"] for row in checked["checkpoints"]] == list(J.CHECKPOINT_IDS)
    else:
        with pytest.raises((TypeError, ValueError)):
            J.validate_protocol(path, repo=tmp_path)
