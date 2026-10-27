from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_job_v4 as J  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _resources(stops: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "owned_instance_id": "3655a80oo",
        "owned_instance_name": "rl-triage-e2e-20261024-c9f4",
        "max_owned_instances": 1,
        "compute_quote_usd_per_h": "1.74",
        "total_program_budget_usd": "20.000000",
        "prior_compute_estimate_usd": "18.330565",
        "fresh_lifecycle_budget_usd": "1.669435",
        "prior_stop_receipts": stops,
    }


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
        {
            "logical_id": identity,
            "isaaclab_relative": f"logs/rsl_rl/{identity}.pt",
            "sha256": str(index + 1) * 64,
        }
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

    old_sources = {name: "a" * 64 for name in J.DIAGNOSTIC.REQUIRED_SOURCE_FILES}
    filler = 0
    while len(old_sources) < 351:
        old_sources[f"bench/ai-source-{filler}.py"] = "a" * 64
        filler += 1
    old_stops = [{"path": f"old-{index}.json", "sha256": "b" * 64} for index in range(21)]
    previous_resources = _resources(old_stops)
    diagnostic = {
        "schema": J.DIAGNOSTIC.PROTOCOL_SCHEMA,
        "protocol_id": "annotator-pull-20261027ai",
        "source_protocol": source_ref,
        "base_record_protocol": base_ref,
        "source_sha256_lf": old_sources,
        "resources": previous_resources,
    }
    diagnostic_path = _write(tmp_path / "diagnostic.json", diagnostic)
    diagnostic_ref = {"path": "diagnostic.json", "sha256_lf": J._sha(diagnostic_path, lf=True)}

    stop_specs = (
        ("ai-stop.json", "v04_owned_vm_stop_receipt_v1", "annotator-pull-20261027ai", "0.150112"),
        ("infra-stop.json", "vision_native_policy_record_pre_candidate_abort_metadata_v1", None, "0.058680"),
        ("retry1-stop.json", "v04_owned_vm_stop_receipt_v1", "native-policy-record-20261027aj", "0.242750"),
        ("retry2-stop.json", "v04_owned_vm_stop_receipt_v1", "native-policy-record-20261027aj", "0.336100"),
        ("ak-stop.json", "v04_owned_vm_stop_receipt_v1", "native-policy-record-20261027ak", "0.333297"),
        ("al-stop.json", "v04_owned_vm_stop_receipt_v1", "native-policy-record-20261027al", "0.140889"),
    )
    expected_stops = []
    stop_refs = []
    for name, schema, run_id, cost in stop_specs:
        if run_id is None:
            body = {
                "schema": schema,
                "protocol_id": "native-policy-record-20261027aj",
                "source_protocol_sha256_lf": source_ref["sha256_lf"],
                "status": "STOPPED",
                "resource_id": previous_resources["owned_instance_id"],
                "resource_name": previous_resources["owned_instance_name"],
                "gpu_measurement_started": False,
                "resource_proof_created": False,
                "training_actions": 0,
                "model_api_requests": 0,
                "full_recordings": 0,
                "capture_approved": False,
                "used_for_decision": False,
                "cost_estimate": {"estimated_compute_cost_usd": cost},
            }
        else:
            body = {
                "schema": schema,
                "status": "STOPPED",
                "execution_identity": {
                    "run_id": run_id,
                    "protocol_sha256": source_ref["sha256_lf"],
                },
                "cost_estimate": {"estimated_compute_cost_usd": cost},
            }
        path = _write(tmp_path / name, body)
        ref = {"path": name, "sha256": J._sha(path)}
        stop_refs.append(ref)
        expected_stops.append({**ref, "schema": schema, "run_id": run_id, "cost": cost})

    sources = dict(old_sources)
    for name in J.REQUIRED_SOURCE_FILES - J.DIAGNOSTIC.REQUIRED_SOURCE_FILES:
        sources[name] = "c" * 64
    assert len(sources) == 383
    resources = deepcopy(previous_resources)
    resources.update(
        {
            "prior_compute_estimate_usd": "19.592393",
            "fresh_lifecycle_budget_usd": "0.407607",
            "prior_stop_receipts": [*old_stops, *stop_refs],
        }
    )
    protocol = {
        "schema": J.PROTOCOL_SCHEMA,
        "protocol_id": J.PROTOCOL_ID,
        "mode": J.MODE,
        "registered_before_measurement": True,
        "source_commit": "d" * 40,
        "source_sha256_lf": sources,
        "predecessor": {"diagnostic": diagnostic_ref, "appended_stops": stop_refs},
        "base_record_protocol": base_ref,
        "source_protocol": source_ref,
        "checkpoints": rows,
        "resources": resources,
        "limits": J.LIMITS,
        "preregistration": {
            "prediction": "두 checkpoint를 분리된 AABA raw receipt 뒤 clean pull로 녹화한다.",
            "outcomes": {
                "recording_complete": "complete",
                "capture_hold": "hold",
                "infrastructure_failure": "infra",
            },
            "fixed_flags": J.FIXED_FLAGS,
        },
    }
    protocol_path = _write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(J, "EXPECTED_DIAGNOSTIC", diagnostic_ref)
    monkeypatch.setattr(J, "EXPECTED_BASE", base_ref)
    monkeypatch.setattr(J, "EXPECTED_SOURCE", source_ref)
    monkeypatch.setattr(J, "EXPECTED_STOPS", tuple(expected_stops))
    monkeypatch.setattr(J.COMMON, "_verify_head_protocol", lambda *_args: None)
    monkeypatch.setattr(J.COMMON, "_verify_source_blobs", lambda *_args: None)
    monkeypatch.setattr(J.COMMON, "_git", lambda *_args: SimpleNamespace(returncode=0))
    return protocol_path


def test_contract_keeps_v1_wire_and_adds_separate_aaba_receipts() -> None:
    assert J.PROTOCOL_SCHEMA == "vision_pull_record_protocol_v4"
    assert J.PROTOCOL_ID == "native-policy-record-20261027am"
    assert J.JOB_SCHEMA == "vision_pull_record_job_v1"
    assert J.RESULT_SCHEMA == "vision_pull_record_result_v1"
    assert J.FAILURE_SCHEMA == "vision_pull_record_failure_v1"
    assert J.CHECKPOINT_IDS == ("source-s2220-none", "source-s2220-f1")
    assert J.LIMITS["max_batch_compute_usd"] == "0.407607"
    assert J.LIMITS["max_wall_s_per_policy"] == 300
    assert J.LIMITS["max_child_s_per_policy"] == 100
    assert J.LIMITS["postflight_reserve_s_per_policy"] == 200
    assert J.LIMITS["max_wall_compute_usd_per_policy"] == "0.145000"
    assert J.LIMITS["max_wall_compute_usd_batch"] == "0.290000"
    assert J.FIXED_FLAGS["aaba_output_separate_from_record"] is True
    assert J.FIXED_FLAGS["aaba_raw_receipts_required"] is True
    assert J.FIXED_FLAGS["first_error_preserved_before_close"] is True
    assert J.FIXED_FLAGS["aaba_verified_terminal_before_fast_shutdown"] is True
    assert J.FIXED_FLAGS["aaba_shutdown_receipt_required"] is True
    assert J.FIXED_FLAGS["aaba_app_close_phase_receipts"] is True
    assert J.FIXED_FLAGS["record_shutdown_policy_unchanged"] is True
    assert J.EXPECTED_PRIOR + J.EXPECTED_FRESH == J.EXPECTED_TOTAL


@pytest.mark.parametrize(
    "mutation",
    [None, "predecessor", "diagnostic_source", "stop_ref", "stop_cost", "al_identity", "meta_boundary", "closure", "budget"],
)
def test_protocol_validates_383_sources_27_stops_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    path = _fixture(tmp_path, monkeypatch)
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "predecessor":
        value["predecessor"]["diagnostic"]["sha256_lf"] = "0" * 64
    elif mutation == "diagnostic_source":
        diagnostic = json.loads((tmp_path / "diagnostic.json").read_text(encoding="utf-8"))
        diagnostic["source_sha256_lf"].pop(next(iter(diagnostic["source_sha256_lf"])))
        _write(tmp_path / "diagnostic.json", diagnostic)
    elif mutation == "stop_ref":
        value["predecessor"]["appended_stops"][2]["sha256"] = "0" * 64
    elif mutation == "stop_cost":
        stop = json.loads((tmp_path / "retry1-stop.json").read_text(encoding="utf-8"))
        stop["cost_estimate"]["estimated_compute_cost_usd"] = "0.242751"
        _write(tmp_path / "retry1-stop.json", stop)
    elif mutation == "al_identity":
        stop = json.loads((tmp_path / "al-stop.json").read_text(encoding="utf-8"))
        stop["execution_identity"]["run_id"] = "native-policy-record-20261027aj"
        _write(tmp_path / "al-stop.json", stop)
    elif mutation == "meta_boundary":
        stop = json.loads((tmp_path / "infra-stop.json").read_text(encoding="utf-8"))
        stop["model_api_requests"] = 1
        _write(tmp_path / "infra-stop.json", stop)
    elif mutation == "closure":
        value["source_sha256_lf"].pop("bench/vision_aaba_shutdown_v1.py")
    elif mutation == "budget":
        value["resources"]["fresh_lifecycle_budget_usd"] = "0.548497"
    path.write_text(json.dumps(value), encoding="utf-8")
    if mutation is None:
        checked = J.validate_protocol(path, repo=tmp_path)
        assert len(checked["source_sha256_lf"]) == 383
        assert len(checked["resources"]["prior_stop_receipts"]) == 27
        assert [row["logical_id"] for row in checked["checkpoints"]] == list(J.CHECKPOINT_IDS)
    else:
        with pytest.raises((TypeError, ValueError)):
            J.validate_protocol(path, repo=tmp_path)


def test_validate_job_preserves_fresh_and_historical_output_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    lab = tmp_path / "isaaclab"
    stage = tmp_path / "stage"
    for folder in (repo, lab, stage):
        folder.mkdir()
    checkpoint = lab / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    resource = _write(tmp_path / "resource.json", {"resource": True})
    protocol_path = _write(tmp_path / "protocol.json", {"protocol": True})
    protocol_sha = J._sha(protocol_path, lf=True)
    row = {
        "logical_id": J.CHECKPOINT_IDS[0],
        "isaaclab_relative": "checkpoint.pt",
        "sha256": J._sha(checkpoint),
    }
    protocol = {
        "source_commit": "a" * 40,
        "source_sha256_lf": {"bench/source.py": "b" * 64},
        "source_protocol": {"sha256_lf": "c" * 64},
        "_source": {"protocol_id": "source"},
    }
    job = {
        "schema": J.JOB_SCHEMA,
        "mode": J.MODE,
        "job_id": f"{J.PROTOCOL_ID}-{row['logical_id']}",
        "source": {
            "repo_root": str(repo.resolve()),
            "commit": protocol["source_commit"],
            "files_lf": protocol["source_sha256_lf"],
        },
        "protocol": {"path": str(protocol_path.resolve()), "sha256_lf": protocol_sha},
        "resource_proof": {"path": str(resource.resolve()), "sha256": J._sha(resource)},
        "isaaclab": str(lab.resolve()),
        "python": str(tmp_path / "python"),
        "output_directory": str((stage / "output").resolve()),
        "profile": J.RECORD.fixed_profile(),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": row["sha256"],
            "logical_id": row["logical_id"],
        },
    }
    job_path = _write(stage / "job.json", job)
    original = job_path.read_bytes()
    monkeypatch.setattr(J, "validate_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(J.RECORD_JOB, "_checkpoint", lambda *args: (row, checkpoint.resolve()))
    monkeypatch.setattr(J.COMMON, "_resource", lambda *args: {"resource_id": "id"})
    monkeypatch.setattr(J.COMMON, "_python_executable", lambda *args: None)
    monkeypatch.setattr(J.COMMON, "_verify_runtime_sources", lambda *args: None)

    assert J.validate_job(job_path)["output_directory"] == stage / "output"
    with pytest.raises((FileNotFoundError, ValueError)):
        J.validate_job(job_path, require_fresh=False)
    (stage / "output").mkdir()
    assert J.validate_job(job_path, require_fresh=False)["output_directory"] == stage / "output"
    with pytest.raises(FileExistsError):
        J.validate_job(job_path)
    assert job_path.read_bytes() == original
