"""C4 PPO ratio 공개 exporter의 격리·원본 불변·변조 거부 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import export_ppo_ratio_repair_v4 as EXPORT


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def invalid_fixture(tmp_path: Path) -> Path:
    run = tmp_path / "invalid"
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    result = {
        "schema": "ppo_ratio_repair_demo_result_v4",
        "tag": EXPORT.TAG,
        "seed": 2165,
        "restart_seed": 3165,
        "experiment_commit": EXPORT.EXPECTED_COMMIT,
        "protocol_sha256_lf": EXPORT.sha256(EXPORT.PROTOCOL, True),
        "conditions": {
            "committed_source_and_protocol": True,
            "fresh_seeds": True,
            "launch_paths_checked": True,
            "costs_recorded": False,
        },
        "outcome": "invalid",
        "model": {},
        "probe_steps": [],
        "costs": {},
        "api_requests": 0,
        "api_requests_in_saved_records": 0,
        "pending_model_records": [],
        "transport_errors": [],
        "demo_success": False,
        "wording": protocol["outcome_wording"]["invalid"],
        "design_coupling": protocol["design_coupling"],
    }
    write(run / "begin.json", {
        "tag": EXPORT.TAG,
        "experiment_commit": EXPORT.EXPECTED_COMMIT,
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "approver": "fixture-operator",
        "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
    })
    write(run / "seed_freshness.json", {
        "schema": "preregistered_seed_freshness_v1",
        "status": "fresh",
        "seeds": [2165, 3165],
        "evidence": [],
    })
    write(run / "launch_paths.json", {"processes_started": 0, "gpu_queries": 0})
    write(run / "result.json", result)
    return run


def clear_gpu_receipt() -> dict:
    sample = {
        "busy": False,
        "busy_reasons": [],
        "admission_policy": EXPORT.V3.EXECUTION_V3.GPU_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "utilization_percent": 0,
        "memory_used_mib": 100,
        "compute_process_count": 0,
        "processes": [],
        "active_engines": [],
    }
    bounded = {
        "attempted": False,
        "accepted": True,
        "reason": "initial_clear",
        "interval_s": EXPORT.V3.EXECUTION_V3.RESAMPLE_INTERVAL_S,
        "max_samples": EXPORT.V3.EXECUTION_V3.MAX_SAMPLES,
        "sample_count": 1,
        "required_consecutive_clear": 2,
        "identities": [],
    }
    return {**sample, "bounded_resample": bounded, "samples": [sample]}


def test_invalid_fixture_exports_v4_without_mutating_sources_or_v2_v3_globals(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    result_before = (run / "result.json").read_bytes()
    protocol_before = EXPORT.PROTOCOL.read_bytes()
    globals_before = (
        EXPORT.V2.TAG, EXPORT.V2.PROTOCOL, deepcopy(EXPORT.V2.EXPECTED_ARMS),
        EXPORT.V3.TAG, EXPORT.V3.PROTOCOL, deepcopy(EXPORT.V3.EXPECTED_ARMS),
    )

    payloads = EXPORT.verify_and_build(run)

    assert set(payloads) == EXPORT.PUBLIC_FILES
    assert payloads["summary.json"]["outcome"] == "invalid"
    assert payloads["summary.json"]["gpu_admission_evidence"]["validated_receipts"] == 0
    assert all(value["schema"].endswith("_v4") for value in payloads.values())
    assert payloads["provenance.json"]["source_result"] == {
        "schema": "ppo_ratio_repair_demo_result_v4",
        "sha256": EXPORT.sha256(run / "result.json"),
    }
    assert (run / "result.json").read_bytes() == result_before
    assert EXPORT.PROTOCOL.read_bytes() == protocol_before
    assert globals_before == (
        EXPORT.V2.TAG, EXPORT.V2.PROTOCOL, EXPORT.V2.EXPECTED_ARMS,
        EXPORT.V3.TAG, EXPORT.V3.PROTOCOL, EXPORT.V3.EXPECTED_ARMS,
    )
    public = EXPORT.serialized(payloads["summary.json"]) + EXPORT.serialized(payloads["provenance.json"])
    assert b"bench/private" not in public and str(ROOT).encode() not in public


@pytest.mark.parametrize("field,value", [
    ("schema", "ppo_ratio_repair_demo_result_v3"),
    ("tag", "other"),
    ("seed", 2164),
    ("restart_seed", 3164),
    ("experiment_commit", "0" * 40),
])
def test_exact_v4_result_identity_tampering_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    run = invalid_fixture(tmp_path)
    result = EXPORT.read(run / "result.json")
    result[field] = value
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="C4 result/protocol/commit identity"):
        EXPORT.verify_and_build(run)


def test_protocol_request_cap_tampering_is_rejected_before_projection(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    protocol["model"]["total_http_request_cap"] = 5
    changed = tmp_path / "protocol.json"
    write(changed, protocol)
    result = EXPORT.read(run / "result.json")
    result["protocol_sha256_lf"] = EXPORT.sha256(changed, True)
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="C4 result/protocol/commit identity"):
        EXPORT.verify_and_build(run, changed)


def test_gpu_raw_status_tampering_is_rejected() -> None:
    receipt = clear_gpu_receipt()
    assert EXPORT.V3._validate_gpu_receipt(receipt) == "initial_clear"
    receipt["bounded_resample"]["accepted"] = False
    with pytest.raises(ValueError, match="GPU"):
        EXPORT.V3._validate_gpu_receipt(receipt)


def test_exact_export_bytes_audit_and_overwrite_are_rejected(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    output, audit = tmp_path / "public", tmp_path / "audit"
    EXPORT.write_new(payloads, output, audit)
    assert set(EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)) == EXPORT.PUBLIC_FILES
    with pytest.raises(FileExistsError):
        EXPORT.write_new(payloads, output, audit)
    with (output / "summary.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact differs"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)
