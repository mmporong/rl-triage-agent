from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from rl_triage import intervention_evidence as E
from rl_triage import intervention_governor as G


@pytest.fixture(scope="module")
def actual_normal():
    return E.verify_public_fixed26_pair(E.SOURCE_COLLECTION, case_id="case-01")


@pytest.fixture(scope="module")
def actual_failure():
    return E.verify_public_fixed26_pair(E.SOURCE_COLLECTION, case_id="case-03")


@pytest.fixture(scope="module")
def actual_recovery_miss():
    return E.verify_public_recovery_miss(
        E.SOURCE_COLLECTION, E.RECOVERY_COLLECTION,
        case_id="case-03", policy="fixed_eligible")


def _gate(tmp_path):
    path = tmp_path / "gate.json"
    raw = json.dumps({"schema": "host_gate_receipt_v1", "gate_id": "test-host",
                      "allowed_actions": ["no_intervention", "investigate", "correct", "reference_review"],
                      "refs": ["test:host-policy"]}, sort_keys=True).encode()
    path.write_bytes(raw)
    return G.verify_host_gate_receipt({"schema": G.HOST_SOURCE_SCHEMA,
                                       "receipt": {"path": str(path),
                                                   "sha256": hashlib.sha256(raw).hexdigest()}})


def test_actual_normal_fixed26_pair_is_reaggregated_and_host_keeps_no_intervention(actual_normal, tmp_path):
    packet = G.build_model_packet(actual_normal)["observation"]
    assert packet["verification_scope"] == "recorded_actual_source_cpu_revalidation"
    assert packet["behavior_outcome"] == "normal"
    assert len(packet["metric_checks"]) == 4
    assert all(row["passed"] for row in packet["metric_checks"])
    decision = G.decide_intervention(actual_normal, None, _gate(tmp_path))
    assert decision["status"] == "host_no_intervention"
    assert decision["approval_granted"] is decision["authority_created"] is False


def test_actual_failure_uses_four_numeric_metrics_without_fault_classification(actual_failure, tmp_path):
    packet = G.build_model_packet(actual_failure)["observation"]
    assert packet["behavior_outcome"] == "failed"
    assert len(packet["candidate"]["metrics"]) == 4
    assert any(not row["passed"] for row in packet["metric_checks"])
    serialized = json.dumps(packet, sort_keys=True)
    for hidden in ("source_type", "classification", "known_fault_types", "fault_labels"):
        assert hidden not in serialized
    decision = G.decide_intervention(actual_failure, None, _gate(tmp_path))
    assert decision["status"] == "investigation_required"
    assert decision["effective_action"] == "investigate"


def test_actual_correction_miss_is_consumed_but_absent_post_metrics_stay_hold(actual_recovery_miss, tmp_path):
    packet = G.build_model_packet(actual_recovery_miss)["observation"]
    assert packet["verifier_id"] == E.RECOVERY_VERIFIER_ID
    assert packet["behavior_outcome"] == "unknown"
    assert packet["candidate"]["metrics"] == {}
    assert packet["missing_reason"] == "public_recovery_export_omits_post_correction_metrics"
    assert packet["correction_receipt"]["status"] == "applied"
    decision = G.decide_intervention(actual_recovery_miss, None, _gate(tmp_path))
    assert decision["status"] == "hold"
    assert decision["reason_code"] == "confirmation_check_unknown"
    assert decision["requested_action"] == "investigate"
    assert decision["effective_action"] == "hold"
    assert decision["approval_granted"] is False


def test_production_adapter_rejects_unregistered_source_and_criteria_paths(tmp_path):
    with pytest.raises(ValueError, match="registered actual public export"):
        E.verify_public_fixed26_pair(tmp_path, case_id="case-01")
    fake_criteria = tmp_path / "summary.json"
    fake_criteria.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="registered recorded source"):
        E.verify_public_fixed26_pair(
            E.SOURCE_COLLECTION, case_id="case-01", criteria_summary=fake_criteria)


def test_registered_artifact_sha_constants_match_recorded_files():
    assert E._sha(E.SOURCE_COLLECTION / "manifest.json") == E.SOURCE_MANIFEST_SHA256
    assert E._sha(E.RECOVERY_COLLECTION / "manifest.json") == E.RECOVERY_MANIFEST_SHA256
    assert E._sha(E.CRITERIA_SUMMARY) == E.CRITERIA_SHA256
    assert E._sha(E.FIXED_PROTOCOL) == E.FIXED_PROTOCOL_SHA256


def test_explicit_registration_selects_another_verified_export_path(tmp_path):
    copied = tmp_path / "registered-source-export"
    shutil.copytree(E.SOURCE_COLLECTION, copied)
    with pytest.raises(ValueError, match="registered actual public export"):
        E.verify_public_fixed26_pair(copied, case_id="case-01")
    observation = E.verify_public_fixed26_pair(
        copied, case_id="case-01", registration=E.DEFAULT_REGISTRATION)
    assert observation._payload["behavior_outcome"] == "normal"
    assert observation._payload["source_refs"]["manifest"] == {
        "path": str(copied / "manifest.json"),
        "sha256": E.SOURCE_MANIFEST_SHA256,
    }
