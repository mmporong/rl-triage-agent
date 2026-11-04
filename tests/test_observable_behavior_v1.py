from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from rl_triage import intervention_evidence as EVIDENCE
from rl_triage import intervention_governor as GOVERNOR
from rl_triage import observable_behavior_v1 as SUBJECT


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _actual(case_id: str = "case-01") -> GOVERNOR.VerifiedObservation:
    return EVIDENCE.verify_public_fixed26_pair(EVIDENCE.SOURCE_COLLECTION, case_id=case_id)


def _payload(observation: GOVERNOR.VerifiedObservation) -> dict:
    return GOVERNOR.build_model_packet(observation)["observation"]


def test_actual_normal_and_failed_pairs_require_absolute_walk_qualification() -> None:
    normal = _payload(SUBJECT.qualify_source_observation(_actual("case-01")))
    failed = _payload(SUBJECT.qualify_source_observation(_actual("case-03")))

    assert normal["behavior_outcome"] == "normal"
    assert normal["absolute_qualification"]["candidate"]["qualified"] is True
    assert normal["absolute_qualification"]["reference"]["qualified"] is True
    assert failed["behavior_outcome"] == "failed"
    assert failed["qualification_reason_code"] == "candidate_unqualified"
    assert failed["absolute_qualification"]["candidate"]["qualified"] is False
    assert normal["criteria"]["qualification_rule_id"] == SUBJECT.RULE_ID


def test_same_relative_ratio_cannot_make_two_stationary_policies_normal() -> None:
    stationary = _read(EVIDENCE.SOURCE_COLLECTION / "sources/source-06/fixed26.json")
    bands = _payload(_actual("case-03"))["criteria"]["bands"]

    result = SUBJECT.classify_actual_fixed26(stationary, stationary, bands)

    assert result["behavior_outcome"] == "unknown"
    assert result["reason_code"] == "reference_unqualified"
    assert result["absolute_qualification"]["candidate"]["qualified"] is False
    assert result["metric_checks"] == []


def test_missing_standstill_is_unknown_not_normal() -> None:
    raw = _read(EVIDENCE.SOURCE_COLLECTION / "sources/source-01/fixed26.json")
    reference = _read(EVIDENCE.SOURCE_COLLECTION / "sources/source-02/fixed26.json")
    del raw["report"]["standstill"]
    bands = _payload(_actual())["criteria"]["bands"]

    result = SUBJECT.classify_actual_fixed26(raw, reference, bands)

    assert result["behavior_outcome"] == "unknown"
    assert result["reason_code"] == "fixed26_missing_or_unverifiable"


def test_source_payload_mutation_and_candidate_reference_swap_are_rejected() -> None:
    original = _actual()
    original._payload["candidate"]["metrics"]["fall_rate"] = 0.9
    with pytest.raises(ValueError, match="mutated"):
        SUBJECT.qualify_source_observation(original)

    fresh = _payload(_actual())
    fresh["candidate"]["paths"] = deepcopy(fresh["reference"]["paths"])
    fresh["candidate"]["refs"] = deepcopy(fresh["reference"]["refs"])
    swapped = GOVERNOR._verified_observation(fresh)
    with pytest.raises(ValueError, match="differs from the sealed source payload"):
        SUBJECT.qualify_source_observation(swapped)


def test_qualified_model_packet_contains_no_hidden_classification() -> None:
    qualified = SUBJECT.qualify_source_observation(_actual("case-03"))
    encoded = json.dumps(GOVERNOR.build_model_packet(qualified), sort_keys=True)

    assert '"classification"' not in encoded
    assert '"source_type"' not in encoded
    assert '"fault_label"' not in encoded
    assert "F1" not in encoded
    assert "criteria_review_only_not_causal_or_recovery_confirmation" in encoded


def test_post_observation_binds_new_checkpoint_and_drops_old_confirmation(tmp_path: Path) -> None:
    source = SUBJECT.qualify_source_observation(_actual())
    source_payload = _payload(source)
    raw = _read(Path(source_payload["source_refs"]["candidate_fixed26"]["path"]))
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"new registered checkpoint")
    checkpoint_ref = {"path": str(checkpoint.resolve()),
                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    raw["source_id"] = raw["report"]["name"] = "post-fixed26"
    raw["report"]["checkpoint"] = {"file": checkpoint.name, "sha256": checkpoint_ref["sha256"]}
    post_ref = _write(tmp_path / "post-fixed26.json", raw)

    post = SUBJECT.classify_post_observation(
        source, post_ref, source_payload["source_refs"]["reference_fixed26"], checkpoint_ref)
    payload = _payload(post)

    assert payload["candidate"]["checkpoint_sha256"] == checkpoint_ref["sha256"]
    assert payload["behavior_outcome"] == "normal"
    assert payload["confirmation_checks"] == []
    assert payload["cause_confirmation_receipt"] is None
    assert payload["correction_receipt"] is None


def test_post_observation_rejects_reference_swap_and_reused_checkpoint(tmp_path: Path) -> None:
    source = SUBJECT.qualify_source_observation(_actual())
    payload = _payload(source)
    candidate_ref = payload["source_refs"]["candidate_fixed26"]
    source_checkpoint = Path(payload["candidate"]["paths"]["fixed26"])

    with pytest.raises(ValueError, match="reference was swapped"):
        SUBJECT.classify_post_observation(source, candidate_ref, candidate_ref,
                                           {"path": str(source_checkpoint),
                                            "sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()})

    with pytest.raises(ValueError, match="post report does not bind the new checkpoint"):
        SUBJECT.classify_post_observation(
            source, candidate_ref, payload["source_refs"]["reference_fixed26"],
            {"path": str(source_checkpoint),
             "sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()})
