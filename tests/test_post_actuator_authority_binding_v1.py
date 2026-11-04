from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import evals.registered_post_actuator_confirmation_v5_core as CORE

REPO = Path(__file__).resolve().parents[1]
RECEIPT = REPO / CORE.DELEGATED_APPROVAL_REL
SOURCE = Path(__file__).resolve()
OUTER_PATH = REPO / CORE.INNER_REL
SOURCE_SHA = "11cd0bc13be4aec43cb31f5f73fde2bdeb33cdfbf0c0788ea4d8dc0d03cad26b"


def _approval_ref() -> dict[str, str]:
    return {"path": CORE.DELEGATED_APPROVAL_REL, "sha256": CORE._sha(RECEIPT)}


def _outer() -> dict[str, object]:
    sources = {relative: CORE._sha_lf(REPO / relative) for relative in CORE.SOURCE_RELATIONS}
    return {
        "schema": CORE.OUTER_SCHEMA,
        "status": "registered",
        "profile_id": CORE.POST.PROFILE_ID,
        "condition": "NONE",
        "max_sdk_launches": 1,
        "child_entrypoint": CORE.CHILD_REL,
        "inner_confirmation_protocol": {
            "path": CORE.INNER_REL,
            "sha256": CORE._sha(REPO / CORE.INNER_REL),
        },
        "primary_plan": {
            "path": "bench/protocols/registered_recovery_comparison_s2664_s2667_20261105_v13.json",
            "sha256": "bd94f8b4cace7bb199a13965dec6a61a1859c4e20c6644d7ae34f3e617fd8a5c",
        },
        "primary_execution": {
            "registration_anchor": "a" * 40,
            "source_anchor": "b" * 40,
            "manifest_sha256": "1" * 64,
            "run_anchor_sha256": "2" * 64,
            "supervisor_sha256": "3" * 64,
            "supervisor_mode": "recovery_v13",
            "supervisor_config_sha256": "4" * 64,
        },
        "source_sha256_lf": sources,
        "result_seen_design": True,
        "automatic_retry": False,
        "post_budget": {
            "schema": "post_actuator_separate_budget_v1",
            "primary_sdk_cap": 32,
            "post_sdk_cap": 1,
            "study_sdk_cap": 33,
            "api_cap": 0,
            "basis": "result_seen_post_actuator_preregistration",
        },
        "interpretation": {
            "checkpoint_association_only": True,
            "producer_loads_checkpoint": False,
            "trained_policy_validated": False,
        },
        "delegated_approval": _approval_ref(),
    }


def _authority(outer: dict[str, object]) -> dict[str, object]:
    return {
        "schema": CORE.AUTHORITY_SCHEMA,
        "status": "approved",
        "profile_id": CORE.POST.PROFILE_ID,
        "case_id": "case-05",
        "condition": "NONE",
        "sdk_launches_approved": 1,
        "approval_id": "ri451-post-actuator-v13-20261105",
        "request_id": "post-actuator-s2664-v13-20261105",
        "effect_key": "post-actuator-s2664-mapping-v1",
        "approved_by": "codex_main_goal_owner",
        "source_inputs": {"path": str(SOURCE), "sha256": SOURCE_SHA},
        "outer_protocol": {"path": str(OUTER_PATH), "sha256": CORE._sha(OUTER_PATH)},
        "primary_result": {"path": str(REPO / "result.json"), "sha256":
                           "d1ad52627b3f295a395ce317fde961572be5a1b1f266db599f3c4d3fb8c9fe97"},
        "primary_manifest": {"path": str(REPO / "manifest.json"), "sha256": "1" * 64},
        "primary_plan": {"path": str(REPO / "plan.json"), "sha256":
                         "bd94f8b4cace7bb199a13965dec6a61a1859c4e20c6644d7ae34f3e617fd8a5c"},
        "primary_supervisor": {"path": str(REPO / "supervisor.json"), "sha256": "2" * 64},
        "source_sha256_lf": outer["source_sha256_lf"],
        "registration_anchor": "a" * 40,
        "result_seen_design": True,
        "automatic_retry": False,
        "approval_basis": "post_primary_completion_result_seen_development",
        "delegated_approval": _approval_ref(),
    }


def _patch_source(monkeypatch) -> None:
    original_read = CORE._read
    original_sha = CORE._sha
    source_body = {
        "checkpoint": {"sha256":
                       "3aaeabbc6a00f4fcb34fe13dff12fdadb557a031fc7ff581d6436b4452f5a3e7"},
        "fixed_report": {"sha256":
                         "8e553408c55053c64e97e83902f806d06586ee388ceb4a3a93be4a73355ee825"},
        "recovery_receipt": {"sha256":
                             "1774de2d8fed9c37cd736277ad40bc4b594b15787c927b972693996620f9353a"},
    }
    monkeypatch.setattr(CORE, "_read", lambda path: source_body if path == SOURCE
                        else original_read(path))
    monkeypatch.setattr(CORE, "_sha", lambda path: SOURCE_SHA if path == SOURCE
                        else original_sha(path))
    monkeypatch.setattr(CORE, "_git_anchor", lambda *_args, **_kwargs: None)


def test_registered_outer_and_authority_accept_exact_delegation(monkeypatch) -> None:
    outer = CORE.validate_outer(_outer())
    _patch_source(monkeypatch)
    authority = CORE._authority(_authority(outer), SOURCE, OUTER_PATH, outer)
    assert authority["approved_by"] == "codex_main_goal_owner"
    assert authority["delegated_approval"] == outer["delegated_approval"]


def test_registration_closure_pins_receipt_and_authority_test() -> None:
    assert CORE.DELEGATED_APPROVAL_REL in CORE.SOURCE_RELATIONS
    assert CORE.AUTHORITY_BINDING_TEST_REL in CORE.SOURCE_RELATIONS


@pytest.mark.parametrize("field", ["actor", "approval_id", "request_id", "effect_key",
                                    "source_inputs_sha256", "post_sdk_cap", "study_sdk_cap"])
def test_delegated_receipt_mutations_are_rejected(monkeypatch, field: str) -> None:
    original_read = CORE._read
    mutated = deepcopy(original_read(RECEIPT))
    mutated[field] = "changed" if isinstance(mutated[field], str) else 999
    monkeypatch.setattr(CORE, "_read", lambda path: mutated if path == RECEIPT
                        else original_read(path))
    with pytest.raises(ValueError, match="delegated approval receipt differs"):
        CORE._delegated_approval(_approval_ref())


def test_unregistered_or_other_receipt_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="delegated approval reference differs"):
        CORE._delegated_approval(None)
    other = {"path": CORE.INNER_REL, "sha256": CORE._sha(REPO / CORE.INNER_REL)}
    with pytest.raises(ValueError, match="delegated approval reference differs"):
        CORE._delegated_approval(other)


@pytest.mark.parametrize("field", ["approved_by", "approval_id", "request_id", "effect_key"])
def test_self_consistent_authority_identity_cannot_replace_delegated_identity(
        monkeypatch, field: str) -> None:
    outer = _outer()
    authority = _authority(outer)
    authority[field] = "self-consistent-but-unregistered"
    _patch_source(monkeypatch)
    with pytest.raises(ValueError, match="authority delegated identity differs"):
        CORE._authority(authority, SOURCE, OUTER_PATH, outer)


def test_authority_target_must_match_delegated_primary_result(monkeypatch) -> None:
    outer = _outer()
    authority = _authority(outer)
    authority["primary_result"] = {"path": str(REPO / "result.json"), "sha256": "9" * 64}
    _patch_source(monkeypatch)
    with pytest.raises(ValueError, match="authority delegated target differs"):
        CORE._authority(authority, SOURCE, OUTER_PATH, outer)
