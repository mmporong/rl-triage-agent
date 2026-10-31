from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_parent_binding_v1 as M


def packet() -> dict:
    return M.C.V.S.build_packet(
        epoch=2, applied_corrections=["restore_body_frame_reward"],
        prior_receipt={"verdict": "not_recovered",
                       "correction_id": "restore_body_frame_reward",
                       "result_sha256": "1" * 64, "checkpoint_sha256": "2" * 64},
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={
            "case": {"checkpoint_sha256": "2" * 64, "telemetry_sha256": "3" * 64},
            "reference": {"checkpoint_sha256": "4" * 64,
                          "telemetry_sha256": "5" * 64}},
        budget_left=3, execution_budget_s=1200)


def setup_inputs(monkeypatch, tmp_path):
    post, reference = tmp_path / "post.json", tmp_path / "reference.json"
    post.write_text("{}\n", encoding="utf-8")
    reference.write_text("{}\n", encoding="utf-8")
    value = packet()
    monkeypatch.setattr(M.POST, "checked_observation", lambda _path: ({}, {}))
    monkeypatch.setattr(M.POST, "make_selection_packet", lambda *_args: {
        "schema": M.POST.INPUT.PACKET_SCHEMA, "packet": value})
    return post, reference, value


def test_scope_injects_exact_parent_and_supports_checked_messages_choice_spec(
        tmp_path, monkeypatch):
    post, reference, value = setup_inputs(monkeypatch, tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    with M.parent_scope(post, reference, "post_registry", "registry_next") as context:
        snapshot = M.C.V.build_snapshot(
            value, "post_registry", "registry_next", M.C.V.file_digest(M.C.V.PROTOCOL), ledger)
        assert snapshot["inputs"]["parent"] == context["parent"]
        assert M.C.V.checked(snapshot) == snapshot
        assert M.C.V.messages(snapshot)[1]["role"] == "user"
        choice = M.C.choose(snapshot, "registry_next")
        if choice["action"] == "continue":
            assert M.C.request_spec(snapshot, choice)["probe"] == choice["probe"]


@pytest.mark.parametrize("mutation", ["packet", "run", "policy", "parent"])
def test_scope_rejects_wrong_packet_run_policy_or_parent(tmp_path, monkeypatch, mutation):
    post, reference, value = setup_inputs(monkeypatch, tmp_path)
    actual = value
    run_id, policy, parent = "post_registry", "registry_next", None
    if mutation == "packet":
        actual = {**value, "budget_left": 2}
    elif mutation == "run":
        run_id = "other"
    elif mutation == "policy":
        policy = "fixed_eligible"
    else:
        parent = {"wrong": True}
    with M.parent_scope(post, reference, "post_registry", "registry_next"), \
            pytest.raises(ValueError):
        M.C.V.build_snapshot(actual, run_id, policy, M.C.V.file_digest(M.C.V.PROTOCOL),
                             tmp_path / "ledger.jsonl", parent=parent)


def test_scope_restores_frozen_builder(tmp_path, monkeypatch):
    post, reference, _value = setup_inputs(monkeypatch, tmp_path)
    original = M.C.V.build_snapshot
    with M.parent_scope(post, reference, "post_registry", "registry_next"):
        assert M.C.V.build_snapshot is not original
    assert M.C.V.build_snapshot is original


def test_candidate_snapshot_helper_uses_same_parent_scope(tmp_path, monkeypatch):
    import lineage_post_recovery_candidate_v1 as candidate

    post, reference, value = setup_inputs(monkeypatch, tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    with M.parent_scope(post, reference, "post_registry", "registry_next"):
        snapshot = M.C.V.build_snapshot(
            value, "post_registry", "registry_next", M.C.V.file_digest(M.C.V.PROTOCOL), ledger)
        rebuilt = candidate.CANDIDATE1._snapshot_at(snapshot, [], [])
    assert rebuilt["inputs"]["parent"] == snapshot["inputs"]["parent"]
