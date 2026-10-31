# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_model_hold_handoff_trial_v1 as T


def _proposal(*, approval: bool = False) -> dict:
    return {
        "schema": T.FOLLOW.SCHEMA, "status": "complete", "case_id": "c02",
        "mode": "enabled_low", "selection_actor": T.SELECTOR_ACTOR,
        "approval_granted": approval, "execution_allowed": False,
        "correction_allowed": False, "automatic_resume": False,
        "parent_closed_HOLD": {"verified": True, "decision_outcome_preserved": "HOLD",
                                "model_valid_hold_preserved": True},
        "budgets": {"model_actions": {"used": 3, "limit": 4, "remaining": 1},
                    "paired_probes": {"used": 1, "limit": 3, "remaining": 2},
                    "shadow_probe_budget_left": 2},
        "proposal": {"action": "run_probe", "target": "P_reward",
                     "fact_ids": list(T.FOLLOW.REWARD_FACT_IDS)},
        "proposal_reason": "counts_normal_and_source_tracking_failure_gate_verified",
        "shadow_state_before": {"path": "unused", "sha256": "x" * 64},
        "source_binding": {"case_checkpoint_sha256": "c" * 64,
                           "reference_checkpoint_sha256": "r" * 64},
    }


def _audit(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "diagnostic_coverage_audit_public_summary_v1", "status": "complete",
        "case_id": "c02", "reference_case_id": "c01", "probe": "P_reward",
        "outcome": "abnormal", "selection_actor": T.AUDIT.APPROVER,
        "model_selected_probe": False, "original_model_result_changed": False,
    }), encoding="utf-8")


def test_prepare_protocol_fixes_actor_budget_owner_and_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"schema": T.R.SCHEMA, "source_sha256_lf": {},
                                  "inherited_source_sha256_lf": {}}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    _audit(audit)
    monkeypatch.setattr(T, "AUDIT_SUMMARY_SHA256",
                        hashlib.sha256(audit.read_bytes()).hexdigest())
    monkeypatch.setattr(T, "ROOT", tmp_path)
    monkeypatch.setattr(T.BASE, "ROOT", tmp_path)
    for relative in T.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    protocol = T.prepare_protocol(parent, audit)
    design = protocol["design"]
    assert design["selector_actor"] == T.SELECTOR_ACTOR
    assert design["approval_actor"] == T.APPROVER
    assert design["resource_lease_owner"] == T.RUN.OWNER
    assert (design["total_actions"], design["total_paired_probes"]) == (4, 2)
    assert design["proof_cache"] == {"kind": "immutable_git_show", "scope": "one_trial",
                                      "mutable_or_authority_cached": False}
    assert protocol["claims"]["new_observation"] is False


def test_proposal_requires_non_authoritative_verified_hold(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T.FOLLOW, "build_proposal", lambda *_args, **_kwargs: _proposal())
    assert T._proposal(Path("epoch"), Path("study"))["proposal"]["target"] == "P_reward"
    monkeypatch.setattr(T.FOLLOW, "build_proposal",
                        lambda *_args, **_kwargs: _proposal(approval=True))
    with pytest.raises(ValueError, match="비권한"):
        T._proposal(Path("epoch"), Path("study"))


def test_proposal_budget_overflow_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _proposal()
    value["budgets"]["model_actions"] = {"used": 4, "limit": 4, "remaining": 0}
    monkeypatch.setattr(T.FOLLOW, "build_proposal", lambda *_args, **_kwargs: value)
    with pytest.raises(ValueError, match="budget"):
        T._proposal(Path("epoch"), Path("study"))


def test_inspect_inputs_binds_proposal_to_same_checkpoints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = tmp_path / "audit.json"
    _audit(audit)
    monkeypatch.setattr(T, "AUDIT_SUMMARY_SHA256",
                        hashlib.sha256(audit.read_bytes()).hexdigest())
    shadow = tmp_path / "shadow.json"
    shadow.write_text("{}", encoding="utf-8")
    value = _proposal()
    value["shadow_state_before"] = {
        "path": str(shadow), "sha256": hashlib.sha256(shadow.read_bytes()).hexdigest()}
    monkeypatch.setattr(T, "_proposal", lambda *_: value)
    monkeypatch.setattr(T.AUDIT, "inspect_inputs", lambda *_: {
        "case": {"checkpoint": {"sha256": "c" * 64}},
        "reference": {"checkpoint": {"sha256": "r" * 64}}})
    state = {"status": "open"}
    monkeypatch.setattr(T.SEL, "_state", lambda _value: state)
    monkeypatch.setattr(T.SEL, "request_spec", lambda current, action: {
        "action": action, "approval_granted": False, "model_generated_args": False})
    result = T.inspect_inputs(Path("p"), Path("s"), Path("m"), Path("c"), Path("e"),
                              Path("study"), audit, Path("lab"))
    assert result["shadow"] is state
    assert result["request_spec"]["approval_granted"] is False


def test_execute_uses_new_operator_approval_and_preserves_parent_hold(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in (
        "protocol", "source", "mode", "corpus", "epoch", "study", "audit",
        "probe_protocol", "view_protocol", "case_input", "reference_input",
        "case_runtime", "reference_runtime")}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    source = {
        "spec": {},
        "case": {"input": paths["case_input"], "metadata": {"runtime_facts": {
            "case": {"path": str(paths["case_runtime"])}}}},
        "reference": {"input": paths["reference_input"], "metadata": {"runtime_facts": {
            "reference": {"path": str(paths["reference_runtime"])}}}},
    }
    inputs = {"source": source, "proposal": _proposal(),
              "shadow": {"state_digest": "shadow"},
              "action": _proposal()["proposal"]}
    monkeypatch.setattr(T, "ROOT", tmp_path)
    monkeypatch.setattr(T, "verify_protocol", lambda *_: {
        "parent_reasoning_protocol": {"path": "parent.json"},
        "prior_coverage_audit": {"path": "audit.json"}})
    (tmp_path / "parent.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(T, "inspect_inputs", lambda *_: inputs)
    monkeypatch.setattr(T.R, "_base_protocol", lambda *_: {"protocols": {
        "paired_probe": {"path": "probe_protocol.json"},
        "view": {"path": "view_protocol.json"}}})
    monkeypatch.setattr(T.RUN, "_lease", lambda: {"leaseId": "lease"})
    monkeypatch.setattr(T.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {"P_reward": "abnormal"}, "state_digest": "after"})
    actors = []

    def host(_state, _action, **kwargs):
        actors.append(kwargs["approver"])
        assert kwargs["verify_context"]({"role": "reference"})["approved"] is True
        out = kwargs["out"]
        out.mkdir(parents=True)
        T.BASE._write(out / "host-result.json", {"schema": "fixture"})
        result = {}
        for name in ("case_plan", "case_receipt", "reference_plan", "reference_receipt"):
            path = out / f"{name}.json"
            path.write_text("{}", encoding="utf-8")
            result[name] = str(path)
        return result

    result = T._execute(paths["protocol"], "a" * 40, paths["source"], paths["mode"],
                        paths["corpus"], paths["epoch"], paths["study"], tmp_path / "lab",
                        tmp_path / "out", execute_host=host)
    assert actors == [T.APPROVER]
    assert result["parent_model_outcome"] == "HOLD"
    assert result["original_model_result_changed"] is False
    assert result["new_observation_claim"] is False


def test_cache_wrapper_preserves_result(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    @contextmanager
    def scope():
        events.append("enter")
        yield {"approval": False, "execution_allowed": False}
        events.append("exit")

    expected = {"schema": "fixture", "parent_model_outcome": "HOLD"}
    monkeypatch.setattr(T.CACHE, "proof_scope", scope)
    monkeypatch.setattr(T, "_execute", lambda *_args, **_kwargs: expected)
    actual = T.execute(Path("p"), "a" * 40, Path("s"), Path("m"), Path("c"),
                       Path("e"), Path("study"), Path("lab"), Path("out"))
    assert actual is expected and events == ["enter", "exit"]
