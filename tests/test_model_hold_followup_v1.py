from __future__ import annotations

# pyright: reportMissingImports=false
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import model_hold_followup_v1 as F


def _write(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture(tmp_path: Path, *, evidence_read: bool, observed: dict[str, str],
             model_actions: int = 2, paired_probes: int = 1,
             gate_run: bool = True) -> tuple[Path, Path, dict[str, Any]]:
    case_sha, ref_sha = "c" * 64, "r" * 64
    facts = {item: 1.0 for item in F.REWARD_FACT_IDS}
    state = {
        "schema": "test-state", "status": "open", "budget_left": 2,
        "evidence_read": evidence_read, "observed": observed, "visible_facts": facts,
        "state_digest": "shadow-digest",
        "fact_ledger": {"source_binding": {
            "case": {"checkpoint_sha256": case_sha},
            "reference": {"checkpoint_sha256": ref_sha},
        }},
    }
    epoch_dir = tmp_path / "epoch"
    _write(epoch_dir / "step-02" / "state-before.json", state)
    closed = {**state, "status": "hold", "state_digest": "closed-digest"}
    state_after = _write(epoch_dir / "step-02" / "state-after.json", closed)
    provider = {"provider_model": F.LOOP.S1.CALL.MODEL, "finish_reason": "stop",
                "content": "hold-content"}
    review = {"status": "valid", "content": "hold-content"}
    dialogue = {
        "schema": "registry_action_options_dialogue_v2",
        "model_valid": True, "provider_incomplete_hold": False,
        "action": {"action": "hold", "target": None, "fact_ids": []},
        "final_review": review, "calls": [{"result": provider}],
    }
    dialogue_ref = _write(epoch_dir / "step-02" / "dialogue.json", dialogue)
    epoch = {
        "schema": "reasoning_diagnostic_loop_result_v1", "status": "complete",
        "case_id": "c-test", "mode": "legacy_none", "decision_outcome": "HOLD",
        "final_status": "hold", "model_actions": model_actions, "paired_probes": paired_probes,
        "steps": [{"step": 2, "action": {"action": "hold", "target": None, "fact_ids": []},
                   "model_valid": True, "format_valid": True, "dialogue": dialogue_ref,
                   "state_before_digest": state["state_digest"], "state_after": state_after,
                   "state_after_digest": closed["state_digest"]}],
    }
    epoch_path = epoch_dir / "result.json"
    _write(epoch_path, epoch)
    study = {
        "schema": "reasoning_diagnosis_study_result_v1", "status": "complete",
        "rows": [{"case_id": "c-test", "mode": "legacy_none", "status": "complete",
                  "decision_outcome": "HOLD", "model_valid_hold": True,
                  "checkpoint_sha256": case_sha, "reference_checkpoint_sha256": ref_sha,
                  "gate": {"run": gate_run,
                           "source_outcome": "not_recovered" if gate_run else "normal_success"}}],
    }
    study_path = tmp_path / "study.json"
    _write(study_path, study)
    return epoch_path, study_path, state


def _context(state: dict[str, Any]) -> dict[str, Any]:
    options = [{"action": "hold", "target": None}]
    if not state["evidence_read"]:
        options.append({"action": "read_evidence", "target": F.SEL.RESOURCE})
    if state["evidence_read"] and F.SEL.NEW_PROBE not in state["observed"]:
        options.append({"action": "run_probe", "target": F.SEL.NEW_PROBE})
    if state["observed"].get(F.SEL.NEW_PROBE) == "normal":
        options.append({"action": "run_probe", "target": "P_reward"})
    return {"action_options": options, "evidence_facts": state["visible_facts"],
            "count_evidence_groups": [["runtime:case:expected_optimizer_steps",
                                       "runtime:case:optimizer_steps_min",
                                       "runtime:case:optimizer_steps_max"]]}


def _build(
    epoch: Path, study: Path, *, epoch_sha: str | None = None, study_sha: str | None = None,
) -> dict[str, Any]:
    def hold_advancer(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        assert action == {"action": "hold", "target": None, "fact_ids": []}
        return {**state, "status": "hold", "state_digest": "closed-digest"}

    return F.build_proposal(
        epoch, study,
        expected_epoch_sha256=epoch_sha or hashlib.sha256(epoch.read_bytes()).hexdigest(),
        expected_study_sha256=study_sha or hashlib.sha256(study.read_bytes()).hexdigest(),
        state_checker=lambda value: value, context_builder=_context,
        hold_advancer=hold_advancer,
        reviewer=lambda content, _state: {"status": "valid", "content": content},
        model_selector=lambda _review, _state: {
            "action": "hold", "target": None, "fact_ids": []},
    )


def test_unread_counts_propose_read_without_mutating_parent(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    before = epoch.read_bytes()
    result = _build(epoch, study)
    assert result["proposal"] == {"action": "read_evidence", "target": F.SEL.RESOURCE,
                                  "fact_ids": []}
    assert epoch.read_bytes() == before
    assert result["approval_granted"] is result["execution_allowed"] is False
    assert result["automatic_resume"] is result["correction_allowed"] is False


def test_read_but_unobserved_counts_propose_recorded_count(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=True, observed={})
    result = _build(epoch, study)
    assert result["proposal"]["target"] == F.SEL.NEW_PROBE
    assert result["proposal"]["fact_ids"] == ["runtime:case:expected_optimizer_steps",
                                                "runtime:case:optimizer_steps_min",
                                                "runtime:case:optimizer_steps_max"]


def test_normal_counts_and_failed_source_gate_propose_reward(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(
        tmp_path, evidence_read=True, observed={F.SEL.NEW_PROBE: "normal"})
    result = _build(epoch, study)
    assert result["proposal"] == {"action": "run_probe", "target": "P_reward",
                                  "fact_ids": list(F.REWARD_FACT_IDS)}
    assert result["parent_closed_HOLD"]["verified"] is True
    assert result["claims"] == {"recovery": False, "correction": False,
                                "execution_effect": False}


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [({"gate_run": False}, "normal_or_unverified_source_gate"),
     ({"model_actions": F.LOOP.MAX_MODEL_ACTIONS}, "no_remaining_model_action_budget"),
     ({"paired_probes": F.LOOP.MAX_PAIRED_PROBES},
      "reward_probe_ineligible_or_visible_facts_incomplete")],
)
def test_gate_or_budget_exhaustion_stays_hold(
    tmp_path: Path, changes: dict[str, Any], expected_reason: str,
) -> None:
    epoch, study, _ = _fixture(
        tmp_path, evidence_read=True, observed={F.SEL.NEW_PROBE: "normal"}, **changes)
    result = _build(epoch, study)
    assert result["proposal"] == {"action": "hold", "target": None, "fact_ids": []}
    assert result["proposal_reason"] == expected_reason


def test_provider_invalid_hold_is_not_reinterpreted(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    data = json.loads(epoch.read_text(encoding="utf-8"))
    dialogue_path = Path(data["steps"][-1]["dialogue"]["path"])
    dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
    dialogue["calls"][-1]["result"]["finish_reason"] = "length"
    data["steps"][-1]["dialogue"] = _write(dialogue_path, dialogue)
    _write(epoch, data)
    result = _build(epoch, study)
    assert result["proposal"]["action"] == "hold"
    assert result["proposal_reason"] == "parent_hold_provider_or_contract_invalid"
    assert result["parent_closed_HOLD"]["model_valid_hold_preserved"] is True


def test_checkpoint_mismatch_is_rejected(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    data = json.loads(study.read_text(encoding="utf-8"))
    data["rows"][0]["checkpoint_sha256"] = "x" * 64
    _write(study, data)
    with pytest.raises(ValueError, match="checkpoint"):
        _build(epoch, study)


@pytest.mark.parametrize("target", ["epoch", "study"])
def test_owner_pinned_input_mutation_is_rejected(tmp_path: Path, target: str) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    epoch_sha = hashlib.sha256(epoch.read_bytes()).hexdigest()
    study_sha = hashlib.sha256(study.read_bytes()).hexdigest()
    path = epoch if target == "epoch" else study
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"expected {target} SHA"):
        _build(epoch, study, epoch_sha=epoch_sha, study_sha=study_sha)


def test_closed_state_stale_digest_field_mutation_is_rejected(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    data = json.loads(epoch.read_text(encoding="utf-8"))
    closed_path = Path(data["steps"][-1]["state_after"]["path"])
    closed = json.loads(closed_path.read_text(encoding="utf-8"))
    closed["evidence_read"] = True
    data["steps"][-1]["state_after"] = _write(closed_path, closed)
    _write(epoch, data)
    with pytest.raises(ValueError, match="exact replay"):
        _build(epoch, study)


def test_final_review_raw_response_mismatch_stays_nonexecuting_hold(tmp_path: Path) -> None:
    epoch, study, _ = _fixture(tmp_path, evidence_read=False, observed={})
    data = json.loads(epoch.read_text(encoding="utf-8"))
    dialogue_path = Path(data["steps"][-1]["dialogue"]["path"])
    dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
    dialogue["final_review"]["content"] = "different-content"
    data["steps"][-1]["dialogue"] = _write(dialogue_path, dialogue)
    _write(epoch, data)
    result = _build(epoch, study)
    assert result["proposal"] == {"action": "hold", "target": None, "fact_ids": []}
    assert result["parent_closed_HOLD"]["verified"] is False
    assert result["execution_allowed"] is False
