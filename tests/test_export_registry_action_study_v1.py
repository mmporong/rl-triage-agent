# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import export_registry_action_study_v1 as X


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": X.PRIV._sha(path)}


def dialogue(state: str, facts: dict[str, object], action: dict[str, object], *,
             valid: bool = True) -> dict[str, object]:
    return {"state_digest": state, "action": action, "model_valid": valid,
            "calls": [{"messages": [{"role": "system", "content": "contract"},
                                     {"role": "user", "content": json.dumps({
                                         "evidence_facts": facts})}],
                       "review": {"status": "valid" if valid else "invalid"}}]}


def producer_fixture(tmp_path: Path, *, action_name: str = "hold",
                     corrupt_state_ref: bool = False) -> tuple[Path, dict, Path]:
    root = tmp_path / "study"
    epoch = root / "c03/eager_model/epoch"
    prior_root = tmp_path / "prior"
    action = {"action": action_name, "target": None, "fact_ids": []}
    before = {"state_digest": "a" * 64, "status": "open"}
    after = {"state_digest": "b" * 64, "status": "hold"}
    write(epoch / "step-01/state-before.json", before)
    write(epoch / "step-01/option-context.json", {"state": "context"})
    dpath = write(epoch / "step-01/dialogue.json", dialogue(
        "a" * 64, {"runtime:case:available": True}, action))
    selected = write(epoch / "step-01/selected-action.json", {
        "action": action, "dialogue": ref(dpath)})
    del selected
    hpath = write(epoch / "step-01/host-action.json", {"action": action})
    apath = write(epoch / "step-01/state-after.json", after)
    write(epoch / "final-state.json", after)
    step = {"step": 1, "action": action, "model_valid": True,
            "state_before_digest": "a" * 64, "state_after_digest": "b" * 64,
            "dialogue": ref(dpath), "host_action": ref(hpath), "state_after": ref(apath)}
    if corrupt_state_ref:
        step["state_after"]["sha256"] = "0" * 64
    raw = {"schema": X.LOOP.SCHEMA, "status": "complete", "run_id": "c03.eager_model",
           "decision_outcome": "HOLD", "model_actions": 1, "paired_probes": 0,
           "parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0,
           "final_status": "hold", "confirmed_family": None, "steps": [step]}
    write(epoch / "result.json", raw)
    prior_epoch = prior_root / "c03/eager_model"
    prior_dialogue = write(prior_epoch / "step-01/dialogue.json", {
        "calls": [{"messages": [{"role": "system", "content": "old"},
                                 {"role": "user", "content": json.dumps({
                                     "evidence_facts": {"runtime:case:available": True}})}]}]})
    write(prior_epoch / "result.json", {"steps": [{"dialogue_sha256": X.PRIV._sha(prior_dialogue)}]})
    row = {"case_id": "c03", "policy": "eager_model", "gate": {"run": True},
           "status": "complete", "decision_outcome": "HOLD", "model_requests": 1,
           "model_actions": 1, "paired_probes": 0, "parent_probe_wall_s": 0.0,
           "owned_child_wall_s": 0.0, "final_status": "hold",
           "first_decision_valid": True, "post_feedback_first_decision_valid": True,
           **X.STUDY._score("R1_ONLY", raw)}
    return root, row, prior_root


def setup_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(X.STUDY.SEL, "context", lambda state: {})
    monkeypatch.setattr(X.OPTIONS, "context", lambda state: {"state": "context"})


def test_full_loop_producer_refs_and_valid_hold_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_replay(monkeypatch)
    root, row, prior = producer_fixture(tmp_path)
    epoch, dialogues, metrics = X._verify_epoch(root, row, "R1_ONLY", prior)
    assert epoch is not None and len(dialogues) == 1
    assert metrics == {"first_action": "hold", "first_response_valid": True,
                       "post_feedback_valid": True, "valid_hold": True}
    assert row["decision_outcome"] == "HOLD"


def test_tampered_state_ref_is_rejected(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    setup_replay(monkeypatch)
    root, row, prior = producer_fixture(tmp_path, corrupt_state_ref=True)
    with pytest.raises(ValueError, match="state after ref"):
        X._verify_epoch(root, row, "R1_ONLY", prior)


def test_normal_host_gate_is_not_model_success(tmp_path: Path) -> None:
    row = {"case_id": "c01", "policy": "eager_model", "gate": {"run": False},
           "model_requests": 0, "paired_probes": 0, "decision_outcome": "HOLD",
           "normal_gate_model_success": False}
    epoch, dialogues, metrics = X._verify_epoch(tmp_path, row, "NONE", tmp_path)
    assert epoch is None and dialogues == []
    assert metrics["first_response_valid"] is None and metrics["valid_hold"] is False


def test_complete_result_cannot_fill_missing_denominator() -> None:
    rows = [{"case_id": case, "policy": policy}
            for case, policy in X._row_order()[:-1]]
    with pytest.raises(ValueError, match="16/12/4"):
        X._rows({"schema": X.STUDY.RESULT_SCHEMA, "status": "complete",
                 "planned_rows": 16, "fault_rows": 12,
                 "normal_host_gate_rows": 4, "rows": rows})


def test_invalid_prefix_preserves_missing_rows_without_zero_fill() -> None:
    expected = X._row_order()
    rows = [{"case_id": case, "policy": policy} for case, policy in expected[:3]]
    assert X._rows({"schema": X.STUDY.RESULT_SCHEMA, "status": "invalid", "rows": rows}) == rows
    rows[1]["policy"] = "wrong"
    with pytest.raises(ValueError, match="순서"):
        X._rows({"schema": X.STUDY.RESULT_SCHEMA, "status": "invalid", "rows": rows})


def test_script_help_bootstraps_repository_package() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(X.ROOT / "evals/export_registry_action_study_v1.py"), "--help"],
        cwd=X.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    assert result.returncode == 0, result.stderr
    assert "--study-result" in result.stdout
