# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import export_diagnostic_option_study_v1 as X


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": X.PRIV._sha(path)}


def rows() -> list[dict]:
    return [{"case_id": f"c{i:02d}", "policy": X.STUDY.POLICY}
            for i in range(1, 9)]


def test_complete_requires_exact_eight_row_registered_denominator() -> None:
    value = {"schema": X.STUDY.RESULT_SCHEMA, "status": "complete", "rows": rows(),
             "planned_rows": 8, "fault_rows": 6, "normal_host_gate_rows": 2}
    assert X._rows(value) == rows()
    value["rows"] = rows()[:-1]
    with pytest.raises(ValueError, match="complete 8행"):
        X._rows(value)
    value["rows"] = rows()
    value["rows"][0]["case_id"] = "c08"
    with pytest.raises(ValueError, match="순서"):
        X._rows(value)


def test_candidate_keeps_format_hold_and_confirmation_separate() -> None:
    current = []
    for index in range(8):
        run = index >= 2
        current.append({"gate": {"run": run}, "first_response_valid": True if run else None,
                        "first_action": "hold" if run else None,
                        "correct_confirmed_family": run and index == 2,
                        "wrong_family_confirmation": False,
                        "model_requests": 1 if run else 0, "paired_probes": 0})
    prior = [{"policy": X.STUDY.POLICY, "gate": {"run": True},
              "correct_confirmed_family": False} for _ in range(6)]
    result = X._candidate(current, prior)
    assert result["first_valid"] == 6 and result["correct"] == 1
    assert result["all_hold"] is True and result["candidate"] is False
    current[2]["first_action"] = "run_probe"
    assert X._candidate(current, prior)["candidate"] is True
    current[3]["wrong_family_confirmation"] = True
    assert X._candidate(current, prior)["candidate"] is False


def producer(tmp_path: Path, *, with_proof: bool = False) -> tuple[Path, dict, Path]:
    root = tmp_path / "study"
    epoch = root / "c03" / X.STUDY.POLICY / "epoch"
    prior = tmp_path / "prior"
    action = {"action": "run_probe" if with_proof else "hold",
              "target": "P_reward" if with_proof else None,
              "fact_ids": ["telemetry:case:mean_reward"] if with_proof else []}
    before = {"state_digest": "a" * 64, "status": "open"}
    after = {"state_digest": "b" * 64, "status": "hold"}
    write(epoch / "step-01/state-before.json", before)
    write(epoch / "step-01/option-context.json", {"context": "v2"})
    dialogue = {"schema": "registry_action_options_dialogue_v2",
                "state_digest": before["state_digest"], "action": action,
                "model_valid": True, "final_review": {"response_text": "raw"},
                "calls": [{"messages": [{"role": "system", "content": "contract"},
                                          {"role": "user", "content": json.dumps({
                                              "evidence_facts": {"fact": 1}})}],
                           "review": {"status": "valid"}}]}
    dialogue_path = write(epoch / "step-01/dialogue.json", dialogue)
    write(epoch / "step-01/selected-action.json", {"action": action,
                                                     "dialogue": ref(dialogue_path)})
    host_path = write(epoch / "step-01/host-action.json", {"action": action})
    after_path = write(epoch / "step-01/state-after.json", after)
    if with_proof:
        proof = {}
        for name in ("case_plan", "case_receipt", "reference_plan", "reference_receipt",
                     "root_approval"):
            proof[name] = ref(write(epoch / "step-01/probe" / f"{name}.json", {"name": name}))
        write(epoch / "step-01/paired-proof.json", proof)
    step = {"action": action, "model_valid": True,
            "state_before_digest": before["state_digest"],
            "state_after_digest": after["state_digest"], "dialogue": ref(dialogue_path),
            "host_action": ref(host_path), "state_after": ref(after_path)}
    raw = {"schema": X.LOOP.SCHEMA, "status": "complete",
           "run_id": f"c03.{X.STUDY.POLICY}", "decision_outcome": "HOLD",
           "model_actions": 1, "paired_probes": 1 if with_proof else 0,
           "parent_probe_wall_s": 0.1 if with_proof else 0.0,
           "owned_child_wall_s": 0.0, "final_status": "hold", "steps": [step]}
    write(epoch / "result.json", raw)
    write(epoch / "final-state.json", after)
    parent_epoch = prior / "c03" / X.STUDY.POLICY / "epoch"
    parent_action = {"action": "hold", "target": None, "fact_ids": []}
    parent_before = write(parent_epoch / "step-01/state-before.json", before)
    del parent_before
    write(parent_epoch / "step-01/option-context.json", {"context": "v1"})
    parent_dialogue = write(parent_epoch / "step-01/dialogue.json", {
        "schema": "registry_action_options_dialogue_v1", "state_digest": before["state_digest"],
        "action": parent_action, "final_review": {"response_text": "old"},
        "calls": [{"messages": [{"role": "system", "content": "old"},
                                  {"role": "user", "content": json.dumps({
                                      "evidence_facts": {"fact": 1}})}]}]})
    write(parent_epoch / "result.json", {"schema": X.OLD.LOOP.SCHEMA, "status": "complete",
          "run_id": f"c03.{X.STUDY.POLICY}", "model_actions": 1,
          "steps": [{"state_before_digest": before["state_digest"], "action": parent_action,
                     "dialogue": ref(parent_dialogue)}]})
    row = {"case_id": "c03", "policy": X.STUDY.POLICY, "gate": {"run": True},
           "status": "complete", "decision_outcome": "HOLD", "model_requests": 1,
           "model_actions": 1, "paired_probes": raw["paired_probes"],
           "parent_probe_wall_s": raw["parent_probe_wall_s"], "owned_child_wall_s": 0.0,
           "final_status": "hold", "first_decision_valid": True,
           "post_feedback_first_decision_valid": True,
           **X.STUDY.BASE._score("R1_ONLY", raw)}
    return root, row, prior


def setup_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(X.STUDY.BASE.SEL, "context", lambda _state: {})
    monkeypatch.setattr(X.OPTIONS, "context", lambda _state: {"context": "v2"})
    monkeypatch.setattr(X.OPTIONS, "select_model",
                        lambda _review, _state: {"action": "hold", "target": None,
                                                 "fact_ids": []})
    monkeypatch.setattr(X.OLD.STUDY.SEL, "context", lambda _state: {})
    monkeypatch.setattr(X.OLD.OPTIONS, "context", lambda _state: {"context": "v1"})
    monkeypatch.setattr(X.OLD.OPTIONS, "select_model",
                        lambda _review, _state: {"action": "hold", "target": None,
                                                 "fact_ids": []})


def test_v2_epoch_replays_state_dialogue_and_parent_visible_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_replay(monkeypatch)
    root, row, prior = producer(tmp_path)
    epoch, dialogues, metrics = X._verify_epoch(root, row, "R1_ONLY", prior)
    assert epoch is not None and len(dialogues) == 1
    assert metrics["first_response_valid"] is True and metrics["valid_hold"] is True


def test_run_probe_requires_all_paired_proof_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_replay(monkeypatch)
    root, row, prior = producer(tmp_path, with_proof=True)
    monkeypatch.setattr(X.OPTIONS, "select_model", lambda _review, _state: row_action(row, root))
    proof = root / "c03" / X.STUDY.POLICY / "epoch/step-01/paired-proof.json"
    proof.unlink()
    with pytest.raises(ValueError, match="paired proof"):
        X._verify_epoch(root, row, "R1_ONLY", prior)


def row_action(_row: dict, root: Path) -> dict:
    return json.loads((root / "c03" / X.STUDY.POLICY / "epoch/step-01/dialogue.json")
                      .read_text(encoding="utf-8"))["action"]


def test_script_and_module_help_bootstrap_before_any_export() -> None:
    script = X.ROOT / "evals/export_diagnostic_option_study_v1.py"
    commands = ([sys.executable, "-B", str(script), "--help"],
                [sys.executable, "-B", "-m", "evals.export_diagnostic_option_study_v1", "--help"])
    for command in commands:
        result = subprocess.run(command, cwd=X.ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False)
        assert result.returncode == 0, result.stderr
        assert "--study-result" in result.stdout
