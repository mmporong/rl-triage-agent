# pyright: reportMissingImports=false
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from evals import export_prospective_agentic_comparison_v1 as X


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": X._sha(path)}


def cases(tmp_path: Path) -> list[dict[str, Any]]:
    rows = []
    categories = ("NONE", "F1", "R1_ONLY", "COMPOUND")
    for index in range(1, 9):
        fixed = write(tmp_path / "raw" / f"c{index:02d}-fixed.json", {
            "behavior": {"err_xy": index / 10, "episode_length": 100 + index}})
        result = write(tmp_path / "raw" / f"c{index:02d}-result.json", {"status": "complete"})
        rows.append({
            "case_id": f"c{index:02d}", "seed": 2320 if index <= 4 else 2321,
            "category": categories[(index - 1) % 4],
            "checkpoint": {"sha256": f"{index:064x}"},
            "fixed_report": json.loads(fixed.read_text()),
            "fixed": fixed, "result": result,
        })
    return rows


def protocol(tmp_path: Path) -> Path:
    view = write(tmp_path / "view.json", {"schema": "view"})
    return write(tmp_path / "protocol.json", {
        "protocols": {"view": {"path": str(view)}}, "claims": {}})


@contextmanager
def view_scope(*args, **kwargs):
    yield {}


def excluded_rows() -> list[dict[str, Any]]:
    return [{
        "case_id": case, "policy": policy, "status": "excluded",
        "gate": {"run": False, "reason": "source_failure_absent"},
        "decision_outcome": "HOLD", "model_requests": 0, "model_actions": 0,
        "first_pass_valid_actions": 0, "post_feedback_valid_actions": 0,
        "semantic_feedback_actions": 0, "paired_probes": 0,
        "correct_confirmed_family": False, "correct_complete_family": False,
        "compound_partial": False, "compound_complete": False,
    } for case, policy in X._row_order()]


def setup_export(monkeypatch: pytest.MonkeyPatch, source_cases: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(X.RUN, "_source_cases", lambda *args: source_cases)
    monkeypatch.setattr(X.RV, "view_scope", view_scope)


def test_complete_export_preserves_24_row_denominator_and_has_no_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_cases = cases(tmp_path)
    source_cases[0]["category"] = "F1"
    setup_export(monkeypatch, source_cases)
    source = write(tmp_path / "source-result.json", {"status": "complete"})
    proto = protocol(tmp_path)
    comparison = write(tmp_path / "comparison/result.json", {
        "schema": X.RUN.RESULT_SCHEMA, "status": "complete",
        "protocol": ref(proto), "source_batch": ref(source), "anchor": "a" * 40,
        "planned_cases": 8, "planned_policy_rows": 24, "rows": excluded_rows(),
    })
    out = tmp_path / "public"
    manifest = X.export(source, comparison, protocol_path=proto, anchor="a" * 40, out=out)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["planned_source_training_jobs"] == 9
    assert summary["planned_fixed_evaluations"] == 8
    assert summary["planned_policy_rows"] == summary["exported_policy_rows"] == 24
    assert summary["missing_policy_rows"] == []
    assert len(list((out / "cases").rglob("*.json"))) == 24
    assert len(list((out / "source").glob("*.json"))) == 8
    assert "raw_byte_copy" not in json.dumps(manifest)
    for path in out.rglob("*.json"):
        assert X.FORBIDDEN.search(path.read_text(encoding="utf-8")) is None


def epoch_fixture(tmp_path: Path, *, wrong_score: bool = False) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "comparison"
    epoch_root = root / "c01/eager_model"
    dialogue = {
        "schema": "prospective_agentic_dialogue_v1", "model_valid": True,
        "semantic_feedback_used": False,
        "calls": [{"messages": [{"role": "user", "content": "literal input"}],
                   "result": {"content": "literal provider result"},
                   "review": {"status": "valid"}}],
    }
    dialogue_path = write(epoch_root / "step-01/dialogue.json", dialogue)
    state = {"status": "confirmed", "hypotheses": ["reward"],
             "observed": {"P_reward": "abnormal"}, "state_digest": "d" * 64}
    write(epoch_root / "step-01/state-after.json", state)
    write(epoch_root / "final-state.json", state)
    epoch = {
        "schema": "prospective_agentic_epoch_v1", "status": "complete",
        "policy": "eager_model", "decision_outcome": "confirmed",
        "confirmed_family": "reward", "request_count": 1, "model_actions": 1,
        "probe_count": 1, "parent_probe_wall_s": 2.0, "owned_child_wall_s": 1.0,
        "final_status": "confirmed", "remaining_candidates": ["reward"],
        "observations": {"P_reward": "abnormal"},
        "steps": [{"dialogue_sha256": X._sha(dialogue_path),
                   "state_after_digest": "d" * 64}],
    }
    write(epoch_root / "result.json", epoch)
    score = X.RUN._score("NONE" if wrong_score else "F1", epoch)
    validity = X.RUN._validity(epoch_root, epoch)
    row = {
        "case_id": "c01", "policy": "eager_model", "status": "complete",
        "gate": {"run": True, "reason": None}, "decision_outcome": "confirmed",
        "model_requests": 1, "model_actions": 1, "paired_probes": 1,
        "parent_probe_wall_s": 2.0, "owned_child_wall_s": 1.0,
        **score, **validity,
    }
    return root, row


def test_partial_failure_exports_verified_literal_dialogue_and_marks_missing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_cases = cases(tmp_path)
    source_cases[0]["category"] = "F1"
    setup_export(monkeypatch, source_cases)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    source = write(tmp_path / "source-result.json", {"status": "complete"})
    proto = protocol(tmp_path)
    comparison_root, row = epoch_fixture(tmp_path)
    failure = write(comparison_root / "failure.json", {
        "schema": X.RUN.RESULT_SCHEMA, "status": "invalid", "rows": [row],
        "error_type": "FixtureStop", "error": "bounded failure"})
    out = tmp_path / "public"
    manifest = X.export(source, failure, protocol_path=proto, anchor="a" * 40, out=out)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "invalid"
    assert summary["exported_policy_rows"] == 1
    assert len(summary["missing_policy_rows"]) == 23
    dialogue = out / "dialogues/c01/eager_model/dialogue-01.json"
    original = comparison_root / "c01/eager_model/step-01/dialogue.json"
    assert dialogue.read_bytes() == original.read_bytes()
    assert manifest["files"]["dialogues/c01/eager_model/dialogue-01.json"]["mode"] == "raw_byte_copy"


def test_path_or_secret_dialogue_is_explicitly_sanitized_with_two_hashes(tmp_path: Path) -> None:
    source = write(tmp_path / "raw.json", {
        "messages": [{"content": r"C:\Users\someone\private.json"}],
        "api_key": "not-a-real-key"})
    target = tmp_path / "public.json"
    entry = X._copy_payload(source, target, "c01/model/dialogue")
    value = json.loads(target.read_text(encoding="utf-8"))
    assert entry["mode"] == "sanitized_derived_json"
    assert value["schema"] == X.SANITIZED_SCHEMA
    assert value["original_sha256"] == X._sha(source)
    assert value["public_payload"]["api_key"] == "<redacted-secret>"
    assert X.FORBIDDEN.search(target.read_text(encoding="utf-8")) is None
    assert entry["sha256"] == X._sha(target)


def test_recomputed_score_mismatch_rejects_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_cases = cases(tmp_path)
    source_cases[0]["category"] = "F1"
    setup_export(monkeypatch, source_cases)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    source = write(tmp_path / "source-result.json", {"status": "complete"})
    proto = protocol(tmp_path)
    comparison_root, row = epoch_fixture(tmp_path, wrong_score=True)
    failure = write(comparison_root / "failure.json", {
        "schema": X.RUN.RESULT_SCHEMA, "status": "invalid", "rows": [row]})
    with pytest.raises(ValueError):
        X.export(source, failure, protocol_path=proto, anchor="a" * 40,
                 out=tmp_path / "public")
