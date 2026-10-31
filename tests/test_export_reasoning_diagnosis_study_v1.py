# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_reasoning_diagnostic_loop_v1 import (
    _complete_factory,
    _load,
)

from evals import export_reasoning_diagnosis_study_v1 as E
from evals import reasoning_diagnostic_loop_v1 as LOOP

pytest_plugins = ("test_evidence_guided_selector_v1", "test_reasoning_diagnostic_loop_v1")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


@pytest.fixture
def epoch(recorded: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    first = _load(recorded, "enabled_low")
    execution = tmp_path / "execution"
    epoch_root = execution / "c02" / "enabled_low" / "epoch"
    result = LOOP.run_epoch(
        first, run_id="c02.enabled_low", mode="enabled_low", out=epoch_root,
        verify_context=lambda: None,
        complete_factory=_complete_factory(
            tmp_path / "bridge", [], '{"option_id":"hold","fact_ids":[]}'),
        execute_pair=lambda *_: pytest.fail("read/hold는 probe를 실행하면 안 된다"))
    row = {"case_id": "c02", "mode": "enabled_low", "category": "F1",
           "gate": {"run": True, "reason": "source_unhealthy"},
           "checkpoint_sha256": "a" * 64, "reference_checkpoint_sha256": "b" * 64,
           "status": "complete", "decision_outcome": result["decision_outcome"],
           "model_actions": result["model_actions"], "paired_probes": result["paired_probes"],
           "parent_probe_wall_s": result["parent_probe_wall_s"],
           "owned_child_wall_s": result["owned_child_wall_s"],
           "final_status": result["final_status"],
           **E.STUDY._metrics(epoch_root, result),
           **E.STUDY.BASE._score("F1", result),
           "wrong_family_confirmation": False, "normal_gate_model_success": False}
    _cases, expected, _proof = E._recorded_prefix(
        recorded["result"], recorded["plan"], recorded["corpus"], mode_anchor="6" * 40)
    monkeypatch.setattr(E.STUDY, "_verify_first_source", lambda *_: None)
    return {"execution": execution, "epoch_root": epoch_root, "row": row,
            "expected": expected, "case": {"category": "F1"}, "reference": {}}


def test_realistic_epoch_replays_recorded_prefix_and_bridge(epoch: dict) -> None:
    dialogues, derived = E._verify_epoch(
        epoch["execution"], epoch["row"], epoch["case"], epoch["reference"],
        epoch["expected"])
    assert len(dialogues) == 2
    assert dialogues[0]["recorded_original"] is True
    assert dialogues[1]["calls"][0]["response"]["final_json"]["option_id"] == "hold"
    assert derived["observed_actions"] == ["read_evidence", "hold"]
    encoded = json.dumps(dialogues, ensure_ascii=False)
    assert "reasoning_content" not in encoded and '"reasoning"' not in encoded


def test_new_result_and_row_tamper_are_rejected(epoch: dict) -> None:
    result_path = epoch["epoch_root"] / "result.json"
    original = json.loads(result_path.read_text(encoding="utf-8"))
    changed = deepcopy(original)
    changed["new_model_requests"] = 2
    _write(result_path, changed)
    with pytest.raises(ValueError, match="new_model_requests|request/probe"):
        E._verify_epoch(epoch["execution"], epoch["row"], epoch["case"],
                        epoch["reference"], epoch["expected"])
    _write(result_path, original)
    bad_row = deepcopy(epoch["row"])
    bad_row["decision_outcome"] = "confirmed"
    with pytest.raises(ValueError, match="decision_outcome"):
        E._verify_epoch(epoch["execution"], bad_row, epoch["case"],
                        epoch["reference"], epoch["expected"])


def test_transport_request_tamper_is_rejected_even_when_response_is_resigned(epoch: dict) -> None:
    dialogue = json.loads((epoch["epoch_root"] / "step-02" / "dialogue.json")
                          .read_text(encoding="utf-8"))
    response_path = Path(dialogue["calls"][0]["result"]["transport_ref"]["path"])
    request_path = response_path.parent.parent / "requests" / response_path.name
    intent = json.loads(request_path.read_text(encoding="utf-8"))
    intent["slot"] = "c02.enabled_low.d3.a1"
    _write(request_path, intent)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["request_sha256"] = E._sha(request_path)
    _write(response_path, response)
    dialogue["calls"][0]["result"]["transport_ref"]["sha256"] = E._sha(response_path)
    _write(epoch["epoch_root"] / "step-02" / "dialogue.json", dialogue)
    step = json.loads((epoch["epoch_root"] / "result.json").read_text(encoding="utf-8"))["steps"][1]
    step["dialogue"]["sha256"] = E._sha(epoch["epoch_root"] / "step-02" / "dialogue.json")
    result = json.loads((epoch["epoch_root"] / "result.json").read_text(encoding="utf-8"))
    result["steps"][1] = step
    _write(epoch["epoch_root"] / "result.json", result)
    with pytest.raises(ValueError, match="slot|intent"):
        E._verify_epoch(epoch["execution"], epoch["row"], epoch["case"],
                        epoch["reference"], epoch["expected"])


def test_incomplete_parent_is_rejected_before_public_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = tmp_path / "result.json"
    _write(result, {"schema": E.STUDY.RESULT_SCHEMA, "status": "invalid", "rows": []})
    monkeypatch.setattr(E.STUDY, "_validate_inputs",
                        lambda *_: ({"mode_plan": {"path": "unused"}}, [], {}))

    class Scope:
        def __enter__(self):
            return {}

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(E.STUDY.BASE.COMP.CACHE, "proof_scope", lambda: Scope())
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="complete 16행"):
        E.export(result, tmp_path / "protocol.json", tmp_path / "source.json",
                 tmp_path / "mode.json", tmp_path / "corpus.json", tmp_path / "lab",
                 anchor="4" * 40, out=out)
    assert not out.exists()


def test_public_response_keeps_final_json_and_usage_but_removes_reasoning_body() -> None:
    public = E._public_response({"content": '{"option_id":"hold","fact_ids":[]}',
        "reasoning_content": "secret chain", "reasoning": "secret",
        "reasoning_tokens": 17, "total_tokens": 31, "wall_s": 1.25,
        "finish_reason": "stop"})
    assert public["final_json"] == {"option_id": "hold", "fact_ids": []}
    assert public["reasoning_tokens"] == 17 and public["wall_s"] == 1.25
    assert "content" not in public and "reasoning" not in public
    assert "reasoning_content" not in public
