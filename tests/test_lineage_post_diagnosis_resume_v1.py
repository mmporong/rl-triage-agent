from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_resume_v1 as R


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_recorded_dialogue_consumes_exactly_one_bound_response(tmp_path: Path, monkeypatch) -> None:
    snapshot = {"binding": {"x": 1}, "view_digest": "view"}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    params = {"model": "m"}
    monkeypatch.setattr(R.C, "dialogue_input", lambda *_: {
        "needs_model": True, "messages": messages, "params": params, "view": {},
        "snapshot_digest": "s", "view_digest": "view", "messages_digest": "m",
        "condition": "production",
    })
    monkeypatch.setattr(R.C.L, "digest", lambda value: "ctx" if value == {"x": 1} else "digest")
    monkeypatch.setattr(R.T, "_session", lambda _root: {})
    result = {"content": "{}", "finish_reason": "stop", "input_tokens": 1,
              "output_tokens": 2, "total_tokens": 3, "wall_s": 0.2,
              "provider_model": "m", "reasoning_content": None, "response_id": "r"}
    monkeypatch.setattr(R.T, "_validate_result", lambda value, _session: value)
    monkeypatch.setattr(R.C.CALL, "safe_result", lambda value: deepcopy(value))

    def engine(_snapshot, _view, sent, _condition, complete):
        assert complete(sent, params) == result
        return {"requests": 1, "calls": [{"result": result}], "selection": {"probe": "P_physics"}}

    monkeypatch.setattr(R.C.CALL, "engine", engine)
    monkeypatch.setattr(R.C, "choose", lambda *_: {"probe": "P_physics"})
    queue = tmp_path / "q"
    request_id = "case-07.model.3-1-abc"
    request = {"schema": R.T.REQUEST_SCHEMA, "slot_id": "case-07.model.3",
               "messages": messages, "params": params, "context_digest": "ctx",
               "view_digest": "view"}
    request_path = queue / "requests" / f"{request_id}.json"
    _write(request_path, request)
    dispatch_path = queue / "dispatches" / f"{request_id}.json"
    _write(dispatch_path, {"request_id": request_id, "request_sha256": R._sha(request_path)})
    response_path = queue / "responses" / f"{request_id}.json"
    _write(response_path, {"schema": R.T.RESPONSE_SCHEMA, "request_id": request_id,
                           "request_sha256": R._sha(request_path),
                           "dispatch_sha256": R._sha(dispatch_path), "result": result,
                           "transport_wall_s": 0.3})
    _write(queue / "response-sha" / f"{request_id}.sha256",
           {"request_id": request_id, "response_sha256": R._sha(response_path)})
    choice, record, refs = R._recorded_dialogue(snapshot, queue, "case-07.model.3")
    assert choice == {"probe": "P_physics"}
    assert record["requests"] == 1
    assert refs["total_tokens"] == 3


def test_recorded_dialogue_rejects_message_swap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(R.C, "dialogue_input", lambda *_: {
        "needs_model": True, "messages": [{"role": "user", "content": "expected"}],
        "params": {}, "view": {}, "snapshot_digest": "s", "view_digest": "v",
        "messages_digest": "m", "condition": "production",
    })
    monkeypatch.setattr(R.C.L, "digest", lambda _value: "ctx")
    monkeypatch.setattr(R.T, "_session", lambda _root: {})
    queue = tmp_path / "q"
    _write(queue / "requests/case-07.model.3-1-x.json",
           {"schema": R.T.REQUEST_SCHEMA, "slot_id": "case-07.model.3",
            "messages": [{"role": "user", "content": "swapped"}], "params": {},
            "context_digest": "ctx", "view_digest": "v"})
    with pytest.raises(ValueError, match="snapshot/messages/params"):
        R._recorded_dialogue({"binding": {}, "view_digest": "v"}, queue, "case-07.model.3")


@pytest.mark.parametrize("outcome,expected", [
    ("normal", ("HOLD", "decision_limit")),
    ("abnormal", ("HOLD", "repair_unsupported")),
])
def test_resume_outcome_contract_is_hold(outcome: str, expected: tuple[str, str]) -> None:
    if outcome == "abnormal":
        final = {"view": {"status": "confirmed", "remaining_candidates": ["physics"]}}
    else:
        final = {"view": {"status": "open", "remaining_candidates": ["actuator"]}}
    if outcome == "abnormal":
        got = ("HOLD", "repair_unsupported")
        assert final["view"]["remaining_candidates"] == ["physics"]
    else:
        got = ("HOLD", "decision_limit")
        assert final["view"]["status"] == "open"
    assert got == expected


def test_prepare_contract_has_no_execution_or_api_authority(monkeypatch, tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    _write(parent, {"schema": R.E3.PROTOCOL_SCHEMA})
    monkeypatch.setattr(R, "ROOT", tmp_path)
    for name in R.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    value = R.prepare_protocol(parent)
    assert value["scope"]["fresh_api_requests"] == 0
    assert value["scope"]["new_logical_probe_cap"] == 1
    assert value["scope"]["correction_execution"] is False
    assert value["scope"]["second_training"] is False


def test_assert_ref_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("one", encoding="utf-8")
    ref = R._ref(path)
    path.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        R._assert_ref(ref)


def test_injected_probe_executor_cannot_claim_actual_backend() -> None:
    assert R._execution_scope(None) == ("actual_backend", True)
    assert R._execution_scope(lambda *_: {}) == ("cpu_fixture", False)
