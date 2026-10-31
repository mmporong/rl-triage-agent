# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import test_evidence_guided_selector_v1 as SELECTOR_FIXTURE
import test_registry_action_loop_v1 as LOOP_FIXTURE

from evals import nemotron_reasoning_options_study_v1 as S1
from evals import nemotron_reasoning_options_study_v2 as S2
from evals import reasoning_diagnostic_loop_v1 as D

selector = SELECTOR_FIXTURE.selector


def _write(path: Path, value: object, *, pretty: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2 if pretty else None,
                               separators=None if pretty else (",", ":")), encoding="utf-8")
    return S1.file_sha(path)


def _corpus(root: Path, state: dict) -> Path:
    entries = []
    for index in range(1, 9):
        case_id = f"c{index:02d}"
        messages = S1.O2.messages(state)
        state_path = root / "states" / f"{case_id}.json"
        messages_path = root / "messages" / f"{case_id}.json"
        entries.append({
            "case_id": case_id,
            "host_gate": ({"run": False, "reason": "healthy_gate"}
                          if index in (1, 5) else {"run": True, "reason": None}),
            "state": {"path": f"states/{case_id}.json", "sha256": _write(state_path, state)},
            "messages": {"path": f"messages/{case_id}.json",
                         "sha256": _write(messages_path, messages, pretty=index == 2)},
            "messages_digest": S1.O2.O1._digest(messages),
        })
    path = root / "corpus.json"
    _write(path, {"schema": S1.CORPUS_SCHEMA, "source_anchor": "6" * 40,
                  "source_batch_sha256": "7" * 64,
                  "source_protocol_sha256_lf": "8" * 64,
                  "same_source_measurements": True, "post_exposure_development": True,
                  "approval_granted": False, "GPU_executed": False,
                  "entries": entries}, pretty=True)
    return path


def _provider(content: str, params: dict | None = None, *, finish: str = "stop") -> dict:
    value = {"provider_model": S1.MODEL, "finish_reason": finish, "content": content,
             "wall_s": 0.01, "input_tokens": 10, "output_tokens": 4,
             "total_tokens": 14, "reasoning_tokens": 0,
             "reasoning_tokens_status": "reported"}
    if params is not None:
        value.update(request_params=deepcopy(params))
    return value


@pytest.fixture
def recorded(selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state, _ledger, _checked = selector
    corpus = _corpus(tmp_path / "corpus-root", state)
    plan = tmp_path / "mode-plan.json"
    S1._write_new(plan, S2.prepare_plan(corpus))

    def complete(request: dict) -> dict:
        content = ('{"option_id":"read_optimizer_counts","fact_ids":[]}'
                   if request["extra_body"].get("reasoning_effort") == "low"
                   else '{"option_id":"hold","fact_ids":[]}')
        return _provider(content)

    run = tmp_path / "mode-run"
    result = S2.execute(plan, corpus, "6" * 40, run, complete=complete,
                        registration_check=lambda p, c, _a: S2.verify_plan(p, c))
    result_path = run / "result.json"
    monkeypatch.setattr(D.S2, "verify_registered", lambda p, c, _a: S2.verify_plan(p, c))
    return {"state": state, "corpus": corpus, "plan": plan, "result": result_path,
            "result_value": result, "sha": S1.file_sha(result_path)}


def _load(recorded: dict, mode: str = "enabled_low") -> dict:
    return D.load_recorded_first(recorded["result"], recorded["plan"], recorded["corpus"],
                                 "6" * 40, recorded["sha"], "c02", mode)


def _bridge_result(root: Path, slot: str, messages: list, params: dict, content: str,
                   *, finish: str = "stop") -> dict:
    index = len(list((root / "requests").glob("[0-9][0-9][0-9].json"))) + 1
    name = f"{index:03d}.json"
    case_id, mode = slot.split(".", 2)[:2]
    intent = {"schema": "registered_reasoning_bridge_request_v1", "index": index,
              "slot": slot, "case_id": case_id, "mode": mode,
              "messages": deepcopy(messages), "messages_sha256": S1.digest(messages),
              "params": deepcopy(params), "params_sha256": S1.digest(params),
              "bridge_protocol_sha256": "b" * 64,
              "context_sha256": S1.digest({"slot": slot, "messages": messages,
                                             "params": params})}
    request = root / "requests" / name
    S1._write_new(request, intent)
    result = _provider(content, params, finish=finish)
    response = {"schema": "registered_reasoning_bridge_response_v1",
                "request_sha256": S1.file_sha(request), "result": deepcopy(result)}
    response_path = root / "responses" / name
    S1._write_new(response_path, response)
    return {**result, "transport_ref": {"path": str(response_path.resolve()),
                                         "sha256": S1.file_sha(response_path)}}


def _complete_factory(root: Path, seen: list[dict], content: str):
    def factory(slot: str, state: dict):
        def complete(messages: list, params: dict) -> dict:
            seen.append({"slot": slot, "state": deepcopy(state),
                         "messages": deepcopy(messages), "params": deepcopy(params)})
            return _bridge_result(root, slot, messages, params, content)
        return complete
    return factory


def test_load_binds_actual_prefix_and_rejects_wrong_sha_normal_and_mode_tamper(recorded):
    first = _load(recorded)
    assert first["schema"] == D.FIRST_SCHEMA
    assert first["raw_response_is_recorded_original"] is True
    assert first["original_model_requests"] == 1 and first["new_model_requests"] == 0
    assert first["refproof"]["result"]["sha256"] == recorded["sha"]
    assert first["refproof"]["private"]["sha256"] == recorded["result_value"]["calls"][5]["private_sha256"]
    with pytest.raises(ValueError, match="result 원문 SHA"):
        D.load_recorded_first(recorded["result"], recorded["plan"], recorded["corpus"],
                              "6" * 40, "0" * 64, "c02", "enabled_low")
    with pytest.raises(ValueError, match="normal_model_only"):
        D.load_recorded_first(recorded["result"], recorded["plan"], recorded["corpus"],
                              "6" * 40, recorded["sha"], "c01", "enabled_low")
    changed = deepcopy(first)
    changed["mode"] = "legacy_none"
    with pytest.raises(ValueError, match="first/mode"):
        D.run_epoch(changed, run_id="c02.enabled_low", mode="enabled_low",
                    out=recorded["result"].parent / "bad-mode", verify_context=lambda: None,
                    complete_factory=lambda *_: pytest.fail("API must not run"),
                    execute_pair=lambda *_: pytest.fail("probe must not run"))


def test_recorded_hold_stops_without_new_api_or_probe(recorded, tmp_path: Path):
    first = _load(recorded, "legacy_none")
    result = D.run_epoch(first, run_id="c02.legacy_none", mode="legacy_none",
                         out=tmp_path / "hold", verify_context=lambda: None,
                         complete_factory=lambda *_: pytest.fail("HOLD must not request model"),
                         execute_pair=lambda *_: pytest.fail("HOLD must not execute SDK"))
    assert result["decision_outcome"] == "HOLD"
    assert result["original_model_requests"] == 1 and result["new_model_requests"] == 0
    assert result["steps"][0]["selected_by"] == "recorded_actual_model_option"


def test_recorded_read_then_new_mode_matched_request(recorded, tmp_path: Path):
    first = _load(recorded, "enabled_low")
    seen: list[dict] = []
    result = D.run_epoch(
        first, run_id="c02.enabled_low", mode="enabled_low", out=tmp_path / "read-hold",
        verify_context=lambda: None,
        complete_factory=_complete_factory(tmp_path / "bridge-read", seen,
                                           '{"option_id":"hold","fact_ids":[]}'),
        execute_pair=lambda *_: pytest.fail("read/hold must not execute probe"))
    assert result["decision_outcome"] == "HOLD" and result["new_model_requests"] == 1
    assert seen[0]["slot"] == "c02.enabled_low.d2.a1"
    assert seen[0]["state"]["evidence_read"] is True
    expected = S1.build_request(seen[0]["messages"], "enabled_low")
    messages = expected.pop("messages")
    assert messages == seen[0]["messages"] and seen[0]["params"] == expected


def test_probe_without_approval_receipt_is_rejected(recorded, tmp_path: Path):
    first = _load(recorded)
    state_after_read = D.SEL.advance(first["state"], first["action"])
    context = D.O2.context(state_after_read)
    option = next(row for row in context["action_options"] if row["action"] == "run_probe")
    fact_id = next(iter(context["evidence_facts"]))
    content = json.dumps({"option_id": option["option_id"], "fact_ids": [fact_id]})
    with pytest.raises(ValueError, match="plan/receipt"):
        D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "no-approval", verify_context=lambda: None,
                    complete_factory=_complete_factory(tmp_path / "bridge-no-approval", [], content),
                    execute_pair=lambda *_: {})


def test_unknown_recorded_outcome_becomes_hold(recorded, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
    first = _load(recorded)
    execute = LOOP_FIXTURE._recorded_pair(tmp_path, monkeypatch)
    original = D.SEL.RECORDED.checked_receipt

    def unknown(plan: Path, receipt: Path) -> dict:
        value = original(plan, receipt)
        return {**value, "outcome": "unknown"}

    monkeypatch.setattr(D.SEL.RECORDED, "checked_receipt", unknown)

    def factory(slot: str, state: dict):
        def complete(messages: list, params: dict) -> dict:
            ids = D.O2.context(state)["count_evidence_groups"][0]
            return _bridge_result(
                tmp_path / "bridge-unknown", slot, messages, params,
                json.dumps({"option_id": f"probe:{D.SEL.NEW_PROBE}", "fact_ids": ids}))
        return complete

    result = D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                         out=tmp_path / "unknown", verify_context=lambda: None,
                         complete_factory=factory, execute_pair=execute)
    assert result["decision_outcome"] == "HOLD"
    assert result["final_status"] == "unidentifiable" and result["paired_probes"] == 1


def test_provider_length_holds_without_retry_and_budget_cap_rejected(recorded, tmp_path: Path):
    first = _load(recorded)
    seen = []

    def factory(slot: str, _state: dict):
        def complete(messages: list, params: dict) -> dict:
            seen.append(slot)
            return _bridge_result(tmp_path / "bridge-length", slot, messages, params, "",
                                  finish="length")
        return complete

    result = D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                         out=tmp_path / "length", verify_context=lambda: None,
                         complete_factory=factory,
                         execute_pair=lambda *_: pytest.fail("length must HOLD"))
    assert seen == ["c02.enabled_low.d2.a1"]
    assert result["new_model_requests"] == 1 and result["automatic_retry"] is False

    over = deepcopy(first)
    over["state"]["budget_left"] = D.MAX_PAIRED_PROBES + 1
    over["state"] = D.SEL._seal(over["state"])
    with pytest.raises(ValueError, match="shape|상한"):
        D.run_epoch(over, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "over", verify_context=lambda: None,
                    complete_factory=lambda *_: pytest.fail("must fail before API"),
                    execute_pair=lambda *_: pytest.fail("must fail before probe"))


def test_state_change_and_callback_transport_or_params_are_rejected(recorded, tmp_path: Path):
    first = _load(recorded)
    changed = deepcopy(first)
    changed["state"]["visible_facts"]["telemetry:case:mean_reward"] = -999
    changed["state"] = D.SEL._seal(changed["state"])
    with pytest.raises(ValueError, match="selector state|review/action"):
        D.run_epoch(changed, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "changed", verify_context=lambda: None,
                    complete_factory=lambda *_: pytest.fail("must fail before API"),
                    execute_pair=lambda *_: pytest.fail("must fail before probe"))

    def wrong_factory(_slot: str, _state: dict):
        def complete(messages: list, _params: dict) -> dict:
            wrong = {"model": "wrong"}
            return _bridge_result(tmp_path / "bridge-wrong", _slot, messages, wrong,
                                  '{"option_id":"hold","fact_ids":[]}')
        return complete

    with pytest.raises(ValueError, match="request_params"):
        D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "wrong-params", verify_context=lambda: None,
                    complete_factory=wrong_factory,
                    execute_pair=lambda *_: pytest.fail("must fail before probe"))


def test_transport_response_and_intent_tamper_are_rejected(recorded, tmp_path: Path):
    first = _load(recorded)

    def returned_tamper(slot: str, _state: dict):
        def complete(messages: list, params: dict) -> dict:
            result = _bridge_result(tmp_path / "bridge-return-tamper", slot, messages, params,
                                    '{"option_id":"hold","fact_ids":[]}')
            result["content"] = '{"option_id":"read_optimizer_counts","fact_ids":[]}'
            return result
        return complete

    with pytest.raises(ValueError, match="response 원문"):
        D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "return-tamper", verify_context=lambda: None,
                    complete_factory=returned_tamper,
                    execute_pair=lambda *_: pytest.fail("must fail before probe"))

    def intent_tamper(slot: str, _state: dict):
        def complete(messages: list, params: dict) -> dict:
            result = _bridge_result(tmp_path / "bridge-intent-tamper", slot, messages, params,
                                    '{"option_id":"hold","fact_ids":[]}')
            response_path = Path(result["transport_ref"]["path"])
            request_path = response_path.parent.parent / "requests" / response_path.name
            intent = json.loads(request_path.read_text(encoding="utf-8"))
            intent["messages_sha256"] = "0" * 64
            request_path.write_text(json.dumps(intent), encoding="utf-8")
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["request_sha256"] = S1.file_sha(request_path)
            response_path.write_text(json.dumps(response), encoding="utf-8")
            result["transport_ref"]["sha256"] = S1.file_sha(response_path)
            return result
        return complete

    with pytest.raises(ValueError, match="intent messages"):
        D.run_epoch(first, run_id="c02.enabled_low", mode="enabled_low",
                    out=tmp_path / "intent-tamper", verify_context=lambda: None,
                    complete_factory=intent_tamper,
                    execute_pair=lambda *_: pytest.fail("must fail before probe"))
