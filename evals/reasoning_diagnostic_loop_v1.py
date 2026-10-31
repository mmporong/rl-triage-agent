"""기록된 reasoning 첫 응답에서 시작하는 등록 검사 진단 루프.

첫 선택은 reasoning mode 비교 실행의 replicate 1 원문을 다시 호출하지 않고
소비한다. 이후 선택만 같은 O2 option 계약과 mode별 요청 파라미터를 사용한다.
이 모듈은 검사 관측과 확정/HOLD까지만 담당하며 교정·복구 권한이 없다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import evidence_guided_selector_v1 as SEL
import nemotron_reasoning_options_study_v1 as S1
import nemotron_reasoning_options_study_v2 as S2
import registry_action_loop_v1 as L1
import registry_action_options_v2 as O2

SCHEMA = "reasoning_diagnostic_loop_result_v1"
FIRST_SCHEMA = "reasoning_diagnostic_recorded_first_v1"
DIALOGUE_SCHEMA = "reasoning_diagnostic_dialogue_v1"
MAX_MODEL_ACTIONS = L1.MAX_MODEL_ACTIONS
MAX_PAIRED_PROBES = L1.MAX_PAIRED_PROBES
SOURCE_FILES = (
    "evals/reasoning_diagnostic_loop_v1.py",
    "tests/test_reasoning_diagnostic_loop_v1.py",
    "evals/registry_action_options_v2.py",
    "evals/registry_action_loop_v2.py",
    "evals/registry_action_loop_v1.py",
    "evals/evidence_guided_selector_v1.py",
    "evals/nemotron_reasoning_options_study_v1.py",
    "evals/nemotron_reasoning_options_study_v2.py",
    "evals/registered_reasoning_bridge_v1.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    S1.require(isinstance(value, dict), f"JSON 객체가 아니다: {path}")
    return value


def _json_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    S1.require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": S1.file_sha(path)}


def _transport_ref(
    value: object, *, slot: str, mode: str, messages: list[dict[str, str]],
    params: dict[str, Any], returned_result: dict[str, Any],
) -> dict[str, str]:
    """Bridge 반환값을 실제 request/response 원문과 다시 결합한다."""
    S1.require(isinstance(value, dict) and set(value) == {"path", "sha256"},
               "provider transport_ref 형식 불일치")
    ref = cast(dict[str, Any], value)
    S1.require(isinstance(ref["path"], str) and bool(ref["path"].strip())
               and isinstance(ref["sha256"], str)
               and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None,
               "provider transport_ref 값 불일치")
    response_path = Path(ref["path"])
    match = re.fullmatch(r"([0-9]{3})\.json", response_path.name)
    S1.require(response_path.is_absolute() and response_path == response_path.resolve()
               and response_path.parent.name == "responses" and match is not None
               and response_path.is_file() and S1.file_sha(response_path) == ref["sha256"],
               "provider transport response 원문 경로/SHA 불일치")
    if match is None:
        raise ValueError("provider transport response index 불일치")
    index = int(match.group(1))
    root = response_path.parent.parent.resolve()
    request_path = (root / "requests" / response_path.name).resolve()
    S1.require(response_path.parent == root / "responses"
               and request_path.parent == root / "requests" and request_path.is_file(),
               "provider transport request/response root 결합 불일치")
    response = _read(response_path)
    intent = _read(request_path)
    response_result = response.get("result")
    raw_returned = deepcopy(returned_result)
    raw_returned.pop("transport_ref", None)
    S1.require(set(response) == {"schema", "request_sha256", "result"}
               and response.get("schema") == "registered_reasoning_bridge_response_v1"
               and response.get("request_sha256") == S1.file_sha(request_path)
               and isinstance(response_result, dict)
               and response_result.get("request_params") == params
               and response_result == raw_returned,
               "provider transport response 원문/result/request SHA 결합 불일치")
    slot_match = re.fullmatch(
        r"(c0[2-8])\.(legacy_none|enabled_low)\.d([2-4])\.a([12])", slot)
    if slot_match is None or slot_match.group(2) != mode:
        raise ValueError("provider transport slot/mode 불일치")
    S1.require(set(intent) == {"schema", "index", "slot", "case_id", "mode",
                               "messages", "messages_sha256", "params", "params_sha256",
                               "bridge_protocol_sha256", "context_sha256"}
               and intent.get("schema") == "registered_reasoning_bridge_request_v1"
               and intent.get("index") == index
               and intent.get("slot") == slot
               and intent.get("case_id") == slot_match.group(1)
               and intent.get("mode") == mode
               and intent.get("messages") == messages
               and intent.get("messages_sha256") == S1.digest(messages)
               and intent.get("params") == params
               and intent.get("params_sha256") == S1.digest(params)
               and isinstance(intent.get("bridge_protocol_sha256"), str)
               and re.fullmatch(r"[0-9a-f]{64}", intent["bridge_protocol_sha256"]) is not None
               and intent.get("context_sha256") == S1.digest({
                   "slot": slot, "messages": messages, "params": params}),
               "provider transport intent messages/params/slot 결합 불일치")
    return cast(dict[str, str], deepcopy(ref))


def _mode_params(messages: list[dict[str, str]], mode: str) -> dict[str, Any]:
    request = S1.build_request(messages, mode)
    sent = request.pop("messages")
    S1.require(sent == messages, "mode request가 O2 messages를 변경했다")
    return request


def _verify_ledger(state: dict[str, Any]) -> dict[str, Any]:
    """초기 state와 Windows fact ledger의 실제 bytes/source 결합을 확인한다."""
    O2.context(state)
    S1.require(state.get("history") == [], "기록 첫 응답은 빈 history state여야 한다")
    ledger = state.get("fact_ledger")
    S1.require(isinstance(ledger, dict), "초기 fact ledger ref 형식 불일치")
    ledger = cast(dict[str, Any], ledger)
    ledger_path = ledger.get("path")
    ledger_sha = ledger.get("sha256")
    source_binding = ledger.get("source_binding")
    S1.require(isinstance(ledger_path, str) and isinstance(ledger_sha, str)
               and isinstance(source_binding, dict), "초기 fact ledger ref 형식 불일치")
    ledger_path = cast(str, ledger_path)
    ledger_sha = cast(str, ledger_sha)
    source_binding = cast(dict[str, Any], source_binding)
    path = Path(ledger_path)
    S1.require(path.is_absolute() and path.is_file()
               and SEL.LEDGER.sha(path) == ledger_sha,
               "초기 fact ledger 원문 SHA 불일치")
    checked = SEL.LEDGER.checked(path)
    S1.require(checked.get("source_binding") == source_binding,
               "초기 fact ledger source binding 불일치")
    return {"fact_ledger": _json_ref(path),
            "source_binding_digest": S1.digest(source_binding)}


def _result_rows(result: dict[str, Any], plan: dict[str, Any], cases: list[dict[str, Any]],
                 mode_result: Path, mode_plan: Path, corpus: Path) -> list[dict[str, Any]]:
    rows = result.get("calls")
    S1.require(isinstance(rows, list), "reasoning mode 32회 결과가 완결되지 않았다")
    rows = cast(list[Any], rows)
    S1.require(result.get("schema") == S2.RESULT_SCHEMA
               and result.get("status") == "complete"
               and result.get("request_count") == S1.API_CAP
               and result.get("api_cap") == S1.API_CAP
               and len(rows) == S1.API_CAP, "reasoning mode 32회 결과가 완결되지 않았다")
    S1.require(result.get("plan") == {"sha256": S1.file_sha(mode_plan)}
               and result.get("corpus") == {"sha256": S1.file_sha(corpus)},
               "reasoning 결과의 plan/corpus 결합 불일치")
    planned_by_index = {row["call_index"]: row for row in plan["calls"]}
    cases_by_id = {row["case_id"]: row for row in cases}
    seen: set[tuple[str, str, int]] = set()
    checked_rows: list[dict[str, Any]] = []
    for index, item in enumerate(rows, 1):
        S1.require(isinstance(item, dict), "reasoning 결과 call_index 순서 불일치")
        item = cast(dict[str, Any], item)
        S1.require(item.get("call_index") == index, "reasoning 결과 call_index 순서 불일치")
        planned = planned_by_index.get(index)
        S1.require(isinstance(planned, dict), "reasoning 결과 행이 등록 call과 다르다")
        planned = cast(dict[str, Any], planned)
        S1.require(all(item.get(key) == planned.get(key)
                       for key in ("call_index", "case_id", "mode", "replicate",
                                   "evaluation_role")),
                   "reasoning 결과 행이 등록 call과 다르다")
        key = (item["case_id"], item["mode"], item["replicate"])
        S1.require(key not in seen, "reasoning 결과 case/mode/replicate 중복")
        seen.add(key)
        call_root = mode_result.parent / "calls" / f"{index:02d}"
        intent_path = call_root / "intent.json"
        private_path = call_root / "private.json"
        public_path = call_root / "public.json"
        S1.require(item.get("private_sha256") == S1.file_sha(private_path)
                   and item.get("public_sha256") == S1.file_sha(public_path),
                   "reasoning 결과 행의 private/public SHA 불일치")
        intent, private, public = (_read(path) for path in
                                   (intent_path, private_path, public_path))
        case = cases_by_id[item["case_id"]]
        expected_request = S1.build_request(case["messages"], item["mode"])
        request_sha = S1.digest(expected_request)
        response = private.get("response")
        S1.require(intent.get("schema") == "nemotron_reasoning_options_intent_v2"
                   and private.get("schema") == S2.CALL_SCHEMA
                   and public == S1._public_call(private)
                   and intent.get("planned") == planned
                   and private.get("planned") == planned,
                   "reasoning 32회 call 원문 identity/projection 불일치")
        S1.require(planned.get("messages_file_sha256") == case["messages_sha256"]
                   and planned.get("messages_digest") == case["messages_digest"]
                   and intent.get("messages_file_sha256") == case["messages_sha256"]
                   and intent.get("messages_digest") == case["messages_digest"]
                   and O2.O1._digest(expected_request["messages"]) == case["messages_digest"],
                   "reasoning 32회 raw/canonical messages 결합 불일치")
        S1.require(intent.get("request") == expected_request
                   and intent.get("request_sha256") == request_sha
                   and private.get("request_sha256") == request_sha,
                   "reasoning 32회 intent/request/mode params 결합 불일치")
        S1.require(isinstance(response, dict),
                   "reasoning 32회 provider/finish/result 요약 불일치")
        response = cast(dict[str, Any], response)
        S1.require(response.get("provider_model") in S1.CALL.ACCEPTED_PROVIDER_MODELS
                   and response.get("finish_reason") == "stop"
                   and not response.get("error_type") and not response.get("infra_error")
                   and item.get("provider_model") == response.get("provider_model")
                   and item.get("finish_reason") == response.get("finish_reason")
                   and item.get("wall_s") == response.get("wall_s"),
                   "reasoning 32회 provider/finish/result 요약 불일치")
        replay_review = O2.assess(response.get("content"), case["state"])
        replay_action = O2.select_model(replay_review, case["state"])
        S1.require(private.get("review") == replay_review
                   and private.get("model_action") == replay_action["action"]
                   and item.get("valid") == (replay_review["status"] == "valid")
                   and item.get("action") == replay_action["action"],
                   "reasoning 32회 state/review/action 재생 불일치")
        checked_rows.append(item)
    S1.require(len(seen) == S1.API_CAP, "reasoning 결과 32개 call identity 불일치")
    return checked_rows


def load_recorded_first(
    mode_result: Path,
    mode_plan: Path,
    corpus: Path,
    anchor: str,
    expected_result_sha256: str,
    case_id: str,
    mode: str,
    replicate: int = 1,
) -> dict[str, Any]:
    """등록된 32회 결과에서 지정 case/mode의 실제 첫 응답을 결합해 불러온다."""
    S1.require(sys.platform == "win32", "recorded diagnostic engine은 Windows에서만 실행한다")
    S1.require(mode in S1.MODES and replicate == 1, "등록된 첫 응답은 replicate 1만 허용한다")
    S1.require(re.fullmatch(r"c0[1-8]", case_id) is not None, "case_id 형식 불일치")
    S1.require(isinstance(expected_result_sha256, str)
               and re.fullmatch(r"[0-9a-f]{64}", expected_result_sha256) is not None,
               "expected result SHA 형식 불일치")
    mode_result, mode_plan, corpus = (Path(value).resolve()
                                        for value in (mode_result, mode_plan, corpus))
    S1.require(mode_result.is_file() and S1.file_sha(mode_result) == expected_result_sha256,
               "reasoning result 원문 SHA 불일치")
    plan = S2.verify_registered(mode_plan, corpus, anchor)
    _corpus_value, cases = S1._corpus_ref(corpus, load_payloads=True)
    by_case = {row["case_id"]: row for row in cases}
    S1.require(case_id in by_case, "corpus에 case_id가 없다")
    case = by_case[case_id]
    S1.require(case["evaluation_role"] == "fault_development",
               "normal_model_only 행은 실제 진단 engine에서 실행하지 않는다")
    result = _read(mode_result)
    rows = _result_rows(result, plan, cases, mode_result, mode_plan, corpus)
    matches = [row for row in rows if row["case_id"] == case_id
               and row["mode"] == mode and row["replicate"] == replicate]
    S1.require(len(matches) == 1, "지정 case/mode/replicate 결과가 단일하지 않다")
    row = matches[0]
    planned = next(item for item in plan["calls"] if item["call_index"] == row["call_index"])
    call_root = mode_result.parent / "calls" / f"{row['call_index']:02d}"
    intent_path, private_path, public_path = (call_root / name for name in
                                               ("intent.json", "private.json", "public.json"))
    intent, private, public = (_read(path) for path in
                               (intent_path, private_path, public_path))
    S1.require(intent.get("schema") == "nemotron_reasoning_options_intent_v2"
               and private.get("schema") == S2.CALL_SCHEMA
               and public == S1._public_call(private),
               "reasoning call 원문 schema/public projection 불일치")
    S1.require(intent.get("planned") == planned and private.get("planned") == planned,
               "reasoning call planned identity 불일치")
    S1.require(planned.get("messages_file_sha256") == case["messages_sha256"]
               and planned.get("messages_digest") == case["messages_digest"]
               and intent.get("messages_file_sha256") == case["messages_sha256"]
               and intent.get("messages_digest") == case["messages_digest"],
               "reasoning call raw/canonical messages 결합 불일치")
    request = S1.build_request(case["messages"], mode)
    request_sha = S1.digest(request)
    S1.require(intent.get("request") == request
               and intent.get("request_sha256") == request_sha
               and private.get("request_sha256") == request_sha
               and O2.O1._digest(request["messages"]) == case["messages_digest"],
               "reasoning intent request/state/mode 결합 불일치")
    response = private.get("response")
    S1.require(isinstance(response, dict),
               "기록 첫 응답 provider/finish/content가 완결되지 않았다")
    response = cast(dict[str, Any], response)
    S1.require(response.get("provider_model") in S1.CALL.ACCEPTED_PROVIDER_MODELS
               and response.get("finish_reason") == "stop"
               and not response.get("error_type") and not response.get("infra_error")
               and isinstance(response.get("content"), str)
               and bool(response["content"].strip()),
               "기록 첫 응답 provider/finish/content가 완결되지 않았다")
    wall = response.get("wall_s")
    if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
            or not math.isfinite(float(wall)) or float(wall) < 0):
        raise ValueError("기록 첫 응답 wall_s 불일치")
    wall_s = float(wall)
    state = deepcopy(case["state"])
    ledger_proof = _verify_ledger(state)
    review = O2.assess(response["content"], state)
    action = O2.select_model(review, state)
    S1.require(private.get("review") == review
               and private.get("model_action") == action["action"]
               and row.get("valid") == (review["status"] == "valid")
               and row.get("action") == action["action"]
               and row.get("finish_reason") == response["finish_reason"]
               and row.get("provider_model") == response["provider_model"]
               and row.get("wall_s") == wall,
               "기록 첫 응답 review/action/result 요약 불일치")
    dialogue = {
        "schema": "reasoning_diagnostic_recorded_dialogue_v1",
        "recorded_original": True,
        "messages": deepcopy(request["messages"]),
        "messages_digest": case["messages_digest"],
        "request_params": _mode_params(case["messages"], mode),
        "raw_response": deepcopy(response),
        "review": deepcopy(review),
        "action": deepcopy(action),
        "model_valid": review["status"] == "valid",
        "semantic_feedback_used": False,
    }
    return {
        "schema": FIRST_SCHEMA,
        "case_id": case_id,
        "mode": mode,
        "replicate": replicate,
        "evaluation_role": case["evaluation_role"],
        "state": state,
        "action": deepcopy(action),
        "dialogue": dialogue,
        "refproof": {
            "anchor": anchor,
            "result": _json_ref(mode_result),
            "plan": _json_ref(mode_plan),
            "corpus": _json_ref(corpus),
            "intent": _json_ref(intent_path),
            "private": _json_ref(private_path),
            "public": _json_ref(public_path),
            "fact_ledger": ledger_proof,
            "call_index": row["call_index"],
            "request_sha256": request_sha,
        },
        "first_wall_s": wall_s,
        "firstwall": wall_s,
        "raw_response": deepcopy(response),
        "raw_response_is_recorded_original": True,
        "original_model_requests": 1,
        "new_model_requests": 0,
        "actual_system_gate_executed_in_parent": False,
        "approval": False,
        "automatic_retry": False,
        "repair": False,
        "recovery": False,
    }


def _provider_ok(result: dict[str, Any]) -> bool:
    return (result.get("provider_model") in S1.CALL.ACCEPTED_PROVIDER_MODELS
            and result.get("finish_reason") == "stop"
            and not result.get("error_type") and not result.get("infra_error")
            and isinstance(result.get("content"), str)
            and bool(result["content"].strip()))


def _dialogue(
    state: dict[str, Any], *, run_id: str, action_index: int, mode: str,
    verify_context: Callable[[], object], complete_factory: Callable[[str, dict], Callable],
) -> dict[str, Any]:
    sent = O2.messages(state)
    calls: list[dict[str, Any]] = []
    provider_ok = False
    used_slots: set[str] = set()
    for attempt in range(1, 3):
        slot = f"{run_id}.d{action_index}.a{attempt}"
        S1.require(slot not in used_slots, "model slot 중복")
        used_slots.add(slot)
        params = _mode_params(sent, mode)
        verify_context()
        complete = complete_factory(slot, deepcopy(state))
        result = complete(deepcopy(sent), deepcopy(params))
        verify_context()
        S1.require(isinstance(result, dict), "provider result 형식 불일치")
        result = deepcopy(result)
        S1.require(result.get("request_params") == params,
                   "provider actual request_params가 mode 요청과 다르다")
        transport = _transport_ref(result.get("transport_ref"), slot=slot, mode=mode,
                                   messages=sent, params=params, returned_result=result)
        provider_ok = _provider_ok(result)
        review = (O2.assess(result.get("content"), state) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["length" if result.get("finish_reason") == "length"
                      else "PROVIDER_UNAVAILABLE"],
            "response_text": result.get("content"),
        })
        calls.append({"attempt": attempt, "slot": slot, "messages": deepcopy(sent),
                      "messages_sha256": O2.O1._digest(sent),
                      "request_params": deepcopy(params), "transport_ref": transport,
                      "result": result, "review": deepcopy(review)})
        if not provider_ok or review["status"] == "valid" or attempt == 2:
            break
        current = O2.context(state)
        sent.extend([{"role": "assistant", "content": result.get("content") or ""},
                     {"role": "user", "content": json.dumps({
                         "validation_errors": review["codes"],
                         "output_schema": current["output_schema"],
                         "action_options": current["action_options"],
                         "count_evidence_groups": current["count_evidence_groups"],
                         "required": "Use one current option_id and exact visible fact_ids only."},
                         ensure_ascii=False, sort_keys=True)}])
    final = calls[-1]["review"]
    action = (O2.select_model(final, state) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    return {"schema": DIALOGUE_SCHEMA, "state_digest": state["state_digest"],
            "context_digest": O2.O1._digest(O2.context(state)), "mode": mode,
            "calls": calls, "action": action, "final_review": deepcopy(final),
            "model_valid": final["status"] == "valid",
            "semantic_feedback_used": len(calls) == 2,
            "automatic_retry": False,
            "provider_incomplete_hold": not provider_ok,
            "host_mapping": "v2_option_contract_only_no_old_response_reinterpretation"}


def _apply_action(
    state: dict[str, Any], action: dict[str, Any], stage: Path, *,
    verify_context: Callable[[], object], execute_pair: Callable[[dict, dict, Path], dict],
    paired_probes: int,
) -> tuple[dict[str, Any], int, float, float, dict[str, Any]]:
    before = deepcopy(state)
    if action["action"] == "run_probe":
        L1.require(paired_probes < MAX_PAIRED_PROBES, "paired probe 상한 초과")
        request = SEL.request_spec(before, action)
        L1.write_new(stage / "request-spec.json", request)
        verify_context()
        pair = execute_pair(before, action, stage / "probe")
        verify_context()
        proof = L1._pair_proof(pair, before, action, stage / "probe")
        L1.write_new(stage / "paired-proof.json", proof)
        state = SEL.advance(before, action, case_plan=Path(pair["case_plan"]),
                            case_receipt=Path(pair["case_receipt"]),
                            reference_plan=Path(pair["reference_plan"]),
                            reference_receipt=Path(pair["reference_receipt"]))
        O2.context(state)
        host = {"action": deepcopy(action), "request_spec": request,
                "root_approval": proof["root_approval"],
                "paired_proof": L1.ref(stage / "paired-proof.json")}
        return (state, paired_probes + 1, L1._time(pair, "parent_wall_s"),
                L1._time(pair, "owned_child_wall_s"), host)
    verify_context()
    state = SEL.advance(before, action)
    verify_context()
    O2.context(state)
    return (state, paired_probes, 0.0, 0.0,
            {"action": deepcopy(action), "request_spec": None,
             "root_approval": None, "paired_proof": None})


def run_epoch(
    first: dict,
    *,
    run_id: str,
    mode: str,
    out: Path,
    verify_context: Callable[[], object],
    complete_factory: Callable[[str, dict], Callable],
    execute_pair: Callable[[dict, dict, Path], dict],
) -> dict[str, Any]:
    """기록 첫 행동을 포함해 최대 네 행동으로 확정 또는 HOLD까지 진행한다."""
    L1.require(isinstance(first, dict) and first.get("schema") == FIRST_SCHEMA,
               "recorded first 형식 불일치")
    L1.require(mode in S1.MODES and first.get("mode") == mode, "first/mode 불일치")
    L1.require(run_id == f"{first.get('case_id')}.{mode}", "run_id는 case_id.mode여야 한다")
    L1.require(first.get("evaluation_role") == "fault_development",
               "normal_model_only engine 실행 거부")
    L1.require(callable(verify_context) and callable(complete_factory) and callable(execute_pair),
               "검증·모델·paired host callback이 필요하다")
    out = Path(out).resolve()
    L1.require(not out.exists(), "epoch out은 fresh 폴더여야 한다")
    verify_context()
    state = deepcopy(first["state"])
    _verify_ledger(state)
    L1.require(state["budget_left"] <= MAX_PAIRED_PROBES, "selector probe 예산 상한 초과")
    raw = first.get("raw_response")
    L1.require(isinstance(raw, dict), "recorded first 원문 표시/provider 불일치")
    raw = cast(dict[str, Any], raw)
    L1.require(first.get("raw_response_is_recorded_original") is True and _provider_ok(raw),
               "recorded first 원문 표시/provider 불일치")
    review = O2.assess(raw["content"], state)
    first_action = O2.select_model(review, state)
    L1.require(first.get("action") == first_action
               and first.get("dialogue", {}).get("review") == review,
               "recorded first state/review/action 재생 불일치")
    out.mkdir(parents=True, exist_ok=False)
    steps: list[dict[str, Any]] = []
    state = deepcopy(state)
    model_actions = 1
    new_requests = 0
    paired_probes = 0
    parent_wall_s = owned_child_wall_s = 0.0
    started = time.monotonic()
    stop_reason: str | None = None
    try:
        index = 1
        stage = out / f"step-{index:02d}"
        L1.write_new(stage / "state-before.json", state)
        L1.write_new(stage / "dialogue.json", first["dialogue"])
        L1.write_new(stage / "reference-proof.json", first["refproof"])
        action = deepcopy(first_action)
        selected_option = review.get("selected_option_id")
        L1.write_new(stage / "selected-action.json", {
            "selected_by": "recorded_actual_model_option",
            "selected_option_id": selected_option, "model_valid": review["status"] == "valid",
            "state_digest": state["state_digest"], "action": action,
            "approval_granted": False, "original_model_requests": 1,
            "new_model_requests": 0})
        before = deepcopy(state)
        state, paired_probes, parent_delta, child_delta, host = _apply_action(
            state, action, stage, verify_context=verify_context, execute_pair=execute_pair,
            paired_probes=paired_probes)
        parent_wall_s += parent_delta
        owned_child_wall_s += child_delta
        L1.write_new(stage / "host-action.json", host)
        L1.write_new(stage / "state-after.json", state)
        steps.append({"step": 1, "selected_by": "recorded_actual_model_option",
                      "selected_option_id": selected_option,
                      "model_valid": review["status"] == "valid",
                      "format_valid": review["format_status"] == "valid",
                      "correctness": "not_assessed_by_engine",
                      "action": action, "state_before_digest": before["state_digest"],
                      "state_after_digest": state["state_digest"],
                      "dialogue": L1.ref(stage / "dialogue.json"),
                      "reference_proof": L1.ref(stage / "reference-proof.json"),
                      "host_action": L1.ref(stage / "host-action.json"),
                      "state_after": L1.ref(stage / "state-after.json")})

        while state["status"] == "open" and state["budget_left"] > 0:
            if model_actions >= MAX_MODEL_ACTIONS:
                stop_reason = "model_action_budget_exhausted"
                break
            model_actions += 1
            index = model_actions
            stage = out / f"step-{index:02d}"
            verify_context()
            O2.context(state)
            L1.write_new(stage / "state-before.json", state)
            L1.write_new(stage / "option-context.json", O2.context(state))
            dialogue = _dialogue(state, run_id=run_id, action_index=index, mode=mode,
                                 verify_context=verify_context,
                                 complete_factory=complete_factory)
            new_requests += len(dialogue["calls"])
            L1.write_new(stage / "dialogue.json", dialogue)
            action = deepcopy(dialogue["action"])
            selected_option = dialogue.get("final_review", {}).get("selected_option_id")
            L1.write_new(stage / "selected-action.json", {
                "selected_by": "new_actual_model_option", "selected_option_id": selected_option,
                "model_valid": dialogue["model_valid"], "state_digest": state["state_digest"],
                "option_context_digest": dialogue["context_digest"], "action": action,
                "approval_granted": False})
            before = deepcopy(state)
            state, paired_probes, parent_delta, child_delta, host = _apply_action(
                state, action, stage, verify_context=verify_context, execute_pair=execute_pair,
                paired_probes=paired_probes)
            parent_wall_s += parent_delta
            owned_child_wall_s += child_delta
            L1.write_new(stage / "host-action.json", host)
            L1.write_new(stage / "state-after.json", state)
            final_review = dialogue["final_review"]
            steps.append({"step": index, "selected_by": "new_actual_model_option",
                          "selected_option_id": selected_option,
                          "model_valid": dialogue["model_valid"],
                          "format_valid": final_review.get("format_status") == "valid",
                          "correctness": "not_assessed_by_engine",
                          "action": action, "state_before_digest": before["state_digest"],
                          "state_after_digest": state["state_digest"],
                          "dialogue": L1.ref(stage / "dialogue.json"),
                          "host_action": L1.ref(stage / "host-action.json"),
                          "state_after": L1.ref(stage / "state-after.json")})
        verify_context()
        O2.context(state)
        L1.write_new(out / "final-state.json", state)
        decision = "confirmed" if state["status"] == "confirmed" else "HOLD"
        if stop_reason is None and decision == "HOLD":
            stop_reason = ("model_invalid_hold" if state["status"] == "hold" and steps
                           and steps[-1]["model_valid"] is False else state["status"])
        result = {
            "schema": SCHEMA, "status": "complete", "artifact_visibility": "private",
            "run_id": run_id, "case_id": first["case_id"], "mode": mode,
            "selection_subject": "recorded_first_then_mode_matched_model_option_v2",
            "model_actions": model_actions, "original_model_requests": 1,
            "new_model_requests": new_requests, "total_model_requests": 1 + new_requests,
            "paired_probes": paired_probes, "decision_outcome": decision,
            "stop_reason": stop_reason, "final_status": state["status"],
            "confirmed_family": (state["hypotheses"][0]
                                 if state["status"] == "confirmed" else None),
            "steps": steps, "recorded_first": {
                "response_is_original": True, "call_index": first["refproof"]["call_index"],
                "wall_s": first["first_wall_s"], "private": first["refproof"]["private"],
                "request_sha256": first["refproof"]["request_sha256"]},
            "parent_probe_wall_s": parent_wall_s,
            "owned_child_wall_s": owned_child_wall_s,
            "epoch_wall_s": time.monotonic() - started,
            "automatic_retry": False, "automatic_repair": False,
            "ground_truth_used": False, "case_category_assessed": False,
            "claims": {"diagnosis_improvement": False, "recovery": False,
                       "new_probe_execution": paired_probes > 0},
        }
        L1.write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        L1.write_new(out / "failure.json", {
            "schema": SCHEMA, "status": "invalid", "run_id": run_id,
            "error_type": type(error).__name__, "error": str(error),
            "model_actions": model_actions, "original_model_requests": 1,
            "new_model_requests": new_requests, "paired_probes": paired_probes,
            "automatic_retry": False, "automatic_repair": False, "recovery": False})
        raise


__all__ = ["DIALOGUE_SCHEMA", "FIRST_SCHEMA", "MAX_MODEL_ACTIONS",
           "MAX_PAIRED_PROBES", "SCHEMA", "SOURCE_FILES",
           "load_recorded_first", "run_epoch"]
