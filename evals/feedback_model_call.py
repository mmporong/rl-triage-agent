"""검증 피드백 1회 재요청 arm: 기존 단일 호출과 같은 1차 호출 뒤, 기존 계약 검증기가 거절하면 문제 코드와 짧은 규칙 설명을
붙여 같은 대화로 한 번만 다시 요청하고 그 결과를 최종으로 검증한다.

evals/active_model_call.py(진단)·evals/correction_model_call.py(교정)는 바꾸지 않는다. 1차 요청은 같은 메시지 함수
(active_diagnosis.decision_messages, correction_model_call.build_messages)·모델·temperature 0·reasoning_effort none·
max_tokens 16384다. 재요청은 1차 메시지 + 1차 응답(assistant) + 피드백(user)이고 요청 설정은 같다.

피드백은 검증기 거절 메시지를 등록된 문제 코드로 바꾼 것이다. 설명은 프롬프트의 계약 규칙을 다시 적은 고정 문장이고, 정답·
카탈로그 값·사실 id·값·등록 교정 같은 packet별 내용이나 프롬프트 밖 힌트를 넣지 않는다. 재요청 뒤에도 무효면 무효다.

  python evals/feedback_model_call.py --kind diagnosis|correction --packet <packet.json> --output <run.json> --model <모델>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

RECORD_SCHEMA = "feedback_model_call_v1"
MAX_TOKENS = 16384
# 검증기 거절 메시지(앞부분) → 문제 코드. 위에서부터 처음 맞는 것을 쓴다.
DIAGNOSIS_CODES = [
    ("decision은 JSON 객체여야 한다", "D_JSON"),
    ("decision 필드는", "D_FIELDS"),
    ("ranking은", "D_RANKING"),
    ("action은", "D_ACTION"),
    ("reason은", "D_REASON"),
    ("expected_outcomes는 JSON 객체여야 한다", "D_EXPECTED_TYPE"),
    ("continue probe는", "D_PROBE"),
    ("남은 예산 없이", "D_BUDGET"),
    ("expected_outcomes는 선택 probe의", "D_EXPECTED"),
    ("hold/confirm의 probe는", "D_HOLD_PROBE"),
    ("hold/confirm의 expected_outcomes는", "D_HOLD_EXPECTED"),
    ("unknown/실패/예산 초과 receipt 뒤에는", "D_HOLD_ONLY"),
    ("초기 판단은 confirm할 수 없다", "D_CONFIRM_INITIAL"),
    ("confirm은", "D_CONFIRM_RULE"),
    ("claims는", "D_CLAIMS_EMPTY"),
    ("claim은 id와 value만", "D_CLAIM_SHAPE"),
    ("알 수 없거나 중복된 claim id", "D_CLAIM_ID"),
    ("claim 값이 evidence fact와 다르다", "D_CLAIM_VALUE"),
    ("post decision은 최신 receipt fact를", "D_LATEST_RECEIPT"),
]
DIAGNOSIS_TEXT = {
    "D_JSON": "The output must be exactly one JSON object.",
    "D_FIELDS": "The object must have exactly these fields: ranking, probe, action, reason, claims, expected_outcomes.",
    "D_RANKING": "ranking must be a permutation of all six public mechanisms.",
    "D_ACTION": "action must be continue, hold, or confirm.",
    "D_REASON": "reason must be a non-empty concise explanation.",
    "D_EXPECTED_TYPE": "expected_outcomes must be a JSON object.",
    "D_PROBE": "For action=continue, probe must be one unobserved public probe.",
    "D_BUDGET": "With budget_left=0, continue is not allowed.",
    "D_EXPECTED": ("For continue, expected_outcomes must exactly copy the selected probe's public catalog expectations for "
                   "the top three ranked mechanisms."),
    "D_HOLD_PROBE": "For hold or confirm, probe must be null.",
    "D_HOLD_EXPECTED": "For hold or confirm, expected_outcomes must be an empty object.",
    "D_HOLD_ONLY": "After an unknown or unusable latest observation, return hold.",
    "D_CONFIRM_INITIAL": "Without a usable observation, never confirm.",
    "D_CONFIRM_RULE": ("Confirm only when the deterministic evidence state is confirmed and its sole hypothesis is ranked "
                       "first."),
    "D_CLAIMS_EMPTY": "claims must contain one or more {\"id\": evidence_fact_id, \"value\": exact_fact_value} objects.",
    "D_CLAIM_SHAPE": "Each claim must be an object with only id and value.",
    "D_CLAIM_ID": "Each claim id must be a distinct available evidence fact id.",
    "D_CLAIM_VALUE": "Every claim must copy an available evidence fact value exactly.",
    "D_LATEST_RECEIPT": "After an observation, cite at least one fact from the latest receipt.",
    "D_OTHER": "The decision does not satisfy the contract in the instructions.",
}
CORRECTION_CODES = [
    ("JSON 파손", "C_JSON"), ("JSON 객체가 아니다", "C_JSON"), ("빈 응답", "C_JSON"),
    ("제안은", "C_KEYS"),
    ("hold_reason은", "C_HOLD_REASON"),
    ("HOLD 제안은", "C_HOLD_IDS"),
    ("correction_id와 recovery_mode는 문자열", "C_TYPES"),
    ("확정 메커니즘이 등록 표에 없다", "C_MECHANISM"),
    ("등록 표에 없는 교정", "C_CORRECTION"), ("확정 메커니즘의 교정이 아니다", "C_CORRECTION"),
    ("등록되지 않은 회복 방식", "C_MODE"),
    ("근거 인용(cited_facts)이", "C_CITED_EMPTY"),
    ("제안 근거에 없는 인용", "C_CITED_UNKNOWN"),
    ("체크리스트는", "C_CHECKLIST_KEYS"),
    ("체크리스트 ", "C_CHECKLIST_VALUE"),
]
CORRECTION_TEXT = {
    "C_JSON": "출력은 JSON 객체 하나뿐이어야 한다.",
    "C_KEYS": "correction_id·recovery_mode·cited_facts·checklist·hold_reason 다섯 키만 쓴다.",
    "C_HOLD_REASON": "hold_reason은 null이거나 비지 않은 문자열이다.",
    "C_HOLD_IDS": "보류하면 correction_id·recovery_mode는 null이다.",
    "C_TYPES": "제안하면 correction_id와 recovery_mode는 문자열이다.",
    "C_MECHANISM": "registry에 확정 메커니즘 항목이 없으면 보류한다.",
    "C_CORRECTION": "correction_id는 registry의 확정 메커니즘 항목 값만 쓴다.",
    "C_MODE": "recovery_mode는 registry의 확정 메커니즘 항목 값만 쓴다.",
    "C_CITED_EMPTY": "cited_facts는 facts의 키를 하나 이상 인용한 문자열 목록이다.",
    "C_CITED_UNKNOWN": "cited_facts와 checklist 값은 facts의 키만 인용한다.",
    "C_CHECKLIST_KEYS": "checklist는 skill_checklist의 모든 번호를 키로 가진다.",
    "C_CHECKLIST_VALUE": "checklist 값은 그 항목을 충족하는 facts 키 하나, 근거가 없으면 \"unmet\"이다.",
    "C_OTHER": "제안이 지시문의 계약을 만족하지 않는다.",
}
FEEDBACK_HEAD = {"diagnosis": ("The previous JSON was rejected by the public contract validator.\nProblems:",
                               "Return one corrected JSON object only, under the same contract."),
                 "correction": ("직전 JSON이 공개 계약 검증에서 거절됐다.\n문제:", "같은 계약을 따르는 고친 JSON 객체 하나만 출력한다.")}


def codes_for(kind: str, reasons: list[str]) -> list[str]:
    table = DIAGNOSIS_CODES if kind == "diagnosis" else CORRECTION_CODES
    other = "D_OTHER" if kind == "diagnosis" else "C_OTHER"
    out = []
    for reason in reasons:
        code = next((c for prefix, c in table if reason.startswith(prefix)), other)
        if code not in out:
            out.append(code)
    return out


def feedback_text(kind: str, codes: list[str]) -> str:
    """문제 코드 → 고정 피드백 문장. packet별 값이나 정답을 넣지 않는다."""
    texts = DIAGNOSIS_TEXT if kind == "diagnosis" else CORRECTION_TEXT
    head, tail = FEEDBACK_HEAD[kind]
    return "\n".join([head, *(f"- {code}: {texts[code]}" for code in codes), tail])


def diagnosis_body(content) -> str:
    """진단 응답 본문. e2e_reward_demo_v2.review_diagnosis_record와 같이 맨 앞의 ```json 울타리만 벗긴다."""
    text = content or ""
    if text.startswith("```json") and text.rstrip().endswith("```"):
        text = text[7:text.rfind("```")].strip()
    return text


def validate(kind: str, content, packet: dict) -> dict:
    """기존 계약 검증기로 판정. {status: valid|invalid, decision?, reasons, codes}."""
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        try:
            decision = D.validate_decision(json.loads(diagnosis_body(content)), packet)
        except json.JSONDecodeError:
            return {"status": "invalid", "reasons": ["JSON 파손"], "codes": ["D_JSON"]}
        except (ValueError, TypeError) as error:
            reasons = [str(error)[:300]]
            return {"status": "invalid", "reasons": reasons, "codes": codes_for(kind, reasons)}
        return {"status": "valid", "decision": decision, "reasons": [], "codes": []}
    import correction_model_call as CM
    proposal, error = CM.parse_content(content)
    verdict = {"accepted": False, "reasons": [error]} if error else CM.validate_model_proposal(proposal, packet,
                                                                                             packet["registry"])
    if not verdict["accepted"]:
        reasons = [r[:300] for r in verdict["reasons"]]
        return {"status": "invalid", "reasons": reasons, "codes": codes_for(kind, reasons)}
    return {"status": "valid", "reasons": [], "codes": [],
            "decision": {"decision": verdict["decision"], "proposal": proposal, "agrees_with_rule": verdict["agrees_with_rule"],
                         "unmet_checklist": verdict["unmet_checklist"]}}


def first_messages(kind: str, data: bytes) -> tuple[list[dict], dict]:
    """1차 메시지와 결합 필드(기존 단일 호출과 같은 함수·같은 결합)."""
    from rl_triage import probe_loop as L
    packet = json.loads(data)
    if kind == "diagnosis":
        from rl_triage.active_diagnosis import decision_messages
        messages = decision_messages(packet)
        return messages, {"packet_sha256": hashlib.sha256(data).hexdigest(), "messages_sha256": L.digest(messages)}
    import correction_model_call as CM
    if packet.get("schema") != CM.PACKET_SCHEMA or packet.get("checklist") != CM.skill_checklist():
        raise ValueError("packet 형식이나 skill 체크리스트가 고정 원문과 다르다")
    messages = CM.build_messages(packet)
    return messages, {"packet_sha256": hashlib.sha256(data).hexdigest(), "packet_digest": CM.packet_digest(packet),
                      "skill_sha256": CM.SKILL_SHA256, "messages_sha256": L.digest(messages)}


def retry_messages(messages: list[dict], first_content: str, kind: str, codes: list[str]) -> list[dict]:
    return [*messages, {"role": "assistant", "content": first_content}, {"role": "user", "content": feedback_text(kind, codes)}]


def request_params(model: str) -> dict:
    return {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
            "extra_body": {"reasoning_effort": "none"}}


def dialogue(kind: str, data: bytes, model: str, complete: Callable[[list[dict], dict], dict]) -> dict:
    """1차 호출 → (무효면) 피드백 재요청 1회. complete(messages, params)는 호출 결과 dict를 돌려준다
    ({content, finish_reason, input_tokens, output_tokens, ..., infra_error?, error_type?})."""
    from rl_triage import probe_loop as L
    packet = json.loads(data)
    messages, binding = first_messages(kind, data)
    params = request_params(model)
    record: dict[str, Any] = {"schema": RECORD_SCHEMA, "kind": kind, **binding, "request": params, "calls": [],
                              "feedback": None, "final": None, "requests": 0}
    for step in ("first", "retry"):
        if step == "retry":
            first = record["calls"][0]
            sent = retry_messages(messages, first["content"], kind, first["codes"])
            text = sent[-1]["content"]
            record["feedback"] = {"codes": first["codes"], "text": text,
                                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "messages_sha256": L.digest(sent)}
        else:
            sent = messages
        result = complete(sent, params)
        record["requests"] += 1
        call = {"step": step, **{k: result.get(k) for k in ("response_id", "provider_model", "finish_reason", "content",
                                                              "reasoning_content", "input_tokens", "output_tokens",
                                                              "total_tokens", "wall_s")}}
        # 인프라 오류(재시도 가능)를 먼저, 그다음 재시도하지 않는 API 오류, 마지막으로 빈 응답·잘림(인프라)을 본다.
        if not result.get("infra_error") and result.get("error_type"):
            call["status"] = "call_error"
            record["calls"].append(call)
            record.update(error_type=result["error_type"], status_code=result.get("status_code"))
            return record
        if result.get("infra_error") or not (result.get("content") or "").strip() or result.get("finish_reason") == "length":
            call["status"] = "infra_error"
            record["calls"].append(call)
            record.update(infra_error=result.get("infra_error") or "empty_or_truncated_response", retryable=True)
            return record
        verdict = validate(kind, result["content"], packet)
        call.update(status=verdict["status"], codes=verdict["codes"], reasons=verdict["reasons"])
        record["calls"].append(call)
        if verdict["status"] == "valid" or step == "retry":
            record["final"] = {"status": verdict["status"], "step": step, "codes": verdict["codes"]}
            return record


def openai_complete(messages: list[dict], params: dict) -> dict:
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
    started = time.perf_counter()
    out: dict[str, Any] = {}
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"], timeout=300,
                        max_retries=0)
        response = client.chat.completions.create(messages=messages, **params)
        out.update(response_id=response.id, provider_model=response.model)
        if response.choices:
            choice = response.choices[0]
            out.update(content=choice.message.content, finish_reason=choice.finish_reason,
                       reasoning_content=getattr(choice.message, "reasoning_content", None))
        if response.usage is not None:
            out.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                       total_tokens=response.usage.total_tokens)
    except Exception as exc:
        # 예외 문자열에는 요청 헤더·자격증명이 섞일 수 있으므로 종류(와 상태 코드)만 보존한다.
        retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        out.update(error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
        if retryable:
            out["infra_error"] = type(exc).__name__
    out["wall_s"] = time.perf_counter() - started
    return out


def call(kind: str, packet_path: Path, output: Path, model: str) -> int:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started_at = datetime.now(timezone.utc).isoformat()
    wall = time.perf_counter()
    record = dialogue(kind, Path(packet_path).read_bytes(), model, openai_complete)
    exit_code = 2 if record.get("infra_error") or record.get("error_type") else 0
    record.update(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                  execution_wall_s=time.perf_counter() - wall, exit_code=exit_code)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="검증 피드백 1회 재요청 arm 호출 한 번(1차 + 최대 1회 재요청)을 기록한다")
    parser.add_argument("--kind", choices=("diagnosis", "correction"), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raise SystemExit(call(args.kind, args.packet, args.output, args.model))
