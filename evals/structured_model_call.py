"""구조화 출력 arm: 진단·교정 단일 호출과 같은 요청에 response_format json_schema(strict) 하나만 더해 한 번 부른다.

evals/active_model_call.py(진단)·evals/correction_model_call.py(교정)는 바꾸지 않는다. 메시지는 같은 함수
(active_diagnosis.decision_messages, correction_model_call.build_messages)로 만들고, 모델·temperature 0·
reasoning_effort none·max_tokens 16384도 같다. 바꾸는 변수는 출력 형식 하나다.

schema는 구조만 정한다: 계약의 필드 이름·필수 여부·형(널 허용 포함)·추가 필드 금지. 값 목록(enum)·최소 개수·
packet에 따라 달라지는 규칙(1위 순열, 카탈로그 기대 결과, 사실 id·값, 등록 교정)은 넣지 않는다. 그 의미 위반은 기존
계약 검증기가 그대로 센다. 기록은 기존 검토 함수(e2e_reward_demo_v2.review_diagnosis_record,
correction_model_call.review_call_record)가 읽는 필드를 같은 이름으로 남긴다.

  python evals/structured_model_call.py --kind diagnosis|correction --packet <packet.json> --output <run.json> --model <모델>
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

RECORD_SCHEMA = "structured_model_call_v1"
NULLABLE_STRING = {"type": ["string", "null"]}
SCHEMAS = {
    "diagnosis": {"type": "object", "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}},
        "probe": NULLABLE_STRING,
        "action": {"type": "string"},
        "reason": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "value": {}},
                                              "required": ["id", "value"], "additionalProperties": False}},
        "expected_outcomes": {"type": "object", "additionalProperties": NULLABLE_STRING}},
        "required": ["ranking", "probe", "action", "reason", "claims", "expected_outcomes"], "additionalProperties": False},
    "correction": {"type": "object", "properties": {
        "correction_id": NULLABLE_STRING,
        "recovery_mode": NULLABLE_STRING,
        "cited_facts": {"type": "array", "items": {"type": "string"}},
        "checklist": {"type": "object", "additionalProperties": {"type": "string"}},
        "hold_reason": NULLABLE_STRING},
        "required": ["correction_id", "recovery_mode", "cited_facts", "checklist", "hold_reason"], "additionalProperties": False},
}
NAMES = {"diagnosis": "diagnosis_decision", "correction": "correction_proposal"}


def response_format(kind: str) -> dict:
    return {"type": "json_schema", "json_schema": {"name": NAMES[kind], "schema": SCHEMAS[kind], "strict": True}}


def response_format_sha256(kind: str) -> str:
    return hashlib.sha256(json.dumps(response_format(kind), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def build(kind: str, data: bytes) -> tuple[list[dict], dict]:
    """(메시지, 기존 검토 함수가 대조하는 결합 필드). 메시지는 기존 단일 호출과 같은 함수로 만든다."""
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


def call(kind: str, packet_path: Path, output: Path, model: str) -> int:
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError

    output = Path(output)
    if output.exists():  # 호출한 뒤 기록을 못 쓰는 일이 없도록 먼저 막는다
        raise FileExistsError(output)
    messages, binding = build(kind, Path(packet_path).read_bytes())
    params: dict[str, Any] = {"model": model, "temperature": 0, "max_tokens": 16384, "stream": False,
                              "extra_body": {"reasoning_effort": "none"}, "response_format": response_format(kind)}
    record = {"schema": RECORD_SCHEMA, "kind": kind, **binding, "response_format_sha256": response_format_sha256(kind),
              "started_at": datetime.now(timezone.utc).isoformat(), "request": params, "response_id": None,
              "provider_model": None, "finish_reason": None, "content": None, "reasoning_content": None,
              "input_tokens": None, "output_tokens": None, "total_tokens": None}
    wall, cpu = time.perf_counter(), time.process_time()
    exit_code = 0
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"], timeout=300,
                        max_retries=0)
        response = client.chat.completions.create(messages=messages, **params)
        record["response_id"] = response.id
        record["provider_model"] = response.model
        if response.choices:
            choice = response.choices[0]
            record.update(content=choice.message.content, finish_reason=choice.finish_reason,
                          reasoning_content=getattr(choice.message, "reasoning_content", None))
        if response.usage is not None:
            record.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens)
        if not (record.get("content") or "").strip() or record["finish_reason"] == "length":
            record["infra_error"] = "empty_or_truncated_response"
            record["retryable"] = True
            exit_code = 2
    except Exception as exc:
        # 예외 문자열에는 요청 헤더·자격증명이 섞일 수 있으므로 종류(와 상태 코드)만 보존한다.
        retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        record["error_type"] = type(exc).__name__
        record["status_code"] = getattr(exc, "status_code", None)
        record["retryable"] = retryable
        if retryable:
            record["infra_error"] = type(exc).__name__
        exit_code = 2
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), execution_wall_s=time.perf_counter() - wall,
                  cpu_s=time.process_time() - cpu, exit_code=exit_code)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="구조화 출력 arm 단일 호출 한 번을 기록한다")
    parser.add_argument("--kind", choices=tuple(SCHEMAS), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raise SystemExit(call(args.kind, args.packet, args.output, args.model))
