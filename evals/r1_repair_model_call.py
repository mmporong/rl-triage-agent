"""확장 검사 계약용 단일 피드백 경로. 기존 모델과 v4의 요청·피드백 규칙을 쓴다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feedback_model_call as F
import r1_repair_contract as C

SCHEMA = "r1_repair_feedback_call_v1"


def validate(kind: str, content, packet: dict) -> dict:
    if kind == "correction":
        return F.validate(kind, content, packet)
    try:
        decision = C.validate_diagnosis(json.loads(F.diagnosis_body(content)), packet)
    except (ValueError, TypeError) as error:
        reasons = [str(error)[:300]]
        return {"status": "invalid", "codes": F.codes_for(kind, reasons), "reasons": reasons}
    return {"status": "valid", "codes": [], "reasons": [], "decision": decision}


def messages_binding(kind: str, data: bytes) -> tuple[list, dict]:
    if kind == "correction":
        return F.first_messages(kind, data)
    if kind != "diagnosis":
        raise ValueError("등록되지 않은 모델 단계")
    messages = C.diagnosis_messages(json.loads(data))
    return messages, {"packet_sha256": hashlib.sha256(data).hexdigest(),
                      "messages_sha256": C.L.digest(messages)}


def dialogue(kind: str, data: bytes, complete=F.openai_complete) -> dict:
    messages, binding = messages_binding(kind, data)
    packet, params = json.loads(data), F.request_params(C.MODEL)
    record = {"schema": SCHEMA, "kind": kind, **binding, "request": params,
              "calls": [], "feedback": None, "final": None, "requests": 0}
    for step in ("first", "retry"):
        sent = messages
        if step == "retry":
            first = record["calls"][0]
            sent = F.retry_messages(messages, first["content"], kind, first["codes"])
            record["feedback"] = {"codes": first["codes"], "text": sent[-1]["content"],
                                  "messages_sha256": C.L.digest(sent)}
        response = complete(sent, params)
        record["requests"] += 1
        call = {"step": step, **{key: response.get(key) for key in (
            "response_id", "provider_model", "finish_reason", "content", "reasoning_content",
            "input_tokens", "output_tokens", "total_tokens", "wall_s")}}
        record["calls"].append(call)
        if response.get("error_type") and not response.get("infra_error"):
            call["status"] = "call_error"
            record["error_type"] = response["error_type"]
            break
        if response.get("infra_error") or not (response.get("content") or "").strip() or response.get("finish_reason") == "length":
            call["status"] = "infra_error"
            record["infra_error"] = response.get("infra_error") or "empty_or_truncated_response"
            break
        if response.get("provider_model") != C.MODEL:
            call["status"] = "call_error"
            record["error_type"] = "ProviderModelMismatch"
            break
        verdict = validate(kind, response["content"], packet)
        call.update({key: verdict[key] for key in ("status", "codes", "reasons")})
        if verdict["status"] == "valid" or step == "retry":
            record["final"] = {"status": verdict["status"], "step": step, "codes": verdict["codes"]}
            break
    return record


def review(record: dict, data: bytes, kind: str, reservation: dict | None = None) -> dict:
    """저장된 상태를 신뢰하지 않고 실제 응답·피드백·결합을 재계산한다."""
    messages, binding = messages_binding(kind, data)
    if reservation is not None:
        if (set(reservation) != {"kind", "packet_sha256", "at"} or reservation["kind"] != kind
                or reservation["packet_sha256"] != hashlib.sha256(data).hexdigest()
                or not isinstance(reservation["at"], str) or not reservation["at"]
                or record.get("reservation_sha256") != C.L.digest(reservation)):
            raise ValueError("모델 예약과 실제 입력 결합이 다르다")
    if (record.get("schema") != SCHEMA or record.get("kind") != kind
            or record.get("request") != F.request_params(C.MODEL)
            or any(record.get(key) != value for key, value in binding.items())):
        raise ValueError("모델 기록의 packet·메시지·요청 결합이 다르다")
    calls = record.get("calls")
    if not isinstance(calls, list) or len(calls) not in (1, 2) or record.get("requests") != len(calls):
        raise ValueError("모델 요청 수가 계약과 다르다")
    if [row.get("step") for row in calls] != ["first", "retry"][:len(calls)]:
        raise ValueError("모델 호출 순서가 다르다")
    packet = json.loads(data)
    first = None
    for index, call in enumerate(calls):
        if index:
            if first is None or first["status"] != "invalid":
                raise ValueError("무효 응답에만 피드백을 요청할 수 있다")
            sent = F.retry_messages(messages, calls[0]["content"], kind, first["codes"])
            expected = {"codes": first["codes"], "text": sent[-1]["content"], "messages_sha256": C.L.digest(sent)}
            if record.get("feedback") != expected:
                raise ValueError("피드백 결합이 다르다")
        if call.get("status") in ("infra_error", "call_error"):
            if index != len(calls) - 1 or record.get("final") is not None:
                raise ValueError("오류 뒤 추가 호출이나 최종 판정이 있다")
            key = "infra_error" if call["status"] == "infra_error" else "error_type"
            if not record.get(key):
                raise ValueError("오류 근거가 없다")
            return {"status": call["status"], "requests": len(calls), "reason": record[key]}
        if (call.get("provider_model") != C.MODEL or not isinstance(call.get("content"), str)
                or not call["content"].strip() or call.get("finish_reason") != "stop"):
            raise ValueError("응답 모델 또는 완료 상태가 다르다")
        verdict = validate(kind, call["content"], packet)
        if any(call.get(key) != verdict[key] for key in ("status", "codes", "reasons")):
            raise ValueError("응답 검증 재계산이 다르다")
        if not index:
            first = verdict
    if len(calls) == 1 and (first["status"] != "valid" or record.get("feedback") is not None):
        raise ValueError("첫 응답이 무효인데 피드백 재요청이 없다")
    expected_final = {"status": verdict["status"], "step": calls[-1]["step"], "codes": verdict["codes"]}
    if record.get("final") != expected_final:
        raise ValueError("최종 모델 판정이 다르다")
    return {**verdict, "requests": len(calls), "first_status": first["status"], "retried": len(calls) == 2}


def reservation_for(kind: str, data: bytes) -> dict:
    return {"kind": kind, "packet_sha256": hashlib.sha256(data).hexdigest(),
            "at": datetime.now(timezone.utc).isoformat()}


def call(kind: str, packet: Path, output: Path, pid_file: Path | None = None) -> int:
    # 예약 파일은 프로세스가 응답 전에 끝나도 같은 호출을 다시 시작하지 못하게 한다.
    output.parent.mkdir(parents=True, exist_ok=True)
    data = packet.read_bytes()
    reservation = reservation_for(kind, data)
    with output.with_suffix(".reserved.json").open("x", encoding="utf-8") as handle:
        json.dump(reservation, handle)
    if output.exists():
        raise FileExistsError(output.name)
    if pid_file is not None:
        if not Path("/proc/self/stat").is_file():
            raise ValueError("모델 worker PID 결합은 WSL에서만 쓴다")
        start_ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        with pid_file.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "start_ticks": start_ticks,
                       "packet_sha256": hashlib.sha256(data).hexdigest()}, handle)
    start = time.perf_counter()
    record = dialogue(kind, data)
    record["reservation_sha256"] = C.L.digest(reservation)
    record["execution_wall_s"] = time.perf_counter() - start
    record["exit_code"] = 2 if record.get("infra_error") or record.get("error_type") else 0
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return record["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("diagnosis", "correction"), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    raise SystemExit(call(args.kind, args.packet, args.output, args.pid_file))
