"""잔여 실패 v2의 선택 전용 모델 호출과 두 seed 공용 20회 요청 원장.

진단 모델은 probe와 fact id만 고른다. 기대값과 실행 인자는 계약 모듈과
controller가 생성하며, 이 모듈은 형식·의미 검증 결과를 한 번의 고정 피드백으로
되돌린다. 교정 호출은 동결된 v1 경로를 그대로 사용한다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import feedback_model_call as F  # noqa: E402
import residual_model_call_v1 as MC1  # noqa: E402
import residual_selection_v2 as C  # noqa: E402

SCHEMA = "residual_selection_feedback_v2"
MODEL = MC1.C.MODEL
TRANSPORT_TIMEOUT_S = MC1.TRANSPORT_TIMEOUT_S

# v1과 같은 단일 원장을 쓴다는 사실을 import 경계에서도 명시한다.
BudgetStop = MC1.BudgetStop
MeteredComplete = MC1.MeteredComplete
API_CAP = MC1.API_CAP
audit_budget = MC1.audit_budget
file_sha = MC1.file_sha
read = MC1.read
write_new = MC1.write_new
wsl_path = MC1.wsl_path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _packet_bytes(packet: dict) -> bytes:
    return (json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _messages_digest(messages: list[dict]) -> str:
    return MC1.C.L.digest(messages)


def _messages_binding(data: bytes) -> tuple[dict, list[dict], dict]:
    packet = json.loads(data)
    C.validate_packet(packet)
    messages = C.diagnosis_messages(packet)
    return packet, messages, {
        "packet_sha256": hashlib.sha256(data).hexdigest(),
        "messages_sha256": _messages_digest(messages),
    }


def _retry_messages(messages: list[dict], first_content: str, review: dict) -> list[dict]:
    feedback = C.feedback_message(review)
    if (not isinstance(feedback, dict) or set(feedback) != {"role", "content"}
            or feedback["role"] != "user" or not isinstance(feedback["content"], str)
            or not feedback["content"]):
        raise ValueError("선택 계약 피드백 메시지 형식이 다르다")
    return [*messages, {"role": "assistant", "content": first_content}, feedback]


def _call_fields(result: dict) -> dict:
    return {key: result.get(key) for key in (
        "response_id", "provider_model", "finish_reason", "content", "reasoning_content",
        "input_tokens", "output_tokens", "total_tokens", "wall_s")}


def _assessment_fields(verdict: dict) -> dict:
    required = {"status", "codes", "reasons", "format_status", "semantic_status"}
    if (not isinstance(verdict, dict) or not required.issubset(verdict)
            or verdict["status"] not in {"valid", "invalid"}
            or not isinstance(verdict["codes"], list)
            or not isinstance(verdict["reasons"], list)
            or any(not isinstance(value, str) or not value for value in verdict["codes"])
            or any(not isinstance(value, str) or not value for value in verdict["reasons"])):
        raise ValueError("선택 계약 검증 결과 형식이 다르다")
    fields = {key: verdict[key] for key in required}
    return fields


def dialogue(data: bytes, complete: Callable[[list[dict], dict], dict] = F.openai_complete) -> dict:
    """진단 1차 호출과 계약 피드백 재요청 한 번을 기록한다."""
    packet, messages, binding = _messages_binding(data)
    params = F.request_params(MODEL)
    record = {"schema": SCHEMA, "kind": "diagnosis", **binding, "request": params,
              "calls": [], "feedback": None, "final": None, "requests": 0}
    for step in ("first", "retry"):
        sent = messages
        if step == "retry":
            first = record["calls"][0]
            first_review = {key: first[key] for key in
                            ("status", "codes", "reasons", "format_status", "semantic_status")}
            sent = _retry_messages(messages, first["content"], first_review)
            text = sent[-1]["content"]
            record["feedback"] = {
                "codes": first["codes"], "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "messages_sha256": _messages_digest(sent),
            }
        result = complete(sent, params)
        record["requests"] += 1
        call = {"step": step, **_call_fields(result)}
        if result.get("error_type") and not result.get("infra_error"):
            call["status"] = "call_error"
            record["calls"].append(call)
            record["error_type"] = result["error_type"]
            record["status_code"] = result.get("status_code")
            break
        if (result.get("infra_error") or not (result.get("content") or "").strip()
                or result.get("finish_reason") == "length"):
            call["status"] = "infra_error"
            record["calls"].append(call)
            record["infra_error"] = result.get("infra_error") or "empty_or_truncated_response"
            break
        if result.get("provider_model") != MODEL:
            call["status"] = "call_error"
            record["calls"].append(call)
            record["error_type"] = "ProviderModelMismatch"
            break
        verdict = C.assess(result["content"], packet)
        call.update(_assessment_fields(verdict))
        record["calls"].append(call)
        if verdict["status"] == "valid" or step == "retry":
            record["final"] = {
                "status": verdict["status"], "step": step, "codes": verdict["codes"],
                "format_status": verdict["format_status"],
                "semantic_status": verdict["semantic_status"],
            }
            break
    return record


def review(record: dict, data: bytes) -> dict:
    """저장값을 신뢰하지 않고 진단 대화의 입력·피드백·검증을 다시 계산한다."""
    packet, messages, binding = _messages_binding(data)
    if (record.get("schema") != SCHEMA or record.get("kind") != "diagnosis"
            or record.get("request") != F.request_params(MODEL)
            or any(record.get(key) != value for key, value in binding.items())):
        raise ValueError("선택 모델 기록의 packet·메시지·요청 결합이 다르다")
    calls = record.get("calls")
    if (not isinstance(calls, list) or len(calls) not in (1, 2)
            or record.get("requests") != len(calls)
            or [row.get("step") for row in calls] != ["first", "retry"][:len(calls)]):
        raise ValueError("선택 모델 호출 수나 순서가 다르다")
    first_verdict = None
    verdict = None
    for index, call in enumerate(calls):
        if index:
            if first_verdict is None or first_verdict["status"] != "invalid":
                raise ValueError("무효 선택 응답에만 피드백을 요청할 수 있다")
            sent = _retry_messages(messages, calls[0]["content"], first_verdict)
            text = sent[-1]["content"]
            expected_feedback = {
                "codes": first_verdict["codes"], "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "messages_sha256": _messages_digest(sent),
            }
            if record.get("feedback") != expected_feedback:
                raise ValueError("선택 모델 피드백 결합이 다르다")
        if call.get("status") in {"infra_error", "call_error"}:
            if index != len(calls) - 1 or record.get("final") is not None:
                raise ValueError("선택 모델 오류 뒤 추가 호출이나 최종 판정이 있다")
            key = "infra_error" if call["status"] == "infra_error" else "error_type"
            if not record.get(key):
                raise ValueError("선택 모델 오류 근거가 없다")
            return {"status": call["status"], "requests": len(calls),
                    "reason": record[key]}
        if (call.get("provider_model") != MODEL or not isinstance(call.get("content"), str)
                or not call["content"].strip() or call.get("finish_reason") != "stop"):
            raise ValueError("선택 응답 모델 또는 완료 상태가 다르다")
        verdict = C.assess(call["content"], packet)
        expected = _assessment_fields(verdict)
        if any(call.get(key) != value for key, value in expected.items()):
            raise ValueError("선택 응답 검증 재계산이 다르다")
        if index == 0:
            first_verdict = expected
    assert first_verdict is not None and verdict is not None
    if len(calls) == 1 and (first_verdict["status"] != "valid" or record.get("feedback") is not None):
        raise ValueError("첫 선택 응답이 무효인데 피드백 재요청이 없다")
    expected_final = {
        "status": verdict["status"], "step": calls[-1]["step"], "codes": verdict["codes"],
        "format_status": verdict["format_status"], "semantic_status": verdict["semantic_status"],
    }
    if record.get("final") != expected_final:
        raise ValueError("최종 선택 모델 판정이 다르다")
    return {**verdict, "requests": len(calls), "first_status": first_verdict["status"],
            "retried": len(calls) == 2}


def _sent_messages(record: dict, data: bytes, kind: str) -> list[list[dict]]:
    """각 API 요청에 실제로 전달됐어야 하는 메시지를 고정 계약에서 재생한다."""
    if kind == "diagnosis":
        packet = json.loads(data)
        C.validate_packet(packet)
        first = C.diagnosis_messages(packet)
        messages = [first]
        if len(record["calls"]) == 2:
            verdict = C.assess(record["calls"][0]["content"], packet)
            messages.append(_retry_messages(first, record["calls"][0]["content"], verdict))
        return messages
    if kind == "correction":
        first, _binding = F.first_messages(kind, data)
        messages = [first]
        if len(record["calls"]) == 2:
            messages.append(F.retry_messages(first, record["calls"][0]["content"], kind,
                                              record["calls"][0]["codes"]))
        return messages
    raise ValueError("등록되지 않은 모델 단계")


def _review_record(record: dict, data: bytes, kind: str, api_root: Path, *,
                   record_path: Path) -> dict:
    """의미 판정 뒤 record 파일·요청 입력·응답 receipt를 함께 재검증한다."""
    if kind == "diagnosis":
        verdict = review(record, data)
    elif kind == "correction":
        verdict = MC1._review_record(record, data, kind, api_root)
    else:
        raise ValueError("등록되지 않은 모델 단계")
    record_path = Path(record_path)
    if not record_path.is_file() or read(record_path) != record:
        raise ValueError("선택 모델 record 경로와 실제 파일 내용이 다르다")
    rows, calls = record.get("request_receipts"), record.get("calls")
    if not isinstance(rows, list) or not isinstance(calls, list) or len(rows) != len(calls):
        raise ValueError("선택 모델 호출과 API receipt 수가 다르다")
    sent_messages = _sent_messages(record, data, kind)
    if len(sent_messages) != len(calls):
        raise ValueError("선택 모델 호출과 전송 메시지 수가 다르다")
    packet_sha = hashlib.sha256(data).hexdigest()
    output_prefix = f"{MC1._output_scope(record_path.parent)}/{record_path.stem}"
    audit = audit_budget(api_root)
    by_index = {row["reservation"]["request_index"]: row for row in audit["rows"]}
    request_indices = []
    for local_index, (binding, call, messages) in enumerate(zip(rows, calls, sent_messages), 1):
        if set(binding) != {"request_index", "reservation_sha256", "response_sha256"}:
            raise ValueError("선택 모델 기록의 API receipt 필드가 다르다")
        request_index = binding["request_index"]
        if (type(request_index) is not int or request_index in request_indices
                or (request_indices and request_index <= request_indices[-1])):
            raise ValueError("선택 모델 API receipt 요청 순서가 다르다")
        request_indices.append(request_index)
        row = by_index.get(request_index)
        if row is None or row["response"] is None:
            raise ValueError("선택 모델 기록의 API receipt가 완료되지 않았다")
        reservation, response = row["reservation"], row["response"]
        expected_reservation = {
            "kind": kind,
            "packet_sha256": packet_sha,
            "messages_sha256": _messages_digest(messages),
            "output_key": f"{output_prefix}/request{local_index}",
        }
        if any(reservation.get(key) != value for key, value in expected_reservation.items()):
            raise ValueError("API 예약이 record의 kind·packet·메시지·출력 경로와 다르다")
        response_path = Path(api_root) / "responses" / f"{request_index:04d}.json"
        if (binding["reservation_sha256"] != MC1.digest(reservation)
                or binding["response_sha256"] != file_sha(response_path)):
            raise ValueError("선택 모델 기록의 API receipt 해시 결합이 다르다")
        raw = response["result"]
        if any(call.get(key) != raw.get(key) for key in (
                "response_id", "provider_model", "finish_reason", "content", "reasoning_content",
                "input_tokens", "output_tokens", "total_tokens", "wall_s")):
            raise ValueError("저장 응답과 선택 모델 기록의 비용·결과가 다르다")
    if verdict["status"] == "valid" and audit["pending"]:
        raise ValueError("미완료 예약이 남은 원장에서는 성공을 확정하지 않는다")
    return verdict


def _output_scope(folder: Path) -> str:
    parts = Path(folder).parts[-3:]
    if not parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("모델 출력 경로의 portable 식별자가 잘못됐다")
    return "/".join(parts)


def _stop_worker(pid_file: Path, packet_path: Path) -> str:
    if not pid_file.is_file():
        return "unverified_no_pid"
    cleanup = r'''import hashlib,json,os,signal,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text()); packet=Path(sys.argv[2]); pid=r['pid']; proc=Path('/proc')/str(pid)
assert type(pid) is int and pid>1 and hashlib.sha256(packet.read_bytes()).hexdigest()==r['packet_sha256']
def same():
 if not (proc/'stat').exists(): return False
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
 if int(fields[19])!=r['start_ticks']: raise RuntimeError('pid identity changed')
 args=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 script=next((i for i,x in enumerate(args) if x.endswith('/residual_model_call_v2.py')),None)
 if script is None or args[script:]!=r['argv'] or str(packet) not in args: raise RuntimeError('worker args differ')
 return True
if same():
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if same(): os.kill(pid,signal.SIGKILL)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if proc.exists(): raise RuntimeError('worker remains')
'''
    command = '"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join(
        [cleanup, wsl_path(pid_file), wsl_path(packet_path)])
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], check=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    return "verified"


def _run_worker(packet_path: Path, output: Path, api_root: Path) -> None:
    pid_file = output.with_suffix(".pid.json")
    argv = [wsl_path(ROOT / "evals/residual_model_call_v2.py"), "--kind", "diagnosis",
            "--packet", wsl_path(packet_path), "--output", wsl_path(output),
            "--api-root", wsl_path(api_root), "--pid-file", wsl_path(pid_file)]
    command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(argv)
    try:
        process = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=TRANSPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        receipt = {"schema": "residual_selection_transport_error_v2", "type": "dialogue_timeout",
                   "worker_cleanup": "unverified", "at": now()}
        try:
            receipt["worker_cleanup"] = _stop_worker(pid_file, packet_path)
        except Exception as error:
            receipt["cleanup_error_type"] = type(error).__name__
        write_new(output.with_suffix(".transport_error.json"), receipt)
        raise TimeoutError("선택 모델 대화 시간 상한 초과") from None
    if not output.is_file():
        raise RuntimeError(f"선택 모델 호출 기록 없음(exit={process.returncode})")
    if pid_file.exists():
        raise RuntimeError("자연 종료한 선택 모델 worker PID 파일이 남았다")


def decide(folder: Path, kind: str, packet: dict, api_root: Path,
           caller: Callable[[list[dict], dict], dict] | None = None) -> dict:
    """인프라 오류만 두 번째 대화를 허용한다. 교정은 v1 호출기에 위임한다."""
    if kind == "correction":
        folder = Path(folder)
        result = MC1.decide(folder, kind, packet, api_root, caller=caller)
        data = (folder / "correction.packet.json").read_bytes()
        for attempt in result["attempts"]:
            record_path = folder / attempt["record"]
            record = read(record_path)
            verdict = _review_record(record, data, kind, api_root, record_path=record_path)
            if (attempt.get("record_sha256") != file_sha(record_path)
                    or attempt.get("request_receipts") != record.get("request_receipts")
                    or attempt.get("review") != verdict):
                raise ValueError("v1 교정 결과와 v2 receipt 재검증 결과가 다르다")
        return result
    if kind != "diagnosis":
        raise ValueError("등록되지 않은 모델 단계")
    C.validate_packet(packet)
    folder, api_root = Path(folder), Path(api_root)
    folder.mkdir(parents=True, exist_ok=True)
    packet_path = folder / "diagnosis.packet.json"
    write_new(packet_path, packet)
    frozen = packet_path.read_bytes()
    if frozen != _packet_bytes(packet):
        raise ValueError("선택 packet 직렬화가 고정 형식과 다르다")
    attempts = []
    for attempt in (1, 2):
        if packet_path.read_bytes() != frozen:
            raise ValueError("호출 사이에 선택 packet이 바뀌었다")
        output = folder / f"diagnosis.dialogue{attempt}.json"
        if caller is None:
            _run_worker(packet_path, output, api_root)
        else:
            metered = MeteredComplete(api_root, "diagnosis", file_sha(packet_path),
                                      f"{_output_scope(folder)}/diagnosis.dialogue{attempt}", caller)
            record = dialogue(frozen, complete=metered)
            record["request_receipts"] = metered.receipts
            write_new(output, record)
        if packet_path.read_bytes() != frozen:
            raise ValueError("선택 모델 호출 중 packet이 바뀌었다")
        record = read(output)
        verdict = _review_record(record, frozen, kind, api_root, record_path=output)
        row = {"record": output.name, "record_sha256": file_sha(output), "review": verdict,
               "request_receipts": record["request_receipts"]}
        write_new(folder / f"diagnosis.review{attempt}.json", row)
        attempts.append(row)
        if verdict["status"] != "infra_error":
            return {"review": verdict, "attempts": attempts}
    return {"review": verdict, "attempts": attempts}


def _worker_identity(packet: Path) -> dict:
    raw = Path("/proc/self/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": os.getpid(), "start_ticks": int(fields[19]), "argv": sys.argv,
            "packet_sha256": file_sha(packet)}


def worker(packet: Path, output: Path, api_root: Path, pid_file: Path) -> int:
    if os.name != "posix":
        raise RuntimeError("선택 모델 worker는 WSL 자격증명 환경에서만 실행한다")
    if output.exists():
        raise FileExistsError(output)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    write_new(pid_file, _worker_identity(packet))
    try:
        data = packet.read_bytes()
        metered = MeteredComplete(api_root, "diagnosis", file_sha(packet),
                                  f"{_output_scope(output.parent)}/{output.stem}", F.openai_complete)
        record = dialogue(data, complete=metered)
        record["request_receipts"] = metered.receipts
        record["worker_identity"] = read(pid_file)
        write_new(output, record)
        return 2 if record.get("infra_error") or record.get("error_type") else 0
    finally:
        pid_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("diagnosis", "correction"), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.kind != "diagnosis":
        raise ValueError("WSL v2 worker는 diagnosis만 실행한다")
    return worker(args.packet, args.output, args.api_root, args.pid_file)


if __name__ == "__main__":
    raise SystemExit(main())
