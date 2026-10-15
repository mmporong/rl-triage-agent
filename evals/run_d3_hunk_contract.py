"""저장된 D3 packet에서 기존 줄 번호 계약과 새 hunk 인용 계약을 3회씩 비교한다.

API 호출은 메인이 새 코드와 프로토콜을 커밋한 뒤 WSL에서 실행한다. 기본 48회와 인프라 재시도
공용 상한 12회를 합쳐 실제 API 요청은 최대 60회다. 의미 오류는 재호출하지 않는다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import d3_agent as OLD  # noqa: E402
import d3_hunk_contract as NEW  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/d3_hunk_contract_v1.json"
SOURCE = ROOT / "bench/private/d3_experiment/ced4302df56bfc4e"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task4"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
REQUEST = {"model": MODEL, "temperature": 0, "max_tokens": 16384, "stream": False,
           "extra_body": {"reasoning_effort": "none"}}
BASE_CALLS = 48
INFRA_RETRY_CAP = 12
TOTAL_API_CAP = 60
CODE = {"evals/d3_agent.py", "evals/d3_hunk_contract.py", "evals/run_d3_hunk_contract.py",
        "bench/protocols/d3_hunk_contract_v1.json"}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def portable_path(path: Path) -> str:
    """원장에는 사용자별 절대경로를 남기지 않는다."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.relative_to(PRIVATE.parent).as_posix()


def packet_paths() -> list[Path]:
    paths = sorted(SOURCE.glob("*/model/B[45].packet.json"), key=lambda p: p.as_posix())
    if len(paths) != 8:
        raise ValueError(f"등록 packet은 8개여야 한다: {len(paths)}")
    return paths


def packet_id(path: Path) -> str:
    return f"{path.parents[1].name}_{path.stem.split('.')[0]}"


def call_order() -> list[dict]:
    rows = []
    for repeat in range(1, 4):
        block = [{"packet_id": packet_id(path), "arm": arm, "repeat": repeat}
                 for path in packet_paths() for arm in ("old", "hunk")]
        block.sort(key=lambda row: hashlib.sha256(
            f"d3_hunk_contract_v1:{repeat}:{row['packet_id']}:{row['arm']}".encode()).hexdigest())
        rows.extend(block)
    return [{"call": index + 1, **row} for index, row in enumerate(rows)]


def read_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def verify_protocol(protocol: dict) -> None:
    if protocol.get("schema") != "d3_hunk_contract_protocol_v1":
        raise ValueError("프로토콜 schema가 다르다")
    expected = {packet_id(path): sha(path) for path in packet_paths()}
    if (protocol.get("packets") != expected
            or protocol.get("call_order_sha256") != canonical_sha(call_order())):
        raise ValueError("packet 결합이나 호출 순서가 사전등록과 다르다")
    if protocol.get("request") != REQUEST or protocol.get("base_calls") != BASE_CALLS:
        raise ValueError("요청 설정이나 기본 호출 수가 다르다")
    if (protocol.get("infra_retry_cap"), protocol.get("total_api_cap"), protocol.get("semantic_retry_cap")) != (12, 60, 0):
        raise ValueError("재시도 상한이 다르다")


def verify_committed() -> str:
    for relative in sorted(CODE):
        subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative],
                       check=True, capture_output=True)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("저장소 밖의 소스 경로다")
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"], check=True,
                                   capture_output=True).stdout
        if lf_bytes(committed) != lf_bytes(path.read_bytes()):
            raise ValueError("API 실행 전 소스와 프로토콜을 커밋해야 한다")
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def messages(arm: str, packet: dict) -> list[dict]:
    return OLD.messages(packet) if arm == "old" else NEW.messages(packet)


def record_path(entry: dict, attempt: int) -> Path:
    return PRIVATE / "calls" / f"c{entry['call']:02d}_{entry['packet_id']}_{entry['arm']}_r{entry['repeat']}_a{attempt}.json"


def reservation_path(entry: dict, attempt: int) -> Path:
    return PRIVATE / "reservations" / f"c{entry['call']:02d}_{entry['packet_id']}_{entry['arm']}_r{entry['repeat']}_a{attempt}.json"


def lock_path() -> Path:
    return PRIVATE / "api_call.lock.json"


def _write_new_durable(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def acquire_lock(entry: dict, attempt: int) -> dict:
    lock = {"schema": "d3_hunk_contract_call_lock_v1", "entry": deepcopy(entry), "attempt": attempt,
            "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        _write_new_durable(lock_path(), lock)
    except FileExistsError as error:
        raise RuntimeError("API 호출 lock이 남아 있다. 상태를 확인하고 resolve-lock 절차를 사용한다") from error
    return lock


def resolve_lock(expected_sha256: str, note: str) -> dict:
    """운영자가 호출 프로세스가 없음을 확인한 뒤 남은 lock만 해제한다."""
    path = lock_path()
    if not path.is_file() or sha(path) != expected_sha256:
        raise ValueError("확인한 lock SHA와 현재 파일이 다르다")
    if not note.strip():
        raise ValueError("프로세스 부재 확인 기록이 필요하다")
    lock = json.loads(path.read_text(encoding="utf-8"))
    resolution = PRIVATE / "lock_resolutions" / f"{expected_sha256}.json"
    _write_new_durable(resolution, {"schema": "d3_hunk_contract_lock_resolution_v1",
                                    "lock_sha256": expected_sha256, "lock": lock,
                                    "operator_note": note.strip(),
                                    "resolved_at": datetime.now(timezone.utc).isoformat()})
    path.unlink()
    return {"status": "resolved", "resolution": portable_path(resolution)}


def _is_infra(error: BaseException) -> bool:
    names = {cls.__name__ for cls in type(error).__mro__}
    status = getattr(error, "status_code", None)
    return bool(names & {"APIConnectionError", "APITimeoutError"}) or (
        isinstance(status, int) and (status == 429 or status >= 500))


def _append_ledger(row: dict) -> None:
    path = PRIVATE / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        previous = json.loads(lines[-1])["event_sha256"] if lines else None
    event = {**row, "previous_event_sha256": previous}
    event["event_sha256"] = canonical_sha(event)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_ledger() -> list[dict]:
    path = PRIVATE / "ledger.jsonl"
    if not path.exists():
        return []
    rows, previous, seen = [], None, set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        recorded_hash = row.pop("event_sha256", None)
        if row.get("previous_event_sha256") != previous or canonical_sha(row) != recorded_hash:
            raise ValueError(f"ledger hash-chain이 다르다: {number}")
        key = (row.get("event"), row.get("call"), row.get("attempt"))
        if key in seen:
            raise ValueError("ledger에 중복 사건이 있다")
        if row.get("event") == "completed" and ("reserved", row.get("call"), row.get("attempt")) not in seen:
            raise ValueError("ledger 완료 사건보다 예약 사건이 먼저 있어야 한다")
        seen.add(key)
        field = "reservation" if row.get("event") == "reserved" else "record"
        hash_field = f"{field}_sha256"
        if row.get("event") not in {"reserved", "completed"} or not isinstance(row.get(field), str):
            raise ValueError("ledger 사건 형식이 다르다")
        artifact = ROOT / row[field]
        if not artifact.exists():
            artifact = PRIVATE.parent / row[field]
        if not artifact.is_file() or sha(artifact) != row.get(hash_field):
            raise ValueError("ledger가 묶은 예약·호출 기록이 없거나 바뀌었다")
        rows.append({**row, "event_sha256": recorded_hash})
        previous = recorded_hash
    return rows


def _used_attempts() -> int:
    return len(list((PRIVATE / "reservations").glob("*.json"))) if (PRIVATE / "reservations").is_dir() else 0


def call(entry: dict, attempt: int, output: Path, *, client_factory: Callable[[], Any] | None = None,
         require_posix: bool = True) -> int:
    if require_posix and os.name != "posix":
        raise RuntimeError("호출은 WSL 자격증명 환경에서 실행한다")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY가 없다")
    protocol = read_protocol()
    verify_protocol(protocol)
    commit = verify_committed()
    if entry not in call_order() or attempt not in (1, 2):
        raise ValueError("등록하지 않은 호출이다")
    if output != record_path(entry, attempt) or output.exists() or reservation_path(entry, attempt).exists():
        raise FileExistsError("호출 기록·예약 경로가 다르거나 이미 존재한다")
    acquire_lock(entry, attempt)
    owned_lock_sha = sha(lock_path())
    try:
        verify_ledger()
        if _used_attempts() >= TOTAL_API_CAP:
            raise ValueError("API 요청 상한 60회에 도달했다")
        if attempt == 2:
            first = json.loads(record_path(entry, 1).read_text(encoding="utf-8"))
            retries = len(list((PRIVATE / "reservations").glob("*_a2.json")))
            if first.get("status") != "infra_error" or retries >= INFRA_RETRY_CAP:
                raise ValueError("인프라 오류 재시도 조건이나 공용 상한을 만족하지 않는다")
        packet_path = next(path for path in packet_paths() if packet_id(path) == entry["packet_id"])
        packet = json.loads(packet_path.read_bytes())
        sent = messages(entry["arm"], packet)
        reservation = {"schema": "d3_hunk_contract_reservation_v1", "entry": deepcopy(entry), "attempt": attempt,
                       "project_commit": commit, "protocol_sha256": sha(PROTOCOL),
                       "packet_sha256": sha(packet_path), "messages_sha256": canonical_sha(sent),
                       "request": deepcopy(REQUEST), "reserved_at": datetime.now(timezone.utc).isoformat()}
        reserved = reservation_path(entry, attempt)
        _write_new_durable(reserved, reservation)
        _append_ledger({"schema": "d3_hunk_contract_ledger_v1", "event": "reserved",
                        "call": entry["call"], "attempt": attempt, "reservation": portable_path(reserved),
                        "reservation_sha256": sha(reserved)})
        record = {**reservation, "schema": "d3_hunk_contract_call_v1",
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": None, "content": None,
                  "response_id": None, "provider_model": None, "finish_reason": None, "input_tokens": None,
                  "output_tokens": None, "total_tokens": None, "retryable": False}
        wall = time.perf_counter()
        try:
            if client_factory is None:
                from openai import OpenAI
                client_factory = lambda: OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                                                 api_key=os.environ["NVIDIA_API_KEY"], timeout=300, max_retries=0)
            response = client_factory().chat.completions.create(messages=sent, **REQUEST)
            record.update(response_id=response.id, provider_model=response.model)
            if response.choices:
                choice = response.choices[0]
                record.update(content=choice.message.content, finish_reason=choice.finish_reason)
            if response.usage is not None:
                record.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                              total_tokens=response.usage.total_tokens)
            if not (record["content"] or "").strip() or record["finish_reason"] == "length":
                record.update(status="infra_error", infra_error="empty_or_truncated_response", retryable=True)
            else:
                record["status"] = "answered"
        except Exception as error:  # 예외 문자열은 자격증명 노출 위험 때문에 기록하지 않는다.
            retryable = _is_infra(error)
            record.update(status="infra_error" if retryable else "call_error", retryable=retryable,
                          infra_error=type(error).__name__ if retryable else None, error_type=type(error).__name__)
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), execution_wall_s=time.perf_counter() - wall)
        _write_new_durable(output, record)
        _append_ledger({"schema": "d3_hunk_contract_ledger_v1", "event": "completed",
                        "call": entry["call"], "attempt": attempt, "record": portable_path(output),
                        "record_sha256": sha(output), "status": record["status"]})
        return 0 if record["status"] == "answered" else 2
    finally:
        if lock_path().exists() and sha(lock_path()) == owned_lock_sha:
            lock_path().unlink()


def review(entry: dict, record: dict) -> dict:
    packet_path = next(path for path in packet_paths() if packet_id(path) == entry["packet_id"])
    packet = json.loads(packet_path.read_bytes())
    expected_messages = messages(entry["arm"], packet)
    if (record.get("entry") != entry or record.get("packet_sha256") != sha(packet_path)
            or record.get("protocol_sha256") != sha(PROTOCOL) or record.get("request") != REQUEST
            or record.get("messages_sha256") != canonical_sha(expected_messages)):
        raise ValueError("호출 기록 결합이 다르다")
    if record.get("status") != "answered":
        return {"status": record.get("status"), "problems": [record.get("infra_error") or record.get("error_type")]}
    output = OLD.parse(record.get("content"))
    problems = OLD.validate(output, packet) if entry["arm"] == "old" else NEW.validate(output, packet)
    return {"status": "invalid" if problems else "valid", "problems": problems}


def plan() -> dict:
    protocol = read_protocol()
    verify_protocol(protocol)
    commit = verify_committed()
    if lock_path().exists():
        return {"status": "locked", "lock_sha256": sha(lock_path()),
                "instruction": "호출 프로세스가 없음을 확인한 뒤 resolve-lock을 실행한다", "commands": []}
    completed = {(row["call"], row["attempt"]) for row in verify_ledger() if row.get("event") == "completed"}
    commands = []
    orphaned = []
    retry_count = len(list((PRIVATE / "reservations").glob("*_a2.json"))) if (PRIVATE / "reservations").is_dir() else 0
    for entry in call_order():
        first_reservation = reservation_path(entry, 1)
        first = record_path(entry, 1)
        if first.exists() and not first_reservation.exists():
            raise ValueError("예약 없이 호출 기록이 존재한다")
        if first_reservation.exists() and (not first.exists() or (entry["call"], 1) not in completed):
            orphaned.append({"call": entry["call"], "attempt": 1,
                             "reservation": portable_path(first_reservation)})
            continue
        if not first_reservation.exists():
            attempt = 1
        else:
            record = json.loads(first.read_text(encoding="utf-8"))
            second_reservation, second = reservation_path(entry, 2), record_path(entry, 2)
            if second.exists() and not second_reservation.exists():
                raise ValueError("재시도 예약 없이 호출 기록이 존재한다")
            if second_reservation.exists() and (not second.exists() or (entry["call"], 2) not in completed):
                orphaned.append({"call": entry["call"], "attempt": 2,
                                 "reservation": portable_path(second_reservation)})
                continue
            if record.get("status") == "infra_error" and not second_reservation.exists() and retry_count < INFRA_RETRY_CAP:
                attempt, retry_count = 2, retry_count + 1
            else:
                continue
        relative = portable_path(record_path(entry, attempt))
        commands.append('. "$HOME/.config/nvidia/env"; ~/.venvs/rl-triage/bin/python '
                        f"evals/run_d3_hunk_contract.py call --call {entry['call']} --attempt {attempt} --output '{relative}'")
    return {"status": "ready" if not orphaned else "orphaned_reservations", "project_commit": commit,
            "planned": BASE_CALLS, "remaining_commands": commands, "orphaned_reservations": orphaned,
            "used_api_attempts": _used_attempts(), "total_api_cap": TOTAL_API_CAP}


def summarize(output: Path) -> dict:
    protocol = read_protocol()
    verify_protocol(protocol)
    if lock_path().exists():
        return {"status": "locked", "lock_sha256": sha(lock_path())}
    completed = {(row["call"], row["attempt"]) for row in verify_ledger() if row.get("event") == "completed"}
    rows, counts = [], {"old": {"valid": 0, "invalid": 0}, "hunk": {"valid": 0, "invalid": 0}}
    retry_pool_used = len(list((PRIVATE / "reservations").glob("*_a2.json"))) if (PRIVATE / "reservations").is_dir() else 0
    incomplete = []
    for entry in call_order():
        attempts = [(2, reservation_path(entry, 2), record_path(entry, 2)),
                    (1, reservation_path(entry, 1), record_path(entry, 1))]
        selected = next(((attempt, reservation, record) for attempt, reservation, record in attempts
                         if reservation.exists()), None)
        if selected is None:
            incomplete.append(entry["call"])
            continue
        attempt, reservation, path = selected
        if not path.exists() or (entry["call"], attempt) not in completed:
            counts[entry["arm"]]["invalid"] += 1
            rows.append({**entry, "attempt": attempt, "status": "orphaned_reservation",
                         "problems": ["reserved_without_committed_record"],
                         "reservation": portable_path(reservation)})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        judged = review(entry, record)
        retry_exhausted = judged["status"] == "infra_error" and (
            path.name.endswith("_a2.json") or retry_pool_used >= INFRA_RETRY_CAP)
        terminal = judged["status"] in {"valid", "invalid", "call_error"} or retry_exhausted
        category = "valid" if judged["status"] == "valid" else "invalid"
        if terminal:
            counts[entry["arm"]][category] += 1
        else:
            incomplete.append(entry["call"])
        rows.append({**entry, "attempt": record["attempt"], **judged})
    if incomplete:
        return {"status": "incomplete", "missing_or_retryable_calls": sorted(set(incomplete)), "rows": rows}
    result = {"schema": "d3_hunk_contract_comparison_v1", "development_measurement": True,
              "design_note": "저장된 D3 응답에서 발견한 줄 번호 근거 무효 사례를 본 뒤 만든 계약이다.",
              "denominator_rule": "각 arm의 24개 사전등록 논리 호출 전체. 거절·파싱 실패·계약 위반·호출 오류·인프라 재시도 소진도 invalid에 포함한다.",
              "transport_condition": "48개 논리 호출이 모두 종단 상태일 때만 두 invalid 비율을 함께 제시한다.",
              "counts": counts, "invalid_rate": {arm: value["invalid"] / 24 for arm, value in counts.items()},
              "api_attempts": _used_attempts(), "api_cap": TOTAL_API_CAP, "rows": rows,
              "wording": "이 저장 응답 개발 측정은 인용 계약의 무효 응답 비율만 비교한다. 모델 능력의 우위나 다른 사례에 대한 일반화 근거로 쓰지 않는다."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "written", "output": output.as_posix(), "counts": counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("plan")
    call_parser = sub.add_parser("call")
    call_parser.add_argument("--call", type=int, required=True)
    call_parser.add_argument("--attempt", type=int, required=True)
    call_parser.add_argument("--output", type=Path, required=True)
    resolve_parser = sub.add_parser("resolve-lock")
    resolve_parser.add_argument("--lock-sha256", required=True)
    resolve_parser.add_argument("--operator-note", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        print(json.dumps({packet_id(path): sha(path) for path in packet_paths()}, indent=2))
        return 0
    if args.command == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))
        return 0
    if args.command == "resolve-lock":
        print(json.dumps(resolve_lock(args.lock_sha256, args.operator_note), ensure_ascii=False, indent=2))
        return 0
    protocol = read_protocol()
    entry = next(row for row in call_order() if row["call"] == args.call)
    return call(entry, args.attempt, ROOT / args.output)


if __name__ == "__main__":
    raise SystemExit(main())
