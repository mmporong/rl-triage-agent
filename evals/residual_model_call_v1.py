"""잔여 실패 재진단의 모델 호출과 두 seed가 공유하는 20회 요청 원장."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import feedback_model_call as F  # noqa: E402
import r1_repair_model_call as R  # noqa: E402
import residual_contract_v1 as C  # noqa: E402

SCHEMA = "residual_feedback_call_v1"
RESERVATION_SCHEMA = "residual_api_reservation_v1"
RESPONSE_SCHEMA = "residual_api_response_v1"
API_CAP = 20
TRANSPORT_TIMEOUT_S = 900
ACTIVE_LEASE_WAIT_S = TRANSPORT_TIMEOUT_S + 30


class BudgetStop(RuntimeError):
    """요청 상한 또는 미완료 예약 때문에 새 요청을 시작할 수 없다."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reservation_files(api_root: Path) -> list[Path]:
    return sorted((Path(api_root) / "reservations").glob("[0-9][0-9][0-9][0-9].json"))


def audit_budget(api_root: Path) -> dict:
    """원장 순서와 응답 결합을 다시 계산한다. 미응답 예약은 호출 소비로 센다."""
    root = Path(api_root)
    reservations = _reservation_files(root)
    responses = {path.name: path for path in (root / "responses").glob("*.json")}
    if len(reservations) > API_CAP:
        raise ValueError("API 요청 원장이 상한을 넘었다")
    counts = {"reserved": len(reservations), "completed": 0, "pending": 0,
              "ok": 0, "error": 0, "timeout": 0}
    rows = []
    output_keys = set()
    for index, path in enumerate(reservations, 1):
        reservation = read(path)
        required = {"schema", "request_index", "kind", "packet_sha256", "messages_sha256",
                    "output_key", "reserved_at"}
        if (path.name != f"{index:04d}.json" or set(reservation) != required
                or reservation["schema"] != RESERVATION_SCHEMA
                or reservation["request_index"] != index
                or reservation["kind"] not in {"diagnosis", "correction"}
                or not all(isinstance(reservation[key], str) and reservation[key]
                           for key in ("packet_sha256", "messages_sha256", "output_key", "reserved_at"))
                or reservation["output_key"] in output_keys):
            raise ValueError("API 요청 예약의 순서·필드·식별자가 다르다")
        output_keys.add(reservation["output_key"])
        response_path = responses.pop(path.name, None)
        response = read(response_path) if response_path else None
        if response is None:
            counts["pending"] += 1
        else:
            expected = {key: reservation[key] for key in
                        ("request_index", "kind", "packet_sha256", "messages_sha256", "output_key")}
            if (set(response) != {"schema", *expected, "reservation_sha256", "status", "result", "finished_at"}
                    or response["schema"] != RESPONSE_SCHEMA
                    or any(response.get(key) != value for key, value in expected.items())
                    or response["reservation_sha256"] != digest(reservation)
                    or response["status"] not in {"ok", "error", "timeout"}
                    or not isinstance(response["result"], dict)
                    or response["status"] != _response_status(response["result"])
                    or not isinstance(response["finished_at"], str) or not response["finished_at"]):
                raise ValueError("API 응답 receipt가 예약과 결합되지 않았다")
            counts["completed"] += 1
            counts[response["status"]] += 1
        rows.append({"reservation": reservation, "response": response})
    if responses:
        raise ValueError("예약이 없는 API 응답 receipt가 있다")
    return {"schema": "residual_api_budget_audit_v1", "cap": API_CAP, **counts,
            "remaining": API_CAP - counts["reserved"], "rows": rows}


def _active_path(api_root: Path) -> Path:
    return Path(api_root) / "active.json"


@contextmanager
def _active_lease(api_root: Path):
    """api-root 하나에서 예약부터 응답 receipt 기록까지 한 호출만 허용한다."""
    root = Path(api_root)
    root.mkdir(parents=True, exist_ok=True)
    path = _active_path(root)
    token = secrets.token_hex(16)
    payload = (json.dumps({"schema": "residual_api_active_lease_v1", "token": token,
                           "pid": os.getpid(), "acquired_at": now()}, ensure_ascii=False,
                          sort_keys=True, allow_nan=False) + "\n").encode()
    deadline = time.monotonic() + ACTIVE_LEASE_WAIT_S
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise BudgetStop("다른 API 호출의 active lease가 끝나지 않았다") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            current = read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if not isinstance(current, dict) or current.get("token") != token:
            raise RuntimeError("API active lease 소유권이 바뀌었다")
        path.unlink()


def _reserve_locked(api_root: Path, kind: str, packet_sha: str,
                    messages_sha: str, output_key: str) -> tuple[Path, dict]:
    root = Path(api_root)
    audit = audit_budget(root)
    if audit["pending"]:
        raise BudgetStop("미완료 요청 예약이 있어 새 호출을 시작하지 않는다")
    if audit["remaining"] <= 0:
        raise BudgetStop("사전등록 API 요청 상한 20회에 도달했다")
    for index in range(1, API_CAP + 1):
        path = root / "reservations" / f"{index:04d}.json"
        row = {"schema": RESERVATION_SCHEMA, "request_index": index, "kind": kind,
               "packet_sha256": packet_sha, "messages_sha256": messages_sha,
               "output_key": output_key, "reserved_at": now()}
        try:
            write_new(path, row)
            # 동시 예약으로 앞 번호가 비는 경우는 원장 검증에서 즉시 멈춘다.
            audit_budget(root)
            return path, row
        except FileExistsError:
            continue
    raise BudgetStop("사전등록 API 요청 상한 20회에 도달했다")


def _reserve(api_root: Path, kind: str, packet_sha: str,
             messages_sha: str, output_key: str) -> tuple[Path, dict]:
    """단독 예약 helper. 실제 호출은 MeteredComplete가 lease를 응답 기록까지 유지한다."""
    with _active_lease(api_root):
        return _reserve_locked(api_root, kind, packet_sha, messages_sha, output_key)


def _response_status(result: dict) -> str:
    if result.get("error_type") in {"TimeoutError", "APITimeoutError", "ReadTimeout"}:
        return "timeout"
    if result.get("error_type") or result.get("infra_error"):
        return "error"
    if (result.get("provider_model") != C.MODEL or result.get("finish_reason") == "length"
            or not (result.get("content") or "").strip()):
        return "error"
    return "ok"


def _retryable_exception(error: Exception) -> bool:
    """NVIDIA 호출의 timeout/connection/429/5xx만 인프라 재시도 대상으로 둔다."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if type(error).__name__ in {
            "APIConnectionError", "APITimeoutError", "ConnectTimeout", "ReadTimeout"}:
        return True
    status = getattr(error, "status_code", None)
    return type(status) is int and (status == 429 or status >= 500)


class MeteredComplete:
    def __init__(self, api_root: Path, kind: str, packet_sha: str, output_key: str,
                 complete: Callable[[list[dict], dict], dict]):
        self.root, self.kind, self.packet_sha = Path(api_root), kind, packet_sha
        self.output_key, self.complete = output_key, complete
        self.receipts: list[dict] = []

    def __call__(self, messages: list[dict], params: dict) -> dict:
        messages_sha = C.L.digest(messages)
        local = len(self.receipts) + 1
        key = f"{self.output_key}/request{local}"
        with _active_lease(self.root):
            path, reservation = _reserve_locked(
                self.root, self.kind, self.packet_sha, messages_sha, key)
            started = time.perf_counter()
            try:
                result = self.complete(messages, params)
                if not isinstance(result, dict):
                    raise TypeError("complete 결과는 dict여야 한다")
            except Exception as error:
                result = {"error_type": type(error).__name__,
                          "wall_s": time.perf_counter() - started}
                if _retryable_exception(error):
                    result["infra_error"] = type(error).__name__
            result = dict(result)
            result.setdefault("wall_s", time.perf_counter() - started)
            response = {"schema": RESPONSE_SCHEMA, "request_index": reservation["request_index"],
                        "kind": self.kind, "packet_sha256": self.packet_sha,
                        "messages_sha256": messages_sha, "output_key": key,
                        "reservation_sha256": digest(reservation), "status": _response_status(result),
                        "result": result, "finished_at": now()}
            response_path = self.root / "responses" / path.name
            write_new(response_path, response)
        self.receipts.append({"request_index": reservation["request_index"],
                              "reservation_sha256": digest(reservation),
                              "response_sha256": file_sha(response_path)})
        return result


@contextmanager
def residual_contract():
    """검증 구현은 재사용하되 전역 모듈 결합을 호출 구간 뒤 반드시 복원한다."""
    old_contract, old_schema = R.C, R.SCHEMA
    R.C, R.SCHEMA = C, SCHEMA
    try:
        yield
    finally:
        R.C, R.SCHEMA = old_contract, old_schema


def _review_record(record: dict, data: bytes, kind: str, api_root: Path) -> dict:
    with residual_contract():
        verdict = R.review(record, data, kind)
    rows = record.get("request_receipts")
    calls = record.get("calls")
    if not isinstance(rows, list) or not isinstance(calls, list) or len(rows) != len(calls):
        raise ValueError("모델 호출과 API receipt 수가 다르다")
    audit = audit_budget(api_root)
    by_index = {row["reservation"]["request_index"]: row for row in audit["rows"]}
    for binding, call in zip(rows, calls):
        if set(binding) != {"request_index", "reservation_sha256", "response_sha256"}:
            raise ValueError("모델 기록의 API receipt 필드가 다르다")
        row = by_index.get(binding["request_index"])
        if row is None or row["response"] is None:
            raise ValueError("모델 기록의 API receipt가 완료되지 않았다")
        reservation, response = row["reservation"], row["response"]
        response_path = Path(api_root) / "responses" / f"{binding['request_index']:04d}.json"
        if (binding["reservation_sha256"] != digest(reservation)
                or binding["response_sha256"] != file_sha(response_path)):
            raise ValueError("모델 기록의 API receipt 해시 결합이 다르다")
        raw = response["result"]
        fields = ("response_id", "provider_model", "finish_reason", "content", "reasoning_content",
                  "input_tokens", "output_tokens", "total_tokens", "wall_s")
        if any(call.get(key) != raw.get(key) for key in fields):
            raise ValueError("저장 응답과 모델 기록의 비용·결과가 다르다")
    if verdict["status"] == "valid" and audit["pending"]:
        raise ValueError("미완료 예약이 남은 원장에서는 성공을 확정하지 않는다")
    return verdict


def _packet_bytes(packet: dict) -> bytes:
    return (json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _output_scope(folder: Path) -> str:
    """공용 원장 안에서 seed/epoch/model을 구분하되 호스트 절대 경로는 기록하지 않는다."""
    parts = Path(folder).parts[-3:]
    if not parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("모델 출력 경로의 portable 식별자가 잘못됐다")
    return "/".join(parts)


def wsl_path(path: Path) -> str:
    value = Path(path).resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", value):
        raise ValueError("모델 호출에는 Windows 저장소 경로가 필요하다")
    return f"/mnt/{value[0].lower()}/{value[3:]}"


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
 script=next((i for i,x in enumerate(args) if x.endswith('/residual_model_call_v1.py')),None)
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


def _run_worker(kind: str, packet_path: Path, output: Path, api_root: Path) -> None:
    pid_file = output.with_suffix(".pid.json")
    argv = [wsl_path(ROOT / "evals/residual_model_call_v1.py"), "--kind", kind,
            "--packet", wsl_path(packet_path), "--output", wsl_path(output),
            "--api-root", wsl_path(api_root), "--pid-file", wsl_path(pid_file)]
    command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(argv)
    try:
        process = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=TRANSPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        receipt = {"schema": "residual_transport_error_v1", "type": "dialogue_timeout",
                   "worker_cleanup": "unverified", "at": now()}
        try:
            receipt["worker_cleanup"] = _stop_worker(pid_file, packet_path)
        except Exception as error:
            receipt["cleanup_error_type"] = type(error).__name__
        write_new(output.with_suffix(".transport_error.json"), receipt)
        raise TimeoutError("모델 대화 시간 상한 초과") from None
    if not output.is_file():
        raise RuntimeError(f"모델 호출 기록 없음(exit={process.returncode})")
    if pid_file.exists():
        raise RuntimeError("자연 종료한 모델 worker PID 파일이 남았다")


def decide(folder: Path, kind: str, packet: dict, api_root: Path,
           caller: Callable[[list[dict], dict], dict] | None = None) -> dict:
    """인프라 오류만 두 번째 대화를 허용하며, 대화마다 계약 피드백은 최대 한 번이다."""
    if kind not in {"diagnosis", "correction"}:
        raise ValueError("등록되지 않은 모델 단계")
    folder, api_root = Path(folder), Path(api_root)
    folder.mkdir(parents=True, exist_ok=True)
    packet_path = folder / f"{kind}.packet.json"
    write_new(packet_path, packet)
    frozen = packet_path.read_bytes()
    if frozen != _packet_bytes(packet):
        raise ValueError("packet 직렬화가 고정 형식과 다르다")
    attempts = []
    for attempt in (1, 2):
        if packet_path.read_bytes() != frozen:
            raise ValueError("호출 사이에 모델 packet이 바뀌었다")
        output = folder / f"{kind}.dialogue{attempt}.json"
        if caller is None:
            _run_worker(kind, packet_path, output, api_root)
        else:
            metered = MeteredComplete(api_root, kind, file_sha(packet_path),
                                      f"{_output_scope(folder)}/{kind}.dialogue{attempt}", caller)
            with residual_contract():
                record = R.dialogue(kind, frozen, complete=metered)
            record["request_receipts"] = metered.receipts
            write_new(output, record)
        if packet_path.read_bytes() != frozen:
            raise ValueError("모델 호출 중 packet이 바뀌었다")
        record = read(output)
        verdict = _review_record(record, frozen, kind, api_root)
        row = {"record": output.name, "record_sha256": file_sha(output), "review": verdict,
               "request_receipts": record["request_receipts"]}
        write_new(folder / f"{kind}.review{attempt}.json", row)
        attempts.append(row)
        if verdict["status"] != "infra_error":
            return {"review": verdict, "attempts": attempts}
    return {"review": verdict, "attempts": attempts}


def _worker_identity(packet: Path) -> dict:
    raw = Path("/proc/self/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": os.getpid(), "start_ticks": int(fields[19]), "argv": sys.argv,
            "packet_sha256": file_sha(packet)}


def worker(kind: str, packet: Path, output: Path, api_root: Path, pid_file: Path) -> int:
    if os.name != "posix":
        raise RuntimeError("모델 worker는 WSL 자격증명 환경에서만 실행한다")
    if output.exists():
        raise FileExistsError(output)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    write_new(pid_file, _worker_identity(packet))
    try:
        data = packet.read_bytes()
        metered = MeteredComplete(api_root, kind, file_sha(packet),
                                  f"{_output_scope(output.parent)}/{output.stem}", F.openai_complete)
        with residual_contract():
            record = R.dialogue(kind, data, complete=metered)
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
    return worker(args.kind, args.packet, args.output, args.api_root, args.pid_file)


if __name__ == "__main__":
    raise SystemExit(main())
