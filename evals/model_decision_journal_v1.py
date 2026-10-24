"""재시작 가능한 모델 결정 호출 원장.

고정된 판단 엔진은 매 실행마다 처음부터 계산한다. 이 모듈은 엔진이 요청한 동일 ordinal의
완료 응답만 재생하고, 새 요청의 예약·응답·재시도 대기와 전역 상한을 파일로 보존한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = "model_decision_journal_v1"
API_SCHEMA = "model_decision_api_v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BINDING_KEYS = {
    "run_id", "source_anchor", "protocol_sha256", "checkpoint_sha256",
    "snapshot_sha256", "initial_messages_sha256",
}
TERMINAL_STATUSES = {"choice_ready", "terminal_hold"}


class JournalError(RuntimeError):
    pass


class JournalIntegrityError(JournalError):
    pass


class JournalBusyError(JournalError):
    pass


class _JournalControl(BaseException):
    """Provider 예외 정규화 경계를 통과하는 journal 전용 제어 흐름."""


class _PauseInfra(_JournalControl):
    def __init__(self, not_before: str) -> None:
        self.not_before = not_before


class _BudgetExhausted(_JournalControl):
    pass


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _validate_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label}에 비유한 수치가 있다")
        return
    if isinstance(value, list):
        for child in value:
            _validate_json(child, label)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{label} object key는 문자열이어야 한다")
        for child in value.values():
            _validate_json(child, label)
        return
    raise TypeError(f"{label}은 JSON 값이어야 한다")


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat(follow_symlinks=False).st_nlink != 1:
        raise JournalIntegrityError(f"원장 경로는 단일 regular file이어야 한다: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"JSON constant 거부: {raw}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise JournalIntegrityError(f"읽을 수 없는 원장 파일: {path.name}") from error
    if not isinstance(value, dict):
        raise JournalIntegrityError(f"원장 파일은 object여야 한다: {path.name}")
    _validate_json(value, path.name)
    return value


def _write_new(path: Path, value: dict) -> None:
    _validate_json(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("now_utc는 timezone-aware datetime을 반환해야 한다")
    result = value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 호환
    if result.utcoffset() != timedelta(0):
        raise ValueError("UTC 시각 변환 실패")
    return result


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat()


def _parse_stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise JournalIntegrityError("저장 시각 형식 불일치")
    match = re.fullmatch(r"(.+\.)(\d{7,})([+-]\d\d:\d\d|Z)", value)
    if match:
        value = match[1] + match[2][:6] + match[3]
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10 호환
    except ValueError as error:
        raise JournalIntegrityError("저장 시각 형식 불일치") from error
    if parsed.tzinfo is None:
        raise JournalIntegrityError("저장 시각에 timezone이 없다")
    return parsed.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 호환


def _binding(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("binding 필드 불일치")
    if not isinstance(value["run_id"], str) or not value["run_id"] or len(value["run_id"]) > 200:
        raise ValueError("run_id 형식 불일치")
    if HEX40.fullmatch(str(value["source_anchor"])) is None:
        raise ValueError("source_anchor 형식 불일치")
    for key in BINDING_KEYS - {"run_id", "source_anchor"}:
        if HEX64.fullmatch(str(value[key])) is None:
            raise ValueError(f"{key} 형식 불일치")
    return deepcopy(value)


@contextmanager
def _lease(path: Path, owner: dict) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema": "model_decision_active_lease_v1", **owner, "pid": os.getpid(),
             "thread_id": threading.get_ident(), "nonce": uuid.uuid4().hex}
    try:
        _write_new(path, value)
    except FileExistsError as error:
        existing = _read(path)
        required = {"schema", "scope", "pid", "thread_id", "nonce", "manifest_sha256"}
        if set(existing) != required or existing.get("schema") != "model_decision_active_lease_v1":
            raise JournalIntegrityError("active lease 형식 불일치") from error
        raise JournalBusyError("active lease가 있어 중복 실행을 거부한다") from error
    try:
        yield
    finally:
        if not path.is_file() or _read(path) != value:
            raise JournalIntegrityError("active lease가 실행 중 변경됐다")
        path.unlink()


class DecisionJournal:
    def __init__(
        self,
        root: Path,
        api_root: Path,
        binding: dict,
        params: dict,
        retry_delay_s: float = 30,
        api_cap: int = 20,
    ) -> None:
        self.root = Path(root).absolute()
        self.api_root = Path(api_root).absolute()
        self.binding = _binding(binding)
        _validate_json(params, "params")
        self.params = deepcopy(params)
        if (isinstance(retry_delay_s, bool) or not isinstance(retry_delay_s, (int, float))
                or not math.isfinite(retry_delay_s) or retry_delay_s < 0):
            raise ValueError("retry_delay_s 범위 불일치")
        if isinstance(api_cap, bool) or not isinstance(api_cap, int) or api_cap < 1:
            raise ValueError("api_cap 범위 불일치")
        self.retry_delay_s = float(retry_delay_s)
        self.api_cap = api_cap
        self._prepare()

    def _manifest_value(self) -> dict:
        return {"schema": SCHEMA, "binding": self.binding, "binding_sha256": digest(self.binding),
                "params": self.params, "params_sha256": digest(self.params),
                "retry_delay_s": self.retry_delay_s, "api_cap": self.api_cap}

    def _prepare_file(self, path: Path, expected: dict, allowed_names: set[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if _read(path) != expected:
                raise JournalIntegrityError(f"기존 {path.name}이 현재 실행 결합과 다르다")
            return
        unknown = {item.name for item in path.parent.iterdir()} - allowed_names
        if unknown:
            raise JournalIntegrityError(f"manifest 없는 기존 상태를 인수하지 않는다: {sorted(unknown)}")
        try:
            _write_new(path, expected)
        except FileExistsError:
            if _read(path) != expected:
                raise JournalIntegrityError(f"경합 중 생성된 {path.name}이 다르다")

    def _prepare(self) -> None:
        if self.root.is_symlink() or self.api_root.is_symlink():
            raise JournalIntegrityError("journal/API root symlink를 거부한다")
        self.root.mkdir(parents=True, exist_ok=True)
        self.api_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.api_manifest_path = self.api_root / "manifest.json"
        self._prepare_file(self.manifest_path, self._manifest_value(), {"manifest.json"})
        api_manifest = {"schema": API_SCHEMA, "api_cap": self.api_cap}
        self._prepare_file(self.api_manifest_path, api_manifest,
                           {"manifest.json", "reservations", "responses", "receipts", ".active"})
        self.manifest_sha256 = file_sha(self.manifest_path)
        self.binding_sha256 = digest(self.binding)
        self.params_sha256 = digest(self.params)

    def _verify_prepared(self) -> None:
        if (_read(self.manifest_path) != self._manifest_value()
                or file_sha(self.manifest_path) != self.manifest_sha256):
            raise JournalIntegrityError("decision manifest가 준비 뒤 변경됐다")
        allowed = {"manifest.json", "events", "terminal.json", ".active"}
        unknown = {path.name for path in self.root.iterdir()} - allowed
        if unknown:
            raise JournalIntegrityError(f"decision root의 등록되지 않은 파일: {sorted(unknown)}")
        allowed_api = {"manifest.json", "reservations", "responses", "receipts", ".active"}
        unknown_api = {path.name for path in self.api_root.iterdir()} - allowed_api
        if unknown_api:
            raise JournalIntegrityError(f"API root의 등록되지 않은 파일: {sorted(unknown_api)}")

    def _events(self) -> list[dict]:
        folder = self.root / "events"
        paths = sorted(folder.glob("*.json")) if folder.exists() else []
        if [path.name for path in paths] != [f"{index:04d}.json" for index in range(1, len(paths) + 1)]:
            raise JournalIntegrityError("journal event 순서 불일치")
        events, previous = [], None
        for index, path in enumerate(paths, 1):
            value = _read(path)
            required = {"schema", "ordinal", "global_index", "reservation_sha256", "response_sha256",
                        "manifest_sha256", "previous_sha256"}
            if (set(value) != required or value.get("schema") != "model_decision_event_v1"
                    or value.get("ordinal") != index or value.get("manifest_sha256") != self.manifest_sha256
                    or value.get("previous_sha256") != previous):
                raise JournalIntegrityError("journal CAS chain 불일치")
            previous = file_sha(path)
            events.append(value)
        return events

    def _audit_api(self) -> dict:
        if _read(self.api_manifest_path) != {"schema": API_SCHEMA, "api_cap": self.api_cap}:
            raise JournalIntegrityError("전역 API manifest 변경")
        reservation_dir, response_dir = self.api_root / "reservations", self.api_root / "responses"
        receipt_dir = self.api_root / "receipts"
        reservations = sorted(reservation_dir.glob("*.json")) if reservation_dir.exists() else []
        responses = sorted(response_dir.glob("*.json")) if response_dir.exists() else []
        receipts = sorted(receipt_dir.glob("*.json")) if receipt_dir.exists() else []
        names = [path.name for path in reservations]
        if names != [f"{index:04d}.json" for index in range(1, len(names) + 1)] or len(names) > self.api_cap:
            raise JournalIntegrityError("전역 API reservation 순서·상한 불일치")
        response_map = {path.name: path for path in responses}
        receipt_map = {path.name: path for path in receipts}
        if (len(response_map) != len(responses) or len(receipt_map) != len(receipts)
                or not set(response_map) <= set(names) or set(receipt_map) != set(response_map)):
            raise JournalIntegrityError("reservation 없는 API response")
        rows = []
        required = {"schema", "index", "manifest_sha256", "binding_sha256", "run_id", "ordinal",
                    "messages_sha256", "params_sha256", "reserved_at"}
        previous_receipt_sha = None
        for index, path in enumerate(reservations, 1):
            reservation = _read(path)
            if (set(reservation) != required or reservation.get("schema") != "model_decision_reservation_v1"
                    or reservation.get("index") != index or not isinstance(reservation.get("ordinal"), int)
                    or reservation["ordinal"] < 1 or HEX64.fullmatch(str(reservation.get("manifest_sha256"))) is None
                    or HEX64.fullmatch(str(reservation.get("binding_sha256"))) is None
                    or HEX64.fullmatch(str(reservation.get("messages_sha256"))) is None
                    or HEX64.fullmatch(str(reservation.get("params_sha256"))) is None):
                raise JournalIntegrityError("API reservation 형식 불일치")
            _parse_stamp(reservation.get("reserved_at"))
            response = None
            if path.name in response_map:
                response_path = response_map[path.name]
                response = _read(response_path)
                required_response = {"schema", "index", "reservation_sha256", "status", "result",
                                     "retry_not_before", "finished_at"}
                if (set(response) != required_response
                        or response.get("schema") != "model_decision_response_v1"
                        or response.get("index") != index
                        or response.get("reservation_sha256") != file_sha(path)
                        or response.get("status") not in {"completed", "infra_error"}
                        or not isinstance(response.get("result"), dict)
                        or response.get("status") != ("infra_error" if self._retryable(response["result"])
                                                       else "completed")):
                    raise JournalIntegrityError("API response 결합 불일치")
                _validate_json(response["result"], "safe_result")
                _parse_stamp(response.get("finished_at"))
                retry_at = response.get("retry_not_before")
                if (response["status"] == "infra_error") != (retry_at is not None):
                    raise JournalIntegrityError("infra retry 시각 결합 불일치")
                if retry_at is not None:
                    _parse_stamp(retry_at)
                receipt = _read(receipt_map[path.name])
                expected_receipt = {"schema": "model_decision_response_receipt_v1", "index": index,
                                    "reservation_sha256": file_sha(path),
                                    "response_sha256": file_sha(response_path),
                                    "previous_receipt_sha256": previous_receipt_sha}
                if receipt != expected_receipt:
                    raise JournalIntegrityError("API response append-only receipt 불일치")
                previous_receipt_sha = file_sha(receipt_map[path.name])
            rows.append({"reservation_path": path, "reservation": reservation,
                         "response_path": response_map.get(path.name), "response": response})
        events = self._events()
        own = [row for row in rows if row["reservation"]["manifest_sha256"] == self.manifest_sha256]
        completed = [row for row in own if row["response"] is not None]
        if len(events) != len(completed):
            raise JournalIntegrityError("완료 response와 journal event 수 불일치")
        for ordinal, (event, row) in enumerate(zip(events, completed, strict=True), 1):
            if (row["reservation"]["binding_sha256"] != self.binding_sha256
                    or row["reservation"]["run_id"] != self.binding["run_id"]
                    or row["reservation"]["params_sha256"] != self.params_sha256
                    or row["reservation"]["ordinal"] != ordinal
                    or event["global_index"] != row["reservation"]["index"]
                    or event["reservation_sha256"] != file_sha(row["reservation_path"])
                    or event["response_sha256"] != file_sha(row["response_path"])):
                raise JournalIntegrityError("decision event와 API 원장 결합 불일치")
        return {"rows": rows, "own": own, "reserved": len(rows),
                "pending": sum(row["response"] is None for row in rows),
                "remaining": self.api_cap - len(rows)}

    @staticmethod
    def _retryable(result: dict) -> bool:
        status = result.get("status_code")
        return (isinstance(result.get("infra_error"), str) and bool(result["infra_error"]) or
                isinstance(status, int) and status >= 500 or
                result.get("error_type") in {"TimeoutError", "ConnectionError"})

    def _append_event(self, ordinal: int, reservation_path: Path, response_path: Path) -> None:
        events = self._events()
        value = {"schema": "model_decision_event_v1", "ordinal": ordinal,
                 "global_index": int(reservation_path.stem),
                 "reservation_sha256": file_sha(reservation_path),
                 "response_sha256": file_sha(response_path), "manifest_sha256": self.manifest_sha256,
                 "previous_sha256": file_sha(self.root / "events" / f"{len(events):04d}.json") if events else None}
        _write_new(self.root / "events" / f"{ordinal:04d}.json", value)

    def _terminal(self, result: dict, status: str, calls: list[dict]) -> dict:
        return {"schema": "model_decision_terminal_v1", "manifest_sha256": self.manifest_sha256,
                "status": status, "engine_result": deepcopy(result),
                "calls_sha256": digest(calls), "completed_calls": len(calls)}

    @staticmethod
    def _result_status(result: dict) -> str:
        if result.get("status") == "terminal_hold":
            return "terminal_hold"
        selection = result.get("selection")
        if (isinstance(selection, dict) and selection.get("action") == "hold") or result.get("stop_reason") in {
            "budget_exhausted", "infra_error",
        }:
            return "terminal_hold"
        return "choice_ready"

    def advance(
        self,
        engine: Callable[[Callable[[list, dict], dict]], dict],
        complete: Callable[[list, dict], dict],
        now_utc: Callable[[], datetime],
    ) -> dict:
        if not callable(engine) or not callable(complete) or not callable(now_utc):
            raise TypeError("engine, complete, now_utc는 callable이어야 한다")
        self._verify_prepared()
        initial = self._audit_api()
        if initial["pending"]:
            return {"status": "blocked_pending", "reason": "pending_reservation", "engine_result": None,
                    "api": self._public_audit(initial)}
        owner = {"scope": "decision", "manifest_sha256": self.manifest_sha256}
        api_owner = {"scope": "global_api", "manifest_sha256": self.manifest_sha256}
        try:
            with _lease(self.root / ".active", owner), _lease(self.api_root / ".active", api_owner):
                self._verify_prepared()
                audit = self._audit_api()
                if audit["pending"]:
                    return {"status": "blocked_pending", "reason": "pending_reservation",
                            "engine_result": None, "api": self._public_audit(audit)}
                terminal_path = self.root / "terminal.json"
                stored_terminal = _read(terminal_path) if terminal_path.exists() else None
                if stored_terminal is not None:
                    required_terminal = {"schema", "manifest_sha256", "status", "engine_result",
                                         "calls_sha256", "completed_calls"}
                    if (set(stored_terminal) != required_terminal
                            or stored_terminal.get("schema") != "model_decision_terminal_v1"
                            or stored_terminal.get("manifest_sha256") != self.manifest_sha256
                            or stored_terminal.get("status") not in TERMINAL_STATUSES
                            or not isinstance(stored_terminal.get("engine_result"), dict)
                            or not isinstance(stored_terminal.get("completed_calls"), int)
                            or stored_terminal["completed_calls"] < 0
                            or HEX64.fullmatch(str(stored_terminal.get("calls_sha256"))) is None):
                        raise JournalIntegrityError("저장 terminal 형식·manifest 결합 불일치")
                ordinal = 0
                used: list[dict] = []
                recorded_this_advance: set[int] = set()

                def completion(messages: list, params: dict) -> dict:
                    nonlocal ordinal, audit
                    _validate_json(messages, "messages")
                    _validate_json(params, "params")
                    if params != self.params:
                        raise JournalIntegrityError("engine params가 준비 manifest와 다르다")
                    ordinal += 1
                    messages_sha = digest(messages)
                    if ordinal == 1 and messages_sha != self.binding["initial_messages_sha256"]:
                        raise JournalIntegrityError("첫 messages가 binding과 다르다")
                    own = audit["own"]
                    if ordinal <= len(own):
                        row = own[ordinal - 1]
                        reservation, response = row["reservation"], row["response"]
                        if response is None:
                            raise JournalIntegrityError("current decision pending reservation")
                        if (reservation["ordinal"] != ordinal or reservation["messages_sha256"] != messages_sha
                                or reservation["params_sha256"] != self.params_sha256):
                            raise JournalIntegrityError("cached completion 입력이 바뀌었다")
                        used.append(deepcopy(response["result"]))
                        return deepcopy(response["result"])
                    if stored_terminal is not None:
                        if (stored_terminal["engine_result"].get("stop_reason") == "budget_exhausted"
                                and audit["remaining"] <= 0):
                            raise _BudgetExhausted
                        raise JournalIntegrityError("terminal replay가 새 API 호출을 요구한다")
                    if ordinal != len(own) + 1:
                        raise JournalIntegrityError("logical completion ordinal 불일치")
                    if own:
                        prior = own[-1]["response"]
                        if prior is not None and self._retryable(prior["result"]):
                            deadline = _parse_stamp(prior["retry_not_before"])
                            if ordinal - 1 in recorded_this_advance or _utc(now_utc()) < deadline:
                                raise _PauseInfra(prior["retry_not_before"])
                    if audit["remaining"] <= 0:
                        raise _BudgetExhausted
                    index = audit["reserved"] + 1
                    reservation = {"schema": "model_decision_reservation_v1", "index": index,
                                   "manifest_sha256": self.manifest_sha256,
                                   "binding_sha256": self.binding_sha256, "run_id": self.binding["run_id"],
                                   "ordinal": ordinal, "messages_sha256": messages_sha,
                                   "params_sha256": self.params_sha256, "reserved_at": _stamp(now_utc())}
                    reservation_path = self.api_root / "reservations" / f"{index:04d}.json"
                    _write_new(reservation_path, reservation)
                    started = time.perf_counter()
                    try:
                        returned = complete(deepcopy(messages), deepcopy(params))
                    except Exception as error:  # noqa: BLE001 - 예외 값 대신 class만 보존한다.
                        name = type(error).__name__
                        safe = {"error_type": name, "wall_s": time.perf_counter() - started}
                        if isinstance(error, (TimeoutError, ConnectionError)):
                            safe["infra_error"] = name
                    else:
                        if not isinstance(returned, dict):
                            raise TypeError("complete 결과는 safe_result object여야 한다")
                        safe = deepcopy(returned)
                    _validate_json(safe, "safe_result")
                    retryable = self._retryable(safe)
                    finished = _utc(now_utc())
                    response = {"schema": "model_decision_response_v1", "index": index,
                                "reservation_sha256": file_sha(reservation_path),
                                "status": "infra_error" if retryable else "completed", "result": safe,
                                "retry_not_before": _stamp(finished + timedelta(seconds=self.retry_delay_s))
                                if retryable else None, "finished_at": _stamp(finished)}
                    response_path = self.api_root / "responses" / reservation_path.name
                    _write_new(response_path, response)
                    _write_new(self.api_root / "receipts" / reservation_path.name,
                               {"schema": "model_decision_response_receipt_v1", "index": index,
                                "reservation_sha256": file_sha(reservation_path),
                                "response_sha256": file_sha(response_path),
                                "previous_receipt_sha256": file_sha(
                                    self.api_root / "receipts" / f"{index - 1:04d}.json") if index > 1 else None})
                    self._append_event(ordinal, reservation_path, response_path)
                    recorded_this_advance.add(ordinal)
                    audit = self._audit_api()
                    used.append(deepcopy(safe))
                    return deepcopy(safe)

                try:
                    engine_result = engine(completion)
                    if not isinstance(engine_result, dict):
                        raise TypeError("engine 결과는 object여야 한다")
                    _validate_json(engine_result, "engine_result")
                    status = self._result_status(engine_result)
                except _PauseInfra as pause:
                    current = self._audit_api()
                    return {"status": "paused_infra", "not_before": pause.not_before,
                            "engine_result": None, "api": self._public_audit(current)}
                except _BudgetExhausted:
                    engine_result = {"stop_reason": "budget_exhausted", "selection": {"action": "hold"}}
                    status = "terminal_hold"
                terminal = self._terminal(engine_result, status, used)
                if stored_terminal is None:
                    _write_new(terminal_path, terminal)
                elif stored_terminal != terminal:
                    raise JournalIntegrityError("저장 terminal과 재계산 결과가 다르다")
                current = self._audit_api()
                return {"status": status, "engine_result": deepcopy(engine_result),
                        "api": self._public_audit(current)}
        except JournalBusyError:
            return {"status": "blocked_pending", "reason": "active_lease",
                    "engine_result": None, "api": self._public_audit(self._audit_api())}

    @staticmethod
    def _public_audit(audit: dict) -> dict:
        return {"reserved": audit["reserved"], "pending": audit["pending"],
                "remaining": audit["remaining"]}
