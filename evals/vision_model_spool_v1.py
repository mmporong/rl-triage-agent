"""GPU 호스트와 NVIDIA provider 호스트 사이의 write-once 모델 호출 spool.

이 모듈은 파일 운반을 수행하지 않는다. 원격 caller는 request를 한 번 기록하고
결합된 response만 기다리며, 로컬 processor는 전역 32회 원장에 예약한 뒤 주입된
provider를 한 번만 호출한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import grounded_vision_observer_v1 as OBSERVER

REQUEST_SCHEMA = "vision_model_spool_request_v1"
RESPONSE_SCHEMA = "vision_model_spool_response_v1"
LEDGER_SCHEMA = "vision_model_spool_ledger_v1"
RESERVATION_SCHEMA = "vision_model_spool_reservation_v1"
TERMINAL_SCHEMA = "vision_model_spool_terminal_v1"
FAILURE_SCHEMA = "vision_model_spool_provider_failure_v1"
API_CAP = 32
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SpoolError(RuntimeError):
    pass


class SpoolIntegrityError(SpoolError):
    pass


class SpoolPendingError(SpoolError):
    pass


class SpoolCapError(SpoolError):
    pass


class ProviderCallError(SpoolError):
    def __init__(self, provider_error_type: str):
        self.provider_error_type = provider_error_type
        super().__init__("provider 호출이 완료되지 않았다")


def _json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SpoolIntegrityError(f"{label}에 비유한 수치가 있다")
        return
    if isinstance(value, list):
        for child in value:
            _json(child, label)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for child in value.values():
            _json(child, label)
        return
    raise SpoolIntegrityError(f"{label}은 JSON 값이어야 한다")


def _bytes(value: Any) -> bytes:
    _json(value, "canonical value")
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


def _stamp(value: object) -> None:
    if not isinstance(value, str):
        raise SpoolIntegrityError("저장 시각 형식 불일치")
    match = re.fullmatch(r"(.+\.)(\d{7,})([+-]\d\d:\d\d|Z)", value)
    normalized = match[1] + match[2][:6] + match[3] if match else value
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    except ValueError as error:
        raise SpoolIntegrityError("저장 시각 형식 불일치") from error
    if parsed.tzinfo is None:
        raise SpoolIntegrityError("저장 시각에 timezone이 없다")


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat(follow_symlinks=False).st_nlink != 1:
        raise SpoolIntegrityError(f"{label}은 단일 regular file이어야 한다")


def _read(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(raw)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SpoolIntegrityError(f"{label}을 읽을 수 없다") from error
    if not isinstance(value, dict):
        raise SpoolIntegrityError(f"{label}은 object여야 한다")
    _json(value, label)
    return value


def _directory(path: Path, label: str, *, create: bool = False) -> Path:
    absolute = path.absolute()
    if absolute.exists() or absolute.is_symlink():
        if absolute.is_symlink() or not absolute.is_dir():
            raise SpoolIntegrityError(f"{label}은 symlink가 아닌 directory여야 한다")
    elif create:
        parent = absolute.parent
        if parent.is_symlink() or not parent.is_dir():
            raise SpoolIntegrityError(f"{label} parent가 안전하지 않다")
        absolute.mkdir()
    else:
        raise SpoolIntegrityError(f"{label} directory가 없다")
    return absolute


def _child_directory(root: Path, relative: tuple[str, ...], *, create: bool) -> Path:
    current = _directory(root, "root")
    for name in relative:
        if name in {"", ".", ".."} or Path(name).name != name:
            raise SpoolIntegrityError("경로 조각이 안전하지 않다")
        current = _directory(current / name, name, create=create)
    if root.resolve() not in (current.resolve(), *current.resolve().parents):
        raise SpoolIntegrityError("경로가 root를 벗어났다")
    return current


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    parent = _directory(path.parent, "output parent")
    raw = _bytes(dict(value))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent / path.name, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _identity(job_id: str, sequence: int, messages: list[dict], params: dict) -> dict[str, Any]:
    if JOB_ID.fullmatch(job_id) is None or job_id in {".", ".."}:
        raise ValueError("job_id 형식 불일치")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= 9999:
        raise ValueError("request sequence 범위 불일치")
    if not isinstance(messages, list) or not isinstance(params, dict):
        raise TypeError("messages/params 형식 불일치")
    _json(messages, "messages")
    _json(params, "params")
    core = {"schema": REQUEST_SCHEMA, "job_id": job_id, "sequence": sequence,
            "messages": deepcopy(messages), "params": deepcopy(params),
            "messages_sha256": digest(messages), "params_sha256": digest(params)}
    canonical = digest(core)
    return {**core, "canonical_sha256": canonical,
            "request_id": f"{job_id}-{sequence:04d}-{canonical[:16]}"}


def _request(value: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "job_id", "sequence", "request_id", "messages", "params",
                "messages_sha256", "params_sha256", "canonical_sha256"}
    if set(value) != required or value.get("schema") != REQUEST_SCHEMA:
        raise SpoolIntegrityError("request schema 불일치")
    try:
        expected = _identity(value.get("job_id"), value.get("sequence"),
                             value.get("messages"), value.get("params"))
    except (TypeError, ValueError, SpoolIntegrityError) as error:
        raise SpoolIntegrityError("request identity 형식 불일치") from error
    if value != expected:
        raise SpoolIntegrityError("request hash 또는 id 불일치")
    return expected


def _paths(spool: Path, job_id: str, sequence: int, *, create: bool) -> tuple[Path, Path]:
    if JOB_ID.fullmatch(job_id) is None or job_id in {".", ".."}:
        raise ValueError("job_id 형식 불일치")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= 9999:
        raise ValueError("request sequence 범위 불일치")
    root = _directory(spool, "spool")
    requests = _child_directory(root, ("requests", job_id), create=create)
    responses = _child_directory(root, ("responses", job_id), create=create)
    name = f"{sequence:04d}.json"
    return requests / name, responses / name


def _response(value: dict[str, Any], request: dict[str, Any], request_path: Path) -> dict[str, Any]:
    required = {"schema", "job_id", "sequence", "request_id", "canonical_sha256",
                "request_file_sha256", "provider"}
    expected = {"schema": RESPONSE_SCHEMA, "job_id": request["job_id"],
                "sequence": request["sequence"], "request_id": request["request_id"],
                "canonical_sha256": request["canonical_sha256"],
                "request_file_sha256": file_sha(request_path)}
    if (set(value) != required or any(value.get(key) != item for key, item in expected.items())
            or not isinstance(value.get("provider"), dict)):
        raise SpoolIntegrityError("response와 request 결합 불일치")
    _json(value["provider"], "provider")
    return deepcopy(value["provider"])


def remote_complete(
    messages: list[dict],
    params: dict,
    *,
    spool: Path,
    job_id: str,
    sequence: int,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.25,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """요청을 한 번 기록하고 정확히 결합된 provider 응답만 반환한다."""
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s < 0):
        raise ValueError("timeout_s 범위 불일치")
    if (isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float))
            or not math.isfinite(poll_interval_s) or poll_interval_s <= 0):
        raise ValueError("poll_interval_s 범위 불일치")
    request = _identity(job_id, sequence, messages, params)
    request_path, response_path = _paths(spool, job_id, sequence, create=True)
    if request_path.exists() or request_path.is_symlink():
        if _read(request_path, "request") != request:
            raise SpoolIntegrityError("기존 request가 현재 호출과 다르다")
    else:
        _write_new(request_path, request)
    deadline = monotonic() + float(timeout_s)
    while True:
        if response_path.exists() or response_path.is_symlink():
            return _response(_read(response_path, "response"), request, request_path)
        now = monotonic()
        if now >= deadline:
            raise SpoolPendingError("응답이 없는 예약 요청을 재전송하지 않는다")
        sleep(min(float(poll_interval_s), deadline - now))


def _ledger_manifest(root: Path) -> Path:
    ledger = _directory(root, "ledger")
    manifest = ledger / "manifest.json"
    expected = {"schema": LEDGER_SCHEMA, "api_cap": API_CAP}
    if manifest.exists() or manifest.is_symlink():
        if _read(manifest, "ledger manifest") != expected:
            raise SpoolIntegrityError("ledger manifest 불일치")
    else:
        unknown = {path.name for path in ledger.iterdir()}
        if unknown:
            raise SpoolIntegrityError("manifest 없는 ledger 상태를 인수하지 않는다")
        _write_new(manifest, expected)
    return ledger


def _numbered(root: Path, name: str, *, contiguous: bool) -> list[Path]:
    folder = root / name
    if not folder.exists() and not folder.is_symlink():
        return []
    directory = _directory(folder, name)
    paths = sorted(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise SpoolIntegrityError(f"{name}에 regular file이 아닌 항목이 있다")
    if (any(re.fullmatch(r"[0-9]{4}\.json", path.name) is None for path in paths)
            or (contiguous and [path.name for path in paths]
                != [f"{index:04d}.json" for index in range(1, len(paths) + 1)])):
        raise SpoolIntegrityError(f"{name} 순서 불일치")
    return paths


def _audit(root: Path) -> dict[str, Any]:
    ledger = _ledger_manifest(root)
    allowed = {"manifest.json", "reservations", "terminals", "failures", ".active"}
    unknown = {path.name for path in ledger.iterdir()} - allowed
    if unknown:
        raise SpoolIntegrityError(f"ledger 등록 밖 항목: {sorted(unknown)}")
    reservations = _numbered(ledger, "reservations", contiguous=True)
    terminals = {path.name: path for path in _numbered(ledger, "terminals", contiguous=False)}
    failures = {path.name: path for path in _numbered(ledger, "failures", contiguous=False)}
    reservation_names = {path.name for path in reservations}
    if (len(reservations) > API_CAP or not set(terminals) <= reservation_names
            or not set(failures) <= reservation_names):
        raise SpoolIntegrityError("ledger 상한 또는 항목 수 불일치")
    rows = []
    for index, path in enumerate(reservations, 1):
        reservation = _read(path, "reservation")
        required = {"schema", "index", "job_id", "sequence", "request_id",
                    "canonical_sha256", "request_file_sha256", "reserved_at"}
        if (set(reservation) != required or reservation.get("schema") != RESERVATION_SCHEMA
                or reservation.get("index") != index
                or JOB_ID.fullmatch(str(reservation.get("job_id"))) is None
                or type(reservation.get("sequence")) is not int
                or not 1 <= reservation["sequence"] <= 9999
                or not isinstance(reservation.get("request_id"), str)
                or not reservation["request_id"] or len(reservation["request_id"]) > 200
                or HEX64.fullmatch(str(reservation.get("canonical_sha256"))) is None
                or HEX64.fullmatch(str(reservation.get("request_file_sha256"))) is None):
            raise SpoolIntegrityError("reservation 형식 불일치")
        _stamp(reservation["reserved_at"])
        terminal_path = terminals.get(path.name)
        terminal = _read(terminal_path, "terminal") if terminal_path is not None else None
        if terminal is not None:
            expected = {"schema": TERMINAL_SCHEMA, "index": index,
                        "reservation_sha256": file_sha(path), "provider": terminal.get("provider")}
            if terminal != expected or not isinstance(terminal.get("provider"), dict):
                raise SpoolIntegrityError("terminal 결합 불일치")
            _json(terminal["provider"], "provider")
        failure_path = failures.get(path.name)
        failure = _read(failure_path, "failure") if failure_path is not None else None
        if failure is not None:
            expected_failure = {"schema": FAILURE_SCHEMA, "index": index,
                                "reservation_sha256": file_sha(path),
                                "error_type": failure.get("error_type")}
            if (failure != expected_failure or not isinstance(failure.get("error_type"), str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,159}", failure["error_type"]) is None):
                raise SpoolIntegrityError("provider failure 결합 불일치")
        if terminal is not None and failure is not None:
            raise SpoolIntegrityError("한 reservation에 terminal과 failure가 함께 있다")
        rows.append({"path": path, "reservation": reservation, "terminal": terminal,
                     "failure": failure})
    return {"rows": rows, "reserved": len(rows),
            "pending": sum(row["terminal"] is None for row in rows),
            "remaining": API_CAP - len(rows)}


def _lock(root: Path) -> tuple[int, Path]:
    path = root / ".active"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SpoolPendingError("local ledger processor가 이미 실행 중이다") from error
    return descriptor, path


def _unlock(descriptor: int, path: Path) -> None:
    os.close(descriptor)
    if path.is_symlink() or not path.is_file():
        raise SpoolIntegrityError("local ledger active lease가 변경됐다")
    path.unlink()


def process_one(
    request_path: Path,
    response_path: Path,
    ledger_root: Path,
    provider: Callable[[list[dict], dict], dict],
) -> dict[str, Any]:
    """검증된 request 하나를 전역 cap 원장 아래 한 번만 provider에 전달한다."""
    request_file = request_path.absolute()
    request = _request(_read(request_file, "request"))
    if request["params"] != OBSERVER.REQUEST_PARAMS:
        raise SpoolIntegrityError("grounded vision request params 불일치")
    spool = request_file.parents[2]
    name = f"{request['sequence']:04d}.json"
    if (request_file != spool / "requests" / request["job_id"] / name
            or response_path.absolute() != spool / "responses" / request["job_id"] / name):
        raise SpoolIntegrityError("request/response 경로와 내부 identity가 다르다")
    expected_request, expected_response = _paths(
        spool, request["job_id"], request["sequence"], create=True)
    if request_file != expected_request:
        raise SpoolIntegrityError("request 경로가 canonical path가 아니다")
    ledger = _ledger_manifest(ledger_root)
    descriptor, active = _lock(ledger)
    try:
        audit = _audit(ledger)
        request_file_sha = file_sha(request_file)
        matches = [row for row in audit["rows"]
                   if row["reservation"]["request_file_sha256"] == request_file_sha]
        if len(matches) > 1:
            raise SpoolIntegrityError("같은 request의 reservation이 중복됐다")
        if matches:
            row = matches[0]
            if (row["reservation"]["request_id"] != request["request_id"]
                    or row["reservation"]["canonical_sha256"] != request["canonical_sha256"]):
                raise SpoolIntegrityError("기존 reservation identity 불일치")
            if row["terminal"] is None:
                raise SpoolPendingError("기존 pending reservation은 provider를 다시 호출하지 않는다")
            provider_result = deepcopy(row["terminal"]["provider"])
            response = _response_value(request, request_file, provider_result)
            if expected_response.exists() or expected_response.is_symlink():
                existing = _response(_read(expected_response, "response"), request, request_file)
                if existing != provider_result:
                    raise SpoolIntegrityError("기존 response와 terminal provider가 다르다")
            else:
                _write_new(expected_response, response)
            return provider_result
        if audit["pending"]:
            raise SpoolPendingError("다른 pending reservation이 있어 provider를 호출하지 않는다")
        if audit["remaining"] <= 0:
            raise SpoolCapError("local provider API cap 32를 소진했다")
        index = audit["reserved"] + 1
        reservation = {"schema": RESERVATION_SCHEMA, "index": index,
                       "job_id": request["job_id"], "sequence": request["sequence"],
                       "request_id": request["request_id"],
                       "canonical_sha256": request["canonical_sha256"],
                       "request_file_sha256": request_file_sha,
                       "reserved_at": datetime.now(timezone.utc).isoformat()}  # noqa: UP017 - Python 3.10
        reservation_dir = _child_directory(ledger, ("reservations",), create=True)
        reservation_path = reservation_dir / f"{index:04d}.json"
        _write_new(reservation_path, reservation)
        try:
            provider_result = provider(deepcopy(request["messages"]), deepcopy(request["params"]))
            if not isinstance(provider_result, dict):
                raise TypeError("provider result type")
            _json(provider_result, "provider")
        except Exception as error:  # noqa: BLE001 - 예외 class만 원장에 보존한다.
            failure_dir = _child_directory(ledger, ("failures",), create=True)
            error_type = f"{type(error).__module__}.{type(error).__qualname__}"
            _write_new(failure_dir / f"{index:04d}.json",
                       {"schema": FAILURE_SCHEMA, "index": index,
                        "reservation_sha256": file_sha(reservation_path),
                        "error_type": error_type})
            raise ProviderCallError(error_type) from None
        terminal_dir = _child_directory(ledger, ("terminals",), create=True)
        terminal = {"schema": TERMINAL_SCHEMA, "index": index,
                    "reservation_sha256": file_sha(reservation_path),
                    "provider": deepcopy(provider_result)}
        _write_new(terminal_dir / f"{index:04d}.json", terminal)
        _write_new(expected_response, _response_value(request, request_file, provider_result))
        return deepcopy(provider_result)
    finally:
        _unlock(descriptor, active)


def _response_value(request: dict[str, Any], request_path: Path,
                    provider: dict[str, Any]) -> dict[str, Any]:
    return {"schema": RESPONSE_SCHEMA, "job_id": request["job_id"],
            "sequence": request["sequence"], "request_id": request["request_id"],
            "canonical_sha256": request["canonical_sha256"],
            "request_file_sha256": file_sha(request_path), "provider": deepcopy(provider)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    process = sub.add_parser("process-one")
    process.add_argument("--request", type=Path, required=True)
    process.add_argument("--response", type=Path, required=True)
    process.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        process_one(args.request, args.response, args.ledger, OBSERVER.default_complete)
        print(json.dumps({"status": "complete"}))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI에는 실패 class만 공개한다.
        print(json.dumps({"status": "invalid", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
