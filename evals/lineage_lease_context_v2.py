"""공용 GPU lease metadata 읽기의 접근 오류만 제한 재시도한다."""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_source_batch_v1 as B

SCHEMA = "lineage_lease_context_read_evidence_v2"
EVIDENCE_NAME = "lease-context-read-evidence.json"
MAX_ATTEMPTS = 20
MAX_WAIT_S = 2.0
RETRY_INTERVAL_S = 0.1
SOURCE_FILES = (
    "evals/lineage_lease_context_v2.py",
    "tests/test_lineage_lease_context_v2.py",
)
_REGISTERED_READ = B.read
_READ = B.read
_LEASE_CONTEXT = B.lease_context
_CLOCK = time.monotonic
_SLEEP = time.sleep
_LOCK = RLock()
_LOCAL = threading.local()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def canonical_path() -> Path:
    return (Path.home() / "CodexSignals" / "gpu-lease.json").resolve()


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _same_path(left: object, right: Path) -> bool:
    if not isinstance(left, (str, os.PathLike)):
        return False
    try:
        return Path(left).resolve() == right
    except (OSError, TypeError, ValueError):
        return False


def _eligible(error: BaseException, path: Path) -> bool:
    return (
        isinstance(error, PermissionError)
        and error.errno == errno.EACCES
        and path.resolve() == canonical_path()
        and _same_path(getattr(error, "filename", None), canonical_path())
    )


def _event(call: int, attempt: int, *, status: str,
           error: BaseException | None = None) -> dict:
    return {
        "call": call,
        "attempt": attempt,
        "status": status,
        "type": None if error is None else type(error).__name__,
        "errno": None if error is None else getattr(error, "errno", None),
        "winerror": None if error is None else getattr(error, "winerror", None),
        "filename": None if error is None else getattr(error, "filename", None),
    }


def _retry_read(path: Path) -> dict:
    path = Path(path)
    events = getattr(_LOCAL, "events", None)
    if not isinstance(events, list):
        raise TypeError("lease read retry scope가 없다")
    call = getattr(_LOCAL, "calls", 0) + 1
    _LOCAL.calls = call
    started = _CLOCK()
    _require(_finite(started), "lease read retry clock이 유한 수치가 아니다")
    last_error: PermissionError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if last_error is not None:
            now = _CLOCK()
            _require(_finite(now) and now >= started,
                     "lease read retry clock이 뒤로 움직였거나 유한하지 않다")
            if now - started >= MAX_WAIT_S:
                raise last_error
        try:
            value = _READ(path)
        except PermissionError as error:
            last_error = error
            events.append(_event(call, attempt, status="permission_error", error=error))
            if not _eligible(error, path):
                raise
            now = _CLOCK()
            _require(_finite(now) and now >= started,
                     "lease read retry clock이 뒤로 움직였거나 유한하지 않다")
            remaining = MAX_WAIT_S - (now - started)
            if attempt >= MAX_ATTEMPTS or remaining <= 0:
                raise
            _SLEEP(min(RETRY_INTERVAL_S, remaining))
        else:
            _require(isinstance(value, dict), "GPU lease metadata는 JSON 객체여야 한다")
            events.append(_event(call, attempt, status="fresh_success"))
            return value
    raise RuntimeError("lease read retry 상한 뒤 제어가 도달할 수 없다")


@contextmanager
def read_scope():
    """현재 thread에서만 B.read를 canonical retry reader로 바꾸고 복원한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(B.read is _retry_read, "재진입 중 lease reader가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(B.read is _REGISTERED_READ, "lease reader가 등록 원본이 아니다")
        original = B.read
        B.read = _retry_read
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            B.read = original


def _failure(error: BaseException) -> dict:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "errno": getattr(error, "errno", None),
        "winerror": getattr(error, "winerror", None),
        "filename": getattr(error, "filename", None),
    }


def _evidence(events: list[dict], *, status: str,
              failure: dict | None = None) -> dict:
    permission_errors = [row for row in events if row["status"] == "permission_error"]
    successes = [row for row in events if row["status"] == "fresh_success"]
    return {
        "schema": SCHEMA,
        "status": status,
        "canonical_path": str(canonical_path()),
        "limits": {
            "max_attempts_per_read": MAX_ATTEMPTS,
            "max_wait_s_per_read": MAX_WAIT_S,
            "retry_interval_s": RETRY_INTERVAL_S,
        },
        "attempts": events,
        "permission_error_attempts": len(permission_errors),
        "fresh_read_success": bool(successes),
        "fresh_read_failure": status == "failed",
        "failure": failure,
        "cached_metadata_accepted": False,
    }


def lease_context(evidence_dir: Path | None = None) -> dict:
    """기존 lease 검증을 유지하고 canonical metadata EACCES만 제한 재시도한다."""
    evidence_path = None
    if evidence_dir is not None:
        evidence_path = Path(evidence_dir).resolve() / EVIDENCE_NAME
        if evidence_path.exists():
            raise FileExistsError(f"lease read evidence가 이미 존재한다: {evidence_path}")
    events: list[dict] = []
    _LOCAL.events = events
    _LOCAL.calls = 0
    try:
        with read_scope():
            result = _LEASE_CONTEXT()
    except BaseException as error:
        if evidence_path is not None:
            _write_new(evidence_path, _evidence(events, status="failed", failure=_failure(error)))
        raise
    else:
        if evidence_path is not None:
            _write_new(evidence_path, _evidence(events, status="complete"))
        return result
    finally:
        _LOCAL.events = None
        _LOCAL.calls = 0
