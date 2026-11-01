"""Windows lease metadata의 일시적 공유 위반만 제한적으로 재시도해 읽는다."""
from __future__ import annotations

import errno
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_ATTEMPTS = 20
MAX_WAIT_S = 2.0
RETRY_INTERVAL_S = 0.05
TRANSIENT_WINERRORS = frozenset({5, 32, 33})


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _read_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def _transient_sharing_error(error: BaseException) -> bool:
    """임대 파일을 여는 동안 발생한 Windows 접근·공유 위반만 분류한다."""
    if not isinstance(error, PermissionError):
        return False
    winerror = getattr(error, "winerror", None)
    return winerror in TRANSIENT_WINERRORS or error.errno == errno.EACCES


def read_lease_metadata(
    path: Path,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    max_wait_s: float = MAX_WAIT_S,
    retry_interval_s: float = RETRY_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    reader: Callable[[Path], bytes] = _read_bytes,
) -> dict[str, Any]:
    """fresh JSON object를 읽고, transient PermissionError만 bounded 재시도한다.

    JSON 파싱 오류, JSON object가 아닌 값, 파일 부재와 그 밖의 I/O 오류는 첫
    시도에서 그대로 전달한다. 이전 metadata나 fallback lease는 사용하지 않는다.
    """
    target = Path(path)
    _require(type(max_attempts) is int and max_attempts >= 1, "max_attempts는 1 이상 정수")
    _require(_finite_number(max_wait_s) and float(max_wait_s) >= 0, "max_wait_s는 0 이상 유한 수치")
    _require(_finite_number(retry_interval_s) and float(retry_interval_s) >= 0,
             "retry_interval_s는 0 이상 유한 수치")
    started = monotonic()
    _require(_finite_number(started), "monotonic 시작값이 유한 수치가 아니다")
    last_error: PermissionError | None = None

    for attempt in range(1, max_attempts + 1):
        if last_error is not None:
            now = monotonic()
            _require(_finite_number(now) and now >= started,
                     "monotonic 시간이 뒤로 움직였거나 유한하지 않다")
            if float(now) - float(started) >= float(max_wait_s):
                raise last_error
        try:
            raw = reader(target)
        except PermissionError as error:
            if not _transient_sharing_error(error):
                raise
            last_error = error
            now = monotonic()
            _require(_finite_number(now) and now >= started,
                     "monotonic 시간이 뒤로 움직였거나 유한하지 않다")
            remaining = float(max_wait_s) - (float(now) - float(started))
            if attempt >= max_attempts or remaining <= 0:
                raise
            sleep(min(float(retry_interval_s), remaining))
            continue

        value = json.loads(raw)
        _require(isinstance(value, dict), "GPU lease metadata는 JSON object여야 한다")
        return value

    raise RuntimeError("lease metadata retry 상한 뒤 제어가 도달할 수 없다")
