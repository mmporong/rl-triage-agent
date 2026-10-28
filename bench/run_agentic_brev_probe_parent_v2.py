"""Parent v1의 검사 lifetime을 JOB4 AWS F1 등록 경로에서 실행하는 제한 adapter."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import agentic_play_job_v4 as JOB4
import run_agentic_brev_probe_parent_v1 as V1
import run_agentic_capture_unit_v8 as CAPTURE8

SCHEMA = V1.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v2.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v2.py"
SOURCES = frozenset(
    set(V1.SOURCES) | set(JOB4.SOURCE_FILES) | set(CAPTURE8.SOURCE_FILES) | {SOURCE, TEST_SOURCE}
)

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE_JOB = V1.JOB2
_BASE_SOURCES = V1.SOURCES


@contextmanager
def aws_parent_scope() -> Iterator[None]:
    """Parent v1의 JOB2 참조와 source closure만 JOB4 등록값으로 교체하고 복원한다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if V1.JOB2 is not _BASE_JOB or V1.SOURCES is not _BASE_SOURCES:
                raise RuntimeError("Parent v1 전역이 이미 교체됐다")
            _STATE["saved"] = (V1.JOB2, V1.SOURCES)
            V1.JOB2, V1.SOURCES = JOB4, SOURCES
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V1.JOB2, V1.SOURCES = _STATE["saved"]
                _STATE["saved"] = None


def scope_active() -> bool:
    return _STATE["depth"] > 0


def prepare(*args, **kwargs):
    with aws_parent_scope():
        return V1.prepare(*args, **kwargs)


def lookup(*args, **kwargs):
    with aws_parent_scope():
        return V1.lookup(*args, **kwargs)


def run(*args, **kwargs):
    with aws_parent_scope():
        return V1.run(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    with aws_parent_scope():
        return V1._verify_completion(*args, **kwargs)


def main() -> None:
    with aws_parent_scope():
        V1.main()


def __getattr__(name: str):
    return getattr(V1, name)


if __name__ == "__main__":
    main()
