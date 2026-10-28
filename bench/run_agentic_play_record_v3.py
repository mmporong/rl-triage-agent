"""agentic play 녹화 실행기 v3: 부모 GPU reader의 관찰 원문 참조(``last_receipt``)를 RUN5까지 전달한다.

RUN1 ``_leased_reader``는 ``StrictRemoteGpuReader``를 함수 closure로 감싸 ``__call__``만 남긴다. 그래서 AABA 뒤
``BASE._gpu_reference``가 ``last_receipt``를 찾지 못하고 ``TypeError``로 멈췄다(v8 실제 실행).

- ``leased_reader_scope()`` 안에서만 ``RUN1._leased_reader``를 ``LeasedReader`` proxy로 바꾸고 finally로 되돌린다.
- proxy 호출은 RUN1 closure와 같다: lease 활성 확인 뒤 원래 reader 호출.
- ``last_receipt``는 접근마다 lease를 확인하고 underlying의 현재 참조 객체를 그대로 돌려준다. 복사·재조립·캐시를
  하지 않으며, 원문 SHA·resource·GPU 검증은 기존 ``_gpu_reference``가 한다. 참조가 없으면 기존처럼 거부된다.
- 실행 본체·자식 entry·live 판정은 RUN2·RUN1 그대로다.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_agentic_play_record_v1 as RUN1
import run_agentic_play_record_v2 as RUN2

SOURCE_FILES = ("bench/run_agentic_play_record_v3.py", "tests/test_run_agentic_play_record_v3.py")
PRODUCTION = RUN1.PRODUCTION
FINAL_NAME = RUN1.FINAL_NAME
_LOCK = threading.RLock()
_BASE_LEASED_READER = RUN1._leased_reader
_STATE = {"depth": 0}


class LeasedReader:
    """lease 확인을 앞에 둔 reader proxy. 관찰 원문 참조만 underlying에서 그대로 읽는다."""

    __slots__ = ("_check", "_reader")

    def __init__(self, reader: Callable[[], list[int]], check: Callable[[], Any]) -> None:
        self._reader = reader
        self._check = check

    def __call__(self) -> list[int]:
        self._check()  # GPU reader 호출 전마다 lease 활성 확인(RUN1과 같다)
        return self._reader()

    @property
    def last_receipt(self) -> Mapping[str, Any] | None:
        self._check()
        return getattr(self._reader, "last_receipt", None)


def _leased_reader(reader: Callable[[], list[int]], check: Callable[[], Any]) -> LeasedReader:
    return LeasedReader(reader, check)


@contextmanager
def leased_reader_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            if RUN1._leased_reader is not _BASE_LEASED_READER:
                raise RuntimeError("RUN1 leased reader가 이미 교체됐다")
            RUN1._leased_reader = _leased_reader
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                RUN1._leased_reader = _BASE_LEASED_READER


def run_job(job_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    with leased_reader_scope():
        return RUN2.run_job(job_path, **kwargs)


def child_main(flag: str, job_path: Path) -> int:
    with leased_reader_scope():
        return RUN2.child_main(flag, job_path)


def main(argv: Sequence[str] | None = None) -> int:
    with leased_reader_scope():
        return RUN2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
