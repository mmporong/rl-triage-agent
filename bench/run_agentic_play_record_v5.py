"""agentic play 녹화 실행기 v5: RUN4 렌더 sync 경로로 F1(보상 결함)·정상 같은 seed 정책을 재생한다.

RUN4·RUN3·RUN2·RUN1·RUN5(vision) 흐름과 lease·GPU reader·sidecar·AABA·종료·render-sync 계약은 그대로다.
``f1_record_scope()`` 안에서만 아래를 바꾸고 finally로 되돌린다.

- ``JOB3.f1_play_scope()``: v1 job 계약 전역(F1 등록·pair·schema·해석)
- ``RUN1.PLAYBACK``: ``agentic_f1_playback_v1``(NONE은 v1 callback, F1은 재생 주입 없음)
- ``RUN1._clip_consistent``: F1은 재생 주입이 없으므로 원본 DCMotor 함수여야 한다(NONE 규칙)
- ``RUN4.ENTRY``: RUN4가 RUN2 entry를 바꿀 대상. 자식 명령이 이 파일로 와서 같은 scope를 다시 연다.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_f1_playback_v1 as PLAYBACK
import agentic_play_job_v3 as JOB3
import run_agentic_play_record_v1 as RUN1
import run_agentic_play_record_v4 as RUN4

SOURCE_FILES = ("bench/run_agentic_play_record_v5.py", "tests/test_run_agentic_play_record_v5.py")
PRODUCTION = RUN1.PRODUCTION
FINAL_NAME = RUN1.FINAL_NAME
ENTRY = Path(__file__).resolve()
_LOCK = threading.RLock()
_BASE: dict[str, Any] = {"playback": RUN1.PLAYBACK, "clip": RUN1._clip_consistent, "entry": RUN4.ENTRY}
_STATE = {"depth": 0}


def clip_consistent(manifest: Mapping[str, Any] | None, fault_id: str) -> bool:
    """F1 재생은 모터 함수를 바꾸지 않는다. 그 밖의 값은 RUN1 규칙 그대로다."""
    return _BASE["clip"](manifest, "NONE" if fault_id == "F1" else fault_id)


@contextmanager
def f1_record_scope() -> Iterator[None]:
    with JOB3.f1_play_scope(), _LOCK:
        if _STATE["depth"] == 0:
            if (RUN1.PLAYBACK is not _BASE["playback"] or RUN1._clip_consistent is not _BASE["clip"]
                    or RUN4.ENTRY != _BASE["entry"]):
                raise RuntimeError("RUN1 playback·clip 또는 RUN4 entry가 이미 교체됐다")
            RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY = PLAYBACK, clip_consistent, ENTRY
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY = (
                    _BASE["playback"], _BASE["clip"], _BASE["entry"])


def run_job(job_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    with f1_record_scope():
        return RUN4.run_job(job_path, **kwargs)


def child_main(flag: str, job_path: Path) -> int:
    with f1_record_scope():
        return RUN4.child_main(flag, job_path)


def main(argv: Sequence[str] | None = None) -> int:
    with f1_record_scope():
        return RUN4.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
