"""agentic play 녹화 실행기 v6: RUN5 F1 경로를 JOB4 AWS target scope에 묶는다."""

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

import agentic_play_job_v4 as JOB4
import run_agentic_play_record_v4 as RUN4
import run_agentic_play_record_v5 as RUN5

SOURCE_FILES = ("bench/run_agentic_play_record_v6.py", "tests/test_run_agentic_play_record_v6.py")
PRODUCTION = RUN5.PRODUCTION
FINAL_NAME = RUN5.FINAL_NAME
ENTRY = Path(__file__).resolve()
_LOCK = threading.RLock()
_BASE_ENTRY = RUN4.ENTRY
_STATE = {"depth": 0}


@contextmanager
def aws_f1_record_scope() -> Iterator[None]:
    with JOB4.aws_f1_play_scope(), RUN5.f1_record_scope(), _LOCK:
        if _STATE["depth"] == 0:
            if RUN4.ENTRY != RUN5.ENTRY:
                raise RuntimeError("RUN5 entry가 이미 교체됐다")
            RUN4.ENTRY = ENTRY
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                RUN4.ENTRY = RUN5.ENTRY


def run_job(job_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    with aws_f1_record_scope():
        return RUN4.run_job(job_path, **kwargs)


def child_main(flag: str, job_path: Path) -> int:
    with aws_f1_record_scope():
        return RUN4.child_main(flag, job_path)


def main(argv: Sequence[str] | None = None) -> int:
    with aws_f1_record_scope():
        return RUN4.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
