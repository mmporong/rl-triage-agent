"""새 seed optimizer 확인 실험용 계측 adapter.

기존 계측 구현은 그대로 쓰되 manifest가 새 사전등록 protocol과 이 adapter를
결합하도록 경계만 바꾼다.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import inspect_optimizer_observability as BASE  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/novel_optimizer_confirmation_v1.json"
BOUND_SOURCES = (
    ROOT / "evals/optimizer_observability.py",
    ROOT / "bench/inspect_optimizer_observability.py",
    ROOT / "bench/hidden_faults.py",
    Path(__file__).resolve(),
    PROTOCOL,
)


@contextmanager
def binding():
    previous = BASE.PROTOCOL, BASE.BOUND_SOURCES
    BASE.PROTOCOL, BASE.BOUND_SOURCES = PROTOCOL, BOUND_SOURCES
    try:
        yield BASE
    finally:
        BASE.PROTOCOL, BASE.BOUND_SOURCES = previous


def main(argv=None) -> int:
    with binding():
        return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
