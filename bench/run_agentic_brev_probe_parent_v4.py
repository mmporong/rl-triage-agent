"""Parent3의 등록 worker 경계를 독립 CLI import 경로에서 실행한다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    _value = str(_folder)
    if _value not in sys.path:
        sys.path.insert(0, _value)

import run_agentic_brev_probe_parent_v3 as V3  # noqa: E402 - 독립 CLI의 경로를 먼저 등록한다.

SCHEMA = V3.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v4.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v4.py"
SOURCES = frozenset(set(V3.SOURCES) | {SOURCE, TEST_SOURCE})

BACKEND = V3.BACKEND
C = V3.C
E = V3.E
U = V3.U
V2 = V3.V2
WORKER1 = V3.WORKER1
WORKER2 = V3.WORKER2
registered_worker_scope = V3.registered_worker_scope
verify_probe_equivalence = V3.verify_probe_equivalence


def prepare(*args, **kwargs):
    with registered_worker_scope():
        return V2.V1.prepare(*args, **kwargs)


def lookup(*args, **kwargs):
    with registered_worker_scope():
        return V2.V1.lookup(*args, **kwargs)


def run(*args, **kwargs):
    with registered_worker_scope():
        return V2.V1.run(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    with registered_worker_scope():
        return V2.V1._verify_completion(*args, **kwargs)


def main() -> None:
    with registered_worker_scope():
        V2.V1.main()


def __getattr__(name: str):
    return getattr(V3, name)


if __name__ == "__main__":
    main()
