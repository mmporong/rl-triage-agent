"""Parent2 AWS 경계에서 등록 worker1을 probe lifetime에만 선택한다."""
from __future__ import annotations

import ast
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import brev_recovery_backend_v4 as BACKEND  # pyright: ignore[reportMissingImports]
import run_agentic_brev_probe_parent_v2 as V2

SCHEMA = V2.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v3.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v3.py"
SOURCES = frozenset(set(V2.SOURCES) | {SOURCE, TEST_SOURCE})
WORKER1 = V2.V1.ROOT / "bench/brev_recovery_worker_v1.py"
WORKER2 = V2.V1.ROOT / "bench/brev_recovery_worker_v2.py"

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE_WORKER = BACKEND.WORKER


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"{name} 함수가 정확히 하나여야 한다")
    return ast.dump(nodes[0], include_attributes=False)


def verify_probe_equivalence() -> None:
    if _function_ast(WORKER1, "run_probe") != _function_ast(WORKER2, "run_probe"):
        raise RuntimeError("worker1/worker2 run_probe 구현이 다르다")


@contextmanager
def registered_worker_scope() -> Iterator[None]:
    """등록 argv의 worker1만 선택하고 BoundedPopen 검사는 그대로 둔다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if BACKEND.WORKER != _BASE_WORKER or _BASE_WORKER != WORKER2:
                raise RuntimeError("backend worker 전역이 이미 교체됐다")
            verify_probe_equivalence()
            _STATE["saved"] = BACKEND.WORKER
            BACKEND.WORKER = WORKER1
        _STATE["depth"] += 1
        try:
            with V2.aws_parent_scope():
                yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                BACKEND.WORKER = _STATE["saved"]
                _STATE["saved"] = None


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
    return getattr(V2.V1, name)


if __name__ == "__main__":
    main()
