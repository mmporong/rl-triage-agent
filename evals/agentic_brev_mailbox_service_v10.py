"""Service9의 계약을 독립 Parent9/Remote9 진입점에 결합한다."""
from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / name).is_dir() for name in ("bench", "evals", "src")):
            return candidate
    raise RuntimeError("repository root unavailable")


ROOT = _repo_root()
for location in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import agentic_brev_mailbox_service_v9 as V9  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v9 as REMOTE  # pyright: ignore[reportMissingImports]

SOURCE = "evals/agentic_brev_mailbox_service_v10.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v10.py"
SOURCES = frozenset(set(V9.SOURCES) | set(REMOTE.SOURCES) | {SOURCE, TEST_SOURCE})
SCHEMA = V9.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}


@contextmanager
def service10_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            lower = (V9.V8.REMOTE, V9.V8.REMOTE_SOURCE, V9.V8.SOURCES, V9.V8.__file__)
            upper = (V9.REMOTE, V9.REMOTE_SOURCE, V9.SOURCES, V9.__file__, V9._BASE)
            _STATE["saved"] = (lower, upper)
            V9.REMOTE, V9.REMOTE_SOURCE, V9.SOURCES, V9.__file__ = (
                REMOTE, REMOTE_SOURCE, SOURCES, __file__,
            )
            V9._BASE = lower
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                lower, upper = _STATE["saved"]
                V9.REMOTE, V9.REMOTE_SOURCE, V9.SOURCES, V9.__file__, V9._BASE = upper
                _STATE["saved"] = None


class BrevMailboxService(V9.BrevMailboxService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with service10_scope():
            super().__init__(*args, **kwargs)

    def _budget(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super()._budget(*args, **kwargs)

    def _validate_host_intent(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super()._validate_host_intent(*args, **kwargs)

    def _provider(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super()._provider(*args, **kwargs)

    def _rpc(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super()._rpc(*args, **kwargs)

    def _registered(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args: Any, **kwargs: Any):
        with service10_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args: Any, **kwargs: Any):
    with service10_scope():
        return V9.unpack_archive(*args, **kwargs)


def __getattr__(name: str):
    return getattr(V9, name)
