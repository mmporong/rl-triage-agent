"""Service3 lifecycle/IO를 Remote4 terminal HOLD 경로에서 실행한다."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import agentic_brev_mailbox_service_v3 as V3  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v4 as REMOTE

SOURCE = "evals/agentic_brev_mailbox_service_v4.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v4.py"
SOURCES = frozenset(set(REMOTE.SOURCES) | set(V3.H.SOURCES) | {SOURCE, TEST_SOURCE})
SCHEMA = V3.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE = (V3.REMOTE, V3.REMOTE_SOURCE, V3.SOURCES, V3.__file__)


@contextmanager
def remote4_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V3.REMOTE, V3.REMOTE_SOURCE, V3.SOURCES, V3.__file__) != _BASE:
                raise RuntimeError("Service3 remote/source 전역이 이미 교체됐다")
            _STATE["saved"] = _BASE
            V3.REMOTE, V3.REMOTE_SOURCE, V3.SOURCES, V3.__file__ = REMOTE, REMOTE_SOURCE, SOURCES, __file__
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V3.REMOTE, V3.REMOTE_SOURCE, V3.SOURCES, V3.__file__ = _STATE["saved"]
                _STATE["saved"] = None


class BrevMailboxService(V3.BrevMailboxService):
    def __init__(self, *args, **kwargs):
        with remote4_scope():
            super().__init__(*args, **kwargs)

    def _rpc(self, *args, **kwargs):
        with remote4_scope():
            return super()._rpc(*args, **kwargs)

    def _registered(self, *args, **kwargs):
        with remote4_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args, **kwargs):
        with remote4_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args, **kwargs):
    with remote4_scope():
        return V3.unpack_archive(*args, **kwargs)


def __getattr__(name: str):
    return getattr(V3, name)
