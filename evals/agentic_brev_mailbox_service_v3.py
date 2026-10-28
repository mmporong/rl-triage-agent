"""Service v2 lifecycle/IO를 Remote3 Parent2 등록 경로에서 실행하는 제한 adapter."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import agentic_brev_mailbox_service_v2 as V2
import agentic_brev_probe_transport_remote_v3 as REMOTE

T, E, R, H = V2.T, V2.E, V2.R, V2.H
SCHEMA = V2.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE
SOURCES = frozenset(set(REMOTE.SOURCES) | set(H.SOURCES))
SOURCE = "evals/agentic_brev_mailbox_service_v3.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v3.py"
SOURCES = frozenset(set(SOURCES) | {SOURCE, TEST_SOURCE})
write_new, logical = V2.write_new, V2.logical

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE_REMOTE = V2.REMOTE
_BASE_REMOTE_SOURCE = V2.REMOTE_SOURCE
_BASE_SOURCES = V2.SOURCES
_BASE_FILE = V2.__file__


@contextmanager
def remote3_scope() -> Iterator[None]:
    """Service v2의 remote/source 전역을 한 호출 lifetime에만 v3로 교체한다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V2.REMOTE is not _BASE_REMOTE or V2.REMOTE_SOURCE != _BASE_REMOTE_SOURCE
                    or V2.SOURCES is not _BASE_SOURCES or V2.__file__ != _BASE_FILE):
                raise RuntimeError("Service v2 remote/source 전역이 이미 교체됐다")
            _STATE["saved"] = (V2.REMOTE, V2.REMOTE_SOURCE, V2.SOURCES, V2.__file__)
            V2.REMOTE, V2.REMOTE_SOURCE, V2.SOURCES, V2.__file__ = REMOTE, REMOTE_SOURCE, SOURCES, __file__
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V2.REMOTE, V2.REMOTE_SOURCE, V2.SOURCES, V2.__file__ = _STATE["saved"]
                _STATE["saved"] = None


class BrevMailboxService(V2.BrevMailboxService):
    def __init__(self, *args, **kwargs):
        with remote3_scope():
            super().__init__(*args, **kwargs)

    def _rpc(self, *args, **kwargs):
        with remote3_scope():
            return super()._rpc(*args, **kwargs)

    def _registered(self, *args, **kwargs):
        with remote3_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args, **kwargs):
        with remote3_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args, **kwargs):
    with remote3_scope():
        return V2.unpack_archive(*args, **kwargs)
