"""Service6의 native budget과 Remote6 Parent4 실행 경계를 결합한다."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import agentic_brev_mailbox_service_v5 as V5
import agentic_brev_mailbox_service_v6 as V6
import agentic_brev_probe_transport_remote_v6 as REMOTE  # pyright: ignore[reportMissingImports]

SOURCE = "evals/agentic_brev_mailbox_service_v7.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v7.py"
SOURCES = frozenset(set(V6.SOURCES) | set(REMOTE.SOURCES) | {SOURCE, TEST_SOURCE})
SCHEMA = V5.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}
_BASE = (V5.REMOTE, V5.REMOTE_SOURCE, V5.SOURCES, V5.__file__)


@contextmanager
def service7_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V5.REMOTE, V5.REMOTE_SOURCE, V5.SOURCES, V5.__file__) != _BASE:
                raise RuntimeError("Service5 remote/source 전역이 이미 교체됐다")
            _STATE["saved"] = _BASE
            V5.REMOTE, V5.REMOTE_SOURCE, V5.SOURCES, V5.__file__ = (
                REMOTE, REMOTE_SOURCE, SOURCES, __file__,
            )
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V5.REMOTE, V5.REMOTE_SOURCE, V5.SOURCES, V5.__file__ = _STATE["saved"]
                _STATE["saved"] = None


class BrevMailboxService(V5.BrevMailboxService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with service7_scope():
            super().__init__(*args, **kwargs)

    def _budget(self, seconds: int = 0) -> None:
        original = self.contract
        reference = original.get("lifecycle_protocol")
        if not isinstance(reference, dict):
            raise TypeError("registered lifecycle reference")
        native = V6._native_lifecycle_reference(self.runtime, reference)
        self.contract = {**original, "lifecycle_protocol": native}
        try:
            V5.BrevMailboxService._budget(self, seconds)
        finally:
            self.contract = original

    def _rpc(self, *args: Any, **kwargs: Any):
        with service7_scope():
            return super()._rpc(*args, **kwargs)

    def _registered(self, *args: Any, **kwargs: Any):
        with service7_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args: Any, **kwargs: Any):
        with service7_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args: Any, **kwargs: Any):
    with service7_scope():
        return V5.unpack_archive(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(V5, name)
