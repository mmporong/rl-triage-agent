"""Service3를 Remote5와 제한된 인프라 실패 원문 보존에 결합한다."""
from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import agentic_brev_mailbox_service_v3 as V3  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v5 as REMOTE  # pyright: ignore[reportMissingImports]

SOURCE = "evals/agentic_brev_mailbox_service_v5.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v5.py"
SOURCES = frozenset(set(REMOTE.SOURCES) | set(V3.H.SOURCES) | {SOURCE, TEST_SOURCE})
SCHEMA = V3.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE = (V3.REMOTE, V3.REMOTE_SOURCE, V3.SOURCES, V3.__file__)


class RemoteInfrastructureFailure(ValueError):
    def __init__(self, code: str, raw_rpc: dict[str, str]):
        self.code, self.raw_rpc = code, dict(raw_rpc)
        super().__init__(json.dumps({"code": code, "raw_rpc": self.raw_rpc},
                                    ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@contextmanager
def remote5_scope() -> Iterator[None]:
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


def _latest_rpc(folder: Path, mode: str) -> Path:
    rows = sorted(folder.glob(f"rpc-{mode}-*.json"), key=lambda path: path.stat().st_mtime_ns)
    V3.T.require(bool(rows), "remote RPC raw response missing")
    return rows[-1]


def _failure_code(raw: dict) -> str:
    if raw.get("timed_out") is True:
        return "remote_helper_timeout"
    stderr = base64.b64decode(raw.get("stderr_base64", ""), validate=True).decode("utf-8", errors="replace")
    if "cannot import name 'UTC'" in stderr and "datetime.py" in stderr:
        return "remote_python_compat_import"
    return "remote_helper_exit_nonzero"


class BrevMailboxService(V3.BrevMailboxService):
    def __init__(self, *args, **kwargs):
        with remote5_scope():
            super().__init__(*args, **kwargs)

    def _rpc(self, folder, plan, registration, mode, *extra):
        with remote5_scope():
            try:
                return super()._rpc(folder, plan, registration, mode, *extra)
            except ValueError as error:
                if str(error) != "remote helper failed; raw preserved":
                    raise
                path = _latest_rpc(Path(folder), mode)
                raw = V3.E.loads(path.read_text(encoding="utf-8"))
                reference = {"path": str(path.resolve(strict=True)), "sha256": V3.E.J.file_sha(path)}
                raise RemoteInfrastructureFailure(_failure_code(raw), reference) from error

    def _registered(self, *args, **kwargs):
        with remote5_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args, **kwargs):
        with remote5_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args, **kwargs):
    with remote5_scope():
        return V3.unpack_archive(*args, **kwargs)


def __getattr__(name: str):
    return getattr(V3, name)
