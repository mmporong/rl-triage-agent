"""Service5를 공개 lifecycle 상대 참조의 실제 budget 소비에 결합한다."""
from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import agentic_brev_mailbox_service_v5 as V5

SOURCE = "evals/agentic_brev_mailbox_service_v6.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v6.py"
SOURCES = frozenset(set(V5.SOURCES) | {SOURCE, TEST_SOURCE})
SCHEMA = V5.SCHEMA
REMOTE_SOURCE = V5.REMOTE_SOURCE

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}
_BASE = (V5.SOURCES, V5.__file__)


@contextmanager
def service6_scope() -> Iterator[None]:
    """하위 constructor가 등록 Service6 source를 계산하는 동안만 V5 source 전역을 교체한다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V5.SOURCES, V5.__file__) != _BASE:
                raise RuntimeError("Service5 source 전역이 이미 교체됐다")
            _STATE["saved"] = _BASE
            V5.SOURCES, V5.__file__ = SOURCES, __file__
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V5.SOURCES, V5.__file__ = _STATE["saved"]
                _STATE["saved"] = None


def _native_lifecycle_reference(runtime: Any, reference: dict[str, Any]) -> dict[str, str]:
    V5.V3.T.require(set(reference) == {"path", "sha256"}, "lifecycle reference fields")
    raw_path = reference["path"]
    V5.V3.T.require(isinstance(raw_path, str) and raw_path != "" and "\\" not in raw_path
                    and ":" not in raw_path, "public lifecycle repo-relative path required")
    relative = PurePosixPath(raw_path)
    V5.V3.T.require(not relative.is_absolute()
                    and all(part not in {"", ".", ".."} for part in relative.parts),
                    "public lifecycle repo-relative path required")
    repo = Path(runtime.repo).resolve(strict=True)
    native = repo.joinpath(*relative.parts)
    V5.V3.T.regular(native)
    resolved = native.resolve(strict=True)
    V5.V3.T.require(resolved == native and repo in resolved.parents,
                    "public lifecycle path outside repository")
    raw = native.read_bytes()
    V5.V3.T.require(hashlib.sha256(raw).hexdigest() == reference["sha256"],
                    "public lifecycle raw SHA changed")
    lf_sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    V5.V3.R.verify_sources(
        {"sources_lf": {relative.as_posix(): lf_sha}}, runtime.anchor, repo=repo,
    )
    return {"path": str(native), "sha256": reference["sha256"]}


class BrevMailboxService(V5.BrevMailboxService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with service6_scope():
            super().__init__(*args, **kwargs)

    def _budget(self, seconds: int = 0) -> None:
        original = self.contract
        reference = original.get("lifecycle_protocol")
        if not isinstance(reference, dict):
            raise TypeError("registered lifecycle reference")
        native = _native_lifecycle_reference(self.runtime, reference)
        adjusted = dict(original)
        adjusted["lifecycle_protocol"] = native
        self.contract = adjusted
        try:
            super()._budget(seconds)
        finally:
            self.contract = original


def __getattr__(name: str) -> Any:
    return getattr(V5, name)
