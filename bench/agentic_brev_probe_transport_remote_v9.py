"""Remote8의 실행 계약을 독립 Parent9 CLI와 결합한다."""
from __future__ import annotations

import argparse
import json
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

import agentic_brev_probe_transport_remote_v8 as V8
import run_agentic_brev_probe_parent_v9 as PARENT

SCHEMA = "agentic_brev_probe_transport_remote_v9"
SOURCE = "bench/agentic_brev_probe_transport_remote_v9.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v9.py"
SERVICE_SOURCE = "evals/agentic_brev_mailbox_service_v10.py"
SERVICE_TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v10.py"
SOURCES = frozenset(set(V8.SOURCES) | set(PARENT.SOURCES) | {
    SOURCE, TEST_SOURCE, SERVICE_SOURCE, SERVICE_TEST_SOURCE,
})
PROVIDER_SCHEMA = PARENT.SCHEMA + "_provider_packet"
PRODUCTION = (PARENT.prepare, V8.PRODUCTION[1], V8.PRODUCTION[2])
MODES, ARMS = V8.MODES, V8.ARMS

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}


@contextmanager
def parent9_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            _STATE["saved"] = (V8.PARENT, V8.SOURCES, V8.PROVIDER_SCHEMA, V8.PRODUCTION)
            V8.PARENT = PARENT
            V8.SOURCES = SOURCES
            V8.PROVIDER_SCHEMA = PROVIDER_SCHEMA
            V8.PRODUCTION = PRODUCTION
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V8.PARENT, V8.SOURCES, V8.PROVIDER_SCHEMA, V8.PRODUCTION = _STATE["saved"]
                _STATE["saved"] = None


def run_transport(*args, **kwargs):
    kwargs.setdefault("preparer", PRODUCTION[0])
    with parent9_scope():
        value = V8.run_transport(*args, **kwargs)
    return {**value, "schema": SCHEMA}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, default="prepare")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--payload-base64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_transport(
        request=args.request, request_sha256=args.request_sha256, mode=args.mode,
        arm=args.arm, payload_base64=args.payload_base64, output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def __getattr__(name: str):
    return getattr(V8, name)


if __name__ == "__main__":
    main()
