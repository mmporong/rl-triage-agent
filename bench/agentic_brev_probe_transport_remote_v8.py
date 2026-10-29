"""Remote7 운반 계약을 실행 가능한 Parent8 fresh pair CLI와 결합한다."""
from __future__ import annotations

import argparse
import json
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import agentic_brev_probe_transport_remote_v7 as V7
import run_agentic_brev_probe_parent_v8 as PARENT

SCHEMA = "agentic_brev_probe_transport_remote_v8"
SOURCE = "bench/agentic_brev_probe_transport_remote_v8.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v8.py"
SERVICE_SOURCE = "evals/agentic_brev_mailbox_service_v9.py"
SERVICE_TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v9.py"
SOURCES = frozenset(set(V7.SOURCES) | set(PARENT.SOURCES) | {
    SOURCE, TEST_SOURCE, SERVICE_SOURCE, SERVICE_TEST_SOURCE,
})
PROVIDER_SCHEMA = PARENT.SCHEMA + "_provider_packet"
PRODUCTION = (PARENT.prepare, V7.PRODUCTION[1], V7.PRODUCTION[2])
MODES, ARMS = V7.MODES, V7.ARMS
V6: Any = V7.V6
V1: Any = V6.V1
V2: Any = V6.V2
require = V7.require

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}
_BASE = (V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION, V1.PROVIDER_SCHEMA)


@contextmanager
def parent8_scope() -> Iterator[None]:
    """Remote v1/v2의 parent와 import-time provider schema를 같은 Parent8로 바꾼다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            current = (V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION, V1.PROVIDER_SCHEMA)
            if current != _BASE:
                raise RuntimeError("Remote parent/provider 전역이 이미 교체됐다")
            _STATE["saved"] = current
            V1.P = V2.P = PARENT
            V1.PRODUCTION = V2.PRODUCTION = PRODUCTION
            V1.PROVIDER_SCHEMA = PROVIDER_SCHEMA
        _STATE["depth"] += 1
        try:
            with PARENT.parent8_scope():
                yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                (V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION,
                 V1.PROVIDER_SCHEMA) = _STATE["saved"]
                _STATE["saved"] = None


def _prepare(request, request_sha256, *, allow_fixture, home, now, preparer):
    with parent8_scope():
        prepared = V2._prepare(
            request, request_sha256, allow_fixture=allow_fixture, home=home, now=now,
            preparer=preparer,
        )
        registered = prepared["protocol"]["sources_lf"]
        require(SOURCES <= registered.keys() and all(
            registered[name] == PARENT.C.digest_lf_file(PARENT.ROOT / name)
            for name in SOURCES
        ), "v8 Parent8/transport/service source closure")
        return prepared


def _terminal_status(prepared: Mapping[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    terminal = V6.V5._terminal_pair_status(value)
    if terminal.get("status") != "pending":
        return terminal
    return V6._dead_parent_status(prepared, terminal)


def run_transport(*, request, request_sha256, mode="prepare", arm=None,
                  payload_base64=None, output=None, allow_fixture=False, home=None,
                  now=None, preparer=PRODUCTION[0],
                  popen_factory: Callable[..., Any] = PRODUCTION[1],
                  identity_fn=PRODUCTION[2], before_add=None):
    require(mode in MODES and type(allow_fixture) is bool, "transport mode/fixture flag")
    if not allow_fixture:
        require(preparer is PRODUCTION[0] and popen_factory is PRODUCTION[1]
                and identity_fn is PRODUCTION[2] and home is None and now is None
                and before_add is None, "actual transport dependency injection forbidden")
        require(os.name == "posix", "actual transport requires POSIX")
    prepared = _prepare(
        request, request_sha256, allow_fixture=allow_fixture, home=home, now=now,
        preparer=preparer,
    )
    with parent8_scope():
        if mode == "prepare":
            value = V1._public_prepare(prepared, request, request_sha256)
        elif mode == "start":
            require(arm is None and payload_base64 is None and output is None, "start arguments")
            value = V1._start(prepared, request, request_sha256, popen_factory, identity_fn)
        elif mode == "status":
            require(payload_base64 is None and output is None, "status arguments")
            value = _terminal_status(prepared, V1._status(prepared, arm))
        elif mode == "provider":
            require(payload_base64 is not None and output is None, "provider arguments")
            value = V1._provider(
                prepared, str(arm), cast(str, payload_base64),
                now or datetime.now(timezone.utc),  # noqa: UP017 - Python 3.10
            )
        else:
            require(output is not None and payload_base64 is None, "export arguments")
            value = V2.export(prepared, str(arm), output, before_add)
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
    return getattr(V7, name)


if __name__ == "__main__":
    main()
