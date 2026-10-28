"""Remote3를 Parent3 worker 선택과 Python 3.10 호환 UTC 경로에 결합한다."""
from __future__ import annotations

import argparse
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import agentic_brev_probe_transport_remote_v3 as V3
import run_agentic_brev_probe_parent_v3 as PARENT

SCHEMA = "agentic_brev_probe_transport_remote_v5"
SOURCE = "bench/agentic_brev_probe_transport_remote_v5.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v5.py"
SOURCES = frozenset(set(V3.SOURCES) | set(PARENT.SOURCES) | {
    SOURCE, TEST_SOURCE,
    "evals/agentic_brev_mailbox_service_v5.py", "tests/test_agentic_brev_mailbox_service_v5.py",
    "evals/agentic_brev_mailbox_transport_v2.py", "tests/test_agentic_brev_mailbox_transport_v2.py",
})
PRODUCTION = (PARENT.prepare, V3.V2.PRODUCTION[1], V3.V2.PRODUCTION[2])
MODES, ARMS = V3.MODES, V3.ARMS
V1: Any = V3.V1
V2: Any = V3.V2
require = V3.require

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE_PARENT = V1.P
_BASE_V1_PRODUCTION = V1.PRODUCTION
_BASE_V2_PRODUCTION = V2.PRODUCTION


@contextmanager
def parent3_scope() -> Iterator[None]:
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V1.P is not _BASE_PARENT or V2.P is not _BASE_PARENT
                    or V1.PRODUCTION is not _BASE_V1_PRODUCTION
                    or V2.PRODUCTION is not _BASE_V2_PRODUCTION):
                raise RuntimeError("Remote parent/production 전역이 이미 교체됐다")
            _STATE["saved"] = (V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION)
            V1.P = V2.P = PARENT
            V1.PRODUCTION = V2.PRODUCTION = PRODUCTION
        _STATE["depth"] += 1
        try:
            with PARENT.registered_worker_scope():
                yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION = _STATE["saved"]
                _STATE["saved"] = None


def _prepare(request, request_sha256, *, allow_fixture, home, now, preparer):
    with parent3_scope():
        prepared = V2._prepare(request, request_sha256, allow_fixture=allow_fixture,
                               home=home, now=now, preparer=preparer)
        registered = prepared["protocol"]["sources_lf"]
        require(SOURCES <= registered.keys() and all(
            registered[name] == PARENT.C.digest_lf_file(PARENT.ROOT / name) for name in SOURCES
        ), "v5 Parent3/transport/service source closure")
        return prepared


def _terminal_pair_status(value: dict[str, Any]) -> dict[str, Any]:
    pair = value.get("pair")
    if value.get("status") == "pending" and isinstance(pair, dict) and pair.get("status") == "hold":
        return {**value, "status": "hold", "terminal_source": "pair"}
    return value


def run_transport(*, request, request_sha256, mode="prepare", arm=None, payload_base64=None,
                  output=None, allow_fixture=False, home=None, now=None,
                  preparer=PRODUCTION[0], popen_factory: Callable[..., Any] = PRODUCTION[1],
                  identity_fn=PRODUCTION[2], before_add=None):
    require(mode in MODES and type(allow_fixture) is bool, "transport mode/fixture flag")
    if not allow_fixture:
        require(preparer is PRODUCTION[0] and popen_factory is PRODUCTION[1]
                and identity_fn is PRODUCTION[2] and home is None and now is None and before_add is None,
                "actual transport dependency injection forbidden")
        require(os.name == "posix", "actual transport requires POSIX")
    prepared = _prepare(request, request_sha256, allow_fixture=allow_fixture,
                        home=home, now=now, preparer=preparer)
    with parent3_scope():
        if mode == "prepare":
            value = V1._public_prepare(prepared, request, request_sha256)
        elif mode == "start":
            require(arm is None and payload_base64 is None and output is None, "start arguments")
            value = V1._start(prepared, request, request_sha256, popen_factory, identity_fn)
        elif mode == "status":
            require(payload_base64 is None and output is None, "status arguments")
            value = _terminal_pair_status(V1._status(prepared, arm))
        elif mode == "provider":
            require(payload_base64 is not None and output is None, "provider arguments")
            value = V1._provider(prepared, str(arm), cast(str, payload_base64),
                                 now or datetime.now(timezone.utc))  # noqa: UP017
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
    try:
        result = run_transport(request=args.request, request_sha256=args.request_sha256,
                               mode=args.mode, arm=args.arm, payload_base64=args.payload_base64,
                               output=args.output)
    except Exception as error:  # noqa: BLE001 - 상위 broker가 구조화한 HOLD를 보존한다.
        result = {"schema": SCHEMA, "status": "HOLD", "dispatched": False,
                  "error_type": type(error).__name__, "reason": str(error),
                  "authority_issued": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def __getattr__(name: str):
    return getattr(V3, name)


if __name__ == "__main__":
    main()
