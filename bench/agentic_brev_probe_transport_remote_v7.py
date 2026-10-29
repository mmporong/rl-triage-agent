"""Remote6의 Parent4 경계에 terminal pair HOLD 투영을 복원한다."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import agentic_brev_probe_transport_remote_v6 as V6

SCHEMA = "agentic_brev_probe_transport_remote_v7"
SOURCE = "bench/agentic_brev_probe_transport_remote_v7.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v7.py"
SERVICE_SOURCE = "evals/agentic_brev_mailbox_service_v8.py"
SERVICE_TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v8.py"
SOURCES = frozenset(set(V6.SOURCES) | {
    SOURCE, TEST_SOURCE, SERVICE_SOURCE, SERVICE_TEST_SOURCE,
})
PRODUCTION = V6.PRODUCTION
MODES, ARMS = V6.MODES, V6.ARMS
V1: Any = V6.V1
V2: Any = V6.V2
require = V6.require


def _terminal_status(
    prepared: Mapping[str, Any], value: dict[str, Any],
) -> dict[str, Any]:
    terminal = V6.V5._terminal_pair_status(value)
    if terminal.get("status") != "pending":
        return terminal
    return V6._dead_parent_status(prepared, terminal)


def _prepare(request, request_sha256, *, allow_fixture, home, now, preparer):
    prepared = V6._prepare(
        request, request_sha256, allow_fixture=allow_fixture, home=home, now=now,
        preparer=preparer,
    )
    registered = prepared["protocol"]["sources_lf"]
    require(SOURCES <= registered.keys() and all(
        registered[name] == V6.PARENT.C.digest_lf_file(V6.PARENT.ROOT / name)
        for name in SOURCES
    ), "v7 Parent4/transport/service source closure")
    return prepared


def run_transport(*, request, request_sha256, mode="prepare", arm=None, payload_base64=None,
                  output=None, allow_fixture=False, home=None, now=None,
                  preparer=PRODUCTION[0], popen_factory: Callable[..., Any] = PRODUCTION[1],
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
    with V6.parent4_scope():
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
    try:
        result = run_transport(
            request=args.request, request_sha256=args.request_sha256, mode=args.mode,
            arm=args.arm, payload_base64=args.payload_base64, output=args.output,
        )
    except Exception as error:  # noqa: BLE001 - broker가 구조화한 HOLD를 보존한다.
        result = {
            "schema": SCHEMA, "status": "HOLD", "dispatched": False,
            "error_type": type(error).__name__, "reason": str(error),
            "authority_issued": False,
        }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def __getattr__(name: str):
    return getattr(V6, name)


if __name__ == "__main__":
    main()
