"""Remote v2 증거 운반을 Parent2 AWS F1 entry와 등록 closure에 결합한다."""
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

import agentic_brev_probe_transport_remote_v2 as V2
import run_agentic_brev_probe_parent_v2 as PARENT

SCHEMA = "agentic_brev_probe_transport_remote_v3"
ARCHIVE_SCHEMA = V2.ARCHIVE_SCHEMA
SOURCE = "bench/agentic_brev_probe_transport_remote_v3.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v3.py"
SOURCES = frozenset(set(V2.SOURCES) | set(PARENT.SOURCES) | {
    SOURCE, TEST_SOURCE,
    "evals/agentic_brev_mailbox_service_v3.py", "tests/test_agentic_brev_mailbox_service_v3.py",
})
PRODUCTION = (PARENT.prepare, V2.PRODUCTION[1], V2.PRODUCTION[2])
MODES, ARMS = V2.MODES, V2.ARMS
V1 = V2.V1
P = PARENT
LAYERS = V2.LAYERS
OutsideRoots = V2.OutsideRoots
producer_path = V2.producer_path
validate_maps = V2.validate_maps
maps_for_export = V2.maps_for_export
maps_for_host = V2.maps_for_host
locate = V2.locate
require = V2.require

_LOCK = threading.RLock()
_STATE = {"depth": 0, "saved": None}
_BASE_PARENT = V1.P
_BASE_V1_PRODUCTION = V1.PRODUCTION
_BASE_V2_PRODUCTION = V2.PRODUCTION


@contextmanager
def parent2_scope() -> Iterator[None]:
    """Remote v1/v2가 참조하는 parent module만 Parent2로 교체하고 중첩 후 복원한다."""
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
            with PARENT.aws_parent_scope():
                yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V1.P, V2.P, V1.PRODUCTION, V2.PRODUCTION = _STATE["saved"]
                _STATE["saved"] = None


def _prepare(request, request_sha256, *, allow_fixture, home, now, preparer):
    with parent2_scope():
        prepared = V2._prepare(
            request, request_sha256, allow_fixture=allow_fixture, home=home, now=now, preparer=preparer
        )
        registered = prepared["protocol"]["sources_lf"]
        require(SOURCES <= registered.keys() and all(
            registered[name] == PARENT.C.digest_lf_file(PARENT.ROOT / name) for name in SOURCES
        ), "v3 Parent2/transport/service source closure")
        return prepared


def export(*args, **kwargs):
    with parent2_scope():
        return V2.export(*args, **kwargs)


def run_transport(*, request, request_sha256, mode="prepare", arm=None, payload_base64=None, output=None,
                  allow_fixture=False, home=None, now=None, preparer=PRODUCTION[0], popen_factory: Callable[..., Any]=PRODUCTION[1],
                  identity_fn=PRODUCTION[2], before_add=None):
    require(mode in MODES and type(allow_fixture) is bool, "transport mode/fixture flag")
    if not allow_fixture:
        require(
            preparer is PRODUCTION[0] and popen_factory is PRODUCTION[1] and identity_fn is PRODUCTION[2]
            and home is None and now is None and before_add is None,
            "actual transport dependency injection forbidden",
        )
        require(os.name == "posix", "actual transport requires POSIX")
    prepared = _prepare(request, request_sha256, allow_fixture=allow_fixture, home=home, now=now, preparer=preparer)
    with parent2_scope():
        if mode == "prepare":
            value = V1._public_prepare(prepared, request, request_sha256)
        elif mode == "start":
            require(arm is None and payload_base64 is None and output is None, "start arguments")
            value = V1._start(prepared, request, request_sha256, popen_factory, identity_fn)
        elif mode == "status":
            require(payload_base64 is None and output is None, "status arguments")
            value = V1._status(prepared, arm)
        elif mode == "provider":
            require(payload_base64 is not None and output is None, "provider arguments")
            value = V1._provider(prepared, str(arm), cast(str, payload_base64), now or datetime.now(timezone.utc))
        else:
            require(output is not None and payload_base64 is None, "export arguments")
            value = export(prepared, str(arm), output, before_add)
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
        result = run_transport(request=args.request, request_sha256=args.request_sha256, mode=args.mode,
                               arm=args.arm, payload_base64=args.payload_base64, output=args.output)
    except Exception as error:  # noqa: BLE001 - 원문을 상위 broker가 보존한다.
        result = {"schema": SCHEMA, "status": "HOLD", "dispatched": False,
                  "error_type": type(error).__name__, "reason": str(error), "authority_issued": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
