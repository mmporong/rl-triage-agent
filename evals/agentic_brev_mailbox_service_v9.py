"""Service8을 등록 Host 판·Parent8 Remote·원 비용 기산점에 결합한다."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import agentic_brev_mailbox_service_v8 as V8
import agentic_brev_probe_transport_remote_v3 as REMOTE3  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v8 as REMOTE  # pyright: ignore[reportMissingImports]
import agentic_brev_runtime_host_v1 as H1
import agentic_brev_runtime_host_v6 as H6

SOURCE = "evals/agentic_brev_mailbox_service_v9.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_service_v9.py"
SOURCES = frozenset(set(V8.SOURCES) | set(REMOTE.SOURCES) | set(H6.SOURCES) | {
    SOURCE, TEST_SOURCE,
})
SCHEMA = V8.SCHEMA
REMOTE_SOURCE = REMOTE.SOURCE

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "saved": None}
_BASE = (V8.REMOTE, V8.REMOTE_SOURCE, V8.SOURCES, V8.__file__)


@contextmanager
def service9_scope() -> Iterator[None]:
    """Service8의 source/remote 선택만 바꾸고 하위 scope의 복원 경계를 유지한다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if (V8.REMOTE, V8.REMOTE_SOURCE, V8.SOURCES, V8.__file__) != _BASE:
                raise RuntimeError("Service8 remote/source 전역이 이미 교체됐다")
            _STATE["saved"] = _BASE
            V8.REMOTE, V8.REMOTE_SOURCE, V8.SOURCES, V8.__file__ = (
                REMOTE, REMOTE_SOURCE, SOURCES, __file__,
            )
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                V8.REMOTE, V8.REMOTE_SOURCE, V8.SOURCES, V8.__file__ = _STATE["saved"]
                _STATE["saved"] = None


def _host_for_registration(registration: dict[str, Any], binding: dict[str, Any],
                           protocol: dict[str, Any]):
    runtime_binding = binding.get("expected_request", {}).get("execution", {}).get(
        "runtime_host_binding", {},
    )
    schema = runtime_binding.get("schema")
    if schema == H1.SCHEMA + "_runtime_binding":
        V8.T.require(protocol.get("schema") == H1.P.SCHEMA + "_protocol",
                     "Host1 parent protocol schema")
        return H1.BrevRuntimeHost
    if schema == H6.SCHEMA + "_runtime_binding":
        V8.T.require(protocol.get("schema") == H6.PARENT_SCHEMA + "_protocol"
                     and H6.SOURCES <= protocol.get("sources_lf", {}).keys(),
                     "Host6 parent protocol/source closure")
        return H6.BrevRuntimeHost
    raise ValueError("registered runtime host schema unsupported")


def _validate_budget_origin(phase: dict[str, Any], contract: dict[str, Any], *,
                            seconds: int, now: datetime) -> None:
    start = V8.E._stamp(phase["registered_at_utc"])
    deadline = V8.E._stamp(phase["deadline_utc"])
    V8.T.require(contract.get("deadline_utc") == phase["deadline_utc"],
                 "contract/lifecycle deadline differs")
    maximum = Decimal(str(phase["max_lifecycle_s"]))
    reserve = Decimal(str(phase["stop_reserve_s"]))
    span = Decimal(str((deadline - start).total_seconds()))
    remaining = Decimal(str((deadline - now).total_seconds()))
    V8.T.require(maximum.is_finite() and reserve.is_finite()
                 and maximum > 0 and reserve > 0 and span == maximum,
                 "lifecycle origin must equal deadline minus max lifecycle")
    V8.T.require(remaining >= Decimal(seconds) + reserve,
                 "deadline must preserve requested work and Stop reserve")


class BrevMailboxService(V8.BrevMailboxService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with service9_scope():
            super().__init__(*args, **kwargs)

    def _budget(self, seconds: int = 0) -> None:
        reference = self.contract.get("lifecycle_protocol")
        if not isinstance(reference, dict):
            raise TypeError("registered lifecycle reference")
        native = V8.V6._native_lifecycle_reference(self.runtime, reference)
        phase = V8.T.read_ref(native)
        _validate_budget_origin(
            phase, self.contract, seconds=seconds,
            now=datetime.now(timezone.utc),  # noqa: UP017 - Python 3.10
        )
        super()._budget(seconds)

    def _validate_host_intent(self, request: dict) -> None:
        operation = request["operation_id"]
        path = self.runtime.folder / "brev-host" / operation / "intent.json"
        V8.T.regular(path)
        value = V8.E.loads(path.read_text(encoding="utf-8"))
        V8.T.require(value["registration"] == request["registration"]
                     and V8.E.digest(value["intent"]) == request["intent_sha256"],
                     "immutable host intent changed")
        registration = V8.T.read_ref(request["registration"])
        binding = V8.T.read_ref(registration["parent_binding"])
        protocol = V8.T.read_ref(registration["parent_protocol"])
        host_type = _host_for_registration(registration, binding, protocol)
        transport = V8.T.BrevMailboxTransport(self.runtime, self.contract_ref)
        with host_type(self.runtime, self.bindings, transport,
                       "registered-main-io-service") as host:
            bound = host._binding(operation, value["intent"])
            if request["phase"] == "case":
                existing = host._authorization(bound)
                V8.T.require(existing is not None and existing[1] is False
                             and existing[0] == request.get("authorization"),
                             "case authorization not consumed/exact")

    def _provider(self, folder, plan, registration, execution, arm) -> None:
        receipt, raw = self._inventory(folder)
        operation = execution["operations"][arm]
        packet = {
            "schema": REMOTE.PROVIDER_SCHEMA,
            "operation_id": operation["operation_id"],
            "observed_at_utc": receipt["request_before_utc"],
            "inventory": {
                "path": operation["provider_packet_path"] + ".inventory.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        }
        payload = base64.b64encode(json.dumps({
            "packet": packet,
            "inventory_base64": base64.b64encode(raw).decode(),
        }).encode()).decode()
        self._rpc(folder, plan, registration, "provider", "--arm", arm,
                  "--payload-base64", payload)

    def _rpc(self, *args: Any, **kwargs: Any):
        with service9_scope():
            return super()._rpc(*args, **kwargs)

    def _registered(self, *args: Any, **kwargs: Any):
        with service9_scope():
            return super()._registered(*args, **kwargs)

    def serve_once(self, *args: Any, **kwargs: Any):
        with service9_scope():
            return super().serve_once(*args, **kwargs)


def unpack_archive(*args: Any, **kwargs: Any):
    with service9_scope():
        return V8.unpack_archive(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(V8, name)


V8.T.require(REMOTE3.SOURCES <= SOURCES, "transport v3 source closure missing")
