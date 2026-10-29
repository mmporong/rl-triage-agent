from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import agentic_brev_mailbox_service_v9 as S9  # pyright: ignore[reportMissingImports]


def _protocol(host6: bool) -> dict:
    if host6:
        return {"schema": S9.H6.PARENT_SCHEMA + "_protocol",
                "sources_lf": {name: "0" * 64 for name in S9.H6.SOURCES}}
    return {"schema": S9.H1.P.SCHEMA + "_protocol", "sources_lf": {}}


def _binding(host6: bool) -> dict:
    schema = S9.H6.SCHEMA if host6 else S9.H1.SCHEMA
    return {"expected_request": {"execution": {
        "runtime_host_binding": {"schema": schema + "_runtime_binding"},
    }}}


def test_host_selection_uses_exact_registered_runtime_schema() -> None:
    assert S9._host_for_registration({}, _binding(False), _protocol(False)) is S9.H1.BrevRuntimeHost
    assert S9._host_for_registration({}, _binding(True), _protocol(True)) is S9.H6.BrevRuntimeHost
    changed = _binding(True)
    changed["expected_request"]["execution"]["runtime_host_binding"]["schema"] = "other"
    with pytest.raises(ValueError, match="unsupported"):
        S9._host_for_registration({}, changed, _protocol(True))


def test_budget_origin_and_stop_reserve_are_both_required() -> None:
    phase = {"registered_at_utc": "2026-01-01T00:00:00+00:00",
             "deadline_utc": "2026-01-01T01:00:00+00:00",
             "max_lifecycle_s": 3600, "stop_reserve_s": 600}
    contract = {"deadline_utc": phase["deadline_utc"]}
    S9._validate_budget_origin(
        phase, contract, seconds=300,
        now=datetime(2026, 1, 1, 0, 40, tzinfo=timezone.utc),  # noqa: UP017 - Python 3.10
    )
    with pytest.raises(ValueError, match="origin"):
        S9._validate_budget_origin(
            {**phase, "registered_at_utc": "2026-01-01T00:10:00+00:00"},
            contract, seconds=300,
            now=datetime(2026, 1, 1, 0, 40, tzinfo=timezone.utc),  # noqa: UP017 - Python 3.10
        )
    with pytest.raises(ValueError, match="Stop reserve"):
        S9._validate_budget_origin(
            phase, contract, seconds=300,
            now=datetime(2026, 1, 1, 0, 52, tzinfo=timezone.utc),  # noqa: UP017 - Python 3.10
        )


def test_validate_host_intent_constructs_selected_host_and_checks_binding(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    operation = "a" * 64
    intent = {"action": "run_probe", "target": "P_reward"}
    protocol_path = tmp_path / "protocol.json"
    binding_path = tmp_path / "binding.json"
    registration_path = tmp_path / "registration.json"
    for path, value in ((protocol_path, _protocol(True)), (binding_path, _binding(True))):
        path.write_text(json.dumps(value), encoding="utf-8")
    registration = {
        "parent_protocol": {"path": str(protocol_path), "sha256": S9.V8.E.J.file_sha(protocol_path)},
        "parent_binding": {"path": str(binding_path), "sha256": S9.V8.E.J.file_sha(binding_path)},
    }
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    request = {
        "operation_id": operation, "phase": "reference",
        "registration": {"path": str(registration_path),
                         "sha256": S9.V8.E.J.file_sha(registration_path)},
        "intent_sha256": S9.V8.E.digest(intent),
    }
    folder = tmp_path / "runtime"
    host_intent = folder / "brev-host" / operation / "intent.json"
    host_intent.parent.mkdir(parents=True)
    host_intent.write_text(json.dumps({"registration": request["registration"],
                                      "intent": intent}), encoding="utf-8")
    seen = []

    class Host:
        def __init__(self, runtime, bindings, transport, approver):
            seen.append((runtime, bindings, type(transport), approver))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def _binding(self, value, payload):
            seen.append((value, payload))
            return {"operation_id": value}

    monkeypatch.setattr(S9.H6, "BrevRuntimeHost", Host)
    monkeypatch.setattr(S9.V8.T, "BrevMailboxTransport", lambda *_args: object())
    service = object.__new__(S9.BrevMailboxService)
    service.runtime = SimpleNamespace(folder=folder)
    service.bindings = {operation: request["registration"]}
    service.contract_ref = {"path": "contract", "sha256": "0" * 64}
    service._validate_host_intent(request)
    assert seen[-1] == (operation, intent)


def test_service9_closure_contains_remote_parent_host_and_transport_v3() -> None:
    assert S9.REMOTE.SOURCES <= S9.SOURCES
    assert S9.H6.SOURCES <= S9.SOURCES
    assert S9.REMOTE3.SOURCES <= S9.SOURCES


def test_provider_packet_uses_remote8_parent8_schema() -> None:
    service = object.__new__(S9.BrevMailboxService)
    raw = b'{"workspaces":[]}'
    service._inventory = lambda _folder: ({"request_before_utc": "2026-01-01T00:00:00+00:00"}, raw)
    calls = []
    service._rpc = lambda *args: calls.append(args)
    execution = {"operations": {"NONE": {
        "operation_id": "a" * 64,
        "provider_packet_path": "/runtime/provider-NONE.json",
    }}}
    service._provider(Path("unused"), {}, {}, execution, "NONE")
    encoded = calls[0][-1]
    envelope = json.loads(base64.b64decode(encoded))
    assert envelope["packet"]["schema"] == S9.REMOTE.PROVIDER_SCHEMA
    assert envelope["packet"]["schema"] == S9.REMOTE.PARENT.SCHEMA + "_provider_packet"
