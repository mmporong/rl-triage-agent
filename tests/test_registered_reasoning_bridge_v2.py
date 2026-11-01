# pyright: reportMissingImports=false
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from evals import registered_reasoning_bridge_v2 as B


def _protocol() -> dict:
    return {"schema": B.SCHEMA, "logical_slots": B._expected_slots(),
            "modes": B._mode_contract(), "max_requests": B.MAX_REQUESTS}


def _request(root: Path, index: int, slot: str, messages: list) -> Path:
    protocol = _protocol()
    mode = next(row["mode"] for row in protocol["logical_slots"] if row["slot_id"] == slot)
    params = deepcopy(protocol["modes"][mode])
    intent = {"schema": B.REQUEST_SCHEMA, "index": index, "slot": slot,
              "case_id": slot.split(".", 1)[0], "mode": mode, "messages": messages,
              "messages_sha256": B.digest(messages), "params": params,
              "params_sha256": B.digest(params), "bridge_protocol_sha256": "a" * 64,
              "context_sha256": B.digest({"slot": slot, "messages": messages,
                                             "params": params})}
    path = root / "requests" / f"{index:03d}.json"
    B._write_new(path, intent)
    return path


def test_registered_slots_cover_two_modes_four_decisions_two_attempts() -> None:
    slots = B._expected_slots()
    assert len(slots) == 96
    assert slots[0]["slot_id"] == "c02.legacy_none.d1.a1"
    assert slots[-1]["slot_id"] == "c08.enabled_low.d4.a2"
    assert B._mode_contract()["enabled_low"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"}


def test_reserve_rejects_unknown_duplicate_and_cap(tmp_path: Path) -> None:
    protocol = _protocol()
    messages = [{"role": "user", "content": "x"}]
    params = protocol["modes"]["legacy_none"]
    first = B._reserve(tmp_path, "c02.legacy_none.d1.a1", messages, params,
                       protocol, "a" * 64)
    assert first.name == "001.json"
    with pytest.raises(ValueError, match="재전송"):
        B._reserve(tmp_path, "c02.legacy_none.d1.a1", messages, params,
                   protocol, "a" * 64)
    with pytest.raises(ValueError, match="미등록"):
        B._reserve(tmp_path, "c01.legacy_none.d1.a1", messages, params,
                   protocol, "a" * 64)
    protocol["max_requests"] = 1
    with pytest.raises(ValueError, match="cap"):
        B._reserve(tmp_path, "c02.legacy_none.d1.a2", messages, params,
                   protocol, "a" * 64)


def test_worker_call_binds_request_claim_and_response(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    session = {"schema": B.SESSION_SCHEMA,
               "protocol": {"path": "x", "sha256": "a" * 64}, "anchor": "b" * 40}
    monkeypatch.setattr(B, "_session", lambda _root: (session, protocol))
    request = _request(tmp_path, 1, "c02.enabled_low.d1.a1",
                       [{"role": "user", "content": "observe"}])
    calls = []

    def provider(value: dict) -> dict:
        calls.append(value)
        return {"provider_model": B.MODE.MODEL, "finish_reason": "stop",
                "content": '{"option_id":"hold","fact_ids":[]}' }

    response = B.worker_call(tmp_path, request, provider)
    assert len(calls) == 1 and response["result"]["request_params"] == protocol["modes"]["enabled_low"]
    assert (tmp_path / "claims/001.json").is_file()
    assert (tmp_path / "slot-claims/c02.enabled_low.d1.a1.json").is_file()
    assert (tmp_path / "responses/001.json").is_file()


def test_worker_duplicate_slot_claim_fails_before_provider(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    session = {"schema": B.SESSION_SCHEMA,
               "protocol": {"path": "x", "sha256": "a" * 64}, "anchor": "b" * 40}
    monkeypatch.setattr(B, "_session", lambda _root: (session, protocol))
    messages = [{"role": "user", "content": "observe"}]
    first = _request(tmp_path, 1, "c02.enabled_low.d1.a1", messages)
    B.worker_call(tmp_path, first, lambda _: {"content": "{}"})
    second = _request(tmp_path, 2, "c02.enabled_low.d1.a1", messages)
    with pytest.raises(FileExistsError):
        B.worker_call(tmp_path, second, lambda _: pytest.fail("provider 호출 금지"))


def test_response_rejects_tampered_params(tmp_path: Path) -> None:
    protocol = _protocol()
    messages = [{"role": "user", "content": "observe"}]
    request = _request(tmp_path, 1, "c02.legacy_none.d1.a1", messages)
    B._write_new(tmp_path / "responses/001.json", {
        "schema": B.RESPONSE_SCHEMA, "request_sha256": B.file_sha(request),
        "result": {"request_params": protocol["modes"]["enabled_low"]}})
    with pytest.raises(ValueError, match="params"):
        B._response(tmp_path, request)
