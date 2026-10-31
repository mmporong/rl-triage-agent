# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import registered_reasoning_bridge_v1 as B

PARENT = B.ROOT / "bench/protocols/nemotron_reasoning_options_development_20261101_v2.json"


def slots() -> list[dict]:
    parent = B.S1._read(PARENT)
    cases = sorted({row["case_id"] for row in parent["calls"]
                    if row["evaluation_role"] == "fault_development"})
    return [{"slot_id": f"{case}.{mode}.d{decision}.a{attempt}",
             "case_id": case, "mode": mode}
            for case in cases for mode in B.S1.MODES
            for decision in range(2, 5) for attempt in range(1, 3)]


@pytest.fixture
def protocol():
    return B.prepare_protocol(PARENT, slots())


def test_protocol_inherits_exact_parent_modes_and_registers_72_unique_slots(protocol):
    parent = B.S1._read(PARENT)
    assert protocol["modes"] == parent["modes"]
    assert protocol["max_requests"] == len(protocol["logical_slots"]) == 72
    assert len({row["slot_id"] for row in protocol["logical_slots"]}) == 72
    assert protocol["authority"] == {"approval": False, "gpu": False,
                                      "repair": False, "recovery": False}
    with pytest.raises(ValueError, match="고정 72"):
        B.prepare_protocol(PARENT, slots()[:-1], max_requests=71)


def _session(monkeypatch, protocol):
    monkeypatch.setattr(B, "_session", lambda _root: (
        {"schema": B.SESSION_SCHEMA, "protocol": {"sha256": "9" * 64}}, protocol))


def test_complete_preserves_mode_params_and_returns_bound_transport(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    seen = []

    def spawn(root: Path, request: Path, _timeout: int) -> int:
        intent = B.S1._read(request)
        seen.append(intent)
        response = {"schema": B.RESPONSE_SCHEMA, "request_sha256": B.S1.file_sha(request),
                    "result": {"provider_model": protocol["model"], "finish_reason": "stop",
                               "content": "{}", "request_params": intent["params"], "wall_s": 0.1}}
        B.S1._write_new(root / "responses" / request.name, response)
        return 0

    monkeypatch.setattr(B, "_spawn_worker", spawn)
    messages = [{"role": "user", "content": "x"}]
    for mode in B.S1.MODES:
        slot = next(row["slot_id"] for row in protocol["logical_slots"] if row["mode"] == mode)
        result = B.complete(tmp_path, slot, messages, deepcopy(protocol["modes"][mode]))
        assert result["request_params"] == protocol["modes"][mode]
        assert Path(result["transport_ref"]["path"]).is_absolute()
        assert Path(result["transport_ref"]["path"]).name in {"001.json", "002.json"}
    assert seen[0]["params"] == protocol["modes"]["legacy_none"]
    assert seen[1]["params"] == protocol["modes"]["enabled_low"]


def test_duplicate_slot_wrong_mode_and_73rd_request_stop_before_worker(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    calls = []
    monkeypatch.setattr(B, "_spawn_worker", lambda *_args: calls.append(True) or 1)
    messages = [{"role": "user", "content": "x"}]
    slot = protocol["logical_slots"][0]
    wrong = deepcopy(protocol["modes"]["enabled_low" if slot["mode"] == "legacy_none" else "legacy_none"])
    with pytest.raises(ValueError, match="params"):
        B.complete(tmp_path, slot["slot_id"], messages, wrong)
    assert calls == [] and not list((tmp_path / "requests").glob("*.json"))

    params = protocol["modes"][slot["mode"]]
    request = B._reserve(tmp_path, slot["slot_id"], messages, params, protocol, "9" * 64)
    with pytest.raises(ValueError, match="재전송"):
        B._reserve(tmp_path, slot["slot_id"], messages, params, protocol, "9" * 64)
    assert request.is_file()
    for index, row in enumerate(protocol["logical_slots"][1:], 2):
        B._reserve(tmp_path, row["slot_id"], messages, protocol["modes"][row["mode"]],
                   protocol, "9" * 64)
        assert index <= 72
    with pytest.raises(ValueError, match="cap"):
        B._reserve(tmp_path, "c02.legacy_none.d2.a1", messages,
                   protocol["modes"]["legacy_none"], protocol, "9" * 64)
    assert calls == []


def test_response_and_request_tamper_are_rejected(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    row = protocol["logical_slots"][0]
    request = B._reserve(tmp_path, row["slot_id"], [], protocol["modes"][row["mode"]],
                         protocol, "9" * 64)
    value = B.S1._read(request)
    value["messages_sha256"] = "0" * 64
    request.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="intent"):
        B.worker_call(tmp_path, request)

    clean = tmp_path / "requests/clean.json"
    value["messages_sha256"] = B.S1.digest(value["messages"])
    clean.write_text(json.dumps(value), encoding="utf-8")
    response = {"schema": B.RESPONSE_SCHEMA, "request_sha256": "0" * 64,
                "result": {"request_params": value["params"]}}
    B.S1._write_new(tmp_path / "responses/clean.json", response)
    with pytest.raises(ValueError, match="response"):
        B._response(tmp_path, clean)


def test_worker_failure_exposes_no_secret_and_preserves_intent(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    monkeypatch.setattr(B, "_spawn_worker", lambda *_args: 7)
    row = protocol["logical_slots"][0]
    with pytest.raises(ValueError, match="비정상 종료"):
        B.complete(tmp_path, row["slot_id"], [], protocol["modes"][row["mode"]])
    raw = (tmp_path / "requests/001.json").read_text(encoding="utf-8")
    assert "NVIDIA_API_KEY" not in raw and "Bearer " not in raw


def test_worker_claim_makes_duplicate_and_cloned_slot_zero_extra_http(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    calls = []
    monkeypatch.setattr(B.S1, "openai_complete", lambda request: calls.append(request) or {
        "provider_model": protocol["model"], "finish_reason": "stop", "content": "{}",
        "wall_s": 0.1})
    row = protocol["logical_slots"][0]
    request = B._reserve(tmp_path, row["slot_id"], [], protocol["modes"][row["mode"]],
                         protocol, "9" * 64)
    B.worker_call(tmp_path, request)
    assert len(calls) == 1
    with pytest.raises((FileExistsError, ValueError)):
        B.worker_call(tmp_path, request)
    clone = tmp_path / "requests/002.json"
    value = B.S1._read(request)
    value["index"] = 2
    value["context_sha256"] = B.S1.digest({"slot": value["slot"],
                                           "messages": value["messages"],
                                           "params": value["params"]})
    clone.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FileExistsError):
        B.worker_call(tmp_path, clone)
    assert len(calls) == 1


def test_worker_rejects_noncanonical_or_cap_outside_filename_before_http(monkeypatch, tmp_path, protocol):
    _session(monkeypatch, protocol)
    calls = []
    monkeypatch.setattr(B.S1, "openai_complete", lambda request: calls.append(request) or {})
    for name in ("clean.json", "073.json"):
        path = tmp_path / "requests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="경로|index"):
            B.worker_call(tmp_path, path)
    assert calls == []


def test_spawn_worker_uses_linux_timeout_wrapper(monkeypatch, tmp_path):
    seen = {}

    class Process:
        def __init__(self, argv, **kwargs):
            seen.update(argv=argv, kwargs=kwargs)

        def wait(self, timeout):
            seen["timeout"] = timeout
            return 0

    monkeypatch.setattr(B.subprocess, "Popen", Process)
    request = tmp_path / "requests/001.json"
    assert B._spawn_worker(tmp_path, request, B.TIMEOUT_S) == 0
    command = seen["argv"][-1]
    assert "/usr/bin/timeout --signal=TERM --kill-after=5s 360s" in command
    assert seen["timeout"] == 395
