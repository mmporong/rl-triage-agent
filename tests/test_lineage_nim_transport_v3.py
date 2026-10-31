from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import lineage_nim_transport_v3 as T

V2_PROTOCOL = ROOT / "bench/protocols/lineage_compound_atomic_nim_20261031_v2.json"
V2_ANCHOR = "fcfaa7c7e1f6b1df10e967ec36f1dd000c6e60c1"
MESSAGES = [{"role": "user", "content": "CPU fixture"}]


def component() -> dict:
    return T._read(V2_PROTOCOL)


def prepare_v3(path: Path) -> dict:
    value = T.prepare_protocol(V2_PROTOCOL)
    path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    return value


def initialize_fixture(root: Path, protocol_path: Path,
                       monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    protocol = prepare_v3(protocol_path)
    original = T.verify_protocol
    monkeypatch.setattr(T, "verify_protocol", lambda path, anchor: deepcopy(protocol))
    session = T.initialize(root, protocol_path, V2_ANCHOR, protocol["plan"])
    return session, {"restore": original, "protocol": protocol}


def test_actual_v2_initialize_then_first_enqueue_reproduces_top_level_binding_bug(
        tmp_path: Path) -> None:
    root = tmp_path / "v2-reproduction"
    value = component()
    T.T2.initialize(root, V2_PROTOCOL, V2_ANCHOR, value["plan"])
    binding = T.T2._read(root / T.T2.BINDING_NAME)
    assert {"anchor", "session_sha256", "component_v1_protocol",
            "atomic_writer_source_sha256_lf"} <= set(binding)
    assert set(binding["protocol"]) == {"path", "sha256_lf"}
    with pytest.raises(KeyError, match="anchor"):
        T.T2.enqueue(root, "case-07.model.1", MESSAGES, value["params"], "a" * 64, "b" * 64)


def test_v3_initialize_enqueue_serve_response_and_close_preserve_t2_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, protocol_path = tmp_path / "queue", tmp_path / "protocol-v3.json"
    session, state = initialize_fixture(root, protocol_path, monkeypatch)
    protocol = state["protocol"]
    assert session["plan"] == protocol["plan"]
    assert session["model"] == protocol["model"]
    assert session["params"] == protocol["params"]
    assert T.T2._binding is T._REGISTERED_BINDING

    ref = T.enqueue(root, "case-07.model.1", MESSAGES, protocol["params"], "a" * 64, "b" * 64)
    result = {"provider_model": protocol["model"], "finish_reason": "stop",
              "content": "fixture", "input_tokens": 1, "output_tokens": 1,
              "total_tokens": 2, "wall_s": 0.01}
    receipt = T.serve_one(root, ref, complete=lambda _messages, _params: result)
    response = T._read(root / receipt["response_relative"])
    assert response["request_sha256"] == ref["request_sha256"]
    assert response["result"] == result
    assert T.status(root)["responses"] == 1
    assert T.close(root, "fixture complete")["reason"] == "fixture complete"
    assert T.T2._binding is T._REGISTERED_BINDING


def test_file_completion_uses_fixed_binding_at_factory_and_callback_time(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, protocol_path = tmp_path / "queue", tmp_path / "protocol-v3.json"
    _session, state = initialize_fixture(root, protocol_path, monkeypatch)
    seen: list[object] = []

    def factory(*_args, **_kwargs):
        seen.append(T.T2._binding)
        return lambda _messages, _params: seen.append(T.T2._binding) or {"content": "ok"}

    monkeypatch.setattr(T.T2, "file_completion", factory)
    callback = T.file_completion(root, "case-07.model.1", "a" * 64, "b" * 64, 1)
    assert callback(MESSAGES, state["protocol"]["params"]) == {"content": "ok"}
    assert seen == [T._fixed_v2_binding, T._fixed_v2_binding]
    assert T.T2._binding is T._REGISTERED_BINDING


@pytest.mark.parametrize("field", [
    "anchor", "session_sha256", "component_v2_protocol",
    "transport_v3_source_sha256_lf", "component_v2_binding_sha256",
])
def test_v3_top_level_binding_tamper_is_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    root, protocol_path = tmp_path / "queue", tmp_path / "protocol-v3.json"
    initialize_fixture(root, protocol_path, monkeypatch)
    binding_path = root / T.BINDING_NAME
    value = T._read(binding_path)
    value[field] = "0" * 40 if field == "anchor" else "tampered"
    binding_path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    with pytest.raises((TypeError, ValueError)):
        T._binding(root)


@pytest.mark.parametrize("field", [
    "anchor", "session_sha256", "component_v1_protocol", "atomic_writer_source_sha256_lf",
])
def test_fixed_v2_top_level_binding_tamper_is_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    root, protocol_path = tmp_path / "queue", tmp_path / "protocol-v3.json"
    initialize_fixture(root, protocol_path, monkeypatch)
    binding_path = root / T.T2.BINDING_NAME
    value = T._read(binding_path)
    value[field] = "0" * 40 if field == "anchor" else "tampered"
    binding_path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    with pytest.raises((TypeError, ValueError)):
        T._binding(root)


def test_binding_scope_restores_registered_t2_function() -> None:
    original = T.T2._binding
    with T.binding_scope():
        assert T.T2._binding is T._fixed_v2_binding
        with T.binding_scope():
            assert T.T2._binding is T._fixed_v2_binding
    assert T.T2._binding is original


def test_protocol_preserves_t2_model_params_plan_caps_and_atomic_publication(tmp_path: Path) -> None:
    protocol = prepare_v3(tmp_path / "protocol-v3.json")
    original = component()
    for field in ("model", "accepted_provider_models", "params", "plan",
                  "operator_approved", "publication"):
        assert protocol[field] == original[field]
    assert protocol["plan"]["global_request_cap"] == 9
    assert protocol["plan"]["per_slot_max"] == 3
    assert protocol["scope"]["duplicate_send"] is False
    assert protocol["scope"]["provider_retry"] is False
    assert set(protocol["source_sha256_lf"]) == set(T.SOURCE_FILES)


def test_verify_rejects_current_source_hash_and_anchor_blob_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = tmp_path / "protocol-v3.json"
    value = prepare_v3(protocol_path)
    value["source_sha256_lf"][T.SOURCE_FILES[0]] = "0" * 64
    protocol_path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    monkeypatch.setattr(T.T2, "verify_protocol", lambda _path, _anchor: component())
    with pytest.raises(ValueError, match="내용 또는 SHA"):
        T.verify_protocol(protocol_path, V2_ANCHOR)

    value = prepare_v3(protocol_path)
    relative = T._relative
    monkeypatch.setattr(T, "_relative", lambda path: (
        "bench/private/protocol-v3.json" if Path(path).resolve() == protocol_path.resolve()
        else relative(path)))
    monkeypatch.setattr(T, "_git", lambda *_args: b"wrong")
    with pytest.raises(ValueError, match="commit blob"):
        T.verify_protocol(protocol_path, V2_ANCHOR)
