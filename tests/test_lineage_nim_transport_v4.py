from __future__ import annotations

# pyright: reportMissingImports=false
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import lineage_nim_transport_v4 as T

V3_PROTOCOL = ROOT / "bench/protocols/runtime_observation_dev_nim_20261031_v3.json"
ANCHOR = "574d85f4945622889375c4cbd49a6a9764195bde"
MESSAGES = [{"role": "user", "content": "portable CPU fixture"}]


@pytest.fixture
def registered_protocol(request) -> tuple[Path, dict]:
    private = ROOT / "bench/private"
    private.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="transport-v4-test-", dir=private))
    request.addfinalizer(lambda: shutil.rmtree(directory, ignore_errors=True))
    path = directory / "protocol-v4.json"
    value = T.prepare_protocol(V3_PROTOCOL)
    path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    return path, value


def initialize(root: Path, protocol_path: Path, protocol: dict,
               monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(T, "verify_protocol", lambda _path, _anchor: deepcopy(protocol))
    return T.initialize(root, protocol_path, ANCHOR, protocol["plan"])


def test_windows_initialize_then_portable_enqueue_serve_response_and_close(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol) -> None:
    protocol_path, protocol = registered_protocol
    root = tmp_path / "queue"
    session = initialize(root, protocol_path, protocol, monkeypatch)
    binding = T._read(root / T.BINDING_NAME)
    assert session["params"] == protocol["params"]
    assert ":" not in binding["protocol"]["path"] and "\\" not in binding["protocol"]["path"]
    assert len(binding["path_aliases"]) == 2
    assert all(":\\" in row["recorded_path"] for row in binding["path_aliases"])

    ref = T.enqueue(root, "telemetry_only", MESSAGES, protocol["params"], "a" * 64, "b" * 64)
    fake = {"provider_model": protocol["model"], "finish_reason": "stop",
            "content": "fixture", "input_tokens": 1, "output_tokens": 1,
            "total_tokens": 2, "wall_s": 0.01}
    receipt = T.serve_one(root, ref, complete=lambda _messages, _params: fake)
    response = T._read(root / receipt["response_relative"])
    assert response["result"] == fake and response["request_sha256"] == ref["request_sha256"]
    assert T.status(root)["responses"] == 1
    assert T.close(root, "portable fixture complete")["reason"] == "portable fixture complete"


def test_portable_verifiers_return_raw_binding_objects_unchanged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol) -> None:
    protocol_path, protocol = registered_protocol
    root = tmp_path / "queue"
    initialize(root, protocol_path, protocol, monkeypatch)
    raw_v3, raw_v2 = T._raw_bindings(root)
    checked_v3, _ = T._portable_v3_binding(root)
    checked_v2, _ = T._portable_v2_binding(root)
    assert checked_v3 == raw_v3 and checked_v2 == raw_v2
    assert T.T2.T1._digest(checked_v2) == raw_v3["component_v2_binding_sha256"]


def test_aliases_are_exact_and_arbitrary_windows_rewrite_is_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol) -> None:
    protocol_path, protocol = registered_protocol
    root = tmp_path / "queue"
    initialize(root, protocol_path, protocol, monkeypatch)
    v4_path = root / T.BINDING_NAME
    value = T._read(v4_path)
    value["path_aliases"][0]["recorded_path"] = "Z:\\invented\\protocol.json"
    v4_path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    with pytest.raises(ValueError, match="alias"):
        T._v4_binding(root)


@pytest.mark.parametrize("field", [
    "anchor", "session_sha256", "component_v3_protocol", "component_v3_binding_sha256",
    "component_v2_binding_sha256", "transport_v4_source_sha256_lf",
])
def test_v4_binding_tamper_is_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol, field: str) -> None:
    protocol_path, protocol = registered_protocol
    root = tmp_path / "queue"
    initialize(root, protocol_path, protocol, monkeypatch)
    path = root / T.BINDING_NAME
    value = T._read(path)
    value[field] = "0" * 40 if field == "anchor" else "tampered"
    path.write_bytes(T.T2.T1._canonical(value) + b"\n")
    with pytest.raises((TypeError, ValueError)):
        T._v4_binding(root)


def test_raw_v2_or_v3_binding_change_breaks_recorded_sha(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol) -> None:
    protocol_path, protocol = registered_protocol
    for name in (T.T3.BINDING_NAME, T.T2.BINDING_NAME):
        root = tmp_path / name
        initialize(root, protocol_path, protocol, monkeypatch)
        path = root / name
        value = T._read(path)
        value["protocol"]["path"] = "C:\\invented\\other.json"
        path.write_bytes(T.T2.T1._canonical(value) + b"\n")
        with pytest.raises(ValueError):
            T._v4_binding(root)


def test_compatibility_scope_reentry_and_restore() -> None:
    originals = (T.T3._binding, T.T3._fixed_v2_binding)
    with T.compatibility_scope():
        assert (T.T3._binding, T.T3._fixed_v2_binding) == (
            T._portable_v3_binding, T._portable_v2_binding)
        with T.compatibility_scope():
            assert T.T3._binding is T._portable_v3_binding
    assert (T.T3._binding, T.T3._fixed_v2_binding) == originals


def test_protocol_preserves_model_params_caps_atomic_and_no_duplicate(registered_protocol) -> None:
    _path, value = registered_protocol
    component = T._read(V3_PROTOCOL)
    for field in ("model", "accepted_provider_models", "params", "plan",
                  "operator_approved", "publication"):
        assert value[field] == component[field]
    assert value["plan"]["global_request_cap"] == 6
    assert value["plan"]["per_slot_max"] == 3
    assert value["scope"]["duplicate_send"] is False
    assert value["scope"]["provider_retry"] is False
    assert value["scope"]["new_queue_only"] is True


def test_file_completion_and_cli_use_v4_compatibility_surface(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_protocol) -> None:
    protocol_path, protocol = registered_protocol
    root = tmp_path / "queue"
    initialize(root, protocol_path, protocol, monkeypatch)
    seen: list[tuple[object, object]] = []

    def factory(*_args, **_kwargs):
        seen.append((T.T3._binding, T.T3._fixed_v2_binding))
        return lambda _messages, _params: seen.append(
            (T.T3._binding, T.T3._fixed_v2_binding)) or {"content": "ok"}

    monkeypatch.setattr(T.T3, "file_completion", factory)
    complete = T.file_completion(root, "telemetry_only", "a" * 64, "b" * 64, 1)
    assert complete(MESSAGES, protocol["params"]) == {"content": "ok"}
    assert seen == [(T._portable_v3_binding, T._portable_v2_binding)] * 2
    source = (ROOT / "evals/lineage_nim_transport_v4.py").read_text(encoding="utf-8")
    assert '"prepare-protocol", "initialize", "serve-next", "once", "close", "status", "worker"' in source
    assert 'Path(protocol_ref["path"])' not in source


def test_no_provider_environment_or_wsl_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert "NVIDIA_API_KEY" not in os.environ
