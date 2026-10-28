"""파일 IO 계약만 검사한다. 합성 결과를 실제 probe 증거로 수용하지 않는다."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import agentic_brev_mailbox_transport_v1 as T


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def world(tmp_path):
    mailbox, results = tmp_path / "transport-mailbox", tmp_path / "results"
    mailbox.mkdir()
    results.mkdir()
    contract = save(tmp_path / "contract.json", {
        "schema": T.SCHEMA + "_contract", "mailbox": str(mailbox), "result_root": str(results),
        "response_wait_s": 0.001, "service_source_sha256_lf": "a" * 64,
    })
    runtime = SimpleNamespace(folder=tmp_path, protocol={
        "host_transport_contract_sha256": contract["sha256"], "host_transport_service_sha256_lf": "a" * 64,
    })
    registration = save(tmp_path / "registration.json", {"schema": "agentic_brev_runtime_host_v1_registration"})
    return T.BrevMailboxTransport(runtime, contract), "1" * 64, {"action": "run_probe", "snapshot": {}}, registration


def test_prepare_and_lookups_never_enqueue(tmp_path):
    transport, op, intent, registration = world(tmp_path)
    assert transport.prepare_pair(op, intent, registration)["status"] == "prepared"
    assert transport.lookup_reference(op, intent, registration)["status"] == "not_found"
    assert transport.lookup_pair(op, intent, registration)["status"] == "not_found"
    assert list(transport.root.iterdir()) == []


def test_reference_intent_is_written_once_and_pending_never_resubmits(tmp_path):
    transport, op, intent, registration = world(tmp_path)
    assert transport.start_reference(op, intent, registration)["status"] == "pending"
    request = transport.root / op / "reference/request.json"
    original = request.read_bytes()
    original_time = request.stat().st_mtime_ns
    assert transport.start_reference(op, intent, registration)["status"] == "pending"
    assert transport.lookup_reference(op, intent, registration)["status"] == "pending"
    assert request.read_bytes() == original and request.stat().st_mtime_ns == original_time
    with pytest.raises(ValueError, match="intent differs"):
        transport.start_reference(op, {**intent, "target": "other"}, registration)
    assert request.read_bytes() == original


def respond(transport, op, phase, result):
    folder = transport.root / op / phase
    save(folder / "response.json", {
        "schema": T.SCHEMA + "_response", "request_sha256": T.E.J.file_sha(folder / "request.json"),
        "operation_id": op, "phase": phase, "status": "completed", "result": result,
    })


def test_completed_result_passes_raw_ref_only_and_detects_change(tmp_path):
    transport, op, intent, registration = world(tmp_path)
    transport.start_reference(op, intent, registration)
    result = save(transport.results / "outcome.json", {"cpu_fixture_only": True})
    respond(transport, op, "reference", result)
    assert transport.lookup_reference(op, intent, registration) == {
        "status": "completed", "operation_id": op, "result": result,
    }
    # 이 전달값은 실제 증거가 아니다. E13와 C host가 별도로 검사해야 한다.
    Path(result["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA changed"):
        transport.lookup_reference(op, intent, registration)


@pytest.mark.parametrize("fault", ["request_sha", "operation", "phase", "outside"])
def test_response_binding_and_result_root_refuse(tmp_path, fault):
    transport, op, intent, registration = world(tmp_path)
    transport.start_reference(op, intent, registration)
    target = tmp_path / "outside.json" if fault == "outside" else transport.results / "outcome.json"
    respond(transport, op, "reference", save(target, {}))
    p = transport.root / op / "reference/response.json"
    value = json.loads(p.read_text())
    if fault == "request_sha":
        value["request_sha256"] = "f" * 64
    elif fault == "operation":
        value["operation_id"] = "2" * 64
    elif fault == "phase":
        value["phase"] = "case"
    save(p, value)
    with pytest.raises(ValueError):
        transport.lookup_reference(op, intent, registration)


def test_case_requires_valid_c_authorization_before_request(tmp_path, monkeypatch):
    transport, op, intent, registration = world(tmp_path)
    def denied(*args):
        raise ValueError("approval denied")
    monkeypatch.setattr(T.C, "validate_probe_authorization", denied)
    with pytest.raises(ValueError, match="approval denied"):
        transport.dispatch_case(op, intent, registration, {"untrusted": True})
    assert not (transport.root / op).exists()


def test_case_binds_consumed_authorization_and_lookup_does_not_approve(tmp_path, monkeypatch):
    transport, op, intent, registration = world(tmp_path)
    calls = []
    monkeypatch.setattr(T.C, "validate_probe_authorization", lambda *args: calls.append(args))
    authorization = {"cpu_fixture_only": "consumed"}
    assert transport.dispatch_case(op, intent, registration, authorization)["status"] == "pending"
    request = json.loads((transport.root / op / "case/request.json").read_text())
    assert request["authorization"] == authorization and len(calls) == 1
    assert transport.lookup_pair(op, intent, registration)["status"] == "pending"
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["operation", "repair", "contract"])
def test_invalid_request_writes_nothing(tmp_path, fault):
    transport, op, intent, registration = world(tmp_path)
    if fault == "operation":
        op = "../escape"
    elif fault == "repair":
        intent["action"] = "request_repair"
    else:
        Path(transport.contract_ref["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        transport.start_reference(op, intent, registration)
    assert not list(transport.root.iterdir())


def test_mailbox_parent_symlink_refused_before_write(tmp_path):
    transport, op, intent, registration = world(tmp_path)
    real = tmp_path / "elsewhere"
    real.mkdir()
    try:
        (transport.root / op).symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("현재 OS의 symlink 생성 권한 없음")
    with pytest.raises(ValueError, match="symlink/reparse"):
        transport.start_reference(op, intent, registration)
    assert not list(real.iterdir())


def test_result_root_dotdot_escape_is_rejected(tmp_path):
    transport, op, intent, registration = world(tmp_path)
    transport.start_reference(op, intent, registration)
    outside = save(tmp_path / "outside.json", {"must_not_be_accepted": True})
    outside["path"] = str(transport.results / ".." / "outside.json")
    respond(transport, op, "reference", outside)
    with pytest.raises(ValueError, match="canonical native"):
        transport.lookup_reference(op, intent, registration)
