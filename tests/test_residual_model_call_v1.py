from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_model_call_v1 as M  # noqa: E402
import r1_repair_contract as RC  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def simple_contract(monkeypatch):
    monkeypatch.setattr(M.C, "diagnosis_messages", lambda packet: [
        {"role": "system", "content": "return ok JSON"},
        {"role": "user", "content": json.dumps(packet, sort_keys=True)},
    ])

    def validate(decision, packet):
        if decision != {"ok": True, "case": packet["case"]}:
            raise ValueError("decision fields are invalid")
        return decision

    monkeypatch.setattr(M.C, "validate_diagnosis", validate)


def answer(packet: dict, *, infra: str | None = None) -> dict:
    if infra:
        return {"infra_error": infra, "error_type": infra, "wall_s": 0.2}
    return {"response_id": "r", "provider_model": M.C.MODEL, "finish_reason": "stop",
            "content": json.dumps({"ok": True, "case": packet["case"]}),
            "reasoning_content": None, "input_tokens": 10, "output_tokens": 4,
            "total_tokens": 14, "wall_s": 0.1}


def test_valid_decision_writes_bound_global_receipt(tmp_path, simple_contract):
    packet = {"case": "compound-a"}
    result = M.decide(tmp_path / "seed-a", "diagnosis", packet, tmp_path / "api",
                      caller=lambda messages, params: answer(packet))
    assert result["review"]["status"] == "valid"
    audit = M.audit_budget(tmp_path / "api")
    assert {key: audit[key] for key in ("reserved", "completed", "pending", "ok", "remaining")} == {
        "reserved": 1, "completed": 1, "pending": 0, "ok": 1, "remaining": 19}
    record = M.read(tmp_path / "seed-a/diagnosis.dialogue1.json")
    assert record["schema"] == M.SCHEMA and len(record["request_receipts"]) == 1
    assert M.R.C is not M.C and M.R.SCHEMA != M.SCHEMA


def test_invalid_first_response_gets_one_feedback_request(tmp_path, simple_contract):
    packet, calls = {"case": "compound-b"}, []

    def complete(messages, params):
        calls.append(messages)
        if len(calls) == 1:
            return {**answer(packet), "content": "{}"}
        return answer(packet)

    result = M.decide(tmp_path / "seed-b", "diagnosis", packet, tmp_path / "api", caller=complete)
    assert result["review"]["status"] == "valid" and result["review"]["retried"] is True
    assert len(calls) == 2 and calls[1][-2]["role"] == "assistant" and calls[1][-1]["role"] == "user"
    assert M.audit_budget(tmp_path / "api")["reserved"] == 2


def test_infrastructure_error_allows_only_second_dialogue(tmp_path, simple_contract):
    packet, calls = {"case": "compound-c"}, []

    def complete(messages, params):
        calls.append(1)
        return answer(packet, infra="APITimeoutError") if len(calls) == 1 else answer(packet)

    result = M.decide(tmp_path / "seed-c", "diagnosis", packet, tmp_path / "api", caller=complete)
    assert result["review"]["status"] == "valid" and len(result["attempts"]) == 2
    audit = M.audit_budget(tmp_path / "api")
    assert (audit["reserved"], audit["timeout"], audit["ok"]) == (2, 1, 1)


def test_non_retryable_exception_is_call_error_without_second_dialogue(tmp_path, simple_contract):
    packet, calls = {"case": "bad-local"}, []

    def complete(messages, params):
        calls.append(1)
        raise ValueError("invalid local input")

    result = M.decide(tmp_path / "bad-local", "diagnosis", packet, tmp_path / "api", caller=complete)
    assert result["review"] == {"status": "call_error", "requests": 1, "reason": "ValueError"}
    assert len(result["attempts"]) == 1 and calls == [1]
    audit = M.audit_budget(tmp_path / "api")
    assert (audit["reserved"], audit["completed"], audit["error"], audit["pending"]) == (1, 1, 1, 0)


@pytest.mark.parametrize(("error", "retryable"), [
    (TimeoutError(), True),
    (ConnectionError(), True),
    (type("Status429", (Exception,), {"status_code": 429})(), True),
    (type("Status503", (Exception,), {"status_code": 503})(), True),
    (type("Status400", (Exception,), {"status_code": 400})(), False),
    (ValueError(), False),
])
def test_exception_retryability_is_narrow(error, retryable):
    assert M._retryable_exception(error) is retryable


def test_active_lease_serializes_concurrent_threads_until_response_receipt(tmp_path):
    root = tmp_path / "api"
    first_entered, release_first, second_entered = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def first(_messages, _params):
        first_entered.set()
        assert release_first.wait(5)
        return answer({"case": "thread-a"})

    def second(_messages, _params):
        second_entered.set()
        return answer({"case": "thread-b"})

    calls = [M.MeteredComplete(root, "diagnosis", "a" * 64, "thread/a", first),
             M.MeteredComplete(root, "diagnosis", "b" * 64, "thread/b", second)]

    def invoke(index):
        try:
            calls[index]([{"role": "user", "content": str(index)}], {})
        except Exception as error:  # pragma: no cover - assertion below reports the captured error
            errors.append(error)

    first_thread = threading.Thread(target=invoke, args=(0,))
    second_thread = threading.Thread(target=invoke, args=(1,))
    first_thread.start()
    assert first_entered.wait(5)
    second_thread.start()
    time.sleep(0.1)
    assert not second_entered.is_set()
    during = M.audit_budget(root)
    assert (during["reserved"], during["pending"]) == (1, 1)
    release_first.set()
    first_thread.join(5)
    second_thread.join(5)
    assert not errors and not first_thread.is_alive() and not second_thread.is_alive()
    assert second_entered.is_set() and not M._active_path(root).exists()
    after = M.audit_budget(root)
    assert (after["reserved"], after["completed"], after["pending"]) == (2, 2, 0)


def test_active_lease_serializes_concurrent_processes(tmp_path):
    root, first_marker, second_marker = tmp_path / "api", tmp_path / "first", tmp_path / "second"
    script = r'''import json,sys,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import residual_model_call_v1 as M
root,marker,delay,key=Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
def complete(messages, params):
 marker.write_text('entered', encoding='utf-8')
 time.sleep(delay)
 return {'response_id':key,'provider_model':M.C.MODEL,'finish_reason':'stop','content':'{}',
         'reasoning_content':None,'input_tokens':1,'output_tokens':1,'total_tokens':2,'wall_s':delay}
M.MeteredComplete(root,'diagnosis',key*64,'process/'+key,complete)([{'role':'user','content':key}],{})
'''
    first = subprocess.Popen([sys.executable, "-c", script, str(ROOT / "evals"), str(root),
                              str(first_marker), "0.5", "a"], cwd=ROOT)
    deadline = time.monotonic() + 5
    while not first_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert first_marker.exists()
    second = subprocess.Popen([sys.executable, "-c", script, str(ROOT / "evals"), str(root),
                               str(second_marker), "0", "b"], cwd=ROOT)
    time.sleep(0.1)
    assert not second_marker.exists()
    assert first.wait(timeout=5) == 0 and second.wait(timeout=5) == 0
    assert second_marker.exists() and not M._active_path(root).exists()
    audit = M.audit_budget(root)
    assert (audit["reserved"], audit["completed"], audit["pending"]) == (2, 2, 0)


def test_crashed_process_preserves_pending_reservation_and_active_lease(tmp_path, monkeypatch):
    root, marker = tmp_path / "api", tmp_path / "crashed"
    script = r'''import os,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import residual_model_call_v1 as M
root,marker=Path(sys.argv[2]),Path(sys.argv[3])
def crash(messages, params):
 marker.write_text('reserved', encoding='utf-8')
 os._exit(9)
M.MeteredComplete(root,'diagnosis','c'*64,'process/crash',crash)([{'role':'user','content':'x'}],{})
'''
    process = subprocess.run([sys.executable, "-c", script, str(ROOT / "evals"), str(root), str(marker)],
                             cwd=ROOT, check=False)
    assert process.returncode == 9 and marker.exists() and M._active_path(root).is_file()
    audit = M.audit_budget(root)
    assert (audit["reserved"], audit["pending"], audit["completed"]) == (1, 1, 0)
    monkeypatch.setattr(M, "ACTIVE_LEASE_WAIT_S", 0.05)
    with pytest.raises(M.BudgetStop, match="active lease"):
        M.MeteredComplete(root, "diagnosis", "d" * 64, "process/after", lambda *_: {}).__call__(
            [{"role": "user", "content": "after"}], {})
    assert M.audit_budget(root)["pending"] == 1


def test_existing_correction_packet_and_registry_remain_compatible(tmp_path):
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    first = {"ranking": ranking, "action": "continue", "probe": RC.NEW_PROBE,
             "reason": "registered check", "claims": [{"id": "fact", "value": 1}],
             "expected_outcomes": {key: RC.public_catalog()[RC.NEW_PROBE]["expect"][key]
                                   for key in ranking[:3]}}
    state = RC.advance(RC.initial_state(first), RC.NEW_PROBE, "abnormal")
    receipt = {"request_id": "request", "outcome": "abnormal", "measurement": {"steps": [4, 4, 4]}}
    report = {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}}
    packet = RC.correction_packet(state, receipt, report, "a" * 64, "b" * 64)
    proposal = {"correction_id": RC.CORRECTION, "recovery_mode": "restart", "hold_reason": None,
                "cited_facts": ["confirmed_subtype"],
                "checklist": {key: "rule_mode" for key in packet["checklist"]}}

    def complete(messages, params):
        return {"response_id": "correction", "provider_model": M.C.MODEL, "finish_reason": "stop",
                "content": json.dumps(proposal), "reasoning_content": None,
                "input_tokens": 20, "output_tokens": 8, "total_tokens": 28, "wall_s": 0.1}

    result = M.decide(tmp_path / "correction", "correction", packet, tmp_path / "api", caller=complete)
    assert result["review"]["status"] == "valid"
    assert result["review"]["decision"]["agrees_with_rule"] is True


def test_api_cap_is_shared_by_two_folders_and_stops_before_callback_21(tmp_path, simple_contract):
    root, calls = tmp_path / "api", []
    for index in range(20):
        packet = {"case": f"case-{index}"}
        folder = tmp_path / ("seed-a" if index % 2 == 0 else "seed-b") / str(index)
        M.decide(folder, "diagnosis", packet, root,
                 caller=lambda messages, params, p=packet: calls.append(p["case"]) or answer(p))
    assert len(calls) == 20 and M.audit_budget(root)["remaining"] == 0
    with pytest.raises(M.BudgetStop, match="상한"):
        M.decide(tmp_path / "seed-c", "diagnosis", {"case": "over"}, root,
                 caller=lambda messages, params: calls.append("over") or answer({"case": "over"}))
    assert len(calls) == 20


def test_pending_reservation_is_counted_and_blocks_new_request(tmp_path, simple_contract):
    root, calls = tmp_path / "api", []
    M._reserve(root, "diagnosis", "a" * 64, "b" * 64, "seed-a/diagnosis/request1")
    audit = M.audit_budget(root)
    assert audit["reserved"] == 1 and audit["pending"] == 1 and audit["remaining"] == 19
    with pytest.raises(M.BudgetStop, match="미완료"):
        M.decide(tmp_path / "seed-b", "diagnosis", {"case": "b"}, root,
                 caller=lambda messages, params: calls.append(1) or answer({"case": "b"}))
    assert calls == []


def test_response_or_reservation_tampering_is_rejected(tmp_path, simple_contract):
    packet, root, folder = {"case": "tamper"}, tmp_path / "api", tmp_path / "seed"
    M.decide(folder, "diagnosis", packet, root, caller=lambda messages, params: answer(packet))
    response = M.read(root / "responses/0001.json")
    response["result"]["content"] = "{}"
    write(root / "responses/0001.json", response)
    with pytest.raises(ValueError, match="해시 결합"):
        M._review_record(M.read(folder / "diagnosis.dialogue1.json"),
                         (folder / "diagnosis.packet.json").read_bytes(), "diagnosis", root)


def test_transport_uses_fixed_wsl_worker_and_timeout_records_cleanup(tmp_path, monkeypatch):
    packet, output, root = tmp_path / "packet.json", tmp_path / "dialogue.json", tmp_path / "api"
    write(packet, {"case": "transport"})
    seen = {}

    def timeout(argv, **kwargs):
        seen["argv"] = argv
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(M, "wsl_path", lambda path: "/mnt/c/" + Path(path).name)
    monkeypatch.setattr(M.subprocess, "run", timeout)
    monkeypatch.setattr(M, "_stop_worker", lambda pid, source: "verified")
    with pytest.raises(TimeoutError):
        M._run_worker("diagnosis", packet, output, root)
    command = seen["argv"][-1]
    assert '. "$HOME/.config/nvidia/env"' in command
    assert "residual_model_call_v1.py" in command and "--api-root" in command and "--pid-file" in command
    receipt = M.read(output.with_suffix(".transport_error.json"))
    assert receipt["worker_cleanup"] == "verified" and receipt["type"] == "dialogue_timeout"


def test_actual_epoch_one_contract_runs_through_model_call_and_review(tmp_path):
    packet = {"schema": M.C.SCHEMA, "facts": {"telemetry:case:error": 1.2},
              "observed": [], "budget_left": 7, "execution_budget_s": 1200,
              "epoch": 1, "applied_corrections": [], "prior_receipt": None}
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    probe = M.C.NEW_PROBE
    decision = {"ranking": ranking, "action": "continue", "probe": probe,
                "reason": "등록 검사를 요청한다",
                "claims": [{"id": "telemetry:case:error", "value": 1.2}],
                "expected_outcomes": {key: M.C.registry()["probes"][probe]["expect"][key]
                                      for key in ranking[:3]}}

    def complete(messages, params):
        assert params["model"] == M.C.MODEL and "applied_corrections" in messages[-1]["content"]
        return {"response_id": "actual", "provider_model": M.C.MODEL, "finish_reason": "stop",
                "content": json.dumps(decision), "reasoning_content": None,
                "input_tokens": 30, "output_tokens": 12, "total_tokens": 42, "wall_s": 0.1}

    result = M.decide(tmp_path / "actual", "diagnosis", packet, tmp_path / "api", caller=complete)
    assert result["review"]["status"] == "valid"
    assert result["review"]["decision"] == decision
