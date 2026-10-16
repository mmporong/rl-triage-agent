from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_model_call_v2 as M  # noqa: E402
import r1_repair_contract as RC  # noqa: E402


def packet() -> dict:
    return M.C.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                            facts={"telemetry:error": 1.2, "telemetry:alive": True},
                            input_bindings={"case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "b" * 64},
                                            "reference": {"checkpoint_sha256": "c" * 64, "telemetry_sha256": "d" * 64}})


def valid_selection(value: dict, *, fact: str = "telemetry:error") -> dict:
    probe = M.C.M._next(M.C.C1.registry(), value["remaining_candidates"], value["observed"])
    decision = {"action": "continue", "probe": probe, "fact_ids": [fact]}
    assert M.C.assess(json.dumps(decision), value)["status"] == "valid"
    return decision


def answer(content: object, *, infra: str | None = None, error: str | None = None) -> dict:
    if infra:
        return {"infra_error": infra, "error_type": infra, "wall_s": 0.2}
    if error:
        return {"error_type": error, "wall_s": 0.2}
    text = content if isinstance(content, str) else json.dumps(content)
    return {"response_id": "response", "provider_model": M.MODEL, "finish_reason": "stop",
            "content": text, "reasoning_content": None, "input_tokens": 10,
            "output_tokens": 4, "total_tokens": 14, "wall_s": 0.1}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_valid_selection_writes_bound_receipt_without_rule_transition(tmp_path):
    value = packet()
    result = M.decide(tmp_path / "seed/epoch1/model", "diagnosis", value, tmp_path / "api",
                      caller=lambda _messages, _params: answer(valid_selection(value)))
    assert result["review"]["status"] == "valid"
    assert result["review"]["format_status"] == "valid"
    assert result["review"]["semantic_status"] == "valid"
    assert result["review"]["decision"] == valid_selection(value)
    assert result["review"]["requests"] == 1
    record = M.read(tmp_path / "seed/epoch1/model/diagnosis.dialogue1.json")
    assert record["schema"] == M.SCHEMA and len(record["request_receipts"]) == 1
    assert M.audit_budget(tmp_path / "api")["remaining"] == 19
    assert "selected_by" not in result["review"]


def test_all_format_and_semantic_codes_are_sent_then_retry_can_be_valid(tmp_path):
    value, calls = packet(), []
    malformed = {"action": "bad", "probe": 3, "fact_ids": [], "extra": True}

    def complete(messages, _params):
        calls.append(messages)
        return answer(malformed if len(calls) == 1 else valid_selection(value))

    result = M.decide(tmp_path / "seed/epoch1/model", "diagnosis", value,
                      tmp_path / "api", caller=complete)
    assert result["review"]["status"] == "valid" and result["review"]["retried"] is True
    expected = M.C.assess(json.dumps(malformed), value)["codes"]
    assert len(expected) >= 3
    feedback = calls[1][-1]
    assert feedback["role"] == "user" and all(code in feedback["content"] for code in expected)
    assert M.audit_budget(tmp_path / "api")["reserved"] == 2


def test_two_invalid_responses_remain_invalid_for_controller_to_select(tmp_path):
    value = packet()
    invalid = {"action": "continue", "probe": "P_missing", "fact_ids": ["missing"]}
    result = M.decide(tmp_path / "seed/epoch2/model", "diagnosis", value,
                      tmp_path / "api", caller=lambda _messages, _params: answer(invalid))
    assert result["review"]["status"] == "invalid"
    assert result["review"]["format_status"] == "valid"
    assert result["review"]["semantic_status"] == "invalid"
    assert result["review"]["requests"] == 2
    assert set(result["review"]["codes"]) == {"S_FACT_UNKNOWN", "S_PROBE_UNKNOWN"}
    assert "selected_by" not in result["review"] and M.audit_budget(tmp_path / "api")["reserved"] == 2


def test_infrastructure_error_gets_second_dialogue_but_call_error_does_not(tmp_path):
    value, calls = packet(), []

    def transient(_messages, _params):
        calls.append(1)
        return answer({}, infra="APITimeoutError") if len(calls) == 1 else answer(valid_selection(value))

    recovered = M.decide(tmp_path / "seed-a/epoch1/model", "diagnosis", value,
                         tmp_path / "api-a", caller=transient)
    assert recovered["review"]["status"] == "valid" and len(recovered["attempts"]) == 2
    assert M.audit_budget(tmp_path / "api-a")["timeout"] == 1

    failed = M.decide(tmp_path / "seed-b/epoch1/model", "diagnosis", value,
                      tmp_path / "api-b", caller=lambda *_: answer({}, error="ValueError"))
    assert failed["review"] == {"status": "call_error", "requests": 1, "reason": "ValueError"}
    assert len(failed["attempts"]) == 1


def test_correction_uses_v1_semantics_and_v2_receipt_binding(tmp_path):
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    first = {"ranking": ranking, "action": "continue", "probe": RC.NEW_PROBE,
             "reason": "registered check", "claims": [{"id": "fact", "value": 1}],
             "expected_outcomes": {key: RC.public_catalog()[RC.NEW_PROBE]["expect"][key]
                                   for key in ranking[:3]}}
    state = RC.advance(RC.initial_state(first), RC.NEW_PROBE, "abnormal")
    receipt = {"request_id": "request", "outcome": "abnormal", "measurement": {"steps": [4, 4, 4]}}
    report = {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}}
    value = RC.correction_packet(state, receipt, report, "a" * 64, "b" * 64)
    proposal = {"correction_id": RC.CORRECTION, "recovery_mode": "restart", "hold_reason": None,
                "cited_facts": ["confirmed_subtype"],
                "checklist": {key: "rule_mode" for key in value["checklist"]}}
    result = M.decide(tmp_path / "seed/epoch1/model", "correction", value, tmp_path / "api",
                      caller=lambda *_: answer(proposal))
    assert result["review"]["status"] == "valid"
    assert result["review"]["decision"]["agrees_with_rule"] is True


def test_api_cap_is_shared_and_stops_before_callback_21(tmp_path):
    root, calls = tmp_path / "api", []
    value = packet()
    decision = valid_selection(value)
    for index in range(M.API_CAP):
        M.decide(tmp_path / f"seed-{index}/epoch1/model", "diagnosis", value, root,
                 caller=lambda _messages, _params: calls.append(1) or answer(decision))
    assert len(calls) == M.API_CAP and M.audit_budget(root)["remaining"] == 0
    with pytest.raises(M.BudgetStop, match="상한"):
        M.decide(tmp_path / "over/epoch1/model", "diagnosis", value, root,
                 caller=lambda *_: calls.append(1) or answer(decision))
    assert len(calls) == M.API_CAP


def test_pending_reservation_blocks_callback_and_packet_validation_is_not_fallback(tmp_path):
    root, calls = tmp_path / "api", []
    M.MC1._reserve(root, "diagnosis", "a" * 64, "b" * 64, "seed/epoch/model/request1")
    with pytest.raises(M.BudgetStop, match="미완료"):
        M.decide(tmp_path / "seed/epoch/model", "diagnosis", packet(), root,
                 caller=lambda *_: calls.append(1) or answer({}))
    assert calls == []

    broken = packet()
    broken["schema"] = "tampered"
    with pytest.raises(ValueError, match="packet"):
        M.decide(tmp_path / "broken/epoch/model", "diagnosis", broken, tmp_path / "other-api",
                 caller=lambda *_: calls.append(1) or answer({}))
    assert calls == []


def test_packet_message_feedback_and_response_tampering_are_rejected(tmp_path, monkeypatch):
    value, root, folder = packet(), tmp_path / "api", tmp_path / "seed/epoch1/model"
    contents = [
        {"action": "bad", "probe": None, "fact_ids": []},
        valid_selection(value),
    ]
    M.decide(folder, "diagnosis", value, root,
             caller=lambda _messages, _params: answer(contents.pop(0)))
    frozen = (folder / "diagnosis.packet.json").read_bytes()
    original = M.read(folder / "diagnosis.dialogue1.json")

    detached = json.loads(json.dumps(original))
    detached["unbound_extra"] = True
    with pytest.raises(ValueError, match="실제 파일 내용"):
        M._review_record(detached, frozen, "diagnosis", root,
                         record_path=folder / "diagnosis.dialogue1.json")

    changed = dict(value)
    changed["facts"] = {**changed["facts"], "new": 2}
    with pytest.raises(ValueError):
        M.review(original, M._packet_bytes(changed))

    original_messages = M.C.diagnosis_messages
    monkeypatch.setattr(M.C, "diagnosis_messages",
                        lambda p: [*original_messages(p), {"role": "user", "content": "changed"}])
    with pytest.raises(ValueError, match="메시지"):
        M.review(original, frozen)
    monkeypatch.setattr(M.C, "diagnosis_messages", original_messages)

    feedback_tamper = json.loads(json.dumps(original))
    feedback_tamper["feedback"]["text"] += " altered"
    with pytest.raises(ValueError, match="피드백"):
        M.review(feedback_tamper, frozen)

    response = M.read(root / "responses/0001.json")
    response["result"]["content"] = "{}"
    write(root / "responses/0001.json", response)
    with pytest.raises(ValueError, match="해시 결합"):
        M._review_record(original, frozen, "diagnosis", root,
                         record_path=folder / "diagnosis.dialogue1.json")


def _replace_record_receipt(record_path: Path, receipt: dict) -> dict:
    record = M.read(record_path)
    record["request_receipts"][0] = receipt
    write(record_path, record)
    return record


def test_same_response_receipt_from_different_packet_is_rejected(tmp_path):
    root = tmp_path / "api"
    first = packet()
    second = packet()
    second["facts"]["telemetry:error"] = 9.8
    folders = [tmp_path / "s1/epoch1/model", tmp_path / "s2/epoch1/model"]
    for folder, value in zip(folders, (first, second)):
        M.decide(folder, "diagnosis", value, root,
                 caller=lambda _messages, _params, v=value: answer(valid_selection(v)))
    first_record = M.read(folders[0] / "diagnosis.dialogue1.json")
    second_path = folders[1] / "diagnosis.dialogue1.json"
    second_record = _replace_record_receipt(second_path, first_record["request_receipts"][0])
    with pytest.raises(ValueError, match="kind·packet·메시지·출력 경로"):
        M._review_record(second_record, (folders[1] / "diagnosis.packet.json").read_bytes(),
                         "diagnosis", root, record_path=second_path)


def test_same_packet_receipt_from_different_output_scope_is_rejected(tmp_path):
    root, value = tmp_path / "api", packet()
    folders = [tmp_path / "s1/epoch1/model", tmp_path / "s2/epoch1/model"]
    for folder in folders:
        M.decide(folder, "diagnosis", value, root,
                 caller=lambda _messages, _params: answer(valid_selection(value)))
    first_record = M.read(folders[0] / "diagnosis.dialogue1.json")
    second_path = folders[1] / "diagnosis.dialogue1.json"
    second_record = _replace_record_receipt(second_path, first_record["request_receipts"][0])
    with pytest.raises(ValueError, match="kind·packet·메시지·출력 경로"):
        M._review_record(second_record, (folders[1] / "diagnosis.packet.json").read_bytes(),
                         "diagnosis", root, record_path=second_path)


@pytest.mark.parametrize(("field", "replacement"), [
    ("kind", "correction"),
    ("messages_sha256", "f" * 64),
])
def test_reservation_kind_or_messages_mismatch_is_rejected(tmp_path, field, replacement):
    root, value, folder = tmp_path / "api", packet(), tmp_path / "seed/epoch1/model"
    M.decide(folder, "diagnosis", value, root,
             caller=lambda _messages, _params: answer(valid_selection(value)))
    reservation_path, response_path = root / "reservations/0001.json", root / "responses/0001.json"
    reservation, response = M.read(reservation_path), M.read(response_path)
    reservation[field] = replacement
    response[field] = replacement
    response["reservation_sha256"] = M.MC1.digest(reservation)
    write(reservation_path, reservation)
    write(response_path, response)
    record_path = folder / "diagnosis.dialogue1.json"
    record = M.read(record_path)
    record["request_receipts"][0] = {
        "request_index": 1, "reservation_sha256": M.MC1.digest(reservation),
        "response_sha256": M.file_sha(response_path),
    }
    write(record_path, record)
    with pytest.raises(ValueError, match="kind·packet·메시지·출력 경로"):
        M._review_record(record, (folder / "diagnosis.packet.json").read_bytes(),
                         "diagnosis", root, record_path=record_path)


def test_duplicate_request_receipt_is_rejected(tmp_path):
    root, value, folder = tmp_path / "api", packet(), tmp_path / "seed/epoch1/model"
    invalid = {"action": "continue", "probe": "P_missing", "fact_ids": ["missing"]}
    M.decide(folder, "diagnosis", value, root, caller=lambda *_: answer(invalid))
    record_path = folder / "diagnosis.dialogue1.json"
    record = M.read(record_path)
    record["request_receipts"][1] = record["request_receipts"][0]
    write(record_path, record)
    with pytest.raises(ValueError, match="요청 순서"):
        M._review_record(record, (folder / "diagnosis.packet.json").read_bytes(),
                         "diagnosis", root, record_path=record_path)


def test_transport_uses_v2_worker_and_timeout_records_verified_cleanup(tmp_path, monkeypatch):
    source, output, root = tmp_path / "packet.json", tmp_path / "dialogue.json", tmp_path / "api"
    write(source, packet())
    seen = {}

    def timeout(argv, **kwargs):
        seen["argv"] = argv
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def stop(pid_file, packet_path):
        assert pid_file == output.with_suffix(".pid.json") and packet_path == source
        return "verified"

    monkeypatch.setattr(M, "wsl_path", lambda path: "/mnt/c/" + Path(path).name)
    monkeypatch.setattr(M.subprocess, "run", timeout)
    monkeypatch.setattr(M, "_stop_worker", stop)
    with pytest.raises(TimeoutError, match="시간 상한"):
        M._run_worker(source, output, root)
    command = seen["argv"][-1]
    assert '. "$HOME/.config/nvidia/env"' in command
    assert "residual_model_call_v2.py" in command and "--kind diagnosis" in command
    receipt = M.read(output.with_suffix(".transport_error.json"))
    assert receipt["worker_cleanup"] == "verified" and receipt["type"] == "dialogue_timeout"
    assert set(receipt) == {"schema", "type", "worker_cleanup", "at"}


def test_worker_identity_contains_exact_argv_and_packet_hash(tmp_path, monkeypatch):
    source = tmp_path / "packet.json"
    write(source, packet())
    fake_stat = "99 (python) S " + " ".join(str(index) for index in range(1, 30))
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.as_posix() == "/proc/self/stat":
            return fake_stat
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(M.sys, "argv", ["residual_model_call_v2.py", "--packet", str(source)])
    identity = M._worker_identity(source)
    assert identity["argv"] == M.sys.argv and identity["packet_sha256"] == M.file_sha(source)
    assert identity["pid"] == M.os.getpid() and type(identity["start_ticks"]) is int
