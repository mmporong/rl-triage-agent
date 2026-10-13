"""E4b NAT 결정 경로: 도구·제출 계약·실제 NAT 워크플로(가짜 스트림, 네트워크 차단)·기록 분류. 모델·GPU 없음."""
import asyncio
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "evals"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rl_triage import nat_demo_tools as T  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


W = load_module("nat_decision_call_under_test", "evals/nat_decision_call.py")


def deny_external_network(monkeypatch):
    original = socket.socket.connect

    def guarded(self, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original(self, address)
        raise AssertionError(f"external network attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)


def diagnosis_packet(tmp_path):
    import run_active_diagnosis as RA
    src = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
           "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    packet = RA.input_packet({"sources": src}, {"max_probes": 4, "probe_execution_budget_s": 180})
    path = tmp_path / "diagnosis.packet.json"
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
    return packet, path


def correction_packet(tmp_path):
    import correction_model_call as CM
    rule = {"mode": "restart", "rule": "reward and standstill_ratio >= 0.9", "correction_id": "restore_body_frame_reward",
            "standstill_ratio": 1.01}
    report = {"protocol": "fixed_eval_v1", "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
              "metrics": {"lin_vel_rmse_mps": 1.19}, "standstill": {"lin_vel_rmse_mps": 1.18}}
    packet = CM.build_packet(case="c", loop_state_digest="s", receipts_digest="r", mechanism="reward",
                             confirming={"probe": "P_reward", "request_id": "q", "outcome": "abnormal",
                                         "measurement": {"track_lin_vel_xy_exp_rel_error": 0.25}},
                             observed={"P_reward": "abnormal"}, rule=rule, source_report=report,
                             source_report_sha256="aa", checkpoint_sha256="bb")
    path = tmp_path / "correction.packet.json"
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
    return packet, path


def bind(monkeypatch, kind, packet_path, tmp_path):
    monkeypatch.setenv("E4B_KIND", kind)
    monkeypatch.setenv("E4B_PACKET", str(packet_path))
    monkeypatch.setenv("E4B_PACKET_SHA256", hashlib.sha256(packet_path.read_bytes()).hexdigest())
    monkeypatch.setenv("E4B_SUBMISSION", str(tmp_path / "run.submission.json"))
    monkeypatch.setenv("E4B_TRACE", str(tmp_path / "run.tools.jsonl"))


def decision(packet, probe="P_reward"):
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    fact = sorted(packet["facts"])[0]
    return {"ranking": ranking, "probe": probe, "action": "continue", "reason": "tracking error stays high",
            "claims": [{"id": fact, "value": packet["facts"][fact]}],
            "expected_outcomes": {m: L.PROBES[probe]["expect"][m] for m in ranking[:3]}}


def proposal():
    return {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
            "cited_facts": ["standstill_ratio"], "checklist": {"1": "eval_metrics", "2": "confirmed_mechanism",
                                                                "3": "confirming_probe", "4": "eval_metrics",
                                                                "5": "unmet", "6": "registered_correction"},
            "hold_reason": None}


# --- 도구 함수 ---
def test_read_tools_bind_packet_sha_and_kind(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    ids = T.list_fact_ids()["fact_ids"]
    assert ids == sorted(packet["facts"])
    got = T.get_facts([ids[0], "telemetry:case:invented"])
    assert got["facts"] == {ids[0]: packet["facts"][ids[0]]} and got["unknown_ids"] == ["telemetry:case:invented"]
    assert set(T.probe_catalog()["probe_catalog"]) == set(L.PROBES)
    with pytest.raises(ValueError):  # 교정 전용 도구
        T.registry()
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):  # packet이 바뀌었다
        T.list_fact_ids()


def test_submit_decision_once_and_reject_stops(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    bad = {**decision(packet), "claims": [{"id": "telemetry:case:invented", "value": 1}]}
    with pytest.raises(T.SubmissionRejected):
        T.submit_decision(bad)
    assert not (tmp_path / "run.submission.json").exists()
    assert json.loads((tmp_path / "run.submission.rejected.json").read_text(encoding="utf-8"))["reasons"]


def test_second_submission_is_rejected(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    saved = json.loads(T.submit_decision(decision(packet)))
    assert saved["saved"] is True
    with pytest.raises(T.SubmissionRejected):
        T.submit_decision(decision(packet, probe="P_noise"))
    body = json.loads((tmp_path / "run.submission.json").read_text(encoding="utf-8"))
    assert body["decision"]["probe"] == "P_reward"


def test_correction_tools_and_proposal_validation(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    bind(monkeypatch, "correction", path, tmp_path)
    assert T.registry()["confirmed_mechanism"] == "reward"
    assert list(T.skill_checklist()["skill_checklist"]) == ["1", "2", "3", "4", "5", "6"]
    with pytest.raises(T.SubmissionRejected):
        T.submit_proposal({**proposal(), "correction_id": "rewrite_reward_code"})
    (tmp_path / "run.submission.rejected.json").unlink()
    json.loads(T.submit_proposal(proposal()))
    body = json.loads((tmp_path / "run.submission.json").read_text(encoding="utf-8"))
    assert body["validation"]["accepted"] and body["validation"]["agrees_with_rule"]


# --- 실제 NAT 워크플로(가짜 스트림) ---
def scripted_stream(monkeypatch, turns, calls):
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI

    async def stream(_self, *args, **kwargs):
        index = len(calls)
        calls.append(kwargs)
        name, args_json, text = turns[min(index, len(turns) - 1)]
        chunk = AIMessageChunk(
            content=text, id=f"offline-{index}",
            tool_call_chunks=[] if name is None else [{"name": name, "args": args_json, "id": f"call-{index}",
                                                         "index": 0, "type": "tool_call_chunk"}],
            usage_metadata={"input_tokens": 100 + index, "output_tokens": 10, "total_tokens": 110 + index},
            chunk_position="last")
        yield ChatGenerationChunk(message=chunk,
                                  generation_info={"finish_reason": "stop" if name is None else "tool_calls"})

    monkeypatch.setattr(ChatOpenAI, "_astream", stream)


def run_call(monkeypatch, tmp_path, kind, packet_path, turns):
    deny_external_network(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    for key in ("E4B_KIND", "E4B_PACKET", "E4B_PACKET_SHA256", "E4B_SUBMISSION", "E4B_TRACE"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    scripted_stream(monkeypatch, turns, calls)
    output = tmp_path / "out" / "run1.json"
    code = W.call(kind, packet_path, output, require_posix=False)
    return code, json.loads(output.read_text(encoding="utf-8")), calls, output


def test_nat_diagnosis_reads_then_submits_valid_decision(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    turns = [("e4b_list_fact_ids", "{}", ""), ("e4b_submit_decision", json.dumps(decision(packet)), ""),
             (None, None, "Submitted.")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 0 and record["status"] == "valid", record
    assert record["decision"]["probe"] == "P_reward" and record["decision"]["ranking"][0] == "reward"
    assert record["tool_sequence"] == ["e4b_list_fact_ids", "e4b_submit_decision"]
    assert record["llm_calls"] == len(calls) == 3 and record["input_tokens"] == 303 and record["output_tokens"] == 30
    assert record["system_prompt_sha256"] and record["events_sha256"] and record["tools_trace_sha256"]
    assert record["atif_sha256"] and record["atif_error"] is None, record.get("atif_error")
    atif = json.loads(output.with_suffix(".atif.json").read_text(encoding="utf-8"))
    assert atif["steps"] and "e4b_submit_decision" in json.dumps(atif)
    events = json.loads(output.with_suffix(".events.json").read_text(encoding="utf-8"))["events"]
    assert {"WORKFLOW_START", "WORKFLOW_END", "LLM_START", "LLM_END", "FUNCTION_START", "FUNCTION_END"} <= {
        e["type"] for e in events}
    assert [t["tool"] for t in record["tool_latencies"]] == record["tool_sequence"]
    assert all(t["latency_s"] is not None and t["latency_s"] >= 0 for t in record["tool_latencies"])
    with pytest.raises(FileExistsError):  # 기록은 덮어쓰지 않는다
        W.call("diagnosis", path, output, require_posix=False)


def test_nat_rejected_submission_stops_without_another_provider_call(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bad = {**decision(packet), "claims": [{"id": "telemetry:case:invented", "value": 1}]}
    turns = [("e4b_submit_decision", json.dumps(bad), ""), (None, None, "should not be reached")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 1 and record["status"] == "invalid" and record["decision"] is None
    assert len(calls) == 1 and record["rejection_sha256"]


def test_nat_correction_submits_valid_proposal(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    turns = [("e4b_skill_checklist", "{}", ""), ("e4b_submit_proposal", json.dumps(proposal()), ""),
             (None, None, "Submitted.")]
    code, record, calls, _ = run_call(monkeypatch, tmp_path, "correction", path, turns)
    assert code == 0 and record["status"] == "valid", record
    assert record["decision"]["decision"] == "propose" and record["decision"]["accepted"]
    assert record["tool_sequence"] == ["e4b_skill_checklist", "e4b_submit_proposal"]


def test_reply_without_submission_is_invalid_and_empty_reply_is_infra(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    code, record, _, _ = run_call(monkeypatch, tmp_path, "diagnosis", path, [(None, None, "I think it is reward.")])
    assert code == 1 and record["status"] == "invalid" and record["error_stage"] == "submission"
    second = tmp_path / "second"
    second.mkdir()
    code, record, _, _ = run_call(monkeypatch, second, "diagnosis", path, [(None, None, "")])
    assert code == 2 and record["status"] == "infra_error"  # NAT가 빈 응답을 예외로 올린다


def test_resubmission_inside_workflow_makes_the_run_invalid(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    good = json.dumps(decision(packet))
    turns = [("e4b_submit_decision", good, ""), ("e4b_submit_decision", good, ""), (None, None, "done")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 1 and record["status"] == "invalid" and record["decision"] is None
    assert output.with_suffix(".submission.json").exists() and record["rejection_sha256"]  # 첫 제출이 있어도 무효
    assert len(calls) == 2


def test_error_after_valid_submission_keeps_the_decision(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    turns = [("e4b_submit_decision", json.dumps(decision(packet)), ""), (None, None, "")]  # 제출 뒤 빈 응답
    code, record, _, _ = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 0 and record["status"] == "valid" and record["decision"]["probe"] == "P_reward"
    assert record["post_submission_error"]["class"] == "infra_error"


def test_observed_packet_and_reasoning_setting(monkeypatch, tmp_path):
    import yaml
    for kind in ("diagnosis", "correction"):
        llm = yaml.safe_load(W.CONFIGS[kind].read_text(encoding="utf-8"))["llms"]["nemotron"]
        assert llm["extra_body"] == {"reasoning_effort": "none"} and llm["temperature"] == 0 and llm["max_retries"] == 0
    packet, _ = diagnosis_packet(tmp_path)
    later = tmp_path / "later.json"
    later.write_text(json.dumps({**packet, "observed": [{"probe": "P_reward"}]}), encoding="utf-8")
    code, record, calls, _ = run_call(monkeypatch, tmp_path, "diagnosis", later, [(None, None, "x")])
    assert code == 1 and record["error_stage"] == "preflight" and calls == []


# --- 분류·요약 단위 ---
def test_classify_error_follows_the_cause_chain():
    class APITimeoutError(Exception):
        pass

    class Wrapped(Exception):
        pass

    try:
        try:
            raise APITimeoutError("timeout")
        except APITimeoutError as inner:
            raise Wrapped("wrapped") from inner
    except Wrapped as error:
        assert W.classify_error(error) == "infra_error"
    status = type("StatusError", (Exception,), {"status_code": 503})("x")
    assert W.classify_error(status) == "infra_error"
    assert W.classify_error(T.SubmissionRejected("bad")) == "invalid"
    assert W.classify_error(asyncio.TimeoutError()) == "infra_error"


def test_summarize_events_pairs_llm_calls_and_measures_tool_latency():
    rows = [{"type": "LLM_START", "uuid": "a", "name": None, "t": 1.0, "span_t": None, "usage": None},
            {"type": "LLM_END", "uuid": "a", "name": None, "t": 2.0, "span_t": 1.0,
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            {"type": "FUNCTION_START", "uuid": "w", "name": "<workflow>", "t": 0.5, "span_t": None, "usage": None},
            {"type": "FUNCTION_START", "uuid": "b", "name": "e4b_get_facts", "t": 2.1, "span_t": None, "usage": None},
            {"type": "FUNCTION_END", "uuid": "b", "name": "e4b_get_facts", "t": 2.4, "span_t": None, "usage": None},
            {"type": "FUNCTION_START", "uuid": "d", "name": "e4b_submit_decision", "t": 2.5, "span_t": None,
             "usage": None},
            {"type": "LLM_START", "uuid": "c", "name": None, "t": 3.0, "span_t": None, "usage": None}]
    out = W.summarize_events(rows)
    assert out["llm_calls"] == 2 and out["llm_call_coverage"] == "incomplete" and out["input_tokens"] is None
    assert out["tool_sequence"] == ["e4b_get_facts", "e4b_submit_decision"]  # 워크플로 자체는 도구가 아니다
    assert out["tool_latencies"][0]["latency_s"] == pytest.approx(0.3) and out["tool_latencies"][1]["latency_s"] is None
    assert out["tool_calls_completed"] == 1


def test_system_prompt_keeps_the_contract_and_user_message_hides_facts(tmp_path):
    from rl_triage import active_diagnosis as D
    packet, _ = diagnosis_packet(tmp_path)
    prompt = W.system_prompt("diagnosis", "TOOL RULES. x")
    assert prompt.startswith(D.SYSTEM_PROMPT.rstrip()) and prompt.rstrip().endswith("TOOL RULES. x")
    message = W.user_message("diagnosis", packet)
    assert "evidence_fact_count" in message and all(fid not in message for fid in packet["facts"])
    assert "probe_catalog" not in message
