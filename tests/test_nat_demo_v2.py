"""E4b v2 NAT 결정 경로: 점검 도구가 v1 실패 형태를 잡는지, 모델이 보는 스키마 설명, 실제 NAT 워크플로(가짜 스트림, 네트워크
차단)에서 점검 뒤 제출·상한 기록. 모델·GPU 없음. 패킷·스트림 보조 함수는 v1 테스트(test_nat_demo.py)의 것을 쓴다."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "evals", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rl_triage import nat_demo_tools_v2 as T  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from test_nat_demo import (bind, correction_packet, decision, deny_external_network, diagnosis_packet,  # noqa: E402
                           proposal, scripted_stream)

spec = importlib.util.spec_from_file_location("nat_decision_call_v2_under_test", ROOT / "evals/nat_decision_call_v2.py")
W = importlib.util.module_from_spec(spec)
sys.modules["nat_decision_call_v2_under_test"] = W
spec.loader.exec_module(W)


def six_expected(packet):
    """v1 진단 무효 형태: 상위 3가설 대신 메커니즘 6개 전부의 기대 결과."""
    return {**decision(packet), "expected_outcomes": {m: L.PROBES["P_reward"]["expect"][m] for m in L.MECHANISMS}}


# --- 점검 도구 ---
def test_precheck_catches_v1_failure_shapes_without_saving(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    bad = T.check_decision(six_expected(packet))
    assert bad["ok"] is False and bad["checked"] and "expected_outcomes" in bad["reasons"][0]
    assert T.check_decision(decision(packet)) == {"ok": True, "checked": True, "reasons": []}
    assert not (tmp_path / "run.submission.json").exists()  # 점검은 저장하지 않는다

    cpacket, cpath = correction_packet(tmp_path)
    bind(monkeypatch, "correction", cpath, tmp_path)
    shapes = {"hold_reason": {**proposal(), "hold_reason": "None"},
              "comma": {**proposal(), "checklist": {**proposal()["checklist"], "4": "eval_metrics, confirming_probe"}},
              "correction id": {**proposal(), "checklist": {**proposal()["checklist"], "6": "restore_body_frame_reward"}}}
    for name, value in shapes.items():
        out = T.check_proposal(value)
        assert out["ok"] is False and out["reasons"], name
    assert "HOLD" in T.check_proposal(shapes["hold_reason"])["reasons"][0]
    assert T.check_proposal(proposal())["ok"] is True


def test_precheck_limit_counts_recorded_checks(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    trace = tmp_path / "run.tools.jsonl"
    trace.write_text("".join(json.dumps({"tool": name}) + "\n" for name in
                             ("e4b2_list_fact_ids", "e4b2_check_decision", "e4b2_check_decision")), encoding="utf-8")
    assert T.precheck_calls(str(trace)) == 2
    out = T.check_decision(decision(packet))
    assert out["ok"] is False and out["checked"] is False and "limit" in out["reasons"][0]


def test_submission_is_judged_by_the_same_contract_once(monkeypatch, tmp_path):
    cpacket, cpath = correction_packet(tmp_path)
    bind(monkeypatch, "correction", cpath, tmp_path)
    with pytest.raises(T.SubmissionRejected):
        T.submit_proposal({**proposal(), "hold_reason": "None"})
    body = json.loads((tmp_path / "run.submission.rejected.json").read_text(encoding="utf-8"))
    assert body["tool"] == "e4b2_submit_proposal"
    with pytest.raises(T.SubmissionRejected):  # 거절 뒤에는 고친 제안도 받지 않는다
        T.submit_proposal(proposal())


# --- 실제 NAT 워크플로(가짜 스트림) ---
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


def test_diagnosis_fixes_after_precheck_then_submits(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    turns = [("e4b2_list_fact_ids", "{}", ""), ("e4b2_check_decision", json.dumps(six_expected(packet)), ""),
             ("e4b2_check_decision", json.dumps(decision(packet)), ""),
             ("e4b2_submit_decision", json.dumps(decision(packet)), ""), (None, None, "Submitted.")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 0 and record["status"] == "valid", record
    assert record["schema"] == "e4b_nat_decision_v2" and record["precheck_calls"] == 2 and record["precheck_limit"] == 2
    assert record["tool_sequence"] == ["e4b2_list_fact_ids", "e4b2_check_decision", "e4b2_check_decision",
                                       "e4b2_submit_decision"]
    assert record["tools_module_sha256_lf"] == W.W1.lf_sha(W.TOOLS_MODULE)
    assert record["base_tools_module_sha256_lf"] == W.W1.lf_sha(W.BASE_TOOLS_MODULE)
    assert record["system_prompt_sha256"] == W.W1.text_sha(W.system_prompt("diagnosis"))
    trace = [json.loads(line) for line in output.with_suffix(".tools.jsonl").read_text(encoding="utf-8").splitlines()]
    first_check = next(row for row in trace if row["tool"] == "e4b2_check_decision")
    assert '"ok": false' in first_check["result_head"]  # 모델이 받은 점검 결과가 기록에 남는다
    tools = json.dumps(calls[0], ensure_ascii=False, default=str)
    assert "TOP THREE" not in tools and "ONLY the top three mechanisms" in tools  # 스키마 설명이 모델에 전달된다


def test_correction_hold_none_is_caught_by_precheck(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    turns = [("e4b2_skill_checklist", "{}", ""), ("e4b2_check_proposal", json.dumps({**proposal(), "hold_reason": "None"}), ""),
             ("e4b2_check_proposal", json.dumps(proposal()), ""), ("e4b2_submit_proposal", json.dumps(proposal()), ""),
             (None, None, "Submitted.")]
    code, record, calls, _ = run_call(monkeypatch, tmp_path, "correction", path, turns)
    assert code == 0 and record["status"] == "valid" and record["decision"]["decision"] == "propose", record
    assert record["precheck_calls"] == 2
    tools = json.dumps(calls[0], ensure_ascii=False, default=str)
    assert "MUST be JSON null" in tools and "exactly ONE fact id" in tools  # hold_reason·체크리스트 설명


def test_over_limit_checks_are_recorded_and_rejection_stops(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    bad = json.dumps({**proposal(), "hold_reason": "None"})
    turns = [("e4b2_check_proposal", bad, ""), ("e4b2_check_proposal", bad, ""), ("e4b2_check_proposal", bad, ""),
             ("e4b2_submit_proposal", bad, ""), (None, None, "should not be reached")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "correction", path, turns)
    assert code == 1 and record["status"] == "invalid" and record["rejection_sha256"]
    assert record["precheck_calls"] == 3 and len(calls) == 4  # 거절 뒤 제공자 호출 없음
    trace = [json.loads(line) for line in output.with_suffix(".tools.jsonl").read_text(encoding="utf-8").splitlines()]
    assert '"checked": false' in [row for row in trace if row["tool"] == "e4b2_check_proposal"][2]["result_head"]


def test_v2_configs_keep_v1_model_settings_and_register_every_tool():
    from nat.cli.type_registry import GlobalTypeRegistry
    registered = {info.local_name for info in GlobalTypeRegistry.get().get_registered_functions()}
    for kind in ("diagnosis", "correction"):
        v1 = yaml.safe_load(W.W1.CONFIGS[kind].read_text(encoding="utf-8"))
        v2 = yaml.safe_load(W.CONFIGS[kind].read_text(encoding="utf-8"))
        assert v2["llms"] == v1["llms"]
        assert set(v2["workflow"]["tool_names"]) == set(v2["functions"]) <= registered
        assert all(name.startswith("e4b2_") for name in v2["functions"])
        assert v2["workflow"]["max_iterations"] == 12 and v2["workflow"]["handle_tool_errors"] is False
