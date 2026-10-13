"""E4b v3: null 없는 교정 제출 스키마의 어댑터·점검·제출과 실제 NAT 워크플로(가짜 스트림, 네트워크 차단). 모델·GPU 없음.
패킷·스트림 보조 함수는 v1 테스트(test_nat_demo.py)의 것을 쓴다."""
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

from rl_triage import nat_demo_tools_v3 as T  # noqa: E402
from test_nat_demo import bind, correction_packet, decision, deny_external_network, diagnosis_packet, scripted_stream  # noqa: E402

spec = importlib.util.spec_from_file_location("nat_decision_call_v3_under_test", ROOT / "evals/nat_decision_call_v3.py")
W = importlib.util.module_from_spec(spec)
sys.modules["nat_decision_call_v3_under_test"] = W
spec.loader.exec_module(W)

CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}


def propose(**over):
    return {"decision": "propose", "correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
            "cited_facts": ["standstill_ratio"], "checklist": dict(CHECKLIST), "hold_reason": "", **over}


def hold(**over):
    return {"decision": "hold", "correction_id": "", "recovery_mode": "", "cited_facts": ["standstill_ratio"],
            "checklist": dict(CHECKLIST), "hold_reason": "confirming evidence is not specific enough", **over}


# --- 어댑터 ---
def test_adapter_maps_to_the_public_contract_and_refuses_mismatched_fields():
    proposal, reasons = T.to_contract(propose())
    assert reasons == [] and proposal["hold_reason"] is None and proposal["recovery_mode"] == "restart"
    proposal, reasons = T.to_contract(hold())
    assert reasons == [] and proposal["correction_id"] is None and proposal["recovery_mode"] is None
    for bad in (propose(hold_reason="None"), propose(correction_id=""), hold(correction_id="restore_body_frame_reward"),
                hold(hold_reason="  ")):
        proposal, reasons = T.to_contract(bad)
        assert proposal is None and reasons  # 판별 칸과 맞지 않으면 고치지 않고 사유를 돌려준다


def test_schema_has_no_null_and_explains_empty_strings():
    schema = json.dumps(T.ProposalInput.model_json_schema(), ensure_ascii=False)
    assert '"null"' not in schema and 'the empty string \\"\\"' in schema


def test_check_and_submit_use_the_same_contract_once(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    bind(monkeypatch, "correction", path, tmp_path)
    assert T.check_proposal(propose(hold_reason="None"))["ok"] is False
    assert T.check_proposal(propose(checklist={**CHECKLIST, "4": "eval_metrics, confirming_probe"}))["ok"] is False
    assert T.check_proposal(propose()) == {"ok": True, "checked": True, "reasons": []}
    json.loads(T.submit_proposal(propose()))
    body = json.loads((tmp_path / "run.submission.json").read_text(encoding="utf-8"))
    assert body["proposal"]["hold_reason"] is None and body["validation"]["accepted"] and body["tool_args"] == propose()
    with pytest.raises(T.SubmissionRejected):
        T.submit_proposal(propose())


def test_precheck_limit_counts_v3_checks(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    bind(monkeypatch, "correction", path, tmp_path)
    (tmp_path / "run.tools.jsonl").write_text("".join(json.dumps({"tool": name}) + "\n" for name in
                                                      ("e4b2_check_proposal", "e4b3_check_proposal", "e4b3_check_proposal")),
                                              encoding="utf-8")
    assert T.precheck_calls(str(tmp_path / "run.tools.jsonl")) == 2  # v2 점검 이름은 세지 않는다
    out = T.check_proposal(propose())
    assert out["checked"] is False and out["ok"] is False and "limit" in out["reasons"][0]


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


def test_correction_propose_and_hold_are_valid_without_null(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    turns = [("e4b2_skill_checklist", "{}", ""), ("e4b3_check_proposal", json.dumps(propose(hold_reason="None")), ""),
             ("e4b3_check_proposal", json.dumps(propose()), ""), ("e4b3_submit_proposal", json.dumps(propose()), ""),
             (None, None, "Submitted.")]
    code, record, calls, _ = run_call(monkeypatch, tmp_path, "correction", path, turns)
    assert code == 0 and record["status"] == "valid" and record["decision"]["decision"] == "propose", record
    assert record["schema"] == "e4b_nat_decision_v3" and record["precheck_calls"] == 2
    assert set(record["tools_modules_sha256_lf"]) == {"v1", "v2", "v3"}
    tools = json.dumps(calls[0], ensure_ascii=False, default=str)
    assert "e4b3_submit_proposal" in tools and "e4b2_submit_proposal" not in tools
    second = tmp_path / "hold"
    second.mkdir()
    code, record, _, _ = run_call(monkeypatch, second, "correction", path,
                                  [("e4b3_submit_proposal", json.dumps(hold()), ""), (None, None, "Held.")])
    assert code == 0 and record["decision"]["decision"] == "HOLD" and not record["decision"]["agrees_with_rule"]


def test_diagnosis_path_is_unchanged_from_v2(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    turns = [("e4b2_check_decision", json.dumps(decision(packet)), ""),
             ("e4b2_submit_decision", json.dumps(decision(packet)), ""), (None, None, "Submitted.")]
    code, record, _, _ = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 0 and record["status"] == "valid" and record["precheck_calls"] == 1
    assert W.CONFIGS["diagnosis"] == W.W2.CONFIGS["diagnosis"]
    assert record["system_prompt_sha256"] == W.W1.text_sha(W.W2.system_prompt("diagnosis"))


def test_v3_correction_config_keeps_model_settings():
    v2 = yaml.safe_load(W.W2.CONFIGS["correction"].read_text(encoding="utf-8"))
    v3 = yaml.safe_load(W.CONFIGS["correction"].read_text(encoding="utf-8"))
    assert v3["llms"] == v2["llms"] and v3["workflow"]["max_iterations"] == v2["workflow"]["max_iterations"]
    assert set(v3["workflow"]["tool_names"]) == set(v3["functions"])
    assert {"e4b3_check_proposal", "e4b3_submit_proposal"} <= set(v3["functions"])
