"""E4b v4: 실수 표기 고정·표기 일치 치환(진단)과 packet 등록 표 검증(교정), 실제 NAT 워크플로(가짜 스트림, 네트워크
차단). 모델·GPU 없음. 패킷·스트림 보조 함수는 v1 테스트, 교정 인자 보조 함수는 v3 테스트의 것을 쓴다."""
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

from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import nat_demo_tools_v4 as T  # noqa: E402
from test_nat_demo import bind, correction_packet, decision, deny_external_network, diagnosis_packet, scripted_stream  # noqa: E402
from test_nat_demo_v3 import propose  # noqa: E402

spec = importlib.util.spec_from_file_location("nat_decision_call_v4_under_test", ROOT / "evals/nat_decision_call_v4.py")
W = importlib.util.module_from_spec(spec)
sys.modules["nat_decision_call_v4_under_test"] = W
spec.loader.exec_module(W)


def float_fact(packet):
    fid = next(k for k, v in sorted(packet["facts"].items()) if isinstance(v, float) and len(repr(v)) > 12)
    return fid, packet["facts"][fid]


def cited(packet, value):
    fid, _ = float_fact(packet)
    return {**decision(packet), "claims": [{"id": fid, "value": value}]}


# --- 표기와 치환 ---
def test_display_keeps_non_floats_and_rounds_floats():
    assert T.display(0.36971601049105324) == "0.369716" and T.display(1.7379e-08) == "1.7379e-08"
    assert T.display({"a": [1, True, "x", 2.0]}) == {"a": [1, True, "x", "2"]}


def test_only_the_shown_representation_maps_to_the_stored_value(tmp_path):
    packet, _ = diagnosis_packet(tmp_path)
    fid, fact = float_fact(packet)
    shown = T.display(fact)
    for value in (shown, float(shown)):  # 보여 준 문자열 또는 그 숫자
        mapped = T.claims_to_contract(cited(packet, value), packet)
        assert mapped["claims"][0]["value"] == fact and D.validate_decision(mapped, packet)
    for value in (repr(fact)[:-1], float(repr(fact)[:-2]), fact * (1 + 1e-12), "0.37"):  # v3 실패형·근사값·다른 표기
        mapped = T.claims_to_contract(cited(packet, value), packet)
        with pytest.raises(ValueError, match="claim"):
            D.validate_decision(mapped, packet)
    exact = T.claims_to_contract(cited(packet, fact), packet)  # 원 값을 그대로 인용하면 그대로 통과(v1과 같다)
    assert D.validate_decision(exact, packet)["claims"][0]["value"] == fact


def test_get_facts_shows_strings_and_check_submit_use_the_mapping(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    bind(monkeypatch, "diagnosis", path, tmp_path)
    fid, fact = float_fact(packet)
    assert T.get_facts([fid])["facts"] == {fid: T.display(fact)}
    assert T.check_decision(cited(packet, repr(fact)[:-1]))["ok"] is False
    assert T.check_decision(cited(packet, T.display(fact)))["ok"] is True
    json.loads(T.submit_decision(cited(packet, T.display(fact))))
    body = json.loads((tmp_path / "run.submission.json").read_text(encoding="utf-8"))
    assert body["decision"]["claims"][0]["value"] == fact and body["tool_args"]["claims"][0]["value"] == T.display(fact)


def test_diagnosis_schema_names_v4_tools_and_keeps_shown_strings(tmp_path):
    packet, _ = diagnosis_packet(tmp_path)
    fid, fact = float_fact(packet)
    schema = json.dumps(T.DecisionInput.model_json_schema())
    assert "e4b4_get_facts" in schema and "e4b2_get_facts" not in schema  # 모델이 보는 설명이 v4 도구만 가리킨다
    args = T.DecisionInput(**cited(packet, T.display(fact))).model_dump(mode="python")
    assert args["claims"][0]["value"] == T.display(fact)  # 입력 스키마가 표기 문자열을 수로 바꾸지 않는다


def test_correction_uses_the_packet_registry(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    widened = {**packet, "registry": {**packet["registry"], "reward": {"correction_id": "restore_body_frame_reward",
                                                                       "recovery_modes": ["restart"]}}}
    path.write_text(json.dumps(widened), encoding="utf-8")
    bind(monkeypatch, "correction", path, tmp_path)
    assert T.check_proposal(propose())["ok"] is True
    assert T.check_proposal(propose(recovery_mode="resume"))["ok"] is False  # packet 표에 없는 회복 방식


def test_precheck_limit_counts_only_v4_checks(monkeypatch, tmp_path):
    for kind, make, args, check in (("diagnosis", diagnosis_packet, None, T.check_decision),
                                    ("correction", correction_packet, propose(), T.check_proposal)):
        packet, path = make(tmp_path)
        bind(monkeypatch, kind, path, tmp_path)
        older = {"diagnosis": "e4b2_check_decision", "correction": "e4b3_check_proposal"}[kind]
        (tmp_path / "run.tools.jsonl").write_text("".join(json.dumps({"tool": name}) + "\n" for name in
                                                          (older, T.PRECHECK_TOOLS[kind][0], T.PRECHECK_TOOLS[kind][0])),
                                                  encoding="utf-8")
        assert T.precheck_calls(kind, str(tmp_path / "run.tools.jsonl")) == 2  # 앞 판 점검 이름은 세지 않는다
        out = check(args or decision(packet))
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


def test_diagnosis_with_v3_failure_value_then_shown_value(monkeypatch, tmp_path):
    packet, path = diagnosis_packet(tmp_path)
    fid, fact = float_fact(packet)
    bad, good = json.dumps(cited(packet, repr(fact)[:-1])), json.dumps(cited(packet, T.display(fact)))
    turns = [("e4b4_list_fact_ids", "{}", ""), ("e4b4_get_facts", json.dumps({"ids": [fid]}), ""),
             ("e4b4_check_decision", bad, ""),
             ("e4b4_check_decision", good, ""), ("e4b4_submit_decision", good, ""), (None, None, "Submitted.")]
    code, record, calls, output = run_call(monkeypatch, tmp_path, "diagnosis", path, turns)
    assert code == 0 and record["status"] == "valid" and record["decision"]["claims"][0]["value"] == fact, record
    assert record["schema"] == "e4b_nat_decision_v4" and record["precheck_calls"] == 2
    assert set(record["tools_modules_sha256_lf"]) == {"v1", "v2", "v3", "v4"}
    trace = [json.loads(line) for line in output.with_suffix(".tools.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace[0]["tool"] == "e4b4_list_fact_ids" and T.display(fact) in trace[1]["result_head"]  # 모델이 받은 값은 6자리 표기다
    tools = json.dumps(calls[0]["tools"], ensure_ascii=False)  # 모델에게 보낸 도구 정의
    assert "e4b4_get_facts" in tools and "e4b2_get_facts" not in tools


def test_correction_and_configs(monkeypatch, tmp_path):
    packet, path = correction_packet(tmp_path)
    turns = [("e4b4_submit_proposal", json.dumps(propose()), ""), (None, None, "Submitted.")]
    code, record, _, _ = run_call(monkeypatch, tmp_path, "correction", path, turns)
    assert code == 0 and record["status"] == "valid" and record["decision"]["decision"] == "propose"
    for kind in ("diagnosis", "correction"):
        v2 = yaml.safe_load(W.W2.CONFIGS[kind].read_text(encoding="utf-8"))
        v4 = yaml.safe_load(W.CONFIGS[kind].read_text(encoding="utf-8"))
        assert v4["llms"] == v2["llms"] and set(v4["workflow"]["tool_names"]) == set(v4["functions"])
        assert all(name.startswith(("e4b2_", "e4b4_")) for name in v4["functions"])
