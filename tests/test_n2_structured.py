"""N2 구조화 출력 arm: schema가 구조만 정해 의미 위반을 지우지 않는지, 기존 검토 함수로 같은 판정을 내는지, 무효 갈래,
실행 상한·재시도, 사전등록 개선 판정. 모델 호출·GPU·비공개 packet 없음(진단 packet은 공개 기록에서 만든다)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_replay_page as B  # noqa: E402
import e4b_compare_v4 as E4  # noqa: E402
import n2_pass_k as N  # noqa: E402
import n2_structured as T  # noqa: E402
import structured_model_call as S  # noqa: E402
from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

SEMANTIC_KEYS = {"enum", "const", "minItems", "maxItems", "minLength", "maxLength", "pattern", "minimum", "maximum",
                 "uniqueItems", "minProperties", "maxProperties", "propertyNames", "format"}


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


# --- schema 정책 ---
@pytest.mark.parametrize("kind", ["diagnosis", "correction"])
def test_schemas_fix_structure_only(kind):
    schema = S.SCHEMAS[kind]
    jsonschema.Draft202012Validator.check_schema(schema)
    assert not {key for node in _walk(schema) for key in node} & SEMANTIC_KEYS  # 값 목록·최소 개수 같은 의미 규칙 없음
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    assert S.response_format(kind) == {"type": "json_schema", "json_schema": {
        "name": S.NAMES[kind], "schema": schema, "strict": True}}
    assert set(schema["properties"]) == ({"ranking", "probe", "action", "reason", "claims", "expected_outcomes"}
                                         if kind == "diagnosis" else set(N2_KEYS))


N2_KEYS = ("correction_id", "recovery_mode", "cited_facts", "checklist", "hold_reason")


def _packet(tmp_path) -> Path:
    path = tmp_path / "h02_s2027.diagnosis.packet.json"
    path.write_text(json.dumps(E4.public_packet("h02_s2027"), ensure_ascii=False), encoding="utf-8")
    return path


def _decision(packet: dict, **change) -> dict:
    fact_id, value = next(iter(D._known_facts(packet).items()))
    ranking = list(L.MECHANISMS)
    decision = {"ranking": ranking, "probe": "P_reward", "action": "continue", "reason": "r",
                "claims": [{"id": fact_id, "value": value}],
                "expected_outcomes": {m: L.PROBES["P_reward"]["expect"][m] for m in ranking[:3]}}
    return {**decision, **change}


def test_semantic_violations_still_pass_the_schema_and_fail_the_validator(tmp_path):
    packet = json.loads(_packet(tmp_path).read_text(encoding="utf-8"))
    good = _decision(packet)
    jsonschema.validate(good, S.SCHEMAS["diagnosis"])
    D.validate_decision(good, packet)
    for bad in (_decision(packet, claims=[]), _decision(packet, expected_outcomes={"reward": "normal"})):
        jsonschema.validate(bad, S.SCHEMAS["diagnosis"])  # 형식은 통과한다
        with pytest.raises(ValueError):
            D.validate_decision(bad, packet)  # 의미 위반은 검증기가 센다
    hold = {"correction_id": None, "recovery_mode": None, "cited_facts": [], "checklist": {}, "hold_reason": None}
    jsonschema.validate(hold, S.SCHEMAS["correction"])  # 빈 인용·빈 체크리스트도 형식은 통과(검증기가 거절)


def test_invalid_category():
    broken = '{\n  "checklist": {\n    "1": "eval_metrics",\n    "6": "rule"\n  ],\n  "hold_reason": null\n}'
    assert T.invalid_category(broken) == "syntax" and T.invalid_category("[1, 2]") == "syntax"
    assert T.invalid_category('```json\n{"claims": []}\n```') == "contract"


# --- 기존 검토 함수로 같은 판정 ---
def _record(kind: str, packet_path: Path, content: str, **change) -> dict:
    messages, binding = S.build(kind, packet_path.read_bytes())
    record = {"schema": S.RECORD_SCHEMA, "kind": kind, **binding, "response_format_sha256": S.response_format_sha256(kind),
              "request": {"model": N.MODEL, "temperature": 0, "max_tokens": 16384, "stream": False,
                          "extra_body": {"reasoning_effort": "none"}, "response_format": S.response_format(kind)},
              "content": content, "finish_reason": "stop", "exit_code": 0, "input_tokens": 2800, "output_tokens": 190,
              "execution_wall_s": 1.9}
    return {**record, **change}


def test_review_reuses_the_existing_reviewers(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    good = T.review("diagnosis", _record("diagnosis", path, json.dumps(_decision(packet))), path)
    assert good["status"] == "valid" and N.decision_key("diagnosis", good["decision"]) == ("continue", "reward")
    empty = T.review("diagnosis", _record("diagnosis", path, json.dumps(_decision(packet, claims=[]))), path)
    assert (empty["status"], empty["category"]) == ("invalid", "contract") and "claims" in empty["reason"]
    broken = T.review("diagnosis", _record("diagnosis", path, '{"ranking": ['), path)
    assert (broken["status"], broken["category"]) == ("invalid", "syntax")
    assert T.review("diagnosis", _record("diagnosis", path, "", infra_error="APITimeoutError"), path)["status"] == "infra_error"
    with pytest.raises(ValueError, match="출력 형식"):  # 출력 형식 없이 부른 기록은 받지 않는다
        T.review("diagnosis", _record("diagnosis", path, "{}", request={"model": N.MODEL, "temperature": 0}), path)
    with pytest.raises(ValueError, match="구조화"):
        T.review("diagnosis", _record("diagnosis", path, "{}", schema="active_model_call_v1"), path)
    with pytest.raises(ValueError):  # 다른 packet의 기록
        T.review("diagnosis", _record("diagnosis", path, "{}", packet_sha256="0" * 64), path)


# --- 실행 ---
@pytest.fixture
def private(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PRIVATE", tmp_path / "n2s")
    monkeypatch.setattr(T, "verify_protocol", lambda: ({}, "c" * 40))
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: tmp_path / "packets" / set_name / f"{kind}.packet.json")
    return tmp_path


def fake_runner(infra_first: set, calls: list):
    def run(argv):
        output = Path("C:/" + argv[-1].split("--output ")[1].split(" ")[0][len("/mnt/c/"):])
        set_name, kind, name = output.parts[-3], output.parts[-2], output.parts[-1]
        run_no, attempt = int(name[3:].split("_")[0]), int(name[-6])
        calls.append((set_name, kind, run_no, attempt))
        record = {"exit_code": 0}
        if attempt == 1 and (set_name, kind, run_no) in infra_first:
            record.update(infra_error="APIConnectionError", exit_code=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record), encoding="utf-8")
        return record["exit_code"]
    return run


def test_execute_follows_the_n2_order_retries_infra_once_and_stops_at_the_cap(private, monkeypatch):
    calls = []
    out = T.execute(fake_runner({("f2_s2033", "correction", 2)}, calls))
    assert out["stopped"] is None and out["requests_used"] == 56 and out["remaining"] == 0
    assert [c[:3] for c in calls if c[3] == 1] == N.schedule() and ("f2_s2033", "correction", 2, 2) in calls
    monkeypatch.setattr(T, "PRIVATE", private / "n2s_cap")
    monkeypatch.setattr(T, "API_CAP", 4)
    capped = []
    out = T.execute(fake_runner(set(), capped))
    assert out["stopped"] == "budget_exhausted" and len(capped) == 4 and out["remaining"] == 51


def test_execute_resumes_a_run_whose_first_attempt_was_an_infra_error(private):
    first = T.run_path("h02_s2027", "diagnosis", 1, 1)
    first.parent.mkdir(parents=True)
    first.write_text(json.dumps({"infra_error": "APIConnectionError", "exit_code": 2}), encoding="utf-8")
    calls = []
    T.execute(fake_runner(set(), calls))
    assert calls[0] == ("h02_s2027", "diagnosis", 1, 2) and ("h02_s2027", "diagnosis", 1, 1) not in calls


def test_call_argv_uses_the_structured_caller_and_sources_credentials(private):
    shell = T.call_argv("u1_s2040", "correction", 2, 1)[-1]
    assert shell.startswith('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" ')
    assert "evals/structured_model_call.py --kind correction --packet " in shell and f"--model {N.MODEL}" in shell
    assert "/n2s/runs/u1_s2040/correction/run2_a1.json" in shell and "NVIDIA_API_KEY" not in shell


# --- 사전등록 ---
def test_fixed_protocol_and_budget():
    assert json.loads(json.dumps(T.FIXED)) == T.FIXED and T.FIXED["schemas"] == S.SCHEMAS
    assert T.FIXED["api_budget"]["planned_requests"] == 55 <= T.API_CAP == 60
    assert T.FIXED["variable"]["changed"].startswith("response_format")
    assert T.N2_SUMMARY in T.public_inputs() and "evals/structured_model_call.py" in T.CODE


@pytest.mark.parametrize("totals,means,complete,decision", [
    ({"invalid": 1, "invalid_contract": 1}, {"diagnosis": 0.8, "correction": 1.0}, True, "improved"),
    ({"invalid": 2, "invalid_contract": 2}, {"diagnosis": 0.9, "correction": 1.0}, True, "not_improved"),
    ({"invalid": 3, "invalid_syntax": 1, "invalid_contract": 2}, {"diagnosis": 0.6, "correction": 0.9}, True, "not_improved"),
    ({"invalid": 0}, {"diagnosis": 0.79, "correction": 1.0}, True, "not_improved"),
    ({"invalid": 0}, {"diagnosis": None, "correction": 1.0}, True, "insufficient"),
    ({"invalid": 0}, {"diagnosis": 1.0, "correction": 1.0}, False, "insufficient")])
def test_improvement_is_claimed_only_past_the_preregistered_rule(totals, means, complete, decision):
    got, wording = T.judge(totals, means, complete)
    assert got == decision
    assert ("개선됐다" in wording[0]) == (decision == "improved")
    if decision == "not_improved":
        assert "개선을 주장하지 않는다" in wording[0]
    assert (T.FIXED["wording"]["syntax_zero"] in wording) == (decision != "insufficient" and not totals.get("invalid_syntax"))


def test_wording_has_no_forbidden_claim():
    texts = [T.FIXED["wording"]["common"], T.FIXED["wording"]["syntax_zero"], T.FIXED["wording"]["insufficient"],
             T.FIXED["wording"]["not_improved"].format(invalid=3, syntax=0, contract=3, p4d="0.60", p4c="0.90"),
             T.FIXED["scope"], T.FIXED["question"]]
    for text in texts:
        assert not B.claim_violations(text) and not B.path_violations(text) and "직접" not in text, text
