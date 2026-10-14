"""N2 검증 피드백 1회 재요청 arm: 검증기 거절 메시지가 모두 등록 코드로 바뀌는지, 피드백에 packet별 값·정답이 없는지,
대화 갈래(1차 유효·재요청 회복·재요청 무효·인프라), 기록 재검증과 변조 탐지, 실행 예산·재시도·재개, 사전등록 개선 판정.
모델 호출·GPU·비공개 packet 없음(진단 packet은 공개 기록에서 만든다)."""
from __future__ import annotations

import inspect
import json
from math import comb
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_replay_page as B  # noqa: E402
import correction_model_call as CM  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import e4b_compare_v4 as E4  # noqa: E402
import feedback_model_call as F  # noqa: E402
import n2_feedback as T  # noqa: E402
import n2_pass_k as N  # noqa: E402
from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_registry import REGISTRY  # noqa: E402

LITERAL = re.compile(r'(?:raise ValueError|reasons\.append|_rejected)\(\[?f?"([^"{]*)')


def _validator_prefixes(*functions) -> list[str]:
    return [m.group(1) for fn in functions for m in LITERAL.finditer(inspect.getsource(fn)) if m.group(1)]


# --- 문제 코드·피드백 정책 ---
def test_every_validator_message_maps_to_a_registered_code():
    diagnosis = _validator_prefixes(D.validate_decision)
    correction = _validator_prefixes(CM.validate_model_proposal) + ["JSON 파손: x", "JSON 객체가 아니다"]
    assert len(diagnosis) == 19 and len(correction) >= 12
    assert "D_OTHER" not in F.codes_for("diagnosis", diagnosis)
    assert "C_OTHER" not in F.codes_for("correction", correction)
    assert set(F.codes_for("diagnosis", diagnosis)) == {c for _, c in F.DIAGNOSIS_CODES}
    assert set(F.codes_for("correction", correction)) == {c for _, c in F.CORRECTION_CODES}
    assert F.codes_for("diagnosis", ["처음 보는 거절"]) == ["D_OTHER"]
    assert F.codes_for("correction", ["체크리스트는 1~6", "체크리스트 3의 근거"]) == ["C_CHECKLIST_KEYS", "C_CHECKLIST_VALUE"]


def test_feedback_texts_carry_no_packet_values_or_answers():
    banned = (set(L.MECHANISMS) | set(L.PROBES) | {e["correction_id"] for e in REGISTRY.values()}
              | {m for e in REGISTRY.values() for m in e["recovery_modes"]})
    for kind, texts in (("diagnosis", F.DIAGNOSIS_TEXT), ("correction", F.CORRECTION_TEXT)):
        assert set(texts) == set(T.FIXED["retry"]["codes"][kind])
        full = F.feedback_text(kind, list(texts))
        assert not {w for w in banned if re.search(rf"\b{re.escape(w)}\b", full)}, kind
        assert "직접" not in full


# --- 대화 갈래 ---
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


def fake_complete(*results):
    sent = []

    def complete(messages, params):
        sent.append((messages, params))
        out = results[len(sent) - 1]
        return {"finish_reason": "stop", "input_tokens": 2800, "output_tokens": 150, "wall_s": 2.0,
                **({"content": out} if isinstance(out, str) else out)}
    return complete, sent


def test_valid_first_answer_is_final_without_retry(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    complete, sent = fake_complete(json.dumps(_decision(packet)))
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["requests"] == 1 and record["feedback"] is None and len(sent) == 1
    assert record["final"] == {"status": "valid", "step": "first", "codes": []}
    out = T.review("diagnosis", record, path)
    assert out["status"] == "valid" and out["key"] == ["continue", "reward"] and not out["retried"]


def test_rejected_answer_gets_one_fixed_feedback_retry(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    fact_id, value = next(iter(D._known_facts(packet).items()))
    bad = json.dumps(_decision(packet, claims=[{"id": fact_id, "value": "wrong"}]))
    complete, sent = fake_complete(bad, json.dumps(_decision(packet)))
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["requests"] == 2 and record["calls"][0]["codes"] == ["D_CLAIM_VALUE"]
    (first_messages, first_params), (retry, retry_params) = sent
    assert retry_params == first_params and retry[:-2] == first_messages
    assert retry[-2] == {"role": "assistant", "content": bad}
    assert retry[-1] == {"role": "user", "content": F.feedback_text("diagnosis", ["D_CLAIM_VALUE"])}
    assert fact_id not in retry[-1]["content"] and str(value) not in retry[-1]["content"]  # 정답 값을 알려주지 않는다
    assert record["final"] == {"status": "valid", "step": "retry", "codes": []}
    out = T.review("diagnosis", record, path)
    assert (out["first_status"], out["status"], out["retried"]) == ("invalid", "valid", True)
    assert out["raw_first_key"] == ["continue", "reward"]


def test_still_invalid_after_retry_is_invalid(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    complete, _ = fake_complete('{"ranking": [', json.dumps(_decision(packet, claims=[])))
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["calls"][0]["codes"] == ["D_JSON"] and record["final"] == {
        "status": "invalid", "step": "retry", "codes": ["D_CLAIMS_EMPTY"]}
    out = T.review("diagnosis", record, path)
    assert out["status"] == "invalid" and out["final_codes"] == ["D_CLAIMS_EMPTY"] and out["raw_first_key"] is None


@pytest.mark.parametrize("second", [{"infra_error": "APITimeoutError"}, {"content": ""},
                                    {"content": "{", "finish_reason": "length"}])
def test_infra_error_on_either_call_makes_the_run_retryable(tmp_path, second):
    path = _packet(tmp_path)
    complete, _ = fake_complete('{"ranking": [', second)
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["infra_error"] and record["retryable"] and record["requests"] == 2 and record["final"] is None
    assert T.review("diagnosis", record, path) == {"status": "infra_error"}


def test_non_retryable_api_error_is_a_call_error(tmp_path):
    path = _packet(tmp_path)
    complete, _ = fake_complete({"error_type": "BadRequestError", "status_code": 400})
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["error_type"] == "BadRequestError" and record["requests"] == 1 and not record.get("infra_error")
    assert T.review("diagnosis", record, path) == {"status": "call_error"}


def test_non_retryable_api_error_on_the_retry_is_a_call_error(tmp_path):
    path = _packet(tmp_path)
    complete, _ = fake_complete('{"ranking": [', {"error_type": "BadRequestError", "status_code": 400})
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    assert record["error_type"] == "BadRequestError" and record["requests"] == 2 and record["final"] is None
    assert record.get("infra_error") is None and [c["status"] for c in record["calls"]] == ["invalid", "call_error"]
    assert T.review("diagnosis", record, path) == {"status": "call_error"}


def test_raw_key_reads_the_body_the_way_the_validator_does(tmp_path):
    packet = json.loads(_packet(tmp_path).read_text(encoding="utf-8"))
    body = json.dumps(_decision(packet, claims=[]))
    assert T.raw_key("diagnosis", f"```json\n{body}\n```") == ["continue", "reward"]
    assert T.raw_key("diagnosis", f" ```json\n{body}\n```") is None  # 검증기도 울타리를 벗기지 않는 응답
    assert T.raw_key("diagnosis", "[1, 2]") is None and T.raw_key("correction", '{"checklist": [') is None


def _correction_packet(tmp_path) -> Path:
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
    return path


def test_correction_retry_uses_the_registry_passed_in_the_packet(tmp_path):
    path = _correction_packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    fact = min(packet["facts"])
    good = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": [fact],
            "checklist": {k: fact for k in packet["checklist"]}, "hold_reason": None}
    bad = {**good, "cited_facts": ["registry"]}
    complete, sent = fake_complete(json.dumps(bad), json.dumps(good))
    record = F.dialogue("correction", path.read_bytes(), N.MODEL, complete)
    assert record["calls"][0]["codes"] == ["C_CITED_UNKNOWN"] and record["final"]["status"] == "valid"
    assert "restore_body_frame_reward" not in sent[1][0][-1]["content"]
    out = T.review("correction", record, path)
    assert out["key"] == ["propose", "restore_body_frame_reward", "restart"]
    assert out["raw_first_key"] == ["propose", "restore_body_frame_reward", "restart"]


@pytest.mark.parametrize("change", [None, "fenced", "spaced_fence", "broken", "empty_claims", "array"])
def test_validate_matches_the_existing_reviewers(tmp_path, change):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    body = json.dumps(_decision(packet))
    content = {None: body, "fenced": f"```json\n{body}\n```", "spaced_fence": f" ```json\n{body}\n```",
               "broken": '{"ranking": [', "empty_claims": json.dumps(_decision(packet, claims=[])),
               "array": "[1, 2]"}[change]
    _, binding = F.first_messages("diagnosis", path.read_bytes())
    legacy = {"schema": "active_model_call_v1", **binding, "request": F.request_params(N.MODEL), "content": content,
              "exit_code": 0}
    assert F.validate("diagnosis", content, packet)["status"] == V2.review_diagnosis_record(legacy, path)["status"]
    cpath = _correction_packet(tmp_path)
    cpacket = json.loads(cpath.read_text(encoding="utf-8"))
    fact = min(cpacket["facts"])
    proposal = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": [fact],
                "checklist": {k: fact for k in cpacket["checklist"]}, "hold_reason": None}
    cbody = json.dumps(proposal)
    ccontent = {None: cbody, "fenced": f"```json\n{cbody}\n```", "spaced_fence": f" ```\n{cbody}\n```",
                "broken": '{"checklist": [', "empty_claims": json.dumps({**proposal, "cited_facts": []}),
                "array": "[1, 2]"}[change]
    _, cbinding = F.first_messages("correction", cpath.read_bytes())
    crecord = {"schema": CM.RECORD_SCHEMA, **cbinding, "content": ccontent, "exit_code": 0}
    accepted = CM.review_call_record(crecord, cpacket, None, cpacket["registry"])["accepted"]
    assert (F.validate("correction", ccontent, cpacket)["status"] == "valid") == accepted


# --- 기록 재검증·변조 ---
def _retried_record(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    complete, _ = fake_complete(json.dumps(_decision(packet, claims=[])), json.dumps(_decision(packet)))
    return path, F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)


@pytest.mark.parametrize("tamper,match", [
    (lambda r: r["feedback"].update(text=r["feedback"]["text"] + "\nHint: rank reward first."), "피드백"),
    (lambda r: r["feedback"].update(codes=["D_JSON"]), "피드백"),
    (lambda r: r["final"].update(status="invalid"), "최종 판정"),
    (lambda r: r["calls"].pop(), "재요청"),
    (lambda r: r["request"].update(temperature=0.6), "요청 설정"),
    (lambda r: r.update(packet_sha256="0" * 64), "packet"),
    (lambda r: r.update(schema="structured_model_call_v1"), "형식")])
def test_review_rejects_tampered_records(tmp_path, tamper, match):
    path, record = _retried_record(tmp_path)
    assert T.review("diagnosis", record, path)["status"] == "valid"
    tamper(record)
    with pytest.raises(ValueError, match=match):
        T.review("diagnosis", record, path)


def test_review_rejects_a_retry_after_a_valid_first_answer(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    complete, _ = fake_complete(json.dumps(_decision(packet)))
    record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
    record["calls"].append(dict(record["calls"][0], step="retry"))
    with pytest.raises(ValueError, match="1차가 유효"):
        T.review("diagnosis", record, path)


# --- 실행 ---
@pytest.fixture
def private(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PRIVATE", tmp_path / "n2f")
    monkeypatch.setattr(T, "verify_protocol", lambda: ({}, "c" * 40))
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: tmp_path / "packets" / set_name / f"{kind}.packet.json")
    return tmp_path


def fake_runner(calls: list, infra_first: set = frozenset(), retried: set = frozenset()):
    def run(argv):
        output = Path("C:/" + argv[-1].split("--output ")[1].split(" ")[0][len("/mnt/c/"):])
        set_name, kind, name = output.parts[-3], output.parts[-2], output.parts[-1]
        run_no, attempt = int(name[3:].split("_")[0]), int(name[-6])
        calls.append((set_name, kind, run_no, attempt))
        record = {"exit_code": 0, "requests": 2 if (set_name, kind, run_no) in retried else 1}
        if attempt == 1 and (set_name, kind, run_no) in infra_first:
            record.update(infra_error="APIConnectionError", exit_code=2, requests=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record), encoding="utf-8")
        return record["exit_code"]
    return run


def test_execute_counts_retry_requests_and_reruns_infra_runs_once(private):
    calls = []
    retried = {(s, k, r) for s, k, r in N.schedule() if r == 1}
    out = T.execute(fake_runner(calls, {("f2_s2033", "correction", 2)}, retried))
    assert out["stopped"] is None and out["remaining"] == 0
    assert out["requests_used"] == 55 + len(retried) + 2  # 재요청 11, 인프라 시도의 2회
    assert [c[:3] for c in calls if c[3] == 1] == N.schedule() and ("f2_s2033", "correction", 2, 2) in calls


def test_execute_stops_when_one_more_run_could_pass_the_cap(private, monkeypatch):
    monkeypatch.setattr(T, "API_CAP", 7)
    calls = []
    out = T.execute(fake_runner(calls, retried={(s, k, r) for s, k, r in N.schedule()}))
    assert out["stopped"] == "budget_exhausted" and len(calls) == 3 and out["requests_used"] == 6
    assert out["remaining"] == 52


def test_execute_resumes_a_run_whose_first_attempt_was_an_infra_error(private):
    first = T.run_path("h02_s2027", "diagnosis", 1, 1)
    first.parent.mkdir(parents=True)
    first.write_text(json.dumps({"infra_error": "APIConnectionError", "exit_code": 2, "requests": 1}), encoding="utf-8")
    calls = []
    T.execute(fake_runner(calls))
    assert calls[0] == ("h02_s2027", "diagnosis", 1, 2) and ("h02_s2027", "diagnosis", 1, 1) not in calls


def test_feedback_runs_mark_retried_key_change_as_not_comparable(private, tmp_path, monkeypatch):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: path)
    first_valid = json.dumps(_decision(packet))
    other = json.dumps(_decision(packet, claims=[], ranking=["actuator"] + [m for m in L.MECHANISMS if m != "actuator"],
                                 expected_outcomes={}))
    plans = {1: (first_valid,), 2: (json.dumps(_decision(packet, claims=[])), first_valid),
             3: (other, first_valid), 4: ('{"ranking": [', first_valid), 5: (other, '{"ranking": [')}
    for run, answers in plans.items():
        complete, _ = fake_complete(*answers)
        record = F.dialogue("diagnosis", path.read_bytes(), N.MODEL, complete)
        target = T.run_path("h02_s2027", "diagnosis", run, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record), encoding="utf-8")
    runs = T.feedback_runs("h02_s2027", "diagnosis")
    assert [r["status"] for r in runs] == ["valid", "valid", "valid", "valid", "invalid"]
    assert "key_change" not in runs[0]
    assert [r["key_change"]["raw_first_key_same"] for r in runs[1:]] == [True, False, None, None]
    assert all(r["key_change"]["comparable"] is False for r in runs[1:])
    assert runs[1]["retry_tokens"] == 2950 and runs[1]["llm_calls"] == 2
    summary = N.repeat_summary("diagnosis", runs, "reward", None)
    assert summary["pass_k"]["5"] == 0 and summary["counts"]["invalid"] == 1


def test_summarize_reports_first_retry_and_final_separately(private, tmp_path, monkeypatch):
    dpath, cpath = _packet(tmp_path), _correction_packet(tmp_path)
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: dpath if kind == "diagnosis" else cpath)
    protocol = {"inputs": {"private": {f"{s}/{k}": {"packet_sha256": N.sha_raw(N.packet_path(s, k))}
                                       for s in N.SET_ORDER for k in N.INPUT_SETS[s]["kinds"]}}}
    monkeypatch.setattr(T, "verify_protocol", lambda: (protocol, "c" * 40))
    monkeypatch.setattr(T, "PROTOCOL", tmp_path / "protocol.json")
    T.PROTOCOL.write_text("{}", encoding="utf-8")
    dpacket, cpacket = (json.loads(p.read_text(encoding="utf-8")) for p in (dpath, cpath))
    fact = min(cpacket["facts"])
    proposal = json.dumps({"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": [fact],
                           "checklist": {k: fact for k in cpacket["checklist"]}, "hold_reason": None})
    for set_name, kind, run in N.schedule():
        if kind == "diagnosis":
            answers = ((json.dumps(_decision(dpacket, claims=[])),) if run == 1 else ()) + (json.dumps(_decision(dpacket)),)
            data = dpath.read_bytes()
        else:
            answers, data = (proposal,), cpath.read_bytes()
        complete, _ = fake_complete(*answers)
        target = T.run_path(set_name, kind, run, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(F.dialogue(kind, data, N.MODEL, complete)), encoding="utf-8")
    out = T.summarize(write=False)
    assert out["complete"] and out["totals"] == {"first_valid": 49, "recovered_by_retry": 6, "retried": 6, "valid": 55}
    assert out["mean_pass4"] == {"diagnosis": 1.0, "correction": 1.0} and out["improvement"]["decision"] == "improved"
    assert out["budget"]["requests_used"] == 61
    h02 = out["sets"]["h02_s2027"]["kinds"]["diagnosis"]["feedback"]
    assert h02["first_only_pass_k"]["4"] == comb(4, 4) / comb(5, 4) and h02["pass_k"]["4"] == 1.0
    assert h02["key_change"] == {"not_comparable": 1, "raw_first_key_same": 1, "raw_first_key_differs": 0}
    assert h02["first_codes"] == {"D_CLAIMS_EMPTY": 1} and h02["final_codes"] == {}
    assert h02["retry_tokens_total"] == 2950 and h02["first_tokens_total"] == 5 * 2950
    n2 = N.read(ROOT / T.N2_SUMMARY)["sets"]["h02_s2027"]["kinds"]["diagnosis"]["single"]["decisions"]
    assert h02["same_as_n2_decision"] == (None if len(n2) != 1 else 5 * (next(iter(n2)) == "continue · reward"))


def test_call_argv_uses_the_feedback_caller_and_sources_credentials(private):
    shell = T.call_argv("u1_s2040", "correction", 2, 1)[-1]
    assert shell.startswith('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" ')
    assert "evals/feedback_model_call.py --kind correction --packet " in shell and f"--model {N.MODEL}" in shell
    assert "/n2f/runs/u1_s2040/correction/run2_a1.json" in shell and "NVIDIA_API_KEY" not in shell


# --- 사전등록 ---
def test_fixed_protocol_and_budget():
    assert json.loads(json.dumps(T.FIXED)) == T.FIXED
    assert T.FIXED["api_budget"]["planned_first_requests"] == T.FIXED["api_budget"]["max_retry_requests"] == 55
    assert 110 <= T.API_CAP == 120 and T.FIXED["retry"]["texts"] == {"diagnosis": F.DIAGNOSIS_TEXT,
                                                                       "correction": F.CORRECTION_TEXT}
    assert F.request_params(N.MODEL) == {"model": N.MODEL, "temperature": 0, "max_tokens": 16384, "stream": False,
                                         "extra_body": {"reasoning_effort": "none"}}
    assert T.N2_SUMMARY in T.public_inputs() and {"evals/feedback_model_call.py", "evals/n2_feedback.py"} <= T.CODE


@pytest.mark.parametrize("totals,means,complete,decision", [
    ({"invalid": 1, "first_valid": 49, "retried": 6}, {"diagnosis": 0.8, "correction": 1.0}, True, "improved"),
    ({"invalid": 2, "first_valid": 49, "retried": 6}, {"diagnosis": 0.9, "correction": 1.0}, True, "not_improved"),
    ({"invalid": 0}, {"diagnosis": 0.79, "correction": 1.0}, True, "not_improved"),
    ({"invalid": 0}, {"diagnosis": None, "correction": 1.0}, True, "insufficient"),
    ({"invalid": 0}, {"diagnosis": 1.0, "correction": 1.0}, False, "insufficient")])
def test_improvement_is_claimed_only_past_the_preregistered_rule(totals, means, complete, decision):
    got, wording = T.judge(totals, means, complete)
    assert got == decision and len(wording) == 1
    assert ("개선됐다" in wording[0]) == (decision == "improved")
    if decision == "improved":
        assert "이미 본 packet의 개발 측정" in wording[0] and "시연 결정 단계에 넣으려면 새 사전등록이 필요" in wording[0]
    if decision == "not_improved":
        assert "개선을 주장하지 않는다" in wording[0] and "이미 본 packet의 개발 측정" in wording[0]


def test_wording_has_no_forbidden_claim():
    fill = {"first_valid": 49, "retries": 6, "invalid": 1, "p4d": "0.80", "p4c": "1.00"}
    texts = [T.FIXED["wording"]["common"], T.FIXED["wording"]["insufficient"],
             T.FIXED["wording"]["improved"].format(**fill), T.FIXED["wording"]["not_improved"].format(**fill),
             T.FIXED["scope"], T.FIXED["question"], T.FIXED["retry"]["feedback_rule"]]
    for text in texts:
        assert not B.claim_violations(text) and not B.path_violations(text) and "직접" not in text, text
