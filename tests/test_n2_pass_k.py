"""N2 반복 일관성(pass^k): 추정식, 실행 순서·API 상한·인프라 재시도, 결정 요약·사전등록 문구, 공개 기록으로 계산하는 NAT
반복·결정적 기준선. 모델 호출·GPU·비공개 packet 없음."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_replay_page as B  # noqa: E402
import n2_pass_k as N  # noqa: E402


# --- 추정식 ---
def test_pass_hat_k_is_the_chance_that_k_draws_are_all_valid_and_identical():
    keys = [["continue", "reward"]] * 3 + [["continue", "optimizer"], None]  # 유효 4(두 결정), 무효 1
    assert N.pass_hat_k(keys, 1) == pytest.approx(4 / 5)
    assert N.pass_hat_k(keys, 2) == pytest.approx(3 / 10)  # C(3,2)/C(5,2)
    assert N.pass_hat_k(keys, 4) == 0 and N.pass_hat_k(keys, 5) == 0
    assert N.pass_hat_k(keys, 6) is None and N.pass_hat_k([], 1) is None
    assert N.pass_hat_k_valid(keys, 4) == pytest.approx(1 / 5)  # C(4,4)/C(5,4)
    assert N.pass_hat_k([["a"]] * 5, 5) == 1


def test_decision_keys():
    assert N.decision_key("diagnosis", {"action": "continue", "ranking": ["reward", "actuator"], "probe": "P_reward"}) == (
        "continue", "reward")
    assert N.decision_key("correction", {"decision": "propose", "proposal": {
        "correction_id": "restore_body_frame_reward", "recovery_mode": "restart"}}) == (
        "propose", "restore_body_frame_reward", "restart")


# --- 사전등록 ---
def test_fixed_plan_stays_under_the_api_cap_and_repeats_over_time():
    assert N.FIXED["api_budget"]["planned_requests"] == len(N.schedule()) == 55 <= N.API_CAP == 80
    assert N.FIXED["api_budget"]["nat_new_requests"] == 0 and N.FIXED["paths"]["nat_v4"]["new_calls"] is False
    order = N.schedule()
    assert order[:11] == [(s, k, 1) for s in N.SET_ORDER for k in N.INPUT_SETS[s]["kinds"]]  # run 번호가 바깥 반복
    assert {r for _, _, r in order} == set(range(1, 6)) and N.RUNS == 5 and N.K_VALUES == (1, 2, 4, 5)
    assert json.loads(json.dumps(N.FIXED)) == N.FIXED


def test_preregistered_wording_has_no_forbidden_claim():
    texts = [v for v in N.FIXED["wording"].values()] + [N.FIXED["scope"], N.FIXED["question"]]
    for text in texts:
        filled = text.format(path="단일 호출", kind="진단", n=5)
        assert not B.claim_violations(filled) and not B.path_violations(filled), filled
        assert "직접" not in filled


def test_check_protocol_rejects_changes():
    dynamic = {"source_files": {"a": "1"}, "inputs": {"public": {}, "private": {}}}
    protocol = {**json.loads(json.dumps(N.FIXED)), **dynamic}
    N.check_protocol(protocol, dynamic)
    for broken, match in (({**protocol, "runs_per_packet": 8}, "fixed"), ({**protocol, "extra": 1}, "fields"),
                          ({**protocol, "source_files": {"a": "2"}}, "bound")):
        with pytest.raises(ValueError, match=match):
            N.check_protocol(broken, dynamic)


# --- 실행(가짜 호출) ---
@pytest.fixture
def private(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "PRIVATE", tmp_path / "n2")
    monkeypatch.setattr(N, "verify_protocol", lambda: ({}, "c" * 40))
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: tmp_path / "packets" / set_name / f"{kind}.packet.json")
    return tmp_path


def fake_runner(infra_first: set, calls: list):
    """argv의 --output 경로에 기록을 쓴다. infra_first의 (세트, 종류, run) 첫 시도는 인프라 오류."""
    def run(argv):
        shell = argv[-1]
        output = Path("C:/" + shell.split("--output ")[1].split(" ")[0][len("/mnt/c/"):])
        parts = output.parts
        set_name, kind, run_no, attempt = parts[-3], parts[-2], int(parts[-1][3:].split("_")[0]), int(parts[-1][-6])
        calls.append((set_name, kind, run_no, attempt))
        record = {"exit_code": 0, "request": {"model": N.MODEL, "temperature": 0}}
        if attempt == 1 and (set_name, kind, run_no) in infra_first:
            record.update(infra_error="APIConnectionError", exit_code=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record), encoding="utf-8")
        return record["exit_code"]
    return run


def test_execute_runs_the_schedule_retries_infra_once_and_resumes(private):
    calls = []
    out = N.execute(fake_runner({("r_s2030", "diagnosis", 1)}, calls))
    assert out["stopped"] is None and out["remaining"] == 0 and out["requests_used"] == 56 == len(calls)
    assert calls[:3] == [("h02_s2027", "diagnosis", 1, 1), ("f2_s2031", "diagnosis", 1, 1), ("f2_s2031", "correction", 1, 1)]
    assert ("r_s2030", "diagnosis", 1, 2) in calls  # 인프라 오류 첫 시도 뒤 같은 run 번호로 한 번 더
    assert N.final_record("r_s2030", "diagnosis", 1)[0].name == "run1_a2.json"
    again = []
    assert N.execute(fake_runner(set(), again))["calls_made"] == 0 and again == []  # 끝난 실행은 다시 부르지 않는다


def test_execute_resumes_a_run_whose_first_attempt_was_an_infra_error(private):
    first = N.run_path("h02_s2027", "diagnosis", 1, 1)  # 이전 실행이 인프라 오류 기록만 남기고 끝났다
    first.parent.mkdir(parents=True)
    first.write_text(json.dumps({"infra_error": "APIConnectionError", "exit_code": 2}), encoding="utf-8")
    calls = []
    N.execute(fake_runner(set(), calls))
    assert calls[0] == ("h02_s2027", "diagnosis", 1, 2) and ("h02_s2027", "diagnosis", 1, 1) not in calls


def test_execute_stops_at_the_api_cap(private, monkeypatch):
    monkeypatch.setattr(N, "API_CAP", 7)
    calls = []
    out = N.execute(fake_runner(set(), calls))
    assert out["stopped"] == "budget_exhausted" and len(calls) == 7 == out["requests_used"]
    assert out["remaining"] == 55 - 7


def test_final_record_waits_for_the_infra_retry(private):
    path = N.run_path("r_s2030", "correction", 2, 1)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"infra_error": "APITimeoutError"}), encoding="utf-8")
    assert N.final_record("r_s2030", "correction", 2) == (None, None)
    N.run_path("r_s2030", "correction", 2, 2).write_text(json.dumps({"infra_error": "APITimeoutError"}), encoding="utf-8")
    assert N.final_record("r_s2030", "correction", 2)[1] == {"infra_error": "APITimeoutError"}  # 두 번째도 인프라면 확정


def test_call_argv_sources_the_credentials_file_and_never_names_the_key(private):
    argv = N.call_argv("u1_s2040", "correction", 3, 1)
    assert argv[:6] == ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]
    shell = argv[-1]
    assert shell.startswith('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" ')
    assert "evals/correction_model_call.py --evidence " in shell and f"--model {N.MODEL}" in shell
    assert shell.split("--output ")[1].startswith("/mnt/") and "/n2/single/u1_s2040/correction/run3_a1.json" in shell
    assert "NVIDIA_API_KEY" not in shell and "cat " not in shell
    assert "evals/active_model_call.py --packet " in N.call_argv("u1_s2040", "diagnosis", 1, 1)[-1]


# --- 요약 ---
def _runs(keys: list, status=None) -> list[dict]:
    rows = []
    for i, key in enumerate(keys, 1):
        row = {"run": i, "status": "valid" if key else (status or "invalid"), "input_tokens": 3000, "output_tokens": 200,
               "wall_s": 3.0, "llm_calls": 1}
        if key:
            row.update(key=list(key), probe="P_reward")
        rows.append(row)
    return rows


def test_repeat_summary_counts_distribution_agreement_and_costs():
    runs = _runs([("continue", "reward")] * 4 + [None]) + [{"run": 6, "status": "infra_error"}]
    out = N.repeat_summary("diagnosis", runs, "reward", {"path": "single", "key": ["continue", "reward"]})
    assert out["n"] == 5 and out["counts"]["infra_error"] == 1 and out["decisions"] == {"continue · reward": 4}
    assert out["pass_k"]["1"] == pytest.approx(0.8) and out["pass_k"]["5"] == 0
    assert out["agree_with_recorded"] == 4 and out["top1_is_registered_mechanism"] == 4
    assert out["input_tokens"]["total"] == 15000 and out["tokens_per_valid_decision"] == pytest.approx(16000 / 4)


def test_outcome_wording_follows_the_preregistered_rule():
    stable = N.repeat_summary("correction", _runs([("propose", "x", "restart")] * 5), "reward", None)
    varied = N.repeat_summary("correction", _runs([("propose", "x", "restart")] * 4 + [("HOLD", None, None)]), "reward", None)
    short = N.repeat_summary("correction", _runs([("propose", "x", "restart")] * 3), "reward", None)
    out = N.outcome_wording("단일 호출", "교정", {"a": stable, "b": stable})
    assert (out["stability"], out["use"], out["mean_pass4"]) == ("stable", "usable", 1)
    out = N.outcome_wording("단일 호출", "교정", {"a": stable, "b": varied})
    assert out["stability"] == "varied" and out["mean_pass4"] == pytest.approx((1 + 0.2) / 2) and out["use"] == "reconsider"
    assert "흔들렸다" in out["wording"][0] and "재검토" in out["wording"][1]
    assert N.outcome_wording("단일 호출", "교정", {"a": short})["use"] == "insufficient"


# --- 공개 기록으로 계산하는 부분 ---
def test_nat_repeats_are_read_from_the_public_e4b_v4_summary():
    rows = N.nat_runs("f2_s2031", "diagnosis")
    out = N.repeat_summary("diagnosis", rows, "actuator", None)
    assert out["decisions"] == {"continue · optimizer": 1, "continue · reward": 3, "continue · termination": 1}
    assert out["pass_k"]["1"] == 1 and out["pass_k"]["4"] == 0 and out["llm_calls"]["median"] == 8
    correction = N.repeat_summary("correction", N.nat_runs("f2_s2031", "correction"), "actuator", None)
    assert correction["pass_k"]["5"] == 1
    assert N.nat_runs("u1_s2040", "diagnosis") is None


def test_deterministic_diagnosis_baselines_on_public_records():
    out = N.diagnosis_baselines("h02_s2027", N.dev_training())
    assert all(out[name]["same_over_runs"] for name in ("rule_prior", "rule_features", "rule_template"))
    assert out["rule_prior"]["top1"] == "reward" and out["rule_template"]["top1"] == "reward"
    assert out["exhaustive_probes"] == {"status": "confirmed", "top1": "reward", "probes_used": 6, "gpu_s_saved_probes": 66.8,
                                        "top1_is_registered_mechanism": True}
    assert N.diagnosis_baselines("u1_s2040", N.dev_training())["exhaustive_probes"] is None  # 저장된 전수 probe가 없다


def test_correction_rule_is_recomputed_and_must_match_the_packet(tmp_path, monkeypatch):
    packet = tmp_path / "correction.packet.json"
    monkeypatch.setattr(N, "packet_path", lambda set_name, kind: packet)
    packet.write_text(json.dumps({"rule_result": {"correction_id": "restore_body_frame_reward", "mode": "restart"}}),
                      encoding="utf-8")
    out = N.correction_baseline("u1_s2040")  # 공개 결함 고정 평가: standstill 1.040 → restart
    assert out["key"] == ["propose", "restore_body_frame_reward", "restart"] and out["standstill_ratio"] > 0.9
    packet.write_text(json.dumps({"rule_result": {"correction_id": "restore_body_frame_reward", "mode": "resume"}}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="rule_result"):
        N.correction_baseline("u1_s2040")
    packet.write_text(json.dumps({"rule_result": {"correction_id": "restore_motor_torque_speed", "mode": "restart"}}),
                      encoding="utf-8")
    assert N.correction_baseline("f2_s2033")["key"][2] == "restart"  # 모터는 standstill과 무관하게 restart(0.891 < 0.9)
