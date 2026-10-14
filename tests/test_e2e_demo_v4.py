"""E4 종단 시연 v4: 두 모델 결정 단계에만 검증 피드백 1회 재요청을 넣었는지(첫 판단 유효·회복·재요청 뒤 무효·인프라, 교정
제안 유효·회복·재요청 뒤 무효의 규칙 요청), 기록 재검증과 변조 거부, model_feedback 기록, 사전등록(새 seed·N2 피드백 arm
결합·v3.1과 같은 규칙), 교정 원장·교정 실행기 결합, 하네스 실패. GPU·모델 없음(호출은 가짜 응답)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import correct as CR  # noqa: E402
import correction_model_call as CM  # noqa: E402
import demo_correction_v4 as DC  # noqa: E402
import e2e_demo_v3_1 as E31  # noqa: E402
import e2e_demo_v4 as E  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import e4b_compare_v4 as E4B  # noqa: E402
import feedback_model_call as F  # noqa: E402
import loop  # noqa: E402
from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_ledger import LedgerError  # noqa: E402
from test_correction_step import make_loop  # noqa: E402

EXPERIMENT = "a" * 40
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
PATH = "single_feedback"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """시연 출력·결과·고리 경로와 ROOT를 임시 폴더로 돌린다. 출력 위치는 저장소와 같은 상대 경로다."""
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "bench/private/e2e_demo_v4")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "PROTOCOL_DIR", tmp_path / "protocols")
    (tmp_path / "protocols").mkdir(exist_ok=True)
    (tmp_path / "protocols/e2e_demo_v4_f2.json").write_text(json.dumps(E.fixed("f2", PATH)), encoding="utf-8")
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(E, "verify_for", lambda t, isaaclab: ({"decision_path": PATH}, EXPERIMENT))
    return E


def fake_complete(*answers):
    """feedback_model_call.dialogue에 넘길 가짜 호출. 문자열은 응답 본문, dict는 호출 결과 그대로다."""
    sent = []

    def complete(messages, params):
        sent.append(messages)
        out = answers[len(sent) - 1]
        return {"finish_reason": "stop", "input_tokens": 1300, "output_tokens": 150, "wall_s": 1.5, "response_id": "r",
                **({"content": out} if isinstance(out, str) else out)}
    return complete


def write_record(path: Path, kind: str, packet_path: Path, *answers) -> Path:
    """evals/feedback_model_call.call과 같은 모양의 기록을 쓴다(실행 시간·종료 코드 포함)."""
    record = F.dialogue(kind, packet_path.read_bytes(), E.MODEL, fake_complete(*answers))
    record.update(execution_wall_s=3.2, exit_code=2 if record.get("infra_error") or record.get("error_type") else 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# --- 사전등록 ---
def test_protocol_changes_only_the_decision_calls_and_binds_the_feedback_arm():
    fixed = E.fixed("f2", PATH)
    assert json.loads(json.dumps(fixed)) == fixed and fixed["schema"] == "e2e_demo_protocol_v4"
    assert E.VARIANTS == {"f2": {"fault": "F2", "case_id": "h06", "mechanism": "actuator", "seed": 2046}}
    used = set(range(2026, 2038)) | {2040, 2042, 3040, 7, 42, 123} | set(range(3029, 3035))
    assert {E.ctx_of("f2").seed, E.ctx_of("f2").restart_seed} == {2046, 3046} and not {2046, 3046} & used
    assert E.ctx_of("f2").protocol.name == "e2e_demo_v4_f2.json" and E.PRIVATE.name == "e2e_demo_v4"
    assert E.RECOVERY_RULES == E31.RECOVERY_RULES and E.SPECIFICITY_GATES == E31.SPECIFICITY_GATES
    assert E.registry_v3() == E31.registry_v3() and set(E.OUTCOMES) == set(E31.OUTCOMES)
    assert set(E.SUCCESS_CONDITIONS) == set(E31.SUCCESS_CONDITIONS) and set(E.PREDICTIONS["f2"]) == set(E31.PREDICTIONS["f2"])
    assert fixed["feedback_policy"]["texts"] == {"diagnosis": F.DIAGNOSIS_TEXT, "correction": F.CORRECTION_TEXT}
    assert set(fixed["feedback_policy"]["codes"]["diagnosis"]) == set(F.DIAGNOSIS_TEXT)
    assert set(fixed["feedback_policy"]["codes"]["correction"]) == set(F.CORRECTION_TEXT)
    assert {"diagnosis", "correction", "infra"} == set(fixed["invalid_after_retry"])
    assert "예측하지 않는다" in fixed["model_feedback_record"]["prediction"]
    assert {"evals/e2e_demo_v4.py", "bench/demo_correction_v4.py", "evals/feedback_model_call.py"} <= E.CODE
    assert not {"evals/e2e_demo_v3_1.py", "bench/demo_correction_v3_1.py", "evals/nat_decision_call_v4.py"} & E.CODE
    assert all((ROOT / name).is_file() for name in E.CODE)
    with pytest.raises(ValueError):
        E.fixed("f2", "single")
    with pytest.raises(ValueError):
        E.ctx_of("f1")


def test_feedback_adoption_requires_the_improved_n2_summary(tmp_path, monkeypatch):
    adoption = E.feedback_adoption()
    assert adoption["decision"] == "improved" and adoption["summary"] == "evals/results/n2_feedback_v1/summary.json"
    assert adoption["final_invalid"] == 1
    summary = json.loads(E.FEEDBACK_SUMMARY.read_text(encoding="utf-8"))
    for change in ({"improvement": {"decision": "not_improved"}}, {"complete": False}):
        fake = tmp_path / "evals/results/n2_feedback_v1/summary.json"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text(json.dumps({**summary, **change}), encoding="utf-8")
        monkeypatch.setattr(E, "FEEDBACK_SUMMARY", fake)
        monkeypatch.setattr(E, "ROOT", tmp_path)
        with pytest.raises(ValueError, match="improved"):
            E.feedback_adoption()


# --- 첫 판단 ---
def ready_for_diagnosis(demo, monkeypatch, tmp_path, tag):
    """packet 단계까지의 기록(진단 packet은 공개 기록에서 만든다)과 고리 시작 대역."""
    ctx = demo.ctx_of("f2")
    demo._write_step(tag, "begin", demo._guard(tag, "begin"), EXPERIMENT, ctx.protocol,
                     {"variant": "f2", "decision_path": PATH, "seed_unused": {"hits": []}})
    packet_path = demo.folder_for(tag) / "model" / "diagnosis.packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(E4B.public_packet("h02_s2027"), ensure_ascii=False), encoding="utf-8")
    reference = tmp_path / "evals/results" / f"{tag}_ref" / "probes" / "none.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"probes": {p: {"measurement": {"x": 1}} for p in L.PROBES}}), encoding="utf-8")
    records = {"evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.13, "FAULT": 0.88}}},
               "reference": {"report": {"path": demo._rel(reference)}},
               "packet": {"packet": {"path": demo._rel(packet_path), "sha256": demo.sha(packet_path)}}}
    for name in demo.STEPS[1:demo.STEPS.index("loop_start")]:
        demo._write_step(tag, name, demo._guard(tag, name), EXPERIMENT, ctx.protocol, records.get(name, {}))
    monkeypatch.setattr(demo, "_check_artifacts", lambda t, arm: {"checkpoint": {"sha256": "c" * 64}})
    started = []
    monkeypatch.setattr(V2, "start_loop", lambda case, ranking, sha, ref, probe: started.append(probe) or {
        "request_id": "q1", "model_probe_used": True, "probe": probe})
    return packet_path, started


def decision(packet: dict, **change) -> dict:
    fact_id, value = next(iter(D._known_facts(packet).items()))
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    body = {"ranking": ranking, "probe": "P_reward", "action": "continue", "reason": "r",
            "claims": [{"id": fact_id, "value": value}],
            "expected_outcomes": {m: L.PROBES["P_reward"]["expect"][m] for m in ranking[:3]}}
    return {**body, **change}


@pytest.mark.parametrize("answers,retried", [((None,), False), (("empty", None), True)])
def test_loop_starts_from_the_final_valid_decision(demo, monkeypatch, tmp_path, answers, retried):
    packet_path, started = ready_for_diagnosis(demo, monkeypatch, tmp_path, "t1")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    texts = [json.dumps(decision(packet, claims=[]) if a == "empty" else decision(packet)) for a in answers]
    record = write_record(demo.folder_for("t1") / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path, *texts)
    step = demo.loop_start("t1", ROOT, record)
    assert step["started"]["request_id"] == "q1" and started == ["P_reward"]
    assert step["feedback"]["retried"] is retried and step["feedback"]["final_status"] == "valid"
    assert step["feedback"]["first_status"] == ("invalid" if retried else "valid")
    assert step["call"]["requests"] == 1 + retried and step["call"]["input_tokens"] == 1300 * (1 + retried)
    view = demo.model_feedback({"loop_start": step}, None, {})["diagnosis"]
    assert view == {"first_valid": not retried, "retries": int(retried), "final_valid": True,
                    "first_codes": ["D_CLAIMS_EMPTY"] if retried else [], "final_codes": [], "requests": 1 + retried,
                    "retry_input_tokens": 1300 if retried else None, "retry_output_tokens": 150 if retried else None,
                    "retry_wall_s": 1.5 if retried else None}


def test_still_invalid_after_the_retry_ends_as_model_invalid_without_a_rule_probe(demo, monkeypatch, tmp_path):
    packet_path, started = ready_for_diagnosis(demo, monkeypatch, tmp_path, "t2")
    record = write_record(demo.folder_for("t2") / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path,
                          '{"ranking": [', '{"ranking": [')
    terminal = demo.loop_start("t2", ROOT, record)
    assert terminal["outcome"] == "model_invalid" and started == [] and not demo.step_path("t2", "loop_start").exists()
    feedback = terminal["detail"]["attempt"]["feedback"]
    assert (feedback["first_codes"], feedback["final_codes"], feedback["retried"]) == (["D_JSON"], ["D_JSON"], True)
    result = demo.summarize("t2")
    assert result["outcome"] == "model_invalid" and result["schema"] == "e2e_demo_result_v4"
    assert result["model_feedback"]["diagnosis"]["final_valid"] is False and result["model_feedback"]["correction"] is None
    assert result["model_feedback"]["diagnosis"]["retries"] == 1


def test_infra_error_on_the_retry_reruns_the_whole_dialogue_at_most_twice(demo, monkeypatch, tmp_path):
    packet_path, _ = ready_for_diagnosis(demo, monkeypatch, tmp_path, "t3")
    model = demo.folder_for("t3") / "model"
    first = write_record(model / "diagnosis.feedback1.json", "diagnosis", packet_path, '{"ranking": [',
                         {"infra_error": "APITimeoutError"})
    with pytest.raises(ValueError, match="feedback1.json이어야"):  # 순서대로 다음 기록만 받는다
        demo.loop_start("t3", ROOT, model / "diagnosis.feedback2.json")
    assert demo.loop_start("t3", ROOT, first) == {"status": "retry_allowed", "attempt": 1}
    second = write_record(model / "diagnosis.feedback2.json", "diagnosis", packet_path, {"infra_error": "APIConnectionError"})
    assert demo.loop_start("t3", ROOT, second)["outcome"] == "model_infra_failed"


def test_non_retryable_api_error_ends_as_model_infra_failed(demo, monkeypatch, tmp_path):
    packet_path, _ = ready_for_diagnosis(demo, monkeypatch, tmp_path, "t4")
    record = write_record(demo.folder_for("t4") / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path,
                          {"error_type": "BadRequestError", "status_code": 400})
    assert demo.loop_start("t4", ROOT, record)["outcome"] == "model_infra_failed"


@pytest.mark.parametrize("tamper,match", [
    (lambda r: r["feedback"].update(text=r["feedback"]["text"] + "\nHint: rank actuator first."), "피드백"),
    (lambda r: r["final"].update(status="invalid"), "최종 판정"),
    (lambda r: r["request"].update(temperature=0.6), "요청 설정"),
    (lambda r: r.update(packet_sha256="0" * 64), "packet"),
    (lambda r: r.update(schema="active_model_call_v1"), "형식"),
    (lambda r: r["calls"].pop(), "재요청")])
def test_tampered_diagnosis_records_are_refused_before_any_step(demo, monkeypatch, tmp_path, tamper, match):
    packet_path, started = ready_for_diagnosis(demo, monkeypatch, tmp_path, "t5")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    path = write_record(demo.folder_for("t5") / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path,
                        json.dumps(decision(packet, claims=[])), json.dumps(decision(packet)))
    record = json.loads(path.read_text(encoding="utf-8"))
    tamper(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        demo.loop_start("t5", ROOT, path)
    assert started == [] and not demo.step_path("t5", "loop_start").exists()


# --- 교정 제안 ---
CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}
GOOD = {"correction_id": "restore_motor_torque_speed", "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
        "checklist": dict(CHECKLIST), "hold_reason": None}


def ready_for_correction(demo, monkeypatch, tmp_path, tag, *answers):
    """확정 고리·특이성 통과까지의 단계 기록, 결함 평가 report, 교정 packet, 교정 피드백 호출 기록을 만든다."""
    ctx = demo.ctx_of("f2")
    demo._write_step(tag, "begin", demo._guard(tag, "begin"), EXPERIMENT, ctx.protocol,
                     {"variant": "f2", "decision_path": PATH, "seed_unused": {"hits": []}})
    report = tmp_path / "evals/results" / tag / "fault.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P0C_RUNS / "h06_s2027.json", report)
    ranking = ["actuator"] + [m for m in L.MECHANISMS if m != "actuator"]
    diagnosis_feedback = {"first_status": "valid", "first_codes": [], "retried": False, "final_status": "valid",
                          "final_codes": [], "requests": 1, "calls": [{"step": "first", "input_tokens": 2800,
                                                                       "output_tokens": 180, "wall_s": 2.0}]}
    records = {
        "alarm": {"results": {"FAULT": {"warned": True}, "NONE": {"warned": False}}},
        "evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.13, "FAULT": 0.88}},
                     "reports": {"FAULT": {"path": demo._rel(report), "sha256_lf": demo.sha(report, True)}}},
        "loop_start": {"decision": {"action": "continue", "ranking": ranking, "probe": "P_torque"},
                       "started": {"request_id": "q", "model_probe_used": True}, "feedback": diagnosis_feedback,
                       "call": {"path": PATH, "record_sha256": "d" * 64, "input_tokens": 2800, "output_tokens": 180,
                                "execution_wall_s": 2.1}},
        "loop_close": {"proceed_to_correction": True,
                       "loop": {"status": "confirmed", "hypotheses": ["actuator"], "observed": {"P_torque": "abnormal"},
                                "requests": [], "probe_costs": []}},
        "specificity": {"required": ["P_noise", "P_value"], "outcomes": {}, "passed": True, "measurements": {}},
    }
    for name in demo.STEPS[1:demo.STEPS.index("correction")]:
        demo._write_step(tag, name, demo._guard(tag, name), EXPERIMENT, ctx.protocol, records.get(name, {}))
    make_loop(tmp_path / "loops", case=ctx.cases["FAULT"], mechanism="actuator")
    info = demo.correction_evidence(tag, ROOT)
    packet_path = demo.ROOT / info["packet"]
    record = write_record(demo.folder_for(tag) / "model" / "correction.feedback1.json", "correction", packet_path, *answers)
    return ctx, record


def fake_executor(calls):
    def make(tag, isaaclab):
        def run(request):
            calls.append(request)
            return {"exit_code": 0, "gpu_s": 100.0, "verdict": "recovered", "result_sha": "f" * 64, "tag": request["tag"]}
        return run
    return make


def isaaclab_root(tmp_path) -> Path:
    root = tmp_path / "isaaclab"
    (root / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True, exist_ok=True)
    return root


def test_retried_correction_proposal_runs_on_the_demo_registry_ledger(demo, monkeypatch, tmp_path):
    bad = {**GOOD, "cited_facts": ["registry"]}
    ctx, record = ready_for_correction(demo, monkeypatch, tmp_path, "t10", json.dumps(bad), json.dumps(GOOD))
    proposed = demo.correction_propose("t10", ROOT, record, budget_gpu_s=2400)
    rid = proposed["request_id"]
    evidence = proposed["evidence"]["model_proposal"]
    assert proposed["recovery_mode"] == "restart" and proposed["correction_id"] == "restore_motor_torque_speed"
    assert proposed["registry_digest"] == R.registry_digest(demo.registry_v3())
    assert evidence["accepted"] is True and evidence["agrees_with_rule"] is True and evidence["proposal"] == GOOD
    assert evidence["feedback"]["first_codes"] == ["C_CITED_UNKNOWN"] and evidence["feedback"]["retried"] is True
    demo.approve_correction("t10", rid, "operator")
    calls = []
    monkeypatch.setattr(DC, "make_executor", fake_executor(calls))
    step = demo.correction("t10", isaaclab_root(tmp_path), rid, record)
    assert [e["event"] for e in demo.correction_ledger(ctx).events()] == ["proposed", "approved", "consumed", "receipt"]
    assert calls[0]["tag"] == "t10_fix" and step["model_proposal"]["feedback"]["final_status"] == "valid"
    assert step["call"]["requests"] == 2 and step["call"]["path"] == PATH
    result = demo.summarize("t10")
    assert result["outcome"] == "recovered" and result["success_conditions"]["correction_call_accepted"] is True
    assert result["model_feedback"]["correction"]["retries"] == 1 and result["model_feedback"]["correction"]["final_valid"]
    assert result["model_feedback"]["diagnosis"]["retries"] == 0
    assert result["correction"]["call"]["requests"] == 2


def test_invalid_after_retry_keeps_the_rule_request_but_fails_the_accepted_condition(demo, monkeypatch, tmp_path):
    _, record = ready_for_correction(demo, monkeypatch, tmp_path, "t11", '{"checklist": [', '{"checklist": [')
    proposed = demo.correction_propose("t11", ROOT, record, budget_gpu_s=2400)
    evidence = proposed["evidence"]["model_proposal"]
    assert proposed["recovery_mode"] == "restart" and proposed["correction_id"] == "restore_motor_torque_speed"
    assert evidence["accepted"] is False and evidence["proposal"] is None and evidence["reasons"]
    assert evidence["feedback"]["final_codes"] == ["C_JSON"]
    demo.approve_correction("t11", proposed["request_id"], "operator")
    monkeypatch.setattr(DC, "make_executor", fake_executor([]))
    demo.correction("t11", isaaclab_root(tmp_path), proposed["request_id"], record)
    result = demo.summarize("t11")
    assert result["outcome"] == "recovered" and result["demo_success"] is False
    assert result["success_conditions"]["correction_call_accepted"] is False
    assert "correction_call_accepted" in result["unmet_conditions"]
    assert result["model_feedback"]["correction"]["final_valid"] is False


def test_correction_feedback_review_matches_the_v3_1_registry_validation(demo, monkeypatch, tmp_path):
    _, record = ready_for_correction(demo, monkeypatch, tmp_path, "t12", json.dumps(GOOD))
    packet_path = demo.folder_for("t12") / "model" / "correction.packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    for proposal in (GOOD, {**GOOD, "recovery_mode": "resume"}, {**GOOD, "cited_facts": ["registry"]},
                     {**GOOD, "correction_id": "restore_body_frame_reward"}):
        ours = F.validate("correction", json.dumps(proposal), packet)["status"] == "valid"
        theirs = CM.validate_model_proposal(proposal, packet, demo.registry_v3())["accepted"]
        assert ours == theirs
    review = demo.review_correction_record({"decision_path": PATH}, record, packet, packet_path)
    assert review["accepted"] and review["feedback"]["retried"] is False and review["rule_visible_to_model"] is True


def test_correction_infra_record_and_tampered_feedback_are_refused(demo, monkeypatch, tmp_path):
    ctx, record = ready_for_correction(demo, monkeypatch, tmp_path, "t13", '{"checklist": [', {"content": ""})
    with pytest.raises(ValueError, match="정상 종료"):
        demo.correction_propose("t13", ROOT, record, budget_gpu_s=2400)
    retry = write_record(record.with_name("correction.feedback2.json"), "correction",
                         demo.folder_for("t13") / "model" / "correction.packet.json", '{"checklist": [', json.dumps(GOOD))
    body = json.loads(retry.read_text(encoding="utf-8"))
    body["feedback"]["text"] += "\n힌트: restore_motor_torque_speed"
    retry.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="피드백"):
        demo.correction_propose("t13", ROOT, retry, budget_gpu_s=2400)
    assert not (tmp_path / "loops" / ctx.cases["FAULT"] / "correction_ledger.jsonl").exists()


def test_budget_floor_terminal_guard_and_orphan_recovery_follow_v3_1(demo, monkeypatch, tmp_path):
    _, record = ready_for_correction(demo, monkeypatch, tmp_path, "t14", json.dumps(GOOD))
    for budget in (2399, 0, 2400.0):
        with pytest.raises(ValueError, match="하한"):
            demo.correction_propose("t14", ROOT, record, budget_gpu_s=budget)
    rid = demo.correction_propose("t14", ROOT, record, budget_gpu_s=2400)["request_id"]
    demo.approve_correction("t14", rid, "operator")

    def crash(demo_tag, isaaclab):
        def run(request):
            path = DC.fix_folder(demo_tag) / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": "e2e_correction_result_v4", "tag": request["tag"],
                                        "correction_request_id": request["request_id"],
                                        "recovery_mode": request["recovery_mode"], "source_commit_sha": EXPERIMENT,
                                        "label": "recovered", "costs": {"train": 900.0, "eval": 60.0}}), encoding="utf-8")
            raise KeyboardInterrupt
        return run
    monkeypatch.setattr(DC, "make_executor", crash)
    isaaclab = isaaclab_root(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        demo.correction("t14", isaaclab, rid, record)
    step = demo.recover_correction("t14", isaaclab, rid, record)
    assert step["recovered_orphan"] is True and step["receipt"]["verdict"] == "recovered"
    (demo.folder_for("t14") / "terminal.json").write_text(json.dumps({"outcome": "harness_failure"}), encoding="utf-8")
    with pytest.raises(ValueError, match="종료"):
        demo.correction_evidence("t14", ROOT)


def test_commands_name_the_feedback_caller_and_this_runner(demo):
    commands = demo.correction_commands(demo.ctx_of("f2"), "t20", {"decision_path": PATH})
    joined = "\n".join(commands)
    assert "evals/feedback_model_call.py --kind correction" in joined and "correction.feedback1.json" in joined
    assert "e2e_demo_v4.py correct-run" in joined and "v3_1" not in joined and "nat_decision" not in joined


def test_no_default_registry_ledger_calls_remain():
    import ast
    tree = ast.parse((ROOT / "evals/e2e_demo_v4.py").read_text(encoding="utf-8"))
    calls = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ("CorrectionLedger", "CR._ledger", "CR.run"):
                    calls.setdefault(function.name, []).append((name, [k.arg for k in node.keywords]))
    assert calls == {"correction_ledger": [("CorrectionLedger", ["registry"])],
                     "record_harness_failure": [("CorrectionLedger", [])]}


def test_correction_executor_is_bound_to_the_v4_demo():
    assert DC._demo() is E
    source = (ROOT / "bench/demo_correction_v4.py").read_text(encoding="utf-8")
    assert 'str(ROOT / "bench/demo_correction_v4.py")' in source and "import e2e_demo_v4 as E" in source
    assert '"schema": "e2e_correction_result_v4"' in source and "e2e_demo_v3_1 as" not in source


def test_begin_sees_outputs_of_the_same_tag_in_other_demos(demo, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    ctx = demo.ctx_of("f2")
    assert demo.seed_unused_evidence(ctx, isaaclab, "t30")["existing_outputs"] == []
    (tmp_path / "bench/private/e2e_demo_v3_1/t30").mkdir(parents=True)
    (demo.RESULTS / "t30_spec").mkdir(parents=True)
    assert len(demo.seed_unused_evidence(ctx, isaaclab, "t30")["existing_outputs"]) == 2


# --- 하네스 실패 ---
def committed_at_experiment(monkeypatch):
    monkeypatch.setattr(E, "committed_blob_lf",
                        lambda commit, relative: E.sha(E.CODE_ROOT / relative, True) if commit == EXPERIMENT else None)
    monkeypatch.setattr(E, "tool_commit", lambda: "b" * 40)


def test_harness_failure_on_a_v4_demo_is_preregistered(demo, monkeypatch, tmp_path):
    committed_at_experiment(monkeypatch)
    ready_for_correction(demo, monkeypatch, tmp_path, "t40", json.dumps(GOOD))
    for step in ("loop_close", "bogus"):
        with pytest.raises(ValueError, match="바로 다음"):
            E.record_harness_failure("v4", "t40", step, "e", "evals/e2e_demo_v4.py:1")
    with pytest.raises(ValueError, match="절대 경로"):
        E.record_harness_failure("v4", "t40", "correction-propose", r"failed at C:\Users\x\a.py", "evals/e2e_demo_v4.py:1")
    record = E.record_harness_failure("v4", "t40", "correction-propose", "e", "evals/e2e_demo_v4.py:1")
    assert record["detail"]["category_preregistered_for_demo"] is True and record["wording"] == E.OUTCOMES["harness_failure"]
    assert E.summarize("t40")["outcome"] == "harness_failure"
    with pytest.raises(ValueError):
        E.record_harness_failure("v3_1", "t40", "correction-propose", "e", "evals/e2e_demo_v4.py:1")


@pytest.mark.parametrize("event", ["approved", "rejected"])
def test_harness_failure_refuses_registered_outcomes(demo, monkeypatch, tmp_path, event):
    committed_at_experiment(monkeypatch)
    _, record = ready_for_correction(demo, monkeypatch, tmp_path, "t41", json.dumps(GOOD))
    rid = demo.correction_propose("t41", ROOT, record, budget_gpu_s=2400)["request_id"]
    if event == "approved":
        demo.approve_correction("t41", rid, "operator")
    else:
        demo.reject_correction("t41", rid, "operator", "r")
    with pytest.raises(ValueError, match="승인·거절"):
        E.record_harness_failure("v4", "t41", "correct-run", "e", "evals/e2e_demo_v4.py:1")
    with pytest.raises(LedgerError):
        demo.approve_correction("t41", rid, "operator") if event == "approved" else demo.approve_correction("t41", rid, "x")
