"""설계자 밖 결함 U1 종단 시연: 교정 범위 점검(대조·강제 모드)과 교정 근거 결합, 대조 범주·예측 판정, 고리 probe 실행
(loop-run·loop-recover), 결함 학습 공용 진입점 연결, 하네스 실패 규칙, 사전등록. GPU·모델 없음. v3.1과 같은 원장·승인·고아
복구 규칙은 v3.1 테스트가 시험한다."""
import ast
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
import demo_correction_u1 as DC  # noqa: E402
import e2e_demo_u1 as E  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import loop  # noqa: E402
import upstream_faults as U  # noqa: E402
import upstream_runs as UR  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402
from rl_triage.correction_ledger import LedgerError  # noqa: E402

EXPERIMENT = "a" * 40
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}
U1_MEASUREMENT = {"track_lin_vel_xy_exp_rel_error": 0.93, "track_ang_vel_z_exp_rel_error": 0.97}  # 두 추종 항 모두 이상
F1_MEASUREMENT = {"track_lin_vel_xy_exp_rel_error": 0.88, "track_ang_vel_z_exp_rel_error": 1.7e-8}  # 등록 표적 항만 이상


def use_tmp(tmp_path, monkeypatch):
    """시연 출력·결과·고리 경로와 ROOT를 임시 폴더로 돌린다."""
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "bench/private/e2e_demo_u1")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "PROTOCOL_DIR", tmp_path / "protocols")
    (tmp_path / "protocols").mkdir(exist_ok=True)
    path = tmp_path / "protocols/e2e_demo_u1_v1.json"
    if not path.exists():
        path.write_text(json.dumps(E.fixed("u1", "single")), encoding="utf-8")
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    return E


@pytest.fixture
def demo(tmp_path, monkeypatch):
    return use_tmp(tmp_path, monkeypatch)


def protocol(mode="contrast"):
    fixed = E.fixed("u1", "single")
    return {**fixed, "scope_gate": {**fixed["scope_gate"], "mode": mode}}


def write_steps(tag, upto, records=None, mode="contrast", monkeypatch=None):
    """begin부터 upto 앞까지 단계 기록을 쓴다(SHA 사슬 포함)."""
    ctx = E.ctx_of("u1")
    if monkeypatch is not None:
        monkeypatch.setattr(E, "verify_for", lambda t, isaaclab: (protocol(mode), EXPERIMENT))
    records = records or {}
    for name in E.STEPS[:E.STEPS.index(upto)]:
        if not E.step_path(tag, name).exists():
            body = records.get(name, {"variant": "u1", "decision_path": "single", "seed_unused": {"hits": []}}
                               if name == "begin" else {})
            E._write_step(tag, name, E._guard(tag, name), EXPERIMENT, ctx.protocol, body)
    return ctx


def make_loop(loops, case, measurement):
    """reward로 확정된 고리: P_reward 요청이 승인·소비되고 abnormal receipt에 항별 측정값이 남아 있다."""
    d = loops / case
    d.mkdir(parents=True)
    prereg = {"case": case, "hypotheses": ["reward"], "checkpoint_sha256": "ab" * 32}
    led = L.Ledger(d / "ledger.jsonl")
    rid = led.propose(prereg, "P_reward", 100)["request_id"]
    led.approve(rid, "tester")
    led.consume(rid, prereg, "P_reward")
    led.receipt(rid, "abnormal", measurement, 10.0, 0)
    state = {"hypotheses": ["reward"], "observed": {"P_reward": "abnormal"}, "skipped": [], "budget": 4,
             "status": "confirmed", "pending": None, "history": []}
    (d / "prereg.json").write_text(json.dumps(prereg), encoding="utf-8")
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return rid


def ready_for_scope(tmp_path, monkeypatch, tag, measurement=U1_MEASUREMENT, mode="contrast"):
    """고리 확정·특이성(게이트 없음)까지의 단계 기록과 확정 고리, 결함 평가 report를 만든다."""
    report = tmp_path / "evals/results" / tag / "fault.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P0C_RUNS / "h02_s2027.json", report)  # 정지 비율 ≥ 0.9인 reward 결함 평가(규칙 restart)
    ranking = ["reward"] + [m for m in L.MECHANISMS if m != "reward"]
    records = {
        "alarm": {"results": {"FAULT": {"warned": True}, "NONE": {"warned": False}}},
        "evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.13, "FAULT": 1.02}},
                     "reports": {"FAULT": {"path": E._rel(report), "sha256_lf": E.sha(report, True)}}},
        "loop_start": {"decision": {"action": "continue", "ranking": ranking, "probe": "P_reward"},
                       "started": {"request_id": "q"}, "call": {"path": "single", "input_tokens": 3000}},
        "loop_close": {"proceed_to_correction": True,
                       "loop": {"status": "confirmed", "hypotheses": ["reward"], "observed": {"P_reward": "abnormal"},
                                "requests": [], "probe_costs": []}},
        "specificity": {"required": [], "passed": True},
    }
    ctx = write_steps(tag, "scope_check", records, mode, monkeypatch)
    make_loop(tmp_path / "loops", ctx.cases["FAULT"], measurement)
    return ctx


def correction_record(tag):
    info = E.correction_evidence(tag, ROOT)
    packet = json.loads((E.ROOT / info["packet"]).read_text(encoding="utf-8"))
    record_path = E.folder_for(tag) / "model" / "correction.call1.json"
    proposal = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
                "checklist": dict(CHECKLIST), "hold_reason": None}
    record_path.write_text(json.dumps({
        "schema": CM.RECORD_SCHEMA, "exit_code": 0, "packet_digest": CM.packet_digest(packet),
        "messages_sha256": L.digest(CM.build_messages(packet)), "skill_sha256": CM.SKILL_SHA256,
        "content": json.dumps(proposal), "request": {"model": E.MODEL}, "execution_wall_s": 1.5,
        "input_tokens": 2000, "output_tokens": 200}), encoding="utf-8")
    return record_path, packet


def fake_executor(verdict, calls):
    def make(tag, isaaclab):
        def run(request):
            calls.append(request)
            return {"exit_code": 0, "gpu_s": 600.0, "verdict": verdict, "result_sha": "f" * 64, "tag": request["tag"]}
        return run
    return make


def run_correction_step(demo, monkeypatch, tmp_path, tag, verdict):
    """범위 점검 뒤 교정 근거·제안·승인·교정 실행(가짜 실행기)까지. (제안 사건, 교정 단계, 실행기 호출)."""
    record_path, _ = correction_record(tag)
    proposed = demo.correction_propose(tag, ROOT, record_path, budget_gpu_s=2400)
    demo.approve_correction(tag, proposed["request_id"], "operator")
    calls = []
    monkeypatch.setattr(DC, "make_executor", fake_executor(verdict, calls))
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True, exist_ok=True)
    return proposed, demo.correction(tag, isaaclab, proposed["request_id"], record_path), calls


# --- 교정 범위 점검과 대조 ---
@pytest.mark.parametrize("verdict,contrast", [("not_recovered", "insufficient_and_not_recovered"),
                                              ("recovered", "insufficient_but_recovered")])
def test_contrast_mode_records_the_scope_then_runs_the_correction_once(demo, monkeypatch, tmp_path, verdict, contrast):
    ready_for_scope(tmp_path, monkeypatch, "u10")
    with pytest.raises(ValueError, match="scope-check"):  # 범위 점검 전에는 교정 근거를 만들지 않는다
        demo.correction_evidence("u10", ROOT)
    step = demo.scope_check("u10", ROOT)
    result = step["result"]
    assert result["verdict"] == "scope_insufficient" and result["uncovered_terms"] == ["track_ang_vel_z_exp"]
    assert result["abnormal_terms"] == ["track_ang_vel_z_exp", "track_lin_vel_xy_exp"]
    assert result["registered_targets"] == ["track_lin_vel_xy_exp"] and step["agent_judgment"] is False
    assert step["proceed_to_correction"] is True and step["mode"] == "contrast"
    assert step["next_commands"][0] == "python evals/e2e_demo_u1.py correction-evidence --tag u10"
    assert not (demo.folder_for("u10") / "terminal.json").exists()
    proposed, fix, calls = run_correction_step(demo, monkeypatch, tmp_path, "u10", verdict)
    bound = proposed["evidence"]["scope_check"]
    assert bound["verdict"] == "scope_insufficient" and bound["mode"] == "contrast"
    assert bound["step_sha256"] == demo.sha(demo.step_path("u10", "scope_check"))
    assert len(calls) == 1 and calls[0]["recovery_mode"] == "restart" and fix["scope_check_bound"] is True
    summary = demo.summarize("u10")
    assert summary["schema"] == "e2e_demo_result_u1_v1" and summary["outcome"] == verdict
    assert summary["scope_contrast"] == {"category": contrast, "wording": demo.SCOPE_WORDING[contrast]}
    assert demo.SCOPE_WORDING[contrast] in summary["wording"]
    assert summary["b_code_wording"] == ("B-code는 우리 주입 방식 때문에 바로 보이며 원전의 백엔드 0 읽기는 지문으로 "
                                         "보이지 않았을 것[추론]")
    checks = summary["success_conditions"]
    assert checks["correction_judged"] is True and checks["scope_check_recorded"] is True
    predicted = summary["prediction_results"]
    assert predicted["scope_check"] is True and predicted["correction"] is (verdict == "not_recovered")
    assert predicted["scope_contrast"] is (verdict == "not_recovered")
    assert summary["scope_check"]["result"]["verdict"] == "scope_insufficient"
    assert json.loads((demo.RESULTS / "u10/demo_result.json").read_text(encoding="utf-8")) == summary


@pytest.mark.parametrize("verdict", ["not_recovered", "recovered"])
def test_a_judged_correction_is_a_demo_success_when_every_condition_holds(demo, monkeypatch, tmp_path, verdict):
    """대조 시연의 성공은 회복이 아니라 교정 판정이다. 성공 조건을 모두 참으로 두면 not_recovered도 demo_success다."""
    ready_for_scope(tmp_path, monkeypatch, "u15")
    demo.scope_check("u15", ROOT)
    run_correction_step(demo, monkeypatch, tmp_path, "u15", verdict)
    monkeypatch.setattr(E, "conditions", lambda steps, outcome: {"all_registered_conditions": True})
    summary = demo.summarize("u15")
    assert summary["outcome"] == verdict and summary["demo_success"] is True and summary["unmet_conditions"] == []


def test_a_scope_record_changed_after_the_proposal_is_not_bound(demo, monkeypatch, tmp_path):
    ready_for_scope(tmp_path, monkeypatch, "u16")
    demo.scope_check("u16", ROOT)
    record_path, _ = correction_record("u16")
    proposed = demo.correction_propose("u16", ROOT, record_path, budget_gpu_s=2400)
    path = demo.step_path("u16", "scope_check")
    changed = {**json.loads(path.read_text(encoding="utf-8")), "mode": "enforce"}  # 사슬(previous_sha256)은 그대로
    path.write_text(json.dumps(changed), encoding="utf-8")
    demo.approve_correction("u16", proposed["request_id"], "operator")
    monkeypatch.setattr(DC, "make_executor", fake_executor("not_recovered", []))
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    fix = demo.correction("u16", isaaclab, proposed["request_id"], record_path)
    assert fix["scope_check_bound"] is False
    assert demo.summarize("u16")["success_conditions"]["scope_check_recorded"] is False


def test_enforce_mode_ends_before_the_correction(demo, monkeypatch, tmp_path):
    ctx = ready_for_scope(tmp_path, monkeypatch, "u11", mode="enforce")
    step = demo.scope_check("u11", ROOT)
    assert step["proceed_to_correction"] is False and "next_commands" not in step
    terminal = json.loads((demo.folder_for("u11") / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["outcome"] == "correction_scope_insufficient"
    assert terminal["detail"]["scope_check"]["uncovered_terms"] == ["track_ang_vel_z_exp"]
    with pytest.raises(ValueError, match="종료"):
        demo.correction_evidence("u11", ROOT)
    summary = demo.summarize("u11")
    assert summary["outcome"] == "correction_scope_insufficient" and summary["demo_success"] is False
    assert summary["scope_contrast"] is None
    assert summary["success_conditions"]["scope_check_recorded"] is False  # 교정 요청 근거에 결합되지 않았다
    assert not (tmp_path / "loops" / ctx.cases["FAULT"] / "correction_ledger.jsonl").exists()  # 교정 요청 없음


def test_enforce_mode_proceeds_when_the_registered_target_covers_the_abnormal_terms(demo, monkeypatch, tmp_path):
    ready_for_scope(tmp_path, monkeypatch, "u12", measurement=F1_MEASUREMENT, mode="enforce")
    step = demo.scope_check("u12", ROOT)
    assert step["result"]["verdict"] == "scope_sufficient" and step["result"]["uncovered_terms"] == []
    assert step["proceed_to_correction"] is True and not (demo.folder_for("u12") / "terminal.json").exists()


def test_scope_check_reads_the_confirming_receipt_and_needs_the_specificity_step(demo, monkeypatch, tmp_path):
    write_steps("u13", "specificity", monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="specificity"):  # SHA 사슬: 특이성 단계가 먼저다
        demo.scope_check("u13", ROOT)
    ready_for_scope(tmp_path, monkeypatch, "u14", measurement={"track_lin_vel_xy_exp_rel_error": 0.9})
    step = demo.scope_check("u14", ROOT)
    assert step["result"]["verdict"] == "scope_unknown"  # 항 하나가 빠지면 판정하지 않는다
    with pytest.raises(FileExistsError):
        demo.scope_check("u14", ROOT)


def test_scope_targets_are_the_registered_reward_functions():
    assert E.scope_targets("restore_body_frame_reward") == ["track_lin_vel_xy_exp"]
    assert E.scope_targets("restore_motor_torque_speed") == []
    assert E.ctx_of("u1").correction_id == "restore_body_frame_reward" == DC.UPSTREAM_TARGETS["U1"]


# --- 고리 probe 실행(loop-run·loop-recover) ---
def started_loop(tmp_path, monkeypatch, tag):
    ctx = write_steps(tag, "loop_close", monkeypatch=monkeypatch)
    started = V2.start_loop(ctx.cases["FAULT"], ["reward", "physics", "optimizer"], "ab" * 32, {}, "P_reward")
    _, _, led = loop._load(ctx.cases["FAULT"])
    led.approve(started["request_id"], "operator")
    return ctx, started["request_id"]


def test_loop_run_measures_one_approved_probe_with_the_upstream_executor(demo, monkeypatch, tmp_path):
    ctx, rid = started_loop(tmp_path, monkeypatch, "u20")
    seen, runs = {}, []

    def executor(isaaclab, **kwargs):
        seen.update(kwargs, isaaclab=isaaclab)
        return (lambda case, probe: runs.append((case, probe)) or (dict(U1_MEASUREMENT), 41.0, 0)), (lambda case, probe: None)
    monkeypatch.setattr(UR, "loop_executor", executor)
    out = demo.loop_run("u20", tmp_path / "isaaclab", rid)
    assert runs == [(ctx.cases["FAULT"], "P_reward")] and "P_reward: abnormal" in out["message"]
    assert seen["fault"] == "U1" and seen["seed"] == 2040 and seen["tag"] == "u20_loop"
    assert seen["results"] == demo.RESULTS and seen["watchdog_s"] == 600
    _, state, led = loop._load(ctx.cases["FAULT"])
    assert state["observed"] == {"P_reward": "abnormal"} and led.requests()[rid]["state"] == "done"
    assert led.requests()[rid]["receipt"]["measurement"] == U1_MEASUREMENT
    with pytest.raises(L.LedgerError):  # 승인은 한 번만 소비된다(CLI는 거부 코드 2)
        demo.loop_run("u20", tmp_path / "isaaclab", rid)
    with pytest.raises(ValueError, match="없는 요청"):  # loop.run의 SystemExit 거부를 시연 거부로 바꾼다
        demo.loop_run("u20", tmp_path / "isaaclab", "nope")


def test_loop_commands_only_between_loop_start_and_close(demo, monkeypatch, tmp_path):
    write_steps("u21", "loop_start", monkeypatch=monkeypatch)
    for call in (lambda: demo.loop_run("u21", tmp_path, "r"), lambda: demo.loop_recover("u21")):
        with pytest.raises(ValueError, match="loop_start"):
            call()
    ctx, rid = started_loop(tmp_path, monkeypatch, "u22")
    (demo.folder_for("u22") / "terminal.json").write_text(json.dumps({"outcome": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="종료"):
        demo.loop_run("u22", tmp_path, rid)


def test_loop_recover_closes_an_orphan_without_rerun(demo, monkeypatch, tmp_path):
    ctx, rid = started_loop(tmp_path, monkeypatch, "u23")
    prereg, _, led = loop._load(ctx.cases["FAULT"])
    led.consume(rid, prereg, "P_reward")  # 실행 중 끊겨 결과 없이 남은 요청
    assert led.orphans() == [rid]
    out = demo.loop_recover("u23")
    assert "산출물 없음" in out["message"]
    _, state, led = loop._load(ctx.cases["FAULT"])
    assert led.orphans() == [] and state["observed"] == {"P_reward": "unknown"}


# --- 결함 학습(공용 진입점) ---
def fake_upstream_train(isaaclab, calls, fail=None):
    def train(root, **kwargs):
        calls.append(kwargs)
        if fail is not None:
            raise fail
        name, seed = kwargs["run_name"], kwargs["seed"]
        run_dir = Path(root) / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-13_00-00-00_{name}"
        (run_dir / "params").mkdir(parents=True)
        for file in ("model_0.pt", "model_299.pt", "params/env.yaml"):
            (run_dir / file).write_text(file, encoding="utf-8")
        (run_dir / "params/agent.yaml").write_text(f"seed: {seed}\n", encoding="utf-8")
        kwargs["telemetry_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["telemetry_path"].write_text(json.dumps({"series": {A.METRIC: [1.0] * 300}}), encoding="utf-8")
        meta = {"case_id": "u1", "seed": seed, "num_envs": 4096, "max_iterations": 300, "task": UR.TASK, "fault": "U1",
                "train_exit_code": 0, "wall_time_s": 600.0, "run_dir": str(run_dir), "entry": "bench/upstream_isaac.py train",
                "upstream_readback": {"fault": "U1", "ok": True}, "upstream_aux": {"used_for_verdict": False}}
        kwargs["meta_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["meta_path"].write_text(json.dumps(meta), encoding="utf-8")
        execution = {"wall_s": 612.34, "exit_code": 0}
        return {"meta": meta, "execution": execution, "extract": {"wall_s": 9.0}, "run_dir": str(run_dir)}
    return train


def test_fault_training_uses_the_shared_upstream_entry(demo, monkeypatch, tmp_path):
    write_steps("u30", "train_FAULT", monkeypatch=monkeypatch)
    isaaclab = tmp_path / "isaaclab"
    calls = []
    monkeypatch.setattr(UR, "train", fake_upstream_train(isaaclab, calls))
    step = demo.train("u30", "FAULT", isaaclab)
    call = calls[0]
    assert call["fault"] == "U1" and call["seed"] == 2040 and call["run_name"] == "u1_s2040"
    assert call["telemetry_path"] == tmp_path / V2.RUNS_DIR / "u1_s2040.telemetry.json"
    assert call["meta_path"] == tmp_path / "bench/private/u1_s2040.meta.json"
    assert call["work_dir"] == demo.folder_for("u30") / "exec/train_FAULT_attempt1" and call["watchdog_s"] == 1800
    assert step["execution"]["parent_wall_s"] == 612.3 and step["upstream"]["readback"]["ok"] is True
    assert step["upstream"]["aux"]["used_for_verdict"] is False and step["artifacts"]["run"] == "u1_s2040"


def test_fault_training_failure_ends_the_demo_as_invalid(demo, monkeypatch, tmp_path):
    write_steps("u31", "train_FAULT", monkeypatch=monkeypatch)
    monkeypatch.setattr(UR, "train", fake_upstream_train(tmp_path, [], ValueError("확인 줄이 하나가 아니다")))
    with pytest.raises(RuntimeError, match="invalid"):
        demo.train("u31", "FAULT", tmp_path / "isaaclab")
    terminal = json.loads((demo.folder_for("u31") / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["outcome"] == "invalid" and terminal["detail"]["error_type"] == "ValueError"
    assert not demo.step_path("u31", "train_FAULT").exists()


# --- 하네스 실패·교정 실행기·원장 ---
def test_harness_failure_allows_loop_and_correction_commands_at_the_right_steps(demo, monkeypatch, tmp_path):
    assert E.harness_allowed_steps("loop_start") == {"loop_close", "loop-run", "loop-recover"}
    assert E.harness_allowed_steps("specificity") == {"scope_check"}
    assert E.harness_allowed_steps("scope_check") == {"correction", *E.CORRECTION_COMMANDS}
    write_steps("u40", "loop_close", monkeypatch=monkeypatch)
    monkeypatch.setattr(E, "tool_commit", lambda: "b" * 40)
    with pytest.raises(ValueError, match="바로 다음"):
        E.record_harness_failure("u1", "u40", "correct-run", "e", "evals/e2e_demo_u1.py:1")
    with pytest.raises(ValueError, match="모르는 시연 판"):
        E.record_harness_failure("v3_1", "u40", "loop-run", "e", "evals/e2e_demo_u1.py:1")


def test_correction_executor_is_bound_to_the_u1_demo():
    assert DC._demo() is E
    source = (ROOT / "bench/demo_correction_u1.py").read_text(encoding="utf-8")
    assert 'str(ROOT / "bench/demo_correction_u1.py")' in source and "import e2e_demo_u1 as E" in source


def test_no_default_registry_ledger_calls_remain():
    tree = ast.parse((ROOT / "evals/e2e_demo_u1.py").read_text(encoding="utf-8"))
    calls = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("CorrectionLedger", "CR._ledger", "CR.run"):
                calls.setdefault(function.name, []).append((ast.unparse(node.func), [k.arg for k in node.keywords]))
    assert calls == {"correction_ledger": [("CorrectionLedger", ["registry"])],
                     "record_harness_failure": [("CorrectionLedger", [])]}


def test_correction_needs_an_approved_request_after_the_scope_check(demo, monkeypatch, tmp_path):
    ready_for_scope(tmp_path, monkeypatch, "u41")
    demo.scope_check("u41", ROOT)
    record_path, _ = correction_record("u41")
    rid = demo.correction_propose("u41", ROOT, record_path, budget_gpu_s=2400)["request_id"]
    with pytest.raises(ValueError, match="승인된"):
        demo.correction("u41", tmp_path / "isaaclab", rid, record_path)
    demo.reject_correction("u41", rid, "operator", "범위 부족을 보고 거절")
    with pytest.raises(LedgerError):
        demo.approve_correction("u41", rid, "operator")
    assert demo.summarize("u41")["outcome"] == "correction_rejected"


# --- 사전등록 ---
def test_variant_seed_protocol_and_bound_sources():
    ctx = E.ctx_of("u1")
    assert (ctx.fault, ctx.case_id, ctx.mechanism, ctx.seed, ctx.restart_seed) == ("U1", "u1", "reward", 2040, 3040)
    used = {2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 3029, 3030, 3031, 3032, 3033, 3034,
            7, 42, 123}
    assert not {ctx.seed, ctx.restart_seed} & used and ctx.cases == {"NONE": "baseline_p0c_s2040", "FAULT": "u1_s2040"}
    assert ctx.protocol.name == "e2e_demo_u1_v1.json" and E.PRIVATE.name == "e2e_demo_u1"
    assert {"evals/e2e_demo_u1.py", "bench/demo_correction_u1.py", "bench/upstream_faults.py", "bench/upstream_isaac.py",
            "bench/upstream_runs.py", "src/rl_triage/correction_scope.py"} <= E.CODE
    assert not {"evals/e2e_demo_v3_1.py", "bench/demo_correction_v3_1.py"} & E.CODE
    assert all((ROOT / name).is_file() for name in E.CODE)
    assert "correction_scope_insufficient" in E.TERMINAL_OUTCOMES and "harness_failure" in E.TERMINAL_OUTCOMES
    for path in E.DECISION_PATHS:
        fixed = E.fixed("u1", path)
        assert json.loads(json.dumps(fixed)) == fixed and fixed["schema"] == "e2e_demo_protocol_u1_v1"
        assert fixed["scope_gate"]["mode"] == "contrast" and fixed["scope_gate"] == E.SCOPE_GATE
        assert fixed["b_code_wording"] == ("B-code는 우리 주입 방식 때문에 바로 보이며 원전의 백엔드 0 읽기는 지문으로 "
                                           "보이지 않았을 것[추론]")
        assert fixed["registered_correction_scope"]["fault_functions"] == sorted(U.SWAPS["U1"])
        assert fixed["registered_correction_scope"]["target"] == "isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp"
        assert fixed["predictions"]["scope_contrast"] == "insufficient_and_not_recovered"
        assert set(fixed["success_conditions"]) == set(E.conditions({}, "in_progress"))
        assert "correction_recovered" not in fixed["success_conditions"]
        assert fixed["watchdogs_s"]["loop_probe"] == 600 and fixed["specificity_gate"]["probes"] == []
        assert fixed["fault_smoke"]["result"] == E.UPSTREAM_SMOKE_REL
    assert E.recommend("reward", json.loads((P0C_RUNS / "h02_s2027.json").read_text(encoding="utf-8")))["mode"] == "restart"


def test_upstream_smoke_binding_reads_the_public_result():
    smoke = E.upstream_smoke()
    assert smoke["outcome"] == "failure_present" and smoke["seed"] == 2026 and smoke["result"] == E.UPSTREAM_SMOKE_REL
    assert round(smoke["standstill_ratio"], 2) == E.fixed("u1", "single")["fault_smoke"]["standstill_ratio"]


def test_cli_requires_the_request_for_loop_run():
    with pytest.raises(SystemExit):
        E.main(["loop-run", "--tag", "u50"])
