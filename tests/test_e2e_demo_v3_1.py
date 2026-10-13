"""E4 종단 시연 v3.1: 교정 원장을 시연 등록 표로 일관되게 쓰는지(F2·F1 restart 실제 원장 회귀), 승인 CLI, 하네스 실패
기록·부분 요약, 새 seed. GPU·모델 없음. v3과 같은 단계·게이트·교정 실행기 검증은 v3 테스트가 시험한다."""
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
import demo_correction_v3_1 as DC  # noqa: E402
import e2e_demo_v3 as E3  # noqa: E402
import e2e_demo_v3_1 as E  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import loop  # noqa: E402
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_ledger import LedgerError  # noqa: E402
from test_correction_step import make_loop  # noqa: E402

EXPERIMENT = "a" * 40
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}
# 변형별: 이미 본 P0-C 결함 평가(규칙이 restart를 내는 것), 확정 메커니즘, 교정 id
CASES = {"f2": ("h06_s2027", "actuator", "restore_motor_torque_speed"),
         "f1": ("h02_s2027", "reward", "restore_body_frame_reward")}


def use_tmp(module, tmp_path, monkeypatch):
    """시연 출력·결과·고리 경로와 ROOT를 임시 폴더로 돌린다. 출력 위치는 저장소와 같은 상대 경로다."""
    prefix = "e2e_demo_v3_1" if module is E else "e2e_demo_v3"
    monkeypatch.setattr(module, "PRIVATE", tmp_path / "bench/private" / prefix)
    monkeypatch.setattr(module, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PROTOCOL_DIR", tmp_path / "protocols")
    (tmp_path / "protocols").mkdir(exist_ok=True)
    for variant in module.VARIANTS:  # 시연 단계 기록이 SHA로 묶는 프로토콜(내용은 하네스 테스트에서 따로 쓴다)
        path = tmp_path / "protocols" / f"{prefix}_{variant}.json"
        if not path.exists():
            path.write_text(json.dumps(module.fixed(variant, "single")), encoding="utf-8")
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    return module


@pytest.fixture
def demo(tmp_path, monkeypatch):
    return use_tmp(E, tmp_path, monkeypatch)


def proposal_for(variant):
    return {"correction_id": CASES[variant][2], "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
            "checklist": dict(CHECKLIST), "hold_reason": None}


def ready_for_correction(module, monkeypatch, tmp_path, tag, variant="f2"):
    """확정 고리·특이성 통과까지의 단계 기록, 결함 평가 report, 교정 packet, 단일 호출 기록을 만든다."""
    ctx = module.ctx_of(variant)
    source, mechanism, _ = CASES[variant]
    module._write_step(tag, "begin", module._guard(tag, "begin"), EXPERIMENT, ctx.protocol,
                       {"variant": variant, "decision_path": "single", "seed_unused": {"hits": []}})
    monkeypatch.setattr(module, "verify_for", lambda t, isaaclab: ({"decision_path": "single"}, EXPERIMENT))
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    report = tmp_path / "evals/results" / tag / "fault.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P0C_RUNS / f"{source}.json", report)
    ranking = [mechanism] + [m for m in L.MECHANISMS if m != mechanism]
    probe = L.confirming_probe(mechanism)
    records = {
        "alarm": {"results": {"FAULT": {"warned": True}, "NONE": {"warned": False}}},
        "evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.13, "FAULT": 0.95}},
                     "reports": {"FAULT": {"path": module._rel(report), "sha256_lf": module.sha(report, True)}}},
        "loop_start": {"decision": {"action": "continue", "ranking": ranking, "probe": probe},
                       "started": {"request_id": "q"}, "call": {"path": "single", "input_tokens": 3000}},
        "loop_close": {"proceed_to_correction": True,
                       "loop": {"status": "confirmed", "hypotheses": [mechanism], "observed": {probe: "abnormal"},
                                "requests": [], "probe_costs": []}},
        "specificity": {"required": variant == "f2", "outcomes": {}, "passed": True, "measurements": {}},
    }
    for name in module.STEPS[1:module.STEPS.index("correction")]:
        module._write_step(tag, name, module._guard(tag, name), EXPERIMENT, ctx.protocol, records.get(name, {}))
    make_loop(tmp_path / "loops", case=ctx.cases["FAULT"], mechanism=mechanism)
    info = module.correction_evidence(tag, ROOT)
    packet = json.loads((module.ROOT / info["packet"]).read_text(encoding="utf-8"))
    record_path = module.folder_for(tag) / "model" / "correction.call1.json"
    record_path.write_text(json.dumps({
        "schema": CM.RECORD_SCHEMA, "exit_code": 0, "packet_digest": CM.packet_digest(packet),
        "messages_sha256": L.digest(CM.build_messages(packet)), "skill_sha256": CM.SKILL_SHA256,
        "content": json.dumps(proposal_for(variant)), "request": {"model": module.MODEL}, "execution_wall_s": 1.5,
        "input_tokens": 2000, "output_tokens": 200}), encoding="utf-8")
    return ctx, record_path


def fake_executor(calls):
    def make(tag, isaaclab):
        def run(request):
            calls.append(request)
            return {"exit_code": 0, "gpu_s": 100.0, "verdict": "recovered", "result_sha": "f" * 64, "tag": request["tag"]}
        return run
    return make


# --- 교정 원장 회귀 ---
@pytest.mark.parametrize("variant", ["f2", "f1"])
def test_restart_correction_runs_on_the_demo_registry_ledger(demo, monkeypatch, tmp_path, variant):
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, "t31", variant)
    proposed = demo.correction_propose("t31", ROOT, record_path, budget_gpu_s=2400)
    rid = proposed["request_id"]
    assert proposed["recovery_mode"] == "restart" and proposed["correction_id"] == CASES[variant][2]
    assert proposed["registry_digest"] == R.registry_digest(demo.registry_v3()) != R.registry_digest()
    with pytest.raises(LedgerError, match="등록 표가 제안 때와 다르다"):  # correct.py 기본 표 원장으로는 승인할 수 없다
        CR._ledger(ctx.cases["FAULT"], CR.LOOPS).approve(rid, "tester")
    demo.approve_correction("t31", rid, "operator")
    calls = []
    monkeypatch.setattr(DC, "make_executor", fake_executor(calls))
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    step = demo.correction("t31", isaaclab, rid, record_path)
    assert [e["event"] for e in demo.correction_ledger(ctx).events()] == ["proposed", "approved", "consumed", "receipt"]
    assert calls[0]["recovery_mode"] == "restart" and calls[0]["tag"] == "t31_fix"
    assert step["receipt"]["verdict"] == "recovered" and step["request"]["recovery_mode"] == "restart"
    assert demo.correction_status("t31")["requests"][rid]["state"] == "done"
    with pytest.raises(LedgerError):  # 승인은 한 번만 소비된다
        demo.run_correction(ctx, rid, fake_executor(calls)("t31", isaaclab), "t31_fix")
    assert len(calls) == 1
    monkeypatch.setattr(E, "tool_commit", lambda: "b" * 40)
    with pytest.raises(ValueError, match="교정 단계"):  # 교정까지 간 시연은 하네스 실패로 닫지 않는다
        E.record_harness_failure("v3_1", "t31", "correct-run", "e", "evals/e2e_demo_v3_1.py:1")


def test_v3_module_reproduces_the_seed_2031_stall(tmp_path, monkeypatch):
    v3 = use_tmp(E3, tmp_path, monkeypatch)
    ctx, record_path = ready_for_correction(v3, monkeypatch, tmp_path, "t3")
    with pytest.raises(LedgerError, match="actuator에 등록되지 않은 회복 방식 'restart'"):
        v3.correction_propose("t3", ROOT, record_path, budget_gpu_s=2400)
    assert not (tmp_path / "loops" / ctx.cases["FAULT"] / "correction_ledger.jsonl").exists()  # 원장 기록 없음


def test_reject_cli_uses_the_same_ledger(demo, monkeypatch, tmp_path):
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, "t32")
    rid = demo.correction_propose("t32", ROOT, record_path, budget_gpu_s=2400)["request_id"]
    demo.reject_correction("t32", rid, "operator", "예산 확인")
    assert demo.correction_status("t32")["requests"][rid]["state"] == "rejected"
    with pytest.raises(LedgerError):
        demo.approve_correction("t32", rid, "operator")


def test_no_default_registry_ledger_calls_remain():
    import ast
    tree = ast.parse((ROOT / "evals/e2e_demo_v3_1.py").read_text(encoding="utf-8"))
    calls = {}  # 함수 이름 → 그 안의 원장 생성·기본 원장 호출
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ("CorrectionLedger", "CR._ledger", "CR.run"):
                    calls.setdefault(function.name, []).append((name, [k.arg for k in node.keywords]))
    assert calls == {"correction_ledger": [("CorrectionLedger", ["registry"])],
                     "record_harness_failure": [("CorrectionLedger", [])]}  # 하네스 실패는 사건 읽기만 한다


# --- 새 seed·사전등록 ---
def test_variants_use_new_seeds_and_keep_v3_rules():
    used = {2026, 2027, 2028, 2029, 2030, 2031, 2032, 3029, 3030, 3031, 3032, 7, 42, 123}
    seeds = {(v, E.ctx_of(v).seed, E.ctx_of(v).restart_seed) for v in E.VARIANTS}
    assert seeds == {("f2", 2033, 3033), ("f1", 2034, 3034)}
    assert not {s for _, a, b in seeds for s in (a, b)} & used
    assert E.ctx_of("f2").protocol.name == "e2e_demo_v3_1_f2.json" and E.PRIVATE.name == "e2e_demo_v3_1"
    assert E.RECOVERY_RULES == E3.RECOVERY_RULES and E.SPECIFICITY_GATES == E3.SPECIFICITY_GATES
    assert E.registry_v3() == E3.registry_v3()
    assert {k: v for k, v in E.OUTCOMES.items() if k != "harness_failure"} == E3.OUTCOMES
    assert "harness_failure" in E.TERMINAL_OUTCOMES and "harness_failure" not in E3.OUTCOMES
    assert {"evals/e2e_demo_v3_1.py", "bench/demo_correction_v3_1.py"} <= E.CODE
    assert not {"evals/e2e_demo_v3.py", "bench/demo_correction_v3.py"} & E.CODE
    assert all((ROOT / name).is_file() for name in E.CODE)
    for variant in E.VARIANTS:
        for path in E.DECISION_PATHS:
            fixed = E.fixed(variant, path)
            assert json.loads(json.dumps(fixed)) == fixed and fixed["schema"] == "e2e_demo_protocol_v3_1"
            assert fixed["harness_failure_rule"] == E.HARNESS_FAILURE_RULE
            assert "approve-correction" in fixed["correction"]["approval"]
            assert {"recover", "budget_floor_checked_at", "terminal_guard"} <= set(fixed["correction"])


# --- 승인 뒤 막다른 경로 방지(예산 하한·종료 가드·고아 복구) ---
def test_budget_below_the_floor_is_refused_before_the_ledger(demo, monkeypatch, tmp_path):
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, "t40")
    for budget in (2399, 0, 2400.0):
        with pytest.raises(ValueError, match="하한"):
            demo.correction_propose("t40", ROOT, record_path, budget_gpu_s=budget)
    assert not (tmp_path / "loops" / ctx.cases["FAULT"] / "correction_ledger.jsonl").exists()


def test_ledger_writes_are_refused_after_the_demo_ended(demo, monkeypatch, tmp_path):
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, "t41")
    rid = demo.correction_propose("t41", ROOT, record_path, budget_gpu_s=2400)["request_id"]
    (demo.folder_for("t41") / "terminal.json").write_text(json.dumps({"outcome": "harness_failure"}), encoding="utf-8")
    for call in (lambda: demo.approve_correction("t41", rid, "operator"),
                 lambda: demo.reject_correction("t41", rid, "operator", "r"),
                 lambda: demo.correction_propose("t41", ROOT, record_path, budget_gpu_s=2400),
                 lambda: demo.correction_evidence("t41", ROOT)):
        with pytest.raises(ValueError, match="종료"):
            call()
    assert [e["event"] for e in demo.correction_ledger(ctx).events()] == ["proposed"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def orphaned(demo, monkeypatch, tmp_path, tag, result=None):
    """승인된 교정 실행이 끊겨(예외가 실행기 밖으로 나가) 고아 요청이 남은 시연. result가 있으면 그 결과 파일을 쓴 뒤 끊긴다."""
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, tag)
    rid = demo.correction_propose(tag, ROOT, record_path, budget_gpu_s=2400)["request_id"]
    demo.approve_correction(tag, rid, "operator")
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="고아"):  # 아직 실행하지 않은 요청은 복구 대상이 아니다
        demo.recover_correction(tag, isaaclab, rid, record_path)

    def crash(demo_tag, isaaclab):
        def run(request):
            if result is not None:
                write_json(DC.fix_folder(demo_tag) / "result.json", {
                    "schema": "e2e_correction_result_v3_1", "tag": request["tag"], "correction_request_id": request["request_id"],
                    "recovery_mode": request["recovery_mode"], "source_commit_sha": EXPERIMENT, **result})
            raise KeyboardInterrupt
        return run
    monkeypatch.setattr(DC, "make_executor", crash)
    with pytest.raises(KeyboardInterrupt):
        demo.correction(tag, isaaclab, rid, record_path)
    assert demo.correction_status(tag)["orphans"] == [rid] and not demo.step_path(tag, "correction").exists()
    return ctx, rid, record_path, isaaclab


@pytest.mark.parametrize("result", [None, {"label": "recovered", "costs": {"train": 900.0, "eval": 60.0}}])
def test_orphaned_correction_is_recovered_without_rerun(demo, monkeypatch, tmp_path, result):
    ctx, rid, record_path, isaaclab = orphaned(demo, monkeypatch, tmp_path, "t42", result)
    with pytest.raises(LedgerError, match="결과 없이"):  # 고아가 있으면 다른 실행을 막는다
        demo.run_correction(ctx, rid, fake_executor([])("t42", isaaclab), "t42_fix")
    step = demo.recover_correction("t42", isaaclab, rid, record_path)
    receipt = step["receipt"]
    assert step["recovered_orphan"] is True and demo.correction_status("t42")["orphans"] == []
    if result is None:
        assert receipt["verdict"] == "unknown" and receipt["reason"] == "orphan_without_artifact"
    else:
        assert receipt["verdict"] == "recovered" and receipt["gpu_s"] == 960.0 and receipt["recovered_from"].endswith(
            "fix/result.json") and receipt["result_sha"] == demo.sha(DC.fix_folder("t42") / "result.json")
    steps = {name: demo.step_record("t42", name) for name in demo.STEPS}
    assert V2.classify(steps, None, demo._correction_ledger_state(ctx)) == (
        "correction_unknown" if result is None else "recovered")


def test_recover_refuses_foreign_record_result_or_an_ended_demo(demo, monkeypatch, tmp_path):
    ctx, rid, record_path, isaaclab = orphaned(demo, monkeypatch, tmp_path, "t43", {"label": "recovered", "costs": {}})
    other = record_path.with_name("correction.call2.json")
    other.write_text(record_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="결합된"):  # 요청에 결합된 교정 호출 기록이 아니다
        demo.recover_correction("t43", isaaclab, rid, other)
    result_path = DC.fix_folder("t43") / "result.json"
    good = json.loads(result_path.read_text(encoding="utf-8"))
    for key, value in (("correction_request_id", "other"), ("tag", "x_fix"), ("source_commit_sha", "d" * 40),
                       ("recovery_mode", "resume"), ("schema", "e2e_correction_result_v3")):
        write_json(result_path, {**good, key: value})
        with pytest.raises(ValueError, match="결과 파일"):
            demo.recover_correction("t43", isaaclab, rid, record_path)
    write_json(result_path, good)
    (demo.folder_for("t43") / "terminal.json").write_text(json.dumps({"outcome": "harness_failure"}), encoding="utf-8")
    with pytest.raises(ValueError, match="종료"):
        demo.recover_correction("t43", isaaclab, rid, record_path)
    assert demo.correction_status("t43")["orphans"] == [rid]  # 거부된 복구는 원장을 바꾸지 않는다


# --- correct.py run과의 동등성·교정 실행기 모듈 ---
@pytest.mark.parametrize("outcome", ["ok", "exit_code_1", "over_budget", "no_gpu_s", "raises"])
def test_run_correction_matches_correct_run(demo, tmp_path, outcome):
    """같은 실행기 출력이면 v3.1 run_correction과 correct.py run의 원장 사건이 같다(등록 표 digest·요청 id만 다르다)."""
    ctx = demo.ctx_of("f2")
    twin = "c_default"
    make_loop(tmp_path / "loops", case=ctx.cases["FAULT"], mechanism="actuator")
    make_loop(tmp_path / "loops", case=twin, mechanism="actuator")

    def proposed(ledger, case, mode):
        prereg, state, _ = CR._loop_state(case, CR.LOOPS)
        rid = ledger.propose(CR.loop_state_digest(prereg, state), "actuator", "r", "restore_motor_torque_speed", mode,
                             R.protocol_case_arm("actuator"), 500, {})["request_id"]
        ledger.approve(rid, "operator")
        return rid
    ours = proposed(demo.correction_ledger(ctx), ctx.cases["FAULT"], "restart")
    theirs = proposed(CR._ledger(twin, CR.LOOPS), twin, "resume")  # 기본 표가 허용하는 방식으로 같은 흐름

    def executor(request):
        if outcome == "raises":
            raise RuntimeError("crash")
        out = {"exit_code": 1 if outcome == "exit_code_1" else 0, "verdict": "recovered", "result_sha": "f" * 64,
               "tag": request["tag"]}
        if outcome != "no_gpu_s":
            out["gpu_s"] = 900.0 if outcome == "over_budget" else 100.0
        return out
    results = []
    for run in (lambda: demo.run_correction(ctx, ours, executor, "x_fix"),
                lambda: CR.run(twin, theirs, executor, loops=CR.LOOPS, tag="x_fix")):
        try:
            results.append(run())
        except RuntimeError as error:
            results.append(type(error).__name__)
    events = [[{k: v for k, v in e.items() if k not in ("at", "request_id", "registry_digest", "recovery_mode",
                                                         "effect_key", "loop_state_digest", "gpu_s")}
               for e in led.events()] for led in (demo.correction_ledger(ctx), CR._ledger(twin, CR.LOOPS))]
    assert events[0] == events[1]
    if outcome == "raises":
        assert results == ["RuntimeError", "RuntimeError"]
        for again in (lambda: demo.run_correction(ctx, ours, executor, "x_fix"),
                      lambda: CR.run(twin, theirs, executor, loops=CR.LOOPS, tag="x_fix")):
            with pytest.raises(LedgerError):  # 남은 고아가 다음 실행을 막는다
                again()
    else:
        keys = ("event", "tag", "verdict", "result_sha", "exit_code", "over_budget")
        assert [{k: r[k] for k in keys} for r in results][0] == [{k: r[k] for k in keys} for r in results][1]
        assert isinstance(results[0]["gpu_s"], float) and isinstance(results[1]["gpu_s"], float)


def test_begin_sees_outputs_of_the_same_tag(demo, tmp_path):
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    ctx = demo.ctx_of("f2")
    assert demo.seed_unused_evidence(ctx, isaaclab, "t50")["existing_outputs"] == []
    (demo.RESULTS / "t50_spec").mkdir(parents=True)
    (tmp_path / "bench/private/e2e_demo_v3/t50").mkdir(parents=True)  # 같은 태그의 v3 시연
    assert len(demo.seed_unused_evidence(ctx, isaaclab, "t50")["existing_outputs"]) == 2


def test_correction_executor_is_bound_to_the_v3_1_demo():
    assert DC._demo() is E
    source = (ROOT / "bench/demo_correction_v3_1.py").read_text(encoding="utf-8")
    assert 'str(ROOT / "bench/demo_correction_v3_1.py")' in source  # Isaac 하위 프로세스도 v3.1 실행기를 부른다
    assert "import e2e_demo_v3_1 as E" in source and "import e2e_demo_v3 as E" not in source
    assert '"schema": "e2e_correction_result_v3_1"' in source


# --- 하네스 실패 ---
def cause_of_v3_stall():
    lines = (ROOT / "evals/e2e_demo_v3.py").read_text(encoding="utf-8").splitlines()
    return f"evals/e2e_demo_v3.py:{next(i + 1 for i, s in enumerate(lines) if 'CR._ledger(case, CR.LOOPS).propose' in s)}"


def committed_at_experiment(monkeypatch):
    """원인 파일이 시연 실험 커밋(EXPERIMENT)에 지금과 같은 내용으로 있다고 본다(git 없이)."""
    monkeypatch.setattr(E, "committed_blob_lf",
                        lambda commit, relative: E.sha(E.CODE_ROOT / relative, True) if commit == EXPERIMENT else None)
    monkeypatch.setattr(E, "tool_commit", lambda: "b" * 40)


@pytest.fixture
def stalled_v3(tmp_path, monkeypatch):
    """seed 2031과 같은 상태의 v3 시연(교정 제안에서 기본 표 원장 거부)과 v3.1 도구를 같은 임시 ROOT에 둔다."""
    v3 = use_tmp(E3, tmp_path, monkeypatch)
    use_tmp(E, tmp_path, monkeypatch)
    ctx, record_path = ready_for_correction(v3, monkeypatch, tmp_path, "t2031")
    with pytest.raises(LedgerError):
        v3.correction_propose("t2031", ROOT, record_path, budget_gpu_s=2400)
    committed_at_experiment(monkeypatch)
    return ctx


def test_harness_failure_records_the_v3_stall_once(stalled_v3, tmp_path, monkeypatch):
    stall = cause_of_v3_stall()
    bad = [("correction-propose", "e", "nofile"), ("correction-propose", "e", "../outside.py:1"),
           ("correction-propose", "e", "evals/e2e_demo_v3.py:999999"), ("specificity", "e", stall),
           ("loop_close", "e", stall), ("bogus", "e", stall), ("correction-propose", " ", stall),
           ("correction-propose", r"failed at C:\Users\someone\x.py", stall),
           ("correction-propose", "see /home/someone/x.py", stall),
           ("correction-propose", "e", "bench/private/e2e_demo_v3/t2031/steps/00_begin.json:1")]
    for step, error, cause in bad:
        with pytest.raises(ValueError):
            E.record_harness_failure("v3", "t2031", step, error, cause)
    with pytest.raises(ValueError, match="하네스 실패"):  # 종료 기록이 없는 시연은 요약하지 않는다
        E.harness_summary("v3", "t2031")
    error = "거부: actuator에 등록되지 않은 회복 방식 'restart'"
    monkeypatch.setattr(E, "committed_blob_lf", lambda commit, relative: "0" * 64)  # 실행된 코드와 다른 원인 파일
    with pytest.raises(ValueError, match="실험 커밋"):
        E.record_harness_failure("v3", "t2031", "correction-propose", error, stall)
    committed_at_experiment(monkeypatch)
    protocol = tmp_path / "protocols/e2e_demo_v3_f2.json"
    original = protocol.read_bytes()
    protocol.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="프로토콜"):  # begin에 묶인 시연 프로토콜과 달라지면 거부
        E.record_harness_failure("v3", "t2031", "correction-propose", error, stall)
    protocol.write_bytes(original)
    record = E.record_harness_failure("v3", "t2031", "correction-propose", error, stall)
    detail = record["detail"]
    assert record["outcome"] == "harness_failure" and detail["approval_consumed"] is False
    assert detail["correction_ledger"] == {"exists": False, "events": {}} and detail["case"] == stalled_v3.cases["FAULT"]
    assert "CR._ledger" in detail["cause"]["text"] and detail["cause"]["commit"] == EXPERIMENT
    assert detail["category_preregistered_for_demo"] is False and "사후 기록" in detail["recorded_by"]
    assert detail["recorded_steps"][-1] == "specificity" and detail["tool_commit"] == "b" * 40
    assert detail["probe_loop_status"] == "confirmed"
    assert record["wording"].endswith("v3 사전등록에 없던 범주를 v3.1 도구로 사후 기록했다.")
    with pytest.raises(FileExistsError, match="종료 기록이 이미"):
        E.record_harness_failure("v3", "t2031", "correction-propose", error, stall)
    with pytest.raises(ValueError, match="종료"):  # v3 실행기는 이 시연을 더 진행하지 않는다
        E3._guard("t2031", "correction")
    with pytest.raises(KeyError):  # v3 summarize는 이 범주를 모른다(그래서 harness-summary를 둔다)
        E3.summarize("t2031")
    summary = E.harness_summary("v3", "t2031")
    proposal = summary["correction_proposal"]
    assert proposal["accepted"] and proposal["recovery_mode"] == "restart" and proposal["agrees_with_rule"]
    assert summary["correction_judged"] is False and summary["prediction_results"]["specificity"] is True
    assert summary["prediction_results"]["correction"] is None and summary["seed"] == 2031
    text = (tmp_path / "evals/results/t2031/harness_failure_summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "isaaclab_root" not in text
    with pytest.raises(FileExistsError):
        E.harness_summary("v3", "t2031")


def test_harness_summary_needs_a_harness_failure_terminal(stalled_v3, tmp_path):
    folder = tmp_path / "bench/private/e2e_demo_v3/t2031"
    (folder / "terminal.json").write_text(json.dumps({"outcome": "loop_not_confirmed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="하네스 실패"):  # 다른 범주로 닫힌 시연은 요약하지 않는다
        E.harness_summary("v3", "t2031")
    with pytest.raises(FileExistsError, match="종료 기록이 이미"):
        E.record_harness_failure("v3", "t2031", "correction-propose", "e", cause_of_v3_stall())
    with pytest.raises(ValueError, match="v3.1 시연은 summarize"):
        E.harness_summary("v3_1", "t2031")
    with pytest.raises(ValueError, match="시연 폴더"):
        E.record_harness_failure("v3", "missing", "correction-propose", "e", cause_of_v3_stall())


def test_harness_cause_must_be_committed_code_inside_the_code_root(tmp_path, monkeypatch):
    code = tmp_path / "code"
    (code / "evals").mkdir(parents=True)
    (code / "evals/x.py").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "outside.py").write_text("a\n", encoding="utf-8")
    for hidden in ("bench/private/x.json", ".git/config"):  # 실험 커밋 대조를 통과해도 비공개·git 내부 파일은 거부
        (code / hidden).parent.mkdir(parents=True, exist_ok=True)
        (code / hidden).write_text("secret\n", encoding="utf-8")
    monkeypatch.setattr(E, "CODE_ROOT", code)
    committed_at_experiment(monkeypatch)
    assert E._cause("evals/x.py:2", EXPERIMENT)["text"] == "b"
    for cause, commit in (("../outside.py:1", EXPERIMENT), ("evals/x.py:3", EXPERIMENT), ("evals/x.py:0", EXPERIMENT),
                          ("evals/x.py", EXPERIMENT), ("evals/y.py:1", EXPERIMENT), ("evals/x.py:2", "c" * 40),
                          (".git/config:1", EXPERIMENT), ("bench/private/x.json:1", EXPERIMENT)):
        with pytest.raises(ValueError):
            E._cause(cause, commit)


@pytest.mark.parametrize("event", ["approved", "rejected", "held"])
def test_harness_failure_refuses_registered_outcomes(demo, monkeypatch, tmp_path, event):
    committed_at_experiment(monkeypatch)
    ctx, record_path = ready_for_correction(demo, monkeypatch, tmp_path, "t33")
    if event == "held":
        prereg, state, _ = CR._loop_state(ctx.cases["FAULT"], CR.LOOPS)
        demo.correction_ledger(ctx).hold(CR.loop_state_digest(prereg, state), "actuator", "test hold")
    else:
        rid = demo.correction_propose("t33", ROOT, record_path, budget_gpu_s=2400)["request_id"]
        if event == "approved":
            demo.approve_correction("t33", rid, "operator")
        else:
            demo.reject_correction("t33", rid, "operator", "r")
    with pytest.raises(ValueError, match="승인·거절"):
        E.record_harness_failure("v3_1", "t33", "correct-run", "e", cause_of_v3_stall())


def test_harness_failure_on_a_v3_1_demo_is_preregistered(demo, monkeypatch, tmp_path):
    committed_at_experiment(monkeypatch)
    ready_for_correction(demo, monkeypatch, tmp_path, "t34")
    record = E.record_harness_failure("v3_1", "t34", "correction-propose", "e", "evals/e2e_demo_v3_1.py:1")
    assert record["detail"]["category_preregistered_for_demo"] is True and record["wording"] == E.OUTCOMES["harness_failure"]
    assert "사후" not in record["detail"]["recorded_by"]
    assert E.summarize("t34")["outcome"] == "harness_failure"  # v3.1 summarize는 이 범주를 안다


def test_harness_failure_during_the_loop_allows_only_loop_close(demo, monkeypatch, tmp_path):
    committed_at_experiment(monkeypatch)
    ctx = demo.ctx_of("f2")
    demo._write_step("t35", "begin", demo._guard("t35", "begin"), EXPERIMENT, ctx.protocol,
                     {"variant": "f2", "decision_path": "single", "seed_unused": {"hits": []}})
    for name in demo.STEPS[1:demo.STEPS.index("loop_close")]:  # 고리를 시작했고 아직 닫지 않았다
        demo._write_step("t35", name, demo._guard("t35", name), EXPERIMENT, ctx.protocol, {})
    for step in ("specificity", "correction-propose", "loop_start", "correction"):
        with pytest.raises(ValueError, match="바로 다음"):
            E.record_harness_failure("v3_1", "t35", step, "e", "evals/e2e_demo_v3_1.py:1")
    record = E.record_harness_failure("v3_1", "t35", "loop_close", "e", "evals/e2e_demo_v3_1.py:1")
    assert record["detail"]["recorded_steps"][-1] == "loop_start" and record["detail"]["probe_loop_status"] is None


def test_error_path_pattern_and_commit_format(monkeypatch):
    cases = {r"C:\Users\x\a.py": True, "c:/Users/x": True, r"\\wsl$\Ubuntu\home": True, "see /c/Users/x": True,
             "(/tmp/x)": True, "~/x.py": True, "path=/home/u/x": True, r"~\x.py": True,
             "https://github.com/a/b": False, "거부: actuator에 등록되지 않은 회복 방식 'restart'": False,
             "evals/e2e_demo_v3.py:694 failed": False, "ratio 0.5/1.0": False, "key:/value": False}
    assert {s: bool(E.ABSOLUTE_PATH.search(s)) for s in cases} == cases
    def no_git(*args, **kwargs):
        raise AssertionError("형식이 틀린 커밋으로 git을 부르면 안 된다(옵션으로 해석될 수 있다)")
    monkeypatch.setattr(E.subprocess, "run", no_git)
    for commit in ("not-a-commit", "--output=x", "A" * 40, None):
        assert E.committed_blob_lf(commit, "evals/e2e_demo_v3.py") is None
