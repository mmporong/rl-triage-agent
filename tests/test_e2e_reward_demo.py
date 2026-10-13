"""E4 새 seed 보상 결함 종단 시연: R1·사전등록·단계 사슬·진단 기록 검증·고리 시작·판정·restart 실행기. GPU·모델 없음."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import e2e_reward_demo as E  # noqa: E402
import demo_restart_correction as DR  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402

P0C = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
REF_MEASURE = {"P_value": {"critic_change": 0.5}, "P_torque": {"low_speed_saturation": 0.002},
               "P_physics": {"mismatch_count": 0}}


def telemetry(name):
    return json.loads((ROOT / "bench/runs_p0c" / f"{name}.telemetry.json").read_text(encoding="utf-8"))["series"][A.METRIC]


def report(name):
    return json.loads((P0C / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """시연 출력·고리·결과 경로를 임시 폴더로 돌린다."""
    loop = importlib.import_module("loop")
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(E, "PROTOCOL", protocol)
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    return E


# --- R1 ---
@pytest.mark.parametrize("seed,first", [(2026, 135), (2027, 134), (2028, 135)])
def test_r1_reproduces_leave_seed_out_analysis(seed, first):
    refs = [telemetry(f"baseline_p0c_s{s}") for s in (2026, 2027, 2028) if s != seed]
    assert A.r1(telemetry(f"h02_s{seed}"), refs)["first_alert_iteration"] == first
    assert not A.r1(telemetry(f"baseline_p0c_s{seed}"), refs)["warned"]


def test_r1_window_eligibility_persistence_and_rearm():
    ref = [1.0] * 120
    run = [1.0] * 40 + [5.0] * 30 + [1.0] * 25 + [5.0] * 25
    out = A.r1(run, [ref])
    # 59 이전은 세지 않는다. 20회 평균이 2를 넘은 상태가 10회 연속인 지점마다 이벤트, 끊긴 뒤 다시 채우면 새 이벤트.
    assert out["alert_iterations"] and all(t >= 60 + 9 for t in out["alert_iterations"])
    assert len(out["alert_iterations"]) == 2
    assert A.smooth([1, 2, 3], 2) == [None, 1.5, 2.5]
    with pytest.raises(ValueError):
        A.r1([1.0] * 10, [[1.0] * 9])
    with pytest.raises(ValueError):
        A.r1([1.0, float("nan")], [[1.0, 1.0]])


# --- 사전등록 ---
def test_protocol_has_all_outcomes_conditions_and_detects_changes():
    assert set(E.FIXED["success_conditions"]) == set(E.SUCCESS_CONDITIONS)
    assert E.TERMINAL_OUTCOMES < set(E.OUTCOMES) and {"recovered", "not_recovered"} <= set(E.OUTCOMES)
    assert all(word not in json.dumps(E.FIXED, ensure_ascii=False) for word in ("직접",))
    dynamic = {k: f"v_{k}" for k in E.DYNAMIC}
    good = {**E.FIXED, **dynamic}
    E.check_protocol(good, dynamic)
    with pytest.raises(ValueError):
        E.check_protocol({**good, "seed": 2027}, dynamic)
    with pytest.raises(ValueError):
        E.check_protocol(good, {**dynamic, "source_files": "changed"})
    with pytest.raises(ValueError):
        E.check_protocol({**good, "extra": 1}, dynamic)


def test_code_set_covers_reused_and_new_files():
    for name in ("evals/e2e_reward_demo.py", "bench/demo_restart_correction.py", "src/rl_triage/tracking_alarm.py",
                 "evals/loop.py", "evals/active_model_call.py", "evals/correction_model_call.py",
                 "bench/train_with_recovery_mode.py", "evals/fixed_eval.py", "bench/hidden_faults.py"):
        assert name in E.CODE and (ROOT / name).is_file()


# --- 단계 사슬 ---
def test_steps_are_ordered_write_once_chained_and_stop_at_terminal(demo):
    tag = "t1"
    with pytest.raises(ValueError):
        demo._guard(tag, "train_NONE")  # begin이 먼저
    prev = demo._guard(tag, "begin")
    demo._write_step(tag, "begin", prev, "c" * 40, {"seed_unused": {"hits": []}})
    with pytest.raises(FileExistsError):
        demo._guard(tag, "begin")
    prev = demo._guard(tag, "train_NONE")
    demo._write_step(tag, "train_NONE", prev, "c" * 40, {})
    begin = demo.step_path(tag, "begin")
    begin.write_text(begin.read_text(encoding="utf-8").replace("[]", "[\"x\"]"), encoding="utf-8")
    with pytest.raises(ValueError):  # 앞 단계를 고치면 사슬이 끊긴다
        demo._guard(tag, "train_F1")
    demo._terminal(tag, "baseline_invalid", {})
    with pytest.raises(ValueError):
        demo._guard(tag, "train_F1")
    with pytest.raises(ValueError):
        demo._terminal("t2", "recovered", {})  # 결과 범주는 조기 종료로 쓸 수 없다
    with pytest.raises(ValueError):
        demo.folder_for("../escape")


# --- 고정 평가 게이트 ---
def test_source_gate_uses_same_seed_none_and_p0c_bands():
    bands = json.loads((ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json").read_text(encoding="utf-8"))["bands"]
    gate = E.source_gate(report("baseline_p0c_s2027"), report("h02_s2027"), bands)
    assert gate["outcome"] == "failure_present" and gate["none_walks"]
    assert gate["standstill_ratio"]["F1"] >= 0.9 > gate["standstill_ratio"]["NONE"]
    assert E.source_gate(report("baseline_p0c_s2027"), report("baseline_p0c_s2027"), bands)["outcome"] == "source_failure_absent"
    assert E.source_gate(report("h02_s2027"), report("h02_s2027"), bands)["outcome"] == "baseline_invalid"


# --- 진단 기록 ---
def _packet(tmp_path):
    import run_active_diagnosis as RA
    src = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
           "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    packet = RA.input_packet({"sources": src}, {"max_probes": E.LOOP_BUDGET,
                                                "probe_execution_budget_s": E.PROBE_EXECUTION_BUDGET_S})
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return packet, path


def _decision(packet, probe="P_reward", action="continue"):
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    fact = sorted(packet["facts"])[0]
    return {"ranking": ranking, "probe": probe if action == "continue" else None, "action": action,
            "reason": "tracking error stays high while episodes survive",
            "claims": [{"id": fact, "value": packet["facts"][fact]}],
            "expected_outcomes": ({m: L.PROBES[probe]["expect"][m] for m in ranking[:3]} if action == "continue" else {})}


def _record(path, content, **kw):
    from rl_triage import active_diagnosis as D
    packet = json.loads(path.read_bytes())
    return {"schema": "active_model_call_v1", "packet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "messages_sha256": L.digest(D.decision_messages(packet)), "request": {"model": E.MODEL, "temperature": 0},
            "exit_code": 0, "content": content, "response_id": "r", "input_tokens": 10, "output_tokens": 5,
            "execution_wall_s": 1.0, **kw}


def test_review_diagnosis_record(tmp_path):
    packet, path = _packet(tmp_path)
    ok = E.review_diagnosis_record(_record(path, json.dumps(_decision(packet))), path)
    assert ok["status"] == "valid" and ok["decision"]["probe"] == "P_reward"
    fenced = "```json\n" + json.dumps(_decision(packet)) + "\n```"
    assert E.review_diagnosis_record(_record(path, fenced), path)["status"] == "valid"
    assert E.review_diagnosis_record(_record(path, json.dumps(_decision(packet, action="hold"))), path)["decision"]["action"] == "hold"
    assert E.review_diagnosis_record(_record(path, None, exit_code=2, infra_error="APITimeoutError"), path)["status"] == "infra_error"
    assert E.review_diagnosis_record(_record(path, None, exit_code=2, error_type="KeyError"), path)["status"] == "call_error"
    assert E.review_diagnosis_record(_record(path, "{broken"), path)["status"] == "invalid"
    bad = _decision(packet)
    bad["claims"] = [{"id": "telemetry:case:invented", "value": 1}]
    assert E.review_diagnosis_record(_record(path, json.dumps(bad)), path)["status"] == "invalid"
    for bad_record in ({**_record(path, "{}"), "packet_sha256": "0" * 64},
                       {**_record(path, "{}"), "messages_sha256": "0" * 64},       # 다른 프롬프트
                       {**_record(path, "{}"), "request": {"model": "other", "temperature": 0}},
                       {**_record(path, "{}"), "request": {"model": E.MODEL, "temperature": 0.7}},
                       {**_record(path, "{}"), "worker_cleanup_failed": True}):
        with pytest.raises(ValueError):
            E.review_diagnosis_record(bad_record, path)


def test_loop_start_enforces_record_order_and_starts_with_model_probe(demo, tmp_path, monkeypatch):
    packet, path = _packet(tmp_path)
    tag = "t4"
    model = demo.folder_for(tag) / "model"
    model.mkdir(parents=True)
    (model / "diagnosis.packet.json").write_bytes(path.read_bytes())
    steps = {"packet": {"packet": {"path": "packet", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
             "reference": {"report": {"path": "ref"}}}
    real_root = demo.ROOT
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    (tmp_path / "packet").write_bytes(path.read_bytes())
    (tmp_path / "ref").write_text(json.dumps({"probes": {p: {"measurement": m} for p, m in REF_MEASURE.items()}}),
                                  encoding="utf-8")
    monkeypatch.setattr(demo, "_guard", lambda tag, name: None)
    monkeypatch.setattr(demo, "verify_protocol", lambda isaaclab: ({}, "c" * 40))
    monkeypatch.setattr(demo, "step_record", lambda tag, name: steps[name])
    monkeypatch.setattr(demo, "_check_artifacts", lambda tag, arm: {"checkpoint": {"sha256": "a" * 64}})
    monkeypatch.setattr(demo, "_rel", lambda p: Path(p).name)
    monkeypatch.setattr(demo, "_write_step", lambda tag, name, prev, commit, record: {"step": name, **record})
    call2 = model / "diagnosis.call2.json"
    call2.write_text(json.dumps(_record(path, json.dumps(_decision(packet)))), encoding="utf-8")
    with pytest.raises(ValueError):  # 첫 기록은 call1이어야 한다
        demo.loop_start(tag, tmp_path, call2)
    call1 = model / "diagnosis.call1.json"
    call1.write_text(json.dumps(_record(path, json.dumps(_decision(packet)))), encoding="utf-8")
    with pytest.raises(ValueError):  # 제출하지 않은 call2가 있다
        demo.loop_start(tag, tmp_path, call1)
    call2.unlink()
    stray = model / "rerun.json"
    stray.write_text(call1.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError):  # 다른 이름으로 저장한 진단 호출 기록
        demo.loop_start(tag, tmp_path, call1)
    stray.unlink()
    out = demo.loop_start(tag, tmp_path, call1)
    assert out["started"]["proposed_probe"] == "P_reward" and out["started"]["model_probe_used"]
    assert any(f"--tag {tag}_loop --metered" in c for c in out["next_commands"])
    assert (real_root / "evals").is_dir()


def test_loop_start_infra_retry_then_terminal(demo, tmp_path, monkeypatch):
    packet, path = _packet(tmp_path)
    tag = "t5"
    model = demo.folder_for(tag) / "model"
    model.mkdir(parents=True)
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    (tmp_path / "packet").write_bytes(path.read_bytes())
    monkeypatch.setattr(demo, "_guard", lambda tag, name: None)
    monkeypatch.setattr(demo, "verify_protocol", lambda isaaclab: ({}, "c" * 40))
    monkeypatch.setattr(demo, "step_record", lambda tag, name: {"packet": {
        "path": "packet", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
    monkeypatch.setattr(demo, "_rel", lambda p: Path(p).name)
    failed = json.dumps(_record(path, None, exit_code=2, infra_error="APITimeoutError"))
    (model / "diagnosis.call1.json").write_text(failed, encoding="utf-8")
    assert demo.loop_start(tag, tmp_path, model / "diagnosis.call1.json")["status"] == "retry_allowed"
    (model / "diagnosis.call2.json").write_text(failed, encoding="utf-8")  # 재시도는 다음 번호로
    assert demo.loop_start(tag, tmp_path, model / "diagnosis.call2.json")["outcome"] == "model_infra_failed"


# --- 고리 시작 ---
def test_start_loop_uses_model_probe_and_matches_loop_start_files(demo, tmp_path):
    loop = importlib.import_module("loop")
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    out = demo.start_loop("h02_s2029", ranking, "a" * 64, REF_MEASURE, "P_reward")
    assert out["model_probe_used"] and out["proposed_probe"] == "P_reward" and out["rule_probe"] == "P_noise"
    loop.start("h02_s2030", ranking, 3, 4, "a" * 64, REF_MEASURE)
    for name in ("prereg.json", "reference.json"):
        a = json.loads((loop.LOOPS / "h02_s2029" / name).read_text(encoding="utf-8"))
        b = json.loads((loop.LOOPS / "h02_s2030" / name).read_text(encoding="utf-8"))
        a.pop("case", None), b.pop("case", None)
        assert a == b
    _, state, led = loop._load("h02_s2029")
    assert state["pending"] == out["request_id"] and led.requests()[out["request_id"]]["probe"] == "P_reward"
    fallback = demo.start_loop("h05_s2029", ranking, "a" * 64, REF_MEASURE, "P_physics")  # 상위 3을 가르지 못한다
    assert not fallback["model_probe_used"] and fallback["proposed_probe"] == "P_noise"
    with pytest.raises(FileExistsError):
        demo.start_loop("h02_s2029", ranking, "a" * 64, REF_MEASURE, "P_reward")


def test_loop_summary_checks_artifacts_against_checkpoint(demo, monkeypatch):
    loop = importlib.import_module("loop")
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    rid = demo.start_loop("h02_s2029", ranking, "a" * 64, REF_MEASURE, "P_reward")["request_id"]
    with pytest.raises(ValueError):  # 아직 대기 중
        demo.loop_summary("t", "a" * 64)
    _, _, led = loop._load("h02_s2029")
    led.approve(rid, "operator")
    measurement = {"track_lin_vel_xy_exp_rel_error": 0.25, "track_ang_vel_z_exp_rel_error": 0.0}
    loop.run("h02_s2029", rid, lambda c, p: (measurement, 18.0, 0))
    probe_file = demo.RESULTS / "t_loop/probes/h02_s2029__P_reward.json"
    probe_file.parent.mkdir(parents=True)
    probe_file.write_text(json.dumps({"name": "h02_s2029", "checkpoint": {"sha256": "a" * 64}}), encoding="utf-8")
    meta = demo.RESULTS / "t_loop/execution/h02_s2029__P_reward.json"
    meta.parent.mkdir(parents=True)
    meta_body = {"checkpoint_sha256": "a" * 64, "report_sha256_lf": E.sha(probe_file, True),
                 "gpu_wall_s": 18.0, "cpu_s": 20.0, "simulator_steps": 51200, "metered": False}
    meta.write_text(json.dumps(meta_body), encoding="utf-8")
    monkeypatch.setattr(demo, "_rel", lambda p: Path(p).name)  # 임시 폴더는 저장소 밖이다
    with pytest.raises(ValueError, match="metered"):  # 등록과 달리 계측 없이 실행
        demo.loop_summary("t", "a" * 64)
    meta.write_text(json.dumps({**meta_body, "metered": True}), encoding="utf-8")
    summary = demo.loop_summary("t", "a" * 64)
    assert summary["status"] == "confirmed" and summary["hypotheses"] == ["reward"]
    assert summary["requests"][0]["approver"] == "operator" and summary["probe_costs"][0]["simulator_steps"] == 51200
    with pytest.raises(ValueError):
        demo.loop_summary("t", "b" * 64)


# --- 판정 ---
def _steps(**overrides):
    call = {"response_id": "r", "execution_wall_s": 2.0, "input_tokens": 100, "output_tokens": 50}
    steps = {
        "begin": {"protocol_sha256_lf": "p", "source_commit_sha": "c", "seed_unused": {"hits": []}},
        **{name: {"protocol_sha256_lf": "p", "source_commit_sha": "c", "execution": {"parent_wall_s": 10.0}}
           for name in ("train_NONE", "train_F1", "reference")},
        "evaluate": {"protocol_sha256_lf": "p", "source_commit_sha": "c", "execution": {"parent_wall_s": 80.0},
                     "gate": {"outcome": "failure_present"}},
        "loop_start": {"protocol_sha256_lf": "p", "source_commit_sha": "c", "call": call,
                       "decision": {"action": "continue"}, "started": {"model_probe_used": True}},
        "loop_close": {"protocol_sha256_lf": "p", "source_commit_sha": "c", "proceed_to_correction": True,
                       "loop": {"requests": [{"state": "done", "approver": "op", "artifact": {"probe": "x"},
                                              "outcome": "abnormal", "gpu_s": 18.0}]}},
        "correction": {"protocol_sha256_lf": "p", "source_commit_sha": "c", "call": call,
                       "model_proposal": {"accepted": True}, "receipt": {"event": "receipt", "verdict": "recovered"},
                       "result": {"costs": {"train_parent_wall_s": 500.0, "eval_parent_wall_s": 50.0}}},
    }
    steps.update(overrides)
    return steps


def test_classify_and_success_conditions():
    empty = {"requests": {}, "holds": []}
    steps = _steps()
    assert E.classify(steps, None, empty) == "recovered"
    assert all(E.conditions(steps, "recovered").values())
    fallback = _steps(loop_start={**steps["loop_start"], "started": {"model_probe_used": False}})
    checks = E.conditions(fallback, "recovered")
    assert not checks["first_probe_from_model"] and sum(not v for v in checks.values()) == 1
    not_rec = _steps(correction={**steps["correction"], "receipt": {"event": "receipt", "verdict": "not_recovered"}})
    assert E.classify(not_rec, None, empty) == "not_recovered"
    unknown = _steps(correction={**steps["correction"], "receipt": {"event": "receipt", "verdict": "unknown"}})
    assert E.classify(unknown, None, empty) == "correction_unknown"
    partial = {k: v for k, v in steps.items() if k != "correction"}
    assert E.classify(partial, None, {"requests": {"r": {"state": "rejected"}}, "holds": []}) == "correction_rejected"
    assert E.classify(partial, None, {"requests": {}, "holds": [{"event": "held"}]}) == "correction_not_proposed"
    assert E.classify(partial, None, empty) == "in_progress"
    assert E.classify(steps, {"outcome": "diagnosis_hold"}, empty) == "diagnosis_hold"
    mixed = _steps(alarm={"protocol_sha256_lf": "other", "source_commit_sha": "c"})
    assert not E.conditions(mixed, "recovered")["protocol_committed"]


def test_summarize_writes_public_and_private_once(demo, monkeypatch):
    tag = "t3"
    prev = demo._guard(tag, "begin")
    demo._write_step(tag, "begin", prev, "c" * 40, {"seed_unused": {"hits": []}, "isaaclab_root": "C:/secret"})
    demo._terminal(tag, "baseline_invalid", {"gate": {"outcome": "baseline_invalid"}})
    out = demo.summarize(tag)
    assert out["outcome"] == "baseline_invalid" and out["demo_success"] is False
    public = (demo.RESULTS / tag / "demo_result.json").read_text(encoding="utf-8")
    assert "C:/secret" not in public and out["wording"].startswith(E.OUTCOMES["baseline_invalid"])
    assert demo.summarize(tag) == out  # 두 번째 호출은 기록을 다시 읽는다


# --- restart 실행기 ---
def test_restart_manifest_satisfies_recovery_mode_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(DR, "source_run", lambda demo_tag: {"environment": {"path": "env.yaml"},
                                                            "agent": {"path": "agent.yaml"}})
    request = {"demo_tag": "d", "tag": DR.fix_tag("d"), "source_commit_sha": "c" * 40, "protocol_sha256_lf": "p" * 64,
               "isaaclab_root": str(tmp_path)}
    manifest = DR.manifest_for(tmp_path / "fix", request)
    assert set(manifest) == R.MANIFEST_FIELDS and R.validate_manifest_mode(manifest) == "restart"
    assert manifest["final_iteration"] == 299 and manifest["run_name"] == "d_fix_restart_corrected"
    assert manifest["seed"] == DR.RESTART_SEED == E.RESTART_SEED == E.FIXED["restart_seed"] != E.SEED
    with pytest.raises(ValueError):
        DR.fix_tag("Bad Tag")


def test_restart_training_configuration_allows_only_run_name(monkeypatch, tmp_path):
    src = tmp_path / "src"
    (src / "params").mkdir(parents=True)
    env = "seed: 2029\nscene:\n  num_envs: 4096\n"
    agent = "seed: 2029\nnum_steps_per_env: 24\nmax_iterations: 300\nrun_name: h02_s2029\n"
    (src / "params/env.yaml").write_text(env, encoding="utf-8")
    (src / "params/agent.yaml").write_text(agent, encoding="utf-8")
    artifacts = {k: {"path": str(src / "params" / f), "sha256": E.sha(src / "params" / f)}
                 for k, f in (("environment", "env.yaml"), ("agent", "agent.yaml"))}
    monkeypatch.setattr(DR, "source_run", lambda demo_tag: artifacts)
    manifest = {"tag": DR.fix_tag("d"), "run_name": "d_fix_restart_corrected", "seed": DR.RESTART_SEED,
                "source_env_config_path": artifacts["environment"]["path"],
                "source_agent_config_path": artifacts["agent"]["path"]}
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    restart_agent = agent.replace("h02_s2029", "d_fix_restart_corrected").replace("seed: 2029", "seed: 3029")
    (run / "params/env.yaml").write_text(env.replace("seed: 2029", "seed: 3029"), encoding="utf-8")
    (run / "params/agent.yaml").write_text(restart_agent, encoding="utf-8")
    DR.validate_training_configuration(run, manifest)
    (run / "params/agent.yaml").write_text(restart_agent.replace("max_iterations: 300", "max_iterations: 600"),
                                           encoding="utf-8")
    with pytest.raises(ValueError):  # seed·run_name 밖의 변경
        DR.validate_training_configuration(run, manifest)
    (run / "params/agent.yaml").write_text(agent.replace("h02_s2029", "d_fix_restart_corrected"), encoding="utf-8")
    (run / "params/env.yaml").write_text(env, encoding="utf-8")
    with pytest.raises(ValueError):  # 실패 seed와 같은 seed로 재학습하면 정상 학습 재현이라 거부
        DR.validate_training_configuration(run, manifest)


def test_executor_returns_unknown_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(DR, "PRIVATE", tmp_path / "private")

    def boom(request, demo_tag, isaaclab):
        raise RuntimeError("GPU busy before restart training")

    monkeypatch.setattr(DR, "_run", boom)
    out = DR.make_executor("d", tmp_path)({"request_id": "r"})
    assert out == {"exit_code": 1, "gpu_s": None, "verdict": "unknown", "result_sha": None, "tag": "d_fix"}
    assert json.loads((DR.fix_folder("d") / "failure.json").read_text(encoding="utf-8"))["error_type"] == "RuntimeError"


def test_restart_judge_against_same_seed_none():
    bands = json.loads((ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json").read_text(encoding="utf-8"))["bands"]
    assert DR.judge(report("baseline_p0c_s2027"), report("baseline_p0c_s2027"), bands)["label"] == "recovered"
    bad = DR.judge(report("h02_s2027"), report("baseline_p0c_s2027"), bands)
    assert bad["label"] == "not_recovered" and bad["standstill_ratio"] >= 0.9


# --- GPU 단계 실행·실패 처리(가짜 소유 실행기) ---
def _owned(code=0, natural=True, calls=None):
    def run(folder, argv, cwd, watchdog_s):
        if calls is not None:
            calls.append(argv)
        return {"job_assigned_before_release": True, "natural_exit": natural, "timed_out": False, "error_type": None,
                "exit_code": code, "cleanup_verified": True, "owned_process_identities": {}, "wall_s": 12.3}
    return run


def _gpu_ready(demo, monkeypatch):
    monkeypatch.setattr(demo, "_guard", lambda tag, name: None)
    monkeypatch.setattr(demo, "verify_protocol", lambda isaaclab: ({"gpu_admission_profile": "x", "bands": {}}, "c" * 40))
    monkeypatch.setattr(demo, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(demo, "_rel", lambda p: Path(p).name)


def test_train_uses_owned_runner_and_failure_ends_demo(demo, monkeypatch, tmp_path):
    _gpu_ready(demo, monkeypatch)
    monkeypatch.setattr(demo, "training_artifacts", lambda arm, isaaclab: {"run": arm})
    monkeypatch.setattr(demo, "_write_step", lambda tag, name, prev, commit, record: {"step": name, **record})
    calls = []
    out = demo.train("t6", "NONE", tmp_path / "IsaacLab", owned=_owned(calls=calls))
    argv = calls[0]
    assert argv[:3] == ["pwsh", "-NoProfile", "-File"] and argv[argv.index("-Fault") + 1] == "NONE"
    assert argv[argv.index("-Seed") + 1] == "2029" and argv[argv.index("-CaseId") + 1] == "baseline_p0c"
    assert out["execution"]["attempt"] == 1 and out["execution"]["parent_wall_s"] == 12.3
    with pytest.raises(RuntimeError):
        demo.train("t6", "F1", tmp_path / "IsaacLab", owned=_owned(code=1))
    assert json.loads((demo.folder_for("t6") / "terminal.json").read_text(encoding="utf-8"))["outcome"] == "invalid"


def test_evaluate_allows_one_infra_retry_without_outputs(demo, monkeypatch, tmp_path):
    _gpu_ready(demo, monkeypatch)
    monkeypatch.setattr(demo, "_check_artifacts", lambda tag, arm: {"checkpoint": {"sha256": "a" * 64}})
    calls = []
    with pytest.raises(RuntimeError, match="재시도"):
        demo.evaluate("t7", tmp_path / "IsaacLab", owned=_owned(code=1, natural=False, calls=calls))
    assert "--isaaclab" in calls[0] and not (demo.folder_for("t7") / "terminal.json").exists()
    with pytest.raises(RuntimeError):
        demo.evaluate("t7", tmp_path / "IsaacLab", owned=_owned(code=1, natural=False))
    terminal = json.loads((demo.folder_for("t7") / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["outcome"] == "invalid" and terminal["detail"]["attempt"] == 2


def test_seed_unused_detects_names_and_logged_restart_seed(demo, tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # 실제 저장소에는 seed 2029 시연 산출물이 있으므로 검사 경로를 임시 폴더로 돌린다
    for folder in (demo.RUNS_DIR, "bench/private", "evals/loops"):
        (repo / folder).mkdir(parents=True)
    monkeypatch.setattr(demo, "ROOT", repo)
    isaaclab = tmp_path / "IsaacLab"
    run = isaaclab / "logs/rsl_rl/unitree_go2_flat/2026-10-14_00-00-00_other_run"
    (run / "params").mkdir(parents=True)
    (run / "params/agent.yaml").write_text("seed: 42\n", encoding="utf-8")
    assert not demo.seed_unused_evidence(isaaclab, "t8")["hits"]
    (run / "params/agent.yaml").write_text("seed: 3029\nrun_name: x\n", encoding="utf-8")
    assert any("seed 3029" in hit for hit in demo.seed_unused_evidence(isaaclab, "t8")["hits"])
    (run / "params/agent.yaml").write_text("seed: 42\n", encoding="utf-8")
    (repo / demo.RUNS_DIR / f"h02_s{demo.SEED}.telemetry.json").write_text("{}", encoding="utf-8")
    assert [Path(hit).name for hit in demo.seed_unused_evidence(isaaclab, "t8")["hits"]] == [
        f"h02_s{demo.SEED}.telemetry.json"]


def test_prediction_results_follow_reached_steps():
    steps = _steps(alarm={"protocol_sha256_lf": "p", "source_commit_sha": "c",
                          "results": {"F1": {"warned": True}, "NONE": {"warned": False}}})
    steps["evaluate"]["gate"] = {"outcome": "failure_present", "standstill_ratio": {"F1": 1.01, "NONE": 0.12}}
    steps["loop_start"]["decision"] = {"action": "continue", "ranking": ["reward", "actuator", "exploration"]}
    steps["loop_close"]["loop"]["observed"] = {"P_reward": "abnormal"}
    steps["correction"]["rule"] = {"rule_mode": "restart"}
    assert all(E.prediction_results(steps).values())
    early = {k: v for k, v in steps.items() if k in ("begin", "train_NONE")}
    assert set(E.prediction_results(early).values()) == {None}


# --- restart 학습 검증: v4 restart 분기와 같은 거부 ---
def _restart_artifacts(tmp_path, monkeypatch):
    folder = tmp_path / "fix"
    training = folder / "train"
    training.mkdir(parents=True)
    run_name = "d_fix_restart_corrected"
    isaaclab = tmp_path / "IsaacLab"
    run_dir = isaaclab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-14_00-00-00_{run_name}"
    (run_dir / "params").mkdir(parents=True)
    for name, text in (("model_299.pt", "ckpt"), ("params/env.yaml", "env"), ("params/agent.yaml", "agent")):
        (run_dir / name).write_text(text, encoding="utf-8")
    manifest = {"tag": "d_fix", "seed": DR.RESTART_SEED, "arm": DR.ARM, "recovery_mode": "restart", "fault": "F1",
                "correction_id": DR.CORRECTION, "source_commit_sha": "c" * 40, "protocol_sha256_lf": "p" * 64,
                "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "run_name": run_name, "final_iteration": 299,
                "isaaclab_root": str(isaaclab)}
    for name, value in (("manifest.json", manifest), ("correction.json", {"before_fault": {}})):
        (training / name).write_text(json.dumps(value), encoding="utf-8")
    rng = {"cpu_sha256": "1" * 64, "cuda_all_sha256": ["2" * 64], "cuda_device_count": 1}
    observed = {"environment_seed": DR.RESTART_SEED, "environment_num_envs": 4096, "action_scale": 0.25,
                "agent_seed": DR.RESTART_SEED, "agent_max_iterations": 300, "runner_num_steps_per_env": 24}
    versions = {"python": "3.10.15", "python_cache_tag": "cpython-310", "rsl_rl_distribution": "2.3.3",
                "torch_distribution": "2.7.0+cu128", "isaaclab_distribution": "0.41.3", "isaaclab_source_commit": "i"}
    initial = {"schema": "training_correction_initial_v2", "recovery_mode": "restart", "resume_checkpoint": None,
               "resume_iteration": None, "source_checkpoint_state": None, "std_reset": None, "start_iteration": 0,
               "init_at_random_ep_len": True, "full_environment_state_restored": False,
               "rng_state_restored_from_checkpoint": False, "rng_state_before_learn": rng, "observed": observed,
               "runtime_versions": versions, "log_dir": str(run_dir)}
    (training / "initial.json").write_text(json.dumps(initial), encoding="utf-8")

    def artifact(path):
        return {"path": str(path), "sha256": DR.sha(path)}

    receipt = {**{k: manifest[k] for k in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id",
                                           "source_commit_sha", "protocol_sha256_lf", "task", "run_name",
                                           "final_iteration")},
               "completed": True, "schema": "training_correction_receipt_v2",
               **{f"{n}_sha256": DR.sha(training / f"{n}.json") for n in ("manifest", "initial", "correction")},
               "process_identity": {"process_id": 7, "process_creation_time_filetime": 99},
               "actual_control_steps": 7200, "environment_make_calls": 1, "aggregate_env_steps": 7200 * 4096,
               "budget": {"seed": DR.RESTART_SEED, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
                          "expected_control_steps": 7200},
               "resume_checkpoint": None, "resume_iteration": None, "source_checkpoint_state": None,
               "checkpoint_load_calls": 0, "std_reset_applied": False, "start_iteration": 0,
               "final_checkpoint_state": {"iteration": 299, "policy_state_sha256": "3" * 64,
                                          "optimizer_state_sha256": "4" * 64},
               "resume_scope": {"policy_optimizer_restored": False, "full_environment_state_restored": False,
                                "rng_state_restored_from_checkpoint": False, "fresh_initialization": True,
                                "action_std_reset": False},
               "rng_state_before_learn": rng, "observed": observed, "actual_action_scale": 0.25,
               "runtime_versions": versions, "checkpoint": artifact(run_dir / "model_299.pt"),
               "configs": {"environment": artifact(run_dir / "params/env.yaml"),
                           "agent": artifact(run_dir / "params/agent.yaml")}}
    protocol = {"correction_source_contract": {"runtime": {"version": [3, 10, 15], "cache_tag": "cpython-310"},
                                               "targets": {}},
                "rsl_rl_version": "2.3.3", "isaaclab_commit": "i"}
    execution = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
                 "exit_code": 0, "cleanup_verified": True, "owned_process_identities": {"7": 99}}
    monkeypatch.setattr(DR, "validate_manifest",
                        lambda path: (json.loads(Path(path).read_text(encoding="utf-8")), protocol))
    monkeypatch.setattr(DR, "validate_readback", lambda *a: None)
    monkeypatch.setattr(DR, "validate_training_configuration", lambda *a: None)
    return folder, training, receipt, initial, execution, protocol, {"torch": "2.7.0+cu128", "isaaclab": "0.41.3"}


@pytest.mark.parametrize("target,key,value", [
    (None, None, None),
    ("receipt", "budget", {"seed": 2029, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
                           "expected_control_steps": 7200}),
    ("receipt", "checkpoint_load_calls", 1),
    ("receipt", "start_iteration", 299),
    ("initial", "full_environment_state_restored", True),
    ("initial", "rng_state_before_learn", {"cpu_sha256": "bad"}),
    ("receipt", "process_identity", {"process_id": 8, "process_creation_time_filetime": 99}),
])
def test_verify_training_restart_checks(tmp_path, monkeypatch, target, key, value):
    folder, training, receipt, initial, execution, protocol, versions = _restart_artifacts(tmp_path, monkeypatch)
    if target == "initial":
        initial[key] = value
        (training / "initial.json").write_text(json.dumps(initial), encoding="utf-8")
        receipt["initial_sha256"] = DR.sha(training / "initial.json")
        if key == "rng_state_before_learn":
            receipt[key] = value
    elif target == "receipt":
        receipt[key] = value
    (training / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    if target is None:
        assert DR.verify_training(folder, execution, protocol, versions)["seed"] == DR.RESTART_SEED
    else:
        with pytest.raises(ValueError):
            DR.verify_training(folder, execution, protocol, versions)


# --- 교정 단계: 승인 소비 전 점검 ---
def test_correction_checks_mode_and_budget_before_consuming(demo, monkeypatch, tmp_path):
    import correct as CR
    from rl_triage.correction_ledger import CorrectionLedger
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    _gpu_ready(demo, monkeypatch)
    record = tmp_path / "correction.call1.json"
    record.write_text(json.dumps({"schema": "correction_model_call_v1"}), encoding="utf-8")
    record_sha = hashlib.sha256(record.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    led = CorrectionLedger(tmp_path / "loops" / "h02_s2029" / "correction_ledger.jsonl")
    evidence = {"model_proposal": {"record_sha256": record_sha, "accepted": True}, "rule_mode": "restart"}
    correction_id = CR.R.REGISTRY["reward"]["correction_id"]
    low = led.propose("s1", "reward", "r", correction_id, "restart", {"case": "case_01", "arm": "corrected"},
                      1000, evidence)["request_id"]
    led.approve(low, "operator")
    with pytest.raises(ValueError, match="예산"):
        demo.correction("t9", tmp_path, low, record)
    assert led.requests()[low]["state"] == "approved"  # 소비하지 않았다
    resume = led.propose("s2", "reward", "r", correction_id, "resume", {"case": "case_01", "arm": "corrected"},
                         2400, evidence)["request_id"]
    led.approve(resume, "operator")
    out = demo.correction("t9", tmp_path, resume, record)
    assert out["outcome"] == "correction_not_proposed" and led.requests()[resume]["state"] == "approved"
