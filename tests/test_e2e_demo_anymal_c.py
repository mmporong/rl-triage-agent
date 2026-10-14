"""두 번째 로봇 task 단계 2(ANYmal-C F1 종단 시연): 시연 v4와 같은 규칙·범주·결정 경로를 쓰는지, 단계 1 결과 결합과 시작 분기
(결과 없음·fail·invalid면 시작하지 않음), task에 묶인 단계(학습·고정 평가·고리 probe·packet)가 ANYmal-C 진입으로 가는지,
교정 원장·교정 실행기 결합. GPU·Isaac·모델 없음(프로세스와 호출은 가짜)."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import correct as CR  # noqa: E402
import demo_correction_anymal_c as DC  # noqa: E402
import e2e_demo_anymal_c as E  # noqa: E402
import e2e_demo_v4 as E4  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import loop  # noqa: E402
import run_active_diagnosis as RA  # noqa: E402
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402
from test_correction_step import make_loop  # noqa: E402
from test_e2e_demo_v4 import CHECKLIST, decision, write_record  # noqa: E402

EXPERIMENT = "a" * 40
TAG = "anymal_c_e2e_t"
P0C = ROOT / "evals/results/p0c_fixed_eval_20261009"
GATE_REPORT = ROOT / "evals/results/anymal_c_gate_20261014/runs/anymal_c_gate_none_s2043.json"
BANDS = json.loads((P0C / "summary.json").read_text(encoding="utf-8"))["bands"]  # 단위 시험용(단계 1 밴드 자리)
GOOD = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
        "checklist": dict(CHECKLIST), "hold_reason": None}


def finished(**change) -> dict:
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
            "cleanup_verified": True, "exit_code": 0, "wall_s": 12.0, "owned_process_identities": {}, **change}


def fake_owned(*behaviors):
    """P.run_owned 대역. 호출마다 다음 동작(폴더·인자에 산출물 쓰기)을 하고 정상 종료 기록을 돌려준다."""
    calls = []

    def owned(folder, argv, cwd, watchdog_s):
        calls.append({"argv": [str(a) for a in argv], "cwd": Path(cwd), "fault": os.environ.get("TRIAGE_FAULT"),
                      "watchdog_s": watchdog_s})
        return behaviors[len(calls) - 1](Path(folder), [str(a) for a in argv]) or finished()
    return owned, calls


def telemetry(value: float = 0.5) -> dict:
    return {"run_dir_name": "x", "summary": {"Train/mean_reward": {"mean_last_20pct": value}},
            "series": {A.METRIC: [value] * 300}}


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def protocol_stub(**extra) -> dict:
    return {**E.fixed("f1"), "bands": BANDS, "eval_protocol_sha256_lf": E.sha(E.EVAL_PROTOCOL, True),
            "stage1": {"result": {"path": E.STAGE1_RESULT, "sha256_lf": "0" * 64},
                       "reference_telemetry": {}}, **extra}


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """시연 출력·결과·고리 경로와 ROOT를 임시 폴더로 돌린다. 출력 위치는 저장소와 같은 상대 경로다."""
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "bench/private/e2e_demo_anymal_c")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "PROTOCOL_DIR", tmp_path / "protocols")
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    write_json(tmp_path / "protocols/e2e_demo_anymal_c_f1.json", protocol_stub())
    state = {"protocol": protocol_stub()}
    monkeypatch.setattr(E, "verify_for", lambda t, isaaclab: (state["protocol"], EXPERIMENT))
    E.state = state
    return E


def isaaclab_root(tmp_path) -> Path:
    root = tmp_path / "isaaclab"
    (root / "logs/rsl_rl/anymal_c_flat").mkdir(parents=True, exist_ok=True)
    return root


def write_steps(tag: str, records: dict, until: str) -> None:
    ctx = E.ctx_of("f1")
    E._write_step(tag, "begin", E._guard(tag, "begin"), EXPERIMENT, ctx.protocol,
                  {"variant": "f1", "decision_path": E.DECISION_PATH, "seed_unused": {"hits": []}})
    for name in E.STEPS[1:E.STEPS.index(until)]:
        E._write_step(tag, name, E._guard(tag, name), EXPERIMENT, ctx.protocol, records.get(name, {}))


# --- 사전등록 ---
def test_protocol_reuses_the_v4_flow_and_binds_only_the_anymal_task():
    fixed = E.fixed("f1")
    assert json.loads(json.dumps(fixed)) == fixed and fixed["schema"] == "e2e_demo_protocol_anymal_c_v1"
    assert E.VARIANTS == {"f1": {"fault": "F1", "case_id": "h02", "mechanism": "reward", "seed": 2047}}
    ctx = E.ctx_of("f1")
    used = (set(range(2026, 2038)) | {2040} | set(range(2042, 2047)) | set(range(3029, 3035)) | {3040, 3046, 7, 42, 123})
    assert (ctx.seed, ctx.restart_seed) == (2047, 3047) and not {2047, 3047} & used
    assert ctx.cases == {"NONE": "anymal_c_none_s2047", "FAULT": "anymal_c_h02_s2047"}
    assert ctx.correction_id == "restore_body_frame_reward" and ctx.protocol.name == "e2e_demo_anymal_c_f1.json"
    assert (fixed["task"], fixed["experiment"], fixed["action_scale"]) == ("Isaac-Velocity-Flat-Anymal-C-v0", "anymal_c_flat", 0.5)
    assert fixed["recovery_rule"] == {"mechanism": "reward", **E4.RECOVERY_RULES["reward"]}
    assert fixed["registry"] == E4.registry_v3() and fixed["outcomes"] == E4.OUTCOMES
    assert fixed["success_conditions"] == E4.SUCCESS_CONDITIONS and fixed["feedback_policy"] == E4.FEEDBACK_POLICY
    assert fixed["invalid_after_retry"] == E4.INVALID_AFTER_RETRY and fixed["specificity_gate"]["probes"] == []
    assert set(fixed["predictions"]) == set(E4.PREDICTIONS["f2"]) and E.STEPS == E4.STEPS
    assert "Go2 시연 수치와의 비교" in fixed["forbidden_claims"] and "fail·invalid" in fixed["stage"]["on_stage1_not_pass"]
    assert {"evals/e2e_demo_anymal_c.py", "bench/demo_correction_anymal_c.py", "evals/metered_probes_task.py",
            "evals/e2e_demo_v4.py", "evals/run_anymal_c_band.py", "evals/fixed_eval_task.py", "evals/probes_task.py",
            "evals/probes.py", "src/rl_triage/probe_loop.py", "evals/feedback_model_call.py"} <= E.CODE
    assert all((ROOT / name).is_file() for name in E.CODE)
    with pytest.raises(ValueError):
        E.ctx_of("f2")


# --- 단계 1 결합과 시작 분기 ---
def stage1_files(root: Path, outcome: str = "pass") -> dict:
    """단계 0 결과·텔레메트리, 단계 1 프로토콜 사본·seed 기록·텔레메트리·결과를 임시 ROOT에 쓴다."""
    import run_anymal_c_band as B
    protocol = root / "bench/protocols/anymal_c_band_v1.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(B.PROTOCOL, protocol)
    stage0_telemetry = write_json(root / B.STAGE0_FILES["telemetry"], telemetry(0.20))
    write_json(root / B.STAGE0_FILES["result"], {"outcome": "pass", "telemetry_sha256": E.sha(stage0_telemetry)})
    rows = {}
    for seed, value in ((2044, 0.21), (2045, 0.22)):
        name = f"anymal_c_band_none_s{seed}"
        path = write_json(root / f"bench/runs_anymal/{name}.telemetry.json", telemetry(value))
        write_json(root / f"evals/results/{E.STAGE1_TAG}/seeds/s{seed}_a1.json",
                   {"valid": True, "run_name": name, "telemetry": f"bench/runs_anymal/{name}.telemetry.json",
                    "telemetry_sha256": E.sha(path)})
        rows[str(seed)] = {"attempt": 1, "run_name": name, "valid": True}
    result = write_json(root / E.STAGE1_RESULT, {
        "schema": "second_task_band_result_v1", "tag": E.STAGE1_TAG, "task": E.G.TASK,
        "protocol_sha256_lf": E.sha(protocol, True), "seed_records": rows, "outcome": outcome,
        "failed_criteria": [] if outcome == "pass" else ["no_tracking_false_alarm"],
        "analysis": {"bands": BANDS} if outcome != "invalid" else None})
    return {"protocol": protocol, "result": result}


@pytest.fixture
def stage1(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "STAGE1_PROTOCOL", tmp_path / "bench/protocols/anymal_c_band_v1.json")
    monkeypatch.setattr(E, "PROTOCOL_DIR", tmp_path / "protocols")
    return tmp_path


def test_stage1_pass_binds_the_bands_and_the_three_normal_runs(stage1):
    stage1_files(stage1)
    bound = E.stage1_inputs()
    assert bound["outcome"] == "pass" and bound["result"]["path"] == E.STAGE1_RESULT
    assert bound["reference_runs"] == ["anymal_c_gate_none_s2043", "anymal_c_band_none_s2044", "anymal_c_band_none_s2045"]
    assert set(bound["reference_telemetry"]) == set(bound["reference_runs"]) and set(bound["seed_records"]) == {"2044", "2045"}
    assert E.stage1_status()["start_stage2"] is True


@pytest.mark.parametrize("outcome", ["fail", "invalid", None])
def test_stage2_does_not_start_unless_stage1_passed(stage1, monkeypatch, capsys, outcome):
    if outcome:
        stage1_files(stage1, outcome)
    with pytest.raises(E.Stage1NotPassed):
        E.stage1_inputs()
    status = E.stage1_status()
    assert status["start_stage2"] is False and (outcome or "없다") in status["reason"]
    monkeypatch.setattr(E.V3, "check_contract", lambda contract, isaaclab: None)
    with pytest.raises(E.Stage1NotPassed):
        E.prepare("f1", stage1 / "isaaclab")
    assert not (stage1 / "protocols").exists()
    assert E.main(["prepare", "--variant", "f1", "--isaaclab", str(stage1 / "isaaclab")]) == 4
    assert "단계 2 시작 안 함" in capsys.readouterr().err


@pytest.mark.parametrize("tamper", ["protocol", "telemetry", "record"])
def test_stage1_results_that_do_not_match_their_sources_are_refused(stage1, tamper):
    files = stage1_files(stage1)
    if tamper == "protocol":
        files["protocol"].write_text(files["protocol"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif tamper == "telemetry":
        write_json(stage1 / "bench/runs_anymal/anymal_c_band_none_s2045.telemetry.json", telemetry(9.0))
    else:
        write_json(stage1 / f"evals/results/{E.STAGE1_TAG}/seeds/s2044_a1.json", {"valid": False})
    with pytest.raises(ValueError) as error:
        E.stage1_inputs()
    assert not isinstance(error.value, E.Stage1NotPassed)


# --- task에 묶인 단계 ---
def test_packet_differs_from_the_shared_builder_only_in_the_task_text(demo):
    sources = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
               "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    shared = RA.input_packet({"sources": sources}, {"max_probes": E.LOOP_BUDGET,
                                                    "probe_execution_budget_s": E.PROBE_EXECUTION_BUDGET_S})
    ours = E.input_packet(sources)
    assert {k: v for k, v in ours.items() if k != "task"} == {k: v for k, v in shared.items() if k != "task"}
    assert list(ours) == list(shared) and ours["task"] == E.TASK_TEXT and "Go2" not in json.dumps(ours)


def run_dir_writer(isaaclab: Path, name: str, fault: str, seed: int = 2047):
    def behave(folder, argv):
        (folder / "stdout.log").write_text(f'x\n[hidden_fault] {json.dumps({"fault": fault})}\n', encoding="utf-8")
        run = isaaclab / "logs/rsl_rl/anymal_c_flat" / f"2026-10-14_10-00-00_{name}"
        (run / "params").mkdir(parents=True)
        for file in ("model_0.pt", "model_299.pt"):
            (run / file).write_bytes(file.encode())
        (run / "params/env.yaml").write_text(f"seed: {seed}\nscene:\n  num_envs: 4096\nactions:\n  joint_pos:\n    scale: 0.5\n",
                                             encoding="utf-8")
        (run / "params/agent.yaml").write_text(f"seed: {seed}\nmax_iterations: 300\nnum_steps_per_env: 24\n", encoding="utf-8")
    return behave


def telemetry_writer(folder, argv):
    write_json(Path(argv[-1]), telemetry())


def test_training_uses_the_anymal_entry_and_checks_the_fault_readback(demo, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    write_steps(TAG, {}, "train_NONE")
    ctx = E.ctx_of("f1")
    owned, calls = fake_owned(run_dir_writer(isaaclab, ctx.cases["NONE"], "NONE"), telemetry_writer)
    E.train(TAG, "NONE", isaaclab, owned)
    owned, calls = fake_owned(run_dir_writer(isaaclab, ctx.cases["FAULT"], "F1"), telemetry_writer)
    step = E.train(TAG, "FAULT", isaaclab, owned)
    argv = calls[0]["argv"]
    assert calls[0]["fault"] == "F1" and "TRIAGE_FAULT" not in os.environ and calls[0]["cwd"] == isaaclab
    assert argv[1].endswith("train_with_fault.py") and argv[argv.index("--task") + 1] == "Isaac-Velocity-Flat-Anymal-C-v0"
    assert argv[argv.index("--seed") + 1] == "2047" and argv[argv.index("--run_name") + 1] == "anymal_c_h02_s2047"
    assert calls[1]["argv"][1].endswith("extract_tb.py") and step["readback"] == [{"fault": "F1"}]
    assert step["artifacts"]["telemetry"]["path"] == "bench/runs_anymal/anymal_c_h02_s2047.telemetry.json"
    assert step["execution"]["parent_wall_s"] == 12.0 and E._check_artifacts(TAG, "FAULT") == step["artifacts"]


def test_training_without_the_registered_fault_readback_ends_invalid(demo, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    write_steps(TAG, {}, "train_NONE")
    owned, _ = fake_owned(run_dir_writer(isaaclab, E.ctx_of("f1").cases["NONE"], "F1"), telemetry_writer)
    with pytest.raises(RuntimeError, match="invalid"):
        E.train(TAG, "NONE", isaaclab, owned)
    assert json.loads((E.folder_for(TAG) / "terminal.json").read_text(encoding="utf-8"))["outcome"] == "invalid"


def fixed_reports(checkpoints: dict, protocol_sha: str) -> dict:
    """NONE은 ANYmal-C 단계 0 정상 보고서, FAULT은 Go2 h02 보고서를 ANYmal-C 평가 칸으로 옮긴 단위 시험용 본문."""
    none = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    fault = json.loads((P0C / "runs/h02_s2027.json").read_text(encoding="utf-8"))
    out = {}
    for arm, body in (("NONE", none), ("FAULT", fault)):
        out[arm] = {**body, "protocol": "fixed_eval_v1_anymal_c", "task": E.G.TASK, "action_scale": 0.5,
                    "protocol_sha256_lf": protocol_sha, "versions": none["versions"],
                    "name": E.ctx_of("f1").cases[arm], "checkpoint": {"file": "model_299.pt", "sha256": checkpoints[arm]}}
    return out


def eval_env_line() -> str:
    names = E.read(E.EVAL_PROTOCOL)["env_events_disabled"]
    line = {"task": E.G.TASK, "disabled": names, "was_enabled": dict.fromkeys(names, True),
            "now_enabled": dict.fromkeys(names, False)}
    return "[eval_env] " + json.dumps(line) + "\n"


def ready_for_evaluate(demo, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    write_steps(TAG, {}, "train_NONE")
    ctx = E.ctx_of("f1")
    for arm, fault in (("NONE", "NONE"), ("FAULT", "F1")):
        owned, _ = fake_owned(run_dir_writer(isaaclab, ctx.cases[arm], fault), telemetry_writer)
        E.train(TAG, arm, isaaclab, owned)
    E.alarm(TAG, isaaclab)
    reports = fixed_reports({arm: E._check_artifacts(TAG, arm)["checkpoint"]["sha256"] for arm in ("NONE", "FAULT")},
                            E.sha(E.EVAL_PROTOCOL, True))
    versions = reports["NONE"]["versions"]
    demo.state["protocol"] = protocol_stub(torch_version=versions["torch"], rsl_rl_version=versions["rsl-rl-lib"])
    return isaaclab, reports


def evaluator(reports: dict, env_line: str):
    def behave(folder, argv):
        (folder / "stdout.log").write_text(env_line, encoding="utf-8")
        for job in json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))["jobs"]:
            arm = "NONE" if job["name"].startswith("anymal_c_none") else "FAULT"
            write_json(Path(job["output"]), reports[arm])
    return behave


def test_evaluation_uses_the_anymal_protocol_and_stage1_bands(demo, tmp_path):
    isaaclab, reports = ready_for_evaluate(demo, tmp_path)
    owned, calls = fake_owned(evaluator(reports, eval_env_line()))
    step = E.evaluate(TAG, isaaclab, owned)
    argv = calls[0]["argv"]
    assert argv[1].endswith("fixed_eval_task.py") and argv[argv.index("--protocol") + 1] == str(E.EVAL_PROTOCOL)
    jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
    assert jobs["action_scale"] == 0.5 and [j["name"] for j in jobs["jobs"]] == ["anymal_c_none_s2047", "anymal_c_h02_s2047"]
    assert step["gate"]["outcome"] == "failure_present" and step["gate"]["none_walks"] is True
    assert step["gate"]["fault_verdict"]["label"] == "unhealthy"
    assert step["eval_env"][0]["now_enabled"] == {"push_robot": False, "base_com": False}


def test_evaluation_without_the_event_readback_ends_invalid(demo, tmp_path):
    isaaclab, reports = ready_for_evaluate(demo, tmp_path)
    owned, _ = fake_owned(evaluator(reports, "no readback\n"))
    with pytest.raises(RuntimeError, match="readback"):
        E.evaluate(TAG, isaaclab, owned)
    assert json.loads((E.folder_for(TAG) / "terminal.json").read_text(encoding="utf-8"))["outcome"] == "invalid"


def test_reports_from_the_go2_protocol_are_refused():
    report = json.loads((P0C / "runs/h02_s2027.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="ANYmal-C"):
        E.validate_report(report, {})


# --- 승인 고리 probe(loop-run) ---
def probe_writer(measurement: dict, checkpoint_sha: str):
    def behave(folder, argv):
        jobs_path = Path(argv[argv.index("--jobs") + 1])
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        job = jobs["jobs"][0]
        write_json(Path(job["output"]), {"name": job["name"], "task": jobs["task"], "seed": jobs["seed"],
                                         "checkpoint": {"sha256": checkpoint_sha},
                                         "probes": {job["probes"][0]: {"exit": 0, "measurement": measurement}}})
        write_json(Path(argv[argv.index("--metrics") + 1]), {
            "completed": True, "cpu_s": 30.0, "phase": "before_app_close", "cpu_scope": "c", "simulator_steps_scope": "s",
            "jobs_sha256": E.sha(jobs_path), "probes_sha256_lf": E.sha(ROOT / "evals/probes.py", True),
            "measurements": [{"probe": job["probes"][0], "checkpoint_sha256": checkpoint_sha, "num_envs": 64,
                              "successful_step_calls": 10, "simulator_steps": 640, "completed": True}]})
    return behave


def test_loop_run_measures_with_the_anymal_driver_and_leaves_loop_py_artifacts(demo, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    write_steps(TAG, {}, "train_NONE")
    ctx = E.ctx_of("f1")
    for arm, fault in (("NONE", "NONE"), ("FAULT", "F1")):
        owned, _ = fake_owned(run_dir_writer(isaaclab, ctx.cases[arm], fault), telemetry_writer)
        E.train(TAG, arm, isaaclab, owned)
    checkpoint = E._check_artifacts(TAG, "FAULT")["checkpoint"]["sha256"]
    reference = {p: {"x": 1} for p in L.PROBES}
    started = V2.start_loop(ctx.cases["FAULT"], ["reward", "exploration", "optimizer"], checkpoint, reference, "P_reward")
    for name in ("alarm", "evaluate", "reference", "packet", "loop_start"):  # 고리 시작까지의 기록(단계 사슬만)
        E._write_step(TAG, name, E._guard(TAG, name), EXPERIMENT, ctx.protocol, {})
    rid = started["request_id"]
    loop.L.Ledger(tmp_path / "loops" / ctx.cases["FAULT"] / "ledger.jsonl").approve(rid, "operator")
    abnormal = {"track_lin_vel_xy_exp_rel_error": 0.35, "track_ang_vel_z_exp_rel_error": 1.0e-8}
    owned, calls = fake_owned(probe_writer(abnormal, checkpoint))
    out = E.loop_run(TAG, isaaclab, rid, owned)
    argv = calls[0]["argv"]
    jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
    assert argv[1].endswith("metered_probes_task.py") and calls[0]["cwd"] == isaaclab
    assert (jobs["task"], jobs["fault"], jobs["seed"]) == ("Isaac-Velocity-Flat-Anymal-C-v0", "F1", 2047)
    case = ctx.cases["FAULT"]
    meta = json.loads(loop._execution_path(case, "P_reward", f"{TAG}_loop").read_text(encoding="utf-8"))
    assert meta["metered"] is True and meta["simulator_steps"] == 640 and meta["exit_code"] == 0
    prereg, state, _ = loop._load(case)
    assert state["observed"] == {"P_reward": "abnormal"} and state["status"] == "confirmed" and state["hypotheses"] == ["reward"]
    assert loop.file_artifact(case, "P_reward", prereg, f"{TAG}_loop")["measurement"] == abnormal
    assert out["next_commands"] == [f"python evals/e2e_demo_anymal_c.py close-loop --tag {TAG}"]
    summary = E.loop_summary(ctx, TAG, checkpoint)
    assert summary["requests"][0]["artifact"]["probe"].endswith(f"{case}__P_reward.json") and summary["probe_costs"][0]["simulator_steps"] == 640
    with pytest.raises(SystemExit):  # 같은 요청을 다시 소비하지 않는다(evals/loop.py와 같은 거부)
        E.loop_run(TAG, isaaclab, rid, fake_owned(probe_writer(abnormal, checkpoint))[0])


# --- 첫 판단 ---
def ready_for_diagnosis(demo, monkeypatch, tmp_path):
    ctx = E.ctx_of("f1")
    packet_path = E.folder_for(TAG) / "model" / "diagnosis.packet.json"
    sources = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
               "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    monkeypatch.setattr(RA, "ROOT", ROOT)
    write_json(packet_path, E.input_packet(sources))
    reference = write_json(tmp_path / "evals/results" / f"{TAG}_ref" / "probes" / "none.json",
                           {"probes": {p: {"measurement": {"x": 1}} for p in L.PROBES}})
    write_steps(TAG, {"evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.15, "FAULT": 0.99}}},
                      "reference": {"report": {"path": E._rel(reference)}},
                      "packet": {"packet": {"path": E._rel(packet_path), "sha256": E.sha(packet_path)}}}, "loop_start")
    monkeypatch.setattr(E, "_check_artifacts", lambda t, arm: {"checkpoint": {"sha256": "c" * 64}})
    started = []
    monkeypatch.setattr(V2, "start_loop", lambda case, ranking, sha, ref, probe: started.append((case, probe)) or {
        "request_id": "q1", "model_probe_used": True, "probe": probe})
    return ctx, packet_path, started


def test_loop_starts_from_a_valid_feedback_decision_and_names_loop_run(demo, monkeypatch, tmp_path):
    ctx, packet_path, started = ready_for_diagnosis(demo, monkeypatch, tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    record = write_record(E.folder_for(TAG) / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path,
                          json.dumps(decision(packet)))
    step = E.loop_start(TAG, ROOT, record)
    assert started == [(ctx.cases["FAULT"], "P_reward")] and step["feedback"]["final_status"] == "valid"
    assert step["call"]["path"] == "single_feedback" and step["call"]["requests"] == 1
    assert any(f"e2e_demo_anymal_c.py loop-run --tag {TAG} --request q1" in c for c in step["next_commands"])


def test_invalid_after_the_retry_ends_as_model_invalid(demo, monkeypatch, tmp_path):
    _, packet_path, started = ready_for_diagnosis(demo, monkeypatch, tmp_path)
    record = write_record(E.folder_for(TAG) / "model" / "diagnosis.feedback1.json", "diagnosis", packet_path,
                          '{"ranking": [', '{"ranking": [')
    assert E.loop_start(TAG, ROOT, record)["outcome"] == "model_invalid" and started == []
    result = E.summarize(TAG)
    assert result["outcome"] == "model_invalid" and result["schema"] == "e2e_demo_result_anymal_c_v1"
    assert result["task"] == "Isaac-Velocity-Flat-Anymal-C-v0" and result["model_feedback"]["diagnosis"]["retries"] == 1


# --- 교정 ---
def ready_for_correction(demo, monkeypatch, tmp_path, *answers):
    ctx = E.ctx_of("f1")
    report = tmp_path / "evals/results" / TAG / "fault.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P0C / "runs/h02_s2027.json", report)  # 정지 대비 비 ≥ 0.9(규칙 restart)인 이미 본 F1 평가
    ranking = ["reward"] + [m for m in L.MECHANISMS if m != "reward"]
    feedback = {"first_status": "valid", "first_codes": [], "retried": False, "final_status": "valid", "final_codes": [],
                "requests": 1, "calls": [{"step": "first", "input_tokens": 2800, "output_tokens": 180, "wall_s": 2.0}]}
    records = {
        "train_NONE": {"execution": {"parent_wall_s": 560.0}}, "train_FAULT": {"execution": {"parent_wall_s": 570.0}},
        "alarm": {"results": {"FAULT": {"warned": True}, "NONE": {"warned": False}}},
        "evaluate": {"execution": {"parent_wall_s": 60.0},
                     "gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.15, "FAULT": 0.99}},
                     "reports": {"FAULT": {"path": E._rel(report), "sha256_lf": E.sha(report, True)}}},
        "reference": {"execution": {"parent_wall_s": 90.0}},
        "loop_start": {"decision": {"action": "continue", "ranking": ranking, "probe": "P_reward"},
                       "started": {"request_id": "q", "model_probe_used": True}, "feedback": feedback,
                       "call": {"path": "single_feedback", "record_sha256": "d" * 64, "input_tokens": 2800,
                                "output_tokens": 180, "execution_wall_s": 2.1}},
        "loop_close": {"proceed_to_correction": True,
                       "loop": {"status": "confirmed", "hypotheses": ["reward"], "observed": {"P_reward": "abnormal"},
                                "requests": [{"state": "done", "approver": "operator", "outcome": "abnormal", "gpu_s": 40.0,
                                              "artifact": {"probe": "p"}}], "probe_costs": []}},
        "specificity": {"required": [], "passed": True},
    }
    write_steps(TAG, records, "correction")
    make_loop(tmp_path / "loops", case=ctx.cases["FAULT"], mechanism="reward")
    info = E.correction_evidence(TAG, ROOT)
    packet_path = E.ROOT / info["packet"]
    record = write_record(E.folder_for(TAG) / "model" / "correction.feedback1.json", "correction", packet_path, *answers)
    return ctx, record


def test_correction_runs_on_the_v4_registry_ledger_with_the_anymal_executor(demo, monkeypatch, tmp_path):
    ctx, record = ready_for_correction(demo, monkeypatch, tmp_path, json.dumps(GOOD))
    proposed = E.correction_propose(TAG, ROOT, record, budget_gpu_s=2400)
    assert proposed["recovery_mode"] == "restart" and proposed["correction_id"] == "restore_body_frame_reward"
    assert proposed["registry_digest"] == R.registry_digest(E4.registry_v3())
    E.approve_correction(TAG, proposed["request_id"], "operator")
    calls = []

    def make(tag, isaaclab):
        def run(request):
            calls.append(request)
            return {"exit_code": 0, "gpu_s": 700.0, "verdict": "recovered", "result_sha": "f" * 64, "tag": request["tag"]}
        return run
    monkeypatch.setattr(DC, "make_executor", make)
    step = E.correction(TAG, isaaclab_root(tmp_path), proposed["request_id"], record)
    assert calls[0]["tag"] == f"{TAG}_fix" and calls[0]["recovery_mode"] == "restart"
    assert [e["event"] for e in E.correction_ledger(ctx).events()] == ["proposed", "approved", "consumed", "receipt"]
    result = E.summarize(TAG)
    assert result["outcome"] == "recovered" and result["success_conditions"]["correction_call_accepted"] is True
    assert result["prediction_results"]["correction_rule"] is True and result["prediction_results"]["specificity"] is True
    assert step["call"]["path"] == "single_feedback"


def test_correction_budget_floor_and_terminal_guard_follow_v4(demo, monkeypatch, tmp_path):
    _, record = ready_for_correction(demo, monkeypatch, tmp_path, json.dumps(GOOD))
    for budget in (2399, 2400.0):
        with pytest.raises(ValueError, match="하한"):
            E.correction_propose(TAG, ROOT, record, budget_gpu_s=budget)
    (E.folder_for(TAG) / "terminal.json").write_text(json.dumps({"outcome": "harness_failure"}), encoding="utf-8")
    with pytest.raises(ValueError, match="종료"):
        E.correction_evidence(TAG, ROOT)


# --- 교정 실행기 결합 ---
def test_correction_executor_is_bound_to_this_runner_and_the_anymal_task(demo, monkeypatch, tmp_path):
    import demo_correction_v4 as DC4
    assert DC._demo() is E and DC.fix_tag(TAG) == f"{TAG}_fix" and DC.ARMS is DC4.ARMS and DC.judge is DC4.judge
    with pytest.raises(ValueError):
        DC.fix_tag("e2e_demo_v4_f2_s2046")
    ctx = E.ctx_of("f1")
    artifacts = {k: {"path": str(tmp_path / f"{k}.bin"), "sha256": "b" * 64} for k in ("checkpoint", "environment", "agent")}
    artifacts["checkpoint"]["path"] = str(tmp_path / "2026-10-14_10-00-00_anymal_c_none_s2047" / "model_299.pt")
    monkeypatch.setattr(E, "step_record", lambda tag, name: {"artifacts": artifacts})
    request = {"variant": "f1", "recovery_mode": "resume", "demo_tag": TAG, "tag": f"{TAG}_fix", "source_commit_sha": EXPERIMENT,
               "protocol_sha256_lf": "p" * 64, "isaaclab_root": str(tmp_path)}
    manifest = DC.manifest_for(tmp_path / "fix", request, "clean_resume_reference")
    assert manifest["task"] == "Isaac-Velocity-Flat-Anymal-C-v0" and manifest["protocol_path"] == str(ctx.protocol.resolve())
    assert manifest["load_run_regex"] == "^2026\\-10\\-14_10\\-00\\-00_anymal_c_none_s2047$" and manifest["final_iteration"] == 598
    source = (ROOT / "bench/demo_correction_anymal_c.py").read_text(encoding="utf-8")
    assert 'str(ROOT / "bench/demo_correction_anymal_c.py")' in source and "import e2e_demo_anymal_c as E" in source
    assert 'str(ROOT / "evals/fixed_eval_task.py")' in source and "unitree_go2_flat" not in source
    assert "unitree_go2_flat" not in (ROOT / "evals/e2e_demo_anymal_c.py").read_text(encoding="utf-8")


def test_begin_refuses_used_seeds_and_existing_outputs(demo, monkeypatch, tmp_path):
    isaaclab = isaaclab_root(tmp_path)
    ctx = E.ctx_of("f1")
    monkeypatch.setattr(E.B, "seed_evidence", lambda root, seed, allowed: {"seed": seed, "checked_roots": ["r"],
                                                                           "hits": [] if seed == 2047 else ["x_s3047"]})
    evidence = E.seed_unused_evidence(ctx, isaaclab, TAG)
    assert evidence["hits"] == ["x_s3047"] and evidence["checked_seeds"] == [2047, 3047]
    (tmp_path / "evals/loops" / ctx.cases["FAULT"]).mkdir(parents=True)
    monkeypatch.setattr(E.B, "seed_evidence", lambda root, seed, allowed: {"seed": seed, "checked_roots": ["r"], "hits": []})
    assert E.seed_unused_evidence(ctx, isaaclab, TAG)["existing_outputs"] == ["evals/loops/anymal_c_h02_s2047"]
