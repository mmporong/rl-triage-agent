"""E4 종단 시연 v3: 변형·회복 방식 규칙·결정 경로 스위치·특이성 게이트·resume 교정 실행기. GPU·모델 없음.
v2에서 그대로 재사용한 함수(진단 기록 검토·고리 시작·실험 커밋 고정)는 v2 테스트가 시험한다."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_correction_v3 as DC  # noqa: E402
import e2e_demo_v3 as E  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import correction_registry as REG  # noqa: E402

EXPERIMENT = "a" * 40
P0C = ROOT / "evals/results/p0c_fixed_eval_20261009"


def report(name):
    return json.loads((P0C / "runs" / f"{name}.json").read_text(encoding="utf-8"))


def bands():
    return json.loads((P0C / "summary.json").read_text(encoding="utf-8"))["bands"]


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """시연 출력·결과·고리 경로와 ROOT를 임시 폴더로 돌린다."""
    import correct as CR
    import loop
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "PROTOCOL_DIR", tmp_path / "protocols")
    (tmp_path / "protocols").mkdir()
    for variant in E.VARIANTS:
        (tmp_path / "protocols" / f"e2e_demo_v3_{variant}.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    return E


def begin_record(demo, tag, variant="f2", commit=EXPERIMENT):
    ctx = demo.ctx_of(variant)
    demo._write_step(tag, "begin", demo._guard(tag, "begin"), commit, ctx.protocol,
                     {"variant": variant, "decision_path": "single", "seed_unused": {"hits": []}})


# --- 변형·규칙·등록 표 ---
def test_variants_bind_case_seed_and_restart_seed():
    f2, f1 = E.ctx_of("f2"), E.ctx_of("f1")
    assert (f2.fault, f2.case_id, f2.mechanism, f2.seed, f2.restart_seed) == ("F2", "h06", "actuator", 2031, 3031)
    assert f2.cases == {"NONE": "baseline_p0c_s2031", "FAULT": "h06_s2031"}
    assert f2.correction_id == "restore_motor_torque_speed" and f1.correction_id == "restore_body_frame_reward"
    assert {f1.seed, f2.seed, f1.restart_seed, f2.restart_seed}.isdisjoint({2026, 2027, 2028, 2029, 2030, 3029, 3030})
    assert all((ROOT / name).is_file() for name in E.CODE)
    with pytest.raises(ValueError):
        E.ctx_of("f9")


def test_recovery_rules_registry_and_pending_actuator(monkeypatch):
    restart = E.recommend("reward", report("h02_s2027"))
    assert restart["mode"] == "restart" and restart["correction_id"] == "restore_body_frame_reward"
    assert E.recommend("reward", report("baseline_p0c_s2027"))["mode"] == "resume"  # 정지 비율 < 0.9
    assert E.recommend("exploration", report("h05_s2027"))["mode"] == REG.HOLD
    assert E.registry_v3()["actuator"]["recovery_modes"] == ["restart", "resume"]
    motor = E.recommend("actuator", report("h06_s2027"))
    assert motor["mode"] == "restart" and motor["correction_id"] == "restore_motor_torque_speed"
    monkeypatch.setitem(E.RECOVERY_RULES, "actuator", {**E.RECOVERY_RULES["actuator"], "mode": None})
    with pytest.raises(ValueError, match="비어"):
        E.recommend("actuator", report("h06_s2027"))
    assert E.registry_v3()["actuator"] == REG.REGISTRY["actuator"]  # 규칙이 비어 있으면 등록 표는 기본 그대로


def test_actuator_rule_follows_the_published_w1_aggregate():
    import run_training_correction_motor as M
    out = M.aggregate_public([ROOT / "evals/results/motor_w1_s2026_20261013", ROOT / "evals/results/motor_w1_s2028_20261013"])
    assert out["rule_update"]["decision"] == "change" and out["rule_update"]["next_rule"].endswith(
        "or fault_mechanism == actuator -> restart, otherwise resume")
    assert E.RECOVERY_RULES["actuator"]["mode"] == "restart"


def test_prepare_refuses_pending_rule_and_unadopted_nat(demo, monkeypatch):
    filled = E.RECOVERY_RULES["actuator"]
    monkeypatch.setitem(E.RECOVERY_RULES, "actuator", {**filled, "mode": None})
    with pytest.raises(ValueError, match="비어"):
        demo.prepare("f2", "single", ROOT)
    monkeypatch.setitem(E.RECOVERY_RULES, "actuator", filled)
    monkeypatch.setattr(demo, "E4B_SUMMARY", demo.ROOT / "missing.json")
    with pytest.raises(ValueError, match="E4b v4"):
        demo.nat_adoption("nat_v4")
    summary = demo.ROOT / "summary.json"
    summary.write_text(json.dumps({"adoption": {"decision": "do_not_adopt"}}), encoding="utf-8")
    monkeypatch.setattr(demo, "E4B_SUMMARY", summary)
    with pytest.raises(ValueError, match="adopt"):
        demo.nat_adoption("nat_v4")
    summary.write_text(json.dumps({"adoption": {"decision": "adopt"}}), encoding="utf-8")
    assert demo.nat_adoption("nat_v4")["decision"] == "adopt" and demo.nat_adoption("single") is None
    with pytest.raises(ValueError, match="결정 경로"):  # E4b v3는 do_not_adopt라 등록 경로에서 뺐다
        demo.fixed("f2", "nat_v3")


def test_nat_v4_correction_review_uses_the_demo_registry(demo, tmp_path, monkeypatch):
    import correction_model_call as CM
    import nat_decision_call as W1
    import nat_decision_call_v4 as W4
    monkeypatch.setitem(E.RECOVERY_RULES, "actuator", {**E.RECOVERY_RULES["actuator"], "mode": "restart"})
    rule = {"mode": "restart", "rule": "r", "correction_id": "restore_motor_torque_speed", "standstill_ratio": 0.5}
    packet = CM.build_packet(case="h06_s2031", loop_state_digest="s", receipts_digest="r", mechanism="actuator",
                             confirming={"probe": "P_torque", "request_id": "q", "outcome": "abnormal", "measurement": {}},
                             observed={"P_torque": "abnormal"}, rule=rule,
                             source_report={"metrics": {"lin_vel_rmse_mps": 1.2}, "standstill": {"lin_vel_rmse_mps": 1.2}},
                             source_report_sha256="aa", checkpoint_sha256="bb", registry=E.registry_v3())
    packet_path = tmp_path / "correction.packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    proposal = {"correction_id": "restore_motor_torque_speed", "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
                "checklist": {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
                              "5": "unmet", "6": "registered_correction"}, "hold_reason": None}
    assert not CM.validate_model_proposal(proposal, packet)["accepted"]  # 기본 표에는 모터 restart가 없다
    record_path = tmp_path / "correction.nat1.json"
    submission = record_path.with_suffix(".submission.json")
    submission.write_text(json.dumps({"kind": "correction", "proposal": proposal,
                                      "validation": CM.validate_model_proposal(proposal, packet, packet["registry"])}),
                          encoding="utf-8")
    record = {"schema": W4.RECORD_SCHEMA, "kind": "correction", "model": E.MODEL, "temperature": 0,
              "packet_sha256": E.sha(packet_path), "config_sha256_lf": W1.lf_sha(W4.CONFIGS["correction"]),
              "tools_modules_sha256_lf": {name: W1.lf_sha(path) for name, path in W4.TOOLS_MODULES.items()},
              "system_prompt_sha256": W1.text_sha(W4.system_prompt("correction")), "status": "valid",
              "submission_sha256": E.sha(submission)}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    review = E.review_nat_record(record_path, packet_path, "correction", E.registry_v3())
    assert review["status"] == "valid" and review["decision"] == "propose" and review["agrees_with_rule"]
    assert E.call_summary(record_path, record)["path"] == "nat_v4"
    record_path.write_text(json.dumps({**record, "schema": "e4b_nat_decision_v3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="NAT 기록"):
        E.review_nat_record(record_path, packet_path, "correction", E.registry_v3())


def test_fixed_survives_json_and_detects_changes():
    for variant in E.VARIANTS:
        for path in E.DECISION_PATHS:
            fixed = E.fixed(variant, path)
            assert json.loads(json.dumps(fixed)) == fixed  # 프로토콜 파일로 왕복해도 같다
    fixed = E.fixed("f1", "single")
    dynamic = {k: f"v_{k}" for k in E.DYNAMIC}
    E.check_protocol({**fixed, **dynamic}, dynamic)
    with pytest.raises(ValueError):
        E.check_protocol({**fixed, **dynamic, "seed": 2030}, dynamic)
    with pytest.raises(ValueError):
        E.check_protocol({**fixed, **dynamic}, {**dynamic, "source_files": "changed"})
    assert set(fixed["success_conditions"]) == set(E.SUCCESS_CONDITIONS) and E.TERMINAL_OUTCOMES < set(E.OUTCOMES)
    assert E.fixed("f2", "single")["specificity_gate"]["probes"] == ["P_noise", "P_value"]


# --- 단계 ---
def _owned(code=0, calls=None, writer=None):
    def run(folder, argv, cwd, watchdog_s):
        if calls is not None:
            calls.append(argv)
        if writer is not None:
            writer(argv)
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
                "exit_code": code, "cleanup_verified": True, "owned_process_identities": {}, "wall_s": 12.3}
    return run


def test_train_uses_variant_case_and_experiment_commit(demo, monkeypatch, tmp_path):
    seen, calls = [], []
    monkeypatch.setattr(demo, "verify_protocol", lambda variant, isaaclab, experiment=None: (
        seen.append((variant, experiment)) or {"gpu_admission_profile": "x", "num_envs": 4096, "max_iterations": 300,
                                               "watchdogs_s": {"train": 1800}}, experiment or "b" * 40))
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(demo, "training_artifacts", lambda ctx, arm, isaaclab: {"run": ctx.cases[arm]})
    begin_record(demo, "t1")
    _steps_through(demo, "t1", "train_FAULT", {})
    out = demo.train("t1", "FAULT", tmp_path / "IsaacLab", owned=_owned(calls=calls))
    argv = calls[0]
    assert seen == [("f2", EXPERIMENT)] and out["source_commit_sha"] == EXPERIMENT
    assert [argv[argv.index(k) + 1] for k in ("-CaseId", "-Seed", "-Fault")] == ["h06", "2031", "F2"]


def _steps_through(demo, tag, upto, records):
    """begin 뒤 upto 직전까지 단계 기록을 사슬로 쓴다(내용은 records에서)."""
    for name in demo.STEPS[1:demo.STEPS.index(upto)]:
        demo._write_step(tag, name, demo._guard(tag, name), EXPERIMENT, demo.ctx_for(tag).protocol, records.get(name, {}))


def _ready(demo, monkeypatch, tag, variant="f2", gate=("P_noise", "P_value")):
    begin_record(demo, tag, variant)
    protocol = {"gpu_admission_profile": "x", "watchdogs_s": {"specificity_probes": 600},
                "specificity_gate": {"probes": list(gate)}, "decision_path": "single"}
    monkeypatch.setattr(demo, "verify_for", lambda t, isaaclab: (protocol, EXPERIMENT))
    monkeypatch.setattr(V2, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(demo, "_check_artifacts", lambda t, arm: {"checkpoint": {"sha256": "c" * 64}})
    ref = demo.ROOT / "ref.json"
    ref.write_text(json.dumps({"probes": {"P_noise": {"measurement": {"noise_ratio": 1.0}},
                                          "P_value": {"measurement": {"critic_change": 1.4}}}}), encoding="utf-8")
    _steps_through(demo, tag, "specificity", {"reference": {"report": {"path": "ref.json"}},
                                              "loop_close": {"proceed_to_correction": True, "loop": {}}})


def _probe_writer(demo, tag, case, values):
    def write(argv):
        out = demo.RESULTS / f"{tag}_spec" / "probes" / f"{case}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"name": case, "checkpoint": {"sha256": "c" * 64}, "seed": 2031,
                                   "probes": {p: {"exit": 0, "measurement": m} for p, m in values.items()}}),
                       encoding="utf-8")
    return write


def test_specificity_gate_passes_or_ends_the_demo(demo, monkeypatch):
    _ready(demo, monkeypatch, "t2")
    calls = []
    good = {"P_noise": {"noise_ratio": 1.0}, "P_value": {"critic_change": 1.3}}
    out = demo.specificity("t2", ROOT, owned=_owned(calls=calls, writer=_probe_writer(demo, "t2", "h06_s2031", good)))
    assert out["passed"] and out["outcomes"] == {"P_noise": "normal", "P_value": "normal"}
    assert calls[0][calls[0].index("--probes") + 1:calls[0].index("--isaaclab")] == ["P_noise", "P_value"]
    assert any("correction-evidence" in c for c in out["next_commands"])
    _ready(demo, monkeypatch, "t3")
    bad = {"P_noise": {"noise_ratio": 1.0}, "P_value": {"critic_change": 0.01}}  # critic이 거의 안 변했다(F4 신호)
    demo.specificity("t3", ROOT, owned=_owned(writer=_probe_writer(demo, "t3", "h06_s2031", bad)))
    terminal = json.loads((demo.folder_for("t3") / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["outcome"] == "confirmation_not_specific" and terminal["detail"]["outcomes"]["P_value"] == "abnormal"
    with pytest.raises(ValueError, match="특이성"):
        demo._require_specificity("t3")


def test_mechanism_without_gate_records_without_gpu(demo, monkeypatch):
    _ready(demo, monkeypatch, "t4", variant="f1", gate=())
    out = demo.specificity("t4", ROOT, owned=_owned(code=1))  # 실행기를 부르면 실패하도록 둔다
    assert out["passed"] is True and out["required"] == [] and "execution" not in out
    demo._require_specificity("t4")


def test_loop_close_requires_the_registered_mechanism(demo, monkeypatch):
    begin_record(demo, "t5")
    monkeypatch.setattr(demo, "verify_for", lambda t, isaaclab: ({}, EXPERIMENT))
    monkeypatch.setattr(demo, "_check_artifacts", lambda t, arm: {"checkpoint": {"sha256": "c" * 64}})
    monkeypatch.setattr(demo, "loop_summary", lambda ctx, t, c: {"status": "confirmed", "hypotheses": ["reward"]})
    _steps_through(demo, "t5", "loop_close", {})
    demo.loop_close("t5", ROOT)
    terminal = json.loads((demo.folder_for("t5") / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["outcome"] == "wrong_confirmation"  # F2 변형에서 reward 확정


def test_conditions_require_specificity_and_one_commit():
    call = {"record_sha256": "r", "execution_wall_s": 2.0, "input_tokens": 100, "output_tokens": 50}
    base = {"protocol_sha256_lf": "p", "source_commit_sha": EXPERIMENT}
    steps = {
        "begin": {**base, "seed_unused": {"hits": []}},
        **{n: {**base, "execution": {"parent_wall_s": 10.0}} for n in ("train_NONE", "train_FAULT", "reference")},
        "evaluate": {**base, "execution": {"parent_wall_s": 80.0}, "gate": {"outcome": "failure_present"}},
        "loop_start": {**base, "call": call, "decision": {"action": "continue"}, "started": {"model_probe_used": True}},
        "loop_close": {**base, "proceed_to_correction": True,
                       "loop": {"requests": [{"state": "done", "approver": "op", "artifact": {"p": 1}, "outcome": "abnormal",
                                              "gpu_s": 18.0}]}},
        "specificity": {**base, "required": ["P_noise", "P_value"], "passed": True, "execution": {"parent_wall_s": 40.0}},
        "correction": {**base, "call": call, "model_proposal": {"accepted": True}, "receipt": {"event": "receipt"},
                       "result": {"costs": {"train_parent_wall_s": 1000.0, "eval_parent_wall_s": 60.0}}},
    }
    checks = E.conditions(steps, "recovered")
    assert set(checks) == set(E.SUCCESS_CONDITIONS) and all(checks.values())
    no_cost = {**steps, "specificity": {**steps["specificity"], "execution": {}}}
    assert not E.conditions(no_cost, "recovered")["costs_recorded"]
    moved = {**steps, "alarm": {"protocol_sha256_lf": "p", "source_commit_sha": "b" * 40}}
    assert not E.conditions(moved, "recovered")["protocol_committed"]


# --- 교정 실행기 ---
def _request(mode, variant="f2"):
    return {"demo_tag": "d", "tag": DC.fix_tag("d"), "variant": variant, "recovery_mode": mode,
            "source_commit_sha": EXPERIMENT, "protocol_sha256_lf": "p" * 64, "isaaclab_root": str(ROOT)}


def _artifacts(tmp_path, name):
    run = tmp_path / f"2026-10-14_00-00-00_{name}"
    (run / "params").mkdir(parents=True)
    return {"checkpoint": {"path": str(run / "model_299.pt"), "sha256": "e" * 64},
            "environment": {"path": str(run / "params/env.yaml"), "sha256": "f" * 64},
            "agent": {"path": str(run / "params/agent.yaml"), "sha256": "0" * 64}}


def test_manifests_for_restart_and_resume(monkeypatch, tmp_path):
    arts = {"train_NONE": _artifacts(tmp_path, "baseline_p0c_s2031"), "train_FAULT": _artifacts(tmp_path, "h06_s2031")}
    monkeypatch.setattr(E, "step_record", lambda tag, name: {"artifacts": arts[name]})
    restart = DC.manifest_for(tmp_path / "fix", _request("restart"), "restart_corrected")
    assert R.validate_manifest_mode(restart) == "restart" and restart["seed"] == 3031 and restart["fault"] == "F2"
    reference = DC.manifest_for(tmp_path / "fix", _request("resume"), "clean_resume_reference")
    treated = DC.manifest_for(tmp_path / "fix", _request("resume"), "resume_corrected")
    for manifest in (reference, treated):
        assert set(manifest) == R.MANIFEST_FIELDS and R.validate_manifest_mode(manifest) == "resume"
        assert manifest["seed"] == 2031 and manifest["final_iteration"] == 598 and manifest["resume_iteration"] == 299
    assert (reference["fault"], reference["correction_id"], reference["config_source_fault"]) == ("NONE", None, "NONE")
    assert reference["load_run_regex"] == "^2026\\-10\\-14_00\\-00\\-00_baseline_p0c_s2031$"
    assert (treated["fault"], treated["correction_id"]) == ("F2", "restore_motor_torque_speed")
    assert treated["resume_checkpoint_path"] == arts["train_FAULT"]["checkpoint"]["path"]
    with pytest.raises(ValueError):  # 승인된 회복 방식 밖의 arm
        DC.manifest_for(tmp_path / "fix", _request("restart"), "resume_corrected")


def test_resume_training_configuration_allows_only_resume_arguments(monkeypatch, tmp_path):
    src = tmp_path / "src"
    (src / "params").mkdir(parents=True)
    env = "seed: 2031\nscene:\n  num_envs: 4096\n"
    agent = "seed: 2031\nnum_steps_per_env: 24\nmax_iterations: 300\nrun_name: h06_s2031\nresume: false\n"
    (src / "params/env.yaml").write_text(env, encoding="utf-8")
    (src / "params/agent.yaml").write_text(agent, encoding="utf-8")
    arts = {k: {"path": str(src / "params" / f), "sha256": DC.sha(src / "params" / f)}
            for k, f in (("environment", "env.yaml"), ("agent", "agent.yaml"))}
    begin = {"variant": "f2"}
    monkeypatch.setattr(E, "step_record", lambda tag, name: begin if name == "begin" else {"artifacts": arts})
    manifest = {"tag": DC.fix_tag("d"), "arm": "resume_corrected", "run_name": "d_fix_resume_corrected",
                "load_run_regex": "^x$", "checkpoint_regex": "^model_299\\.pt$", "max_iterations": 300,
                "source_env_config_path": arts["environment"]["path"], "source_agent_config_path": arts["agent"]["path"]}
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    resumed = agent.replace("h06_s2031", "d_fix_resume_corrected").replace("resume: false", "resume: true")
    resumed += "load_run: ^x$\nload_checkpoint: ^model_299\\.pt$\n"
    (run / "params/env.yaml").write_text(env, encoding="utf-8")
    (run / "params/agent.yaml").write_text(resumed, encoding="utf-8")
    DC.validate_training_configuration(run, manifest)
    (run / "params/env.yaml").write_text(env.replace("2031", "3031"), encoding="utf-8")
    with pytest.raises(ValueError):  # resume은 환경 설정(seed 포함)을 바꾸지 않는다
        DC.validate_training_configuration(run, manifest)
    (run / "params/env.yaml").write_text(env, encoding="utf-8")
    (run / "params/agent.yaml").write_text(resumed.replace("num_steps_per_env: 24", "num_steps_per_env: 48"),
                                           encoding="utf-8")
    with pytest.raises(ValueError):
        DC.validate_training_configuration(run, manifest)


def _resume_artifacts(tmp_path, monkeypatch):
    folder = tmp_path / "fix"
    training = folder / "resume_corrected" / "train"
    training.mkdir(parents=True)
    run_name = "d_fix_resume_corrected"
    isaaclab = tmp_path / "IsaacLab"
    run_dir = isaaclab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-14_00-00-00_{run_name}"
    (run_dir / "params").mkdir(parents=True)
    for name, text in (("model_598.pt", "ckpt"), ("params/env.yaml", "env"), ("params/agent.yaml", "agent")):
        (run_dir / name).write_text(text, encoding="utf-8")
    resume = {"path": str(tmp_path / "source/model_299.pt"), "sha256": "9" * 64}
    manifest = {"tag": "d_fix", "seed": 2031, "arm": "resume_corrected", "recovery_mode": "resume", "fault": "F2",
                "correction_id": "restore_motor_torque_speed", "source_commit_sha": EXPERIMENT,
                "protocol_sha256_lf": "p" * 64, "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "run_name": run_name,
                "final_iteration": 598, "isaaclab_root": str(isaaclab), "resume_checkpoint_path": resume["path"],
                "resume_checkpoint_sha256": resume["sha256"]}
    for name, value in (("manifest.json", manifest), ("correction.json", {"before_fault": {}})):
        (training / name).write_text(json.dumps(value), encoding="utf-8")
    rng = {"cpu_sha256": "1" * 64, "cuda_all_sha256": ["2" * 64], "cuda_device_count": 1}
    observed = {"environment_seed": 2031, "environment_num_envs": 4096, "action_scale": 0.25, "agent_seed": 2031,
                "agent_max_iterations": 300, "runner_num_steps_per_env": 24}
    versions = {"python": "3.10.15", "python_cache_tag": "cpython-310", "rsl_rl_distribution": "2.3.3",
                "torch_distribution": "2.7.0+cu128", "isaaclab_distribution": "0.41.3", "isaaclab_source_commit": "i"}
    source_state = {"iteration": 299, "policy_state_sha256": "5" * 64, "optimizer_state_sha256": "6" * 64}
    initial = {"schema": "training_correction_initial_v2", "recovery_mode": "resume",
               "resume_checkpoint": {**resume, "load_optimizer": True}, "resume_iteration": 299, "start_iteration": 299,
               "source_checkpoint_state": source_state, "std_reset": None, "init_at_random_ep_len": True,
               "full_environment_state_restored": False, "rng_state_restored_from_checkpoint": False,
               "rng_state_before_learn": rng, "observed": observed, "runtime_versions": versions, "log_dir": str(run_dir),
               "policy_state": {"sha256": "5" * 64}, "optimizer_state": {"sha256": "6" * 64}}
    (training / "initial.json").write_text(json.dumps(initial), encoding="utf-8")

    def artifact(path):
        return {"path": str(path), "sha256": DC.sha(path)}

    receipt = {**{k: manifest[k] for k in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id",
                                           "source_commit_sha", "protocol_sha256_lf", "task", "run_name",
                                           "final_iteration")},
               "completed": True, "schema": "training_correction_receipt_v2",
               **{f"{n}_sha256": DC.sha(training / f"{n}.json") for n in ("manifest", "initial", "correction")},
               "process_identity": {"process_id": 7, "process_creation_time_filetime": 99},
               "actual_control_steps": 7200, "environment_make_calls": 1, "aggregate_env_steps": 7200 * 4096,
               "budget": {"seed": 2031, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
                          "expected_control_steps": 7200},
               "resume_checkpoint": resume, "resume_iteration": 299, "start_iteration": 299, "checkpoint_load_calls": 1,
               "source_checkpoint_state": source_state, "std_reset_applied": False,
               "final_checkpoint_state": {"iteration": 598, "policy_state_sha256": "3" * 64,
                                          "optimizer_state_sha256": "4" * 64},
               "resume_scope": {"policy_optimizer_restored": True, "full_environment_state_restored": False,
                                "rng_state_restored_from_checkpoint": False, "fresh_initialization": False,
                                "action_std_reset": False},
               "rng_state_before_learn": rng, "observed": observed, "actual_action_scale": 0.25,
               "runtime_versions": versions, "checkpoint": artifact(run_dir / "model_598.pt"),
               "configs": {"environment": artifact(run_dir / "params/env.yaml"),
                           "agent": artifact(run_dir / "params/agent.yaml")}}
    protocol = {"correction_source_contract": {"runtime": {"version": [3, 10, 15], "cache_tag": "cpython-310"},
                                               "targets": {}},
                "rsl_rl_version": "2.3.3", "isaaclab_commit": "i"}
    execution = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
                 "exit_code": 0, "cleanup_verified": True, "owned_process_identities": {"7": 99}}
    monkeypatch.setattr(DC, "validate_manifest", lambda path: (json.loads(Path(path).read_text(encoding="utf-8")), protocol))
    monkeypatch.setattr(R, "validate_manifest_mode", lambda manifest: manifest["recovery_mode"])
    monkeypatch.setattr(DC, "validate_readback", lambda *a: None)
    monkeypatch.setattr(DC, "validate_training_configuration", lambda *a: None)
    return folder, training, receipt, execution, protocol, {"torch": "2.7.0+cu128", "isaaclab": "0.41.3"}


@pytest.mark.parametrize("key,value", [
    (None, None), ("checkpoint_load_calls", 0), ("resume_iteration", 0), ("std_reset_applied", True),
    ("start_iteration", 0), ("resume_checkpoint", {"path": "other", "sha256": "9" * 64}),
])
def test_verify_training_resume_checks(tmp_path, monkeypatch, key, value):
    folder, training, receipt, execution, protocol, versions = _resume_artifacts(tmp_path, monkeypatch)
    if key is not None:
        receipt[key] = value
    (training / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    if key is None:
        assert DC.verify_training(folder, "resume_corrected", execution, protocol, versions)["final_iteration"] == 598
    else:
        with pytest.raises(ValueError):
            DC.verify_training(folder, "resume_corrected", execution, protocol, versions)


def test_reference_and_judge_by_mode():
    none, fault = report("baseline_p0c_s2027"), report("h06_s2027")
    restart = DC.reference_for("restart", {}, none, bands())
    assert restart["valid"] and restart["report"] is none
    good = DC.reference_for("resume", {"clean_resume_reference": report("baseline_p0c_s2028")}, none, bands())
    assert good["valid"] and good["against_original_none"]["label"] == "healthy"
    bad = DC.reference_for("resume", {"clean_resume_reference": fault}, none, bands())
    assert not bad["valid"]  # 기준이 정상 정책 대비 healthy가 아니면 판정하지 않는다
    assert DC.judge(fault, none, bands())["label"] == "not_recovered"
    assert DC.judge(report("baseline_p0c_s2028"), none, bands())["label"] == "recovered"
