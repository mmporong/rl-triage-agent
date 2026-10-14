"""두 번째 task 단계 0(ANYmal-C 이식 관문): 평가·probe 구동기, 평가 프로토콜 파생, 관문 판정 행렬, 실행 흐름(GPU 없이)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import fixed_eval_task as FE  # noqa: E402
import probes_task as PT  # noqa: E402
import run_anymal_c_gate as G  # noqa: E402

U = G.U
ISAACLAB = Path.home() / "IsaacLab"
needs_isaaclab = pytest.mark.skipif(not (ISAACLAB / G.ISAACLAB_FILES["agent_cfg"]).is_file(), reason="로컬 Isaac Lab 소스가 없다")
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
P0C_PROBES = ROOT / "evals/results/p1a_probes_holdout_20261009/probes"
NAME = G.RUN_NAMES[1]


# --- 구동기 ---
def test_the_eval_driver_disables_only_the_listed_events_once_and_restores_the_parser(tmp_path, capsys):
    protocol = tmp_path / "p.json"
    protocol.write_text(json.dumps({"env_events_disabled": ["push_robot", "base_com"]}), encoding="utf-8")
    cfg = SimpleNamespace(events=SimpleNamespace(push_robot=object(), base_com=object(), add_base_mass=object()))
    utils = ModuleType("isaaclab_tasks.utils")
    utils.parse_env_cfg = lambda task, device=None, num_envs=None: cfg
    seen = {}

    def evaluate(args):
        seen["cfg"] = utils.parse_env_cfg(G.TASK, device="cuda:0", num_envs=1040)
        return "done"
    original = utils.parse_env_cfg
    assert FE.wrap(evaluate, utils)(SimpleNamespace(protocol=protocol)) == "done"
    assert utils.parse_env_cfg is original  # 호출이 끝나면 되돌린다
    assert seen["cfg"].events.push_robot is None and seen["cfg"].events.base_com is None
    assert seen["cfg"].events.add_base_mass is not None  # add_base_mass는 fixed_eval.py가 끈다
    lines = FE.parse(capsys.readouterr().out)
    assert lines == [{"task": G.TASK, "disabled": ["push_robot", "base_com"], "was_enabled": {"push_robot": True, "base_com": True},
                      "now_enabled": {"push_robot": False, "base_com": False}}]

    def twice(args):
        utils.parse_env_cfg(G.TASK)
        utils.parse_env_cfg(G.TASK)
    with pytest.raises(RuntimeError, match="두 번"):
        FE.wrap(twice, utils)(SimpleNamespace(protocol=protocol))
    assert utils.parse_env_cfg is original


def test_the_frozen_eval_reads_the_parser_inside_evaluate_and_main_calls_the_module_global():
    """감싸기가 동작하는 전제: fixed_eval.evaluate 안에서 parse_env_cfg를 import하고 main은 모듈 전역 evaluate를 부른다."""
    text = (ROOT / "evals/fixed_eval.py").read_text(encoding="utf-8")
    assert "    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg" in text
    assert "        evaluate(args)" in text and 'env_cfg = parse_env_cfg(proto["task"]' in text


def test_the_probe_driver_reads_the_task_from_jobs_and_the_frozen_probe_reads_the_global(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"task": G.TASK, "fault": "NONE", "seed": 2043, "jobs": []}), encoding="utf-8")
    assert PT.task_from_argv(["x", "--jobs", str(jobs), "--headless"]) == G.TASK
    jobs.write_text(json.dumps({"fault": "NONE"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        PT.task_from_argv(["x", "--jobs", str(jobs)])
    text = (ROOT / "evals/probes.py").read_text(encoding="utf-8")
    assert "env_cfg = parse_env_cfg(TASK," in text and '"task": TASK' in text and 'load_cfg_from_registry(TASK,' in text


def test_the_eval_protocol_keeps_the_fixed_eval_v1_condition_and_only_changes_task_and_events():
    base = U.read(ROOT / "bench/protocols/fixed_eval_v1.json")
    ours = U.read(G.EVAL_PROTOCOL)
    for key in ("eval_seed", "envs_per_condition", "horizon_steps", "step_dt", "episode_length_s", "command_grid",
                "policy_interface_from_run", "accounting"):
        assert ours[key] == base[key], key
    assert {k: v for k, v in ours["env_changes"].items() if k in base["env_changes"]} == base["env_changes"]
    assert set(ours["env_changes"]) - set(base["env_changes"]) == {"events.push_robot", "events.base_com"}
    assert ours["task"] == G.TASK and ours["name"] != base["name"] and ours["env_events_disabled"] == ["push_robot", "base_com"]


@needs_isaaclab
def test_local_isaaclab_sources_carry_the_registered_lines(tmp_path):
    sources = G.isaaclab_sources(ISAACLAB)
    assert set(sources) == set(G.ISAACLAB_FILES)
    fake = tmp_path / "lab"
    for key, relative in G.ISAACLAB_FILES.items():
        (fake / relative).parent.mkdir(parents=True, exist_ok=True)
        (fake / relative).write_bytes((ISAACLAB / relative).read_bytes())
    cfg = fake / G.ISAACLAB_FILES["agent_cfg"]
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("self.max_iterations = 300", "self.max_iterations = 1500"),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="agent_cfg"):
        G.isaaclab_sources(fake)


# --- 판정 행렬 ---
def _protocol() -> dict:
    return {**copy.deepcopy(G.FIXED), "source_files": {}, "isaaclab_sources": {},
            "eval_protocol_sha256_lf": U.sha(G.EVAL_PROTOCOL, True)}


def _report(source: str = "baseline_p0c_s2026", sha: str = "a" * 64) -> dict:
    report = U.read(P0C_RUNS / f"{source}.json")
    report.update(name=NAME, protocol=U.read(G.EVAL_PROTOCOL)["name"], task=G.TASK,
                  protocol_sha256_lf=U.sha(G.EVAL_PROTOCOL, True), action_scale=G.ACTION_SCALE,
                  checkpoint={"file": G.FINAL_CHECKPOINT, "sha256": sha})
    return report


def _probe_report(sha: str = "a" * 64) -> dict:
    report = U.read(P0C_PROBES / "baseline_p0c_s2026.json")
    report.update(name=NAME, task=G.TASK, seed=G.SEED, checkpoint={"file": G.FINAL_CHECKPOINT, "sha256": sha})
    return report


PARAMS = {"agent.yaml:seed": 2043, "env.yaml:seed": 2043, "env.yaml:scene.num_envs": 4096, "agent.yaml:max_iterations": 300,
          "env.yaml:actions.joint_pos.scale": 0.5}
EVAL_ENV = [{"task": G.TASK, "disabled": ["push_robot", "base_com"], "was_enabled": {"push_robot": True, "base_com": True},
             "now_enabled": {"push_robot": False, "base_com": False}}]


def _evidence(**changes) -> dict:
    evidence = {"run_name": NAME, "failure": None, "train_exit_code": 0, "checkpoint_sha256": "a" * 64,
                "first_checkpoint": True, "readback": [{"fault": "NONE"}], "params": dict(PARAMS), "eval_env": copy.deepcopy(EVAL_ENV),
                "report": _report(), "probe_report": _probe_report(), "train_wall_s": 900.0}
    evidence.update(changes)
    return evidence


def test_a_walking_policy_with_normal_absolute_probes_within_cost_passes():
    out = G.judge(_protocol(), _evidence())
    assert (out["outcome"], out["failed_checks"], out["failed_criteria"], out["prediction_match"]) == ("pass", [], [], True)
    assert out["gate"]["absolute_probes"] == dict.fromkeys(G.ABSOLUTE_PROBES, "normal")
    assert set(out["gate"]["ratio_probe_measurements"]) == {"P_value", "P_torque"} and out["gate"]["standstill_ratio"] < 0.5


@pytest.mark.parametrize(("change", "criterion"), [
    ({"report": _report("h02_s2026")}, "walks"),  # 추종하지 못하는 정책
    ({"train_wall_s": G.COST_MAX_S + 1}, "cost_within_plan"),
])
def test_gate_criteria_fail_without_invalidating(change, criterion):
    out = G.judge(_protocol(), _evidence(**change))
    assert out["outcome"] == "fail" and out["failed_criteria"] == [criterion] and out["invalid_cause"] is None
    assert out["wording"] == G.FIXED["wording"]["fail"]


def test_an_abnormal_absolute_probe_or_an_unmeasured_probe_fails_the_gate():
    probe = _probe_report()
    probe["probes"]["P_episode"]["measurement"]["timeout_ratio"] = 0.5
    out = G.judge(_protocol(), _evidence(probe_report=probe))
    assert out["failed_criteria"] == ["absolute_probes_normal"] and out["gate"]["absolute_probes"]["P_episode"] == "abnormal"
    probe = _probe_report()
    probe["probes"]["P_torque"].update(exit=1, measurement=None)  # 비 기준 probe가 안 재져도 관문은 어긋난다
    assert G.judge(_protocol(), _evidence(probe_report=probe))["failed_criteria"] == ["probes_measured"]


def _break(check: str) -> dict:
    if check == "pipeline_completed":
        return {"failure": {"stage": "fixed_evaluation", "error_type": "RuntimeError"}}
    if check == "training_exit_zero":
        return {"train_exit_code": 1}
    if check == "final_checkpoint_present":
        return {"first_checkpoint": False}
    if check == "training_readback":
        return {"readback": [{"fault": "F1"}]}
    if check == "params_expected":
        return {"params": {**PARAMS, "env.yaml:actions.joint_pos.scale": 0.25}}
    if check == "eval_env_readback":
        return {"eval_env": [{**EVAL_ENV[0], "now_enabled": {"push_robot": True, "base_com": False}}]}
    if check == "fixed_report_valid":
        return {"report": {**_report(), "task": "Isaac-Velocity-Flat-Unitree-Go2-v0"}}
    if check == "probe_report_bound":
        return {"probe_report": {**_probe_report(), "seed": 2026}}
    raise AssertionError(check)


@pytest.mark.parametrize("check", list(G.FIXED["validity_checks"]))
def test_each_validity_failure_is_invalid_with_its_cause(check):
    out = G.judge(_protocol(), _evidence(**_break(check)))
    assert out["outcome"] == "invalid" and check in out["failed_checks"] and out["gate"] is None
    assert out["invalid_cause"] == G.FIXED["invalid_causes"]["by_check"][out["failed_checks"][0]]


def test_an_event_that_was_never_enabled_does_not_pass_the_eval_readback():
    line = {**EVAL_ENV[0], "was_enabled": {"push_robot": False, "base_com": True}}
    out = G.judge(_protocol(), _evidence(eval_env=[line]))
    assert out["outcome"] == "invalid" and out["failed_checks"] == ["eval_env_readback"]


@needs_isaaclab
def test_the_external_force_event_is_a_zero_range_training_event_in_both_tasks():
    """push 결정 문장의 근거: base_external_force_torque는 기본 범위가 0이고, Go2·ANYmal-C는 PLAY 설정에서만 끈다."""
    base = ISAACLAB / G.ISAACLAB_FILES["velocity_env_cfg"]
    assert '"force_range": (0.0, 0.0),' in base.read_text(encoding="utf-8")
    for robot in ("go2", "anymal_c"):
        text = (base.parent / "config" / robot / "flat_env_cfg.py").read_text(encoding="utf-8")
        play = text.index("_PLAY(")
        assert "base_external_force_torque = None" not in text[:play] and "base_external_force_torque = None" in text[play:]


def test_a_later_stage_failure_reports_the_training_cause_first():
    out = G.judge(_protocol(), _evidence(failure={"stage": "params", "error_type": "KeyError"}, readback=[]))
    assert out["failed_checks"][0] == "pipeline_completed" and out["invalid_cause"] == "injection"
    out = G.judge(_protocol(), _evidence(failure={"stage": "training", "error_type": "RuntimeError"}, train_exit_code=124))
    assert out["invalid_cause"] == "infrastructure"


def test_the_report_validator_rejects_another_protocol_name_or_checkpoint():
    assert G.report_valid(_report(), _evidence(), _protocol())
    assert not G.report_valid({**_report(), "protocol": "fixed_eval_v1"}, _evidence(), _protocol())
    assert not G.report_valid(_report(sha="b" * 64), _evidence(), _protocol())
    assert not G.report_valid({**_report(), "protocol_sha256_lf": "0" * 64}, _evidence(), _protocol())


def test_the_protocol_records_the_push_decision_stop_rule_and_no_go2_comparison():
    fixed = G.FIXED
    assert "끈다" in fixed["fixed_evaluation"]["push_decision"] and "결과를 보기 전에" in fixed["fixed_evaluation"]["push_decision"]
    assert "단계 1·2를 하지 않는다" in fixed["stop_rule"] and fixed["seed"] == 2043
    assert set(fixed["gate_criteria"]) == {"walks", "probes_measured", "absolute_probes_normal", "cost_within_plan"}
    assert set(fixed["validity_checks"]) == set(fixed["invalid_causes"]["by_check"])
    text = json.dumps(fixed, ensure_ascii=False)
    assert "직접" not in text and "Go2와의 수치 비교가 아니다" in fixed["scope"]
    dynamic = {"source_files": {"a": "1"}, "isaaclab_sources": {}, "eval_protocol_sha256_lf": "e"}
    G.check_protocol({**copy.deepcopy(fixed), **dynamic}, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        G.check_protocol({**copy.deepcopy(fixed), **dynamic, "seed": 2044}, dynamic)


# --- 실행 흐름(GPU 없이 소유 Job을 대신한다) ---
@pytest.fixture()
def lab(tmp_path, monkeypatch):
    import run_runtime_readback_smoke as P
    fake = tmp_path / "IsaacLab"
    protocol = _protocol()
    protocol_path = tmp_path / "gate.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    for name, value in (("RESULTS", tmp_path / "results"), ("PRIVATE", tmp_path / "private"), ("RUNS_DIR", tmp_path / "runs"),
                        ("PROTOCOL", protocol_path), ("ROOT", tmp_path)):
        monkeypatch.setattr(G, name, value)
    monkeypatch.setattr(G, "EVAL_PROTOCOL", ROOT / "bench/protocols/fixed_eval_v1_anymal_c.json")
    state = {"calls": [], "gpu_busy": False, "gpu_processes": [], "timeout": set(), "train_exit": 0, "readback": True,
             "eval_source": "baseline_p0c_s2026", "verify_calls": 0}

    def verify(_isaaclab):
        state["verify_calls"] += 1
        return protocol, "0" * 40
    monkeypatch.setattr(G, "verify", verify)
    monkeypatch.setattr(U, "_admission", lambda: {"busy": state["gpu_busy"], "processes": state["gpu_processes"],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(U, "_process_created", lambda _pid: 1)
    monkeypatch.setattr(U.time, "sleep", lambda _s: None)

    def run_owned(folder, args, cwd, watchdog_s):
        name, stdout, code = Path(args[1]).name, "", 0
        state["calls"].append(name)
        if name == "train_with_fault.py":
            assert os.environ["TRIAGE_FAULT"] == "NONE" and args[args.index("--task") + 1] == G.TASK
            run_name = args[args.index("--run_name") + 1]
            run_dir = fake / "logs/rsl_rl/anymal_c_flat" / f"2026-10-15_00-00-00_{run_name}"
            (run_dir / "params").mkdir(parents=True)
            (run_dir / "params/env.yaml").write_text("seed: 2043\nscene:\n  num_envs: 4096\nactions:\n  joint_pos:\n    scale: 0.5\n",
                                                     encoding="utf-8")
            (run_dir / "params/agent.yaml").write_text("seed: 2043\nmax_iterations: 300\n", encoding="utf-8")
            (run_dir / G.FIRST_CHECKPOINT).write_bytes(b"zero")
            (run_dir / G.FINAL_CHECKPOINT).write_bytes(b"final")
            stdout = '[hidden_fault] {"fault": "NONE"}\n' if state["readback"] else ""
            code = state["train_exit"]
        elif name == "extract_tb.py":
            Path(args[3]).write_text(json.dumps({"series": {"Train/mean_reward": [1.0, 2.0]}}), encoding="utf-8")
        elif name == "fixed_eval_task.py":
            assert Path(args[args.index("--protocol") + 1]) == G.EVAL_PROTOCOL
            job = U.read(Path(args[args.index("--jobs") + 1]))["jobs"][0]
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({**_report(state["eval_source"], U.sha(Path(job["checkpoint"]))), "name": job["name"]}),
                           encoding="utf-8")
            stdout = FE.PREFIX + json.dumps(EVAL_ENV[0]) + "\n"
        else:
            jobs = U.read(Path(args[args.index("--jobs") + 1]))
            assert name == "probes_task.py" and jobs["task"] == G.TASK and jobs["fault"] == "NONE" and jobs["seed"] == 2043
            job = jobs["jobs"][0]
            assert job["probes"] == list(G.PROBE_ORDER)
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({**_probe_report(U.sha(Path(job["checkpoint"]))), "name": job["name"]}), encoding="utf-8")
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        timed_out = ("training" if name == "train_with_fault.py" else name) in state["timeout"]
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out, "error_type": None,
                "exit_code": 124 if timed_out else code, "cleanup_verified": True, "owned_process_identities": {"4242": 1},
                "wall_s": 950.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake, state=state)


TAG = "anymal_c_gate_20261015"


def test_execute_trains_evaluates_measures_probes_and_writes_once(lab):
    out = G.execute(TAG, lab.path)
    assert out["outcome"] == "pass" and out["failed_criteria"] == [] and out["failure"] is None
    assert lab.state["calls"] == ["train_with_fault.py", "extract_tb.py", "fixed_eval_task.py", "probes_task.py"]
    result = U.read(G.RESULTS / TAG / "result.json")
    assert result["train_wall_per_iteration_s"] == round(950.0 / 300, 3) and result["model_calls"] == 0
    assert result["seed_evidence"]["hits"] == [] and result["eval_env"] == EVAL_ENV and lab.state["verify_calls"] == 3
    text = json.dumps(result)
    assert "C:" not in text and "LIMMM" not in text and "TRIAGE_FAULT" not in os.environ
    with pytest.raises(FileExistsError):
        G.execute(TAG + "b", lab.path)


def test_a_busy_gpu_or_a_used_seed_launches_nothing(lab):
    lab.state["gpu_busy"] = True
    assert G.execute(TAG, lab.path)["status"] == "gpu_busy_no_launch" and lab.state["calls"] == []
    lab.state["gpu_busy"] = False
    other = lab.path / "logs/rsl_rl/unitree_go2_flat/2026-10-01_00-00-00_x_s2043"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="seed 2043"):
        G.execute(TAG, lab.path)
    assert lab.state["calls"] == []


def test_a_missing_training_readback_skips_evaluation_and_probes(lab):
    lab.state["readback"] = False
    out = G.execute(TAG, lab.path)
    assert (out["outcome"], out["invalid_cause"]) == ("invalid", "injection")
    assert lab.state["calls"] == ["train_with_fault.py", "extract_tb.py"]


def test_a_non_walking_policy_fails_the_gate(lab):
    lab.state["eval_source"] = "h02_s2026"
    out = G.execute(TAG, lab.path)
    assert out["outcome"] == "fail" and out["failed_criteria"] == ["walks"]
    with pytest.raises((ValueError, FileExistsError)):  # 처음 나온 fail이 최종이다
        G.execute(TAG + "b", lab.path, retry_of=TAG)


def test_a_training_timeout_allows_one_retry_under_a_new_run_name(lab):
    lab.state["timeout"] = {"training"}
    first = G.execute(TAG, lab.path)
    assert (first["outcome"], first["invalid_cause"]) == ("invalid", "infrastructure")
    lab.state["timeout"] = set()
    with pytest.raises(ValueError, match="시도 2"):
        G.execute(TAG + "b", lab.path, retry_of="anymal_c_gate_20261099")
    second = G.execute(TAG + "b", lab.path, retry_of=TAG)
    result = U.read(G.RESULTS / (TAG + "b") / "result.json")
    assert second["outcome"] == "pass" and result["run_name"] == G.RUN_NAMES[2] and result["seed_evidence"]["allowed_hits"]


def test_an_owned_process_left_on_the_gpu_is_invalid(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    out = G.execute(TAG, lab.path)
    assert (out["outcome"], out["failure"]["stage"]) == ("invalid", "postflight")
