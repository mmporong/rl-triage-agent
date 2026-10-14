"""X-R1: 옵티마이저 외부 결함 R1 주입기(가짜 PPO), smoke 판정 행렬, 프로토콜·고리 예측, 실행 흐름(GPU 없이)."""
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

import optimizer_faults as O  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
import run_upstream_r1_smoke as R  # noqa: E402
import run_upstream_smoke as U  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
needs_isaaclab = pytest.mark.skipif(not (ISAACLAB / R.SOURCE_FILES["ppo"]).is_file(),
                                    reason="로컬 Isaac Lab v2.1.1·rsl_rl 2.3.3 소스가 없다")
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
NAMES = R.RUN_NAMES[1]


# --- 주입기(가짜 PPO) ---
class _Storage:
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        for _ in range(num_epochs * num_mini_batches):
            yield object()


class _Optimizer:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


def _ppo_class():
    class PPO:
        def __init__(self, policy=None, num_learning_epochs=1, num_mini_batches=1, schedule="fixed", desired_kl=0.01):
            self.policy = SimpleNamespace(is_recurrent=False)
            self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
            self.schedule, self.desired_kl = schedule, desired_kl
            self.storage, self.optimizer = _Storage(), _Optimizer()

        def update(self):
            for _ in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
                self.optimizer.step()
            return {"value_function": 1.0}
    PPO.__module__, PPO.__qualname__ = O.PPO_MODULE, "PPO"
    return PPO


def _make(cls):
    return cls(num_learning_epochs=5, num_mini_batches=4, schedule="adaptive")


def _lines(capsys) -> list[dict]:
    return O.parse(capsys.readouterr().out)


@pytest.mark.parametrize(("fault", "effective"), [("R1", 1), ("NONE", 5)])
def test_patched_ppo_keeps_the_configured_epochs_but_runs_one_pass_for_r1(capsys, fault, effective):
    cls = _ppo_class()
    O.patch_ppo(fault, cls)
    ppo = _make(cls)
    assert ppo.update() == {"value_function": 1.0} and ppo.optimizer.calls == effective * 4
    ppo.update()  # 두 번째 update는 세지 않고 확인 줄도 없다
    lines = _lines(capsys)
    assert [line["event"] for line in lines] == ["construct", "first_update"]
    assert lines[0] | {"class": None} == {"event": "construct", "fault": fault, "class": None, "configured_epochs": 5,
                                          "effective_epochs": effective, "num_mini_batches": 4, "schedule": "adaptive",
                                          "desired_kl": 0.01}
    assert lines[1] == {"event": "first_update", "fault": fault, "minibatches": effective * 4,
                        "optimizer_steps": effective * 4, "expected": effective * 4}
    assert O.readback_ok(lines, fault, 5, 4) and not O.readback_ok(lines, "R1" if fault == "NONE" else "NONE", 5, 4)
    assert "mini_batch_generator" not in vars(ppo.storage) and "step" not in vars(ppo.optimizer)  # 셈 도구는 걷힌다


def test_readback_refuses_missing_extra_broken_or_inconsistent_lines(capsys):
    cls = _ppo_class()
    O.patch_ppo("R1", cls)
    _make(cls).update()
    lines = _lines(capsys)
    assert O.readback_ok(lines, "R1", 5, 4)
    assert not O.readback_ok(lines[:1], "R1", 5, 4) and not O.readback_ok(lines + lines[1:], "R1", 5, 4)
    assert not O.readback_ok([lines[0], None], "R1", 5, 4)
    for key, value in (("optimizer_steps", 20), ("minibatches", 20), ("expected", 20)):
        assert not O.readback_ok([lines[0], {**lines[1], key: value}], "R1", 5, 4)
    assert not O.readback_ok([{**lines[0], "configured_epochs": 1}, lines[1]], "R1", 5, 4)
    assert not O.readback_ok([{**lines[0], "class": "x.PPO"}, lines[1]], "R1", 5, 4)
    assert O.parse("[optimizer_fault] {broken\nother\n") == [None]


def test_arm_patches_the_ppo_class_only_at_the_first_env_and_refuses_unknown_faults(monkeypatch):
    with pytest.raises(SystemExit):
        O.arm("R2")
    cls = _ppo_class()
    ppo_module = ModuleType(O.PPO_MODULE)
    ppo_module.PPO = cls
    gym = ModuleType("gymnasium")
    gym.make = lambda *args, **kwargs: ("env", args)
    imported = []
    real_import = __import__("importlib").import_module

    def import_module(name, *args):
        imported.append(name)
        return ppo_module if name == O.PPO_MODULE else real_import(name, *args)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setattr("importlib.import_module", import_module)
    original_init = cls.__init__
    O.arm("R1")
    assert cls.__init__ is original_init and imported == []  # 앱 실행 전에는 rsl_rl·torch를 부르지 않는다
    assert gym.make("Go2") == ("env", ("Go2",)) and cls.__init__ is not original_init and imported == [O.PPO_MODULE]
    patched = cls.__init__
    gym.make("again")
    assert cls.__init__ is patched and imported == [O.PPO_MODULE]


# --- 결합 소스 ---
def test_the_registered_ppo_and_go2_lines_match_the_parsers():
    ppo = "\n".join(["x"] * 104 + ["        self.num_learning_epochs = num_learning_epochs"] + ["x"] * 99
                    + ["            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
                       "        else:",
                       "            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)"]
                    + ["x"] * 197 + ["        num_updates = self.num_learning_epochs * self.num_mini_batches"])
    assert R.epoch_lines(ppo) == R.PPO_EPOCH_LINES
    assert R.epoch_lines(ppo + "\n        self.num_learning_epochs = 1") != R.PPO_EPOCH_LINES  # 새 사용처는 거부 대상


@needs_isaaclab
def test_local_isaaclab_and_rsl_rl_sources_bind_and_a_changed_line_is_refused(tmp_path):
    sources = R.isaaclab_sources(ISAACLAB)
    assert set(sources) == set(R.SOURCE_FILES) and all(len(v["sha256_lf"]) == 64 for v in sources.values())
    fake = tmp_path / "lab"
    for key, relative in R.SOURCE_FILES.items():
        (fake / relative).parent.mkdir(parents=True, exist_ok=True)
        (fake / relative).write_bytes((ISAACLAB / relative).read_bytes())
    cfg = fake / R.SOURCE_FILES["go2_agent_cfg"]
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("num_learning_epochs=5,", "num_learning_epochs=1,"), encoding="utf-8")
    with pytest.raises(ValueError, match="Go2 PPO 설정"):
        R.isaaclab_sources(fake)


# --- 판정 행렬 ---
def _protocol() -> dict:
    return {**copy.deepcopy(R.FIXED), "source_files": {}, "isaaclab_sources": {}, "reference_inputs": R.reference_inputs()}


def _readback(fault: str) -> list[dict]:
    effective = R.CONFIGURED_EPOCHS if fault == "NONE" else 1
    steps = effective * R.NUM_MINI_BATCHES
    return [{"event": "construct", "fault": fault, "class": f"{O.PPO_MODULE}.PPO", "configured_epochs": 5,
             "effective_epochs": effective, "num_mini_batches": 4, "schedule": "adaptive", "desired_kl": 0.01},
            {"event": "first_update", "fault": fault, "minibatches": steps, "optimizer_steps": steps, "expected": steps}]


def _report(source_run: str, name: str, checkpoint_sha: str) -> dict:
    report = U.read(P0C_RUNS / f"{source_run}.json")
    report["name"] = name
    report["checkpoint"] = {"file": R.FINAL_CHECKPOINT, "sha256": checkpoint_sha}
    return report


def _evidence(fault_source: str = "h02_s2026") -> dict:
    runs = {"reference": {"run_name": NAMES["reference"], "train_exit_code": 0, "checkpoint_sha256": "a" * 64,
                          "readback": _readback("NONE")},
            "fault": {"run_name": NAMES["fault"], "train_exit_code": 0, "checkpoint_sha256": "b" * 64,
                      "readback": _readback("R1")}}
    return {"failure": None, "runs": runs, "params_differences": ["agent.yaml:run_name"],
            "epoch_params": {arm: {R.EPOCH_PARAM: 5} for arm in R.ARMS},
            "reports": {"reference": _report("baseline_p0c_s2026", NAMES["reference"], "a" * 64),
                        "fault": _report(fault_source, NAMES["fault"], "b" * 64)}}


def test_valid_runs_are_judged_against_the_same_day_reference_with_the_p0c_bands():
    protocol = _protocol()
    present = R.judge(protocol, _evidence("h02_s2026"))  # 추종하지 못하는 정책
    assert (present["outcome"], present["prediction_match"], present["failed_checks"]) == ("failure_present", True, [])
    assert "err_xy_ratio" in present["verdict"]["failed"] and present["standstill_ratio"]["fault"] > 0.9
    absent = R.judge(protocol, _evidence("baseline_p0c_s2026"))  # 기준과 같은 행동
    assert (absent["outcome"], absent["prediction_match"]) == ("failure_absent", False)
    assert absent["wording"] == protocol["wording"]["failure_absent"]


def _break(evidence: dict, check: str) -> dict:
    e = copy.deepcopy(evidence)
    reference, fault = e["runs"]["reference"], e["runs"]["fault"]
    if check == "pipeline_completed":
        e["failure"] = {"stage": "fixed_evaluation", "error_type": "RuntimeError"}
    elif check == "reference_training_exit_zero":
        reference["train_exit_code"] = 1
    elif check == "reference_final_checkpoint_present":
        reference["checkpoint_sha256"] = None
    elif check == "reference_injection_readback":
        reference["readback"] = _readback("R1")  # 정상 학습에 결함이 걸렸다
    elif check == "training_exit_zero":
        fault["train_exit_code"] = 1
    elif check == "final_checkpoint_present":
        fault["checkpoint_sha256"] = None
    elif check == "injection_readback":
        fault["readback"] = _readback("NONE")  # 결함이 걸리지 않았다
    elif check == "params_match_reference":
        e["params_differences"] = ["agent.yaml:algorithm.num_mini_batches", "agent.yaml:run_name"]
    elif check == "configured_epochs_in_params":
        e["epoch_params"]["fault"][R.EPOCH_PARAM] = 1
    elif check == "reference_walks":
        e["reports"]["reference"] = _report("h02_s2027", NAMES["reference"], "a" * 64)
    elif check == "fixed_report_valid":
        e["reports"]["fault"]["action_scale"] = 0.5
    elif check == "runtime_matches_reference":
        e["reports"]["fault"]["versions"] = {**e["reports"]["fault"]["versions"], "torch": "0.0"}
    return e


@pytest.mark.parametrize("check", [c for c in R.FIXED["validity_checks"] if c != "verdict_determined"])
def test_each_validity_failure_makes_the_smoke_invalid_with_its_cause(check):
    out = R.judge(_protocol(), _break(_evidence(), check))
    assert out["outcome"] == "invalid" and check in out["failed_checks"] and out["prediction_match"] is None
    assert out["invalid_cause"] == R.FIXED["invalid_causes"]["by_check"][out["failed_checks"][0]]


def test_an_undetermined_band_verdict_is_invalid(monkeypatch):
    monkeypatch.setattr(R.RC, "verdict", lambda _c, _bands: {"label": "undetermined", "failed": []})
    out = R.judge(_protocol(), _evidence())
    assert out["checks"]["verdict_determined"] is False and out["failed_checks"] == ["verdict_determined"]
    assert (out["outcome"], out["invalid_cause"]) == ("invalid", "evaluation")


def test_a_post_training_stage_failure_reports_the_training_cause_first():
    evidence = _break(_break(_evidence(), "injection_readback"), "pipeline_completed")
    out = R.judge(_protocol(), evidence)
    assert out["failed_checks"][0] == "pipeline_completed" and out["invalid_cause"] == "injection"
    training = _break(_evidence(), "pipeline_completed")
    training["failure"]["stage"] = "fault_training"
    assert R.judge(_protocol(), _break(training, "training_exit_zero"))["invalid_cause"] == "infrastructure"


def test_missing_evidence_is_invalid_not_a_crash():
    out = R.judge(_protocol(), {"failure": {"stage": "reference_training", "error_type": "RuntimeError"}})
    assert out["outcome"] == "invalid" and out["invalid_cause"] == "infrastructure" and out["verdict"] is None


def test_telemetry_summary_and_the_auxiliary_learning_rate_record():
    telemetry = {"series": {"Loss/learning_rate": [1e-3, float("nan"), 3e-3, 2e-3], "Train/mean_reward": [1.0]}}
    summary = R.telemetry_summary(telemetry)
    assert summary["Loss/learning_rate"] == {"first": 1e-3, "last": 2e-3, "min": 1e-3, "max": 3e-3, "median": 2e-3,
                                             "nonfinite": 1}
    reference = R.telemetry_summary({"series": {"Loss/learning_rate": [5e-4, 6e-4, 5.5e-4]}})
    assert R.auxiliary({"reference": reference, "fault": summary}) == {
        "learning_rate_median": {"reference": 5.5e-4, "fault": 2e-3}, "fault_higher": True}
    assert R.auxiliary({"reference": None, "fault": summary})["fault_higher"] is None


# --- 프로토콜 내용 ---
def test_the_loop_stage_predictions_match_the_frozen_loop_simulation():
    stage = R.FIXED["loop_stage"]
    predicted = {probe: spec["prediction"] for probe, spec in stage["probe_predictions"].items()}
    assert set(predicted) == set(L.PROBES)
    for strategy in ("fixed", "exhaustive"):
        assert L.simulate(strategy, predicted)["status"] == stage["predicted_conclusion"][strategy]
    flips = {"P_torque": "P_torque abnormal(나머지 예측대로)", "P_episode": "P_episode abnormal(나머지 예측대로)",
             "P_value": "P_value abnormal(나머지 예측대로)"}
    scenarios = {row["if"]: row for row in stage["scenarios"]}
    for probe, label in flips.items():
        for strategy in ("fixed", "exhaustive"):
            run = L.simulate(strategy, {**predicted, probe: "abnormal"})
            assert f"{run['status']} {run['conclusion']}" == scenarios[label][strategy]
    assert L.PROBES["P_value"]["expect"]["optimizer"] == "abnormal" and L.THRESHOLDS["P_value"] == ("critic_change", "ratio_below", 0.1)


def test_the_protocol_records_sources_competing_prediction_and_a_fault_hidden_from_params():
    fixed = R.FIXED
    assert fixed["prediction"]["outcome"] == "failure_present"
    assert fixed["prediction"]["competing"]["outcome"] == "failure_absent"
    assert fixed["source"]["issue"]["url"].endswith("/issues/38") and "없음" in fixed["source"]["issue"]["maintainer_response"]
    assert '("num_learning_epochs", None),' in fixed["source"]["code"]["legacy_runner_mapping"]["excerpts"]
    assert fixed["mapping"]["effect"]["gradient_steps_per_iteration"] == [20, 4] and fixed["seed"] == 2042
    assert set(fixed["validity_checks"]) == set(fixed["invalid_causes"]["by_check"])
    assert set(fixed["outcomes"]) == set(fixed["wording"]) == {"failure_present", "failure_absent", "invalid"}
    assert O.FAULTS["R1"]["category"] == fixed["fault"]["category"] == "optimizer"
    for text in json.dumps(fixed, ensure_ascii=False).split("다."):
        assert "직접" not in text


def test_protocol_changes_are_refused(monkeypatch, tmp_path):
    dynamic = {"source_files": {"a": "1"}, "isaaclab_sources": {}, "reference_inputs": {}}
    protocol = {**copy.deepcopy(R.FIXED), **dynamic}
    R.check_protocol(protocol, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        R.check_protocol({**protocol, "seed": 2043}, dynamic)
    with pytest.raises(ValueError, match="결합한"):
        R.check_protocol(protocol, {**dynamic, "source_files": {"a": "2"}})
    with pytest.raises(ValueError, match="필드"):
        R.check_protocol({**protocol, "extra": 1}, dynamic)


# --- 실행 흐름(GPU 없이 소유 Job 실행을 대신한다) ---
@pytest.fixture()
def lab(tmp_path, monkeypatch):
    """가짜 Isaac Lab·결과 폴더·GPU·소유 Job. state로 시나리오별 실패를 넣는다."""
    import run_runtime_readback_smoke as P
    fake_lab = tmp_path / "IsaacLab"
    protocol = _protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    for name, value in (("RESULTS", tmp_path / "results"), ("PRIVATE", tmp_path / "private"),
                        ("RUNS_DIR", tmp_path / "runs"), ("PROTOCOL", protocol_path)):
        monkeypatch.setattr(R, name, value)
    monkeypatch.setattr(R, "ROOT", tmp_path)  # 흔적 검사가 실제 저장소를 보지 않게 한다
    state = {"calls": [], "envs": [], "gpu_busy": False, "verify_calls": 0, "timeout": set(), "train_exit": {},
             "readback": {}, "epochs": {}, "tamper_checkpoint": False, "gpu_processes": [], "eval_source": "h02_s2026"}

    def verify(_isaaclab):
        state["verify_calls"] += 1
        return protocol, "0" * 40
    monkeypatch.setattr(R, "verify", verify)
    monkeypatch.setattr(U, "_admission", lambda: {"busy": state["gpu_busy"], "processes": state["gpu_processes"],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(U, "_process_created", lambda _pid: 1)
    monkeypatch.setattr(U.time, "sleep", lambda _s: None)

    def run_owned(folder, args, cwd, watchdog_s):
        name = Path(args[1]).name
        stdout, exit_code = "", 0
        if name == "train_with_optimizer_fault.py":
            fault = os.environ["TRIAGE_OPTIMIZER_FAULT"]
            arm = "reference" if fault == "NONE" else "fault"
            state["calls"].append(f"train:{fault}")
            assert watchdog_s == R.WATCHDOG_S["training"] and args[args.index("--seed") + 1] == "2042"
            run_name = args[args.index("--run_name") + 1]
            run_dir = fake_lab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-14_00-00-00_{run_name}"
            (run_dir / "params").mkdir(parents=True)
            (run_dir / "params/env.yaml").write_text("seed: 2042\nscene:\n  num_envs: 4096\n", encoding="utf-8")
            epochs = state["epochs"].get(arm, 5)
            (run_dir / "params/agent.yaml").write_text(
                f"seed: 2042\nrun_name: {run_name}\nalgorithm:\n  num_learning_epochs: {epochs}\n  num_mini_batches: 4\n",
                encoding="utf-8")
            (run_dir / R.FINAL_CHECKPOINT).write_bytes(f"checkpoint {run_name}".encode())
            lines = state["readback"].get(arm, _readback(fault))
            stdout = "".join(O.PREFIX + json.dumps(line) + "\n" for line in lines)
            exit_code = state["train_exit"].get(arm, 0)
        elif name == "extract_tb.py":
            state["calls"].append("extract")
            lr = [5e-4] * R.MAX_ITERATIONS if "none" in args[3] else [2e-3] * R.MAX_ITERATIONS
            Path(args[3]).write_text(json.dumps({"series": {"Loss/learning_rate": lr}}), encoding="utf-8")
        else:
            state["calls"].append("fixed_eval")
            jobs = U.read(Path(args[args.index("--jobs") + 1]))["jobs"]
            assert [job["name"] for job in jobs] == [NAMES["reference"], NAMES["fault"]] or "_a2" in jobs[0]["name"]
            for job, source in zip(jobs, ("baseline_p0c_s2026", state["eval_source"])):
                out = Path(job["output"])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(_report(source, job["name"], U.sha(Path(job["checkpoint"])))), encoding="utf-8")
                if state["tamper_checkpoint"]:
                    Path(job["checkpoint"]).write_bytes(b"changed during evaluation")
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        key = "training" if name == "train_with_optimizer_fault.py" else name
        timed_out = key in state["timeout"]
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out,
                "error_type": None, "exit_code": 124 if timed_out else exit_code, "cleanup_verified": True,
                "owned_process_identities": {"4242": 1}, "wall_s": 1.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake_lab, state=state)


def _result(tag: str) -> dict:
    return U.read(R.RESULTS / tag / "result.json")


def test_execute_trains_both_arms_evaluates_once_and_writes_once(lab):
    with pytest.raises(ValueError, match="tag"):
        R.execute("r1", lab.path)
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert out["outcome"] == "failure_present" and out["failed_checks"] == [] and out["failure"] is None
    assert lab.state["calls"] == ["train:NONE", "train:R1", "extract", "extract", "fixed_eval"]
    result = _result("upstream_r1_smoke_20261015")
    assert result["params_differences"] == ["agent.yaml:run_name"] and result["model_calls"] == 0
    assert result["run_names"] == NAMES and result["seed_evidence"]["hits"] == []
    assert result["auxiliary"] == {"learning_rate_median": {"reference": 5e-4, "fault": 2e-3}, "fault_higher": True}
    assert result["runs"]["fault"]["readback"] == _readback("R1") and lab.state["verify_calls"] == 3
    text = json.dumps(result)
    assert "C:" not in text and "LIMMM" not in text and "TRIAGE_OPTIMIZER_FAULT" not in os.environ
    with pytest.raises(FileExistsError):
        R.execute("upstream_r1_smoke_20261015", lab.path)
    with pytest.raises(FileExistsError):  # 같은 시도는 tag를 바꿔도 한 번
        R.execute("upstream_r1_smoke_20261015b", lab.path)


def test_a_busy_gpu_launches_and_consumes_nothing(lab):
    lab.state["gpu_busy"] = True
    assert R.execute("upstream_r1_smoke_20261015", lab.path)["status"] == "gpu_busy_no_launch"
    assert lab.state["calls"] == [] and not R.PRIVATE.exists()


def test_a_used_seed_is_refused_before_launch(lab):
    R.RUNS_DIR.mkdir(parents=True)
    (R.RUNS_DIR / "other_s2042.telemetry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="seed 2042"):
        R.execute("upstream_r1_smoke_20261015", lab.path)
    assert lab.state["calls"] == []


def test_a_broken_reference_readback_skips_the_fault_training_and_the_evaluation(lab):
    lab.state["readback"]["reference"] = _readback("R1")
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert (out["outcome"], out["invalid_cause"]) == ("invalid", "injection") and lab.state["calls"] == ["train:NONE"]


@pytest.mark.parametrize(("setup", "cause"), [
    (lambda s: s["train_exit"].update(fault=1), "training_process"),
    (lambda s: s["readback"].update(fault=_readback("NONE")), "injection"),
    (lambda s: s["epochs"].update(fault=1), "comparability"),
])
def test_a_fault_side_training_failure_skips_the_evaluation(lab, setup, cause):
    setup(lab.state)
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert (out["outcome"], out["invalid_cause"]) == ("invalid", cause) and "fixed_eval" not in lab.state["calls"]


def test_a_training_timeout_allows_exactly_one_retry_under_new_run_names(lab):
    lab.state["timeout"] = {"training"}
    first = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert (first["outcome"], first["invalid_cause"], first["failure"]["stage"]) == ("invalid", "infrastructure",
                                                                                      "reference_training")
    with pytest.raises(ValueError, match="시도 2"):
        R.execute("upstream_r1_smoke_20261015b", lab.path, retry_of="upstream_r1_smoke_20261099")
    lab.state["timeout"] = set()
    second = R.execute("upstream_r1_smoke_20261015b", lab.path, retry_of="upstream_r1_smoke_20261015")
    result = _result("upstream_r1_smoke_20261015b")
    assert second["outcome"] == "failure_present" and result["run_names"] == R.RUN_NAMES[2] and result["attempt"] == 2
    assert result["seed_evidence"]["hits"] == [] and result["seed_evidence"]["allowed_hits"]
    with pytest.raises(FileExistsError):  # 처음 나온 유효 결과가 최종이다
        R.execute("upstream_r1_smoke_20261015c", lab.path, retry_of="upstream_r1_smoke_20261015")


def test_a_valid_result_cannot_be_retried(lab):
    R.execute("upstream_r1_smoke_20261015", lab.path)
    with pytest.raises((ValueError, FileExistsError)):
        R.execute("upstream_r1_smoke_20261015b", lab.path, retry_of="upstream_r1_smoke_20261015")


def test_a_checkpoint_changed_during_evaluation_is_invalid(lab):
    lab.state["tamper_checkpoint"] = True
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]["stage"]) == ("invalid", "infrastructure", "postflight")


def test_an_owned_process_left_on_the_gpu_is_invalid(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert (out["outcome"], out["failure"]["stage"]) == ("invalid", "postflight")


def test_the_failure_absent_path(lab):
    lab.state["eval_source"] = "baseline_p0c_s2026"
    out = R.execute("upstream_r1_smoke_20261015", lab.path)
    assert out["outcome"] == "failure_absent"
    assert _result("upstream_r1_smoke_20261015")["prediction_match"] is False
