"""X-U1: 외부 결함 U1 주입 본문의 의미 대조, smoke 판정 행렬, 프로토콜 결합, 실행 흐름(GPU 없이)."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import run_upstream_smoke as S  # noqa: E402
import upstream_faults as U  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
REWARDS = ISAACLAB / S.ISAACLAB_FILES["rewards"]
needs_isaaclab = pytest.mark.skipif(not REWARDS.is_file(), reason="로컬 Isaac Lab v2.1.1 소스가 없다")
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
needs_reference = pytest.mark.skipif(not (ROOT / "bench/private/baseline_p0c_s2026.meta.json").is_file(),
                                     reason="비공개 P0-C 기준 실행 meta가 없다")
FAKE_TORCH = SimpleNamespace(square=np.square, exp=np.exp, zeros_like=np.zeros_like,
                             sum=lambda x, dim: np.sum(x, axis=dim))
RUN_NAME = S.RUN_NAMES[1]


class _Cfg:
    def __init__(self, name):
        self.name = name


def _env(lin, ang, command):
    data = SimpleNamespace(root_lin_vel_b=np.array(lin, dtype=float), root_ang_vel_b=np.array(ang, dtype=float))
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                           command_manager=SimpleNamespace(get_command=lambda name: np.array(command, dtype=float)))


def _isaaclab_originals() -> dict:
    """로컬 v2.1.1 rewards.py에서 4함수 정의만 꺼내 numpy 가짜 torch로 실행할 수 있게 만든다."""
    tree = ast.parse(REWARDS.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in U.SWAPS["U1"]]
    namespace = {"torch": FAKE_TORCH, "SceneEntityCfg": _Cfg, "ManagerBasedRLEnv": object, "RigidObject": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REWARDS), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in U.SWAPS["U1"]}


def _replacement(name: str):
    return FunctionType(U.SWAPS["U1"][name].__code__, {"torch": FAKE_TORCH}, name)


CALLS = {"lin_vel_z_l2": {}, "ang_vel_xy_l2": {},
         "track_lin_vel_xy_exp": {"std": 0.5, "command_name": "base_velocity"},
         "track_ang_vel_z_exp": {"std": 0.5, "command_name": "base_velocity"}}


@needs_isaaclab
def test_replacements_equal_the_isaaclab_terms_with_zero_body_velocity():
    originals = _isaaclab_originals()
    command = [[0.8, -0.3, 0.6], [0.0, 0.0, 0.0]]
    moving = _env([[0.5, 0.1, 0.2], [-0.4, 0.3, -0.1]], [[0.3, -0.2, 0.4], [0.1, 0.1, -0.5]], command)
    still = _env(np.zeros((2, 3)), np.zeros((2, 3)), command)
    for name, kwargs in CALLS.items():
        replaced = _replacement(name)(moving, asset_cfg=_Cfg("robot"), **kwargs)
        assert np.allclose(replaced, originals[name](still, asset_cfg=_Cfg("robot"), **kwargs)), name
        assert not np.allclose(replaced, originals[name](moving, asset_cfg=_Cfg("robot"), **kwargs)), name
    assert np.all(_replacement("lin_vel_z_l2")(moving, asset_cfg=_Cfg("robot")) == 0)
    assert np.all(_replacement("ang_vel_xy_l2")(moving, asset_cfg=_Cfg("robot")) == 0)


@needs_isaaclab
def test_code_swap_keeps_the_original_defaults_and_refuses_other_signatures():
    originals = _isaaclab_originals()
    moving = _env([[0.5, 0.1, 0.2]], [[0.3, -0.2, 0.4]], [[0.8, -0.3, 0.6]])
    for name, kwargs in CALLS.items():
        target = originals[name]
        expected = _replacement(name)(moving, asset_cfg=_Cfg("robot"), **kwargs)
        U.swap_code(target, U.SWAPS["U1"][name])
        target.__globals__["torch"] = FAKE_TORCH
        assert np.allclose(target(moving, **kwargs), expected), name  # asset_cfg는 원본 기본값 SceneEntityCfg("robot")
    with pytest.raises(RuntimeError, match="인자"):
        U.swap_code(_isaaclab_originals()["lin_vel_z_l2"], U.SWAPS["U1"]["track_lin_vel_xy_exp"])


def test_the_registered_isaaclab_lines_match_the_parser(tmp_path):
    if REWARDS.is_file():
        assert S.function_lines(REWARDS.read_text(encoding="utf-8")) == S.FUNCTION_LINES
    for relative in S.ISAACLAB_FILES.values():  # 다른 파일은 있어야 줄 범위 검사만 거부 사유가 된다
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("", encoding="utf-8")
    source = REWARDS.read_text(encoding="utf-8") if REWARDS.is_file() else "\n".join(
        f"def {n}():\n    pass" for n in S.FUNCTION_LINES)
    (tmp_path / S.ISAACLAB_FILES["rewards"]).write_text("# shifted\n" + source, encoding="utf-8")
    with pytest.raises(ValueError, match="4함수 위치"):
        S.isaaclab_sources(tmp_path)


# --- Isaac 안에서 도는 arm·readback 경로(가짜 isaaclab·gymnasium 모듈) ---
SIGNATURES = {"lin_vel_z_l2": "env, asset_cfg=None", "ang_vel_xy_l2": "env, asset_cfg=None",
              "track_lin_vel_xy_exp": "env, std, command_name, asset_cfg=None",
              "track_ang_vel_z_exp": "env, std, command_name, asset_cfg=None"}


def _fake_isaac(monkeypatch, bind=None, terms=None):
    """가짜 보상 모듈과 gymnasium. bind로 보상 항에 묶일 함수를, terms로 활성 항 목록을 바꾼다."""
    rewards = ModuleType(U.REWARDS_MODULE)
    exec("\n".join(f"def {name}({args}):\n    return 'original'" for name, args in SIGNATURES.items()),  # noqa: S102
         rewards.__dict__)
    monkeypatch.setitem(sys.modules, U.REWARDS_MODULE, rewards)
    names = list(SIGNATURES) + ["feet_air_time"] if terms is None else terms

    def make(*_args, **_kwargs):
        funcs = {name: getattr(rewards, name, None) for name in names}
        funcs.update(bind or {})
        manager = SimpleNamespace(active_terms=names, get_term_cfg=lambda name: SimpleNamespace(func=funcs[name]))
        return SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=manager))
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    return rewards, gymnasium


def _readback_lines(capsys) -> list[dict]:
    return [json.loads(line[len("[upstream_fault] "):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith("[upstream_fault] ")]


def test_arm_swaps_once_before_the_first_env_and_prints_one_readback(monkeypatch, capsys):
    rewards, gymnasium = _fake_isaac(monkeypatch)
    swaps = []
    original_before_env = U.before_env
    monkeypatch.setattr(U, "before_env", lambda fault: swaps.append(fault) or original_before_env(fault))
    U.arm("U1")
    gymnasium.make("task")
    gymnasium.make("task")
    assert swaps == ["U1"]
    lines = _readback_lines(capsys)
    assert len(lines) == 1 and S.readback_ok(lines[0])
    assert all(getattr(rewards, name).__code__ is code.__code__ for name, code in U.SWAPS["U1"].items())
    with pytest.raises(SystemExit):
        U.arm("U9")


@pytest.mark.parametrize("case", ["other_function_bound", "term_missing"])
def test_readback_is_not_ok_when_the_reward_manager_does_not_use_the_swapped_function(monkeypatch, capsys, case):
    if case == "other_function_bound":
        _, gymnasium = _fake_isaac(monkeypatch, bind={"track_ang_vel_z_exp": lambda env, std, command_name: 0})
    else:
        _, gymnasium = _fake_isaac(monkeypatch, terms=[n for n in SIGNATURES if n != "ang_vel_xy_l2"])
    U.arm("U1")
    gymnasium.make("task")
    (line,) = _readback_lines(capsys)
    assert line["ok"] is False and not S.readback_ok(line)


# --- 판정 행렬 ---
def _protocol() -> dict:
    return S.read(S.PROTOCOL)


def _report(source_run: str, checkpoint_sha: str, name: str = RUN_NAME) -> dict:
    report = S.read(P0C_RUNS / f"{source_run}.json")
    report["name"] = name
    report["checkpoint"] = {"file": S.FINAL_CHECKPOINT, "sha256": checkpoint_sha}
    return report


READBACK = {"fault": "U1", "ok": True, "swapped": dict.fromkeys(U.SWAPS["U1"], True),
            "bound_to_reward_terms": dict.fromkeys(U.SWAPS["U1"], True)}


def _evidence(source_run: str = "h02_s2026") -> dict:
    sha = "a" * 64
    series = {key: [0.0] * S.MAX_ITERATIONS for key in S.PENALTY_SERIES}
    return {"run_name": RUN_NAME, "failure": None, "train_exit_code": 0, "checkpoint_sha256": sha,
            "readback": copy.deepcopy(READBACK), "telemetry": {"series": series},
            "params_differences": ["agent.yaml:run_name"], "report": _report(source_run, sha),
            "reference_report": S.read(P0C_RUNS / f"{S.REFERENCE_RUN}.json")}


def test_valid_runs_are_judged_with_the_p0c_bands():
    protocol = _protocol()
    present = S.judge(protocol, _evidence("h02_s2026"))  # F1처럼 추종하지 못하는 정책
    assert (present["outcome"], present["prediction_match"], present["failed_checks"]) == ("failure_present", True, [])
    assert "err_xy_ratio" in present["verdict"]["failed"] and present["invalid_cause"] is None
    absent = S.judge(protocol, _evidence(S.REFERENCE_RUN))  # 기준과 같은 행동
    assert (absent["outcome"], absent["prediction_match"]) == ("failure_absent", False)
    assert absent["wording"] == protocol["wording"]["failure_absent"]


def _break(evidence: dict, check: str) -> dict:
    e = copy.deepcopy(evidence)
    if check == "pipeline_completed":
        e["failure"] = {"stage": "fixed_evaluation", "error_type": "RuntimeError"}
    elif check == "training_exit_zero":
        e["train_exit_code"] = 1
    elif check == "final_checkpoint_present":
        e["checkpoint_sha256"] = None
    elif check == "injection_readback":
        e["readback"]["swapped"]["track_ang_vel_z_exp"] = False
    elif check == "penalties_zero_in_telemetry":
        e["telemetry"]["series"][S.PENALTY_SERIES[1]][150] = -1e-6
    elif check == "params_match_reference":
        e["params_differences"] = ["agent.yaml:run_name", "env.yaml:rewards.lin_vel_z_l2.weight"]
    elif check == "fixed_report_valid":
        e["report"]["checkpoint"]["sha256"] = "b" * 64
    elif check == "runtime_matches_reference":
        e["report"]["versions"] = {**e["report"]["versions"], "torch": "0.0.0"}
    return e


BREAKABLE = [c for c in S.FIXED["validity_checks"] if c not in ("reference_walks", "verdict_determined")]


@pytest.mark.parametrize("check", BREAKABLE)
def test_each_validity_failure_makes_the_smoke_invalid_with_its_cause(check):
    protocol = _protocol()
    result = S.judge(protocol, _break(_evidence(), check))
    # 뒤 검사가 앞 검사에 기대면(체크포인트 → 보고서 → 런타임) 함께 어긋나지만 첫 항목은 깨뜨린 검사다.
    assert result["outcome"] == "invalid" and result["failed_checks"][0] == check and result["prediction_match"] is None
    assert result["verdict"] is None and result["checks"]["verdict_determined"] is None
    assert result["invalid_cause"] == protocol["invalid_causes"]["by_check"][check]


def test_the_first_failed_check_in_registered_order_names_the_cause():
    e = _break(_break(_evidence(), "fixed_report_valid"), "training_exit_zero")
    result = S.judge(_protocol(), e)
    assert result["failed_checks"][0] == "training_exit_zero" and "fixed_report_valid" in result["failed_checks"]
    assert result["invalid_cause"] == "training_process"
    crashed = _break(_break(_evidence(), "training_exit_zero"), "pipeline_completed")  # 평가 단계에서 멈춤
    assert S.judge(_protocol(), crashed)["invalid_cause"] == "training_process"
    crashed["failure"]["stage"] = "training"  # 학습 watchdog·실행 실패는 인프라 원인
    assert S.judge(_protocol(), crashed)["invalid_cause"] == "infrastructure"


def test_an_undetermined_band_verdict_is_invalid(monkeypatch):
    real = S.RC.verdict
    monkeypatch.setattr(S.RC, "verdict", lambda *args: {**real(*args), "label": "undetermined"})
    result = S.judge(_protocol(), _evidence())
    assert (result["outcome"], result["failed_checks"], result["invalid_cause"]) == (
        "invalid", ["verdict_determined"], "evaluation")


def test_missing_or_partial_evidence_is_invalid_not_a_crash():
    protocol = _protocol()
    for key in ("readback", "telemetry", "params_differences", "report"):
        e = _evidence()
        e[key] = None
        assert S.judge(protocol, e)["outcome"] == "invalid", key
    e = _evidence()
    e["telemetry"]["series"][S.PENALTY_SERIES[0]] = [0.0] * 299
    assert "penalties_zero_in_telemetry" in S.judge(protocol, e)["failed_checks"]
    walking = copy.deepcopy(protocol)
    walking["reference_inputs"]["walks"] = False
    assert "reference_walks" in S.judge(walking, _evidence())["failed_checks"]


def test_readback_parsing_needs_exactly_one_line():
    line = "[upstream_fault] " + json.dumps(READBACK)
    assert S.readback_ok(S.parse_readback(f"x\n{line}\ny"))
    assert S.parse_readback(f"{line}\n{line}") is None and S.parse_readback("no line") is None
    assert S.parse_readback("[upstream_fault] {broken") is None


def test_train_command_matches_the_run_case_arguments():
    cmd = S.train_command(Path("C:/IsaacLab"), RUN_NAME)
    assert Path(cmd[0]).name == "python.bat" and Path(cmd[1]).name == "train_with_upstream_fault.py"
    assert cmd[2:] == ["--task", S.TASK, "--num_envs", "4096", "--max_iterations", "300", "--seed", "2026",
                       "--headless", "--run_name", "upstream_u1_s2026"]


def test_params_differences_flattens_nested_values():
    a = {"env.yaml:rewards.lin_vel_z_l2.weight": -2.0, "agent.yaml:run_name": "x"}
    assert S.params_differences(a, {**a, "agent.yaml:run_name": "y"}) == ["agent.yaml:run_name"]
    assert S.flat_params({"a": {"b": [1, {"c": 2}]}}) == {"a.b[0]": 1, "a.b[1].c": 2}


def test_telemetry_summary_keeps_nonfinite_values_out_of_json():
    series = {S.TRACKING_SERIES[0]: [0.1, float("nan"), 0.3, float("inf")]}
    summary = S.telemetry_summary({"series": series})[S.TRACKING_SERIES[0]]
    assert summary == {"first": 0.1, "last": None, "min": 0.1, "max": 0.3, "nonfinite": 2}
    json.dumps(summary, allow_nan=False)


# --- 프로토콜 ---
def test_protocol_records_sources_mapping_seed_choice_rules_and_later_uses():
    protocol = _protocol()
    assert protocol["prediction"]["outcome"] == "failure_present" and set(protocol["outcomes"]) == set(S.OUTCOMES)
    assert [m["isaaclab_function"] for m in protocol["mapping"]] == list(U.SWAPS["U1"])
    assert protocol["isaaclab_locations"]["rewards_function_lines"] == S.FUNCTION_LINES
    assert any("28.3" in e and "0.44" in e for e in protocol["source"]["workaround_commit"]["excerpts"])
    assert protocol["source"]["fix_pr"]["squash_commit"].endswith("2a3bcea728e71fa1225211ae9c4cf8dcf94be5de")
    assert protocol["seed_choice"]["decision"].startswith("seed 2026") and protocol["seed_choice"]["cons"]
    assert set(protocol["later_uses"]) == set(S.OUTCOMES) and protocol["used_for"].startswith("실패 유무")
    assert set(protocol["invalid_causes"]["by_check"]) == set(protocol["validity_checks"])
    assert protocol["execution"]["attempts"] == {str(k): v for k, v in S.RUN_NAMES.items()}
    assert protocol["execution"]["watchdog_s"] == S.WATCHDOG_S
    assert protocol["reference_inputs"]["fixed_eval_protocol_sha256_lf"] == protocol["source_files"][
        "bench/protocols/fixed_eval_v1.json"]
    from build_replay_page import claim_violations
    texts = [protocol["scope"], *protocol["wording"].values(), protocol["prediction"]["reason"]]
    assert not [p for t in texts for p in claim_violations(t)]


def test_protocol_changes_and_uncommitted_code_are_refused(monkeypatch):
    protocol = _protocol()
    dynamic = {k: protocol[k] for k in S.DYNAMIC}
    S.check_protocol(protocol, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        S.check_protocol({**protocol, "seed": 2041}, dynamic)
    with pytest.raises(ValueError, match="결합한"):
        S.check_protocol(protocol, {**dynamic, "source_files": {}})
    monkeypatch.setattr(S, "dynamic_fields", lambda isaaclab: dynamic)

    def uncommitted(_repo, _paths):
        raise subprocess.CalledProcessError(1, ["git"])
    with pytest.raises(ValueError, match="커밋"):
        S.verify(ISAACLAB, uncommitted)
    assert S.verify(ISAACLAB, lambda _repo, paths: "0" * 40) == (protocol, "0" * 40)


@needs_reference
def test_reference_with_zero_penalties_or_another_eval_protocol_cannot_anchor_the_smoke(tmp_path, monkeypatch):
    telemetry = S.read(S.REFERENCE_TELEMETRY)
    telemetry["series"][S.PENALTY_SERIES[0]] = [0.0] * S.MAX_ITERATIONS
    fake = tmp_path / "ref.telemetry.json"
    fake.write_text(json.dumps(telemetry), encoding="utf-8")
    monkeypatch.setattr(S, "REFERENCE_TELEMETRY", fake)
    with pytest.raises(ValueError, match="페널티"):
        S.reference_inputs(ISAACLAB)
    other = tmp_path / "fixed_eval_other.json"
    other.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(S, "FIXED_EVAL_PROTOCOL", other)
    with pytest.raises(ValueError, match="프로토콜 SHA"):
        S.reference_inputs(ISAACLAB)


# --- 실행 흐름(GPU 없이 소유 Job 실행을 대신한다) ---
@pytest.fixture()
def lab(tmp_path, monkeypatch):
    """가짜 Isaac Lab·결과 폴더·GPU·소유 Job. state로 시나리오별 실패를 넣는다."""
    import run_runtime_readback_smoke as P
    reference = S.reference_paths(ISAACLAB)
    fake_lab = tmp_path / "IsaacLab"
    monkeypatch.setattr(S, "reference_paths", lambda _isaaclab: reference)
    for name, value in (("RESULTS", tmp_path / "results"), ("PRIVATE", tmp_path / "private"),
                        ("RUNS_DIR", tmp_path / "runs")):
        monkeypatch.setattr(S, name, value)
    state = {"calls": [], "gpu_busy": False, "verify_calls": 0, "change_commit_at": None, "change_protocol": False,
             "bound_same": False, "timeout": set(), "train_exit": 0, "readback": READBACK, "launch_error": False,
             "tamper_checkpoint": False, "gpu_processes": [], "created": 1, "penalty": 0.0, "runtime_torch": None,
             "training_cleanup_fails": False}

    def verify(_isaaclab):
        state["verify_calls"] += 1
        changed = state["change_commit_at"] is not None and state["verify_calls"] >= state["change_commit_at"]
        protocol = _protocol()
        if changed and state["change_protocol"]:
            protocol["source_files"]["evals/fixed_eval.py"] = "0" * 64
        return protocol, ("1" if changed else "0") * 40
    monkeypatch.setattr(S, "verify", verify)
    monkeypatch.setattr(S, "bound_sources_same", lambda _start, _now: state["bound_same"])
    monkeypatch.setattr(S, "_process_created", lambda _pid: state["created"])
    monkeypatch.setattr(S, "_admission", lambda: {"busy": state["gpu_busy"], "processes": state["gpu_processes"],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(S.time, "sleep", lambda _s: None)

    def run_owned(folder, args, cwd, watchdog_s):
        name = Path(args[1]).name
        state["calls"].append(name)
        stdout = ""
        if state["launch_error"]:
            raise OSError("job could not be created")
        if name == "train_with_upstream_fault.py":
            assert os.environ["TRIAGE_UPSTREAM_FAULT"] == "U1" and watchdog_s == S.WATCHDOG_S["training"]
            run_name = args[args.index("--run_name") + 1]
            run_dir = fake_lab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-13_00-00-00_{run_name}"
            (run_dir / "params").mkdir(parents=True)
            shutil.copy(reference["environment"], run_dir / "params/env.yaml")
            agent = reference["agent"].read_text(encoding="utf-8").replace(S.REFERENCE_RUN, run_name)
            (run_dir / "params/agent.yaml").write_text(agent, encoding="utf-8")
            (run_dir / S.FINAL_CHECKPOINT).write_bytes(b"checkpoint")
            if state["readback"] is not None:
                stdout = "[upstream_fault] " + json.dumps(state["readback"]) + "\n"
        elif name == "extract_tb.py":
            series = {key: [state["penalty"]] * S.MAX_ITERATIONS for key in S.PENALTY_SERIES}
            Path(args[3]).write_text(json.dumps({"series": series}), encoding="utf-8")
        else:
            job = S.read(Path(args[args.index("--jobs") + 1]))["jobs"][0]
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            report = _report("h02_s2026", S.sha(Path(job["checkpoint"])), job["name"])
            if state["runtime_torch"]:
                report["versions"] = {**report["versions"], "torch": state["runtime_torch"]}
            out.write_text(json.dumps(report), encoding="utf-8")
            if state["tamper_checkpoint"]:
                Path(job["checkpoint"]).write_bytes(b"changed during evaluation")
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        timed_out = name in state["timeout"]
        exit_code = 124 if timed_out else state["train_exit"] if name == "train_with_upstream_fault.py" else 0
        cleanup = not (state["training_cleanup_fails"] and name == "train_with_upstream_fault.py")
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out,
                "error_type": None, "exit_code": exit_code, "cleanup_verified": cleanup,
                "owned_process_identities": {"4242": 1}, "wall_s": 1.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake_lab, state=state)


def _result(tag: str) -> dict:
    return S.read(S.RESULTS / tag / "result.json")


@needs_reference
def test_execute_trains_extracts_evaluates_and_writes_once(lab):
    with pytest.raises(ValueError, match="tag"):
        S.execute("u1", lab.path)
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert out["outcome"] == "failure_present" and out["failed_checks"] == [] and out["failure"] is None
    result = _result("upstream_u1_smoke_20261013")
    assert result["params_differences"] == ["agent.yaml:run_name"] and result["model_calls"] == 0
    assert (result["attempt"], result["run_name"], result["invalid_cause"]) == (1, RUN_NAME, None)
    assert lab.state["calls"] == ["train_with_upstream_fault.py", "extract_tb.py", "fixed_eval.py"]
    assert lab.state["verify_calls"] == 3  # 시작·평가 직전·평가 뒤
    assert "C:" not in json.dumps(result) and "LIMMM" not in json.dumps(result)
    private = S.PRIVATE / "upstream_u1_smoke_20261013"
    assert {p.name for p in private.iterdir()} >= {"train", "extract", "eval", "execution.json", "gpu_before_training.json",
                                                   "gpu_before_fixed_evaluation.json", "gpu_after.json"}
    assert not (private / "failure.json").exists() and "TRIAGE_UPSTREAM_FAULT" not in os.environ
    with pytest.raises(FileExistsError):
        S.execute("upstream_u1_smoke_20261013", lab.path)
    with pytest.raises(FileExistsError):  # 같은 시도는 tag를 바꿔도 한 번
        S.execute("upstream_u1_smoke_20261013b", lab.path)


@needs_reference
def test_a_busy_gpu_before_training_launches_and_consumes_nothing(lab):
    lab.state["gpu_busy"] = True
    assert S.execute("upstream_u1_smoke_20261013", lab.path)["status"] == "gpu_busy_no_launch"
    assert lab.state["calls"] == [] and not S.PRIVATE.exists() and not S.RESULTS.exists()


@needs_reference
def test_failed_injection_skips_the_fixed_evaluation(lab):
    lab.state["readback"] = None
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]) == ("invalid", "injection", None)
    assert lab.state["calls"] == ["train_with_upstream_fault.py", "extract_tb.py"]


@needs_reference
def test_a_busy_gpu_before_evaluation_leaves_invalid_and_resume_finishes_once(lab, monkeypatch):
    monkeypatch.setattr(S, "EVAL_GPU_WAIT_S", 0)
    original_extract = S.params_differences

    def busy_after_training(*args):
        lab.state["gpu_busy"] = True
        return original_extract(*args)
    monkeypatch.setattr(S, "params_differences", busy_after_training)
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]["stage"]) == ("invalid", "infrastructure", "pre_evaluation")
    failure = S.read(S.PRIVATE / "upstream_u1_smoke_20261013/failure.json")
    assert failure["stage"] == "pre_evaluation" and "GPU" in failure["detail"]
    monkeypatch.setattr(S, "params_differences", original_extract)
    lab.state["gpu_busy"] = False
    resumed = S.resume("upstream_u1_smoke_20261013r", "upstream_u1_smoke_20261013", lab.path)
    assert resumed["outcome"] == "failure_present"
    result = _result("upstream_u1_smoke_20261013r")
    assert result["resumed_from"] == "upstream_u1_smoke_20261013" and result["checks"]["injection_readback"] is True
    assert lab.state["calls"] == ["train_with_upstream_fault.py", "extract_tb.py", "fixed_eval.py"]  # 학습·추출은 다시 안 함
    with pytest.raises(FileExistsError):
        S.resume("upstream_u1_smoke_20261013s", "upstream_u1_smoke_20261013", lab.path)
    with pytest.raises(ValueError, match="이어 하기"):
        S.resume("upstream_u1_smoke_20261013t", "upstream_u1_smoke_20261013r", lab.path)


@needs_reference
def test_a_training_watchdog_timeout_allows_exactly_one_retry_under_a_new_run_name(lab):
    lab.state["timeout"] = {"train_with_upstream_fault.py"}
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["invalid_cause"], out["failure"]["stage"]) == ("infrastructure", "training")
    with pytest.raises(ValueError, match="이어 하기"):  # 학습이 끝나지 않았으면 이어 하지 않는다
        S.resume("upstream_u1_smoke_20261013r", "upstream_u1_smoke_20261013", lab.path)
    lab.state["timeout"] = set()
    retried = S.execute("upstream_u1_smoke_20261013b", lab.path, retry_of="upstream_u1_smoke_20261013")
    assert retried["outcome"] == "failure_present"
    result = _result("upstream_u1_smoke_20261013b")
    assert (result["attempt"], result["run_name"], result["retry_of"]) == (2, S.RUN_NAMES[2], "upstream_u1_smoke_20261013")
    assert result["params_differences"] == ["agent.yaml:run_name"]
    with pytest.raises(ValueError, match="시도 2"):
        S.execute("upstream_u1_smoke_20261013c", lab.path, retry_of="upstream_u1_smoke_20261013b")


@needs_reference
def test_a_training_stage_failure_is_retried_not_resumed_even_with_good_artifacts(lab):
    lab.state["training_cleanup_fails"] = True  # 체크포인트·확인 줄은 있지만 학습 Job이 정리되지 않음
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"]["stage"], out["invalid_cause"]) == ("training", "infrastructure")
    assert all(_result("upstream_u1_smoke_20261013")["checks"][name] for name in S.TRAINING_CHECKS)
    with pytest.raises(ValueError, match="이어 하기"):
        S.resume("upstream_u1_smoke_20261013r", "upstream_u1_smoke_20261013", lab.path)
    lab.state["training_cleanup_fails"] = False
    retried = S.execute("upstream_u1_smoke_20261013b", lab.path, retry_of="upstream_u1_smoke_20261013")
    assert retried["outcome"] == "failure_present"


@needs_reference
@pytest.mark.parametrize("change", ["bound_paths", "protocol"])
def test_a_source_change_during_evaluation_is_recorded_as_invalid(lab, change):
    lab.state["change_commit_at"] = 3
    lab.state["bound_same"] = change == "protocol"
    lab.state["change_protocol"] = change == "protocol"
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["failure"]["stage"]) == ("invalid", "postflight")
    assert _result("upstream_u1_smoke_20261013")["checks"]["pipeline_completed"] is False


@needs_reference
def test_an_unrelated_commit_during_the_run_is_recorded_not_rejected(lab):
    lab.state["change_commit_at"] = 2
    lab.state["bound_same"] = True
    assert S.execute("upstream_u1_smoke_20261013", lab.path)["outcome"] == "failure_present"
    result = _result("upstream_u1_smoke_20261013")
    assert result["commits_seen"] == ["0" * 40, "1" * 40] and result["source_commit_sha"] == "0" * 40


@needs_reference
@pytest.mark.parametrize("case", ["tamper_checkpoint", "gpu_processes"])
def test_postflight_catches_a_changed_checkpoint_or_an_owned_process_left_on_the_gpu(lab, case):
    lab.state[case] = True if case == "tamper_checkpoint" else [{"pid": 4242}]
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["failure"]["stage"], out["invalid_cause"]) == ("invalid", "postflight", "infrastructure")


@needs_reference
def test_a_reused_pid_on_the_gpu_is_not_an_owned_process(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    lab.state["created"] = 2  # 같은 PID, 다른 생성 시각
    assert S.execute("upstream_u1_smoke_20261013", lab.path)["outcome"] == "failure_present"


@needs_reference
@pytest.mark.parametrize("order", ["resume_then_retry", "retry_then_resume"])
def test_the_first_valid_result_is_final(lab, order):
    lab.state["gpu_processes"] = [{"pid": 4242}]  # 시도 1은 사후 확인에서 인프라 원인 invalid
    first = "upstream_u1_smoke_20261013"
    assert S.execute(first, lab.path)["failure"]["stage"] == "postflight"
    lab.state["gpu_processes"] = []
    if order == "resume_then_retry":
        assert S.resume(first + "r", first, lab.path)["outcome"] == "failure_present"
        for retry_of in (first, first + "r"):  # 지난 기록이든 마지막(유효) 결과든 시도 2는 없다
            with pytest.raises(ValueError, match="마지막 결과"):
                S.execute(first + "b", lab.path, retry_of=retry_of)
    else:
        assert S.execute(first + "b", lab.path, retry_of=first)["outcome"] == "failure_present"
        calls = list(lab.state["calls"])
        with pytest.raises(FileExistsError):
            S.resume(first + "r", first, lab.path)
        assert lab.state["calls"] == calls  # 평가를 돌리기 전에 거부한다
    assert S.read(S._consumed("final"))["outcome"] == "failure_present"


@needs_reference
@pytest.mark.parametrize("ending", ["runner", "comparability"])
def test_retry_follows_the_last_result_of_attempt_one(lab, monkeypatch, ending):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    first = "upstream_u1_smoke_20261013"
    assert S.execute(first, lab.path)["invalid_cause"] == "infrastructure"
    lab.state["gpu_processes"] = []
    if ending == "runner":
        def broken(*_args):
            raise ZeroDivisionError("band")
        monkeypatch.setattr(S.RC, "compare", broken)
    else:
        lab.state["runtime_torch"] = "0.0.0"
    assert S.resume(first + "r", first, lab.path)["invalid_cause"] == ending
    for retry_of in (first, first + "r"):  # 처음 시도 1 기록을 주어도 마지막 결과로 판단한다
        with pytest.raises(ValueError, match="마지막 결과"):
            S.execute(first + "b", lab.path, retry_of=retry_of)


@needs_reference
def test_retry_waits_for_a_running_resume_and_names_its_tag(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    first = "upstream_u1_smoke_20261013"
    S.execute(first, lab.path)
    S.write_new(S._consumed("resume1"), {"tag": first + "r", "resumed_from": first, "attempt": 1})  # 이어 하기 진행 중
    for retry_of in (first, first + "r"):
        with pytest.raises(ValueError, match="마지막 결과"):
            S.execute(first + "b", lab.path, retry_of=retry_of)
    S._consumed("resume1").unlink()
    S.resume(first + "r", first, lab.path)  # 이어 하기도 사후 확인에서 인프라 원인 invalid
    lab.state["gpu_processes"] = []
    with pytest.raises(ValueError, match="마지막 결과"):
        S.execute(first + "b", lab.path, retry_of=first)
    assert S.execute(first + "b", lab.path, retry_of=first + "r")["outcome"] == "failure_present"


@needs_reference
def test_a_params_failure_after_nonzero_penalties_keeps_the_injection_cause(lab, monkeypatch):
    lab.state["penalty"] = -0.01

    def broken(_run_dir):
        raise OSError("params unreadable")
    monkeypatch.setattr(S, "load_params", broken)
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"]["stage"], out["invalid_cause"]) == ("params", "injection")


@needs_reference
def test_the_final_marker_is_claimed_before_the_result_is_written(lab, monkeypatch):
    original = S.write_new

    def no_result(path, value):
        if path.name == "result.json":
            raise OSError("disk full")
        original(path, value)
    monkeypatch.setattr(S, "write_new", no_result)
    with pytest.raises(OSError):
        S.execute("upstream_u1_smoke_20261013", lab.path)
    assert S.read(S._consumed("final"))["tag"] == "upstream_u1_smoke_20261013"


@needs_reference
def test_a_judge_that_keeps_failing_still_leaves_a_minimal_invalid_result(lab, monkeypatch):
    def broken(*_args):
        raise RuntimeError("judge")
    monkeypatch.setattr(S, "judge", broken)
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]["stage"]) == ("invalid", "runner", "judge")
    assert _result("upstream_u1_smoke_20261013")["checks"] is None


@needs_reference
def test_a_crashed_training_keeps_its_cause_when_a_later_stage_also_fails(lab):
    lab.state["train_exit"] = 1
    lab.state["timeout"] = {"extract_tb.py"}
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"]["stage"], out["invalid_cause"]) == ("telemetry_extract", "training_process")
    assert out["failed_checks"][:2] == ["pipeline_completed", "training_exit_zero"]
    with pytest.raises(ValueError, match="시도 2"):
        S.execute("upstream_u1_smoke_20261013b", lab.path, retry_of="upstream_u1_smoke_20261013")


@needs_reference
def test_resume_refuses_telemetry_left_by_a_failed_extraction(lab):
    lab.state["timeout"] = {"extract_tb.py"}  # 가짜 추출은 파일을 쓴 뒤 시간 초과로 끝난다
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"]["stage"], out["invalid_cause"]) == ("telemetry_extract", "infrastructure")
    lab.state["timeout"] = set()
    with pytest.raises(ValueError, match="텔레메트리"):
        S.resume("upstream_u1_smoke_20261013r", "upstream_u1_smoke_20261013", lab.path)


@needs_reference
def test_an_unreadable_reference_after_a_failure_still_writes_the_invalid_result(lab, monkeypatch, tmp_path):
    lab.state["gpu_processes"] = [{"pid": 4242}]
    monkeypatch.setattr(S, "P0C", tmp_path / "missing")
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["failure"]["stage"]) == ("invalid", "postflight")
    assert S.read(S.PRIVATE / "upstream_u1_smoke_20261013/judge_error.json")["error_type"] == "FileNotFoundError"


def test_the_gpu_wait_returns_once_the_gpu_is_free(tmp_path, monkeypatch):
    answers = [True, True, False]
    monkeypatch.setattr(S, "_admission", lambda: {"busy": answers.pop(0), "busy_reasons": [], "processes": []})
    monkeypatch.setattr(S.time, "sleep", lambda _s: None)
    S._wait_for_gpu(tmp_path)
    assert answers == [] and S.read(tmp_path / "gpu_before_fixed_evaluation.json")["busy"] is False


def test_a_busy_gpu_is_not_reported_as_success(monkeypatch, capsys):
    monkeypatch.setattr(S, "execute", lambda *_args: {"status": "gpu_busy_no_launch", "busy_reasons": ["x"]})
    assert S.main(["execute", "--tag", "upstream_u1_smoke_20261013"]) == 3
    assert "gpu_busy_no_launch" in capsys.readouterr().out


@needs_reference
def test_a_launch_failure_still_consumes_the_attempt(lab):
    lab.state["launch_error"] = True
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"], out["invalid_cause"]) == ({"stage": "training", "error_type": "OSError"}, "infrastructure")
    lab.state["launch_error"] = False
    with pytest.raises(FileExistsError):  # 실행 폴더가 없어도 시도 1은 한 번
        S.execute("upstream_u1_smoke_20261013b", lab.path)


@needs_reference
def test_a_missing_readback_is_not_retried_or_resumed_when_extraction_also_fails(lab):
    lab.state["readback"] = None
    lab.state["timeout"] = {"extract_tb.py"}
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["failure"]["stage"], out["invalid_cause"]) == ("telemetry_extract", "injection")
    with pytest.raises(ValueError, match="이어 하기"):
        S.resume("upstream_u1_smoke_20261013r", "upstream_u1_smoke_20261013", lab.path)
    with pytest.raises(ValueError, match="시도 2"):
        S.execute("upstream_u1_smoke_20261013b", lab.path, retry_of="upstream_u1_smoke_20261013")


@needs_reference
def test_a_crash_while_judging_still_writes_an_invalid_result(lab, monkeypatch):
    def broken(*_args):
        raise ZeroDivisionError("band")
    monkeypatch.setattr(S.RC, "compare", broken)
    out = S.execute("upstream_u1_smoke_20261013", lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]) == (
        "invalid", "runner", {"stage": "judge", "error_type": "ZeroDivisionError"})
    with pytest.raises(ValueError, match="시도 2"):  # 실행기 결함은 재시도 대상이 아니다
        S.execute("upstream_u1_smoke_20261013b", lab.path, retry_of="upstream_u1_smoke_20261013")
