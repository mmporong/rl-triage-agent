"""D3 무해성 smoke 실행기: 판정 행렬과 실행 흐름(GPU 없이 소유 Job 실행을 대신한다)을 고정한다."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import d3_changes as D  # noqa: E402
import d3_diff as R  # noqa: E402
import run_d3_harmless_smoke as H  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
needs_inputs = pytest.mark.skipif(
    not (ISAACLAB / D.REWARDS[1]).is_file()
    or not all((ROOT / "bench/private" / f"baseline_p0c_s{seed}.meta.json").is_file() for seed in H.SEEDS),
    reason="로컬 Isaac Lab v2.1.1 소스나 비공개 P0-C 기준 실행 meta가 없다")
TAG = "d3_harmless_smoke_20261013_s2026"
READBACK = {"changes": list(D.HARMLESS),
            "code": {cid: {t["qualname"]: {"swapped": True, "bound": True} for t in D.code_targets(cid)}
                     for cid in D.HARMLESS if D.CHANGES[cid]["kind"] == "code"},
            "config": {cid: {key: True if key.startswith("env.") else "params" for key in D.CHANGES[cid]["overrides"]}
                       for cid in D.HARMLESS if D.CHANGES[cid]["kind"] == "config"},
            "ok": True}


@pytest.fixture(scope="module")
def protocol():
    return json.loads(json.dumps({**H.FIXED, **H.dynamic_fields(ISAACLAB)}))


def _report(source_run: str, checkpoint_sha: str, name: str) -> dict:
    report = H.read(P0C_RUNS / f"{source_run}.json")
    report["name"] = name
    report["checkpoint"] = {"file": H.FINAL_CHECKPOINT, "sha256": checkpoint_sha}
    return report


def _scaled(report: dict, lin: float = 1.0, yaw: float = 1.0, falls: int = 0) -> dict:
    """행 합계와 요약 지표를 함께 바꿔 고정 평가 검증을 통과하는 보고서를 만든다."""
    report = copy.deepcopy(report)
    rows = report["by_condition"]
    for row in rows:
        row["lin_error_sq_sum"] *= lin ** 2
        row["yaw_error_sq_sum"] *= yaw ** 2
        row["lin_vel_rmse_mps"] = math.sqrt(row["lin_error_sq_sum"] / row["alive_steps_sum"])
        row["yaw_rate_rmse_radps"] = math.sqrt(row["yaw_error_sq_sum"] / row["alive_steps_sum"])
    rows[0]["fall_count"] = falls
    rows[0]["fall_rate"] = falls / rows[0]["num_envs"]
    total, alive = sum(r["num_envs"] for r in rows), sum(r["alive_steps_sum"] for r in rows)
    report["metrics"].update(fall_rate=sum(r["fall_count"] for r in rows) / total,
                             lin_vel_rmse_mps=math.sqrt(sum(r["lin_error_sq_sum"] for r in rows) / alive),
                             yaw_rate_rmse_radps=math.sqrt(sum(r["yaw_error_sq_sum"] for r in rows) / alive))
    return report


def _params(seed: int, run_name: str) -> dict:
    out = {key: list(pair) for key, pair in D.expected_params(D.HARMLESS).items()}
    return {**out, H.RUN_NAME_ALLOWED_DIFF: [H.reference_run(seed), run_name]}


def _evidence(seed: int = 2026, source_run: str | None = None, **report_scale) -> dict:
    sha, name = "a" * 64, H.run_name(seed, 1)
    report = _report(source_run or H.reference_run(seed), sha, name)
    return {"run_name": name, "seed": seed, "failure": None, "train_exit_code": 0, "checkpoint_sha256": sha,
            "readback": copy.deepcopy(READBACK), "params_differences": _params(seed, name),
            "report": _scaled(report, **report_scale) if report_scale else report,
            "reference_report": H.read(P0C_RUNS / f"{H.reference_run(seed)}.json")}


# --- 판정 행렬 ---
@needs_inputs
@pytest.mark.parametrize("seed", H.SEEDS)
def test_a_run_like_its_reference_passes_with_margins(protocol, seed):
    out = H.judge(protocol, _evidence(seed))
    assert (out["outcome"], out["failed_checks"], out["invalid_cause"], out["prediction_match"]) == (
        "harmless_pass", [], None, True)
    assert out["verdict"]["label"] == "healthy" and all(out["margins"].values())
    assert out["comparison"]["err_xy_ratio"] == pytest.approx(1.0)


@needs_inputs
@pytest.mark.parametrize("scale, margin", [({"lin": 1.08}, "err_xy_ratio"), ({"yaw": 1.07}, "err_yaw_ratio"),
                                           ({"falls": 32}, "fall_frac")])
def test_a_healthy_run_outside_a_margin_is_a_harmless_failure(protocol, scale, margin):
    out = H.judge(protocol, _evidence(**scale))
    assert out["verdict"]["label"] == "healthy"  # p0c 밴드 안이지만
    assert out["outcome"] == "harmless_fail" and out["margins"][margin] is False and out["prediction_match"] is False
    assert [k for k, ok in out["margins"].items() if not ok] == [margin]


@needs_inputs
def test_a_run_inside_the_margins_passes_and_an_unhealthy_run_fails(protocol):
    assert H.judge(protocol, _evidence(lin=1.04, yaw=1.04, falls=20))["outcome"] == "harmless_pass"  # 20/1040 ≤ 0.02
    out = H.judge(protocol, _evidence(source_run="h02_s2026"))  # F1처럼 추종하지 못하는 정책
    assert (out["outcome"], out["verdict"]["label"]) == ("harmless_fail", "unhealthy")


def _break(evidence: dict, protocol: dict, check: str) -> tuple[dict, dict]:
    e, p = copy.deepcopy(evidence), copy.deepcopy(protocol)
    if check == "pipeline_completed":
        e["failure"] = {"stage": "fixed_evaluation", "error_type": "RuntimeError"}
    elif check == "training_exit_zero":
        e["train_exit_code"] = 1
    elif check == "final_checkpoint_present":
        e["checkpoint_sha256"] = None
    elif check == "injection_readback":
        e["readback"]["code"]["H3"]["feet_air_time"]["bound"] = False
    elif check == "params_overrides_applied":
        e["params_differences"].pop("agent.yaml:algorithm.entropy_coef")
    elif check == "params_otherwise_match":
        e["params_differences"]["env.yaml:sim.dt"] = "값 생략"
    elif check == "reference_walks":
        p["reference_inputs"]["2026"]["walks"] = False
    elif check == "fixed_report_valid":
        e["report"]["checkpoint"]["sha256"] = "b" * 64
    elif check == "runtime_matches_reference":
        e["report"]["versions"]["torch"] = "0.0.0"
    return e, p


@needs_inputs
@pytest.mark.parametrize("check", [c for c in H.FIXED["validity_checks"] if c != "verdict_determined"])
def test_each_validity_failure_makes_the_seed_invalid_with_its_cause(protocol, check):
    evidence, broken_protocol = _break(_evidence(), protocol, check)
    out = H.judge(broken_protocol, evidence)
    assert out["outcome"] == "invalid" and out["failed_checks"][0] == check, out["failed_checks"]
    assert out["invalid_cause"] == protocol["invalid_causes"]["by_check"][check] and out["prediction_match"] is None


@needs_inputs
def test_each_seed_is_judged_against_its_own_reference(protocol):
    broken = copy.deepcopy(protocol)
    broken["reference_inputs"]["2027"]["walks"] = False
    assert H.judge(broken, _evidence(2027))["invalid_cause"] == "comparability"
    assert H.judge(broken, _evidence(2026))["outcome"] == "harmless_pass"


@needs_inputs
def test_an_undetermined_band_verdict_is_invalid(protocol, monkeypatch):
    real = H.RC.verdict
    monkeypatch.setattr(H.RC, "verdict", lambda *args: {**real(*args), "label": "undetermined"})
    out = H.judge(protocol, _evidence())
    assert (out["outcome"], out["invalid_cause"]) == ("invalid", "evaluation")


@needs_inputs
def test_readback_for_other_changes_or_a_partial_readback_is_not_ok():
    assert H.readback_ok(READBACK)
    for broken in ({**READBACK, "changes": [*D.HARMLESS, "U1"]}, {**READBACK, "ok": False},
                   {**READBACK, "config": {**READBACK["config"], "H7": {"agent.algorithm.entropy_coef": True}}},
                   {**READBACK, "code": {k: v for k, v in READBACK["code"].items() if k != "H6"}}, None, []):
        assert not H.readback_ok(broken), broken
    assert H.parse_readback("x\n" + D.READBACK_PREFIX + json.dumps(READBACK) + "\n") == READBACK
    assert H.parse_readback((D.READBACK_PREFIX + "{}\n") * 2) is None


def test_params_differences_show_values_only_for_registered_keys():
    reference = {"env.yaml:rewards.track_lin_vel_xy_exp.weight": 1.5, "env.yaml:sim.dt": 0.005,
                 "agent.yaml:run_name": "baseline_p0c_s2026", "env.yaml:same": 1}
    run = {**reference, "env.yaml:rewards.track_lin_vel_xy_exp.weight": 1.4, "env.yaml:sim.dt": 0.02,
           "agent.yaml:run_name": "d3_harmless_v1_s2026"}
    assert H.params_differences(run, reference) == {
        "agent.yaml:run_name": ["baseline_p0c_s2026", "d3_harmless_v1_s2026"], "env.yaml:sim.dt": "값 생략",
        "env.yaml:rewards.track_lin_vel_xy_exp.weight": [1.5, 1.4]}


# --- 프로토콜 ---
@needs_inputs
def test_the_protocol_records_changes_reserves_amendments_margins_and_rules(protocol):
    assert list(protocol["changes"]) == list(D.HARMLESS) and list(protocol["reserves"]) == list(D.RESERVE)
    assert [a["id"] for a in protocol["amendments_from_design_draft"]] == ["H1", "H4", "H6", "reserve"]
    assert protocol["apply"]["hydra_overrides"] == D.hydra_overrides(D.HARMLESS) and protocol["apply"]["spec"] == H.SPEC
    assert protocol["judgment"]["margins"] == {"err_xy_ratio_max": 1.05, "err_yaw_ratio_max": 1.05, "fall_frac_max": 0.02}
    assert protocol["seeds"] == [2026, 2027] and set(protocol["reference_inputs"]) == {"2026", "2027"}
    assert set(protocol["invalid_causes"]["by_check"]) == set(protocol["validity_checks"])
    assert set(protocol["rendered_diffs"]) == set(D.CHANGES)
    assert protocol["prediction"] == {**protocol["prediction"], "seed_outcome": "harmless_pass", "overall": "pass"}
    from build_replay_page import claim_violations
    texts = [protocol["scope"], *protocol["wording"].values(), protocol["prediction"]["reason"], protocol["failure_rule"]]
    assert not [p for t in texts for p in claim_violations(t)]
    assert "C:" not in json.dumps(protocol) and "LIMMM" not in json.dumps(protocol)


@needs_inputs
def test_protocol_changes_uncommitted_code_and_clues_in_diffs_are_refused(protocol, monkeypatch):
    dynamic = {k: protocol[k] for k in H.DYNAMIC}
    H.check_protocol(protocol, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        H.check_protocol({**protocol, "seeds": [2026]}, dynamic)
    with pytest.raises(ValueError, match="결합한"):
        H.check_protocol(protocol, {**dynamic, "rendered_diffs": {}})
    monkeypatch.setattr(H, "read", lambda _path: protocol)
    monkeypatch.setattr(H, "dynamic_fields", lambda _isaaclab: dynamic)

    def uncommitted(_repo, _paths):
        raise subprocess.CalledProcessError(1, ["git"])
    with pytest.raises(ValueError, match="커밋"):
        H.verify(ISAACLAB, uncommitted)
    assert H.verify(ISAACLAB, lambda _repo, paths: "0" * 40) == (protocol, "0" * 40)
    monkeypatch.setattr(R, "canary", lambda diff: ["word:fault"])
    with pytest.raises(ValueError, match="단서"):
        H.rendered_diffs(ISAACLAB)


@needs_inputs
def test_a_reference_whose_params_differ_from_the_catalog_baseline_is_refused(monkeypatch):
    monkeypatch.setitem(D.CHANGES, "H7", {**D.CHANGES["H7"], "overrides": {"agent.algorithm.entropy_coef": (0.02, 0.008)}})
    with pytest.raises(ValueError, match="기준값"):
        H.reference_inputs(ISAACLAB)


@needs_inputs
def test_a_written_protocol_file_is_current(protocol):
    if not H.PROTOCOL.is_file():
        pytest.skip("프로토콜을 아직 prepare하지 않았다")
    H.check_protocol(H.read(H.PROTOCOL), {k: protocol[k] for k in H.DYNAMIC})


def test_the_training_command_carries_no_change_ids():
    command = H.train_command(ISAACLAB, 2026, H.run_name(2026, 1))
    assert Path(command[1]).name == "train_with_d3_changes.py" and command[command.index("--seed") + 1] == "2026"
    assert not [part for part in command for cid in D.CHANGES if re.search(rf"\b{cid}\b", part)]
    assert command[-2:] == ["--run_name", "d3_harmless_v1_s2026"] and H.run_name(2027, 2) == "d3_harmless_v1_s2027_a2"


# --- 실행 흐름(GPU 없이 소유 Job 실행을 대신한다) ---
@pytest.fixture()
def lab(tmp_path, monkeypatch, protocol):
    """가짜 Isaac Lab·결과 폴더·GPU·소유 Job. state로 시나리오별 실패를 넣는다."""
    import run_runtime_readback_smoke as P
    references = {seed: H.reference_paths(ISAACLAB, seed) for seed in H.SEEDS}
    reference_params = {seed: H.load_params(references[seed]["run_dir"]) for seed in H.SEEDS}
    fake_lab = tmp_path / "IsaacLab"
    monkeypatch.setattr(H, "reference_paths", lambda _isaaclab, seed: references[seed])
    for name, value in (("RESULTS", tmp_path / "results"), ("PRIVATE", tmp_path / "private"), ("RUNS_DIR", tmp_path / "runs"),
                        ("PROTOCOL", tmp_path / "d3_harmless_smoke_v1.json")):
        monkeypatch.setattr(H, name, value)
    H.PROTOCOL.write_text(json.dumps(protocol), encoding="utf-8")  # 시도 표식은 프로토콜 SHA별이다
    state = {"calls": [], "gpu_busy": False, "verify_calls": 0, "change_commit_at": None, "change_protocol": False,
             "bound_same": False, "timeout": set(), "train_exit": 0, "readback": READBACK, "launch_error": False,
             "tamper_checkpoint": False, "gpu_processes": [], "created": 1, "report": {}, "skip_override": None,
             "extra_param": None}

    def verify(_isaaclab):
        state["verify_calls"] += 1
        changed = state["change_commit_at"] is not None and state["verify_calls"] >= state["change_commit_at"]
        current = copy.deepcopy(protocol)
        if changed and state["change_protocol"]:
            current["source_files"]["evals/fixed_eval.py"] = "0" * 64
        return current, ("1" if changed else "0") * 40
    monkeypatch.setattr(H, "verify", verify)
    monkeypatch.setattr(H, "bound_sources_same", lambda _start, _now: state["bound_same"])
    monkeypatch.setattr(H, "_process_created", lambda _pid: state["created"])
    monkeypatch.setattr(H, "_admission", lambda: {"busy": state["gpu_busy"], "processes": state["gpu_processes"],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(H.time, "sleep", lambda _s: None)

    def load_params(run_dir):
        for seed, paths in references.items():
            if Path(run_dir) == paths["run_dir"]:
                return reference_params[seed]
        seed = int(re.search(r"_s(\d{4})", Path(run_dir).name).group(1))
        params = {**reference_params[seed], H.RUN_NAME_ALLOWED_DIFF: Path(run_dir).name.split("_", 2)[2]}
        params.update({key: new for key, (_old, new) in D.expected_params(D.HARMLESS).items()
                       if key != state["skip_override"]})
        if state["extra_param"]:
            params[state["extra_param"]] = "changed"
        return params
    monkeypatch.setattr(H, "load_params", load_params)

    def run_owned(folder, args, cwd, watchdog_s):
        name = Path(args[1]).name
        state["calls"].append(name)
        stdout = ""
        if state["launch_error"]:
            raise OSError("job could not be created")
        if name == "train_with_d3_changes.py":
            assert os.environ[D.ENV_VAR] == H.SPEC and watchdog_s == H.WATCHDOG_S["training"]
            run_name = args[args.index("--run_name") + 1]
            run_dir = fake_lab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-13_00-00-00_{run_name}"
            run_dir.mkdir(parents=True)
            (run_dir / H.FINAL_CHECKPOINT).write_bytes(b"checkpoint " + run_name.encode())
            if state["readback"] is not None:
                stdout = D.READBACK_PREFIX + json.dumps(state["readback"]) + "\n"
        elif name == "extract_tb.py":
            Path(args[3]).write_text(json.dumps({"series": {"Train/mean_episode_length": [1.0, 2.0]}}), encoding="utf-8")
        else:
            job = H.read(Path(args[args.index("--jobs") + 1]))["jobs"][0]
            seed = int(re.search(r"_s(\d{4})", job["name"]).group(1))
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            report = _report(H.reference_run(seed), H.sha(Path(job["checkpoint"])), job["name"])
            out.write_text(json.dumps(_scaled(report, **state["report"]) if state["report"] else report), encoding="utf-8")
            if state["tamper_checkpoint"]:
                Path(job["checkpoint"]).write_bytes(b"changed during evaluation")
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        timed_out = name in state["timeout"]
        exit_code = 124 if timed_out else state["train_exit"] if name == "train_with_d3_changes.py" else 0
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out,
                "error_type": None, "exit_code": exit_code, "cleanup_verified": True,
                "owned_process_identities": {"4242": 1}, "wall_s": 1.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake_lab, state=state)


def _result(tag: str) -> dict:
    return H.read(H.RESULTS / tag / "result.json")


@needs_inputs
def test_execute_trains_extracts_evaluates_and_writes_once(lab):
    with pytest.raises(ValueError, match="tag"):
        H.execute("u1", 2026, lab.path)
    with pytest.raises(ValueError, match="seed"):
        H.execute(TAG, 2035, lab.path)
    out = H.execute(TAG, 2026, lab.path)
    assert (out["outcome"], out["failed_checks"], out["failure"]) == ("harmless_pass", [], None)
    result = _result(TAG)
    assert (result["seed"], result["attempt"], result["run_name"], result["changes"]) == (
        2026, 1, "d3_harmless_v1_s2026", list(D.HARMLESS))
    assert result["params_differences"] == _params(2026, "d3_harmless_v1_s2026") and result["model_calls"] == 0
    assert lab.state["calls"] == ["train_with_d3_changes.py", "extract_tb.py", "fixed_eval.py"]
    assert lab.state["verify_calls"] == 3  # 시작·평가 직전·평가 뒤
    assert "C:" not in json.dumps(result) and "LIMMM" not in json.dumps(result)
    assert D.ENV_VAR not in os.environ and not (H.PRIVATE / TAG / "failure.json").exists()
    assert H.read(H._consumed(2026, "final")) == {"tag": TAG, "outcome": "harmless_pass"}
    with pytest.raises(FileExistsError):
        H.execute(TAG, 2026, lab.path)
    with pytest.raises(FileExistsError):  # 같은 seed·시도는 tag를 바꿔도 한 번
        H.execute(TAG + "b", 2026, lab.path)
    assert H.execute("d3_harmless_smoke_20261013_s2027", 2027, lab.path)["outcome"] == "harmless_pass"  # seed마다 따로


@needs_inputs
def test_a_busy_gpu_before_training_launches_and_consumes_nothing(lab):
    lab.state["gpu_busy"] = True
    assert H.execute(TAG, 2026, lab.path)["status"] == "gpu_busy_no_launch"
    assert lab.state["calls"] == [] and not H.PRIVATE.exists() and not H.RESULTS.exists()


@needs_inputs
@pytest.mark.parametrize("case, cause", [("readback", "injection"), ("skip_override", "injection"),
                                         ("extra_param", "comparability")])
def test_a_change_that_did_not_apply_skips_the_fixed_evaluation(lab, case, cause):
    if case == "readback":
        lab.state["readback"] = None
    elif case == "skip_override":
        lab.state["skip_override"] = "env.yaml:observations.policy.joint_vel.noise.n_min"
    else:
        lab.state["extra_param"] = "env.yaml:sim.dt"
    out = H.execute(TAG, 2026, lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]) == ("invalid", cause, None)
    assert lab.state["calls"] == ["train_with_d3_changes.py", "extract_tb.py"]
    assert H.seed_state(2026)["next"] == []  # 인프라 원인이 아니라 시도 2·이어 하기가 없다
    with pytest.raises(ValueError, match="시도 2"):
        H.execute(TAG + "b", 2026, lab.path, retry_of=TAG)


@needs_inputs
def test_a_busy_gpu_before_evaluation_leaves_invalid_and_resume_finishes_once(lab, monkeypatch):
    monkeypatch.setattr(H, "EVAL_GPU_WAIT_S", 0)
    original = H.params_differences

    def busy_after_training(*args):
        lab.state["gpu_busy"] = True
        return original(*args)
    monkeypatch.setattr(H, "params_differences", busy_after_training)
    out = H.execute(TAG, 2026, lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]["stage"]) == ("invalid", "infrastructure", "pre_evaluation")
    assert H.seed_state(2026)["next"] == ["resume", "retry"]
    monkeypatch.setattr(H, "params_differences", original)
    lab.state["gpu_busy"] = False
    assert H.resume(TAG + "r", TAG, lab.path)["outcome"] == "harmless_pass"
    assert _result(TAG + "r")["resumed_from"] == TAG
    assert lab.state["calls"] == ["train_with_d3_changes.py", "extract_tb.py", "fixed_eval.py"]  # 학습·추출은 다시 안 함
    with pytest.raises(FileExistsError):
        H.resume(TAG + "s", TAG, lab.path)
    with pytest.raises(ValueError, match="이어 하기"):
        H.resume(TAG + "t", TAG + "r", lab.path)


@needs_inputs
def test_a_resumed_result_is_not_resumed_again_even_when_it_ends_in_infrastructure(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]  # 시도 1과 이어 하기 모두 사후 확인에서 인프라 원인 invalid
    H.execute(TAG, 2026, lab.path)
    assert H.resume(TAG + "r", TAG, lab.path)["failure"]["stage"] == "postflight"
    with pytest.raises(ValueError, match="이어 하기"):
        H.resume(TAG + "s", TAG + "r", lab.path)
    assert H.seed_state(2026)["next"] == ["retry"]
    lab.state["gpu_processes"] = []
    assert H.execute(TAG + "b", 2026, lab.path, retry_of=TAG + "r")["outcome"] == "harmless_pass"


@needs_inputs
def test_a_training_watchdog_timeout_allows_exactly_one_retry_under_a_new_run_name(lab):
    lab.state["timeout"] = {"train_with_d3_changes.py"}
    out = H.execute(TAG, 2026, lab.path)
    assert (out["invalid_cause"], out["failure"]["stage"]) == ("infrastructure", "training")
    assert H.seed_state(2026)["next"] == ["retry"]
    with pytest.raises(ValueError, match="이어 하기"):
        H.resume(TAG + "r", TAG, lab.path)
    lab.state["timeout"] = set()
    with pytest.raises(ValueError, match="시도 2"):  # 다른 seed의 기록으로는 재시도하지 않는다
        H.execute(TAG + "b", 2027, lab.path, retry_of=TAG)
    assert H.execute(TAG + "b", 2026, lab.path, retry_of=TAG)["outcome"] == "harmless_pass"
    assert (_result(TAG + "b")["attempt"], _result(TAG + "b")["run_name"]) == (2, "d3_harmless_v1_s2026_a2")
    with pytest.raises(ValueError, match="시도 2"):
        H.execute(TAG + "c", 2026, lab.path, retry_of=TAG + "b")


@needs_inputs
def test_the_first_valid_result_is_final_for_its_seed(lab):
    lab.state["gpu_processes"] = [{"pid": 4242}]  # 사후 확인에서 인프라 원인 invalid
    assert H.execute(TAG, 2026, lab.path)["failure"]["stage"] == "postflight"
    lab.state["gpu_processes"] = []
    assert H.execute(TAG + "b", 2026, lab.path, retry_of=TAG)["outcome"] == "harmless_pass"
    calls = list(lab.state["calls"])
    with pytest.raises(FileExistsError):
        H.resume(TAG + "r", TAG, lab.path)
    assert lab.state["calls"] == calls and H.seed_state(2026)["final"] is True


@needs_inputs
@pytest.mark.parametrize("change", ["bound_paths", "protocol"])
def test_a_source_change_during_evaluation_is_recorded_as_invalid(lab, change):
    lab.state["change_commit_at"] = 3
    lab.state["bound_same"] = change == "protocol"
    lab.state["change_protocol"] = change == "protocol"
    out = H.execute(TAG, 2026, lab.path)
    assert (out["outcome"], out["failure"]["stage"]) == ("invalid", "postflight")


@needs_inputs
def test_an_unrelated_commit_during_the_run_is_recorded_not_rejected(lab):
    lab.state["change_commit_at"] = 2
    lab.state["bound_same"] = True
    assert H.execute(TAG, 2026, lab.path)["outcome"] == "harmless_pass"
    assert _result(TAG)["commits_seen"] == ["0" * 40, "1" * 40]


@needs_inputs
def test_a_crash_while_judging_still_writes_an_invalid_result(lab, monkeypatch):
    def broken(*_args):
        raise RuntimeError("judge")
    monkeypatch.setattr(H, "judge", broken)
    out = H.execute(TAG, 2026, lab.path)
    assert (out["outcome"], out["invalid_cause"], out["failure"]["stage"]) == ("invalid", "runner", "judge")
    assert _result(TAG)["checks"] is None and H.seed_state(2026)["next"] == []


# --- 두 seed 종합 ---
@needs_inputs
def test_summary_waits_until_both_seeds_are_final_and_then_passes(lab):
    with pytest.raises(ValueError, match="결정되지"):
        H.summarize("d3_harmless_summary_20261013")
    H.execute(TAG, 2026, lab.path)
    with pytest.raises(ValueError, match="결정되지"):
        H.summarize("d3_harmless_summary_20261013")
    H.execute("d3_harmless_smoke_20261013_s2027", 2027, lab.path)
    with pytest.raises(ValueError, match="tag"):
        H.summarize(TAG)
    assert H.summarize("d3_harmless_summary_20261013")["outcome"] == "pass"
    summary = H.read(H.RESULTS / "d3_harmless_summary_20261013/summary.json")
    assert summary["prediction_match"] is True and summary["wording"] == H.FIXED["wording"]["pass"]
    assert {seed: row["outcome"] for seed, row in summary["seeds"].items()} == {"2026": "harmless_pass",
                                                                               "2027": "harmless_pass"}
    with pytest.raises(FileExistsError):
        H.summarize("d3_harmless_summary_20261013")


@needs_inputs
def test_one_harmless_failure_decides_the_summary_without_the_other_seed(lab):
    lab.state["report"] = {"lin": 1.08}
    assert H.execute(TAG, 2026, lab.path)["outcome"] == "harmless_fail"
    out = H.summarize("d3_harmless_summary_20261013")
    summary = H.read(H.RESULTS / "d3_harmless_summary_20261013/summary.json")
    assert out["outcome"] == "fail" and summary["seeds"]["2027"] is None
    assert summary["seeds"]["2026"]["margins"]["err_xy_ratio"] is False


@needs_inputs
def test_a_seed_without_a_valid_result_and_no_next_step_makes_the_summary_invalid(lab):
    lab.state["train_exit"] = 1
    assert H.execute(TAG, 2026, lab.path)["invalid_cause"] == "training_process"
    with pytest.raises(ValueError, match="결정되지"):  # 남은 seed가 harmless_fail이면 fail이므로 아직 정하지 않는다
        H.summarize("d3_harmless_summary_20261013")
    lab.state["train_exit"] = 0
    H.execute("d3_harmless_smoke_20261013_s2027", 2027, lab.path)
    assert H.summarize("d3_harmless_summary_20261013")["outcome"] == "invalid"
    with pytest.raises(FileExistsError):  # 프로토콜마다 공개 요약은 하나
        H.summarize("d3_harmless_summary_20261013b")


@needs_inputs
def test_a_harmless_failure_on_the_other_seed_turns_an_invalid_seed_into_a_failed_summary(lab):
    lab.state["train_exit"] = 1
    H.execute(TAG, 2026, lab.path)
    lab.state["train_exit"] = 0
    lab.state["report"] = {"falls": 32}
    assert H.execute("d3_harmless_smoke_20261013_s2027", 2027, lab.path)["outcome"] == "harmless_fail"
    assert H.summarize("d3_harmless_summary_20261013")["outcome"] == "fail"


def test_main_reports_a_busy_gpu_and_requires_a_seed(monkeypatch, capsys):
    monkeypatch.setattr(H, "execute", lambda *_args: {"status": "gpu_busy_no_launch", "busy_reasons": ["x"]})
    assert H.main(["execute", "--tag", TAG, "--seed", "2026"]) == 3
    assert "gpu_busy_no_launch" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        H.main(["execute", "--tag", TAG])
