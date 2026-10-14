"""두 번째 task 단계 1(ANYmal-C 정상 seed 셋): 단계 0 입력 결합, seed별 실행·재시도, 밴드·비 기준 probe·R1 오경보 판정(GPU 없이)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import run_anymal_c_band as B  # noqa: E402

G, U, FE = B.G, B.U, B.FE
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
P0C_PROBES = ROOT / "evals/results/p1a_probes_holdout_20261009/probes"
P0C_TELEMETRY = ROOT / "bench/runs_p0c"
EVAL_ENV = {"task": G.TASK, "disabled": ["push_robot", "base_com"], "was_enabled": {"push_robot": True, "base_com": True},
            "now_enabled": {"push_robot": False, "base_com": False}}
TAG = "anymal_c_band_20261015"


def test_the_stage0_inputs_must_be_the_committed_pass(tmp_path, monkeypatch):
    assert set(B.stage0_inputs()) == set(B.STAGE0_FILES)
    for relative in B.STAGE0_FILES.values():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    monkeypatch.setattr(B, "ROOT", tmp_path)
    result = U.read(tmp_path / B.STAGE0_FILES["result"])
    (tmp_path / B.STAGE0_FILES["result"]).write_text(json.dumps({**result, "outcome": "fail"}), encoding="utf-8")
    with pytest.raises(ValueError, match="pass"):
        B.stage0_inputs()
    (tmp_path / B.STAGE0_FILES["result"]).write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / B.STAGE0_FILES["telemetry"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="텔레메트리"):
        B.stage0_inputs()


def test_the_protocol_records_criteria_stop_rule_and_no_go2_comparison():
    fixed = B.FIXED
    assert set(fixed["criteria"]) == {"new_runs_walk", "absolute_probes_normal", "ratio_probes_cross_seed_normal",
                                      "no_tracking_false_alarm"}
    assert fixed["seeds"] == [2044, 2045] and fixed["stage0_seed"] == 2043 and "단계 2를 하지 않는다" in fixed["stop_rule"]
    assert "Go2와의 수치 비교가 아니다" in fixed["scope"] and "직접" not in json.dumps(fixed, ensure_ascii=False)
    dynamic = {"source_files": {}, "isaaclab_sources": {}, "eval_protocol_sha256_lf": "e", "stage0_inputs": {}}
    B.check_protocol({**copy.deepcopy(fixed), **dynamic}, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        B.check_protocol({**copy.deepcopy(fixed), **dynamic, "seeds": [2044, 2046]}, dynamic)
    assert "src/rl_triage/tracking_alarm.py" in B.CODE and "evals/run_anymal_c_gate.py" in B.CODE


# --- 실행 흐름(가짜 Isaac Lab·소유 Job) ---
def _report(source: str, name: str, sha: str) -> dict:
    report = U.read(P0C_RUNS / f"{source}.json")
    report.update(name=name, protocol=U.read(G.EVAL_PROTOCOL)["name"], task=G.TASK, protocol_sha256_lf=U.sha(G.EVAL_PROTOCOL, True),
                  action_scale=G.ACTION_SCALE, checkpoint={"file": G.FINAL_CHECKPOINT, "sha256": sha})
    return report


def _probe_report(source: str, name: str, seed: int, sha: str, torque: float | None = None,
                  episode: float | None = None) -> dict:
    report = U.read(P0C_PROBES / f"{source}.json")
    report.update(name=name, task=G.TASK, seed=seed, checkpoint={"file": G.FINAL_CHECKPOINT, "sha256": sha})
    if torque is not None:
        report["probes"]["P_torque"]["measurement"]["low_speed_saturation"] = torque
    if episode is not None:
        report["probes"]["P_episode"]["measurement"]["timeout_ratio"] = episode
    return report


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    import run_runtime_readback_smoke as P
    fake = tmp_path / "IsaacLab"
    for relative in B.STAGE0_FILES.values():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    protocol = {**copy.deepcopy(B.FIXED), "source_files": {}, "isaaclab_sources": {},
                "eval_protocol_sha256_lf": U.sha(G.EVAL_PROTOCOL, True), "stage0_inputs": {}}
    protocol_path = tmp_path / "band.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    for module, name, value in ((B, "ROOT", tmp_path), (B, "PROTOCOL", protocol_path), (B, "PRIVATE", tmp_path / "private"),
                                (B, "RESULTS", tmp_path / "evals/results"), (G, "ROOT", tmp_path),
                                (G, "RESULTS", tmp_path / "evals/results"), (G, "RUNS_DIR", tmp_path / "bench/runs_anymal")):
        monkeypatch.setattr(module, name, value)
    sources = {2044: "baseline_p0c_s2027", 2045: "baseline_p0c_s2028"}
    state = {"calls": [], "gpu_busy": False, "timeout": set(), "readback": {}, "eval": dict(sources), "probes": dict(sources),
             "telemetry": dict(sources), "torque": {}, "episode": {}}
    monkeypatch.setattr(B, "verify", lambda _isaaclab: (protocol, "0" * 40))
    monkeypatch.setattr(U, "_admission", lambda: {"busy": state["gpu_busy"], "processes": [],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(U.time, "sleep", lambda _s: None)

    def run_owned(folder, args, cwd, watchdog_s):
        name, stdout, seed = Path(args[1]).name, "", None
        if name == "train_with_fault.py":
            assert os.environ["TRIAGE_FAULT"] == "NONE"
            seed, run_name = int(args[args.index("--seed") + 1]), args[args.index("--run_name") + 1]
            state["calls"].append(f"train:{seed}")
            run_dir = fake / "logs/rsl_rl/anymal_c_flat" / f"2026-10-15_00-00-00_{run_name}"
            (run_dir / "params").mkdir(parents=True)
            (run_dir / "params/env.yaml").write_text(
                f"seed: {seed}\nscene:\n  num_envs: 4096\nactions:\n  joint_pos:\n    scale: 0.5\n", encoding="utf-8")
            (run_dir / "params/agent.yaml").write_text(f"seed: {seed}\nmax_iterations: 300\n", encoding="utf-8")
            (run_dir / G.FIRST_CHECKPOINT).write_bytes(b"zero")
            (run_dir / G.FINAL_CHECKPOINT).write_bytes(f"final {run_name}".encode())
            stdout = "" if state["readback"].get(seed) is False else '[hidden_fault] {"fault": "NONE"}\n'
            timed_out = seed in state["timeout"]
            if timed_out:
                state["timeout"].discard(seed)
        else:
            timed_out = False
            if name == "extract_tb.py":
                seed = int(Path(args[3]).name.split("_s")[1][:4])
                state["calls"].append(f"extract:{seed}")
                shutil.copy(P0C_TELEMETRY / f"{state['telemetry'][seed]}.telemetry.json", args[3])
            elif name == "fixed_eval_task.py":
                job = U.read(Path(args[args.index("--jobs") + 1]))["jobs"][0]
                seed = int(job["name"].split("_s")[1][:4])
                state["calls"].append(f"eval:{seed}")
                Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(job["output"]).write_text(json.dumps(_report(state["eval"][seed], job["name"], U.sha(Path(job["checkpoint"])))),
                                               encoding="utf-8")
                stdout = FE.PREFIX + json.dumps(EVAL_ENV) + "\n"
            else:
                jobs = U.read(Path(args[args.index("--jobs") + 1]))
                job, seed = jobs["jobs"][0], jobs["seed"]
                assert jobs["task"] == G.TASK and jobs["fault"] == "NONE"
                state["calls"].append(f"probes:{seed}")
                Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(job["output"]).write_text(json.dumps(_probe_report(state["probes"][seed], job["name"], seed,
                                                                        U.sha(Path(job["checkpoint"])), state["torque"].get(seed),
                                                                        state["episode"].get(seed))),
                                               encoding="utf-8")
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out, "error_type": None,
                "exit_code": 124 if timed_out else 0, "cleanup_verified": True, "owned_process_identities": {"4242": 1},
                "wall_s": 560.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake, state=state, root=tmp_path)


def _result(lab) -> dict:
    return U.read(lab.root / "evals/results" / TAG / "result.json")


def test_both_seeds_run_then_the_summary_derives_bands_and_passes(lab):
    out = B.execute(TAG, lab.path)
    assert out["outcome"] == "pass" and out["failed_criteria"] == []
    assert lab.state["calls"] == ["train:2044", "extract:2044", "eval:2044", "probes:2044",
                                  "train:2045", "extract:2045", "eval:2045", "probes:2045"]
    result = _result(lab)
    analysis = result["analysis"]
    assert len(analysis["comparisons"]) == 6 and len(analysis["ratio_probes"]) == 12
    assert set(analysis["tracking_alarm"]) == {"2043", "2044", "2045"} and not any(a["warned"] for a in analysis["tracking_alarm"].values())
    assert set(analysis["bands"]) == {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}
    assert result["prediction_match"] is True and result["model_calls"] == 0
    text = json.dumps(result)
    assert "C:" not in text and "LIMMM" not in text and "TRIAGE_FAULT" not in os.environ
    with pytest.raises(FileExistsError):
        B.summarize(TAG, lab.path)


def test_a_busy_gpu_or_a_used_seed_launches_nothing(lab):
    lab.state["gpu_busy"] = True
    assert B.execute(TAG, lab.path)["status"] == "gpu_busy_no_launch" and lab.state["calls"] == []
    lab.state["gpu_busy"] = False
    (lab.path / "logs/rsl_rl/unitree_go2_flat/2026-10-01_00-00-00_x_s2044").mkdir(parents=True)
    with pytest.raises(ValueError, match="seed 2044"):
        B.execute(TAG, lab.path)
    assert lab.state["calls"] == []


def test_an_infrastructure_failure_of_one_seed_is_retried_once_then_summarized(lab):
    lab.state["timeout"] = {2044}
    out = B.execute(TAG, lab.path)
    assert out["status"] == "retry_needed" and out["retry"] == [f"retry --tag {TAG} --seed 2044"]
    assert "train:2045" in lab.state["calls"] and not (lab.root / "evals/results" / TAG / "result.json").exists()
    with pytest.raises(ValueError, match="시도 2"):
        B.retry(TAG, 2045, lab.path)  # 2045는 유효했다
    final = B.retry(TAG, 2044, lab.path)
    result = _result(lab)
    assert final["outcome"] == "pass" and result["seed_records"]["2044"]["attempt"] == 2
    assert result["seed_records"]["2044"]["run_name"] == "anymal_c_band_none_s2044_a2"


@pytest.mark.parametrize(("setup", "criterion"), [
    (lambda s: s["eval"].update({2045: "h02_s2027"}), "new_runs_walk"),
    (lambda s: s["episode"].update({2044: 0.5}), "absolute_probes_normal"),
    (lambda s: s["torque"].update({2044: 0.05}), "ratio_probes_cross_seed_normal"),
    (lambda s: s["telemetry"].update({2045: "h02_s2027"}), "no_tracking_false_alarm"),
])
def test_a_criterion_miss_fails_and_stops(lab, setup, criterion):
    setup(lab.state)
    out = B.execute(TAG, lab.path)
    assert out["outcome"] == "fail" and criterion in out["failed_criteria"]
    assert _result(lab)["wording"] == B.FIXED["wording"]["fail"]


def test_a_non_infrastructure_invalid_seed_makes_the_summary_invalid(lab):
    lab.state["readback"][2045] = False
    out = B.execute(TAG, lab.path)
    result = _result(lab)
    assert out["outcome"] == "invalid" and result["seed_records"]["2045"]["invalid_cause"] == "injection"
    assert result["analysis"] is None and "eval:2045" not in lab.state["calls"]


def test_summarize_waits_for_every_seed_and_records_commits(lab):
    with pytest.raises(ValueError, match="최종"):
        B.summarize(TAG, lab.path)
    with pytest.raises(ValueError, match="최종 기록"):
        B.normal_runs(TAG)
    B.execute(TAG, lab.path)
    record = U.read(lab.root / "evals/results" / TAG / "seeds" / "s2044_a1.json")
    assert record["commits_seen"] == ["0" * 40] and record["seed_evidence"]["hits"] == []
