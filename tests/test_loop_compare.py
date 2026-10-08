"""P1-A 선택 방식 비교: probe 결과표·진단 순위·정답표로 방식별 표를 만든다(합성 자료)."""
import json

from test_offline_replay import _offline

REF_MEAS = {"P_noise": {"noise_ratio": 1.0}, "P_value": {"critic_change": 1.5},
            "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0},
            "P_torque": {"low_speed_saturation": 0.001}, "P_physics": {"mismatch_count": 0},
            "P_episode": {"timeout_ratio": 1.0}}


def _probe_file(d, name, override):
    meas = {**REF_MEAS, **override}
    rep = {"name": name, "probes": {p: {"measurement": m, "exit": 0, "elapsed_s": 10.0} for p, m in meas.items()}}
    (d / f"{name}.json").write_text(json.dumps(rep), encoding="utf-8")


def test_compare_counts_correct_and_wrong_conclusions(tmp_path):
    probes = tmp_path / "probes_tag" / "probes"
    probes.mkdir(parents=True)
    _probe_file(probes, "baseline_p0c_s2026", {})
    _probe_file(probes, "h01_s2026", {"P_episode": {"timeout_ratio": 0.05}})   # termination
    _probe_file(probes, "h02_s2026", {"P_value": {"critic_change": 0.0}})      # optimizer
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"cases": {"h01": {"category": "termination"}, "h02": {"category": "optimizer"}}}),
                   encoding="utf-8")
    ranks = tmp_path / "agent_res"
    ranks.mkdir()
    rows = [{"case_id": "h01", "seed": 2026, "mode": "agent", "infra_error": False,
             "ranking": ["termination", "reward", "physics"]},
            {"case_id": "h02", "seed": 2026, "mode": "agent", "infra_error": False,
             "ranking": ["reward", "physics", "actuator"]}]  # 참 범주가 상위 3에 없다
    (ranks / "seed2026.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    proc = _offline("evals/loop_compare.py", "--probes", tmp_path / "probes_tag", "--out", out, "--key", key,
                    "--ranking", f"agent={ranks}")
    assert proc.returncode == 0, proc.stderr
    table = json.loads((out / "loop_compare.json").read_text(encoding="utf-8"))["table"]
    assert table["exhaustive"]["correct"] == 2 and table["fixed"]["correct"] == 2
    d = table["discriminate:agent"]
    assert d["correct"] == 1 and d["none_supported"] == 1 and d["mean_probes"] <= 3
    assert table["random"]["cells"] == 2 * 5
    rows = [json.loads(x) for x in (out / "loop_compare.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"run", "seed", "truth", "strategy", "status", "conclusion", "probes_used", "gpu_s_est",
            "gpu_s_measured", "steps"} <= set(rows[0])
