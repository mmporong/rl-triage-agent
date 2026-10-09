"""T1 외부 원인 결함: Isaac 없이 확인할 수 있는 부분(결함 표, 본문 교체 호환, 설정 적용, 위임, 사례·판정)."""
import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from test_offline_replay import ROOT, _offline

from rl_triage import triage_tools as T


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ext():
    return _load("bench/external_faults.py", "external_faults_under_test")


def test_fault_table_categories_and_pairs():
    X = _ext()
    assert {k for k, f in X.FAULTS.items() if f["category"] is None} == {"X1_REF", "X3_REF"}
    for k in ("X1", "X2", "X3"):
        assert X.FAULTS[k]["category"] in T.MECHANISMS and X.FAULTS[k]["source"].startswith("https://github.com/isaac-sim/")
    assert X.FAULTS["X1"]["pair"] == X.FAULTS["X1_REF"]["pair"] == "X1" and X.FAULTS["X2"]["pair"] is None


def test_replacements_match_upstream_signatures_and_have_no_closures():
    """본문 교체는 인자 이름·개수가 같아야 교체 대상의 기본 인자(asset_cfg=SceneEntityCfg("robot"))가 맞는다."""
    X = _ext()
    for fault, (module, name, fn) in X.SWAPS.items():
        code = fn.__code__
        assert module == "isaaclab.envs.mdp.events"
        assert code.co_varnames[:code.co_argcount] == X.UPSTREAM_ARGS[name], fault
        assert not code.co_freevars


def _cfg(base_com=None, push=None, history=3):
    return SimpleNamespace(events=SimpleNamespace(base_com=base_com, push_robot=push),
                           scene=SimpleNamespace(contact_forces=SimpleNamespace(history_length=history)))


def _upstream():
    return SimpleNamespace(
        base_com=SimpleNamespace(mode="startup", params={"com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05),
                                                                       "z": (-0.01, 0.01)}}),
        push_robot=SimpleNamespace(mode="interval", interval_range_s=(10.0, 15.0),
                                   params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}))


def test_apply_cfg_uses_upstream_terms_and_only_on_stock_go2():
    X = _ext()
    for fault in ("X1_REF", "X1"):
        cfg = _cfg()
        changed = X.apply_cfg(fault, cfg, _upstream())
        assert cfg.events.base_com.mode == "reset" and changed["events.base_com"]["com_range"]["x"] == (-0.05, 0.05)
        assert cfg.events.push_robot is None and cfg.scene.contact_forces.history_length == 3
    cfg = _cfg()
    X.apply_cfg("X3", cfg, _upstream())
    assert cfg.events.push_robot.interval_range_s == (10.0, 15.0) and cfg.events.base_com is None
    cfg = _cfg()
    assert X.apply_cfg("X2", cfg, _upstream()) == {"scene.contact_forces.history_length": 0}
    assert cfg.scene.contact_forces.history_length == 0
    with pytest.raises(RuntimeError, match="순정 Go2 flat"):
        X.apply_cfg("X1", _cfg(base_com=object()), _upstream())
    with pytest.raises(RuntimeError, match="순정 Go2 flat"):
        X.apply_cfg("X3_REF", _cfg(push=object()), _upstream())
    with pytest.raises(RuntimeError, match="history_length=0"):
        X.apply_cfg("X2", _cfg(history=0), _upstream())
    with pytest.raises(SystemExit, match="모르는 외부 결함"):
        X.apply_cfg("X9", _cfg(), _upstream())


def test_push_overwrite_replaces_velocity_with_sample():
    """#1584 이전 의미: 현재 속도와 무관하게 표본값이 그대로 쓰인다(numpy 대역으로 본문만 실행)."""
    X = _ext()
    written = {}

    class Asset:
        device = "cpu"
        data = SimpleNamespace(root_vel_w=np.full((4, 6), 1.0))

        def write_root_velocity_to_sim(self, vel, env_ids):
            written["vel"], written["ids"] = vel.copy(), env_ids

    asset = Asset()
    env = SimpleNamespace(scene={"robot": asset})
    fake = {"torch": SimpleNamespace(tensor=lambda x, device=None: np.array(x, dtype=float)),
            "math_utils": SimpleNamespace(sample_uniform=lambda lo, hi, shape, device=None: np.broadcast_to(hi, shape))}
    fn = types.FunctionType(X._push_overwrite_pre1584.__code__, fake)
    fn(env, np.array([0, 2]), {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, SimpleNamespace(name="robot"))
    assert written["vel"].tolist() == [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]] * 2  # 1.0 + 0.5가 아니다
    assert written["ids"].tolist() == [0, 2]


def test_hidden_faults_delegates_x_names_and_keeps_p0c_table():
    H = _load("bench/hidden_faults.py", "hidden_faults_t1_test")
    assert sorted(H.FAULTS) == ["F1", "F2", "F3", "F4", "F5", "F6", "NONE"]  # P0-C 사례 생성이 바뀌지 않는다
    assert H._external("F5") is None and H._external("NONE") is None
    assert "X3" in H._external("X3").FAULTS
    with pytest.raises(SystemExit, match="모르는 TRIAGE_FAULT"):
        H.arm("X9")
    sys.modules.pop("external_faults", None)


def test_t1_key_plan_and_run_mapping():
    C = _load("bench/t1_cases.py", "t1_cases_under_test")
    a, b = C.make_key(5, ["X1", "X2", "X3"]), C.make_key(5, ["X1", "X2", "X3"])
    assert a == b and sorted(a["cases"]) == ["e01", "e02", "e03"]
    by_fault = {c["fault"]: (cid, c) for cid, c in a["cases"].items()}
    assert by_fault["X2"][1]["reference"] == "baseline_p0c" and by_fault["X2"][1]["reference_fault"] == "NONE"
    x1_id, x1 = by_fault["X1"]
    assert x1["reference"] == f"ref_{x1_id}" and x1["reference_fault"] == "X1_REF" and x1["category"] == "physics"
    runs = C.plan(a)
    assert len(runs) == 3 * (3 + 2) and not any(r["case_id"] == "baseline_p0c" for r in runs)
    assert all("fault" not in r for r in runs)
    assert C.fault_of_run(x1_id, a) == "X1" and C.fault_of_run(f"ref_{x1_id}", a) == "X1_REF"
    assert C.fault_of_run("baseline_p0c", a) == "NONE"
    stock = C.make_key(5, ["X1", "X3"], stock_reference=("X1",))
    assert {c["fault"]: c["reference"] for c in stock["cases"].values()}["X1"] == "baseline_p0c"
    with pytest.raises(SystemExit):
        C.make_key(5, ["X1", "X1"])
    with pytest.raises(SystemExit):
        C.make_key(5, ["X4"])


def test_run_probes_resolves_t1_names():
    R = _load("evals/run_probes.py", "run_probes_under_test")
    C = _load("bench/t1_cases.py", "t1_cases_for_probes")
    key = C.make_key(5, ["X1", "X2", "X3"])
    x2 = next(cid for cid, c in key["cases"].items() if c["fault"] == "X2")
    assert R.fault_of("t1cal_X1_REF_s7", None) == "X1_REF" and R.fault_of("t1cal_X2_s7", None) == "X2"
    assert R.fault_of(f"{x2}_s2026", None, key) == "X2"
    assert R.fault_of("baseline_p0c_s2027", None, key) == "NONE"
    with pytest.raises(SystemExit, match="answer_key_t1"):
        R.fault_of("e01_s2026", None, None)


REF_MEAS = {"P_noise": {"noise_ratio": 1.0}, "P_value": {"critic_change": 1.5},
            "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0},
            "P_torque": {"low_speed_saturation": 0.001}, "P_physics": {"mismatch_count": 0},
            "P_episode": {"timeout_ratio": 1.0}}


def _probe_file(d, name, override):
    meas = {**REF_MEAS, **override}
    rep = {"name": name, "probes": {p: {"measurement": m, "exit": 0, "elapsed_s": 10.0} for p, m in meas.items()}}
    (d / f"{name}.json").write_text(json.dumps(rep), encoding="utf-8")


def test_loop_compare_uses_each_case_reference(tmp_path):
    """짝 기준(ref_e01)과 순정 기준(baseline_p0c)이 섞여도 사례마다 정답표의 기준과 비교한다."""
    probes = tmp_path / "t" / "probes"
    probes.mkdir(parents=True)
    _probe_file(probes, "baseline_p0c_s2026", {})
    _probe_file(probes, "ref_e01_s2026", {"P_value": {"critic_change": 0.2}})  # 짝 기준 자체가 작다
    _probe_file(probes, "e01_s2026", {"P_value": {"critic_change": 0.2}})      # 짝 기준과 같다 → 정상
    _probe_file(probes, "e02_s2026", {"P_episode": {"timeout_ratio": 0.05}})
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"cases": {"e01": {"category": "physics", "reference": "ref_e01"},
                                         "e02": {"category": "termination", "reference": "baseline_p0c"}}}),
                   encoding="utf-8")
    out = tmp_path / "out"
    proc = _offline("evals/loop_compare.py", "--probes", tmp_path / "t", "--out", out, "--key", key)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(x) for x in (out / "loop_compare.jsonl").read_text(encoding="utf-8").splitlines()]
    ex = {r["run"]: r for r in rows if r["strategy"] == "exhaustive"}
    assert set(ex) == {"e01_s2026", "e02_s2026"}  # 기준 실행은 채점하지 않는다
    assert all(s["outcome"] == "normal" for s in ex["e01_s2026"]["steps"])  # baseline과 비교했다면 P_value가 abnormal
    assert ex["e02_s2026"]["conclusion"] == "termination"
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "probes").mkdir()
    _probe_file(missing / "probes", "e01_s2026", {})
    proc = _offline("evals/loop_compare.py", "--probes", missing, "--out", tmp_path / "o2", "--key", key)
    assert proc.returncode != 0 and "ref_e01_s2026" in proc.stderr


def _report(lin, fall=0.0, survival=20.0, yaw=0.3):
    return {"protocol": "fixed_eval_v1", "protocol_sha256_lf": "x", "horizon_steps": 1000, "step_dt": 0.02,
            "standstill": {"lin_vel_rmse_mps": 1.18, "yaw_rate_rmse_radps": 0.83},
            "metrics": {"fall_rate": fall, "mean_survival_s": survival, "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw}}


def test_calibration_judge_requires_walking_reference_and_breakage():
    K = _load("evals/t1_calibration.py", "t1_calibration_under_test")
    smoke = {"X1": {"decision": "pair"}, "X2": {"decision": "include"}, "X3": {"decision": "exclude"}}
    reports = {"t1cal_X1_REF_s7": _report(0.15), "t1cal_X1_s7": _report(0.30),
               "p0ccal_NONE_s7": _report(0.145), "t1cal_X2_s7": _report(0.155)}
    x1 = K.judge("X1", smoke, reports)
    assert x1["include"] and x1["reference"] == "t1cal_X1_REF_s7" and x1["failed"] == ["err_xy_ratio"]
    x2 = K.judge("X2", smoke, reports)  # 0.155/0.145 = 1.07 < 1.10 → 상류 크기에서 안 깨짐
    assert not x2["include"] and x2["reason"] == "상류 크기에서 깨지지 않음"
    assert K.judge("X3", smoke, reports) == {"include": False, "reason": "smoke exclude"}
    reports["t1cal_X1_REF_s7"] = _report(0.9)  # 기준이 제자리(1.18)의 절반 이상 → 걷지 않음
    assert K.judge("X1", smoke, reports)["reason"] == "기준이 걷지 않음"
    stock = {**smoke, "X1": {"decision": "stock_reference"}}
    assert K.reference_of("X1", stock) == "p0ccal_NONE_s7"


def test_smoke_judgement_table():
    S = _load("bench/smoke_external.py", "smoke_external_under_test")
    r = {"NONE": {"teleport": {"lost_frac": 0.0}, "contact": {"touched_frac": 1.0, "stale_frac_of_touched": 0.0}},
         "X1_REF": {"com": {"outside_range_frac": 0.0}, "teleport": {"lost_frac": 0.0}},
         "X1": {"com": {"outside_range_frac": 0.9}, "teleport": {"lost_frac": 0.0}},
         "X2": {"contact": {"touched_frac": 1.0, "stale_frac_of_touched": 1.0}},
         "X3_REF": {"push": {"vx_after_mean": 0.98}}, "X3": {"push": {"vx_after_mean": 0.02}}}
    d = S.judge(r)
    assert d["X1"]["decision"] == "pair" and d["X2"]["decision"] == "include" and d["X3"]["decision"] == "include"
    r["X1_REF"]["teleport"]["lost_frac"] = 1.0  # Bug 2 재현 → 순정 기준
    assert S.judge(r)["X1"]["decision"] == "stock_reference"
    r["X1"]["com"]["outside_range_frac"] = 0.1  # 누적 안 됨 → 제외
    assert S.judge(r)["X1"]["decision"] == "exclude"
    r["NONE"]["teleport"]["lost_frac"] = 1.0  # 순정에서도 텔레포트가 안 되면 측정 절차 문제
    assert S.judge(r)["X1"]["decision"] == "inconclusive"
    r["X2"]["contact"]["stale_frac_of_touched"] = 0.0
    assert S.judge(r)["X2"]["decision"] == "exclude"
    r["X3"]["push"]["vx_after_mean"] = 1.0
    assert S.judge(r)["X3"]["decision"] == "exclude"
