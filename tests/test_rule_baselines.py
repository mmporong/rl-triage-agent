"""P0-A2 결정적 기준선: 규칙·템플릿·B0가 offline으로 돌고, dev 표·임계값 절차·holdout 동결이 지켜지는지 검사한다."""
import importlib.util
import json
from collections import Counter

import pytest
import yaml

from rl_triage import rules as R
from rl_triage import triage_tools as T
from test_offline_replay import CASES, KEY, ROOT, USER_PATH, _offline

# 규칙을 dev에 맞춰 정했으므로 dev 만점은 성능 근거가 아니다. 이 표가 바뀌면 규칙이 바뀐 것이다.
DEV_EXPECTED = {"rule_prior": (20, 6, 10), "rule_features": (20, 20, 20), "rule_template": (20, 20, 20),
                "rule_b0": (20, 20, 20)}


def _summary(name):
    return json.loads((ROOT / "bench" / "runs" / f"{name}.telemetry.json").read_text(encoding="utf-8"))["summary"]


def _dev_cases():
    out = []
    for seed in R.DEV_SEEDS:
        env = yaml.load((ROOT / "bench" / "reference" / "params" / f"seed{seed}" / "env.yaml").read_text(encoding="utf-8"),
                        Loader=T._TolerantLoader)
        cfg, ref = R.ref_config(env), _summary(f"baseline_s{seed}")
        for c in CASES:
            out.append((f"s{seed}/{c['case_id']}", R.features(_summary(f"{c['case_id']}_s{seed}"), ref, cfg),
                        KEY["cases"][c["case_id"]]["category"]))
    return out


DEV = _dev_cases()


def _baselines_module():
    spec = importlib.util.spec_from_file_location("baselines_under_test", ROOT / "evals" / "baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("p0a2") / "dev"
    proc = _offline("evals/baselines.py", "--seeds", "7", "42", "--out", out)
    assert proc.returncode == 0, proc.stderr
    return out, proc


# ---------- 실행과 기록 ----------

def test_dev_baselines_run_offline_and_match_the_frozen_table(dev_run):
    out, proc = dev_run
    report = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    for mode, want in DEV_EXPECTED.items():
        t = report["table"][mode]["total"]
        assert (t["cells"], t["top1"], t["top2"]) == want, mode
    assert report["environment"]["model_calls"] == 0
    assert report["template_train_seeds"] == {"s7": [42], "s42": [7]}  # 평가 seed의 정답은 template에 쓰지 않는다
    text = proc.stdout + (out / "baselines.json").read_text(encoding="utf-8")
    assert not USER_PATH.search(text) and str(ROOT) not in text and ROOT.as_posix() not in text


def test_baseline_rows_rescore_with_replay(dev_run):
    out, _ = dev_run
    proc = _offline("evals/replay.py", out, "--json")
    assert proc.returncode == 0, proc.stderr
    tasks = json.loads(proc.stdout)["inputs"]["results"][0]["tasks"]["blind"]
    for mode, want in DEV_EXPECTED.items():
        assert (tasks[mode]["cells"], tasks[mode]["top1"], tasks[mode]["top2"]) == want, mode


def test_existing_output_and_unknown_seed_are_input_errors(dev_run, tmp_path):
    out, _ = dev_run
    proc = _offline("evals/baselines.py", "--seeds", "7", "--out", out)
    assert proc.returncode == 2 and "덮어쓰지 않는다" in proc.stderr
    proc = _offline("evals/baselines.py", "--seeds", "999", "--out", tmp_path / "x")
    assert proc.returncode == 2 and not (tmp_path / "x").exists()


def test_holdout_seed_requires_committed_rules(monkeypatch, tmp_path):
    B = _baselines_module()
    assert B.freeze_problem({"git_head": "abc", "git_dirty": True}, [7, 123])
    assert B.freeze_problem({"git_head": None, "git_dirty": None}, [123])
    assert B.freeze_problem({"git_head": "abc", "git_dirty": False}, [123]) is None
    assert B.freeze_problem({"git_head": None, "git_dirty": None}, [7, 42]) is None
    monkeypatch.setattr(B, "code_version", lambda *a, **k: {"git_head": "abc", "git_dirty": True, "sha256_lf": {}})
    assert B.main(["--seeds", "123", "--out", str(tmp_path / "h")]) == 2
    assert not (tmp_path / "h").exists()


# ---------- 규칙과 임계값 ----------

def test_prior_order_is_dev_label_frequency():
    counts = Counter(truth for _, _, truth in DEV)
    assert list(R.PRIOR_ORDER) == sorted(R.MECHANISMS, key=lambda m: (-counts[m], R.MECHANISMS.index(m)))
    assert list(R.MECHANISMS) == T.MECHANISMS


def test_every_rule_fires_on_dev_and_wrong_mechanism_rules_are_outranked():
    """임계값 절차의 음성 정의: 다른 범주 규칙이 발화한 dev 사례에서는 정답 범주 점수가 더 높다."""
    seen = set()
    for cell, f, truth in DEV:
        ranking, fired = R.feature_ranking(f)
        seen |= {r["rule"] for r in fired}
        assert ranking[0] == truth, (cell, fired)
        points = Counter()
        for r in fired:
            points[r["mechanism"]] += r["points"]
        assert all(points[truth] > p for m, p in points.items() if m != truth), (cell, fired)
    assert seen == {f"R{i}" for i in range(1, 11)}


def test_healthy_run_fires_no_rule_and_falls_back_to_prior_order():
    for seed in R.DEV_SEEDS:
        env = yaml.load((ROOT / "bench" / "reference" / "params" / f"seed{seed}" / "env.yaml").read_text(encoding="utf-8"),
                        Loader=T._TolerantLoader)
        ref = _summary(f"baseline_s{seed}")
        ranking, fired = R.feature_ranking(R.features(ref, ref, R.ref_config(env)))
        assert fired == [] and ranking == list(R.PRIOR_ORDER)


@pytest.mark.parametrize("factor", [0.8, 1.25])
@pytest.mark.parametrize("name", sorted(R.THRESHOLDS))
def test_dev_decisions_do_not_sit_on_a_threshold(monkeypatch, name, factor):
    monkeypatch.setitem(R.THRESHOLDS, name, R.THRESHOLDS[name] * factor)
    for cell, f, truth in DEV:
        assert R.feature_ranking(f)[0][0] == truth, (cell, name, factor)


def test_features_normalize_by_survival_time_and_reset_count():
    """Episode_Reward는 에피소드 합/제한 시간, Episode_Termination은 리셋 개수다(IMPLEMENTATION-ORDER 2절 C)."""
    def s(mean, mx=None):
        return {"mean_last_20pct": mean, "max": mx if mx is not None else mean}
    cfg = {"step_dt": 0.02, "episode_length_s": 20.0, "reward_weights": {"track_lin_vel_xy_exp": 1.5}}
    ref = {R.LEN: s(1000.0), "Episode_Reward/track_lin_vel_xy_exp": s(0.6),
           "Episode_Termination/time_out": s(1.0), "Episode_Termination/base_contact": s(0.0)}
    run = {R.LEN: s(500.0, 900.0), "Episode_Reward/track_lin_vel_xy_exp": s(0.3),
           "Episode_Termination/time_out": s(1.0), "Episode_Termination/base_contact": s(3.0)}
    f = R.features(run, ref, cfg)
    assert f["len_ratio"] == 0.5 and f["rate_ratio"]["track_lin_vel_xy_exp"] == pytest.approx(1.0)
    assert f["timeout_frac"] == 0.25 and f["len_pinned"] == pytest.approx(0.8)


# ---------- B0 설정 diff ----------

def test_b0_ranks_the_largest_mapped_change_and_ignores_identity_keys():
    ref = {"env.rewards.a.weight": 1.0, "agent.algorithm.learning_rate": 0.001, "agent.save_interval": 50,
           "agent.run_name": "baseline", "env.events.m.params.r": [-1.0, 3.0]}
    run = R.apply_overrides(ref, ["agent.algorithm.learning_rate=0.0012", "agent.save_interval=25",
                                  "env.events.m.params.r=[25.0,30.0]"])
    run["agent.run_name"] = "c02"
    diff = R.config_diff(ref, run)
    assert [d["key"] for d in diff] == ["agent.algorithm.learning_rate", "agent.save_interval", "env.events.m.params.r"]
    ranking, scored = R.b0_ranking(diff)
    assert ranking[:2] == ["physics", "optimizer"] and sorted(ranking) == sorted(R.MECHANISMS)
    assert {d["key"]: d["mechanism"] for d in scored}["agent.save_interval"] is None
    with pytest.raises(ValueError, match="없는 키"):
        R.apply_overrides(ref, ["env.rewards.missing.weight=1.0"])


def test_b0_change_magnitude():
    assert R._magnitude(1.5, -1.5) == R._magnitude(1.0, 0.0) == float("inf")
    assert R._magnitude(0.99, 0.99) == 0.0 and R._magnitude(0.25, 1.5) == pytest.approx(1.7918, abs=1e-4)
    assert R._magnitude([-1.0, 3.0], [25.0, 30.0]) == float("inf") and R._magnitude("a", "b") == 1.0
