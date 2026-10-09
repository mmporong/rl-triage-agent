"""진단 시간 재생: 반복 분모, 누락 비용, 비교 집합, 비공개 정보 경계를 검사한다."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_offline_replay import _offline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostic_cost", ROOT / "evals" / "diagnostic_cost.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def row(strategy, run="h01_s2026", elapsed=10.0, conclusion="physics", status="confirmed"):
    return {"strategy": strategy, "run": run, "gpu_s_measured": elapsed, "probes_used": 2,
            "truth": "physics", "conclusion": conclusion, "correct": conclusion == "physics", "status": status}


def test_random_repeats_do_not_inflate_cases_or_cohort_time():
    rows = [row("exhaustive", elapsed=20)] + [row(f"random{s}", elapsed=10 + s) for s in range(5)]
    result = D.summarize(rows)["random"]
    assert (result["unique_cases"], result["evaluation_cells"], result["order_repeats"]) == (1, 5, 5)
    assert result["probe_measurement_s_per_repeat"] == 12
    assert result["correct_per_repeat"] == 1
    assert result["relative_probe_time_vs_exhaustive"] == pytest.approx(20 / 12, abs=0.0001)


@pytest.mark.parametrize("bad", [None, -1, float("nan"), True])
def test_missing_or_invalid_time_is_rejected(bad):
    with pytest.raises(ValueError):
        D.summarize([row("exhaustive", elapsed=bad)])


def test_missing_cases_and_different_confirmation_rates_remain_visible():
    rows = [row("exhaustive"), row("exhaustive", "h02_s2026"), row("partial"),
            row("agent"), row("agent", "h02_s2026", conclusion=None, status="none_supported")]
    result = D.summarize(rows)
    assert not result["partial"]["same_cases_as_exhaustive"]
    assert result["partial"]["relative_probe_time_vs_exhaustive"] is None
    assert result["agent"]["correct_rate"] == 0.5
    assert result["agent"]["none_supported_per_repeat"] == 1
    with pytest.raises(ValueError, match="사례 집합"):
        D.summarize([row("exhaustive"), row("random0"), row("random1", "h02_s2026")])
    with pytest.raises(ValueError, match="중복"):
        D.summarize([row("exhaustive"), row("exhaustive")])


def test_offline_snapshot_excludes_private_fields_and_replays_without_meta(tmp_path):
    loop, meta, out = (tmp_path / p for p in ("loop", "private", "out"))
    loop.mkdir()
    meta.mkdir()
    (loop / "loop_compare.jsonl").write_text(json.dumps(row("exhaustive")) + "\n", encoding="utf-8")
    for name, elapsed in (("h01_s2026", 100), ("baseline_p0c_s2026", 200)):
        data = {"num_envs": 4096, "max_iterations": 300, "train_exit_code": 0, "wall_time_s": elapsed,
                "run_dir": "PRIVATE_USER_PATH", "fault": "PRIVATE_FAULT"}
        (meta / f"{name}.meta.json").write_text(json.dumps(data), encoding="utf-8")
    proc = _offline("evals/diagnostic_cost.py", "--loop", loop, "--training-meta", meta, "--out", out)
    assert proc.returncode == 0, proc.stderr
    text = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert "PRIVATE_USER_PATH" not in text and "PRIVATE_FAULT" not in text
    report = json.loads((out / "diagnostic_cost.json").read_text(encoding="utf-8"))
    assert report["training"]["mean_wall_s"] == 150
    assert report["retraining_elimination_scenario"]["wall_s_per_case"] == 900
    assert report["retraining_elimination_scenario"]["measured_savings_ratio"] is None
    assert report["limitations"][0].startswith("1건")
    assert not any("18건" in text or "90건" in text for text in report["limitations"])
    contaminated = tmp_path / "contaminated_times.json"
    records = json.loads((out / "training_times.json").read_text(encoding="utf-8"))
    for record in records:
        record.update(run_dir="PRIVATE_USER_PATH", fault="PRIVATE_FAULT", strategy="PRIVATE_STRATEGY")
    contaminated.write_text(json.dumps(records), encoding="utf-8")
    replay = tmp_path / "replay"
    proc = _offline("evals/diagnostic_cost.py", "--loop", loop, "--training-times", contaminated, "--out", replay)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((replay / "diagnostic_cost.json").read_text(encoding="utf-8"))["table"] == report["table"]
    text = "".join(p.read_text(encoding="utf-8") for p in replay.iterdir())
    assert not any(secret in text for secret in ("PRIVATE_USER_PATH", "PRIVATE_FAULT", "PRIVATE_STRATEGY"))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    proc = _offline("evals/diagnostic_cost.py", "--loop", loop, "--training-meta", meta, "--out", out)
    assert proc.returncode == 2
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
