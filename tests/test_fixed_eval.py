"""P0-B2 고정 평가: 명령 격자·제자리 기준선과 판정 집계를 Isaac 없이 검사한다(평가 실행 자체는 GPU가 필요하다)."""
import importlib.util
import json
import math

import pytest

from test_offline_replay import ROOT, _offline


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "evals" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROTOCOL = json.loads((ROOT / "bench" / "protocols" / "fixed_eval_v1.json").read_text(encoding="utf-8"))


def test_grid_and_standstill_reference():
    FE = _module("fixed_eval")
    grid = FE.command_grid(PROTOCOL["command_grid"])
    assert len(grid) == 26 and (0.0, 0.0, 0.0) not in grid
    still = FE.standstill_rmse(grid)
    assert still["lin_vel_rmse_mps"] == pytest.approx(math.sqrt(36 / 26))
    assert still["yaw_rate_rmse_radps"] == pytest.approx(math.sqrt(18 / 26))
    assert PROTOCOL["horizon_steps"] * PROTOCOL["step_dt"] == PROTOCOL["episode_length_s"]


def _report(name, fall=0.0, surv=20.0, lin=1.17, yaw=0.82):
    return {"protocol": "fixed_eval_v1", "protocol_sha256_lf": "x", "horizon_steps": 1000, "step_dt": 0.02,
            "standstill": {"lin_vel_rmse_mps": 1.1767, "yaw_rate_rmse_radps": 0.8321}, "name": name,
            "checkpoint": {"file": "model_99.pt", "sha256": "0" * 64},
            "metrics": {"fall_rate": fall, "mean_survival_s": surv, "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw}}


def _write(folder, reports):
    d = folder / "runs"
    d.mkdir(parents=True)
    for r in reports:
        (d / f"{r['name']}.json").write_text(json.dumps(r), encoding="utf-8")


def test_summary_bands_labels_and_no_overwrite(tmp_path):
    folder = tmp_path / "fe"
    _write(folder, [_report("baseline_s7"), _report("baseline_s42", lin=1.18), _report("benign_all_s7", fall=0.02),
                    _report("baseline_repeat1_s42", lin=1.18), _report("c04_s7", fall=1.0, surv=0.2),
                    _report("c03_s42", lin=1.20), _report("baseline_s42__model_50", lin=9.9)])
    relabel = tmp_path / "relabel.json"
    relabel.write_text(json.dumps({"runs": [{"run": "c03_s42", "behavior_label": "healthy", "current_recovered": False},
                                            {"run": "c04_s7", "behavior_label": "unhealthy", "current_recovered": False}]}),
                       encoding="utf-8")
    proc = _offline("evals/fixed_eval_summary.py", folder, "--relabel", relabel)
    assert proc.returncode == 0, proc.stderr
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    by = {r["run"]: r for r in s["runs"]}
    assert "baseline_s42__model_50" not in by  # 체크포인트 접미사 결과는 실행 판정에 쓰지 않는다
    assert {n["pair"] for n in s["normal"]} == {"baseline_s7 vs baseline_s42", "baseline_s42 vs baseline_s7",
                                               "benign_all_s7 vs baseline_s7", "baseline_repeat1_s42 vs baseline_s42"}
    assert s["bands"]["fall_frac_max"] == pytest.approx(0.07)
    assert by["c04_s7"]["fixed_label"] == "unhealthy" and by["c03_s42"]["fixed_label"] == "healthy"
    assert s["changes_vs_p0b1"] == [] and s["environment"]["model_calls"] == 0
    proc = _offline("evals/fixed_eval_summary.py", folder)
    assert proc.returncode == 2 and "덮어쓰지 않는다" in proc.stderr


def test_summary_rejects_mixed_protocols(tmp_path):
    other = dict(_report("baseline_s42"), protocol_sha256_lf="y")
    _write(tmp_path / "fe", [_report("baseline_s7"), other])
    proc = _offline("evals/fixed_eval_summary.py", tmp_path / "fe")
    assert proc.returncode == 2 and "프로토콜" in proc.stderr
