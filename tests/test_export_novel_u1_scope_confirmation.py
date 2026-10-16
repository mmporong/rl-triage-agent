from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals/results/novel_u1_scope_confirmation_s2062_20261016"
PRIVATE_RESULT = ROOT / "bench/private/codex_support_20261015/task11/u1/runs/novel_u1_scope_s2062_20261016/result.json"


def read(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def test_public_export_is_complete_and_keeps_seed_roles_separate():
    summary = read("summary.json")
    assert summary["status"] == "complete" and summary["outcome"] == "recovered"
    assert summary["measurement_kind"] == "post_result_development_confirmation"
    assert summary["source_training_seed"] == 2062
    assert summary["none_control_training_seed"] == 2062
    assert summary["corrected_u1_restart_training_seed"] == 3062
    assert summary["source_failure"] == {"outcome": "failure_present", "none_walks": True, "u1_label": "unhealthy"}
    assert summary["none_control"] == {"label": "healthy", "healthy": True, "walks": True}
    assert summary["corrected_u1_restart"] == {"label": "healthy", "recovery_label": "recovered"}


def test_public_evaluations_match_measured_metrics_and_all_gates():
    none = read("evaluations/none_family_control.json")
    corrected = read("evaluations/u1_family_restart.json")
    assert none["training"]["seed"] == 2062 and corrected["training"]["seed"] == 3062
    assert none["fixed_evaluation"]["metrics"] == {
        "fall_rate": 0.0, "mean_survival_s": 20.0,
        "lin_vel_rmse_mps": 0.13649091128467256, "yaw_rate_rmse_radps": 0.294454962870671,
    }
    assert corrected["fixed_evaluation"]["metrics"] == {
        "fall_rate": 0.0, "mean_survival_s": 20.0,
        "lin_vel_rmse_mps": 0.1372356520698349, "yaw_rate_rmse_radps": 0.2964537417502349,
    }
    for row in (none, corrected):
        assert row["label"] == "healthy"
        assert row["verdict"]["failed_checks"] == []
        assert all(row["verdict"]["checks"].values())
        assert len(row["fixed_evaluation"]["conditions"]) == 26


def test_source_failure_and_scope_measurements_are_preserved():
    source = read("source_failure_evidence.json")
    scope = read("scope_evidence.json")
    assert source["arms"]["NONE"]["metrics"]["lin_vel_rmse_mps"] == 0.13649091128467256
    assert source["arms"]["U1"]["metrics"]["lin_vel_rmse_mps"] == 1.2410838184345923
    assert source["gate"]["u1_failed_checks"] == ["err_xy_ratio", "err_yaw_ratio"]
    assert scope["scope_verdict"] == "scope_insufficient"
    assert scope["probe_outcome"] == "abnormal"
    assert len(scope["measured_probe_terms"]) == 2
    assert len(scope["source_registered_terms"]) == 4


def test_public_files_have_no_private_or_user_paths_and_use_allowlisted_shapes():
    forbidden = re.compile(r"[A-Za-z]:[\\/]|/home/|/Users/|/mnt/[a-z]/|bench/private", re.IGNORECASE)
    files = sorted(RESULTS.rglob("*.json"))
    assert len(files) == 7
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert forbidden.search(text) is None
    provenance = read("provenance.json")
    assert provenance["export_contract"] == "allowlisted_fields_only_no_private_or_user_paths"
    assert set(provenance["source_artifact_sha256"]) and len(provenance["public_training_telemetry"]) == 2


def test_telemetry_is_safe_and_bound_to_source_run_names():
    provenance = read("provenance.json")
    expected = {
        "source_none": ("bench/runs_p0c/baseline_p0c_s2062.telemetry.json", "2026-10-16_10-52-03_baseline_p0c_s2062"),
        "source_u1": ("bench/runs_p0c/u1_s2062.telemetry.json", "2026-10-16_11-02-23_u1_s2062"),
    }
    for row in provenance["public_training_telemetry"]:
        path, run_name = expected[row["logical_id"]]
        assert row["path"] == path and row["metric_count"] == 24 and row["points_per_metric"] == 300
        telemetry = json.loads((ROOT / path).read_text(encoding="utf-8"))
        assert telemetry["run_dir_name"] == run_name
        assert set(telemetry) == {"run_dir_name", "summary", "series"}


@pytest.mark.skipif(not PRIVATE_RESULT.is_file(), reason="private source result is unavailable")
def test_cpu_export_verifier_passes_with_original_private_artifacts():
    python = Path.home() / "IsaacLab/_isaac_sim/python.bat"
    if not python.is_file():
        pytest.skip("Isaac Lab Python is unavailable")
    completed = subprocess.run(
        [str(python), "evals/export_novel_u1_scope_confirmation.py", "check", "--isaaclab", str(Path.home() / "IsaacLab")],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=240,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["outcome"] == "recovered"
