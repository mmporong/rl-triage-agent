# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import export_evidence_guided_recovery_v1 as X


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_missing_complete_result_creates_no_public_output(tmp_path: Path) -> None:
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="complete recovery result"):
        X.export(tmp_path / "protocol.json", "a" * 40, tmp_path / "candidate.json",
                 tmp_path / "approval.json", tmp_path / "source.json",
                 tmp_path / "study.json", tmp_path / "missing-result.json", out=out)
    assert not out.exists()


def test_fixed26_metrics_require_all_four_and_26_conditions() -> None:
    report = {"metrics": {"lin_vel_rmse_mps": 1.0, "yaw_rate_rmse_radps": 2.0,
                          "mean_survival_s": 20.0, "fall_rate": 0.0},
              "by_condition": [{} for _ in range(26)]}
    assert X._metrics(report)["mean_survival_s"] == 20.0
    report["by_condition"].pop()
    with pytest.raises(ValueError, match="fixed26"):
        X._metrics(report)


def test_approval_must_bind_exact_candidate(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_path = write(tmp_path / "candidate.json", {"eligible": True})
    approval_path = write(tmp_path / "approval.json", {
        "schema": X.R.APPROVAL_SCHEMA, "approval_granted": True,
        "candidate": X.R.ref(candidate_path), "correction_id": X.R.CORRECTION,
        "recovery_seed": X.R.RECOVERY_SEED, "actor": "operator"})
    monkeypatch.setattr(X.R, "verify_candidate", lambda path: {"eligible": True})
    assert X._approval(candidate_path, approval_path)[1]["actor"] == "operator"
    value = json.loads(approval_path.read_text())
    value["candidate"]["sha256"] = "0" * 64
    write(approval_path, value)
    with pytest.raises(ValueError, match="candidate/approval"):
        X._approval(candidate_path, approval_path)


def test_owned_execution_rejects_unclean_child() -> None:
    clean = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
             "timed_out": False}
    assert X._owned(clean, "training") == clean
    with pytest.raises(ValueError, match="owned cleanup"):
        X._owned({**clean, "cleanup_verified": False}, "training")


def test_script_help_bootstraps_repository_package() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(X.ROOT / "evals/export_evidence_guided_recovery_v1.py"),
         "--help"], cwd=X.ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False)
    assert result.returncode == 0, result.stderr
    assert "--candidate" in result.stdout and "--result" in result.stdout
