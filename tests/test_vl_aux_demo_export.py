from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys

import pytest

from evals import export_vl_aux_demo_v1 as EXPORT


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals/results/vl_aux_demo_public_s2080_20261016"
PRIVATE_RESULT = ROOT / "bench/private/codex_support_20261015/task23/runs/vl_aux_f1_s2080_20261016/result.json"


def read(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_summary_keeps_base_decision_and_auxiliary_observation_separate():
    summary = read("summary.json")
    assert summary["outcome"] == "recovered" and summary["demo_success"] is True
    assert summary["base_success_conditions"] == {"passed": 12, "total": 12}
    assert summary["auxiliary"] == {
        "complete": True, "valid_observations": 3, "expected_observations": 3,
        "fixed_label_correspondence": {"matched": 3, "total": 3}, "used_for_decision": False,
    }
    assert summary["api"]["requests"] == {"decision": 2, "observer": 3, "total": 5, "cap": 16}
    assert summary["api"]["tokens"] == {"input": 9749, "output": 429, "total": 10178}
    observations = read("observations.json")["observations"]
    assert sum(row["fixed_label_correspondence"] for row in observations) == summary["auxiliary"]["fixed_label_correspondence"]["matched"]
    assert sum(row["requests"] for row in observations) == summary["api"]["requests"]["observer"]
    assert len(read("provider_identity.json")["calls"]) == summary["api"]["requests"]["total"]


def test_public_observations_preserve_three_measured_decisions_and_frame_bindings():
    rows = read("observations.json")["observations"]
    assert [(row["arm"], row["fixed_evaluation_label"], row["decision"])
            for row in rows] == [
        ("none", "healthy", {"walking": "yes", "in_place": "no", "falling": "no"}),
        ("fault", "unhealthy", {"walking": "no", "in_place": "yes", "falling": "no"}),
        ("corrected", "healthy", {"walking": "yes", "in_place": "no", "falling": "no"}),
    ]
    assert all(row["used_for_decision"] is False and row["fixed_label_correspondence"] is True for row in rows)
    for row in rows:
        assert len(row["frames"]) == 6
        for frame in row["frames"]:
            path = RESULTS / frame["path"]
            assert path.is_file() and path.stat().st_size == frame["bytes"]
            assert sha256(path) == frame["sha256"]


def test_public_provider_identity_preserves_registered_raw_alias_without_weights_claim():
    identity = read("provider_identity.json")
    assert identity["identity_policy"] == "registered endpoint alias; model weights were not attested"
    assert identity["timing_scope"]["timing_comparison_allowed"] is False
    assert "not a comparable latency" in identity["timing_scope"]["combined"]
    observer = identity["calls"][2:]
    assert len(observer) == 3
    assert all(row["requested_model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b" for row in observer)
    assert all(row["returned_provider_model_raw"] == "private/nvidia/nemotron-3.5-super-120b-a12b" for row in observer)
    assert all(row["registered_alias"] is True for row in observer)


def test_public_fixed_evaluations_preserve_measured_rmse_and_labels():
    none, fault, corrected = (read(f"evaluations/{name}.json") for name in ("none", "fault", "corrected"))
    assert (none["label"], none["metrics"]["lin_vel_rmse_mps"]) == ("healthy", 0.17660087657880483)
    assert (fault["label"], fault["metrics"]["lin_vel_rmse_mps"]) == ("unhealthy", 1.1788660688485655)
    assert (corrected["label"], corrected["metrics"]["lin_vel_rmse_mps"]) == ("healthy", 0.15074493114093)
    assert all(len(row["conditions"]) == 26 for row in (none, fault, corrected))


def test_public_inventory_has_no_private_or_user_paths_and_contains_exactly_18_jpegs():
    forbidden = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support", re.IGNORECASE)
    files = sorted(path.relative_to(RESULTS).as_posix() for path in RESULTS.rglob("*") if path.is_file())
    assert len(files) == 25
    assert len([name for name in files if name.endswith(".jpg")]) == 18
    assert len([name for name in files if name.endswith(".json")]) == 7
    for path in RESULTS.rglob("*.json"):
        assert forbidden.search(path.read_text(encoding="utf-8")) is None
    provenance = read("provenance.json")
    assert provenance["source_commit_sha"] == "110720c2b0188d48e39bbc8cdbc90a3d88fcc6bb"
    assert provenance["protocol_sha256_lf"] == "278a8c8310079c9227de724b3504f07e588d97b9014222c0521cb5a42bb05c7d"
    assert len(provenance["step_graph"]) == 11
    assert provenance["model_record_validation"] == {
        "diagnosis": "valid", "correction": "valid", "observer_records": 3,
        "observer_ledger_entries": 3,
    }


def test_export_refuses_existing_output_and_public_validator_rejects_sensitive_paths(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(EXPORT, "OUTPUT", output)
    monkeypatch.setattr(EXPORT, "AUDIT", tmp_path / "audit")
    with pytest.raises(FileExistsError):
        EXPORT.write_new({}, {})
    payloads = {relative: read(relative) for relative in EXPORT.JSON_FILES}
    frames = {relative: RESULTS / relative for relative in EXPORT.FRAME_FILES}
    payloads["summary.json"] = {**payloads["summary.json"], "leak": r"bench\private\redacted.json"}
    with pytest.raises(ValueError, match="private or user path"):
        EXPORT.verify_public_payloads(payloads, frames)


@pytest.mark.skipif(not PRIVATE_RESULT.is_file(), reason="private source result is unavailable")
def test_private_source_replay_and_public_export_verifier_pass():
    completed = subprocess.run(
        [sys.executable, "evals/export_vl_aux_demo_v1.py", "verify"], cwd=ROOT,
        capture_output=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "pass"
