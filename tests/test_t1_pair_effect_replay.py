"""과거 숫자를 새 효과로 승격하지 않고 입력·판정 변조를 검출한다."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import replay_t1_pair_effects as R  # noqa: E402


def test_public_replay_retains_all_pairs_and_worsening_metric():
    result = R.replay()
    assert result["pair_count"] == 3 and result["input_count"] == 6
    assert result["original_unhealthy"] == 2 and result["original_failure_absent"] == 1
    assert result["rows"][1]["relative_error_reduction"]["lin_vel_rmse_mps"] < 0
    assert result["agent_diagnosis_or_recovery_proven"] is False


@pytest.mark.parametrize("tamper", ["label", "count", "raw_metric"])
def test_tampered_evidence_is_rejected(monkeypatch, tmp_path, tamper):
    shutil.copytree(R.SOURCE, tmp_path / "source")
    monkeypatch.setattr(R, "SOURCE", tmp_path / "source")
    path = R.SOURCE / "holdout_summary.json"
    value = json.loads(path.read_bytes())
    if tamper == "label":
        value["runs"][0]["label"] = "unhealthy"
    elif tamper == "count":
        value["unhealthy"] = 3
    else:
        path = R.SOURCE / "runs/e01_s2028.json"
        value = json.loads(path.read_bytes())
        value["metrics"]["fall_rate"] = 0.5
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|labels|counts"):
        R.replay()
