"""공개 재집계가 음성 결과와 실행 출처의 검증 경계를 보존하는지 확인한다."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import replay_recovery_feasibility as R  # noqa: E402


def test_public_negative_result_replays_without_execution_proof():
    result = R.replay()
    assert result["status"] == "pass"
    assert result["experiment_status"] == "no_failure_observed"
    assert result["validated_reports"] == 3
    assert result["summary"]["sham_equivalent"] is True
    assert result["summary"]["recovery_conditions_met"] is False
    assert result["execution_provenance_verified_by_this_replay"] is False


@pytest.mark.parametrize("tamper", ["status", "missing_arm", "raw_metric", "raw_metric_rehashed",
                                   "checkpoint_rehashed", "agent_claim", "source_commit", "scope",
                                   "attempt", "private_provenance", "private_verification", "full_raw",
                                   "cost", "unknown_claim", "invalid_attempt_cost",
                                   "attempt_unknown_claim", "attempt_behavioral_result", "attempt_source"])
def test_altered_public_result_is_rejected(tmp_path, tamper):
    source = tmp_path / "source"
    shutil.copytree(R.SOURCE, source)
    path = source / "result.json"
    value = json.loads(path.read_bytes())
    if tamper == "status":
        value["status"] = "recovery_conditions_met"
    elif tamper == "missing_arm":
        del value["reports"]["fault_sham"]
    elif tamper == "agent_claim":
        value["judgment_influence_proven"] = True
    elif tamper == "source_commit":
        value["source_commit_sha"] = "0" * 40
    elif tamper in ("scope", "attempt"):
        value[tamper] = "different_experiment"
    elif tamper == "private_provenance":
        value["private_execution_audit"]["public_replay_verifies_execution_provenance"] = True
    elif tamper == "private_verification":
        value["private_execution_audit"]["verification_passed"] = False
    elif tamper == "full_raw":
        value["private_execution_audit"]["full_raw_execution_artifacts_public"] = True
    elif tamper == "cost":
        value["cost"]["all_attempts_sdk_jobs"] = 999
    elif tamper == "unknown_claim":
        value["recovery_proven"] = True
    elif tamper in ("invalid_attempt_cost", "attempt_unknown_claim", "attempt_behavioral_result", "attempt_source"):
        failure_path = source / "attempt_a.json"
        failure = json.loads(failure_path.read_bytes())
        field, changed = {
            "invalid_attempt_cost": ("parent_wall_s", 999),
            "attempt_unknown_claim": ("recovery_proven", True),
            "attempt_behavioral_result": ("behavioral_result", {"status": "recovery_conditions_met"}),
            "attempt_source": ("source_commit_sha", "0" * 40),
        }[tamper]
        failure[field] = changed
        failure_path.write_text(json.dumps(failure), encoding="utf-8")
    else:
        report_path = source / "runs/fault_before.json"
        report = json.loads(report_path.read_bytes())
        if tamper == "checkpoint_rehashed":
            report["checkpoint"]["sha256"] = "0" * 64
        else:
            report["metrics"]["fall_rate"] = 0.5
        report_path.write_text(json.dumps(report), encoding="utf-8")
        if tamper.endswith("rehashed"):
            value["reports"]["fault_before"]["sha256_lf"] = R.sha_lf(report_path)
    if tamper not in ("raw_metric", "invalid_attempt_cost", "attempt_unknown_claim",
                      "attempt_behavioral_result", "attempt_source"):
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        R.replay(source)
