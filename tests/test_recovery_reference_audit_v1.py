"""공개 원문의 SHA 결합과 사후 민감도 계산을 검사한다."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_reference_audit_v1", ROOT / "evals/recovery_reference_audit_v1.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_audit_binds_original_bytes_and_keeps_sensitivity_separate():
    report = M.build_report(ROOT)
    assert report["saved_actor_reset_label"] == "not_recovered"
    labels = {key: value["registered_verdict"]["label"]
              for key, value in report["reference_sensitivity"].items()}
    assert labels == {"registered_clean_resume_598": "unhealthy", "sensitivity_original_none_299": "healthy"}
    warnings = report["reference_condition_checks"]["registered_clean_resume_598"]["condition_warnings"]
    assert warnings["conditions_checked"] == 26
    assert len(warnings["all_fell"]) == 1
    assert warnings["all_fell"][0]["fall_count"] == warnings["all_fell"][0]["num_envs"] == 40
    costs = report["measured_arm_costs"]
    assert costs["model_adaptive"]["classified_wall_s"] > costs["fixed_eligible"]["classified_wall_s"]
    assert costs["model_first_then_rule"]["recovered"] is False
    for name, ref in report["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == ref["sha256"]
