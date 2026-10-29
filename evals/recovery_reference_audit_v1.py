"""공개 seed 2028의 회복 기준 민감도를 재계산한다. 새 실험이나 라벨 변경은 하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_triage.recovery_diagnostics import diagnose

INPUTS = {
    "reset": "evals/results/reset_interventions_v2_20261015/seeds/seed_2028.json",
    "reward": "evals/results/reward_v4_s2028_20261013/result.json",
    "resume_report": "evals/results/reward_v4_s2028_20261013/reports/clean_resume_reference.json",
    "initial_report": "evals/results/reward_v4_s2028_20261013/original_reports/clean_reference.json",
    "costs": "evals/results/selection_comparison_s2171_20261023/costs.json",
    "policies": "evals/results/selection_comparison_s2171_20261023/policies.json",
}


def build_report(repo: Path = ROOT) -> dict:
    sources, inputs = {}, {}
    for key, name in INPUTS.items():
        raw = (repo / name).read_bytes()
        inputs[key] = json.loads(raw)
        sources[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    original = inputs["reward"]["numeric"]
    actor = inputs["reset"]["arms"]["actor_last_layer_reset"]
    metrics = actor["fixed_evaluation"]
    run = {"survival_s": metrics["mean_survival_s"], "horizon_s": 20.0,
           "fall_frac": metrics["fall_rate"], "err_xy": metrics["lin_vel_rmse_mps"],
           "err_yaw": metrics["yaw_rate_rmse_radps"]}
    references = {
        "registered_clean_resume_598": original["behavior"]["clean_resume_reference"],
        "sensitivity_original_none_299": original["original_behavior"]["clean_reference"],
    }
    audits = {key: diagnose(run, reference, original["bands"])
              for key, reference in references.items()}
    reference_checks = {}
    for key, report in (("registered_clean_resume_598", inputs["resume_report"]),
                        ("sensitivity_original_none_299", inputs["initial_report"])):
        value = references[key]
        reference_checks[key] = diagnose(value, value, original["bands"], conditions=report["by_condition"])
    costs = inputs["costs"]["costs"]["arms"]
    measured = {}
    for arm, value in costs.items():
        policy = inputs["policies"]["policies"][arm]
        probe_s, repair_s = sum(value["probes"]), sum(value["repairs"])
        measured[arm] = {"status": policy["status"], "probe_count": len(value["probes"]),
                         "probe_wall_s": probe_s, "repair_wall_s": repair_s,
                         "classified_wall_s": probe_s + repair_s,
                         "recovered": policy["status"] == "recovered"}
    return {
        "schema": "public_recovery_reference_audit_v1",
        "scope": "posthoc_seen_results_numeric_sensitivity_only",
        "new_training_runs": 0, "new_model_calls": 0, "used_for_decision": False,
        "existing_results_modified": False, "sources": sources,
        "actor_reset_seed": 2028, "saved_actor_reset_label": actor["label"],
        "actor_reset_behavior": run, "reference_behaviors": references,
        "bands_unchanged": original["bands"], "reference_sensitivity": audits,
        "reference_condition_checks": reference_checks,
        "selection_comparison_seed": 2171, "measured_arm_costs": measured,
        "cost_scope": "classified_probe_and_repair_wall_time_excludes_api_wait_and_shared_source",
        "wording": {
            "reference_sensitivity": "같은 actor 관측도 기준 598에서는 unhealthy, 기준 299에서는 healthy다. 이미 본 결과의 사후 민감도이며 회복 증거를 추가하지 않는다.",
            "label_policy": "기존 seed 2028의 not_recovered 라벨을 유지한다. 기준 변경은 향후 새 사전등록에서만 다룬다.",
            "reference_warning": "집계 healthy와 특정 조건 전원 낙상은 함께 성립할 수 있으므로 조건별 경고를 별도로 기록한다.",
            "cost": "13.6%와 44%는 실행하지 않은 대안의 추정치이며 관측된 절감량이 아니다. 공개 시간은 GPU 커널 시간이 아닌 작업 벽시계다.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "saved_label": report["saved_actor_reset_label"]}))


if __name__ == "__main__":
    main()
