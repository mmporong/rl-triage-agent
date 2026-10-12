"""공개 T1 세 짝의 기존 평가를 재집계한다. 새 성능 실험이 아니다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.grounded_recovery import validate_fixed_report  # noqa: E402
from t1_holdout_summary import summarize  # noqa: E402

SOURCE = ROOT / "evals/results/t1_fixed_eval_20261009"


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def replay() -> dict:
    summary = json.loads((SOURCE / "holdout_summary.json").read_bytes())
    hashes = {row["name"]: row["sha256_lf"] for row in summary["inputs"]}
    if len(hashes) != 6 or [row["seed"] for row in summary["runs"]] != [2026, 2027, 2028]:
        raise ValueError("the complete historical three-pair sample is required")
    rows = []
    for pair in summary["runs"]:
        reports = {}
        for field in ("run", "reference"):
            name = pair[field]
            path = SOURCE / "runs" / f"{name}.json"
            if sha_lf(path) != hashes[name]:
                raise ValueError("historical input hash changed")
            report = json.loads(path.read_bytes())
            expected = {"name": name, "checkpoint_file": "model_299.pt",
                        "checkpoint_sha256": report["checkpoint"]["sha256"], "action_scale": 0.25,
                        "protocol_sha256_lf": sha_lf(ROOT / "bench/protocols/fixed_eval_v1.json")}
            checked = validate_fixed_report(report, expected)
            stored_metrics = pair["metrics" if field == "run" else "reference_metrics"]
            if checked["metrics"] != stored_metrics:
                raise ValueError("summary differs from the validated raw report")
            reports[field] = checked
        before, corrected = (reports[field]["metrics"] for field in ("run", "reference"))
        rows.append({"training_seed": pair["seed"], "original_label": pair["label"],
                     "original_failed_metrics": pair["failed"],
                     "fault_trained_policy": before, "reference_trained_policy": corrected,
                     "relative_error_reduction": {
                         metric: (before[metric] - corrected[metric]) / before[metric]
                         for metric in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps")},
                     "checkpoint_sha256": {field: report["checkpoint"]["sha256"]
                                            for field, report in reports.items()}})
    key_path = ROOT / "bench/answer_key_t1.json"
    if sha_lf(key_path) != summary["answer_key_sha256_lf"]:
        raise ValueError("historical pairing key changed")
    recomputed = summarize(SOURCE, json.loads(key_path.read_bytes()),
                           sha_lf(ROOT / "bench/protocols/fixed_eval_v1.json"))
    if any(summary[key] != value for key, value in recomputed.items()):
        raise ValueError("historical labels or counts differ from raw report reaggregation")
    return {"schema": "t1_pair_effect_replay_v1", "scope": "retrospective_existing_three_training_pairs",
            "new_model_calls": 0, "new_gpu_runs": 0, "input_count": 6, "pair_count": 3,
            "original_unhealthy": summary["unhealthy"], "original_failure_absent": summary["failure_absent"],
            "rows": rows, "input_sha256_lf": hashes,
            "source_summary_sha256_lf": sha_lf(SOURCE / "holdout_summary.json"),
            "agent_diagnosis_or_recovery_proven": False, "statistical_superiority_proven": False,
            "interpretation": "These are different policies trained under fault and reference callbacks, then evaluated in the same stock environment. They are not the same-policy runtime correction experiment, new training, an agent-selected repair, or new held-out evidence. All three existing pairs, including the failure-absent pair and worsened metrics, are retained."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "replayed", "pairs": result["pair_count"], "output": str(args.output)}))
