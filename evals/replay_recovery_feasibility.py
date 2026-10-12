"""공개된 세 행동 보고서를 재집계한다. 비공개 실행 원장을 인증하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage import grounded_recovery as C  # noqa: E402

SOURCE = ROOT / "evals/results/recovery_feasibility_20261012"
PROTOCOL = ROOT / "bench/protocols/recovery_feasibility_v3.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
# 이 공개 기록의 무결성 결속값이다. 비공개 실행의 인증이나 새 결과의 수용 기준이 아니다.
PUBLISHED_RESULT_SHA256_LF = "d276858014f131cffaddc3f5d804fb58c629db1f0acc0a562ecb1424163ca314"
PRESERVED_ATTEMPT_SHA256_LF = {
    "a": "7c97ac45fb88acf4ddb0308745c53d3b3544f1ef8a48aa9e8b6657f4a8179af0",
    "b": "d49fe1ce45b5ebe20b3ad9dfca473feda06c9d6d28cf7ec0f0c5aad2280fa293",
}
RESULT_KEYS = {
    "schema", "attempt", "scope", "source_commit_sha", "protocol_path", "protocol_sha256_lf",
    "checkpoint_sha256", "status", "summary", "reports", "relative_error_reduction",
    "result_wording", "private_execution_audit", "cost", "agent_entry_or_approval_evidence",
    "judgment_influence_proven", "claims", "next_action",
}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def error_reduction(reports: dict) -> dict:
    before = reports["fault_before"]["metrics"]
    corrected = reports["callback_fixed"]["metrics"]
    return {key: (before[key] - corrected[key]) / before[key]
            for key in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps")}


def replay(source: Path = SOURCE) -> dict:
    result_path = source / "result.json"
    result = json.loads(result_path.read_bytes())
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        raise ValueError("public result fields changed")
    if sha_lf(result_path) != PUBLISHED_RESULT_SHA256_LF:
        raise ValueError("published result hash changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    if (result["schema"] != "recovery_feasibility_public_result_v1"
            or result["protocol_path"] != PROTOCOL.relative_to(ROOT).as_posix()
            or result["protocol_sha256_lf"] != sha_lf(PROTOCOL)
            or result["checkpoint_sha256"] != protocol["checkpoint_sha256"]):
        raise ValueError("registered protocol or checkpoint binding changed")
    audit = result["private_execution_audit"]
    if (audit["public_replay_verifies_execution_provenance"] is not False
            or audit["full_raw_execution_artifacts_public"] is not False
            or audit["verified_at_source_commit"] != result["source_commit_sha"]):
        raise ValueError("private execution audit boundary changed")
    cost = result["cost"]
    for name, saved in zip(("a", "b"), cost["preserved_invalid_attempts"], strict=True):
        path = f"attempt_{name}.json"
        if sha_lf(source / path) != PRESERVED_ATTEMPT_SHA256_LF[name]:
            raise ValueError("preserved infrastructure attempt hash changed")
        original = json.loads((source / path).read_bytes())
        expected = {"path": path, **{key: original[key] for key in
                    ("status", "sdk_jobs_launched", "parent_wall_s")}}
        if saved != expected or original["status"] != "invalid" or original["model_calls"] != 0:
            raise ValueError("preserved infrastructure attempt differs from its public record")
    completed_wall = sum(audit["arms"][arm]["wall_s"] for arm in C.ARMS)
    failed = cost["preserved_invalid_attempts"]
    if (cost["model_calls"] != 0 or cost["completed_triplet_sdk_jobs"] != len(C.ARMS)
            or cost["completed_triplet_parent_wall_s"] != completed_wall
            or cost["all_attempts_sdk_jobs"] != len(C.ARMS) + sum(row["sdk_jobs_launched"] for row in failed)
            or cost["all_attempts_parent_wall_s"] != completed_wall + sum(row["parent_wall_s"] for row in failed)):
        raise ValueError("published execution cost totals differ from their records")
    if set(result["reports"]) != set(C.ARMS):
        raise ValueError("all three arms are required")
    reports = {}
    for arm in C.ARMS:
        entry = result["reports"][arm]
        if entry["path"] != f"runs/{arm}.json":
            raise ValueError("report path differs from the fixed portable layout")
        path = source / entry["path"]
        if sha_lf(path) != entry["sha256_lf"]:
            raise ValueError("public report hash changed")
        report = json.loads(path.read_bytes())
        reports[arm] = C.validate_fixed_report(report, {
            "name": arm, "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "action_scale": 0.25, "protocol_sha256_lf": sha_lf(FIXED_EVAL),
        })
    summary = C.summarize_triplet(reports)
    if (result["summary"] != summary or result["status"] != summary["status"]
            or result["relative_error_reduction"] != error_reduction(reports)
            or result["result_wording"] != protocol["result_wording"][summary["status"]]):
        raise ValueError("stored outcome differs from raw report reaggregation")
    if (result["agent_entry_or_approval_evidence"] is not False
            or result["judgment_influence_proven"] is not False
            or result["claims"] != C.CLAIMS):
        raise ValueError("numeric replay cannot support agent or causal claims")
    return {"status": "pass", "experiment_status": summary["status"],
            "validated_reports": len(reports), "conditions_per_report": len(C.GRID),
            "relative_error_reduction": error_reduction(reports),
            "summary": summary, "new_model_calls": 0, "new_gpu_runs": 0,
            "execution_provenance_verified_by_this_replay": False,
            "scope": "pinned_public_record_hashes_numeric_aggregation_cost_sums_and_registered_outcome_only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(json.dumps(replay(args.source), ensure_ascii=False, allow_nan=False))
