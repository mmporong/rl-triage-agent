"""actor 출력층 재설정 단축 예산 결과를 공개 allowlist JSON으로 내보낸다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import actor_reset_budget_v1 as SOURCE  # noqa: E402
from rl_triage import grounded_recovery as RECOVERY_INPUT  # noqa: E402
from rl_triage import recovery as RECOVERY  # noqa: E402


RUN = ROOT / "bench/private/codex_support_20261015/task25/runs/actor_reset_budget_20261016"
OUTPUT = ROOT / "evals/results/actor_reset_budget_public_20261017"
AUDIT = ROOT / "bench/private/codex_support_20261015/task26/export/actor_reset_budget_public_20261017"
PROTOCOL = ROOT / "bench/protocols/actor_reset_budget_v1.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
EXPECTED_SHA256 = {
    "result": "237df85e495bec75ad4e741d2b8f83e80cbb6f65d44c648da280291a37ce0d10",
    "protocol": "1cf9c53a1623c6aaf58932bc6dbab0d7a6938f1acc27f78ea4d86c81d7544397",
    "fixed_eval": "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46",
}
SEEDS = (2090, 2091)
PUBLIC_ARMS = ("none", "f1", "reset100", "reset150", "reset200", "restart300")
RESET_BUDGETS = (100, 150, 200)
BANDS = {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.1513,
         "err_yaw_ratio_max": 1.1719, "fall_frac_max": 0.05}
PUBLIC_FILES = {
    "summary.json", "costs.json", "state_evidence.json", "provenance.json",
    *(f"evaluations/s{seed}/{arm}.json" for seed in SEEDS for arm in PUBLIC_ARMS),
}
FORBIDDEN = re.compile(
    r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support|task2[56]|actor_reset_budget_20261016",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_relative(seed: int, arm: str, kind: str) -> str:
    if arm in {"none", "f1"}:
        return f"source/s{seed}/{arm}/eval/report.json"
    source_arm = "restart_300" if arm == "restart300" else f"actor_last_layer_reset_{arm[5:]}"
    return f"arms/s{seed}/{source_arm}/eval/report.json" if kind == "report" \
        else f"arms/s{seed}/{source_arm}/train/{kind}.json"


def _source_path(seed: int, arm: str, kind: str = "report") -> Path:
    return RUN / _source_relative(seed, arm, kind)


def _require_source(path: Path, relative: str, evidence: dict[str, str]) -> str:
    expected = evidence.get(relative)
    actual = sha256(path) if path.is_file() else None
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"source artifact differs: {relative}")
    return expected


def _verdict(report: dict[str, Any], none: dict[str, Any], bands: dict[str, Any]) -> dict[str, Any]:
    return RECOVERY.verdict(
        RECOVERY.compare(RECOVERY_INPUT._behavior(report), RECOVERY_INPUT._behavior(none)), bands)


def _public_evaluation(seed: int, arm: str, report: dict[str, Any], label: str,
                       verdict: dict[str, Any], source_sha: str) -> dict[str, Any]:
    conditions = [{
        "command": row["command"], "num_envs": row["num_envs"], "fall_count": row["fall_count"],
        "alive_steps_sum": row["alive_steps_sum"], "fall_rate": row["fall_rate"],
        "lin_vel_rmse_mps": row["lin_vel_rmse_mps"],
        "yaw_rate_rmse_radps": row["yaw_rate_rmse_radps"],
    } for row in report["by_condition"]]
    if len(conditions) != 26:
        raise ValueError("fixed evaluation condition count differs")
    return {
        "schema": "public_actor_reset_budget_evaluation_v1", "seed": seed, "arm": arm,
        "label": label, "verdict": {"label": verdict["label"], "checks": verdict["checks"],
                                      "failed_checks": verdict["failed"],
                                      "short_horizon": verdict["short_horizon"]},
        "protocol": {"name": report["protocol"], "sha256_lf": report["protocol_sha256_lf"],
                     "task": report["task"], "eval_seed": report["eval_seed"],
                     "num_envs": report["num_envs"], "horizon_steps": report["horizon_steps"],
                     "step_dt": report["step_dt"], "action_scale": report["action_scale"]},
        "checkpoint": report["checkpoint"], "metrics": report["metrics"],
        "standstill": report["standstill"], "conditions": conditions,
        "source_artifact_sha256": source_sha,
    }


def _state_evidence(result: dict[str, Any], evidence: dict[str, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    equivalence_relative = "initial_equivalence.json"
    _require_source(RUN / equivalence_relative, equivalence_relative, evidence)
    equivalence = read(RUN / equivalence_relative)
    for seed in SEEDS:
        seed_rows: dict[str, Any] = {}
        for budget in RESET_BUDGETS:
            arm = f"reset{budget}"
            initial_relative = _source_relative(seed, arm, "initial")
            intervention_relative = _source_relative(seed, arm, "intervention")
            initial_sha = _require_source(RUN / initial_relative, initial_relative, evidence)
            intervention_sha = _require_source(RUN / intervention_relative, intervention_relative, evidence)
            initial, intervention = read(RUN / initial_relative), read(RUN / intervention_relative)
            seed_rows[str(budget)] = {
                "learning_iterations": budget,
                "initial": {"policy_state_sha256": initial["policy_state"]["sha256"],
                            "optimizer_state_sha256": initial["optimizer_state"]["sha256"],
                            "rng_state_before_learn": initial["rng_state_before_learn"]},
                "intervention": {
                    "name": intervention["intervention"], "reset_seed": intervention["reset_seed"],
                    "target_modules": intervention["target_modules"],
                    "target_parameter_count": intervention["target_parameter_count"],
                    "policy_state_before": intervention["policy_state_before"],
                    "policy_state_after": intervention["policy_state_after"],
                    "optimizer_state_before": intervention["optimizer_state_before"],
                    "optimizer_state_after": intervention["optimizer_state_after"],
                    "critic_preserved": intervention["critic_state_before"] == intervention["critic_state_after"],
                    "std_preserved": intervention["std_state_before"] == intervention["std_state_after"],
                    "optimizer_target_moments_removed": intervention["optimizer_moments_present_after"] is False,
                },
                "source_artifact_sha256": {"initial": initial_sha, "intervention": intervention_sha},
            }
        restart_initial_relative = _source_relative(seed, "restart300", "initial")
        restart_receipt_relative = _source_relative(seed, "restart300", "receipt")
        restart_initial_sha = _require_source(RUN / restart_initial_relative, restart_initial_relative, evidence)
        restart_receipt_sha = _require_source(RUN / restart_receipt_relative, restart_receipt_relative, evidence)
        restart_initial, restart_receipt = read(RUN / restart_initial_relative), read(RUN / restart_receipt_relative)
        rows[str(seed)] = {
            "reset_arms": seed_rows,
            "initial_equivalence": equivalence[str(seed)],
            "restart_300": {
                "learning_iterations": 300,
                "fresh_initialization": restart_receipt["resume_scope"]["fresh_initialization"],
                "checkpoint_load_calls": restart_receipt["checkpoint_load_calls"],
                "initial_policy_state_sha256": restart_initial["policy_state"]["sha256"],
                "initial_optimizer_state_sha256": restart_initial["optimizer_state"]["sha256"],
                "final_checkpoint_sha256": restart_receipt["checkpoint"]["sha256"],
                "matches_source_none_checkpoint_sha256": result["restart_matches_source_none_sha"][str(seed)],
                "source_artifact_sha256": {"initial": restart_initial_sha, "receipt": restart_receipt_sha},
            },
        }
    return {"schema": "public_actor_reset_budget_state_evidence_v1", "seeds": rows,
            "interpretation": "같은 seed의 단축 개입은 동일한 F1 checkpoint에서 시작했다. restart300은 checkpoint를 불러오지 않았다."}


def verify_and_build(isaaclab: Path) -> dict[str, dict[str, Any]]:
    for name, path in (("result", RUN / "result.json"), ("protocol", PROTOCOL), ("fixed_eval", FIXED_EVAL)):
        if not path.is_file() or sha256(path) != EXPECTED_SHA256[name]:
            raise ValueError(f"fixed source differs: {name}")
    verified = SOURCE.verify(isaaclab.resolve(), SOURCE.TAG)["result"]
    stored = read(RUN / "result.json")
    if verified != stored:
        raise ValueError("stored result differs from source verifier replay")
    expected_labels = {
        "100": {"2090": "not_recovered", "2091": "not_recovered"},
        "150": {"2090": "not_recovered", "2091": "not_recovered"},
        "200": {"2090": "not_recovered", "2091": "not_recovered"},
        "restart_300": {"2090": "recovered", "2091": "recovered"},
    }
    if (stored.get("status") != "complete" or stored.get("decision") != "keep_current_rule"
            or stored.get("candidate_budget_learning_iterations") is not None
            or stored.get("labels") != expected_labels or stored.get("provenance_valid") is not True
            or stored.get("automatic_global_change") is not False or stored.get("model_calls") != 0):
        raise ValueError("registered budget outcome differs")
    protocol = read(PROTOCOL)
    evidence = stored["evidence_sha256"]
    evaluations: dict[str, dict[str, Any]] = {}
    source_roles: dict[str, str] = {"registered_result": EXPECTED_SHA256["result"],
                                    "registered_protocol": EXPECTED_SHA256["protocol"],
                                    "fixed_evaluation_protocol": EXPECTED_SHA256["fixed_eval"]}
    for seed in SEEDS:
        reports = {arm: read(_source_path(seed, arm)) for arm in PUBLIC_ARMS}
        none = reports["none"]
        for arm, report in reports.items():
            relative = _source_relative(seed, arm, "report")
            source_sha = _require_source(RUN / relative, relative, evidence)
            verdict = _verdict(report, none, protocol["bands"])
            if arm == "none":
                label = "healthy"
            elif arm == "f1":
                label = "unhealthy"
            else:
                source_key = "restart_300" if arm == "restart300" else arm[5:]
                label = "healthy" if stored["labels"][source_key][str(seed)] == "recovered" else "unhealthy"
            if verdict["label"] != label:
                raise ValueError(f"public fixed-evaluation label differs: s{seed}/{arm}")
            public_name = f"evaluations/s{seed}/{arm}.json"
            evaluations[public_name] = _public_evaluation(seed, arm, report, label, verdict, source_sha)
            source_roles[f"fixed_evaluation.s{seed}.{arm}"] = source_sha

    costs = deepcopy(stored["costs"])
    costs_doc = {
        "schema": "public_actor_reset_budget_costs_v1",
        "measurement": "GPU 작업을 실행한 부모 프로세스의 시작부터 자연 종료와 소유 프로세스 정리 확인까지의 벽시계 시간",
        "not_gpu_kernel_time": True, "used_for_candidate_decision": False,
        "source_preparation": costs["source_preparation"], "recovery_arms": costs["recovery_arms"],
        "totals_wall_s": {"source_preparation": costs["source_preparation_total_wall_s"],
                          "recovery_experiment": costs["recovery_experiment_total_wall_s"],
                          "whole_experiment_accounted": costs["whole_experiment_accounted_wall_s"]},
        "training_wall_ratio_vs_same_seed_restart": stored["training_wall_ratio_vs_same_seed_restart"],
        "short_arms_were_faster_but_recovered": False,
        "whole_experiment_cheaper_claimed": False,
    }
    state = _state_evidence(stored, evidence)
    for seed in SEEDS:
        for budget in RESET_BUDGETS:
            for kind in ("initial", "intervention"):
                relative = _source_relative(seed, f"reset{budget}", kind)
                source_roles[f"state.s{seed}.reset{budget}.{kind}"] = evidence[relative]
        for kind in ("initial", "receipt"):
            relative = _source_relative(seed, "restart300", kind)
            source_roles[f"state.s{seed}.restart300.{kind}"] = evidence[relative]
    source_roles["initial_equivalence"] = evidence["initial_equivalence.json"]

    summary = {
        "schema": "public_actor_reset_budget_summary_v1", "status": "complete",
        "measurement_kind": "post_result_novel_seed_budget_confirmation",
        "seeds": list(SEEDS), "fault": "F1", "correction": "restore_body_frame_reward",
        "fixed_evaluation": {"conditions": 26, "labels": expected_labels},
        "outcome": {"decision": "keep_current_rule", "reset_100_recovered": "0/2",
                    "reset_150_recovered": "0/2", "reset_200_recovered": "0/2",
                    "restart_300_recovered": "2/2", "candidate_budget_learning_iterations": None,
                    "automatic_global_change": False},
        "prediction_match": stored["prediction_match"],
        "wording": stored["wording"],
        "interpretation": "100·150·조건부 200회 출력층 재설정은 두 seed 모두 미회복이었고 restart300은 두 seed 모두 회복했다. 기존 회복 규칙을 유지한다.",
        "comparison_boundary": "이번 seed에서 300회 출력층 재설정은 실행하지 않았다. 과거 seed의 300회 결과와 학습 예산의 인과 관계를 비교하지 않는다.",
        "cost_boundary": "단축 arm의 벽시계가 restart300보다 짧았지만 모두 미회복이었다. 단일 arm 시간과 전체 탐색 비용을 구분한다.",
        "claims": {"generalization": False, "causal_budget_comparison": False,
                   "gpu_savings": False, "model_superiority": False},
        "source_commit_sha": stored["source_commit_sha"], "protocol_sha256_lf": stored["protocol_sha256_lf"],
    }
    payloads: dict[str, dict[str, Any]] = {"summary.json": summary, "costs.json": costs_doc,
                                           "state_evidence.json": state, **evaluations}
    public_hashes = {name: hashlib.sha256(
        (json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest() for name, body in payloads.items()}
    payloads["provenance.json"] = {
        "schema": "public_actor_reset_budget_provenance_v1",
        "source_artifacts": dict(sorted(source_roles.items())),
        "public_artifact_sha256": dict(sorted(public_hashes.items())),
        "source_verification": "registered CPU replay passed before export",
        "private_paths_disclosed": False,
    }
    verify_public_payloads(payloads)
    return payloads


def verify_public_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(payloads) != PUBLIC_FILES:
        raise ValueError("public artifact inventory differs")
    encoded = json.dumps(payloads, ensure_ascii=False, allow_nan=False)
    if FORBIDDEN.search(encoded):
        raise ValueError("public artifact contains a private or absolute path")
    summary, costs, state, provenance = (payloads[name] for name in
                                         ("summary.json", "costs.json", "state_evidence.json", "provenance.json"))
    if (set(summary) != {"schema", "status", "measurement_kind", "seeds", "fault", "correction",
                         "fixed_evaluation", "outcome", "prediction_match", "wording", "interpretation",
                         "comparison_boundary", "cost_boundary", "claims", "source_commit_sha",
                         "protocol_sha256_lf"}
            or summary["status"] != "complete" or summary["seeds"] != list(SEEDS)
            or summary["outcome"]["decision"] != "keep_current_rule"
            or summary["outcome"]["restart_300_recovered"] != "2/2"
            or any(summary["outcome"][f"reset_{budget}_recovered"] != "0/2" for budget in RESET_BUDGETS)
            or any(summary["claims"].values())):
        raise ValueError("public summary differs")
    if (costs.get("schema") != "public_actor_reset_budget_costs_v1"
            or costs.get("not_gpu_kernel_time") is not True
            or costs.get("used_for_candidate_decision") is not False
            or costs.get("short_arms_were_faster_but_recovered") is not False
            or costs.get("whole_experiment_cheaper_claimed") is not False):
        raise ValueError("public cost scope differs")
    source_costs, recovery_costs = costs.get("source_preparation"), costs.get("recovery_arms")
    if (not isinstance(source_costs, dict) or not isinstance(recovery_costs, dict)
            or set(source_costs) != {"2090", "2091"}
            or any(not isinstance(rows, dict) or set(rows) != {"NONE", "F1"} for rows in source_costs.values())
            or set(recovery_costs) != {"100", "150", "200", "restart_300"}
            or any(not isinstance(rows, dict) or set(rows) != {"2090", "2091"}
                   for rows in recovery_costs.values())):
        raise ValueError("public cost arm/seed inventory differs")
    cost_rows = [row for seed in source_costs.values() for row in seed.values()]
    cost_rows += [row for arm in recovery_costs.values() for row in arm.values()]
    for row in cost_rows:
        if (set(row) != {"training_wall_s", "fixed_eval_wall_s", "total_wall_s"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in row.values())
                or not math.isclose(row["training_wall_s"] + row["fixed_eval_wall_s"],
                                    row["total_wall_s"], abs_tol=1e-9)):
            raise ValueError("public cost row differs")
    source_total = sum(row["total_wall_s"] for seed in source_costs.values() for row in seed.values())
    recovery_total = sum(row["total_wall_s"] for arm in recovery_costs.values() for row in arm.values())
    totals = costs["totals_wall_s"]
    if (not math.isclose(source_total, totals["source_preparation"], abs_tol=1e-9)
            or not math.isclose(recovery_total, totals["recovery_experiment"], abs_tol=1e-9)
            or not math.isclose(source_total + recovery_total, totals["whole_experiment_accounted"], abs_tol=1e-9)):
        raise ValueError("public cost totals differ")
    if state.get("schema") != "public_actor_reset_budget_state_evidence_v1" or set(state.get("seeds", {})) != {"2090", "2091"}:
        raise ValueError("public state evidence differs")
    recomputed_labels: dict[str, dict[str, str]] = {
        "100": {}, "150": {}, "200": {}, "restart_300": {}}
    for seed in SEEDS:
        none_row = payloads[f"evaluations/s{seed}/none.json"]
        none_behavior = {"survival_s": none_row["metrics"]["mean_survival_s"],
                         "horizon_s": none_row["protocol"]["horizon_steps"] * none_row["protocol"]["step_dt"],
                         "fall_frac": none_row["metrics"]["fall_rate"],
                         "err_xy": none_row["metrics"]["lin_vel_rmse_mps"],
                         "err_yaw": none_row["metrics"]["yaw_rate_rmse_radps"]}
        for arm in PUBLIC_ARMS:
            row = payloads[f"evaluations/s{seed}/{arm}.json"]
            if (set(row) != {"schema", "seed", "arm", "label", "verdict", "protocol", "checkpoint",
                             "metrics", "standstill", "conditions", "source_artifact_sha256"}
                    or row.get("schema") != "public_actor_reset_budget_evaluation_v1" or row.get("seed") != seed
                    or row.get("arm") != arm or len(row.get("conditions", [])) != 26
                    or row.get("label") not in {"healthy", "unhealthy"}):
                raise ValueError("public evaluation differs")
            behavior = {"survival_s": row["metrics"]["mean_survival_s"],
                        "horizon_s": row["protocol"]["horizon_steps"] * row["protocol"]["step_dt"],
                        "fall_frac": row["metrics"]["fall_rate"],
                        "err_xy": row["metrics"]["lin_vel_rmse_mps"],
                        "err_yaw": row["metrics"]["yaw_rate_rmse_radps"]}
            verdict = RECOVERY.verdict(RECOVERY.compare(behavior, none_behavior), BANDS)
            expected_public_verdict = {"label": verdict["label"], "checks": verdict["checks"],
                                       "failed_checks": verdict["failed"],
                                       "short_horizon": verdict["short_horizon"]}
            if row["verdict"] != expected_public_verdict or row["label"] != verdict["label"]:
                raise ValueError("public fixed-evaluation verdict replay differs")
            if arm == "f1" and verdict["label"] != "unhealthy":
                raise ValueError("public source fault gate differs")
            if arm == "none" and verdict["label"] != "healthy":
                raise ValueError("public source normal gate differs")
            if arm.startswith("reset"):
                recomputed_labels[arm[5:]][str(seed)] = (
                    "recovered" if verdict["label"] == "healthy" else "not_recovered")
            elif arm == "restart300":
                recomputed_labels["restart_300"][str(seed)] = (
                    "recovered" if verdict["label"] == "healthy" else "not_recovered")
        seed_state = state["seeds"][str(seed)]
        if set(seed_state.get("reset_arms", {})) != {"100", "150", "200"}:
            raise ValueError("public reset state budget set differs")
        signatures = []
        for budget in RESET_BUDGETS:
            reset = seed_state["reset_arms"][str(budget)]
            if (reset.get("learning_iterations") != budget
                    or reset.get("intervention", {}).get("name") != "actor_last_layer_reset"
                    or reset["intervention"].get("critic_preserved") is not True
                    or reset["intervention"].get("std_preserved") is not True
                    or reset["intervention"].get("optimizer_target_moments_removed") is not True):
                raise ValueError("public reset intervention state differs")
            signatures.append(reset["initial"])
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("public reset initial equivalence differs")
        equivalence = seed_state.get("initial_equivalence", {})
        if equivalence.get("status") != "identical" or equivalence.get("budgets") != [100, 150, 200] \
                or equivalence.get("signature") != signatures[0]:
            raise ValueError("public initial equivalence receipt differs")
        restart = seed_state.get("restart_300", {})
        if (restart.get("learning_iterations") != 300 or restart.get("fresh_initialization") is not True
                or restart.get("checkpoint_load_calls") != 0
                or restart.get("matches_source_none_checkpoint_sha256") is not True):
            raise ValueError("public restart state differs")
        none_checkpoint = payloads[f"evaluations/s{seed}/none.json"]["checkpoint"]["sha256"]
        restart_checkpoint = payloads[f"evaluations/s{seed}/restart300.json"]["checkpoint"]["sha256"]
        if restart.get("final_checkpoint_sha256") != none_checkpoint or restart_checkpoint != none_checkpoint:
            raise ValueError("public restart/source NONE checkpoint identity differs")
    if recomputed_labels != summary["fixed_evaluation"]["labels"]:
        raise ValueError("public summary labels differ from fixed evaluations")
    restart_ok = all(recomputed_labels["restart_300"][str(seed)] == "recovered" for seed in SEEDS)
    common = [budget for budget in RESET_BUDGETS
              if all(recomputed_labels[str(budget)][str(seed)] == "recovered" for seed in SEEDS)]
    decision = "conditional_candidate" if restart_ok and common else (
        "keep_current_rule" if restart_ok else "restart_control_not_recovered")
    expected_outcome = {"decision": decision,
                        **{f"reset_{budget}_recovered": f"{sum(recomputed_labels[str(budget)][str(seed)] == 'recovered' for seed in SEEDS)}/2"
                           for budget in RESET_BUDGETS},
                        "restart_300_recovered": f"{sum(recomputed_labels['restart_300'][str(seed)] == 'recovered' for seed in SEEDS)}/2",
                        "candidate_budget_learning_iterations": min(common) if decision == "conditional_candidate" else None,
                        "automatic_global_change": False}
    if summary["outcome"] != expected_outcome:
        raise ValueError("public decision replay differs")
    predictions = {"100": "not_recovered", "150": "recovered", "200": "recovered",
                   "restart_300": "recovered"}
    expected_prediction_match = {arm: {str(seed): recomputed_labels[arm][str(seed)] == prediction
                                       for seed in SEEDS} for arm, prediction in predictions.items()}
    if summary["prediction_match"] != expected_prediction_match:
        raise ValueError("public prediction match replay differs")
    expected_ratios = {str(seed): {str(budget):
        costs["recovery_arms"][str(budget)][str(seed)]["training_wall_s"]
        / costs["recovery_arms"]["restart_300"][str(seed)]["training_wall_s"]
        for budget in RESET_BUDGETS} for seed in SEEDS}
    if costs["training_wall_ratio_vs_same_seed_restart"] != expected_ratios:
        raise ValueError("public training wall ratio differs")
    if not all(expected_ratios[str(seed)][str(budget)] < 1 for seed in SEEDS for budget in RESET_BUDGETS):
        raise ValueError("public shorter-arm time relation differs")
    source_artifacts = provenance.get("source_artifacts")
    expected_source_roles = {"registered_result", "registered_protocol", "fixed_evaluation_protocol",
                             "initial_equivalence"}
    expected_source_roles |= {f"fixed_evaluation.s{seed}.{arm}"
                              for seed in SEEDS for arm in PUBLIC_ARMS}
    expected_source_roles |= {f"state.s{seed}.reset{budget}.{kind}"
                              for seed in SEEDS for budget in RESET_BUDGETS
                              for kind in ("initial", "intervention")}
    expected_source_roles |= {f"state.s{seed}.restart300.{kind}"
                              for seed in SEEDS for kind in ("initial", "receipt")}
    if (not isinstance(source_artifacts, dict) or set(source_artifacts) != expected_source_roles
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source_artifacts.values())
            or source_artifacts["registered_result"] != EXPECTED_SHA256["result"]
            or source_artifacts["registered_protocol"] != EXPECTED_SHA256["protocol"]
            or source_artifacts["fixed_evaluation_protocol"] != EXPECTED_SHA256["fixed_eval"]):
        raise ValueError("public source provenance inventory differs")
    for seed in SEEDS:
        for arm in PUBLIC_ARMS:
            if (payloads[f"evaluations/s{seed}/{arm}.json"]["source_artifact_sha256"]
                    != source_artifacts[f"fixed_evaluation.s{seed}.{arm}"]):
                raise ValueError("public evaluation source provenance differs")
        seed_state = state["seeds"][str(seed)]
        for budget in RESET_BUDGETS:
            artifact = seed_state["reset_arms"][str(budget)]["source_artifact_sha256"]
            if (not isinstance(artifact, dict)
                    or artifact.get("initial") != source_artifacts[f"state.s{seed}.reset{budget}.initial"]
                    or artifact.get("intervention") != source_artifacts[f"state.s{seed}.reset{budget}.intervention"]):
                raise ValueError("public reset state provenance differs")
        restart_artifact = seed_state["restart_300"]["source_artifact_sha256"]
        if (not isinstance(restart_artifact, dict)
                or restart_artifact.get("initial") != source_artifacts[f"state.s{seed}.restart300.initial"]
                or restart_artifact.get("receipt") != source_artifacts[f"state.s{seed}.restart300.receipt"]):
            raise ValueError("public restart state provenance differs")
    hashes = provenance.get("public_artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != PUBLIC_FILES - {"provenance.json"}:
        raise ValueError("public provenance inventory differs")
    for name, expected in hashes.items():
        body = json.dumps(payloads[name], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != expected:
            raise ValueError("public provenance hash differs")
    return {"status": "valid", "files": len(payloads), "evaluations": 12}


def export(isaaclab: Path) -> dict[str, Any]:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("public export output already exists")
    payloads = verify_and_build(isaaclab)
    for relative, body in payloads.items():
        write_new(OUTPUT / relative, body)
    verify_public(OUTPUT)
    receipt = {"schema": "actor_reset_budget_export_receipt_v1", "status": "complete",
               "public_files": {name: sha256(OUTPUT / name) for name in sorted(PUBLIC_FILES)}}
    write_new(AUDIT / "export_receipt.json", receipt)
    return {"status": "exported", "output": OUTPUT.relative_to(ROOT).as_posix(),
            "files": len(PUBLIC_FILES), "evaluations": 12}


def verify_public(folder: Path = OUTPUT) -> dict[str, Any]:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("public output folder is unavailable")
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != PUBLIC_FILES:
        raise ValueError("public output file inventory differs")
    payloads = {name: read(folder / name) for name in PUBLIC_FILES}
    return verify_public_payloads(payloads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = (verify_and_build(args.isaaclab) and {"status": "ready"}) if args.command == "check" else \
            export(args.isaaclab) if args.command == "export" else verify_public(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
