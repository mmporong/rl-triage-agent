"""ANYmal-C F6 smoke 결과를 공개 allowlist JSON으로 내보낸다."""
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

import anymal_termination_smoke_v1 as SOURCE  # noqa: E402
from rl_triage import recovery as RECOVERY  # noqa: E402


RUN = ROOT / "bench/private/codex_support_20261015/task27/runs/anymal_f6_smoke_s2100_20261017"
OUTPUT = ROOT / "evals/results/anymal_f6_smoke_public_s2100_20261017"
AUDIT = ROOT / "bench/private/codex_support_20261015/task28/export/anymal_f6_smoke_public_s2100_20261017"
PROTOCOL = ROOT / "bench/protocols/anymal_termination_smoke_v1.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1_anymal_c.json"
EXPECTED_SHA256 = {
    "result": "11808e18b73855fff9fe2d2e27c128d7e9b41d67c53f3fed7ba23ad989676611",
    "protocol": "66cf669f708e30fee2c3515b6aad975c452572c5177fd9c993bf493f5118e412",
    "fixed_eval": "d5dbc5c92bb6e14c47c73bd70a6df52ece3de4bc3557558c35534f878cffcbd7",
}
EXPECTED_SOURCE_ROLES = {
    "registered_result": EXPECTED_SHA256["result"],
    "registered_protocol": EXPECTED_SHA256["protocol"],
    "fixed_evaluation_protocol": EXPECTED_SHA256["fixed_eval"],
    "cpu_semantics": "1ba3e5b31e8e620da854fd1d209e0102eb1dc598f934d4ad99da833f64263e4f",
    "telemetry.none": "52ff7fa0d5fc21faa82635b60fabab2b6cc9da10a7ee4b7e167a2f2b31621e8c",
    "telemetry.f6": "cc95fa800024dd937f7ff9998c1eea4248ecadc296769060e7fed3fbae9a473b",
    "fixed_evaluation.none": "b0ebc0777803a66f4130f7c35f820d043a1492cfc54240285c07e30b48f74826",
    "fixed_evaluation.f6": "ac56053593babdb2847a71429695b3472e6eba93f850e7af2c49b2bbc6d2e67b",
}
ARMS = ("NONE", "F6")
SEED = 2100
BANDS = {
    "survival_ratio_min": 0.8809606037764762,
    "err_xy_ratio_max": 1.1767207428933193,
    "err_yaw_ratio_max": 1.248528339710145,
    "fall_frac_max": 0.0826923094689846,
}
PUBLIC_FILES = {
    "summary.json", "runtime_signature.json", "costs.json", "provenance.json",
    "evaluations/none.json", "evaluations/f6.json",
}
# 고정된 source replay로 생성한 공개본의 내용 결합. payload와 그 자체의 manifest를
# 함께 바꾸는 경우도 거부한다. 다른 실행은 새 exporter/결과 경로로 등록한다.
EXPECTED_PUBLIC_CANONICAL_SHA256 = {
    "costs.json": "bb0359824e46816e6380aa475be6defc3331e68eb469407654ddc5cfefd0a9c5",
    "evaluations/f6.json": "e1ee91bc1f632c76c1a0657599c89a7b4667cd4ed8e0e16f89d4d7bf95a54e1c",
    "evaluations/none.json": "cf76209dc60f714db35274633c6f1fa218e68fb3ea99cec8d37c08a5245a715b",
    "provenance.json": "48551bdb368786756e173208b28d11e93da5ce1da4c4e0654c730e3a0bdbffdd",
    "runtime_signature.json": "df784f68151b67f163278270900e4b6659b24a8cac1c8cc5b8f4b83d0fb669b0",
    "summary.json": "dcabc8bbf11a9f02bd61370babd6171976675115f3f7d0d83347d3d18f31e9f4",
}
FORBIDDEN = re.compile(
    r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support|task27|anymal_f6_smoke_s2100_20261017",
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


def _encoded(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _snapshot(folder: Path) -> dict[str, Any]:
    files = {path.relative_to(folder).as_posix(): sha256(path)
             for path in sorted(folder.rglob("*")) if path.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(files), "sha256": digest, "files": files}


def _source_hash(result: dict[str, Any], relative: str) -> str:
    expected = result["evidence_sha256"].get(relative)
    path = RUN / relative
    actual = sha256(path) if path.is_file() else None
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"source artifact differs: {relative}")
    return expected


def _public_evaluation(arm: str, report: dict[str, Any], source_sha: str) -> dict[str, Any]:
    conditions = [{
        "command": row["command"], "num_envs": row["num_envs"],
        "fall_count": row["fall_count"], "alive_steps_sum": row["alive_steps_sum"],
        "lin_error_sq_sum": row["lin_error_sq_sum"], "yaw_error_sq_sum": row["yaw_error_sq_sum"],
        "fall_rate": row["fall_rate"], "lin_vel_rmse_mps": row["lin_vel_rmse_mps"],
        "yaw_rate_rmse_radps": row["yaw_rate_rmse_radps"],
    } for row in report["by_condition"]]
    if len(conditions) != 26:
        raise ValueError("fixed evaluation condition count differs")
    return {
        "schema": "public_anymal_termination_smoke_evaluation_v1",
        "seed": SEED,
        "arm": arm,
        "protocol": {
            "name": report["protocol"], "sha256_lf": report["protocol_sha256_lf"],
            "task": report["task"], "eval_seed": report["eval_seed"],
            "num_envs": report["num_envs"], "horizon_steps": report["horizon_steps"],
            "step_dt": report["step_dt"], "action_scale": report["action_scale"],
        },
        "checkpoint": report["checkpoint"],
        "metrics": report["metrics"],
        "standstill": report["standstill"],
        "conditions": conditions,
        "source_artifact_sha256": source_sha,
    }


def verify_and_build(isaaclab: Path) -> dict[str, dict[str, Any]]:
    for name, path in (("result", RUN / "result.json"), ("protocol", PROTOCOL), ("fixed_eval", FIXED_EVAL)):
        if not path.is_file() or sha256(path) != EXPECTED_SHA256[name]:
            raise ValueError(f"fixed source differs: {name}")
    if SOURCE.verify_result(isaaclab.resolve(), SOURCE.TAG) != {
            "status": "valid", "outcome": "source_failure_absent", "may_preregister_demo": False}:
        raise ValueError("registered source replay differs")
    result = read(RUN / "result.json")
    expected = {
        "status": "complete", "seed": SEED, "outcome": "source_failure_absent",
        "prediction_match": False, "model_calls": 0, "may_preregister_demo": False,
        "demo_executed": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("registered smoke outcome differs")

    reports = {arm: read(RUN / f"evaluate/{arm}.json") for arm in ARMS}
    source_roles = {
        "registered_result": EXPECTED_SHA256["result"],
        "registered_protocol": EXPECTED_SHA256["protocol"],
        "fixed_evaluation_protocol": EXPECTED_SHA256["fixed_eval"],
        "cpu_semantics": _source_hash(result, "cpu/report.json"),
        "telemetry.none": _source_hash(result, "NONE/telemetry.json"),
        "telemetry.f6": _source_hash(result, "F6/telemetry.json"),
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        relative = f"evaluate/{arm}.json"
        report_sha = _source_hash(result, relative)
        name = f"evaluations/{arm.lower()}.json"
        evaluations[name] = _public_evaluation(arm, reports[arm], report_sha)
        source_roles[f"fixed_evaluation.{arm.lower()}"] = report_sha
    if source_roles != EXPECTED_SOURCE_ROLES:
        raise ValueError("registered source role hashes differ")

    signature = deepcopy(result["runtime_injection_signature"])
    runtime = {
        "schema": "public_anymal_termination_smoke_runtime_v1",
        "fault": "F6",
        "configured_episode_length_s": 20.0,
        "fault_timeout_s": 1.0,
        **signature,
        "interpretation": "F6 학습의 에피소드 길이는 등록한 50 step 경계에 도달했고 timeout 항도 관측됐다.",
        "behavior_decision_source": "clean fixed evaluation",
    }
    costs = deepcopy(result["costs"])
    training_total = sum(costs["training"][arm]["training_wall_s"] for arm in ARMS)
    extract_total = sum(costs["training"][arm]["extract_wall_s"] for arm in ARMS)
    costs_doc = {
        "schema": "public_anymal_termination_smoke_costs_v1",
        "measurement": "부모 실행 시작부터 자연 종료와 소유 프로세스 정리 확인까지의 벽시계 시간",
        "not_gpu_kernel_time": True,
        "training": costs["training"],
        "fixed_eval_wall_s": costs["fixed_eval_wall_s"],
        "totals_wall_s": {
            "training": training_total,
            "telemetry_extract": extract_total,
            "training_and_fixed_evaluation": training_total + costs["fixed_eval_wall_s"],
            "accounted_with_extract": training_total + extract_total + costs["fixed_eval_wall_s"],
        },
    }
    summary = {
        "schema": "public_anymal_termination_smoke_summary_v1",
        "status": "complete",
        "measurement_kind": "post_f2_applicability_development_smoke",
        "seed": SEED,
        "robot": "ANYmal-C",
        "fault": "F6 termination timeout",
        "prediction": "failure_present",
        "prediction_match": False,
        "outcome": "source_failure_absent",
        "may_preregister_demo": False,
        "demo_executed": False,
        "model_calls": 0,
        "fixed_evaluation": {"conditions": 26, "bands": BANDS, "none_walks": True,
                             "fault_verdict": "healthy"},
        "wording": result["wording"],
        "interpretation": "F6 조기 종료는 학습 로그에서 확인됐지만 새 seed 정책은 깨끗한 고정 평가에서 정상 범위를 벗어나지 않았다. 등록 규칙에 따라 종단 시연을 중단했다.",
        "evidence_boundary": "새 seed 하나의 개발 smoke다. F6가 다른 조건에서도 실패를 만들지 않는다는 결론은 내리지 않는다.",
        "claims": {"generalization": False, "gpu_savings": False, "model_superiority": False},
        "source_commit_sha": result["source_commit_sha"],
        "protocol_sha256_lf": result["protocol_sha256_lf"],
    }
    payloads: dict[str, dict[str, Any]] = {
        "summary.json": summary, "runtime_signature.json": runtime, "costs.json": costs_doc, **evaluations,
    }
    public_hashes = {name: hashlib.sha256(_encoded(body)).hexdigest() for name, body in payloads.items()}
    payloads["provenance.json"] = {
        "schema": "public_anymal_termination_smoke_provenance_v1",
        "source_artifacts": dict(sorted(source_roles.items())),
        "public_artifact_sha256": dict(sorted(public_hashes.items())),
        "source_verification": "registered CPU replay passed before export",
        "private_paths_disclosed": False,
    }
    verify_public_payloads(payloads)
    return payloads


def _finite_number(value: Any, *, minimum: float = 0.0) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _aggregate(row: dict[str, Any]) -> dict[str, float]:
    conditions = row["conditions"]
    num_envs = sum(condition["num_envs"] for condition in conditions)
    alive = sum(condition["alive_steps_sum"] for condition in conditions)
    return {
        "fall_rate": sum(condition["fall_count"] for condition in conditions) / num_envs,
        "mean_survival_s": alive * row["protocol"]["step_dt"] / num_envs,
        "lin_vel_rmse_mps": math.sqrt(sum(condition["lin_error_sq_sum"] for condition in conditions) / alive),
        "yaw_rate_rmse_radps": math.sqrt(sum(condition["yaw_error_sq_sum"] for condition in conditions) / alive),
    }


def _behavior(row: dict[str, Any]) -> dict[str, float]:
    return {
        "survival_s": row["metrics"]["mean_survival_s"],
        "horizon_s": row["protocol"]["horizon_steps"] * row["protocol"]["step_dt"],
        "fall_frac": row["metrics"]["fall_rate"],
        "err_xy": row["metrics"]["lin_vel_rmse_mps"],
        "err_yaw": row["metrics"]["yaw_rate_rmse_radps"],
    }


def verify_public_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(payloads) != PUBLIC_FILES:
        raise ValueError("public artifact inventory differs")
    encoded = json.dumps(payloads, ensure_ascii=False, allow_nan=False)
    if FORBIDDEN.search(encoded):
        raise ValueError("public artifact contains a private or absolute path")
    for name, expected_sha in EXPECTED_PUBLIC_CANONICAL_SHA256.items():
        canonical = json.dumps(payloads[name], sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != expected_sha:
            raise ValueError(f"public registered content differs: {name}")

    summary = payloads["summary.json"]
    runtime = payloads["runtime_signature.json"]
    costs = payloads["costs.json"]
    provenance = payloads["provenance.json"]
    if (summary.get("schema") != "public_anymal_termination_smoke_summary_v1"
            or summary.get("status") != "complete" or summary.get("seed") != SEED
            or summary.get("prediction") != "failure_present" or summary.get("prediction_match") is not False
            or summary.get("outcome") != "source_failure_absent"
            or summary.get("may_preregister_demo") is not False or summary.get("demo_executed") is not False
            or summary.get("model_calls") != 0 or summary.get("fixed_evaluation", {}).get("bands") != BANDS
            or summary.get("source_commit_sha") != "c914acc0361a0b71fce0860980fcd940cee5c929"
            or summary.get("protocol_sha256_lf") != EXPECTED_SHA256["protocol"]
            or any(summary.get("claims", {}).values())):
        raise ValueError("public summary differs")

    rows = {arm: payloads[f"evaluations/{arm.lower()}.json"] for arm in ARMS}
    for arm, row in rows.items():
        if (set(row) != {"schema", "seed", "arm", "protocol", "checkpoint", "metrics", "standstill",
                         "conditions", "source_artifact_sha256"}
                or row.get("schema") != "public_anymal_termination_smoke_evaluation_v1"
                or row.get("seed") != SEED or row.get("arm") != arm
                or len(row.get("conditions", [])) != 26
                or provenance.get("source_artifacts", {}).get(f"fixed_evaluation.{arm.lower()}")
                    != row.get("source_artifact_sha256")):
            raise ValueError("public evaluation differs")
        protocol = row["protocol"]
        if (protocol.get("num_envs") != 1040 or protocol.get("horizon_steps") != 1000
                or protocol.get("step_dt") != 0.02 or protocol.get("action_scale") != 0.5):
            raise ValueError("public evaluation protocol differs")
        commands = set()
        for condition in row["conditions"]:
            if (set(condition) != {"command", "num_envs", "fall_count", "alive_steps_sum",
                                   "lin_error_sq_sum", "yaw_error_sq_sum", "fall_rate",
                                   "lin_vel_rmse_mps", "yaw_rate_rmse_radps"}
                    or type(condition["num_envs"]) is not int or condition["num_envs"] != 40
                    or type(condition["fall_count"]) is not int or condition["fall_count"] < 0
                    or condition["fall_count"] > condition["num_envs"]
                    or not all(_finite_number(condition[key]) for key in (
                        "alive_steps_sum", "lin_error_sq_sum", "yaw_error_sq_sum", "fall_rate",
                        "lin_vel_rmse_mps", "yaw_rate_rmse_radps"))):
                raise ValueError("public condition differs")
            command = tuple(condition["command"])
            if len(command) != 3 or any(value not in (-1.0, 0.0, 1.0) for value in command) \
                    or command == (0.0, 0.0, 0.0):
                raise ValueError("public command grid differs")
            commands.add(command)
            expected_condition = {
                "fall_rate": condition["fall_count"] / condition["num_envs"],
                "lin_vel_rmse_mps": math.sqrt(condition["lin_error_sq_sum"] / condition["alive_steps_sum"]),
                "yaw_rate_rmse_radps": math.sqrt(condition["yaw_error_sq_sum"] / condition["alive_steps_sum"]),
            }
            if any(not math.isclose(condition[key], value, rel_tol=1e-6, abs_tol=1e-8)
                   for key, value in expected_condition.items()):
                raise ValueError("public condition metric replay differs")
        if len(commands) != 26:
            raise ValueError("public command inventory differs")
        aggregated = _aggregate(row)
        if any(not math.isclose(aggregated[key], row["metrics"][key], rel_tol=1e-6, abs_tol=1e-8)
               for key in aggregated):
            raise ValueError("public aggregate metric replay differs")

    none, fault = rows["NONE"], rows["F6"]
    verdict = RECOVERY.verdict(RECOVERY.compare(_behavior(fault), _behavior(none)), BANDS)
    none_walks = (none["metrics"]["lin_vel_rmse_mps"] < 0.5 * none["standstill"]["lin_vel_rmse_mps"]
                  and none["metrics"]["fall_rate"] <= 0.05)
    outcome = "baseline_invalid" if not none_walks else (
        "failure_present" if verdict["label"] == "unhealthy" else "source_failure_absent")
    expected_fixed = {"conditions": 26, "bands": BANDS, "none_walks": none_walks,
                      "fault_verdict": verdict["label"]}
    if outcome != summary["outcome"] or summary["fixed_evaluation"] != expected_fixed:
        raise ValueError("public source gate replay differs")

    if (runtime.get("schema") != "public_anymal_termination_smoke_runtime_v1"
            or runtime.get("fault") != "F6" or runtime.get("fault_timeout_s") != 1.0
            or runtime.get("tail_iterations") != 60
            or runtime.get("checks") != {"f6_episodes_at_most_50_steps": True,
                                          "f6_time_out_observed": True,
                                          "none_episode_median_above_50": True}
            or runtime.get("episode_steps_median", {}).get("F6") != 50.0
            or runtime.get("episode_steps_median", {}).get("NONE", 0) <= 50.0
            or runtime.get("f6_episode_steps_max", math.inf) > 50.0 + 1e-6
            or runtime.get("f6_time_out_max", 0) <= 0
            or runtime.get("behavior_decision_source") != "clean fixed evaluation"):
        raise ValueError("public runtime signature differs")

    if (costs.get("schema") != "public_anymal_termination_smoke_costs_v1"
            or costs.get("not_gpu_kernel_time") is not True
            or set(costs.get("training", {})) != set(ARMS)):
        raise ValueError("public costs differ")
    training_total = extract_total = 0.0
    for arm in ARMS:
        row = costs["training"][arm]
        if set(row) != {"training_wall_s", "extract_wall_s"} or not all(_finite_number(v) for v in row.values()):
            raise ValueError("public cost row differs")
        training_total += row["training_wall_s"]
        extract_total += row["extract_wall_s"]
    fixed = costs.get("fixed_eval_wall_s")
    totals = costs.get("totals_wall_s", {})
    if isinstance(fixed, bool) or not isinstance(fixed, (int, float)) \
            or not math.isfinite(fixed) or fixed < 0:
        raise ValueError("public fixed evaluation cost differs")
    fixed_value = float(fixed)
    expected_totals = {"training": training_total, "telemetry_extract": extract_total,
                       "training_and_fixed_evaluation": training_total + fixed_value,
                       "accounted_with_extract": training_total + extract_total + fixed_value}
    if set(totals) != set(expected_totals) or any(
            not math.isclose(totals[key], value, abs_tol=1e-9) for key, value in expected_totals.items()):
        raise ValueError("public cost total differs")

    if (provenance.get("schema") != "public_anymal_termination_smoke_provenance_v1"
            or provenance.get("private_paths_disclosed") is not False
            or provenance.get("source_artifacts") != EXPECTED_SOURCE_ROLES):
        raise ValueError("public source provenance differs")
    public_hashes = provenance.get("public_artifact_sha256")
    expected_public = PUBLIC_FILES - {"provenance.json"}
    if not isinstance(public_hashes, dict) or set(public_hashes) != expected_public:
        raise ValueError("public hash inventory differs")
    for name in expected_public:
        if hashlib.sha256(_encoded(payloads[name])).hexdigest() != public_hashes[name]:
            raise ValueError("public artifact hash differs")
    return {"status": "valid", "files": len(PUBLIC_FILES), "evaluations": 2,
            "outcome": outcome, "may_preregister_demo": False}


def export(isaaclab: Path) -> dict[str, Any]:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("public output or audit receipt already exists")
    before = _snapshot(RUN)
    payloads = verify_and_build(isaaclab)
    for relative, body in payloads.items():
        write_new(OUTPUT / relative, body)
    verified = verify_public(OUTPUT)
    after = _snapshot(RUN)
    if before != after:
        raise ValueError("source result changed during export")
    receipt = {
        "schema": "anymal_termination_smoke_export_receipt_v1",
        "status": "complete",
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_unchanged": True,
        "public_files": {name: sha256(OUTPUT / name) for name in sorted(PUBLIC_FILES)},
        "verification": verified,
    }
    write_new(AUDIT / "export_receipt.json", receipt)
    return {"status": "exported", "output": OUTPUT.relative_to(ROOT).as_posix(),
            "files": len(PUBLIC_FILES), "evaluations": 2, "source_files": before["file_count"]}


def verify_public(folder: Path = OUTPUT) -> dict[str, Any]:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("public output folder is unavailable")
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != PUBLIC_FILES:
        raise ValueError("public output file inventory differs")
    return verify_public_payloads({name: read(folder / name) for name in PUBLIC_FILES})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = ({"status": "ready", "files": len(verify_and_build(args.isaaclab))}
                  if args.command == "check" else export(args.isaaclab)
                  if args.command == "export" else verify_public(args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
