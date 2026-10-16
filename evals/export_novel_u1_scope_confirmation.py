"""새 seed U1 범위 확장 확인 결과를 공개용 allowlist JSON으로 내보낸다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import e2e_reward_demo_v2 as DEMO  # noqa: E402
import novel_u1_scope_confirmation as NOVEL  # noqa: E402
from rl_triage import recovery as RECOVERY  # noqa: E402

BASE = NOVEL.BASE
TAG = "novel_u1_scope_s2062_20261016"
PRIVATE_ROOT = ROOT / "bench/private/codex_support_20261015/task11/u1"
RUN = PRIVATE_ROOT / "runs" / TAG
SOURCE_RESULTS = PRIVATE_ROOT / "source_results"
OUTPUT = ROOT / "evals/results/novel_u1_scope_confirmation_s2062_20261016"
TELEMETRY = {
    "source_none": ROOT / "bench/runs_p0c/baseline_p0c_s2062.telemetry.json",
    "source_u1": ROOT / "bench/runs_p0c/u1_s2062.telemetry.json",
}
INPUTS = {
    "protocol": ROOT / "bench/protocols/novel_u1_scope_confirmation_v1.json",
    "fixed_eval_protocol": ROOT / "bench/protocols/fixed_eval_v1.json",
    "bands": ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json",
    "source.none": SOURCE_RESULTS / "novel_u1_scope_source_s2062/runs/baseline_p0c_s2062.json",
    "source.u1": SOURCE_RESULTS / "novel_u1_scope_source_s2062/runs/u1_s2062.json",
    "source.probe": SOURCE_RESULTS / "novel_u1_scope_source_s2062_scope/probes/u1_s2062__P_reward.json",
    "source.probe_receipt": PRIVATE_ROOT / "source/novel_u1_scope_source_s2062/scope_probe.json",
    "run.result": RUN / "result.json",
    "run.scope": RUN / "scope.json",
    "run.approval": RUN / "approval.json",
    "run.consumed": RUN / "consumed.json",
    "run.prereg": RUN / "prereg.json",
    "run.eval.execution": RUN / "eval/execution.json",
    "run.eval.jobs": RUN / "eval/jobs.json",
    "run.eval.launch": RUN / "eval/launch.json",
    "run.eval.gpu_before": RUN / "eval/gpu_before.json",
    "run.eval.gpu_after": RUN / "eval/gpu_after.json",
    "run.eval.none": RUN / "eval/none_family_control.json",
    "run.eval.corrected": RUN / "eval/u1_family_restart.json",
    "telemetry.none": TELEMETRY["source_none"],
    "telemetry.u1": TELEMETRY["source_u1"],
}
for arm in NOVEL.ARMS:
    for name in ("manifest", "execution", "receipt", "correction", "family_environment", "training_attestation"):
        INPUTS[f"run.{arm}.{name}"] = RUN / arm / "train" / f"{name}.json"

EXPECTED_SHA256 = {
    "protocol": "e3680a6006ecf0fff60ef312df4c9fbadc2a847bd69649c9847964897ed73e5d",
    "fixed_eval_protocol": "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46",
    "bands": "7589a132d4e1a55e159527389a8b635a1766472ecde558da0a1a704c286f3f23",
    "source.none": "9da98c2aabcd15904d310ba0286a52e0f05c2d26b519ab012d322f073f97d221",
    "source.u1": "7cb8d404d21054d805de7d8194232d75534001b3777e4fd793486a3d2b999f5a",
    "source.probe": "46ed287e086deeb60dc302f82c76f82e3ebd9c0aa5f83d43d48641d0937b4fae",
    "source.probe_receipt": "35083c4d784d5fc8ea70a02dee8a0e8db6a0b0da6197b6650d37c8fe1a873563",
    "run.result": "d53524c0980e4d869f88da122df6943a6edbe549c60b3f1629a3082da87a011c",
    "run.scope": "1ec91b66dc96a9e18a5d42a0d48efae0eaefcf3d9d5689dab9c190d3c2f543cd",
    "run.approval": "bf341b27947887965253a26390ed5da2fc6492525f25a96c4937ebf321031615",
    "run.consumed": "203d357434d135fbfc60a22a52b93fd3daf93e958b0de17efe4a0a1f39c8cbda",
    "run.prereg": "f56b7423c31e894679d6c2a1ae39ee18345bdd28a8c5e5bb8bdce196c65cd02e",
    "run.eval.execution": "50c69ed2fc8ea8cb9e48414fe686a51b67c7ddda01437bfea326ea111160cadb",
    "run.eval.jobs": "595877fbdcaf4d5e49b09df5eaf5313719b0c31d2cb1efeefb1418567714b184",
    "run.eval.launch": "07fe23345bcbca3a621c0442d99334266642430cda11e59f9b6895619ca1dce0",
    "run.eval.gpu_before": "223eb8abb24e3988f357333a32702da2be29f98dc9cbbf0e6e10602944c2a9ad",
    "run.eval.gpu_after": "fa00d5ba6a948f89888cbc2c4d705b0ae2a0d0d196b30b9ddb7f65697ed5b7f1",
    "run.eval.none": "06585abc871b9377259c00d831c8108d1b3aebeb191120d484e49abf62f3425f",
    "run.eval.corrected": "2c448af82043335f6ee3d11bcd778aee7b6dcd8d958b424dea2da71dd47b888c",
    "telemetry.none": "2b0359e1c47d6e54b4b3f6490fafc60a517e550c12a89e0979a62034ce0a543a",
    "telemetry.u1": "9b6d0f0094d82fccd1520a42cee3b0c63a1ecdca2716e686823b0144d31d076b",
    "run.none_family_control.manifest": "b7873a0aecc57c13642988946da9d09d94441c7c11a1cee3077f21c04a65696c",
    "run.none_family_control.execution": "69f3fdba2f7295b4eae47e9de75602c917b5c881afa7f61f419248126de98b09",
    "run.none_family_control.receipt": "a1be0e60378e103ced572895afa1a881cb083e3768cfa34a7d8f40e3091bc4a3",
    "run.none_family_control.correction": "8e7247f61ddf396de08121c008ab5d6e90987e57e8896c5977bf397c4fc2ef25",
    "run.none_family_control.family_environment": "c797fbf337ee57853c808668437cea6dca9813e134eb2e51a7fdf9d3c903fda1",
    "run.none_family_control.training_attestation": "30a69acc7a840d8f0ef5ba991fcfc89bd351d244f3ee3842464f665034e7ef5a",
    "run.u1_family_restart.manifest": "6104f5be8519ed6b0abcd3432189a4d584b7e6d98dfaa6193dbc162f1f9baba2",
    "run.u1_family_restart.execution": "a2cc6fd6331d8592896ac951b8f594d0c48e9a1363f699fc3ee19695f1bec8e6",
    "run.u1_family_restart.receipt": "576c02bd74b83ec924a52dd64d59582b382b0292650a522ab3fe34f7ddd1cb15",
    "run.u1_family_restart.correction": "44d5c1292e99ccc0408ad8b0eba831fc18b46c6acd8db71e955af89728509787",
    "run.u1_family_restart.family_environment": "c797fbf337ee57853c808668437cea6dca9813e134eb2e51a7fdf9d3c903fda1",
    "run.u1_family_restart.training_attestation": "5e2c4a052c6675f3c5075ed48439185b0671f6239b70522730ff294711212dae",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_rows(report: dict) -> list[dict]:
    return [{
        "target_velocity": row["command"],
        "fall_rate": row["fall_rate"],
        "lin_vel_rmse_mps": row["lin_vel_rmse_mps"],
        "yaw_rate_rmse_radps": row["yaw_rate_rmse_radps"],
        "num_envs": row["num_envs"],
    } for row in report["by_condition"]]


def _public_evaluation(arm: str, report: dict, verdict: dict, receipt: dict, attestation: dict) -> dict:
    return {
        "schema": "public_novel_u1_scope_evaluation_v1",
        "arm": arm,
        "training": {
            "seed": receipt["seed"], "mode": receipt["recovery_mode"],
            "max_iterations": receipt["budget"]["max_iterations"],
            "num_envs": receipt["budget"]["num_envs"],
            "num_steps_per_env": receipt["budget"]["num_steps_per_env"],
            "aggregate_env_steps": receipt["aggregate_env_steps"],
        },
        "label": verdict["label"],
        "verdict": {"label": verdict["label"], "checks": verdict["checks"], "failed_checks": verdict["failed"]},
        "fixed_evaluation": {
            "condition_count": len(report["by_condition"]), "environments_per_condition": 40,
            "horizon_steps": report["horizon_steps"], "action_scale": report["action_scale"],
            "metrics": report["metrics"], "standstill": report["standstill"],
            "conditions": _condition_rows(report),
        },
        "checkpoint": report["checkpoint"],
        "artifact_sha256": {
            "manifest": EXPECTED_SHA256[f"run.{arm}.manifest"],
            "receipt": EXPECTED_SHA256[f"run.{arm}.receipt"],
            "training_attestation": EXPECTED_SHA256[f"run.{arm}.training_attestation"],
            "fixed_evaluation": EXPECTED_SHA256["run.eval.none" if arm == "none_family_control" else "run.eval.corrected"],
        },
    }


def verify_and_build(isaaclab: Path) -> dict[str, dict]:
    if set(INPUTS) != set(EXPECTED_SHA256):
        raise ValueError("source artifact inventory differs")
    for logical_id, path in INPUTS.items():
        if not path.is_file() or sha256(path) != EXPECTED_SHA256[logical_id]:
            raise ValueError(f"source artifact differs: {logical_id}")

    NOVEL.configure()
    protocol = NOVEL.validate_protocol(NOVEL.fixed_protocol(), isaaclab)
    scope, _ = NOVEL.validate_authorization(RUN / "scope.json", RUN / "approval.json", isaaclab)
    prereg = read(RUN / "prereg.json")
    BASE._validate_source_inputs(prereg, isaaclab)
    source_none, source_u1 = read(INPUTS["source.none"]), read(INPUTS["source.u1"])
    bands = read(INPUTS["bands"])["bands"]
    source_verdict = RECOVERY.verdict(
        RECOVERY.compare(BASE.C._behavior(source_u1), BASE.C._behavior(source_none)), bands
    )
    if source_verdict["label"] != "unhealthy" or not DEMO.walks(source_none):
        raise ValueError("source failure gate differs")

    versions = source_none["versions"]
    reports, verdicts, attestations = {}, {}, {}
    for arm in NOVEL.ARMS:
        execution = read(RUN / arm / "train/execution.json")
        attestations[arm] = BASE._strict_verify(RUN, arm, execution, protocol, versions)
        report_path = RUN / "eval" / f"{arm}.json"
        reports[arm] = BASE.C.validate_fixed_report(read(report_path), {
            "name": arm, "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": attestations[arm]["checkpoint"]["sha256"],
            "action_scale": 0.25, "protocol_sha256_lf": sha256_lf(INPUTS["fixed_eval_protocol"]),
        })
        if reports[arm]["versions"] != versions:
            raise ValueError("fixed evaluation runtime differs")
        verdicts[arm] = RECOVERY.verdict(
            RECOVERY.compare(BASE.C._behavior(reports[arm]), BASE.C._behavior(source_none)), bands
        )

    stored = read(RUN / "result.json")
    recomputed = {
        "schema": "u1_scope_extension_result_v1", "tag": TAG, "outcome": "recovered",
        "none_control_healthy": verdicts["none_family_control"]["label"] == "healthy" and DEMO.walks(reports["none_family_control"]),
        "none_control_walks": DEMO.walks(reports["none_family_control"]),
        "arms": {arm: {"label": verdicts[arm]["label"], "verdict": verdicts[arm]} for arm in NOVEL.ARMS},
        "wording": protocol["wording"]["recovered"], "generalization_claim": False,
        "protocol_sha256_lf": sha256_lf(INPUTS["protocol"]), "source_commit_sha": prereg["source_commit_sha"],
    }
    if recomputed != stored or not recomputed["none_control_healthy"] or verdicts["u1_family_restart"]["label"] != "healthy":
        raise ValueError("stored recovery result differs from CPU recomputation")

    probe = read(INPUTS["source.probe"])["probes"]["P_reward"]
    probe_receipt = read(INPUTS["source.probe_receipt"])
    if probe["measurement"] != probe_receipt["measurement"] or probe_receipt["outcome"] != "abnormal":
        raise ValueError("scope probe binding differs")

    receipts = {arm: read(RUN / arm / "train/receipt.json") for arm in NOVEL.ARMS}
    corrections = {arm: read(RUN / arm / "train/correction.json") for arm in NOVEL.ARMS}
    environments = {arm: read(RUN / arm / "train/family_environment.json") for arm in NOVEL.ARMS}
    summary = {
        "schema": "public_novel_u1_scope_confirmation_summary_v1", "status": "complete",
        "measurement_kind": "post_result_development_confirmation", "outcome": "recovered",
        "source_training_seed": 2062, "none_control_training_seed": 2062,
        "corrected_u1_restart_training_seed": 3062,
        "training_protocol": {"max_iterations": 300, "num_envs": 4096, "num_steps_per_env": 24},
        "fixed_evaluation_protocol": {"condition_count": 26, "environments_per_condition": 40, "horizon_steps": 1000, "action_scale": 0.25},
        "source_failure": {"outcome": "failure_present", "none_walks": True, "u1_label": source_verdict["label"]},
        "none_control": {"label": verdicts["none_family_control"]["label"], "healthy": True, "walks": True},
        "corrected_u1_restart": {"label": verdicts["u1_family_restart"]["label"], "recovery_label": "recovered"},
        "scope": {"measured_probe_terms": 2, "source_registered_terms": 4, "restored_terms": 4},
        "design_limit": "기존 U1 결과를 본 뒤 넓힌 함수군 교정을 처음 쓰는 seed에서 확인한 개발 측정이다. API 호출은 0회이고 모델 판단 고리는 비활성이다.",
        "claim_limit": "이 결과만으로 다른 seed, 결함, 로봇에 대한 일반화나 원인 규명, 비용 절감, 모델 우월성을 주장하지 않는다.",
        "source_commit_sha": stored["source_commit_sha"], "protocol_sha256_lf": stored["protocol_sha256_lf"],
    }
    source_evidence = {
        "schema": "public_novel_u1_source_failure_evidence_v1",
        "fixed_evaluation": summary["fixed_evaluation_protocol"],
        "arms": {
            "NONE": {"case": source_none["name"], "training_seed": 2062, "metrics": source_none["metrics"],
                     "checkpoint": source_none["checkpoint"], "report_sha256": EXPECTED_SHA256["source.none"]},
            "U1": {"case": source_u1["name"], "training_seed": 2062, "metrics": source_u1["metrics"],
                   "checkpoint": source_u1["checkpoint"], "report_sha256": EXPECTED_SHA256["source.u1"]},
        },
        "gate": {"none_walks": True, "u1_label": source_verdict["label"], "u1_checks": source_verdict["checks"],
                 "u1_failed_checks": source_verdict["failed"], "outcome": "failure_present"},
    }
    scope_evidence = {
        "schema": "public_novel_u1_scope_evidence_v1", "fault": "U1", "source_training_seed": 2062,
        "fault_family": scope["fault_family"], "diagnosis": scope["diagnosis"],
        "scope_verdict": scope["result"]["verdict"], "measured_probe_terms": scope["observed_probe_terms"],
        "probe_measurement": probe["measurement"], "probe_outcome": probe_receipt["outcome"],
        "source_registered_terms": scope["source_registered_terms"], "uncovered_terms": scope["result"]["uncovered_terms"],
        "design_coupling": "known_u1_post_result_development_registration", "api_calls": 0, "model_loop": "inactive",
        "artifact_sha256": {"scope": EXPECTED_SHA256["run.scope"], "probe_receipt": EXPECTED_SHA256["source.probe_receipt"]},
    }
    readback = {"schema": "public_novel_u1_reward_family_readback_v1", "arms": {}}
    for arm in NOVEL.ARMS:
        correction, environment = corrections[arm], environments[arm]
        readback["arms"][arm] = {
            "registered_function_count": len(correction["registered_functions"]),
            "changed_by_fault_count": len(correction["changed_targets"]),
            "restored_count": len(correction["restored_targets"]), "restored_functions": correction["restored_targets"],
            "clean_function_count": sum(environment["clean"].values()),
            "bound_reward_term_count": sum(environment["bound_to_reward_terms"].values()),
            "environment_binding_ok": environment["ok"],
            "artifact_sha256": {"correction_readback": EXPECTED_SHA256[f"run.{arm}.correction"],
                                "family_environment": EXPECTED_SHA256[f"run.{arm}.family_environment"]},
        }
    evaluations = {
        arm: _public_evaluation(arm, reports[arm], verdicts[arm], receipts[arm], attestations[arm])
        for arm in NOVEL.ARMS
    }
    telemetry_rows = []
    for logical_id, path in TELEMETRY.items():
        telemetry = read(path)
        if len(telemetry["summary"]) != 24 or any(len(values) != 300 for values in telemetry["series"].values()):
            raise ValueError("public telemetry shape differs")
        telemetry_rows.append({"logical_id": logical_id, "path": path.relative_to(ROOT).as_posix(),
                               "sha256": sha256(path), "sha256_lf": sha256_lf(path),
                               "metric_count": 24, "points_per_metric": 300})
    provenance = {
        "schema": "public_novel_u1_scope_confirmation_provenance_v1",
        "source_commit_sha": stored["source_commit_sha"], "protocol_sha256_lf": stored["protocol_sha256_lf"],
        "source_artifact_sha256": deepcopy(EXPECTED_SHA256), "public_training_telemetry": telemetry_rows,
        "export_contract": "allowlisted_fields_only_no_private_or_user_paths",
    }
    return {
        "summary.json": summary, "source_failure_evidence.json": source_evidence,
        "scope_evidence.json": scope_evidence, "correction_readback.json": readback,
        "evaluations/none_family_control.json": evaluations["none_family_control"],
        "evaluations/u1_family_restart.json": evaluations["u1_family_restart"],
        "provenance.json": provenance,
    }


def write_new(folder: Path, payloads: dict[str, dict]) -> None:
    if folder.exists():
        raise FileExistsError("public result folder already exists")
    for relative, payload in payloads.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    payloads = verify_and_build(args.isaaclab.resolve())
    if args.command == "export":
        write_new(args.output.resolve(), payloads)
    print(json.dumps({"status": "verified" if args.command == "check" else "exported",
                      "outcome": payloads["summary.json"]["outcome"], "file_count": len(payloads)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
