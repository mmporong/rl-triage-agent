"""U1 확장 교정을 처음 적용할 seed 2062의 원 결함 증거를 기록한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import u1_scope_source as BASE  # noqa: E402
import upstream_runs as UR  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

SEED = 2062
RESTART_SEED = 3062
SOURCE_PRIVATE = ROOT / "bench/private/codex_support_20261015/task11/u1/source"
SOURCE_RESULTS = ROOT / "bench/private/codex_support_20261015/task11/u1/source_results"
SOURCE_PROTOCOL_DIR = ROOT / "bench/protocols/novel_u1_scope_source"
SOURCE_CODE = {
    "evals/novel_u1_scope_source.py",
    "evals/novel_u1_scope_confirmation.py",
    "evals/u1_scope_source.py",
    "bench/novel_u1_family_training.py",
    "bench/reward_family_correction.py",
    "bench/protocols/novel_u1_scope_confirmation_v1.json",
    "evals/preregistered_seed_guard.py",
}
FRESHNESS_RECEIPT = ROOT / "bench/private/codex_support_20261015/task11/u1/seed_freshness.json"
_BASE_SCOPE_BINDINGS = BASE._scope_bindings


def source_fixed(variant: str, decision_path: str) -> dict:
    row = deepcopy(BASE._BASE_FIXED(variant, decision_path))
    row.update({
        "question": "처음 쓰는 seed 2062 U1에서 실패와 P_reward 두 추종 항을 재측정하고 네 항 확장 교정을 시험할 입력을 결합할 수 있는가",
        "scope": "known_u1_family_novel_seed_confirmation_source_only",
        "seed": SEED,
        "restart_seed": RESTART_SEED,
        "seed_selection": "source seed 2062와 restart seed 3062를 결과 전에 고정했고 GPU 실행 직전에 새 seed 여부를 다시 검사한다",
        "enabled_steps": BASE.ENABLED_STEPS,
        "api_calls": 0,
        "model_loop": {"status": "inactive", "reason": "저장된 규칙과 측정만 사용한다"},
        "diagnosis": {"status": "inactive", "api_calls": 0},
        "loop": {"status": "inactive", "api_calls": 0},
        "specificity_gate": {"status": "inactive"},
        "scope_gate": {"status": "development_registration", "probe": "P_reward"},
        "correction": {"status": "inactive", "reason": "교정 학습은 확인 프로토콜에서 실행한다"},
        "common_wording": "기존 U1 결과를 본 뒤 만든 함수군 등록을 처음 쓰는 seed에서 확인한다.",
        "decision_path_rule": "모델/API 경로는 비활성이다",
        "changes_from_v3_1": ["source seed 2062", "write-once private outputs", "novel-seed confirmation"],
        "forbidden_claims": ["진단 정확도·우월성", "일반화", "교정 성공률", "GPU 절감", "모델 판단을 사용했다는 표기"],
    })
    row["predictions"] = {
        "source_failure": "NONE은 걷고 U1은 고정 평가에서 unhealthy",
        "p_reward": "두 추종 항 중 큰 상대 오차가 5%를 넘어서 abnormal",
        "source_family": "소스와 학습 readback에서 U1 함수군 네 항이 일치",
    }
    row["success_conditions"] = {
        "source_failure_present": "동일 seed NONE은 걷고 U1은 unhealthy",
        "p_reward_observed": "P_reward가 abnormal",
        "source_family_bound": "네 등록 함수가 소스와 환경에 결합됨",
    }
    row["success_rule"] = "세 source-only 조건을 모두 검증해야 확장 교정 학습으로 진행한다"
    return row


def configure() -> None:
    BASE.SOURCE_PRIVATE = SOURCE_PRIVATE
    BASE.SOURCE_RESULTS = SOURCE_RESULTS
    BASE.SOURCE_PROTOCOL_DIR = SOURCE_PROTOCOL_DIR
    BASE.SOURCE_CODE = SOURCE_CODE
    BASE.E.VARIANTS["u1"] = {"fault": "U1", "case_id": "u1", "mechanism": "reward", "seed": SEED}
    BASE.E.PRIVATE = SOURCE_PRIVATE
    BASE.E.RESULTS = SOURCE_RESULTS
    BASE.E.PROTOCOL_DIR = SOURCE_PROTOCOL_DIR
    BASE.E.CODE |= SOURCE_CODE
    BASE.E.fixed = source_fixed


def scope_bindings(tag: str, isaaclab: Path, probe_path: Path) -> dict:
    row = _BASE_SCOPE_BINDINGS(tag, isaaclab, probe_path)
    freshness = BASE.E.read(FRESHNESS_RECEIPT)
    if freshness.get("status") != "fresh" or freshness.get("seeds") != [SEED, RESTART_SEED]:
        raise ValueError("fresh-seed receipt differs")
    row["seed_freshness_receipt_path"] = BASE._canonical(FRESHNESS_RECEIPT)
    row["seed_freshness_receipt_sha256"] = BASE.E.sha(FRESHNESS_RECEIPT)
    return row


def validate_scope_record(scope_path: Path, isaaclab: Path) -> dict:
    configure()
    scope_path = Path(scope_path).resolve()
    scope = BASE.E.read(scope_path)
    tag = scope.get("bindings", {}).get("source_tag")
    if (scope.get("schema") != "novel_u1_scope_confirmation_scope_v1" or scope.get("fault") != "U1"
            or scope.get("seed") != SEED or not isinstance(tag, str)):
        raise ValueError("scope record identity differs")
    canonical = (BASE.E.folder_for(tag) / "scope.json").resolve()
    if not canonical.is_file() or BASE.E.sha(canonical) != BASE.E.sha(scope_path):
        raise ValueError("scope record differs from its canonical source run")
    probe_path = Path(scope["bindings"].get("probe_receipt_path", "")).resolve()
    if probe_path != (BASE.E.folder_for(tag) / "scope_probe.json").resolve():
        raise ValueError("scope probe receipt escapes its source run")
    if scope.get("bindings") != scope_bindings(tag, isaaclab, probe_path):
        raise ValueError("scope bindings differ from source artifacts")
    probe = BASE.E.read(probe_path)
    evidence = BASE.source_evidence(tag, isaaclab)
    expected_paths = UR.probe_paths(f"{tag}_scope", evidence["ctx"].cases["FAULT"], "P_reward", SOURCE_RESULTS)
    execution_path = Path(probe.get("execution_receipt_path", ""))
    report = BASE.E.read(expected_paths["report"])
    measurement = report.get("probes", {}).get("P_reward", {}).get("measurement")
    if (probe.get("schema") != "novel_u1_scope_probe_receipt_v1"
            or probe.get("case") != evidence["ctx"].cases["FAULT"] or probe.get("probe") != "P_reward"
            or probe.get("exit_code") != 0 or probe.get("measurement") != measurement
            or execution_path.resolve() != expected_paths["execution"].resolve()
            or probe.get("execution_receipt_sha256") != BASE.E.sha(execution_path)
            or probe.get("checkpoint_sha256") != evidence["artifacts"]["FAULT"]["checkpoint"]["sha256"]
            or L.classify("P_reward", measurement,
                          evidence["reference"]["probes"]["P_reward"]["measurement"]) != "abnormal"
            or probe.get("outcome") != "abnormal"):
        raise ValueError("scope P_reward receipt differs from source evidence")
    return scope


def diagnose_scope(tag: str, isaaclab: Path) -> dict:
    configure()
    evidence = BASE.source_evidence(tag, isaaclab)
    ctx = evidence["ctx"]
    result = UR.probe(isaaclab, fault="U1", seed=ctx.seed, run_name=ctx.cases["FAULT"], probe="P_reward",
                      tag=f"{tag}_scope", work_dir=BASE.E.folder_for(tag) / "exec/scope_probe",
                      results=SOURCE_RESULTS, watchdog_s=600)
    measurement, exit_code = result["measurement"], result["exit_code"]
    outcome = L.classify("P_reward", measurement if exit_code == 0 else None,
                         evidence["reference"]["probes"]["P_reward"]["measurement"])
    if outcome != "abnormal":
        raise ValueError("P_reward did not confirm the preregistered source failure")
    probe_path = BASE.E.folder_for(tag) / "scope_probe.json"
    BASE.E.write_new(probe_path, {
        "schema": "novel_u1_scope_probe_receipt_v1", "case": ctx.cases["FAULT"], "probe": "P_reward",
        "measurement": measurement, "outcome": outcome, "exit_code": exit_code, "gpu_s": result["gpu_s"],
        "checkpoint_sha256": evidence["artifacts"]["FAULT"]["checkpoint"]["sha256"],
        "execution_receipt_path": BASE._canonical(Path(result["paths"]["execution"])),
        "execution_receipt_sha256": BASE.E.sha(Path(result["paths"]["execution"])),
    })
    row = {
        "schema": "novel_u1_scope_confirmation_scope_v1", "fault": "U1", "seed": SEED,
        "fault_family": "root_body_velocity_reward_family", "diagnosis": "known_u1_reward_family",
        "observed_probe_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"],
        "source_registered_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"],
        "result": {"verdict": "scope_insufficient", "uncovered_terms": [
            "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]},
        "bindings": scope_bindings(tag, isaaclab, probe_path),
        "design_coupling": "기존 U1 결과를 본 뒤 만든 개발 등록이며 측정 두 항과 소스 확인 네 항을 구분한다",
    }
    path = BASE.E.folder_for(tag) / "scope.json"
    BASE.E.write_new(path, row)
    validate_scope_record(path, isaaclab)
    return {"scope": str(path), "scope_sha256": BASE.E.sha(path), "probe": str(probe_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fresh-seeds", "begin", "train", "alarm", "evaluate", "reference", "diagnose-scope", "status"))
    parser.add_argument("--tag")
    parser.add_argument("--arm", choices=("NONE", "FAULT"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    configure()
    BASE.configure = configure
    BASE._scope_bindings = scope_bindings
    if args.command == "prepare": result = BASE.E.prepare("u1", "single", args.isaaclab)
    elif args.command == "fresh-seeds":
        from preregistered_seed_guard import assert_fresh_seeds
        result = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
    elif args.command == "begin":
        from preregistered_seed_guard import assert_fresh_seeds
        freshness = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
        BASE.E.write_new(FRESHNESS_RECEIPT, freshness)
        result = BASE.E.begin(args.tag, "u1", args.isaaclab)
    elif args.command == "train": result = BASE.E.train(args.tag, args.arm, args.isaaclab)
    elif args.command == "alarm": result = BASE.E.alarm(args.tag, args.isaaclab)
    elif args.command == "evaluate": result = BASE._private_fixed_evaluate(args.tag, args.isaaclab)
    elif args.command == "reference": result = BASE._private_reference(args.tag, args.isaaclab)
    elif args.command == "diagnose-scope": result = diagnose_scope(args.tag, args.isaaclab)
    else: result = BASE.E.status(args.tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
