"""U1 함수군 확장용 새 seed 2051 원 결함 증거를 비공개 경로에 기록한다."""
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

import e2e_demo_u1 as E  # noqa: E402
import upstream_runs as UR  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

SOURCE_PRIVATE = ROOT / "bench/private/codex_support_20261015/task5/source"
SOURCE_RESULTS = ROOT / "bench/private/codex_support_20261015/task5/source_results"
SOURCE_PROTOCOL_DIR = ROOT / "bench/protocols/u1_scope_extension_source"
SOURCE_CODE = {
    "evals/u1_scope_source.py", "evals/u1_scope_extension.py", "bench/reward_family_correction.py",
    "bench/demo_correction_u1_family.py", "bench/protocols/u1_scope_extension_v1.json",
}
ENABLED_STEPS = ["begin", "train_NONE", "train_FAULT", "alarm", "evaluate", "reference"]
_BASE_FIXED = E.fixed


def source_fixed(variant: str, decision_path: str) -> dict:
    row = deepcopy(_BASE_FIXED(variant, decision_path))
    row.update({
        "question": "새 seed 2051 U1에서 실패와 P_reward 두 추종 항 이상을 재측정하고 소스 함수군 네 항을 결합할 수 있는가",
        "scope": "known_u1_family_development_registration_source_only",
        "seed": 2051,
        "restart_seed": 3051,
        "seed_selection": "source failure seed 2051과 후속 restart seed 3051을 사전 고정했다",
        "enabled_steps": ENABLED_STEPS,
        "api_calls": 0,
        "model_loop": {"status": "inactive", "reason": "source-only 실행은 모델 판단·승인 probe 고리를 실행하지 않는다"},
        "diagnosis": {"status": "inactive", "api_calls": 0},
        "loop": {"status": "inactive", "api_calls": 0},
        "specificity_gate": {"status": "inactive"},
        "scope_gate": {"status": "development_registration", "probe": "P_reward"},
        "correction": {"status": "inactive", "reason": "교정 학습은 별도 확장 프로토콜에서 실행한다"},
        "common_wording": "알려진 U1 함수군을 본 뒤 만든 개발 등록에서 실패와 P_reward 두 추종 항을 새 seed로 재측정한다.",
        "decision_path_rule": "source-only 실행은 저장된 규칙과 측정만 사용하며 모델/API 경로는 비활성이다",
        "changes_from_v3_1": ["source seed 2051", "private fixed-eval/probe outputs", "known-U1 family development registration"],
        "forbidden_claims": ["진단 정확도·우월성", "일반화", "교정 성공률", "GPU 절감", "모델 판단을 사용했다는 표기"],
    })
    row["predictions"] = {
        "source_failure": "NONE은 걷고 U1은 고정 평가에서 unhealthy",
        "p_reward": "두 추종 항을 측정하고 둘 중 큰 상대 오차가 5%를 넘으면 P_reward abnormal",
        "source_family": "소스·학습 readback에서 등록된 네 함수가 U1 함수군과 일치",
    }
    row["success_conditions"] = {
        "source_failure_present": "동일 seed NONE이 걷고 U1이 unhealthy",
        "p_reward_observed": "두 추종 항을 측정하고 동결 P_reward 판정이 abnormal",
        "source_family_bound": "학습 readback과 고정 소스 계약에서 네 등록 함수가 확인됨",
    }
    row["success_rule"] = "세 source-only 조건을 모두 검증해야 scope 개발 기록을 만든다"
    return row


def configure() -> None:
    E.VARIANTS["u1"] = {"fault": "U1", "case_id": "u1", "mechanism": "reward", "seed": 2051}
    E.PRIVATE = SOURCE_PRIVATE
    E.RESULTS = SOURCE_RESULTS
    E.PROTOCOL_DIR = SOURCE_PROTOCOL_DIR
    E.CODE |= SOURCE_CODE
    E.fixed = source_fixed


def _digest(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _canonical(path: Path) -> str:
    return str(Path(path).resolve())


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _step_chain(tag: str, protocol_path: Path, commit: str) -> dict[str, dict]:
    previous = None
    rows: dict[str, dict] = {}
    for name in ENABLED_STEPS:
        path = E.step_path(tag, name)
        row = E.step_record(tag, name)
        expected_previous = None if previous is None else E.sha(previous)
        if (row.get("step") != name or row.get("previous_sha256") != expected_previous
                or row.get("source_commit_sha") != commit
                or row.get("protocol_sha256_lf") != E.sha(protocol_path, True)):
            raise ValueError(f"{name} source step chain differs")
        rows[name] = row
        previous = path
    return rows


def _validate_report(path: Path, name: str, checkpoint_sha: str, action_scale: float, fixed_sha: str) -> dict:
    return C.validate_fixed_report(E.read(path), {
        "name": name, "checkpoint_file": "model_299.pt", "checkpoint_sha256": checkpoint_sha,
        "action_scale": action_scale, "protocol_sha256_lf": fixed_sha,
    })


def source_evidence(tag: str, isaaclab: Path) -> dict:
    configure()
    ctx = E.ctx_for(tag)
    protocol, commit = E.verify_for(tag, isaaclab)
    steps = _step_chain(tag, ctx.protocol, commit)
    for name in ("train_NONE", "train_FAULT", "evaluate", "reference"):
        if not E.V3.execution_valid(steps[name].get("execution", {})):
            raise ValueError(f"{name} source execution receipt is invalid")
    artifacts = {}
    for arm in ("NONE", "FAULT"):
        actual = E.training_artifacts(ctx, arm, isaaclab)
        if steps[f"train_{arm}"].get("artifacts") != actual:
            raise ValueError(f"{arm} source training artifacts differ from current files")
        artifacts[arm] = actual

    readback = steps["train_FAULT"].get("upstream", {}).get("readback", {})
    exact_four = dict.fromkeys(("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"), True)
    if (readback.get("swapped") != exact_four or readback.get("bound_to_reward_terms") != exact_four
            or readback.get("ok") is not True):
        raise ValueError("U1 source readback did not bind the exact four reward functions")

    evaluate = steps["evaluate"]
    reports, report_paths = {}, {}
    for arm in ("NONE", "FAULT"):
        entry = evaluate.get("reports", {}).get(arm, {})
        path = _resolve_repo_path(entry.get("path", ""))
        expected_path = (SOURCE_RESULTS / tag / "runs" / f"{ctx.cases[arm]}.json").resolve()
        if path.is_symlink() or path != expected_path or entry.get("sha256_lf") != E.sha(path, True):
            raise ValueError(f"{arm} fixed report source binding differs")
        reports[arm] = _validate_report(path, ctx.cases[arm], artifacts[arm]["checkpoint"]["sha256"],
                                        protocol["action_scale"], protocol["fixed_eval_sha256_lf"])
        report_paths[arm] = path
    if (reports["NONE"]["versions"] != reports["FAULT"]["versions"]
            or reports["NONE"]["versions"]["torch"] != protocol["torch_version"]
            or reports["NONE"]["versions"]["rsl-rl-lib"] != protocol["rsl_rl_version"]
            or evaluate.get("gate") != E.source_gate(reports["NONE"], reports["FAULT"], protocol["bands"])
            or evaluate["gate"]["outcome"] != "failure_present"):
        raise ValueError("source fixed evaluation or failure gate differs")

    reference_entry = steps["reference"].get("report", {})
    reference_path = _resolve_repo_path(reference_entry.get("path", ""))
    expected_reference = (SOURCE_RESULTS / f"{tag}_ref" / "probes" / f"{ctx.cases['NONE']}.json").resolve()
    if (reference_path.is_symlink() or reference_path != expected_reference
            or reference_entry.get("sha256_lf") != E.sha(reference_path, True)):
        raise ValueError("NONE probe reference source binding differs")
    reference = E.read(reference_path)
    probes = reference.get("probes", {})
    if (reference.get("name") != ctx.cases["NONE"]
            or reference.get("checkpoint", {}).get("sha256") != artifacts["NONE"]["checkpoint"]["sha256"]
            or reference.get("seed") != ctx.seed or set(probes) != set(L.PROBES)
            or any(value.get("exit") != 0 or value.get("measurement") is None for value in probes.values())):
        raise ValueError("NONE probe reference does not bind six complete probes")
    return {"ctx": ctx, "protocol": protocol, "commit": commit, "steps": steps, "artifacts": artifacts,
            "readback": readback, "report_paths": report_paths, "reference_path": reference_path,
            "reference": reference}


def _scope_bindings(tag: str, isaaclab: Path, probe_path: Path) -> dict:
    evidence = source_evidence(tag, isaaclab)
    ctx, artifacts = evidence["ctx"], evidence["artifacts"]
    return {
        "source_tag": tag, "source_commit_sha": evidence["commit"],
        "source_protocol_path": _canonical(ctx.protocol), "source_protocol_sha256_lf": E.sha(ctx.protocol, True),
        "source_folder_path": _canonical(E.folder_for(tag)),
        "none_checkpoint_path": _canonical(Path(artifacts["NONE"]["checkpoint"]["path"])),
        "none_checkpoint_sha256": artifacts["NONE"]["checkpoint"]["sha256"],
        "u1_checkpoint_path": _canonical(Path(artifacts["FAULT"]["checkpoint"]["path"])),
        "u1_checkpoint_sha256": artifacts["FAULT"]["checkpoint"]["sha256"],
        "source_env_config_path": _canonical(Path(artifacts["NONE"]["environment"]["path"])),
        "source_env_config_sha256": artifacts["NONE"]["environment"]["sha256"],
        "source_agent_config_path": _canonical(Path(artifacts["NONE"]["agent"]["path"])),
        "source_agent_config_sha256": artifacts["NONE"]["agent"]["sha256"],
        "none_fixed_report_path": _canonical(evidence["report_paths"]["NONE"]),
        "none_fixed_report_sha256": E.sha(evidence["report_paths"]["NONE"]),
        "u1_fixed_report_path": _canonical(evidence["report_paths"]["FAULT"]),
        "u1_fixed_report_sha256": E.sha(evidence["report_paths"]["FAULT"]),
        "reference_probe_report_path": _canonical(evidence["reference_path"]),
        "reference_probe_report_sha256": E.sha(evidence["reference_path"]),
        "none_training_step_sha256": E.sha(E.step_path(tag, "train_NONE")),
        "fault_training_step_sha256": E.sha(E.step_path(tag, "train_FAULT")),
        "evaluation_step_sha256": E.sha(E.step_path(tag, "evaluate")),
        "reference_step_sha256": E.sha(E.step_path(tag, "reference")),
        "fault_readback_sha256": _digest(evidence["readback"]),
        "probe_receipt_path": _canonical(probe_path), "probe_receipt_sha256": E.sha(probe_path),
    }


def validate_scope_record(scope_path: Path, isaaclab: Path) -> dict:
    scope_path = Path(scope_path).resolve()
    scope = E.read(scope_path)
    tag = scope.get("bindings", {}).get("source_tag")
    if (scope.get("schema") != "u1_scope_extension_scope_v1" or scope.get("fault") != "U1"
            or scope.get("seed") != 2051 or not isinstance(tag, str)):
        raise ValueError("scope record identity differs")
    configure()
    canonical_scope = (E.folder_for(tag) / "scope.json").resolve()
    if not canonical_scope.is_file() or E.sha(canonical_scope) != E.sha(scope_path):
        raise ValueError("scope record differs from its canonical source run")
    probe_path = Path(scope["bindings"].get("probe_receipt_path", "")).resolve()
    if probe_path != (E.folder_for(tag) / "scope_probe.json").resolve():
        raise ValueError("scope P_reward receipt is outside its canonical source run")
    if scope.get("bindings") != _scope_bindings(tag, isaaclab, probe_path):
        raise ValueError("scope bindings differ from current source artifacts")
    probe = E.read(probe_path)
    evidence = source_evidence(tag, isaaclab)
    expected_keys = {"schema", "case", "probe", "measurement", "outcome", "exit_code", "gpu_s", "checkpoint_sha256",
                     "execution_receipt_path", "execution_receipt_sha256"}
    execution_path = Path(probe.get("execution_receipt_path", ""))
    expected_paths = UR.probe_paths(f"{tag}_scope", evidence["ctx"].cases["FAULT"], "P_reward", SOURCE_RESULTS)
    if execution_path.resolve() != expected_paths["execution"].resolve():
        raise ValueError("scope P_reward execution receipt path differs")
    execution_meta = E.read(execution_path)
    report_path = expected_paths["report"]
    report = E.read(report_path)
    report_probe = report.get("probes", {}).get("P_reward", {})
    expected_meta = {"schema": "probe_execution_v1", "case": evidence["ctx"].cases["FAULT"],
                     "probe": "P_reward", "checkpoint_sha256": probe["checkpoint_sha256"],
                     "report_sha256_lf": E.sha(report_path, True), "exit_code": 0,
                     "upstream_readback_ok": True}
    if (set(probe) != expected_keys or probe["schema"] != "u1_scope_probe_receipt_v1"
            or probe["case"] != evidence["ctx"].cases["FAULT"] or probe["probe"] != "P_reward"
            or probe["exit_code"] != 0
            or probe["checkpoint_sha256"] != evidence["artifacts"]["FAULT"]["checkpoint"]["sha256"]
            or execution_path.is_symlink() or probe["execution_receipt_sha256"] != E.sha(execution_path)
            or any(execution_meta.get(key) != value for key, value in expected_meta.items())
            or report.get("name") != evidence["ctx"].cases["FAULT"]
            or report.get("checkpoint", {}).get("sha256") != probe["checkpoint_sha256"]
            or report_probe.get("exit") != 0 or report_probe.get("measurement") != probe["measurement"]
            or L.classify("P_reward", probe["measurement"],
                          evidence["reference"]["probes"]["P_reward"]["measurement"]) != probe["outcome"]
            or probe["outcome"] != "abnormal"):
        raise ValueError("scope P_reward receipt differs from source evidence")
    return scope


def _private_fixed_evaluate(tag: str, isaaclab: Path) -> dict:
    ctx = E.ctx_for(tag)
    previous = E._guard(tag, "evaluate")
    protocol, commit = E.verify_for(tag, isaaclab)
    artifacts = {arm: E._check_artifacts(tag, arm) for arm in ("NONE", "FAULT")}
    gpu = E.V2._admit_gpu(protocol)
    folder = E.folder_for(tag)
    jobs_path = folder / "jobs/fixed_eval.json"
    outputs = {arm: SOURCE_RESULTS / tag / "runs" / f"{ctx.cases[arm]}.json" for arm in ("NONE", "FAULT")}
    if jobs_path.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError("private fixed-evaluation jobs or outputs already exist")
    E.write_new(jobs_path, {"action_scale": protocol["action_scale"], "jobs": [
        {"name": ctx.cases[arm], "checkpoint": artifacts[arm]["checkpoint"]["path"], "output": str(outputs[arm])}
        for arm in ("NONE", "FAULT")]})
    argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()), str((ROOT / "evals/fixed_eval.py").resolve()),
            "--protocol", str((ROOT / "bench/protocols/fixed_eval_v1.json").resolve()),
            "--jobs", str(jobs_path.resolve()), "--headless"]
    execution, attempt = E._execute(tag, "evaluate", argv, protocol["watchdogs_s"]["fixed_eval"])
    if not E.V3.execution_valid(execution) or not all(path.is_file() for path in outputs.values()):
        E._gpu_step_failed(tag, "evaluate", execution, attempt, any(path.exists() for path in outputs.values()))
    reports, paths = {}, {}
    for arm, path in outputs.items():
        reports[arm] = _validate_report(path, ctx.cases[arm], artifacts[arm]["checkpoint"]["sha256"],
                                        protocol["action_scale"], protocol["fixed_eval_sha256_lf"])
        paths[arm] = {"path": E._rel(path), "sha256_lf": E.sha(path, True)}
    gate = E.source_gate(reports["NONE"], reports["FAULT"], protocol["bands"])
    record = E._write_step(tag, "evaluate", previous, commit, ctx.protocol,
                           {"gpu_before": gpu, "execution": execution, "reports": paths, "gate": gate})
    if gate["outcome"] != "failure_present":
        E._terminal(tag, gate["outcome"], {"gate": gate})
    return record


def _private_reference(tag: str, isaaclab: Path) -> dict:
    ctx = E.ctx_for(tag)
    previous = E._guard(tag, "reference")
    protocol, commit = E.verify_for(tag, isaaclab)
    checkpoint = E._check_artifacts(tag, "NONE")["checkpoint"]
    folder = E.folder_for(tag)
    output = SOURCE_RESULTS / f"{tag}_ref" / "probes" / f"{ctx.cases['NONE']}.json"
    jobs_path = folder / "jobs/reference_probes.json"
    if jobs_path.exists() or output.exists():
        raise FileExistsError("private reference jobs or output already exist")
    E.write_new(jobs_path, {"fault": "NONE", "seed": ctx.seed, "jobs": [{
        "name": ctx.cases["NONE"], "checkpoint": checkpoint["path"], "probes": sorted(L.PROBES), "output": str(output)}]})
    gpu = E.V2._admit_gpu(protocol)
    argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()), str((ROOT / "evals/probes.py").resolve()),
            "--jobs", str(jobs_path.resolve()), "--headless"]
    execution, attempt = E._execute(tag, "reference", argv, protocol["watchdogs_s"]["reference_probes"])
    if not E.V3.execution_valid(execution) or not output.is_file():
        E._gpu_step_failed(tag, "reference", execution, attempt, output.exists())
    report = E.read(output)
    probes = report.get("probes", {})
    if (report.get("name") != ctx.cases["NONE"] or report.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]
            or report.get("seed") != ctx.seed or set(probes) != set(L.PROBES)
            or any(value.get("exit") != 0 or value.get("measurement") is None for value in probes.values())):
        raise ValueError("NONE probe reference did not produce six complete measurements")
    return E._write_step(tag, "reference", previous, commit, ctx.protocol, {
        "gpu_before": gpu, "execution": execution, "report": {"path": E._rel(output), "sha256_lf": E.sha(output, True)}})


def diagnose_scope(tag: str, isaaclab: Path) -> dict:
    configure()
    evidence = source_evidence(tag, isaaclab)
    ctx = evidence["ctx"]
    probe_result = UR.probe(isaaclab, fault="U1", seed=ctx.seed, run_name=ctx.cases["FAULT"], probe="P_reward",
                            tag=f"{tag}_scope", work_dir=E.folder_for(tag) / "exec/scope_probe",
                            results=SOURCE_RESULTS, watchdog_s=600)
    measurement, gpu_s, exit_code = (probe_result["measurement"], probe_result["gpu_s"], probe_result["exit_code"])
    outcome = L.classify("P_reward", measurement if exit_code == 0 else None,
                         evidence["reference"]["probes"]["P_reward"]["measurement"])
    if outcome != "abnormal":
        raise ValueError("P_reward did not confirm the new-seed reward failure")
    probe_path = E.folder_for(tag) / "scope_probe.json"
    E.write_new(probe_path, {"schema": "u1_scope_probe_receipt_v1", "case": ctx.cases["FAULT"],
                              "probe": "P_reward", "measurement": measurement, "outcome": outcome,
                              "exit_code": exit_code, "gpu_s": gpu_s,
                              "checkpoint_sha256": evidence["artifacts"]["FAULT"]["checkpoint"]["sha256"],
                              "execution_receipt_path": _canonical(Path(probe_result["paths"]["execution"])),
                              "execution_receipt_sha256": E.sha(Path(probe_result["paths"]["execution"]))})
    row = {
        "schema": "u1_scope_extension_scope_v1", "fault": "U1", "seed": 2051,
        "fault_family": "root_body_velocity_reward_family", "diagnosis": "known_u1_reward_family",
        "observed_probe_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"],
        "source_registered_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"],
        "result": {"verdict": "scope_insufficient", "uncovered_terms": [
            "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]},
        "bindings": _scope_bindings(tag, isaaclab, probe_path),
        "design_coupling": "기존 U1 결과를 본 뒤 만든 개발 등록이며 P_reward 관측 두 항과 소스 확인 네 항을 구분한다",
    }
    path = E.folder_for(tag) / "scope.json"
    E.write_new(path, row)
    validate_scope_record(path, isaaclab)
    return {"scope": str(path), "scope_sha256": E.sha(path), "probe": str(probe_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "begin", "train", "alarm", "evaluate", "reference",
                                            "diagnose-scope", "status"))
    parser.add_argument("--tag")
    parser.add_argument("--arm", choices=("NONE", "FAULT"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    configure()
    if args.command == "prepare":
        result = E.prepare("u1", "single", args.isaaclab)
    elif args.command == "begin":
        result = E.begin(args.tag, "u1", args.isaaclab)
    elif args.command == "train":
        result = E.train(args.tag, args.arm, args.isaaclab)
    elif args.command == "alarm":
        result = E.alarm(args.tag, args.isaaclab)
    elif args.command == "evaluate":
        result = _private_fixed_evaluate(args.tag, args.isaaclab)
    elif args.command == "reference":
        result = _private_reference(args.tag, args.isaaclab)
    elif args.command == "diagnose-scope":
        result = diagnose_scope(args.tag, args.isaaclab)
    else:
        result = E.status(args.tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
