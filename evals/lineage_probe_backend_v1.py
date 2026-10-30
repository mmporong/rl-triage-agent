"""검증된 lineage source를 동결 probe GPU child에 연결하는 fail-closed backend.

이 모듈은 승인이나 원장 event를 만들지 않는다. 실행에는 호스트가 미리 만든
``lineage_probe_execution_authorization_v1``과 running 상태 원장이 필요하다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import brev_recovery_backend_v4 as FROZEN
import lineage_selection_inputs_v1 as INPUT
import lineage_source_batch_v1 as B
import lineage_source_training_v1 as SOURCE
import optimizer_observability as OPTIMIZER
import residual_training_v1 as RESIDUAL
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_probe_plan_v1"
RECEIPT_SCHEMA = "lineage_probe_receipt_v1"
AUTH_SCHEMA = "lineage_probe_execution_authorization_v1"
PROTOCOL_SCHEMA = "lineage_probe_backend_protocol_v1"
LEGACY_PROBES = tuple(FROZEN.LEGACY_PROBES)
OPTIMIZER_PROBE = FROZEN.OPTIMIZER_PROBE
PROBES = (*LEGACY_PROBES, OPTIMIZER_PROBE)
SOURCE_ARMS = ("NONE", "F1", "F2", "F3", "F5", "COMPOUND")
LEGACY_FAULT = {arm: ("F1" if arm == "COMPOUND" else arm) for arm in SOURCE_ARMS}
OPTIMIZER_ARM = {"COMPOUND": "COMPOUND", "F1": "F1", "NONE": "NONE",
                 "F2": "NONE", "F3": "NONE", "F5": "NONE"}
OPTIMIZER_FAULTS = {"COMPOUND": ["F1", "R1"], "F1": ["F1"], "NONE": [],
                    "F2": [], "F3": [], "F5": []}
PROTOCOL_FIELDS = {
    "schema", "source_sha256_lf", "runtime_source_sha256_lf",
    "watchdogs_s", "isaac_python_relative", "scope",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check, capture_output=True,
                          text=True, encoding="utf-8", errors="strict")


def _relative(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not Path(value).is_absolute() and ".." not in Path(value).parts)


def _hash_map(value: object, name: str) -> dict[str, str]:
    require(isinstance(value, dict) and bool(value), f"{name}이 비었다")
    require(all(_relative(path) and re.fullmatch(r"[0-9a-f]{64}", str(item))
                for path, item in value.items()), f"{name} 형식")
    return dict(value)


def verify_protocol(protocol_path: Path, lab: Path, anchor: str) -> dict:
    protocol_path, lab = Path(protocol_path).resolve(), Path(lab).resolve()
    require(protocol_path.is_relative_to(ROOT) and protocol_path.is_file(), "protocol은 저장소 파일이어야 한다")
    value = read(protocol_path)
    require(set(value) == PROTOCOL_FIELDS and value.get("schema") == PROTOCOL_SCHEMA,
            "probe protocol 필드 또는 schema")
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "registration anchor")
    source = _hash_map(value["source_sha256_lf"], "source_sha256_lf")
    runtime = _hash_map(value["runtime_source_sha256_lf"], "runtime_source_sha256_lf")
    required = {"evals/lineage_probe_backend_v1.py", "evals/probes.py", "bench/hidden_faults.py",
                *RESIDUAL.MINIMUM_BOUND_SOURCES}
    require(required <= set(source), "probe source closure 누락")
    watchdogs = value.get("watchdogs_s")
    require(isinstance(watchdogs, dict) and set(watchdogs) == {"legacy", "optimizer"}
            and all(type(watchdogs[key]) is int and watchdogs[key] > 0 for key in watchdogs), "watchdog 계약")
    require(_relative(value.get("isaac_python_relative")) and isinstance(value.get("scope"), str)
            and bool(value["scope"]), "Isaac Python 또는 scope")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    require(_git("merge-base", "--is-ancestor", anchor, "HEAD", check=False).returncode == 0,
            "anchor가 현재 HEAD 조상이 아니다")
    relative_protocol = protocol_path.relative_to(ROOT).as_posix()
    bindings = {relative_protocol: sha(protocol_path, lf=True), **source}
    for relative, expected in bindings.items():
        path = ROOT / relative
        require(path.is_file() and sha(path, lf=True) == expected, f"현재 source가 다르다: {relative}")
        blob = _git("show", f"{anchor}:{relative}").stdout.encode("utf-8").replace(b"\r\n", b"\n")
        require(hashlib.sha256(blob).hexdigest() == expected, f"anchor blob이 다르다: {relative}")
    for relative, expected in runtime.items():
        path = lab / relative
        require(path.is_file() and sha(path, lf=True) == expected, f"runtime source가 다르다: {relative}")
    python = lab / value["isaac_python_relative"]
    require(python.is_file(), "Isaac Python이 없다")
    return deepcopy(value)


def _source(collected_input: Path, lab: Path | None = None) -> tuple[dict, dict, dict]:
    collected, _telemetry = INPUT._checked_collected(Path(collected_input))
    source_map = collected["private_source_map"]
    stage = Path(source_map["stage"])
    manifest = SOURCE.validate_manifest(read(Path(source_map["manifest"])))
    if lab is not None:
        SOURCE.verify_binding(manifest, repo=ROOT, lab=Path(lab).resolve())
    evidence = B.completed(stage, manifest, read(Path(source_map["owned_result"])))
    require(manifest["purpose"] == "source" and manifest["arm"] in SOURCE_ARMS,
            "probe source arm 또는 purpose")
    checkpoint = evidence["checkpoint"]
    require(Path(checkpoint["path"]).resolve() == Path(source_map["checkpoint"]).resolve()
            and checkpoint["sha256"] == collected["source_binding"]["checkpoint_sha256"]
            and sha(Path(checkpoint["path"])) == checkpoint["sha256"], "source checkpoint 결합")
    return collected, manifest, evidence


def _residual_manifest(protocol: dict, manifest: dict, evidence: dict, stage: Path,
                       anchor: str, protocol_path: Path) -> dict:
    arm = OPTIMIZER_ARM[manifest["arm"]]
    job_token = re.sub(r"[^A-Za-z0-9_]", "_", manifest["job_id"])
    value = {
        "schema": RESIDUAL.SCHEMA, "mode": "inspection", "arm": arm,
        "seed": manifest["seed"], "run_name": f"lineage_probe_{job_token}_optimizer",
        "task": RESIDUAL.TASK, **RESIDUAL.FIXED, "max_iterations": 3,
        "source_commit_sha": anchor,
        "protocol_path": protocol_path.resolve().relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((stage / "records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": evidence["checkpoint"]["sha256"],
        "original_faults": OPTIMIZER_FAULTS[manifest["arm"]], "applied_corrections": [],
    }
    return RESIDUAL.validate_manifest(value)


def _auxiliary_inputs(probe: str, checkpoint: dict) -> list[dict]:
    """동결 probe가 최종 checkpoint 외에 읽는 source 파일을 결합한다."""
    if probe != "P_value":
        return []
    args = C.RC.registry()["probes"][probe]["args"]
    source_name = args.get("from")
    require(isinstance(source_name, str) and source_name not in {"", ".", ".."}
            and Path(source_name).name == source_name and len(Path(source_name).parts) == 1,
            "P_value from은 checkpoint와 같은 디렉터리의 파일명이어야 한다")
    checkpoint_path = Path(checkpoint["path"]).resolve()
    source_path = (checkpoint_path.parent / source_name).resolve()
    require(source_path.parent == checkpoint_path.parent, "P_value 보조 checkpoint 디렉터리")
    require(source_path != checkpoint_path, "P_value 최종 checkpoint와 보조 checkpoint가 같다")
    require(source_path.is_file(), "P_value 보조 checkpoint가 없다")
    return [{"role": "P_value:from", "path": str(source_path),
             "sha256": sha(source_path), "size_bytes": source_path.stat().st_size}]


def _verify_auxiliary_inputs(plan: dict) -> None:
    expected = _auxiliary_inputs(plan["probe"], plan["checkpoint"])
    require(plan.get("auxiliary_inputs") == expected, "probe 보조 입력 결합")


def prepare_probe(collected_input: Path, probe: str, *, stage: Path, lab: Path,
                  protocol_path: Path, anchor: str, role: str = "case") -> dict:
    """검증된 source에서 paired case/reference 단일 probe 실행 계획을 만든다."""
    require(role in {"case", "reference"}, "probe role은 case 또는 reference여야 한다")
    require(probe in PROBES and probe in C.RC.registry()["probes"], "등록 probe가 아니다")
    stage, lab, protocol_path = Path(stage).resolve(), Path(lab).resolve(), Path(protocol_path).resolve()
    require(not stage.exists(), "probe stage는 항상 새 경로여야 한다")
    protocol = verify_protocol(protocol_path, lab, anchor)
    collected, manifest, evidence = _source(Path(collected_input), lab)
    require(role != "reference" or manifest["arm"] == "NONE",
            "reference 단일 probe는 검증된 NONE source만 허용한다")
    checkpoint = evidence["checkpoint"]
    auxiliary_inputs = _auxiliary_inputs(probe, checkpoint)
    job_token = re.sub(r"[^A-Za-z0-9_]", "_", manifest["job_id"])
    run_name = f"lineage_probe_{job_token}_{probe.lower()}"
    python = (lab / protocol["isaac_python_relative"]).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    if probe in LEGACY_PROBES:
        report_path = stage / "report.json"
        jobs = {"fault": LEGACY_FAULT[manifest["arm"]], "seed": manifest["seed"],
                "task": FROZEN.TASK, "jobs": [{"name": run_name, "checkpoint": checkpoint["path"],
                "probes": [probe], "output": str(report_path.resolve())}]}
        write_new(stage / "jobs.json", jobs)
        command = [str(python), str((ROOT / "evals/probes.py").resolve()),
                   "--jobs", str((stage / "jobs.json").resolve()), "--headless"]
        projection = {"backend": "frozen_legacy_probes", "source_arm": manifest["arm"],
                      "fault": jobs["fault"], "compound_projection": "F1_only" if manifest["arm"] == "COMPOUND" else None}
        raw_files = {"jobs": str((stage / "jobs.json").resolve()), "report": str(report_path.resolve())}
        watchdog = protocol["watchdogs_s"]["legacy"]
    else:
        inspection = _residual_manifest(protocol, manifest, evidence, stage, anchor, protocol_path)
        write_new(stage / "inspection.manifest.json", inspection)
        command = [str(python), str((ROOT / "bench/residual_training_v1.py").resolve()),
                   "--manifest", str((stage / "inspection.manifest.json").resolve())]
        projection = {"backend": "optimizer_observability_3_update", "source_arm": manifest["arm"],
                      "inspection_arm": inspection["arm"], "original_faults": inspection["original_faults"],
                      "checkpoint_loaded": False,
                      "projection_reason": "count_code_only_NONE" if manifest["arm"] in {"F2", "F3", "F5"} else "source_fault_family"}
        raw_files = {"manifest": str((stage / "inspection.manifest.json").resolve()),
                     "records": inspection["records_path"], "readback": inspection["readback_path"]}
        watchdog = protocol["watchdogs_s"]["optimizer"]
    plan = {
        "schema": SCHEMA, "role": role, "probe": probe, "stage": str(stage), "lab": str(lab),
        "run_name": run_name, "seed": manifest["seed"], "source_arm": manifest["arm"],
        "checkpoint": {"path": str(Path(checkpoint["path"]).resolve()), "sha256": checkpoint["sha256"]},
        "auxiliary_inputs": auxiliary_inputs,
        "telemetry_sha256": collected["source_binding"]["telemetry_sha256"],
        "collected_input": {"path": str(Path(collected_input).resolve()), "sha256": sha(Path(collected_input))},
        "source_manifest_sha256": sha(Path(collected["private_source_map"]["manifest"])),
        "protocol": {"path": str(protocol_path), "sha256_lf": sha(protocol_path, lf=True), "anchor": anchor},
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "command": command, "watchdog_s": watchdog, "projection": projection, "raw_files": raw_files,
        "scope": {"approval_created": False, "ledger_event_created": False,
                  "paired_single_probe_role": role, "reference6_shared_mode": "unsupported"},
    }
    write_new(stage / "plan.json", plan)
    return deepcopy(plan)


def _verify_plan(plan_path: Path) -> dict:
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    require(plan.get("schema") == SCHEMA and plan_path == Path(plan.get("stage", "")) / "plan.json",
            "plan schema 또는 stage 위치")
    protocol_path = Path(plan["protocol"]["path"])
    protocol = verify_protocol(protocol_path, Path(plan["lab"]), plan["protocol"]["anchor"])
    require(plan["protocol"]["sha256_lf"] == sha(protocol_path, lf=True)
            and plan["source_sha256_lf"] == protocol["source_sha256_lf"]
            and plan["runtime_source_sha256_lf"] == protocol["runtime_source_sha256_lf"], "plan protocol 결합")
    collected_path = Path(plan["collected_input"]["path"])
    require(sha(collected_path) == plan["collected_input"]["sha256"], "collected input 변경")
    collected, manifest, evidence = _source(collected_path, Path(plan["lab"]))
    require(plan["probe"] in PROBES and plan["role"] in {"case", "reference"}
            and (plan["role"] != "reference" or manifest["arm"] == "NONE")
            and plan["seed"] == manifest["seed"]
            and plan["source_arm"] == manifest["arm"] and plan["checkpoint"]["sha256"] == evidence["checkpoint"]["sha256"]
            and sha(Path(plan["checkpoint"]["path"])) == plan["checkpoint"]["sha256"]
            and plan["telemetry_sha256"] == collected["source_binding"]["telemetry_sha256"]
            and plan["source_manifest_sha256"] == sha(Path(collected["private_source_map"]["manifest"])),
            "plan source identity 변경")
    _verify_auxiliary_inputs(plan)
    return plan


def _authorization(path: Path, plan_path: Path, plan: dict) -> dict:
    path = Path(path).resolve()
    value = read(path)
    required = {"schema", "plan_sha256", "snapshot", "choice", "request_spec", "ledger_path", "request_id"}
    plan_binding = value.get("plan_sha256")
    plan_bound = (plan_binding == sha(plan_path)
                  or (isinstance(plan_binding, dict) and set(plan_binding) == {"case", "reference"}
                      and plan_binding.get(plan["role"]) == sha(plan_path)
                      and all(re.fullmatch(r"[0-9a-f]{64}", str(item)) for item in plan_binding.values())))
    require(set(value) == required and value.get("schema") == AUTH_SCHEMA and plan_bound,
            "실행 authorization 형식 또는 plan SHA")
    snapshot, choice = value["snapshot"], value["choice"]
    spec = C._frozen_request_spec(snapshot, choice)
    bindings = snapshot["inputs"]["packet"]["input_bindings"]
    role_binding = bindings[plan["role"]]
    expected_checkpoint = (spec["checkpoint_sha256"] if plan["role"] == "case"
                           else spec["reference_checkpoint_sha256"])
    require(spec == value["request_spec"] and spec["probe"] == plan["probe"]
            and spec["args"] == C.RC.registry()["probes"][plan["probe"]]["args"]
            and expected_checkpoint == role_binding["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
            and role_binding["telemetry_sha256"] == plan["telemetry_sha256"]
            and Path(value["ledger_path"]).resolve() == Path(snapshot["inputs"]["ledger_path"]).resolve(),
            "authorization request_spec/context/args/checkpoint 결합")
    ledger = C.S.M.RegistryLedger(Path(value["ledger_path"]), C.RC.registry())
    request = ledger.requests().get(value["request_id"], {})
    events = [row for row in ledger.events() if row.get("request_id") == value["request_id"]]
    require(request.get("state") == "running" and [row.get("event") for row in events]
            == ["proposed", "approved", "consumed"] and sum(row.get("event") == "approved" for row in events) == 1
            and sum(row.get("event") == "consumed" for row in events) == 1
            and request.get("probe") == plan["probe"] and request.get("prereg_digest") == C.L.digest(spec["prereg"])
            and request.get("args_digest") == C.L.digest(spec["args"]), "authorization 원장이 running 단회 소비 상태가 아니다")
    return value


def _claim(plan: dict, plan_path: Path, authorization: dict) -> tuple[Path, dict]:
    ledger = C.S.M.RegistryLedger(Path(authorization["ledger_path"]), C.RC.registry())
    request = ledger.requests()[authorization["request_id"]]
    namespace = digest({"request_id": authorization["request_id"],
                        "effect_key": request["effect_key"], "registry_digest": ledger.registry_digest})
    claim_path = Path(authorization["ledger_path"]).resolve().parent / ".lineage_probe_claims" / namespace / f"{plan['role']}.json"
    claim = {"schema": "lineage_probe_execution_claim_v1", "role": plan["role"],
             "request_id": authorization["request_id"], "effect_key": request["effect_key"],
             "registry_digest": ledger.registry_digest, "plan_sha256": sha(plan_path),
             "checkpoint_sha256": plan["checkpoint"]["sha256"],
             "auxiliary_inputs": deepcopy(plan["auxiliary_inputs"]),
             "telemetry_sha256": plan["telemetry_sha256"],
             "context_digest": C.L.digest(authorization["request_spec"]["prereg"]),
             "args_digest": C.L.digest(authorization["request_spec"]["args"]),
             "status": "unknown_until_parent_settlement"}
    try:
        write_new(claim_path, claim)
    except FileExistsError as error:
        raise ValueError("동일 logical request와 role의 실행 claim이 이미 존재한다") from error
    return claim_path, claim


def _execution_ok(value: dict) -> bool:
    return (value.get("exit_code") == 0 and value.get("natural_exit") is True
            and value.get("timed_out") is False and value.get("error_type") is None
            and value.get("cleanup_verified") is True and value.get("job_assigned_before_release") is True)


def _legacy_measurement(plan: dict) -> dict:
    jobs, report = read(Path(plan["raw_files"]["jobs"])), read(Path(plan["raw_files"]["report"]))
    expected = {"fault": LEGACY_FAULT[plan["source_arm"]], "seed": plan["seed"], "task": FROZEN.TASK,
                "jobs": [{"name": plan["run_name"], "checkpoint": plan["checkpoint"]["path"],
                          "probes": [plan["probe"]], "output": plan["raw_files"]["report"]}]}
    require(jobs == expected, "frozen probes jobs가 plan과 다르다")
    artifact = {"checkpoint": plan["checkpoint"], "seed": plan["seed"]}
    row = report.get("probes", {}).get(plan["probe"], {})
    measurement = row.get("measurement")
    return FROZEN.validate_probe_result(artifact, plan["probe"], plan["run_name"],
                                        {"report": report, "measurement": measurement})


def _optimizer_measurement(plan: dict) -> dict:
    manifest = RESIDUAL.validate_manifest(read(Path(plan["raw_files"]["manifest"])))
    rows = [json.loads(line) for line in Path(plan["raw_files"]["records"]).read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 3 and [row.get("update_index") for row in rows] == [0, 1, 2],
            "optimizer observed rows가 3 update가 아니다")
    measured = OPTIMIZER.assemble_inspection(rows, RESIDUAL.observer_manifest(manifest))
    readback = read(Path(plan["raw_files"]["readback"]))
    require(readback.get("observer_manifest") == RESIDUAL.observer_manifest(manifest), "optimizer readback manifest")
    return measured


def execute_probe(plan_path: Path, authorization_path: Path, *, owned: Callable | None = None,
                  check_lease: Callable = B.lease_context, check_gpu: Callable | None = None) -> dict:
    """running host authorization 하나를 소비하지 않은 채 실제 child 결과만 봉인한다."""
    plan_path = Path(plan_path).resolve()
    plan = _verify_plan(plan_path)
    authorization = _authorization(Path(authorization_path), plan_path, plan)
    stage = Path(plan["stage"])
    receipt_path = stage / "probe_receipt.json"
    require(not receipt_path.exists() and not (stage / "owned").exists(), "probe 실행 cache 또는 stage 재사용")
    _verify_auxiliary_inputs(plan)
    generated = ([Path(plan["raw_files"]["report"])] if plan["probe"] in LEGACY_PROBES else
                 [Path(plan["raw_files"][name]) for name in ("records", "readback")])
    require(not any(path.exists() for path in generated), "기존 frozen probe 출력 또는 cache 재사용")
    claim_path, claim = _claim(plan, plan_path, authorization)
    if owned is None:
        import run_runtime_readback_smoke as owned_module
        owned = owned_module.run_owned
    if check_gpu is None:
        import lineage_gpu_admission_v1 as gpu
        check_gpu = gpu.snapshot
    lease_before = check_lease()
    gpu_before = check_gpu(stage)
    error: Exception | None = None
    execution: dict = {}
    measurement: dict | None = None
    try:
        owned_stage = stage / "owned"
        owned_stage.mkdir()
        execution = dict(owned(owned_stage, list(plan["command"]), Path(plan["lab"]), plan["watchdog_s"]))
        write_new(stage / "owned-result.json", execution)
        require(_execution_ok(execution), "owned child의 exit0·자연 종료·Job 정리가 확인되지 않았다")
        _verify_auxiliary_inputs(plan)
        lease_after = check_lease()
        require(lease_after.get("leaseId") == lease_before.get("leaseId"), "probe 도중 GPU lease가 바뀌었다")
        measurement = _legacy_measurement(plan) if plan["probe"] in LEGACY_PROBES else _optimizer_measurement(plan)
    except Exception as exc:  # noqa: BLE001 - 실패 receipt를 봉인한 뒤 원예외를 다시 올린다.
        error = exc
    raw_refs = {name: {"path": str(Path(path).resolve()), "sha256": sha(Path(path))}
                for name, path in plan["raw_files"].items() if Path(path).is_file()}
    if (stage / "owned-result.json").is_file():
        raw_refs["owned_result"] = {"path": str((stage / "owned-result.json").resolve()),
                                    "sha256": sha(stage / "owned-result.json")}
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "complete" if error is None else "unknown",
        "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "authorization": {"path": str(Path(authorization_path).resolve()), "sha256": sha(Path(authorization_path)),
                          "request_id": authorization["request_id"]},
        "claim": {"path": str(claim_path), "sha256": sha(claim_path), "body": claim},
        "probe": plan["probe"], "role": plan["role"], "source_arm": plan["source_arm"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"], "seed": plan["seed"],
        "source_refs": {"collected_input": deepcopy(plan["collected_input"]),
                        "source_manifest_sha256": plan["source_manifest_sha256"],
                        "auxiliary_inputs": deepcopy(plan["auxiliary_inputs"])},
        "projection": deepcopy(plan["projection"]), "measurement": deepcopy(measurement),
        "execution": deepcopy(execution), "gpu_before": deepcopy(gpu_before), "raw_refs": raw_refs,
        "error_type": None if error is None else type(error).__name__,
        "scope": {"approval_created": False, "ledger_receipt_created": False,
                  "controller_settlement_owner": "future_host_controller"},
    }
    write_new(receipt_path, receipt)
    if error is not None:
        raise error
    return deepcopy(receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--collected-input", type=Path, required=True)
    prepare.add_argument("--probe", choices=PROBES, required=True)
    prepare.add_argument("--stage", type=Path, required=True)
    prepare.add_argument("--lab", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--anchor", required=True)
    prepare.add_argument("--role", choices=("case", "reference"), default="case")
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_probe(args.collected_input, args.probe, stage=args.stage, lab=args.lab,
                               protocol_path=args.protocol, anchor=args.anchor, role=args.role)
        print(json.dumps({"status": "prepared", "plan": str(Path(result["stage"]) / "plan.json")}))
    else:
        result = execute_probe(args.plan, args.authorization)
        print(json.dumps({"status": result["status"], "receipt": str(Path(result["plan"]["path"]).parent / "probe_receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
