"""교정 후 typed 관측을 기존 lineage probe 실행 경계에 연결한다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_retry_v1 as RETRY
import lineage_gpu_wait_v1 as GPU_WAIT
import lineage_lease_context_v2 as LEASE
import lineage_post_correction_inputs_v1 as POST
import lineage_probe_backend_v1 as P1

SCHEMA = "lineage_post_probe_plan_v1"
RECEIPT_SCHEMA = "lineage_post_probe_receipt_v1"
AUTH_SCHEMA = P1.AUTH_SCHEMA
PROTOCOL_SCHEMA = "lineage_post_probe_backend_protocol_v1"
LEGACY_PROBES = P1.LEGACY_PROBES
OPTIMIZER_PROBE = P1.OPTIMIZER_PROBE
PROBES = P1.PROBES
PROTOCOL_FIELDS = {"schema", "component_probe_protocol", "source_sha256_lf", "scope"}
SOURCE_FILES = (
    "evals/lineage_post_probe_backend_v1.py",
    "tests/test_lineage_post_probe_backend_v1.py",
    "evals/lineage_post_correction_inputs_v1.py",
    "tests/test_lineage_post_correction_inputs_v1.py",
    "evals/lineage_compound_retry_v1.py",
    "evals/lineage_lease_context_v2.py",
)
_LOCK = RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("component protocol은 저장소 파일이어야 한다") from exc


def prepare_protocol(base_probe_protocol: Path) -> dict:
    """기존 probe protocol과 typed 후속 입력 closure를 묶는다."""
    base = Path(base_probe_protocol).resolve()
    value = P1.read(base)
    require(value.get("schema") == P1.PROTOCOL_SCHEMA, "기존 probe protocol schema가 다르다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "component_probe_protocol": {"path": _relative(base), "sha256_lf": sha(base, lf=True)},
        "source_sha256_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "scope": "epoch2_post_correction_typed_input_only_no_approval_or_execution_authority",
    }


def verify_protocol(protocol_path: Path, lab: Path, anchor: str) -> dict:
    path = Path(protocol_path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "protocol은 저장소 파일이어야 한다")
    value = read(path)
    require(set(value) == PROTOCOL_FIELDS and value.get("schema") == PROTOCOL_SCHEMA,
            "post probe protocol 필드 또는 schema가 다르다")
    component = value.get("component_probe_protocol")
    if not isinstance(component, dict) or set(component) != {"path", "sha256_lf"}:
        raise ValueError("기존 probe protocol 결합이 없다")
    base_path = ROOT / str(component["path"])
    require(base_path.is_file() and sha(base_path, lf=True) == component["sha256_lf"],
            "기존 probe protocol 원문이 바뀌었다")
    P1.verify_protocol(base_path, Path(lab), anchor)
    expected = {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES}
    require(value.get("source_sha256_lf") == expected, "post probe source closure가 바뀌었다")
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "registration anchor")
    for relative, expected_sha in expected.items():
        committed = P1._git("show", f"{anchor}:{relative}").stdout.encode("utf-8")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(committed).hexdigest() == expected_sha,
                f"anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def _lease_wait(stage: Path) -> dict:
    counter = 0

    def lease_check() -> dict:
        nonlocal counter
        counter += 1
        if getattr(LEASE._LOCAL, "events", None) is not None:
            return P1.B.lease_context()
        evidence = Path(stage) / "lease-reads" / f"{counter:04d}.json"
        with RETRY.lease_read_scope(evidence):
            return P1.B.lease_context()

    return GPU_WAIT.await_clear(
        stage, max_wait_s=600, interval_s=2, consecutive_clear=2, lease_check=lease_check)


def _failure(path: Path, phase: str, error: BaseException) -> None:
    filename = getattr(error, "filename", None)
    value = {
        "schema": "lineage_post_probe_failure_v1", "status": "blocked", "phase": phase,
        "error_type": type(error).__name__, "error_message": str(error),
        "filename": None if filename is None else str(filename),
        "errno": getattr(error, "errno", None), "traceback": traceback.format_exc(),
        "post_block_roles_started": 0,
    }
    if not path.exists():
        write_new(path, value)


def _admit(stage: Path, phase: str) -> dict:
    root = Path(stage).resolve().parent / f"{Path(stage).name}-admission-before-{phase}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _lease_wait(root)
    except BaseException as error:
        _failure(root / "failure.json", phase, error)
        raise


def _case_source(path: Path, lab: Path) -> tuple[dict, dict, dict]:
    observation, _telemetry = POST.checked_observation(path)
    sources = observation["private_metadata"]["sources"]
    receipt = read(Path(sources["repair_receipt"]))
    recovery = receipt["collected_recovery"]
    checkpoint = receipt["new_checkpoint"]
    require(Path(checkpoint["path"]).is_file() and sha(Path(checkpoint["path"]))
            == checkpoint["sha256"] == observation["source_binding"]["checkpoint_sha256"],
            "post-correction checkpoint 결합이 다르다")
    require(Path(sources["lab"]).resolve() == Path(lab).resolve(),
            "post-correction Isaac Lab 결합이 다르다")
    private = observation["private_metadata"]["host_only"]
    source = {
        "kind": "post_correction", "seed": receipt["seed"], "source_arm": "FIRST",
        "checkpoint": deepcopy(checkpoint),
        "telemetry_sha256": observation["source_binding"]["telemetry_sha256"],
        "original_faults": deepcopy(private["original_faults"]),
        "remaining_faults": deepcopy(private["remaining_faults"]),
        "applied_corrections": deepcopy(private["applied_correction_ids"]),
        "readback": private["readback"],
        "prior_not_recovered_receipt_sha256": observation["source_binding"][
            "repair_receipt_sha256"],
        "source_manifest_sha256": recovery["source_binding"]["manifest_sha256"],
    }
    return observation, receipt, source


def _reference_source(path: Path, lab: Path) -> tuple[dict, dict, dict]:
    collected, manifest, evidence = P1._source(path, lab)
    require(manifest["arm"] == "NONE", "reference 입력은 검증된 NONE source여야 한다")
    source = {
        "kind": "none_reference", "seed": manifest["seed"], "source_arm": "NONE",
        "checkpoint": deepcopy(evidence["checkpoint"]),
        "telemetry_sha256": collected["source_binding"]["telemetry_sha256"],
        "original_faults": [], "remaining_faults": [], "applied_corrections": [],
        "readback": None, "prior_not_recovered_receipt_sha256": None,
        "source_manifest_sha256": sha(Path(collected["private_source_map"]["manifest"])),
    }
    return collected, manifest, source


def _effective_fault(source: dict) -> str:
    remaining = source["remaining_faults"]
    require(isinstance(remaining, list) and set(remaining) <= {"F1", "R1"},
            "등록되지 않은 잔여 결함은 legacy probe에 투영할 수 없다")
    return "F1" if "F1" in remaining else "NONE"


def _residual_manifest(protocol: dict, source: dict, stage: Path, anchor: str,
                       protocol_path: Path) -> dict:
    arm = "FIRST" if source["kind"] == "post_correction" else "NONE"
    value = {
        "schema": P1.RESIDUAL.SCHEMA, "mode": "inspection", "arm": arm,
        "seed": source["seed"], "run_name": f"lineage_post_probe_s{source['seed']}_optimizer",
        "task": P1.RESIDUAL.TASK, **P1.RESIDUAL.FIXED, "max_iterations": 3,
        "source_commit_sha": anchor,
        "protocol_path": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((stage / "records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "original_faults": source["original_faults"],
        "applied_corrections": source["applied_corrections"],
    }
    return P1.RESIDUAL.validate_manifest(value)


def prepare_probe(collected_input: Path, probe: str, *, stage: Path, lab: Path,
                  protocol_path: Path, anchor: str, role: str = "case") -> dict:
    require(role in {"case", "reference"}, "probe role은 case 또는 reference여야 한다")
    require(probe in PROBES and probe in P1.C.RC.registry()["probes"], "등록 probe가 아니다")
    stage, lab = Path(stage).resolve(), Path(lab).resolve()
    require(not stage.exists(), "probe stage는 항상 새 경로여야 한다")
    wrapper = verify_protocol(Path(protocol_path), lab, anchor)
    component = ROOT / wrapper["component_probe_protocol"]["path"]
    base = P1.verify_protocol(component, lab, anchor)
    if role == "case":
        _value, _receipt, source = _case_source(Path(collected_input), lab)
    else:
        _value, _manifest, source = _reference_source(Path(collected_input), lab)
    _admit(stage, "prepare")
    stage.mkdir(parents=True, exist_ok=False)
    checkpoint = source["checkpoint"]
    auxiliary = P1._auxiliary_inputs(probe, checkpoint)
    run_name = f"lineage_post_probe_{role}_s{source['seed']}_{probe.lower()}"
    python = (lab / base["isaac_python_relative"]).resolve()
    if probe in LEGACY_PROBES:
        report = stage / "report.json"
        fault = _effective_fault(source) if role == "case" else "NONE"
        jobs = {"fault": fault, "seed": source["seed"], "task": P1.FROZEN.TASK,
                "jobs": [{"name": run_name, "checkpoint": checkpoint["path"],
                          "probes": [probe], "output": str(report)}]}
        write_new(stage / "jobs.json", jobs)
        command = [str(python), str((ROOT / "evals/probes.py").resolve()),
                   "--jobs", str((stage / "jobs.json").resolve()), "--headless"]
        projection = {"backend": "frozen_legacy_probes", "effective_fault": fault,
                      "already_repaired_fault_reintroduced": False}
        raw_files = {"jobs": str((stage / "jobs.json").resolve()), "report": str(report)}
        watchdog = base["watchdogs_s"]["legacy"]
    else:
        inspection = _residual_manifest(base, source, stage, anchor, component)
        write_new(stage / "inspection.manifest.json", inspection)
        command = [str(python), str((ROOT / "bench/residual_training_v1.py").resolve()),
                   "--manifest", str((stage / "inspection.manifest.json").resolve())]
        projection = {"backend": "optimizer_observability_3_update",
                      "inspection_arm": inspection["arm"],
                      "remaining_faults": deepcopy(source["remaining_faults"]),
                      "checkpoint_loaded": False}
        raw_files = {"manifest": str((stage / "inspection.manifest.json").resolve()),
                     "records": inspection["records_path"],
                     "readback": inspection["readback_path"]}
        watchdog = base["watchdogs_s"]["optimizer"]
    plan = {
        "schema": SCHEMA, "role": role, "probe": probe, "stage": str(stage),
        "lab": str(lab), "run_name": run_name, "seed": source["seed"],
        "source_arm": source["source_arm"], "source_kind": source["kind"],
        "checkpoint": {"path": str(Path(checkpoint["path"]).resolve()),
                       "sha256": checkpoint["sha256"]},
        "auxiliary_inputs": auxiliary, "telemetry_sha256": source["telemetry_sha256"],
        "collected_input": {"path": str(Path(collected_input).resolve()),
                            "sha256": sha(Path(collected_input))},
        "source_manifest_sha256": source["source_manifest_sha256"],
        "protocol": {"path": str(Path(protocol_path).resolve()),
                     "sha256_lf": sha(Path(protocol_path), lf=True), "anchor": anchor},
        "component_protocol": {"path": str(component), "sha256_lf": sha(component, lf=True)},
        "source_sha256_lf": base["source_sha256_lf"],
        "runtime_source_sha256_lf": base["runtime_source_sha256_lf"],
        "command": command, "watchdog_s": watchdog, "projection": projection,
        "raw_files": raw_files,
        "private_lineage": {
            "original_faults": source["original_faults"],
            "applied_corrections": source["applied_corrections"],
            "remaining_faults": source["remaining_faults"], "readback": source["readback"],
            "prior_valid_not_recovered_receipt_sha256": source[
                "prior_not_recovered_receipt_sha256"],
        },
        "scope": {"approval_created": False, "ledger_event_created": False,
                  "paired_single_probe_role": role, "source_purpose_fabricated": False},
    }
    write_new(stage / "plan.json", plan)
    return deepcopy(plan)


def _verify_plan(plan_path: Path) -> dict:
    path = Path(plan_path).resolve()
    plan = read(path)
    require(plan.get("schema") == SCHEMA and path == Path(plan.get("stage", "")) / "plan.json",
            "post probe plan schema 또는 stage 위치가 다르다")
    wrapper = verify_protocol(Path(plan["protocol"]["path"]), Path(plan["lab"]),
                              plan["protocol"]["anchor"])
    component = ROOT / wrapper["component_probe_protocol"]["path"]
    base = P1.verify_protocol(component, Path(plan["lab"]), plan["protocol"]["anchor"])
    require(plan["protocol"]["sha256_lf"] == sha(Path(plan["protocol"]["path"]), lf=True),
            "post probe protocol SHA가 다르다")
    require(plan.get("component_protocol") == {
                "path": str(component), "sha256_lf": sha(component, lf=True)}
            and plan.get("source_sha256_lf") == base["source_sha256_lf"]
            and plan.get("runtime_source_sha256_lf") == base["runtime_source_sha256_lf"],
            "post probe component/source/runtime 결합이 다르다")
    source_path = Path(plan["collected_input"]["path"])
    require(source_path.is_file() and sha(source_path) == plan["collected_input"]["sha256"],
            "post probe collected input이 바뀌었다")
    if plan["role"] == "case":
        _value, _receipt, source = _case_source(source_path, Path(plan["lab"]))
    else:
        _value, _manifest, source = _reference_source(source_path, Path(plan["lab"]))
    require(plan.get("role") in {"case", "reference"}
            and plan.get("source_kind") == source["kind"]
            and plan.get("source_arm") == source["source_arm"]
            and plan.get("seed") == source["seed"]
            and plan.get("source_manifest_sha256") == source["source_manifest_sha256"]
            and plan["checkpoint"]["sha256"] == source["checkpoint"]["sha256"]
            and sha(Path(plan["checkpoint"]["path"])) == plan["checkpoint"]["sha256"]
            and plan["telemetry_sha256"] == source["telemetry_sha256"]
            and plan["private_lineage"] == {
                "original_faults": source["original_faults"],
                "applied_corrections": source["applied_corrections"],
                "remaining_faults": source["remaining_faults"], "readback": source["readback"],
                "prior_valid_not_recovered_receipt_sha256": source[
                    "prior_not_recovered_receipt_sha256"]},
            "post probe checkpoint/telemetry/private lineage가 바뀌었다")
    _verify_auxiliary_inputs(plan)
    return plan


def _verify_auxiliary_inputs(plan: dict) -> None:
    expected = P1._auxiliary_inputs(plan["probe"], plan["checkpoint"])
    require(plan.get("auxiliary_inputs") == expected, "post probe 보조 입력 결합이 다르다")


def _legacy_measurement(plan: dict) -> dict:
    jobs = read(Path(plan["raw_files"]["jobs"]))
    report = read(Path(plan["raw_files"]["report"]))
    fault = plan["projection"]["effective_fault"]
    expected = {"fault": fault, "seed": plan["seed"], "task": P1.FROZEN.TASK,
                "jobs": [{"name": plan["run_name"], "checkpoint": plan["checkpoint"]["path"],
                          "probes": [plan["probe"]], "output": plan["raw_files"]["report"]}]}
    require(jobs == expected, "post frozen probes jobs가 plan과 다르다")
    artifact = {"checkpoint": plan["checkpoint"], "seed": plan["seed"]}
    row = report.get("probes", {}).get(plan["probe"], {})
    return P1.FROZEN.validate_probe_result(
        artifact, plan["probe"], plan["run_name"],
        {"report": report, "measurement": row.get("measurement")})


_optimizer_measurement = P1._optimizer_measurement
_execution_ok = P1._execution_ok


@contextmanager
def _backend_scope() -> Iterator[None]:
    with _LOCK:
        originals = (P1._verify_plan, P1._verify_auxiliary_inputs,
                     P1._legacy_measurement, P1.RECEIPT_SCHEMA)
        P1._verify_plan = _verify_plan
        P1._verify_auxiliary_inputs = _verify_auxiliary_inputs
        P1._legacy_measurement = _legacy_measurement
        P1.RECEIPT_SCHEMA = RECEIPT_SCHEMA
        try:
            yield
        finally:
            (P1._verify_plan, P1._verify_auxiliary_inputs,
             P1._legacy_measurement, P1.RECEIPT_SCHEMA) = originals


def _lease_reader(stage: Path) -> Callable[[], dict]:
    counter = 0

    def read_lease() -> dict:
        nonlocal counter
        counter += 1
        if getattr(LEASE._LOCAL, "events", None) is not None:
            return P1.B.lease_context()
        evidence = Path(stage) / "lease-execution" / f"{counter:04d}.json"
        with RETRY.lease_read_scope(evidence):
            return P1.B.lease_context()

    return read_lease


def execute_probe(plan: Path, authorization: Path, *, owned: Callable | None = None,
                  check_lease: Callable | None = None,
                  check_gpu: Callable | None = None) -> dict:
    plan_path = Path(plan).resolve()
    value = _verify_plan(plan_path)
    stage = Path(value["stage"])
    _admit(stage, "claim")
    lease = check_lease or _lease_reader(stage)
    try:
        with _backend_scope():
            receipt = P1.execute_probe(plan_path, Path(authorization), owned=owned,
                                       check_lease=lease, check_gpu=check_gpu)
        require(receipt.get("schema") == RECEIPT_SCHEMA, "post probe receipt schema가 다르다")
        return receipt
    except BaseException as error:
        _failure(stage / "post-probe-failure.json", "execute", error)
        raise


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(P1, name)
