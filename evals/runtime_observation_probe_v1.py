"""검증된 runtime training 입력을 기존 paired probe 실행 경계에 연결한다."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_probe_backend_v1 as POST
import lineage_probe_backend_v1 as P1
import optimizer_runtime_facts_v1 as FACT
import runtime_training_inputs_v1 as INPUT

SCHEMA = "runtime_observation_probe_plan_v1"
RECEIPT_SCHEMA = "runtime_observation_probe_receipt_v1"
AUTH_SCHEMA = P1.AUTH_SCHEMA
PROTOCOL_SCHEMA = "runtime_observation_probe_protocol_v1"
PROTOCOL_PURPOSE = "registered_training_runtime_observation_probe"
LEGACY_PROBES = P1.LEGACY_PROBES
OPTIMIZER_PROBE = P1.OPTIMIZER_PROBE
PROBES = P1.PROBES
PROTOCOL_FIELDS = {
    "schema", "purpose", "component_probe_protocol", "source_sha256_lf", "scope",
}
SOURCE_FILES = (
    "evals/runtime_observation_probe_v1.py",
    "tests/test_runtime_observation_probe_v1.py",
    "evals/runtime_training_inputs_v1.py",
    "tests/test_runtime_training_inputs_v1.py",
    "evals/optimizer_runtime_facts_v1.py",
    "tests/test_optimizer_runtime_facts_v1.py",
    "evals/lineage_post_probe_backend_v1.py",
    "tests/test_lineage_post_probe_backend_v1.py",
)
RUNTIME_PINS = {
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab/isaaclab/envs/mdp/rewards.py",
    "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
}
_LOCK = RLock()
_BASE_AUTHORIZATION = P1._authorization


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
    POST.write_new(path, value)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("protocol source는 저장소 파일이어야 한다") from exc


def prepare_protocol(base_probe_protocol: Path) -> dict[str, Any]:
    """동결 P1 protocol을 generic registered-training 입력과 결합한다."""
    base = Path(base_probe_protocol).resolve()
    value = P1.read(base)
    require(value.get("schema") == P1.PROTOCOL_SCHEMA,
            "component는 원본 lineage probe protocol이어야 한다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "purpose": PROTOCOL_PURPOSE,
        "component_probe_protocol": {"path": _relative(base), "sha256_lf": sha(base, lf=True)},
        "source_sha256_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "scope": "paired_single_probe_from_verified_registered_300_update_training_no_authority",
    }


def verify_protocol(path: Path, lab: Path, anchor: str) -> dict[str, Any]:
    """wrapper·component·source closure·9 runtime pin을 등록 commit에 대조한다."""
    path, lab = Path(path).resolve(), Path(lab).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "protocol은 저장소 파일이어야 한다")
    value = read(path)
    require(set(value) == PROTOCOL_FIELDS and value.get("schema") == PROTOCOL_SCHEMA
            and value.get("purpose") == PROTOCOL_PURPOSE,
            "runtime observation probe protocol schema/purpose가 다르다")
    component = value.get("component_probe_protocol")
    require(isinstance(component, dict) and set(component) == {"path", "sha256_lf"},
            "component probe protocol 결합이 없다")
    component = cast(dict[str, Any], component)
    base_path = ROOT / str(component["path"])
    require(value == prepare_protocol(base_path), "protocol 내용 또는 source closure가 바뀌었다")
    base = P1.verify_protocol(base_path, lab, anchor)
    require(set(base["runtime_source_sha256_lf"]) == RUNTIME_PINS,
            "component protocol의 runtime pin은 정확히 9개여야 한다")
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "registration anchor 형식")
    bindings = {_relative(path): sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, expected in bindings.items():
        current = ROOT / relative
        require(current.is_file() and sha(current, lf=True) == expected,
                f"현재 source가 다르다: {relative}")
        committed = P1._git("show", f"{anchor}:{relative}").stdout.encode("utf-8")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(committed).hexdigest() == expected,
                f"anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def _source(path: Path, lab: Path, role: str) -> tuple[dict, dict, dict]:
    require(role in {"case", "reference"}, "probe role은 case 또는 reference여야 한다")
    collected, metadata = INPUT.checked(Path(path).resolve())
    require(Path(metadata["lab"]).resolve() == Path(lab).resolve(), "collected Isaac Lab 결합이 다르다")
    private = metadata.get("private_training")
    raw = metadata.get("raw_sources")
    require(isinstance(private, dict) and isinstance(raw, dict),
            "collected private training/raw source metadata가 없다")
    private = cast(dict[str, Any], private)
    raw = cast(dict[str, Any], raw)
    checkpoint_path = Path(raw["checkpoint"]).resolve()
    checkpoint_sha = collected["source_binding"]["checkpoint_sha256"]
    require(checkpoint_path.is_file() and sha(checkpoint_path) == checkpoint_sha,
            "collected current checkpoint bytes가 다르다")
    facts_envelope, _facts_metadata = FACT.build(
        Path(metadata["manifest"]), Path(metadata["evidence_dir"]),
        Path(metadata["owned_result"]), role=role, lab=Path(metadata["lab"]),
        expected_checkpoint_sha256=checkpoint_sha)
    telemetry_path = Path(metadata["telemetry"]).resolve()
    require(telemetry_path.is_file()
            and sha(telemetry_path) == collected["source_binding"]["telemetry_sha256"],
            "collected telemetry bytes가 다르다")
    source = {
        "kind": "registered_training_observation",
        "seed": private["seed"],
        "source_arm": private["arm"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "telemetry_sha256": collected["source_binding"]["telemetry_sha256"],
        "binding_digest": collected["source_binding"]["binding_digest"],
        "source_manifest_sha256": sha(Path(metadata["manifest"])),
        "runtime_facts": facts_envelope["model_view"]["facts"],
        "original_faults": deepcopy(private["original_faults"]),
        "applied_corrections": deepcopy(private["applied_corrections"]),
        "remaining_faults": deepcopy(private["remaining_faults"]),
        "readback": str(Path(raw["readback"]).resolve()),
    }
    return collected, metadata, source


def _effective_fault(source: dict[str, Any]) -> str:
    remaining = source["remaining_faults"]
    require(isinstance(remaining, list) and set(remaining) <= {"F1", "R1"},
            "등록되지 않은 잔여 결함은 legacy probe에 투영할 수 없다")
    return "F1" if "F1" in remaining else "NONE"


def _residual_manifest(
    protocol: dict[str, Any], source: dict[str, Any], stage: Path, anchor: str,
    protocol_path: Path,
) -> dict[str, Any]:
    value = {
        "schema": P1.RESIDUAL.SCHEMA, "mode": "inspection", "arm": source["source_arm"],
        "seed": source["seed"],
        "run_name": f"runtime_observation_probe_s{source['seed']}_optimizer",
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


def prepare_probe(
    collected_input: Path, probe: str, *, stage: Path, lab: Path,
    protocol_path: Path, anchor: str, role: str,
) -> dict[str, Any]:
    """검증된 generic training 입력에서 단일 role probe 계획을 만든다."""
    require(role in {"case", "reference"}, "probe role은 case 또는 reference여야 한다")
    require(probe in PROBES and probe in P1.C.RC.registry()["probes"], "등록 probe가 아니다")
    stage, lab = Path(stage).resolve(), Path(lab).resolve()
    protocol_path = Path(protocol_path).resolve()
    require(not stage.exists(), "probe stage는 항상 새 경로여야 한다")
    wrapper = verify_protocol(protocol_path, lab, anchor)
    component = ROOT / wrapper["component_probe_protocol"]["path"]
    base = P1.verify_protocol(component, lab, anchor)
    _collected, _metadata, source = _source(Path(collected_input), lab, role)
    POST._admit(stage, "prepare")
    stage.mkdir(parents=True, exist_ok=False)
    checkpoint = source["checkpoint"]
    auxiliary = P1._auxiliary_inputs(probe, checkpoint)
    run_name = f"runtime_observation_probe_{role}_s{source['seed']}_{probe.lower()}"
    python = (lab / base["isaac_python_relative"]).resolve()
    if probe in LEGACY_PROBES:
        report = stage / "report.json"
        fault = _effective_fault(source)
        jobs = {"fault": fault, "seed": source["seed"], "task": P1.FROZEN.TASK,
                "jobs": [{"name": run_name, "checkpoint": checkpoint["path"],
                          "probes": [probe], "output": str(report)}]}
        write_new(stage / "jobs.json", jobs)
        command = [str(python), str((ROOT / "evals/probes.py").resolve()),
                   "--jobs", str((stage / "jobs.json").resolve()), "--headless"]
        projection = {"backend": "frozen_legacy_probes", "effective_fault": fault,
                      "hidden_fault_mask_exposed_to_model": False}
        raw_files = {"jobs": str((stage / "jobs.json").resolve()), "report": str(report)}
        watchdog = base["watchdogs_s"]["legacy"]
    else:
        inspection = _residual_manifest(base, source, stage, anchor, component)
        write_new(stage / "inspection.manifest.json", inspection)
        command = [str(python), str((ROOT / "bench/residual_training_v1.py").resolve()),
                   "--manifest", str((stage / "inspection.manifest.json").resolve())]
        projection = {"backend": "optimizer_observability_3_update",
                      "checkpoint_loaded": False, "hidden_fault_mask_exposed_to_model": False}
        raw_files = {"manifest": str((stage / "inspection.manifest.json").resolve()),
                     "records": inspection["records_path"],
                     "readback": inspection["readback_path"]}
        watchdog = base["watchdogs_s"]["optimizer"]
    plan = {
        "schema": SCHEMA, "role": role, "probe": probe, "stage": str(stage),
        "lab": str(lab), "run_name": run_name, "seed": source["seed"],
        "source_arm": source["source_arm"], "source_kind": source["kind"],
        "checkpoint": checkpoint, "auxiliary_inputs": auxiliary,
        "telemetry_sha256": source["telemetry_sha256"],
        "binding_digest": source["binding_digest"],
        "runtime_facts": source["runtime_facts"],
        "collected_input": {"path": str(Path(collected_input).resolve()),
                            "sha256": sha(Path(collected_input))},
        "source_manifest_sha256": source["source_manifest_sha256"],
        "protocol": {"path": str(protocol_path), "sha256_lf": sha(protocol_path, lf=True),
                     "anchor": anchor},
        "component_protocol": {"path": str(component), "sha256_lf": sha(component, lf=True)},
        "source_sha256_lf": base["source_sha256_lf"],
        "runtime_source_sha256_lf": base["runtime_source_sha256_lf"],
        "command": command, "watchdog_s": watchdog, "projection": projection,
        "raw_files": raw_files,
        "private_lineage": {
            "original_faults": source["original_faults"],
            "applied_corrections": source["applied_corrections"],
            "remaining_faults": source["remaining_faults"], "readback": source["readback"],
        },
        "scope": {"approval_created": False, "ledger_event_created": False,
                  "paired_single_probe_role": role, "source_purpose_fabricated": False,
                  "model_hidden_fault_mask_exposed": False},
    }
    write_new(stage / "plan.json", plan)
    return deepcopy(plan)


def _verify_plan(plan_path: Path) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan = read(path)
    require(plan.get("schema") == SCHEMA and path == Path(plan.get("stage", "")) / "plan.json",
            "runtime observation probe plan schema/stage가 다르다")
    wrapper = verify_protocol(Path(plan["protocol"]["path"]), Path(plan["lab"]),
                              plan["protocol"]["anchor"])
    component = ROOT / wrapper["component_probe_protocol"]["path"]
    base = P1.verify_protocol(component, Path(plan["lab"]), plan["protocol"]["anchor"])
    require(plan["protocol"]["sha256_lf"] == sha(Path(plan["protocol"]["path"]), lf=True)
            and plan.get("component_protocol") == {
                "path": str(component), "sha256_lf": sha(component, lf=True)}
            and plan.get("source_sha256_lf") == base["source_sha256_lf"]
            and plan.get("runtime_source_sha256_lf") == base["runtime_source_sha256_lf"],
            "runtime observation component/source/runtime 결합이 다르다")
    source_path = Path(plan["collected_input"]["path"])
    require(source_path.is_file() and sha(source_path) == plan["collected_input"]["sha256"],
            "runtime collected input bytes가 바뀌었다")
    _collected, _metadata, source = _source(source_path, Path(plan["lab"]), plan["role"])
    require(plan.get("role") in {"case", "reference"}
            and plan.get("source_kind") == source["kind"]
            and plan.get("source_arm") == source["source_arm"]
            and plan.get("seed") == source["seed"]
            and plan.get("source_manifest_sha256") == source["source_manifest_sha256"]
            and plan.get("binding_digest") == source["binding_digest"]
            and plan.get("runtime_facts") == source["runtime_facts"]
            and plan["checkpoint"] == source["checkpoint"]
            and sha(Path(plan["checkpoint"]["path"])) == plan["checkpoint"]["sha256"]
            and plan.get("telemetry_sha256") == source["telemetry_sha256"]
            and plan.get("private_lineage") == {
                "original_faults": source["original_faults"],
                "applied_corrections": source["applied_corrections"],
                "remaining_faults": source["remaining_faults"], "readback": source["readback"]},
            "runtime observation source/checkpoint/facts/private lineage가 바뀌었다")
    _verify_auxiliary_inputs(plan)
    return plan


def _verify_auxiliary_inputs(plan: dict[str, Any]) -> None:
    expected = P1._auxiliary_inputs(plan["probe"], plan["checkpoint"])
    require(plan.get("auxiliary_inputs") == expected, "runtime probe 보조 입력 결합이 다르다")


def _legacy_measurement(plan: dict[str, Any]) -> dict[str, Any]:
    jobs = read(Path(plan["raw_files"]["jobs"]))
    report = read(Path(plan["raw_files"]["report"]))
    fault = plan["projection"]["effective_fault"]
    expected = {"fault": fault, "seed": plan["seed"], "task": P1.FROZEN.TASK,
                "jobs": [{"name": plan["run_name"], "checkpoint": plan["checkpoint"]["path"],
                          "probes": [plan["probe"]], "output": plan["raw_files"]["report"]}]}
    require(jobs == expected, "runtime frozen probe jobs가 plan과 다르다")
    artifact = {"checkpoint": plan["checkpoint"], "seed": plan["seed"]}
    row = report.get("probes", {}).get(plan["probe"], {})
    return P1.FROZEN.validate_probe_result(
        artifact, plan["probe"], plan["run_name"],
        {"report": report, "measurement": row.get("measurement")})


def _authorization(path: Path, plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    value = _BASE_AUTHORIZATION(path, plan_path, plan)
    packet = value.get("snapshot", {}).get("inputs", {}).get("packet", {})
    snapshot_facts = packet.get("facts")
    require(isinstance(snapshot_facts, dict), "authorization snapshot runtime facts가 없다")
    prefix = f"runtime:{plan['role']}:"
    role_facts = {key: item for key, item in snapshot_facts.items() if key.startswith(prefix)}
    require(role_facts == plan["runtime_facts"],
            "authorization snapshot runtime facts가 검증 원문과 다르다")
    return value


_optimizer_measurement = P1._optimizer_measurement
_execution_ok = P1._execution_ok
classify_raw = P1.C.classify_raw


@contextmanager
def _backend_scope() -> Iterator[None]:
    with _LOCK:
        originals = (P1._verify_plan, P1._verify_auxiliary_inputs, P1._legacy_measurement,
                     P1._authorization, P1.RECEIPT_SCHEMA)
        P1._verify_plan = _verify_plan
        P1._verify_auxiliary_inputs = _verify_auxiliary_inputs
        P1._legacy_measurement = _legacy_measurement
        P1._authorization = _authorization
        P1.RECEIPT_SCHEMA = RECEIPT_SCHEMA
        try:
            yield
        finally:
            (P1._verify_plan, P1._verify_auxiliary_inputs, P1._legacy_measurement,
             P1._authorization, P1.RECEIPT_SCHEMA) = originals


def execute_probe(
    plan: Path, authorization: Path, *, owned: Callable | None = None,
    gpu_reader: Callable | None = None,
) -> dict[str, Any]:
    """기존 authorization/claim/lease/GPU/unknown 경계 안에서 probe를 실행한다."""
    plan_path = Path(plan).resolve()
    value = _verify_plan(plan_path)
    stage = Path(value["stage"])
    POST._admit(stage, "claim")
    try:
        with _backend_scope():
            receipt = P1.execute_probe(
                plan_path, Path(authorization), owned=owned,
                check_lease=POST._lease_reader(stage), check_gpu=gpu_reader)
        require(receipt.get("schema") == RECEIPT_SCHEMA,
                "runtime observation probe receipt schema가 다르다")
        return receipt
    except BaseException as error:
        POST._failure(stage / "runtime-observation-probe-failure.json", "execute", error)
        raise


def _file_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"receipt 원문 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def _done_authorization(path: Path, plan_path: Path, plan: dict[str, Any]) -> tuple[dict, list]:
    authorization = read(path)
    required = {"schema", "plan_sha256", "snapshot", "choice", "request_spec",
                "ledger_path", "request_id"}
    binding = authorization.get("plan_sha256")
    plan_bound = (binding == sha(plan_path)
                  or (isinstance(binding, dict) and set(binding) == {"case", "reference"}
                      and binding.get(plan["role"]) == sha(plan_path)
                      and all(re.fullmatch(r"[0-9a-f]{64}", str(item))
                              for item in binding.values())))
    require(set(authorization) == required and authorization.get("schema") == AUTH_SCHEMA
            and plan_bound, "완료 authorization 형식 또는 plan SHA가 다르다")
    snapshot = authorization["snapshot"]
    choice = authorization["choice"]
    spec = P1.C._frozen_request_spec(snapshot, choice)
    require(spec == authorization["request_spec"] and spec["probe"] == plan["probe"]
            and spec["args"] == P1.C.RC.registry()["probes"][plan["probe"]]["args"],
            "완료 authorization request spec/args/probe가 다르다")
    packet = snapshot.get("inputs", {}).get("packet", {})
    bindings = packet.get("input_bindings", {})
    role_binding = bindings.get(plan["role"], {})
    expected_checkpoint = (spec["checkpoint_sha256"] if plan["role"] == "case"
                           else spec["reference_checkpoint_sha256"])
    require(expected_checkpoint == role_binding.get("checkpoint_sha256")
            == plan["checkpoint"]["sha256"]
            and role_binding.get("telemetry_sha256") == plan["telemetry_sha256"],
            "완료 authorization checkpoint/telemetry 결합이 다르다")
    snapshot_facts = packet.get("facts", {})
    prefix = f"runtime:{plan['role']}:"
    role_facts = {key: item for key, item in snapshot_facts.items()
                  if isinstance(key, str) and key.startswith(prefix)}
    require(role_facts == plan["runtime_facts"],
            "완료 authorization runtime facts가 검증 원문과 다르다")
    ledger_path = Path(authorization["ledger_path"]).resolve()
    require(ledger_path == Path(snapshot["inputs"]["ledger_path"]).resolve(),
            "완료 authorization ledger 경로가 snapshot과 다르다")
    rows, _ledger_digest, pending = P1.C.V.read_history(
        ledger_path, snapshot["binding"], packet["execution_budget_s"])
    require(pending is None, "완료 receipt 뒤 미완결 probe 요청이 있다")
    ledger = P1.C.S.M.RegistryLedger(ledger_path, P1.C.RC.registry())
    request = ledger.requests().get(authorization["request_id"], {})
    events = [row for row in ledger.events()
              if row.get("request_id") == authorization["request_id"]]
    require(request.get("state") == "done" and request.get("probe") == plan["probe"]
            and request.get("prereg_digest") == P1.C.L.digest(spec["prereg"])
            and request.get("args_digest") == P1.C.L.digest(spec["args"])
            and [row.get("event") for row in events]
            == ["proposed", "approved", "consumed", "receipt"],
            "현재 ledger가 승인→consume→receipt done 상태가 아니다")
    return authorization, rows


def verify_receipt(plan_path: Path, receipt_path: Path) -> dict[str, Any]:
    """현재 done ledger와 raw proof로 probe receipt를 read-only 재검증한다."""
    plan_path, receipt_path = Path(plan_path).resolve(), Path(receipt_path).resolve()
    plan = _verify_plan(plan_path)
    receipt = read(receipt_path)
    require(receipt.get("schema") == RECEIPT_SCHEMA
            and receipt.get("status") in {"complete", "unknown"},
            "runtime observation probe receipt schema/status가 다르다")
    require(receipt.get("plan") == _file_ref(plan_path), "receipt plan ref가 다르다")
    authorization_ref = receipt.get("authorization")
    require(isinstance(authorization_ref, dict)
            and set(authorization_ref) == {"path", "sha256", "request_id"},
            "receipt authorization ref 형식이 다르다")
    authorization_ref = cast(dict[str, Any], authorization_ref)
    authorization_path = Path(authorization_ref["path"]).resolve()
    require(authorization_ref["sha256"] == sha(authorization_path),
            "receipt authorization bytes가 바뀌었다")
    authorization, history = _done_authorization(authorization_path, plan_path, plan)
    require(authorization_ref["request_id"] == authorization["request_id"],
            "receipt authorization request ID가 다르다")
    require(receipt.get("probe") == plan["probe"] and receipt.get("role") == plan["role"]
            and receipt.get("source_arm") == plan["source_arm"]
            and receipt.get("checkpoint_sha256") == plan["checkpoint"]["sha256"]
            and receipt.get("seed") == plan["seed"],
            "receipt role/probe/source/checkpoint/seed 결합이 다르다")
    expected_source_refs = {
        "collected_input": deepcopy(plan["collected_input"]),
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "auxiliary_inputs": deepcopy(plan["auxiliary_inputs"]),
    }
    require(receipt.get("source_refs") == expected_source_refs,
            "receipt collected/manifest/auxiliary source ref가 다르다")
    require(receipt.get("projection") == plan["projection"],
            "receipt probe projection이 plan과 다르다")
    expected_raw = {name: _file_ref(Path(path)) for name, path in plan["raw_files"].items()
                    if Path(path).is_file()}
    owned_result = Path(plan["stage"]) / "owned-result.json"
    if owned_result.is_file():
        expected_raw["owned_result"] = _file_ref(owned_result)
    require(receipt.get("raw_refs") == expected_raw, "receipt raw refs SHA가 다르다")
    claim = receipt.get("claim")
    require(isinstance(claim, dict) and set(claim) == {"path", "sha256", "body"},
            "receipt execution claim 형식이 다르다")
    claim = cast(dict[str, Any], claim)
    claim_path = Path(claim["path"]).resolve()
    claim_body = read(claim_path)
    spec = authorization["request_spec"]
    require(claim.get("sha256") == sha(claim_path) and claim.get("body") == claim_body
            and claim_body.get("role") == plan["role"]
            and claim_body.get("request_id") == authorization["request_id"]
            and claim_body.get("plan_sha256") == sha(plan_path)
            and claim_body.get("checkpoint_sha256") == plan["checkpoint"]["sha256"]
            and claim_body.get("telemetry_sha256") == plan["telemetry_sha256"]
            and claim_body.get("context_digest") == P1.C.L.digest(spec["prereg"])
            and claim_body.get("args_digest") == P1.C.L.digest(spec["args"]),
            "receipt claim/plan/context 결합이 다르다")
    status = receipt["status"]
    if status == "complete":
        execution = read(owned_result)
        require(receipt.get("error_type") is None and receipt.get("execution") == execution
                and _execution_ok(execution),
                "complete receipt owned exit0/natural exit/cleanup 증거가 다르다")
        with _backend_scope():
            measurement = (P1._legacy_measurement(plan) if plan["probe"] in LEGACY_PROBES
                           else P1._optimizer_measurement(plan))
        require(receipt.get("measurement") == measurement,
                "complete receipt raw measurement 재계산이 다르다")
        require(any(row.get("probe") == plan["probe"] for row in history),
                "done ledger history에 완료 probe가 없다")
        correction_allowed = True
    else:
        require(receipt.get("error_type") is not None,
                "unknown receipt에는 오류 유형이 있어야 한다")
        measurement = None
        correction_allowed = False
    return {
        "schema": "runtime_observation_probe_verified_receipt_v1",
        "status": status,
        "correction_allowed": correction_allowed,
        "role": plan["role"], "probe": plan["probe"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        "telemetry_sha256": plan["telemetry_sha256"],
        "binding_digest": plan["binding_digest"],
        "measurement": deepcopy(measurement),
        "plan": _file_ref(plan_path), "receipt": _file_ref(receipt_path),
        "authorization": _file_ref(authorization_path), "claim": _file_ref(claim_path),
        "raw_refs": deepcopy(expected_raw),
    }


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(P1, name)
