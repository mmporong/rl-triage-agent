"""case-07 복합 결함의 실행 전 unknown 두 행만 개발용으로 재측정한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_comparison_batch_v1 as BATCH
import lineage_diagnosis_epoch_v1 as E
import lineage_lease_context_v2 as LEASE
import lineage_nim_transport_v1 as T
import lineage_probe_failure_evidence_v3 as F3

SCHEMA = "lineage_compound_retry_protocol_v1"
MANIFEST_SCHEMA = "lineage_compound_retry_manifest_v1"
RESULT_SCHEMA = "lineage_compound_retry_result_v1"
TARGET_CASE = "case-07"
POLICIES = ("registry_next", "model_adaptive")
PLAN = (
    {"order": 1, "case_id": TARGET_CASE, "policy": "registry_next"},
    {"order": 2, "case_id": TARGET_CASE, "policy": "model_adaptive"},
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DRIVER_SOURCE_FILES = (
    "evals/lineage_compound_retry_v1.py",
    "tests/test_lineage_compound_retry_v1.py",
)
BOUND_SOURCE_FILES = (*LEASE.SOURCE_FILES, *DRIVER_SOURCE_FILES)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON 객체가 아니다: {path}")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    payload = Path(path).read_bytes()
    if lf:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("component protocol은 repo 내부여야 한다") from error


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def transport_plan() -> dict:
    return {
        "case_ids": [TARGET_CASE],
        "logical_slots": [
            {"slot_id": f"{TARGET_CASE}.model.{index}", "case_id": TARGET_CASE}
            for index in range(1, 4)
        ],
        "global_request_cap": 9,
        "per_slot_max": 3,
    }


def prepare_protocol(comparison_protocol: Path, nim_protocol: Path,
                     failure_protocol: Path) -> dict:
    """T.initialize가 소비하는 transport protocol에 retry 계약을 함께 등록한다."""
    comparison_path = Path(comparison_protocol).resolve()
    nim_path = Path(nim_protocol).resolve()
    failure_path = Path(failure_protocol).resolve()
    comparison, nim = _read(comparison_path), _read(nim_path)
    failure = _read(failure_path)
    _require(comparison.get("schema") == BATCH.SCHEMA, "comparison protocol schema가 다르다")
    _require(nim.get("schema") == T.SCHEMA and nim.get("plan") == transport_plan(),
             "새 case-07/3-slot/9-cap NIM protocol이 다르다")
    _require(failure.get("schema") == F3.SCHEMA, "failure protocol schema가 다르다")
    _require(comparison.get("budget", {}).get("max_decisions_per_policy_case") == 3
             and comparison.get("budget", {}).get("execution_request_budget_s") == 1200,
             "원래 비교의 3-decision/1200초 논리 예산이 다르다")
    admission_path = ROOT / failure["component_admission_protocol"]["path"]
    admission = _read(admission_path)
    _require(admission.get("component_probe_protocol")
             == comparison["component_protocols"]["paired_probe"],
             "failure evidence와 comparison의 paired probe가 다르다")
    cases = [row for row in comparison.get("cases", []) if row.get("case_id") == TARGET_CASE]
    _require(len(cases) == 1 and cases[0].get("arm") == "COMPOUND"
             and cases[0].get("seed") == 2307, "case-07 COMPOUND seed 2307 등록이 다르다")
    components = {
        "comparison": {"path": _relative(comparison_path), "sha256_lf": _sha(comparison_path, lf=True)},
        "source_collection": deepcopy(comparison["component_protocols"]["source_collection"]),
        "paired_probe": deepcopy(comparison["component_protocols"]["paired_probe"]),
        "nim_transport": {"path": _relative(nim_path), "sha256_lf": _sha(nim_path, lf=True)},
        "probe_failure_evidence": {"path": _relative(failure_path), "sha256_lf": _sha(failure_path, lf=True)},
    }
    return {
        "schema": SCHEMA,
        "phase": "development_infrastructure_retry_epoch1_first_diagnosis_only",
        "model": nim["model"],
        "accepted_provider_models": deepcopy(nim["accepted_provider_models"]),
        "params": deepcopy(nim["params"]),
        "driver_source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in BOUND_SOURCE_FILES},
        "plan": transport_plan(),
        "target": {"case_id": TARGET_CASE, "seed": 2307, "arm": "COMPOUND",
                   "policies": list(POLICIES), "execution_plan": list(PLAN)},
        "component_protocols": components,
        "runtime_source_sha256_lf": deepcopy(comparison["runtime_source_sha256_lf"]),
        "budget": {"max_decisions_per_policy_case": 3, "execution_request_budget_s": 1200,
                   "model_logical_slots": 3, "model_post_cap_per_slot": 3,
                   "model_post_cap_total": 9, "replacement_seeds": 0,
                   "host_gate_override": None},
        "retry_contract": {
            "original_results_immutable": True,
            "original_hold_required": "probe_outcome_unknown",
            "original_failure_stage": "gpu_admission_before_probe_execution",
            "automatic_retry": False,
            "resume_supported": False,
            "first_diagnosis_only": True,
            "correction": False,
            "training": False,
            "epoch2": False,
            "unknown_or_backend_failure_stops_remaining_rows": True,
        },
        "claims": {"measurement_scope": "development_after_observed_infrastructure_failure",
                   "correction": False, "recovery": False, "generalization": False},
        "outcome_wording": {
            "complete": "case-07 두 정책의 첫 진단을 새 실행으로 보존했다.",
            "invalid": "입력·계보·입장 또는 실행 계약이 불완전해 자동 재시도 없이 중단했다.",
        },
    }


def verify_protocol(protocol_path: Path, *, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "execution anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    _require(value.get("schema") == SCHEMA, "compound retry protocol schema가 다르다")
    components = value.get("component_protocols")
    _require(isinstance(components, dict), "component protocol 표가 없다")
    expected = prepare_protocol(
        ROOT / components["comparison"]["path"],
        ROOT / components["nim_transport"]["path"],
        ROOT / components["probe_failure_evidence"]["path"],
    )
    _require(value == expected, "compound retry protocol 내용 또는 component SHA가 다르다")
    BATCH.verify_protocol(ROOT / components["comparison"]["path"], anchor)
    T._verify_sources(ROOT / components["nim_transport"]["path"], anchor)
    F3.verify_protocol(ROOT / components["probe_failure_evidence"]["path"], anchor=anchor)
    bound = {_relative(path): _sha(path, lf=True), **value["driver_source_sha256_lf"]}
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 retry source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"execution anchor retry blob이 다르다: {relative}")
    return deepcopy(value)


def _ref(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"증거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _verify_runtime(lab: Path, expected: dict) -> None:
    lab = Path(lab).resolve()
    _require(isinstance(expected, dict) and bool(expected), "Isaac runtime source 결합이 없다")
    for relative, expected_sha in expected.items():
        path = lab / relative
        _require(path.is_file() and _sha(path, lf=True) == expected_sha,
                 f"Isaac runtime source SHA가 다르다: {relative}")


def _original_failure(row: dict) -> dict:
    result_path = Path(str(row.get("result_path", ""))).resolve()
    result = row.get("result")
    _require(result_path.is_file() and HEX64.fullmatch(str(row.get("result_sha256", ""))) is not None
             and _sha(result_path) == row["result_sha256"] and _read(result_path) == result,
             "원래 결과 파일·SHA·embedded result가 다르다")
    _require(row.get("case_id") == TARGET_CASE and row.get("arm") == "COMPOUND"
             and row.get("policy") in POLICIES and result.get("end_state") == "HOLD"
             and result.get("hold_reason") == "probe_outcome_unknown"
             and result.get("decision_count") == 1 and len(result.get("steps", [])) == 1,
             "원래 case-07 HOLD 첫 단계 계약이 다르다")
    step = result["steps"][0]
    receipts = step.get("paired_receipts")
    _require(step.get("outcome") == "unknown" and isinstance(receipts, dict)
             and set(receipts) == {"case", "reference"}, "원래 paired unknown 계약이 다르다")
    evidence = {}
    causes = []
    step_root = Path(step["authorization_path"]).resolve().parent
    for role in ("reference", "case"):
        receipt = receipts[role]
        _require(receipt == {"status": "unknown", "measurement": None, "execution": {},
                             "error_type": "RuntimeError", "role": role},
                 f"원래 {role} unknown receipt가 다르다")
        stage = step_root / role
        forbidden = (stage / "probe_receipt.json", stage / "owned-result.json", stage / "owned")
        _require(not any(path.exists() for path in forbidden),
                 f"원래 {role} stage에 full receipt 또는 owned 실행이 있다")
        gpu_path = stage / "gpu_before.json"
        gpu = _read(gpu_path)
        offenders = [proc for proc in gpu.get("processes", [])
                     if proc.get("pid") == 139312 and str(proc.get("name", "")).lower() == "svchost.exe"]
        reasons = gpu.get("busy_reasons")
        _require(gpu.get("compute_process_count") == 0 and len(offenders) == 1
                 and reasons == ["unapproved_gpu_process:139312", "unverified_process_identity:139312"],
                 f"원래 {role} GPU 입장 실패 원인이 다르다")
        causes.append((tuple(reasons), offenders[0]["pid"], offenders[0]["name"].lower()))
        evidence[role] = {"gpu_before": _ref(gpu_path), "full_receipt_absent": True,
                          "owned_execution_absent": True}
    _require(len(set(causes)) == 1, "원래 양 role의 svchost 입장 실패 원인이 다르다")
    return {
        "policy": row["policy"], "result": _ref(result_path),
        "authorization": _ref(Path(step["authorization_path"])),
        "probe": step["probe"], "failure_cause": {
            "pid": 139312, "process": "svchost.exe", "busy_reasons": list(causes[0][0]),
        },
        "roles": evidence,
        "original_model_record": {
            "requests": result.get("requests"), "tokens": deepcopy(result.get("tokens")),
            "wall": deepcopy(result.get("wall")),
            "selection_actor": deepcopy(result.get("selection_actor")),
            "model_validity": deepcopy(result.get("model_validity")),
            "model_observation_use": deepcopy(result.get("model_observation_use")),
        } if row["policy"] == "model_adaptive" else None,
    }


def _original_model_transport(transport_root: Path) -> dict:
    root = Path(transport_root).resolve()
    session = T._session(root)
    _require(session["plan"].get("global_request_cap") == 90,
             "원래 90요청 queue session이 아니다")
    requests = []
    for path in (root / "requests").glob("*.json"):
        value = _read(path)
        if value.get("slot_id") == f"{TARGET_CASE}.model.1":
            requests.append((path, value))
    _require(len(requests) == 1 and requests[0][1].get("ordinal") == 1,
             "원래 case-07 첫 model 요청이 하나가 아니다")
    request_path, request = requests[0]
    response_path = root / "responses" / request_path.name
    _require(response_path.is_file(), "원래 case-07 model 응답이 없다")
    return {
        "session": _ref(root / "session.json"),
        "request": _ref(request_path),
        "response": _ref(response_path),
        "slot_id": request["slot_id"],
        "request_id": request["request_id"],
    }


def _fresh_view(case: dict, collected: dict, manifest: dict, output: Path) -> dict:
    envelope = BATCH.INPUT.make_selection_packet(
        Path(collected[case["source_job"]]["path"]),
        Path(collected[case["reference_job"]]["path"]),
        budget_left=3,
        execution_budget_s=1200,
    )
    rows = {}
    for policy in POLICIES:
        snapshot = E.C.V.build_snapshot(
            envelope["packet"], f"compound_retry_{policy}", policy,
            E.C.V.file_digest(E.C.V.PROTOCOL), output / "view-ledgers" / f"{policy}.jsonl",
        )
        rows[policy] = snapshot["view_digest"]
    _require(len(set(rows.values())) == 1, "두 정책의 fresh public view가 다르다")
    return {"digest": next(iter(rows.values())), "policies": rows}


def prepare_inputs(out: Path, protocol_path: Path, original_manifest: Path,
                   original_result: Path, *, lab: Path, execution_anchor: str) -> dict:
    """원래 30행·18 source/fixed/gate를 재검증하고 두 행 retry manifest를 만든다."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = verify_protocol(Path(protocol_path), anchor=execution_anchor)
    old_manifest, comparison, collected, gate_rows = BATCH._validate_manifest(Path(original_manifest))
    result_path = Path(original_result).resolve()
    original = _read(result_path)
    expected_result = Path(old_manifest["output_root"]).resolve() / "execution" / "result.json"
    _require(result_path == expected_result and original.get("schema") == BATCH.RESULT_SCHEMA
             and original.get("status") == "complete" and len(original.get("rows", [])) == 30,
             "원래 30행 result 또는 manifest 결합이 다르다")
    target_rows = [row for row in original["rows"]
                   if row.get("case_id") == TARGET_CASE and row.get("policy") in POLICIES]
    _require(len(target_rows) == 2 and {row["policy"] for row in target_rows} == set(POLICIES),
             "원래 case-07 대상 두 행이 다르다")
    originals = {row["policy"]: _original_failure(row) for row in target_rows}
    originals["model_adaptive"]["original_model_transport"] = _original_model_transport(
        Path(old_manifest["transport_root"])
    )
    all_causes = [json.dumps(role, sort_keys=True)
                  for item in originals.values() for role in [item["failure_cause"]]]
    _require(len(set(all_causes)) == 1, "원래 두 정책의 svchost 입장 실패 원인이 다르다")
    case = next(row for row in comparison["cases"] if row["case_id"] == TARGET_CASE)
    _require(case["seed"] == 2307 and case["arm"] == "COMPOUND"
             and all(gate_rows[TARGET_CASE].get(key) == case[key] for key in
                     ("case_id", "seed", "arm", "source_job", "reference_job")),
             "case-07 source/gate 결합이 다르다")
    _verify_runtime(Path(lab), protocol["runtime_source_sha256_lf"])
    fresh = _fresh_view(case, collected, old_manifest, out)
    transport_root = out / "nim-rpc-v1"
    nim_protocol = ROOT / protocol["component_protocols"]["nim_transport"]["path"]
    session = T.initialize(transport_root, nim_protocol, execution_anchor, transport_plan())
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": _ref(Path(protocol_path)),
        "original_manifest": _ref(Path(original_manifest)),
        "original_result": _ref(result_path),
        "lab": str(Path(lab).resolve()), "execution_anchor": execution_anchor,
        "gate": _ref(Path(old_manifest["gate_path"])),
        "probe_protocol": _ref(Path(old_manifest["probe_protocol"])),
        "failure_protocol": deepcopy(protocol["component_protocols"]["probe_failure_evidence"]),
        "transport_root": str(transport_root), "transport_session": _ref(transport_root / "session.json"),
        "transport_plan": deepcopy(session["plan"]),
        "case": deepcopy(case),
        "collected": {job: deepcopy(collected[job]) for job in (case["source_job"], case["reference_job"])},
        "fresh_public_view": fresh,
        "execution_plan": list(PLAN),
        "original_failures": originals,
        "max_decisions": 3, "execution_budget_s": 1200,
        "resume_supported": False, "automatic_retry": False,
    }
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "intent.json", {
        "mode": "development_infrastructure_retry_first_diagnosis_only",
        "planned_rows": 2, "api_executed": False, "gpu_executed": False,
        "correction": False, "training": False, "epoch2": False,
        "original_results_immutable": True, "automatic_retry": False,
    })
    return deepcopy(manifest)


def _load_manifest(path: Path) -> tuple[dict, dict]:
    manifest = _read(path)
    required = {"schema", "protocol", "original_manifest", "original_result", "lab",
                "execution_anchor", "gate", "probe_protocol", "failure_protocol",
                "transport_root", "transport_session", "transport_plan", "case", "collected",
                "fresh_public_view", "execution_plan", "original_failures", "max_decisions",
                "execution_budget_s", "resume_supported", "automatic_retry"}
    _require(set(manifest) == required and manifest.get("schema") == MANIFEST_SCHEMA,
             "retry manifest 필드가 다르다")
    for key in ("protocol", "original_manifest", "original_result", "gate", "probe_protocol",
                "transport_session"):
        ref = manifest[key]
        path_value = Path(ref.get("path", ""))
        _require(set(ref) == {"path", "sha256"} and path_value.is_file()
                 and _sha(path_value) == ref["sha256"], f"manifest 입력 SHA가 바뀌었다: {key}")
    protocol = verify_protocol(Path(manifest["protocol"]["path"]), anchor=manifest["execution_anchor"])
    _verify_runtime(Path(manifest["lab"]), protocol["runtime_source_sha256_lf"])
    old_manifest, comparison, collected, gate_rows = BATCH._validate_manifest(
        Path(manifest["original_manifest"]["path"])
    )
    original = _read(Path(manifest["original_result"]["path"]))
    target_rows = [row for row in original.get("rows", [])
                   if row.get("case_id") == TARGET_CASE and row.get("policy") in POLICIES]
    originals = {row["policy"]: _original_failure(row) for row in target_rows}
    originals["model_adaptive"]["original_model_transport"] = _original_model_transport(
        Path(old_manifest["transport_root"])
    )
    case = next(row for row in comparison["cases"] if row["case_id"] == TARGET_CASE)
    _require(manifest["case"] == case and manifest["original_failures"] == originals
             and manifest["execution_plan"] == list(PLAN)
             and manifest["max_decisions"] == 3 and manifest["execution_budget_s"] == 1200
             and manifest["transport_plan"] == transport_plan()
             and manifest["resume_supported"] is False and manifest["automatic_retry"] is False,
             "retry manifest 계획·원래 failure 결합이 다르다")
    _require(manifest["collected"] == {job: collected[job]
             for job in (case["source_job"], case["reference_job"])},
             "retry manifest collected 결합이 다르다")
    _require(all(gate_rows[TARGET_CASE].get(key) == case[key] for key in
                 ("case_id", "seed", "arm", "source_job", "reference_job")),
             "retry manifest gate 결합이 다르다")
    session = T._session(Path(manifest["transport_root"]))
    _require(session["plan"] == transport_plan()
             and _sha(Path(manifest["transport_root"]) / "session.json")
             == manifest["transport_session"]["sha256"], "새 9요청 queue 결합이 다르다")
    failure_ref = protocol["component_protocols"]["probe_failure_evidence"]
    _require(manifest["failure_protocol"] == failure_ref,
             "failure evidence protocol 결합이 다르다")
    return manifest, protocol


@contextmanager
def lease_read_scope(evidence_path: Path) -> Iterator[None]:
    """canonical lease metadata EACCES만 재시도하고 모든 read 사건을 별도 봉인한다."""
    evidence_path = Path(evidence_path).resolve()
    if evidence_path.exists():
        raise FileExistsError(f"lease read evidence가 이미 존재한다: {evidence_path}")
    events: list[dict] = []
    _require(getattr(LEASE._LOCAL, "events", None) in (None, []),
             "다른 lease read evidence scope가 활성화됐다")
    LEASE._LOCAL.events = events
    LEASE._LOCAL.calls = 0
    try:
        with LEASE.read_scope():
            yield
    except BaseException as error:
        _write_new(evidence_path, LEASE._evidence(
            events, status="failed", failure=LEASE._failure(error)
        ))
        raise
    else:
        _write_new(evidence_path, LEASE._evidence(events, status="complete"))
    finally:
        LEASE._LOCAL.events = None
        LEASE._LOCAL.calls = 0


def execute(manifest_path: Path, *, epoch_runner=F3.run_epoch) -> dict:
    """두 행을 순서대로 실행하며 unknown/예외 뒤에는 다음 행을 시작하지 않는다."""
    manifest, protocol = _load_manifest(Path(manifest_path))
    root = Path(manifest_path).resolve().parent
    execution = root / "execution"
    if execution.exists():
        raise RuntimeError("resume_not_supported: 기존 retry 실행을 보존하고 다시 시작하지 않는다")
    execution.mkdir()
    _write_new(execution / "started.json", {"status": "running", "planned_rows": 2,
                                             "resume_supported": False, "automatic_retry": False})
    case = manifest["case"]
    rows: list[dict] = []
    synthetic = epoch_runner is not F3.run_epoch
    failure_protocol = ROOT / protocol["component_protocols"]["probe_failure_evidence"]["path"]
    try:
        with lease_read_scope(execution / "lease-read-evidence.json"):
            for planned in manifest["execution_plan"]:
                policy = planned["policy"]
                row_root = execution / "runs" / f"{planned['order']:02d}_{TARGET_CASE}_{policy}"
                slots = tuple(f"{TARGET_CASE}.model.{index}" for index in range(1, 4))
                result = epoch_runner(
                    row_root,
                    Path(manifest["collected"][case["source_job"]]["path"]),
                    Path(manifest["collected"][case["reference_job"]]["path"]),
                    policy,
                    run_id=f"compound_retry_{policy}",
                    lab=Path(manifest["lab"]),
                    probe_protocol=Path(manifest["probe_protocol"]["path"]),
                    failure_protocol=failure_protocol,
                    anchor=manifest["execution_anchor"],
                    gate_path=Path(manifest["gate"]["path"]),
                    gate_sha256=manifest["gate"]["sha256"],
                    case_id=TARGET_CASE,
                    transport_root=Path(manifest["transport_root"]),
                    slot_ids=slots,
                    max_decisions=3,
                    execution_budget_s=1200,
                )
                result_path = row_root / "result.json"
                _require(result_path.is_file() and _read(result_path) == result,
                         "epoch result 파일과 반환값이 다르다")
                if not synthetic:
                    _require(result.get("evidence_scope") == "actual_backend",
                             "retry epoch가 실제 기본 backend 경로가 아니다")
                row = {**deepcopy(planned), "arm": "COMPOUND",
                       "fresh_view_digest": manifest["fresh_public_view"]["digest"],
                       "result": deepcopy(result), "result_path": str(result_path),
                       "result_sha256": _sha(result_path),
                       "original_failure": deepcopy(manifest["original_failures"][policy]),
                       "classification": BATCH._classification(case, result)}
                rows.append(row)
                _write_new(execution / "rows" / f"{planned['order']:02d}.json", row)
                hold_reason = result.get("hold_reason")
                if hold_reason == "probe_outcome_unknown" or (
                        isinstance(hold_reason, str) and hold_reason.startswith("probe_prepare_")):
                    raise RuntimeError(f"{hold_reason}: 다음 정책 행을 시작하지 않는다")
        _require(len(rows) == 2 and len({row["fresh_view_digest"] for row in rows}) == 1,
                 "두 정책의 fresh input 짝이 불완전하다")
        result = {
            "schema": RESULT_SCHEMA, "status": "complete",
            "evidence_scope": "cpu_fixture" if synthetic else "actual_backend",
            "measurement_scope": "development_after_observed_infrastructure_failure",
            "rows": rows, "original_results_immutable": True, "correction_executed": False,
            "training_executed": False, "epoch2_registered": False,
            "resume_supported": False, "automatic_retry": False,
        }
        _write_new(execution / "result.json", result)
        _write_new(execution / "completed.json", {"status": "complete", "rows": 2,
                                                   "result_sha256": _sha(execution / "result.json")})
        return deepcopy(result)
    except BaseException as error:
        if not (execution / "failed.json").exists():
            _write_new(execution / "failed.json", {
                "status": "invalid", "completed_rows": len(rows),
                "error_type": type(error).__name__, "error": str(error),
                "resume_supported": False, "automatic_retry": False,
            })
        raise
    finally:
        if not (Path(manifest["transport_root"]) / "closed.json").exists():
            T.close(Path(manifest["transport_root"]), "compound_retry_finished")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    public = commands.add_parser("prepare-protocol")
    public.add_argument("--comparison-protocol", type=Path, required=True)
    public.add_argument("--nim-protocol", type=Path, required=True)
    public.add_argument("--failure-protocol", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--protocol", type=Path, required=True)
    prepare_cmd.add_argument("--original-manifest", type=Path, required=True)
    prepare_cmd.add_argument("--original-result", type=Path, required=True)
    prepare_cmd.add_argument("--out", type=Path, required=True)
    prepare_cmd.add_argument("--lab", type=Path, required=True)
    prepare_cmd.add_argument("--execution-anchor", required=True)
    execute_cmd = commands.add_parser("execute")
    execute_cmd.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.comparison_protocol, args.nim_protocol,
                                              args.failure_protocol))
        return 0
    if args.command == "prepare":
        prepare_inputs(args.out, args.protocol, args.original_manifest, args.original_result,
                       lab=args.lab, execution_anchor=args.execution_anchor)
        return 0
    execute(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
