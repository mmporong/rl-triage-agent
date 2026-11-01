# pyright: reportMissingImports=false
"""R3 paired probe receipt에서 owned SDK role 시간을 읽기 전용으로 보충 감사한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_five_policy_summary_v1 as SUMMARY

DRIVER = SUMMARY.DRIVER
RUNNER = SUMMARY.RUNNER
BASE_HOST = DRIVER.HOST
HOST = RUNNER.HOST
OBS = HOST._PROBE

SCHEMA = "lineage_probe_cost_audit_v1"
PAIR_SCHEMA = SCHEMA + "_pair"
ROW_SCHEMA = SCHEMA + "_row"
SUMMARY_SCHEMA = SCHEMA + "_summary"
CASES_SCHEMA = SCHEMA + "_cases"
PROVENANCE_SCHEMA = SCHEMA + "_provenance"
MANIFEST_SCHEMA = SCHEMA + "_manifest"
PUBLIC_FILES = ("summary.json", "cases.json", "provenance.json", "manifest.json")
WALL_SCOPE = "parent_launch_to_exit_and_owned_job_empty_verification"
ROLES = ("reference", "case")
PRIVATE_VALUE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp|root|private)/|"
    r"bench[\\/]private|Bearer\s+|nvapi-|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)
SECRET_KEY = re.compile(
    r"^(?:api_?key|authorization|credential|secret|access_?token|pid|process_id|"
    r"owned_process_identities)$",
    re.IGNORECASE,
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")),
    )
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    require(not target.exists(), f"{target.name}은 fresh 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(json_bytes(value))


def _bound(value: object, label: str) -> Path:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            f"{label} ref 형식이 다르다")
    row = cast(Mapping[str, Any], value)
    target = Path(str(row["path"])).resolve(strict=True)
    require(target.is_file() and sha(target) == row["sha256"], f"{label} SHA가 다르다")
    return target


def _wall(value: object, label: str) -> float:
    require(type(value) in {int, float} and math.isfinite(float(value)) and value >= 0,
            f"{label}은 유한한 0 이상 수치여야 한다")
    return float(cast(float, value))


def _assert_public(*values: object) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                require(not (isinstance(key, str) and SECRET_KEY.fullmatch(key)),
                        "공개 JSON에 credential/PID key가 있다")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            require(PRIVATE_VALUE.search(value) is None,
                    "공개 JSON에 개인·private 경로 또는 credential 값이 있다")

    for value in values:
        visit(value)


def _execution(execution: object, label: str) -> dict[str, Any]:
    require(isinstance(execution, dict), f"{label} owned execution이 없다")
    row = cast(dict[str, Any], execution)
    require(row.get("job_assigned_before_release") is True
            and row.get("natural_exit") is True
            and row.get("timed_out") is False
            and row.get("error_type") is None
            and row.get("exit_code") == 0
            and row.get("cleanup_verified") is True
            and row.get("wall_scope") == WALL_SCOPE,
            f"{label} exit0/natural/cleanup/time scope가 다르다")
    _wall(row.get("wall_s"), f"{label} wall_s")
    return row


def _anchor_blob(module: Any, anchor: str, relative: str, expected: str) -> None:
    blob = module._git("show", f"{anchor}:{relative}").stdout.encode("utf-8")
    blob = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(blob).hexdigest() == expected,
            f"등록 anchor source/blob이 다르다: {relative}")


def _historical_base_protocol(path: Path, lab: Path, anchor: str) -> dict[str, Any]:
    """실행 뒤 공유 source 변경과 분리해 등록 anchor blob과 runtime pin을 검증한다."""
    module = OBS.P1
    protocol_path, lab = Path(path).resolve(strict=True), Path(lab).resolve(strict=True)
    value = module.read(protocol_path)
    require(set(value) == module.PROTOCOL_FIELDS and value.get("schema") == module.PROTOCOL_SCHEMA,
            "historical component probe protocol 형식이 다르다")
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "registration anchor 형식")
    source = module._hash_map(value["source_sha256_lf"], "source_sha256_lf")
    runtime = module._hash_map(value["runtime_source_sha256_lf"], "runtime_source_sha256_lf")
    required = {"evals/lineage_probe_backend_v1.py", "evals/probes.py",
                "bench/hidden_faults.py", *module.RESIDUAL.MINIMUM_BOUND_SOURCES}
    require(required <= set(source), "historical probe source closure 누락")
    watchdogs = value.get("watchdogs_s")
    require(isinstance(watchdogs, dict) and set(watchdogs) == {"legacy", "optimizer"}
            and all(type(watchdogs[name]) is int and watchdogs[name] > 0 for name in watchdogs),
            "historical watchdog 계약이 다르다")
    relative_protocol = protocol_path.relative_to(module.ROOT).as_posix()
    for relative, expected in {relative_protocol: module.sha(protocol_path, lf=True),
                               **source}.items():
        _anchor_blob(module, anchor, relative, expected)
    for relative, expected in runtime.items():
        target = lab / relative
        require(target.is_file() and module.sha(target, lf=True) == expected,
                f"runtime source가 다르다: {relative}")
    require((lab / value["isaac_python_relative"]).is_file(), "Isaac Python이 없다")
    return deepcopy(value)


def _historical_wrapper_protocol(path: Path, lab: Path, anchor: str) -> dict[str, Any]:
    protocol_path = Path(path).resolve(strict=True)
    value = OBS.read(protocol_path)
    require(set(value) == OBS.PROTOCOL_FIELDS
            and value.get("schema") == OBS.PROTOCOL_SCHEMA
            and value.get("purpose") == OBS.PROTOCOL_PURPOSE,
            "historical runtime wrapper protocol 형식이 다르다")
    component = value.get("component_probe_protocol")
    require(isinstance(component, dict) and set(component) == {"path", "sha256_lf"},
            "historical component protocol ref가 없다")
    base_path = OBS.ROOT / component["path"]
    require(OBS.sha(base_path, lf=True) == component["sha256_lf"],
            "historical component protocol bytes가 다르다")
    base = _historical_base_protocol(base_path, lab, anchor)
    require(set(base["runtime_source_sha256_lf"]) == OBS.RUNTIME_PINS,
            "historical runtime pin은 정확히 9개여야 한다")
    relative_protocol = protocol_path.relative_to(OBS.ROOT).as_posix()
    for relative, expected in {relative_protocol: OBS.sha(protocol_path, lf=True),
                               **value["source_sha256_lf"]}.items():
        _anchor_blob(OBS.P1, anchor, relative, expected)
    return deepcopy(value)


@contextmanager
def _historical_protocol_scope(plan_path: Path):
    plan = read(plan_path)
    protocol = plan.get("protocol")
    require(isinstance(protocol, dict), "plan protocol ref가 없다")
    path, lab, anchor = Path(protocol["path"]), Path(plan["lab"]), protocol["anchor"]
    _historical_wrapper_protocol(path, lab, anchor)
    original_wrapper, original_base = OBS.verify_protocol, OBS.P1.verify_protocol
    OBS.verify_protocol = _historical_wrapper_protocol
    OBS.P1.verify_protocol = _historical_base_protocol
    try:
        yield
    finally:
        OBS.verify_protocol, OBS.P1.verify_protocol = original_wrapper, original_base


def inspect_pair(result_path: Path) -> dict[str, Any]:
    """pair/plan/request/receipt/owned 파일 결합에서 역할별 시간을 복원한다."""
    path = Path(result_path).resolve(strict=True)
    pair = read(path)
    require(pair.get("schema") == BASE_HOST.SCHEMA
            and pair.get("status") in {"complete", "unknown_hold"}
            and pair.get("evidence_kind") == "actual_runtime_observation"
            and pair.get("retry_count") == 0,
            "paired host actual result가 아니다")
    authorization_path = _bound(pair.get("authorization"), "pair authorization")
    authorization = read(authorization_path)
    require(authorization.get("request_id") == pair.get("backend_request_id")
            and authorization.get("request_spec", {}).get("probe") == pair.get("probe"),
            "pair authorization/request 결합이 다르다")
    roles: list[dict[str, Any]] = []
    known_wall = 0.0
    unknown_roles: list[str] = []
    observed_roles = 0
    for role in ROLES:
        plan_path = _bound(pair["plans"][role], f"{role} plan")
        plan = read(plan_path)
        require(plan.get("role") == role and plan.get("probe") == pair.get("probe")
                and isinstance(plan.get("checkpoint"), dict),
                f"{role} plan role/probe/checkpoint 결합이 다르다")
        plan_binding = authorization.get("plan_sha256")
        require(isinstance(plan_binding, dict) and plan_binding.get(role) == sha(plan_path),
                f"{role} authorization plan SHA가 다르다")
        expected_checkpoint = (authorization["request_spec"]["checkpoint_sha256"]
                               if role == "case" else
                               authorization["request_spec"]["reference_checkpoint_sha256"])
        require(plan["checkpoint"].get("sha256") == expected_checkpoint,
                f"{role} request/checkpoint 결합이 다르다")
        receipt_ref = pair["receipts"][role]
        if receipt_ref is None:
            unknown = pair["outer_unknown_receipts"][role]
            require(isinstance(unknown, dict) and unknown.get("status") == "unknown",
                    f"{role} explicit unknown receipt가 없다")
            unknown_roles.append(role)
            roles.append({
                "role": role, "status": "unknown", "wall_s": None,
                "timing_source": "explicit_outer_unknown", "plan_sha256": sha(plan_path),
                "receipt_sha256": None, "owned_result_sha256": None,
                "checkpoint_sha256": None,
            })
            continue
        receipt_path = _bound(receipt_ref, f"{role} receipt")
        receipt = read(receipt_path)
        receipt_authorization = receipt.get("authorization")
        require(receipt.get("status") == "complete" and receipt.get("role") == role
                and receipt.get("probe") == pair.get("probe")
                and receipt.get("plan") == pair["plans"][role]
                and receipt.get("checkpoint_sha256") == expected_checkpoint
                and isinstance(receipt_authorization, dict)
                and receipt_authorization.get("sha256") == sha(authorization_path)
                and receipt_authorization.get("request_id") == pair.get("backend_request_id"),
                f"{role} receipt/plan/role 결합이 다르다")
        claim = receipt.get("claim")
        require(isinstance(claim, dict) and isinstance(claim.get("body"), dict),
                f"{role} receipt claim이 없다")
        claim_path = Path(str(claim.get("path", ""))).resolve(strict=True)
        claim_body = read(claim_path)
        require(claim.get("sha256") == sha(claim_path) and claim.get("body") == claim_body
                and claim_body.get("role") == role
                and claim_body.get("request_id") == pair.get("backend_request_id")
                and claim_body.get("plan_sha256") == sha(plan_path)
                and claim_body.get("checkpoint_sha256") == expected_checkpoint,
                f"{role} claim/request/plan/checkpoint 결합이 다르다")
        raw_refs = receipt.get("raw_refs")
        require(isinstance(raw_refs, dict) and isinstance(raw_refs.get("owned_result"), dict),
                f"{role} owned-result ref가 없다")
        owned_path = _bound(raw_refs["owned_result"], f"{role} owned-result")
        require(owned_path.parent == plan_path.parent,
                f"{role} owned-result가 plan stage와 다르다")
        owned = read(owned_path)
        execution = _execution(receipt.get("execution"), role)
        require(execution == owned, f"{role} inline execution과 owned-result가 다르다")
        wall = _wall(execution["wall_s"], f"{role} wall_s")
        observed_roles += 1
        known_wall += wall
        roles.append({
            "role": role, "status": "complete", "wall_s": wall,
            "wall_scope": WALL_SCOPE, "timing_source": "receipt_reconstruction",
            "plan_sha256": sha(plan_path), "receipt_sha256": sha(receipt_path),
            "owned_result_sha256": sha(owned_path),
            "checkpoint_sha256": expected_checkpoint,
        })
    total = known_wall if not unknown_roles else None
    reported_unknown = pair.get("owned_child_unknown_roles")
    require(isinstance(reported_unknown, list), "reported unknown role 목록이 없다")
    return {
        "schema": PAIR_SCHEMA, "status": pair.get("status"),
        "probe": pair.get("probe"), "outcome": pair.get("outcome"),
        "pair_result_sha256": sha(path),
        "pair_parent_wall_s": _wall(pair.get("parent_wall_s"), "pair parent wall_s"),
        "reported": {
            "owned_child_wall_s": pair.get("owned_child_wall_s"),
            "owned_child_known_wall_s": pair.get("owned_child_known_wall_s"),
            "owned_child_unknown_roles": deepcopy(reported_unknown),
        },
        "reconstructed": {
            "timing_source": "receipt_reconstruction",
            "sdk_roles_observed": observed_roles,
            "sdk_roles_timed": observed_roles,
            "known_owned_wall_s": known_wall,
            "owned_wall_s": total,
            "unknown_roles": unknown_roles,
            "roles": roles,
        },
    }


def inspect_orphan_role(interruption: dict[str, Any]) -> dict[str, Any]:
    """pair result 없이 완료된 reference role과 미실행 case role을 분리 검증한다."""
    require(interruption.get("schema") == "lineage_five_policy_partial_role_interruption_v1"
            and interruption.get("stage") == "paired_probe_role_execution"
            and interruption.get("pair_result") is None
            and interruption.get("case_id") in {f"case-{index:02d}" for index in range(1, 11)}
            and interruption.get("arm") in DRIVER.ARMS
            and type(interruption.get("step")) is int and interruption["step"] >= 1
            and isinstance(interruption.get("probe"), str),
            "partial role interruption 형식이 다르다")
    roles = interruption.get("roles")
    require(isinstance(roles, dict) and set(roles) == {"reference", "case"},
            "partial role 두 역할이 없다")
    authorization_path = _bound(interruption.get("authorization"), "orphan authorization")
    authorization = read(authorization_path)
    probe = interruption["probe"]
    request_id = authorization.get("request_id")
    require(authorization.get("request_spec", {}).get("probe") == probe
            and isinstance(request_id, str), "orphan authorization request/probe가 다르다")

    reference = roles["reference"]
    require(isinstance(reference, dict) and reference.get("status") == "complete",
            "orphan reference 완료 표식이 없다")
    reference_plan = _bound(reference.get("plan"), "orphan reference plan")
    reference_receipt = _bound(reference.get("receipt"), "orphan reference receipt")
    reference_owned = _bound(reference.get("owned_result"), "orphan reference owned-result")
    reference_claim = _bound(reference.get("claim"), "orphan reference claim")
    plan = read(reference_plan)
    receipt = read(reference_receipt)
    claim_body = read(reference_claim)
    owned = read(reference_owned)
    expected_checkpoint = authorization["request_spec"]["reference_checkpoint_sha256"]
    require(plan.get("role") == "reference" and plan.get("probe") == probe
            and plan.get("checkpoint", {}).get("sha256") == expected_checkpoint
            and authorization.get("plan_sha256", {}).get("reference") == sha(reference_plan),
            "orphan reference plan/request/checkpoint 결합이 다르다")
    receipt_authorization = receipt.get("authorization")
    claim = receipt.get("claim")
    require(receipt.get("status") == "complete" and receipt.get("role") == "reference"
            and receipt.get("probe") == probe and receipt.get("plan") == reference["plan"]
            and receipt.get("checkpoint_sha256") == expected_checkpoint
            and receipt.get("raw_refs", {}).get("owned_result") == reference["owned_result"]
            and isinstance(receipt_authorization, dict)
            and receipt_authorization.get("sha256") == sha(authorization_path)
            and receipt_authorization.get("request_id") == request_id
            and isinstance(claim, dict) and claim.get("path") == str(reference_claim)
            and claim.get("sha256") == sha(reference_claim) and claim.get("body") == claim_body
            and claim_body.get("role") == "reference"
            and claim_body.get("request_id") == request_id
            and claim_body.get("plan_sha256") == sha(reference_plan)
            and claim_body.get("checkpoint_sha256") == expected_checkpoint,
            "orphan reference receipt/claim/request 결합이 다르다")
    execution = _execution(receipt.get("execution"), "orphan reference")
    require(execution == owned, "orphan reference inline execution과 owned-result가 다르다")

    case = roles["case"]
    require(isinstance(case, dict) and case.get("status") == "not_executed"
            and case.get("receipt") is None and case.get("owned_result") is None
            and case.get("claim") is None
            and case.get("reason") == "lease_read_permission_error_before_case_execute_approval",
            "orphan case 미실행 증거가 다르다")
    case_plan = _bound(case.get("plan"), "orphan case plan")
    case_value = read(case_plan)
    expected_case_checkpoint = authorization["request_spec"]["checkpoint_sha256"]
    require(case_value.get("role") == "case" and case_value.get("probe") == probe
            and case_value.get("checkpoint", {}).get("sha256") == expected_case_checkpoint
            and authorization.get("plan_sha256", {}).get("case") == sha(case_plan),
            "orphan case plan/request/checkpoint 결합이 다르다")
    require(not (case_plan.parent / "probe_receipt.json").exists()
            and not (case_plan.parent / "owned-result.json").exists()
            and not (case_plan.parent / "owned/launch.json").exists(),
            "orphan case에 실행 산출물이 있다")
    case_claim = reference_claim.with_name("case.json")
    require(not case_claim.exists(), "orphan case claim이 존재한다")
    wall = _wall(execution["wall_s"], "orphan reference wall_s")
    return {
        "schema": SCHEMA + "_orphan_role", "status": "verified",
        "case_id": interruption["case_id"], "policy": interruption["arm"],
        "step": interruption["step"], "probe": probe,
        "policy_diagnosis_step": False, "additional_execution": False,
        "orphan_role_cost": {
            "role": "reference", "status": "complete", "wall_s": wall,
            "wall_scope": WALL_SCOPE, "timing_source": "receipt_reconstruction",
            "authorization_sha256": sha(authorization_path),
            "plan_sha256": sha(reference_plan), "receipt_sha256": sha(reference_receipt),
            "owned_result_sha256": sha(reference_owned), "claim_sha256": sha(reference_claim),
            "checkpoint_sha256": expected_checkpoint,
        },
        "unexecuted_role": {
            "role": "case", "status": "not_executed", "wall_s": None,
            "plan_sha256": sha(case_plan), "checkpoint_sha256": expected_case_checkpoint,
            "reason": case["reason"], "receipt_present": False,
            "owned_result_present": False, "owned_launch_present": False,
            "claim_present": False,
        },
    }


def _historical_snapshot(detail: dict[str, Any], source: dict[str, Any], step_index: int,
                         pair: dict[str, Any], temp_root: Path) -> dict[str, Any]:
    ledger_path = Path(pair["public_ledger"]["path"]).resolve(strict=True)
    prefix = b""
    if step_index:
        previous_ref = detail["steps"][step_index - 1]["probe_result"]
        previous = read(_bound(previous_ref, "previous pair result"))
        require(Path(previous["public_ledger"]["path"]).resolve() == ledger_path,
                "step 사이 public ledger가 바뀌었다")
        proof = previous["public_ledger"]["prefix"]
        prefix = ledger_path.read_bytes()[: proof["byte_length"]]
        require(hashlib.sha256(prefix).hexdigest() == proof["sha256"],
                "previous ledger prefix가 바뀌었다")
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"step-{step_index + 1:02d}.ledger.jsonl"
    temporary.write_bytes(prefix)
    snapshot = DRIVER.C.V.build_snapshot(
        source["packet"], detail["run_id"], detail["host_storage_policy"],
        DRIVER.C.V.file_digest(DRIVER.C.V.PROTOCOL), temporary)
    snapshot["inputs"]["ledger_path"] = str(ledger_path)
    require(digest(snapshot) == pair.get("public_snapshot_sha256")
            and snapshot.get("view_digest") == detail["steps"][step_index]["before_view_digest"],
            "historical public snapshot 복원이 다르다")
    return snapshot


def inspect_step(detail_path: Path, source: dict[str, Any], step_index: int,
                 temp_root: Path) -> dict[str, Any]:
    """driver step의 pair result를 역사적 snapshot과 결합해 읽기 전용 검사한다."""
    detail = read(Path(detail_path).resolve(strict=True))
    steps = detail.get("steps")
    require(isinstance(steps, list) and 0 <= step_index < len(steps), "step index가 다르다")
    step = cast(dict[str, Any], steps[step_index])
    pair_path = _bound(step.get("probe_result"), "step pair result")
    pair = read(pair_path)
    snapshot = _historical_snapshot(detail, source, step_index, pair, temp_root)
    choice = pair.get("public_choice")
    require(isinstance(choice, dict), "pair public choice가 없다")
    case_plan = _bound(pair["plans"]["case"], "case plan")
    with _historical_protocol_scope(case_plan):
        verified = HOST.verify_pair(pair_path, snapshot, choice)
    inspected = inspect_pair(pair_path)
    require(verified.get("result_ref", {}).get("sha256") == inspected["pair_result_sha256"],
            "기존 verify_pair와 raw pair SHA가 다르다")
    require(inspected["probe"] == step.get("probe")
            and inspected["outcome"] == step.get("outcome")
            and inspected["reported"]["owned_child_wall_s"] == step.get("owned_child_wall_s"),
            "driver step/pair result 결합이 다르다")
    return {
        "step": step.get("step"), "probe": step.get("probe"),
        "outcome": step.get("outcome"), "pair": inspected,
    }


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    output = deepcopy(rows)
    for row in output:
        candidates = [role for step in row["steps"]
                      for role in step["pair"]["reconstructed"]["roles"]]
        candidates.extend(item["orphan_role_cost"]
                          for item in row.get("partial_role_interruptions", []))
        for role in candidates:
            receipt = role.get("receipt_sha256")
            role["counted_once"] = receipt is not None and receipt not in seen
            if receipt is None:
                continue
            step_value = next((step["step"] for step in row["steps"]
                               if role in step["pair"]["reconstructed"]["roles"]),
                              next((item["step"] for item in row.get(
                                  "partial_role_interruptions", [])
                                    if role is item["orphan_role_cost"]), None))
            identity = {"case_id": row["case_id"], "policy": row["policy"],
                        "step": step_value, "role": role["role"]}
            if receipt in seen:
                role["duplicate_of"] = deepcopy(seen[receipt])
                duplicates.append({"receipt_sha256": receipt,
                                   "first": deepcopy(seen[receipt]), "duplicate": identity,
                                   "reason": "same_receipt_referenced_multiple_times"})
            else:
                seen[receipt] = identity
    return output, duplicates


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(f"case-{case:02d}", policy)
                for case in range(1, 11) for policy in DRIVER.ARMS}
    observed = {(row.get("case_id"), row.get("policy")) for row in rows}
    require(len(rows) == 50 and observed == expected, "audit는 complete case10×policy5가 필요하다")
    deduped, duplicates = _deduplicate(rows)

    def totals(selected: list[dict[str, Any]]) -> dict[str, Any]:
        roles = [role for row in selected for step in row["steps"]
                 for role in step["pair"]["reconstructed"]["roles"]]
        orphan_roles = [item["orphan_role_cost"] for row in selected
                        for item in row.get("partial_role_interruptions", [])]
        roles.extend(orphan_roles)
        counted = [role for role in roles if role.get("counted_once")]
        unknown = [role for role in roles if role.get("status") == "unknown"]
        unexecuted = [item["unexecuted_role"] for row in selected
                      for item in row.get("partial_role_interruptions", [])]
        return {
            "planned_rows": len(selected),
            "zero_probe_rows": sum(not row["steps"] for row in selected),
            "paired_probe_steps": sum(len(row["steps"]) for row in selected),
            "sdk_roles_counted_once": len(counted),
            "orphan_roles_counted_once": sum(
                role.get("counted_once") is True for role in orphan_roles),
            "known_owned_wall_s": sum(role["wall_s"] for role in counted),
            "unknown_role_receipts": len(unknown),
            "explicit_not_executed_roles": len(unexecuted),
            "pair_parent_wall_s": sum(step["pair"]["pair_parent_wall_s"]
                                      for row in selected for step in row["steps"]),
            "row_parent_wall_s": sum(row["row_parent_wall_s"] for row in selected),
        }

    policies = {policy: totals([row for row in deduped if row["policy"] == policy])
                for policy in DRIVER.ARMS}
    cases = {f"case-{case:02d}": totals(
        [row for row in deduped if row["case_id"] == f"case-{case:02d}"])
        for case in range(1, 11)}
    return {"rows": deduped, "duplicates": duplicates,
            "totals": totals(deduped), "by_policy": policies, "by_case": cases}


def _intent_accounting(paired_intents: object, completed_pairs: int, *,
                       timed_roles: int, orphan_roles: int,
                       explicit_not_executed_roles: int) -> dict[str, Any]:
    require(type(paired_intents) is int and paired_intents >= completed_pairs >= 0,
            "paired intent/completed pair 수가 다르다")
    unexecuted = cast(int, paired_intents) - completed_pairs
    reserved = cast(int, paired_intents) * 2
    unresolved = reserved - timed_roles - explicit_not_executed_roles
    require(unresolved >= 0 and orphan_roles <= timed_roles,
            "reserved/timed/orphan/unexecuted role 수가 다르다")
    return {
        "registered_pair_intents": paired_intents,
        "completed_pair_results": completed_pairs,
        "pair_intents_without_pair_result": unexecuted,
        "reserved_sdk_roles": reserved,
        "verified_timed_sdk_roles": timed_roles,
        "verified_orphan_sdk_roles": orphan_roles,
        "explicit_not_executed_roles": explicit_not_executed_roles,
        "unresolved_reserved_roles": unresolved,
        "whole_pair_sdk_zero_inferred": False,
        "missing_wall_filled_with_zero": False,
    }


def audit_batch(batch_path: Path, protocol_path: Path, config_path: Path,
                anchor: str) -> dict[str, Any]:
    """complete actual 50행을 기존 preflight/grid/pair verifier로 재검증한다."""
    batch_path = Path(batch_path).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    protocol, config, prepared = RUNNER.preflight(protocol_path, anchor, config_path)
    batch, _verified_grid = SUMMARY._grid(batch_path, allow_fixture=False)
    require(batch.get("execution_anchor") == anchor, "batch execution anchor가 다르다")
    raw_rows = batch.get("rows")
    require(isinstance(raw_rows, list) and len(raw_rows) == 50, "complete batch50가 아니다")
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lineage-probe-cost-audit-") as folder:
        temp_root = Path(folder)
        for outer in cast(list[dict[str, Any]], raw_rows):
            case_id, policy = outer["case_id"], outer["arm"]
            detail_path = SUMMARY._bound_file(outer["result"], f"{case_id}/{policy} result")
            detail = read(detail_path)
            source = prepared[case_id]["source"]
            require(isinstance(source, dict), f"{case_id} source가 actual preflight에 없다")
            steps = [inspect_step(detail_path, source, index, temp_root / f"row-{len(rows):02d}")
                     for index in range(len(detail.get("steps", [])))]
            interruptions = []
            if detail.get("interruption") is not None:
                require(isinstance(detail["interruption"], dict), "interruption 형식이 다르다")
                interruptions.append(inspect_orphan_role(detail["interruption"]))
            rows.append({
                "schema": ROW_SCHEMA, "case_id": case_id, "policy": policy,
                "end_state": detail.get("end_state"),
                "execution_state": "no_probe_executed" if not steps else "paired_probe_executed",
                "steps": steps,
                "partial_role_interruptions": interruptions,
                "row_parent_wall_s": _wall(detail.get("parent_wall_s"), "row parent wall_s"),
                "source_digest": detail.get("source_digest"),
            })
    aggregate = _aggregate(rows)
    totals = aggregate["totals"]
    aggregate["intent_accounting"] = _intent_accounting(
        batch.get("paired_intents"), totals["paired_probe_steps"],
        timed_roles=totals["sdk_roles_counted_once"],
        orphan_roles=totals["orphan_roles_counted_once"],
        explicit_not_executed_roles=totals["explicit_not_executed_roles"])
    return {"batch": batch, "protocol": protocol, "config": config, **aggregate}


def _public_relative(path: Path, label: str) -> str:
    target = Path(path).resolve(strict=True)
    try:
        relative = target.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label}은 저장소 공개 파일이어야 한다") from error
    require(not relative.startswith("bench/private/"), f"{label}은 공개 상대경로여야 한다")
    return relative


def export(batch_path: Path, protocol_path: Path, config_path: Path, anchor: str,
           out: Path) -> dict[str, Any]:
    """complete actual50 감사 결과를 evals/results의 fresh 공개 tag에 쓴다."""
    output = Path(out).resolve()
    require(not output.exists(), "공개 output은 fresh 경로여야 한다")
    require(output.parent == (ROOT / "evals/results").resolve(),
            "공개 output은 evals/results/<fresh-tag>여야 한다")
    paths = tuple(Path(path).resolve(strict=True)
                  for path in (batch_path, protocol_path, config_path))
    before = {path: sha(path) for path in paths}
    audited = audit_batch(*paths, anchor)
    totals = audited["totals"]
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete",
        "scope": "actual_backend_probe_cost_receipt_audit",
        "timing_source": "receipt_reconstruction",
        "wall_scope": WALL_SCOPE,
        "totals": totals, "by_policy": audited["by_policy"], "by_case": audited["by_case"],
        "intent_accounting": audited["intent_accounting"],
        "duplicate_receipt_references": audited["duplicates"],
        "interpretation_limits": {
            "sdk_wall_is_gpu_kernel_time": False,
            "sdk_wall_is_gpu_busy_time": False,
            "diagnostic_efficacy_measured": False,
            "model_comparison_claimed": False,
            "new_gpu_savings_claimed": False,
            "missing_wall_filled_with_zero": False,
        },
    }
    cases = {"schema": CASES_SCHEMA, "status": "complete", "rows": audited["rows"]}
    config = audited["config"]
    source_index = SUMMARY._bound_file(config["source_index"], "source index")
    provenance = {
        "schema": PROVENANCE_SCHEMA, "status": "complete",
        "execution_anchor": anchor,
        "public_protocol": {"path": _public_relative(paths[1], "protocol"),
                            "sha256": sha(paths[1])},
        "sha256_only": {"batch": sha(paths[0]), "execution_config": sha(paths[2]),
                        "source_index": sha(source_index)},
        "raw_receipts_included": False, "raw_metadata_included": False,
        "private_paths_included": False, "credentials_included": False,
        "pid_values_included": False,
    }
    _assert_public(summary, cases, provenance)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", summary)
    write_new(output / "cases.json", cases)
    write_new(output / "provenance.json", provenance)
    inventory = {name: {"sha256": sha(output / name),
                        "size_bytes": (output / name).stat().st_size}
                 for name in PUBLIC_FILES if name != "manifest.json"}
    manifest = {"schema": MANIFEST_SCHEMA, "status": "complete",
                "public_files": list(PUBLIC_FILES), "files": inventory,
                "overwrite_allowed": False}
    _assert_public(manifest)
    write_new(output / "manifest.json", manifest)
    require(before == {path: sha(path) for path in paths}, "audit 중 등록 원문이 바뀌었다")
    verify_export(output)
    return deepcopy(manifest)


def verify_export(out: Path) -> dict[str, Any]:
    root = Path(out).resolve(strict=True)
    manifest = read(root / "manifest.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA
            and manifest.get("status") == "complete"
            and manifest.get("public_files") == list(PUBLIC_FILES)
            and set(manifest.get("files", {})) == set(PUBLIC_FILES) - {"manifest.json"}
            and manifest.get("overwrite_allowed") is False,
            "공개 manifest 계약이 다르다")
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file())
    require(actual == sorted(PUBLIC_FILES), "공개 tag에 미등록 파일이 있다")
    values = {}
    for name in PUBLIC_FILES:
        path = root / name
        value = read(path)
        require(path.read_bytes() == json_bytes(value), f"canonical JSON이 아니다: {name}")
        if name != "manifest.json":
            item = manifest["files"][name]
            require(sha(path) == item.get("sha256")
                    and path.stat().st_size == item.get("size_bytes"),
                    f"공개 파일 SHA/size가 바뀌었다: {name}")
        values[name] = value
    _assert_public(*values.values())
    summary, cases = values["summary.json"], values["cases.json"]
    require(summary.get("schema") == SUMMARY_SCHEMA
            and summary.get("timing_source") == "receipt_reconstruction"
            and summary.get("interpretation_limits", {}).get(
                "missing_wall_filled_with_zero") is False,
            "공개 summary timing 경계가 다르다")
    rows = cases.get("rows")
    require(cases.get("schema") == CASES_SCHEMA and isinstance(rows, list)
            and len(rows) == 50, "공개 cases 50행이 아니다")
    _aggregate(cast(list[dict[str, Any]], rows))
    return deepcopy(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--batch", type=Path, required=True)
    export_parser.add_argument("--protocol", type=Path, required=True)
    export_parser.add_argument("--config", type=Path, required=True)
    export_parser.add_argument("--anchor", required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.batch, args.protocol, args.config, args.anchor, args.out)
    else:
        verify_export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
