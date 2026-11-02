# pyright: reportMissingImports=false, reportUnknownMemberType=false
"""중단된 R4 FIRST를 기존 2 SDK 근거부터 이어 실행하는 등록 host."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v1 as EXPORT_V1
import recovery_outcome_contract_v1 as OUTCOME
import run_five_policy_recovery_loop_v1 as RUN
import run_five_policy_recovery_with_lease_v1 as HOST
from lease_timestamps import parse_lease_timestamp

FIRST = RUN.FIRST
SECOND = RUN.SECOND
FIRST_CONTINUATION_PROTOCOL_SCHEMA = "five_policy_recovery_first_continuation_protocol_v2"
FIRST_CONTINUATION_PLAN_SCHEMA = "five_policy_recovery_first_continuation_private_plan_v2"
FIRST_CONTINUATION_RESULT_SCHEMA = "five_policy_recovery_first_continuation_result_v2"
SECOND_COMPATIBILITY_PROTOCOL_SCHEMA = "five_policy_recovery_second_compatibility_protocol_v2"
ORIGINAL_FIRST_ANCHOR = "a8f6dbe8679e1f9b54a47c4d7db415b35fd8c55e"
ORIGINAL_HOST_ANCHOR = "ccf70bac3b01a003b7863618a3cef9b78bfdc0a7"
FIRST_WALL_S = 14_400.0
OLD_SDK_EXECUTIONS = 2
NEW_SDK_EXECUTION_CAP = 6
SOURCE_FILES = (
    "bench/run_five_policy_recovery_resume_v2.py",
    "tests/test_five_policy_recovery_resume_v2.py",
    *OUTCOME.SOURCE_FILES,
    "evals/lease_timestamps.py",
    "tests/test_lease_timestamps.py",
    "evals/core_seed_guard_v2.py",
    "tests/test_core_seed_guard_v2.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), "JSON object required")
    return value


def _write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    _require(target.is_file(), f"근거 파일이 아니다: {target}")
    return {"path": str(target), "sha256": _sha(target)}


def _bound(value: object, label: str) -> Path:
    return FIRST.bound(value, label)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("등록 source는 저장소 내부여야 한다") from error


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True).stdout


def _verify_blob(anchor: str, relative: str, expected_lf: str) -> None:
    raw = _git("show", f"{anchor}:{relative}")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    _require(hashlib.sha256(normalized).hexdigest() == expected_lf,
             f"등록 anchor blob이 다르다: {relative}")


def _verify_anchor(anchor: str) -> None:
    _require(isinstance(anchor, str) and len(anchor) == 40, "execution anchor 형식")
    head = _git("rev-parse", "HEAD").decode().strip()
    ancestry = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                               anchor, head], check=False)
    _require(ancestry.returncode == 0, "execution anchor ancestry")


def _source_map() -> dict[str, str]:
    return {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _lease_id(value: Mapping[str, Any]) -> str:
    metadata = value.get("metadata")
    candidates = [value.get("leaseId"), value.get("lease_id")]
    if isinstance(metadata, Mapping):
        candidates.extend((metadata.get("leaseId"), metadata.get("lease_id")))
    found = [row for row in candidates if isinstance(row, str) and row]
    _require(bool(found) and len(set(found)) == 1, "old lease ID 원문")
    return cast(str, found[0])


def _parse_time(value: object, label: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{label} timestamp")
    return parse_lease_timestamp(cast(str, value))


def _old_elapsed(lease: Mapping[str, Any]) -> float:
    metadata = lease.get("metadata")
    rows = [lease, metadata] if isinstance(metadata, Mapping) else [lease]
    start = end = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        start = (start or row.get("acquiredUtc") or row.get("acquired_at")
                 or row.get("acquiredAt"))
        end = (end or row.get("completedUtc") or row.get("completed_at")
               or row.get("completedAt"))
    elapsed = (_parse_time(end, "completed") - _parse_time(start, "acquired")).total_seconds()
    _require(elapsed >= 0, "old lease elapsed가 음수다")
    return elapsed


def _checked_ref(value: object, label: str) -> Path:
    path = _bound(value, label)
    return path


def _training_from_completed(stage: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = stage / "manifest.json"
    evidence_path = stage / "training" / "training_evidence.json"
    owned_path = stage / "owned-result.json"
    collected_path = stage / "collected" / "collected.json"
    evidence = _read(evidence_path)
    checkpoint = evidence.get("checkpoint")
    _require(isinstance(checkpoint, dict), "training checkpoint evidence")
    runtime = {role: _ref(stage / "runtime-facts" / role / "artifact.json")
               for role in ("case", "reference")}
    training = {
        "manifest": _ref(manifest_path), "evidence": _ref(evidence_path),
        "owned_result": _ref(owned_path), "checkpoint": deepcopy(checkpoint),
        "execution": deepcopy(_read(owned_path)), "collected": _ref(collected_path),
        "runtime_facts": runtime,
    }
    return training, evidence


def finalize_completed_target(old_phase: Path, *, plan_path: Path, recovery_protocol: Path,
                              anchor: str, lab: Path, out: Path,
                              original_approval: Path, original_lease: Path,
                              inventory_path: Path | None = None,
                              normalized_recovery: Callable[[dict, dict, dict], dict] =
                              OUTCOME.normalized_recovery) -> dict[str, Any]:
    """완료된 FIRST300/fixed26 원문을 재실행 없이 새 leaf로 판정한다."""
    _require(anchor == ORIGINAL_FIRST_ANCHOR, "원 FIRST anchor")
    old_phase, out = Path(old_phase).resolve(strict=True), Path(out).resolve()
    _require(not out.exists(), "CPU finalize output은 fresh 경로여야 한다")
    plan = _read(plan_path)
    _require(plan.get("schema") == RUN.FIRST_PLAN_SCHEMA and plan.get("anchor") == anchor,
             "원 FIRST plan")
    rows = cast(list[dict[str, Any]], plan.get("targets"))
    _require(len(rows) == 4 and [(row["case_id"], row["policy"]) for row in rows]
             == list(RUN.TARGETS), "원 four-target 순서")
    row = rows[0]
    _require((row["case_id"], row["policy"], row["registered_seed"])
             == ("case-03", "fixed_eligible", 4552), "완료 target/seed")
    candidate_path = _bound(row["candidate"], "completed candidate")
    candidate = FIRST.checked_candidate(candidate_path)
    protocol = FIRST.verify_protocol(recovery_protocol, anchor=anchor,
                                     lab=Path(lab).resolve(strict=True))
    _require(_ref(recovery_protocol) == plan["recovery_protocol"], "원 recovery protocol ref")
    lease_raw = _read(original_lease)
    old_lease_id = _lease_id(lease_raw)
    RUN._approval(Path(original_approval), schema=FIRST.APPROVAL_SCHEMA,
                  candidate=candidate_path, protocol=Path(recovery_protocol),
                  anchor=anchor, seed=4552, lease_id=old_lease_id)
    target = old_phase / "targets" / "case-03_fixed_eligible"
    intent = _read(target / "intent.json")
    _require(intent.get("candidate") == _ref(candidate_path)
             and intent.get("approval") == _ref(original_approval)
             and intent.get("protocol") == _ref(recovery_protocol)
             and intent.get("lease_id") == old_lease_id
             and intent.get("registered_seed") == 4552, "원 intent 결합")
    failure = _read(target / "failure.json")
    _require(failure.get("status") == "invalid" and failure.get("error_type") == "ValueError",
             "원 ABI failure")
    training, evidence = _training_from_completed(target / "training")
    rebuilt = FIRST.TRAINING.build_manifest(
        Path(recovery_protocol), protocol, anchor, stage=target / "training", seed=4552,
        arm="FIRST", original_faults=candidate["original_faults"],
        applied_corrections=[candidate["correction"]],
        source_checkpoint_sha256=candidate["source_checkpoint_sha256"],
        run_name="five_policy_recovery_case_03_fixed_eligible")
    _require(rebuilt == _read(target / "training" / "manifest.json"), "원 training manifest")
    _require(evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False
             and evidence.get("source_checkpoint_sha256") == candidate["source_checkpoint_sha256"],
             "fresh300/load false/source CP")
    checkpoint = cast(dict[str, Any], training["checkpoint"])
    checkpoint_path = Path(cast(str, checkpoint["path"])).resolve(strict=True)
    _require(_sha(checkpoint_path) == checkpoint.get("sha256")
             == "26f9c0f03aa31f932b41bc1b507e58360208fe57db5dc7d598d158d1a5fcf915"
             and checkpoint_path.stat().st_size == 983_787, "완료 checkpoint raw")
    corrected_path = target / "evaluation" / "corrected.json"
    corrected = _read(corrected_path)
    jobs = _read(target / "evaluation" / "jobs.json")
    job_rows = jobs.get("jobs")
    _require(isinstance(job_rows, list) and len(job_rows) == 1
             and Path(job_rows[0]["checkpoint"]).resolve(strict=True) == checkpoint_path
             and Path(job_rows[0]["output"]).resolve(strict=True) == corrected_path,
             "fixed26 jobs/checkpoint/report")
    eval_owned = _read(target / "evaluation" / "owned-result.json")
    _require(FIRST.TRAINING.EXEC._owned_ok(_read(target / "training" / "owned-result.json"),
                                           FIRST.TRAINING.EXEC.WATCHDOG["train"])
             and FIRST.TRAINING.EXEC._owned_ok(eval_owned,
                                               FIRST.TRAINING.EXEC.WATCHDOG["eval"]),
             "training/fixed owned natural exit/cleanup")
    _bridge, reference_raw = FIRST.SOURCE_INPUT.checked(
        _bound(candidate["inputs"]["reference_probe_input"], "reference probe input"))
    reference_report = _read(Path(reference_raw["raw"]["fixed_result"]["path"]))["report"]
    normalized = normalized_recovery(
        corrected, reference_report, _read(FIRST.OLD_RECOVERY.BANDS)["bands"])
    verdict = cast(dict[str, Any], normalized["verdict"])
    outcome = verdict["outcome"]
    status = "HOLD" if outcome == "unknown" else outcome
    _require(status in {"recovered", "not_recovered", "HOLD"}, "normalized recovery outcome")
    inventory_ref = _ref(inventory_path) if inventory_path is not None else None
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": FIRST.RECEIPT_SCHEMA, "status": status,
        "case_id": "case-03", "policy": "fixed_eligible", "registered_seed": 4552,
        "source_seed": candidate["source_seed"], "candidate": _ref(candidate_path),
        "approval": _ref(original_approval), "protocol": _ref(recovery_protocol),
        "lease_id": old_lease_id, "confirmed_family": candidate["confirmed_family"],
        "correction": candidate["correction"], "training_mode": "fresh_restart",
        "source_checkpoint_sha256": candidate["source_checkpoint_sha256"],
        "source_checkpoint_loaded": False, "training": training,
        "new_checkpoint": deepcopy(training["checkpoint"]),
        "fixed26": {"conditions": 26, "corrected_report": corrected,
                    "reference_report_sha256": FIRST.digest(reference_report),
                    "execution": eval_owned},
        "recovery": verdict, "automatic_retry": False, "second_epoch_registered": False,
        "next_step": "second_diagnosis_required" if status == "not_recovered" else "stop",
        "continuation_provenance": {
            "kind": "verified_prior_roles_finalize", "original_failed_phase": str(old_phase),
            "original_failure": _ref(target / "failure.json"),
            "original_approval": _ref(original_approval), "original_closed_lease": _ref(original_lease),
            "original_raw_inventory": inventory_ref,
            "abi_normalization": {"source": _ref(ROOT / "evals/recovery_outcome_contract_v1.py"),
                                  "canonical_recovery_preserved": True,
                                  "inner_outcome_added_from_outer": True},
        },
    }
    receipt_path = out / "receipt.json"
    _write_new(receipt_path, receipt)
    epoch2 = None
    if status == "not_recovered":
        epoch2 = {
            "schema": FIRST.EPOCH2_SCHEMA, "status": "input_only", "case_id": "case-03",
            "policy": "fixed_eligible", "prior_receipt": _ref(receipt_path),
            "candidate": _ref(candidate_path), "checkpoint": deepcopy(training["checkpoint"]),
            "collected": deepcopy(training["collected"]),
            "runtime_facts": deepcopy(training["runtime_facts"]),
            "remaining_faults": deepcopy(evidence["remaining_faults"]),
            "new_protocol_required": True, "approval_granted": False,
            "execution_allowed": False, "automatic_second_execution": False,
        }
        _write_new(out / "epoch2-input.json", epoch2)
    result = {
        "schema": FIRST.RESULT_SCHEMA, "status": "complete", "receipt": _ref(receipt_path),
        "outcome": status, "epoch2_input": _ref(out / "epoch2-input.json") if epoch2 else None,
        "overall_goal_complete": False, "automatic_retry": False,
        "execution_kind": "verified_prior_roles_finalize",
    }
    _write_new(out / "result.json", result)
    with OUTCOME.recovery_outcome_scope():
        EXPORT_V1._training(receipt, candidate, result_path=target / "result.json",
                            lab=Path(lab), second=False, allow_fixture=False)
    return deepcopy(result)


def build_continuation_protocol(*, plan_path: Path, first_protocol: Path,
                                original_host_protocol: Path,
                                old_failed_phase: Path, finalized_result: Path,
                                old_closed_lease: Path, inventory_path: Path,
                                continuation_plan_out: Path,
                                out: Path) -> dict[str, Any]:
    """old 2 SDK와 남은 6 SDK를 새 parent source closure에 등록한다."""
    plan = _read(plan_path)
    _require(plan.get("schema") == RUN.FIRST_PLAN_SCHEMA
             and plan.get("anchor") == ORIGINAL_FIRST_ANCHOR, "원 first plan")
    _require(_ref(first_protocol) == plan.get("recovery_protocol"), "원 first protocol ref")
    lease = _read(old_closed_lease)
    finalized = _read(finalized_result)
    _require(finalized.get("schema") == FIRST.RESULT_SCHEMA
             and finalized.get("execution_kind") == "verified_prior_roles_finalize",
             "CPU finalized result")
    elapsed = _old_elapsed(lease)
    targets = [{"case_id": row["case_id"], "policy": row["policy"],
                "registered_seed": row["registered_seed"],
                "candidate_sha256": row["candidate"]["sha256"]}
               for row in plan["targets"]]
    budgets = {"api_calls": 0, "first_wall_s": FIRST_WALL_S,
               "old_lease_elapsed_s": elapsed,
               "remaining_wall_s": FIRST_WALL_S - elapsed,
               "old_sdk_executions": OLD_SDK_EXECUTIONS,
               "new_sdk_execution_cap": NEW_SDK_EXECUTION_CAP,
               "total_sdk_execution_cap": 8}
    private_plan = {
        "schema": FIRST_CONTINUATION_PLAN_SCHEMA, "status": "prepared_no_execution",
        "original_plan": _ref(plan_path), "original_first_protocol": _ref(first_protocol),
        "original_host_protocol": _ref(original_host_protocol),
        "old_failed_phase": str(Path(old_failed_phase).resolve(strict=True)),
        "old_raw_inventory": _ref(inventory_path),
        "finalized_result": _ref(finalized_result),
        "old_closed_lease": _ref(old_closed_lease), "budgets": budgets,
    }
    _write_new(continuation_plan_out, private_plan)
    value = {
        "schema": FIRST_CONTINUATION_PROTOCOL_SCHEMA, "status": "preregistered_continuation",
        "private_plan_sha256": _sha(continuation_plan_out),
        "original": {
            "first_protocol": {"path": _relative(first_protocol),
                               "sha256": _sha(first_protocol)},
            "anchor": ORIGINAL_FIRST_ANCHOR,
            "host_protocol": {"path": _relative(original_host_protocol),
                              "sha256": _sha(original_host_protocol)},
            "host_anchor": ORIGINAL_HOST_ANCHOR,
        },
        "prior": {"inventory_sha256": _sha(inventory_path),
                  "closed_lease_sha256": _sha(old_closed_lease),
                  "finalized_result_sha256": _sha(finalized_result),
                  "case_id": "case-03", "policy": "fixed_eligible",
                  "lease_id": _lease_id(lease)},
        "targets": targets, "budgets": budgets,
        "sources_lf": _source_map(), "automatic_retry": False,
        "second_phase_requires_new_registered_protocol": True,
    }
    _require(value["budgets"]["remaining_wall_s"] > 0, "continuation wall budget")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    _require("bench/private" not in encoded and "C:\\" not in encoded,
             "public continuation protocol에 private/절대 경로가 있다")
    _write_new(out, value)
    return deepcopy(value)


def verify_continuation_protocol(path: Path, *, continuation_plan_path: Path,
                                 execution_anchor: str,
                                 lab: Path) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("schema") == FIRST_CONTINUATION_PROTOCOL_SCHEMA
             and value.get("status") == "preregistered_continuation", "continuation schema")
    _verify_anchor(execution_anchor)
    relative = _relative(path)
    _verify_blob(execution_anchor, relative, _sha(path, lf=True))
    _require(value.get("sources_lf") == _source_map(), "continuation source LF closure")
    for name, digest in value["sources_lf"].items():
        _verify_blob(execution_anchor, name, digest)
    private = _read(continuation_plan_path)
    _require(private.get("schema") == FIRST_CONTINUATION_PLAN_SCHEMA
             and private.get("status") == "prepared_no_execution"
             and _sha(continuation_plan_path) == value.get("private_plan_sha256"),
             "public/private continuation plan SHA")
    original = value.get("original")
    _require(isinstance(original, dict) and original.get("anchor") == ORIGINAL_FIRST_ANCHOR,
             "원 FIRST binding")
    plan_path = _checked_ref(private["original_plan"], "original plan")
    plan = _read(plan_path)
    _require(plan.get("schema") == RUN.FIRST_PLAN_SCHEMA
             and plan.get("anchor") == ORIGINAL_FIRST_ANCHOR, "원 plan schema/anchor")
    protocol_path = _checked_ref(private["original_first_protocol"], "original first protocol")
    host_protocol = _checked_ref(private["original_host_protocol"], "original host protocol")
    _require(original.get("first_protocol") == {
                 "path": _relative(protocol_path), "sha256": _sha(protocol_path)}
             and original.get("host_protocol") == {
                 "path": _relative(host_protocol), "sha256": _sha(host_protocol)},
             "public/private original protocol refs")
    _require(original.get("host_anchor") == ORIGINAL_HOST_ANCHOR,
             "원 outer host anchor")
    host = HOST.validate_host_protocol(
        host_protocol, ORIGINAL_HOST_ANCHOR, phase="first", lab=Path(lab).resolve(strict=True))
    first_stage = host["stage_protocols"]["first"]
    _require((ROOT / first_stage["path"]).resolve() == protocol_path
             and first_stage["source_anchor"] == ORIGINAL_FIRST_ANCHOR,
             "원 outer host/first protocol 결합")
    identities = [(row["case_id"], row["policy"]) for row in value.get("targets", [])]
    seeds = [row["registered_seed"] for row in value.get("targets", [])]
    _require(identities == list(RUN.TARGETS) and seeds == [4552, 4553, 4556, 4557],
             "continuation four targets/seeds")
    _require([row["candidate_sha256"] for row in value["targets"]]
             == [row["candidate"]["sha256"] for row in plan["targets"]],
             "continuation candidate SHA")
    budgets = value.get("budgets")
    _require(isinstance(budgets, dict) and budgets.get("api_calls") == 0
             and budgets.get("first_wall_s") == FIRST_WALL_S
             and budgets.get("old_sdk_executions") == 2
             and budgets.get("new_sdk_execution_cap") == 6
             and budgets.get("total_sdk_execution_cap") == 8
             and budgets.get("remaining_wall_s")
             == FIRST_WALL_S - budgets.get("old_lease_elapsed_s"), "continuation budgets")
    _require(private.get("budgets") == budgets, "public/private budgets")
    lease_path = _checked_ref(private["old_closed_lease"], "old closed lease")
    lease = _read(lease_path)
    _require(_sha(lease_path) == value["prior"]["closed_lease_sha256"]
             and _lease_id(lease) == value["prior"]["lease_id"]
             and _old_elapsed(lease) == budgets["old_lease_elapsed_s"], "old lease elapsed/id")
    finalized_path = _checked_ref(private["finalized_result"], "finalized result")
    finalized = _read(finalized_path)
    _require(finalized.get("schema") == FIRST.RESULT_SCHEMA
             and finalized.get("execution_kind") == "verified_prior_roles_finalize",
             "verified prior finalize")
    inventory = _checked_ref(private["old_raw_inventory"], "old raw inventory")
    _require(_sha(inventory) == value["prior"]["inventory_sha256"]
             and _sha(finalized_path) == value["prior"]["finalized_result_sha256"],
             "prior public/private SHA")
    return {"protocol": deepcopy(value), "plan": deepcopy(private)}


def run_first(plan_path: Path, continuation_protocol: Path,
              continuation_plan: Path, approvals: Mapping[str, Path] | None, *, execution_anchor: str,
              lab: Path, isaac_python: Path, out: Path, runner: Callable,
              wait: Callable, lease_reader: Callable[[], Mapping[str, Any]],
              approver: str | None = None,
              candidate_executor: Callable[..., dict[str, Any]] = FIRST.execute_candidate,
              protocol_verifier: Callable[..., dict[str, Any]] = verify_continuation_protocol,
              use_host_binding: bool = True) -> dict[str, Any]:
    """검증된 prior 한 행을 재사용하고 남은 세 후보만 새 lease로 실행한다."""
    material = protocol_verifier(
        continuation_protocol, continuation_plan_path=continuation_plan,
        execution_anchor=execution_anchor, lab=lab)
    overlay, private = material["protocol"], material["plan"]
    _require(_ref(plan_path) == private["original_plan"], "overlay/original plan")
    plan = _read(plan_path)
    out = Path(out).resolve()
    _require(not out.exists(), "continuation first output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    lease = RUN._lease(lease_reader)
    started = time.monotonic()
    original_protocol = _bound(plan["recovery_protocol"], "original recovery protocol")
    old_result_path = _bound(private["finalized_result"], "finalized target")
    old_result = _read(old_result_path)
    old_lease = overlay["prior"]["lease_id"]
    records: list[dict[str, Any]] = []
    executed = 0
    binding = HOST.retry_lease_binding() if use_host_binding else nullcontext()
    with binding, OUTCOME.recovery_outcome_scope():
        for index, row in enumerate(plan["targets"]):
            if index == 0:
                _require((row["case_id"], row["policy"]) == ("case-03", "fixed_eligible"),
                         "prior target position")
                records.append({**row, "outcome": old_result["outcome"],
                                "result": _ref(old_result_path),
                                "epoch2_input": old_result.get("epoch2_input"),
                                "lease_id": old_lease,
                                "execution_kind": "verified_prior_roles_finalize"})
                continue
            _require(executed * 2 + 2 <= overlay["budgets"]["new_sdk_execution_cap"]
                     and time.monotonic() - started + 3600
                     <= overlay["budgets"]["remaining_wall_s"], "new FIRST SDK/wall budget")
            candidate_path = _bound(row["candidate"], "first candidate")
            candidate = FIRST.checked_candidate(candidate_path)
            _require(candidate.get("status") == "validated_candidate", "원 후보가 HOLD로 바뀌었다")
            key = f"{row['case_id']}:{row['policy']}"
            approval = RUN._approval_for(
                candidate_path, original_protocol, schema=FIRST.APPROVAL_SCHEMA,
                anchor=ORIGINAL_FIRST_ANCHOR, seed=row["registered_seed"], lease=lease,
                approvals=approvals, key=key, approver=approver, root=out)
            RUN._approval(approval, schema=FIRST.APPROVAL_SCHEMA, candidate=candidate_path,
                          protocol=original_protocol, anchor=ORIGINAL_FIRST_ANCHOR,
                          seed=row["registered_seed"], lease_id=lease["leaseId"])
            target = out / "targets" / key.replace(":", "_")
            result = candidate_executor(
                candidate_path, original_protocol, approval, anchor=ORIGINAL_FIRST_ANCHOR,
                lab=lab, isaac_python=isaac_python, registered_seed=row["registered_seed"],
                out=target, runner=runner, wait=wait, lease_reader=lease_reader)
            RUN._same_lease(lease, lease_reader)
            executed += 1
            records.append({**row, "outcome": result["outcome"],
                            "result": _ref(target / "result.json"),
                            "epoch2_input": result.get("epoch2_input"),
                            "lease_id": lease["leaseId"],
                            "execution_kind": "new_registered_execution"})
    _require(len(records) == 4 and executed == 3, "continuation은 prior1/new3이어야 한다")
    new_wall = time.monotonic() - started
    result = {
        "schema": FIRST_CONTINUATION_RESULT_SCHEMA, "status": "complete",
        "plan": _ref(plan_path), "phase_protocol": _ref(continuation_protocol),
        "continuation_plan": _ref(continuation_plan),
        "execution_anchor": execution_anchor, "anchor": ORIGINAL_FIRST_ANCHOR,
        "lease_id": lease["leaseId"], "records": records,
        "old_sdk_executions": 2, "new_sdk_executions": executed * 2,
        "total_sdk_execution_upper_bound": 2 + executed * 2,
        "old_wall_s": overlay["budgets"]["old_lease_elapsed_s"],
        "new_parent_wall_s": new_wall,
        "total_parent_wall_s": overlay["budgets"]["old_lease_elapsed_s"] + new_wall,
        "api_calls": 0, "automatic_epoch2": False,
        "prior_reuse_count": 1, "new_executed_targets": executed,
        "second_phase_requires_new_registered_protocol": any(
            row["outcome"] == "not_recovered" for row in records),
    }
    _write_new(out / "result.json", result)
    return deepcopy(result)


def run_first_operational(plan_path: Path, *, continuation_protocol: Path,
                          continuation_plan: Path,
                          execution_anchor: str, lab: Path, isaac_python: Path,
                          out: Path, approver: str) -> dict[str, Any]:
    """bounded retry lease reader와 기존 owned runner로 continuation FIRST를 실행한다."""
    return run_first(
        plan_path, continuation_protocol, continuation_plan, None,
        execution_anchor=execution_anchor,
        lab=lab, isaac_python=isaac_python, out=out, runner=RUN.OWNED.run_owned,
        wait=RUN.WAIT._lease_wait, lease_reader=lambda: RUN.LEASE.lease_context(),
        approver=approver)


def prepare_second(first_result_path: Path, *, loop_protocol: Path, anchor: str,
                   lab: Path, out: Path,
                   source_preparer: Callable[..., dict[str, Any]] = SECOND.prepare_source,
                   source_checker: Callable[..., dict[str, Any]] = SECOND.checked_source,
                   protocol_verifier: Callable[..., dict[str, Any]] =
                   SECOND.verify_loop_protocol) -> dict[str, Any]:
    first = _read(first_result_path)
    _require(first.get("schema") == FIRST_CONTINUATION_RESULT_SCHEMA
             and first.get("status") == "complete", "FIRST continuation v2 result")
    _require(len(first.get("records", [])) == 4
             and {(row["case_id"], row["policy"]) for row in first["records"]}
             == set(RUN.TARGETS), "FIRST v2 exact records")
    protocol = protocol_verifier(loop_protocol, anchor, lab=lab)
    out = Path(out).resolve()
    _require(not out.exists(), "second plan output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    rows = []
    for first_row in first["records"]:
        if first_row["outcome"] != "not_recovered":
            rows.append({"case_id": first_row["case_id"], "policy": first_row["policy"],
                         "status": "not_in_second_subset", "source": None})
            continue
        result = _read(_bound(first_row["result"], "FIRST leaf result"))
        _require(result.get("outcome") == "not_recovered"
                 and result.get("epoch2_input") == first_row.get("epoch2_input"),
                 "FIRST receipt/epoch2 결합")
        epoch2 = _bound(first_row["epoch2_input"], "epoch2 input")
        source = source_preparer(epoch2, lab=lab)
        source_path = out / "sources" / f"{source['case_id']}_{source['policy']}.json"
        _write_new(source_path, source)
        source_checker(source)
        key = f"{source['case_id']}:{source['policy']}"
        _require(protocol["source_digests"].get(key) == SECOND.digest(source),
                 "second source가 등록 subset에 없다")
        rows.append({"case_id": source["case_id"], "policy": source["policy"],
                     "status": "registered", "source": _ref(source_path),
                     "registered_seed": protocol["second_seeds"][key]})
    plan = {
        "schema": RUN.SECOND_PLAN_SCHEMA, "status": "prepared_no_execution",
        "first_result": _ref(first_result_path), "loop_protocol": _ref(loop_protocol),
        "anchor": anchor, "records": rows,
        "registered_count": sum(row["status"] == "registered" for row in rows),
        "api_cap": 32, "probe_pair_cap": 12, "probe_sdk_cap": 24,
        "training_fixed_sdk_cap": 8, "total_sdk_cap": 32,
        "automatic_retry": False, "approval_granted": False,
    }
    _write_new(out / "plan.json", plan)
    return deepcopy(plan)


def build_second_compatibility_protocol(*, second_plan: Path, host_protocol: Path,
                                        out: Path) -> dict[str, Any]:
    plan = _read(second_plan)
    _require(plan.get("schema") == RUN.SECOND_PLAN_SCHEMA, "second plan")
    value = {
        "schema": SECOND_COMPATIBILITY_PROTOCOL_SCHEMA,
        "status": "preregistered_second_compatibility",
        "second_plan_sha256": _sha(second_plan),
        "loop_protocol": {"path": _relative(_bound(plan["loop_protocol"], "loop protocol")),
                          "sha256": plan["loop_protocol"]["sha256"]},
        "host_protocol": {"path": _relative(host_protocol), "sha256": _sha(host_protocol)},
        "sources_lf": _source_map(),
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    _require("bench/private" not in encoded and "C:\\" not in encoded,
             "public second compatibility protocol에 private/절대 경로가 있다")
    _write_new(out, value)
    return deepcopy(value)


def verify_second_compatibility_protocol(path: Path, *, anchor: str,
                                         second_plan_path: Path,
                                         host_protocol: Path) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("schema") == SECOND_COMPATIBILITY_PROTOCOL_SCHEMA
             and value.get("status") == "preregistered_second_compatibility",
             "second compatibility schema")
    _verify_anchor(anchor)
    _verify_blob(anchor, _relative(path), _sha(path, lf=True))
    _require(value.get("sources_lf") == _source_map(), "second compatibility sources")
    for name, digest in value["sources_lf"].items():
        _verify_blob(anchor, name, digest)
    plan = _read(second_plan_path)
    _require(plan.get("schema") == RUN.SECOND_PLAN_SCHEMA
             and value.get("second_plan_sha256") == _sha(second_plan_path)
             and value.get("loop_protocol") == {
                 "path": _relative(_bound(plan["loop_protocol"], "loop protocol")),
                 "sha256": plan["loop_protocol"]["sha256"]},
             "second compatibility plan/loop")
    _require(value.get("host_protocol") == {
                 "path": _relative(host_protocol), "sha256": _sha(host_protocol)},
             "second compatibility host protocol")
    return deepcopy(value)


def run_second_operational(plan_path: Path, *, host_protocol: Path, host_anchor: str,
                           compatibility_protocol: Path, compatibility_anchor: str,
                           transport_root: Path, lab: Path, isaac_python: Path,
                           out: Path, approver: str) -> dict[str, Any]:
    """별도 second host와 compatibility overlay를 검증한 뒤 동결 실행기를 호출한다."""
    _require(host_anchor == compatibility_anchor, "second host/compat anchor가 다르다")
    compat = verify_second_compatibility_protocol(
        compatibility_protocol, anchor=compatibility_anchor,
        second_plan_path=plan_path, host_protocol=host_protocol)
    _require(compat["second_plan_sha256"] == _sha(plan_path), "second compat plan")
    HOST.validate_host_protocol(host_protocol, host_anchor, phase="second", lab=lab)
    with HOST.retry_lease_binding(), OUTCOME.recovery_outcome_scope():
        return RUN.run_second_operational(
            plan_path, transport_root=transport_root, lab=lab,
            isaac_python=isaac_python, out=out, approver=approver)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    finalize = sub.add_parser("finalize-completed-target")
    for name in ("old_phase", "plan", "recovery_protocol", "anchor", "lab", "out",
                 "original_approval", "original_lease", "inventory"):
        finalize.add_argument("--" + name.replace("_", "-"), required=True)
    build = sub.add_parser("build-continuation")
    for name in ("plan", "first_protocol", "original_host_protocol", "old_failed_phase",
                 "finalized_result", "old_closed_lease", "inventory",
                 "continuation_plan_out", "out"):
        build.add_argument("--" + name.replace("_", "-"), required=True)
    first = sub.add_parser("run-first")
    for name in ("plan", "continuation_protocol", "continuation_plan", "execution_anchor", "lab",
                 "isaac_python", "out"):
        first.add_argument("--" + name.replace("_", "-"), required=True)
    first.add_argument("--approver", choices=[RUN.APPROVER], required=True)
    prepare = sub.add_parser("prepare-second")
    for name in ("first_result", "loop_protocol", "anchor", "lab", "out"):
        prepare.add_argument("--" + name.replace("_", "-"), required=True)
    second = sub.add_parser("run-second")
    for name in ("plan", "host_protocol", "host_anchor", "compatibility_protocol",
                 "compatibility_anchor", "transport_root", "lab", "isaac_python", "out"):
        second.add_argument("--" + name.replace("_", "-"), required=True)
    second.add_argument("--approver", choices=[RUN.APPROVER], required=True)
    compat = sub.add_parser("build-second-compatibility")
    for name in ("second_plan", "host_protocol", "out"):
        compat.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "finalize-completed-target":
        finalize_completed_target(
            Path(args.old_phase), plan_path=Path(args.plan),
            recovery_protocol=Path(args.recovery_protocol), anchor=args.anchor,
            lab=Path(args.lab), out=Path(args.out),
            original_approval=Path(args.original_approval),
            original_lease=Path(args.original_lease), inventory_path=Path(args.inventory))
    elif args.command == "build-continuation":
        build_continuation_protocol(
            plan_path=Path(args.plan), first_protocol=Path(args.first_protocol),
            original_host_protocol=Path(args.original_host_protocol),
            old_failed_phase=Path(args.old_failed_phase),
            finalized_result=Path(args.finalized_result),
            old_closed_lease=Path(args.old_closed_lease),
            inventory_path=Path(args.inventory),
            continuation_plan_out=Path(args.continuation_plan_out), out=Path(args.out))
    elif args.command == "run-first":
        run_first_operational(
            Path(args.plan), continuation_protocol=Path(args.continuation_protocol),
            continuation_plan=Path(args.continuation_plan),
            execution_anchor=args.execution_anchor, lab=Path(args.lab),
            isaac_python=Path(args.isaac_python), out=Path(args.out),
            approver=args.approver)
    elif args.command == "prepare-second":
        prepare_second(Path(args.first_result), loop_protocol=Path(args.loop_protocol),
                       anchor=args.anchor, lab=Path(args.lab), out=Path(args.out))
    elif args.command == "run-second":
        run_second_operational(
            Path(args.plan), host_protocol=Path(args.host_protocol),
            host_anchor=args.host_anchor,
            compatibility_protocol=Path(args.compatibility_protocol),
            compatibility_anchor=args.compatibility_anchor,
            transport_root=Path(args.transport_root), lab=Path(args.lab),
            isaac_python=Path(args.isaac_python), out=Path(args.out), approver=args.approver)
    else:
        build_second_compatibility_protocol(
            second_plan=Path(args.second_plan), host_protocol=Path(args.host_protocol),
            out=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
