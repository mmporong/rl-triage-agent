# pyright: reportMissingImports=false
"""완료된 SECOND 진단을 보존하고 미실행 training 하나만 재개한다."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import recovery_epoch2_training_boundary_v1 as TRAIN_BOUNDARY
import run_five_policy_recovery_resume_v8 as V8

V5 = V8.V5
C = V8.C
PARENT = V5.V3.CONT
RUN = PARENT.RUN
SECOND = PARENT.SECOND
CACHE = V5.V3.CACHE
SCHEMA = "five_policy_recovery_second_training_only_retry_protocol_v1"
PLAN_SCHEMA = "five_policy_recovery_second_training_only_retry_plan_v1"
SOURCE_FILES = tuple(sorted({
    *V8.SOURCE_FILES,
    *TRAIN_BOUNDARY.SOURCE_FILES,
    "bench/run_five_policy_recovery_resume_v9.py",
    "tests/test_five_policy_recovery_resume_v9.py",
}))
EXPECTED_FILES = 902
EXPECTED_INVENTORY_DIGEST = (
    "20d9ed4c8fec5fa9dac30aab7254b593517d1492fc414781d339238e25710c1c")
EXPECTED_PREVIOUS_WALL_US = 3_833_489_176
EXPECTED_FAILURE_WALL_US = 6_950_256_152
EXPECTED_CUMULATIVE_WALL_US = 10_783_745_328
TOTAL_WALL_US = 43_200_000_000
TARGET_KEY = "case-07:logs_only"
TARGET_SEED = 5557


def _source_map() -> dict[str, str]:
    return {name: C._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _inventory(phase: Path) -> dict[str, dict[str, Any]]:
    root = Path(phase).resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}

    def failed(error: OSError) -> None:
        raise ValueError("training 실패 원문 읽기 실패") from error

    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=failed):
        base = Path(directory)
        C._require(not any((base / name).is_symlink()
                           or getattr(base / name, "is_junction", lambda: False)()
                           for name in [*subdirs, *names]),
                   "training 실패 원문 link 거부")
        for name in names:
            path = base / name
            files[path.relative_to(root).as_posix()] = {
                "sha256": C._sha(path), "size_bytes": path.stat().st_size}
    return dict(sorted(files.items()))


def _failure_state(phase: Path, inventory_path: Path,
                   closed_lease: Path) -> dict[str, Any]:
    root = Path(phase).resolve(strict=True)
    inventory = C._read(inventory_path)
    lease = C._read(closed_lease)
    files = _inventory(root)
    C._require(inventory.get("schema") == "epoch2_training_binding_failure_inventory_v1"
               and inventory.get("status") == "sealed"
               and inventory.get("files") == files
               and inventory.get("file_count") == EXPECTED_FILES
               and inventory.get("inventory_digest") == EXPECTED_INVENTORY_DIGEST
               == V5.C.SECOND.digest(files),
               "training 실패 inventory가 다르다")
    error = C._read(root / "actual-v1/parent-error.json")
    original_failure = inventory.get("original_failure")
    C._require(isinstance(original_failure, dict)
               and original_failure.get("status") == "invalid"
               and all(error.get(name) == original_failure.get(name)
                       for name in ("automatic_retry", "error", "error_type"))
               and error.get("error") == "protocol에 guarded training source closure가 없다"
               and C._read(root / "exit.json").get("exit_code") == 1,
               "training 실패 오류/exit가 다르다")
    C._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == C.RUN.LEASE.OWNER
               and C._lease_id(lease) == inventory.get("lease_id")
               and C._sha(closed_lease) == inventory.get("closed_lease_sha256"),
               "training 실패 lease가 다르다")
    wall_us = math.ceil((C._old_elapsed(lease) + 0.000001) * 1_000_000)
    C._require(wall_us == inventory.get("old_wall_upper_bound_us")
               == EXPECTED_FAILURE_WALL_US
               and inventory.get("previous_wall_us") == EXPECTED_PREVIOUS_WALL_US
               and inventory.get("cumulative_wall_us") == EXPECTED_CUMULATIVE_WALL_US
               and round(float(inventory["remaining_wall_s"]) * 1_000_000)
               == TOTAL_WALL_US - EXPECTED_CUMULATIVE_WALL_US,
               "training 실패 wall 상계가 다르다")
    records = inventory.get("records")
    C._require(isinstance(records, list) and len(records) == 4
               and [row.get("key") for row in records] == [
                   "case-03_fixed_eligible", "case-03_logs_only",
                   "case-07_fixed_eligible", "case-07_logs_only"],
               "training 실패 records가 다르다")
    records = cast(list[dict[str, Any]], records)
    for index, row in enumerate(records):
        candidate = C._bound(row["candidate"], "old second candidate")
        diagnosis = C._bound(row["diagnosis"], "old second diagnosis")
        C._require(C._sha(candidate) == row["candidate"]["sha256"]
                   and C._sha(diagnosis) == row["diagnosis"]["sha256"]
                   and row.get("candidate_status")
                   == ("validated_candidate" if index == 3 else "HOLD"),
                   "training 실패 candidate/diagnosis ref가 다르다")
    target = records[-1]
    C._require(target.get("confirmed_family") == "optimizer_update_count"
               and target.get("second_correction") == "restore_configured_ppo_epochs"
               and target.get("probes") == [
                   {"probe": "P_episode", "outcome": "normal"},
                   {"probe": "P_value", "outcome": "normal"},
                   {"probe": "P_optimizer_updates", "outcome": "abnormal"}],
               "미실행 training 후보가 다르다")
    C._require(inventory.get("API_started") == 6
               and inventory.get("prior_probe_physical_roles") == 2
               and inventory.get("new_probe_physical_roles") == 22
               and inventory.get("new_training_physical_roles") == 0
               and inventory.get("cumulative_physical_roles") == 24
               and inventory.get("remaining_physical_cap") == 8
               and inventory.get("cumulative_sdk_cap") == 32
               and inventory.get("model_or_probe_rerun_allowed") is False,
               "training 실패 API/SDK 상계가 다르다")
    return {
        "inventory_digest": EXPECTED_INVENTORY_DIGEST,
        "inventory_sha256": C._sha(inventory_path),
        "closed_lease_sha256": C._sha(closed_lease),
        "lease_id": inventory["lease_id"],
        "records": deepcopy(records),
        "old_wall_us": wall_us,
        "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US,
        "remaining_wall_s": inventory["remaining_wall_s"],
        "prior_api_calls": 6,
        "prior_physical_roles": 24,
        "prior_nim_calls": 6,
        "new_training_physical_cap": 2,
        "new_api_cap": 0,
    }


def _public_failure(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    for row in result["records"]:
        row["candidate"] = {"sha256": row["candidate"]["sha256"]}
        row["diagnosis"] = {"sha256": row["diagnosis"]["sha256"]}
    return result


def build_protocol(*, parent_protocol: Path, parent_anchor: str,
                   parent_private_plan: Path, old_phase: Path,
                   old_inventory: Path, old_closed_lease: Path,
                   training_protocol: Path, training_anchor: str,
                   private_out: Path, out: Path) -> dict[str, Any]:
    parent = V8.verify_protocol(parent_protocol, anchor=parent_anchor,
                                private_plan=parent_private_plan)
    _v7, _v6, _v5, second_plan, _sources = V8._parents(parent)
    second = C._read(second_plan)
    loop = C._bound(second["loop_protocol"], "loop protocol")
    TRAIN_BOUNDARY.verify_protocol(
        training_protocol, training_anchor, loop_protocol=loop,
        loop_anchor=second["anchor"])
    failure = _failure_state(old_phase, old_inventory, old_closed_lease)
    plan = {
        "schema": PLAN_SCHEMA,
        "parent_protocol": C._ref(parent_protocol),
        "parent_anchor": parent_anchor,
        "parent_private_plan": C._ref(parent_private_plan),
        "second_plan": C._ref(second_plan),
        "old_phase": str(Path(old_phase).resolve(strict=True)),
        "old_inventory": C._ref(old_inventory),
        "old_closed_lease": C._ref(old_closed_lease),
        "training_protocol": C._ref(training_protocol),
        "training_anchor": training_anchor,
        "prior_failure": failure,
        "approval_granted": False,
        "automatic_retry": False,
    }
    value = {
        "schema": SCHEMA,
        "status": "preregistered_training_only_retry",
        "parent": {"path": C._relative(parent_protocol),
                   "sha256": C._sha(parent_protocol)},
        "training_component": {"path": C._relative(training_protocol),
                               "sha256": C._sha(training_protocol),
                               "anchor": training_anchor},
        "sources_lf": _source_map(),
        "prior": _public_failure(failure),
        "budget": {
            "api_cap": 32, "prior_api_calls": 6, "new_api_calls": 0,
            "sdk_cap": 32, "prior_physical_sdk": 24,
            "new_training_fixed_sdk_cap": 2, "cumulative_sdk_upper_bound": 26,
            "nim_cap": 32, "prior_nim_calls": 6, "new_nim_calls": 0,
            "cumulative_parent_wall_cap_s": 43200,
            "remaining_parent_wall_s": 32416.254672,
        },
        "preserved_diagnosis_count": 4,
        "preserved_probe_physical_roles": 24,
        "model_or_probe_rerun_allowed": False,
        "semantic_retry": False,
        "epoch3_allowed": False,
        "binding_change": "guarded_training_component_only",
    }
    C._write_new(private_out, plan)
    C._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str,
                    private_plan: Path) -> dict[str, Any]:
    value, plan = C._read(path), C._read(private_plan)
    C._require(value.get("schema") == SCHEMA
               and value.get("status") == "preregistered_training_only_retry"
               and plan.get("schema") == PLAN_SCHEMA
               and plan.get("approval_granted") is False
               and plan.get("automatic_retry") is False,
               "training-only protocol/plan schema")
    parent_path = C._bound(plan["parent_protocol"], "parent protocol")
    parent_private = C._bound(plan["parent_private_plan"], "parent private plan")
    parent = V8.verify_protocol(parent_path, anchor=plan["parent_anchor"],
                                private_plan=parent_private)
    _v7, _v6, _v5, second_plan, _sources = V8._parents(parent)
    C._require(C._ref(second_plan) == plan["second_plan"], "원 second plan 결합")
    second = C._read(second_plan)
    training_path = C._bound(plan["training_protocol"], "training protocol")
    TRAIN_BOUNDARY.verify_protocol(
        training_path, plan["training_anchor"],
        loop_protocol=C._bound(second["loop_protocol"], "loop protocol"),
        loop_anchor=second["anchor"])
    failure = _failure_state(
        Path(plan["old_phase"]), C._bound(plan["old_inventory"], "old inventory"),
        C._bound(plan["old_closed_lease"], "old closed lease"))
    C._require(plan["prior_failure"] == failure, "training 실패 private 결합")
    C._verify_anchor(anchor)
    C._verify_blob(anchor, C._relative(path), C._sha(path, lf=True))
    C._require(value.get("sources_lf") == _source_map(), "training-only source SHA")
    for name, digest in value["sources_lf"].items():
        C._verify_blob(anchor, name, digest)
    expected = {
        "schema": SCHEMA, "status": "preregistered_training_only_retry",
        "parent": {"path": C._relative(parent_path), "sha256": C._sha(parent_path)},
        "training_component": {"path": C._relative(training_path),
                               "sha256": C._sha(training_path),
                               "anchor": plan["training_anchor"]},
        "sources_lf": _source_map(), "prior": _public_failure(failure),
        "budget": {
            "api_cap": 32, "prior_api_calls": 6, "new_api_calls": 0,
            "sdk_cap": 32, "prior_physical_sdk": 24,
            "new_training_fixed_sdk_cap": 2, "cumulative_sdk_upper_bound": 26,
            "nim_cap": 32, "prior_nim_calls": 6, "new_nim_calls": 0,
            "cumulative_parent_wall_cap_s": 43200,
            "remaining_parent_wall_s": 32416.254672},
        "preserved_diagnosis_count": 4, "preserved_probe_physical_roles": 24,
        "model_or_probe_rerun_allowed": False, "semantic_retry": False,
        "epoch3_allowed": False, "binding_change": "guarded_training_component_only",
    }
    C._require(len(SOURCE_FILES) == 40 and value == expected,
               "training-only budget/science")
    return deepcopy(plan)


def _checked_records(failure: dict[str, Any], second_plan: Path
                     ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    second = C._read(second_plan)
    plan_rows = second.get("records")
    C._require(isinstance(plan_rows, list) and len(plan_rows) == 4,
               "원 second plan records")
    plan_rows = cast(list[dict[str, Any]], plan_rows)
    checked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan_row, old_row in zip(plan_rows, failure["records"], strict=True):
        key = f"{plan_row['case_id']}:{plan_row['policy']}"
        C._require(old_row["key"] == key.replace(":", "_")
                   and plan_row.get("status") == "registered",
                   "원 second plan/실패 record 순서가 다르다")
        candidate_path = C._bound(old_row["candidate"], "old second candidate")
        candidate = SECOND.checked_second_candidate(candidate_path)
        C._require(candidate["case_id"] == plan_row["case_id"]
                   and candidate["policy"] == plan_row["policy"]
                   and candidate["inputs"]["source"] == plan_row["source"]
                   and candidate["inputs"]["diagnosis_result"] == old_row["diagnosis"]
                   and candidate["status"] == old_row["candidate_status"],
                   "원 second candidate/plan/diagnosis가 다르다")
        checked.append((deepcopy(plan_row), candidate))
    C._require(checked[-1][0]["registered_seed"] == TARGET_SEED
               and checked[-1][1]["status"] == "validated_candidate"
               and all(candidate["status"] == "HOLD"
                       for _row, candidate in checked[:-1]),
               "유일 미실행 training 대상이 다르다")
    return checked


def execute(*, protocol: Path, anchor: str, private_plan: Path,
            lab: Path, isaac_python: Path, out: Path,
            approver: str) -> dict[str, Any]:
    started = time.monotonic()
    with CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        parent = V8.verify_protocol(C._bound(plan["parent_protocol"], "parent protocol"),
                                    anchor=plan["parent_anchor"],
                                    private_plan=C._bound(
                                        plan["parent_private_plan"], "parent private plan"))
        v7, _v6, v5, second_plan, sources = V8._parents(parent)
    target = Path(out).resolve()
    C._require(not target.exists(), "training-only output은 fresh 경로여야 한다")
    with ExitStack() as stack:
        V8._scopes(v7, v5, sources, stack)
        stack.enter_context(V5.V4.remaining_wall_scope(plan["prior_failure"]["remaining_wall_s"]))
        with CACHE.proof_scope(repo=ROOT):
            checked = _checked_records(plan["prior_failure"], second_plan)
        second = C._read(second_plan)
        loop_path = C._bound(second["loop_protocol"], "loop protocol")
        training_path = C._bound(plan["training_protocol"], "training protocol")
        target.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        executed = 0
        with PARENT.HOST.retry_lease_binding(), PARENT.OUTCOME.recovery_outcome_scope():
            lease = RUN._lease(RUN.LEASE.lease_context)

            def lease_reader() -> dict[str, Any]:
                C._require(time.monotonic() - started
                           < plan["prior_failure"]["remaining_wall_s"],
                           "training-only 누적 wall 상한 초과")
                current = RUN.LEASE.lease_context()
                RUN._same_lease(lease, lambda: current)
                return current

            for row, candidate in checked:
                key = f"{row['case_id']}:{row['policy']}"
                old = next(item for item in plan["prior_failure"]["records"]
                           if item["key"] == key.replace(":", "_"))
                base = {**row, "diagnosis": deepcopy(old["diagnosis"]),
                        "candidate": deepcopy(old["candidate"])}
                if candidate["status"] == "HOLD":
                    records.append({**base, "outcome": "HOLD", "result": None})
                    continue
                C._require(key == TARGET_KEY and row["registered_seed"] == TARGET_SEED
                           and executed == 0, "등록되지 않은 training 실행")
                watchdogs = SECOND.TRAINING.EXEC.WATCHDOG
                reserve = watchdogs["train"] + watchdogs["eval"] + 2 * watchdogs["gpu_wait"]
                C._require(time.monotonic() - started + reserve
                           <= plan["prior_failure"]["remaining_wall_s"],
                           "training-only 학습·평가 wall 예약 부족")
                approval = RUN._approval_for(
                    C._bound(old["candidate"], "old second candidate"), loop_path,
                    schema=SECOND.SECOND_APPROVAL_SCHEMA, anchor=second["anchor"],
                    seed=TARGET_SEED, lease=lease, approvals=None, key=key,
                    approver=approver, root=target)
                RUN._approval(
                    approval, schema=SECOND.SECOND_APPROVAL_SCHEMA,
                    candidate=C._bound(old["candidate"], "old second candidate"),
                    protocol=loop_path, anchor=second["anchor"], seed=TARGET_SEED,
                    lease_id=lease["leaseId"])
                result_dir = target / "recovery" / key.replace(":", "_")
                with TRAIN_BOUNDARY.training_boundary_scope(
                        loop_protocol=loop_path, loop_anchor=second["anchor"],
                        training_protocol=training_path,
                        training_anchor=plan["training_anchor"], lab=lab):
                    result = SECOND.execute_second_candidate(
                        C._bound(old["candidate"], "old second candidate"), loop_path,
                        approval, anchor=second["anchor"], lab=lab,
                        isaac_python=isaac_python, registered_seed=TARGET_SEED,
                        out=result_dir, runner=RUN.OWNED.run_owned,
                        wait=RUN.WAIT._lease_wait, lease_reader=lease_reader)
                RUN._same_lease(lease, lease_reader)
                executed += 1
                records.append({**base, "outcome": result["outcome"],
                                "result": C._ref(result_dir / "result.json")})
        C._require(len(records) == 4 and executed == 1,
                   "training-only 결과 수가 다르다")
        result = {
            "schema": RUN.SECOND_RESULT_SCHEMA, "status": "complete",
            "plan": C._ref(second_plan), "lease_id": lease["leaseId"],
            "records": records, "probe_pairs": 12,
            "probe_sdk_upper_bound": 24, "executed_second_targets": 1,
            "training_fixed_sdk_upper_bound": 2, "total_sdk_upper_bound": 26,
            "transport_status": {"preserved_api_calls": 6, "new_api_calls": 0},
            "automatic_retry": False, "epoch3_allowed": False,
            "parent_wall_s": time.monotonic() - started,
            "prior_parent_wall_us": EXPECTED_CUMULATIVE_WALL_US,
            "prior_diagnoses_preserved": 4, "prior_probe_pairs_preserved": 12,
            "prior_physical_roles": 24, "new_physical_roles_upper_bound": 2,
            "prior_nim_calls": 6, "new_nim_calls": 0,
            "model_or_probe_reruns": 0, "new_api_calls": 0,
        }
        C._write_new(target / "result.json", result)
        return deepcopy(result)


def export(*, protocol: Path, anchor: str, private_plan: Path,
           first_result: Path, second_result: Path, lab: Path,
           out: Path) -> dict[str, Any]:
    with CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        parent = V8.verify_protocol(C._bound(plan["parent_protocol"], "parent protocol"),
                                    anchor=plan["parent_anchor"],
                                    private_plan=C._bound(
                                        plan["parent_private_plan"], "parent private plan"))
        v7, v6, v5, second_plan, sources = V8._parents(parent)
    second = C._read(second_plan)
    loop_path = C._bound(second["loop_protocol"], "loop protocol")
    with ExitStack() as stack:
        V8._scopes(v7, v5, sources, stack)
        stack.enter_context(TRAIN_BOUNDARY.training_boundary_scope(
            loop_protocol=loop_path, loop_anchor=second["anchor"],
            training_protocol=C._bound(plan["training_protocol"], "training protocol"),
            training_anchor=plan["training_anchor"], lab=lab))
        with CACHE.proof_scope(repo=ROOT):
            return V8.V5.EXPORT.export(
                first_result, out, lab=lab, second_result=second_result,
                second_compatibility_protocol=C._bound(
                    v5["compatibility_protocol"], "compatibility"),
                second_compatibility_anchor=v6["parent_anchor"],
                second_host_protocol=C._bound(v5["host_protocol"], "host protocol"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    export_parser = sub.add_parser("export")
    for item in (run, export_parser):
        item.add_argument("--protocol", type=Path, required=True)
        item.add_argument("--anchor", required=True)
        item.add_argument("--private-plan", type=Path, required=True)
        item.add_argument("--lab", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
    run.add_argument("--isaac-python", type=Path, required=True)
    run.add_argument("--approver", choices=[RUN.APPROVER], required=True)
    export_parser.add_argument("--first-result", type=Path, required=True)
    export_parser.add_argument("--second-result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            execute(protocol=args.protocol, anchor=args.anchor,
                    private_plan=args.private_plan, lab=args.lab,
                    isaac_python=args.isaac_python, out=args.out,
                    approver=args.approver)
        else:
            export(protocol=args.protocol, anchor=args.anchor,
                   private_plan=args.private_plan, first_result=args.first_result,
                   second_result=args.second_result, lab=args.lab, out=args.out)
    except BaseException as error:
        V5._record_failure(args.out, args.command, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
