# pyright: reportMissingImports=false
"""완료된 첫 pair 영수증을 한 번 재사용해 SECOND를 재개한다."""
from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import recovery_receipt_snapshot_routes_v1 as ROUTE
import recovery_recorded_pair_reuse_v1 as REUSER
import run_five_policy_recovery_resume_v6 as V6

V5 = V6.V5
C = V6.C
SCHEMA = "five_policy_recovery_recorded_pair_retry_protocol_v1"
PLAN_SCHEMA = "five_policy_recovery_recorded_pair_retry_plan_v1"
SOURCE_FILES = tuple(sorted({
    *V6.SOURCE_FILES,
    "evals/recovery_receipt_snapshot_routes_v1.py",
    "tests/test_recovery_receipt_snapshot_routes_v1.py",
    "evals/recovery_recorded_pair_reuse_v1.py",
    "tests/test_recovery_recorded_pair_reuse_v1.py",
    "bench/run_five_policy_recovery_resume_v7.py",
    "tests/test_five_policy_recovery_resume_v7.py",
}))
EXPECTED_PARENT_WALL_US = 1_761_960_669
EXPECTED_FAILURE_WALL_US = 1_120_155_721
EXPECTED_CUMULATIVE_WALL_US = 2_882_116_390
TOTAL_WALL_US = 43_200_000_000
PAIR_RELATIVE = (
    "actual-v1/diagnosis/case-03_fixed_eligible/step01/probe/result.json")
CONTEXT_RELATIVE = "actual-v1/diagnosis/case-03_fixed_eligible/step01/context.json"
CHOICE_RELATIVE = "actual-v1/diagnosis/case-03_fixed_eligible/step01/choice.json"


def _source_map() -> dict[str, str]:
    return {name: C._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _inventory(phase: Path) -> dict[str, dict[str, Any]]:
    root = Path(phase).resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}

    def failed(error: OSError) -> None:
        raise ValueError("receipt 재사용 실패 원문 읽기 실패") from error

    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=failed):
        base = Path(directory)
        C._require(not any((base / name).is_symlink()
                           or getattr(base / name, "is_junction", lambda: False)()
                           for name in [*subdirs, *names]),
                   "receipt 재사용 실패 원문 link 거부")
        for name in names:
            path = base / name
            files[path.relative_to(root).as_posix()] = {
                "sha256": C._sha(path), "size_bytes": path.stat().st_size}
    return dict(sorted(files.items()))


def _failure_state(phase: Path, inventory_path: Path,
                   closed_lease: Path) -> dict[str, Any]:
    root = Path(phase).resolve(strict=True)
    files = _inventory(root)
    inventory = C._read(inventory_path)
    lease = C._read(closed_lease)
    error = C._read(root / "actual-v1/parent-error.json")
    C._require(inventory.get("schema")
               == "actual_epoch2_public_snapshot_verifier_failure_v1"
               and inventory.get("status") == "sealed"
               and inventory.get("files") == files
               and inventory.get("file_count") == 78
               and inventory.get("inventory_digest") == V5.C.SECOND.digest(files),
               "receipt 재사용 실패 inventory가 다르다")
    C._require(error == inventory.get("original_error")
               and error.get("error") == "backend packet이 등록 source와 다르다"
               and C._read(root / "exit.json").get("exit_code") == 1,
               "receipt 재사용 실패 오류/exit가 다르다")
    C._require(inventory.get("API_requests") == 0
               and inventory.get("SDK_started") == 2
               and inventory.get("SDK_completed") == 2
               and inventory.get("paired_intents") == 1
               and inventory.get("sdk_roles_reserved") == 2
               and inventory.get("reserved_role_terminal_state")
               == "completed_physical_roles_reuse_required"
               and inventory.get("new_physical_role_cap") == 30
               and inventory.get("prior_physical_roles_reused") == 2
               and inventory.get("physical_cumulative_cap") == 32,
               "완료 pair의 physical SDK 상계가 다르다")
    pair, context, choice = (root / name for name in (
        PAIR_RELATIVE, CONTEXT_RELATIVE, CHOICE_RELATIVE))
    result = C._read(pair)
    C._require(inventory.get("completed_pair") == C._ref(pair)
               and result.get("schema") == "local_pair_probe_host_result_v1"
               and result.get("status") == "complete"
               and result.get("outcome") == "normal"
               and result.get("probe") == "P_episode"
               and result.get("correction_executed") is False
               and result.get("recovery_claimed") is False
               and result.get("retry_count") == 0,
               "재사용할 완료 pair가 다르다")
    for role in ("case", "reference"):
        role_state = inventory.get("completed_roles", {}).get(role, {})
        receipt = C._bound(role_state.get("receipt", {}), f"{role} receipt")
        C._require(result.get("receipts", {}).get(role) == C._ref(receipt)
                   and role_state.get("exit_code") == 0
                   and role_state.get("cleanup_verified") is True,
                   f"완료 {role} physical receipt가 다르다")
    C._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == C.RUN.LEASE.OWNER
               and C._lease_id(lease) == inventory.get("lease_id")
               and inventory.get("closed_lease", {}).get("sha256")
               == C._sha(closed_lease),
               "receipt 재사용 실패 lease가 다르다")
    wall_us = math.ceil((C._old_elapsed(lease) + 0.000001) * 1_000_000)
    C._require(wall_us == inventory.get("old_wall_upper_bound_us")
               == EXPECTED_FAILURE_WALL_US
               and inventory.get("previous_wall_us") == EXPECTED_PARENT_WALL_US
               and inventory.get("cumulative_wall_us") == EXPECTED_CUMULATIVE_WALL_US
               and round(float(inventory["remaining_wall_s"]) * 1_000_000)
               == TOTAL_WALL_US - EXPECTED_CUMULATIVE_WALL_US,
               "receipt 재사용 실패 wall 상계가 다르다")
    return {
        "inventory_digest": inventory["inventory_digest"],
        "inventory_sha256": C._sha(inventory_path),
        "closed_lease_sha256": C._sha(closed_lease),
        "lease_id": inventory["lease_id"],
        "wall_us": wall_us,
        "api_calls": 0, "sdk_started": 2, "sdk_completed": 2,
        "paired_intents_completed": 1, "sdk_roles_completed": 2,
        "completed_pair": C._ref(pair),
        "original_context": C._ref(context),
        "original_choice": C._ref(choice),
        "new_physical_role_cap": 30,
        "prior_physical_roles_reused": 2,
        "physical_cumulative_cap": 32,
    }


def _public_failure(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    for name in ("completed_pair", "original_context", "original_choice"):
        result[name] = {"sha256": result[name]["sha256"]}
    return result


def build_protocol(*, parent_protocol: Path, parent_anchor: str,
                   parent_private_plan: Path, old_phase: Path,
                   old_inventory: Path, old_closed_lease: Path,
                   private_out: Path, out: Path) -> dict[str, Any]:
    parent = V6.verify_protocol(parent_protocol, anchor=parent_anchor,
                                private_plan=parent_private_plan)
    failure = _failure_state(old_phase, old_inventory, old_closed_lease)
    C._require(parent.get("cumulative_wall_us") == EXPECTED_PARENT_WALL_US,
               "V6 누적 wall이 다르다")
    remaining = (TOTAL_WALL_US - EXPECTED_CUMULATIVE_WALL_US) / 1_000_000
    plan = {
        "schema": PLAN_SCHEMA,
        "parent_protocol": C._ref(parent_protocol),
        "parent_anchor": parent_anchor,
        "parent_private_plan": C._ref(parent_private_plan),
        "old_phase": str(Path(old_phase).resolve(strict=True)),
        "old_inventory": C._ref(old_inventory),
        "old_closed_lease": C._ref(old_closed_lease),
        "prior_failure": failure,
        "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US,
        "remaining_wall_s": remaining,
        "approval_granted": False, "automatic_retry": False,
    }
    value = {
        "schema": SCHEMA,
        "status": "preregistered_recorded_pair_reuse_retry",
        "parent": {"path": C._relative(parent_protocol),
                   "sha256": C._sha(parent_protocol)},
        "sources_lf": _source_map(),
        "prior": {
            "parent_cumulative_wall_us": EXPECTED_PARENT_WALL_US,
            **_public_failure(failure),
            "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US,
        },
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "new_physical_sdk_cap": 30, "prior_physical_sdk_reused": 2,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": remaining,
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "receipt_snapshot_routes_and_exact_pair_reuse",
    }
    C._write_new(private_out, plan)
    C._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str, private_plan: Path) -> dict[str, Any]:
    value, plan = C._read(path), C._read(private_plan)
    C._require(value.get("schema") == SCHEMA
               and value.get("status") == "preregistered_recorded_pair_reuse_retry"
               and plan.get("schema") == PLAN_SCHEMA
               and plan.get("approval_granted") is False
               and plan.get("automatic_retry") is False,
               "pair reuse protocol/plan schema")
    parent_protocol = C._bound(plan["parent_protocol"], "parent protocol")
    parent_private = C._bound(plan["parent_private_plan"], "parent private plan")
    parent = V6.verify_protocol(parent_protocol, anchor=plan["parent_anchor"],
                                private_plan=parent_private)
    failure = _failure_state(
        Path(plan["old_phase"]), C._bound(plan["old_inventory"], "old inventory"),
        C._bound(plan["old_closed_lease"], "old closed lease"))
    C._require(set(plan) == {
        "schema", "parent_protocol", "parent_anchor", "parent_private_plan",
        "old_phase", "old_inventory", "old_closed_lease", "prior_failure",
        "cumulative_wall_us", "remaining_wall_s", "approval_granted",
        "automatic_retry"}
        and parent.get("cumulative_wall_us") == EXPECTED_PARENT_WALL_US
        and plan["prior_failure"] == failure
        and plan["cumulative_wall_us"] == EXPECTED_CUMULATIVE_WALL_US
        and plan["remaining_wall_s"] == 40317.88361,
        "pair reuse private plan 결합")
    C._verify_anchor(anchor)
    C._verify_blob(anchor, C._relative(path), C._sha(path, lf=True))
    C._require(value.get("sources_lf") == _source_map(), "pair reuse source SHA")
    for name, digest in value["sources_lf"].items():
        C._verify_blob(anchor, name, digest)
    expected = {
        "schema": SCHEMA, "status": "preregistered_recorded_pair_reuse_retry",
        "parent": {"path": C._relative(parent_protocol),
                   "sha256": C._sha(parent_protocol)},
        "sources_lf": _source_map(),
        "prior": {"parent_cumulative_wall_us": EXPECTED_PARENT_WALL_US,
                  **_public_failure(failure),
                  "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US},
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "new_physical_sdk_cap": 30, "prior_physical_sdk_reused": 2,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": 40317.88361,
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "receipt_snapshot_routes_and_exact_pair_reuse",
    }
    C._require(len(SOURCE_FILES) == 32 and value == expected,
               "pair reuse budget/science")
    return deepcopy(plan)


def _parents(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path,
                                            list[dict[str, Any]]]:
    v6 = V6.verify_protocol(C._bound(plan["parent_protocol"], "parent protocol"),
                            anchor=plan["parent_anchor"],
                            private_plan=C._bound(
                                plan["parent_private_plan"], "parent private plan"))
    v5 = V5.verify_protocol(C._bound(v6["parent_protocol"], "V5 protocol"),
                            anchor=v6["parent_anchor"],
                            private_plan=C._bound(
                                v6["parent_private_plan"], "V5 private plan"))
    second_plan = C._bound(v5["second_plan"], "second plan")
    return v6, v5, second_plan, V5.V4.registered_sources(second_plan)


def _scopes(plan: dict[str, Any], v5: dict[str, Any], sources: list[dict[str, Any]],
            stack: ExitStack) -> None:
    stack.enter_context(V6.LEGACY.source_validation_scope())
    with V5.V3.CACHE.proof_scope(repo=ROOT):
        parents = stack.enter_context(V5.BIND.epoch2_snapshot_scope(sources))
    stack.enter_context(V5.PROBE_BOUNDARY.probe_boundary_scope(
        sources, parents=parents,
        protocol_path=C._bound(v5["inner_loop_protocol"], "inner loop"),
        anchor=v5["inner_anchor"]))
    stack.enter_context(ROUTE.receipt_snapshot_routes_scope(sources, parents=parents))
    failure = plan["prior_failure"]
    stack.enter_context(REUSER.recorded_pair_reuse_scope(
        original_result=C._bound(failure["completed_pair"], "completed pair"),
        original_context=C._bound(failure["original_context"], "original context"),
        original_choice=C._bound(failure["original_choice"], "original choice")))


def execute(*, protocol: Path, anchor: str, private_plan: Path, transport_root: Path,
            lab: Path, isaac_python: Path, out: Path, approver: str) -> dict[str, Any]:
    with V5.V3.CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        v6, v5, second_plan, sources = _parents(plan)
    with ExitStack() as stack:
        _scopes(plan, v5, sources, stack)
        stack.enter_context(V5.V4.remaining_wall_scope(plan["remaining_wall_s"]))
        return V5.V3.run_second_operational(
            second_plan,
            host_protocol=C._bound(v5["host_protocol"], "host protocol"),
            host_anchor=v6["parent_anchor"],
            compatibility_protocol=C._bound(v5["compatibility_protocol"], "compatibility"),
            compatibility_anchor=v6["parent_anchor"],
            cache_protocol=C._bound(v5["cache_protocol"], "cache protocol"),
            cache_anchor=v6["parent_anchor"], transport_root=transport_root,
            lab=lab, isaac_python=isaac_python, out=out, approver=approver)


def export(*, protocol: Path, anchor: str, private_plan: Path, first_result: Path,
           second_result: Path, lab: Path, out: Path) -> dict[str, Any]:
    with V5.V3.CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        v6, v5, _second_plan, sources = _parents(plan)
    with ExitStack() as stack:
        _scopes(plan, v5, sources, stack)
        with V5.V3.CACHE.proof_scope(repo=ROOT):
            return V5.EXPORT.export(
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
    run.add_argument("--transport-root", type=Path, required=True)
    run.add_argument("--isaac-python", type=Path, required=True)
    run.add_argument("--approver", choices=[C.RUN.APPROVER], required=True)
    export_parser.add_argument("--first-result", type=Path, required=True)
    export_parser.add_argument("--second-result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            execute(protocol=args.protocol, anchor=args.anchor,
                    private_plan=args.private_plan, transport_root=args.transport_root,
                    lab=args.lab, isaac_python=args.isaac_python,
                    out=args.out, approver=args.approver)
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
