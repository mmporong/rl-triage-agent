# pyright: reportMissingImports=false
"""운영 callback의 legacy source 재검증을 격리해 SECOND를 재개한다."""
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

import recovery_legacy_validation_isolation_v1 as LEGACY
import run_five_policy_recovery_resume_v5 as V5

C = V5.C
SCHEMA = "five_policy_recovery_legacy_validation_retry_protocol_v1"
PLAN_SCHEMA = "five_policy_recovery_legacy_validation_retry_plan_v1"
SOURCE_FILES = tuple(sorted({
    *V5.SOURCE_FILES,
    "evals/recovery_legacy_validation_isolation_v1.py",
    "tests/test_recovery_legacy_validation_isolation_v1.py",
    "bench/run_five_policy_recovery_resume_v6.py",
    "tests/test_five_policy_recovery_resume_v6.py",
}))
EXPECTED_PARENT_WALL_US = 1_025_528_011
EXPECTED_FAILURE_WALL_US = 736_432_658
EXPECTED_CUMULATIVE_WALL_US = 1_761_960_669
TOTAL_WALL_US = 43_200_000_000


def _source_map() -> dict[str, str]:
    return {name: C._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _inventory(phase: Path) -> dict[str, dict[str, Any]]:
    root = Path(phase).resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}

    def failed(error: OSError) -> None:
        raise ValueError("legacy 재검증 실패 원문 읽기 실패") from error

    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=failed):
        base = Path(directory)
        C._require(not any((base / name).is_symlink()
                           or getattr(base / name, "is_junction", lambda: False)()
                           for name in [*subdirs, *names]),
                   "legacy 재검증 실패 원문 link 거부")
        for name in names:
            path = base / name
            files[path.relative_to(root).as_posix()] = {
                "sha256": C._sha(path), "size_bytes": path.stat().st_size}
    return dict(sorted(files.items()))


def _failure_state(phase: Path, inventory_path: Path,
                   closed_lease: Path) -> dict[str, Any]:
    files = _inventory(phase)
    inventory = C._read(inventory_path)
    lease = C._read(closed_lease)
    error = C._read(Path(phase) / "actual-v1/parent-error.json")
    intent = C._read(Path(phase) / "actual-v1/paired-intents/0001.json")
    C._require(inventory.get("schema") == "actual_epoch2_legacy_revalidation_failure_v1"
               and inventory.get("status") == "sealed"
               and inventory.get("files") == files
               and inventory.get("file_count") == 14
               and inventory.get("inventory_digest") == V5.C.SECOND.digest(files),
               "legacy 재검증 실패 inventory가 다르다")
    C._require(error == inventory.get("original_error")
               and error.get("error") == "다른 runtime view protocol SHA"
               and C._read(Path(phase) / "exit.json").get("exit_code") == 1,
               "legacy 재검증 오류/exit가 다르다")
    C._require(intent.get("sdk_roles_reserved") == 2
               and inventory.get("paired_intents") == 1
               and inventory.get("sdk_roles_reserved") == 2
               and inventory.get("reserved_role_terminal_state")
               == "cancelled_before_backend_admission"
               and inventory.get("API_requests") == 0
               and inventory.get("SDK_started") == 0,
               "legacy 재검증 실패가 backend 입장 전 취소 상태가 아니다")
    forbidden = ("/jobs.json", "/plan.json", "/authorization.json",
                 "/probes.ledger.jsonl", "/api-intent.json", "/owned/")
    C._require(not any(any(token in f"/{name}" for token in forbidden)
                       for name in files),
               "legacy 재검증 실패에 backend 실행 파일이 있다")
    C._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == C.RUN.LEASE.OWNER
               and C._lease_id(lease) == inventory.get("lease_id")
               and inventory.get("closed_lease", {}).get("sha256")
               == C._sha(closed_lease),
               "legacy 재검증 실패 lease가 다르다")
    wall_us = math.ceil((C._old_elapsed(lease) + 0.000001) * 1_000_000)
    C._require(wall_us == inventory.get("old_wall_upper_bound_us")
               == EXPECTED_FAILURE_WALL_US
               and inventory.get("previous_wall_us") == EXPECTED_PARENT_WALL_US
               and inventory.get("cumulative_wall_us") == EXPECTED_CUMULATIVE_WALL_US
               and round(float(inventory["remaining_wall_s"]) * 1_000_000)
               == TOTAL_WALL_US - EXPECTED_CUMULATIVE_WALL_US,
               "legacy 재검증 실패 wall 상계가 다르다")
    return {
        "inventory_digest": inventory["inventory_digest"],
        "inventory_sha256": C._sha(inventory_path),
        "closed_lease_sha256": C._sha(closed_lease),
        "lease_id": inventory["lease_id"],
        "wall_us": wall_us,
        "api_calls": 0, "sdk_started": 0,
        "paired_intents_cancelled_before_backend": 1,
        "sdk_roles_reserved_cancelled": 2,
    }


def build_protocol(*, parent_protocol: Path, parent_anchor: str,
                   parent_private_plan: Path, old_phase: Path,
                   old_inventory: Path, old_closed_lease: Path,
                   private_out: Path, out: Path) -> dict[str, Any]:
    parent = V5.verify_protocol(parent_protocol, anchor=parent_anchor,
                                private_plan=parent_private_plan)
    failure = _failure_state(old_phase, old_inventory, old_closed_lease)
    C._require(parent.get("cumulative_wall_us") == EXPECTED_PARENT_WALL_US,
               "V5 누적 wall이 다르다")
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
        "approval_granted": False,
        "automatic_retry": False,
    }
    value = {
        "schema": SCHEMA,
        "status": "preregistered_legacy_validation_isolation_retry",
        "parent": {"path": C._relative(parent_protocol),
                   "sha256": C._sha(parent_protocol)},
        "sources_lf": _source_map(),
        "prior": {
            "parent_cumulative_wall_us": EXPECTED_PARENT_WALL_US,
            **failure,
            "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US,
            "api_calls": 0, "sdk_started": 0,
            "total_paired_intents_cancelled_before_backend": 2,
            "total_sdk_roles_reserved_cancelled": 4,
        },
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": remaining,
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "legacy_source_validation_isolation",
    }
    C._write_new(private_out, plan)
    C._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str, private_plan: Path) -> dict[str, Any]:
    value, plan = C._read(path), C._read(private_plan)
    C._require(value.get("schema") == SCHEMA
               and value.get("status")
               == "preregistered_legacy_validation_isolation_retry"
               and plan.get("schema") == PLAN_SCHEMA
               and plan.get("approval_granted") is False
               and plan.get("automatic_retry") is False,
               "legacy retry protocol/plan schema")
    parent_protocol = C._bound(plan["parent_protocol"], "parent protocol")
    parent_private = C._bound(plan["parent_private_plan"], "parent private plan")
    parent = V5.verify_protocol(parent_protocol, anchor=plan["parent_anchor"],
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
        and plan["remaining_wall_s"] == 41438.039331,
        "legacy retry private plan 결합")
    C._verify_anchor(anchor)
    C._verify_blob(anchor, C._relative(path), C._sha(path, lf=True))
    C._require(value.get("sources_lf") == _source_map(), "legacy retry source SHA")
    for name, digest in value["sources_lf"].items():
        C._verify_blob(anchor, name, digest)
    expected = {
        "schema": SCHEMA,
        "status": "preregistered_legacy_validation_isolation_retry",
        "parent": {"path": C._relative(parent_protocol),
                   "sha256": C._sha(parent_protocol)},
        "sources_lf": _source_map(),
        "prior": {
            "parent_cumulative_wall_us": EXPECTED_PARENT_WALL_US,
            **failure,
            "cumulative_wall_us": EXPECTED_CUMULATIVE_WALL_US,
            "api_calls": 0, "sdk_started": 0,
            "total_paired_intents_cancelled_before_backend": 2,
            "total_sdk_roles_reserved_cancelled": 4,
        },
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": 41438.039331,
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "legacy_source_validation_isolation",
    }
    C._require(len(SOURCE_FILES) == 26 and value == expected,
               "legacy retry budget/science")
    return deepcopy(plan)


def _parent(plan: dict[str, Any]) -> tuple[dict[str, Any], Path,
                                           list[dict[str, Any]]]:
    parent = V5.verify_protocol(
        C._bound(plan["parent_protocol"], "parent protocol"),
        anchor=plan["parent_anchor"],
        private_plan=C._bound(plan["parent_private_plan"], "parent private plan"))
    second_plan = C._bound(parent["second_plan"], "second plan")
    return parent, second_plan, V5.V4.registered_sources(second_plan)


def execute(*, protocol: Path, anchor: str, private_plan: Path, transport_root: Path,
            lab: Path, isaac_python: Path, out: Path, approver: str) -> dict[str, Any]:
    with V5.V3.CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        parent, second_plan, sources = _parent(plan)
    with ExitStack() as stack:
        stack.enter_context(LEGACY.source_validation_scope())
        V5._scopes(parent, sources, stack)
        stack.enter_context(V5.V4.remaining_wall_scope(plan["remaining_wall_s"]))
        return V5.V3.run_second_operational(
            second_plan,
            host_protocol=C._bound(parent["host_protocol"], "host protocol"),
            host_anchor=plan["parent_anchor"],
            compatibility_protocol=C._bound(
                parent["compatibility_protocol"], "compatibility"),
            compatibility_anchor=plan["parent_anchor"],
            cache_protocol=C._bound(parent["cache_protocol"], "cache protocol"),
            cache_anchor=plan["parent_anchor"], transport_root=transport_root,
            lab=lab, isaac_python=isaac_python, out=out, approver=approver)


def export(*, protocol: Path, anchor: str, private_plan: Path, first_result: Path,
           second_result: Path, lab: Path, out: Path) -> dict[str, Any]:
    with V5.V3.CACHE.proof_scope(repo=ROOT):
        plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
        parent, _second_plan, sources = _parent(plan)
    with ExitStack() as stack:
        stack.enter_context(LEGACY.source_validation_scope())
        V5._scopes(parent, sources, stack)
        with V5.V3.CACHE.proof_scope(repo=ROOT):
            return V5.EXPORT.export(
                first_result, out, lab=lab, second_result=second_result,
                second_compatibility_protocol=C._bound(
                    parent["compatibility_protocol"], "compatibility"),
                second_compatibility_anchor=plan["parent_anchor"],
                second_host_protocol=C._bound(parent["host_protocol"], "host protocol"))


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
