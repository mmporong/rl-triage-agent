# pyright: reportMissingImports=false
"""두 보존 실패 뒤 typed epoch-2 probe 경계로 SECOND를 재개한다."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v2 as EXPORT
import recovery_epoch2_probe_boundary_v1 as PROBE_BOUNDARY
import run_five_policy_recovery_resume_v4 as V4

C = V4.C
BIND = V4.BIND
V3 = V4.V3
SCHEMA = "five_policy_recovery_second_probe_boundary_retry_protocol_v1"
PLAN_SCHEMA = "five_policy_recovery_second_probe_boundary_retry_plan_v1"
SOURCE_FILES = tuple(sorted({
    *V4.SOURCE_FILES,
    *PROBE_BOUNDARY.SOURCE_FILES,
    "bench/run_five_policy_recovery_resume_v5.py",
    "tests/test_five_policy_recovery_resume_v5.py",
    "evals/export_five_policy_recovery_v2.py",
    "tests/test_export_five_policy_recovery_v2.py",
}))
SECOND_FAILURE_FILES = frozenset({
    "actual-v1-git-proof-cache.json",
    "actual-v1/diagnosis/case-03_fixed_eligible/step01/busy-gate.json",
    "actual-v1/diagnosis/case-03_fixed_eligible/step01/choice.json",
    "actual-v1/diagnosis/case-03_fixed_eligible/step01/context.json",
    "actual-v1/diagnosis/case-03_fixed_eligible/step01/request-spec.json",
    "actual-v1/gpu-observations/r4_case_03_fixed_eligible_1/gpu_before.json",
    "actual-v1/paired-intents/0001.json",
    "actual-v1/parent-error.json",
    "actual-v1/parent-traceback.txt",
    "dispatch.json", "execute.stderr.log", "execute.stdout.log", "exit.json",
    "transport-v1/manifest.json",
})
POST_FAILURE_CPU_LEDGER = (
    "actual-v1/diagnosis/case-03_fixed_eligible/probes.ledger.jsonl"
)
TOTAL_WALL_US = 43_200_000_000
EXPECTED_PRIOR_WALL_US = 1_025_528_011


def _source_map() -> dict[str, str]:
    return {name: C._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _inventory(phase: Path, *, permitted_extra: dict[str, Any] | None = None
               ) -> dict[str, dict[str, Any]]:
    root = Path(phase).resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}

    def failed(error: OSError) -> None:
        raise ValueError("두 번째 실패 원문 읽기 실패") from error

    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=failed):
        base = Path(directory)
        C._require(not any((base / name).is_symlink()
                           or getattr(base / name, "is_junction", lambda: False)()
                           for name in [*subdirs, *names]), "두 번째 실패 원문 link 거부")
        for name in names:
            path = base / name
            files[path.relative_to(root).as_posix()] = {
                "sha256": C._sha(path), "size_bytes": path.stat().st_size}
    expected_extra = set() if permitted_extra is None else {POST_FAILURE_CPU_LEDGER}
    extras = set(files) - SECOND_FAILURE_FILES
    C._require(SECOND_FAILURE_FILES <= set(files) and extras == expected_extra,
               "두 번째 실패 원문 14개 또는 등록된 CPU 원장이 다르다")
    if permitted_extra is not None:
        ledger = files.pop(POST_FAILURE_CPU_LEDGER)
        C._require(permitted_extra == {
            "path": POST_FAILURE_CPU_LEDGER,
            "sha256": ledger["sha256"],
            "size_bytes": ledger["size_bytes"],
        }, "사후 CPU 원장 등록값 또는 내용이 다르다")
    originals = dict(sorted(files.items()))
    C._require(set(originals) == SECOND_FAILURE_FILES, "두 번째 실패 원문 14개가 다르다")
    return originals


def _cpu_provenance(path: Path, phase: Path) -> dict[str, Any]:
    provenance_path = Path(path).resolve(strict=True)
    value = C._read(provenance_path)
    C._require(value.get("schema") == "post_failure_CPU_host_ledger_provenance_v1"
               and value.get("status") == "sealed"
               and value.get("original_failure14_bytes_unchanged") is True
               and value.get("historical_inventory_digest")
               == "ca300c2dfac3c4523093405148d4ddbfc9aa65c32b398229509ef34b2b72e3e0"
               and value.get("classification")
               == "post_failure_CPU_fixture_pending_not_production_execution"
               and value.get("SDK_started") == 0
               and value.get("API_requests") == 0
               and value.get("deletion") is False
               and value.get("historical_inventory_rewrite") is False,
               "사후 CPU provenance 상태가 다르다")

    def bound(name: str) -> Path:
        ref = value.get(name)
        if not isinstance(ref, dict):
            raise TypeError(f"사후 CPU {name} ref")
        target = Path(ref.get("path", "")).resolve(strict=True)
        C._require(ref.get("sha256") == C._sha(target), f"사후 CPU {name} SHA")
        return target

    cpu_path, context_path, choice_path = (bound(name) for name in (
        "cpu_result", "context", "choice"))
    authorization_path, harness_path = (bound(name) for name in (
        "authorization", "cpu_harness"))
    ledger_ref = value.get("extra_ledger")
    if not isinstance(ledger_ref, dict):
        raise TypeError("사후 CPU ledger ref")
    ledger = (Path(phase).resolve(strict=True) / POST_FAILURE_CPU_LEDGER).resolve(strict=True)
    C._require(Path(ledger_ref.get("path", "")).resolve(strict=True) == ledger
               and value.get("extra_ledger_relative") == POST_FAILURE_CPU_LEDGER
               and ledger_ref.get("sha256") == C._sha(ledger)
               and value.get("extra_ledger_size_bytes") == ledger.stat().st_size,
               "사후 CPU ledger path/SHA/size")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    C._require([row.get("event") for row in rows] == ["proposed", "approved", "consumed"]
               and value.get("events") == ["proposed", "approved", "consumed"]
               and all(row.get("request_id") == value.get("request_id") for row in rows)
               and rows[0].get("prereg_digest") == value.get("prereg_digest")
               and rows[0].get("registry_digest") == rows[1].get("registry_digest")
               == rows[2].get("registry_digest")
               and rows[0].get("effect_key") == rows[1].get("effect_key")
               == rows[2].get("effect_key"),
               "사후 CPU ledger 3-event 결합")
    cpu, context, choice, authorization = map(C._read, (
        cpu_path, context_path, choice_path, authorization_path))
    C._require(cpu.get("schema") == "recovery_epoch2_probe_boundary_actual_cpu_v1"
               and cpu.get("status") == "owned_boundary_reached"
               and cpu.get("sdk_executed") is False
               and cpu.get("gpu_executed") is False
               and cpu.get("api_calls") == 0
               and cpu.get("selected_probe") == "P_episode"
               and cpu.get("context") == value["context"]
               and cpu.get("choice") == value["choice"]
               and cpu.get("authorization") == value["authorization"],
               "사후 CPU result가 owned sentinel 증거가 아니다")
    C._require(Path(context["snapshot"]["inputs"]["ledger_path"]).resolve(strict=True)
               == ledger
               and choice.get("context_digest") == value.get("prereg_digest")
               and choice.get("probe") == "P_episode"
               and choice.get("approval_granted") is False
               and authorization.get("schema") == "lineage_probe_execution_authorization_v1"
               and authorization.get("choice", {}).get("probe") == "P_episode"
               and authorization.get("snapshot", {}).get("execution_authorized") is False
               and authorization.get("snapshot", {}).get("inputs", {}).get("parent")
               == cpu.get("backend_parent"),
               "사후 CPU context/choice/authorization 결합")
    source_ref = cpu.get("source")
    C._require(isinstance(source_ref, dict)
               and source_ref.get("sha256") == C._sha(
                   Path(source_ref.get("path", "")).resolve(strict=True)),
               "사후 CPU source SHA")
    return {
        "status": "post_failure_cpu_fixture_pending_not_production_execution",
        "provenance_sha256": C._sha(provenance_path),
        "ledger": {"path": POST_FAILURE_CPU_LEDGER,
                   "sha256": C._sha(ledger), "size_bytes": ledger.stat().st_size},
        "events": ["proposed", "approved", "consumed"],
        "request_id": value["request_id"],
        "prereg_digest": value["prereg_digest"],
        "cpu_result_sha256": C._sha(cpu_path),
        "context_sha256": C._sha(context_path),
        "choice_sha256": C._sha(choice_path),
        "authorization_sha256": C._sha(authorization_path),
        "cpu_harness_sha256": C._sha(harness_path),
        "sdk_started": 0, "api_requests": 0,
    }


def _second_failure_state(phase: Path, inventory_path: Path,
                          closed_lease: Path, *, registration_anchor: str,
                          parent_anchor: str,
                          cpu_provenance: Path | None = None) -> dict[str, Any]:
    cpu = None if cpu_provenance is None else _cpu_provenance(cpu_provenance, phase)
    files = _inventory(phase, permitted_extra=None if cpu is None else cpu["ledger"])
    inventory = C._read(inventory_path)
    C._require(inventory.get("schema") == "actual_epoch2_view_protocol_failure_v1"
               and inventory.get("status") == "sealed"
               and inventory.get("files") == files
               and inventory.get("file_count") == 14
               and inventory.get("inventory_digest") == C.SECOND.digest(files),
               "봉인된 두 번째 실패 inventory가 다르다")
    error = C._read(Path(phase) / "actual-v1/parent-error.json")
    intent = C._read(Path(phase) / "actual-v1/paired-intents/0001.json")
    lease = C._read(closed_lease)
    dispatch = C._read(Path(phase) / "dispatch.json")
    cache_evidence = C._read(Path(phase) / "actual-v1-git-proof-cache.json")
    C._require(error == inventory.get("original_error")
               and error.get("error") == "runtime view protocol schema 또는 필드 범위 불일치"
               and C._read(Path(phase) / "exit.json").get("exit_code") == 1,
               "두 번째 실패 오류/exit가 다르다")
    C._require(dispatch.get("registration_anchor") == registration_anchor
               and dispatch.get("parent_anchor") == parent_anchor
               and cache_evidence.get("execution_anchor") == parent_anchor,
               "두 번째 실패의 registration/parent anchor가 다르다")
    C._require(intent.get("sdk_roles_reserved") == 2
               and inventory.get("paired_intents") == 1
               and inventory.get("sdk_roles_reserved") == 2
               and inventory.get("reserved_role_terminal_state")
               == "cancelled_before_backend_admission"
               and inventory.get("API_requests") == 0
               and inventory.get("SDK_started") == 0,
               "두 번째 실패가 backend 입장 전 취소 상태가 아니다")
    C._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == C.RUN.LEASE.OWNER
               and C._lease_id(lease) == inventory.get("lease_id"),
               "두 번째 실패 lease가 다르다")
    second_us = round(float(inventory["old_wall_upper_bound_s"]) * 1_000_000)
    measured_us = math.ceil((C._old_elapsed(lease) + 0.000001) * 1_000_000)
    previous_us = round(float(inventory["previous_wall_upper_bound_s"]) * 1_000_000)
    cumulative_us = round(float(inventory["cumulative_wall_upper_bound_s"]) * 1_000_000)
    remaining_us = round(float(inventory["remaining_parent_wall_s"]) * 1_000_000)
    C._require(second_us == measured_us == 647_737_403
               and previous_us == 377_790_608
               and cumulative_us == EXPECTED_PRIOR_WALL_US
               and remaining_us == TOTAL_WALL_US - EXPECTED_PRIOR_WALL_US
               and round(float(dispatch["remaining_parent_wall_s"]) * 1_000_000)
               == TOTAL_WALL_US - previous_us,
               "두 실패의 wall 원문 합이 다르다")
    return {
        "inventory": files,
        "inventory_digest": inventory["inventory_digest"],
        "inventory_sha256": C._sha(inventory_path),
        "post_failure_cpu_provenance": cpu,
        "lease_id": inventory["lease_id"],
        "closed_lease_sha256": C._sha(closed_lease),
        "wall_us": second_us,
        "api_calls": 0,
        "sdk_started": 0,
        "paired_intents": 1,
        "sdk_roles_reserved_cancelled": 2,
    }


def _component(ref: dict[str, Any]) -> dict[str, str]:
    path = C._bound(ref, "component")
    return {"path": C._relative(path), "sha256": C._sha(path)}


def _components(plan: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "second_plan": {"sha256": plan["second_plan"]["sha256"]},
        **{name: _component(plan[name]) for name in (
            "host_protocol", "compatibility_protocol", "cache_protocol",
            "inner_loop_protocol")},
    }


def build_protocol(*, old_protocol: Path, old_anchor: str, old_private_plan: Path,
                   old_phase: Path, old_inventory: Path, old_closed_lease: Path,
                   cpu_provenance: Path | None = None,
                   second_plan: Path, host_protocol: Path, compatibility_protocol: Path,
                   cache_protocol: Path, inner_loop_protocol: Path, inner_anchor: str,
                   private_out: Path, out: Path,
                   component_anchor: str | None = None) -> dict[str, Any]:
    previous = V4.verify_protocol(old_protocol, anchor=old_anchor,
                                  private_plan=old_private_plan)
    second = _second_failure_state(
        old_phase, old_inventory, old_closed_lease,
        registration_anchor=old_anchor, parent_anchor=previous["parent_anchor"],
        cpu_provenance=cpu_provenance)
    first_us = round(float(previous["old_wall_upper_bound_s"]) * 1_000_000)
    cumulative_us = first_us + second["wall_us"]
    C._require(first_us == 377_790_608 and cumulative_us == EXPECTED_PRIOR_WALL_US,
               "두 보존 실패의 누적 wall이 다르다")
    C._require(component_anchor is None or (
        isinstance(component_anchor, str) and len(component_anchor) == 40
        and set(component_anchor) <= set("0123456789abcdef")),
        "component anchor 형식")
    plan = {
        "schema": PLAN_SCHEMA,
        "old_protocol": C._ref(old_protocol), "old_anchor": old_anchor,
        "old_private_plan": C._ref(old_private_plan),
        "old_phase": str(Path(old_phase).resolve(strict=True)),
        "old_inventory": C._ref(old_inventory),
        "old_closed_lease": C._ref(old_closed_lease),
        "cpu_provenance": None if cpu_provenance is None else C._ref(cpu_provenance),
        "second_plan": C._ref(second_plan), "host_protocol": C._ref(host_protocol),
        "compatibility_protocol": C._ref(compatibility_protocol),
        "cache_protocol": C._ref(cache_protocol),
        "inner_loop_protocol": C._ref(inner_loop_protocol),
        "inner_anchor": inner_anchor, "component_anchor": component_anchor,
        "prior_first": {"inventory_digest": previous["inventory_digest"],
                          "lease_id": previous["lease_id"],
                          "wall_us": first_us},
        "prior_second": second,
        "cumulative_wall_us": cumulative_us,
        "remaining_wall_s": (TOTAL_WALL_US - cumulative_us) / 1_000_000,
        "approval_granted": False, "automatic_retry": False,
    }
    components = _components(plan)
    value = {
        "schema": SCHEMA,
        "status": "preregistered_typed_probe_boundary_retry",
        "component_anchor": component_anchor,
        "components": components,
        "sources_lf": _source_map(),
        "prior": {
            "first_inventory_digest": previous["inventory_digest"],
            "second_inventory_digest": second["inventory_digest"],
            "second_inventory_sha256": second["inventory_sha256"],
            "second_closed_lease_sha256": second["closed_lease_sha256"],
            "post_failure_cpu_provenance": second["post_failure_cpu_provenance"],
            "first_lease_id": previous["lease_id"],
            "second_lease_id": second["lease_id"],
            "cumulative_wall_us": cumulative_us,
            "api_calls": 0, "sdk_started": 0,
            "paired_intents_cancelled_before_backend": 1,
            "sdk_roles_reserved_cancelled": 2,
        },
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": plan["remaining_wall_s"],
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "typed_runtime_view_and_registered_backend_parent",
    }
    C._write_new(private_out, plan)
    C._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str, private_plan: Path) -> dict[str, Any]:
    value, plan = C._read(path), C._read(private_plan)
    C._require(value.get("schema") == SCHEMA
               and value.get("status") == "preregistered_typed_probe_boundary_retry"
               and plan.get("schema") == PLAN_SCHEMA
               and plan.get("approval_granted") is False
               and plan.get("automatic_retry") is False,
               "typed retry protocol/plan schema")
    previous = V4.verify_protocol(C._bound(plan["old_protocol"], "old protocol"),
                                  anchor=plan["old_anchor"],
                                  private_plan=C._bound(
                                      plan["old_private_plan"], "old private plan"))
    second = _second_failure_state(
        Path(plan["old_phase"]), C._bound(plan["old_inventory"], "old inventory"),
        C._bound(plan["old_closed_lease"], "old closed lease"),
        registration_anchor=plan["old_anchor"], parent_anchor=previous["parent_anchor"],
        cpu_provenance=None if plan["cpu_provenance"] is None else C._bound(
            plan["cpu_provenance"], "CPU provenance"))
    expected_plan_keys = {
        "schema", "old_protocol", "old_anchor", "old_private_plan", "old_phase",
        "old_inventory", "old_closed_lease", "cpu_provenance", "second_plan", "host_protocol",
        "compatibility_protocol", "cache_protocol", "inner_loop_protocol",
        "inner_anchor", "component_anchor", "prior_first", "prior_second",
        "cumulative_wall_us", "remaining_wall_s", "approval_granted",
        "automatic_retry",
    }
    C._require(set(plan) == expected_plan_keys
               and plan["prior_first"] == {
                   "inventory_digest": previous["inventory_digest"],
                   "lease_id": previous["lease_id"],
                   "wall_us": 377_790_608}
               and plan["prior_second"] == second
               and plan["cumulative_wall_us"] == EXPECTED_PRIOR_WALL_US
               and plan["remaining_wall_s"] == 42174.471989,
               "typed retry old evidence/wall")
    C._verify_anchor(anchor)
    C._verify_blob(anchor, C._relative(path), C._sha(path, lf=True))
    C._require(plan.get("component_anchor") in (None, anchor)
               and value.get("component_anchor") == plan.get("component_anchor"),
               "typed retry component anchor")
    C._require(value["sources_lf"] == _source_map(), "typed retry source SHA")
    for name, digest in value["sources_lf"].items():
        C._verify_blob(anchor, name, digest)
    components = _components(plan)
    C._require(value["components"] == components, "typed retry components")
    for item in components.values():
        if "path" not in item:
            continue
        C._verify_blob(anchor, item["path"], C._sha(ROOT / item["path"], lf=True))
    second_plan = C._bound(plan["second_plan"], "second plan")
    sources = V4.registered_sources(second_plan)
    inner = C._bound(plan["inner_loop_protocol"], "inner loop")
    second_plan_value = C._read(second_plan)
    C._require(C._bound(second_plan_value["loop_protocol"], "plan loop") == inner
               and second_plan_value.get("anchor") == plan["inner_anchor"],
               "second plan과 typed inner loop/anchor가 다르다")
    PROBE_BOUNDARY.verify_loop_protocol(inner, plan["inner_anchor"])
    V3.verify_protocol(C._bound(plan["cache_protocol"], "cache protocol"),
                       anchor=anchor, second_plan=second_plan,
                       compatibility_protocol=C._bound(
                           plan["compatibility_protocol"], "compatibility protocol"))
    C.verify_second_compatibility_protocol(
        C._bound(plan["compatibility_protocol"], "compatibility protocol"),
        anchor=anchor, second_plan_path=second_plan,
        host_protocol=C._bound(plan["host_protocol"], "host protocol"))
    expected_value = {
        "schema": SCHEMA, "status": "preregistered_typed_probe_boundary_retry",
        "component_anchor": plan["component_anchor"], "components": components,
        "sources_lf": _source_map(),
        "prior": {
            "first_inventory_digest": previous["inventory_digest"],
            "second_inventory_digest": second["inventory_digest"],
            "second_inventory_sha256": second["inventory_sha256"],
            "second_closed_lease_sha256": second["closed_lease_sha256"],
            "post_failure_cpu_provenance": second["post_failure_cpu_provenance"],
            "first_lease_id": previous["lease_id"],
            "second_lease_id": second["lease_id"],
            "cumulative_wall_us": EXPECTED_PRIOR_WALL_US,
            "api_calls": 0, "sdk_started": 0,
            "paired_intents_cancelled_before_backend": 1,
            "sdk_roles_reserved_cancelled": 2,
        },
        "budget": {"api_cap": 32, "sdk_cap": 32,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": 42174.471989,
                   "per_arm_pair_cap": 3, "targets": 4},
        "same_registered_sources_model_seeds_criteria": True,
        "semantic_retry": False, "epoch3_allowed": False,
        "binding_change": "typed_runtime_view_and_registered_backend_parent",
    }
    C._require(len(sources) == 4 and value == expected_value,
               "typed retry budget/science")
    return deepcopy(plan)


def _scopes(plan: dict[str, Any], sources: list[dict[str, Any]], stack: ExitStack) -> None:
    with V3.CACHE.proof_scope(repo=ROOT):
        parents = stack.enter_context(BIND.epoch2_snapshot_scope(sources))
    stack.enter_context(PROBE_BOUNDARY.probe_boundary_scope(
        sources, parents=parents,
        protocol_path=C._bound(plan["inner_loop_protocol"], "inner loop"),
        anchor=plan["inner_anchor"]))


def execute(*, protocol: Path, anchor: str, private_plan: Path, transport_root: Path,
            lab: Path, isaac_python: Path, out: Path, approver: str) -> dict[str, Any]:
    plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
    second_plan = C._bound(plan["second_plan"], "second plan")
    sources = V4.registered_sources(second_plan)
    with ExitStack() as stack:
        _scopes(plan, sources, stack)
        stack.enter_context(V4.remaining_wall_scope(plan["remaining_wall_s"]))
        return V3.run_second_operational(
            second_plan,
            host_protocol=C._bound(plan["host_protocol"], "host protocol"),
            host_anchor=anchor,
            compatibility_protocol=C._bound(plan["compatibility_protocol"], "compatibility"),
            compatibility_anchor=anchor,
            cache_protocol=C._bound(plan["cache_protocol"], "cache protocol"),
            cache_anchor=anchor, transport_root=transport_root, lab=lab,
            isaac_python=isaac_python, out=out, approver=approver)


def export(*, protocol: Path, anchor: str, private_plan: Path, first_result: Path,
           second_result: Path, lab: Path, out: Path) -> dict[str, Any]:
    plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
    second_plan = C._bound(plan["second_plan"], "second plan")
    sources = V4.registered_sources(second_plan)
    with ExitStack() as stack:
        _scopes(plan, sources, stack)
        with V3.CACHE.proof_scope(repo=ROOT):
            return EXPORT.export(
                first_result, out, lab=lab, second_result=second_result,
                second_compatibility_protocol=C._bound(
                    plan["compatibility_protocol"], "compatibility"),
                second_compatibility_anchor=anchor,
                second_host_protocol=C._bound(plan["host_protocol"], "host protocol"))


def _record_failure(out: Path, command: str, error: BaseException) -> None:
    target = Path(out).resolve()
    target.mkdir(parents=True, exist_ok=True)
    failure = target / "parent-error.json"
    if not failure.exists():
        C._write_new(failure, {
            "schema": "five_policy_recovery_parent_error_v1", "command": command,
            "error_type": type(error).__name__, "error": str(error),
            "automatic_retry": False})
    trace = target / "parent-traceback.txt"
    if not trace.exists():
        with trace.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(traceback.format_exc())


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
    for name in ("transport-root", "isaac-python"):
        run.add_argument("--" + name, type=Path, required=True)
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
        _record_failure(args.out, args.command, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
