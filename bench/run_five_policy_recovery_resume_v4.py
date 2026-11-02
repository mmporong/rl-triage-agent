# pyright: reportMissingImports=false
"""검사 전 부모 결합 오류를 보존하고 같은 SECOND 입력을 새 경로에서 실행한다."""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import traceback
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import recovery_epoch2_parent_binding_v1 as BIND
import run_five_policy_recovery_resume_v3 as V3

C = V3.CONT
SCHEMA = "five_policy_recovery_second_parent_retry_protocol_v1"
PLAN_SCHEMA = "five_policy_recovery_second_parent_retry_plan_v1"
SOURCE_FILES = tuple(sorted({
    *V3.SOURCE_FILES,
    "evals/recovery_epoch2_parent_binding_v1.py",
    "tests/test_recovery_epoch2_parent_binding_v1.py",
    "bench/run_five_policy_recovery_resume_v4.py",
    "tests/test_five_policy_recovery_resume_v4.py",
}))
OLD_FILES = {
    "dispatch.json", "execute.stdout.log", "execute.stderr.log", "exit.json",
    "actual-v1/parent-error.json", "actual-v1-git-proof-cache.json",
    "transport-v1/manifest.json",
}
_LOCK = threading.RLock()


def _source_map() -> dict[str, str]:
    return {name: C._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def _inventory(phase: Path) -> dict[str, Any]:
    phase = Path(phase).resolve(strict=True)
    files = {}

    def fail(error: OSError) -> None:
        raise ValueError("old phase 읽기 실패") from error

    for directory, subdirs, names in os.walk(phase, followlinks=False, onerror=fail):
        base = Path(directory)
        C._require(not any((base / name).is_symlink()
                           or getattr(base / name, "is_junction", lambda: False)()
                           for name in [*subdirs, *names]), "old phase link 거부")
        for name in names:
            path = base / name
            files[path.relative_to(phase).as_posix()] = {
                "sha256": C._sha(path), "size_bytes": path.stat().st_size}
    C._require(set(files) == OLD_FILES, "검사 전 실패의 원문 집합이 다르다")
    return dict(sorted(files.items()))


def _old_state(phase: Path, closed_lease: Path, parent_anchor: str) -> dict[str, Any]:
    files = _inventory(phase)
    error = C._read(phase / "actual-v1/parent-error.json")
    C._require(error.get("schema") == "five_policy_recovery_parent_error_v1"
               and error.get("error_type") == "ValueError"
               and error.get("error") == "다른 arm 또는 출력 checkpoint의 부모"
               and C._read(phase / "exit.json").get("exit_code") == 1,
               "등록된 snapshot 부모 결합 오류가 아니다")
    lease = C._read(closed_lease)
    C._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == C.RUN.LEASE.OWNER,
               "종료된 원 lease가 아니다")
    C._require(C._read(phase / "dispatch.json").get("outer_anchor") == parent_anchor
               and C._read(phase / "actual-v1-git-proof-cache.json").get("execution_anchor")
               == parent_anchor, "원 phase execution anchor")
    transport = C._read(phase / "transport-v1/manifest.json")
    C._require(transport.get("schema") == C.RUN.TRANSPORT.MANIFEST_SCHEMA
               and transport.get("api_cap") == 32,
               "원 32-call transport manifest")
    # 원문 집합에는 요청·전송 intent·paired intent·SDK launch/receipt가 없다.
    old_wall = math.ceil((C._old_elapsed(lease) + 0.000001) * 1_000_000) / 1_000_000
    C._require(math.isfinite(old_wall) and 0 <= old_wall < 43200 - 3600,
               "새 고리의 최소 남은 wall 예약이 없다")
    return {"inventory": files, "inventory_digest": C.SECOND.digest(files),
            "lease_id": C._lease_id(lease), "old_wall_upper_bound_s": old_wall,
            "remaining_wall_s": 43200.0 - old_wall,
            "old_api_calls": 0, "old_sdk_executions": 0}


def _components(plan: dict[str, Any]) -> dict[str, Any]:
    return {name: {"path": C._relative(C._bound(plan[name], name)),
                   "sha256": plan[name]["sha256"]}
            for name in ("host_protocol", "compatibility_protocol", "cache_protocol")}


def build_protocol(*, second_plan: Path, host_protocol: Path, compatibility_protocol: Path,
                   cache_protocol: Path, parent_anchor: str, old_phase: Path,
                   old_closed_lease: Path, private_out: Path, out: Path) -> dict[str, Any]:
    old = _old_state(old_phase, old_closed_lease, parent_anchor)
    plan = {"schema": PLAN_SCHEMA, "second_plan": C._ref(second_plan),
            "host_protocol": C._ref(host_protocol),
            "compatibility_protocol": C._ref(compatibility_protocol),
            "cache_protocol": C._ref(cache_protocol), "parent_anchor": parent_anchor,
            "old_phase": str(Path(old_phase).resolve(strict=True)),
            "old_closed_lease": C._ref(old_closed_lease), **old,
            "approval_granted": False, "automatic_retry": False}
    value = {"schema": SCHEMA, "status": "preregistered_parent_binding_retry",
             "second_plan_sha256": C._sha(second_plan), "parent_anchor": parent_anchor,
             "parent_components": _components(plan), "sources_lf": _source_map(),
             "prior": {name: old[name] for name in (
                 "inventory_digest", "lease_id", "old_wall_upper_bound_s",
                 "old_api_calls", "old_sdk_executions")},
             "closed_lease_sha256": C._sha(old_closed_lease),
             "budget": {"api_cap": 32, "sdk_cap": 32,
                        "cumulative_parent_wall_cap_s": 43200,
                        "remaining_parent_wall_s": old["remaining_wall_s"]},
             "same_registered_sources_model_seeds_criteria": True,
             "semantic_retry": False, "epoch3_allowed": False,
             "binding_change": "supply_verified_epoch2_parent_to_existing_snapshot_validator"}
    C._write_new(private_out, plan)
    C._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str, private_plan: Path) -> dict[str, Any]:
    value, plan = C._read(path), C._read(private_plan)
    C._require(value.get("schema") == SCHEMA
               and value.get("status") == "preregistered_parent_binding_retry"
               and plan.get("schema") == PLAN_SCHEMA
               and plan.get("approval_granted") is False
               and plan.get("automatic_retry") is False, "retry protocol/plan schema")
    C._verify_anchor(anchor)
    C._verify_blob(anchor, C._relative(path), C._sha(path, lf=True))
    C._require(value["sources_lf"] == _source_map(), "retry current source SHA")
    for name, digest in value["sources_lf"].items():
        C._verify_blob(anchor, name, digest)
    old = _old_state(Path(plan["old_phase"]), C._bound(plan["old_closed_lease"], "closed lease"),
                     plan["parent_anchor"])
    C._require(all(plan.get(name) == val for name, val in old.items())
               and value["prior"] == {name: old[name] for name in (
                   "inventory_digest", "lease_id", "old_wall_upper_bound_s",
                   "old_api_calls", "old_sdk_executions")}
               and value["parent_anchor"] == plan["parent_anchor"]
               and value["parent_components"] == _components(plan)
               and value["closed_lease_sha256"] == plan["old_closed_lease"]["sha256"]
               and value["second_plan_sha256"] == plan["second_plan"]["sha256"]
               and value["budget"] == {"api_cap": 32, "sdk_cap": 32,
                   "cumulative_parent_wall_cap_s": 43200,
                   "remaining_parent_wall_s": old["remaining_wall_s"]}
               and value.get("same_registered_sources_model_seeds_criteria") is True
               and value.get("semantic_retry") is False and value.get("epoch3_allowed") is False
               and value.get("binding_change") ==
                   "supply_verified_epoch2_parent_to_existing_snapshot_validator",
               "retry prior/parent/cap binding")
    C._bound(plan["second_plan"], "second plan")
    return deepcopy(plan)


def registered_sources(plan_path: Path) -> list[dict[str, Any]]:
    plan = C._read(plan_path)
    C._require(plan.get("schema") == C.RUN.SECOND_PLAN_SCHEMA, "SECOND plan schema")
    loop = C._read(C._bound(plan["loop_protocol"], "loop protocol"))
    sources = []
    for row in plan["records"]:
        C._require(row.get("status") == "registered", "등록된 four sources만 허용한다")
        source = C._read(C._bound(row["source"], "epoch2 source"))
        key = f"{row['case_id']}:{row['policy']}"
        C._require((source["case_id"], source["policy"]) == (row["case_id"], row["policy"])
                   and C.SECOND.digest(source) == loop["source_digests"][key],
                   "registered source digest")
        sources.append(source)
    C._require(len(sources) == 4
               and {(s["case_id"], s["policy"]) for s in sources} == set(C.RUN.TARGETS),
               "등록 대상 네 개의 exact 결합")
    return sources


@contextmanager
def remaining_wall_scope(remaining: float):
    C._require(type(remaining) in (int, float) and math.isfinite(remaining)
               and 3600 <= remaining <= 43200, "남은 wall 범위")
    with _LOCK:
        original = C.RUN.SECOND_WALL_S
        C._require(original == 43200, "동결 SECOND wall 상수가 바뀌었다")
        C.RUN.SECOND_WALL_S = float(remaining)
        try:
            yield
        finally:
            changed = C.RUN.SECOND_WALL_S != float(remaining)
            C.RUN.SECOND_WALL_S = original
            C._require(not changed, "scope 안 SECOND wall이 바뀌었다")


def execute(*, protocol: Path, anchor: str, private_plan: Path, transport_root: Path,
            lab: Path, isaac_python: Path, out: Path, approver: str) -> dict[str, Any]:
    plan = verify_protocol(protocol, anchor=anchor, private_plan=private_plan)
    second_plan = C._bound(plan["second_plan"], "SECOND plan")
    sources = registered_sources(second_plan)
    parent = plan["parent_anchor"]
    with ExitStack() as stack:
        # parent scope의 실제 원문 검증도 immutable Git 조회만 재사용한다.
        # V3는 별도 scope를 여므로 proxy를 원복한 뒤 기존 실행기를 호출한다.
        with V3.CACHE.proof_scope(repo=ROOT):
            stack.enter_context(BIND.epoch2_snapshot_scope(sources))
        with remaining_wall_scope(plan["remaining_wall_s"]):
            return V3.run_second_operational(
                second_plan, host_protocol=C._bound(plan["host_protocol"], "host protocol"),
                host_anchor=parent,
                compatibility_protocol=C._bound(plan["compatibility_protocol"], "compatibility"),
                compatibility_anchor=parent, cache_protocol=C._bound(plan["cache_protocol"], "cache"),
                cache_anchor=parent, transport_root=transport_root,
                lab=lab, isaac_python=isaac_python, out=out, approver=approver)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    for name in ("protocol", "private-plan", "transport-root", "lab", "isaac-python", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approver", choices=[C.RUN.APPROVER], required=True)
    args = parser.parse_args(argv)
    try:
        execute(protocol=args.protocol, anchor=args.anchor, private_plan=args.private_plan,
                transport_root=args.transport_root, lab=args.lab, isaac_python=args.isaac_python,
                out=args.out, approver=args.approver)
    except BaseException as error:
        C.RUN._record_cli_failure(args.out, "run-second-parent-retry", error)
        trace = args.out / "parent-traceback.txt"
        if not trace.exists():
            with trace.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
