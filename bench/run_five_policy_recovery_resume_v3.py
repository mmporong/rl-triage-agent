# pyright: reportMissingImports=false
"""R4 SECOND에서 불변 Git blob 조회만 재사용하는 등록 실행 경계."""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import recovery_git_blob_cache_v1 as CACHE
import run_five_policy_recovery_resume_v2 as CONT

SCHEMA = "five_policy_recovery_second_git_proof_protocol_v1"
SOURCE_FILES = tuple(sorted({
    *CONT.SOURCE_FILES,
    "evals/recovery_git_blob_cache_v1.py",
    "tests/test_recovery_git_blob_cache_v1.py",
    "bench/run_five_policy_recovery_resume_v3.py",
    "tests/test_five_policy_recovery_resume_v3.py",
}))
POLICY = {
    "cached_value": "successful_exact_commit_git_show_only",
    "scope": "one_second_parent_process",
    "current_source_checkpoint_receipt_runtime_checks_cached": False,
    "lease_approval_and_api_checks_cached": False,
    "model_inputs_and_decisions_changed": False,
    "recovery_criteria_changed": False,
    "api_cap": 32,
    "sdk_cap": 32,
    "wall_cap_s": 43200,
}


def _source_map() -> dict[str, str]:
    return {name: CONT._sha(ROOT / name, lf=True) for name in SOURCE_FILES}


def build_protocol(*, second_plan: Path, compatibility_protocol: Path,
                   out: Path) -> dict[str, Any]:
    plan = CONT._read(second_plan)
    CONT._require(plan.get("schema") == CONT.RUN.SECOND_PLAN_SCHEMA, "SECOND plan schema")
    compatibility = CONT._read(compatibility_protocol)
    CONT._require(
        compatibility.get("schema") == CONT.SECOND_COMPATIBILITY_PROTOCOL_SCHEMA
        and compatibility.get("second_plan_sha256") == CONT._sha(second_plan),
        "SECOND compatibility plan binding")
    value = {
        "schema": SCHEMA,
        "status": "preregistered_immutable_git_proof_reuse",
        "second_plan_sha256": CONT._sha(second_plan),
        "compatibility_protocol": {
            "path": CONT._relative(compatibility_protocol),
            "sha256": CONT._sha(compatibility_protocol),
        },
        "sources_lf": _source_map(),
        "policy": deepcopy(POLICY),
    }
    # 공개 protocol에는 private plan 경로 대신 원문 SHA만 결합한다.
    CONT._write_new(out, value)
    return deepcopy(value)


def verify_protocol(path: Path, *, anchor: str, second_plan: Path,
                    compatibility_protocol: Path) -> dict[str, Any]:
    value = CONT._read(path)
    CONT._require(set(value) == {
        "schema", "status", "second_plan_sha256", "compatibility_protocol",
        "sources_lf", "policy",
    } and value.get("schema") == SCHEMA
        and value.get("status") == "preregistered_immutable_git_proof_reuse",
        "Git proof protocol shape/status")
    CONT._verify_anchor(anchor)
    CONT._verify_blob(anchor, CONT._relative(path), CONT._sha(path, lf=True))
    CONT._require(value["sources_lf"] == _source_map(), "Git proof current source SHA")
    for name, digest in value["sources_lf"].items():
        CONT._verify_blob(anchor, name, digest)
    CONT._require(value["policy"] == POLICY
                  and value["second_plan_sha256"] == CONT._sha(second_plan)
                  and value["compatibility_protocol"] == {
                      "path": CONT._relative(compatibility_protocol),
                      "sha256": CONT._sha(compatibility_protocol)},
                  "Git proof policy/plan/compatibility binding")
    return deepcopy(value)


def run_second_operational(plan_path: Path, *, host_protocol: Path, host_anchor: str,
                           compatibility_protocol: Path, compatibility_anchor: str,
                           cache_protocol: Path, cache_anchor: str,
                           transport_root: Path, lab: Path, isaac_python: Path,
                           out: Path, approver: str) -> dict[str, Any]:
    CONT._require(cache_anchor == host_anchor == compatibility_anchor,
                  "SECOND outer anchors must match")
    verify_protocol(cache_protocol, anchor=cache_anchor, second_plan=plan_path,
                    compatibility_protocol=compatibility_protocol)
    evidence_path = out.parent / (out.name + "-git-proof-cache.json")
    CONT._require(not evidence_path.exists(), "새 Git proof evidence 경로가 필요하다")
    stats: dict[str, Any] | None = None
    try:
        with CACHE.proof_scope(repo=ROOT) as stats:
            return CONT.run_second_operational(
                plan_path, host_protocol=host_protocol, host_anchor=host_anchor,
                compatibility_protocol=compatibility_protocol,
                compatibility_anchor=compatibility_anchor,
                transport_root=transport_root, lab=lab, isaac_python=isaac_python,
                out=out, approver=approver)
    finally:
        if stats is not None:
            CONT._write_new(evidence_path, {
                "schema": "five_policy_recovery_second_git_proof_evidence_v1",
                "protocol": CONT._ref(cache_protocol), "execution_anchor": cache_anchor,
                "stats": stats,
                "scope": "parent_CPU_validation_not_GPU_kernel_or_model_gain",
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-protocol")
    for name in ("second-plan", "compatibility-protocol", "out"):
        build.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("run-second")
    for name in ("plan", "host-protocol", "compatibility-protocol", "cache-protocol",
                 "transport-root", "lab", "isaac-python", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    for name in ("host-anchor", "compatibility-anchor", "cache-anchor"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--approver", choices=[CONT.RUN.APPROVER], required=True)
    args = parser.parse_args(argv)
    if args.command == "build-protocol":
        build_protocol(second_plan=args.second_plan,
                       compatibility_protocol=args.compatibility_protocol, out=args.out)
    else:
        try:
            run_second_operational(
                args.plan, host_protocol=args.host_protocol, host_anchor=args.host_anchor,
                compatibility_protocol=args.compatibility_protocol,
                compatibility_anchor=args.compatibility_anchor,
                cache_protocol=args.cache_protocol, cache_anchor=args.cache_anchor,
                transport_root=args.transport_root, lab=args.lab,
                isaac_python=args.isaac_python, out=args.out, approver=args.approver)
        except BaseException as error:
            CONT.RUN._record_cli_failure(args.out, "run-second", error)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
