# pyright: reportMissingImports=false
"""R3 성과와 분리해 두 F2 사례의 P_torque 관측 능력만 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_lease_context_v2 as LEASE
import lineage_probe_cost_audit_v1 as AUDIT
import registered_optimizer_first_policy_v1 as OPT
import run_lineage_five_policy_v1 as RUN

B, DRIVER, HOST = RUN.B, RUN.DRIVER, RUN.HOST
SCHEMA = "f2_probe_coverage_overlay_protocol_v1"
PLAN_SCHEMA = "f2_probe_coverage_private_plan_v1"
RESULT_SCHEMA = "f2_probe_coverage_result_v1"
ORACLE_POLICY = "registered_f2_torque_oracle_control"
HOST_STORAGE_POLICY = "registry_next"
PROBE = "P_torque"
TARGETS = {
    "case-05": {"arm": "F2", "seed": 2554},
    "case-06": {"arm": "F2", "seed": 2555},
}
SOURCE_FILES = (
    "bench/run_f2_probe_coverage_v1.py",
    "tests/test_f2_probe_coverage_v1.py",
    "evals/lineage_probe_cost_audit_v1.py",
    "tests/test_lineage_probe_cost_audit_v1.py",
    *OPT.SOURCE_FILES,
    *LEASE.SOURCE_FILES,
)
HEX40 = frozenset("0123456789abcdef")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object가 필요하다")
    return value


def write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    return {"path": str(target), "sha256": sha(target)}


def bound(value: object, label: str) -> Path:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, f"{label} ref")
    row = cast(Mapping[str, Any], value)
    path = Path(cast(str, row["path"])).resolve(strict=True)
    require(sha(path) == row["sha256"], f"{label} SHA")
    return path


def _hex40(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX40


def _source_map() -> dict[str, str]:
    return {name: sha(ROOT / name, lf=True) for name in sorted(set(SOURCE_FILES))}


def _validate_plan(plan: dict[str, Any]) -> None:
    targets = plan.get("targets", [])
    require(plan.get("schema") == PLAN_SCHEMA
            and [row.get("case_id") for row in targets] == list(TARGETS)
            and all(row.get("oracle_label", "F2") == "F2"
                    and row.get("action") == ("reuse" if row.get("historical") is not None
                                               else "execute_one_pair")
                    for row in targets), "private oracle plan")
    require(plan.get("oracle_budget") == {
                "pair_cap": 2, "sdk_role_cap": 4, "api_cap": 0,
                "wall_cap_s": 3600, "automatic_retry": False,
            }
            and plan.get("selection_performance_credit") == 0
            and plan.get("model_calls") == 0
            and plan.get("training") is False
            and plan.get("correction") is False, "oracle fixed budget/scope")
    original = plan.get("original_budget", {})
    require(original.get("paired_cap") == 150
            and original.get("sdk_role_cap") == 300
            and original.get("wall_cap_s") == 144000, "original R3 caps")


def _target_material(source: dict[str, Any], inputs: Mapping[str, Path], *,
                     case_id: str, checker: Callable = RUN.HOST.checked) -> dict[str, Any]:
    require(source.get("case_id") == case_id
            and source.get("qualification", {}).get("arm") == "F2", "oracle F2 label")
    case_bridge, case_raw = checker(inputs["case"])
    reference_bridge, reference_raw = checker(inputs["reference"])
    expected_seed = TARGETS[case_id]["seed"]
    require(case_raw.get("manifest", {}).get("arm") == "F2"
            and reference_raw.get("manifest", {}).get("arm") == "NONE"
            and case_raw.get("manifest", {}).get("seed") == expected_seed
            and reference_raw.get("manifest", {}).get("seed") == expected_seed,
            "oracle source role/seed")
    bindings = source["packet"]["input_bindings"]
    require(case_bridge.get("binding", {}).get("checkpoint_sha256")
            == bindings["case"]["checkpoint_sha256"]
            and reference_bridge.get("binding", {}).get("checkpoint_sha256")
            == bindings["reference"]["checkpoint_sha256"], "oracle role checkpoint")
    return {"case_checkpoint_sha256": bindings["case"]["checkpoint_sha256"],
            "reference_checkpoint_sha256": bindings["reference"]["checkpoint_sha256"]}


def _historical_torque(batch: dict[str, Any], case_id: str, source: dict[str, Any],
                       *, inspect: Callable = AUDIT.inspect_step) -> dict[str, Any] | None:
    matches = []
    with tempfile.TemporaryDirectory(prefix="f2-oracle-history-") as folder:
        temp = Path(folder)
        for outer_index, outer in enumerate(batch["rows"]):
            if outer.get("case_id") != case_id:
                continue
            detail_path = bound(outer.get("result"), "R3 row result")
            detail = read(detail_path)
            for step_index, step in enumerate(detail.get("steps", [])):
                if step.get("probe") != PROBE:
                    continue
                checked = inspect(detail_path, source, step_index,
                                  temp / f"{outer_index:02d}-{step_index:02d}")
                require(checked.get("probe") == PROBE
                        and checked.get("outcome") in {"normal", "abnormal", "unknown"},
                        "historical P_torque receipt")
                matches.append({"policy": outer["arm"], "detail": ref(detail_path),
                                "step_index": step_index, "verified": checked})
    if not matches:
        return None
    return matches[0]


def prepare(batch_result: Path, r3_protocol: Path, config_path: Path,
            semantic_anchor: str, out: Path, *,
            preflight: Callable = RUN.preflight,
            input_checker: Callable = RUN.HOST.checked,
            inspect: Callable = AUDIT.inspect_step) -> dict[str, Any]:
    require(_hex40(semantic_anchor) and not Path(out).exists(), "semantic anchor/fresh plan")
    protocol, config, prepared = preflight(r3_protocol, semantic_anchor, config_path)
    require(protocol.get("paired_cap") == 150
            and protocol.get("sdk_role_cap") == 300
            and protocol.get("wall_cap_s") == 144000, "original R3 caps")
    batch = read(Path(batch_result).resolve(strict=True))
    rows = batch.get("rows")
    require(batch.get("schema") == "lineage_five_policy_batch_result_v1"
            and batch.get("status") == "complete" and batch.get("planned_cases") == 10
            and batch.get("planned_rows") == 50 and isinstance(rows, list) and len(rows) == 50
            and batch.get("execution_protocol") == B.ref(r3_protocol)
            and batch.get("execution_config") == B.ref(config_path)
            and batch.get("execution_anchor") == semantic_anchor
            and batch.get("repair") is False and batch.get("recovery") is False,
            "complete original R3 batch50")
    require(type(batch.get("paired_intents")) is int and 0 <= batch["paired_intents"] <= 150
            and batch.get("sdk_roles_reserved") == batch["paired_intents"] * 2
            and isinstance(batch.get("parent_wall_s"), (int, float))
            and 0 <= batch["parent_wall_s"] <= 144000, "original R3 cumulative budget")
    targets = []
    for case_id in TARGETS:
        item = prepared[case_id]
        source, inputs = item.get("source"), item.get("inputs")
        require(isinstance(source, dict) and isinstance(inputs, Mapping)
                and set(inputs) == {"case", "reference"}, "oracle actual source/input")
        bindings = _target_material(source, cast(Mapping[str, Path], inputs),
                                    case_id=case_id, checker=input_checker)
        historical = _historical_torque(batch, case_id, source, inspect=inspect)
        targets.append({"case_id": case_id, "oracle_label": "F2",
                        "source_digest": B.digest(source), "bindings": bindings,
                        "inputs": {role: ref(Path(path)) for role, path in inputs.items()},
                        "historical": historical,
                        "action": "reuse" if historical is not None else "execute_one_pair"})
    plan = {"schema": PLAN_SCHEMA, "status": "prepared_private_not_executable",
            "batch_result": ref(Path(batch_result)), "r3_protocol": ref(Path(r3_protocol)),
            "execution_config": ref(Path(config_path)), "semantic_anchor": semantic_anchor,
            "source_index": deepcopy(config["source_index"]),
            "source_index_sha256": config["source_index"]["sha256"],
            "original_budget": {"paired_cap": protocol["paired_cap"],
                                "sdk_role_cap": protocol["sdk_role_cap"],
                                "wall_cap_s": protocol["wall_cap_s"],
                                "paired_consumed": batch["paired_intents"],
                                "sdk_roles_consumed": batch["sdk_roles_reserved"],
                                "wall_consumed_s": batch["parent_wall_s"]},
            "oracle_budget": {"pair_cap": 2, "sdk_role_cap": 4, "api_cap": 0,
                              "wall_cap_s": 3600, "automatic_retry": False},
            "targets": targets, "selection_performance_credit": 0,
            "model_calls": 0, "training": False, "correction": False}
    write_new(out, plan)
    return deepcopy(plan)


def build_protocol(plan_path: Path, out: Path) -> dict[str, Any]:
    value = build_protocol_value(plan_path)
    write_new(out, value)
    return deepcopy(value)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def verify_protocol(protocol_path: Path, registration_anchor: str, plan_path: Path, *,
                    lab: Path, repo: Path = ROOT,
                    preflight: Callable = RUN.preflight) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(_hex40(registration_anchor), "overlay registration anchor")
    protocol, plan = read(protocol_path), read(plan_path)
    expected = build_protocol_value(plan_path)
    require(protocol == expected and protocol.get("status") == "draft_not_executable_until_committed",
            "overlay protocol/plan 결합")
    repo = Path(repo).resolve()
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                            registration_anchor, head], check=False).returncode == 0,
            "overlay registration ancestry")
    relative = Path(protocol_path).resolve(strict=True).relative_to(repo).as_posix()
    for name, expected_lf in {**protocol["source_files_lf"], relative: sha(protocol_path, lf=True)}.items():
        blob = _git(repo, "show", f"{registration_anchor}:{name}")
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(normalized).hexdigest() == expected_lf,
                "overlay committed blob 변경: " + name)
    batch = read(bound(plan["batch_result"], "R3 batch"))
    require(batch.get("status") == "complete" and len(batch.get("rows", [])) == 50,
            "original complete50 required")
    r3_protocol = bound(plan["r3_protocol"], "R3 protocol")
    config = bound(plan["execution_config"], "R3 config")
    original, raw_config, prepared = preflight(r3_protocol, plan["semantic_anchor"], config)
    require(protocol["r3"] == {
                "batch_sha256": plan["batch_result"]["sha256"],
                "protocol_sha256": plan["r3_protocol"]["sha256"],
                "config_sha256": plan["execution_config"]["sha256"],
                "source_index_sha256": plan["source_index_sha256"],
                "semantic_anchor": plan["semantic_anchor"],
            }
            and batch.get("execution_protocol") == plan["r3_protocol"]
            and batch.get("execution_config") == plan["execution_config"]
            and batch.get("execution_anchor") == plan["semantic_anchor"],
            "overlay/original R3 binding")
    require(Path(raw_config["lab"]).resolve() == Path(lab).resolve(strict=True), "registered lab")
    return protocol, plan, {"protocol": original, "config": raw_config, "prepared": prepared,
                            "batch": batch}


def build_protocol_value(plan_path: Path) -> dict[str, Any]:
    plan = read(Path(plan_path).resolve(strict=True))
    _validate_plan(plan)
    return {"schema": SCHEMA, "status": "draft_not_executable_until_committed",
            "source_files_lf": _source_map(),
            "private_plan_sha256": sha(plan_path), "private_plan_digest": digest(plan),
            "r3": {"batch_sha256": plan["batch_result"]["sha256"],
                   "protocol_sha256": plan["r3_protocol"]["sha256"],
                   "config_sha256": plan["execution_config"]["sha256"],
                   "source_index_sha256": plan["source_index_sha256"],
                   "semantic_anchor": plan["semantic_anchor"]},
            "targets": [{"case_id": row["case_id"], "oracle_label": "F2",
                         "case_checkpoint_sha256": row["bindings"]["case_checkpoint_sha256"],
                         "reference_checkpoint_sha256": row["bindings"]["reference_checkpoint_sha256"],
                         "action": row["action"]} for row in plan["targets"]],
            "oracle_policy": {"name": ORACLE_POLICY, "probe": PROBE,
                              "host_storage_policy": HOST_STORAGE_POLICY,
                              "selection_performance_credit": 0,
                              "new_model_calls": 0, "training": False, "correction": False},
            "budget": deepcopy(plan["oracle_budget"]),
            "original_budget": deepcopy(plan["original_budget"])}


def oracle_choice(snapshot: dict[str, Any]) -> dict[str, Any]:
    checked = DRIVER.C.V.checked(snapshot)
    require(checked["inputs"]["policy"] == HOST_STORAGE_POLICY
            and checked["view"]["status"] == "open"
            and PROBE in checked["view"]["eligible_probes"],
            "P_torque oracle eligibility")
    choice = OPT._continue(checked, PROBE, "registered_f2_oracle_coverage")
    choice["policy"] = HOST_STORAGE_POLICY
    return choice


def classify(outcome: str) -> str:
    require(outcome in {"normal", "abnormal", "unknown"}, "P_torque outcome")
    return {"abnormal": "confirmed_signature", "normal": "signature_not_observed",
            "unknown": "unknown_HOLD"}[outcome]


def execute(protocol_path: Path, registration_anchor: str, plan_path: Path, *, lab: Path,
            out: Path, pair_executor: Callable = HOST.execute_pair,
            pair_verifier: Callable = HOST.verify_pair,
            lease_reader: Callable = LEASE.lease_context,
            validator: Callable = verify_protocol) -> dict[str, Any]:
    protocol, plan, material = validator(protocol_path, registration_anchor, plan_path, lab=lab)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = lease_reader()
    require(initial.get("owner") == RUN.LEASE.OWNER and isinstance(initial.get("leaseId"), str),
            "oracle common lease")
    started, new_pairs = time.monotonic(), 0
    remaining_wall = (plan["original_budget"]["wall_cap_s"]
                      - plan["original_budget"]["wall_consumed_s"])
    require(remaining_wall >= 0, "oracle cumulative wall cap")
    results = []

    def boundary() -> dict[str, Any]:
        current = lease_reader()
        require(current.get("owner") == initial["owner"]
                and current.get("leaseId") == initial["leaseId"], "oracle lease identity")
        elapsed = time.monotonic() - started
        require(elapsed <= protocol["budget"]["wall_cap_s"]
                and elapsed <= remaining_wall, "oracle cumulative wall cap")
        require(sha(plan_path) == protocol["private_plan_sha256"]
                and digest(read(plan_path)) == protocol["private_plan_digest"], "oracle plan 변경")
        return {"approved": True}

    try:
        for target in plan["targets"]:
            case_id = target["case_id"]
            source = material["prepared"][case_id]["source"]
            require(B.digest(source) == target["source_digest"], "oracle source digest")
            historical = target.get("historical")
            if historical is not None:
                checked = AUDIT.inspect_step(bound(historical["detail"], "historical detail"), source,
                                             historical["step_index"], out / "history" / case_id)
                require(checked == historical["verified"], "historical receipt 재검증")
                outcome, execution = checked["outcome"], "reused_existing_receipt"
            else:
                remaining_pairs = plan["original_budget"]["paired_cap"] - plan["original_budget"]["paired_consumed"]
                remaining_sdk = plan["original_budget"]["sdk_role_cap"] - plan["original_budget"]["sdk_roles_consumed"]
                require(new_pairs < protocol["budget"]["pair_cap"]
                        and new_pairs + 1 <= remaining_pairs
                        and (new_pairs + 1) * 2 <= remaining_sdk, "oracle cumulative pair/SDK cap")
                new_pairs += 1
                intent = {"case_id": case_id, "probe": PROBE, "pair": new_pairs,
                          "sdk_roles_reserved": 2, "automatic_retry": False,
                          "control_policy": ORACLE_POLICY,
                          "host_storage_policy": HOST_STORAGE_POLICY}
                write_new(out / "paired-intents" / f"{new_pairs:02d}.json", intent)
                ledger = out / case_id / "oracle.ledger.jsonl"
                snapshot = DRIVER.C.V.build_snapshot(
                    source["packet"], f"f2_oracle_{case_id.replace('-', '_')}",
                    HOST_STORAGE_POLICY,
                    DRIVER.C.V.file_digest(DRIVER.C.V.PROTOCOL), ledger)
                choice = oracle_choice(snapshot)
                inputs = material["prepared"][case_id]["inputs"]
                tools = material["protocol"]["tool_protocols"]
                pair = pair_executor(
                    snapshot, choice, backend_packet=source["backend_packet"],
                    case_input=inputs["case"], reference_input=inputs["reference"],
                    probe_protocol=ROOT / tools["paired_probe"]["path"],
                    view_protocol=ROOT / tools["view"]["path"],
                    anchor=plan["semantic_anchor"], lab=lab, out=out / case_id / "pair",
                    verify_context=lambda *_a, **_k: boundary())
                verified = pair_verifier(Path(pair["result_ref"]["path"]), snapshot, choice)
                require(verified.get("probe") == PROBE and verified.get("retry_count") == 0,
                        "oracle pair receipt")
                outcome, execution = verified["outcome"], "executed_one_pair"
            results.append({"case_id": case_id, "oracle_label": "F2", "probe": PROBE,
                            "outcome": outcome, "coverage": classify(outcome),
                            "execution": execution, "control_policy": ORACLE_POLICY,
                            "host_storage_policy": HOST_STORAGE_POLICY,
                            "selected_by": "registered_rule",
                            "selection_performance_credit": 0})
        result = {"schema": RESULT_SCHEMA, "status": "complete", "targets": results,
                  "new_pairs": new_pairs, "sdk_roles_reserved": new_pairs * 2, "api_calls": 0,
                  "model_calls": 0, "training": False, "correction": False,
                  "original_comparison_modified": False, "selection_performance_credit": 0,
                  "lease_id": initial["leaseId"], "parent_wall_s": time.monotonic() - started}
        require(result["parent_wall_s"] <= protocol["budget"]["wall_cap_s"]
                and result["parent_wall_s"] <= remaining_wall,
                "oracle cumulative wall cap")
        write_new(out / "result.json", result)
        return result
    except BaseException as error:
        write_new(out / "failure.json", {"status": "invalid", "error_type": type(error).__name__,
                                         "error": str(error), "new_pairs": new_pairs,
                                         "automatic_retry": False})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("batch-result", "r3-protocol", "config", "semantic-anchor", "out"):
        prep.add_argument("--" + name, required=True, type=None if "anchor" in name else Path)
    build = sub.add_parser("build-protocol")
    build.add_argument("--plan", type=Path, required=True); build.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    execute_parser = sub.add_parser("execute")
    for child in (verify, execute_parser):
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--registration-anchor", required=True)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--lab", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.batch_result, args.r3_protocol, args.config, args.semantic_anchor, args.out)
    elif args.command == "build-protocol":
        build_protocol(args.plan, args.out)
    elif args.command == "verify":
        protocol, plan, _material = verify_protocol(
            args.protocol, args.registration_anchor, args.plan, lab=args.lab)
        write_new(args.out, {"status": "complete", "gpu": 0, "api": 0,
                             "protocol_digest": digest(protocol), "plan_digest": digest(plan)})
    else:
        execute(args.protocol, args.registration_anchor, args.plan, lab=args.lab, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
