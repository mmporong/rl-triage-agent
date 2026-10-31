"""보관된 optimizer count 검사와 새 3-update 검사의 일치·비용을 독립 비교한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import diagnostic_fact_ledger_v1 as FACTS
import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import prospective_diagnosis_inputs_v2 as INPUT
import recorded_optimizer_probe_v1 as RECORDED
import run_prospective_agentic_comparison_v1 as PARENT
import runtime_observation_probe_v1 as OBS

SCHEMA = "recorded_optimizer_probe_audit_protocol_v1"
PREFLIGHT_SCHEMA = "recorded_optimizer_probe_audit_preflight_v1"
RESULT_SCHEMA = "recorded_optimizer_probe_audit_result_v1"
FRESH_PROBE = "P_optimizer_updates"
SEEDS = (2320, 2321)
ORDER = {2320: (RECORDED.PROBE, FRESH_PROBE), 2321: (FRESH_PROBE, RECORDED.PROBE)}
MAX_WALL_S = 2400
MAX_SDK_LAUNCHES = 4
SOURCE_FILES = tuple(sorted(set(PARENT.SOURCE_FILES) | {
    "evals/recorded_optimizer_probe_audit_v1.py",
    "tests/test_recorded_optimizer_probe_audit_v1.py",
}))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _repo_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("audit protocol 입력은 저장소 안에 있어야 한다") from error
    return {"path": relative, "sha256_lf": _sha(path, lf=True)}


def _git(*args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True).stdout


def prepare_protocol(comparison_protocol: Path, training_protocol: Path,
                     source_batch_result: Path, probe_protocol: Path,
                     view_protocol: Path) -> dict[str, Any]:
    comparison = _read(comparison_protocol)
    _require(comparison.get("schema") == PARENT.SCHEMA, "comparison V3 protocol이 아니다")
    refs = comparison.get("protocols", {})
    expected = {"training": _repo_ref(training_protocol), "paired_probe": _repo_ref(probe_protocol),
                "view": _repo_ref(view_protocol)}
    _require(all(refs.get(name) == row for name, row in expected.items()),
             "comparison transitive protocol ref 불일치")
    source = _read(source_batch_result)
    _require(source.get("schema") == PARENT.SOURCE.BATCH_SCHEMA
             and source.get("status") == "complete", "source batch가 complete가 아니다")
    for relative in SOURCE_FILES:
        _require((ROOT / relative).is_file(), f"audit source closure가 없다: {relative}")
    return {
        "schema": SCHEMA, "purpose": "recorded_vs_fresh_optimizer_count_consistency_cost_audit",
        "protocols": {"comparison": _repo_ref(comparison_protocol),
                      "training": expected["training"], "paired_probe": expected["paired_probe"],
                      "view": expected["view"]},
        "source_batch": {"sha256": _sha(source_batch_result),
                         "source_anchor": source["anchor"], "jobs": 17},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(comparison["runtime_source_sha256_lf"]),
        "design": {"seeds": list(SEEDS), "case_category": "R1_ONLY",
                   "reference_category": "NONE",
                   "order": {str(seed): list(ORDER[seed]) for seed in SEEDS},
                   "model_requests": 0, "repairs": 0,
                   "source_failure_required": False,
                   "main_comparison_healthy_gate_unchanged": True},
        "bounds": {"wall_s": MAX_WALL_S, "new_sdk_launches": MAX_SDK_LAUNCHES,
                   "fresh_roles": 4, "recorded_sdk_launches": 0},
        "claims": {"gpu_kernel_savings": False, "early_detection": False,
                   "recovery": False, "model_advantage": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path, *,
                    source_batch_result: Path) -> dict[str, Any]:
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", anchor)), "audit anchor 형식")
    value = _read(path)
    refs = cast(dict[str, dict[str, str]], value.get("protocols"))
    expected = prepare_protocol(
        ROOT / refs["comparison"]["path"], ROOT / refs["training"]["path"],
        source_batch_result, ROOT / refs["paired_probe"]["path"], ROOT / refs["view"]["path"])
    _require(value == expected, "audit protocol/current source가 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    self_ref = _repo_ref(path)
    self_blob = _git("show", f"{anchor}:{self_ref['path']}").replace(b"\r\n", b"\n")
    _require(hashlib.sha256(self_blob).hexdigest() == self_ref["sha256_lf"],
             "audit protocol anchor blob 불일치")
    for relative, digest in value["source_sha256_lf"].items():
        blob = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest, f"audit anchor blob 불일치: {relative}")
    comparison = ROOT / refs["comparison"]["path"]
    PARENT.verify_protocol(comparison, anchor, lab)
    return deepcopy(value)


def _source_path(protocol: dict[str, Any]) -> Path:
    expected = protocol.get("source_batch", {}).get("sha256")
    path_text = protocol.get("_source_batch_path")
    if isinstance(path_text, str):
        path = Path(path_text).resolve()
        _require(path.is_file() and _sha(path) == expected, "audit source batch ref 불일치")
        return path
    raise ValueError("source batch path는 실행 인자로 결합해야 한다")


def _cases(protocol: dict[str, Any], anchor: str) -> dict[int, tuple[dict, dict]]:
    comparison = _read(ROOT / protocol["protocols"]["comparison"]["path"])
    rows = PARENT._source_cases(_source_path(protocol), comparison, anchor)
    result = {}
    for seed in SEEDS:
        case = next(row for row in rows if row["seed"] == seed and row["category"] == "R1_ONLY")
        reference = next(row for row in rows if row["seed"] == seed and row["category"] == "NONE")
        result[seed] = (case, reference)
    return result


def _verify_current(protocol_path: Path, protocol: dict[str, Any], source_batch_result: Path,
                    lab: Path) -> None:
    public = {key: value for key, value in protocol.items() if key != "_source_batch_path"}
    _require(_read(protocol_path) == public, "audit 실행 중 protocol이 바뀌었다")
    refs = public["protocols"]
    expected = prepare_protocol(
        ROOT / refs["comparison"]["path"], ROOT / refs["training"]["path"],
        source_batch_result, ROOT / refs["paired_probe"]["path"], ROOT / refs["view"]["path"])
    _require(public == expected, "audit 실행 중 source/protocol closure가 바뀌었다")
    for relative, digest in public["runtime_source_sha256_lf"].items():
        target = Path(lab).resolve() / relative
        _require(target.is_file() and _sha(target, lf=True) == digest,
                 f"audit 실행 중 runtime pin이 바뀌었다: {relative}")


def _harness(case: dict, reference: dict, folder: Path, target: str) -> tuple[dict, dict]:
    packet = INPUT.make_packet(case["input"], reference["input"], budget_left=3)
    ledger = FACTS.build(Path(case["metadata"]["runtime_facts"]["case"]["path"]),
                         Path(reference["metadata"]["runtime_facts"]["reference"]["path"]))
    ledger_path = folder / "fact-ledger.json"
    FACTS.write_new(ledger_path, ledger)
    state = SEL.initialize(packet["packet"], ledger_path)
    if target == RECORDED.PROBE:
        state = SEL.advance(state, {"action": "read_evidence", "target": SEL.RESOURCE,
                                    "fact_ids": []})
        action = SEL.select_strong_rule(state)
        _require(action["target"] == RECORDED.PROBE, "recorded audit rule 조건이 성립하지 않는다")
    else:
        action = {"action": "run_probe", "target": FRESH_PROBE, "fact_ids": []}
        # 기존 registry rule과 같은 host 생성 action이며 모델 assessor를 통과한 척하지 않는다.
        SEL.request_spec(state, action)
    return state, action


def preflight(protocol_path: Path, anchor: str, source_batch_result: Path,
              lab: Path, out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "audit preflight out은 fresh여야 한다")
    checked = verify_protocol(protocol_path, anchor, lab,
                              source_batch_result=source_batch_result)
    protocol = {**checked, "_source_batch_path": str(Path(source_batch_result).resolve())}
    cases = _cases(protocol, anchor)
    out.mkdir(parents=True, exist_ok=False)
    plans = []
    for seed in SEEDS:
        for target in ORDER[seed]:
            state, action = _harness(*cases[seed], out / f"s{seed}/{target}", target)
            plans.append({"seed": seed, "target": target, "state_digest": state["state_digest"],
                          "request_spec": SEL.request_spec(state, action)})
    value = {"schema": PREFLIGHT_SCHEMA, "status": "ready", "plans": plans,
             "model_requests": 0, "sdk_launches": 0, "gpu_started": False}
    _write(out / "preflight.json", value)
    return value


def _canonical(target: str, role: str, plan: Path, receipt: Path) -> dict[str, Any]:
    if target == RECORDED.PROBE:
        checked = RECORDED.checked_receipt(plan, receipt)
        measurement = checked["measurement"]
        return {"role": role, "expected_optimizer_steps": measurement["expected_optimizer_steps"],
                "optimizer_steps": [measurement["optimizer_steps_min"],
                                    measurement["optimizer_steps_max"]],
                "expected_minibatches": measurement["expected_minibatches"],
                "minibatches": [measurement["minibatches_min"], measurement["minibatches_max"]],
                "outcome": checked["outcome"], "records": 3}
    checked = OBS.verify_receipt(plan, receipt)
    if checked["status"] != "complete":
        return {"role": role, "outcome": "unknown", "records": None}
    records_path = Path(checked["raw_refs"]["records"]["path"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == 3 and [row.get("update_index") for row in rows] == [0, 1, 2],
             "fresh optimizer rows가 3 update가 아니다")
    expected_steps = {row["expected"]["optimizer_steps"] for row in rows}
    expected_batches = {row["expected"]["minibatches"] for row in rows}
    steps = [row["observed"]["optimizer_steps"] for row in rows]
    batches = [row["observed"]["minibatches"] for row in rows]
    _require(len(expected_steps) == len(expected_batches) == 1, "fresh expected count 불일치")
    normal = (len(set(steps)) == len(set(batches)) == 1
              and steps[0] == next(iter(expected_steps)) and batches[0] == next(iter(expected_batches)))
    return {"role": role, "expected_optimizer_steps": next(iter(expected_steps)),
            "optimizer_steps": [min(steps), max(steps)],
            "expected_minibatches": next(iter(expected_batches)),
            "minibatches": [min(batches), max(batches)],
            "outcome": "normal" if normal else "abnormal", "records": 3}


def execute(protocol_path: Path, anchor: str, source_batch_result: Path, lab: Path, out: Path,
            *, execute_pair: Callable = HOST.execute_pair) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "audit out은 fresh여야 한다")
    checked = verify_protocol(protocol_path, anchor, lab,
                              source_batch_result=source_batch_result)
    protocol = {**checked, "_source_batch_path": str(Path(source_batch_result).resolve())}
    cases = _cases(protocol, anchor)
    lease = PARENT._lease()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in SEEDS:
        case, reference = cases[seed]
        methods = {}
        for target in ORDER[seed]:
            state, action = _harness(case, reference, out / f"s{seed}/{target}/harness", target)
            def verify(_context=None):
                elapsed = time.monotonic() - started
                _require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "audit wall 상한 종료")
                _verify_current(protocol_path, protocol, source_batch_result, lab)
                _require(PARENT._lease().get("lease_id") == lease.get("lease_id"),
                         "audit lease identity 변경")
                _require(_sha(source_batch_result) == protocol["source_batch"]["sha256"],
                         "audit source batch 변경")
                return {"approved": True}
            pair = execute_pair(
                state, action, case_input=case["input"], reference_input=reference["input"],
                case_runtime=Path(case["metadata"]["runtime_facts"]["case"]["path"]),
                reference_runtime=Path(reference["metadata"]["runtime_facts"]["reference"]["path"]),
                probe_protocol=ROOT / protocol["protocols"]["paired_probe"]["path"],
                view_protocol=ROOT / protocol["protocols"]["view"]["path"], anchor=anchor,
                lab=lab, out=out / f"s{seed}/{target}/probe", verify_context=verify,
                approver="registered-audit-rule")
            next_state = SEL.advance(state, action, **{name: Path(pair[name]) for name in (
                "case_plan", "case_receipt", "reference_plan", "reference_receipt")})
            measurements = {role: _canonical(target, role, Path(pair[f"{role}_plan"]),
                                               Path(pair[f"{role}_receipt"]))
                            for role in ("case", "reference")}
            methods[target] = {"measurements": measurements,
                               "paired_outcome": next_state["observed"][target],
                               "parent_wall_s": pair["parent_wall_s"],
                               "owned_child_wall_s": pair["owned_child_wall_s"],
                               "sdk_launches": 0 if target == RECORDED.PROBE else 2}
        recorded, fresh = methods[RECORDED.PROBE], methods[FRESH_PROBE]
        agreement = (recorded["measurements"] == fresh["measurements"]
                     and recorded["paired_outcome"] == fresh["paired_outcome"])
        rows.append({"seed_id": f"s{seed}", "order": list(ORDER[seed]), "methods": methods,
                     "agreement": agreement,
                     "recorded_parent_wall_lower": recorded["parent_wall_s"] < fresh["parent_wall_s"]})
    sdk = sum(method["sdk_launches"] for row in rows for method in row["methods"].values())
    _require(sdk <= MAX_SDK_LAUNCHES, "audit SDK launch 상한 초과")
    candidate = all(row["agreement"] and row["recorded_parent_wall_lower"] for row in rows)
    value = {"schema": RESULT_SCHEMA, "status": "complete", "rows": rows,
             "agreement_count": sum(row["agreement"] for row in rows), "planned_seeds": 2,
             "sdk_launches": sdk, "model_requests": 0, "repairs": 0,
             "conditional_candidate": candidate,
             "candidate_statement": ("보관된 계측이 있는 조건의 더 가벼운 검사 경로"
                                     if candidate else "미해결 또는 비용 동률"),
             "source_failure_required": False, "main_healthy_gate_changed": False,
             "claims": deepcopy(protocol["claims"]), "wall_s": time.monotonic() - started}
    _write(out / "result.json", value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("comparison-protocol", "training-protocol", "source-batch-result",
                 "probe-protocol", "view-protocol", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for command in ("preflight", "execute"):
        run = sub.add_parser(command)
        for name in ("protocol", "source-batch-result", "lab", "out"):
            run.add_argument(f"--{name}", type=Path, required=True)
        run.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        value = prepare_protocol(args.comparison_protocol, args.training_protocol,
                                 args.source_batch_result, args.probe_protocol, args.view_protocol)
        _write(args.out, value)
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.source_batch_result, args.lab, args.out)
    else:
        execute(args.protocol, args.anchor, args.source_batch_result, args.lab, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
