# pyright: reportMissingImports=false
"""R3의 네 고정 target을 first 교정과 조건부 second 교정으로 실행한다.

first와 second는 별도 등록·승인·lease 단계다. first 결과를 보기 전 second를
만들거나 실행하지 않으며, HOLD·unknown·비등록 후보를 다른 target으로 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import five_policy_recovery_candidate_v1 as FIRST
import five_policy_recovery_epoch2_v1 as SECOND
import five_policy_recovery_nim_transport_v1 as TRANSPORT
import lineage_compound_first_recovery_v1 as WAIT
import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v1 as LEASE
import run_local_pair_four_arm_v1 as LOCAL_PARENT
import run_runtime_readback_smoke as OWNED

FIRST_PLAN_SCHEMA = "five_policy_recovery_parent_first_plan_v1"
FIRST_RESULT_SCHEMA = "five_policy_recovery_parent_first_result_v1"
SECOND_PLAN_SCHEMA = "five_policy_recovery_parent_second_plan_v1"
SECOND_RESULT_SCHEMA = "five_policy_recovery_parent_second_result_v1"
TARGETS = tuple(FIRST.TARGETS)
SOURCE_FILES = tuple(sorted({
    *FIRST.SOURCE_FILES, *SECOND.SOURCE_FILES, *TRANSPORT.SOURCE_FILES,
    "bench/run_five_policy_recovery_loop_v1.py",
    "tests/test_run_five_policy_recovery_loop_v1.py",
}))
FIRST_WALL_S = 14_400.0
SECOND_WALL_S = 43_200.0
APPROVER = "codex-main-user-delegated"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required")
    return value


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _ref(path: Path) -> dict[str, str]:
    return FIRST.ref(Path(path).resolve(strict=True))


def _bound(value: object, label: str) -> Path:
    return FIRST.bound(value, label)


def _verify_sources(protocol: Mapping[str, Any]) -> None:
    pins_value = protocol.get("sources_lf")
    _require(isinstance(pins_value, dict) and bool(pins_value), "등록 source 결합이 없다")
    pins = cast(dict[Any, Any], pins_value)
    for name, expected in pins.items():
        _require(isinstance(name, str) and isinstance(expected, str)
                 and FIRST.sha(ROOT / name, lf=True) == expected,
                 "실행 source가 바뀌었다: " + str(name))


def _lease(reader: Callable[[], Mapping[str, Any]], *, owner: str = LEASE.OWNER) -> dict[str, Any]:
    value = dict(reader())
    _require(isinstance(value.get("leaseId"), str) and bool(value["leaseId"]),
             "공용 GPU lease가 없다")
    _require(value.get("owner") == owner, "공용 GPU lease owner가 다르다")
    return value


def _same_lease(before: Mapping[str, Any], reader: Callable[[], Mapping[str, Any]]) -> None:
    after = _lease(reader, owner=cast(str, before["owner"]))
    _require(after["leaseId"] == before["leaseId"], "phase 실행 중 GPU lease가 바뀌었다")


def _approval(path: Path, *, schema: str, candidate: Path, protocol: Path,
              anchor: str, seed: int, lease_id: str) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("schema") == schema and value.get("decision") == "approve"
             and isinstance(value.get("approver"), str) and bool(value["approver"].strip())
             and value.get("candidate") == _ref(candidate)
             and value.get("protocol") == _ref(protocol) and value.get("anchor") == anchor
             and value.get("registered_seed") == seed and value.get("lease_id") == lease_id,
             "operator approval/candidate/protocol/seed/lease 결합")
    return value


def _approval_for(candidate: Path, protocol: Path, *, schema: str, anchor: str, seed: int,
                  lease: Mapping[str, Any], approvals: Mapping[str, Path] | None, key: str,
                  approver: str | None, root: Path) -> Path:
    if approvals is not None and key in approvals:
        return Path(approvals[key]).resolve(strict=True)
    _require(approver == APPROVER, "명시된 사용자 위임 approver가 없다")
    target = root / "operator-approvals" / f"{key.replace(':', '_')}.json"
    _write_new(target, {"schema": schema, "decision": "approve", "approver": approver,
                        "candidate": _ref(candidate), "protocol": _ref(protocol),
                        "anchor": anchor, "registered_seed": seed,
                        "lease_id": lease["leaseId"], "generated_by": "host_operator_boundary",
                        "model_generated": False, "protocol_generated": False})
    return target


def _worker_command(root: Path, call: Path, protocol: Path, anchor: str) -> list[str]:
    args = ["-B", LOCAL_PARENT.mounted(ROOT / "evals/five_policy_recovery_nim_transport_v1.py"),
            "--worker", "--root", LOCAL_PARENT.mounted(root), "--call-dir", LOCAL_PARENT.mounted(call),
            "--protocol", LOCAL_PARENT.mounted(protocol), "--anchor", anchor]
    shell = 'set -euo pipefail; . "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
    return ["wsl.exe", "--distribution", "Ubuntu", "--exec", "bash", "-lc", shell + shlex.join(args)]


def _worker_instance():
    spec = importlib.util.spec_from_file_location("_five_policy_recovery_parent_worker_v1",
                                                  LOCAL_PARENT.__file__)
    if spec is None or spec.loader is None:
        raise ImportError("one-shot worker parent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__["worker_command"] = _worker_command
    return module


_WORKER = _worker_instance()


def prepare_first(batch_result: Path, *, five_policy_protocol: Path, config: Path,
                  recovery_protocol: Path, anchor: str, out: Path,
                  candidate_preparer: Callable[..., dict[str, Any]] = FIRST.prepare_candidate,
                  candidate_checker: Callable[..., dict[str, Any]] = FIRST.checked_candidate) -> dict[str, Any]:
    """등록된 네 target을 결과 순서와 무관하게 모두 후보/HOLD로 고정한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "first plan output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    protocol = FIRST.read(recovery_protocol)
    rows = []
    for case_id, policy in TARGETS:
        candidate = candidate_preparer(
            batch_result, protocol_path=five_policy_protocol, config_path=config,
            anchor=anchor, case_id=case_id, policy=policy)
        path = out / "candidates" / f"{case_id}_{policy}.json"
        _write_new(path, candidate)
        candidate_checker(path)
        rows.append({"case_id": case_id, "policy": policy, "candidate": _ref(path),
                     "status": candidate["status"], "hold_reason": candidate.get("hold_reason"),
                     "registered_seed": protocol["recovery_seeds"][f"{case_id}:{policy}"]})
    plan = {
        "schema": FIRST_PLAN_SCHEMA, "status": "prepared_no_execution",
        "batch_result": _ref(batch_result), "five_policy_protocol": _ref(five_policy_protocol),
        "config": _ref(config), "recovery_protocol": _ref(recovery_protocol),
        "anchor": anchor, "targets": rows, "max_sdk_executions": 8,
        "api_calls": 0, "automatic_epoch2": False, "approval_granted": False,
    }
    _write_new(out / "plan.json", plan)
    return deepcopy(plan)


def run_first(plan_path: Path, approvals: Mapping[str, Path] | None, *, lab: Path,
              isaac_python: Path, out: Path, runner: Callable, wait: Callable,
              lease_reader: Callable[[], Mapping[str, Any]],
              approver: str | None = None,
              candidate_executor: Callable[..., dict[str, Any]] = FIRST.execute_candidate) -> dict[str, Any]:
    """HOLD는 보존하고 승인된 validated target만 FIRST300/fixed26으로 실행한다."""
    plan = _read(plan_path)
    _require(plan.get("schema") == FIRST_PLAN_SCHEMA and len(plan.get("targets", [])) == 4
             and plan.get("automatic_epoch2") is False, "first parent plan")
    out = Path(out).resolve()
    _require(not out.exists(), "first result output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    lease = _lease(lease_reader)
    started = time.monotonic()
    records, executed = [], 0
    protocol = _bound(plan["recovery_protocol"], "recovery protocol")
    config = _bound(plan["config"], "five-policy config")
    protocol_sha = FIRST.sha(protocol)
    _require(len({(row["case_id"], row["policy"]) for row in plan["targets"]}) == 4
             and {(row["case_id"], row["policy"]) for row in plan["targets"]} == set(TARGETS),
             "first exact four targets")
    for row in plan["targets"]:
        key = f"{row['case_id']}:{row['policy']}"
        candidate_path = _bound(row["candidate"], "first candidate")
        candidate = FIRST.checked_candidate(candidate_path)
        if candidate["status"] == "HOLD":
            records.append({**row, "outcome": "HOLD", "result": None})
            continue
        _require(executed * 2 + 2 <= 8 and time.monotonic() - started + 3600 <= FIRST_WALL_S,
                 "first SDK/wall 예약 상한 부족")
        _require(FIRST.sha(protocol) == protocol_sha
                 and FIRST.sha(config) == plan["config"]["sha256"],
                 "first protocol/config가 바뀌었다")
        approval = _approval_for(
            candidate_path, protocol, schema=FIRST.APPROVAL_SCHEMA, anchor=plan["anchor"],
            seed=row["registered_seed"], lease=lease, approvals=approvals, key=key,
            approver=approver, root=out)
        _approval(approval, schema=FIRST.APPROVAL_SCHEMA, candidate=candidate_path,
                  protocol=protocol, anchor=plan["anchor"], seed=row["registered_seed"],
                  lease_id=lease["leaseId"])
        result = candidate_executor(
            candidate_path, protocol, approval, anchor=plan["anchor"], lab=lab,
            isaac_python=isaac_python, registered_seed=row["registered_seed"],
            out=out / "targets" / key.replace(":", "_"), runner=runner, wait=wait,
            lease_reader=lease_reader)
        _same_lease(lease, lease_reader)
        executed += 1
        records.append({**row, "outcome": result["outcome"],
                        "result": _ref(out / "targets" / key.replace(":", "_") / "result.json"),
                        "epoch2_input": result.get("epoch2_input")})
    _require(executed <= 4, "first 실행 target 상한")
    result = {"schema": FIRST_RESULT_SCHEMA, "status": "complete", "plan": _ref(plan_path),
              "lease_id": lease["leaseId"], "records": records,
              "executed_targets": executed, "sdk_execution_upper_bound": executed * 2,
              "api_calls": 0, "automatic_epoch2": False, "parent_wall_s": time.monotonic() - started,
              "second_phase_requires_new_registered_protocol": any(
                  row["outcome"] == "not_recovered" for row in records)}
    _write_new(out / "result.json", result)
    return deepcopy(result)


def prepare_second(first_result_path: Path, *, loop_protocol: Path, anchor: str,
                   lab: Path, out: Path,
                   source_preparer: Callable[..., dict[str, Any]] = SECOND.prepare_source,
                   source_checker: Callable[..., dict[str, Any]] = SECOND.checked_source,
                   protocol_verifier: Callable[..., dict[str, Any]] = SECOND.verify_loop_protocol) -> dict[str, Any]:
    """first not_recovered receipt만 새 loop protocol의 등록 subset으로 만든다."""
    first = _read(first_result_path)
    _require(first.get("schema") == FIRST_RESULT_SCHEMA and first.get("status") == "complete",
             "first parent result")
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
        epoch2 = _bound(first_row["epoch2_input"], "epoch2 input")
        source = source_preparer(epoch2, lab=lab)
        source_path = out / "sources" / f"{source['case_id']}_{source['policy']}.json"
        _write_new(source_path, source)
        source_checker(source)
        key = f"{source['case_id']}:{source['policy']}"
        _require(protocol["source_digests"].get(key) == SECOND.digest(source),
                 "second source가 loop protocol subset에 없다")
        rows.append({"case_id": source["case_id"], "policy": source["policy"],
                     "status": "registered", "source": _ref(source_path),
                     "registered_seed": protocol["second_seeds"][key]})
    plan = {"schema": SECOND_PLAN_SCHEMA, "status": "prepared_no_execution",
            "first_result": _ref(first_result_path), "loop_protocol": _ref(loop_protocol),
            "anchor": anchor, "records": rows, "registered_count": sum(
                row["status"] == "registered" for row in rows),
            "api_cap": 32, "probe_pair_cap": 12, "probe_sdk_cap": 24,
            "training_fixed_sdk_cap": 8, "total_sdk_cap": 32,
            "automatic_retry": False, "approval_granted": False}
    _write_new(out / "plan.json", plan)
    return deepcopy(plan)


def run_second(plan_path: Path, approvals: Mapping[str, Path] | None, *, transport_root: Path,
               lab: Path, isaac_python: Path, out: Path, send: Callable,
               execute_probe: Callable, busy_reader: Callable, runner: Callable,
               wait: Callable, lease_reader: Callable[[], Mapping[str, Any]],
               approver: str | None = None,
               probe_factory: Callable[[dict[str, Any]], Callable] | None = None,
               diagnosis_runner: Callable[..., dict[str, Any]] = SECOND.run_diagnosis,
               candidate_preparer: Callable[..., dict[str, Any]] = SECOND.prepare_second_candidate,
               candidate_executor: Callable[..., dict[str, Any]] = SECOND.execute_second_candidate) -> dict[str, Any]:
    """등록 subset만 진단하고 validated second candidate만 BOTH300/fixed26으로 실행한다."""
    plan = _read(plan_path)
    _require(plan.get("schema") == SECOND_PLAN_SCHEMA and plan.get("api_cap") == 32,
             "second parent plan")
    out, protocol = Path(out).resolve(), _bound(plan["loop_protocol"], "loop protocol")
    _require(not out.exists(), "second result output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    lease = _lease(lease_reader)
    started = time.monotonic()
    protocol_sha = FIRST.sha(protocol)
    identities = [(row["case_id"], row["policy"]) for row in plan["records"]]
    _require(len(identities) == 4 and len(set(identities)) == 4 and set(identities) == set(TARGETS),
             "second exact four records")
    TRANSPORT.initialize(transport_root, protocol_path=protocol,
                         execution_anchor=plan["anchor"], api_cap=32)
    records, probe_pairs, executed = [], 0, 0
    for row in plan["records"]:
        if row["status"] != "registered":
            records.append({**row, "outcome": "not_in_second_subset"})
            continue
        source_path = _bound(row["source"], "epoch2 source")
        source = SECOND.checked_source(_read(source_path))
        key = f"{row['case_id']}:{row['policy']}"
        _require(probe_pairs * 2 + 6 <= 24 and time.monotonic() - started + 3600 <= SECOND_WALL_S,
                 "second probe/wall 예약 상한 부족")
        _require(FIRST.sha(protocol) == protocol_sha, "second protocol이 바뀌었다")
        complete = (None if row["policy"] == "fixed_eligible" else
                    lambda request: TRANSPORT.complete(transport_root, request, send=send))
        dispatch = probe_factory(source) if probe_factory is not None else execute_probe
        diagnosis_dir = out / "diagnosis" / key.replace(":", "_")
        diagnosis = diagnosis_runner(
            source, arm=row["policy"], run_id=f"r4_{row['case_id'].replace('-', '_')}_{row['policy']}",
            output=diagnosis_dir, protocol_path=protocol, execution_anchor=plan["anchor"],
            lab=lab, complete=complete, execute_probe=dispatch, busy_reader=busy_reader)
        _same_lease(lease, lease_reader)
        steps = diagnosis.get("steps", [])
        _require(isinstance(steps, list) and len(steps) <= 3, "second probe pair 상한")
        probe_pairs += len(steps)
        diagnosis_path = diagnosis_dir / "result.json"
        candidate = candidate_preparer(source_path, diagnosis_path, protocol_path=protocol,
                                       anchor=plan["anchor"], lab=lab)
        candidate_path = out / "candidates" / f"{key.replace(':', '_')}.json"
        _write_new(candidate_path, candidate)
        if candidate["status"] == "HOLD":
            records.append({**row, "outcome": "HOLD", "diagnosis": _ref(diagnosis_path),
                            "candidate": _ref(candidate_path), "result": None})
            continue
        _require(executed * 2 + 2 <= 8 and time.monotonic() - started + 3600 <= SECOND_WALL_S,
                 "second training/fixed wall 예약 상한 부족")
        _require(FIRST.sha(protocol) == protocol_sha
                 and FIRST.sha(source_path) == row["source"]["sha256"],
                 "second protocol/source가 바뀌었다")
        approval = _approval_for(
            candidate_path, protocol, schema=SECOND.SECOND_APPROVAL_SCHEMA,
            anchor=plan["anchor"], seed=row["registered_seed"], lease=lease,
            approvals=approvals, key=key, approver=approver, root=out)
        _approval(approval, schema=SECOND.SECOND_APPROVAL_SCHEMA, candidate=candidate_path,
                  protocol=protocol, anchor=plan["anchor"], seed=row["registered_seed"],
                  lease_id=lease["leaseId"])
        execution_dir = out / "recovery" / key.replace(":", "_")
        result = candidate_executor(
            candidate_path, protocol, approval, anchor=plan["anchor"], lab=lab,
            isaac_python=isaac_python, registered_seed=row["registered_seed"], out=execution_dir,
            runner=runner, wait=wait, lease_reader=lease_reader)
        _same_lease(lease, lease_reader)
        executed += 1
        records.append({**row, "outcome": result["outcome"], "diagnosis": _ref(diagnosis_path),
                        "candidate": _ref(candidate_path),
                        "result": _ref(execution_dir / "result.json")})
    _require(probe_pairs <= 12 and probe_pairs * 2 <= 24 and executed <= 4,
             "second SDK upper bound exceeded")
    transport = TRANSPORT.status(transport_root)
    _require(transport["reservations_charged"] <= 32, "R4 API cap exceeded")
    result = {"schema": SECOND_RESULT_SCHEMA, "status": "complete", "plan": _ref(plan_path),
              "lease_id": lease["leaseId"], "records": records,
              "probe_pairs": probe_pairs, "probe_sdk_upper_bound": probe_pairs * 2,
              "executed_second_targets": executed, "training_fixed_sdk_upper_bound": executed * 2,
              "total_sdk_upper_bound": probe_pairs * 2 + executed * 2,
              "transport_status": transport, "automatic_retry": False,
              "epoch3_allowed": False, "parent_wall_s": time.monotonic() - started}
    _write_new(out / "result.json", result)
    return deepcopy(result)


def _operational_probe_factory(*, protocol: dict[str, Any], protocol_path: Path, anchor: str,
                               lab: Path, output: Path, lease: Mapping[str, Any],
                               lease_reader: Callable[[], Mapping[str, Any]], started: float):
    state = {"pairs": 0}
    protocol_sha = FIRST.sha(protocol_path)

    def factory(source: dict[str, Any]) -> Callable:
        source = SECOND.checked_source(source)
        def context(_request: dict | None = None) -> dict[str, bool]:
            _require(time.monotonic() - started < SECOND_WALL_S, "second wall 상한")
            _same_lease(lease, lease_reader)
            _require(FIRST.sha(protocol_path) == protocol_sha, "second protocol이 바뀌었다")
            _verify_sources(protocol)
            SECOND.checked_source(source)
            return {"approved": True}

        def dispatch(snapshot: dict, choice: dict, *, out: Path) -> dict[str, Any]:
            context()
            _require(state["pairs"] < 12 and state["pairs"] * 2 + 2 <= 24,
                     "paired probe SDK 예약 상한")
            state["pairs"] += 1
            _write_new(output / "paired-intents" / f"{state['pairs']:04d}.json", {
                "case_id": source["case_id"], "policy": source["policy"],
                "source_digest": SECOND.digest(source), "snapshot_digest": SECOND.digest(snapshot),
                "choice_digest": SECOND.digest(choice), "sdk_roles_reserved": 2,
                "automatic_retry": False})
            return SECOND.execute_pair(
                snapshot, choice, backend_packet=source["backend_packet"],
                case_input=_bound(source["inputs"]["case_collected"], "case collected"),
                reference_input=_bound(source["inputs"]["reference_source_v2"], "reference source"),
                probe_protocol=ROOT / protocol["probe_protocol"]["path"],
                view_protocol=ROOT / protocol["view_protocol"]["path"], anchor=anchor,
                lab=lab, out=out, verify_context=context)
        return dispatch
    return factory, state


def _busy_reader(output: Path, lease: Mapping[str, Any],
                 lease_reader: Callable[[], Mapping[str, Any]]) -> Callable:
    def read(operation_id: str, _intent: dict) -> dict[str, str]:
        _same_lease(lease, lease_reader)
        stage = output / "gpu-observations" / operation_id
        try:
            GPU.snapshot(stage)
        except RuntimeError:
            receipt = _read(stage / "gpu_before.json")
            _require(receipt.get("bounded_resample", {}).get("accepted") is False,
                     "GPU busy 예외 원문이 없다")
            return {"status": "busy", "observation_id": FIRST.sha(stage / "gpu_before.json")}
        return {"status": "available", "observation_id": FIRST.sha(stage / "gpu_before.json")}
    return read


def run_first_operational(plan_path: Path, *, lab: Path, isaac_python: Path,
                          out: Path, approver: str) -> dict[str, Any]:
    return run_first(
        plan_path, None, lab=lab, isaac_python=isaac_python, out=out,
        runner=OWNED.run_owned, wait=WAIT._lease_wait, lease_reader=LEASE.lease_context,
        approver=approver)


def run_second_operational(plan_path: Path, *, transport_root: Path, lab: Path,
                           isaac_python: Path, out: Path, approver: str) -> dict[str, Any]:
    plan = _read(plan_path)
    protocol_path = _bound(plan["loop_protocol"], "loop protocol")
    protocol = SECOND.verify_loop_protocol(protocol_path, plan["anchor"], lab=lab)
    lease = _lease(LEASE.lease_context)
    started = time.monotonic()

    def lease_reader() -> Mapping[str, Any]:
        current = LEASE.lease_context()
        _same_lease(lease, lambda: current)
        return current

    factory, probe_state = _operational_probe_factory(
        protocol=protocol, protocol_path=protocol_path, anchor=plan["anchor"], lab=lab,
        output=Path(out).resolve(), lease=lease, lease_reader=lease_reader, started=started)
    result = run_second(
        plan_path, None, transport_root=transport_root, lab=lab, isaac_python=isaac_python,
        out=out, send=lambda body, call: _WORKER._worker_send(
            transport_root, protocol_path, plan["anchor"], body, call),
        execute_probe=lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("operational probe factory required")),
        probe_factory=factory, busy_reader=_busy_reader(Path(out).resolve(), lease, lease_reader),
        runner=OWNED.run_owned, wait=WAIT._lease_wait, lease_reader=lease_reader,
        approver=approver)
    _require(probe_state["pairs"] == result["probe_pairs"],
             "paired probe 예약/실행 수가 다르다")
    return result


def _record_cli_failure(out: Path, command: str, error: BaseException) -> None:
    target = Path(out).resolve()
    target.mkdir(parents=True, exist_ok=True)
    failure = target / "parent-error.json"
    if not failure.exists():
        _write_new(failure, {"schema": "five_policy_recovery_parent_error_v1",
                             "command": command, "error_type": type(error).__name__,
                             "error": str(error), "automatic_retry": False})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare-first")
    for name in ("batch", "five_policy_protocol", "config", "recovery_protocol", "anchor", "out"):
        first.add_argument("--" + name.replace("_", "-"), required=True)
    second = sub.add_parser("prepare-second")
    for name in ("first_result", "loop_protocol", "anchor", "lab", "out"):
        second.add_argument("--" + name.replace("_", "-"), required=True)
    run_first_parser = sub.add_parser("run-first")
    for name in ("plan", "lab", "isaac_python", "out"):
        run_first_parser.add_argument("--" + name.replace("_", "-"), required=True)
    run_first_parser.add_argument("--approver", choices=[APPROVER], required=True)
    run_second_parser = sub.add_parser("run-second")
    for name in ("plan", "transport_root", "lab", "isaac_python", "out"):
        run_second_parser.add_argument("--" + name.replace("_", "-"), required=True)
    run_second_parser.add_argument("--approver", choices=[APPROVER], required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-first":
            prepare_first(Path(args.batch), five_policy_protocol=Path(args.five_policy_protocol),
                          config=Path(args.config), recovery_protocol=Path(args.recovery_protocol),
                          anchor=args.anchor, out=Path(args.out))
        elif args.command == "prepare-second":
            prepare_second(Path(args.first_result), loop_protocol=Path(args.loop_protocol),
                           anchor=args.anchor, lab=Path(args.lab), out=Path(args.out))
        elif args.command == "run-first":
            run_first_operational(Path(args.plan), lab=Path(args.lab),
                                  isaac_python=Path(args.isaac_python), out=Path(args.out),
                                  approver=args.approver)
        else:
            run_second_operational(Path(args.plan), transport_root=Path(args.transport_root),
                                   lab=Path(args.lab), isaac_python=Path(args.isaac_python),
                                   out=Path(args.out), approver=args.approver)
    except Exception as error:
        if hasattr(args, "out"):
            _record_cli_failure(Path(args.out), args.command, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
