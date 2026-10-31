"""유효한 첫 미회복 receipt에서 두 번째 진단·교정만 재개한다."""
# ruff: noqa: E402
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import immutable_git_proof_cache_v1 as CACHE
import lineage_nim_transport_v4 as TRANSPORT
import runtime_observation_diagnosis_v1 as DIAG
import runtime_observation_probe_v1 as PROBE
import runtime_observation_recovery_v1 as R
import runtime_observation_training_v1 as TRAIN
import runtime_observation_view_v1 as RV
from rl_triage.correction_ledger import CorrectionLedger

SCHEMA = "runtime_observation_continuation_protocol_v1"
INPUT_SCHEMA = "runtime_observation_continuation_inputs_v1"
RESULT_SCHEMA = "runtime_observation_continuation_result_v1"
MAX_WALL_S = 14400
SOURCE_FILES = tuple(sorted({
    *R.SOURCE_FILES,
    "evals/immutable_git_proof_cache_v1.py", "tests/test_immutable_git_proof_cache_v1.py",
    "evals/runtime_observation_continuation_v1.py",
    "tests/test_runtime_observation_continuation_v1.py",
    "bench/run_runtime_observation_continuation_v1.ps1",
}))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": sha(path)}


def component(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256_lf": sha(path, lf=True)}


def prepare_protocol(origin_protocol: Path, probe_protocol: Path,
                     transport_protocol: Path) -> dict[str, Any]:
    origin = read(origin_protocol)
    require(origin.get("schema") == R.SCHEMA and origin.get("seed") == R.SEED,
            "origin prospective protocol이 아니다")
    transport = read(transport_protocol)
    require(transport.get("schema") == TRANSPORT.SCHEMA,
            "continuation은 transport V4 protocol을 요구한다")
    slots = transport.get("plan", {}).get("logical_slots")
    require(transport.get("model") == origin["model"] and transport.get("params") == origin["params"],
            "continuation transport model/params가 origin과 다르다")
    require(transport.get("plan", {}).get("case_ids") == ["continuation-s2311"]
            and transport.get("plan", {}).get("global_request_cap") == 9
            and transport.get("plan", {}).get("per_slot_max") == 3
            and slots == [{"case_id": "continuation-s2311", "slot_id": f"epoch2.decision{index}"}
                          for index in (1, 2, 3)],
            "continuation transport plan은 epoch2 3 slot/cap9여야 한다")
    runtime = origin.get("runtime_source_sha256_lf")
    require(isinstance(runtime, dict) and len(runtime) == 9, "origin runtime pin 9개가 없다")
    return {
        "schema": SCHEMA, "purpose": "continue_verified_first_not_recovered_epoch_only",
        "seed": R.SEED, "origin": component(origin_protocol),
        "components": {"probe": component(probe_protocol),
                       "transport": component(transport_protocol),
                       "view": component(RV.PROTOCOL), "fixed": component(R.OLD.FIXED_PROTOCOL)},
        "model": origin["model"], "params": deepcopy(origin["params"]),
        "inherited": {"conditions": deepcopy(origin["conditions"]),
                      "training": deepcopy(origin["training"]),
                      "predictions": deepcopy(origin["predictions"]),
                      "criteria": deepcopy(origin["criteria"])},
        "bounds": {"source_training": 0, "repeat_first_probe": 0, "first_repair_training": 0,
                   "second_repair_training_max": 1, "epoch2_model_slots": 3,
                   "epoch2_model_requests_max": 9, "combined_provider_requests_max": 24,
                   "development_model_requests": 2,
                   "wall_s": MAX_WALL_S, "train_watchdog_s": 1800,
                   "eval_watchdog_s": 600, "gpu_wait_s": 600,
                   "paired_probe_budget_s": 1200},
        "coupling": {"same_seen_seed": True, "new_independent_seed": False,
                     "requires_original_first_receipt": True,
                     "origin_overall_result_complete_required": False,
                     "immutable_git_cache_operational_only": True},
        "outcomes": {"recovered": origin["wording"]["recovered"],
                     "not_recovered": origin["wording"]["not_recovered"],
                     "HOLD": origin["wording"]["HOLD"]},
        "source_sha256_lf": {name: R.D.sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(runtime),
        "external_effects": {"tokenfactory_requests": 0, "brev_start": 0,
                             "local_gpu_only": True, "lease_owner": R.OWNER},
        "transport_runtime_observation": transport.get("schema"),
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = read(path)
    parts = value.get("components")
    require(isinstance(parts, dict), "continuation components가 없다")
    parts = cast(dict[str, dict[str, str]], parts)
    origin_ref = value.get("origin")
    require(isinstance(origin_ref, dict), "continuation origin ref가 없다")
    origin_ref = cast(dict[str, str], origin_ref)
    expected = prepare_protocol(ROOT / origin_ref["path"], ROOT / parts["probe"]["path"],
                                ROOT / parts["transport"]["path"])
    require(value == expected, "continuation protocol 내용 또는 source 결합이 다르다")
    TRAIN._registered_sources(path, value, anchor)
    for item in [origin_ref, *parts.values()]:
        current = ROOT / item["path"]
        require(sha(current, lf=True) == item["sha256_lf"], "component 현재 SHA가 다르다")
        blob = R.OLD._git("show", f"{anchor}:{item['path']}").replace(b"\r\n", b"\n")
        require(hashlib.sha256(blob).hexdigest() == item["sha256_lf"],
                "component anchor blob이 다르다")
    PROBE.verify_protocol(ROOT / parts["probe"]["path"], lab, anchor)
    TRANSPORT.verify_protocol(ROOT / parts["transport"]["path"], anchor)
    return value


def _inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _closed_origin_queue(root: Path, expected_requests: int) -> dict[str, Any]:
    root = Path(root).resolve()
    status = TRANSPORT.status(root)
    require(isinstance(status.get("closed"), dict), "origin NIM queue가 닫히지 않았다")
    require(status.get("requests") == status.get("dispatches") == status.get("responses")
            == expected_requests and status.get("pending") == status.get("unknown")
            == status.get("failures") == 0 and 0 <= expected_requests <= 9,
            "origin NIM queue는 완료된 epoch1 request/dispatch/response만 가져야 한다")
    slots = []
    for path in sorted((root / "requests").glob("*.json")):
        request = read(path)
        slots.append(request.get("slot_id"))
    require(all(slot in {f"epoch1.decision{index}" for index in (1, 2, 3)} for slot in slots),
            "origin queue에 epoch1 밖 intent가 있다")
    return {"root": str(root), "status": status, "inventory": _inventory(root),
            "epoch2": {"requests": 0, "dispatches": 0, "responses": 0}}


def _origin_phase(root: Path) -> dict[str, Any]:
    candidates = [root / "failure.json", root / "result.json"]
    present = [path for path in candidates if path.is_file()]
    require(len(present) == 1, "origin phase result/failure는 정확히 하나여야 한다")
    value = read(present[0])
    wall = value.get("wall_s")
    if type(wall) not in (int, float):
        raise ValueError("origin phase wall_s가 유효하지 않다")
    wall_value = float(cast(int | float, wall))
    require(math.isfinite(wall_value) and 0 <= wall_value <= R.MAX_TOTAL_S,
            "origin phase wall_s가 유효하지 않다")
    return {"artifact": ref(present[0]), "status": value.get("status"),
            "wall_s": wall_value, "mtime_ns": present[0].stat().st_mtime_ns}


def _check_origin_training(training: dict[str, Any], *, arm: str, faults: list[str],
                           corrections: list[str]) -> None:
    _collected, metadata = R._training_checked(training)
    private = metadata.get("private_training")
    require(isinstance(private, dict) and private.get("seed") == R.SEED
            and private.get("arm") == arm and private.get("original_faults") == faults
            and private.get("applied_corrections") == corrections,
            f"origin {arm} 학습 seed/arm/fault/correction 결합이 다르다")


def _origin(origin_root: Path, origin_queue: Path, origin_protocol: Path,
            origin_anchor: str, lab: Path) -> dict[str, Any]:
    origin_root = Path(origin_root).resolve()
    R.verify_protocol(origin_protocol, origin_anchor, lab)
    receipt_path = origin_root / "repair1/receipt.json"
    receipt = read(receipt_path)
    ledger_path = origin_root / "corrections.ledger.jsonl"
    reference_path = origin_root / "none/training-result.json"
    compound_path = origin_root / "compound/training-result.json"
    first_path = origin_root / "repair1/training-result.json"
    reference, compound, first = read(reference_path), read(compound_path), read(first_path)
    _check_origin_training(reference, arm="NONE", faults=[], corrections=[])
    _check_origin_training(compound, arm="COMPOUND", faults=["F1", "R1"], corrections=[])
    require(receipt.get("schema") == R.RECEIPT_SCHEMA and receipt.get("epoch") == 1
            and receipt.get("parent") is None and receipt.get("recovery", {}).get("outcome") == "not_recovered"
            and receipt.get("input_training") == compound and receipt.get("output_training") == first,
            "origin 첫 repair는 유효한 미회복 epoch1 receipt여야 한다")
    with CACHE.proof_scope(), RV.view_scope(RV.PROTOCOL, origin_anchor):
        R.verify_repair(receipt_path, ledger_path, reference, read(R.OLD.BANDS)["bands"])
    ledger = CorrectionLedger(ledger_path, R.C.RC.correction_registry())
    events = ledger.events()
    require(len(events) == 4 and [row["event"] for row in events]
            == ["proposed", "approved", "consumed", "receipt"],
            "origin first correction ledger는 완료 4행이어야 한다")
    diagnosis_path = Path(receipt["diagnosis"]["path"])
    diagnosis = read(diagnosis_path)
    require(type(diagnosis.get("requests")) is int and 0 <= diagnosis["requests"] <= 13,
            "origin+development+continuation provider 요청 상한이 24를 넘는다")
    first_correction = receipt["applied_after"][-1]
    _check_origin_training(first, arm="FIRST", faults=["F1", "R1"],
                           corrections=[first_correction])
    reference_report_path = origin_root / "none/fixed/corrected.json"
    compound_report_path = origin_root / "compound/fixed/corrected.json"
    reference_report = R._check_report(reference_report_path, reference)
    compound_report = R._check_report(compound_report_path, compound)
    bands = read(R.OLD.BANDS)["bands"]
    normal = R.C.D1.recovery(reference_report, reference_report, bands)
    source = R.C.D1.recovery(compound_report, reference_report, bands)
    require(R.C.D1.V.R1.U.walks(reference_report)
            and normal["verdict"]["label"] == "healthy",
            "origin NONE 정상 대조가 healthy/walks가 아니다")
    require(source["verdict"]["label"] == "unhealthy",
            "origin COMPOUND source failure가 재계산되지 않는다")
    return {"receipt": ref(receipt_path), "ledger": ref(ledger_path),
            "reference": ref(reference_path), "compound": ref(compound_path), "first": ref(first_path),
            "reference_report": ref(reference_report_path),
            "compound_report": ref(compound_report_path),
            "diagnosis": ref(diagnosis_path), "diagnosis_inventory": _inventory(diagnosis_path.parent),
            "queue": _closed_origin_queue(origin_queue, diagnosis["requests"]),
            "phase": _origin_phase(origin_root),
            "prior": {"verdict": "not_recovered", "correction_id": receipt["applied_after"][-1],
                      "result_sha256": sha(receipt_path),
                      "checkpoint_sha256": first["checkpoint"]["sha256"]},
            "origin_requests": diagnosis.get("requests", 0)}


def prepare_inputs(origin_root: Path, origin_queue: Path, origin_protocol: Path, continuation_protocol: Path,
                   origin_anchor: str, continuation_anchor: str, lab: Path, out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    require(not out.exists(), "continuation input out은 fresh 경로여야 한다")
    verify_protocol(continuation_protocol, continuation_anchor, lab)
    origin = _origin(origin_root, origin_queue, origin_protocol, origin_anchor, lab)
    prepared_ns = time.time_ns()
    gap_s = max(0.0, (prepared_ns - origin["phase"]["mtime_ns"]) / 1_000_000_000)
    require(origin["phase"]["wall_s"] + gap_s <= R.MAX_TOTAL_S,
            "origin 실행과 continuation 준비 간격이 전체 wall 상한을 넘는다")
    value = {"schema": INPUT_SCHEMA, "origin_root": str(Path(origin_root).resolve()),
             "origin_protocol": ref(origin_protocol), "continuation_protocol": ref(continuation_protocol),
             "origin_anchor": origin_anchor, "continuation_anchor": continuation_anchor,
             "lab": str(Path(lab).resolve()), "origin": origin,
             "budget": {"origin_wall_s": origin["phase"]["wall_s"],
                        "prepare_gap_s": gap_s, "prepared_time_ns": prepared_ns,
                        "continuation_wall_s_max": MAX_WALL_S,
                        "combined_wall_s_max": R.MAX_TOTAL_S},
             "scope": {"source_training": 0, "first_repair_training": 0,
                       "approval": False, "execution_allowed": False}}
    out.mkdir(parents=True)
    R.D.write(out / "inputs.json", value)
    return value


def verify_origin(inputs_path: Path) -> dict[str, Any]:
    stored = read(inputs_path)
    require(stored.get("schema") == INPUT_SCHEMA, "continuation input schema가 다르다")
    verify_protocol(Path(stored["continuation_protocol"]["path"]),
                    stored["continuation_anchor"], Path(stored["lab"]))
    current = _origin(Path(stored["origin_root"]), Path(stored["origin"]["queue"]["root"]),
                      Path(stored["origin_protocol"]["path"]),
                      stored["origin_anchor"], Path(stored["lab"]))
    require(current == stored["origin"], "origin first receipt 또는 raw 진단 원문이 바뀌었다")
    return stored


def _execute_scoped(inputs_path: Path, protocol_path: Path, anchor: str, queue: Path,
                    out: Path, *, lab: Path, isaac_python: Path, runner: Callable,
                    wait: Callable, diagnose: Callable) -> dict[str, Any]:
    inputs = verify_origin(inputs_path)
    protocol = verify_protocol(protocol_path, anchor, lab)
    require(str(Path(protocol_path).resolve()) == inputs["continuation_protocol"]["path"],
            "execute protocol이 prepared input과 다르다")
    queue_status = TRANSPORT.status(queue)
    require(queue_status.get("closed") is None
            and all(queue_status.get(key) == 0 for key in (
                "requests", "dispatches", "responses", "pending", "unknown", "failures")),
            "continuation NIM queue는 열린 fresh 0/0/0 상태여야 한다")
    out = Path(out).resolve()
    require(not out.exists(), "continuation output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    lease = R.FIRST.LEASE.lease_context(out / "lease-initial")
    require(lease.get("owner") == R.OWNER, "공용 GPU lease 소유자가 다르다")
    ledger_path = out / "corrections.ledger.jsonl"
    shutil.copyfile(Path(inputs["origin"]["ledger"]["path"]), ledger_path)
    require(sha(ledger_path) == inputs["origin"]["ledger"]["sha256"], "origin ledger 사본 불일치")
    first = read(Path(inputs["origin"]["first"]["path"]))
    reference = read(Path(inputs["origin"]["reference"]["path"]))
    parent_path = Path(inputs["origin"]["receipt"]["path"])
    prior = inputs["origin"]["prior"]
    start = time.monotonic()
    execution_time_ns = time.time_ns()
    execution_gap_s = max(0.0, (execution_time_ns - inputs["budget"]["prepared_time_ns"])
                          / 1_000_000_000)
    stages: list[dict[str, Any]] = []
    repairs: list[dict[str, str]] = []
    summary = {"schema": RESULT_SCHEMA, "status": "running", "outcome": "HOLD",
               "seed": R.SEED, "origin_inputs": ref(inputs_path),
               "origin_status": "valid_first_not_recovered", "continuation_status": "running",
               "source_training": 0, "first_repair_training": 0, "second_repair_training": 0,
               "reason": None, "diagnosis": None, "stages": stages, "repairs": repairs,
               "wall": {"origin_wall_s": inputs["budget"]["origin_wall_s"],
                        "prepare_gap_s": inputs["budget"]["prepare_gap_s"],
                        "execution_gap_s": execution_gap_s, "phase_wall_s": 0.0}}

    def budget(reserve: float = 0.0) -> None:
        elapsed = time.monotonic() - start
        require(elapsed + reserve <= MAX_WALL_S, "continuation phase wall 예약 상한 초과")
        require(inputs["budget"]["origin_wall_s"] + inputs["budget"]["prepare_gap_s"]
                + execution_gap_s + elapsed + reserve
                <= inputs["budget"]["combined_wall_s_max"], "누적 전체 wall 예약 상한 초과")

    def context() -> None:
        budget()
        verify_origin(inputs_path)
        with CACHE.proof_scope(), RV.view_scope(RV.PROTOCOL, inputs["origin_anchor"]):
            R.verify_repair(parent_path, ledger_path, reference, read(R.OLD.BANDS)["bands"])

    try:
        require(inputs["budget"]["origin_wall_s"] + inputs["budget"]["prepare_gap_s"]
                + execution_gap_s <= inputs["budget"]["combined_wall_s_max"],
                "prepare 이후 실행 대기까지 누적 wall 상한을 넘는다")
        diagnosis = diagnose(
            out / "diagnosis-e2", Path(first["collected"]["path"]),
            Path(reference["collected"]["path"]),
            case_runtime_path=Path(first["runtime_facts"]["case"]["path"]),
            reference_runtime_path=Path(reference["runtime_facts"]["reference"]["path"]),
            run_id="runtime_observation_s2311_v1", view_protocol=RV.PROTOCOL,
            probe_protocol=ROOT / protocol["components"]["probe"]["path"], anchor=anchor,
            lab=lab, transport_root=queue,
            slot_ids=tuple(f"epoch2.decision{i}" for i in (1, 2, 3)), verify_context=context,
            applied_corrections=(prior["correction_id"],), prior_receipt=prior,
            parent={"run_id": "runtime_observation_s2311_v1", "policy": "model_adaptive",
                    "epoch": 1, **prior})
        summary["diagnosis"] = ref(out / "diagnosis-e2/result.json")
        if diagnosis["end_state"] != "confirmed":
            summary.update(status="complete", continuation_status="HOLD",
                           reason=diagnosis.get("hold_reason") or "diagnosis_unconfirmed")
        else:
            context()
            registry = R.C.RC.correction_registry()
            component_id = diagnosis.get("confirmed_subtype")
            if not isinstance(component_id, str) or component_id not in registry:
                summary.update(status="complete", continuation_status="HOLD",
                               reason="repair_unsupported")
            elif registry[component_id]["correction_id"] == prior["correction_id"]:
                summary.update(status="complete", continuation_status="HOLD",
                               reason="repeated_correction")
            else:
                component_id, correction = R.verify_diagnosis(diagnosis, [prior["correction_id"]])
                snapshot = R.load_verified_snapshot(diagnosis)
                entry = registry[component_id]
                budget(3600)
                ledger = CorrectionLedger(ledger_path, registry)
                request = ledger.propose(R.C.L.digest(snapshot), component_id, R.C.L.digest(diagnosis["steps"]),
                    correction, "restart", {"case": entry["protocol_case"], "arm": "corrected"}, 3600,
                    evidence={"actual_diagnosis": ref(out / "diagnosis-e2/result.json"),
                              "selection_source": "registered_host_rule"})
                ledger.approve(request["request_id"], "codex-main-user-delegated")
                repair_start = time.monotonic()
                ledger.consume(request["request_id"], R.C.L.digest(snapshot))
                fixed = TRAIN.build_manifest(protocol_path, protocol, anchor, stage=out / "repair2/train",
                    seed=R.SEED, arm="BOTH", original_faults=["F1", "R1"],
                    applied_corrections=[prior["correction_id"], correction],
                    source_checkpoint_sha256=first["checkpoint"]["sha256"],
                    run_name="ro_cont_repair2_s2311")
                output = TRAIN.run_training(out / "repair2/train", fixed, lab=lab,
                                            isaac_python=isaac_python, runner=runner, wait=wait)
                summary["second_repair_training"] = 1
                training_result_path = out / "repair2/training-result.json"
                R.D.write(training_result_path, output)
                stages.append({"stage": "repair2", "training": ref(training_result_path)})
                evidence = read(Path(output["evidence"]["path"]))
                report, _execution = TRAIN.run_fixed(out / "repair2/fixed", evidence,
                    name="ro_cont_repair2_s2311", lab=lab, isaac_python=isaac_python,
                    runner=runner, wait=wait)
                report_path = out / "repair2/fixed/corrected.json"
                reference_receipt = read(parent_path)["reference_report"]
                reference_report = read(Path(reference_receipt["path"]))
                measured = R.C.D1.recovery(report, reference_report, read(R.OLD.BANDS)["bands"])
                receipt_path = out / "repair2/receipt.json"
                receipt = {"schema": R.RECEIPT_SCHEMA, "epoch": 2,
                    "request_id": request["request_id"], "applied_before": [prior["correction_id"]],
                    "applied_after": [prior["correction_id"], correction], "input_training": first,
                    "output_training": output, "corrected_report": ref(report_path),
                    "reference_report": reference_receipt, "recovery": measured,
                    "parent": ref(parent_path), "diagnosis": ref(out / "diagnosis-e2/result.json"),
                    "continuation_purpose": protocol["purpose"]}
                R.D.write(receipt_path, receipt)
                ledger.receipt(request["request_id"], "runtime_cont_s2311_e2", measured["outcome"],
                               sha(receipt_path), time.monotonic() - repair_start, 0)
                R.verify_repair(receipt_path, ledger_path, reference, read(R.OLD.BANDS)["bands"])
                repairs.append(ref(receipt_path))
                summary.update(status="complete", continuation_status="complete",
                               outcome=measured["outcome"], receipt=ref(receipt_path))
        summary["wall"]["phase_wall_s"] = time.monotonic() - start
        summary["wall_s"] = summary["wall"]["phase_wall_s"]
        R.D.write(out / "result.json", summary)
        return summary
    except BaseException as error:
        R.D.write(out / "failure.json", {**summary, "status": "invalid",
                  "error_type": type(error).__name__, "error": str(error),
                  "wall_s": time.monotonic() - start})
        raise
    finally:
        TRANSPORT.close(queue, "runtime_observation_continuation_finished")


def execute(inputs_path: Path, protocol_path: Path, anchor: str, queue: Path, out: Path, *,
            lab: Path, isaac_python: Path, runner: Callable, wait: Callable,
            diagnose: Callable = DIAG.run_epoch) -> dict[str, Any]:
    with CACHE.proof_scope(), RV.view_scope(RV.PROTOCOL, anchor):
        return _execute_scoped(inputs_path, protocol_path, anchor, queue, out, lab=lab,
                               isaac_python=isaac_python, runner=runner, wait=wait,
                               diagnose=diagnose)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-inputs")
    for name in ("origin-root", "origin-queue", "origin-protocol", "continuation-protocol", "lab", "out"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--origin-anchor", required=True)
    prepare.add_argument("--continuation-anchor", required=True)
    run = sub.add_parser("run")
    for name in ("inputs", "protocol", "queue", "out", "lab", "isaac-python"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--anchor", required=True)
    args = parser.parse_args()
    if args.command == "prepare-inputs":
        prepare_inputs(args.origin_root, args.origin_queue, args.origin_protocol, args.continuation_protocol,
                       args.origin_anchor, args.continuation_anchor, args.lab, args.out)
    else:
        import run_runtime_readback_smoke as OWNED
        execute(args.inputs, args.protocol, args.anchor, args.queue, args.out, lab=args.lab,
                isaac_python=args.isaac_python, runner=OWNED.run_owned, wait=R.FIRST._lease_wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
