"""갱신 관측을 사용하는 새 seed의 제한된 두 교정 고리."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import core_seed_guard_v1 as SEEDS
import lineage_compound_first_recovery_v1 as FIRST
import lineage_nim_transport_v4 as T
import lineage_recovery_execution_v1 as OLD
import runtime_observation_development_v1 as D
import runtime_observation_training_v1 as TRAIN
import runtime_observation_view_v1 as RV
import runtime_training_inputs_v1 as INPUT
import selection_comparison_controller_v1 as C

from rl_triage.correction_ledger import CorrectionLedger

SCHEMA = "runtime_observation_recovery_protocol_v1"
RECEIPT_SCHEMA = "runtime_observation_repair_receipt_v1"
SEED = 2311
SMOKE_SEED = 2398
OWNER = "rl-triage-lineage-sources-v1"
MAX_TOTAL_S = 28800
SOURCE_FILES = tuple(sorted({
    *OLD.SOURCE_FILES, *RV.SOURCE_FILES, *T.T3.T2.T1.SOURCE_FILES,
    *T.T3.T2.SOURCE_FILES, *T.T3.SOURCE_FILES, *T.SOURCE_FILES,
    "evals/runtime_observation_recovery_v1.py", "tests/test_runtime_observation_recovery_v1.py",
    "evals/runtime_observation_training_v1.py", "tests/test_runtime_observation_training_v1.py",
    "evals/runtime_training_inputs_v1.py", "tests/test_runtime_training_inputs_v1.py",
    "evals/runtime_observation_diagnosis_v1.py", "tests/test_runtime_observation_diagnosis_v1.py",
    "evals/runtime_observation_probe_v1.py", "tests/test_runtime_observation_probe_v1.py",
    "evals/runtime_observation_development_v1.py", "tests/test_runtime_observation_development_v1.py",
    "evals/core_seed_guard_v1.py", "tests/test_core_seed_guard_v1.py",
    "evals/preregistered_seed_guard.py", "evals/lineage_lease_context_v2.py",
    "evals/lineage_compound_first_recovery_v1.py", "evals/lineage_gpu_wait_v1.py",
    "evals/lineage_gpu_admission_v1.py", "evals/lineage_recovery_execution_v2.py",
    "bench/run_runtime_observation_v1.ps1",
}))


def require(ok: bool, message: str) -> None:
    D.require(ok, message)


def component(path: Path) -> dict:
    path = path.resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256_lf": D.sha(path, lf=True)}


def prepare_protocol(probe_protocol: Path, transport_protocol: Path,
                     development_protocol: Path) -> dict:
    old_probe = D.read(probe_protocol)
    runtime = old_probe.get("runtime_source_sha256_lf")
    base = old_probe
    if runtime is None:
        base = D.read(ROOT / old_probe["component_probe_protocol"]["path"])
        runtime = base.get("runtime_source_sha256_lf")
    require(isinstance(runtime, dict) and len(runtime) == 9, "검사 protocol의 runtime 9개 결합이 필요하다")
    return {
        "schema": SCHEMA, "purpose": "prospective_same_family_runtime_observation_recovery",
        "seed": SEED, "smoke_seed": SMOKE_SEED,
        "conditions": ["NONE", "COMPOUND", "FIRST_if_confirmed", "BOTH_if_valid_not_recovered"],
        "same_seed_pairing": True, "original_faults": ["F1", "R1"],
        "training": {"num_envs": 4096, "updates": 300, "source_checkpoint_loaded": False,
                     "recovery_mode": "fresh_restart", "configured_epochs": 5, "minibatches": 4},
        "smoke": {"updates": 3, "arm": "NONE", "source_checkpoint_role": "provenance_only"},
        "model": C.CALL.MODEL, "params": C.CALL.F.request_params(C.CALL.MODEL),
        "bounds": {"epochs": 2, "different_corrections": 2, "probes_per_epoch": 3,
                   "model_requests": 18, "requests_per_choice": 3, "paired_probe_budget_s": 1200,
                   "training_jobs_max": 4, "fixed_evaluations_max": 4,
                   "smoke_jobs": 1, "total_wall_s": MAX_TOTAL_S,
                   "train_watchdog_s": 1800, "eval_watchdog_s": 600, "gpu_wait_s": 600},
        "predictions": {
            "NONE": "healthy_fixed26_host_gate_no_model_no_probe_no_correction",
            "COMPOUND": "source_failure_expected",
            "diagnosis": "runtime_counts_expected_to_prioritize_optimizer_probe_in_at_least_one_epoch",
            "first_correction": "not_recovered_expected_with_one_remaining_fault",
            "two_corrections": "recovered_expected_but_not_required_for_valid_measurement",
        },
        "criteria": {
            "confirmed": "existing_registered_single_subtype_and_actual_abnormal_confirmation_receipt",
            "recovery": "unchanged_fixed26_and_original_bands_against_same_seed_NONE",
            "epoch2": "only_valid_not_recovered_receipt_with_current_checkpoint_and_consumed_correction_ledger",
            "HOLD": "invalid_model_or_unknown_or_unconfirmed_or_unsupported_or_exhausted_budget",
            "failure_absent": "preserve_and_stop_no_replacement_seed",
            "semantic_retry": False,
        },
        "wording": {
            "recovered": "등록된 새 seed의 복합 결함에서 계측을 포함한 진단·확인 검사·누적 교정·fresh 학습 뒤 고정26 회복 기준을 통과했다.",
            "not_recovered": "등록된 교정 뒤에도 새 seed의 고정26 회복 기준을 충족하지 못했다.",
            "HOLD": "등록된 경계에서 판단 또는 실행을 보류했으며 회복으로 계산하지 않는다.",
            "scope": "이미 본 F1/R1 계열의 한 새 seed 확인이다. 미등록 결함·모델 버전 우위·영상 기여·일반화를 주장하지 않는다.",
        },
        "components": {"probe": component(probe_protocol), "transport": component(transport_protocol),
                       "view": component(RV.PROTOCOL), "development": component(development_protocol),
                       "fixed": component(OLD.FIXED_PROTOCOL)},
        "bands": {"path": OLD.BANDS.relative_to(ROOT).as_posix(), "sha256": D.sha(OLD.BANDS)},
        "source_sha256_lf": {name: D.sha(ROOT / name, lf=True) for name in sorted({
            *SOURCE_FILES, *old_probe["source_sha256_lf"], *base["source_sha256_lf"]})},
        "runtime_source_sha256_lf": runtime,
        "coupling": {"design_after_seen_seed2307_failure": True, "registered_fault_families": True,
                     "freshness_scope": "core_records_and_local_IsaacLab_logs_excluding_stopped_provider"},
        "external_effects": {"tokenfactory_requests": 0, "brev_start": 0, "local_gpu_only": True,
                             "lease_owner": OWNER, "commits_during_gpu": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict:
    import runtime_observation_probe_v1 as PROBE

    path = path.resolve()
    value = D.read(path)
    parts = value["components"]
    require(value == prepare_protocol(ROOT / parts["probe"]["path"],
        ROOT / parts["transport"]["path"], ROOT / parts["development"]["path"]),
        "prospective protocol 또는 source 결합이 다르다")
    TRAIN._registered_sources(path, value, anchor)
    for item in parts.values():
        file = ROOT / item["path"]
        require(D.sha(file, lf=True) == item["sha256_lf"], "component 현재 SHA가 다르다")
        blob = OLD._git("show", f"{anchor}:{item['path']}").replace(b"\r\n", b"\n")
        require(hashlib.sha256(blob).hexdigest() == item["sha256_lf"], "component anchor blob이 다르다")
    require(D.sha(OLD.BANDS) == value["bands"]["sha256"], "bands 원문 변경")
    RV.verify_protocol(RV.PROTOCOL, anchor)
    T.verify_protocol(ROOT / parts["transport"]["path"], anchor)
    PROBE.verify_protocol(ROOT / parts["probe"]["path"], lab, anchor)
    return value


def _check_report(report_path: Path, training: dict) -> dict:
    report = D.read(report_path)
    cp = training["checkpoint"]
    return C.G.validate_fixed_report(report, {"name": report["name"],
        "checkpoint_file": Path(cp["path"]).name, "checkpoint_sha256": cp["sha256"],
        "action_scale": 0.25, "protocol_sha256_lf": D.sha(OLD.FIXED_PROTOCOL, lf=True)})


def _training_checked(training: dict) -> tuple[dict, dict]:
    for key in ("manifest", "evidence", "owned_result", "collected"):
        item = training[key]
        require(D.sha(Path(item["path"])) == item["sha256"], "학습 산출물 원문 SHA 불일치")
    value, metadata = INPUT.checked(Path(training["collected"]["path"]))
    require(value["source_binding"]["checkpoint_sha256"] == training["checkpoint"]["sha256"],
            "학습 collected와 checkpoint 불일치")
    return value, metadata


def load_verified_snapshot(result: dict) -> dict:
    item = result.get("final_snapshot")
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("진단 final snapshot 원문 참조가 없다")
    path = Path(item["path"])
    require(path.is_file() and D.sha(path) == item["sha256"], "final snapshot 원문 SHA 불일치")
    return C.V.checked(D.read(path))


def verify_diagnosis(result: dict, applied: list[str]) -> tuple[str, str]:
    import runtime_observation_probe_v1 as PROBE

    snapshot = load_verified_snapshot(result)
    component_id, correction = C.registered_correction(snapshot, applied)
    expected = "P_reward" if component_id == "reward" else C.RC.NEW_PROBE
    ledger = C.S.M.RegistryLedger(Path(snapshot["inputs"]["ledger_path"]), C.RC.registry())
    requests = ledger.requests()
    confirmations = []
    for step in result["steps"]:
        if not step.get("request_id"):
            continue
        roles = {}
        for role in ("reference", "case"):
            item = step["receipts"][role]
            path = Path(item["path"])
            require(D.sha(path) == item["sha256"], "probe receipt 원문 SHA 불일치")
            roles[role] = PROBE.verify_receipt(Path(step["plans"][role]["path"]), path)
        require(all(item.get("status") == "complete" for item in roles.values()),
                "unknown 검사로 교정할 수 없다")
        outcome = C.classify_raw(step["probe"], roles["case"]["measurement"], roles["reference"]["measurement"])
        request = requests.get(step["request_id"], {})
        receipt = request.get("receipt", {})
        require(request.get("state") == "done" and receipt.get("outcome") == outcome
                and receipt.get("measurement") == roles["case"]["measurement"]
                and receipt.get("over_budget") is False and receipt.get("exit_code") == 0,
                "probe 원장과 raw 측정 재계산 불일치")
        confirmations.append(step["probe"] == expected and outcome == "abnormal")
    require(any(confirmations), "등록 교정의 abnormal 확인 검사 원문이 없다")
    return component_id, correction


def verify_repair(path: Path, ledger_path: Path, reference: dict, bands: dict) -> dict:
    value = D.read(path)
    require(value.get("schema") == RECEIPT_SCHEMA, "새 repair receipt schema가 다르다")
    before, after = value["applied_before"], value["applied_after"]
    require(len(after) == value["epoch"] in (1, 2) and after[:-1] == before
            and len(set(after)) == len(after), "누적 교정 epoch/중복 불일치")
    input_training, output = value["input_training"], value["output_training"]
    _in, input_metadata = _training_checked(input_training)
    _out, metadata = _training_checked(output)
    training = metadata["private_training"]
    require(input_metadata["private_training"]["seed"] == SEED
            and input_metadata["private_training"]["applied_corrections"] == before,
            "교정 입력 학습의 seed/누적 교정 불일치")
    require(training["seed"] == SEED and training["applied_corrections"] == after
            and training["original_faults"] == ["F1", "R1"], "출력 학습 교정 mask/seed 불일치")
    fixed = D.read(Path(output["manifest"]["path"]))
    require(fixed["source_checkpoint_sha256"] == input_training["checkpoint"]["sha256"],
            "교정 입력 checkpoint provenance 불일치")
    if value["epoch"] == 2:
        parent_ref = value["parent"]
        parent_path = Path(parent_ref["path"])
        require(D.sha(parent_path) == parent_ref["sha256"], "부모 receipt SHA 불일치")
        parent = verify_repair(parent_path, ledger_path, reference, bands)
        require(parent["recovery"]["outcome"] == "not_recovered"
                and parent["applied_after"] == before
                and parent["output_training"]["checkpoint"]["sha256"] == input_training["checkpoint"]["sha256"],
                "두 번째 epoch의 유효 미회복 부모가 아니다")
    else:
        require(value["parent"] is None and before == [], "첫 교정 부모 불일치")
    for item in (value["corrected_report"], value["reference_report"]):
        require(D.sha(Path(item["path"])) == item["sha256"], "평가 원문 SHA 불일치")
    measured = C.D1.recovery(_check_report(Path(value["corrected_report"]["path"]), output),
                            _check_report(Path(value["reference_report"]["path"]), reference), bands)
    require(measured == value["recovery"], "교정 회복 재계산 불일치")
    ledger = CorrectionLedger(ledger_path, C.RC.correction_registry())
    request = ledger.requests().get(value["request_id"], {})
    diagnosis_ref = value["diagnosis"]
    diagnosis_path = Path(diagnosis_ref["path"])
    require(diagnosis_path.is_file() and D.sha(diagnosis_path) == diagnosis_ref["sha256"],
            "교정 근거 진단 원문 SHA 불일치")
    diagnosis = D.read(diagnosis_path)
    _component, registered = verify_diagnosis(diagnosis, before)
    require(registered == after[-1]
            and request.get("loop_state_digest") == C.L.digest(load_verified_snapshot(diagnosis))
            and request.get("evidence") == {"actual_diagnosis": diagnosis_ref,
                "selection_source": "registered_host_rule"}, "교정 승인과 확인 진단 결합 불일치")
    events = [row["event"] for row in ledger.events() if row.get("request_id") == value["request_id"]]
    require(events == ["proposed", "approved", "consumed", "receipt"] and request["state"] == "done"
            and request["correction_id"] == after[-1] and request["recovery_mode"] == "restart"
            and request["receipt"]["result_sha"] == D.sha(path)
            and request["receipt"]["verdict"] == measured["outcome"]
            and request["receipt"]["over_budget"] is False and request["receipt"]["exit_code"] == 0,
            "교정 승인·단회 소비·판정 원장 불일치")
    return value


def run(protocol_path: Path, anchor: str, queue: Path, out: Path, *, lab: Path,
        isaac_python: Path, smoke_checkpoint_sha256: str) -> dict:
    import run_runtime_readback_smoke as OWNED
    import runtime_observation_diagnosis_v1 as DIAG

    protocol = verify_protocol(protocol_path, anchor, lab)
    require(not out.exists(), "종단 output은 fresh 경로여야 한다")
    fresh = SEEDS.assert_fresh_seeds([SEED, SMOKE_SEED], lab)
    out.mkdir(parents=True)
    D.write(out / "freshness-before-execution.json", fresh)
    lease = FIRST.LEASE.lease_context(out / "lease-initial")
    require(lease["owner"] == OWNER, "공용 GPU lease 소유자가 다르다")
    start = time.monotonic()
    stages = []
    repairs = []
    bands = D.read(OLD.BANDS)["bands"]
    summary = {"schema": "runtime_observation_recovery_result_v1", "status": "running",
               "seed": SEED, "outcome": "HOLD", "reason": None,
               "normal_control": None, "stages": stages, "repairs": repairs,
               "claims": {"new_fault_family": False, "model_version_superiority": False,
                          "video_effect": False, "tokenfactory": False}}

    def budget(reserve: float) -> None:
        require(time.monotonic() - start + reserve <= MAX_TOTAL_S, "전체 실행 시간 예약 상한 부족")

    def train(label: str, arm: str, corrections: list[str], source: str | None,
              *, smoke: bool = False) -> dict:
        budget(900 if smoke else 2400)
        stage = out / label / "train"
        fixed = TRAIN.build_manifest(protocol_path, protocol, anchor, stage=stage,
            seed=SMOKE_SEED if smoke else SEED, arm=arm,
            original_faults=[] if arm == "NONE" else ["F1", "R1"],
            applied_corrections=corrections, source_checkpoint_sha256=source,
            run_name=f"ro_{label}_s{SMOKE_SEED if smoke else SEED}",
            mode="inspection" if smoke else "train")
        result = TRAIN.run_training(stage, fixed, lab=lab, isaac_python=isaac_python,
                                    runner=OWNED.run_owned, wait=FIRST._lease_wait)
        D.write(out / label / "training-result.json", result)
        stages.append({"stage": label, "training": D.ref(out / label / "training-result.json")})
        return result

    def evaluate(label: str, training: dict) -> tuple[dict, Path]:
        budget(1200)
        evidence = D.read(Path(training["evidence"]["path"]))
        report, _execution = TRAIN.run_fixed(out / label / "fixed", evidence,
            name=f"ro_{label}_s{SEED}", lab=lab, isaac_python=isaac_python,
            runner=OWNED.run_owned, wait=FIRST._lease_wait)
        path = out / label / "fixed/corrected.json"
        return report, path

    ledger_path = out / "corrections.ledger.jsonl"
    try:
        train("smoke", "NONE", [], smoke_checkpoint_sha256, smoke=True)
        reference = train("none", "NONE", [], None)
        reference_report, reference_report_path = evaluate("none", reference)
        healthy = C.D1.V.R1.U.walks(reference_report)
        normal_recovery = C.D1.recovery(reference_report, reference_report, bands)
        summary["normal_control"] = {"host_gate": True, "model_called": False,
            "probes": 0, "corrections": 0, "recovery": normal_recovery}
        if not healthy or normal_recovery["verdict"]["label"] != "healthy":
            summary["reason"] = "baseline_invalid"
        else:
            current = train("compound", "COMPOUND", [], None)
            source_report, _source_report_path = evaluate("compound", current)
            source_recovery = C.D1.recovery(source_report, reference_report, bands)
            summary["source_recovery"] = source_recovery
            if source_recovery["verdict"]["label"] != "unhealthy":
                summary["reason"] = "source_failure_absent_or_unknown"
            else:
                applied: list[str] = []
                parent_path = None
                prior = None
                run_id = "runtime_observation_s2311_v1"
                with RV.view_scope(RV.PROTOCOL, anchor):
                    for epoch in (1, 2):
                        def context(current_input=current, parent_input=parent_path) -> None:
                            verify_protocol(protocol_path, anchor, lab)
                            _training_checked(current_input)
                            _training_checked(reference)
                            if parent_input is not None:
                                verify_repair(parent_input, ledger_path, reference, bands)

                        budget(1800)
                        parent = None if prior is None else {"run_id": run_id,
                            "policy": "model_adaptive", "epoch": 1, **prior}
                        diagnosis = DIAG.run_epoch(out / f"diagnosis-e{epoch}",
                            Path(current["collected"]["path"]), Path(reference["collected"]["path"]),
                            case_runtime_path=Path(current["runtime_facts"]["case"]["path"]),
                            reference_runtime_path=Path(reference["runtime_facts"]["reference"]["path"]),
                            run_id=run_id, view_protocol=RV.PROTOCOL,
                            probe_protocol=ROOT / protocol["components"]["probe"]["path"],
                            anchor=anchor, lab=lab, transport_root=queue,
                            slot_ids=tuple(f"epoch{epoch}.decision{i}" for i in (1, 2, 3)),
                            verify_context=context, applied_corrections=tuple(applied),
                            prior_receipt=prior, parent=parent)
                        if diagnosis["end_state"] != "confirmed":
                            summary["reason"] = diagnosis.get("hold_reason") or "diagnosis_unconfirmed"
                            break
                        context()
                        component_id, correction = verify_diagnosis(diagnosis, applied)
                        snapshot = load_verified_snapshot(diagnosis)
                        registry = C.RC.correction_registry()
                        entry = registry[component_id]
                        ledger = CorrectionLedger(ledger_path, registry)
                        state_digest = C.L.digest(snapshot)
                        evidence_ref = D.ref(out / f"diagnosis-e{epoch}/result.json")
                        request = ledger.propose(state_digest, component_id, C.L.digest(diagnosis["steps"]),
                            correction, "restart", {"case": entry["protocol_case"], "arm": "corrected"},
                            3600, evidence={"actual_diagnosis": evidence_ref,
                                           "selection_source": "registered_host_rule"})
                        ledger.approve(request["request_id"], "codex-main-user-delegated")
                        ledger.consume(request["request_id"], state_digest)
                        before = list(applied)
                        applied.append(correction)
                        correction_start = time.monotonic()
                        output = train(f"repair{epoch}", "FIRST" if epoch == 1 else "BOTH",
                                       applied, current["checkpoint"]["sha256"])
                        corrected_report, report_path = evaluate(f"repair{epoch}", output)
                        measured = C.D1.recovery(corrected_report, reference_report, bands)
                        receipt_path = out / f"repair{epoch}/receipt.json"
                        receipt = {"schema": RECEIPT_SCHEMA, "epoch": epoch,
                            "request_id": request["request_id"], "applied_before": before,
                            "applied_after": list(applied), "input_training": current,
                            "output_training": output, "corrected_report": D.ref(report_path),
                            "reference_report": D.ref(reference_report_path), "recovery": measured,
                            "parent": None if parent_path is None else D.ref(parent_path),
                            "diagnosis": evidence_ref}
                        D.write(receipt_path, receipt)
                        ledger.receipt(request["request_id"], f"runtime_s{SEED}_e{epoch}", measured["outcome"],
                                       D.sha(receipt_path), time.monotonic() - correction_start, 0)
                        verify_repair(receipt_path, ledger_path, reference, bands)
                        repairs.append(D.ref(receipt_path))
                        summary["outcome"] = measured["outcome"]
                        if measured["outcome"] != "not_recovered" or epoch == 2:
                            break
                        parent_path = receipt_path
                        prior = {"verdict": "not_recovered", "correction_id": correction,
                                 "result_sha256": D.sha(receipt_path),
                                 "checkpoint_sha256": output["checkpoint"]["sha256"]}
                        current = output
                        summary["outcome"] = "HOLD"
        summary["status"] = "complete"
        summary["wall_s"] = time.monotonic() - start
        D.write(out / "result.json", summary)
        return summary
    except BaseException as error:
        D.write(out / "failure.json", {**summary, "status": "invalid", "error_type": type(error).__name__,
                                      "error": str(error), "wall_s": time.monotonic() - start})
        raise
    finally:
        T.close(queue, "prospective_runtime_observation_finished")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("probe-protocol", "transport-protocol", "development-protocol", "out"):
        prepare.add_argument("--" + name, type=Path, required=True)
    execute = sub.add_parser("run")
    for name in ("protocol", "queue", "out", "lab", "isaac-python"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--anchor", required=True)
    execute.add_argument("--smoke-checkpoint-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        D.write(args.out, prepare_protocol(args.probe_protocol, args.transport_protocol, args.development_protocol))
    else:
        run(args.protocol, args.anchor, args.queue, args.out, lab=args.lab,
            isaac_python=args.isaac_python, smoke_checkpoint_sha256=args.smoke_checkpoint_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
