"""residual v2 성공 실행을 CPU로 재검증해 경로 없는 공개 결과로 추출한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_demo_v2 as D  # noqa: E402

RUN = ROOT / "bench/private/codex_support_20261015/task31/runs/residual_v2_20261017"
OUTPUT = ROOT / "evals/results/residual_success_public_20261017"
AUDIT = ROOT / "bench/private/codex_support_20261015/task32/export_receipt.json"
ANCHOR = "499eb28f975fe20dc34cf2add8ce160d1de3078f"
SOURCE_RESULT_SHA = "f856801a76f71e990955adf7c5856d10cc9d68395a63b699f59e7340d865e043"
PROTOCOL_SHA_LF = "d2012679fc29be9a0cb687940a4577fc33212b655d4d89fe45db3489f9983ef4"
FORBIDDEN = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support", re.I)

# build 명령으로 고정 원본을 재생한 뒤 계산한 canonical payload SHA다. 공개 payload와
# provenance의 자체 manifest를 함께 바꿔도 verify를 통과하지 못하도록 코드에 결합한다.
PUBLIC_SHA256 = {
    "evaluations/s2120_none.json": "73d356d9329627579ebcaf55d70facb871b1d576ebd9095c7e5db615ca71a8e8",
    "evaluations/s2120_compound.json": "71f4b661181fd0fac93d07fddd05d7056de38b918c9b782ed1e62dd3c52dabd3",
    "evaluations/s2120_first.json": "d305a7ed5677b65ea8ec36aee990a42a8bf75f1d23b68611519aa92c43ee6d6e",
    "evaluations/s2120_both.json": "9ead9944541b68312a9fac0f7eaa1f24d50b3bdc4e91ff75cd650fbcb353b8ca",
    "evaluations/s2121_none.json": "fcb34e5b78fe087dcffd876c553cd1d88e80526141269e04793d6b94553ab4cf",
    "evaluations/s2121_compound.json": "a3ddb57ddeaf06b7a9aa888bfb54ae3cabd96866550817ce820e0ddf38856aca",
    "evaluations/s2121_first.json": "dd766d7612243616a70bcab15644549f3196c49486c6d671f1410f503879220b",
    "evaluations/s2121_both.json": "eaf6210953a271481788ec4266d049a74e3c49e53d2c9e7d678f6561219fbf91",
    "summary.json": "8b902b771d1e05dd8ece65ea6ffff750a1c4f6bbcc052bb0b2eec9d5cd1e64a2",
    "model_audit.json": "1ecb9844d9b30dd82096207ecf73e7adcb2b40dd1c651aa2460e45625f4fb680",
    "costs.json": "d5e5d4880f6c4583eaea12660ea14cc168c19dd174d8ac9e0063e44f1c068497",
    "provenance.json": "9178bed801456745e4dbeec7e05e300dfafc4c3550190c12d20c56a4e26de9a5",
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def public_safe(value: object) -> None:
    if isinstance(value, str) and FORBIDDEN.search(value):
        raise ValueError("공개 문자열에 사용자 또는 비공개 경로가 있다")
    if isinstance(value, dict):
        for key, child in value.items():
            public_safe(key)
            public_safe(child)
    elif isinstance(value, list):
        for child in value:
            public_safe(child)


def snapshot(folder: Path) -> dict:
    files = {path.relative_to(folder).as_posix(): D.sha(path)
             for path in sorted(folder.rglob("*")) if path.is_file()}
    return {"file_count": len(files), "sha256": digest(files)}


def public_evaluation(seed: int, arm: str, report: dict, report_sha: str,
                      artifact_sha: str, outcome: str) -> dict:
    fields = ("protocol", "protocol_sha256_lf", "task", "eval_seed", "num_envs",
              "horizon_steps", "step_dt", "action_scale", "versions", "checkpoint",
              "metrics", "standstill", "by_condition")
    return {"schema": "residual_success_public_evaluation_v2", "seed": seed, "arm": arm,
            **{key: report[key] for key in fields}, "recovery_outcome": outcome,
            "source_evaluation_report_sha256": report_sha,
            "source_training_artifact_sha256": artifact_sha}


def public_call(call: dict, receipt: dict) -> dict:
    fields = ("step", "provider_model", "finish_reason", "content", "status", "codes",
              "reasons", "input_tokens", "output_tokens", "total_tokens", "wall_s")
    return {**receipt, **{key: call.get(key) for key in fields}}


def _model_evidence(folder: Path, epoch: int, kind: str, packet: dict,
                    api_root: Path) -> tuple[dict, list[dict]]:
    model_folder = folder / f"epoch{epoch}/model"
    packet_path = model_folder / f"{kind}.packet.json"
    if D.read(packet_path) != packet:
        raise ValueError("모델 packet이 재생 입력과 다르다")
    saved = D.read(model_folder / f"{kind}.review1.json")
    record_path = model_folder / f"{kind}.dialogue1.json"
    record = D.read(record_path)
    review = D.MC._review_record(record, packet_path.read_bytes(), kind, api_root,
                                 record_path=record_path)
    expected = {"record": record_path.name, "record_sha256": D.sha(record_path),
                "review": review, "request_receipts": record["request_receipts"]}
    if saved != expected:
        raise ValueError("모델 review와 record·API receipt 재생이 다르다")
    calls = [public_call(call, receipt)
             for call, receipt in zip(record["calls"], record["request_receipts"])]
    return {"status": review["status"], "codes": review.get("codes", []),
            "requests": review["requests"], "first_status": review.get("first_status"),
            "retried": review.get("retried", False), "packet_sha256": D.sha(packet_path),
            "record_sha256": D.sha(record_path), "api_receipts_verified": True}, calls


def verify_and_build(isaaclab: Path) -> dict[str, dict]:
    if D.sha(RUN / "result.json") != SOURCE_RESULT_SHA or D.sha(D.PROTOCOL, True) != PROTOCOL_SHA_LF:
        raise ValueError("고정 원본 또는 사전등록 SHA가 다르다")
    protocol, anchor = D.verify(isaaclab, ANCHOR)
    source = D.read(RUN / "result.json")
    if (source.get("status") != "complete" or source.get("demo_success") is not True
            or source.get("successes") != 2 or source.get("experiment_commit") != anchor
            or source.get("seeds") != list(D.SEEDS)):
        raise ValueError("원본 실행 상태가 다르다")
    budget = D.MC.audit_budget(RUN / "api")
    if budget != source["api"] or (budget["reserved"], budget["ok"], budget["pending"]) != (8, 8, 0):
        raise ValueError("API 요청 원장 8회 재검증 불일치")
    bands = D.read(ROOT / D.BANDS)["bands"]
    payloads: dict[str, dict] = {}
    cases, model_rows, api_calls, costs, roles = [], [], [], [], {}
    repair_count = selection_count = evaluation_count = 0
    for seed, recorded in zip(D.SEEDS, source["cases"]):
        folder = RUN / f"s{seed}"
        case = D.read(folder / "result.json")
        if case != recorded or case["outcome"] != "recovered" or not case["demo_success"]:
            raise ValueError("seed별 회복 결과가 다르다")
        reference = D.read(folder / "source_evaluation/none.json")
        ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", D.C.correction_registry())
        previous = None
        verified = []
        for epoch, row in enumerate(case["epochs"], 1):
            bundle_path = folder / f"epoch{epoch}/repair_receipt.json"
            bundle = D.read(bundle_path)
            binding = D.binding(protocol, anchor, applied=bundle["applied_after"],
                                faults=["F1", "R1"], checkpoint=bundle["input_checkpoint_sha256"])
            receipt = D.verify_repair(bundle_path, ledger, expected_epoch=epoch,
                expected_applied=bundle["applied_before"],
                expected_input_sha=bundle["input_checkpoint_sha256"], previous=previous,
                reference=reference, bands=bands, expected_binding=binding, seed=seed)
            if receipt != row["verified_receipt"]:
                raise ValueError("교정 receipt 재생 결과가 저장 결과와 다르다")
            expected_verdict = "not_recovered" if epoch == 1 else "recovered"
            if receipt["verdict"] != expected_verdict:
                raise ValueError("두 교정의 판정 순서가 사전등록 성공 흐름과 다르다")
            verified.append(receipt)
            previous = receipt["result_sha256"]
            repair_count += 1
            source_artifact = D.read(folder / ("artifact_compound.json" if epoch == 1 else "epoch1/artifact.json"))
            reference_artifact = D.read(folder / "artifact_none.json")
            diagnosis_packet = D.diagnosis_packet(source_artifact, reference_artifact, epoch,
                                                   bundle["applied_before"], None if epoch == 1 else verified[0])
            selection = D.replay_selection(folder / f"epoch{epoch}/model", diagnosis_packet, RUN / "api")
            if selection != row["selection"] or selection["selected_by"] != "model":
                raise ValueError("검사 선택 receipt 또는 선택 주체가 다르다")
            selection_count += 1
            diag_summary, diag_calls = _model_evidence(folder, epoch, "diagnosis",
                                                        diagnosis_packet, RUN / "api")
            correction_packet = D.read(folder / f"epoch{epoch}/model/correction.packet.json")
            correction_summary, correction_calls = _model_evidence(
                folder, epoch, "correction", correction_packet, RUN / "api")
            model_rows.append({"seed": seed, "epoch": epoch,
                "applied_before": bundle["applied_before"], "selected_by": selection["selected_by"],
                "probe": selection["probe"], "fact_ids": selection["fact_ids"],
                "host_claims": selection["host_decision"]["claims"],
                "host_expected_outcomes": selection["host_decision"]["expected_outcomes"],
                "confirmed_component": row["confirmed_component"],
                "correction_id": receipt["correction_id"], "recovery_verdict": receipt["verdict"],
                "diagnosis": diag_summary, "correction": correction_summary})
            api_calls.extend({"seed": seed, "epoch": epoch, "kind": "diagnosis", **item}
                             for item in diag_calls)
            api_calls.extend({"seed": seed, "epoch": epoch, "kind": "correction", **item}
                             for item in correction_calls)
            roles[f"s{seed}.epoch{epoch}.repair_receipt"] = D.sha(bundle_path)
            roles[f"s{seed}.epoch{epoch}.selection"] = D.sha(folder / f"epoch{epoch}/model/selection.json")
        reports = {
            "NONE": folder / "source_evaluation/none.json",
            "COMPOUND": folder / "source_evaluation/compound.json",
            "FIRST": folder / "epoch1/evaluation/first.json",
            "BOTH": folder / "epoch2/evaluation/both.json",
        }
        artifacts = {
            "NONE": folder / "artifact_none.json", "COMPOUND": folder / "artifact_compound.json",
            "FIRST": folder / "epoch1/artifact.json", "BOTH": folder / "epoch2/artifact.json",
        }
        outcomes, metrics = {}, {}
        for arm in ("NONE", "COMPOUND", "FIRST", "BOTH"):
            report, artifact = D.read(reports[arm]), D.read(artifacts[arm])
            D.V.validated_reports({"reports": {arm: report}}, {arm: artifact})
            outcome = D.recovery(report, reference, bands)["outcome"]
            outcomes[arm], metrics[arm] = outcome, report["metrics"]
            name = f"evaluations/s{seed}_{arm.lower()}.json"
            payloads[name] = public_evaluation(seed, arm, report, D.sha(reports[arm]),
                                               D.sha(artifacts[arm]), outcome)
            roles[f"s{seed}.{arm}.evaluation"] = D.sha(reports[arm])
            roles[f"s{seed}.{arm}.training_artifact"] = D.sha(artifacts[arm])
            evaluation_count += 1
        if outcomes != {"NONE": "recovered", "COMPOUND": "not_recovered",
                        "FIRST": "not_recovered", "BOTH": "recovered"}:
            raise ValueError("고정 평가 4개 재계산 결과가 다르다")
        none_report, both_report = D.read(reports["NONE"]), D.read(reports["BOTH"])
        if (none_report["checkpoint"]["sha256"] != both_report["checkpoint"]["sha256"]
                or none_report["metrics"] != both_report["metrics"]):
            raise ValueError("두 교정 뒤 checkpoint SHA·고정 평가 지표가 NONE과 다르다")
        cases.append({"seed": seed, "outcome": case["outcome"], "demo_success": True,
            "correction_order": [row["correction_id"] for row in verified],
            "repair_verdicts": [row["verdict"] for row in verified],
            "selection_origin": [row["selection"]["selected_by"] for row in case["epochs"]],
            "final_matches_none": {"checkpoint_sha256_equal": True, "metrics_equal": True,
                                   "checkpoint_sha256": both_report["checkpoint"]["sha256"]},
            "metrics": metrics, "evaluation_outcomes": outcomes})
        costs.append({"seed": seed, "stages_wall_s": case["costs"],
                      "owned_process_wall_sum_s": sum(case["costs"].values())})
    api_calls.sort(key=lambda row: row["request_index"])
    if [row["request_index"] for row in api_calls] != list(range(1, 9)):
        raise ValueError("API 원장과 공개 모델 호출 8회의 일대일 결합이 다르다")
    if source["decision_metrics"] != D.decision_metrics(
            [epoch for case in source["cases"] for epoch in case["epochs"]]):
        raise ValueError("모델 유효율·규칙 전환율 재계산이 다르다")
    payloads["summary.json"] = {
        "schema": "residual_success_public_summary_v2", "status": "complete",
        "demo_success": True, "seeds": list(D.SEEDS), "successful_cases": 2, "cases": cases,
        "decision_metrics": source["decision_metrics"],
        "model_validity": source["decision_metrics"]["model_validity"],
        "rule_transition": source["decision_metrics"]["rule_transition"],
        "end_to_end_recovery": source["end_to_end_recovery"],
        "interpretation": "모델 선택 유효율, 등록 규칙 전환율, 두 교정 뒤 회복률을 서로 다른 지표로 보고한다.",
        "rule_transition_effect_claimed": False, "design_coupling": source["design_coupling"],
        "wording": source["wording"]}
    payloads["model_audit.json"] = {
        "schema": "residual_success_model_decisions_v2", "model": D.SELECT.MODEL,
        "diagnosis_rows": model_rows, "raw_calls": api_calls,
        "decision_metrics": source["decision_metrics"],
        "model_validity": source["decision_metrics"]["model_validity"],
        "format_validity": source["decision_metrics"]["format_validity"],
        "semantic_validity_given_format": source["decision_metrics"]["semantic_validity_given_format"],
        "first_pass_validity": source["decision_metrics"]["first_pass_validity"],
        "rule_transition": source["decision_metrics"]["rule_transition"],
        "rule_transition_effect_claimed": False}
    payloads["costs.json"] = {
        "schema": "residual_success_costs_v2", "cases": costs,
        "owned_process_wall_sum_s": sum(row["owned_process_wall_sum_s"] for row in costs),
        "wall_definition": "각 소유 프로세스 실행 구간의 벽시계 합계다.",
        "api": {"requests": len(api_calls), "cap": D.MC.API_CAP,
                **{key: sum(row[key] for row in api_calls)
                   for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s")}},
        "export_gpu_executions": 0, "export_model_requests": 0}
    payloads["provenance.json"] = {
        "schema": "residual_success_public_provenance_v2", "experiment_commit": anchor,
        "protocol": "bench/protocols/residual_demo_v2.json", "protocol_sha256_lf": PROTOCOL_SHA_LF,
        "source_result_sha256": SOURCE_RESULT_SHA, "source_roles_sha256": roles,
        "source_snapshot_sha256": snapshot(RUN)["sha256"],
        "validation": {"protocol_and_bound_sources": True, "api_receipt_crosschecks": len(api_calls),
            "selection_receipts_replayed": selection_count, "repair_receipts_replayed": repair_count,
            "fixed_evaluations_recalculated": evaluation_count},
        "payload_sha256": {name: digest(value) for name, value in payloads.items()}}
    for value in payloads.values():
        public_safe(value)
    return payloads


def verify_payloads(payloads: dict[str, dict]) -> None:
    if not PUBLIC_SHA256 or set(payloads) != set(PUBLIC_SHA256):
        raise ValueError("공개 파일 목록이 canonical 내용 결합과 다르다")
    for name, value in payloads.items():
        public_safe(value)
        if digest(value) != PUBLIC_SHA256[name]:
            raise ValueError("공개 payload의 canonical 내용 결합 불일치")
    provenance = payloads["provenance.json"]
    expected_manifest = {name: digest(value) for name, value in payloads.items()
                         if name != "provenance.json"}
    if provenance.get("payload_sha256") != expected_manifest:
        raise ValueError("공개 provenance manifest가 payload와 다르다")


def verify_public(folder: Path = OUTPUT) -> dict:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = {path.relative_to(folder).as_posix(): path
             for path in folder.rglob("*") if path.is_file()}
    payloads = {name: D.read(path) for name, path in paths.items()}
    verify_payloads(payloads)
    summary = payloads["summary.json"]
    return {"status": "valid", "files": len(paths), "model_validity": summary["model_validity"],
            "rule_transition": summary["rule_transition"],
            "end_to_end_recovery": summary["end_to_end_recovery"]}


def export(isaaclab: Path) -> dict:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("기존 공개 결과 또는 감사 기록은 덮어쓰지 않는다")
    before = snapshot(RUN)
    payloads = verify_and_build(isaaclab)
    verify_payloads(payloads)
    after = snapshot(RUN)
    if before != after:
        raise ValueError("공개 추출 중 원본 파일이 바뀌었다")
    for name, value in payloads.items():
        D.write_new(OUTPUT / name, value)
    D.write_new(AUDIT, {"source_before": before, "source_after": after,
                        "public_sha256": {name: digest(value) for name, value in payloads.items()},
                        "gpu_executions": 0, "model_requests": 0})
    return {"status": "exported", "files": len(payloads), "source_unchanged": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command == "build":
        payloads = verify_and_build(args.isaaclab)
        result = {"files": len(payloads), "canonical_sha256":
                  {name: digest(value) for name, value in payloads.items()}}
    elif args.command == "export":
        result = export(args.isaaclab)
    else:
        result = verify_public()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
