"""두 번째 진단에서 멈춘 residual v1의 CPU 감사와 공개 결과 추출."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_demo_v1 as D

C, MC = D.C, D.MC
RUN = ROOT / "bench/private/codex_support_20261015/task29/runs/residual_v1_20261017"
OUTPUT = ROOT / "evals/results/residual_failure_public_20261017_v2"
AUDIT = ROOT / "bench/private/codex_support_20261015/task30/export_receipt_v2.json"
SOURCE_RESULT_SHA = "9c6195ebfe94222e67e11508c4b5238a9f9c6c379a885d3184b490f1cc545f05"
ANCHOR = "4198c80f4ee4c6bdf0bfa779ae2b412e5aa55233"
PROTOCOL_SHA_LF = "b1b3f231f82c31f28c1b3971d4ab167bbcb4d5e137115ab6b8a14a8cb7c01f0c"
FORBIDDEN = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support", re.I)
# 고정 원본을 검증해 만든 공개 payload의 canonical SHA. 공개본 자체의 manifest와
# payload를 함께 바꿔도 검증을 통과하지 못하도록 코드에 결합한다.
PUBLIC_SHA256 = {
    "evaluations/s2110_none.json": "69bc079a21c129e660f78e56923511f2e2f71b7363af35e7b3f57be3a466dca5",
    "evaluations/s2110_compound.json": "0900c9c3b16f3ee48d6097185753cdf124e5a75e974fa62b2a687319778e5129",
    "evaluations/s2110_first.json": "736560dc0224baac5d4e37d92b9a31f340d1e307601b844e8693aa9f384f142b",
    "evaluations/s2111_none.json": "b28bf2ad483eb337e25993a0a08c485d3023ada2b6a2c4ededb6c3ca29db52b5",
    "evaluations/s2111_compound.json": "d30bd075ca364d0abfe151570b60e8fd188803ac5b6d727d398765f5d5449141",
    "evaluations/s2111_first.json": "509573dc9beeef03d6792c1408683c36f525fdd69c45992c600df4e259612891",
    "summary.json": "1645c3e0ea10785d947ebfec2d57529909808b738175d23e6ef3b9d7e127afa0",
    "model_contract_audit.json": "d18f1ac8b8bfa5b4a5213732a3e5838f57faf2ec859070bbd40d0b0456dd8763",
    "costs.json": "bd7c0ac89abe812f309ea48c21c3aa791c397ee9bb6b7647f346c94935bf9b8b",
    "provenance.json": "c1a6531479f9b07df2d5b8bdaf19c2909ea77fbe134347886181848d096a2640",
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def public_safe(value: object) -> None:
    """JSON escape를 푼 뒤 중첩 문자열까지 경로 누출을 검사한다."""
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
    files = {p.relative_to(folder).as_posix(): D.sha(p)
             for p in sorted(folder.rglob("*")) if p.is_file()}
    return {"file_count": len(files), "sha256": digest(files), "files": files}


def diagnosis_details(content: str, packet: dict) -> dict:
    """원래 fail-fast 판정과 별도로 복수 형식 오류를 조사한다. 결정을 실행하지 않는다."""
    value = json.loads(MC.F.diagnosis_body(content))
    ranking = value["ranking"]
    public = list(C.L.MECHANISMS)
    probe = value["probe"]
    expected = ({key: C.R.public_catalog()[probe]["expect"].get(key)
                 for key in ranking[:3]} if value["action"] == "continue" else {})
    facts = packet["facts"]
    return {
        "ranking_is_public_permutation": len(ranking) == len(public) and set(ranking) == set(public),
        "expected_outcomes_matches_top_three": value["expected_outcomes"] == expected,
        "actual_expected_keys": sorted(value["expected_outcomes"]),
        "required_expected_keys": sorted(expected),
        "claims_match_packet": all(row["id"] in facts and row["value"] == facts[row["id"]]
                                   for row in value["claims"]),
    }


def boundary_analysis(content: str, packet: dict) -> dict:
    """사후 형식 보정으로 잠재 경계를 재현한다. 원래 호출 판정은 바꾸지 않는다."""
    decision = json.loads(MC.F.diagnosis_body(content))
    patched = deepcopy(decision)
    patched["expected_outcomes"] = {
        key: C.R.public_catalog()[decision["probe"]]["expect"][key]
        for key in decision["ranking"][:3]}
    C.validate_diagnosis(patched, packet)
    state = C.initial_state(patched, packet["applied_corrections"])
    expectations = {key: C.registry()["probes"][decision["probe"]]["expect"][key]
                    for key in state["hypotheses"]}
    return {"analysis_only": True, "original_experimental_verdict_changed": False,
            "only_expected_outcomes_subset_changed": True,
            "format_corrected_decision_accepted_by_v1": True,
            "probe": state["next_probe"], "remaining_hypotheses": state["hypotheses"],
            "probe_expectations": expectations,
            "partitions_remaining_hypotheses": len(set(expectations.values())) > 1,
            "registered_rule_next_probe": C.M._next(C.registry(), state["hypotheses"], {}),
            "rule_probe_executed": False}


def evaluation(seed: int, arm: str, report: dict, report_sha: str, artifact_sha: str) -> dict:
    return {"schema": "residual_failure_public_evaluation_v1", "seed": seed, "arm": arm,
            **{key: report[key] for key in ("protocol", "protocol_sha256_lf", "task", "eval_seed",
                "num_envs", "horizon_steps", "step_dt", "action_scale", "versions", "checkpoint",
                "metrics", "standstill", "by_condition")},
            "source_evaluation_report_sha256": report_sha, "source_training_artifact_sha256": artifact_sha}


def verify_and_build(isaaclab: Path) -> dict:
    if D.sha(RUN / "result.json") != SOURCE_RESULT_SHA or D.sha(D.PROTOCOL, True) != PROTOCOL_SHA_LF:
        raise ValueError("고정 원본 또는 사전등록 SHA가 다르다")
    protocol, anchor = D.verify(isaaclab, ANCHOR)
    source = D.read(RUN / "result.json")
    if (source["status"] != "complete" or source["demo_success"] is not False
            or source["experiment_commit"] != anchor or source["seeds"] != [2110, 2111]):
        raise ValueError("원본 실행 상태가 다르다")
    budget = MC.audit_budget(RUN / "api")
    if budget != source["api"] or (budget["reserved"], budget["ok"], budget["pending"]) != (10, 10, 0):
        raise ValueError("요청 원장 재검증 불일치")
    bands = D.read(ROOT / D.BANDS)["bands"]
    payloads, cases, calls, dialogues, boundaries, costs, roles = {}, [], [], [], [], [], {}
    for seed, recorded_case in zip(D.SEEDS, source["cases"]):
        folder = RUN / f"s{seed}"
        case = D.read(folder / "result.json")
        if case != recorded_case or (case["outcome"], case["reason"]) != ("HOLD", "diagnosis_invalid"):
            raise ValueError("seed별 종료 상태 불일치")
        first, second = case["epochs"]
        bundle_path = folder / "epoch1/repair_receipt.json"
        bundle = D.read(bundle_path)
        reference = D.read(folder / "source_evaluation/none.json")
        binding = D.binding(protocol, anchor, applied=[C.REWARD], checkpoint=bundle["input_checkpoint_sha256"])
        receipt = D.verify_repair(bundle_path, D.CorrectionLedger(folder / "correction.ledger.jsonl",
            C.correction_registry()), expected_epoch=1, expected_applied=[],
            expected_input_sha=bundle["input_checkpoint_sha256"], previous=None, reference=reference,
            bands=bands, expected_binding=binding, seed=seed)
        if receipt != first["verified_receipt"] or receipt["verdict"] != "not_recovered":
            raise ValueError("첫 교정의 비회복 receipt 재생 불일치")
        if second["epoch"] != 2 or second["applied_before"] != [C.REWARD] or second["probe_steps"]:
            raise ValueError("두 번째 epoch 중단 지점이 다르다")
        for epoch, kind in ((1, "diagnosis"), (1, "correction"), (2, "diagnosis")):
            model = folder / f"epoch{epoch}/model"
            packet_path, dialogue_path = model / f"{kind}.packet.json", model / f"{kind}.dialogue1.json"
            data, raw = packet_path.read_bytes(), D.read(dialogue_path)
            packet = json.loads(data)
            review = MC._review_record(raw, data, kind, RUN / "api")
            recorded_review = D.read(model / f"{kind}.review1.json")
            if (recorded_review["review"] != review or recorded_review["record_sha256"] != D.sha(dialogue_path)
                    or recorded_review["request_receipts"] != raw["request_receipts"]):
                raise ValueError("모델 검토 기록 불일치")
            if epoch == 2 and packet["prior_receipt"] != receipt:
                raise ValueError("두 번째 진단 입력의 앞선 receipt 불일치")
            dialogues.append({"seed": seed, "epoch": epoch, "kind": kind,
                "status": review["status"], "codes": review["codes"], "requests": review["requests"],
                "packet_sha256": D.sha(packet_path), "record_sha256": D.sha(dialogue_path),
                "api_receipts_verified": True})
            for call, request in zip(raw["calls"], raw["request_receipts"]):
                entry = {"seed": seed, "epoch": epoch, "kind": kind, **request,
                    **{key: call[key] for key in ("step", "provider_model", "finish_reason", "content",
                        "status", "codes", "reasons", "input_tokens", "output_tokens", "total_tokens", "wall_s")}}
                if kind == "diagnosis":
                    entry["independent_format_checks"] = diagnosis_details(call["content"], packet)
                calls.append(entry)
            if epoch == 2:
                boundaries.append({"seed": seed, **boundary_analysis(raw["calls"][-1]["content"], packet)})
        metrics, outcomes = {}, {}
        for arm, relative in (("NONE", "source_evaluation/none.json"),
                              ("COMPOUND", "source_evaluation/compound.json"),
                              ("FIRST", "epoch1/evaluation/first.json")):
            report_path = folder / relative
            report = D.read(report_path)
            artifact_path = folder / ("epoch1/artifact.json" if arm == "FIRST" else f"artifact_{arm.lower()}.json")
            artifact = D.read(artifact_path)
            D.V.validated_reports({"reports": {arm: report}}, {arm: artifact})
            measured = D.recovery(report, reference, bands)
            metrics[arm], outcomes[arm] = report["metrics"], measured["outcome"]
            name = f"evaluations/s{seed}_{arm.lower()}.json"
            payloads[name] = evaluation(seed, arm, report, D.sha(report_path), D.sha(artifact_path))
            roles[f"s{seed}.{arm}.evaluation"] = D.sha(report_path)
            roles[f"s{seed}.{arm}.training_artifact"] = D.sha(artifact_path)
        if outcomes != {"NONE": "recovered", "COMPOUND": "not_recovered", "FIRST": "not_recovered"}:
            raise ValueError("고정 평가 재계산이 예상 기록과 다르다")
        roles[f"s{seed}.first_repair_receipt"] = D.sha(bundle_path)
        costs.append({"seed": seed, "stages_wall_s": case["costs"],
                      "owned_process_wall_sum_s": sum(case["costs"].values())})
        cases.append({"seed": seed, "outcome": "HOLD", "reason": case["reason"],
            "stop_epoch": 2, "demo_success": False, "first_diagnosis_valid_after_feedback": True,
            "first_probe": first["probe_steps"][0]["probe"],
            "first_probe_outcome": first["probe_steps"][0]["outcome"],
            "first_probe_measurement": first["probe_steps"][0]["measurement"],
            "applied_corrections": [C.REWARD], "recovery_mode": "restart", "training_updates": 300,
            "first_receipt_verified": True, "epoch2_opened": True, "epoch2_probes_executed": 0,
            "second_correction_executed": False, "metrics": metrics, "evaluation_outcomes": outcomes})
    calls.sort(key=lambda x: x["request_index"])
    if [row["request_index"] for row in calls] != list(range(1, 11)):
        raise ValueError("원장 요청과 모델 호출의 일대일 결합 불일치")
    payloads["summary.json"] = {"schema": "residual_failure_public_summary_v1", "status": "complete",
        "demo_success": False, "successful_cases": 0, "cases": cases, "seeds": list(D.SEEDS),
        "classification": "model_contract_violation", "infrastructure_failure_observed": False,
        "retry_as_infrastructure_failure": False, "existing_result_reclassified": False,
        "design_coupling": protocol["design_coupling"],
        "audit_scope": "저장 응답·원장·교정 receipt·고정 평가를 CPU로 재검증한 사후 감사. 새 모델 성능 측정은 아니다.",
        "prediction_met": False, "api_requests": budget["reserved"], "api_cap": budget["cap"],
        "wording": "두 seed 모두 첫 보상 교정 뒤 비회복 평가로 두 번째 고리를 열었지만, 모델이 재요청에서도 출력 계약을 충족하지 못해 HOLD했다."}
    payloads["model_contract_audit.json"] = {"schema": "residual_failure_model_audit_v1",
        "model": C.MODEL, "raw_calls": calls, "dialogues": dialogues, "boundary_analysis": boundaries,
        "raw_valid": sum(row["status"] == "valid" for row in calls),
        "raw_invalid": sum(row["status"] == "invalid" for row in calls),
        "dialogues_valid": sum(row["status"] == "valid" for row in dialogues),
        "dialogues_invalid": sum(row["status"] == "invalid" for row in dialogues),
        "contract": {"public_ranking_categories": list(C.L.MECHANISMS), "expected_outcomes_count": 3,
                     "expected_outcomes_scope": "ranking의 상위 세 공개 범주",
                     "feedback_requests_per_dialogue": 1, "validator_reports_first_error_only": True},
        "causal_classification": "두 번째 재요청의 D_EXPECTED는 모델 계약 위반이다. 이미 교정한 성분의 검사 재선택 허용은 별도의 잠재 경계 결함이며 이번 종료 원인은 아니다."}
    payloads["costs.json"] = {"schema": "residual_failure_costs_v1", "cases": costs,
        "owned_process_wall_sum_s": sum(row["owned_process_wall_sum_s"] for row in costs),
        "wall_definition": "각 소유 프로세스 실행 구간의 벽시계 합계. GPU 커널 시간이나 전체 lease 경과 시간이 아니다.",
        "api": {"requests": len(calls), "cap": MC.API_CAP,
                **{key: sum(row[key] for row in calls)
                   for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s")}},
        "audit_gpu_executions": 0, "audit_model_requests": 0}
    payloads["provenance.json"] = {"schema": "residual_failure_public_provenance_v1",
        "experiment_commit": anchor, "protocol": "bench/protocols/residual_demo_v1.json",
        "protocol_sha256_lf": PROTOCOL_SHA_LF, "source_result_sha256": SOURCE_RESULT_SHA,
        "source_roles_sha256": roles, "source_snapshot_sha256": snapshot(RUN)["sha256"],
        "validation": {"protocol_and_bound_sources": True, "api_receipt_crosschecks": len(calls),
            "feedback_dialogues_replayed": len(dialogues), "repair_receipts_replayed": len(cases),
            "fixed_evaluations_recalculated": 6},
        "payload_sha256": {name: digest(value) for name, value in payloads.items()}}
    for value in payloads.values():
        public_safe(value)
    return payloads


def verify_payloads(payloads: dict) -> None:
    if not PUBLIC_SHA256 or set(payloads) != set(PUBLIC_SHA256):
        raise ValueError("공개 파일 목록이 고정 내용 결합과 다르다")
    for name, value in payloads.items():
        public_safe(value)
        if digest(value) != PUBLIC_SHA256[name]:
            raise ValueError("공개 payload의 고정 내용 결합 불일치")


def verify_public(folder: Path = OUTPUT) -> dict:
    paths = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
    verify_payloads({name: D.read(path) for name, path in paths.items()})
    return {"status": "valid", "files": len(paths), "classification": "model_contract_violation",
            "stop_epoch": 2, "demo_success": False}


def export(isaaclab: Path) -> dict:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("기존 공개 결과 또는 감사 기록은 덮어쓰지 않는다")
    before = snapshot(RUN)
    payloads = verify_and_build(isaaclab)
    verify_payloads(payloads)
    after = snapshot(RUN)
    if before != after:
        raise ValueError("감사 중 원본 파일이 바뀌었다")
    for name, value in payloads.items():
        MC.write_new(OUTPUT / name, value)
    MC.write_new(AUDIT, {"source_before": before, "source_after": after,
                        "public_sha256": {name: digest(value) for name, value in payloads.items()},
                        "gpu_executions": 0, "model_requests": 0})
    return {"status": "exported", "files": len(payloads), "source_unchanged": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.isaaclab) if args.command == "export" else verify_public(),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
