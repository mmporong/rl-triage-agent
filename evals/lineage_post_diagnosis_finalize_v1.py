"""완료된 세 검사와 실패한 종료 처리를 분리해 CPU 집계한다. 실행 권한은 없다."""
from __future__ import annotations

import argparse
from pathlib import Path

import lineage_post_diagnosis_resume_v1 as R


def terminal(view: dict) -> dict:
    """세 검사 예산을 소진한 미확정 상태는 HOLD로 기록한다."""
    R._require(view.get("budget_left") == 0 and len(view.get("observations", [])) == 3,
               "완료된 세 검사와 소진된 예산이 필요하다")
    outcomes = [row["outcome"] for row in view["observations"]]
    if "unknown" in outcomes:
        reason = "probe_outcome_unknown"
    elif view["status"] == "unidentifiable" and len(view["remaining_candidates"]) > 1:
        reason = "decision_limit"
    elif view["status"] == "confirmed" and view["remaining_candidates"] == ["physics"]:
        reason = "repair_unsupported"
    else:
        raise ValueError("등록된 사후 집계 상태가 아니다")
    return {"end_state": "HOLD", "hold_reason": reason,
            "selection_status": view["status"], "approval_allowed": False}


def analyze(plan_path: Path, stage: Path, failed_lease: Path, out: Path) -> dict:
    """원문을 바꾸지 않고 prefix·실측·종료 상태를 재검산한다."""
    plan, stage = R._read(plan_path), Path(stage).resolve()
    R._require(not Path(out).exists() and not (stage / "result.json").exists(),
               "새 분석 파일과 원래 미작성 result가 필요하다")
    lease = R._read(failed_lease)
    R._require(lease.get("state") == "failed" and lease.get("exitCode") == 1
               and lease.get("owner") == "rl-triage-lineage-sources-v1"
               and lease.get("completedUtc") is not None,
               "실패 종료한 controller 임대 원문이 필요하다")
    post = R._assert_ref(plan["inputs"]["post"])
    reference = R._assert_ref(plan["inputs"]["reference"])
    old_ledger = R._assert_ref(plan["old_execution"]["ledger"])
    current = R._prepare_value(
        old_stage=Path(plan["old_execution"]["stage"]),
        transport_root=Path(plan["old_execution"]["transport_root"]),
        operator_failure=R._assert_ref(plan["old_execution"]["operator_failure"]),
        post_input=post, reference_input=reference,
        gate_path=R._assert_ref(plan["inputs"]["gate"]),
        gate_sha256=plan["inputs"]["gate_sha256"], lab=Path(plan["lab"]),
        protocol_path=R._assert_ref(plan["protocol"]), anchor=plan["anchor"],
        run_id=plan["run_id"], policy=plan["policy"],
    )
    R._require(current == plan, "준비 원문 재검증이 다르다")
    ledger_path = stage / "probes.ledger.jsonl"
    before = ledger_path.read_bytes()
    R._require(before.startswith(old_ledger.read_bytes()), "원래 8사건 prefix가 다르다")
    ledger = R.C.S.M.RegistryLedger(ledger_path, R.C.RC.registry())
    events, requests = ledger.events(), ledger.requests()
    R._require(len(events) == 12 and len(requests) == 3
               and all(row["state"] == "done" for row in requests.values()),
               "세 검사와 12사건이 완료되지 않았다")
    auth_path = stage / "probes/step_03/authorization.json"
    auth = R._read(auth_path)
    request = requests[auth["request_id"]]
    request_events = [row for row in events if row["request_id"] == auth["request_id"]]
    with R.PARENT.parent_scope(post, reference, plan["run_id"], plan["policy"]) as context:
        snapshot = R.CANDIDATE1._snapshot_at(
            auth["snapshot"], events[:8], plan["prefix_rows"])
        R._require(auth["choice"] == plan["third_choice"]
                   and auth["request_spec"] == plan["third_request_spec"],
                   "기록된 세 번째 선택과 실제 승인이 다르다")
        R.CANDIDATE1._done_authorization(
            auth_path, R._sha(auth_path), snapshot, auth["choice"],
            auth["request_spec"], auth["request_id"], ledger_path,
            request, request_events,
        )
        paired = {}
        with R.CANDIDATE2.typed_backend_scope():
            for role, source in (("case", post), ("reference", reference)):
                root = auth_path.parent / role
                raw = R._read(root / "probe_receipt.json")
                measurement, _ = R.CANDIDATE1._receipt(
                    raw, role=role, plan_path=root / "plan.json",
                    plan=R.POST_BACKEND._verify_plan(root / "plan.json"),
                    auth_path=auth_path, auth_sha=R._sha(auth_path),
                    request_id=auth["request_id"], collected_input=source,
                    source_binding=plan["binding"][role],
                )
                paired[role] = {"status": raw["status"], "measurement": measurement,
                                "execution": raw["execution"],
                                "raw_receipt_sha256": R._sha(root / "probe_receipt.json")}
        outcome = R.C.classify_raw("P_physics", paired["case"]["measurement"],
                                    paired["reference"]["measurement"])
        R._require(request["receipt"]["outcome"] == outcome,
                   "실측 재분류와 원장 결과가 다르다")
        final = R.C.V.build_snapshot(context["packet"], plan["run_id"], plan["policy"],
                                     R.C.V.file_digest(R.C.V.PROTOCOL), ledger_path)
        result = terminal(final["view"])
    R._require(ledger_path.read_bytes() == before, "분석이 원장을 바꿨다")
    value = {"schema": "lineage_post_diagnosis_finalization_analysis_v1",
             "status": "complete", **result,
             "controller_status": "invalid_finalization",
             "finalization_error": "normal_third_probe_expected_open_instead_of_unidentifiable",
             "measurements_complete": True, "case_id": "case-07", "seed": 2307,
             "probe_order": ["P_episode", "P_noise", "P_physics"],
             "outcomes": ["normal", "normal", outcome],
             "remaining_candidates": final["view"]["remaining_candidates"],
             "total_logical_probes": 3, "reused_logical_probes": 2,
             "new_logical_probes": 1, "new_api_requests": 0,
             "new_training_jobs": 0, "new_correction_approvals": 0,
             "model_responses_reused": 3, "paired_third": paired,
             "plan": R._ref(plan_path), "execution_ledger": R._ref(ledger_path),
             "failed_lease": R._ref(failed_lease), "original_result_written": False,
             "analysis_only": True, "source_sha256": R._sha(Path(__file__)),
             "final_snapshot_digest": R.C.L.digest(final)}
    R._write_new(out, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "stage", "failed-lease", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.plan, args.stage, args.failed_lease, args.out)


if __name__ == "__main__":
    main()
