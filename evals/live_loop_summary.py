"""저장된 실제 승인 고리 8건을 다시 감사하고 분모·누락·비용표를 쓴다(GPU 없음)."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import live_loop_evidence as E


def validated_runs(case, runs, requests):
    """receipt 원장을 분모로 삼고 결과표의 누락·중복·probe 치환을 거부한다."""
    receipts = {rid: req for rid, req in requests.items() if "receipt" in req}
    ids = [run["request_id"] for run in runs]
    if len(ids) != len(set(ids)) or set(ids) != set(receipts):
        raise ValueError(f"{case}: receipt와 runs의 요청 ID가 1:1이 아니다")
    by_id = {run["request_id"]: run for run in runs}
    if any(by_id[rid]["probe"] != req["probe"] for rid, req in receipts.items()):
        raise ValueError(f"{case}: runs의 probe가 승인 원장과 다르다")
    return [by_id[rid] for rid in receipts]


def budget_acceptance(rows):
    return {"all_case_cost_budgets_within": all(r["cost_budget_status"] == "within" for r in rows),
            "all_end_to_end_budgets_within": all(r["end_to_end_budget_status"] == "within" for r in rows),
            "all_probe_budgets_within": all(r["probe_budgets_within"] for r in rows)}


def summarize(prereg_path: Path, tag: str, *, write=True):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag에는 영문·숫자·밑줄·하이픈만 쓴다")
    prereg = E.read(prereg_path)
    E.verify_prereg(prereg_path, prereg)
    root = E.ROOT / "evals/results" / tag
    p0_key = E.read(E.ROOT / "bench/answer_key_p0c.json")
    t1_key = E.read(E.ROOT / "bench/answer_key_t1.json")
    rows, audits, failures, recovered = [], [], [], []
    agreements, meters, run_times = [], [], []
    for spec in prereg["cases"]:
        case = spec["case"]
        folder = root / "cases" / case
        result, stored = E.read(folder / "result.json"), E.read(folder / "audit.json")
        source = {k: E.ROOT / v["path"] for k, v in spec["sources"].items()}
        fresh = E.A.finalize(SimpleNamespace(loop_dir=E.loop._dir(case), contract=folder / "contract.json",
                         reference=source["reference"], before=source["before"], after=None, abstain=True,
                         events=folder / "events.json", timeline=folder / "timeline.json" if (folder / "timeline.json").exists() else None))
        for key in ("loop", "behavior", "costs", "cost_events", "end_to_end_wall", "sources"):
            if fresh[key] != stored[key]:
                raise ValueError(f"{case}: 저장된 감사와 재계산 {key}가 다르다")
        if result["costs"] != fresh["costs"]["totals"] or result["observed"] != fresh["loop"]["observed"]:
            raise ValueError(f"{case}: 결과표와 원장이 다르다")
        ledger = E.loop._load(case)[2]
        events = ledger.events()
        if fresh["loop"]["pending_requests"] or fresh["loop"]["incomplete_requests"]:
            raise ValueError(f"{case}: 미완료 승인·고아 요청이 있다")
        counts = {kind: sum(e["event"] == kind for e in events) for kind in ("approved", "rejected", "consumed", "receipt")}
        if counts["rejected"] != 1 or counts["approved"] != counts["consumed"] or counts["consumed"] != counts["receipt"]:
            raise ValueError(f"{case}: 사전등록된 승인·거절 수가 다르다")
        requests = ledger.requests()
        runs = validated_runs(case, result["runs"], requests)
        if len(runs) != counts["receipt"]:
            raise ValueError(f"{case}: receipt 이벤트 수와 요청 수가 다르다")
        prior, ref = E.read(source["prior_probes"]), E.read(source["probe_reference"])["probes"]
        computed = []
        for run in runs:
            run_times.append(run.get("process_to_receipt_s"))
            probe, rid = run["probe"], run["request_id"]
            report_path = E.loop._probe_path(case, probe, tag)
            report, meta = E.read(report_path), E.read(E.loop._execution_path(case, probe, tag))
            if (report["name"] != case or report["seed"] != spec["seed"]
                    or report["checkpoint"]["sha256"] != spec["checkpoint_sha256"]
                    or meta.get("report_sha256_lf") != E.sha(report_path)):
                raise ValueError(f"{case}/{probe}: 실제 입력·출력 SHA가 다르다")
            measurement = report["probes"][probe]["measurement"] or {}
            agreement = E.measurement_agreement(probe, measurement, prior["probes"][probe]["measurement"], ref[probe]["measurement"], prereg)
            receipt = requests[rid]["receipt"]
            agreement.update(receipt_outcome=receipt["outcome"], receipt_agreement=receipt["outcome"] == agreement["prior_outcome"])
            computed.append(agreement)
            agreements.append({"case": case, **agreement})
            if receipt["gpu_s"] != meta["gpu_wall_s"]:
                raise ValueError(f"{case}/{probe}: 원래 GPU 비용이 바뀌었다")
            resource = root / "resources" / report_path.name
            if resource.exists():
                meter = E.read(resource)
                if meter["simulator_steps"] != meter["successful_step_calls"] * meter["num_envs"] or meter["num_envs"] != report["num_envs"]:
                    raise ValueError(f"{case}/{probe}: 실제 제어 전이 수가 다르다")
                meters.append({"case": case, "probe": probe, "app_close_cpu_included": meter["app_close_cpu_included"],
                               "isaac_cpu_scope": meter["isaac_cpu_scope"]})
        if computed != result["agreements"]:
            raise ValueError(f"{case}: 판정·수치 일치 표가 재계산과 다르다")
        stem = case.rsplit("_s", 1)[0]
        truth = "NONE" if stem == "baseline_p0c" else (t1_key if stem.startswith("e") else p0_key)["cases"][stem]["category"]
        status, hypotheses = fresh["loop"]["diagnosis_status"], fresh["loop"]["hypotheses"]
        conclusion = hypotheses[0] if status == "confirmed" else None
        row = {"case": case, "status": status, "truth": truth, "conclusion": conclusion,
               "wrong_confirmation": status == "confirmed" and conclusion != truth,
               "before_label": fresh["behavior"]["before"]["label"], "probes": counts["consumed"],
               "raw_classification_agreement": all(a["live_outcome"] == a["prior_outcome"] for a in computed),
               "receipt_agreement": all(a["receipt_agreement"] for a in computed), "scalar_agreement": all(a["scalar_agreement"] for a in computed),
               "cost_complete": all(v is not None for v in fresh["costs"]["totals"].values()),
               "cost_budget_status": fresh["costs"]["budget_status"],
               "end_to_end_budget_status": fresh["end_to_end_wall"]["budget_status"],
               "probe_budgets_within": not fresh["loop"]["probe_budget_exceeded"] and not fresh["loop"]["request_budget_overruns"],
               "end_to_end_wall_s": fresh["end_to_end_wall"]["elapsed_s"], **result["costs"]}
        rows.append(row)
        audits.append(fresh)
        if result["recovery"]:
            recovered.append({"case": case, **result["recovery"]})
        if result.get("infrastructure_errors"):
            failures.append({"case": case, "errors": result["infrastructure_errors"], "missing_costs": fresh["costs"]["missing_attempts"]})
    all_events = [event for audit in audits for event in audit["cost_events"]]
    costs = E.A.C.summarize_costs(all_events, {})
    costs["budget_status"] = "not_evaluated"
    costs["budget_scope"] = "sum_only; preregistered_limits_apply_per_case_and_are_evaluated_in_rows"
    scalar_fields = [v for a in agreements for v in a["scalar_comparison"].values()]
    acceptance = {"at_least_four_terminal_loops": len(rows) >= 4, "finalize_errors_zero": True,
                  "receipt_classification_matches_all": all(a["receipt_agreement"] for a in agreements),
                  "scalar_tolerance_matches_all": all(v["within_tolerance"] for v in scalar_fields),
                  "process_to_receipt_time_all": all(value is not None for value in run_times),
                  "recovery_once_no_extra_gpu": len(recovered) == 1 and all(r["extra_gpu_wall_s"] == 0 and r["extra_simulator_steps"] == 0
                        and r["execution_records_before"] == r["execution_records_after"] and r["artifact_unchanged"] and not r["orphans_after"] for r in recovered),
                  "all_eight_cost_fields_complete": all(r["cost_complete"] for r in rows),
                  "none_wrong_confirmation_zero": not any(r["wrong_confirmation"] for r in rows if r["truth"] == "NONE"),
                  **budget_acceptance(rows)}
    result = {"schema": "live_loop_summary_v2", "prereg_sha256_lf": E.sha(prereg_path), "cases": len(rows), "rows": rows,
              "probe_attempts": len(agreements), "raw_classification_matches": sum(a["live_outcome"] == a["prior_outcome"] for a in agreements),
              "receipt_matches": sum(a["receipt_agreement"] for a in agreements), "scalar_fields": len(scalar_fields),
              "scalar_field_matches": sum(v["within_tolerance"] for v in scalar_fields), "complete_cost_cases": sum(r["cost_complete"] for r in rows),
              "wrong_confirmations": sum(r["wrong_confirmation"] for r in rows), "recovery": recovered, "infrastructure_failures": failures,
              "costs": costs, "acceptance": acceptance, "accepted": all(acceptance.values()), "meters": meters,
              "scope": "cached_rankings; user_authorized_codex_operator; no_fresh_model; no_human_review; no_repair",
              "cpu_scope_limit": "reported Python process intervals; app-close CPU excluded where flag false; shell/service CPU excluded"}
    if write:
        outputs = (root / "summary_v2.json", root / "costs_v2.csv")
        if any(path.exists() for path in outputs):
            raise FileExistsError("기존 summary_v2/costs_v2는 덮어쓰지 않는다")
        E.A._write_new(outputs[0], result)
        with outputs[1].open("x", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--check-only", action="store_true", help="파일 저장 없이 같은 검증과 요약을 수행")
    args = parser.parse_args(argv)
    result = summarize(args.prereg, args.tag, write=not args.check_only)
    print(json.dumps({k: result[k] for k in ("cases", "probe_attempts", "raw_classification_matches", "receipt_matches", "scalar_fields", "scalar_field_matches", "complete_cost_cases", "wrong_confirmations", "accepted")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
