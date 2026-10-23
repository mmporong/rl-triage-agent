"""고정한 과거 입력 여섯 대화의 준비·재계산. 실제 호출에는 커밋 anchor와 명시 승인이 필요하다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_call_v2 as C  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "bench/protocols/selection_comparison_runtime_v2.json"
RUN_ID = "selection_runtime_20261020_v1"
CLAIMS = ROOT / ".omx/selection_runtime_claims"
ORDER = (("c1", "production"), ("c1", "exposure_full"), ("c1", "exposure_masked"),
         ("c2", "production"), ("c2", "exposure_masked"), ("c2", "exposure_full"))


def read_bundle(path: Path, expected: str | None = None) -> dict:
    raw = path.read_bytes()
    C.require(expected is None or C.raw_sha(raw) == expected, "입력 bundle SHA 불일치")
    value = json.loads(raw)
    C.require(set(value) == {"schema", "run_id", "archive_integrity_sha256", "cases"}
              and value["schema"] == "selection_runtime_inputs_v1" and value["run_id"] == RUN_ID
              and C.V.S._valid_sha(value["archive_integrity_sha256"])
              and isinstance(value["cases"], list) and len(value["cases"]) == 2
              and [row["context_id"] for row in value["cases"]] == ["c1", "c2"], "입력 bundle 범위")
    for row in value["cases"]:
        C.require(set(row) == {"context_id", "packet", "history_utf8", "archive_packet_sha256", "archive_receipt_sha256"}
                  and C.V.S._valid_sha(row["archive_packet_sha256"])
                  and C.V.S._valid_sha(row["archive_receipt_sha256"])
                  and isinstance(row["history_utf8"], str), "과거 context 입력 범위")
        packet = C.V.S.validate_packet(row["packet"])
        C.require(packet["epoch"] == 1 and packet["execution_budget_s"] == 1200, "첫 epoch 입력만 사용")
    return value


def materialize(root: Path, bundle: dict) -> list[Path]:
    snapshots = {}
    for row in bundle["cases"]:
        name = row["context_id"]
        folder = C.inside(root, f"live/{name}")
        folder.mkdir(parents=True, exist_ok=False)
        initial = C.V.build_snapshot(row["packet"], RUN_ID, "model_adaptive", C.V.file_digest(C.V.PROTOCOL),
                                     folder / "initial.jsonl")
        history = folder / "observed.jsonl"
        with history.open("xb") as handle:
            handle.write(row["history_utf8"].encode("utf-8"))
        observed = C.V.build_snapshot(**{**initial["inputs"], "ledger_path": history})
        C.require(len(observed["view"]["observations"]) == 1
                  and observed["view"]["observations"][0]["outcome"] == "abnormal", "단일 과거 abnormal receipt 필요")
        snapshots[name] = {"production": initial, "exposure_full": observed, "exposure_masked": observed}
    return [C.freeze(root, snapshots[name][condition], condition) for name, condition in ORDER]


def execution_plan(bundle_path: Path, scratch: Path) -> dict:
    bundle = read_bundle(bundle_path)
    protocol_sha = C.V.file_digest(C.PROTOCOL)
    C.initialize(scratch, RUN_ID, protocol_sha, fixture=True)
    paths = materialize(scratch, bundle)
    rows = [{"row_id": f"row{index}", "pair_id": None if condition == "production" else name,
             "order": index, "state": C.planned_state(C.prepared(scratch, path)[0]), "max_requests": 3}
            for index, ((name, condition), path) in enumerate(zip(ORDER, paths, strict=True), 1)]
    value = {"schema": "selection_call_execution_v2", "run_id": RUN_ID, "caller_protocol_sha": protocol_sha,
             "model": C.MODEL, "accepted_provider_models": sorted(C.ACCEPTED_PROVIDER_MODELS),
             "params": C.F.request_params(C.MODEL), "api_cap": 20,
             "conditions": list(C.CONDITIONS), "trusted_host_input_required": True, "planned_calls": rows,
             "predictions": {
                 "development_disclosure": C.DEVELOPMENT_DISCLOSURE,
                 "bundle_sha256": C.MC.file_sha(bundle_path), "archive_integrity_sha256": bundle["archive_integrity_sha256"],
                 "runner_sha256_lf": C.V.file_digest(Path(__file__)),
                 "tests_sha256_lf": C.V.file_digest(ROOT / "tests/test_selection_comparison_runtime_v2.py"),
                 "final_model_contract_valid": {"production": "2/2", "exposure": "4/4"},
                 "global_execution_claim": "semantic_plan_digest_single_start_across_roots",
                 "request_bound": 18, "new_GPU_runs": 0, "new_training_runs": 0,
                 "historical_costs_combined_with_API": False, "used_for_recovery_decision": False},
             "result_wording": C.RESULT_WORDING}
    return C.validate_plan(value, protocol_sha)


def summarize(root: Path, paths: list[Path], plan: dict) -> dict:
    C.validate_plan(plan, C.V.file_digest(C.PROTOCOL))
    owner = C.session(root)
    C.require(len(paths) == len(plan["planned_calls"]) == 6, "여섯 고정 대화 필요")
    records = []
    for path, row in zip(paths, plan["planned_calls"], strict=True):
        capsule = C.prepared(root, path)[0]
        C.require(C.planned_state(capsule) == row["state"], "계획에 없는 입력")
        if not owner["fixture"]:
            receipt = C.MC.read(C.inside(root, f"plan_receipts/{row['row_id']}.json"))
            C.require(receipt == {"schema": "selection_call_plan_consumed_v2", "row_digest": C.V.L.digest(row),
                                 "experiment_sha": owner["experiment_sha"], "session_digest": C.V.L.digest(owner),
                                 "capsule_sha256": path.stem}, "계획 소비 receipt 불일치")
        records.append(C.review(root, path, C.inside(root, f"calls/{path.stem}.json")))
    budget = C.audit(root)
    expected_groups = {path.stem for path in paths}
    C.require(set(budget["groups"]) == expected_groups, "계획 밖 또는 누락된 logical call 원장")
    C.require(budget["reserved"] == sum(row["requests"] for row in records),
              "logical call 기록과 공용 요청 원장 합계 불일치")
    fields = ("input_tokens", "output_tokens", "total_tokens", "wall_s")
    costs = {key: (sum(row["costs"][key] for row in records)
                   if all(row["costs"][key] is not None for row in records) else None) for key in fields}
    provider_ok = all(any(call["result"].get("provider_model") in C.ACCEPTED_PROVIDER_MODELS for call in row["calls"])
                      and all(call["result"].get("provider_model") is None
                              or call["result"].get("provider_model") in C.ACCEPTED_PROVIDER_MODELS
                              for call in row["calls"])
                      for row in records)
    cleanup = not C.MC._active_path(root / "api").exists() and not any((root / "workers").glob("*.json"))
    receipt_names = {path.name for path in (root / "plan_receipts").glob("*.json")}
    expected_receipts = ({row["row_id"] + ".json" for row in plan["planned_calls"]}
                         if not owner["fixture"] else set())
    C.require(receipt_names == expected_receipts, "계획 밖 또는 누락된 소비 receipt")
    runtime_ok = (not owner["fixture"] and provider_ok and cleanup
                  and budget["pending"] == 0 and budget["completed"] == budget["reserved"]
                  and budget["ok"] == budget["reserved"] and budget["error"] == 0
                  and budget["timeout"] == 0 and budget["reserved"] <= 18)
    validity = {condition: {
        "planned": sum(c == condition for _name, c in ORDER),
        "first_valid": sum(row["condition"] == condition and bool(row["calls"])
                           and row["calls"][0]["choice"]["model_status"] == "valid" for row in records),
        "final_valid": sum(row["condition"] == condition and row["selection"]["model_status"] == "valid" for row in records)}
        for condition in C.CONDITIONS}
    return {"schema": "selection_runtime_summary_v2", "status": "complete", "runtime_verified": runtime_ok,
            "measurement_kind": "synthetic_fixture" if owner["fixture"] else "registered_real_provider_run",
            "model_contract_prediction_met": all(row["planned"] == row["final_valid"] for row in validity.values()),
            "contract_validity": validity, "logical_calls_planned": 6, "logical_calls_completed": 6,
            "api_requests": budget["reserved"], "api_cap": 20, "worst_planned_requests": 18,
            "api_costs": costs, "owned_worker_cleanup_verified": cleanup,
            "exposure_pairs": [C.exposure_summary(root, paths[1], paths[2]), C.exposure_summary(root, paths[5], paths[4])],
            "new_GPU_runs": 0, "new_training_runs": 0, "new_probe_runs": 0,
            "claims": {"selector_effect": False, "observation_causal_contribution": False,
                       "recovery": False, "generalization": False, "used_for_recovery_decision": False}}


def verify_controller(plan: dict, anchor: str) -> None:
    """호출기 source 봉인에 더해 새 controller·회귀도 commit/index/worktree에 결합한다."""
    expected = plan["predictions"]
    sources = {"evals/selection_comparison_runtime_v2.py": expected["runner_sha256_lf"],
               "tests/test_selection_comparison_runtime_v2.py": expected["tests_sha256_lf"]}
    for path, sha in sources.items():
        C.require(C.V.file_digest(ROOT / path) == sha, "검수한 controller 변경")
        for version in (f"{anchor}:{path}", f"HEAD:{path}", f":{path}"):
            result = C.subprocess.run(["git", "show", version], cwd=ROOT, capture_output=True)
            C.require(result.returncode == 0 and C.raw_sha(result.stdout.replace(b"\r\n", b"\n")) == sha,
                      "controller source commit/index 불일치")


def claim_execution(plan: dict, root: Path) -> Path:
    """같은 계획은 root를 바꾸어도 한 번만 시작한다. 실패한 claim도 보존한다."""
    identity = C.V.L.digest(plan)
    path = CLAIMS / f"{identity}.json"
    C.MC.write_new(path, {"schema": "selection_runtime_execution_claim_v2", "execution_digest": identity,
                          "run_id": plan["run_id"], "bundle_sha256": plan["predictions"]["bundle_sha256"],
                          "root_digest": C.raw_sha(str(Path(root).resolve()).encode("utf-8"))})
    return path


def execute(bundle_path: Path, root: Path, anchor: str, approve: bool) -> dict:
    C.require(approve, "실제 모델 호출 명시 승인 필요")
    plan = C.execution_manifest(PROTOCOL, anchor, C.V.file_digest(C.PROTOCOL))
    verify_controller(plan, anchor)
    expected = plan["predictions"]
    bundle = read_bundle(bundle_path, expected["bundle_sha256"])
    C.require(bundle["archive_integrity_sha256"] == expected["archive_integrity_sha256"], "과거 결합 증거 변경")
    C.initialize(root, RUN_ID, plan["caller_protocol_sha"], fixture=False,
                 experiment_sha=C.MC.file_sha(PROTOCOL), anchor=anchor)
    claim_execution(plan, root)
    C.authorize_controller(root, plan, PROTOCOL, anchor)
    stage, completed = "materialize", 0
    try:
        paths = materialize(root, bundle)
        for path, row in zip(paths, plan["planned_calls"], strict=True):
            C.require(C.planned_state(C.prepared(root, path)[0]) == row["state"], "봉인된 순서·입력 변경")
        for index, path in enumerate(paths, 1):
            stage = f"logical_call_{index}"
            C.launch(root, path, PROTOCOL, anchor)
            completed += 1
        stage = "summarize"
        result = summarize(root, paths, plan)
    except Exception as error:
        result = {"schema": "selection_runtime_summary_v2", "status": "invalid", "runtime_verified": False,
                  "stage": stage, "error_type": type(error).__name__, "logical_calls_planned": 6,
                  "logical_calls_completed": completed, "logical_calls_not_completed": 6 - completed,
                  "api_requests": None, "api_pending": None, "model_contract_prediction_met": None,
                  "api_costs": {key: None for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s")},
                  "api_cost_scope": "unknown_in_incomplete_run_see_preserved_request_receipts",
                  "new_GPU_runs": 0, "new_training_runs": 0, "new_probe_runs": 0}
        try:
            budget = C.audit_budget(root / "api")
            result.update(api_requests=budget["reserved"], api_pending=budget["pending"])
        except Exception as audit_error:
            result["api_audit_error_type"] = type(audit_error).__name__
    C.MC.write_new(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--approve-model-calls", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        C.require(args.output is not None and not args.approve_model_calls and args.anchor is None, "prepare는 CPU 전용")
        if args.output is None:
            raise ValueError("실행 계획 출력 경로 필요")
        C.MC.write_new(args.output, execution_plan(args.bundle, args.root))
        print("CPU 입력·순서 계획 저장")
    else:
        C.require(args.anchor is not None, "실제 실행에는 commit anchor 필요")
        print(json.dumps(execute(args.bundle, args.root, args.anchor, args.approve_model_calls), ensure_ascii=False))
