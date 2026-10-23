"""D4 선택 정책 비교 실행을 재검산해 최소 공개 JSON으로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

API = importlib.import_module("selection_comparison_conditional_call_v1")
CONTROL = importlib.import_module("selection_comparison_controller_v1")
EXECUTION = importlib.import_module("selection_comparison_execution_v1")
FLOW = importlib.import_module("selection_comparison_flow_v1")

TAG = "selection_comparison_s2170_20261023"
RUN = (ROOT / "bench/private/claude_reward_track_20261013/selection_comparison/runs" / TAG)
PROTOCOL = ROOT / "bench/protocols/selection_comparison_demo_v1.json"
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261023/d4/export/audit")
EXPECTED_ANCHOR = "5a469549b07d8f4ab14c208725feccbc1b377757"
EXPECTED_PROTOCOL_SHA = "bad02f9ff9f131388a4b7f22d106c7fa6664a4a04a51a9dfc350244459a1c5c2"
PUBLIC_FILES = {"summary.json", "policies.json", "model_evidence.json", "costs.json", "provenance.json"}
STATUSES = {"complete", "source_failure_absent", "baseline_invalid", "HOLD", "invalid"}


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serialized(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256(path: Path, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def _hash_manifest(paths: list[Path]) -> str:
    return hashlib.sha256(serialized([sha256(path) for path in sorted(paths)])).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _inside(run: Path, value: str | Path) -> Path:
    root, path = Path(run).resolve(), Path(value).resolve()
    require(path == root or root in path.parents, "실행 폴더 밖 artifact 참조")
    require(not path.is_symlink(), "symlink artifact 거부")
    return path


def _number(value: Any) -> float | None:
    if value is None:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "비용 숫자 형식")
    result = float(value)
    require(math.isfinite(result) and result >= 0, "비용 숫자 범위")
    return result


def _cost_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _cost_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_cost_tree(item) for item in value]
    return _number(value)


def _flatten_costs(value: Any) -> list[float | None]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten_costs(child)]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_costs(child)]
    return [_number(value)]


def _identity(run: Path, protocol_path: Path) -> tuple[dict, dict, dict]:
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    result, protocol, begin = read(run / "result.json"), read(protocol_path), read(run / "begin.json")
    require(sha256(protocol_path, True) == EXPECTED_PROTOCOL_SHA, "D4 protocol SHA 불일치")
    require(protocol.get("schema") == "selection_comparison_demo_protocol_v1"
            and protocol.get("seed") == 2170
            and protocol.get("api", {}).get("http_cap") == 60
            and protocol.get("api", {}).get("slot_plan", {}).get("slots") is not None,
            "D4 protocol identity 불일치")
    require(result.get("schema") == "selection_comparison_result_v1"
            and result.get("tag") == TAG and result.get("seed") == 2170
            and result.get("status") in STATUSES, "D4 result identity 불일치")
    require(begin == {"tag": TAG, "seed": 2170, "anchor": EXPECTED_ANCHOR,
                      "protocol_sha256_lf": EXPECTED_PROTOCOL_SHA}, "D4 begin 결합 불일치")
    freshness_path = _inside(run, result.get("preflight", {}).get("seed_freshness_path", ""))
    require(freshness_path == run / "seed_freshness.json"
            and sha256(freshness_path) == result["preflight"].get("seed_freshness_sha256"),
            "seed freshness 결합 불일치")
    freshness = read(freshness_path)
    require(freshness.get("status") == "fresh" and freshness.get("seeds") == [2170],
            "seed freshness 내용 불일치")
    return result, protocol, begin


def _stage_evidence(run: Path) -> tuple[dict[str, str], dict[Path, float]]:
    hashes: dict[str, str] = {}
    costs: dict[Path, float] = {}
    for index, path in enumerate(sorted(run.rglob("stage_seal.json")), 1):
        stage = path.parent
        seal = read(path)
        kind = seal.get("kind")
        require(kind in {"train", "inspect", "fixed_evaluate", "legacy_probe"}, "stage seal kind")
        raw = {}
        for name in EXECUTION._execution_names(kind):
            raw[name.removesuffix(".json")] = read(stage / name)
        EXECUTION.verify_stage_seal(stage, kind, raw, seal)
        costs[stage.resolve()] = FLOW._wall(raw)
        hashes[f"stage_seal_{index:02d}_{kind}"] = sha256(path)
    return hashes, costs


def _owned_audit(run: Path, stage_costs: dict[Path, float]) -> tuple[dict, dict[str, str]]:
    root = run / "owned_audit"
    if not root.is_dir():
        require(not stage_costs, "stage seal은 있으나 owned audit이 없다")
        return {"known_subtotal_s": 0.0, "total_s": 0.0, "unknown_stage_count": 0,
                "terminal_stage_count": 0}, {}
    batch_path = root / "batch.json"
    batch = read(batch_path)
    require(batch.get("schema") == EXECUTION.SCHEMA
            and _number(batch.get("owned_parent_wall_cap_s")) == EXECUTION.OWNED_PARENT_WALL_CAP_S
            and _number(batch.get("overall_elapsed_cap_s")) == EXECUTION.OVERALL_ELAPSED_CAP_S,
            "owned batch 결합")
    terminals: dict[Path, list[float | None]] = {}
    hashes = {"owned_batch": sha256(batch_path)}
    groups: dict[tuple[int, str], dict[str, Path]] = {}
    for path in root.iterdir():
        if path.name in {"batch.json", "batch_failure.json"}:
            continue
        match = EXECUTION._AUDIT.fullmatch(path.name)
        if not path.is_file() or path.is_symlink() or match is None:
            raise ValueError("owned audit inventory")
        groups.setdefault((int(match["index"]), match["stage"]), {})[match["kind"]] = path
    require([index for index, _stage in sorted(groups)] == list(range(1, len(groups) + 1)),
            "owned audit sequence")
    known = 0.0
    unknown = 0
    for ordinal, ((index, stage_name), paths) in enumerate(sorted(groups.items()), 1):
        require("reservation" in paths and len({"completed", "failed"} & paths.keys()) == 1,
                "owned audit terminal 상태")
        terminal_kind = "completed" if "completed" in paths else "failed"
        reservation, terminal = read(paths["reservation"]), read(paths[terminal_kind])
        require(reservation.get("schema") == EXECUTION.SCHEMA and reservation.get("index") == index
                and reservation.get("stage") == stage_name and terminal.get("schema") == EXECUTION.SCHEMA
                and terminal.get("index") == index and terminal.get("stage") == stage_name
                and terminal.get("status") == terminal_kind
                and terminal.get("reservation_sha256") == sha256(paths["reservation"]),
                "owned audit reservation·terminal 결합")
        if "execution" in paths:
            require(terminal.get("execution_sha256") == sha256(paths["execution"]),
                    "owned execution SHA 결합")
        else:
            require(terminal_kind == "failed" and "execution_sha256" not in terminal,
                    "owned execution 누락 경계")
        folder = _inside(run, reservation.get("folder", ""))
        stage_path = folder.parent.resolve()
        charged = _number(terminal.get("charged_owned_wall_s"))
        terminals.setdefault(stage_path, []).append(charged)
        if charged is None:
            require(terminal_kind == "failed", "완료 stage 비용 null")
            unknown += 1
        else:
            known += charged
        hashes[f"owned_terminal_{ordinal:02d}_{terminal_kind}"] = sha256(paths[terminal_kind])
    for stage, classified in stage_costs.items():
        charges = terminals.get(stage)
        if charges is None or any(value is None for value in charges):
            raise ValueError("stage 비용과 owned audit 중복관계")
        charged = sum(value for value in charges if value is not None)
        if charged + 1e-9 < classified:
            raise ValueError("stage 비용과 owned audit 중복관계")
    require(set(stage_costs) <= set(terminals), "분류 stage와 owned audit 결합")
    failure_path = root / "batch_failure.json"
    if failure_path.is_file():
        failure = read(failure_path)
        require(failure.get("schema") == EXECUTION.SCHEMA and isinstance(failure.get("reason"), str)
                and _number(failure.get("consumed_owned_parent_wall_s")) == known,
                "owned batch failure 결합")
        hashes["owned_batch_failure"] = sha256(failure_path)
    result = read(run / "result.json")
    budget = result.get("owned_budget")
    if budget is None:
        require(result.get("status") == "invalid" and not groups,
                "result owned budget 누락")
    else:
        require(isinstance(budget, dict)
                and _number(budget.get("consumed_owned_parent_wall_s")) == known,
                "result owned budget 소비 불일치")
    return {"known_subtotal_s": known, "total_s": None if unknown else known,
            "unknown_stage_count": unknown, "terminal_stage_count": sum(map(len, terminals.values()))}, hashes


def _source_evidence(run: Path, result: dict) -> tuple[dict[str, str], dict[str, dict]]:
    source = result.get("shared", {}).get("source")
    if source is None:
        require(result.get("status") == "invalid" and result.get("source_gate") is None,
                "source gate가 있는 결과의 source evidence 누락")
        return {}, {}
    require(isinstance(source, dict), "공유 source 형식")
    hashes: dict[str, str] = {}
    artifacts = {}
    for arm in ("NONE", FLOW.SOURCE_ARM):
        artifact_ref = source.get("artifacts", {}).get(arm, {})
        artifact_path = _inside(run, artifact_ref.get("path", ""))
        require(sha256(artifact_path) == artifact_ref.get("sha256"), "source artifact SHA")
        artifact = read(artifact_path)
        expected_policy = "shared_reference" if arm == "NONE" else CONTROL.POLICIES[0]
        CONTROL.validate_artifact(artifact, arm="NONE" if arm == "NONE" else "shared_compound",
                                  policy=expected_policy, epoch=0, corrections=[],
                                  source_checkpoint_sha256=None,
                                  expected_kind=CONTROL.ACTUAL_EVIDENCE)
        artifacts[arm] = artifact
        report_ref = source.get("reports", {}).get(arm, {})
        report_path = _inside(run, report_ref.get("path", ""))
        require(sha256(report_path) == report_ref.get("sha256"), "source fixed report SHA")
        CONTROL.fixed_report(report_path, artifact)
        hashes[f"source_artifact_{arm.lower()}"] = sha256(artifact_path)
        hashes[f"source_fixed_evaluation_{arm.lower()}"] = sha256(report_path)
    bands = read(ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json")["bands"]
    reports = {arm: CONTROL.fixed_report(_inside(run, source["reports"][arm]["path"]), artifacts[arm])
               for arm in ("NONE", FLOW.SOURCE_ARM)}
    expected_gate = {**FLOW.RESIDUAL.recovery(reports[FLOW.SOURCE_ARM], reports["NONE"], bands),
                     "none_walks": FLOW.RESIDUAL.D1.V.R1.U.walks(reports["NONE"])}
    require(result.get("source_gate") == expected_gate, "source gate 재계산 불일치")
    return hashes, artifacts


def _reference_evidence(run: Path, result: dict, source_artifacts: dict[str, dict]) -> dict[str, str]:
    references = result.get("shared", {}).get("reference_evidence")
    if references is None:
        require(result.get("status") != "complete", "완료 결과의 shared reference 누락")
        return {}
    expected = set(CONTROL.L.PROBES) | {CONTROL.RC.NEW_PROBE}
    require(isinstance(references, dict) and set(references) == expected
            and "NONE" in source_artifacts, "shared reference exact 7 inventory")
    none_sha = source_artifacts["NONE"]["checkpoint"]["sha256"]
    hashes = {}
    for probe in sorted(expected):
        path = _inside(run, references[probe])
        require(path == run / "shared" / "evidence" / f"{probe}.json", "shared reference 위치")
        value = read(path)
        CONTROL.validate_actual_probe_evidence(value)
        require(value.get("probe") == probe, "shared reference probe identity")
        if probe == CONTROL.RC.NEW_PROBE:
            require(value.get("backend_kind") == "optimizer"
                    and value.get("source_checkpoint_sha256") == none_sha
                    and Path(value.get("stage", "")).resolve() == run / "shared/reference_optimizer",
                    "optimizer reference NONE checkpoint·stage 결합")
        else:
            artifact = value.get("artifact", {})
            require(value.get("backend_kind") == "legacy"
                    and artifact.get("checkpoint", {}).get("sha256") == none_sha
                    and artifact.get("arm") == "NONE" and artifact.get("policy") == "shared_reference"
                    and Path(value.get("stage", "")).resolve() == run / "shared/reference_legacy",
                    "legacy reference NONE checkpoint·stage 결합")
        hashes[f"shared_reference_{probe.lower()}"] = sha256(path)
    return hashes


def _ledger_and_policy_evidence(run: Path, result: dict) -> tuple[dict[str, dict], dict[str, str]]:
    bands = read(ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json")["bands"]
    public: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    arms = result.get("arms", {})
    require(isinstance(arms, dict) and set(arms) <= set(CONTROL.POLICIES), "정책 arm inventory")
    for policy in CONTROL.POLICIES:
        arm = arms.get(policy)
        if arm is None:
            public[policy] = {"status": "not_started", "epochs": [], "probe_order": [],
                              "probe_count": 0, "corrections": []}
            continue
        require(isinstance(arm, dict) and arm.get("status") in {
            "running", "HOLD", "recovered", "not_recovered", "unknown"
        }, "정책 결과 status")
        verified_epochs = []
        corrections = []
        for row in arm.get("epochs", []):
            require(set(row) == {"epoch", "outcome", "correction_id", "repair_path"}, "정책 epoch 형식")
            repair_path = _inside(run, row["repair_path"])
            verified = CONTROL.verify_repair(
                repair_path, ledger_path=run / "arms" / policy / "corrections.ledger.jsonl", bands=bands)
            require(row["epoch"] == verified["epoch"] and row["outcome"] == verified["outcome"]
                    and row["correction_id"] == verified["applied_after"][-1], "repair 결과 재계산")
            verified_epochs.append({"epoch": row["epoch"], "outcome": row["outcome"],
                                    "correction_id": row["correction_id"]})
            corrections.append(row["correction_id"])
            hashes[f"{policy}_repair_epoch{row['epoch']}"] = sha256(repair_path)
        probe_rows = []
        for seal_path in sorted((run / "arms" / policy).glob("epoch*/probe_*.json")):
            value = read(seal_path)
            checked = CONTROL.verify_probe(seal_path, value.get("snapshot"))
            probe_rows.append(checked)
            hashes[f"{policy}_probe_{len(probe_rows):02d}"] = sha256(seal_path)
        require([row["probe"] for row in probe_rows] == arm.get("probe_order", []), "probe 순서 재계산")
        public[policy] = {"status": arm["status"], "reason": arm.get("reason"),
                          "epochs": verified_epochs, "probe_order": deepcopy(arm.get("probe_order", [])),
                          "probe_count": len(probe_rows), "probe_outcomes": [row["outcome"] for row in probe_rows],
                          "corrections": corrections}
        for ledger_path in sorted((run / "arms" / policy).glob("epoch*/probes.ledger.jsonl")):
            ledger = CONTROL.S.M.RegistryLedger(ledger_path, CONTROL.RC.registry())
            require(all(row["state"] not in {"pending", "approved", "running"}
                        for row in ledger.requests().values()), "미종결 probe ledger")
            hashes[f"{policy}_probe_ledger_{ledger_path.parent.name}"] = sha256(ledger_path)
        correction_path = run / "arms" / policy / "corrections.ledger.jsonl"
        if correction_path.is_file():
            ledger = CONTROL.D1.CorrectionLedger(correction_path, CONTROL.RC.correction_registry())
            require(all(row["state"] not in {"pending", "approved", "running"}
                        for row in ledger.requests().values()), "미종결 correction ledger")
            hashes[f"{policy}_correction_ledger"] = sha256(correction_path)
    if result.get("status") == "complete":
        require(set(arms) == set(CONTROL.POLICIES)
                and all(public[policy]["status"] not in {"not_started", "running"}
                        for policy in CONTROL.POLICIES), "완료 결과의 정책 terminal 누락")
    audit = FLOW._ledger_audit(run)
    require(audit == result.get("ledger_audit") and audit["pending_count"] == 0, "result ledger audit 불일치")
    return public, hashes


def _model_evidence(run: Path, protocol: dict, result: dict) -> tuple[dict, dict[str, str]]:
    root = run / "model"
    if not root.exists():
        require(result.get("status") == "invalid" and "model_audit" not in result,
                "종결 결과 또는 model audit의 model 폴더 누락")
        return {"status": "not_started", "api": {"reserved": 0, "completed": 0,
                "pending": 0}, "costs": {key: 0 for key in (
                    "input_tokens", "output_tokens", "total_tokens", "wall_s")}, "decisions": [],
                "production_decision_count": 0, "production_valid_count": 0,
                "production_valid_rate": None}, {}
    plan = API.validate_plan(read(root / "plan.json"))
    require(plan == protocol["api"]["slot_plan"], "model slot plan 불일치")
    audit = API.audit_api(root, plan)
    decisions = []
    receipts = []
    for path in sorted((root / "slots").glob("[0-9][0-9].json")):
        receipt = read(path)
        slot, stored = API._load_receipt(root, plan, receipt.get("slot_id", ""))
        require(receipt == stored, "model slot receipt 불일치")
        receipts.append(receipt)
        if receipt["disposition"] == "executed":
            dialogue = API.review_dialogue(root, plan, receipt)
            selection = dialogue["engine"].get("selection", {})
            require(isinstance(selection.get("model_status"), str), "model selection status 누락")
            selected_by = selection.get("selected_by")
            if slot["condition"] == "production":
                require(isinstance(selected_by, str), "production model 선택 주체 누락")
            else:
                require(selected_by is None and dialogue["engine"].get("used_for_decision") is False,
                        "보조 관측 선택 주체·판정 비사용 경계")
            decisions.append({"slot_id": slot["slot_id"], "policy": slot["policy"],
                              "epoch": slot["epoch"], "condition": slot["condition"],
                              "requests": dialogue["engine"]["requests"],
                              "action": selection.get("action"), "probe": selection.get("probe"),
                              "model_status": selection["model_status"], "selected_by": selected_by,
                              "used_for_decision": dialogue["engine"].get("used_for_decision")})
    finalize_path = root / "finalize.json"
    if finalize_path.is_file():
        recomputed = API.finalize(root, plan)
        stored = read(finalize_path)
        expected_stored = {**recomputed, "controller_status": result["status"],
                           "policies_recorded": sorted(result.get("arms", {}))}
        require(stored == expected_stored and result.get("model_audit") == expected_stored,
                "controller model finalize exact 결합")
        transports = [read(path) for path in (root / "workers").glob("*.transport.json")]
        executed_slots = {row["slot_id"] for row in receipts if row["disposition"] == "executed"}
        require({row.get("slot_id") for row in transports} == executed_slots
                and len(transports) == len(executed_slots)
                and all(row.get("status") == "complete"
                        and row.get("cleanup") == "natural_exit_pid_released" for row in transports)
                and not list((root / "workers").glob("*.pid.json")), "model worker transport·cleanup 결합")
        model_status = stored["status"]
        costs = deepcopy(stored["costs"])
        api = deepcopy(stored["api"])
        dispositions = deepcopy(stored.get("dispositions", {}))
        slots_closed = stored.get("slots_closed")
    else:
        require(result.get("status") == "invalid" and "model_audit" not in result,
                "완료 결과의 model finalize 누락")
        model_status, costs = "partial_invalid", {key: None for key in (
            "input_tokens", "output_tokens", "total_tokens", "wall_s")}
        api = {key: audit[key] for key in ("reserved", "completed", "pending", "ok", "error", "timeout")}
        dispositions = {name: sum(row.get("disposition") == name for row in receipts)
                        for name in sorted(API.DISPOSITIONS)}
        slots_closed = len(receipts)
    hashes = {"model_plan": sha256(root / "plan.json")}
    for label, paths in {
        "model_slot_receipts_manifest": list((root / "slots").glob("[0-9][0-9].json")),
        "model_dialogues_manifest": list((root / "dialogues").glob("[0-9][0-9].json")),
        "model_api_reservations_manifest": list((root / "api/reservations").glob("[0-9][0-9][0-9][0-9].json")),
        "model_api_responses_manifest": list((root / "api/responses").glob("[0-9][0-9][0-9][0-9].json")),
    }.items():
        hashes[label] = _hash_manifest(paths)
    if finalize_path.is_file():
        hashes["model_finalize"] = sha256(finalize_path)
    production = [row for row in decisions if row["condition"] == "production"]
    valid = sum(row["model_status"] == "valid" for row in production)
    return {"status": model_status, "slots_closed": slots_closed, "dispositions": dispositions,
            "api": api, "costs": costs, "decisions": decisions,
            "production_decision_count": len(production), "production_valid_count": valid,
            "production_valid_rate": (valid / len(production) if production else None)}, hashes


def _public_safe(payloads: dict[str, dict]) -> None:
    require(set(payloads) == PUBLIC_FILES, "공개 파일 inventory")
    raw = b"".join(serialized(value) for value in payloads.values())
    forbidden = (b"bench/private", b"C:\\Users", b"/home/", Path.home().name.encode(), b"request_id",
                 b"response_id", b"function_call", b"api_key", b"authorization_path")
    require(not any(item.lower() in raw.lower() for item in forbidden), "공개 payload 민감 정보")


def verify_and_build(run: Path = RUN, protocol_path: Path = PROTOCOL) -> dict[str, dict]:
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    before = {path: sha256(path) for path in (run / "result.json", protocol_path)}
    result, protocol, _begin = _identity(run, protocol_path)
    stage_hashes, stage_costs = _stage_evidence(run)
    owned_costs, owned_hashes = _owned_audit(run, stage_costs)
    source_hashes, source_artifacts = _source_evidence(run, result)
    reference_hashes = _reference_evidence(run, result, source_artifacts)
    policies, ledger_hashes = _ledger_and_policy_evidence(run, result)
    model, model_hashes = _model_evidence(run, protocol, result)
    costs = _cost_tree(result.get("costs", {"shared": {}, "arms": {}}))
    values = _flatten_costs(costs)
    known_subtotal = sum(value for value in values if value is not None)
    has_unknown = any(value is None for value in values)
    stage_subtotal = sum(stage_costs.values())
    classified_exact = (not has_unknown
                        and math.isclose(known_subtotal, stage_subtotal, rel_tol=0.0, abs_tol=1e-9))
    if result["status"] != "invalid":
        require(classified_exact, "result 분류 비용과 stage seal 비용 불일치")
    source_gate = result.get("source_gate")
    summary = {
        "schema": "public_selection_comparison_summary_v1", "tag": TAG, "seed": 2170,
        "status": result["status"], "source_gate": deepcopy(source_gate),
        "paired_case": {"faults": ["F1", "R1"], "registered_development_comparison": True,
                        "independent_case_count": 1, "policy_count": 4},
        "policy_status": {policy: policies[policy]["status"] for policy in CONTROL.POLICIES},
        "interpretation": "한 paired seed의 등록 복합 결함 개발 비교이며 네 독립 사례가 아니다.",
    }
    payloads = {
        "summary.json": summary,
        "policies.json": {"schema": "public_selection_comparison_policies_v1", "policies": policies},
        "model_evidence.json": {"schema": "public_selection_comparison_model_evidence_v1", **model,
                                "invalid_decision_policy": "HOLD_without_rule_fallback"},
        "costs.json": {"schema": "public_selection_comparison_costs_v1", "scope": {
            "shared": "source_and_reference_once", "arms": "isolated_policy_owned_costs"},
            "costs": costs, "result_known_subtotal_s": known_subtotal,
            "result_classified_subtotal_s": known_subtotal if classified_exact else None,
            "stage_seal_subtotal_s": stage_subtotal,
            "classified_cost_status": "complete" if classified_exact else "incomplete",
            "classified_has_unknown": has_unknown, "owned_audit": owned_costs,
            "overall_s": None if has_unknown else owned_costs["total_s"],
            "cost_relationship": "owned_total_contains_classified_stage_costs_not_added"},
        "provenance.json": {"schema": "public_selection_comparison_provenance_v1",
            "anchor_commit": EXPECTED_ANCHOR,
            "source_protocol": {"schema": protocol["schema"], "sha256_lf": EXPECTED_PROTOCOL_SHA},
            "source_result": {"schema": result["schema"], "sha256": sha256(run / "result.json")},
            "evidence_sha256": {**stage_hashes, **owned_hashes, **source_hashes,
                                **reference_hashes, **ledger_hashes, **model_hashes},
            "export_contract": "allowlisted_fields_only_no_private_or_user_paths"},
    }
    _public_safe(payloads)
    require(before == {path: sha256(path) for path in before}, "export 검증 중 원본 변경")
    return payloads


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    _public_safe(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in sorted(payloads.items()):
        (output / name).write_bytes(serialized(payload))
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "selection_comparison_public_export_receipt_v1", "tag": TAG,
               "source_result_sha256": payloads["provenance.json"]["source_result"]["sha256"],
               "public_sha256": {name: sha256(output / name) for name in sorted(payloads)}}
    (audit / "receipt.json").write_bytes(serialized(receipt))


def verify_export(output: Path = OUTPUT, run: Path = RUN, protocol_path: Path = PROTOCOL,
                  audit: Path = AUDIT) -> dict[str, str]:
    expected = verify_and_build(run, protocol_path)
    require(output.is_dir() and {path.name for path in output.iterdir()} == PUBLIC_FILES
            and all(path.is_file() and not path.is_symlink() for path in output.iterdir()),
            "공개 output inventory")
    for name, payload in expected.items():
        require((output / name).read_bytes() == serialized(payload), f"공개 artifact 불일치: {name}")
    hashes = {name: sha256(output / name) for name in sorted(expected)}
    receipt = {"schema": "selection_comparison_public_export_receipt_v1", "tag": TAG,
               "source_result_sha256": expected["provenance.json"]["source_result"]["sha256"],
               "public_sha256": hashes}
    require(audit.is_dir() and [path.name for path in audit.iterdir()] == ["receipt.json"]
            and (audit / "receipt.json").read_bytes() == serialized(receipt), "private export receipt 불일치")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        result = {"status": "pass", "public_files": sorted(verify_and_build())}
    elif args.command == "export":
        payloads = verify_and_build()
        write_new(payloads)
        result = {"status": "exported", "public_files": sorted(payloads)}
    else:
        result = {"status": "pass", "public_sha256": verify_export()}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
