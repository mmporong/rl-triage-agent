"""D4 v2 공개 exporter의 최소 공개·불변·변조 거부 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
EXPORT = importlib.import_module("export_selection_comparison_v2")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def invalid_fixture(tmp_path: Path) -> Path:
    run = tmp_path / EXPORT.TAG
    freshness = {"schema": "preregistered_seed_freshness_v1", "status": "fresh",
                 "seeds": [2171], "evidence": []}
    write(run / "begin.json", {"tag": EXPORT.TAG, "seed": 2171, "anchor": EXPORT.EXPECTED_ANCHOR,
                                "protocol_sha256_lf": EXPORT.EXPECTED_PROTOCOL_SHA})
    write(run / "seed_freshness.json", freshness)
    result = {"schema": "selection_comparison_result_v2", "tag": EXPORT.TAG, "seed": 2171,
              "status": "invalid", "source_gate": None, "shared": {}, "arms": {},
              "costs": {"shared": {}, "arms": {}}, "failure": {"type": "FixtureFailure"},
              "ledger_audit": {"inventories": {"probe": 0, "correction": 0},
                               "pending": [], "pending_count": 0},
              "preflight": {"seed_freshness_path": str(run / "seed_freshness.json"),
                            "seed_freshness_sha256": EXPORT.sha256(run / "seed_freshness.json")}}
    write(run / "result.json", result)
    return run


def owned_terminal(run: Path, index: int, stage_name: str, stage: Path, *, charged: float | None,
                   terminal: str = "completed") -> None:
    root = run / "owned_audit"
    if not (root / "batch.json").exists():
        write(root / "batch.json", {"schema": EXPORT.EXECUTION.SCHEMA,
              "owned_parent_wall_cap_s": EXPORT.EXECUTION.OWNED_PARENT_WALL_CAP_S,
              "overall_elapsed_cap_s": EXPORT.EXECUTION.OVERALL_ELAPSED_CAP_S,
              "batch_started_clock_s": 1.0})
    stem = f"{index:04d}_{stage_name}"
    reservation_path = root / f"{stem}.reservation.json"
    write(reservation_path, {"schema": EXPORT.EXECUTION.SCHEMA, "index": index,
          "stage": stage_name, "folder": str(stage / "owned"), "cwd": str(run), "args": ["fixture"],
          "requested_watchdog_s": 10.0, "effective_watchdog_s": 10.0,
          "consumed_owned_parent_wall_s_before": 0.0,
          "remaining_before": {"owned_parent_wall_s": 100.0, "overall_elapsed_s": 100.0}})
    row = {"schema": EXPORT.EXECUTION.SCHEMA, "index": index, "stage": stage_name,
           "reservation_sha256": EXPORT.sha256(reservation_path), "status": terminal,
           "wrapper_observed_wall_s": charged or 0.0, "charged_owned_wall_s": charged}
    if terminal == "completed":
        execution = root / f"{stem}.execution.json"
        write(execution, {"wall_s": charged})
        row["execution_sha256"] = EXPORT.sha256(execution)
    else:
        row["reason"] = "runner_exception"
    write(root / f"{stem}.{terminal}.json", row)
    result_path = run / "result.json"
    if result_path.is_file():
        result = EXPORT.read(result_path)
        known = sum(EXPORT.read(path).get("charged_owned_wall_s") or 0.0
                    for path in root.glob("*.completed.json"))
        known += sum(EXPORT.read(path).get("charged_owned_wall_s") or 0.0
                     for path in root.glob("*.failed.json"))
        result["owned_budget"] = {"consumed_owned_parent_wall_s": known,
                                  "remaining": {"owned_parent_wall_s": 100.0,
                                                "overall_elapsed_s": 100.0}}
        write(result_path, result)


def test_partial_invalid_is_public_with_unknown_boundary_and_no_private_paths(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    result_before = (run / "result.json").read_bytes()
    protocol_before = EXPORT.PROTOCOL.read_bytes()

    payloads = EXPORT.verify_and_build(run)

    assert set(payloads) == EXPORT.PUBLIC_FILES
    assert payloads["summary.json"]["status"] == "invalid"
    assert payloads["summary.json"]["paired_case"]["independent_case_count"] == 1
    assert payloads["model_evidence.json"]["status"] == "not_started"
    assert payloads["model_evidence.json"]["verified_request_count"] == 0
    assert payloads["model_evidence.json"]["usage_known_subtotal"] == {
        "input_tokens": None, "output_tokens": None, "total_tokens": None}
    assert payloads["model_evidence.json"]["usage_unknown_request_count"] == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert payloads["costs.json"]["overall_s"] == 0
    public = b"".join(EXPORT.serialized(value) for value in payloads.values())
    assert b"bench/private" not in public and str(ROOT).encode() not in public
    assert (run / "result.json").read_bytes() == result_before
    assert EXPORT.PROTOCOL.read_bytes() == protocol_before


def usage_fixture(tmp_path: Path) -> tuple[Path, dict]:
    run = tmp_path / "run"
    plan = {"fixture": "plan"}
    write(run / "model/plan.json", plan)
    return run, {"api": {"slot_plan": plan}}


def test_verified_usage_preserves_known_subtotal_across_timeout(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, protocol = usage_fixture(tmp_path)
    rows = [
        {"response": {"status": "timeout", "result": {
            "error_type": "APITimeoutError", "wall_s": 300.84}}},
        {"response": {"status": "ok", "result": {
            "input_tokens": 2500, "output_tokens": 226,
            "total_tokens": 2726, "wall_s": 10.0}}},
    ]
    monkeypatch.setattr(EXPORT.API, "validate_plan", lambda value: value)
    monkeypatch.setattr(EXPORT.API, "audit_api", lambda *args: {"reserved": 2, "rows": rows})

    usage = EXPORT._verified_usage(run, protocol)
    model_evidence = {"costs": {"total_tokens": None}}
    model_evidence.update(usage)

    assert model_evidence["costs"]["total_tokens"] is None
    assert usage == {
        "verified_request_count": 2,
        "usage_known_subtotal": {
            "input_tokens": 2500, "output_tokens": 226, "total_tokens": 2726},
        "usage_unknown_request_count": {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 1},
    }


def test_verified_usage_pending_request_is_unknown_not_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, protocol = usage_fixture(tmp_path)
    monkeypatch.setattr(EXPORT.API, "validate_plan", lambda value: value)
    monkeypatch.setattr(EXPORT.API, "audit_api", lambda *args: {
        "reserved": 1, "rows": [{"response": None}]})

    usage = EXPORT._verified_usage(run, protocol)

    assert usage["usage_known_subtotal"] == {
        "input_tokens": None, "output_tokens": None, "total_tokens": None}
    assert usage["usage_unknown_request_count"] == {
        "input_tokens": 1, "output_tokens": 1, "total_tokens": 1}


def test_verified_usage_fully_known_sums_each_field(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, protocol = usage_fixture(tmp_path)
    rows = [
        {"response": {"result": {
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14}}},
        {"response": {"result": {
            "input_tokens": 20, "output_tokens": 6, "total_tokens": 26}}},
    ]
    monkeypatch.setattr(EXPORT.API, "validate_plan", lambda value: value)
    monkeypatch.setattr(EXPORT.API, "audit_api", lambda *args: {"reserved": 2, "rows": rows})

    usage = EXPORT._verified_usage(run, protocol)

    assert usage["usage_known_subtotal"] == {
        "input_tokens": 30, "output_tokens": 10, "total_tokens": 40}
    assert usage["usage_unknown_request_count"] == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_v2_projection_and_isolated_v1_globals(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    before = (EXPORT.BASE.TAG, EXPORT.BASE.RUN, EXPORT.BASE.PROTOCOL, EXPORT.BASE.FLOW)

    payloads = EXPORT.verify_and_build(run)

    assert all(value["schema"].endswith("_v2") for value in payloads.values())
    assert payloads["summary.json"]["seed"] == 2171
    assert payloads["provenance.json"]["source_result"]["schema"] == "selection_comparison_result_v2"
    assert payloads["provenance.json"]["source_protocol"] == {
        "schema": "selection_comparison_demo_protocol_v2",
        "sha256_lf": EXPORT.EXPECTED_PROTOCOL_SHA,
    }
    assert before == (EXPORT.BASE.TAG, EXPORT.BASE.RUN, EXPORT.BASE.PROTOCOL, EXPORT.BASE.FLOW)


def test_configured_base_uses_v2_flow_without_mutating_shared_base(tmp_path: Path) -> None:
    before = EXPORT.BASE.FLOW
    configured = EXPORT._configured_base(tmp_path / "run", EXPORT.PROTOCOL)
    assert configured.FLOW is EXPORT.FLOW
    assert configured.TAG == EXPORT.TAG and configured.EXPECTED_ANCHOR == EXPORT.EXPECTED_ANCHOR
    assert EXPORT.BASE.FLOW is before


def test_v2_protocol_tampering_is_rejected_before_projection(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    protocol["infrastructure_retry"]["prior_seed"] = 9999
    changed = tmp_path / "protocol.json"
    write(changed, protocol)
    with pytest.raises(ValueError, match="protocol SHA"):
        EXPORT.verify_and_build(run, changed)


@pytest.mark.parametrize("field,value", [("schema", "other"), ("tag", "other"),
                                          ("seed", 2172), ("status", "success")])
def test_result_identity_tampering_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    run = invalid_fixture(tmp_path)
    result = EXPORT.read(run / "result.json")
    result[field] = value
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="result identity"):
        EXPORT.verify_and_build(run)


def test_freshness_path_escape_is_rejected(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    write(outside, {"status": "fresh", "seeds": [2171]})
    result = EXPORT.read(run / "result.json")
    result["preflight"] = {"seed_freshness_path": str(outside),
                           "seed_freshness_sha256": EXPORT.sha256(outside)}
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="실행 폴더 밖"):
        EXPORT.verify_and_build(run)


def test_cost_unknown_is_not_zero_filled(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    result = EXPORT.read(run / "result.json")
    result["costs"] = {"shared": {"source": {"training": {"NONE": None}}},
                       "arms": {}}
    write(run / "result.json", result)
    payloads = EXPORT.verify_and_build(run)
    assert payloads["costs.json"]["result_known_subtotal_s"] == 0
    assert payloads["costs.json"]["overall_s"] is None
    assert payloads["costs.json"]["classified_has_unknown"] is True


def test_failed_owned_cost_is_partial_and_unknown_not_zero(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    owned_terminal(run, 1, "train.none", run / "shared/training/none", charged=2.5)
    owned_terminal(run, 2, "train.compound", run / "shared/training/compound",
                   charged=None, terminal="failed")
    payloads = EXPORT.verify_and_build(run)
    owned = payloads["costs.json"]["owned_audit"]
    assert owned == {"known_subtotal_s": 2.5, "total_s": None,
                     "unknown_stage_count": 1, "terminal_stage_count": 2}
    assert payloads["costs.json"]["overall_s"] is None


def test_duplicate_owned_terminal_is_rejected(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    stage = run / "shared/training/none"
    owned_terminal(run, 1, "train.none", stage, charged=1.0)
    completed = run / "owned_audit/0001_train.none.completed.json"
    failed = EXPORT.read(completed)
    failed["status"] = "failed"
    failed["reason"] = "forged_duplicate"
    write(run / "owned_audit/0001_train.none.failed.json", failed)
    with pytest.raises(ValueError, match="terminal 상태"):
        EXPORT.verify_and_build(run)


def test_noninvalid_or_nonnull_gate_requires_source_storage(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    result = EXPORT.read(run / "result.json")
    result["source_gate"] = {"none_walks": True, "outcome": "not_recovered"}
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="source evidence 누락"):
        EXPORT.verify_and_build(run)


@pytest.mark.parametrize("status", ["HOLD", "baseline_invalid", "source_failure_absent"])
def test_noninvalid_terminal_cannot_omit_model_root(tmp_path: Path, status: str) -> None:
    run = invalid_fixture(tmp_path)
    result = EXPORT.read(run / "result.json")
    result["status"] = status
    write(run / "result.json", result)
    with pytest.raises(ValueError, match="model 폴더 누락"):
        EXPORT._model_evidence(run, EXPORT.read(EXPORT.PROTOCOL), result)


def test_source_gate_is_recomputed_from_both_fixed_reports(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = (tmp_path / "run").resolve()
    source = {"artifacts": {}, "reports": {}}
    for arm in ("NONE", EXPORT.FLOW.SOURCE_ARM):
        artifact_path = run / "shared" / f"artifact_{arm.lower()}.json"
        report_path = run / "shared" / f"report_{arm.lower()}.json"
        write(artifact_path, {"arm": arm})
        write(report_path, {"arm": arm})
        source["artifacts"][arm] = {"path": str(artifact_path), "sha256": EXPORT.sha256(artifact_path)}
        source["reports"][arm] = {"path": str(report_path), "sha256": EXPORT.sha256(report_path)}
    result = {"status": "HOLD", "shared": {"source": source},
              "source_gate": {"outcome": "not_recovered", "marker": "both", "none_walks": True}}
    monkeypatch.setattr(EXPORT.CONTROL, "validate_artifact", lambda value, **kwargs: value)
    monkeypatch.setattr(EXPORT.CONTROL, "fixed_report", lambda path, artifact: {"arm": artifact["arm"]})
    monkeypatch.setattr(EXPORT.FLOW.RESIDUAL, "recovery",
                        lambda case, none, bands: {"outcome": "not_recovered", "marker": "both"})
    monkeypatch.setattr(EXPORT.FLOW.RESIDUAL.D1.V.R1.U, "walks", lambda report: True)
    _hashes, artifacts = EXPORT._source_evidence(run, result)
    assert set(artifacts) == {"NONE", EXPORT.FLOW.SOURCE_ARM}
    result["source_gate"]["outcome"] = "recovered"
    with pytest.raises(ValueError, match="gate 재계산"):
        EXPORT._source_evidence(run, result)


def test_write_verify_and_overwrite_rejection(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    output, audit = tmp_path / "public", tmp_path / "audit"
    EXPORT.write_new(payloads, output, audit)
    assert set(EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)) == EXPORT.PUBLIC_FILES
    with pytest.raises(FileExistsError):
        EXPORT.write_new(payloads, output, audit)
    with (output / "summary.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact 불일치"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)


def sealed_fixed_stage(run: Path, name: str = "stage", wall_s: float = 1.0) -> Path:
    stage = run / name
    stage.mkdir(parents=True)
    receipt = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
               "error_type": None, "exit_code": 0, "cleanup_verified": True,
               "owned_process_identities": {}, "wall_s": wall_s}
    write(stage / "execution.json", receipt)
    clear = {"busy": False, "busy_reasons": [],
             "admission_policy": EXPORT.EXECUTION.GPU_BACKEND.GPU_PROFILE,
           "gpu_exclusive": False, "performance_measurement": False, "utilization_percent": 0,
           "memory_used_mib": 0, "compute_process_count": 0, "processes": [], "active_engines": []}
    gpu = {**clear,
           "bounded_resample": {"attempted": False, "accepted": True, "reason": "initial_clear",
                                "interval_s": 2, "max_samples": 3, "sample_count": 1,
                                "required_consecutive_clear": 2, "identities": []}, "samples": [clear]}
    write(stage / "gpu_before.json", gpu)
    write(stage / "gpu_after.json", gpu)
    seal = EXPORT.EXECUTION._build_stage_seal(stage, "fixed_evaluate", {"execution": receipt})
    write(stage / "stage_seal.json", seal)
    return stage


def test_stage_seal_recalculation_rejects_receipt_tampering(tmp_path: Path) -> None:
    stage = sealed_fixed_stage(tmp_path / "run")
    assert EXPORT._stage_evidence(tmp_path / "run")
    receipt = EXPORT.read(stage / "execution.json")
    receipt["wall_s"] = 2.0
    write(stage / "execution.json", receipt)
    with pytest.raises(ValueError, match="stage seal"):
        EXPORT._stage_evidence(tmp_path / "run")


def test_invalid_after_completed_stage_preserves_sealed_and_owned_costs(tmp_path: Path) -> None:
    run = invalid_fixture(tmp_path)
    stage = sealed_fixed_stage(run, "completed_stage", 1.0)
    owned_terminal(run, 1, "fixed_evaluate.completed_stage", stage, charged=1.1)
    owned_terminal(run, 2, "train.next_failed", run / "next_failed", charged=None, terminal="failed")

    payloads = EXPORT.verify_and_build(run)

    costs = payloads["costs.json"]
    assert costs["classified_cost_status"] == "incomplete"
    assert costs["result_classified_subtotal_s"] is None
    assert costs["stage_seal_subtotal_s"] == 1.0
    assert costs["owned_audit"]["known_subtotal_s"] == 1.1
    assert costs["overall_s"] is None


def reference_storage(run: Path) -> tuple[dict, dict]:
    none_sha = "a" * 64
    source = {"NONE": {"checkpoint": {"sha256": none_sha}}}
    references = {}
    expected = sorted(set(EXPORT.CONTROL.L.PROBES) | {EXPORT.CONTROL.RC.NEW_PROBE})
    for probe in expected:
        path = run / "shared/evidence" / f"{probe}.json"
        if probe == EXPORT.CONTROL.RC.NEW_PROBE:
            value = {"schema": "selection_actual_probe_evidence_v1", "backend_kind": "optimizer",
                     "probe": probe, "source_checkpoint_sha256": none_sha,
                     "stage": str(run / "shared/reference_optimizer")}
        else:
            value = {"schema": "selection_actual_probe_evidence_v1", "backend_kind": "legacy",
                     "probe": probe, "stage": str(run / "shared/reference_legacy"),
                     "artifact": {"arm": "NONE", "policy": "shared_reference",
                                  "checkpoint": {"sha256": none_sha}}}
        write(path, value)
        references[probe] = str(path)
    return {"status": "complete", "shared": {"reference_evidence": references}}, source


def test_completed_controller_reference_storage_is_exact_seven_and_none_bound(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = (tmp_path / "run").resolve()
    result, source = reference_storage(run)
    monkeypatch.setattr(EXPORT.CONTROL, "validate_actual_probe_evidence", lambda value: value)
    hashes = EXPORT._reference_evidence(run, result, source)
    assert len(hashes) == 7

    missing = EXPORT.read(run / "shared/evidence/P_reward.json")
    del result["shared"]["reference_evidence"]["P_reward"]
    with pytest.raises(ValueError, match="exact 7"):
        EXPORT._reference_evidence(run, result, source)
    result["shared"]["reference_evidence"]["P_reward"] = str(run / "shared/evidence/P_reward.json")
    missing["artifact"]["checkpoint"]["sha256"] = "b" * 64
    write(run / "shared/evidence/P_reward.json", missing)
    with pytest.raises(ValueError, match="NONE checkpoint"):
        EXPORT._reference_evidence(run, result, source)


def model_storage(run: Path, result: dict) -> tuple[dict, dict, dict]:
    root = run / "model"
    plan = {"plan_sha256": "p", "slots": [{"slot_id": "model_adaptive:e1:open0",
             "policy": "model_adaptive", "epoch": 1, "condition": "production", "ordinal": 1}]}
    receipt = {"slot_id": "model_adaptive:e1:open0", "disposition": "executed", "ordinal": 1}
    write(root / "plan.json", plan)
    write(root / "slots/01.json", receipt)
    write(root / "workers/01.transport.json", {"slot_id": receipt["slot_id"], "status": "complete",
                                                "cleanup": "natural_exit_pid_released"})
    recomputed = {"schema": "selection_conditional_finalize_v1", "status": "complete",
                  "slots_closed": 20, "executed": 1,
                  "dispositions": {"executed": 1},
                  "api": {"reserved": 1, "completed": 1, "pending": 0,
                          "ok": 1, "error": 0, "timeout": 0},
                  "costs": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "wall_s": 0.1}}
    stored = {**recomputed, "controller_status": result["status"],
              "policies_recorded": sorted(result["arms"])}
    write(root / "finalize.json", stored)
    result["model_audit"] = stored
    dialogue = {"engine": {"requests": 1, "used_for_decision": True,
                            "selection": {"action": "continue", "probe": "P_reward",
                                          "model_status": "valid", "selected_by": "model"}}}
    return plan, receipt, {"recomputed": recomputed, "dialogue": dialogue}


def test_completed_model_storage_exact_finalize_transport_and_selection_authority(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = (tmp_path / "run").resolve()
    result = {"status": "complete", "arms": {name: {} for name in EXPORT.CONTROL.POLICIES}}
    plan, receipt, fixture = model_storage(run, result)
    monkeypatch.setattr(EXPORT.API, "validate_plan", lambda value: value)
    monkeypatch.setattr(EXPORT.API, "audit_api", lambda *args: {
        "reserved": 1, "completed": 1, "pending": 0, "ok": 1, "error": 0, "timeout": 0})
    monkeypatch.setattr(EXPORT.API, "_load_receipt", lambda *args: (plan["slots"][0], receipt))
    monkeypatch.setattr(EXPORT.API, "review_dialogue", lambda *args: fixture["dialogue"])
    monkeypatch.setattr(EXPORT.API, "finalize", lambda *args: fixture["recomputed"])

    public, _hashes = EXPORT._model_evidence(run, {"api": {"slot_plan": plan}}, result)
    assert public["decisions"][0]["model_status"] == "valid"
    assert public["decisions"][0]["selected_by"] == "model"

    stored = EXPORT.read(run / "model/finalize.json")
    stored["controller_status"] = "invalid"
    write(run / "model/finalize.json", stored)
    with pytest.raises(ValueError, match="finalize exact"):
        EXPORT._model_evidence(run, {"api": {"slot_plan": plan}}, result)


def test_auxiliary_selection_keeps_selected_by_null_and_decision_false(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = (tmp_path / "run").resolve()
    result = {"status": "complete", "arms": {name: {} for name in EXPORT.CONTROL.POLICIES}}
    plan, receipt, fixture = model_storage(run, result)
    slot = plan["slots"][0]
    slot.update(slot_id="model_adaptive:e1:exposure_full", condition="exposure_full")
    receipt["slot_id"] = slot["slot_id"]
    write(run / "model/plan.json", plan)
    write(run / "model/slots/01.json", receipt)
    write(run / "model/workers/01.transport.json", {"slot_id": receipt["slot_id"],
          "status": "complete", "cleanup": "natural_exit_pid_released"})
    fixture["dialogue"]["engine"]["used_for_decision"] = False
    fixture["dialogue"]["engine"]["selection"].pop("selected_by")
    monkeypatch.setattr(EXPORT.API, "validate_plan", lambda value: value)
    monkeypatch.setattr(EXPORT.API, "audit_api", lambda *args: {
        "reserved": 1, "completed": 1, "pending": 0, "ok": 1, "error": 0, "timeout": 0})
    monkeypatch.setattr(EXPORT.API, "_load_receipt", lambda *args: (slot, receipt))
    monkeypatch.setattr(EXPORT.API, "review_dialogue", lambda *args: fixture["dialogue"])
    monkeypatch.setattr(EXPORT.API, "finalize", lambda *args: fixture["recomputed"])

    public, _hashes = EXPORT._model_evidence(run, {"api": {"slot_plan": plan}}, result)

    assert public["decisions"][0]["selected_by"] is None
    assert public["decisions"][0]["used_for_decision"] is False

    slot["condition"] = "production"
    write(run / "model/plan.json", plan)
    with pytest.raises(ValueError, match="production model 선택 주체"):
        EXPORT._model_evidence(run, {"api": {"slot_plan": plan}}, result)
