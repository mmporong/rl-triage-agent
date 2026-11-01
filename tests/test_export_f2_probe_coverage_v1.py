from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_f2_probe_coverage_v1", ROOT / "evals/export_f2_probe_coverage_v1.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
RUN = M.RUN


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def packet() -> dict:
    bindings = {
        "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
        "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
    }
    return RUN.DRIVER.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=bindings,
        budget_left=3, execution_budget_s=1200)


def fixture(tmp_path: Path, outcomes=("normal", "abnormal"),
            executions=("executed_one_pair", "reused_existing_receipt")):
    protocol_path = write(tmp_path / "overlay.json", {"fixture": True})
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "run" / "result.json"
    sources = {case_id: {"packet": packet()} for case_id in M.TARGETS}
    plan_targets = []
    result_targets = []
    inspected_history = {}
    fresh_audits = {}
    for case_id, outcome, execution in zip(M.TARGETS, outcomes, executions, strict=True):
        historical = None
        action = "execute_one_pair"
        if execution == "reused_existing_receipt":
            detail = write(tmp_path / "history" / case_id / "detail.json", {"fixture": True})
            verified = {"probe": RUN.PROBE, "outcome": outcome,
                        "pair": {"pair_result_sha256": ("a" if case_id == "case-05" else "b") * 64}}
            historical = {"detail": {"path": str(detail.resolve()), "sha256": M.sha(detail)},
                          "step_index": 0, "verified": verified}
            inspected_history[case_id] = verified
            action = "reuse"
        else:
            ledger = result_path.parent / case_id / "oracle.ledger.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text("", encoding="utf-8")
            snapshot = RUN.DRIVER.C.V.build_snapshot(
                sources[case_id]["packet"], f"f2_oracle_{case_id.replace('-', '_')}",
                RUN.HOST_STORAGE_POLICY,
                RUN.DRIVER.C.V.file_digest(RUN.DRIVER.C.V.PROTOCOL), ledger)
            choice = RUN.oracle_choice(snapshot)
            pair_path = write(result_path.parent / case_id / "pair" / "result.json", {
                "public_ledger": {"path": str(ledger.resolve())},
                "public_snapshot_sha256": RUN.B.digest(snapshot),
                "public_choice": choice,
                "fixture_outcome": outcome,
            })
            fresh_audits[case_id] = {
                "probe": RUN.PROBE, "outcome": outcome,
                "pair_result_sha256": M.sha(pair_path),
                "reconstructed": {"sdk_roles_observed": 2,
                                  "known_owned_wall_s": 2.5},
            }
        plan_targets.append({"case_id": case_id, "action": action,
                             "historical": historical,
                             "source_digest": RUN.B.digest(sources[case_id])})
        result_targets.append({
            "case_id": case_id, "oracle_label": "F2", "probe": RUN.PROBE,
            "outcome": outcome, "coverage": M.OUTCOMES[outcome], "execution": execution,
            "control_policy": RUN.ORACLE_POLICY,
            "host_storage_policy": RUN.HOST_STORAGE_POLICY,
            "selected_by": "registered_rule", "selection_performance_credit": 0,
        })
    plan = {"targets": plan_targets, "batch_result": {"sha256": "c" * 64}}
    write(plan_path, plan)
    new_pairs = sum(row == "executed_one_pair" for row in executions)
    result = {
        "schema": RUN.RESULT_SCHEMA, "status": "complete", "targets": result_targets,
        "new_pairs": new_pairs, "sdk_roles_reserved": new_pairs * 2,
        "api_calls": 0, "model_calls": 0, "training": False, "correction": False,
        "original_comparison_modified": False, "selection_performance_credit": 0,
        "parent_wall_s": 5.0, "lease_id": "must-not-export",
    }
    write(result_path, result)
    protocol = {"r3": {"batch_sha256": "c" * 64},
                "oracle_policy": {"name": RUN.ORACLE_POLICY,
                                  "selection_performance_credit": 0}}
    material = {"prepared": {case_id: {"source": source}
                             for case_id, source in sources.items()}}

    def validator(*_args, **_kwargs):
        return protocol, plan, material

    def inspect_step(_detail, source, _index, _temp):
        case_id = next(key for key, value in sources.items() if value is source)
        return deepcopy(inspected_history[case_id])

    def pair_verifier(path, snapshot, choice):
        raw = M.read(path)
        assert raw["public_snapshot_sha256"] == RUN.B.digest(snapshot)
        assert raw["public_choice"] == choice
        return {"probe": RUN.PROBE, "outcome": raw["fixture_outcome"], "retry_count": 0}

    def inspect_pair(path):
        case_id = path.parents[1].name
        return deepcopy(fresh_audits[case_id])

    callbacks = {"validator": validator, "inspect_step": inspect_step,
                 "inspect_pair": inspect_pair, "pair_verifier": pair_verifier,
                 "source_verifier": lambda _anchor: {
                     name: (f"{index:064x}")[-64:]
                     for index, name in enumerate(M.SOURCE_FILES, start=1)}}
    return result_path, protocol_path, plan_path, result, callbacks


def build(tmp_path: Path, **kwargs):
    result_path, protocol_path, plan_path, _result, callbacks = fixture(tmp_path, **kwargs)
    public = M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                            lab=tmp_path, **callbacks)
    return public, (result_path, protocol_path, plan_path, callbacks)


@pytest.mark.parametrize("outcome", ["normal", "abnormal", "unknown"])
def test_exports_all_outcomes_with_sha_only_evidence(tmp_path: Path, outcome: str):
    public, _inputs = build(tmp_path, outcomes=(outcome, outcome))
    assert [row["coverage"] for row in public["targets"]] == [
        M.OUTCOMES[outcome], M.OUTCOMES[outcome]]
    assert public["new_cost"] == {"pairs": 1, "sdk_roles_reserved": 2,
                                  "sdk_roles_observed": 2,
                                  "known_owned_wall_s": 2.5, "api_calls": 0}
    assert public["claims"] == {
        "original_50_row_comparison_modified": False,
        "selection_performance_credit": 0,
        "diagnosis_success_credit": 0,
        "recovery_or_media_credit": 0,
    }
    serialized = json.dumps(public, sort_keys=True)
    assert str(tmp_path) not in serialized and "lease_id" not in serialized.lower()
    assert all(len(row["pair_result_sha256"]) == 64 for row in public["targets"])


def test_reuses_both_historical_receipts_without_new_cost(tmp_path: Path):
    public, _inputs = build(
        tmp_path, executions=("reused_existing_receipt", "reused_existing_receipt"))
    assert public["new_cost"] == {"pairs": 0, "sdk_roles_reserved": 0,
                                  "sdk_roles_observed": 0,
                                  "known_owned_wall_s": 0.0, "api_calls": 0}
    assert all(row["cost"]["reused_prior_receipt"] for row in public["targets"])


def test_rejects_tamper_and_different_case(tmp_path: Path):
    result_path, protocol_path, plan_path, result, callbacks = fixture(tmp_path)
    result["targets"][0]["coverage"] = "confirmed_signature"
    write(result_path, result)
    with pytest.raises(ValueError, match="target summary"):
        M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                       lab=tmp_path, **callbacks)

    other = tmp_path / "other"
    result_path, protocol_path, plan_path, result, callbacks = fixture(other)
    result["targets"][0]["case_id"] = "case-07"
    write(result_path, result)
    with pytest.raises(ValueError, match="complete result"):
        M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                       lab=other, **callbacks)


def test_missing_fresh_pair_receipt_is_rejected(tmp_path: Path):
    result_path, protocol_path, plan_path, _result, callbacks = fixture(tmp_path)
    (result_path.parent / "case-05" / "pair" / "result.json").unlink()
    with pytest.raises(ValueError, match="receipt missing"):
        M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                       lab=tmp_path, **callbacks)


def test_rejects_source_digest_and_original_batch_binding_tamper(tmp_path: Path):
    result_path, protocol_path, plan_path, _result, callbacks = fixture(tmp_path)
    original_validator = callbacks["validator"]

    def changed_source_validator(*args, **kwargs):
        protocol, plan, material = original_validator(*args, **kwargs)
        changed = deepcopy(material)
        changed["prepared"]["case-05"]["source"] = {"packet": packet(), "tampered": True}
        return protocol, plan, changed

    callbacks["validator"] = changed_source_validator
    with pytest.raises(ValueError, match="source digest"):
        M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                       lab=tmp_path, **callbacks)

    callbacks["validator"] = lambda *_a, **_k: (
        {"r3": {"batch_sha256": "e" * 64},
         "oracle_policy": {"name": RUN.ORACLE_POLICY,
                           "selection_performance_credit": 0}},
        original_validator()[1], original_validator()[2])
    with pytest.raises(ValueError, match="original batch"):
        M.build_export(result_path, protocol_path, "d" * 40, plan_path,
                       lab=tmp_path, **callbacks)


def test_verify_rejects_modified_public_export(tmp_path: Path, monkeypatch):
    public, inputs = build(tmp_path)
    result_path, protocol_path, plan_path, _callbacks = inputs
    export_path = write(tmp_path / "public.json", public)
    changed = deepcopy(public)
    changed["claims"]["selection_performance_credit"] = 1
    write(export_path, changed)
    monkeypatch.setattr(M, "build_export", lambda *_a, **_k: public)
    with pytest.raises(ValueError, match="export changed"):
        M.verify(export_path, result_path, protocol_path, "d" * 40, plan_path, lab=tmp_path)


def test_public_path_guard_rejects_path_and_lease_fields():
    with pytest.raises(ValueError, match="path/lease"):
        M._reject_paths({"result_path": "secret"})
    with pytest.raises(ValueError, match="path/lease"):
        M._reject_paths({"lease_id": "secret"})
    with pytest.raises(ValueError, match="path string"):
        M._reject_paths({"value": "C:\\private\\result.json"})
