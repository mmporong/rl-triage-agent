from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_recovery_export_v1 as X


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retry_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    manifest_path = _write(tmp_path / "retry-manifest.json", {"placeholder": True})
    row_results = []
    specifications = (
        ("registry_next", ["P_value", "P_physics", "P_episode"], "HOLD",
         "decision_limit", None, 0, ["registered_rule"] * 3, False),
        ("model_adaptive", ["P_episode", "P_reward"], "confirmed",
         None, "reward", 2, ["model", "model"], True),
    )
    rows = []
    for order, spec in enumerate(specifications, 1):
        policy, probes, end, hold, confirmed, requests, actors, partial = spec
        result = {
            "probe_order": probes,
            "steps": [{"probe": probe, "paired_receipts": {},
                       "ledger_receipt": {"gpu_s": 1.0}} for probe in probes],
            "decision_count": len(probes), "requests": requests, "end_state": end,
            "hold_reason": hold, "confirmed_subtype": confirmed,
            "selection_actor": actors,
            "tokens": {"input_tokens": 11 * requests, "output_tokens": requests,
                       "total_tokens": 12 * requests,
                       "known_subtotal": {"input_tokens": 11 * requests,
                                          "output_tokens": requests,
                                          "total_tokens": 12 * requests},
                       "unknown_cost_requests": {"input_tokens": [],
                                                 "output_tokens": [],
                                                 "total_tokens": []}},
            "wall": {"known_model_wall_subtotal_s": float(requests),
                     "model_wall_s": float(requests), "probe_child_known_s": 3.0,
                     "probe_parent_known_s": 4.0, "unknown_model_cost_requests": []},
        }
        result_path = _write(tmp_path / f"{policy}.json", result)
        row_results.append(result_path)
        rows.append({
            "order": order, "case_id": "case-07", "policy": policy,
            "arm": "COMPOUND", "result": result, "result_path": str(result_path),
            "result_sha256": _sha(result_path),
            "classification": {"confirmed": confirmed is not None,
                               "compound_partial": partial,
                               "correct_confirmation": partial,
                               "false_confirmation": False,
                               "full_cause_confirmed": False,
                               "none_intervention": False},
        })
    full = {
        "schema": X.RETRY.RESULT_SCHEMA, "status": "complete",
        "evidence_scope": "actual_backend",
        "measurement_scope": "development_after_observed_infrastructure_failure",
        "rows": rows, "original_results_immutable": True,
        "correction_executed": False, "training_executed": False,
        "epoch2_registered": False, "automatic_retry": False,
    }
    result_path = _write(tmp_path / "retry-result.json", full)
    manifest = {"execution_anchor": "a" * 40,
                "execution_plan": [{"policy": "registry_next"},
                                   {"policy": "model_adaptive"}]}
    return manifest_path, result_path, manifest


def test_retry_summary_keeps_hold_partial_and_numeric_measurements(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, result_path, manifest = _retry_fixture(tmp_path)
    monkeypatch.setattr(X.RETRY, "_load_manifest", lambda _path: (manifest, {}))

    actual_manifest, public = X._retry_summary(manifest_path, result_path)

    assert actual_manifest == manifest
    assert public["rows"][0]["end_state"] == "HOLD"
    assert public["rows"][0]["hold_reason"] == "decision_limit"
    assert public["rows"][1]["classification"]["compound_partial"] is True
    assert public["rows"][1]["tokens"]["total_tokens"] == 24
    assert public["rows"][1]["wall"]["model_wall_s"] == 2.0
    assert public["model_diagnostic_selection"]["correction_choice"] is None


def test_retry_summary_rejects_result_sha_swap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, result_path, manifest = _retry_fixture(tmp_path)
    monkeypatch.setattr(X.RETRY, "_load_manifest", lambda _path: (manifest, {}))
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["rows"][0]["result_sha256"] = "0" * 64
    _write(result_path, value)

    with pytest.raises(ValueError, match="원문·SHA"):
        X._retry_summary(manifest_path, result_path)


def test_first_hold_requires_followup_false() -> None:
    row = {"case_id": "case-07", "policy": "registry_next", "arm": "COMPOUND",
           "seed": 2307, "status": "HOLD", "hold_reason": "decision_limit"}
    receipt = X.FIRST._hold_receipt(row, "decision_limit")
    assert X._first_hold(row, receipt)["followup_epoch_allowed"] is False
    receipt["followup_epoch_allowed"] = True
    with pytest.raises(ValueError, match="followup=false"):
        X._first_hold(row, receipt)


def _patch_validation(monkeypatch: pytest.MonkeyPatch, *, first_status: str,
                      second_correction: str | None = None) -> None:
    diagnosis = {"schema": "d", "model_diagnostic_selection": {
        "policy": "model_adaptive", "probe_order": ["P_reward"],
        "confirmed_subtype": "reward", "correction_choice": None}}
    monkeypatch.setattr(X, "_retry_summary", lambda *_args: (
        {"execution_anchor": "anchor"}, diagnosis))
    first_receipts = [
        {"schema": X.RECEIPT_SCHEMA, "epoch": 1, "case_id": "case-07",
         "policy": "registry_next", "status": "HOLD"},
        {"schema": X.RECEIPT_SCHEMA, "epoch": 1, "case_id": "case-07",
         "policy": "model_adaptive", "status": first_status,
         "correction": "restore_body_frame_reward"},
    ]
    monkeypatch.setattr(X, "_first", lambda *_args: (
        {"anchor": "anchor"}, first_receipts, {}, ["restore_body_frame_reward"]))
    if second_correction is not None:
        monkeypatch.setattr(X, "_second", lambda *_args: (
            {"schema": X.RECEIPT_SCHEMA, "epoch": 2, "case_id": "case-07",
             "policy": "model_adaptive", "status": "recovered",
             "correction": second_correction}, {}, second_correction,
            {"parent_binding_verified": True},
            {"end_state": "confirmed", "hold_reason": None,
             "confirmed_subtype": "optimizer_update_count"}))


def test_recovered_first_rejects_second_stage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validation(monkeypatch, first_status="recovered")
    inputs = [_write(tmp_path / name, {}) for name in ("rm", "rr", "fm", "fr", "sm", "sr")]
    with pytest.raises(ValueError, match="FIRST recovered"):
        X._validated(*inputs[:4], inputs[4], inputs[5])


def test_not_recovered_can_stop_without_second_and_preserves_host_choice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validation(monkeypatch, first_status="not_recovered")
    inputs = [_write(tmp_path / name, {}) for name in ("rm", "rr", "fm", "fr")]

    summary, _provenance, files = X._validated(*inputs, None, None)

    assert summary["terminal_status"] == "not_recovered"
    assert summary["second_stage_supplied"] is False
    diagnosis = json.loads(files["diagnosis.json"])
    assert diagnosis["model_diagnostic_selection"]["correction_choice"] == {
        "selected_by": "host_registered_correction_from_confirmed_subtype",
        "correction": "restore_body_frame_reward",
    }


def test_second_correction_must_be_distinct(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validation(monkeypatch, first_status="not_recovered",
                      second_correction="restore_body_frame_reward")
    inputs = [_write(tmp_path / name, {}) for name in ("rm", "rr", "fm", "fr", "sm", "sr")]
    with pytest.raises(ValueError, match="서로 다른 최대 2개"):
        X._validated(*inputs[:4], inputs[4], inputs[5])


def test_second_diagnosis_is_linked_without_reinterpreting_outcome(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validation(monkeypatch, first_status="not_recovered",
                      second_correction="restore_configured_ppo_epochs")
    inputs = [_write(tmp_path / name, {}) for name in ("rm", "rr", "fm", "fr", "sm", "sr")]

    summary, provenance, files = X._validated(*inputs[:4], inputs[4], inputs[5])

    assert summary["epoch2_diagnosis_outcome"] == {
        "end_state": "confirmed", "hold_reason": None,
        "confirmed_subtype": "optimizer_update_count",
        "path": "post-correction-diagnosis.json",
        "sha256": hashlib.sha256(files["post-correction-diagnosis.json"]).hexdigest(),
    }
    assert provenance["second_adapter"] == {"parent_binding_verified": True}
    assert summary["distinct_corrections"] == [
        "restore_body_frame_reward", "restore_configured_ppo_epochs"]


def test_epoch2_adapter_chain_is_bound_to_exact_result_and_context(tmp_path: Path) -> None:
    result = _write(tmp_path / "epoch2/result.json", {"schema": X.SECOND.E2.SCHEMA})
    v3_protocol = _write(tmp_path / "v3.json", {"protocol": 3})
    v2_protocol = _write(tmp_path / "v2.json", {"protocol": 2})
    parent_protocol = _write(tmp_path / "parent.json", {"protocol": 1})
    context = {
        "post_input": {"path": str(tmp_path / "post.json"), "sha256": "1" * 64},
        "reference_input": {"path": str(tmp_path / "reference.json"), "sha256": "2" * 64},
        "run_id": "post_model_adaptive", "policy": "model_adaptive",
        "parent_binding": {"packet_digest": "3" * 64, "run_id": "post_model_adaptive",
                           "policy": "model_adaptive", "parent": {"epoch": 1}},
    }
    v2_ref = {"path": str(v2_protocol), "sha256_lf": _sha(v2_protocol)}
    _write(tmp_path / "epoch2/v3-adapter-receipt.json", {
        "schema": X.EPOCH2_V3.WRAPPER_RECEIPT_SCHEMA, "status": "complete",
        "v3_protocol": {"path": str(v3_protocol), "sha256": _sha(v3_protocol)},
        "v2_protocol": v2_ref, "anchor": "a" * 40,
        "post_observation_sha256": "1" * 64,
        "parent_context": {"packet_digest": "3" * 64,
                           "run_id": "post_model_adaptive", "policy": "model_adaptive",
                           "parent": {"epoch": 1}},
        "parent_result": {"path": str(result), "sha256": _sha(result)},
        "parent_result_schema": X.SECOND.E2.SCHEMA,
    })
    v2_receipt = _write(tmp_path / "epoch2/v2-wrapper-receipt.json", {
        "schema": X.EPOCH2_V2.WRAPPER_RECEIPT_SCHEMA, "status": "complete",
        "wrapper_protocol": {"path": str(v2_protocol), "sha256": _sha(v2_protocol)},
        "parent_protocol": {"path": str(parent_protocol),
                            "sha256_lf": _sha(parent_protocol)},
        "parent_result": {"path": str(result), "sha256": _sha(result)},
        "parent_result_schema": X.SECOND.E2.SCHEMA,
        "preflight_inside_output": False, "diagnosis_replayed": False,
        "training_replayed": False,
    })

    public = X._checked_epoch2_adapters(result, context)

    assert public["parent_binding_verified"] is True
    assert public["v2_wrapper_receipt_sha256"] == _sha(v2_receipt)
    broken = json.loads(v2_receipt.read_text(encoding="utf-8"))
    broken["parent_result"]["sha256"] = "0" * 64
    _write(v2_receipt, broken)
    with pytest.raises(ValueError, match="v2 wrapper"):
        X._checked_epoch2_adapters(result, context)


def test_post_diagnosis_projection_keeps_hold_and_removes_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    step = tmp_path / "epoch2/probes/step_01"
    auth = _write(step / "authorization.json", {"schema": "authorization"})
    paired = {}
    for role, measurement in (("case", {"noise_ratio": 1.1}),
                              ("reference", {"noise_ratio": 1.0})):
        paired[role] = {"status": "complete", "role": role,
                        "measurement": measurement, "execution": {"wall_s": 1.0}}
        _write(step / role / "probe_receipt.json", paired[role])
    result = {
        "schema": X.SECOND.E2.SCHEMA, "status": "complete",
        "evidence_scope": "actual_backend", "epoch": 2, "case_id": "case-07",
        "policy": "model_adaptive", "correction_executed": False,
        "training_executed": False, "run_id": "post_model_adaptive",
        "end_state": "HOLD", "hold_reason": "decision_limit",
        "confirmed_subtype": None, "probe_order": ["P_noise"], "decision_count": 1,
        "steps": [{"decision": 1, "probe": "P_noise", "outcome": "normal",
                   "request_id": "r1", "authorization_path": str(auth),
                   "authorization_sha256": _sha(auth), "paired_receipts": paired}],
        "requests": 1, "model_validity": ["valid"], "selection_actor": ["model"],
        "tokens": {"total_tokens": 10}, "wall": {"model_wall_s": 0.2},
        "source_binding": {
            "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64,
                     "seed": 2307},
            "reference": {"checkpoint_sha256": "3" * 64,
                          "telemetry_sha256": "4" * 64, "seed": 2307}},
        "prior_not_recovered_receipt_sha256": "5" * 64,
    }
    result_path = _write(tmp_path / "epoch2/result.json", result)
    monkeypatch.setattr(X.SECOND.CANDIDATE2, "_completed_ledger",
                        lambda _path, _steps: ([{}] * 4, {"r1": {}}))

    public = X._post_diagnosis_projection({
        "policy": "model_adaptive",
        "epoch_result": {"path": str(result_path), "sha256": _sha(result_path)}})

    assert public["end_state"] == "HOLD"
    assert public["hold_reason"] == "decision_limit"
    assert public["steps"][0]["roles"]["case"]["measurement"] == {"noise_ratio": 1.1}
    assert public["outcome_reinterpreted"] is False
    X.BASE._privacy("post-correction-diagnosis.json", X.BASE._bytes(public))


def test_export_is_fresh_private_free_and_verifiable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"schema": X.SCHEMA, "status": "complete", "diagnosis_rows": 2,
               "distinct_corrections": ["restore_body_frame_reward"],
               "max_distinct_corrections": 2}
    provenance = {"schema": X.SCHEMA + "_provenance_v1", "source_anchor": "abc"}
    monkeypatch.setattr(X, "_validated", lambda *_args: (
        summary, provenance, {"summary.json": X.BASE._bytes(summary),
                              "provenance.json": X.BASE._bytes(provenance)}))
    inputs = [_write(tmp_path / name, {}) for name in ("rm", "rr", "fm", "fr")]
    out = tmp_path / "public"

    X.export_completed(*inputs, out=out, repo=tmp_path)

    assert X.verify_export(out)["schema"] == X.MANIFEST_SCHEMA
    with pytest.raises(ValueError, match="새 하위 경로"):
        X.export_completed(*inputs, out=out, repo=tmp_path)


def test_privacy_rejects_private_path() -> None:
    with pytest.raises(ValueError, match="private 경로"):
        X.BASE._privacy("bad.json", b'{"path":"bench/private/input.json"}')
