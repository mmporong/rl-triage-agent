from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
import lineage_recovery_execution_v1 as T


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _protocol() -> dict:
    return {"source_sha256_lf": {name: "a" * 64 for name in T.TRAIN.BOUND_SOURCES},
            "runtime_source_sha256_lf": {f"runtime/{index}.py": "b" * 64 for index in range(9)}}


def _row(tmp_path: Path, *, case: str = "case-03", policy: str = "registry_next",
         status: str = "candidate") -> dict:
    arm, seed = ("F1", 2303) if case == "case-03" else ("COMPOUND", 2307)
    candidate = _json(tmp_path / f"{case}_{policy}_candidate.json", {
        "schema": T.CANDIDATE.SCHEMA, "status": "validated_candidate", "case_id": case,
        "seed": seed, "source_arm": arm, "component": "reward",
        "correction": "restore_body_frame_reward", "approval_granted": False,
        "execution_allowed": False, "inspection_evidence": {"steps": []},
        "restart_candidate": {"source_checkpoint_sha256": "c" * 64},
    })
    reference = _json(tmp_path / f"{case}_reference.json", {
        "checkpoint": {"sha256": "d" * 64}, "versions": {"runtime": "same"},
        "horizon_steps": 1000, "step_dt": .02,
        "metrics": {"mean_survival_s": 20.0, "fall_rate": 0.0,
                    "lin_vel_rmse_mps": 1.0, "yaw_rate_rmse_radps": 1.0},
    })
    epoch = _json(tmp_path / f"{case}_{policy}_epoch.json", {"status": "complete"})
    case_input = _json(tmp_path / f"{case}_case.json", {"case": case})
    reference_input = _json(tmp_path / f"{case}_none.json", {"case": case})
    return {"case_id": case, "policy": policy, "arm": arm, "seed": seed,
            "source_job": "source", "reference_job": "none", "status": status,
            "hold_reason": "unconfirmed" if status == "HOLD" else None,
            "candidate": _ref(candidate) if status == "candidate" else None,
            "epoch_result": _ref(epoch), "case_input": _ref(case_input),
            "reference_input": _ref(reference_input),
            "source_checkpoint_sha256": "c" * 64,
            "reference_checkpoint_sha256": "d" * 64,
            "reference_report": {**_ref(reference), "checkpoint_sha256": "d" * 64}}


def _manifest(tmp_path: Path, rows: list[dict]) -> dict:
    protocol = _json(tmp_path / "protocol.json", {"schema": T.SCHEMA})
    batch_manifest = _json(tmp_path / "batch-manifest.json", {})
    batch_result = _json(tmp_path / "batch-result.json", {})
    target = _json(tmp_path / "targets.json", {})
    return {"schema": T.MANIFEST_SCHEMA, "protocol": _ref(protocol), "anchor": "a" * 40,
            "batch_manifest": _ref(batch_manifest), "batch_result": _ref(batch_result),
            "target_record": _ref(target), "lab": str(tmp_path / "lab"),
            "isaac_python": str(tmp_path / "python.bat"), "rows": rows,
            "same_seed_pairing": True, "new_independent_seed": False,
            "source18_verified": True, "fixed18_verified": True, "batch30_verified": True,
            "scope": "first_registered_correction_only", "epoch2_registered": False}


def test_prepare_protocol_has_fixed_targets_scope_budgets_and_no_private_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _json(tmp_path / "batch.json", {
        "schema": T.BATCH.SCHEMA,
        "runtime_source_sha256_lf": {f"runtime/{index}.py": "a" * 64 for index in range(9)},
    })
    monkeypatch.setattr(T, "_relative", lambda path: "batch.json" if Path(path) == batch else
                        Path(path).resolve().relative_to(ROOT).as_posix())
    value = T.prepare_protocol(batch)
    assert value["scope"] == {"epoch": 1, "second_epoch": False,
                              "overall_goal_complete": False,
                              "second_epoch_requires": "new_protocol_after_valid_not_recovered_receipt"}
    assert value["targets"] == list(T.TARGETS) and value["policies"] == list(T.POLICIES)
    assert value["target_commitment"]["sha256"] == T.TARGET_SHA256
    assert "private" not in json.dumps(value).lower()
    assert value["watchdogs_s"] == T.WATCHDOG
    assert value["training"]["seed_reuse_claim"].endswith("not_independent_seed")
    assert value["predictions"] == {
        "f1_valid_reward_candidate": "same_seed_fresh_restart_300_expected_recovered",
        "compound_first_single_correction": (
            "same_seed_fresh_restart_300_expected_not_recovered_due_to_remaining_fault"),
        "unconfirmed_or_unsupported_candidate": "HOLD_with_zero_training_jobs",
        "contrary_healthy_compound_result": (
            "preserve_observed_result_without_prediction_change_or_forced_second_correction"),
    }
    wording = value["outcome_wording"]
    assert set(wording) == {"recovered", "not_recovered", "HOLD", "invalid", "claim_scope"}
    assert "두 번째 교정을 실행하지 않는다" in wording["not_recovered"]
    assert all(term in wording["claim_scope"] for term in ("원인 확정", "LLM 우위", "독립 seed"))


@pytest.mark.parametrize(("arm", "correction", "faults", "epochs"), [
    ("F1", "restore_body_frame_reward", ["F1"], 5),
    ("COMPOUND", "restore_body_frame_reward", ["F1", "R1"], 1),
    ("COMPOUND", "restore_configured_ppo_epochs", ["F1", "R1"], 5),
])
def test_training_manifest_is_first_fresh_same_seed_and_source_cp_is_provenance_only(
        tmp_path: Path, arm: str, correction: str, faults: list[str], epochs: int) -> None:
    row = _row(tmp_path, case="case-03" if arm == "F1" else "case-07")
    candidate = json.loads(Path(row["candidate"]["path"]).read_text(encoding="utf-8"))
    candidate["correction"] = correction
    Path(row["candidate"]["path"]).write_text(json.dumps(candidate), encoding="utf-8")
    row["candidate"] = _ref(Path(row["candidate"]["path"]))
    fixed = T._training_manifest(T.FIXED_PROTOCOL, _protocol(), "a" * 40, row, tmp_path / "stage")
    assert fixed["arm"] == "FIRST" and fixed["mode"] == "train"
    assert fixed["seed"] == row["seed"] and fixed["max_iterations"] == 300
    assert fixed["original_faults"] == faults and fixed["applied_corrections"] == [correction]
    assert fixed["source_checkpoint_sha256"] == row["source_checkpoint_sha256"]
    remaining = T.TRAIN.RESIDUAL.FAULTS.remaining_faults(faults, [correction])
    assert (1 if "R1" in remaining else 5) == epochs


def test_smoke_manifest_is_none_inspection_three_and_not_a_training_control(tmp_path: Path) -> None:
    row = _row(tmp_path)
    fixed = T._training_manifest(T.FIXED_PROTOCOL, _protocol(), "a" * 40, row,
                                 tmp_path / "smoke", smoke_mode=True)
    assert fixed["arm"] == "NONE" and fixed["mode"] == "inspection"
    assert fixed["max_iterations"] == 3 and fixed["applied_corrections"] == []
    assert fixed["source_checkpoint_sha256"] == row["reference_checkpoint_sha256"]


def test_operator_approval_is_exact_candidate_digest_boundary(tmp_path: Path) -> None:
    row = _row(tmp_path)
    good = _json(tmp_path / "approval.json", {"schema": T.APPROVAL_SCHEMA, "approver": "operator",
        "entries": [{"case_id": row["case_id"], "policy": row["policy"],
                     "candidate_sha256": row["candidate"]["sha256"], "decision": "approve"}]})
    assert T._approval(good, [row])["approver"] == "operator"
    value = json.loads(good.read_text(encoding="utf-8")); value["entries"][0]["candidate_sha256"] = "0" * 64
    bad = _json(tmp_path / "bad.json", value)
    with pytest.raises(ValueError, match="candidate digest"):
        T._approval(bad, [row])


def test_unconfirmed_candidate_holds_but_confirmed_source_mismatch_is_not_hidden(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = {"result_path": str(tmp_path / "epoch.json")}
    status, reason, ref = T._materialize_candidate(
        {**common, "result": {"status": "complete", "end_state": "HOLD",
                               "hold_reason": "probe_outcome_unknown"}},
        case_input=tmp_path / "case.json", reference_input=tmp_path / "none.json",
        gate_path=tmp_path / "gate.json", gate_sha256="a" * 64, case_id="case-03",
        candidate_path=tmp_path / "candidate.json")
    assert (status, reason, ref) == ("HOLD", "probe_outcome_unknown", None)

    confirmed = {"schema": T.CANDIDATE.E.SCHEMA, "status": "complete",
                 "end_state": "confirmed", "hold_reason": None, "repair_supported": True,
                 "correction_executed": False, "training_executed": False,
                 "epoch2_registered": False}
    monkeypatch.setattr(T.CANDIDATE, "prepare_candidate",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("source mismatch")))
    with pytest.raises(ValueError, match="source mismatch"):
        T._materialize_candidate(
            {**common, "result": confirmed}, case_input=tmp_path / "case.json",
            reference_input=tmp_path / "none.json", gate_path=tmp_path / "gate.json",
            gate_sha256="a" * 64, case_id="case-03", candidate_path=tmp_path / "candidate.json")


def test_unconfirmed_rows_hold_with_zero_sdk_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(tmp_path, case=case, policy=policy, status="HOLD")
            for case in ("case-03", "case-07") for policy in T.POLICIES]
    manifest = _manifest(tmp_path, rows)
    manifest_path = _json(tmp_path / "manifest.json", manifest)
    approval = _json(tmp_path / "approval.json", {"schema": T.APPROVAL_SCHEMA,
                                                    "approver": "operator", "entries": []})
    monkeypatch.setattr(T, "_load_manifest", lambda _path: (manifest, _protocol()))
    monkeypatch.setattr(T, "verify_smoke", lambda *_args: {"path": "smoke", "result_sha256": "f" * 64})
    monkeypatch.setattr(T, "_relative", lambda path: Path(path).name)
    called = 0

    def runner(*_args):
        nonlocal called
        called += 1
        raise AssertionError("HOLD에서 SDK child를 시작하면 안 된다")

    result = T.execute(manifest_path, approval, tmp_path / "smoke", tmp_path / "out", runner=runner)
    assert result["status"] == "complete" and called == 0
    assert all(row["status"] == "HOLD" and row["sdk_jobs"] == 0 for row in result["receipts"])


def _execute_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str = "recovered"):
    candidate_row = _row(tmp_path)
    hold_rows = [_row(tmp_path, case=case, policy=policy, status="HOLD")
                 for case, policy in (("case-03", "model_adaptive"),
                                      ("case-07", "registry_next"),
                                      ("case-07", "model_adaptive"))]
    rows = [candidate_row, *hold_rows]
    manifest = _manifest(tmp_path, rows)
    manifest_path = _json(tmp_path / "manifest.json", manifest)
    approval = _json(tmp_path / "approval.json", {"schema": T.APPROVAL_SCHEMA, "approver": "delegated",
        "entries": [{"case_id": candidate_row["case_id"], "policy": candidate_row["policy"],
                     "candidate_sha256": candidate_row["candidate"]["sha256"], "decision": "approve"}]})
    monkeypatch.setattr(T, "_load_manifest", lambda _path: (manifest, _protocol()))
    monkeypatch.setattr(T, "verify_smoke", lambda *_args: {"path": "smoke", "result_sha256": "f" * 64})
    monkeypatch.setattr(T, "_relative", lambda path: Path(path).name)
    candidate = json.loads(Path(candidate_row["candidate"]["path"]).read_text(encoding="utf-8"))
    monkeypatch.setattr(T, "_candidate", lambda *_args: candidate)

    def train(stage, fixed, _manifest_value, _protocol_value, **_kwargs):
        stage.mkdir(parents=True)
        _json(stage / "manifest.json", fixed)
        _json(stage / "owned-result.json", {"exit_code": 0})
        checkpoint = stage / "model_299.pt"; checkpoint.write_bytes(b"checkpoint")
        return ({"wall_s": 1.0}, {"checkpoint": {"path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "iteration": 299}})

    def collect(_manifest_path, _evidence, _owned, out, _python, *, lab):
        out.mkdir(parents=True)
        value = {"schema": T.RECOVERY_INPUTS.SCHEMA, "lab": str(lab)}
        _json(out / "collected.json", value)
        return value

    def evaluate(stage, _row_value, _manifest_value, _evidence, **_kwargs):
        stage.mkdir(parents=True)
        report = json.loads(Path(candidate_row["reference_report"]["path"]).read_text(encoding="utf-8"))
        _json(stage / "corrected.json", report)
        return report, {"wall_s": 1.0}

    monkeypatch.setattr(T, "_run_training", train)
    monkeypatch.setattr(T, "_fixed_evaluate", evaluate)
    monkeypatch.setattr(T.RECOVERY_INPUTS, "checked_recovery", lambda _path: ({}, {}))
    monkeypatch.setattr(T.C.D1, "recovery", lambda *_args: {"outcome": outcome, "verdict": {"label": outcome}})
    result = T.execute(manifest_path, approval, tmp_path / "smoke", tmp_path / "out",
                       runner=lambda *_args: {}, collector=collect)
    return result, tmp_path / "out"


def test_normal_result_uses_actual_correction_ledger_order_and_seals_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, out = _execute_fixture(tmp_path, monkeypatch)
    assert result["status"] == "complete" and result["overall_goal_complete"] is False
    recovered = next(row for row in result["receipts"] if row["status"] == "recovered")
    assert recovered["same_seed_pairing"] is True and recovered["new_independent_seed"] is False
    ledger = next((out / "ledgers").glob("*.jsonl"))
    events = [json.loads(line)["event"] for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert events == ["proposed", "approved", "consumed", "receipt"]
    assert recovered["second_epoch_registered"] is False and recovered["model_calls"] == 0


def test_unknown_is_hold_failure_and_starts_no_later_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="unknown"):
        _execute_fixture(tmp_path, monkeypatch, outcome="unknown")
    failure = json.loads((tmp_path / "out/failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "invalid" and len(failure["completed_receipts"]) == 1
    receipt = failure["completed_receipts"][0]
    assert receipt["status"] == "HOLD" and receipt["automatic_retry"] is False


def test_missing_smoke_stops_before_output_or_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row(tmp_path)
    manifest = _manifest(tmp_path, [row, _row(tmp_path, policy="model_adaptive"),
        _row(tmp_path, case="case-07"), _row(tmp_path, case="case-07", policy="model_adaptive")])
    monkeypatch.setattr(T, "_load_manifest", lambda _path: (manifest, _protocol()))
    with pytest.raises(FileNotFoundError):
        T.execute(tmp_path / "manifest.json", tmp_path / "approval.json", tmp_path / "missing-smoke",
                  tmp_path / "out", runner=lambda *_args: pytest.fail("SDK 실행 금지"))
    assert not (tmp_path / "out").exists()


def test_manifest_raw_source_reference_mismatch_is_rejected_before_protocol_verify(tmp_path: Path) -> None:
    rows = [_row(tmp_path, case=case, policy=policy, status="HOLD")
            for case in ("case-03", "case-07") for policy in T.POLICIES]
    value = _manifest(tmp_path, rows)
    manifest = _json(tmp_path / "manifest.json", value)
    Path(value["protocol"]["path"]).write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="protocol 원문"):
        T._load_manifest(manifest)


def test_training_failure_preserves_failure_and_starts_no_next_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(tmp_path, policy=policy) for policy in T.POLICIES]
    rows += [_row(tmp_path, case="case-07", policy=policy, status="HOLD") for policy in T.POLICIES]
    manifest = _manifest(tmp_path, rows)
    manifest_path = _json(tmp_path / "manifest.json", manifest)
    approval = _json(tmp_path / "approval.json", {"schema": T.APPROVAL_SCHEMA, "approver": "operator",
        "entries": [{"case_id": row["case_id"], "policy": row["policy"],
                     "candidate_sha256": row["candidate"]["sha256"], "decision": "approve"}
                    for row in rows if row["status"] == "candidate"]})
    monkeypatch.setattr(T, "_load_manifest", lambda _path: (manifest, _protocol()))
    monkeypatch.setattr(T, "verify_smoke", lambda *_args: {"path": "smoke", "result_sha256": "f" * 64})
    monkeypatch.setattr(T, "_relative", lambda path: Path(path).name)
    monkeypatch.setattr(T, "_candidate", lambda row, _manifest: json.loads(
        Path(row["candidate"]["path"]).read_text(encoding="utf-8")))
    calls = 0

    def failed(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("training failed")

    monkeypatch.setattr(T, "_run_training", failed)
    with pytest.raises(RuntimeError, match="training failed"):
        T.execute(manifest_path, approval, tmp_path / "smoke", tmp_path / "out",
                  runner=lambda *_args: {})
    assert calls == 1
    failure = json.loads((tmp_path / "out/failure.json").read_text(encoding="utf-8"))
    assert failure["automatic_retry"] is False and failure["completed_receipts"] == []


@pytest.mark.parametrize(("metrics", "horizon", "expected"), [
    ({"mean_survival_s": 20.0, "fall_rate": 0.0, "lin_vel_rmse_mps": 1.0,
      "yaw_rate_rmse_radps": 1.0}, 1000, "recovered"),
    ({"mean_survival_s": 5.0, "fall_rate": 0.5, "lin_vel_rmse_mps": 3.0,
      "yaw_rate_rmse_radps": 3.0}, 1000, "not_recovered"),
    ({"mean_survival_s": 10.0, "fall_rate": 0.0, "lin_vel_rmse_mps": 1.0,
      "yaw_rate_rmse_radps": 1.0}, 500, "unknown"),
])
def test_frozen_recovery_maps_recovered_not_recovered_unknown(metrics, horizon, expected) -> None:
    reference = {"versions": {"runtime": "same"}, "horizon_steps": 1000, "step_dt": .02,
                 "metrics": {"mean_survival_s": 20.0, "fall_rate": 0.0,
                             "lin_vel_rmse_mps": 1.0, "yaw_rate_rmse_radps": 1.0}}
    report = {"versions": {"runtime": "same"}, "horizon_steps": horizon,
              "step_dt": .02, "metrics": metrics}
    bands = {"survival_ratio_min": .9, "fall_frac_max": .05,
             "err_xy_ratio_max": 1.1513, "err_yaw_ratio_max": 1.1719}
    assert T.C.D1.recovery(report, reference, bands)["outcome"] == expected


def test_owned_receipt_rejects_false_zero_timeout_and_cleanup() -> None:
    base = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "job_assigned_before_release": True, "timed_out": False,
            "error_type": None, "wall_s": 1.0}
    assert T._owned_ok(base, 10)
    for key, value in (("natural_exit", False), ("cleanup_verified", False),
                       ("timed_out", True), ("wall_s", 11.0)):
        bad = {**base, key: value}
        assert not T._owned_ok(bad, 10)
