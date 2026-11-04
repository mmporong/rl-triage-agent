import json
from pathlib import Path

import pytest

from evals import registered_recovery_resources_v1 as resources
from rl_triage import intervention_governor as governor
from rl_triage import investigation_recovery as recovery
from rl_triage.correction_ledger import CorrectionLedger
from rl_triage.correction_registry_v2 import REGISTRY
from rl_triage.investigation_session import file_sha


def test_consumed_recovery_without_artifact_closes_unknown_without_rerun(tmp_path: Path) -> None:
    ledger = CorrectionLedger(tmp_path / "recovery.jsonl", registry=REGISTRY)
    proposal = ledger.propose("state", "optimizer_update_count", "receipt",
                              "restore_configured_ppo_epochs", "restart",
                              {"case": "optimizer_update_count", "arm": "corrected"}, 10)
    ledger.approve(proposal["request_id"], "host")
    ledger.consume(proposal["request_id"], "state")
    receipt = ledger.recover(proposal["request_id"], None)
    assert receipt["verdict"] == "unknown"
    assert receipt["reason"] == "orphan_without_artifact"
    assert ledger.requests()[proposal["request_id"]]["state"] == "done"


@pytest.mark.parametrize("already_qualified", [False, True])
@pytest.mark.parametrize(("runtime_outcome", "relative_gap", "expected_status"), [
    (None, False, "hold"), ("normal", False, "host_no_intervention"),
    ("normal", True, "reference_review_required"), ("abnormal", True, "investigation_required"),
    ("unknown", True, "hold"),
])
def test_actual_post_sealer_uses_actual_criteria_and_new_checkpoint_without_copying_checks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_qualified: bool,
        runtime_outcome: str | None, relative_gap: bool, expected_status: str) -> None:
    from rl_triage import intervention_evidence as evidence
    from rl_triage import observable_behavior_v1 as behavior

    source = evidence.verify_public_fixed26_pair(evidence.SOURCE_COLLECTION, case_id="case-01")
    qualified = behavior.qualify_source_observation(source)
    payload = governor.build_model_packet(qualified)["observation"]
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"new checkpoint for actual-shape CPU adapter test")
    checkpoint_ref = {"path": str(checkpoint), "sha256": file_sha(checkpoint)}
    report = json.loads(Path(payload["source_refs"]["candidate_fixed26"]["path"]).read_text(
        encoding="utf-8"))["report"]
    report["name"] = "corrected"
    report["checkpoint"] = {"file": checkpoint.name, "sha256": checkpoint_ref["sha256"]}
    report_path = tmp_path / "fixed26.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    fixed_ref = {"path": str(report_path), "sha256": file_sha(report_path)}
    reference = json.loads(Path(payload["source_refs"]["reference_fixed26"]["path"]).read_text(
        encoding="utf-8"))["report"]
    reference["name"] = "private-original-reference"
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    post = {"mechanism": "optimizer_update_count", "correction_id": "restore_configured_ppo_epochs",
            "fixed_report": fixed_ref, "reference_report": {"path": str(reference_path),
            "sha256": file_sha(reference_path)}, "new_checkpoint": checkpoint_ref}
    if relative_gap:
        target_yaw = reference["metrics"]["yaw_rate_rmse_radps"] * 2.0
        for row in report["by_condition"]:
            row["yaw_error_sq_sum"] = target_yaw ** 2 * row["alive_steps_sum"]
            row["yaw_rate_rmse_radps"] = target_yaw
        report["metrics"]["yaw_rate_rmse_radps"] = target_yaw
        report_path.write_text(json.dumps(report), encoding="utf-8")
        fixed_ref["sha256"] = file_sha(report_path)
    if runtime_outcome is not None:
        post["runtime_confirmation"] = {
            "scope": "corrected_training_first_three_updates",
            "checks": [{"check_id": "optimizer_update_count", "outcome": runtime_outcome,
                        "refs": ["sha256:" + "a" * 64]}]}
    monkeypatch.setattr(recovery, "verified_post_observation", lambda *_args: post)
    input_observation = qualified if already_qualified else source
    sealed = recovery.sealed_post_observation(tmp_path, "r1", input_observation)
    result = governor.build_model_packet(sealed)["observation"]
    assert result["behavior_outcome"] == ("below_reference" if relative_gap else "normal")
    assert result["criteria"]["qualification_rule_id"] == behavior.RULE_ID
    assert result["candidate"]["checkpoint_sha256"] == checkpoint_ref["sha256"]
    expected_checks = (post["runtime_confirmation"]["checks"] if runtime_outcome is not None else [
        {"check_id": "post_cause_confirmation", "outcome": "unknown",
         "refs": ["sha256:" + fixed_ref["sha256"]]}])
    assert result["confirmation_checks"] == expected_checks
    assert result["cause_confirmation_receipt"] is None
    assert result["correction_receipt"]["status"] == "applied"
    gate_path = tmp_path / "host-gate.json"
    gate_path.write_text(json.dumps({"schema": "host_gate_receipt_v1", "gate_id": "post-host",
                                    "allowed_actions": ["no_intervention", "investigate", "correct",
                                                        "reference_review"], "refs": ["host:policy"]}),
                         encoding="utf-8")
    gate = governor.verify_host_gate_receipt({"schema": governor.HOST_SOURCE_SCHEMA,
        "receipt": {"path": str(gate_path), "sha256": file_sha(gate_path)}})
    decision = governor.decide_intervention(sealed, None, gate)
    assert decision["status"] == expected_status
    assert decision["approval_granted"] is decision["authority_created"] is False
    if expected_status == "reference_review_required":
        assert decision["correctability"] == "reference_review_first"
        assert decision["investigation_need"] == "undetermined"
        assert decision["additional_checks"] == ["review_reference_and_evaluation_criteria",
                                                  "verify_unobserved_registered_components"]
    with pytest.raises(ValueError, match="registered new-checkpoint verifier"):
        recovery.sealed_post_observation(tmp_path, "r1", input_observation,
                                         post_confirmation_refs=[{"path": "fixture", "sha256": "0" * 64}])
    reference["checkpoint"]["sha256"] = "f" * 64
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    post["reference_report"]["sha256"] = file_sha(reference_path)
    with pytest.raises(ValueError, match="qualified source reference"):
        recovery.sealed_post_observation(tmp_path, "r1", input_observation)


def test_dispatch_lease_admission_failure_keeps_request_approved(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = CorrectionLedger(tmp_path / "recovery.jsonl", registry=REGISTRY)
    proposal = ledger.propose("state", "optimizer_update_count", "receipt",
                              "restore_configured_ppo_epochs", "restart",
                              {"case": "optimizer_update_count", "arm": "corrected"}, 10)
    ledger.approve(proposal["request_id"], "registered-host")
    request_id = proposal["request_id"]
    (tmp_path / "approvals").mkdir()
    (tmp_path / "authority_costs").mkdir()
    (tmp_path / "approvals" / f"{request_id}.json").write_text(
        json.dumps({"proof": {"status": "authorized"}}), encoding="utf-8")
    (tmp_path / "authority_costs" / f"{request_id}.json").write_text(
        json.dumps({"authority_s": 0.1}), encoding="utf-8")
    manifest = {"loop_state_digest": "state", "prepare_s": 0.1,
                "recovery_context": {"protocol_body": {}}}
    monkeypatch.setattr(recovery, "_load", lambda _folder: (manifest, ledger))
    monkeypatch.setattr(recovery, "_backend", lambda: {
        "admit": lambda _context, _request: (_ for _ in ()).throw(ValueError("no live lease"))})
    with pytest.raises(ValueError, match="no live lease"):
        recovery.dispatch(tmp_path, request_id)
    assert ledger.requests()[request_id]["state"] == "approved"


@pytest.mark.parametrize("execution_fails", [False, True])
def test_dispatch_measures_sampler_boundary_and_owned_execution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution_fails: bool) -> None:
    ledger = CorrectionLedger(tmp_path / "recovery.jsonl", registry=REGISTRY)
    proposal = ledger.propose("state", "optimizer_update_count", "receipt",
                              "restore_configured_ppo_epochs", "restart",
                              {"case": "optimizer_update_count", "arm": "corrected"}, 10)
    ledger.approve(proposal["request_id"], "registered-host")
    request_id = proposal["request_id"]
    (tmp_path / "approvals").mkdir()
    (tmp_path / "authority_costs").mkdir()
    (tmp_path / "approvals" / f"{request_id}.json").write_text(
        json.dumps({"proof": {"status": "authorized"}}), encoding="utf-8")
    (tmp_path / "authority_costs" / f"{request_id}.json").write_text(
        json.dumps({"authority_s": 0.1}), encoding="utf-8")
    output = tmp_path / "execution"
    context = {"output_root": str(output), "protocol_body": {}}
    manifest = {"loop_state_digest": "state", "prepare_s": 0.1, "recovery_context": context}
    events = []

    class Sampler:
        def __init__(self, folder):
            self.folder = Path(folder)

        def start(self):
            events.append("start")
            return {"scope": "device_wide"}

        def stop(self):
            events.append("stop")
            path = self.folder / "stop.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"schema": "sampler-stop"}), encoding="utf-8")
            return {"receipt_ref": {"path": str(path.resolve()), "sha256": file_sha(path)}}

    def execute(_intent):
        output.mkdir()
        if execution_fails:
            raise FileNotFoundError(f"missing owned launch folder: {output / 'training/owned/launch.json'}")
        training = output / "training/execution.json"
        training.parent.mkdir()
        training.write_text(json.dumps({"wall_s": 2.0}), encoding="utf-8")
        receipt = output / "receipt.json"
        receipt.write_text(json.dumps({"schema": "raw"}), encoding="utf-8")
        return {"verdict": "unknown", "gpu_s": 2.0, "exit_code": 0,
                "receipt_ref": {"path": str(receipt.resolve()), "sha256": file_sha(receipt)}}

    fake_backend = {"admit": lambda _context, request: {
                        "schema": "registered_recovery_lease_admission_v1", "request_id": request["request_id"],
                        "effect_key": request["effect_key"], "owner": "rl-triage-lineage-sources-v1",
                        "lease_id": "lease-1"},
                    "execute": execute, "verify_receipt": lambda raw, _intent: raw}
    monkeypatch.setattr(recovery, "_load", lambda _folder: (manifest, ledger))
    monkeypatch.setattr(recovery, "_backend", lambda: fake_backend)
    monkeypatch.setattr(recovery, "status", lambda _folder: {"requests": ledger.requests()})
    monkeypatch.setattr(resources, "PeriodicDeviceSampler", Sampler)
    result = recovery.dispatch(tmp_path, request_id)
    assert events == ["start", "stop"]
    if execution_fails:
        error = json.loads((tmp_path / "dispatch_errors" / f"{request_id}.json").read_text(encoding="utf-8"))
        assert error["error_type"] == "FileNotFoundError"
        assert str(output / "training/owned/launch.json") in error["error_message"]
        assert "Traceback (most recent call last)" in error["traceback"]
        assert "raise FileNotFoundError" in error["traceback"]
        assert error["reexecute_allowed"] is False
        assert result["requests"][request_id]["state"] == "running"
        recovery.settle(tmp_path, request_id, None)
        assert ledger.requests()[request_id]["receipt"]["reason"] == "orphan_without_artifact"
        with pytest.raises(ValueError, match="approved unused"):
            recovery.dispatch(tmp_path, request_id)
        assert events == ["start", "stop"]
        return
    assert result["requests"][request_id]["state"] == "done"
    resource_ref = {"path": str((tmp_path / f"resources/{request_id}.json").resolve()),
                    "sha256": file_sha(tmp_path / f"resources/{request_id}.json")}
    measured = resources.verify(resource_ref, request_id=request_id,
                                effect_key=proposal["effect_key"])
    assert measured["owned_gpu_s"] == 2.0 and measured["simulator_sdk_launches"] == 1


def test_prepare_revalidates_source_and_confirming_raw_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_folder = tmp_path / "session"
    host = session_folder / "host_receipts/r1.json"
    host.parent.mkdir(parents=True)
    host.write_text(json.dumps({"raw": True}), encoding="utf-8")
    (session_folder / "manifest.json").write_text("{}", encoding="utf-8")
    calls = {"source": 0, "receipt": 0}

    class Ledger:
        @staticmethod
        def requests():
            return {"r1": {"request_id": "r1", "state": "done", "probe": "P_optimizer_count",
                           "receipt": {"outcome": "abnormal"}}}

        @staticmethod
        def events():
            return [{"event": "consumed", "request_id": "r1"}]

    class Session:
        _manifest_sha256 = "1" * 64

        def __init__(self, _path):
            self.manifest = {"source": {"sha256": "2" * 64}}
            self.source = {"checkpoint_sha256": "3" * 64, "case_id": "case03"}
            self.ledger = Ledger()

        @staticmethod
        def _verify_operation(verifier):
            assert verifier is not None
            calls["source"] += 1

        @staticmethod
        def status():
            return {"status": "confirmed", "confirmed_mechanism": "optimizer_update_count"}

        @staticmethod
        def _intent(request, consume):
            return {"request": request, "consumption": consume, "host_receipt_path": str(host)}

    monkeypatch.setattr(recovery, "InvestigationSession", Session)

    def verify_receipt(raw, _intent):
        assert raw == {"raw": True}
        calls["receipt"] += 1
        return {"outcome": "abnormal", "probe": "P_optimizer_count",
                "receipt_ref": {"path": str(host), "sha256": file_sha(host)}}

    monkeypatch.setattr(recovery.registered_diagnostic_backend, "verify_receipt", verify_receipt)
    context = {"protocol_body": {"diagnostic": {
        "session_manifest_sha256": "1" * 64, "source_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64, "case_id": "case03",
        "mechanism": "optimizer_update_count", "probe": "P_optimizer_count",
        "diagnostic_receipt_sha256": file_sha(host)}}}
    monkeypatch.setattr(recovery, "_backend", lambda: {"prepare": lambda _payload: context})
    result = recovery.prepare(session_folder, {}, tmp_path / "recovery")
    assert result["mechanism"] == "optimizer_update_count"
    assert calls == {"source": 1, "receipt": 1}


def test_sealed_post_observation_is_governor_consumable_and_missing_post_cause_stays_unknown(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_path = tmp_path / "reference.json"
    metrics = {"fall_rate": 0.0, "mean_survival_s": 20.0,
               "lin_vel_rmse_mps": 0.5, "yaw_rate_rmse_radps": 0.4}
    reference_path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    reference_ref = {"path": str(reference_path), "sha256": file_sha(reference_path)}
    candidate = {"source_id": "case", "task": "task", "protocol": "fixed_eval_v1", "seed": 1,
                 "checkpoint_sha256": "1" * 64, "source_binding_sha256": "2" * 64,
                 "metrics": metrics, "refs": ["source"]}
    criteria = {"task": "task", "protocol": "fixed_eval_v1", "source_binding_sha256": "2" * 64,
                "metrics": {name: {"direction": "min" if name == "mean_survival_s" else "max",
                                    "absolute": 20.0 if name == "mean_survival_s" else 1.0,
                                    "relative": 0.8 if name == "mean_survival_s" else 1.2}
                            for name in metrics}, "refs": ["criteria"]}
    source = governor._verified_observation({
        "schema": governor.NORMALIZED_SCHEMA, "receipt_id": "source", "verifier_id": "test",
        "verification_scope": "test", "candidate": candidate, "reference": dict(candidate),
        "criteria": criteria, "reference_available": True, "lineage_matches": True,
        "behavior_outcome": "normal", "metric_checks": [], "confirmation_checks": [],
        "cause_confirmation_receipt": None, "correction_receipt": None,
        "source_refs": {"candidate_eval": None, "reference_eval": None, "evaluation_criteria": None}})
    monkeypatch.setattr(recovery, "verified_post_observation", lambda *_args, **_kwargs: {
        "mechanism": "optimizer_update_count", "correction_id": "restore_configured_ppo_epochs",
        "metrics": metrics, "fixed_report": {"path": "post", "sha256": "3" * 64},
        "reference_report": reference_ref, "new_checkpoint": {"path": "cp", "sha256": "4" * 64}})
    sealed = recovery.sealed_post_observation(tmp_path, "r1", source)
    packet = governor.build_model_packet(sealed)
    assert packet["observation"]["confirmation_checks"] == [{
        "check_id": "post_cause_confirmation", "outcome": "unknown", "refs": ["sha256:" + "3" * 64]}]
    assert packet["observation"]["correction_receipt"]["status"] == "applied"
