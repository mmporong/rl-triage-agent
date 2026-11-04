from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from evals import registered_post_actuator_confirmation_v1 as M


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _protocol() -> dict:
    path = Path(__file__).with_name("fixtures") / "registered_post_actuator_confirmation_draft_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _request(tmp_path: Path) -> dict:
    refs = {}
    for name in ("checkpoint", "fixed_report", "training_receipt", "correction_readback",
                 "recovery_receipt", "source_protocol", "runtime_a", "runtime_b"):
        refs[name] = _json(tmp_path / f"{name}.json", {"role": name})
    return {
        "schema": M.REQUEST_SCHEMA, "profile_id": M.PROFILE_ID,
        "case_id": "case-05", "source_case_id": "F2s2664", "seed": 2664,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "condition": "NONE",
        "checkpoint": refs["checkpoint"], "fixed_report": refs["fixed_report"],
        "training_receipt": refs["training_receipt"],
        "correction_readback": refs["correction_readback"],
        "recovery_receipt": refs["recovery_receipt"], "source_protocol": refs["source_protocol"],
        "runtime_closure": {"runtime_a": refs["runtime_a"], "runtime_b": refs["runtime_b"]},
        "lease": {"profile_id": "registered_desktop_shared_baseline_v1",
                  "lease_id": "lease-1", "caller_pid": 42},
        "effect": {"effect_key": "effect-1", "request_id": "effect-request-1"},
        "request": {"approval_id": "approval-1", "request_id": "request-1"},
        "sdk_budget": 1, "result_seen_design": True,
    }


def _plan(tmp_path: Path) -> dict:
    return M.prepare(_request(tmp_path), _protocol())


def _execution() -> dict:
    return {"job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
            "cleanup_verified": True, "timed_out": False, "error_type": None,
            "owned_process_identities": {}}


def _component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, assessment: str = "normal") -> tuple[dict, dict]:
    plan = _plan(tmp_path)
    manifest = tmp_path / "manifest.json"
    body = tmp_path / "body.json"
    execution = tmp_path / "execution.json"
    manifest.write_text("{}", encoding="utf-8")
    body.write_text(json.dumps({"measurement_status": "valid", "curve_assessment": assessment,
                                "helper_result": {"status": "valid"},
                                "state_restoration": {"all_restored": True}}), encoding="utf-8")
    execution.write_text(json.dumps(_execution()), encoding="utf-8")
    calls = []

    def validate(path, **kwargs):
        calls.append(("manifest", path, kwargs))
        return {"manifest": {"condition": "NONE"}, "paths": {"output_path": body}}

    def verify(value, validated, owned):
        calls.append(("body", value, validated, owned))
        return []

    monkeypatch.setattr(M.PRODUCER, "validate_manifest", validate)
    monkeypatch.setattr(M.PRODUCER, "verify_body", verify)
    receipt = M.verify_component_receipt(plan, manifest, body, execution,
                                         repo_root=M.ROOT, isaaclab_root=tmp_path)
    assert [row[0] for row in calls] == ["manifest", "body"]
    return plan, receipt


def _observation(checkpoint_sha: str, behavior: str = "below_reference"):
    payload = {
        "schema": M.governor.NORMALIZED_SCHEMA, "receipt_id": "recovery:r1",
        "verifier_id": "registered_recovery_absolute_fixed26_v1",
        "verification_scope": "registered_recovery_absolute_fixed26_post_evidence",
        "candidate": {"checkpoint_sha256": checkpoint_sha, "refs": ["sha256:" + "1" * 64]},
        "reference": {"refs": ["sha256:" + "2" * 64]}, "criteria": {"refs": ["sha256:" + "3" * 64]},
        "reference_available": True, "lineage_matches": True, "behavior_outcome": behavior,
        "metric_checks": [],
        "confirmation_checks": [{"check_id": "actuator_mapping", "outcome": "unknown",
                                  "refs": ["sha256:" + "4" * 64]}],
        "cause_confirmation_receipt": None,
        "correction_receipt": {"correction_id": "restore_motor_parameters", "status": "applied",
                               "refs": ["sha256:" + "5" * 64]},
        "source_refs": {}, "post_confirmation_scope": "correction_readback_only",
        "confirmation_catalog_complete": False,
        "unobserved_confirmation_components": ["actuator_mapping", "optimizer_update_count",
                                                "ppo_surrogate_gradient"],
    }
    return M.governor._verified_observation(payload)


def _host_gate(tmp_path: Path):
    receipt = tmp_path / "gate.json"
    receipt.write_text(json.dumps({"schema": "host_gate_receipt_v1", "gate_id": "g1",
                                   "allowed_actions": ["no_intervention", "investigate", "correct",
                                                       "reference_review"],
                                   "refs": ["sha256:" + "a" * 64]}), encoding="utf-8")
    return M.governor.verify_host_gate_receipt({
        "schema": M.governor.HOST_SOURCE_SCHEMA,
        "receipt": {"path": str(receipt), "sha256": _sha(receipt)},
    })


def _supplement(tmp_path: Path, plan: dict, source, receipt: dict):
    return M.reverify_and_supplement(
        plan, source, tmp_path, "r1", receipt, _host_gate(tmp_path),
        repo_root=M.ROOT, isaaclab_root=tmp_path,
    )


def test_prepare_consumes_real_observer_preflight_and_keeps_execution_disabled(tmp_path: Path):
    result = _plan(tmp_path)
    assert result["component_entrypoint"]["child"] == "bench/actuator_envelope_runtime.py"
    assert result["component_invocation"]["callable"] == "bench.actuator_envelope_runtime.run_sdk"
    assert result["status"] == "not_registered_not_executable"
    assert result["execution_authority"] is False
    assert result["interpretation"]["actual_checkpoint_execution"] is False


@pytest.mark.parametrize("role", ["checkpoint", "fixed_report", "training_receipt",
                                  "correction_readback", "source_protocol"])
def test_prepare_rejects_changed_bound_source(tmp_path: Path, role: str):
    request = _request(tmp_path)
    Path(request[role]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reference changed"):
        M.prepare(request, _protocol())


def test_prepare_rejects_changed_runtime_closure(tmp_path: Path):
    request = _request(tmp_path)
    Path(request["runtime_closure"]["runtime_a"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_closure"):
        M.prepare(request, _protocol())


def test_component_receipt_uses_existing_validator_and_maps_measurement(tmp_path: Path, monkeypatch):
    _plan_value, receipt = _component(tmp_path, monkeypatch)
    assert receipt["outcome"] == "normal"
    assert receipt["joint_count"] == 12
    assert receipt["state_restored"] is True
    assert receipt["actual_checkpoint_execution"] is False


def test_curve_difference_is_abnormal_not_recovery_claim(tmp_path: Path, monkeypatch):
    _plan_value, receipt = _component(tmp_path, monkeypatch, assessment="curve_difference")
    assert receipt["outcome"] == "abnormal"
    assert receipt["trained_policy_validated"] is False


def test_measurement_free_normal_is_rejected(tmp_path: Path):
    plan = _plan(tmp_path)
    receipt = M.unknown_receipt(plan, "not_executed")
    forged = deepcopy(receipt)
    forged["outcome"] = "normal"
    core = {key: forged[key] for key in forged if key not in {"schema", "receipt_sha256"}}
    forged["receipt_sha256"] = M._digest(core)
    with pytest.raises(ValueError, match="Measurement-free|measurement-free"):
        M._component_receipt(forged, plan)


def test_self_digested_normal_with_empty_source_refs_is_rejected(tmp_path: Path):
    plan = _plan(tmp_path)
    forged = M.unknown_receipt(plan, "not_executed")
    forged.update({"outcome": "normal", "measurement_present": True,
                   "joint_count": 12, "state_restored": True, "validator_reasons": []})
    core = {key: forged[key] for key in forged if key not in {"schema", "receipt_sha256"}}
    forged["receipt_sha256"] = M._digest(core)
    with pytest.raises(ValueError, match="source ref set"):
        M._component_receipt(forged, plan)


def test_forbidden_private_subtree_is_rejected_before_read():
    value = {"path": "C:/x/ri_codex2_tokenfactory_live_160/secret.json", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="forbidden private subtree"):
        M._ref(value, "receipt")


def test_receipt_and_checkpoint_swap_are_rejected(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    changed = deepcopy(receipt)
    changed["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding"):
        M._component_receipt(changed, plan)
    other = deepcopy(receipt)
    other["checkpoint"] = {**other["checkpoint"], "sha256": "f" * 64}
    core = {key: other[key] for key in other if key not in {"schema", "receipt_sha256"}}
    other["receipt_sha256"] = M._digest(core)
    with pytest.raises(ValueError, match="checkpoint"):
        M._component_receipt(other, plan)


def test_plan_invocation_and_component_source_swap_are_rejected(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    changed_plan = deepcopy(plan)
    changed_plan["component_invocation"]["condition"] = "F2"
    with pytest.raises(ValueError, match="invocation"):
        M._validate_plan(changed_plan)
    Path(receipt["source_refs"]["body"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="component.body"):
        M._component_receipt(receipt, plan)


@pytest.mark.parametrize("role", ["body", "execution"])
def test_supplement_rejects_raw_component_mutation(tmp_path: Path, monkeypatch, role: str):
    plan, receipt = _component(tmp_path, monkeypatch)
    source = _observation(plan["request"]["checkpoint"]["sha256"])
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    Path(receipt["source_refs"][role]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match=f"component.{role} reference changed"):
        _supplement(tmp_path, plan, source, receipt)


def test_supplement_rejects_different_manifest_swap(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    original_manifest = Path(receipt["source_refs"]["manifest"]["path"])
    body = Path(receipt["source_refs"]["body"]["path"])
    replacement = tmp_path / "other-manifest.json"
    replacement.write_text("{}", encoding="utf-8")

    def validate(path, **_kwargs):
        if Path(path) != original_manifest:
            raise ValueError("manifest lineage differs")
        return {"manifest": {"condition": "NONE"}, "paths": {"output_path": body}}

    monkeypatch.setattr(M.PRODUCER, "validate_manifest", validate)
    changed = deepcopy(receipt)
    changed["source_refs"]["manifest"] = {"path": str(replacement), "sha256": _sha(replacement)}
    core = {key: changed[key] for key in changed if key not in {"schema", "receipt_sha256"}}
    changed["receipt_sha256"] = M._digest(core)
    source = _observation(plan["request"]["checkpoint"]["sha256"])
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    with pytest.raises(ValueError, match="manifest lineage differs"):
        _supplement(tmp_path, plan, source, changed)


def test_supplement_rejects_changed_producer_binding(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    source = _observation(plan["request"]["checkpoint"]["sha256"])
    original = M.OBSERVER.production_entrypoints

    def changed(root):
        value = original(root)
        value["actuator"]["source_sha256"]["child"] = "0" * 64
        return value

    monkeypatch.setattr(M.OBSERVER, "production_entrypoints", changed)
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    with pytest.raises(ValueError, match="producer source binding changed"):
        _supplement(tmp_path, plan, source, receipt)


def test_normal_mapping_below_reference_uses_scoped_reference_review(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    source = _observation(plan["request"]["checkpoint"]["sha256"])
    calls = []

    def sealed(folder, request_id, original):
        calls.append((folder, request_id))
        M.governor.build_model_packet(original)
        return source

    monkeypatch.setattr(M.recovery, "sealed_post_observation", sealed)
    result = _supplement(tmp_path, plan, source, receipt)
    assert calls == [(tmp_path, "r1")]
    assert result["status"] == "reference_review_required"
    assert result["effective_action"] == "reference_review"
    assert result["governor_decision"]["reason_code"] == "post_correction_relative_gap_with_scoped_checks"
    assert result["observation"]["confirmation_catalog_complete"] is False
    assert result["observation"]["unobserved_confirmation_components"] == [
        "optimizer_update_count", "ppo_surrogate_gradient"]


def test_normal_mapping_and_behavior_is_noop_candidate(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    source = _observation(plan["request"]["checkpoint"]["sha256"], behavior="normal")
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    result = _supplement(tmp_path, plan, source, receipt)
    assert result["effective_action"] == "no_intervention"
    assert result["approval_granted"] is False


@pytest.mark.parametrize("outcome", ["abnormal", "unknown"])
def test_abnormal_or_unknown_mapping_is_conservative_hold(tmp_path: Path, monkeypatch, outcome: str):
    plan, observed = _component(tmp_path, monkeypatch, assessment=(
        "curve_difference" if outcome == "abnormal" else "normal"))
    receipt = observed if outcome == "abnormal" else M.unknown_receipt(plan, "not_executed")
    source = _observation(plan["request"]["checkpoint"]["sha256"], behavior="normal")
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    result = _supplement(tmp_path, plan, source, receipt)
    assert result["status"] == "hold"
    assert result["effective_action"] == "hold"
    assert result["authority_created"] is False
    assert result["api_calls"] == 0


def test_existing_non_actuator_unknown_is_preserved(tmp_path: Path, monkeypatch):
    plan, receipt = _component(tmp_path, monkeypatch)
    source = _observation(plan["request"]["checkpoint"]["sha256"])
    payload = M.governor.build_model_packet(source)["observation"]
    payload["confirmation_checks"].append({"check_id": "optimizer_update_count", "outcome": "unknown",
                                           "refs": ["sha256:" + "6" * 64]})
    source = M.governor._verified_observation(payload)
    monkeypatch.setattr(M.recovery, "sealed_post_observation", lambda *_args: source)
    result = _supplement(tmp_path, plan, source, receipt)
    assert result["effective_action"] == "hold"
    checks = {row["check_id"]: row["outcome"] for row in result["observation"]["confirmation_checks"]}
    assert checks == {"actuator_mapping": "normal", "optimizer_update_count": "unknown"}


def test_cli_prepare_is_write_once(tmp_path: Path):
    request_path, protocol_path, output = tmp_path / "request.json", tmp_path / "protocol.json", tmp_path / "out.json"
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")
    protocol_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    assert M.main(["prepare", "--request", str(request_path), "--protocol", str(protocol_path),
                   "--out", str(output)]) == 0
    assert output.is_file()
    assert M.main(["prepare", "--request", str(request_path), "--protocol", str(protocol_path),
                   "--out", str(output)]) == 2
