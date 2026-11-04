from __future__ import annotations

import hashlib
import json

import pytest

from rl_triage import intervention_governor as G

BINDING = "b" * 64
TASK = "Go2"
PROTOCOL = "fixed_eval_v1"


def _write(tmp_path, name, value):
    path = tmp_path / name
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _criteria():
    return {
        "schema": "cpu_fixture_evaluation_criteria_v1", "task": TASK, "protocol": PROTOCOL,
        "source_binding_sha256": BINDING, "refs": ["criteria:fixed"],
        "metrics": {
            "fall_rate": {"direction": "max", "absolute": 0.5, "relative": 2.0},
            "mean_survival_s": {"direction": "min", "absolute": 5.0, "relative": 0.5},
            "lin_vel_rmse_mps": {"direction": "max", "absolute": 1.0, "relative": 2.0},
            "yaw_rate_rmse_radps": {"direction": "max", "absolute": 1.0, "relative": 2.0},
        },
    }


def _eval(role, criteria_sha, *, seed, metrics=None, task=TASK, binding=BINDING):
    return {
        "schema": "cpu_fixture_fixed_eval_result_v1", "role": role, "task": task,
        "protocol": PROTOCOL, "evaluation_criteria_sha256": criteria_sha,
        "source_binding_sha256": binding, "seed": seed,
        "metrics": metrics or {"fall_rate": 0.1, "mean_survival_s": 9.0,
                               "lin_vel_rmse_mps": 0.2, "yaw_rate_rmse_radps": 0.2},
        "refs": [f"eval:{role}"],
    }


def _artifact(tmp_path, name, schema, *, binding=BINDING, **values):
    return _write(tmp_path, name, {"schema": schema, "source_binding_sha256": binding,
                                   "refs": [f"receipt:{name}"], **values})


def _sources(tmp_path, *, candidate_metrics=None, reference=True, task=TASK, binding=BINDING,
             check_outcome="normal", cause=False, correction=False):
    criteria = _write(tmp_path, "criteria.json", _criteria())
    candidate = _write(tmp_path, "candidate.json", _eval(
        "candidate", criteria["sha256"], seed=101, metrics=candidate_metrics, task=task, binding=binding))
    reference_ref = None
    if reference:
        reference_ref = _write(tmp_path, "reference.json", _eval(
            "reference", criteria["sha256"], seed=999))
    check = _artifact(tmp_path, "check.json", "cpu_fixture_confirmation_check_v1", binding=binding,
                      check_id="runtime_check", outcome=check_outcome)
    cause_ref = None
    if cause:
        cause_ref = _artifact(tmp_path, "cause.json", "cause_confirmation_receipt_v1", binding=binding,
                              confirmation_id="confirm-1", status="confirmed", mechanism="actuator",
                              registered_correction_id="restore_motor_torque_speed")
    correction_ref = None
    if correction:
        correction_ref = _artifact(tmp_path, "correction.json", "correction_execution_receipt_v1", binding=binding,
                                   correction_id="restore_motor_torque_speed", status="applied")
    return {"schema": G.SOURCE_SCHEMA, "receipt_id": "obs-1", "candidate_eval": candidate,
            "reference_eval": reference_ref, "evaluation_criteria": criteria,
            "confirmation_checks": [check], "cause_confirmation_receipt": cause_ref,
            "correction_receipt": correction_ref}


def _gate(tmp_path, allowed=("no_intervention", "investigate", "correct", "reference_review")):
    ref = _write(tmp_path, "gate.json", {"schema": "host_gate_receipt_v1", "gate_id": "host-1",
                                         "allowed_actions": list(allowed), "refs": ["host:policy"]})
    return G.verify_host_gate_receipt({"schema": G.HOST_SOURCE_SCHEMA, "receipt": ref})


def _decide(tmp_path, sources, model=None, allowed=("no_intervention", "investigate", "correct", "reference_review")):
    return G.decide_intervention(G.verify_observation_receipt(sources), model, _gate(tmp_path, allowed))


def test_normal_bundle_uses_independent_reference_seed_without_claiming_model_success(tmp_path):
    row = _decide(tmp_path, _sources(tmp_path))
    assert row["status"] == "host_no_intervention"
    assert row["effective_action"] == "no_intervention"
    assert row["raw_model_proposal"] is None
    assert row["approval_granted"] is row["authority_created"] is False
    assert row["verification_scope"] == "cpu_fixture_development_only"


def test_obvious_failure_requires_investigation_without_inventing_a_cause(tmp_path):
    metrics = {"fall_rate": 0.8, "mean_survival_s": 4.0,
               "lin_vel_rmse_mps": 1.2, "yaw_rate_rmse_radps": 1.1}
    row = _decide(tmp_path, _sources(tmp_path, candidate_metrics=metrics, check_outcome="abnormal"))
    assert row["status"] == "investigation_required"
    assert row["effective_action"] == "investigate"
    assert row["correctability"] == "unconfirmed"
    assert row["additional_checks"] == ["obtain_cause_confirmation_receipt"]


def test_missing_reference_holds_with_a_machine_consumable_followup(tmp_path):
    row = _decide(tmp_path, _sources(tmp_path, reference=False))
    assert row["status"] == "hold"
    assert row["reason_code"] == "reference_missing"
    assert row["additional_checks"] == ["provide_source_bound_reference"]


@pytest.mark.parametrize("change", ["task", "binding"])
def test_unmatched_reference_lineage_holds_but_independent_seed_does_not(tmp_path, change):
    sources = (_sources(tmp_path, task="OtherTask") if change == "task"
               else _sources(tmp_path, binding="c" * 64))
    row = _decide(tmp_path, sources)
    assert row["status"] == "hold"
    assert row["reason_code"] == "reference_lineage_mismatch"


def test_nonfinite_metric_is_rejected_by_the_registered_verifier(tmp_path):
    metrics = {"fall_rate": float("nan"), "mean_survival_s": 9.0,
               "lin_vel_rmse_mps": 0.2, "yaw_rate_rmse_radps": 0.2}
    with pytest.raises(ValueError, match="finite"):
        G.verify_observation_receipt(_sources(tmp_path, candidate_metrics=metrics))


def test_unknown_confirmation_check_holds_and_names_the_check(tmp_path):
    row = _decide(tmp_path, _sources(tmp_path, check_outcome="unknown"))
    assert row["status"] == "hold"
    assert row["additional_checks"] == ["resolve_check:runtime_check"]


def test_post_correction_normal_checks_and_reference_only_gap_require_reference_review(tmp_path):
    metrics = {"fall_rate": 0.3, "mean_survival_s": 9.0,
               "lin_vel_rmse_mps": 0.2, "yaw_rate_rmse_radps": 0.2}
    row = _decide(tmp_path, _sources(tmp_path, candidate_metrics=metrics, correction=True))
    assert row["status"] == "reference_review_required"
    assert row["effective_action"] == "reference_review"
    assert row["correctability"] == "reference_review_only"
    assert row["approval_granted"] is False


def test_model_role_is_separate_from_host_normal_and_failure_decisions(tmp_path):
    model_correct = {"schema": G.MODEL_SCHEMA, "action": "correct", "reason": "model guess", "refs": []}
    normal = _decide(tmp_path, _sources(tmp_path), model_correct)
    assert normal["raw_model_proposal"]["action"] == "correct"
    assert normal["effective_action"] == "no_intervention"
    metrics = {"fall_rate": 0.8, "mean_survival_s": 4.0,
               "lin_vel_rmse_mps": 1.2, "yaw_rate_rmse_radps": 1.1}
    model_none = {"schema": G.MODEL_SCHEMA, "action": "no_intervention", "reason": "model guess", "refs": []}
    failed = _decide(tmp_path, _sources(tmp_path, candidate_metrics=metrics, check_outcome="abnormal"), model_none)
    assert failed["raw_model_proposal"]["action"] == "no_intervention"
    assert failed["effective_action"] == "investigate"


@pytest.mark.parametrize("model", [None, {"schema": G.MODEL_SCHEMA, "action": "correct",
                                           "reason": "proposal", "refs": []}])
def test_same_host_gate_blocks_model_and_no_agent_paths(model, tmp_path):
    metrics = {"fall_rate": 0.8, "mean_survival_s": 4.0,
               "lin_vel_rmse_mps": 1.2, "yaw_rate_rmse_radps": 1.1}
    row = _decide(tmp_path, _sources(tmp_path, candidate_metrics=metrics, check_outcome="abnormal"),
                  model, allowed=("no_intervention",))
    assert row["host_blocked"] is True
    assert row["effective_action"] == "hold"


def test_confirmed_cause_only_makes_correction_proposal_eligible(tmp_path):
    metrics = {"fall_rate": 0.8, "mean_survival_s": 4.0,
               "lin_vel_rmse_mps": 1.2, "yaw_rate_rmse_radps": 1.1}
    row = _decide(tmp_path, _sources(tmp_path, candidate_metrics=metrics,
                                     check_outcome="abnormal", cause=True))
    assert row["status"] == "proposal_eligible"
    assert row["proposal_eligible"] is True
    assert row["approval_granted"] is row["authority_created"] is False


@pytest.mark.parametrize("hidden_key", ["fault_labels", "benchmark_scoring",
                                         "original_faults", "remaining_faults"])
def test_hidden_benchmark_fields_are_rejected_recursively(tmp_path, hidden_key):
    sources = _sources(tmp_path)
    sources["candidate_eval"][hidden_key] = ["secret"]
    with pytest.raises(ValueError, match="hidden benchmark"):
        G.verify_observation_receipt(sources)


def test_unregistered_verifier_and_fabricated_verified_object_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unregistered"):
        G.verify_observation_receipt(_sources(tmp_path), verifier_id="caller_claimed_healthy")
    with pytest.raises(TypeError):
        G.VerifiedObservation({"healthy": True}, "0" * 64, object())


def test_source_sha_is_checked_before_any_decision(tmp_path):
    sources = _sources(tmp_path)
    sources["candidate_eval"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        G.verify_observation_receipt(sources)


def test_model_packet_keeps_metric_bundle_and_no_execution_authority(tmp_path):
    observation = G.verify_observation_receipt(_sources(tmp_path))
    packet = G.build_model_packet(observation)
    assert packet["execution_authority"] is False
    assert len(packet["observation"]["metric_checks"]) == 4
    assert {row["metric"] for row in packet["observation"]["metric_checks"]} == set(G._METRICS)


def test_mutating_an_opaque_verified_payload_is_detected(tmp_path):
    observation = G.verify_observation_receipt(_sources(tmp_path))
    observation._payload["behavior_outcome"] = "failed"
    with pytest.raises(ValueError, match="mutated"):
        G.decide_intervention(observation, None, _gate(tmp_path))
