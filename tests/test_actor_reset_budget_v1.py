from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from evals import actor_reset_budget_v1 as B
from bench import train_actor_reset_budget_v1 as CHILD


def test_frozen_budget_contract_and_predictions():
    assert B.SEEDS == (2090, 2091)
    assert B.RESET_BUDGETS == (100, 150) and B.CONDITIONAL_BUDGET == 200
    assert B.RESTART_BUDGET == 300 and B.FIXED["automatic_global_change"] is False
    assert B.FIXED["prediction"] == {
        "reset_100": ["not_recovered", "not_recovered"],
        "reset_150": ["recovered", "recovered"],
        "reset_200_if_needed": ["recovered", "recovered"],
        "restart_300": ["recovered", "recovered"],
    }


@pytest.mark.parametrize(("mode", "budget", "arm", "final"), [
    ("reset", 100, "actor_last_layer_reset_100", 398),
    ("reset", 150, "actor_last_layer_reset_150", 448),
    ("reset", 200, "actor_last_layer_reset_200", 498),
    ("restart", 300, "restart_300", 299),
])
def test_budget_names_and_off_by_one_final_iterations(mode, budget, arm, final, tmp_path, monkeypatch):
    folder = tmp_path / B.TAG
    checkpoint = tmp_path / "logs/run/model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source_path = folder / "source/s2090/source_evidence.json"
    source_path.parent.mkdir(parents=True)
    source = {"F1": {"checkpoint": str(checkpoint), "checkpoint_sha256": "a" * 64,
                     "environment": {"path": str(tmp_path / "env.yaml"), "sha256": "b" * 64},
                     "agent": {"path": str(tmp_path / "agent.yaml"), "sha256": "c" * 64}}}
    source_path.write_text(json.dumps(source), encoding="utf-8")
    (folder / "source_selection.json").write_text(json.dumps({"seeds": {"2090": {
        "source_evidence_sha256": B.sha(source_path), "checkpoint_sha256": "a" * 64}}}), encoding="utf-8")
    monkeypatch.setattr(B, "PROTOCOL", tmp_path / "protocol.json")
    request = {"source_commit_sha": "d" * 40, "protocol_sha256_lf": "e" * 64,
               "isaaclab_root": str(tmp_path / "IsaacLab")}
    assert B.arm_name(mode, budget) == arm
    manifest = B.manifest_for(folder, request, 2090, arm)
    assert manifest["max_iterations"] == budget and manifest["final_iteration"] == final
    assert R_validate(manifest) == ("resume" if mode == "reset" else "restart")
    assert (manifest["resume_checkpoint_path"] is not None) is (mode == "reset")


def R_validate(manifest):
    return B.R.validate_manifest_mode(manifest)


@pytest.mark.parametrize(("labels", "expected"), [
    ({"100": {"2090": "recovered", "2091": "recovered"},
      "150": {"2090": "not_recovered", "2091": "not_recovered"}}, False),
    ({"100": {"2090": "recovered", "2091": "not_recovered"},
      "150": {"2090": "not_recovered", "2091": "recovered"}}, True),
    ({"100": {"2090": "not_recovered", "2091": "not_recovered"},
      "150": {"2090": "not_recovered", "2091": "not_recovered"}}, True),
])
def test_conditional_200_branch_requires_no_common_two_of_two_budget(labels, expected):
    assert B.should_run_200(labels) is expected


def test_remaining_arm_order_runs_conditional_200_before_restart():
    split = {"100": {"2090": "recovered", "2091": "not_recovered"},
             "150": {"2090": "not_recovered", "2091": "recovered"}}
    common = {"100": {"2090": "recovered", "2091": "recovered"},
              "150": {"2090": "not_recovered", "2091": "not_recovered"}}
    assert B.remaining_arm_order(split) == ("200", "restart_300")
    assert B.remaining_arm_order(common) == ("restart_300",)


def test_result_label_gate_rejects_extra_seed_and_unknown_label():
    labels = {"100": {"2090": "not_recovered", "2091": "recovered"},
              "150": {"2090": "recovered", "2091": "recovered"},
              "restart_300": {"2090": "recovered", "2091": "recovered"}}
    assert B.validate_result_labels(labels) == labels
    extra_seed = json.loads(json.dumps(labels))
    extra_seed["100"]["9999"] = "recovered"
    with pytest.raises(ValueError, match="seed/값"):
        B.validate_result_labels(extra_seed)
    unknown = json.loads(json.dumps(labels))
    unknown["150"]["2091"] = "invalid"
    with pytest.raises(ValueError, match="seed/값"):
        B.validate_result_labels(unknown)


def test_candidate_gate_uses_minimum_common_budget_restart_and_provenance():
    labels = {"100": {"2090": "not_recovered", "2091": "not_recovered"},
              "150": {"2090": "recovered", "2091": "recovered"},
              "restart_300": {"2090": "recovered", "2091": "recovered"}}
    assert B.candidate_decision(labels, True) == ("conditional_candidate", 150)
    labels["restart_300"]["2091"] = "not_recovered"
    assert B.candidate_decision(labels, True) == ("restart_control_not_recovered", None)
    labels["restart_300"]["2091"] = "recovered"
    assert B.candidate_decision(labels, False) == ("keep_current_rule", None)


@pytest.mark.parametrize(("labels", "expected"), [
    ({"100": {"2090": "recovered", "2091": "recovered"},
      "150": {"2090": "recovered", "2091": "recovered"},
      "restart_300": {"2090": "recovered", "2091": "recovered"}},
     ("conditional_candidate", 100)),
    ({"100": {"2090": "not_recovered", "2091": "not_recovered"},
      "150": {"2090": "recovered", "2091": "not_recovered"},
      "200": {"2090": "recovered", "2091": "recovered"},
      "restart_300": {"2090": "recovered", "2091": "recovered"}},
     ("conditional_candidate", 200)),
])
def test_candidate_gate_selects_minimum_common_100_or_conditional_200(labels, expected):
    assert B.candidate_decision(labels, True) == expected


def _initial(value="a"):
    return {"policy_state": {"sha256": value * 64}, "optimizer_state": {"sha256": "b" * 64},
            "rng_state_before_learn": {"torch": "c" * 64}}


def test_initial_equivalence_requires_independent_reset_arms_to_match():
    rows = {"100": {"initial": _initial()}, "150": {"initial": _initial()}}
    assert B.validate_initial_equivalence(rows)["status"] == "identical"
    rows["150"]["initial"] = _initial("d")
    with pytest.raises(ValueError, match="초기 policy"):
        B.validate_initial_equivalence(rows)


def test_wall_receipt_rejects_wrong_scope_and_non_finite_values():
    valid = {"wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification", "wall_s": 1.25}
    assert B.validated_wall_s(valid, "test") == 1.25
    for invalid in ({**valid, "wall_s": float("inf")}, {**valid, "wall_s": float("nan")},
                    {**valid, "wall_s": True}, {**valid, "wall_scope": "child_only"}):
        with pytest.raises(ValueError, match="벽시계"):
            B.validated_wall_s(invalid, "test")


def test_result_document_recomputes_predictions_and_evidence_hashes(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(B, "PROTOCOL", protocol_path)
    evidence = tmp_path / "receipt.json"
    evidence.write_text('{"value": 1}', encoding="utf-8")
    labels = {"100": {"2090": "not_recovered", "2091": "not_recovered"},
              "150": {"2090": "recovered", "2091": "recovered"},
              "restart_300": {"2090": "recovered", "2091": "recovered"}}
    costs = {"recovery_arms": {key: {str(seed): {"training_wall_s": float(index + 1)}
                                      for seed in B.SEEDS}
                                for index, key in enumerate(labels)}}
    protocol = {"prediction": B.FIXED["prediction"], "budget_unit": B.FIXED["budget_unit"],
                "candidate_wording": B.FIXED["candidate_wording"],
                "restart_control_not_recovered_wording": B.FIXED["restart_control_not_recovered_wording"],
                "keep_wording": B.FIXED["keep_wording"], "time_scope": B.FIXED["time_scope"],
                "limits": B.FIXED["limits"]}
    first = B.result_document(tmp_path, protocol, "a" * 40, {"commit": "a" * 40},
                              labels, {key: {} for key in labels}, costs,
                              {"2090": True, "2091": False}, True)
    assert first["prediction_match"]["100"] == {"2090": True, "2091": True}
    assert first["prediction_match"]["150"] == {"2090": True, "2091": True}
    assert first["restart_matches_source_none_sha"] == {"2090": True, "2091": False}
    assert first["training_wall_ratio_vs_same_seed_restart"]["2090"] == {"100": 1 / 3, "150": 2 / 3}
    assert "150회" in first["wording"]
    restart_failed = json.loads(json.dumps(labels))
    restart_failed["restart_300"]["2091"] = "not_recovered"
    undecided = B.result_document(tmp_path, protocol, "a" * 40, {"commit": "a" * 40},
                                  restart_failed, {key: {} for key in labels}, costs,
                                  {"2090": True, "2091": False}, True)
    assert undecided["decision"] == "restart_control_not_recovered"
    assert undecided["wording"] == B.FIXED["restart_control_not_recovered_wording"]
    evidence.write_text('{"value": 2}', encoding="utf-8")
    replay = B.result_document(tmp_path, protocol, "a" * 40, {"commit": "a" * 40},
                               labels, {key: {} for key in labels}, costs,
                               {"2090": True, "2091": False}, True)
    assert replay["evidence_sha256"] != first["evidence_sha256"]


def test_intervention_rejects_critic_or_earlier_actor_change():
    actor_before = {"actor.0": "1" * 64, "actor.2": "2" * 64}
    actor_after = {**actor_before, "actor.2": "3" * 64}
    row = {"schema": "actor_reset_budget_intervention_v1", "arm": "actor_last_layer_reset_100",
           "intervention": "actor_last_layer_reset", "reset_seed": 122090,
           "target_modules": ["actor.2"], "target_parameter_count": 2,
           "optimizer_param_indices": [7, 8], "optimizer_moment_indices_present_before": [7, 8],
           "optimizer_moments_present_after": False, "parameter_identity_preserved": True,
           "policy_state_before": "4" * 64, "policy_state_after": "5" * 64,
           "optimizer_state_before": "6" * 64, "optimizer_state_expected_after": "7" * 64,
           "optimizer_state_after": "7" * 64, "optimizer_param_groups_before": "8" * 64,
           "optimizer_param_groups_after": "8" * 64, "critic_state_before": "9" * 64,
           "critic_state_after": "9" * 64, "std_state_before": "a" * 64, "std_state_after": "a" * 64,
           "actor_linear_states_before": actor_before, "actor_linear_states_after": actor_after}
    manifest = {"arm": "actor_last_layer_reset_100", "seed": 2090, "recovery_mode": "resume"}
    source = {"policy_state_sha256": row["policy_state_before"],
              "optimizer_state_sha256": row["optimizer_state_before"]}
    initial = {"policy_state": {"sha256": row["policy_state_after"]},
               "optimizer_state": {"sha256": row["optimizer_state_after"]}}
    B.validate_intervention(row, manifest, source, initial)
    initial["policy_state"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        B.validate_intervention(row, manifest, source, initial)
    initial["policy_state"]["sha256"] = row["policy_state_after"]
    row["critic_state_after"] = "0" * 64
    with pytest.raises(ValueError):
        B.validate_intervention(row, manifest, source, initial)


def test_freshness_and_invalid_result_inputs_fail_closed():
    row = {"schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": [2090, 2091],
           "checked_at_utc": "2026-10-16T00:00:00+00:00",
           "checked_roots": ["<repo>/bench/**/*.telemetry.json", "<repo>/bench/private/execution_records",
                             "<repo>/evals/results", "<repo>/evals/loops", "<isaaclab>/logs"],
           "checked_file_count": 5169, "evidence": [],
           "scope": "현재 저장소의 실행 기록과 지정 Isaac Lab 로그에 한정한다"}
    assert B.validate_freshness(row) == row
    row["seeds"] = [2090]
    with pytest.raises(ValueError):
        B.validate_freshness(row)
    with pytest.raises(ValueError):
        B.arm_spec("actor_last_layer_reset_300")


def test_protocol_has_no_personal_absolute_path_if_prepared():
    if not B.PROTOCOL.is_file():
        pytest.skip("prepare 전")
    text = B.PROTOCOL.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text and "/home/" not in text and "/mnt/c/Users/" not in text


def _v4_test_helpers():
    path = B.ROOT / "tests/test_training_correction_v4.py"
    spec = importlib.util.spec_from_file_location("task25_v4_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("arm", "budget", "final", "loads"), [
    ("resume_corrected", 100, 398, 1),
    ("resume_corrected", 150, 448, 1),
    ("resume_corrected", 200, 498, 1),
    ("restart_corrected", 300, 299, 0),
])
def test_shared_training_harness_receipt_supports_each_budget(
        monkeypatch, tmp_path, arm, budget, final, loads):
    helpers = _v4_test_helpers()
    manifest = helpers._child_manifest(tmp_path, arm, seed=2090)
    manifest.update({
        "max_iterations": budget,
        "final_iteration": final,
        "run_name": f"task25_{arm}_{budget}",
        "output_dir": str((tmp_path / f"private/{arm}_{budget}/train").resolve()),
    })
    events = []
    helpers._install_child_fakes(monkeypatch, tmp_path, manifest, events)
    manifest_path = Path(manifest["output_dir"]) / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    receipt = B.R.execute(manifest_path, helpers.FullH(manifest, helpers._protocol(), events))

    assert receipt["checkpoint_load_calls"] == loads
    assert receipt["start_iteration"] == (299 if loads else 0)
    assert receipt["final_iteration"] == final
    assert receipt["actual_control_steps"] == budget * B.NUM_STEPS
    assert receipt["aggregate_env_steps"] == budget * B.NUM_STEPS * B.NUM_ENVS
    assert receipt["budget"]["max_iterations"] == budget
    assert receipt["budget"]["expected_control_steps"] == budget * B.NUM_STEPS


def _strict_receipt_fixture(monkeypatch, tmp_path, *, resume=False):
    helpers = _v4_test_helpers()
    harness_arm = "resume_corrected" if resume else "restart_corrected"
    manifest = helpers._child_manifest(tmp_path, harness_arm, seed=2090)
    arm = "actor_last_layer_reset_100" if resume else "restart_300"
    manifest.update({"arm": arm, "max_iterations": 100 if resume else 300,
                     "final_iteration": 398 if resume else 299,
                     "run_name": f"task25_{arm}",
                     "output_dir": str((tmp_path / f"private/{arm}/train").resolve())})
    events = []
    helpers._install_child_fakes(monkeypatch, tmp_path, manifest, events)
    training = Path(manifest["output_dir"])
    manifest_path = training / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol = helpers._protocol()
    protocol["runtime_versions"] = {
        "torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "isaaclab": "0.41.3"}
    receipt = B.R.execute(manifest_path, helpers.FullH(manifest, protocol, events))
    initial = json.loads((training / "initial.json").read_text(encoding="utf-8"))
    return training, manifest, protocol, receipt, initial


@pytest.mark.parametrize("mutation", [
    "extra_field", "tag", "budget", "resume_scope", "config_sha", "initial_schema", "rng", "state_iteration",
])
def test_strict_receipt_contract_rejects_identity_and_provenance_tampering(
        monkeypatch, tmp_path, mutation):
    training, manifest, protocol, receipt, initial = _strict_receipt_fixture(monkeypatch, tmp_path)
    if mutation == "extra_field":
        receipt["unexpected"] = True
    elif mutation == "tag":
        receipt["tag"] = "other"
    elif mutation == "budget":
        receipt["budget"]["max_iterations"] = 299
    elif mutation == "resume_scope":
        receipt["resume_scope"]["fresh_initialization"] = False
    elif mutation == "config_sha":
        receipt["configs"]["agent"]["sha256"] = "f" * 64
    elif mutation == "initial_schema":
        initial["schema"] = "other"
    elif mutation == "rng":
        initial["rng_state_before_learn"]["cpu_sha256"] = "f" * 64
    else:
        receipt["final_checkpoint_state"]["iteration"] = 298
    with pytest.raises(ValueError):
        B.validate_receipt_contract(training, manifest, protocol, receipt, initial)


@pytest.mark.parametrize("resume", [False, True])
def test_strict_receipt_contract_accepts_unmodified_harness_output(monkeypatch, tmp_path, resume):
    training, manifest, protocol, receipt, initial = _strict_receipt_fixture(
        monkeypatch, tmp_path, resume=resume)
    checkpoint = B.validate_receipt_contract(training, manifest, protocol, receipt, initial)
    assert checkpoint.name == ("model_398.pt" if resume else "model_299.pt")


def test_child_adapter_restores_shared_hook_after_failure(monkeypatch, tmp_path):
    manifest = {"arm": "actor_last_layer_reset_100", "seed": 2090,
                "output_dir": str(tmp_path / "train")}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_install = object()
    monkeypatch.setattr(CHILD.R, "_install_runner_hooks", original_install)

    def fail_after_adapter_install(path, helper):
        assert CHILD.R._install_runner_hooks is not original_install
        raise RuntimeError("synthetic child failure")

    monkeypatch.setattr(CHILD.R, "execute", fail_after_adapter_install)
    with pytest.raises(RuntimeError, match="synthetic child failure"):
        CHILD.execute(manifest_path)
    assert CHILD.R._install_runner_hooks is original_install
