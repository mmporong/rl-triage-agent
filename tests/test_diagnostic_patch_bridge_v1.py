# pyright: reportMissingImports=false
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evals"), str(ROOT / "bench")]
import diagnostic_patch_bridge_v1 as B
import ppo_patch_cases_v1 as C

from evals import ppo_patch_feedback_v1 as F


def inspector(config, *, updates):
    raw = json.loads((ROOT / "evals/results/ppo_generated_patch_cpu_20261106_v1/cpu-before.json").read_text(encoding="utf-8"))
    value = copy.deepcopy(raw["records"][0]["measurement"])
    epochs, batches = config.get("num_learning_epochs", 1), config["num_mini_batches"]
    value["input"].update(config=copy.deepcopy(config), updates=updates)
    value["resolved"].update(num_learning_epochs=epochs, num_mini_batches=batches)
    value["resolved"]["num_learning_epochs_source"] = "config" if "num_learning_epochs" in config else "installed_ppo_signature_default"
    m = value["measurement"]
    m.update(updates=updates, minibatches=epochs*batches*updates, optimizer_steps=epochs*batches*updates,
             expected_per_update=epochs*batches, expected_total=epochs*batches*updates, counts_match=True)
    m["per_update"] = [{"index": i+1, "minibatches": epochs*batches, "optimizer_steps": epochs*batches,
                        "return_values_finite": True} for i in range(updates)]
    return value


def inputs(healthy=False):
    source = C.SOURCES["healthy" if healthy else "literal_override"]
    context = B.diagnosis_context(source, F.validation_plan()[0], case_id="fixture")
    choice = {"action": "inspect", "probe": B.PROBE, "source_sha256": context["source_sha256"],
              "fact_ids": ["configured.epochs"], "rationale": "verify configured work"}
    baseline = F.validate_on_cpu(source, inspector=inspector)
    reference = F.validate_on_cpu(C.SOURCES["healthy"], inspector=inspector)
    authorization = {"approved": True, "principal": "user_delegated_host",
                     "scope": "one_cpu_adapter_inspection", "context_sha256": B.digest(context)}
    return context, choice, baseline, reference, authorization


def test_abnormal_confirmation_hands_exact_source_and_facts_to_patch():
    args = inputs()
    receipt = B.bind_confirmation(*args)
    assert receipt["outcome"] == "abnormal" and receipt["patch_eligible"]
    context = B.patch_context(args[0], receipt, args[2], args[3])
    assert context["source"] == args[0]["source"]
    assert context["facts"]["inspection.actual_steps"] == [3, 3, 3]
    assert receipt["GPU_approval_granted"] is False


def test_normal_reference_does_not_authorize_patch():
    args = inputs(healthy=True)
    receipt = B.bind_confirmation(*args)
    assert receipt["outcome"] == "normal" and not receipt["patch_eligible"]
    with pytest.raises(ValueError, match="abnormal"):
        B.patch_context(args[0], receipt, args[2], args[3])


@pytest.mark.parametrize("kind", ["missing_authority", "integer_true", "wrong_source", "wrong_raw", "wrong_fact"])
def test_binding_or_raw_tampering_is_rejected(kind):
    context, choice, baseline, reference, auth = inputs()
    if kind == "missing_authority":
        auth["approved"] = False
    elif kind == "integer_true":
        auth["approved"] = 1
    elif kind == "wrong_source":
        baseline["source_sha256"] = "0"*64
    elif kind == "wrong_raw":
        reference["records"][0]["measurement"]["measurement"]["per_update"][0]["optimizer_steps"] = 0
    else:
        choice["fact_ids"] = ["invented"]
    with pytest.raises(ValueError):
        B.bind_confirmation(context, choice, baseline, reference, auth)


def test_legacy_path_consumes_existing_verifier_without_registered_correction_execution():
    seen = []
    def verifier(result, applied):
        seen.append((result, applied))
        return "optimizer_update_count", "restore_configured_ppo_epochs"
    value = B.verify_legacy_diagnosis({"fixture": True}, [], verifier=verifier)
    assert seen == [({"fixture": True}, [])]
    assert not value["registered_correction_executed"] and not value["execution_allowed"]


def runtime_fixture():
    root = ROOT / "evals/results/ppo_generated_patch_behavior_s5750_20261106_v1"
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    protocol = read(ROOT / "bench/protocols/generated_ppo_patch_behavior_s5750_20261106_v3.json")
    pair = {f"{role}_training": read(root / "training" / role / "training-result.json")
            for role in ("healthy", "fault")}
    pair.update({f"{role}_fixed": read(root / "fixed" / f"{role}.json") for role in ("healthy", "fault")})
    return {"pairs": [pair], "fixed_protocol_sha256_lf": protocol["fixed_eval"]["sha256_lf"],
            "bands": protocol["fixed_eval"]["bands"]}


def test_actual_robot_failure_is_bound_before_model_followup():
    runtime = runtime_fixture()
    context = B.runtime_diagnosis_context(C.SOURCES["literal_override"], F.validation_plan()[0],
                                          case_id="observed-fixture", runtime=runtime)
    assert context["facts"]["runtime.seed5750.actual_optimizer_steps"] == [4, 4, 4]
    assert context["facts"]["runtime.seed5750.fault_tracking_RMSE"] > 1
    assert context["runtime_binding_sha256"] == B.digest(runtime)


@pytest.mark.parametrize("mode", ["checkpoint", "healthy_fault", "source", "counts"])
def test_runtime_substitution_or_absent_failure_is_rejected(mode):
    runtime = runtime_fixture()
    pair = runtime["pairs"][0]
    if mode == "checkpoint":
        pair["fault_fixed"]["checkpoint"]["sha256"] = "0"*64
    elif mode == "healthy_fault":
        pair["fault_fixed"]["metrics"] = copy.deepcopy(pair["healthy_fixed"]["metrics"])
    elif mode == "source":
        pair["fault_training"]["adapter_source_sha256"] = "0"*64
    else:
        pair["fault_training"]["counts"]["optimizer_steps"] = 6000
    with pytest.raises(ValueError):
        B.runtime_diagnosis_context(C.SOURCES["literal_override"], F.validation_plan()[0],
                                    case_id="observed-fixture", runtime=runtime)
