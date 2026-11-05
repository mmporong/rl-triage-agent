import argparse
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
C = importlib.import_module("stairs_agent_repair_v2")
TRAIN = importlib.import_module("train_stairs_agent_adaptation_v2")


def source():
    return {"schema": "stairs_agent_source_v1", "observed_symptom": "stability_or_tracking_failure",
            "assessment": {"command_preserved": True, "accepted": False}}


def test_actual_nested_fact_is_accepted_and_invented_leaf_is_rejected():
    s = source()
    p = C.packet(s, [C.probe(C.PROBES[0], s)])
    decision = {"action": C.PROBES[1], "cited_facts": ["source.observed_symptom"], "reason": "자세 확인"}
    assert C.validate_decision(decision, p) == decision
    decision["cited_facts"] = ["source.invented_cause"]
    with pytest.raises(ValueError):
        C.validate_decision(decision, p)


def test_adaptation_requires_verified_receipts_from_both_families_and_host_approval():
    s = source()
    receipts = [C.probe(p, s) for p in C.PROBES]
    decision = {"action": C.ADAPT, "cited_facts": ["P_stairs_task_integrity.result", "P_stairs_stability.result"],
                "reason": "명령은 정상이며 동작 기준은 실패했다"}
    assert C.authorize_adaptation(s, receipts, decision, True)["decision_digest"] == C.digest(decision)
    with pytest.raises(ValueError):
        C.authorize_adaptation(s, receipts, decision, False)
    decision["cited_facts"] = ["P_stairs_stability.result"]
    with pytest.raises(ValueError):
        C.authorize_adaptation(s, receipts, decision, True)


def test_incomplete_and_normal_receipts_cannot_authorize_training():
    s = source()
    assert C.ADAPT not in C.packet(s, [C.probe(C.PROBES[0], s)])["eligible_actions"]
    s["assessment"]["accepted"] = True
    assert C.packet(s, [C.probe(p, s) for p in C.PROBES])["eligible_actions"] == ["HOLD"]


def test_launcher_arguments_are_passed_once_and_original_namespace_is_preserved():
    args = argparse.Namespace(kit_args="", headless=True, device="cuda:0")
    config = TRAIN.launcher_arguments(args)
    assert config["kit_args"] == "--/app/vulkan=false --/app/window/hideUi=true"
    assert config["fast_shutdown"] is False and config["headless"] is True
    assert vars(args) == {"kit_args":"", "headless":True, "device":"cuda:0"}
