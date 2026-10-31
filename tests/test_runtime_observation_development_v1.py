from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import runtime_observation_development_v1 as D


def packet():
    return D.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25},
        input_bindings={role: {"checkpoint_sha256": token * 64, "telemetry_sha256": "c" * 64}
                        for role, token in (("case", "a"), ("reference", "b"))},
        budget_left=3, execution_budget_s=1200)


def runtime():
    facts = {**D.FACT.unavailable_facts("case"), **D.FACT.unavailable_facts("reference")}
    facts.update({"runtime:case:available": True, "runtime:case:updates_observed": 3,
                  "runtime:case:expected_optimizer_steps": 20, "runtime:case:optimizer_steps_min": 4,
                  "runtime:case:optimizer_steps_max": 4, "runtime:case:expected_minibatches": 20,
                  "runtime:case:minibatches_min": 4, "runtime:case:minibatches_max": 4})
    return facts


def test_conditions_differ_only_in_registered_runtime_numeric_facts():
    base = packet()
    baseline = D.condition_packet(base, runtime(), "telemetry_only")
    added = D.condition_packet(base, runtime(), "telemetry_and_runtime")
    without = deepcopy(added)
    without["facts"] = {k: v for k, v in without["facts"].items() if not k.startswith("runtime:")}
    assert baseline == without == base
    assert all(value is None for key, value in added["facts"].items()
               if key.startswith("runtime:reference:") and not key.endswith(":available"))


def test_unregistered_or_preexisting_runtime_facts_reject():
    facts = runtime()
    facts["runtime:case:answer"] = "R1"
    with pytest.raises(ValueError):
        D.condition_packet(packet(), facts, "telemetry_and_runtime")
    base = packet()
    base["facts"]["runtime:case:available"] = True
    with pytest.raises(ValueError):
        D.condition_packet(base, runtime(), "telemetry_only")


def test_real_caller_accepts_runtime_citation_but_never_grants_execution(tmp_path):
    protocol = tmp_path / "view.json"
    protocol.write_bytes(D.RV.canonical(D.RV.prepare_protocol()))
    with D.RV.view_scope(protocol, None, fixture=True):
        value = D.condition_packet(packet(), runtime(), "telemetry_and_runtime")
        snapshot = D.C.V.build_snapshot(value, "cpu_fixture", "model_adaptive",
                                       D.RV.file_digest(protocol), tmp_path / "ledger.jsonl")
        content = {"action": "continue", "probe": "P_optimizer_updates",
                   "fact_ids": ["runtime:case:expected_optimizer_steps", "runtime:case:optimizer_steps_min"]}
        fixture = D.C.CALL.FixtureResponses([{"provider_model": D.C.CALL.MODEL,
            "finish_reason": "stop", "content": json.dumps(content), "wall_s": 0.1}])
        choice, _engine, audit = D.select(snapshot, fixture)
        assert choice["model_status"] == "valid"
        assert audit["request_spec"]["probe"] == "P_optimizer_updates"
        assert audit["request_spec"]["approval_granted"] is False
        assert not (tmp_path / "ledger.jsonl").exists()
