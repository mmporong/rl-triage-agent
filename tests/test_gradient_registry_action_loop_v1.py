# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import gradient_hypothesis_contract_v1 as GH
import gradient_registry_action_loop_v1 as LOOP
import gradient_registry_action_options_v1 as O
import recorded_ppo_gradient_probe_v1 as REC

RUNS = ROOT / "evals/results/ppo_gradient_v5_20261017/runs"


@pytest.fixture()
def pair(tmp_path: Path) -> dict:
    endpoints = {}
    for role, run in (("case", "inverse_ratio_s2144_on"), ("reference", "none_s2144_on")):
        plan, receipt = tmp_path / role / "plan.json", tmp_path / role / "receipt.json"
        REC.prepare(RUNS / run, request_id=f"ri310-{role}", out=plan)
        REC.execute(plan, out=receipt)
        endpoints[role] = {"role": role, "plan": plan, "receipt": receipt}
    return {"schema": GH.PAIR_SCHEMA, "subtype_schema": GH.SUBTYPE_SCHEMA, **endpoints}


def provider_factory(sequence: list[str], calls: list[str]):
    def factory(slot: str, state: dict):
        index = len(calls)
        target = sequence[index]

        def complete(messages: list, params: dict) -> dict:
            del messages, params
            calls.append(slot)
            if target == "hold":
                response = {"option_id": "hold", "fact_ids": []}
            else:
                context = O.context(state)
                option = next(row for row in context["action_options"] if row["target"] == target)
                response = {"option_id": option["option_id"],
                            "fact_ids": option["required_fact_ids"]}
            return {"provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
                    "finish_reason": "stop", "content": json.dumps(response),
                    "provider_kind": "fixture"}
        return complete
    return factory


def run_fixture(tmp_path: Path, binding: dict, sequence: list[str], *,
                consume=LOOP.consume_immutable_pair) -> tuple[dict, list[str]]:
    run_id = f"fixture-{tmp_path.name}"
    state = GH.initialize(binding, 2)
    calls: list[str] = []
    result = LOOP.run_epoch(
        state, run_id=run_id, out=tmp_path / "epoch", verify_context=lambda: True,
        complete_factory=provider_factory(sequence, calls), consume_pair=consume)
    return result, calls


def test_actual_s2144_pair_count_then_gradient_confirms_surrogate(
    tmp_path: Path, pair: dict,
) -> None:
    result, calls = run_fixture(tmp_path, pair, [GH.COUNT_PROBE, GH.GRADIENT_PROBE])
    assert result["decision_outcome"] == "confirmed"
    assert result["confirmed_subtype"] == "optimizer_surrogate"
    assert result["model_actions"] == result["recorded_pair_consumptions"] == 2
    assert result["scope"] == {
        "fixture_provider": True, "actual_model": False, "new_observation": False,
        "new_training": False, "gpu": False, "correction": False, "recovery": False,
    }
    assert len(calls) == 2
    first = result["steps"][0]["hypotheses_after"]
    assert "optimizer_update_count" not in first
    assert "optimizer_surrogate" in first
    assert result["steps"][1]["hypotheses_after"] == ["optimizer_surrogate"]


def test_hold_and_invalid_provider_stop_without_consumption(tmp_path: Path, pair: dict) -> None:
    result, calls = run_fixture(tmp_path / "hold", pair, ["hold"])
    assert result["decision_outcome"] == "HOLD"
    assert result["recorded_pair_consumptions"] == 0
    assert len(calls) == 1

    state = GH.initialize(pair, 2)
    provider_calls = []
    def factory(_slot, _state):
        def complete(_messages, _params):
            provider_calls.append(1)
            return {"provider_model": "wrong", "finish_reason": "stop", "content": "{}",
                    "provider_kind": "fixture"}
        return complete
    result = LOOP.run_epoch(state, run_id="bad-provider", out=tmp_path / "invalid",
                            verify_context=lambda: True, complete_factory=factory,
                            consume_pair=lambda *_args: pytest.fail("must not consume"))
    assert result["stop_reason"] == "model_invalid_hold"
    assert result["recorded_pair_consumptions"] == 0
    assert provider_calls == [1]


def test_exact_two_keys_and_registry_fact_ids_are_required(pair: dict) -> None:
    state = GH.initialize(pair, 2)
    context = O.context(state)
    option = next(row for row in context["action_options"] if row["target"] == GH.COUNT_PROBE)
    missing = json.dumps({"option_id": option["option_id"],
                          "fact_ids": option["required_fact_ids"][:-1]})
    assert O.assess(missing, state)["codes"] == ["S_FACT_IDS_EXACT"]
    reordered = json.dumps({"option_id": option["option_id"],
                            "fact_ids": list(reversed(option["required_fact_ids"]))})
    reordered_review = O.assess(reordered, state)
    assert reordered_review["status"] == "valid"
    assert reordered_review["action"]["fact_ids"] == list(reversed(option["required_fact_ids"]))
    extra = json.dumps({"option_id": option["option_id"],
                        "fact_ids": option["required_fact_ids"], "cause": "count"})
    assert O.assess(extra, state)["codes"] == ["F_OPTION_FIELDS"]
    review = O.assess(json.dumps({"option_id": option["option_id"],
                                  "fact_ids": option["required_fact_ids"]}), state)
    review["context_digest"] = "0" * 64
    with pytest.raises(ValueError, match="review"):
        O.select_model(review, state)


def test_legacy_or_resealed_initial_state_is_rejected_before_output(tmp_path: Path, pair: dict) -> None:
    state = GH.initialize(pair, 2)
    legacy = dict(state)
    legacy["subtype_schema"] = "legacy_7_subtype_v1"
    with pytest.raises(ValueError, match="7-subtype"):
        LOOP.run_epoch(legacy, run_id="legacy", out=tmp_path / "legacy",
                       verify_context=lambda: True, complete_factory=lambda *_: None,
                       consume_pair=lambda *_: None)
    assert not (tmp_path / "legacy").exists()

    tampered = GH._finish({**{key: value for key, value in state.items() if key != "state_digest"},
                           "hypotheses": ["optimizer_surrogate"]})
    with pytest.raises(ValueError, match="재구성"):
        LOOP.run_epoch(tampered, run_id="tampered", out=tmp_path / "tampered",
                       verify_context=lambda: True, complete_factory=lambda *_: None,
                       consume_pair=lambda *_: None)


def test_unknown_reference_stops_without_second_model_call(tmp_path: Path, pair: dict) -> None:
    swapped = {**pair, "case": pair["reference"],
               "reference": {"role": "reference", "plan": pair["case"]["plan"],
                             "receipt": pair["case"]["receipt"]}}
    swapped["case"] = {"role": "case", "plan": pair["reference"]["plan"],
                       "receipt": pair["reference"]["receipt"]}
    result, calls = run_fixture(tmp_path, swapped, [GH.GRADIENT_PROBE])
    assert result["decision_outcome"] == "HOLD"
    assert result["stop_reason"] == "unknown_or_reference_not_normal"
    assert result["recorded_pair_consumptions"] == 1
    assert len(calls) == 1


def test_two_normal_probes_end_as_budget_exhausted_not_unknown(
    tmp_path: Path, pair: dict,
) -> None:
    normal = {
        "schema": GH.PAIR_SCHEMA, "subtype_schema": GH.SUBTYPE_SCHEMA,
        "case": {"role": "case", "plan": pair["reference"]["plan"],
                 "receipt": pair["reference"]["receipt"]},
        "reference": pair["reference"],
    }
    result, calls = run_fixture(tmp_path, normal, [GH.COUNT_PROBE, GH.GRADIENT_PROBE])
    assert result["decision_outcome"] == "HOLD"
    assert result["stop_reason"] == "budget_exhausted_or_multiple_hypotheses"
    assert result["recorded_pair_consumptions"] == 2
    assert len(calls) == 2
    final = json.loads((tmp_path / "epoch/final-state.json").read_text(encoding="utf-8"))
    assert [row["outcome"] for row in final["history"]] == ["normal", "normal"]
    assert final["status"] == "hold"


def test_tampered_approval_or_reused_proof_is_rejected(tmp_path: Path, pair: dict) -> None:
    def tamper(state, action, request, out):
        proof = LOOP.consume_immutable_pair(state, action, request, out)
        path = Path(proof["approval"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["actor"] = "other"
        path.write_text(json.dumps(value), encoding="utf-8")
        return proof
    with pytest.raises(ValueError, match="approval"):
        run_fixture(tmp_path / "tamper", pair, [GH.COUNT_PROBE], consume=tamper)

    first_proof = None
    def reuse(state, action, request, out):
        nonlocal first_proof
        if first_proof is None:
            first_proof = LOOP.consume_immutable_pair(state, action, request, out)
        return first_proof
    with pytest.raises(ValueError, match="probe output"):
        run_fixture(tmp_path / "reuse", pair, [GH.COUNT_PROBE, GH.GRADIENT_PROBE], consume=reuse)


def test_raw_tamper_after_initialization_is_rejected(tmp_path: Path, pair: dict) -> None:
    state = GH.initialize(pair, 2)
    receipt = Path(pair["case"]["receipt"])
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["outcome"] = "normal"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        LOOP.run_epoch(state, run_id="raw-tamper", out=tmp_path / "epoch",
                       verify_context=lambda: True, complete_factory=lambda *_: None,
                       consume_pair=lambda *_: None)
