# pyright: reportMissingImports=false
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from evals import run_prospective_reasoning_confirmation_v1 as R

SOURCE_PROTOCOL = R.ROOT / "bench/protocols/prospective_diagnosis_training_20261101_v1.json"


def _state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    ledger = tmp_path / "facts.json"
    ledger.write_text("facts", encoding="utf-8")
    facts = {}
    for role, observed in (("case", 4), ("reference", 20)):
        prefix = f"runtime:{role}:"
        facts.update({prefix + "available": True, prefix + "updates_observed": 3,
                      prefix + "expected_optimizer_steps": 20,
                      prefix + "optimizer_steps_min": observed,
                      prefix + "optimizer_steps_max": observed,
                      prefix + "expected_minibatches": 20,
                      prefix + "minibatches_min": observed,
                      prefix + "minibatches_max": observed,
                      f"derived:{role}:step_count_matches_config": observed == 20,
                      f"derived:{role}:minibatch_count_matches_config": observed == 20})
    checked = {"source_binding": {
        "case": {"checkpoint_sha256": "a" * 64, "binding_digest": "c" * 64},
        "reference": {"checkpoint_sha256": "b" * 64, "binding_digest": "d" * 64}},
        "model_view": {"facts": facts}}
    monkeypatch.setattr(R.SEL.LEDGER, "checked", lambda _path: deepcopy(checked))
    packet = {"facts": {"telemetry:case:mean_reward": 1.0},
              "remaining_candidates": list(R.SEL.S.C1.registry()["mechanisms"]),
              "observed": {}, "budget_left": 3,
              "input_bindings": {"case": {"checkpoint_sha256": "a" * 64},
                                   "reference": {"checkpoint_sha256": "b" * 64}}}
    monkeypatch.setattr(R.SEL.S, "validate_packet", lambda value: deepcopy(value))
    return R.SEL.initialize(packet, ledger)


def test_prepare_design_fixes_first_call_modes_budgets_and_rule() -> None:
    design = R.prepare_design(SOURCE_PROTOCOL)
    assert design["seeds"] == [2320, 2321]
    assert design["categories"] == ["NONE", "F1", "R1_ONLY", "COMPOUND"]
    assert len(design["bridge"]["logical_slots"]) == 96
    assert design["budgets"] == {"model_actions_per_row": 4, "paired_probes_per_row": 3,
                                  "api_requests": 96, "paired_probes": 54,
                                  "sdk_role_executions": 108, "wall_s": 14400}
    assert design["rule_policy"] == ["read_evidence", "P_optimizer_updates",
                                      "if_count_normal_then_P_reward"]
    assert design["tool_protocols"]["view"]["path"].endswith("runtime_observation_view_v1.json")


def test_stage_a_closure_includes_model_and_input_dependencies() -> None:
    design = R.prepare_design(SOURCE_PROTOCOL)
    required = {"evals/registry_action_options_v1.py",
                "evals/nemotron_reasoning_options_study_v1.py",
                "evals/selection_comparison_call_v2.py",
                "evals/feedback_model_call.py",
                "src/rl_triage/probe_loop.py",
                "evals/prospective_diagnosis_inputs_v2.py",
                "evals/runtime_observation_view_v1.py"}
    assert required <= set(design["source_sha256_lf"])


@pytest.mark.parametrize("mode", ["legacy_none", "enabled_low"])
def test_first_state_starts_with_selected_mode_and_holds_without_probe(
        mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path, monkeypatch)
    seen = []

    def factory(slot: str, state: dict):
        def complete(messages: list, params: dict) -> dict:
            seen.append({"slot": slot, "state": deepcopy(state), "messages": messages,
                         "params": deepcopy(params)})
            result = {"provider_model": R.BRIDGE.MODE.MODEL, "finish_reason": "stop",
                      "content": '{"option_id":"hold","fact_ids":[]}',
                      "request_params": deepcopy(params), "wall_s": 0.01,
                      "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "reasoning_tokens": 0, "reasoning_tokens_status": "reported"}
            root = tmp_path / "transport" / mode
            index = len(list((root / "requests").glob("*.json"))) + 1
            request = root / "requests" / f"{index:03d}.json"
            R._write_new(request, {"schema": R.BRIDGE.REQUEST_SCHEMA, "index": index,
                "slot": slot, "case_id": slot.split(".", 1)[0], "mode": mode,
                "messages": deepcopy(messages), "messages_sha256": R.BRIDGE.digest(messages),
                "params": deepcopy(params), "params_sha256": R.BRIDGE.digest(params),
                "bridge_protocol_sha256": "a" * 64,
                "context_sha256": R.BRIDGE.digest({
                    "slot": slot, "messages": messages, "params": params})})
            response = root / "responses" / request.name
            R._write_new(response, {"schema": R.BRIDGE.RESPONSE_SCHEMA,
                                    "request_sha256": R._sha(request), "result": result})
            return {**result, "transport_ref": {"path": str(response.resolve()),
                                                  "sha256": R._sha(response)}}
        return complete

    result = R._run_model_epoch(
        state, run_id=f"c02.{mode}", mode=mode,
        out=tmp_path / mode, verify_context=lambda: True, complete_factory=factory,
        execute_pair=lambda *_args, **_kwargs: pytest.fail("HOLD는 probe를 실행하지 않는다"))
    assert result["decision_outcome"] == "HOLD" and result["model_requests"] == 1
    assert seen[0]["slot"] == f"c02.{mode}.d1.a1"
    expected = R.BRIDGE.MODE.build_request(seen[0]["messages"], mode)
    expected.pop("messages")
    assert seen[0]["params"] == expected


def test_assessment_keeps_not_confirmed_and_indeterminate() -> None:
    rows = []
    for case in range(1, 9):
        for policy in R.POLICIES:
            rows.append({"case_id": f"c{case:02d}", "policy": policy, "status": "complete",
                         "model_requests": 0, "paired_probes": 0,
                         "wrong_family_confirmation": False,
                         "correct_confirmed_family": False})
    assert R._assessment(rows) == "not_confirmed"
    rows[0]["status"] = "invalid"
    assert R._assessment(rows) == "indeterminate"


def test_compound_partial_is_correct_and_other_family_is_wrong() -> None:
    reward = R._score("COMPOUND", {"decision_outcome": "confirmed",
                                   "confirmed_family": "reward"})
    physics = R._score("COMPOUND", {"decision_outcome": "confirmed",
                                    "confirmed_family": "physics"})
    assert reward == {"correct_confirmed_family": True,
                      "wrong_family_confirmation": False,
                      "compound_partial": True, "compound_complete": False}
    assert physics["correct_confirmed_family"] is False
    assert physics["wrong_family_confirmation"] is True


def test_provider_failure_keeps_aggregate_indeterminate() -> None:
    rows = []
    for case in range(1, 9):
        for policy in R.POLICIES:
            rows.append({"case_id": f"c{case:02d}", "policy": policy, "status": "complete",
                         "model_requests": 0, "paired_probes": 0,
                         "provider_unusable_count": 0,
                         "wrong_family_confirmation": False,
                         "correct_confirmed_family": policy == "enabled_low"})
    rows[4]["provider_unusable_count"] = 1
    assert R._assessment(rows) == "indeterminate"


def test_four_actions_end_in_hold_instead_of_raising(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path, monkeypatch)
    fake = {"action": "read_evidence", "target": R.SEL.RESOURCE, "fact_ids": []}
    monkeypatch.setattr(R, "_dialogue_v2", lambda *_args, **_kwargs: {
        "action": deepcopy(fake), "calls": [{}], "model_valid": True,
        "provider_incomplete_hold": False})
    monkeypatch.setattr(R.RLOOP, "_apply_action", lambda current, _action, _stage, **_kwargs: (
        deepcopy(current), 0, 0.0, 0.0, {"action": deepcopy(fake)}))
    result = R._run_model_epoch(state, run_id="c02.enabled_low", mode="enabled_low",
                                out=tmp_path / "four", verify_context=lambda: True,
                                complete_factory=lambda *_: None, execute_pair=lambda *_: None)
    assert result["model_actions"] == 4 and result["decision_outcome"] == "HOLD"
    assert result["stop_reason"] == "action_budget_exhausted"


def test_source_corpus_binding_rejects_swapped_batch(tmp_path: Path) -> None:
    source_protocol = R.ROOT / "bench/protocols/prospective_diagnosis_training_20261101_v1.json"
    design = {"source_protocol": {"path": source_protocol.relative_to(R.ROOT).as_posix(),
                                  "sha256": R._sha(source_protocol)}}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text("{}", encoding="utf-8")
    batch = {"protocol": deepcopy(design["source_protocol"]), "anchor": "a" * 40}
    corpus = {"source_batch_sha256": "0" * 64,
              "source_protocol_sha256_lf": R._sha(source_protocol, lf=True),
              "source_anchor": "a" * 40}
    with pytest.raises(ValueError, match="다른 source batch"):
        R._verify_source_corpus_binding(design, batch, corpus, batch_path)


def test_current_source_change_is_rejected(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = tmp_path / "execution.json"
    design_path, bridge_path = tmp_path / "design.json", tmp_path / "bridge.json"
    design = {"x": 2, "source_protocol": {
        "path": "bench/protocols/prospective_diagnosis_training_20261101_v1.json",
        "sha256": R._sha(SOURCE_PROTOCOL)}, "source_sha256_lf": {
            "evals/run_prospective_reasoning_confirmation_v1.py": "f" * 64},
        "tool_protocols": {}}
    for path, value in ((protocol, {"x": 1}), (design_path, design),
                        (bridge_path, {"x": 3})):
        R._write_new(path, value)
    monkeypatch.setattr(R, "_sha", lambda path, **_kwargs: (
        R.hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if Path(path).resolve() == SOURCE_PROTOCOL.resolve() else "0" * 64))
    with pytest.raises(ValueError, match="실행 중 source 변경"):
        R._verify_current_bytes(protocol, {"x": 1}, design_path, design,
                                bridge_path, {"x": 3})


def test_unqualified_source_stops_before_packet_build(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    design_path = tmp_path / "design.json"
    R._write_new(design_path, R.prepare_design(SOURCE_PROTOCOL))
    source_batch = tmp_path / "source.json"
    R._write_new(source_batch, {"anchor": "a" * 40})
    rows = [{"case_id": f"c{index:02d}", "seed": 2320 if index <= 4 else 2321,
             "category": ("NONE", "F1", "R1_ONLY", "COMPOUND")[(index - 1) % 4],
             "fixed_report": {}} for index in range(1, 9)]
    monkeypatch.setattr(R, "_source_rows", lambda *_args: deepcopy(rows))
    monkeypatch.setattr(R.BASE.RUN, "_gate", lambda *_args: {
        "run": False, "reason": "source_failure_absent", "source_outcome": "unknown"})
    monkeypatch.setattr(R.BASE, "_inputs",
                        lambda *_args, **_kwargs: pytest.fail("미적격 source는 packet 생성 금지"))
    result = R.build_corpus(design_path, source_batch, tmp_path / "out")
    assert result["status"] == "source_not_qualified"
    assert result["comparison_outcome"] == "indeterminate"
    assert not (tmp_path / "out/corpus.json").exists()


def test_provider_messages_do_not_include_seed_path_or_fault_label(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages = R.O2.messages(_state(tmp_path, monkeypatch))
    text = str(messages)
    assert "2320" not in text and "C:\\Users" not in text
    assert "R1_ONLY" not in text and "COMPOUND" not in text
