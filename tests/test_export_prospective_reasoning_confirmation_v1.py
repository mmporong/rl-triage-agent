# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import export_prospective_reasoning_confirmation_v1 as E
from evals import run_prospective_reasoning_confirmation_v1 as R


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


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
    monkeypatch.setattr(R.SEL.S, "validate_packet", lambda value: deepcopy(value))
    return R.SEL.initialize({
        "facts": {"telemetry:case:mean_reward": 1.0},
        "remaining_candidates": list(R.SEL.S.C1.registry()["mechanisms"]),
        "observed": {}, "budget_left": 3,
        "input_bindings": {"case": {"checkpoint_sha256": "a" * 64},
                           "reference": {"checkpoint_sha256": "b" * 64}}}, ledger)


def _completion(tmp_path: Path, mode: str):
    def factory(slot: str, _state: dict):
        def complete(messages: list, params: dict) -> dict:
            result = {"provider_model": R.BRIDGE.MODE.MODEL, "finish_reason": "stop",
                      "content": '{"option_id":"hold","fact_ids":[]}',
                      "request_params": deepcopy(params), "wall_s": 0.01,
                      "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "reasoning_tokens": 0, "reasoning_tokens_status": "reported"}
            root = tmp_path / "transport"
            index = len(list((root / "requests").glob("*.json"))) + 1
            request = root / "requests" / f"{index:03d}.json"
            R._write_new(request, {
                "schema": R.BRIDGE.REQUEST_SCHEMA, "index": index, "slot": slot,
                "case_id": slot.split(".", 1)[0], "mode": mode,
                "messages": deepcopy(messages),
                "messages_sha256": R.BRIDGE.digest(messages),
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
    return factory


def _hold_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                mode: str = "enabled_low") -> tuple[Path, dict, dict]:
    state = _state(tmp_path, monkeypatch)
    root = tmp_path / "epoch"
    epoch = R._run_model_epoch(
        state, run_id=f"c02.{mode}", mode=mode, out=root,
        verify_context=lambda: True, complete_factory=_completion(tmp_path, mode),
        execute_pair=lambda *_args, **_kwargs: pytest.fail("HOLD는 probe를 실행하지 않는다"))
    row = {"case_id": "c02", "policy": mode, "status": "complete",
           **{name: epoch[name] for name in ("decision_outcome", "confirmed_family",
              "model_requests", "paired_probes", "provider_unusable_count",
              "semantic_invalid_count")}, **R._score("R1_ONLY", epoch)}
    return root, state, row


def _parent_rows() -> list[dict]:
    rows = []
    for index in range(1, 9):
        for policy in R.POLICIES:
            if index in (1, 5):
                row = {"case_id": f"c{index:02d}", "policy": policy,
                       "status": "complete", "decision_outcome": "HOLD",
                       "model_requests": 0, "paired_probes": 0,
                       "correct_confirmed_family": False,
                       "wrong_family_confirmation": False,
                       "provider_unusable_count": 0, "semantic_invalid_count": 0,
                       "compound_partial": False, "compound_complete": False}
            else:
                row = {"case_id": f"c{index:02d}", "policy": policy,
                       "status": "complete", "decision_outcome": "HOLD",
                       "confirmed_family": None, "model_requests": 0,
                       "paired_probes": 0, "provider_unusable_count": 0,
                       "semantic_invalid_count": 0,
                       "correct_confirmed_family": False,
                       "wrong_family_confirmation": False,
                       "compound_partial": False, "compound_complete": False}
            rows.append(row)
    return rows


def test_model_epoch_replays_dialogue_and_transport(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state, row = _hold_epoch(tmp_path, monkeypatch)
    detail, calls = E._verify_model_epoch(root, state, row, case_id="c02",
                                           policy="enabled_low", category="R1_ONLY")
    assert detail["epoch"]["decision_outcome"] == "HOLD"
    assert len(calls) == 1 and calls[0]["provider_usable"] is True
    assert calls[0]["usage"]["status"] == "complete"
    assert calls[0]["reasoning_body_included"] is False


@pytest.mark.parametrize("target", ["summary", "receipt", "mode", "unknown"])
def test_model_epoch_rejects_tampered_summary_receipt_mode_or_missing_unknown(
        target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state, row = _hold_epoch(tmp_path, monkeypatch)
    if target == "summary":
        row["model_requests"] = 2
    elif target == "receipt":
        host = R._read(root / "step-01/host-action.json")
        host["request_spec"] = {"unexpected": True}
        _write(root / "step-01/host-action.json", host)
    elif target == "mode":
        dialogue = R._read(root / "step-01/dialogue.json")
        dialogue["mode"] = "legacy_none"
        _write(root / "step-01/dialogue.json", dialogue)
    else:
        row.pop("provider_unusable_count")
    with pytest.raises(ValueError):
        E._verify_model_epoch(root, state, row, case_id="c02", policy="enabled_low",
                              category="R1_ONLY")


def test_model_epoch_rejects_tampered_provider_response(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state, row = _hold_epoch(tmp_path, monkeypatch)
    dialogue = R._read(root / "step-01/dialogue.json")
    response = Path(dialogue["calls"][0]["result"]["transport_ref"]["path"])
    value = R._read(response)
    value["result"]["content"] = "tampered"
    _write(response, value)
    with pytest.raises(ValueError, match="response ref"):
        E._verify_model_epoch(root, state, row, case_id="c02", policy="enabled_low",
                              category="R1_ONLY")


def test_usage_marks_missing_fields_unknown_instead_of_zero() -> None:
    value = E._usage({})
    assert value["status"] == "unknown"
    assert value["reported_fields"] == []
    assert set(value["values"].values()) == {None}


def test_pair_proof_uses_the_same_ledger_loop_as_runtime(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    seen = []
    monkeypatch.setattr(R.RLOOP.L1, "_pair_proof", lambda *args: seen.append(args) or {"ok": True})
    assert E._pair_proof({"pair": 1}, {"state": 1}, {"action": "run_probe"}, tmp_path) == {
        "ok": True}
    assert seen[0][-1] == tmp_path


def test_rule_epoch_replays_pre_action_receipt_and_final_state(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rule"
    action = {"action": "hold", "target": None, "fact_ids": []}
    initial = {"state_digest": "a" * 64, "status": "open", "hypotheses": ["reward"]}
    final = {"state_digest": "b" * 64, "status": "open", "hypotheses": ["reward"]}
    monkeypatch.setattr(R.SEL, "select_strong_rule", lambda state: deepcopy(action))
    monkeypatch.setattr(R.SEL, "advance", lambda state, selected, **_kwargs: deepcopy(final))
    selected = {"state_before": initial, "selected_by": "rule", "action": action}
    record = {**selected, "state_after_digest": final["state_digest"]}
    epoch = {"schema": "prospective_agentic_epoch_v1", "status": "complete",
             "run_id": "c02.strong_registered_rule", "policy": "strong_registered_rule",
             "request_count": 0, "steps": [record], "probe_count": 0,
             "model_actions": 0, "decision_outcome": "HOLD", "confirmed_family": None,
             "parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0,
             "epoch_wall_s": 0.01}
    _write(root / "step-01/selected-action.json", selected)
    _write(root / "step-01/state-after.json", final)
    _write(root / "final-state.json", final)
    _write(root / "result.json", epoch)
    row = {"case_id": "c02", "policy": "strong_registered_rule", "status": "complete",
           "decision_outcome": "HOLD", "confirmed_family": None,
           "model_requests": 0, "paired_probes": 0, "provider_unusable_count": 0,
           "semantic_invalid_count": 0, **R._score("R1_ONLY", epoch)}
    detail = E._verify_rule_epoch(root, initial, row, case_id="c02", category="R1_ONLY")
    assert detail["epoch"]["decision_outcome"] == "HOLD"
    selected["state_after_digest"] = final["state_digest"]
    _write(root / "step-01/selected-action.json", selected)
    with pytest.raises(ValueError, match="선택"):
        E._verify_rule_epoch(root, initial, row, case_id="c02", category="R1_ONLY")


def test_parent_aggregate_rejects_tampered_summary() -> None:
    rows = _parent_rows()
    result = {"planned_rows": 24, "model_requests": 1, "paired_probes": 0,
              "sdk_role_executions_upper_bound": 0,
              "assessment": R._assessment(rows), "claims": {"recovery": False}}
    with pytest.raises(ValueError, match="aggregate"):
        E._verify_parent_aggregate(result, rows, {"claims": {"recovery": False}})


def test_qualification_rejects_unqualified_source_and_bad_gate_count() -> None:
    gates = ([{"case_id": "c01", "category": "NONE",
               "gate": {"reason": "healthy_gate"}},
              {"case_id": "c05", "category": "NONE",
               "gate": {"reason": "healthy_gate"}}]
             + [{"case_id": f"c{i:02d}", "category": "F1", "gate": {"run": True}}
                for i in (2, 3, 4, 6, 7, 8)])
    good = {"schema": "prospective_reasoning_source_qualification_v1",
            "status": "qualified", "gates": gates, "api_requests": 0,
            "replacement_seed": False, "comparison_outcome": None}
    E._verify_qualification(good)
    bad = deepcopy(good)
    bad["status"] = "source_not_qualified"
    with pytest.raises(ValueError, match="qualification"):
        E._verify_qualification(bad)
    bad = deepcopy(good)
    bad["gates"] = gates[:-1]
    with pytest.raises(ValueError):
        E._verify_qualification(bad)


def test_export_writes_path_free_projection_and_refuses_overwrite(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    protocols = tmp_path / "bench/protocols"
    paths = {name: protocols / f"{name}.json" for name in ("execution", "design", "bridge")}
    source, corpus = tmp_path / "inputs/source.json", tmp_path / "inputs/corpus.json"
    bridge_root, result_path = tmp_path / "private/bridge", tmp_path / "private/run/result.json"
    claims = {"new_fault_family_generalization": False, "llm_superiority": False,
              "default_mode_adoption": False, "repair": False, "recovery": False}
    design = {"seeds": [2430, 2431], "claims": claims,
              "analysis": {"post_exposure_design": True},
              "bridge": {"modes": {"legacy_none": {}, "enabled_low": {}}},
              "source_sha256_lf": {"evals/frozen.py": "a" * 64}}
    for path, value in ((paths["execution"], {"x": 1}), (paths["design"], design),
                        (paths["bridge"], {"bridge": 1}), (source, {"source": 1}),
                        (corpus, {"corpus": 1})):
        _write(path, value)
    gates = ([{"case_id": "c01", "category": "NONE",
               "gate": {"reason": "healthy_gate"}},
              {"case_id": "c05", "category": "NONE",
               "gate": {"reason": "healthy_gate"}}]
             + [{"case_id": f"c{i:02d}", "category": "FAULT", "gate": {"run": True}}
                for i in (2, 3, 4, 6, 7, 8)])
    _write(corpus.parent / "source-qualification.json", {
        "schema": "prospective_reasoning_source_qualification_v1", "status": "qualified",
        "gates": gates, "api_requests": 0, "replacement_seed": False,
        "comparison_outcome": None})
    rows = _parent_rows()
    raw = {"schema": R.RESULT_SCHEMA, "status": "complete", "rows": rows,
           "planned_rows": 24, "model_requests": 0, "paired_probes": 0,
           "sdk_role_executions_upper_bound": 0, "assessment": R._assessment(rows),
           "claims": claims}
    _write(result_path, raw)
    public_rows = []
    for row in rows:
        case_index = int(row["case_id"][1:])
        public_rows.append({"schema": E.CASE_SCHEMA, **deepcopy(row),
                            "seed": 2430 if case_index <= 4 else 2431,
                            "category": "NONE" if case_index in (1, 5) else "FAULT",
                            "checkpoint_sha256": "c" * 64})
    spec = {"planned_rows": 24, "fault_rows": 18, "normal_host_gate_rows": 6,
            "design_anchor": "d" * 40}
    monkeypatch.setattr(R.BRIDGE, "_session", lambda _root: ({
        "protocol": {"path": paths["bridge"].relative_to(tmp_path).as_posix(),
                     "sha256": E._sha(paths["bridge"])}, "anchor": "d" * 40},
        E._read(paths["bridge"])))
    def verify(*_args):
        return deepcopy(spec)

    def collect(*_args):
        return deepcopy(public_rows), [], {
            "api_wall_s": 0.0, "parent_probe_wall_s": 0.0,
            "owned_child_wall_s": 0.0,
            "scope": "component_sums_not_whole_execution_wall"}
    out = tmp_path / "public"
    summary = E.export(result_path, paths["execution"], paths["design"], paths["bridge"],
                       source, corpus, bridge_root, anchor="e" * 40, out=out,
                       verify_execution_fn=verify, collect_fn=collect)
    assert summary["independent_source_seeds"] == 2
    assert summary["uncertainty"]["p_value_reported"] is False
    assert summary["claims"]["llm_superiority"] is False
    assert len(list((out / "cases").glob("*.json"))) == 24
    encoded = "".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
    assert "reasoning_content" not in encoded and "bench/private" not in encoded
    with pytest.raises(ValueError, match="fresh"):
        E.export(result_path, paths["execution"], paths["design"], paths["bridge"],
                 source, corpus, bridge_root, anchor="e" * 40, out=out,
                 verify_execution_fn=verify, collect_fn=collect)


def test_export_delegates_source_and_corpus_integrity_before_writing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    paths = [tmp_path / name for name in ("result.json", "protocol.json", "design.json",
                                          "bridge.json", "source.json", "corpus.json")]
    for path in paths:
        _write(path, {})
    def verify(_protocol: Path, _anchor: str, _design: Path, _bridge: Path,
               source: Path, corpus: Path) -> dict:
        if E._read(source) != {"expected": True} or E._read(corpus) != {"expected": True}:
            raise ValueError("source/corpus SHA 불일치")
        return {}
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="source/corpus"):
        E.export(paths[0], paths[1], paths[2], paths[3], paths[4], paths[5],
                 tmp_path / "bridge-root", anchor="a" * 40, out=out,
                 verify_execution_fn=verify)
    assert not out.exists()
