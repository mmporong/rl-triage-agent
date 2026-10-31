# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import test_evidence_guided_selector_v1 as SELECTOR_FIXTURE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_diagnostic_option_study_v1 as S


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_prepare_protocol_binds_parent_prior_and_v2_source_closure() -> None:
    value = S.prepare_protocol(
        ROOT / "bench/protocols/registry_action_contract_development_20261101_v1.json",
        "a" * 64)
    assert value["schema"] == S.SCHEMA
    assert value["prior_option_result_sha256"] == "a" * 64
    assert value["design"] == {
        "opaque_cases": [f"c{i:02d}" for i in range(1, 9)],
        "policy": "evidence_guided_model", "planned_rows": 8, "fault_rows": 6,
        "normal_host_gate_rows": 2, "same_cases_as_parent": True,
        "same_source_measurements_tools_and_budgets": True,
        "registry_prediction_metadata_newly_exposed": True,
        "probe_budget_per_row": 3, "model_action_budget_per_row": 4,
        "registered_candidate_partitions": True, "probability_or_eig": False,
        "hold_always_allowed": True, "normal_gate_is_not_model_success": True,
        "automatic_repair": False,
    }
    assert value["bounds"] == {"total_wall_s": 7200, "global_api_requests": 48,
                               "paired_probes": 18, "sdk_role_executions": 36}
    assert value["analysis"]["prediction"].endswith("improvement not guaranteed")
    assert value["analysis"]["reporting"]["fault_denominator"] == 6
    assert value["analysis"]["development_candidate_rule"] == {
        "normal_rows_api_and_probe_zero": True, "wrong_family_confirmations": 0,
        "first_valid_at_least": "5/6",
        "correct_confirmed_family_strictly_above_parent_guided": "0/6",
        "all_six_hold": "not_improved", "format_only_improvement": "not_adopted",
        "positive_result": "not_new_seed_effect_or_default_adoption"}
    assert value["analysis"]["research"]["paper_effect_sizes_borrowed"] is False
    assert set(value["source_sha256_lf"]) == set(S.SOURCE_FILES)
    assert "evals/registry_action_options_v1.py" in value["inherited_source_sha256_lf"]
    assert "bench/private" not in json.dumps(value)


def test_prepare_rejects_old_or_unregistered_parent_schema(tmp_path: Path) -> None:
    parent = _json(tmp_path / "old.json", {"schema": "prospective_agentic_comparison_protocol_v1"})
    with pytest.raises(ValueError, match="parent option study schema"):
        S.prepare_protocol(parent, "b" * 64)
    with pytest.raises(ValueError, match="prior option result SHA"):
        S.prepare_protocol(
            ROOT / "bench/protocols/registry_action_contract_development_20261101_v1.json", "bad")


def test_prior_requires_complete_parent_rows_and_exact_protocol(tmp_path: Path) -> None:
    parent = ROOT / "bench/protocols/registry_action_contract_development_20261101_v1.json"
    rows = [{"case_id": f"c{i:02d}", "policy": policy}
            for i in range(1, 9) for policy in S.BASE.POLICIES]
    prior = _json(tmp_path / "prior.json", {
        "schema": S.BASE.RESULT_SCHEMA, "status": "complete", "rows": rows,
        "protocol": S.BASE._ref(parent), "validity": {"first_decision_denominator": 12},
    })
    spec = S.prepare_protocol(parent, S.BASE._sha(prior))
    assert len(S._prior(prior, spec)["rows"]) == 16
    changed = json.loads(prior.read_text(encoding="utf-8"))
    changed["rows"].pop()
    _json(prior, changed)
    spec["prior_option_result_sha256"] = S.BASE._sha(prior)
    with pytest.raises(ValueError, match="8×2"):
        S._prior(prior, spec)


def test_transport_guard_refuses_request_when_global_cap_is_already_spent() -> None:
    called = []
    factory = S._guarded_complete_factory(
        lambda _slot, _state: lambda messages, params: called.append((messages, params)) or {},
        lambda: {"requests": 48})
    with pytest.raises(ValueError, match="global API cap"):
        factory("c02.d1", {})([], {})
    assert called == []


def test_run_id_matches_registered_transport_v4_logical_slot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, _path, _ledger = cast(Any, SELECTOR_FIXTURE.selector).__wrapped__(tmp_path, monkeypatch)
    captured = []

    def file_completion(root, slot, state_digest, context_digest, timeout):
        captured.append((Path(root), slot, state_digest, context_digest, timeout))
        return lambda *_: {}

    monkeypatch.setattr(S.COMP.TRANSPORT, "file_completion", file_completion)
    slot = f"{S._run_id('c02')}.d1"
    complete = S.COMP.transport_factory(tmp_path / "queue")(slot, state)
    assert callable(complete)
    assert captured[0][1] == "c02.evidence_guided_model.d1"
    assert captured[0][4] == 180
    with pytest.raises(ValueError, match="case_id"):
        S._run_id("c02.v2")


def test_execute_preserves_six_fault_two_healthy_gate_and_v2_context(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _json(tmp_path / "protocol.json", {"schema": S.SCHEMA})
    source = _json(tmp_path / "source.json", {"source": True})
    prior_path = _json(tmp_path / "prior.json", {"prior": True})
    runtime = _json(tmp_path / "runtime.json", {"runtime": True})
    state, _path, _ledger = cast(Any, SELECTOR_FIXTURE.selector).__wrapped__(tmp_path, monkeypatch)
    categories = ("NONE", "F1", "R1_ONLY", "COMPOUND") * 2
    cases = [{
        "case_id": f"c{index + 1:02d}", "seed": 2320 + index // 4,
        "category": category, "input": tmp_path / f"input-{index}.json",
        "checkpoint": {"sha256": f"{index + 1:064x}"},
        "metadata": {"runtime_facts": {
            "case": {"path": str(runtime)}, "reference": {"path": str(runtime)}}},
    } for index, category in enumerate(categories)]
    prior_rows = [{"case_id": f"c{i:02d}", "policy": policy,
                   "gate": {"run": category != "NONE"}, "first_decision_valid": True}
                  for i, category in enumerate(categories, 1) for policy in S.BASE.POLICIES]
    parent = _json(tmp_path / "parent.json", {
        "base_comparison_protocol": {"path": str(tmp_path / "base.json")}})
    _json(tmp_path / "base.json", {"protocols": {}})
    spec = {"parent_study_protocol": {"path": str(parent)},
            "claims": {"diagnosis_improvement": False}}
    monkeypatch.setattr(S, "_validate_inputs", lambda *args: (
        spec, cases, {"rows": prior_rows, "validity": {"first_decision_denominator": 12}}))
    monkeypatch.setattr(S, "_verify_current_protocol", lambda *args: None)
    monkeypatch.setattr(S.RUN, "_lease", lambda: {"leaseId": "lease-1"})
    monkeypatch.setattr(S.RUN, "BANDS", _json(tmp_path / "bands.json", {"bands": {}}))
    monkeypatch.setattr(S.T4, "status", lambda root: {"requests": 0})
    monkeypatch.setattr(S.COMP, "transport_factory", lambda root: object())
    monkeypatch.setattr(S.RUN, "_gate", lambda case, reference, bands: (
        {"run": False, "reason": "healthy_gate", "source_outcome": "recovered"}
        if case["category"] == "NONE" else
        {"run": True, "reason": None, "source_outcome": "not_recovered"}))
    monkeypatch.setattr(S.BASE, "_inputs", lambda *args, **kwargs: {
        "state": deepcopy(state), "case_runtime": runtime, "reference_runtime": runtime})
    monkeypatch.setattr(S.BASE, "_dialogue_metrics", lambda *args: {
        "first_decision_valid": True, "post_feedback_first_decision_valid": True,
        "first_decision_invalid_codes": [], "feedback_actions": 0,
        "model_requests": 1, "read_actions": 0, "probe_actions": 0, "hold_actions": 1})
    def fake_epoch(initial_state: dict, **_kwargs: object) -> dict:
        context = S.O2.context(initial_state)
        assert context["contract"]["candidate_partitions_from_registered_semantics"] is True
        return {"status": "complete", "decision_outcome": "HOLD", "model_actions": 1,
                "paired_probes": 0, "parent_probe_wall_s": 0.0,
                "owned_child_wall_s": 0.0, "final_status": "hold",
                "confirmed_family": None, "steps": []}

    result = S.execute(protocol, "b" * 40, source, prior_path, tmp_path,
                       tmp_path / "out", tmp_path / "queue", run_epoch=fake_epoch)
    assert (result["planned_rows"], result["fault_rows"], result["normal_host_gate_rows"]) == (8, 6, 2)
    assert result["model_requests"] == 6 and result["paired_probes"] == 0
    normal = [row for row in result["rows"] if not row["gate"]["run"]]
    assert len(normal) == 2 and all(row["model_requests"] == 0 for row in normal)
    assert result["validity"] == {"first_decision_valid": 6, "first_decision_denominator": 6,
                                  "prior_guided_valid_fault_rows": 6}
    assert result["claims"]["diagnosis_improvement"] is False


def test_cli_prepare_is_write_once(tmp_path: Path) -> None:
    out = tmp_path / "protocol.json"
    args = ["prepare-protocol", "--parent-study-protocol",
            str(ROOT / "bench/protocols/registry_action_contract_development_20261101_v1.json"),
            "--prior-option-result-sha256", "c" * 64, "--out", str(out)]
    assert S.main(args) == 0 and out.is_file()
    with pytest.raises(ValueError, match="write-once"):
        S.main(args)
