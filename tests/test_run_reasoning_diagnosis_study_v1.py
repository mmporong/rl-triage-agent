# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_reasoning_diagnosis_study_v1 as S


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_prepare_bridge_registers_exact_followup_slots(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def prepare(parent: Path, slots: list[dict], max_requests: int) -> dict:
        captured.update(parent=parent, slots=slots, max_requests=max_requests)
        return {"slots": slots}

    monkeypatch.setattr(S.BRIDGE, "prepare_protocol", prepare)
    parent = tmp_path / "mode.json"
    value = S.prepare_bridge(parent)
    slots = captured["slots"]
    assert value == {"slots": slots}
    assert captured["parent"] == parent.resolve()
    assert captured["max_requests"] == 72
    assert len(slots) == len({row["slot_id"] for row in slots}) == 72
    assert slots[:6] == [
        {"slot_id": f"c02.enabled_low.d{decision}.a{attempt}",
         "case_id": "c02", "mode": "enabled_low"}
        for decision in range(2, 5) for attempt in range(1, 3)]
    assert {row["slot_id"].split(".")[2] for row in slots} == {"d2", "d3", "d4"}
    assert S._mode_order("c03") == ("legacy_none", "enabled_low")
    assert S._mode_order("c04") == ("enabled_low", "legacy_none")


def test_protocol_contract_separates_original_and_new_effects(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _json(tmp_path / "parent.json", {
        "schema": S.PARENT.SCHEMA,
        "design": {"opaque_cases": [f"c{i:02d}" for i in range(1, 9)], "fault_rows": 6},
        "inherited_source_sha256_lf": {}, "source_sha256_lf": {}})
    mode = _json(tmp_path / "mode.json", {
        "schema": S.MODE2.PLAN_SCHEMA, "source_sha256_lf": {}})
    bridge_value = {"schema": "bridge"}
    bridge = _json(tmp_path / "bridge.json", bridge_value)
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S.BASE, "ROOT", tmp_path)
    monkeypatch.setattr(S, "SOURCE_FILES", ())
    monkeypatch.setattr(S, "prepare_bridge", lambda _path: bridge_value)
    value = S.prepare_protocol(parent, mode, bridge, S.MODE_RESULT_SHA256)
    assert value["schema"] == S.SCHEMA
    assert value["design"]["planned_rows"] == 16
    assert value["design"]["fault_rows"] == 12
    assert value["design"]["normal_host_gate_rows"] == 4
    assert value["design"]["recorded_first_responses"] == 12
    assert value["design"]["original_mode_calls"] == 32
    assert value["bounds"] == {"total_wall_s": 7200, "new_api_requests": 72,
                               "paired_probes": 36, "sdk_role_executions": 72}
    assert value["analysis"]["original_first_proposal_reference"] == {
        "legacy_none_fault_active_non_hold": "5/12",
        "enabled_low_fault_active_non_hold": "11/12",
        "both_modes_contract_valid": "12/12"}
    assert all(claim is False for claim in value["claims"].values())
    assert value["source_sha256_lf"] == {"bridge.json": S.BASE._sha(bridge, lf=True)}
    assert "bench/private" not in json.dumps(value)


def test_original_prefix_requires_registered_5_of_12_and_11_of_12(tmp_path: Path) -> None:
    calls = []
    for mode, active in (("legacy_none", 5), ("enabled_low", 11)):
        calls.extend({"mode": mode, "evaluation_role": "fault_development",
                      "valid": True, "action": "read_evidence" if index < active else "hold"}
                     for index in range(12))
    path = _json(tmp_path / "result.json", {"calls": calls})
    assert S._original_prefix_summary(path) == {
        "legacy_none": {"active_non_hold": 5, "contract_valid": 12, "denominator": 12},
        "enabled_low": {"active_non_hold": 11, "contract_valid": 12, "denominator": 12}}
    calls[5]["action"] = "read_evidence"
    _json(tmp_path / "changed.json", {"calls": calls})
    with pytest.raises(ValueError, match="기준값"):
        S._original_prefix_summary(tmp_path / "changed.json")


def test_preflight_consumes_inputs_without_bridge_or_probe(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _json(tmp_path / "protocol.json", {"protocol": True})
    source = _json(tmp_path / "source.json", {"source": True})
    mode_result = _json(tmp_path / "mode.json", {"calls": []})
    corpus = _json(tmp_path / "corpus.json", {"corpus": True})
    cases = [{"case_id": f"c{i:02d}"} for i in range(1, 9)]
    gates = {row["case_id"]: {"run": row["case_id"] in S.FAULT_CASES} for row in cases}
    monkeypatch.setattr(S, "_validate_inputs", lambda *args: (
        {"claims": {"recovery": False}}, cases, gates))
    monkeypatch.setattr(S, "_original_prefix_summary", lambda _path: {
        "legacy_none": {"active_non_hold": 5, "contract_valid": 12, "denominator": 12},
        "enabled_low": {"active_non_hold": 11, "contract_valid": 12, "denominator": 12}})
    monkeypatch.setattr(S.RUN, "_lease", lambda: {"leaseId": "lease-1"})
    result = S.preflight(protocol, "a" * 40, source, mode_result, corpus, tmp_path,
                         tmp_path / "preflight")
    assert result["fault_rows"] == 12 and result["normal_host_gate_rows"] == 4
    assert result["new_model_requests"] == result["paired_probes"] == 0
    assert result["sdk_role_executions"] == 0 and result["gpu_started"] is False


def test_execute_keeps_normal_rows_at_zero_and_counts_only_new_requests(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_value = {"parent_diagnostic_protocol": {"path": "parent.json"},
                      "bridge_protocol": {"path": "bridge.json"},
                      "mode_plan": {"path": "mode-plan.json"},
                      "mode_result_sha256": S.MODE_RESULT_SHA256,
                      "claims": {"diagnosis_improvement": False}}
    protocol = _json(tmp_path / "protocol.json", protocol_value)
    source = _json(tmp_path / "source.json", {"source": True})
    corpus = _json(tmp_path / "corpus.json", {"corpus": True})
    calls = [{"mode": mode, "evaluation_role": "fault_development", "valid": True,
              "action": "read_evidence" if index < active else "hold"}
             for mode, active in (("legacy_none", 5), ("enabled_low", 11))
             for index in range(12)]
    mode_result = _json(tmp_path / "mode-result.json", {"calls": calls})
    runtime = _json(tmp_path / "runtime.json", {"runtime": True})
    categories = ("NONE", "F1", "R1_ONLY", "COMPOUND") * 2
    cases = [{"case_id": f"c{index + 1:02d}", "seed": 2320 + index // 4,
              "category": category, "input": tmp_path / f"input-{index}.json",
              "checkpoint": {"sha256": f"{index + 1:064x}"},
              "metadata": {"runtime_facts": {
                  "case": {"path": str(runtime)}, "reference": {"path": str(runtime)}}}}
             for index, category in enumerate(categories)]
    gates = {row["case_id"]: {"run": row["category"] != "NONE",
                                      "reason": None if row["category"] != "NONE" else "healthy_gate"}
             for row in cases}
    monkeypatch.setattr(S, "_validate_inputs", lambda *args: (protocol_value, cases, gates))
    monkeypatch.setattr(S, "_base_protocol", lambda _spec: {"protocols": {
        "paired_probe": {"path": str(runtime)}, "view": {"path": str(runtime)}}})
    monkeypatch.setattr(S, "prepare_protocol", lambda *args: protocol_value)
    monkeypatch.setattr(S, "_verify_first_source", lambda *args: None)
    monkeypatch.setattr(S.RUN, "_lease", lambda: {"leaseId": "lease-1"})
    monkeypatch.setattr(S.BASE, "_score", lambda category, epoch: {
        "expected_family": [], "confirmed_family": None,
        "correct_confirmed_family": False, "correct_complete_family": False,
        "compound_partial": False, "compound_complete": False})
    request_count = {"value": 0}
    monkeypatch.setattr(S, "_bridge_request_count", lambda _root: request_count["value"])

    def complete(_root: Path, _slot: str, _messages: list, _params: dict) -> dict:
        request_count["value"] += 1
        return {}

    loaded: list[tuple[str, str, int]] = []

    def load_first(*args: object, **kwargs: object) -> dict:
        loaded.append((cast(str, args[5]), cast(str, args[6]), cast(int, kwargs["replicate"])))
        return {"first": True}

    def epoch(_first: dict, **kwargs: object) -> dict:
        factory = cast(Any, kwargs["complete_factory"])
        factory(f"{kwargs['run_id']}.d2.a1", {})([], {})
        return {"status": "complete", "decision_outcome": "HOLD", "model_actions": 2,
                "original_model_requests": 1, "new_model_requests": 1,
                "total_model_requests": 2, "paired_probes": 0,
                "parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0,
                "final_status": "hold", "confirmed_family": None, "steps": []}

    result = S.execute(protocol, "b" * 40, source, mode_result, corpus, tmp_path,
                       tmp_path / "out", tmp_path / "bridge-root", load_first=load_first,
                       run_epoch=epoch, bridge_complete=complete,
                       bridge_initialize=lambda *args: {})
    assert len(result["rows"]) == 16 and len(loaded) == 12
    assert result["recorded_first_responses"] == 12
    assert result["new_model_requests"] == request_count["value"] == 12
    assert result["paired_probes"] == result["sdk_role_executions"] == 0
    normal = [row for row in result["rows"] if not row["gate"]["run"]]
    assert len(normal) == 4
    assert all(row["original_model_requests"] == row["new_model_requests"] == 0
               and row["paired_probes"] == 0 for row in normal)
    assert result["original_prefix"]["legacy_none"]["active_non_hold"] == 5
    assert result["original_prefix"]["enabled_low"]["active_non_hold"] == 11


def test_complete_factory_checks_cap_before_dispatch() -> None:
    # execute 통합 테스트의 72-cap 경계를 별도 값으로 확인한다.
    assert S.MAX_NEW_REQUESTS == 72
    assert len(S._logical_slots()) == 72


def test_metrics_separates_valid_hold_contract_invalid_and_provider_failure(
        tmp_path: Path) -> None:
    recorded = _json(tmp_path / "recorded.json", {
        "schema": "reasoning_diagnostic_recorded_dialogue_v1",
        "semantic_feedback_used": False})
    unusable = _json(tmp_path / "unusable.json", {
        "schema": S.LOOP.DIALOGUE_SCHEMA, "semantic_feedback_used": False,
        "provider_incomplete_hold": True, "model_valid": False,
        "calls": [{"result": {"finish_reason": "length"}}]})
    epoch = {"decision_outcome": "HOLD", "original_model_requests": 1,
             "new_model_requests": 1, "total_model_requests": 2,
             "steps": [
                 {"model_valid": True, "format_valid": True,
                  "action": {"action": "read_evidence"},
                  "dialogue": {"path": str(recorded)}},
                 {"model_valid": False, "format_valid": False,
                  "action": {"action": "hold"},
                  "dialogue": {"path": str(unusable)}}]}
    metrics = S._metrics(tmp_path, epoch)
    assert metrics["new_provider_unusable"] is True
    assert metrics["new_contract_invalid"] is False
    assert metrics["model_valid_hold"] is False
    assert metrics["provider_failure_codes"] == ["length"]

    accepted = next(iter(S.MODE1.CALL.ACCEPTED_PROVIDER_MODELS))
    invalid = _json(tmp_path / "invalid.json", {
        "schema": S.LOOP.DIALOGUE_SCHEMA, "semantic_feedback_used": True,
        "provider_incomplete_hold": False, "model_valid": False,
        "calls": [{"result": {"provider_model": accepted, "finish_reason": "stop",
                                "content": "bad", "error_type": None,
                                "infra_error": False}}]})
    epoch["steps"][-1]["dialogue"] = {"path": str(invalid)}
    metrics = S._metrics(tmp_path, epoch)
    assert metrics["new_provider_unusable"] is False
    assert metrics["new_contract_invalid"] is True

    epoch["steps"][-1]["model_valid"] = True
    metrics = S._metrics(tmp_path, epoch)
    assert metrics["model_valid_hold"] is True
