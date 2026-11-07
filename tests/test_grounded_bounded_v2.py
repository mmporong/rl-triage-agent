import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from rl_triage import grounded_bounded_v2 as B
from rl_triage import grounded_evidence as G
from rl_triage import triage_tools as T
from rl_triage.nat_grounded_bounded_v2 import NextCheck


@pytest.fixture
def bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    (tmp_path / "cases/one").mkdir(parents=True)
    (tmp_path / "reference").mkdir()
    for path, value in [
        ("cases/one/telemetry.json", {"summary": {"Train/mean_reward": {"n": 1, "last": 0.5}}, "series": {}}),
        ("reference/telemetry.json", {"reference_status": "absent", "summary": {}, "series": {}}),
    ]:
        (tmp_path / path).write_text(json.dumps(value), encoding="utf-8")
    return B.BoundedSession("one")


def page_args(session):
    session.call("grounded_active_registry", {})
    index = session.call("grounded_evidence_group_index", {"case_id": "one"})
    return {"case_id": "one", "catalog_sha256": index["catalog_sha256"], "group_id": index["groups"][0]["group_id"]}


def assessment_args(session):
    page = page_args(session)
    session.call("grounded_evidence_group_page", page)
    evidence = "telemetry.summary::Train/mean_reward::last"
    selected = session.call("grounded_evidence_details", {"case_id": "one", "catalog_sha256": page["catalog_sha256"], "evidence_ids": [evidence]})
    premise = {"evidence_id": evidence, "relation": "eq", "operand": {"value": 0.5, "unit": "telemetry_native"}}
    hypotheses = [{"id": h, "rank": rank, "narrative": "development fixture", "premises": [premise]} for rank, h in enumerate(["h1", "h2"], 1)]
    predictions = [{"hypothesis_id": h, "predictions": [{"measurement_id": m, "relation": relation, "operand": {"value": 0.0, "unit": "reward_native"}} for m in ("reward_mean", "reward_min")]} for h, relation in [("h1", "lt"), ("h2", "gte")]]
    return {"case_id": "one", "catalog_sha256": page["catalog_sha256"], "selection_binding": selected["selection_binding"], "hypotheses": hypotheses, "next_check": {"check_id": "runtime_reward_window", "hypothesis_predictions": predictions}}


def test_third_page_is_rejected_before_read(bounded, monkeypatch):
    args = page_args(bounded)
    original = G.evidence_catalog_group_page
    calls = []
    def read(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(G, "evidence_catalog_group_page", read)
    for _ in range(2):
        bounded.call("grounded_evidence_group_page", args)
    with pytest.raises(B.ToolBoundaryError, match="tool_budget_exhausted"):
        bounded.call("grounded_evidence_group_page", args)
    assert len(calls) == 2
    assert bounded.receipt()["events"][-1]["backend_entered"] is False


def test_parallel_page_calls_cannot_exceed_budget(bounded):
    args = page_args(bounded)
    def invoke(_):
        try:
            bounded.call("grounded_evidence_group_page", args)
            return True
        except B.ToolBoundaryError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(8)))
    assert sum(results) == 2
    assert bounded.counts["grounded_evidence_group_page"] == 2


def test_check_schema_rejects_registry_id():
    schema = NextCheck.model_json_schema()
    assert set(schema["properties"]["check_id"]["enum"]) == set(B.CHECK_IDS)
    with pytest.raises(ValidationError):
        NextCheck(check_id="grounded_active_v1", hypothesis_predictions=[])
    assert "registry_id" not in schema["properties"]


def test_valid_registration_retains_host_validation_and_no_authority(bounded):
    result = bounded.call("write_grounded_active_assessment", assessment_args(bounded))
    assert result["status"] == "hypotheses_only"
    assert result["next_check"]["registry_id"] == "runtime_reward_window"
    receipt = bounded.receipt()
    assert receipt["completed"] and receipt["terminal_reason"] is None
    assert not any(receipt[k] for k in ["approval_created", "inspection_executed", "gpu_runs", "cause_confirmed", "recovery_confirmed"])


@pytest.mark.parametrize("kind", ["wrong_evidence", "nonexclusive", "wrong_case", "wrong_catalog", "registry_id"])
def test_invalid_proposals_do_not_create_assessment(bounded, kind):
    args = assessment_args(bounded)
    if kind == "wrong_evidence":
        args["hypotheses"][0]["premises"][0]["evidence_id"] = "invented"
    elif kind == "nonexclusive":
        args["next_check"]["hypothesis_predictions"][1]["predictions"] = args["next_check"]["hypothesis_predictions"][0]["predictions"]
    elif kind == "wrong_case":
        args["case_id"] = "another"
    elif kind == "wrong_catalog":
        args["catalog_sha256"] = "0" * 64
    else:
        args["next_check"]["check_id"] = "grounded_active_v1"
    with pytest.raises((B.ToolBoundaryError, ValueError)):
        bounded.call("write_grounded_active_assessment", args)
    assert not (T.WORKSPACE / "preregistrations/grounded_active_assessments/one.json").exists()


def test_changed_telemetry_stops_before_backend(bounded):
    p = T.WORKSPACE / "cases/one/telemetry.json"
    p.write_text('{"summary":{},"series":{}}', encoding="utf-8")
    with pytest.raises(B.ToolBoundaryError, match="source_changed"):
        bounded.call("grounded_active_registry", {})
    assert not bounded.counts


def test_unregistered_action_and_further_calls_are_terminal(bounded):
    before = hashlib.sha256((T.WORKSPACE / "cases/one/telemetry.json").read_bytes()).hexdigest()
    with pytest.raises(B.ToolBoundaryError, match="tool_not_registered"):
        bounded.call("shell", {"command": "synthetic unauthorized action"})
    with pytest.raises(B.ToolBoundaryError, match="session_closed"):
        bounded.call("grounded_active_registry", {})
    assert not bounded.counts
    assert before == hashlib.sha256((T.WORKSPACE / "cases/one/telemetry.json").read_bytes()).hexdigest()


def test_repeated_write_cannot_change_saved_file(bounded):
    args = assessment_args(bounded)
    bounded.call("write_grounded_active_assessment", args)
    p = T.WORKSPACE / "preregistrations/grounded_active_assessments/one.json"
    before = p.read_bytes()
    with pytest.raises(B.ToolBoundaryError, match="session_closed"):
        bounded.call("write_grounded_active_assessment", args)
    assert p.read_bytes() == before
