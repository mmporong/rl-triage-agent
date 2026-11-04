from __future__ import annotations

import json
from pathlib import Path

import pytest
import test_investigation_session as F

from evals.diagnostic_terminal_status_v1 import terminal_aware_status
from rl_triage.investigation_session import InvestigationSession


def session(tmp_path,method="iterative_agent"):
    catalog=F.registry()
    catalog["mechanisms"].append("m3")
    for probe in catalog["probes"].values(): probe["expect"]["m3"]="normal"
    return InvestigationSession.prepare(F.source(tmp_path),tmp_path/"session",registry=catalog,
        method=method,max_checks=2,gpu_budget_per_check_s=1,
        source_verifier=F.verifier,model_binding=F.binding())


def snapshot(folder:Path):
    return {p.relative_to(folder).as_posix():p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def content(action="hold",probe=None,facts=None):
    return {"content":json.dumps({"action":action,"probe":probe,"probes":[],
        "fact_ids":["visible_metric"] if facts is None else facts})}


def test_fresh_status_does_not_create_model_journal(tmp_path):
    s=session(tmp_path);before=snapshot(tmp_path)
    assert terminal_aware_status(s)["status"]=="open"
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize("method",["one_shot_llm","iterative_agent"])
@pytest.mark.parametrize("facts",[["visible_metric"],[],["invented"]])
def test_saved_hold_is_revalidated_without_transport_or_persistent_change(tmp_path,method,facts):
    s=session(tmp_path,method);calls=[]
    def complete(*_): calls.append(1);return content(facts=facts)
    assert s.next(complete=complete,source_verifier=F.verifier)["status"]=="model_hold"
    assert s.status()["status"]=="open"
    before=snapshot(tmp_path)
    result=terminal_aware_status(s)
    assert result["status"]=="model_hold" and result["new_model_call"] is False
    assert result["pending"] is None and result["confirmed_mechanism"] is None
    assert len(calls)==1 and snapshot(tmp_path)==before


def test_repeated_probe_hold_does_not_turn_into_diagnostic_resume(tmp_path):
    s=session(tmp_path);calls=[]
    def complete(*_): calls.append(1);return content("run_probe","p1")
    assert s.next(complete=complete,source_verifier=F.verifier)["status"]=="approval_pending"
    F.execute_outcome(s,"normal")
    assert s.status()["status"]=="open"
    assert s.next(complete=complete,source_verifier=F.verifier)["status"]=="model_hold"
    before=snapshot(tmp_path);result=terminal_aware_status(s)
    assert result["status"]=="model_hold" and result["model"]["model_status"]=="contract_invalid"
    assert len(calls)==2 and snapshot(tmp_path)==before


def test_pending_probe_is_not_reclassified_as_hold(tmp_path):
    s=session(tmp_path)
    s.next(complete=lambda *_:content("run_probe","p1"),source_verifier=F.verifier)
    before=snapshot(tmp_path)
    assert terminal_aware_status(s)["status"]=="approval_pending"
    assert snapshot(tmp_path)==before


def test_choice_ready_without_request_remains_open(tmp_path):
    s=session(tmp_path)
    result=s._model_choice(s.status(),complete=lambda *_:content("run_probe","p1"))
    assert result["status"]=="choice_ready"
    before=snapshot(tmp_path)
    assert terminal_aware_status(s)["status"]=="open"
    assert snapshot(tmp_path)==before


def test_tampered_terminal_engine_result_is_rejected_by_original_replay(tmp_path):
    s=session(tmp_path)
    s.next(complete=lambda *_:content(facts=[]),source_verifier=F.verifier)
    terminal=s.folder/"model/0000/terminal.json"
    d=json.loads(terminal.read_text());d["engine_result"]["model_status"]="valid"
    terminal.write_text(json.dumps(d),encoding="utf-8")
    before=snapshot(tmp_path)
    with pytest.raises(ValueError,match="terminal"):
        terminal_aware_status(s)
    assert snapshot(tmp_path)==before


def test_tampered_response_is_rejected_without_new_call(tmp_path):
    s=session(tmp_path)
    s.next(complete=lambda *_:content(),source_verifier=F.verifier)
    response=s.folder/"api/responses/0001.json"
    d=json.loads(response.read_text());d["result"]["content"]="{}"
    response.write_text(json.dumps(d),encoding="utf-8")
    before=snapshot(tmp_path)
    with pytest.raises(ValueError): terminal_aware_status(s)
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize("consumer",["preparer","controller"])
def test_production_recovery_consumers_agree_on_saved_model_hold(tmp_path,monkeypatch,consumer):
    from evals import prepare_registered_recovery_comparison_v1 as prepare
    from evals import run_registered_recovery_comparison_v1 as controller
    from rl_triage import registered_diagnostic_backend
    s=session(tmp_path)
    s.next(complete=lambda *_:content(facts=[]),source_verifier=F.verifier)
    monkeypatch.setattr(registered_diagnostic_backend,"verify_source",F.verifier)
    before=snapshot(tmp_path)
    observed=(prepare._diagnostic_status(s.folder) if consumer=="preparer"
              else controller._verified_diagnostic_status(s.folder))
    assert observed["status"]=="model_hold"
    assert controller._TERMINAL_WITHOUT_RECOVERY[observed["status"]]=="hold"
    assert snapshot(tmp_path)==before
