"""실제 기존 원장·raw seal을 쓰는 CPU 조사 고리 회귀. 새 API/GPU 실행 없음."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import agentic_investigation_contract_v1 as A
import test_selection_comparison_controller_v1 as F
from agentic_investigation_loop_v1 import Investigation

C = A.C
POLICY = "model_adaptive"
FACT = "config:action_scale"


def response(action="run_probe", target="P_reward", facts=None):
    return json.dumps({"action": action, "target": target, "fact_ids": facts or [FACT]})


def make_probe(root, session, probe, outcome, *, epoch=1, applied=None, prior=None):
    before = deepcopy(session.snapshot)
    session.propose(response(target=probe))
    choice = session.probe_choice()
    ordinal = len(before["view"]["observations"])
    path = root / "raw" / f"e{epoch}_probe_{ordinal}"
    path.mkdir(parents=True)
    case, reference = path / "case.json", path / "reference.json"
    case.write_text(json.dumps({} if outcome == "unknown" else F.measurement(probe, outcome)), encoding="utf-8")
    reference.write_text(json.dumps(F.measurement(probe, "normal")), encoding="utf-8")
    seal = C.record_probe(root, "case", epoch, before, choice, case, reference, "cpu-operator", 1.0)
    return seal, F.snapshot(root, "case", POLICY, epoch=epoch, applied=applied, prior=prior)


def test_normal_fixture_can_hold_without_any_execution(tmp_path):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="조사 후 필요할 때만 교정")
    review = session.propose(response("hold", None))
    assert review["status"] == "valid" and review["approval_granted"] is False
    assert session.status == "hold" and session.pending is None
    assert session.audit()["gpu_jobs_started_by_this_module"] == 0
    with pytest.raises(ValueError, match="종료"):
        session.propose(response())


def test_read_frames_then_select_actual_registered_probe(tmp_path):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="명령 추종 실패 조사",
                            evidence={"evidence:clip": "frames"}, allow_fixture=True)
    session.propose(response("inspect_frames", "evidence:clip"))
    with pytest.raises(ValueError, match="미완료"):
        session.propose(response())
    with pytest.raises(ValueError, match="frame"):
        session.complete_evidence("evidence:clip", {"vision:play:f01:timestamp_s": 0.0})
    session.complete_evidence("evidence:clip", {"vision:frame:f01": {
        "evidence_type": "raw_frame", "timestamp_s": 0.0, "source_group": "visual_source_1"}})
    assert session.context()["model_view"]["available_actions"]["inspect_frames"] == []
    session.propose(response(facts=["vision:frame:f01"]))
    choice = session.probe_choice()
    spec = C.request_spec(session.snapshot, choice)
    assert spec["probe"] == "P_reward" and spec["approval_granted"] is False
    assert spec["args"] == C.RC.registry()["probes"]["P_reward"]["args"]


def test_real_raw_probe_receipt_changes_next_available_action(tmp_path):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="고장 조사", allow_fixture=True)
    seal, updated = make_probe(tmp_path, session, "P_value", "normal")
    session.complete_probe(seal, updated)
    assert "P_value" not in session.context()["model_view"]["available_actions"]["run_probe"]
    seal, updated = make_probe(tmp_path, session, "P_reward", "abnormal")
    session.complete_probe(seal, updated)
    available = session.context()["model_view"]["available_actions"]
    assert available["run_probe"] == []
    assert available["propose_correction"] == [C.RC.REWARD]
    assert len([x for x in session.events if x["event"] == "probe"]) == 2
    with pytest.raises(ValueError, match="미완료"):
        session.complete_probe(seal, updated)


@pytest.mark.parametrize("content", [
    '{"action":"hold","action":"run_probe","target":"P_reward","fact_ids":["config:action_scale"]}',
    response("run_shell", "rm"), response("propose_correction", "invented"),
    response(facts=["answer:reward"]), response(target="P_missing"),
    '{"action":"run_probe","target":"P_reward","fact_ids":["config:action_scale"],"approval":true}',
])
def test_invalid_model_actions_hold_without_fallback(tmp_path, content):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="고장 조사")
    result = session.propose(content)
    assert result["status"] == "invalid" and session.status == "hold"
    assert session.pending is None and result["approval_granted"] is False


def test_model_messages_exclude_host_binding_and_artifact_paths(tmp_path):
    snapshot = F.snapshot(tmp_path, "case", POLICY)
    context = A.prepare(snapshot, goal="명령 추종 실패 조사", evidence={"evidence:training": "training"})
    payload = A.messages(context)[1]["content"]
    assert str(tmp_path) not in payload
    assert "input_bindings" not in payload and "checkpoint_sha256" not in payload
    assert "source_failure_absent" not in payload
    assert "evidence:training" in payload
    changed = deepcopy(context)
    changed["model_view"]["eligible_probes"].append("P_fake")
    with pytest.raises(ValueError, match="결합"):
        A.messages(changed)


@pytest.mark.parametrize("facts", [
    {"answer:label": 1}, {"path:source": "/private/file"},
    {"vision:play:f01:pose:healthy": 1},
    {"telemetry:case:Metrics/base_velocity/error_vel_xy": float("nan")},
    {"vision:frame:f01": {"evidence_type": "raw_frame", "timestamp_s": 0,
                           "source_group": "visual_source_1", "path": "fault.png"}},
])
def test_extra_evidence_rejects_labels_paths_and_nonfinite(facts):
    with pytest.raises(ValueError):
        A.public_facts(facts)


def test_probe_seal_tampering_and_fixture_in_production_are_rejected(tmp_path):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="조사")
    seal, updated = make_probe(tmp_path, session, "P_reward", "abnormal")
    with pytest.raises(ValueError, match="fixture"):
        session.complete_probe(seal, updated)
    session.allow_fixture = True
    wrong = {**seal, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA"):
        session.complete_probe(wrong, updated)
    other = F.snapshot(tmp_path / "other", "case", POLICY)
    with pytest.raises(ValueError, match="계보"):
        session.complete_probe(seal, other)


def test_decision_budget_has_no_extra_model_or_probe_request(tmp_path):
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="조사", max_decisions=0)
    result = session.propose(response())
    assert result == {"status": "hold", "approval_granted": False}
    assert session.status == "hold" and session.pending is None


def test_unknown_receipt_and_missing_evidence_do_not_continue(tmp_path):
    missing = Investigation(F.snapshot(tmp_path / "missing", "case", POLICY), goal="조사")
    assert missing.propose(response("inspect_frames", "evidence:missing"))["status"] == "invalid"
    assert missing.status == "hold"
    session = Investigation(F.snapshot(tmp_path, "case", POLICY), goal="조사", allow_fixture=True)
    seal, updated = make_probe(tmp_path, session, "P_reward", "unknown")
    session.complete_probe(seal, updated)
    assert session.status == "hold" and session.pending is None
    with pytest.raises(ValueError, match="종료"):
        session.propose(response("propose_correction", C.RC.REWARD))


def test_two_corrections_require_verified_nonrecovery_and_new_checkpoint(tmp_path, monkeypatch):
    arm = "case"
    current = F.artifact(tmp_path, arm, POLICY, 0, [], b"fault-source")
    reference = F.artifact(tmp_path, "NONE", "shared_reference", 0, [], b"healthy-reference")
    bindings = deepcopy(F.BINDINGS)
    bindings["case"]["checkpoint_sha256"] = current["checkpoint"]["sha256"]
    bindings["reference"]["checkpoint_sha256"] = reference["checkpoint"]["sha256"]
    monkeypatch.setattr(F, "BINDINGS", bindings)
    reference_report = F.report(tmp_path, "s2120_none.json", "normal", reference)
    applied, prior, parent = [], None, None
    session = Investigation(F.snapshot(tmp_path, arm, POLICY), goal="두 성분 실패 조사", allow_fixture=True)
    for epoch, component, report_name in [(1, "reward", "s2120_first.json"),
                                          (2, C.RC.COUNT, "s2120_both.json")]:
        probe = "P_reward" if component == "reward" else C.RC.NEW_PROBE
        seal, updated = make_probe(tmp_path, session, probe, "abnormal", epoch=epoch, applied=applied, prior=prior)
        session.complete_probe(seal, updated)
        final, seals = deepcopy(session.snapshot), [seal]
        cid = C.RC.CORRECTIONS[component]
        session.propose(response("propose_correction", cid))
        ledger = tmp_path / arm / "corrections.ledger.jsonl"
        auth = C.authorize_repair(ledger, arm, final, applied, seals, "cpu-operator", 3600)
        output = F.artifact(tmp_path, arm, POLICY, epoch, [*applied, cid],
                            f"corrected-{epoch}".encode(), current["checkpoint"]["sha256"])
        report = F.report(tmp_path, report_name, f"out{epoch}", output)
        path = tmp_path / arm / f"epoch{epoch}/repair.json"
        C.seal_repair(path, ledger_path=ledger, authorization=auth, arm=arm, policy=POLICY,
                      epoch=epoch, applied_before=applied, input_artifact=current, output_artifact=output,
                      output_report_path=report, reference_artifact=reference,
                      reference_report_path=reference_report, bands=F.BANDS, gpu_s=30.0, parent_path=parent)
        next_snapshot = None
        if epoch == 1:
            prior = {"verdict": "not_recovered", "correction_id": cid,
                     "result_sha256": C.digest_file(path), "checkpoint_sha256": output["checkpoint"]["sha256"]}
            next_snapshot = F.snapshot(tmp_path, arm, POLICY, epoch=2, applied=[cid], prior=prior)
            with pytest.raises(ValueError, match="次|다음"):
                session.complete_repair(path, ledger_path=ledger, bands=F.BANDS)
        session.complete_repair(path, ledger_path=ledger, bands=F.BANDS, next_snapshot=next_snapshot)
        assert session.status == ("active" if epoch == 1 else "recovered")
        current, applied, parent = output, [*applied, cid], path
    assert [x["probe"] for x in session.events if x["event"] == "probe"] == ["P_reward", C.RC.NEW_PROBE]
    assert [x["outcome"] for x in session.events if x["event"] == "repair"] == ["not_recovered", "recovered"]
