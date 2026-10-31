from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import evidence_guided_recovery_v1 as R


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _study_protocol() -> dict:
    return {
        "schema": R.STUDY.SCHEMA,
        "purpose": "post_exposure_registry_action_contract_development_study",
        "base_comparison_protocol": {"path": R._relative(R.COMPARISON_PROTOCOL),
                                     "sha256_lf": R.COMPARISON_SHA256_LF},
        "base_comparison_anchor": R.COMPARISON_ANCHOR,
        "prior_comparison_result_sha256": R.PRIOR_COMPARISON_RESULT_SHA256,
    }


def _prepare(monkeypatch: pytest.MonkeyPatch) -> dict:
    study_path = R.ROOT / "bench/protocols" / R.STUDY_PROTOCOL_NAME
    original_read = R.read
    monkeypatch.setattr(R, "read", lambda path: (
        _study_protocol() if Path(path).resolve() == study_path.resolve() else original_read(path)))
    monkeypatch.setattr(R.STUDY, "prepare_protocol", lambda *args: _study_protocol())
    monkeypatch.setattr(R, "sha", lambda path, lf=False: "a" * 64)
    return R.prepare_protocol(study_path, {f"runtime/{i}": "b" * 64 for i in range(9)})


def test_protocol_fixes_single_case_rule_and_exact_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _prepare(monkeypatch)
    assert value["preselected_case"] == {
        "case_id": "c03", "source_seed": 2320, "category": "R1_ONLY", "policy": "eager_model"}
    assert value["repair"] == {
        "correction_id": "restore_configured_ppo_epochs", "selected_by": "registered_host_rule",
        "mode": "fresh_restart", "arm": "FIRST", "original_faults": ["R1"],
        "applied_corrections": ["restore_configured_ppo_epochs"], "checkpoint_loaded": False,
        "source_checkpoint_provenance_only": True}
    assert value["selection_gate"]["option_id"] == "probe:P_recorded_optimizer_updates"
    assert value["selection_gate"]["selected_by"] == "model_option_id"
    assert value["preselected_recovery_seed"] == 3332 and value["smoke_seed"] == 4332
    assert value["bounds"] == {"sdk_jobs": 3, "api_requests": 0, "corrections": 1,
                               "wall_s": 5400, "automatic_retry": False}


def test_current_sources_prepare_and_real_manifest_contract(
    tmp_path: Path,
) -> None:
    study = R.ROOT / "bench/protocols" / R.STUDY_PROTOCOL_NAME
    comparison = json.loads(R.COMPARISON_PROTOCOL.read_text(encoding="utf-8"))
    prepared = R.prepare_protocol(study, comparison["runtime_source_sha256_lf"])
    assert prepared["source_sha256_lf"]["evals/evidence_guided_recovery_v1.py"] \
        == R.sha(R.ROOT / "evals/evidence_guided_recovery_v1.py", lf=True)
    protocol_path = R.ROOT / "bench/protocols/runtime_observation_recovery_s2311_20261031_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fixed = R.TRAIN.build_manifest(
        protocol_path, protocol, "b3e60b6704b94c4b6f9ff7e8d550ee1792082cfd",
        stage=tmp_path / "stage",
        seed=3332, arm="FIRST", original_faults=["R1"],
        applied_corrections=["restore_configured_ppo_epochs"],
        source_checkpoint_sha256="c" * 64, run_name="egr_repair_s3332")
    assert fixed["arm"] == "FIRST"
    assert fixed["original_faults"] == ["R1"]
    assert fixed["applied_corrections"] == ["restore_configured_ppo_epochs"]
    assert fixed["source_checkpoint_sha256"] == "c" * 64


def test_actual_lease_schema_accepts_registered_owner_and_rejects_other_owner() -> None:
    caller, child, wrapper = 40202, 40201, 40200
    metadata = {"owner": R.OWNER, "state": "running", "childPid": child,
                "wrapperPid": wrapper, "completedUtc": None,
                "acquiredUtc": "2026-10-30T12:00:00.1234567Z",
                "heartbeatUtc": "2026-10-30T12:00:30.7654321Z",
                "leaseId": "11111111-1111-1111-1111-111111111111"}
    ancestors = [{"pid": caller, "parent": child, "name": "python.exe"},
                 {"pid": child, "parent": wrapper, "name": "python.exe"},
                 {"pid": wrapper, "parent": 1, "name": "pwsh.exe"}]
    now = datetime(2026, 10, 30, 12, 0, 31, tzinfo=UTC)
    checked = R.FIRST.LEASE.B.validate_lease(metadata, ancestors, caller, now)
    assert R._lease_admission(checked)["owner"] == R.RUN.OWNER
    wrong = {**metadata, "owner": "rl-triage-agent"}
    with pytest.raises(ValueError, match="소유자|신원"):
        R.FIRST.LEASE.B.validate_lease(wrong, ancestors, caller, now)
    with pytest.raises(ValueError, match="owner/state"):
        R._lease_admission(wrong)


def test_missing_comparison_is_hold_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _write(tmp_path / "protocol.json", {})
    source = _write(tmp_path / "source.json", {"status": "complete"})
    monkeypatch.setattr(R, "verify_protocol", lambda *args: {"repair": {}})
    value = R.prepare_candidate(protocol, "a" * 40, source, tmp_path / "missing.json",
                                tmp_path / "study", lab=tmp_path, out=tmp_path / "out")
    assert value["status"] == "HOLD"
    assert value["reason"] == "required_result_missing"
    assert value["approval_granted"] is False


def test_old_comparison_result_is_not_reinterpreted_as_option_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _write(tmp_path / "protocol.json", {})
    source = _write(tmp_path / "source.json", {"status": "complete"})
    old = _write(tmp_path / "old-comparison.json", {
        "schema": R.RUN.RESULT_SCHEMA, "status": "complete"})
    monkeypatch.setattr(R, "verify_protocol", lambda *args: {"repair": {}})
    value = R.prepare_candidate(protocol, "a" * 40, source, old, tmp_path / "study",
                                lab=tmp_path, out=tmp_path / "out")
    assert value["status"] == "HOLD"
    assert value["reason"] == "selection_study_not_complete"
    assert value["approval_granted"] is False


def test_other_option_or_hold_does_not_open_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _write(tmp_path / "protocol.json", {})
    source = _write(tmp_path / "source.json", {"status": "complete"})
    study_protocol = R.ROOT / "bench/protocols" / R.STUDY_PROTOCOL_NAME
    rows = [{"case_id": f"c{index:02d}", "policy": "eager_model",
             "status": "complete", "decision_outcome": "HOLD",
             "confirmed_family": None, "model_actions": 1, "paired_probes": 0}
            for index in range(1, 9)]
    rows += [{**row, "policy": "evidence_guided_model"} for row in rows]
    study = _write(tmp_path / "study.json", {
        "schema": R.STUDY.RESULT_SCHEMA, "status": "complete",
        "protocol": R.ref(study_protocol), "source_batch": R.ref(source),
        "planned_rows": 16, "rows": rows})
    monkeypatch.setattr(R, "verify_protocol", lambda *args: {
        "selection_study": {"path": R._relative(study_protocol)}, "repair": {}})
    value = R.prepare_candidate(protocol, "a" * 40, source, study, tmp_path / "study-root",
                                lab=tmp_path, out=tmp_path / "out")
    assert value["status"] == "HOLD"
    assert value["reason"] == "model_selection_not_confirmed"
    assert value["approval_granted"] is False


def test_approve_requires_eligible_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = _write(tmp_path / "held.json", {
        "schema": R.CANDIDATE_SCHEMA, "status": "HOLD", "eligible": False,
        "approval_granted": False})
    with pytest.raises(ValueError, match="eligible"):
        R.approve(held, actor="root", out=tmp_path / "approval")
    eligible = _write(tmp_path / "eligible.json", {
        "schema": R.CANDIDATE_SCHEMA, "status": "eligible", "eligible": True,
        "approval_granted": False})
    monkeypatch.setattr(R, "verify_candidate", lambda path: json.loads(path.read_text()))
    approved = R.approve(eligible, actor="root", out=tmp_path / "approved")
    assert approved["correction_id"] == R.CORRECTION
    assert approved["recovery_seed"] == 3332
    assert approved["approval_granted"] is True


def test_model_selection_rejects_other_tool_or_invalid_response(
    tmp_path: Path,
) -> None:
    epoch = tmp_path / "epoch"
    step = epoch / "step-01"
    dialogue = {
        "model_valid": True,
        "action": {"action": "run_probe", "target": "P_reward", "fact_ids": ["x"]},
        "calls": [{"messages_sha256": "a" * 64,
                   "result": {"provider_model": next(iter(R.CALL.ACCEPTED_PROVIDER_MODELS)),
                              "response_id": "r"},
                   "review": {"status": "valid"}}],
    }
    _write(step / "dialogue.json", dialogue)
    result = {"steps": [{"selected_by": "model", "action": dialogue["action"]}]}
    state = {"status": "confirmed", "hypotheses": [R.COMPONENT]}
    with pytest.raises(ValueError, match="정확히 하나"):
        R._model_selection(epoch, result, state)


def test_model_selection_replays_option_response_and_root_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch = tmp_path / "epoch"
    stage = epoch / "step-01"
    before = {"state_digest": "s", "status": "open"}
    action = {"action": "run_probe", "target": R.PROBE, "fact_ids": ["f"]}
    review = {"status": "valid", "selected_option_id": R.OPTION_ID}
    dialogue = {
        "schema": "registry_action_options_dialogue_v1", "model_valid": True,
        "state_digest": "s", "action": action, "final_review": review,
        "calls": [{"messages_sha256": "a" * 64,
                   "result": {"provider_model": next(iter(R.CALL.ACCEPTED_PROVIDER_MODELS)),
                              "response_id": "r", "content": "raw"},
                   "review": review}],
    }
    _write(stage / "state-before.json", before)
    _write(stage / "option-context.json", {"context": True})
    _write(stage / "dialogue.json", dialogue)
    _write(stage / "selected-action.json", {
        "selected_by": "model_option_id", "selected_option_id": R.OPTION_ID,
        "dialogue": R.ref(stage / "dialogue.json"), "action": action})
    _write(stage / "request-spec.json", {"spec": True})
    approval = _write(stage / "probe/root-approval.json", {"approval_granted": True})
    _write(stage / "paired-proof.json", {"root_approval": R.ref(approval)})
    _write(stage / "state-after.json", {"status": "confirmed"})
    monkeypatch.setattr(R.OPTIONS, "context", lambda state: {"context": True})
    monkeypatch.setattr(R.OPTIONS, "assess", lambda content, state: review)
    monkeypatch.setattr(R.SEL, "request_spec", lambda state, selected: {"spec": True})
    result = {"steps": [{"selected_by": "model_option_id", "selected_option_id": R.OPTION_ID,
                          "action": action, "dialogue": R.ref(stage / "dialogue.json"),
                          "state_after": R.ref(stage / "state-after.json")}]}
    model = R._model_selection(epoch, result, {"status": "confirmed", "hypotheses": [R.COMPONENT]})
    assert model["option_id"] == R.OPTION_ID
    assert model["selection_subject"] == "model_option_id"
    assert model["root_approval"] == R.ref(approval)


def test_tag_contract_rejects_hyphen() -> None:
    assert R.TAG.fullmatch("egr_repair_s3332")
    assert R.TAG.fullmatch("egr-repair-s3332") is None


def test_runtime_20_requires_all_six_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    facts = {f"runtime:case:{name}": 20 for name in (
        "expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max",
        "expected_minibatches", "minibatches_min", "minibatches_max")}
    facts["runtime:case:updates_observed"] = 3
    monkeypatch.setattr(R.FACT, "checked", lambda *args, **kwargs: ({"ok": True}, facts))
    assert R._runtime_20(tmp_path / "facts.json", "a" * 64) == {"ok": True}
    facts["runtime:case:optimizer_steps_min"] = 5
    with pytest.raises(ValueError, match="20/20"):
        R._runtime_20(tmp_path / "facts.json", "a" * 64)
