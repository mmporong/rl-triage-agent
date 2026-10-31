# pyright: reportMissingImports=false
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from evals import export_model_hold_handoff_trial_v1 as E


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


@pytest.fixture
def trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    execution = tmp_path / "execution"
    probe = execution / "probe"
    refs = {}
    for role in ("case", "reference"):
        plan = probe / role / "plan.json"
        receipt = probe / role / "probe_receipt.json"
        _write(plan, {"schema": "plan"})
        _write(receipt, {"schema": "runtime_observation_probe_receipt_v1",
            "status": "complete", "probe": E.TRIAL.PROBE, "role": role,
            "checkpoint_sha256": ("a" if role == "case" else "b") * 64,
            "measurement": {"track_lin_vel_xy_exp_rel_error": 0.32 if role == "case" else 0.01,
                            "track_ang_vel_z_exp_rel_error": 0.0}})
        refs[f"{role}_plan"], refs[f"{role}_receipt"] = str(plan), str(receipt)
    action = {"action": "run_probe", "target": "P_reward",
              "fact_ids": list(E.TRIAL.FOLLOW.REWARD_FACT_IDS)}
    shadow = {"state_digest": "shadow"}
    approval = probe / "root-approval.json"
    _write(approval, {"actor": E.TRIAL.APPROVER, "approval_granted": True,
        "action": action, "state_digest": "shadow", "request_spec": {
            "action": action, "approval_granted": False, "model_generated_args": False,
            "model_generated_cause": False, "model_generated_repair": False}})
    host = probe / "host-result.json"
    _write(host, {"schema": "evidence_guided_probe_host_result_v1",
                  "scope": "legacy_registered_paired_probe", **refs})
    result = execution / "result.json"
    _write(result, {"schema": E.TRIAL.RESULT_SCHEMA, "status": "complete",
        "case_id": "c02", "reference_case_id": "c01",
        "selector_actor": E.TRIAL.SELECTOR_ACTOR, "approval_actor": E.TRIAL.APPROVER,
        "original_model_selected_probe": False, "original_model_result_changed": False,
        "parent_model_outcome": "HOLD", "proposal": action, "probe_outcome": "abnormal",
        "host_result": {"path": str(host), "sha256": E._sha(host)},
        "state_after_digest": "after", "api_requests": 0, "paired_probes": 1,
        "sdk_role_executions": 2, "training": False, "correction": False,
        "recovery": False, "new_observation_claim": False,
        "model_aggregate_changed": False, "parent_wall_s": 136.765})
    proof = {name: {"path": refs[name], "sha256": E._sha(Path(refs[name]))} for name in refs}
    proof["root_approval"] = {"path": str(approval), "sha256": E._sha(approval)}
    monkeypatch.setattr(E.COVER.LOOP, "_pair_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(E.TRIAL.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {"P_reward": "abnormal"}, "state_digest": "after"})
    proposal = {"proposal": action, "approval_granted": False, "execution_allowed": False,
                "correction_allowed": False, "automatic_resume": False}
    source = {"case": {"checkpoint": {"sha256": "a" * 64}},
              "reference": {"checkpoint": {"sha256": "b" * 64}}}
    return {"result": result, "inputs": {"proposal": proposal, "shadow": shadow,
                                           "action": action, "source": source},
            "approval": approval}


def test_export_writes_five_safe_files_and_separates_actors(
        trial: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    files = {name: tmp_path / f"{name}.json" for name in (
        "protocol", "source", "mode", "corpus", "epoch", "study", "audit")}
    for path in files.values():
        _write(path, {})
    monkeypatch.setattr(E.TRIAL, "verify_protocol", lambda *_: {
        "parent_reasoning_protocol": {"path": "parent.json"},
        "prior_coverage_audit": {"path": "audit.json"}})
    monkeypatch.setattr(E.TRIAL, "inspect_inputs", lambda *_: trial["inputs"])

    @contextmanager
    def scope():
        yield {"approval": False, "execution_allowed": False, "hits": 1}

    monkeypatch.setattr(E.TRIAL.CACHE, "proof_scope", scope)
    out = tmp_path / "public"
    manifest = E.export(
        trial["result"], files["protocol"], files["source"], files["mode"],
        files["corpus"], files["epoch"], files["study"], tmp_path / "lab",
        anchor="1" * 40, out=out)
    assert len(list(out.rglob("*.json"))) == 5
    summary = E._read(out / "summary.json")
    assert summary["selector_actor"] == E.TRIAL.SELECTOR_ACTOR
    assert summary["approval_actor"] == E.TRIAL.APPROVER
    assert summary["parent_model_result_changed"] is False
    assert summary["claims"]["model_aggregate_improvement"] is False
    assert summary["scope"]["integration_trial_not_independent_case"] is True
    assert manifest["artifact_inventory"] == sorted(manifest["files"])
    assert "C:\\" not in json.dumps(manifest)


def test_root_approval_actor_tamper_is_rejected(trial: dict) -> None:
    approval = E._read(trial["approval"])
    approval["actor"] = "model"
    _write(trial["approval"], approval)
    with pytest.raises(ValueError, match="root approval"):
        E._verify_execution(trial["result"], trial["inputs"])


def test_original_proposal_authority_or_replayed_outcome_tamper_is_rejected(
        trial: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    trial["inputs"]["proposal"]["approval_granted"] = True
    with pytest.raises(ValueError, match="비권한"):
        E._proposal_authority(trial["inputs"]["proposal"])
    trial["inputs"]["proposal"]["approval_granted"] = False
    monkeypatch.setattr(E.TRIAL.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {"P_reward": "normal"}, "state_digest": "wrong"})
    with pytest.raises(ValueError, match="outcome/state digest"):
        E._verify_execution(trial["result"], trial["inputs"])
