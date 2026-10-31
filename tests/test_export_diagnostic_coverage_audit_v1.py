# pyright: reportMissingImports=false
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from evals import export_diagnostic_coverage_audit_v1 as E


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    execution = tmp_path / "execution"
    probe = execution / "probe"
    refs = {}
    for role in ("case", "reference"):
        plan = probe / role / "plan.json"
        receipt = probe / role / "probe_receipt.json"
        _write(plan, {"schema": "plan"})
        _write(receipt, {"schema": "runtime_observation_probe_receipt_v1",
            "status": "complete", "probe": E.AUDIT.PROBE, "role": role,
            "checkpoint_sha256": ("a" if role == "case" else "b") * 64,
            "measurement": {"track_lin_vel_xy_exp_rel_error": 0.32 if role == "case" else 0.01,
                            "track_ang_vel_z_exp_rel_error": 0.0}})
        refs[f"{role}_plan"], refs[f"{role}_receipt"] = str(plan), str(receipt)
    approval = probe / "root-approval.json"
    action = {"action": "run_probe", "target": E.AUDIT.PROBE,
              "fact_ids": list(E.AUDIT.FACT_IDS)}
    _write(approval, {"actor": E.AUDIT.APPROVER, "approval_granted": True,
                      "action": action})
    host_path = probe / "host-result.json"
    _write(host_path, {"schema": "evidence_guided_probe_host_result_v1",
                       "scope": "legacy_registered_paired_probe", **refs})
    result_path = execution / "result.json"
    _write(result_path, {"schema": E.AUDIT.RESULT_SCHEMA, "status": "complete",
        "case_id": E.AUDIT.CASE_ID, "reference_case_id": E.AUDIT.REFERENCE_ID,
        "selection_actor": E.AUDIT.APPROVER, "model_selected_probe": False,
        "original_model_outcome": "HOLD", "probe": E.AUDIT.PROBE,
        "outcome": "abnormal", "interpretation": E.AUDIT._interpret("abnormal"),
        "host_result": {"path": str(host_path), "sha256": E._sha(host_path)},
        "state_after_digest": "after", "api_requests": 0, "paired_probes": 1,
        "sdk_role_executions": 2, "training": False, "correction": False,
        "recovery": False, "other_probe_outcomes_inferred": False,
        "parent_wall_s": 115.938})
    proof = {name: {"path": refs[name], "sha256": E._sha(Path(refs[name]))} for name in refs}
    proof["root_approval"] = {"path": str(approval), "sha256": E._sha(approval)}
    monkeypatch.setattr(E.LOOP, "_pair_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(E.AUDIT.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {E.AUDIT.PROBE: "abnormal"}, "state_digest": "after"})
    inputs = {"first": {"state": {"state_digest": "before"}}, "action": action,
              "case": {"checkpoint": {"sha256": "a" * 64}},
              "reference": {"checkpoint": {"sha256": "b" * 64}}}
    return {"execution": execution, "result": result_path, "inputs": inputs,
            "host": host_path, "proof": proof}


def test_export_writes_five_safe_files_and_operator_scope(
    audit: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    protocol = tmp_path / "protocol.json"
    for path in (protocol, tmp_path / "source.json", tmp_path / "mode.json",
                 tmp_path / "corpus.json", tmp_path / "epoch.json"):
        _write(path, {})
    monkeypatch.setattr(E.AUDIT, "verify_protocol", lambda *_: {
        "parent_reasoning_protocol": {"path": "parent.json"}})
    monkeypatch.setattr(E.AUDIT, "inspect_inputs", lambda *_: audit["inputs"])

    @contextmanager
    def scope():
        yield {"immutable_git_show_cache_hits": 1}

    monkeypatch.setattr(E.AUDIT.CACHE, "proof_scope", scope)
    out = tmp_path / "public"
    manifest = E.export(audit["result"], protocol, tmp_path / "source.json",
                        tmp_path / "mode.json", tmp_path / "corpus.json",
                        tmp_path / "epoch.json", tmp_path / "lab", anchor="4" * 40, out=out)
    assert len(list(out.rglob("*.*"))) == 5
    summary = E._read(out / "summary.json")
    assert summary["selection_actor"] == E.AUDIT.APPROVER
    assert summary["model_selected_probe"] is False
    assert summary["registered_prior_prediction_supported"] is False
    assert summary["claims"]["other_probe_outcomes_inferred"] is False
    assert manifest["artifact_inventory"] == sorted(manifest["files"])
    assert "C:\\" not in json.dumps(manifest)


def test_host_result_sha_tamper_is_rejected(audit: dict) -> None:
    result = E._read(audit["result"])
    result["host_result"]["sha256"] = "0" * 64
    _write(audit["result"], result)
    with pytest.raises(ValueError, match="host_result 경로/SHA"):
        E._verify_execution(audit["result"], audit["inputs"])


def test_replayed_outcome_or_state_tamper_is_rejected(
    audit: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E.AUDIT.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {E.AUDIT.PROBE: "normal"}, "state_digest": "wrong"})
    with pytest.raises(ValueError, match="outcome/state digest"):
        E._verify_execution(audit["result"], audit["inputs"])
