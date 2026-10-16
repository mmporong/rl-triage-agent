from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import export_residual_success_v2 as X  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def payloads() -> dict[str, dict]:
    decision_metrics = {"diagnosis_opportunities": 4, "completed_diagnoses": 4,
        "infrastructure_or_call_errors": 0,
        "model_validity": {"numerator": 4, "denominator": 4, "rate": 1.0},
        "format_validity": {"numerator": 4, "denominator": 4, "rate": 1.0},
        "semantic_validity_given_format": {"numerator": 4, "denominator": 4, "rate": 1.0},
        "first_pass_validity": {"numerator": 4, "denominator": 4, "rate": 1.0},
        "rule_transition": {"numerator": 0, "denominator": 4, "rate": 0.0},
        "rule_transition_given_invalid": {"numerator": 0, "denominator": 0, "rate": None}}
    summary = {"schema": "summary", "value": 1,
               "decision_metrics": decision_metrics,
               "model_validity": decision_metrics["model_validity"],
               "rule_transition": decision_metrics["rule_transition"],
               "end_to_end_recovery": {"numerator": 2, "denominator": 2, "rate": 1.0}}
    provenance = {"schema": "provenance", "payload_sha256": {"summary.json": X.digest(summary)}}
    return {"summary.json": summary, "provenance.json": provenance}


def pin(monkeypatch, values: dict[str, dict]) -> None:
    monkeypatch.setattr(X, "PUBLIC_SHA256", {name: X.digest(value) for name, value in values.items()})


def test_public_inventory_and_source_identifiers_are_pinned():
    checked = X.verify_public()
    assert checked["status"] == "valid" and checked["files"] == 12
    provenance = X.D.read(X.OUTPUT / "provenance.json")
    assert provenance["source_result_sha256"] == X.SOURCE_RESULT_SHA
    assert provenance["protocol_sha256_lf"] == X.PROTOCOL_SHA_LF
    assert provenance["experiment_commit"] == X.ANCHOR


def test_public_summary_keeps_decision_and_recovery_metrics_separate():
    summary = X.D.read(X.OUTPUT / "summary.json")
    metrics = summary["decision_metrics"]
    assert metrics["diagnosis_opportunities"] == metrics["completed_diagnoses"] == 4
    assert metrics["infrastructure_or_call_errors"] == 0
    assert metrics["model_validity"] == {"numerator": 4, "denominator": 4, "rate": 1.0}
    assert metrics["rule_transition"] == {"numerator": 0, "denominator": 4, "rate": 0.0}
    assert metrics["rule_transition_given_invalid"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert summary["end_to_end_recovery"] == {"numerator": 2, "denominator": 2, "rate": 1.0}


def test_public_safe_rejects_user_private_and_nested_escaped_paths():
    windows_path = "C:" + "\\" + "Users" + "\\name\\repo"
    escaped_path = "C:" + "\\\\" + "Users" + "\\\\name\\\\secret"
    for value in (windows_path, "/" + "home/name/data", "bench/" + "private/case",
                  {"nested": [escaped_path]}):
        with pytest.raises(ValueError, match="경로"):
            X.public_safe(value)
    X.public_safe({"path_role": "source evaluation", "content": "registered probe"})


def test_payload_and_manifest_cannot_be_changed_together(monkeypatch):
    original = payloads()
    pin(monkeypatch, original)
    X.verify_payloads(original)

    changed = payloads()
    changed["summary.json"]["value"] = 2
    changed["provenance.json"]["payload_sha256"]["summary.json"] = X.digest(changed["summary.json"])
    with pytest.raises(ValueError, match="canonical"):
        X.verify_payloads(changed)


def test_verify_public_rejects_extra_file_and_reads_only_canonical_set(tmp_path, monkeypatch):
    values = payloads()
    pin(monkeypatch, values)
    for name, value in values.items():
        write(tmp_path / name, value)
    result = X.verify_public(tmp_path)
    assert result["status"] == "valid" and result["files"] == 2
    write(tmp_path / "extra.json", {"extra": True})
    with pytest.raises(ValueError, match="파일 목록"):
        X.verify_public(tmp_path)


def test_export_writes_allowlisted_payload_and_immutable_source_receipt(tmp_path, monkeypatch):
    run, output, audit = tmp_path / "run", tmp_path / "public", tmp_path / "audit.json"
    write(run / "source.json", {"fixed": True})
    values = payloads()
    pin(monkeypatch, values)
    monkeypatch.setattr(X, "RUN", run)
    monkeypatch.setattr(X, "OUTPUT", output)
    monkeypatch.setattr(X, "AUDIT", audit)
    monkeypatch.setattr(X, "verify_and_build", lambda _isaaclab: values)
    before = X.snapshot(run)
    result = X.export(tmp_path / "isaaclab")
    assert result == {"status": "exported", "files": 2, "source_unchanged": True}
    assert X.snapshot(run) == before and X.verify_public(output)["status"] == "valid"
    receipt = X.D.read(audit)
    assert receipt["source_before"] == receipt["source_after"] == before
    assert receipt["gpu_executions"] == receipt["model_requests"] == 0
    with pytest.raises(FileExistsError):
        X.export(tmp_path / "isaaclab")


def test_public_evaluation_is_allowlist_not_private_report_copy():
    report = {
        "protocol": "fixed_eval_v1", "protocol_sha256_lf": "a" * 64, "task": "task",
        "eval_seed": 2026, "num_envs": 1040, "horizon_steps": 1000, "step_dt": 0.02,
        "action_scale": 0.25, "versions": {}, "checkpoint": {"file": "model.pt", "sha256": "b" * 64},
        "metrics": {"lin_vel_rmse_mps": 0.1}, "standstill": {}, "by_condition": [],
        "source_report_path": "C:" + "\\" + "Users" + "\\name\\private.json",
        "inventory_sha256": {"secret": "c" * 64},
    }
    public = X.public_evaluation(2120, "NONE", report, "d" * 64, "e" * 64, "recovered")
    assert "source_report_path" not in public and "inventory_sha256" not in public
    X.public_safe(public)


def test_export_rejects_changed_payload_before_creating_output(tmp_path, monkeypatch):
    run, output, audit = tmp_path / "run", tmp_path / "public", tmp_path / "audit.json"
    write(run / "source.json", {"fixed": True})
    values = payloads()
    pin(monkeypatch, values)
    values["summary.json"]["value"] = 2
    values["provenance.json"]["payload_sha256"]["summary.json"] = X.digest(values["summary.json"])
    monkeypatch.setattr(X, "RUN", run)
    monkeypatch.setattr(X, "OUTPUT", output)
    monkeypatch.setattr(X, "AUDIT", audit)
    monkeypatch.setattr(X, "verify_and_build", lambda _isaaclab: values)
    before = X.snapshot(run)
    with pytest.raises(ValueError, match="canonical"):
        X.export(tmp_path / "isaaclab")
    assert not output.exists() and not audit.exists()
    assert X.snapshot(run) == before
