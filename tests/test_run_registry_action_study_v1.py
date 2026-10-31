# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_registry_action_study_v1 as S


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_prepare_protocol_binds_prior_and_new_option_sources() -> None:
    value = S.prepare_protocol(
        ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json", "a" * 64)
    assert value["schema"] == S.SCHEMA
    assert value["prior_comparison_result_sha256"] == "a" * 64
    assert value["design"]["planned_rows"] == 16
    assert value["design"]["normal_host_gate_rows"] == 4
    assert value["bounds"] == {"total_wall_s": 14400, "global_api_requests": 96,
                               "paired_probes": 36, "sdk_role_executions": 72}
    assert set(value["source_sha256_lf"]) == set(S.SOURCE_FILES)
    rendered = json.dumps(value)
    assert "bench/private" not in rendered and "C:\\Users" not in rendered


def test_prepare_rejects_bad_prior_digest() -> None:
    with pytest.raises(ValueError, match="prior result SHA"):
        S.prepare_protocol(
            ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json", "no")


def test_prior_requires_complete_24_rows_and_exact_sha(tmp_path: Path) -> None:
    base = ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json"
    rows = [{"case_id": f"c{i:02d}", "policy": policy}
            for i in range(1, 9) for policy in S.COMP.POLICIES]
    path = _json(tmp_path / "prior.json", {
        "schema": S.RUN.RESULT_SCHEMA, "status": "complete", "rows": rows,
        "protocol": S._ref(base),
    })
    protocol = S.prepare_protocol(base, S._sha(path))
    assert len(S._prior(path, protocol)["rows"]) == 24
    value = json.loads(path.read_text())
    value["rows"].pop()
    _json(path, value)
    protocol["prior_comparison_result_sha256"] = S._sha(path)
    with pytest.raises(ValueError, match="8×3"):
        S._prior(path, protocol)


def test_dialogue_metrics_preserves_first_and_feedback_denominators(tmp_path: Path) -> None:
    dialogue = {
        "calls": [
            {"review": {"status": "invalid", "codes": ["F_ACTION"]}},
            {"review": {"status": "valid", "codes": []}},
        ],
        "model_valid": True,
        "semantic_feedback_used": True,
    }
    path = _json(tmp_path / "step-01" / "dialogue.json", dialogue)
    epoch = {"steps": [{"action": {"action": "run_probe"},
                         "dialogue": {"path": str(path), "sha256": S._sha(path)}}]}
    assert S._dialogue_metrics(tmp_path, epoch) == {
        "first_decision_valid": False,
        "post_feedback_first_decision_valid": True,
        "first_decision_invalid_codes": ["F_ACTION"],
        "feedback_actions": 1, "model_requests": 2,
        "read_actions": 0, "probe_actions": 1, "hold_actions": 0,
    }


def test_dialogue_ref_cannot_escape_epoch(tmp_path: Path) -> None:
    path = _json(tmp_path / "outside.json", {"calls": [{"review": {"status": "valid"}}]})
    with pytest.raises(ValueError, match="dialogue raw ref"):
        S._dialogue_metrics(tmp_path / "epoch", {"steps": [{
            "action": {"action": "hold"},
            "dialogue": {"path": str(path), "sha256": S._sha(path)}}]})


def test_prior_first_decision_does_not_credit_feedback_or_later_action(tmp_path: Path) -> None:
    rows = []
    for case in range(1, 7):
        for policy in S.POLICIES:
            epoch_root = tmp_path / f"c{case:02d}" / policy
            first = _json(epoch_root / "step-01" / "dialogue.json", {
                "calls": [{"review": {"status": "invalid", "codes": ["F_ACTION"]}},
                          {"review": {"status": "valid", "codes": []}}],
                "model_valid": True, "semantic_feedback_used": True})
            later = _json(epoch_root / "step-02" / "dialogue.json", {
                "calls": [{"review": {"status": "valid", "codes": []}}],
                "model_valid": True, "semantic_feedback_used": False})
            epoch = {
                "schema": "prospective_agentic_epoch_v1", "status": "complete",
                "run_id": f"c{case:02d}.{policy}",
                "steps": [{"dialogue_sha256": S._sha(first)},
                          {"dialogue_sha256": S._sha(later)}],
            }
            _json(epoch_root / "result.json", epoch)
            rows.append({"case_id": f"c{case:02d}", "policy": policy,
                         "gate": {"run": True}, "first_pass_valid_actions": 1,
                         "post_feedback_valid_actions": 2, "semantic_feedback_actions": 1})
    assert S._prior_first_decisions(tmp_path / "result.json", rows) == 0


def test_anchor_check_rejects_modified_or_uncommitted_protocol(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    protocol = _json(tmp_path / "study.json", {"schema": "study"})
    subprocess.run(["git", "add", "study.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "test"], cwd=tmp_path, check=True)
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                            capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(S, "ROOT", tmp_path)
    S._verify_anchor_file(protocol, anchor)
    _json(protocol, {"schema": "changed"})
    with pytest.raises(ValueError, match="anchor protocol blob"):
        S._verify_anchor_file(protocol, anchor)
    untracked = _json(tmp_path / "replacement.json", {"schema": "study"})
    with pytest.raises(subprocess.CalledProcessError):
        S._verify_anchor_file(untracked, anchor)


def test_execute_keeps_12_fault_and_4_normal_rows_separate(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = _json(tmp_path / "protocol.json", {"schema": S.SCHEMA})
    source = _json(tmp_path / "source.json", {"source": True})
    prior_path = _json(tmp_path / "prior.json", {"prior": True})
    runtime = _json(tmp_path / "runtime.json", {"runtime": True})
    categories = ("NONE", "F1", "R1_ONLY", "COMPOUND")
    cases = []
    for index in range(8):
        category = categories[index % 4]
        cases.append({
            "case_id": f"c{index + 1:02d}", "seed": 2320 + index // 4,
            "category": category, "input": tmp_path / f"input-{index}.json",
            "checkpoint": {"sha256": f"{index + 1:064x}"},
            "metadata": {"runtime_facts": {
                "case": {"path": str(runtime)}, "reference": {"path": str(runtime)}}},
        })
    prior_rows = [{"case_id": f"c{i:02d}", "policy": policy,
                   "gate": {"run": category != "NONE"}, "first_pass_valid_actions": 0}
                  for i, category in enumerate(categories * 2, 1)
                  for policy in S.COMP.POLICIES]
    spec = {"base_comparison_protocol": {"path": "base.json"},
            "claims": {"generalization": False}}
    monkeypatch.setattr(S, "_validate_inputs", lambda *args: (spec, cases, {"rows": prior_rows}))
    monkeypatch.setattr(S, "_verify_current_protocol", lambda *args: None)
    monkeypatch.setattr(S.RUN, "_lease", lambda: {"leaseId": "lease-1"})
    monkeypatch.setattr(S.RUN, "BANDS", _json(tmp_path / "bands.json", {"bands": {}}))
    monkeypatch.setattr(S.T4, "status", lambda root: {"requests": 0})
    monkeypatch.setattr(S.COMP, "transport_factory", lambda root: object())
    monkeypatch.setattr(S.RUN, "_gate", lambda case, reference, bands: (
        {"run": False, "reason": "healthy_gate", "source_outcome": "recovered"}
        if case["category"] == "NONE" else
        {"run": True, "reason": None, "source_outcome": "not_recovered"}))
    monkeypatch.setattr(S, "_inputs", lambda case, reference, root, eager: {
        "state": {"eager": eager}, "envelope": {}, "case_runtime": runtime,
        "reference_runtime": runtime})
    monkeypatch.setattr(S, "_dialogue_metrics", lambda root, epoch: {
        "first_decision_valid": True, "post_feedback_first_decision_valid": True,
        "first_decision_invalid_codes": [], "feedback_actions": 0,
        "model_requests": 1, "read_actions": 0, "probe_actions": 0, "hold_actions": 1})
    monkeypatch.setattr(S, "_prior_first_decisions", lambda path, rows: 0)

    def fake_epoch(state: dict, **kwargs: object) -> dict:
        return {"status": "complete", "decision_outcome": "HOLD", "model_actions": 1,
                "paired_probes": 0, "parent_probe_wall_s": 0.0,
                "owned_child_wall_s": 0.0, "final_status": "hold",
                "confirmed_family": None, "steps": []}

    result = S.execute(protocol_path, "b" * 40, source, prior_path, tmp_path,
                       tmp_path / "out", tmp_path / "queue", run_epoch=fake_epoch)
    assert result["planned_rows"] == 16
    assert result["fault_rows"] == 12
    assert result["normal_host_gate_rows"] == 4
    normal = [row for row in result["rows"] if not row["gate"]["run"]]
    assert len(normal) == 4
    assert all(row["model_requests"] == 0 and row["normal_gate_model_success"] is False
               for row in normal)
    assert result["validity"]["first_decision_denominator"] == 12
    assert result["format_candidate"] is True
    assert result["claims"]["generalization"] is False
    assert len(list((tmp_path / "out").glob("c*/**/proof-cache-stats.json"))) == 12


def test_cli_prepare_is_write_once(tmp_path: Path) -> None:
    out = tmp_path / "protocol.json"
    args = ["prepare-protocol", "--base-comparison-protocol",
            str(ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json"),
            "--prior-comparison-result-sha256", "c" * 64, "--out", str(out)]
    assert S.main(args) == 0 and out.is_file()
    with pytest.raises(ValueError, match="write-once"):
        S.main(args)
