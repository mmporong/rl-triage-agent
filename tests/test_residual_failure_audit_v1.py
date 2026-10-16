from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import audit_residual_failure_v1 as A


def packet():
    return {"schema": A.C.SCHEMA, "facts": {"eval:lin_error": 1.2}, "observed": [],
            "budget_left": A.C.MAX_PROBES, "execution_budget_s": 1200,
            "epoch": 2, "applied_corrections": [A.C.REWARD],
            "prior_receipt": {"verdict": "not_recovered", "correction_id": A.C.REWARD,
                              "result_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}}


def decision():
    categories = list(A.C.L.MECHANISMS)
    return {"ranking": categories, "probe": "P_reward", "action": "continue",
            "reason": "학습 평가의 추종 오차를 확인한다",
            "claims": [{"id": "eval:lin_error", "value": 1.2}],
            "expected_outcomes": {key: A.C.R.public_catalog()["P_reward"]["expect"][key]
                                  for key in categories}}


def test_exact_top_three_contract_is_satisfiable_on_same_epoch2_packet():
    bad = decision()
    content = json.dumps(bad)
    with A.MC.residual_contract():
        assert A.MC.R.validate("diagnosis", content, packet())["codes"] == ["D_EXPECTED"]
    checks = A.diagnosis_details(content, packet())
    assert checks["ranking_is_public_permutation"] and checks["claims_match_packet"]
    assert not checks["expected_outcomes_matches_top_three"]
    good = deepcopy(bad)
    good["expected_outcomes"] = {key: bad["expected_outcomes"][key] for key in bad["ranking"][:3]}
    assert A.C.validate_diagnosis(good, packet()) == good
    assert A.diagnosis_details(json.dumps(good), packet())["expected_outcomes_matches_top_three"]
    assert len(bad["expected_outcomes"]) == 6


def test_subtype_ranking_and_expected_keys_are_two_distinct_errors():
    bad = decision()
    bad["ranking"][3:4] = ["optimizer_value", "optimizer_update_count"]
    with A.MC.residual_contract():
        assert A.MC.R.validate("diagnosis", json.dumps(bad), packet())["codes"] == ["D_RANKING"]
    checks = A.diagnosis_details(json.dumps(bad), packet())
    assert not checks["ranking_is_public_permutation"]
    assert not checks["expected_outcomes_matches_top_three"]
    bad["claims"][0]["value"] = 9.9
    assert not A.diagnosis_details(json.dumps(bad), packet())["claims_match_packet"]


def test_boundary_analysis_is_hypothetical_and_does_not_change_input():
    p, d = packet(), decision()
    before = deepcopy((p, d))
    result = A.boundary_analysis(json.dumps(d), p)
    assert (p, d) == before
    assert result["format_corrected_decision_accepted_by_v1"]
    assert result["probe"] == "P_reward"
    assert set(result["probe_expectations"].values()) == {"normal"}
    assert "reward" not in result["remaining_hypotheses"]
    assert not result["partitions_remaining_hypotheses"]
    assert result["registered_rule_next_probe"] == "P_value"
    assert result["analysis_only"] and not result["original_experimental_verdict_changed"]
    assert not result["rule_probe_executed"]


def test_public_record_keeps_epoch2_hold_and_all_calls_visible():
    assert A.verify_public()["files"] == 10
    summary = A.D.read(A.OUTPUT / "summary.json")
    audit = A.D.read(A.OUTPUT / "model_contract_audit.json")
    assert summary["classification"] == "model_contract_violation"
    assert not summary["demo_success"] and not summary["retry_as_infrastructure_failure"]
    for case in summary["cases"]:
        assert case["first_diagnosis_valid_after_feedback"] and case["first_receipt_verified"]
        assert case["epoch2_opened"] and case["stop_epoch"] == 2
        assert case["epoch2_probes_executed"] == 0 and not case["second_correction_executed"]
    assert (audit["raw_valid"], audit["raw_invalid"]) == (4, 6)
    assert (audit["dialogues_valid"], audit["dialogues_invalid"]) == (4, 2)
    assert {row["finish_reason"] for row in audit["raw_calls"]} == {"stop"}
    for seed in (2110, 2111):
        assert [row["codes"] for row in audit["raw_calls"]
                if row["seed"] == seed and row["epoch"] == 2] == [["D_RANKING"], ["D_EXPECTED"]]


@pytest.mark.parametrize("with_manifest", [False, True])
def test_public_verifier_rejects_payload_and_manifest_tamper(tmp_path, with_manifest):
    folder = tmp_path / "public"
    shutil.copytree(A.OUTPUT, folder)
    summary = A.D.read(folder / "summary.json")
    summary["demo_success"] = True
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_manifest:
        provenance = A.D.read(folder / "provenance.json")
        provenance["payload_sha256"]["summary.json"] = A.digest(summary)
        (folder / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="고정 내용 결합"):
        A.verify_public(folder)


@pytest.mark.parametrize("value", [r"C:\Users\sample\secret", "/home/sample/file", "bench/private/run"])
def test_public_path_guard_checks_nested_decoded_json(value):
    decoded = json.loads(json.dumps({"row": [{"nested": value}]}))
    with pytest.raises(ValueError, match="비공개 경로"):
        A.public_safe(decoded)


def test_export_never_overwrites_existing_results(tmp_path, monkeypatch):
    folder = tmp_path / "existing"
    folder.mkdir()
    sentinel = folder / "sentinel"
    sentinel.write_bytes(b"keep")
    monkeypatch.setattr(A, "OUTPUT", folder)
    with pytest.raises(FileExistsError):
        A.export(tmp_path)
    assert sentinel.read_bytes() == b"keep"


def test_cli_public_verification_needs_no_private_run_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "evals/audit_residual_failure_v1.py"), "verify"],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"
