from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from evals import export_actor_reset_budget_v1 as EXPORT


@pytest.fixture(scope="module")
def payloads():
    if not EXPORT.OUTPUT.is_dir():
        pytest.fail("public Task26 export is missing")
    return {name: EXPORT.read(EXPORT.OUTPUT / name) for name in EXPORT.PUBLIC_FILES}


def test_private_integration_recomputes_registered_outcome():
    isaaclab = Path.home() / "IsaacLab"
    if not EXPORT.RUN.is_dir() or not isaaclab.is_dir():
        pytest.skip("private Task25 integration input unavailable")
    payloads = EXPORT.verify_and_build(isaaclab)
    summary = payloads["summary.json"]
    assert summary["outcome"] == {
        "decision": "keep_current_rule", "reset_100_recovered": "0/2",
        "reset_150_recovered": "0/2", "reset_200_recovered": "0/2",
        "restart_300_recovered": "2/2", "candidate_budget_learning_iterations": None,
        "automatic_global_change": False,
    }
    assert payloads["costs.json"]["totals_wall_s"] == {
        "source_preparation": pytest.approx(2097.440999998944),
        "recovery_experiment": pytest.approx(2710.5630000005476),
        "whole_experiment_accounted": pytest.approx(4808.0039999994915),
    }
    assert EXPORT.verify_public_payloads(payloads) == {"status": "valid", "files": 16, "evaluations": 12}


def _rehash(payloads, name):
    body = json.dumps(payloads[name], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    import hashlib
    payloads["provenance.json"]["public_artifact_sha256"][name] = hashlib.sha256(body.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("mutation", [
    "private_path", "missing_file", "extra_seed", "unknown_label", "cost_total", "condition_count", "public_hash",
    "semantic_metric", "semantic_ratio", "cost_row", "checkpoint_identity",
    "source_inventory", "source_crosslink",
])
def test_public_verifier_rejects_tampering(payloads, mutation):
    changed = deepcopy(payloads)
    if mutation == "private_path":
        changed["summary.json"]["interpretation"] = r"bench\private\hidden.json"
    elif mutation == "missing_file":
        changed.pop("costs.json")
    elif mutation == "extra_seed":
        changed["state_evidence.json"]["seeds"]["9999"] = {}
    elif mutation == "unknown_label":
        changed["evaluations/s2090/reset100.json"]["label"] = "invalid"
    elif mutation == "cost_total":
        changed["costs.json"]["totals_wall_s"]["whole_experiment_accounted"] += 1
    elif mutation == "condition_count":
        changed["evaluations/s2090/none.json"]["conditions"].pop()
    elif mutation == "public_hash":
        changed["provenance.json"]["public_artifact_sha256"]["summary.json"] = "0" * 64
    elif mutation == "semantic_metric":
        name = "evaluations/s2090/reset100.json"
        changed[name]["metrics"]["lin_vel_rmse_mps"] = 0.01
        _rehash(changed, name)
    elif mutation == "semantic_ratio":
        changed["costs.json"]["training_wall_ratio_vs_same_seed_restart"]["2090"]["100"] = 0.01
        _rehash(changed, "costs.json")
    elif mutation == "cost_row":
        changed["costs.json"]["recovery_arms"]["100"]["2090"]["total_wall_s"] += 1
        _rehash(changed, "costs.json")
    elif mutation == "checkpoint_identity":
        name = "evaluations/s2090/restart300.json"
        changed[name]["checkpoint"]["sha256"] = "f" * 64
        _rehash(changed, name)
    elif mutation == "source_inventory":
        changed["provenance.json"]["source_artifacts"]["unexpected"] = "f" * 64
    else:
        changed["provenance.json"]["source_artifacts"]["fixed_evaluation.s2090.none"] = "f" * 64
    with pytest.raises(ValueError):
        EXPORT.verify_public_payloads(changed)


def test_public_only_verification_needs_no_private_inputs(payloads, tmp_path):
    output = tmp_path / "public"
    for relative, body in payloads.items():
        EXPORT.write_new(output / relative, body)
    assert EXPORT.verify_public(output) == {"status": "valid", "files": 16, "evaluations": 12}


def test_export_refuses_existing_output(payloads, tmp_path, monkeypatch):
    output, audit = tmp_path / "public", tmp_path / "audit"
    output.mkdir()
    monkeypatch.setattr(EXPORT, "OUTPUT", output)
    monkeypatch.setattr(EXPORT, "AUDIT", audit)
    monkeypatch.setattr(EXPORT, "verify_and_build", lambda isaaclab: payloads)
    with pytest.raises(FileExistsError):
        EXPORT.export(tmp_path / "IsaacLab")


def test_committed_public_folder_is_self_consistent_if_present():
    assert EXPORT.verify_public() == {"status": "valid", "files": 16, "evaluations": 12}
