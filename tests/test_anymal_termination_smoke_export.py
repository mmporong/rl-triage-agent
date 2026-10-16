from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from evals import export_anymal_termination_smoke_v1 as EXPORT


@pytest.fixture(scope="module")
def payloads():
    if not EXPORT.OUTPUT.is_dir():
        pytest.fail("public Task28 export is missing")
    return {name: EXPORT.read(EXPORT.OUTPUT / name) for name in EXPORT.PUBLIC_FILES}


def _rehash(payloads, name):
    payloads["provenance.json"]["public_artifact_sha256"][name] = hashlib.sha256(
        EXPORT._encoded(payloads[name])).hexdigest()


def test_private_integration_replays_registered_smoke():
    isaaclab = Path.home() / "IsaacLab"
    if not EXPORT.RUN.is_dir() or not isaaclab.is_dir():
        pytest.skip("private Task27 integration input unavailable")
    payloads = EXPORT.verify_and_build(isaaclab)
    assert payloads["summary.json"]["outcome"] == "source_failure_absent"
    assert payloads["summary.json"]["may_preregister_demo"] is False
    assert payloads["runtime_signature.json"]["episode_steps_median"] == {
        "NONE": pytest.approx(978.9799805), "F6": pytest.approx(50.0)}
    assert payloads["costs.json"]["totals_wall_s"] == {
        "training": pytest.approx(943.688000000082),
        "telemetry_extract": pytest.approx(2.1570000001229346),
        "training_and_fixed_evaluation": pytest.approx(1027.094000000041),
        "accounted_with_extract": pytest.approx(1029.251000000164),
    }


@pytest.mark.parametrize("mutation", [
    "private_path", "missing_file", "metric", "condition", "condition_metric", "gate", "runtime", "cost",
    "public_hash", "source_inventory", "source_crosslink",
])
def test_public_verifier_rejects_tampering(payloads, mutation):
    changed = deepcopy(payloads)
    if mutation == "private_path":
        changed["summary.json"]["interpretation"] = r"bench\private\hidden.json"
    elif mutation == "missing_file":
        changed.pop("costs.json")
    elif mutation == "metric":
        changed["evaluations/f6.json"]["metrics"]["lin_vel_rmse_mps"] = 0.01
        _rehash(changed, "evaluations/f6.json")
    elif mutation == "condition":
        changed["evaluations/f6.json"]["conditions"].pop()
        _rehash(changed, "evaluations/f6.json")
    elif mutation == "condition_metric":
        changed["evaluations/f6.json"]["conditions"][0]["fall_rate"] = 0.5
        _rehash(changed, "evaluations/f6.json")
    elif mutation == "gate":
        changed["summary.json"]["outcome"] = "failure_present"
        _rehash(changed, "summary.json")
    elif mutation == "runtime":
        changed["runtime_signature.json"]["f6_episode_steps_max"] = 51.0
        _rehash(changed, "runtime_signature.json")
    elif mutation == "cost":
        changed["costs.json"]["totals_wall_s"]["accounted_with_extract"] += 1
        _rehash(changed, "costs.json")
    elif mutation == "public_hash":
        changed["provenance.json"]["public_artifact_sha256"]["summary.json"] = "0" * 64
    elif mutation == "source_inventory":
        changed["provenance.json"]["source_artifacts"]["unexpected"] = "f" * 64
    else:
        changed["provenance.json"]["source_artifacts"]["fixed_evaluation.f6"] = "f" * 64
    with pytest.raises(ValueError):
        EXPORT.verify_public_payloads(changed)


def test_public_only_verification_needs_no_private_inputs(payloads, tmp_path):
    output = tmp_path / "public"
    for relative, body in payloads.items():
        EXPORT.write_new(output / relative, body)
    assert EXPORT.verify_public(output) == {
        "status": "valid", "files": 6, "evaluations": 2,
        "outcome": "source_failure_absent", "may_preregister_demo": False,
    }


@pytest.mark.parametrize("name,keys,replacement", [
    ("evaluations/f6.json", ("checkpoint", "sha256"), "0" * 64),
    ("evaluations/f6.json", ("protocol", "task"), "Other-Task"),
    ("evaluations/f6.json", ("protocol", "eval_seed"), 9999),
    ("evaluations/f6.json", ("standstill", "lin_vel_rmse_mps"), 999),
    ("runtime_signature.json", ("configured_episode_length_s",), 999),
    ("summary.json", ("interpretation",), "unsupported success claim"),
    ("costs.json", ("measurement",), "GPU kernel time"),
    ("provenance.json", ("source_verification",), "unverified"),
])
def test_coordinated_content_and_manifest_mutation_is_refused(payloads, name, keys, replacement):
    changed = deepcopy(payloads)
    target = changed[name]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = replacement
    if name != "provenance.json":
        _rehash(changed, name)
    with pytest.raises(ValueError, match="registered content"):
        EXPORT.verify_public_payloads(changed)


def test_export_refuses_existing_output(payloads, tmp_path, monkeypatch):
    output, audit = tmp_path / "public", tmp_path / "audit"
    output.mkdir()
    monkeypatch.setattr(EXPORT, "OUTPUT", output)
    monkeypatch.setattr(EXPORT, "AUDIT", audit)
    monkeypatch.setattr(EXPORT, "verify_and_build", lambda isaaclab: payloads)
    with pytest.raises(FileExistsError):
        EXPORT.export(tmp_path / "IsaacLab")


def test_export_rejects_source_change_during_copy(payloads, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("before", encoding="utf-8")
    monkeypatch.setattr(EXPORT, "RUN", source)
    monkeypatch.setattr(EXPORT, "OUTPUT", tmp_path / "public")
    monkeypatch.setattr(EXPORT, "AUDIT", tmp_path / "audit")

    def mutate_source(_):
        (source / "evidence.txt").write_text("after", encoding="utf-8")
        return payloads

    monkeypatch.setattr(EXPORT, "verify_and_build", mutate_source)
    with pytest.raises(ValueError, match="source result changed"):
        EXPORT.export(tmp_path / "IsaacLab")


def test_committed_public_folder_is_self_consistent_if_present():
    assert EXPORT.verify_public() == {
        "status": "valid", "files": 6, "evaluations": 2,
        "outcome": "source_failure_absent", "may_preregister_demo": False,
    }
