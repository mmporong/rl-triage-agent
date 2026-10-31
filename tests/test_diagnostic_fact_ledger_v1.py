# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import diagnostic_fact_ledger_v1 as D


def _artifact(path: Path, checkpoint: str, binding: str) -> None:
    path.write_text(json.dumps({"envelope": {"source_binding": {
        "checkpoint_sha256": checkpoint, "binding_digest": binding}}}), encoding="utf-8")


def _facts(role: str, observed: int) -> dict[str, bool | int | None]:
    prefix = f"runtime:{role}:"
    return {
        prefix + "available": True, prefix + "updates_observed": 3,
        prefix + "expected_optimizer_steps": 20,
        prefix + "optimizer_steps_min": observed, prefix + "optimizer_steps_max": observed,
        prefix + "expected_minibatches": 20,
        prefix + "minibatches_min": observed, prefix + "minibatches_max": observed,
    }


def test_build_keeps_raw_counts_and_derives_bools_without_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "case.json"
    checkpoint = "a" * 64
    _artifact(artifact, checkpoint, "b" * 64)
    monkeypatch.setattr(D.FACT, "checked", lambda path, **kwargs: (
        {"source_binding": {"checkpoint_sha256": checkpoint, "binding_digest": "b" * 64}},
        _facts("case", 4),
    ))
    ids = [
        "runtime:case:expected_optimizer_steps",
        "runtime:case:optimizer_steps_min",
        "runtime:case:optimizer_steps_max",
    ]
    value = D.build(artifact, None, proposal_fact_ids=ids)
    facts = value["model_view"]["facts"]
    assert len(facts) == 20
    assert facts["runtime:case:expected_optimizer_steps"] == 20
    assert facts["runtime:case:optimizer_steps_min"] == 4
    assert facts["derived:case:step_count_matches_config"] is False
    assert facts["derived:reference:step_count_matches_config"] is None
    assert facts["runtime:reference:optimizer_steps_min"] is None
    assert value["scope"]["bayesian_probability"] is False
    assert value["scope"]["new_observation"] is False


def test_proposal_requires_expected_and_all_observed_ids() -> None:
    facts = {**_facts("case", 4), **D.FACT.unavailable_facts("reference")}
    facts.update({**D._derived("case", _facts("case", 4)),
                  **D._derived("reference", D.FACT.unavailable_facts("reference"))})
    with pytest.raises(ValueError, match="expected와 observed"):
        D.validate_proposal_fact_ids(facts, [
            "runtime:case:expected_optimizer_steps", "runtime:case:optimizer_steps_min"])
    with pytest.raises(ValueError, match="unavailable"):
        D.validate_proposal_fact_ids(facts, [
            "runtime:reference:expected_optimizer_steps",
            "runtime:reference:optimizer_steps_min",
            "runtime:reference:optimizer_steps_max",
        ])


def test_write_once_checked_and_artifact_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "case.json"
    checkpoint = "c" * 64
    _artifact(artifact, checkpoint, "d" * 64)
    monkeypatch.setattr(D.FACT, "checked", lambda path, **kwargs: (
        {"source_binding": {"checkpoint_sha256": checkpoint, "binding_digest": "d" * 64}},
        _facts("case", 20),
    ))
    out = tmp_path / "ledger.json"
    value = D.build(artifact, None)
    D.write_new(out, value)
    assert D.checked(out) == value
    with pytest.raises(ValueError, match="fresh"):
        D.write_new(out, value)
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        D.checked(out)
