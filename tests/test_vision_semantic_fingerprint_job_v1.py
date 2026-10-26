from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import test_vision_semantic_job_v1 as FIXTURE

from bench import vision_semantic_fingerprint_job_v1 as J

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(FIXTURE, "J", J)
    value: dict[str, Any] = FIXTURE._fixture(tmp_path, monkeypatch)
    repo, parent, protocol = value["repo"], value["parent"], value["protocol"]
    parent["resources"]["whole_batch_budget_usd"] = 11.267092
    parent_path = repo / "bench/protocols/parent.json"
    FIXTURE._write(parent_path, parent)
    FIXTURE._git(repo, "add", "bench/protocols/parent.json")
    FIXTURE._git(repo, "commit", "-qm", "new lifecycle source")
    protocol["source_commit"] = FIXTURE._git(repo, "rev-parse", "HEAD")
    protocol["parent_source_protocol"]["sha256_lf"] = FIXTURE._sha(parent_path, lf=True)
    protocol["resources"].update(
        prior_compute_estimate_usd="8.732908", fresh_lifecycle_budget_usd="11.267092"
    )
    stop_path = repo / "stop.json"
    stop = J._read(stop_path)
    stop["cost_estimate"]["estimated_compute_cost_usd"] = "8.732908"
    FIXTURE._write(stop_path, stop)
    protocol["resources"]["prior_stop_receipts"][0]["sha256"] = FIXTURE._sha(stop_path)
    FIXTURE._write(value["protocol_path"], protocol)
    FIXTURE._commit_protocol(value)
    return value


def test_new_budget_validates_and_keeps_frozen_source_members(tmp_path, monkeypatch):
    value = _fixture(tmp_path, monkeypatch)
    validated = J.validate_protocol(value["protocol_path"], repo=value["repo"])
    assert validated["resources"]["prior_compute_estimate_usd"] == "8.732908"
    assert validated["resources"]["fresh_lifecycle_budget_usd"] == "11.267092"
    assert J.REQUIRED_SOURCE_FILES <= validated["source_sha256_lf"].keys()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "prior_compute_estimate_usd": "8.500674",
            "fresh_lifecycle_budget_usd": "11.499326",
        },
        {"fresh_lifecycle_budget_usd": "11.267093"},
        {"prior_compute_estimate_usd": "8.732907"},
        {"total_program_budget_usd": "20.000001"},
    ],
)
def test_new_budget_rejects_stale_or_oversized_allowance(
    tmp_path, monkeypatch, changes
):
    value = _fixture(tmp_path, monkeypatch)
    value["protocol"]["resources"].update(changes)
    FIXTURE._write(value["protocol_path"], value["protocol"])
    FIXTURE._commit_protocol(value)
    with pytest.raises(ValueError, match="resource budget"):
        J.validate_protocol(value["protocol_path"], repo=value["repo"])


def test_new_budget_requires_receipt_sum_to_include_consumed_interval(
    tmp_path, monkeypatch
):
    value = _fixture(tmp_path, monkeypatch)
    stop_path = value["repo"] / "stop.json"
    stop = J._read(stop_path)
    stop["cost_estimate"]["estimated_compute_cost_usd"] = "8.500674"
    FIXTURE._write(stop_path, stop)
    value["protocol"]["resources"]["prior_stop_receipts"][0]["sha256"] = FIXTURE._sha(
        stop_path
    )
    FIXTURE._write(value["protocol_path"], value["protocol"])
    FIXTURE._commit_protocol(value)
    with pytest.raises(ValueError, match="비용 합계"):
        J.validate_protocol(value["protocol_path"], repo=value["repo"])


def test_versioned_parent_changes_only_registered_validator_import():
    original = (ROOT / "bench/run_vision_semantic_diagnostic_v1.py").read_text(
        encoding="utf-8"
    )
    versioned = (ROOT / "bench/run_vision_semantic_fingerprint_base_v1.py").read_text(
        encoding="utf-8"
    )
    expected = original.replace(
        "import vision_semantic_job_v1 as JOB",
        "import vision_semantic_fingerprint_job_v1 as JOB",
    )
    assert ast.dump(ast.parse(versioned)) == ast.dump(ast.parse(expected))


def test_versioned_validator_preserves_frozen_guards_except_budget_and_source_members():
    original = ast.parse(
        (ROOT / "bench/vision_semantic_job_v1.py").read_text(encoding="utf-8")
    )
    versioned = ast.parse(
        (ROOT / "bench/vision_semantic_fingerprint_job_v1.py").read_text(
            encoding="utf-8"
        )
    )
    for tree in (original, versioned):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_FILES"
                for target in node.targets
            ):
                node.value = ast.Set(elts=[])
            if isinstance(node, ast.Constant) and node.value in {
                "8.500674",
                "8.732908",
            }:
                node.value = "PRIOR"
            if isinstance(node, ast.Constant) and node.value in {
                "11.499326",
                "11.267092",
            }:
                node.value = "FRESH"
    assert ast.dump(original) == ast.dump(versioned)
