from __future__ import annotations

import copy
import json
import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import early_recovery_decider as E  # noqa: E402


def protocol() -> dict:
    return json.loads((ROOT / "bench/protocols/early_recovery_decider_v1.json").read_text(encoding="utf-8"))


def test_feature_extraction_cannot_read_full_run_summary_or_suffix():
    source = E.read_json(ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json")
    altered = copy.deepcopy(source)
    altered["summary"] = {"forbidden": {"last": 10**30}}
    for values in altered["series"].values():
        if isinstance(values, list) and len(values) > 50:
            values[50:] = [10**30] * (len(values) - 50)
    assert E.extract_features(source, protocol(), 50) == E.extract_features(altered, protocol(), 50)


def test_public_catalog_joins_all_files_then_deduplicates_checkpoint_lineages():
    catalog = E.build_catalog(protocol())
    assert catalog["inventory_count"] == 72
    assert catalog["joined_before_dedup"] == 72
    assert catalog["included_after_dedup"] < 72
    assert catalog["excluded_missing_label"] == []
    checkpoints = [row["checkpoint_sha256"] for row in catalog["runs"]]
    assert len(checkpoints) == len(set(checkpoints))
    assert {row["behavior_label"] for row in catalog["runs"]} == {"healthy", "unhealthy"}
    assert all(row["recovery_label"] is None for row in catalog["runs"])


def test_seed_lineage_never_crosses_splits_and_anymal_is_separate():
    catalog = E.build_catalog(protocol())
    groups = {}
    for row in catalog["runs"]:
        groups.setdefault((row["task"], row["seed"]), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert {row["split"] for row in catalog["runs"] if "Anymal" in row["task"]} == {"transfer_audit"}


def test_offline_evaluation_keeps_behavior_and_recovery_claims_separate():
    catalog = E.build_catalog(protocol())
    report, predictions = E.evaluate(protocol(), catalog)
    assert report["measurement_class"] == "development_measurement_after_results_were_seen"
    assert report["reliable_recovery_decider"] is False
    assert report["recovery_label_counts"] == {}
    assert "GPU" in report["counterfactual_scope"]
    assert len(predictions) == catalog["included_after_dedup"]
    assert all(row["n"] in protocol()["features"]["candidate_prefix_iterations"] for row in predictions)


def test_critical_false_healthy_is_counted_explicitly():
    rows = [
        {"behavior_label": "unhealthy", "prediction": "healthy", "n": 50},
        {"behavior_label": "healthy", "prediction": "abstain", "n": 50},
    ]
    result = E.metrics(rows)
    assert result["false_healthy"] == 1
    assert result["confusion"]["unhealthy_as_healthy"] == 1
    assert result["counterfactual_early_behavior_iterations"] == 250


def test_missing_input_is_abstention_with_zero_counterfactual_iterations():
    result = E.metrics([{"behavior_label": "healthy", "prediction": "abstain_missing_input", "n": 50}])
    assert result["abstention_rate"] == 1
    assert result["confusion"]["healthy_as_abstain"] == 1
    assert result["counterfactual_early_behavior_iterations"] == 0


def test_protocol_provenance_is_actual_input(tmp_path):
    custom = protocol()
    custom["sentinel"] = "custom"
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    report, _ = E.evaluate(custom, E.build_catalog(custom), path)
    assert report["protocol_sha256_lf"] == E.sha256(path)
    with pytest.raises(ValueError):
        E.evaluate(custom, E.build_catalog(custom))


def test_wrong_contract_or_unhealthy_reference_is_rejected():
    source = E.read_json(ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json")
    report = E.read_json(ROOT / "evals/results/p0c_fixed_eval_20261009/runs/baseline_p0c_s2026.json")
    bad = copy.deepcopy(report)
    bad["metrics"]["fall_rate"] = 0.9
    with pytest.raises(ValueError):
        E.validate_label_binding("baseline_p0c_s2026", source, bad, bad)
    bad = copy.deepcopy(report)
    bad["action_scale"] = 0.5
    with pytest.raises(ValueError):
        E.validate_label_binding("baseline_p0c_s2026", source, bad, report)
