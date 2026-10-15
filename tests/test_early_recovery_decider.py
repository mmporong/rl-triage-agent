from __future__ import annotations

import copy
import hashlib
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
    report, predictions = E.evaluate(protocol(), catalog, ROOT / "bench/protocols/early_recovery_decider_v1.json")
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


def _bound_input_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ROOT", tmp_path)
    name = "baseline_p0c_s2026"
    telemetry = tmp_path / f"bench/runs_p0c/{name}.telemetry.json"
    report = tmp_path / f"evals/results/original/runs/{name}.json"
    for path, data in ((telemetry, {"run_dir_name": "original_" + name}), (report, {"name": name})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    registered = protocol()
    registered["schema"] = "early_recovery_decider_protocol_v2"
    registered["inventory"].update({
        "expected_file_count": 1,
        "telemetry_files": [{"path": telemetry.relative_to(tmp_path).as_posix(),
                             "sha256_lf": E.sha256(telemetry), "run": name}],
        "fixed_reports": [{"path": report.relative_to(tmp_path).as_posix(),
                           "sha256_lf": E.sha256(report), "run": name}],
    })
    return registered, telemetry, report


def test_unregistered_telemetry_and_non_evaluation_report_do_not_change_inputs(tmp_path, monkeypatch):
    registered, telemetry, report = _bound_input_fixture(tmp_path, monkeypatch)
    (telemetry.parent / "u1_s2051.telemetry.json").write_text("not JSON", encoding="utf-8")
    later = tmp_path / "evals/results/000_optimizer/runs" / report.name
    later.parent.mkdir(parents=True)
    later.write_text(json.dumps({"name": "baseline_p0c_s2026", "schema": "optimizer_measurement"}), encoding="utf-8")
    assert E.inventory(registered) == [telemetry]
    assert E.report_index(registered) == {"baseline_p0c_s2026": [report]}


@pytest.mark.parametrize("kind", ["telemetry", "report"])
@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_registered_input_missing_or_hash_changed_is_rejected(tmp_path, monkeypatch, kind, damage):
    registered, telemetry, report = _bound_input_fixture(tmp_path, monkeypatch)
    target = telemetry if kind == "telemetry" else report
    if damage == "missing":
        target.unlink()
    else:
        target.write_text('{"changed": true}', encoding="utf-8")
    load = E.inventory if kind == "telemetry" else E.report_index
    with pytest.raises(ValueError):
        load(registered)


@pytest.mark.parametrize("key,loader", [("telemetry_files", "inventory"), ("fixed_reports", "report_index")])
def test_duplicate_registered_paths_are_rejected(tmp_path, monkeypatch, key, loader):
    registered, _, _ = _bound_input_fixture(tmp_path, monkeypatch)
    registered["inventory"][key] *= 2
    registered["inventory"]["expected_file_count"] = 2
    with pytest.raises(ValueError):
        getattr(E, loader)(registered)


@pytest.mark.parametrize("unsafe", ["../outside.json", "/outside.json", "C:/outside.json", "bench\\runs_p0c\\outside.json"])
def test_registered_paths_must_be_repository_relative(tmp_path, monkeypatch, unsafe):
    registered, _, _ = _bound_input_fixture(tmp_path, monkeypatch)
    registered["inventory"]["telemetry_files"][0]["path"] = unsafe
    with pytest.raises(ValueError):
        E.inventory(registered)


def test_bound_report_run_name_mismatch_is_rejected_even_with_matching_hash(tmp_path, monkeypatch):
    registered, _, report = _bound_input_fixture(tmp_path, monkeypatch)
    report.write_text('{"name": "other_s2026"}', encoding="utf-8")
    registered["inventory"]["fixed_reports"][0]["sha256_lf"] = E.sha256(report)
    with pytest.raises(ValueError):
        E.report_index(registered)


def test_bound_report_malformed_json_is_not_silently_excluded(tmp_path, monkeypatch):
    registered, _, report = _bound_input_fixture(tmp_path, monkeypatch)
    report.write_text("not JSON", encoding="utf-8")
    registered["inventory"]["fixed_reports"][0]["sha256_lf"] = E.sha256(report)
    with pytest.raises(ValueError):
        E.report_index(registered)


def test_frozen_v2_preserves_v1_policy_and_catalog():
    original = protocol()
    frozen = E.read_json(ROOT / "bench/protocols/early_recovery_decider_v2.json")
    assert E.sha256(ROOT / "bench/protocols/early_recovery_decider_v1.json") == "53e6546bebfa35f83186279f4644ede629de45672ef314a62bd33fa192330a63"
    for key, value in original.items():
        if key not in {"schema", "status", "inventory"}:
            assert frozen[key] == value
    for key, value in original["inventory"].items():
        assert frozen["inventory"][key] == value
    assert len(frozen["inventory"]["telemetry_files"]) == 72
    assert len(frozen["inventory"]["fixed_reports"]) == 72
    catalog = E.build_catalog(frozen)
    assert catalog == E.build_catalog(original)
    assert (catalog["joined_before_dedup"], catalog["included_after_dedup"], len(catalog["duplicates"])) == (72, 64, 8)
    # a46be95 당시 검증된 catalog의 전체 lineage·라벨·split·입력 SHA를 고정한다.
    canonical = json.dumps(catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == "ccc9f93202f28023c4e7cf7fb75be939f8e31193ba897d8bf664f85430bae0d1"


def test_modified_band_summary_is_rejected(tmp_path, monkeypatch):
    registered, _, _ = _bound_input_fixture(tmp_path, monkeypatch)
    registered["inventory"]["band_summaries"] = {}
    for key in ("go2", "anymal"):
        band = tmp_path / f"evals/results/{key}_bands.json"
        band.write_text('{"bands": {}}', encoding="utf-8")
        registered["inventory"]["band_summaries"][key] = {
            "path": band.relative_to(tmp_path).as_posix(), "sha256_lf": E.sha256(band)}
    band.write_text('{"bands": {"changed": true}}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        E.build_catalog(registered)
