# pyright: reportMissingImports=false
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evals import lineage_matched_analysis_v1 as A

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "evals/results/lineage_diagnosis_dev_20261031_v1"
SOURCE = ROOT / "evals/results/lineage_source_fixed_dev_20261031_v1/summary.json"
PROTOCOL = ROOT / "bench/protocols/lineage_comparison_20261031_v1.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_export(tmp_path: Path) -> Path:
    target = tmp_path / "export"
    shutil.copytree(EXPORT, target)
    return target


def _resign_row(export: Path, order: int) -> None:
    manifest_path = export / "manifest.json"
    manifest = _read(manifest_path)
    relative = f"rows/{order:02d}.json"
    entry = next(row for row in manifest["rows_manifest"] if row["path"] == relative)
    entry["sha256"] = A.sha(export / relative)
    entry["size_bytes"] = (export / relative).stat().st_size
    _write(manifest_path, manifest)


def _mutate_row(export: Path, order: int, mutate: Callable[[dict[str, Any]], None]) -> None:
    row_path = export / f"rows/{order:02d}.json"
    row = _read(row_path)
    mutate(row)
    _write(row_path, row)
    _resign_row(export, order)


def test_actual_export_is_recomputed_as_matched_intention_to_diagnose() -> None:
    result = A.analyze(EXPORT, SOURCE, PROTOCOL)

    assert result["inputs"]["rows"] == 30
    assert result["inputs"]["observations"] == 88
    assert result["by_policy"]["model_adaptive"]["correct_component_confirmations"] == 2
    assert result["by_policy"]["fixed_eligible"]["correct_component_confirmations"] == 2
    assert result["by_policy"]["registry_next"]["correct_component_confirmations"] == 0
    pair = result["matched_pairs"]["model_adaptive_vs_fixed_eligible"]
    complete = pair["technical_complete_paired_subset"]
    assert complete["case_ids"] == ["case-01", "case-02", "case-03", "case-04"]
    assert complete["denominator"] == 4
    assert pair["intention_to_diagnose_all_10"]["denominator"] == 10
    assert pair["intention_to_diagnose_all_10"]["exact_two_sided_sign_p"] == 1.0
    assert result["posthoc_f1_probe_comparison"]["model_vs_fixed"][
        "logical_probe_differences"
    ] == [
        {"case_id": "case-03", "left": 2, "right": 3, "left_minus_right": -1},
        {"case_id": "case-04", "left": 2, "right": 3, "left_minus_right": -1},
    ]
    assert result["scope"]["overall_policy_superiority_claim"] is False
    assert result["scope"]["normal_host_gate_counted_as_llm_accuracy"] is False


def test_unknown_subtypes_and_normal_host_gate_remain_distinct() -> None:
    result = A.analyze(EXPORT, SOURCE, PROTOCOL)
    cases = {case["case_id"]: case for case in result["cases"]}

    assert cases["case-01"]["normal_host_no_intervention"] is True
    assert cases["case-01"]["policies"]["model_adaptive"]["primary"]["state"] == (
        "known_zero"
    )
    assert cases["case-05"]["policies"]["model_adaptive"]["technical_details"] == [
        "post_run_validation_unknown"
    ]
    assert cases["case-06"]["policies"]["fixed_eligible"]["technical_details"] == [
        "parent_without_receipt"
    ]
    assert result["unknown_by_arm"]["F3"]["model_adaptive"]["case_ids"] == ["case-10"]
    assert result["unknown_by_arm"]["F5"]["registry_next"]["case_ids"] == ["case-09"]
    assert result["scope"]["out_of_support_unknown_counted_as_success"] is False


def test_model_invalid_is_known_zero_and_not_excluded(tmp_path: Path) -> None:
    export = _copy_export(tmp_path)
    _mutate_row(export, 3, lambda row: row.__setitem__("model_validity", ["invalid_json"]))

    result = A.analyze(export, SOURCE, PROTOCOL)
    row = result["cases"][0]["policies"]["model_adaptive"]
    pair = result["matched_pairs"]["model_adaptive_vs_fixed_eligible"]
    assert row["model_failure"] is True
    assert row["primary"] == {
        "score": 0,
        "state": "known_zero",
        "correct_component_confirmation": False,
        "full_cause_confirmed": False,
        "compound_partial": False,
        "false_confirmation": False,
    }
    assert "case-01" in pair["technical_complete_paired_subset"]["case_ids"]


def test_invalid_selection_cannot_claim_later_confirmation(tmp_path: Path) -> None:
    export = _copy_export(tmp_path)
    _mutate_row(export, 7, lambda row: row.__setitem__("model_validity", ["invalid_json", "valid"]))

    with pytest.raises(ValueError, match="모델 무효 뒤 원인 확정"):
        A.analyze(export, SOURCE, PROTOCOL)


def test_spoofed_compound_partial_without_valid_receipts_is_rejected(tmp_path: Path) -> None:
    export = _copy_export(tmp_path)
    order = 19
    row_path = export / f"rows/{order:02d}.json"
    row = _read(row_path)
    row["confirmed_subtype"] = "reward"
    row["end_state"] = "confirmed"
    row["hold_reason"] = None
    row["classification"] = A.BATCH._classification({"arm": "COMPOUND"}, row)
    for step_index, step in enumerate(row["steps"]):
        step["outcome"] = "abnormal" if step_index == len(row["steps"]) - 1 else "normal"
        for ref in step["observations"].values():
            observation_path = export / ref["path"]
            observation = _read(observation_path)
            observation["status"] = "complete"
            observation["probe"] = step["probe"]
            observation["measurement"] = {"synthetic_test": 1}
            _write(observation_path, observation)
            ref["sha256"] = A.sha(observation_path)
            ref["size_bytes"] = observation_path.stat().st_size
    _write(row_path, row)
    _resign_row(export, order)
    with pytest.raises(ValueError, match="성공 receipt/measurement"):
        A.analyze(export, SOURCE, PROTOCOL)

    classification = A.BATCH._classification(
        {"arm": "COMPOUND"}, {"confirmed_subtype": "reward", "decision_count": 1}
    )
    assert classification["correct_confirmation"] is True
    assert classification["compound_partial"] is True
    assert classification["full_cause_confirmed"] is False


@pytest.mark.parametrize("tamper", ["outcome", "measurement"])
def test_resigned_raw_measurement_or_outcome_tamper_is_rejected(
    tmp_path: Path, tamper: str
) -> None:
    export = _copy_export(tmp_path)
    order = 7
    row_path = export / f"rows/{order:02d}.json"
    row = _read(row_path)
    if tamper == "outcome":
        row["steps"][0]["outcome"] = "abnormal"
    else:
        ref = row["steps"][0]["observations"]["case"]
        observation_path = export / ref["path"]
        observation = _read(observation_path)
        observation["measurement"] = {"synthetic_test": 1}
        _write(observation_path, observation)
        ref["sha256"] = A.sha(observation_path)
        ref["size_bytes"] = observation_path.stat().st_size
    _write(row_path, row)
    _resign_row(export, order)

    with pytest.raises(ValueError):
        A.analyze(export, SOURCE, PROTOCOL)


@pytest.mark.parametrize("field", ["seed", "checkpoint", "observation_role"])
def test_recombined_row_sha_cannot_hide_binding_tamper(tmp_path: Path, field: str) -> None:
    export = _copy_export(tmp_path)
    order = 7
    row_path = export / f"rows/{order:02d}.json"
    row = _read(row_path)
    if field == "seed":
        row["seed"] += 1
    elif field == "checkpoint":
        row["source_binding"]["case"]["checkpoint_sha256"] = "0" * 64
    else:
        ref = row["steps"][0]["observations"]["case"]
        observation_path = export / ref["path"]
        observation = _read(observation_path)
        observation["role"] = "reference"
        _write(observation_path, observation)
        ref["sha256"] = A.sha(observation_path)
        ref["size_bytes"] = observation_path.stat().st_size
    _write(row_path, row)
    _resign_row(export, order)

    with pytest.raises(ValueError):
        A.analyze(export, SOURCE, PROTOCOL)


def test_missing_observation_and_path_traversal_fail_closed(tmp_path: Path) -> None:
    export = _copy_export(tmp_path)
    row = _read(export / "rows/07.json")
    (export / row["steps"][0]["observations"]["case"]["path"]).unlink()
    with pytest.raises(ValueError, match="파일이 없다"):
        A.analyze(export, SOURCE, PROTOCOL)

    export = _copy_export(tmp_path / "second")
    manifest = _read(export / "manifest.json")
    manifest["rows_manifest"][0]["path"] = "../rows/01.json"
    _write(export / "manifest.json", manifest)
    with pytest.raises(ValueError):
        A.analyze(export, SOURCE, PROTOCOL)


def test_exact_sign_test_bounds_and_write_once(tmp_path: Path) -> None:
    assert A.exact_sign_p(0, 0) == 1.0
    assert A.exact_sign_p(2, 0) == 0.5
    assert A.exact_sign_p(3, 1) == 0.625
    result = A.analyze(EXPORT, SOURCE, PROTOCOL)
    bounds = result["matched_pairs"]["model_adaptive_vs_registry_next"][
        "unknown_identification_bounds_all_10"
    ]
    assert bounds["minimum_difference"] == -0.2
    assert bounds["maximum_difference"] == 0.7
    out = tmp_path / "analysis.json"
    A.write_new(out, result)
    assert _read(out)["schema"] == A.SCHEMA
    with pytest.raises(ValueError, match="write-once"):
        A.write_new(out, result)
