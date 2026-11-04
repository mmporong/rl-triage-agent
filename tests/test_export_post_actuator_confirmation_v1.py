from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("post_export", HERE.parent / "evals" / "export_post_actuator_confirmation_v1.py")
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _verified() -> dict:
    return {
        "post": {
            "condition": "NONE",
            "intent": {"sha256": "1" * 64},
            "consumption": {"sha256": "2" * 64},
            "execution": {"sha256": "3" * 64},
            "component_receipt": {
                "case_id": "case-05", "outcome": "normal", "receipt_sha256": "4" * 64,
                "source_refs": {"body": {"sha256": "5" * 64}},
            },
            "supplementary_decision": {
                "status": "reference_review_required", "effective_action": "reference_review",
            },
        },
        "meter": {
            "parent_full_wall_s": 153.2, "owned_wall_s": 153.1, "phase_wall_s": 150.3,
            "parent_process_cpu_s": 1.8,
        },
        "device": {
            "energy": {"devices": {"gpu": {"sample_count": 73, "window_s": 148.1,
                "covered_s": 148.0, "coverage_fraction": 0.9996,
                "gaps": [{"reason": "window_boundary_uncovered"}], "max_gap_s": 6.0,
                "energy_wh": 1.67}}},
            "device_summary": {"gpu": {"metrics": {
                "power_draw_w_peak": {"value": 43.01},
                "memory_used_mib_peak": {"value": 3385.0},
                "gpu_util_percent_sample_mean": {"value": 15.7},
            }}},
        },
        "refs": {"post_result": {"sha256": "6" * 64}},
    }


def test_public_projection_keeps_policy_and_energy_scope_separate() -> None:
    result, metrics, _sources = EXPORT._public(_verified())
    assert result["mapping_outcome"] == "normal"
    assert result["trained_policy_validated"] is False
    assert result["new_recovery_claim"] is False
    assert result["effective_action"] == "reference_review"
    assert metrics["parent"]["energy_wh"] is None
    assert metrics["parent"]["peak_power_w"] is None
    assert metrics["device_sample_window"]["energy_wh"] == 1.67
    assert metrics["device_sample_window"]["process_attribution"] is None
    assert metrics["device_window_is_not_whole_parent_energy"] is True


def test_public_guard_rejects_path() -> None:
    with pytest.raises(EXPORT.ExportError, match="path"):
        EXPORT._assert_public({"value": "C:/private/result.json"})


def test_device_window_requires_one_device() -> None:
    value = _verified()["device"]
    value["energy"]["devices"]["gpu2"] = value["energy"]["devices"]["gpu"]
    with pytest.raises(EXPORT.ExportError, match="denominator"):
        EXPORT._device_window(value)


def test_number_rejects_unknown_or_negative() -> None:
    for value in (None, -1, float("nan"), True):
        with pytest.raises(EXPORT.ExportError):
            EXPORT._number(value, "fixture")


def test_registered_operation_scope_adds_wrapper_closure_and_restores_core() -> None:
    operation = EXPORT.operation
    original = operation.CORE.SOURCE_RELATIONS
    assert len(operation.SOURCE_RELATIONS) == len(original) + 4
    with operation.retry_observation_scope():
        assert operation.CORE.SOURCE_RELATIONS == operation.SOURCE_RELATIONS
    assert operation.CORE.SOURCE_RELATIONS is original
