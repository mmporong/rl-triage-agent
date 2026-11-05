from __future__ import annotations

from copy import deepcopy

import pytest

from evals import summarize_balanced_recovery_continuation_v2 as summary


def _plan() -> dict:
    return {
        "limits": {
            "cpu_checkpoint_launch_cap": 8,
            "cpu_telemetry_launch_cap": 8,
            "gpu_sdk_launch_cap": 16,
            "total_owned_launch_cap": 32,
        },
        "continuation": {
            "prior_whole_study_sdk_launches": 49,
            "cumulative_gpu_sdk_launch_cap": 65,
        },
    }


def _totals() -> dict:
    return {
        "historical_sdk_launches": 15,
        "new_sdk_launches": 15,
        "historical_completed_effects": 7,
        "historical_completed_training": 1,
        "current_dispatches": 8,
        "cpu_checkpoint_launches": 7,
        "cpu_telemetry_launches": 7,
        "total_owned_children": 29,
    }


def test_actual_seven_epoch_cpu_observations_are_valid_under_caps() -> None:
    assert summary._validate_accounting(_plan(), _totals()) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_checkpoint_launches", 0),
        ("cpu_telemetry_launches", 0),
        ("cpu_checkpoint_launches", 8),
        ("cpu_telemetry_launches", 8),
    ],
)
def test_cpu_launch_counts_from_zero_through_cap_are_valid(field: str, value: int) -> None:
    totals = _totals()
    totals[field] = value
    assert summary._validate_accounting(_plan(), totals) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_checkpoint_launches", 9),
        ("cpu_telemetry_launches", 9),
        ("cpu_checkpoint_launches", -1),
        ("cpu_telemetry_launches", -1),
    ],
)
def test_cpu_launch_counts_outside_caps_are_rejected(field: str, value: int) -> None:
    totals = _totals()
    totals[field] = value
    with pytest.raises(summary.E.ExportError, match="accounting differs"):
        summary._validate_accounting(_plan(), totals)


def test_other_sdk_and_owned_limits_remain_unchanged() -> None:
    for field, value in (
        ("new_sdk_launches", 16),
        ("historical_sdk_launches", 14),
        ("current_dispatches", 7),
        ("total_owned_children", 33),
    ):
        totals = deepcopy(_totals())
        totals[field] = value
        with pytest.raises(summary.E.ExportError, match="accounting differs"):
            summary._validate_accounting(_plan(), totals)
