from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evals.summarize_mechanism_resource_comparison_v1 as S


def comparison_rows() -> list[dict]:
    rows = []
    for case in S.CASES:
        for method in S.METHODS:
            outcome = "hold" if (case, method) == ("case-07", "iterative_agent") else "correct_confirmation"
            rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "diagnostic_outcome": outcome,
                    "status": "model_hold" if outcome == "hold" else "confirmed",
                }
            )
    return rows


def cost_rows() -> list[dict]:
    rows = []
    for case_index, case in enumerate(S.CASES, 1):
        for method_index, method in enumerate(S.METHODS, 1):
            outcome = "hold" if (case, method) == ("case-07", "iterative_agent") else "correct_confirmation"
            rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "diagnostic_outcome": outcome,
                    "SDK_launches": 1,
                    "owned_wall_s": float(case_index + method_index),
                    "end_to_end_s": float(10 + case_index + method_index),
                    "model_calls": int("llm" in method or method == "iterative_agent"),
                    "input_tokens": 100 if "llm" in method or method == "iterative_agent" else 0,
                    "output_tokens": 10 if "llm" in method or method == "iterative_agent" else 0,
                    "device_energy_wh": 0.5,
                }
            )
    return rows


def step_rows() -> list[dict]:
    return [
        {
            "case_id": case,
            "method": method,
            "process_cpu_s": 0.25,
            "wall_s": 1.0,
            "source_journal_sha256": f"{case}:{method}",
            "source_line": 1,
            "step": "prepare",
            "status": "returned",
        }
        for case in S.CASES
        for method in S.METHODS
    ]


def source_summary() -> dict:
    return {
        "schema": "registered_source_costs_v1",
        "status": "complete",
        "allocation": {"charged_once": True, "per_method": False},
        "counts": {"training_jobs": 20, "fixed_evaluations": 20, "telemetry_extractions": 20},
        "wall_clock": {"simulator_owned_wall_s": 1000.0, "source_parent_span_s": 1100.0},
        "gpu_measurement": {
            "process_gpu_seconds": None,
            "provider_gpu_seconds": None,
            "scope": "device_wide_including_other_loads_not_attributable_to_source_jobs",
            "devices": [
                {"energy_wh": 20.0, "scope": "whole_device_including_other_loads"}
            ],
        },
        "provider_cost": {"actual_billing_usd": None},
    }


def recovery_summary(*, complete: bool = True) -> dict:
    arms = []
    for case in S.RECOVERY_CASES:
        for method_index, method in enumerate(S.METHODS, 1):
            outcome = "recovered" if complete else "not_started"
            arms.append(
                {
                    "case_id": case,
                    "method": method,
                    "diagnostic_status": "confirmed",
                    "recovery_outcome": outcome,
                    "recovery_attempted": complete,
                    "endpoint": {"checkpoint_sha256": "a" * 64, "fixed_report_sha256": "b" * 64}
                    if complete
                    else None,
                    "costs": {
                        "api_calls": 0 if complete else None,
                        "checkpoint_cpu_owned_wall_s": None,
                        "current_dispatch_sdk_launches": 1 if complete else None,
                        "current_dispatch_simulator_owned_wall_s": 10.0 if complete else None,
                        "lease_wait_s": None,
                        "locked_wall_s": 12.0 if complete else None,
                        "owned_cost_complete": complete,
                        "phase_elapsed_s": 12.0 if complete else None,
                        "reused_training_owned_wall_s": 20.0 if complete else None,
                        "reused_training_sdk_launches": 1 if complete else None,
                        "simulator_owned_wall_s": 30.0 if complete else None,
                        "status": "complete" if complete else "not_applicable",
                        "telemetry_cpu_owned_wall_s": None,
                        "whole_effect_active_phase_s": 35.0 if complete else None,
                        "device_energy": None,
                    },
                }
            )
    return {
        "schema": "registered_recovery_comparison_public_summary_v1",
        "planned_cases": list(S.RECOVERY_CASES),
        "planned_methods": list(S.METHODS),
        "planned_arms": 16,
        "completed_arms": 16 if complete else 0,
        "controller_halted": not complete,
        "arms": arms,
    }


def attempt_timing() -> dict:
    return {
        "schema": "recovery_attempt_timing_aggregate_v1",
        "status": "complete",
        "attempt_count": 3,
        "all_attempt_wall_span_is_algorithm_cost": False,
        "all_attempt_wall_span_is_gpu_owned_time": False,
        "all_attempt_wall_span_s": 500.0,
        "per_arm": [
            {
                "case_id": case,
                "method": method,
                "transition_count": 1 if case == "case-05" else 0,
                "unknown_transition_count": 0,
                "known_active_command_wall_s": 2.0 if case == "case-05" else 0.0,
                "active_command_wall_s": 2.0 if case == "case-05" else 0.0,
                "observation_status": "observed" if case == "case-05" else "not_observed",
            }
            for case in S.RECOVERY_CASES
            for method in S.METHODS
        ],
    }


def summarize(recovery: dict | None = None) -> dict:
    return S.summarize(
        comparison_rows=comparison_rows(),
        cost_rows=cost_rows(),
        step_rows=step_rows(),
        source_summary=source_summary(),
        recovery_summary=recovery or recovery_summary(),
        bindings={"fixture": "f" * 64},
    )


def test_separates_six_case_diagnosis_four_case_recovery_and_common_source() -> None:
    result = summarize()
    assert result["status"] == "complete"
    assert result["common_source_cost"]["training_jobs"] == 20
    assert result["common_source_cost"]["fixed_evaluations"] == 20
    assert result["common_source_cost"]["allocation"] == "charged_once_not_repeated_per_method"
    assert result["common_source_cost"]["gpu_device_time_s"] is None
    assert result["common_source_cost"]["provider_gpu_seconds"] is None
    assert result["diagnostic_primary"]["arm_count"] == 24
    assert result["recovery_primary"]["arm_count"] == 16
    fixed = result["diagnostic_primary"]["methods"]["fixed_sequence"]
    assert fixed["resources"]["sdk_launches"] == 6.0
    assert fixed["resources"]["recorded_controller_process_cpu_s"] == 1.5
    assert fixed["normal_success"] is None
    recovery_fixed = result["recovery_primary"]["methods"]["fixed_sequence"]
    assert recovery_fixed["costs"]["simulator_owned_wall_s"]["value"] == 120.0
    assert recovery_fixed["costs"]["gpu_device_time_s"] is None
    assert recovery_fixed["costs"]["actual_billing_usd"] is None
    expected_diagnostic_owned = sum(
        row["owned_wall_s"]
        for row in cost_rows()
        if row["method"] == "fixed_sequence" and row["case_id"] in S.RECOVERY_CASES
    )
    assert (
        recovery_fixed["four_case_cost_components"][
            "diagnostic_plus_scientific_effect_simulator_owned_wall_s"
        ]
        == expected_diagnostic_owned + 120.0
    )
    assert recovery_fixed["recovery_attempted_count"] == 4
    assert (
        recovery_fixed["four_case_cost_components"][
            "diagnostic_plus_recorded_attempt_active_command_wall_s"
        ]
        is None
    )


def test_incomplete_recovery_keeps_missing_numbers_null_and_no_complete_claim() -> None:
    result = summarize(recovery_summary(complete=False))
    assert result["status"] == "incomplete"
    assert result["recovery_primary"]["all_recovery_arms_complete"] is False
    fixed = result["recovery_primary"]["methods"]["fixed_sequence"]
    assert fixed["outcomes"] == {"not_started": 4}
    assert fixed["costs"]["simulator_owned_wall_s"] == {
        "known_count": 0,
        "unknown_count": 4,
        "known_sum": None,
        "value": None,
    }
    assert (
        fixed["four_case_cost_components"][
            "diagnostic_plus_scientific_effect_simulator_owned_wall_s"
        ]
        is None
    )


def test_optional_attempt_timing_adds_recorded_active_commands_not_user_wall_span() -> None:
    result = S.summarize(
        comparison_rows=comparison_rows(),
        cost_rows=cost_rows(),
        step_rows=step_rows(),
        source_summary=source_summary(),
        recovery_summary=recovery_summary(),
        recovery_attempt_timing=attempt_timing(),
        bindings={"fixture": "f" * 64},
    )
    fixed = result["recovery_primary"]["methods"]["fixed_sequence"]
    costs = fixed["four_case_cost_components"]
    assert costs["recorded_recovery_attempt_active_command_wall_s"]["value"] == 2.0
    assert costs["diagnostic_plus_recorded_attempt_active_command_wall_s"] == pytest.approx(
        costs["diagnostic_end_to_end_arm_sum_s"] + 2.0
    )
    assert costs["is_full_user_elapsed_time"] is False
    assert result["recovery_primary"]["attempt_timing"]["all_attempt_wall_span_s"] == 500.0


def test_model_hold_is_not_normal_or_success_even_when_cost_is_low() -> None:
    recovery = recovery_summary()
    row = next(
        row
        for row in recovery["arms"]
        if row["case_id"] == "case-07" and row["method"] == "iterative_agent"
    )
    row["diagnostic_status"] = "model_hold"
    row["recovery_outcome"] = "not_started"
    row["recovery_attempted"] = False
    row["endpoint"] = None
    for key in S.RECOVERY_COST_FIELDS:
        row["costs"][key] = 0
    result = summarize(recovery)
    iterative = result["recovery_primary"]["methods"]["iterative_agent"]
    assert iterative["outcomes"]["hold"] == 1
    assert iterative["decision_terminal_arm_count"] == 4
    assert iterative["recovery_attempted_count"] == 3
    assert iterative["normal_success"] is None
    assert result["interpretation"]["hold_low_cost_is_not_success"] is True


def test_paired_rows_require_same_terminal_outcome_and_endpoint_shape() -> None:
    recovery = recovery_summary()
    result = summarize(recovery)
    assert len(result["recovery_primary"]["paired_cost_rows"]) == 24

    changed = deepcopy(recovery)
    row = next(
        row
        for row in changed["arms"]
        if row["case_id"] == "case-05" and row["method"] == "static_rule"
    )
    row["endpoint"] = {"checkpoint_sha256": "c" * 64}
    result = summarize(changed)
    pairs = result["recovery_primary"]["paired_cost_rows"]
    assert not any(
        pair["case_id"] == "case-05" and "static_rule" in {pair["left_method"], pair["right_method"]}
        for pair in pairs
    )


def test_duplicate_case_method_and_step_source_are_rejected() -> None:
    comparisons = comparison_rows()
    with pytest.raises(S.ComparisonError, match="중복 case/method"):
        S.summarize(
            comparison_rows=[*comparisons, deepcopy(comparisons[0])],
            cost_rows=cost_rows(),
            step_rows=step_rows(),
            source_summary=source_summary(),
            recovery_summary=recovery_summary(),
            bindings={},
        )
    steps = step_rows()
    with pytest.raises(S.ComparisonError, match="source가 중복"):
        S.summarize(
            comparison_rows=comparisons,
            cost_rows=cost_rows(),
            step_rows=[*steps, deepcopy(steps[0])],
            source_summary=source_summary(),
            recovery_summary=recovery_summary(),
            bindings={},
        )


def test_missing_or_out_of_scope_case_is_rejected() -> None:
    comparisons = comparison_rows()[1:]
    with pytest.raises(S.ComparisonError, match="누락"):
        S.summarize(
            comparison_rows=comparisons,
            cost_rows=cost_rows(),
            step_rows=step_rows(),
            source_summary=source_summary(),
            recovery_summary=recovery_summary(),
            bindings={},
        )
    recovery = recovery_summary()
    recovery["planned_cases"] = [*S.RECOVERY_CASES, "case-09"]
    with pytest.raises(S.ComparisonError, match="계획 계약"):
        summarize(recovery)


def test_common_source_device_energy_rejects_malformed_or_non_device_wide_rows() -> None:
    malformed = source_summary()
    malformed["gpu_measurement"]["devices"] = ["not-an-object"]
    with pytest.raises(S.ComparisonError, match="object가 아니다"):
        S._validate_source(malformed)

    wrong_scope = source_summary()
    wrong_scope["gpu_measurement"]["devices"][0]["scope"] = "process_attributed"
    with pytest.raises(S.ComparisonError, match="device-wide"):
        S._validate_source(wrong_scope)


def test_cli_is_write_once_and_records_only_logical_input_bindings(tmp_path: Path) -> None:
    files = {
        "comparison": comparison_rows(),
        "cost": cost_rows(),
        "steps": step_rows(),
    }
    paths: dict[str, Path] = {}
    for name, rows in files.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths[name] = path
    source = tmp_path / "source.json"
    source.write_text(json.dumps(source_summary()), encoding="utf-8")
    recovery = tmp_path / "recovery.json"
    recovery.write_text(json.dumps(recovery_summary()), encoding="utf-8")
    output = tmp_path / "output.json"
    argv = [
        "--comparison-rows",
        str(paths["comparison"]),
        "--cost-rows",
        str(paths["cost"]),
        "--step-rows",
        str(paths["steps"]),
        "--source-summary",
        str(source),
        "--recovery-summary",
        str(recovery),
        "--out",
        str(output),
    ]
    assert S.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(result["input_bindings"]) == [
        "common_source_summary",
        "diagnostic_comparison_case_rows",
        "diagnostic_controller_step_rows",
        "diagnostic_cost_arm_rows",
        "recovery_summary",
    ]
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        S.main(argv)
