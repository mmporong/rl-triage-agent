"""진단 6-case와 회복 4-case의 실제 자원·결과를 분리 집계한다."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CASES = tuple(f"case-{number:02d}" for number in range(5, 11))
RECOVERY_CASES = CASES[:4]
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
TERMINAL_RECOVERY = {
    "recovered",
    "not_recovered",
    "hold",
    "unsupported",
    "normal_or_host_gate",
}
RECOVERY_OUTCOMES = TERMINAL_RECOVERY | {"unknown", "infrastructure_failure", "not_started"}
RECOVERY_COST_FIELDS = (
    "api_calls",
    "checkpoint_cpu_owned_wall_s",
    "current_dispatch_sdk_launches",
    "current_dispatch_simulator_owned_wall_s",
    "lease_wait_s",
    "locked_wall_s",
    "phase_elapsed_s",
    "reused_training_owned_wall_s",
    "reused_training_sdk_launches",
    "simulator_owned_wall_s",
    "telemetry_cpu_owned_wall_s",
    "whole_effect_active_phase_s",
)


class ComparisonError(ValueError):
    """입력 계보 또는 집계 계약이 잘못됐다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError("JSON 입력을 읽을 수 없다") from error
    if not isinstance(value, dict):
        raise ComparisonError("JSON 입력 최상위가 object가 아니다")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ComparisonError(f"JSONL {line_number}행이 object가 아니다")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError("JSONL 입력을 읽을 수 없다") from error
    return rows


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label}이 기록된 숫자가 아니다")
    return float(value)


def _grid(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[str], methods: Sequence[str], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected = set(itertools.product(cases, methods))
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        case_id, method = row.get("case_id"), row.get("method")
        if not isinstance(case_id, str) or not isinstance(method, str):
            raise ComparisonError(f"{label} case/method 형식이 잘못됐다")
        key = (case_id, method)
        if key in indexed:
            raise ComparisonError(f"{label}에 중복 case/method가 있다: {key}")
        if key not in expected:
            raise ComparisonError(f"{label}에 범위 밖 case/method가 있다: {key}")
        indexed[key] = row
    missing = sorted(expected - indexed.keys())
    if missing:
        raise ComparisonError(f"{label}에 case/method가 누락됐다: {missing}")
    return indexed


def _known(values: Iterable[Any]) -> dict[str, Any]:
    known: list[float] = []
    unknown = 0
    for value in values:
        if value is None:
            unknown += 1
        else:
            known.append(_number(value, "aggregate value"))
    known_sum = sum(known) if known else None
    return {
        "known_count": len(known),
        "unknown_count": unknown,
        "known_sum": known_sum,
        "value": known_sum if unknown == 0 else None,
    }


def _device_energy(costs: Mapping[str, Any]) -> float | None:
    value = costs.get("device_energy")
    if value is None:
        return None
    try:
        devices = value["energy"]["devices"]
    except (KeyError, TypeError) as error:
        raise ComparisonError("recovery device energy 구조가 잘못됐다") from error
    if not isinstance(devices, dict) or not devices:
        raise ComparisonError("recovery device energy device 집합이 비었다")
    total = 0.0
    for device in devices.values():
        if not isinstance(device, Mapping):
            raise ComparisonError("recovery device energy 행이 object가 아니다")
        total += _number(device.get("energy_wh"), "recovery device energy Wh")
        if device.get("scope") != "device_wide_estimate":
            raise ComparisonError("recovery energy scope가 device-wide가 아니다")
    return total


def _recovery_class(row: Mapping[str, Any]) -> str:
    if row.get("diagnostic_status") == "model_hold":
        return "hold"
    outcome = row.get("recovery_outcome")
    if outcome not in RECOVERY_OUTCOMES:
        raise ComparisonError(f"등록되지 않은 recovery outcome이다: {outcome}")
    return str(outcome)


def _validate_attempt_timing(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    if (
        summary.get("schema") != "recovery_attempt_timing_aggregate_v1"
        or summary.get("status") not in {"complete", "incomplete"}
        or summary.get("attempt_count") in {None, 0}
        or summary.get("all_attempt_wall_span_is_algorithm_cost") is not False
        or summary.get("all_attempt_wall_span_is_gpu_owned_time") is not False
        or not isinstance(summary.get("per_arm"), list)
    ):
        raise ComparisonError("recovery attempt timing 계약이 다르다")
    rows = _grid(summary["per_arm"], RECOVERY_CASES, METHODS, "recovery attempt timing")
    for row in rows.values():
        if row.get("observation_status") not in {"observed", "not_observed", "incomplete"}:
            raise ComparisonError("recovery attempt timing observation status가 다르다")
        if row.get("active_command_wall_s") is not None:
            _number(row.get("active_command_wall_s"), "attempt active command wall")
    return dict(summary)


def _validate_source(summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = summary.get("counts")
    allocation = summary.get("allocation")
    wall = summary.get("wall_clock")
    gpu = summary.get("gpu_measurement")
    if (
        summary.get("schema") != "registered_source_costs_v1"
        or summary.get("status") != "complete"
        or not isinstance(counts, Mapping)
        or counts.get("training_jobs") != 20
        or counts.get("fixed_evaluations") != 20
        or not isinstance(allocation, Mapping)
        or allocation.get("charged_once") is not True
        or allocation.get("per_method") is not False
        or not isinstance(wall, Mapping)
        or not isinstance(gpu, Mapping)
    ):
        raise ComparisonError("공통 source summary 계약이 다르다")
    devices = gpu.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ComparisonError("공통 source GPU device 집합이 비었다")
    device_energy_wh = 0.0
    for device in devices:
        if not isinstance(device, Mapping):
            raise ComparisonError("공통 source GPU device 행이 object가 아니다")
        if device.get("scope") != "whole_device_including_other_loads":
            raise ComparisonError("공통 source energy scope가 device-wide가 아니다")
        device_energy_wh += _number(device.get("energy_wh"), "source device energy Wh")
    return {
        "allocation": "charged_once_not_repeated_per_method",
        "training_jobs": 20,
        "fixed_evaluations": 20,
        "telemetry_extractions": counts.get("telemetry_extractions"),
        "simulator_owned_wall_s": wall.get("simulator_owned_wall_s"),
        "source_parent_span_s": wall.get("source_parent_span_s"),
        "controller_or_child_cpu_s": None,
        "gpu_device_time_s": None,
        "process_gpu_seconds": gpu.get("process_gpu_seconds"),
        "provider_gpu_seconds": gpu.get("provider_gpu_seconds"),
        "device_wide_energy_wh": device_energy_wh,
        "device_energy_scope": gpu.get("scope"),
        "actual_billing_usd": summary.get("provider_cost", {}).get("actual_billing_usd")
        if isinstance(summary.get("provider_cost"), Mapping)
        else None,
    }


def _diagnostic_summary(
    comparison_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], Mapping[str, Any]]]:
    comparison = _grid(comparison_rows, CASES, METHODS, "diagnostic comparison")
    costs = _grid(cost_rows, CASES, METHODS, "diagnostic cost")
    steps: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        key: [] for key in comparison
    }
    seen_steps: set[tuple[str, str, str, int]] = set()
    for row in step_rows:
        case_id, method = row.get("case_id"), row.get("method")
        source_sha, source_line = row.get("source_journal_sha256"), row.get("source_line")
        if (
            not isinstance(case_id, str)
            or not isinstance(method, str)
            or not isinstance(source_sha, str)
            or type(source_line) is not int
        ):
            raise ComparisonError("controller step identity 형식이 잘못됐다")
        pair = (case_id, method)
        if pair not in steps:
            raise ComparisonError(f"controller step에 범위 밖 case/method가 있다: {pair}")
        identity = (*pair, source_sha, source_line)
        if identity in seen_steps:
            raise ComparisonError(f"controller step source가 중복됐다: {identity}")
        seen_steps.add(identity)
        steps[pair].append(row)
    if any(not rows for rows in steps.values()):
        raise ComparisonError("controller step가 없는 diagnostic arm이 있다")

    methods: dict[str, Any] = {}
    for method in METHODS:
        method_costs = [costs[(case, method)] for case in CASES]
        method_comparison = [comparison[(case, method)] for case in CASES]
        for comp, cost in zip(method_comparison, method_costs, strict=True):
            if comp.get("diagnostic_outcome") != cost.get("diagnostic_outcome"):
                raise ComparisonError("comparison과 cost의 diagnostic outcome이 다르다")
        outcomes = Counter(str(row.get("diagnostic_outcome")) for row in method_comparison)
        controller_cpu = sum(
            _number(row.get("process_cpu_s"), "controller process CPU")
            for case in CASES
            for row in steps[(case, method)]
        )
        methods[method] = {
            "cases": 6,
            "correct_confirmation_count": outcomes.get("correct_confirmation", 0),
            "correct_confirmation_rate": outcomes.get("correct_confirmation", 0) / 6,
            "outcomes": dict(sorted(outcomes.items())),
            "hold_count": outcomes.get("hold", 0),
            "normal_success": None,
            "resources": {
                "sdk_launches": sum(
                    _number(row.get("SDK_launches"), "diagnostic SDK launches")
                    for row in method_costs
                ),
                "simulator_owned_wall_s": sum(
                    _number(row.get("owned_wall_s"), "diagnostic owned wall")
                    for row in method_costs
                ),
                "end_to_end_arm_sum_s": sum(
                    _number(row.get("end_to_end_s"), "diagnostic end-to-end wall")
                    for row in method_costs
                ),
                "recorded_controller_process_cpu_s": controller_cpu,
                "owned_child_cpu_s": None,
                "model_calls": sum(
                    _number(row.get("model_calls"), "diagnostic model calls")
                    for row in method_costs
                ),
                "input_tokens": sum(
                    _number(row.get("input_tokens"), "diagnostic input tokens")
                    for row in method_costs
                ),
                "output_tokens": sum(
                    _number(row.get("output_tokens"), "diagnostic output tokens")
                    for row in method_costs
                ),
                "actual_billing_usd": None,
                "provider_gpu_s": None,
                "gpu_device_time_s": None,
                "device_energy_wh": sum(
                    _number(row.get("device_energy_wh"), "diagnostic device energy Wh")
                    for row in method_costs
                ),
                "device_energy_scope": "device_wide_not_process_attributed",
            },
        }
    return {
        "scope": "six_cases_four_methods_24_arms",
        "cases": list(CASES),
        "arm_count": 24,
        "normal_evidence_scope": "not_observed_in_primary_six_cases",
        "methods": methods,
    }, costs


def _recovery_summary(
    summary: Mapping[str, Any],
    diagnostic_costs: Mapping[tuple[str, str], Mapping[str, Any]],
    attempt_timing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        summary.get("schema") != "registered_recovery_comparison_public_summary_v1"
        or summary.get("planned_cases") != list(RECOVERY_CASES)
        or summary.get("planned_methods") != list(METHODS)
        or summary.get("planned_arms") != 16
        or not isinstance(summary.get("arms"), list)
    ):
        raise ComparisonError("recovery summary 계획 계약이 다르다")
    arms = _grid(summary["arms"], RECOVERY_CASES, METHODS, "recovery")
    classes = {key: _recovery_class(row) for key, row in arms.items()}
    all_complete = all(value in TERMINAL_RECOVERY for value in classes.values())
    verified_timing = _validate_attempt_timing(attempt_timing)
    timing_rows = (
        _grid(verified_timing["per_arm"], RECOVERY_CASES, METHODS, "recovery attempt timing")
        if verified_timing is not None
        else None
    )

    methods: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [arms[(case, method)] for case in RECOVERY_CASES]
        outcomes = Counter(classes[(case, method)] for case in RECOVERY_CASES)
        cost_maps: list[Mapping[str, Any]] = []
        for row in method_rows:
            costs = row.get("costs")
            if not isinstance(costs, Mapping):
                raise ComparisonError("recovery costs가 object가 아니다")
            cost_maps.append(costs)
        aggregate: dict[str, Any] = {
            field: _known(cost.get(field) for cost in cost_maps) for field in RECOVERY_COST_FIELDS
        }
        aggregate["device_energy_wh"] = _known(_device_energy(cost) for cost in cost_maps)
        aggregate["device_energy_scope"] = "device_wide_not_process_attributed"
        aggregate["gpu_device_time_s"] = None
        aggregate["provider_gpu_s"] = None
        aggregate["actual_billing_usd"] = None

        diag_subset = [diagnostic_costs[(case, method)] for case in RECOVERY_CASES]
        diagnostic_owned = sum(
            _number(row.get("owned_wall_s"), "four-case diagnostic owned wall")
            for row in diag_subset
        )
        diagnostic_end_to_end = sum(
            _number(row.get("end_to_end_s"), "four-case diagnostic end-to-end")
            for row in diag_subset
        )
        simulator = aggregate["simulator_owned_wall_s"]["value"]
        effect_wall = aggregate["whole_effect_active_phase_s"]["value"]
        recorded_attempt = (
            _known(timing_rows[(case, method)].get("active_command_wall_s") for case in RECOVERY_CASES)
            if timing_rows is not None
            else None
        )
        recorded_attempt_value = recorded_attempt["value"] if recorded_attempt is not None else None
        methods[method] = {
            "arms": 4,
            "outcomes": dict(sorted(outcomes.items())),
            "decision_terminal_arm_count": sum(
                value in TERMINAL_RECOVERY for value in outcomes.elements()
            ),
            "recovery_attempted_count": sum(row.get("recovery_attempted") is True for row in method_rows),
            "all_arms_complete": all(
                classes[(case, method)] in TERMINAL_RECOVERY for case in RECOVERY_CASES
            ),
            "normal_success": None,
            "costs": aggregate,
            "four_case_cost_components": {
                "diagnostic_simulator_owned_wall_s": diagnostic_owned,
                "diagnostic_end_to_end_arm_sum_s": diagnostic_end_to_end,
                "recovery_scientific_effect_simulator_owned_wall_s": simulator,
                "recovery_whole_effect_active_phase_s": effect_wall,
                "diagnostic_plus_scientific_effect_simulator_owned_wall_s": (
                    diagnostic_owned + simulator if simulator is not None else None
                ),
                "recorded_recovery_attempt_active_command_wall_s": recorded_attempt,
                "diagnostic_plus_recorded_attempt_active_command_wall_s": (
                    diagnostic_end_to_end + recorded_attempt_value
                    if recorded_attempt_value is not None
                    else None
                ),
                "scope": "four_recovery_cases_only_not_six_case_diagnostic_total",
                "is_full_user_elapsed_time": False,
            },
        }

    paired: list[dict[str, Any]] = []
    for case in RECOVERY_CASES:
        for left, right in itertools.combinations(METHODS, 2):
            left_row, right_row = arms[(case, left)], arms[(case, right)]
            left_class, right_class = classes[(case, left)], classes[(case, right)]
            left_endpoint, right_endpoint = left_row.get("endpoint"), right_row.get("endpoint")
            if (
                left_class != right_class
                or left_class not in {"recovered", "not_recovered"}
                or not isinstance(left_endpoint, Mapping)
                or not isinstance(right_endpoint, Mapping)
                or set(left_endpoint) != set(right_endpoint)
            ):
                continue
            left_cost = left_row.get("costs", {}).get("simulator_owned_wall_s")
            right_cost = right_row.get("costs", {}).get("simulator_owned_wall_s")
            if left_cost is None or right_cost is None:
                continue
            left_value = _number(left_cost, "paired left simulator wall")
            right_value = _number(right_cost, "paired right simulator wall")
            paired.append(
                {
                    "case_id": case,
                    "outcome": left_class,
                    "left_method": left,
                    "right_method": right,
                    "left_simulator_owned_wall_s": left_value,
                    "right_simulator_owned_wall_s": right_value,
                    "right_minus_left_s": right_value - left_value,
                    "endpoint_fields": sorted(left_endpoint),
                    "scope": "descriptive_same_case_same_terminal_outcome",
                }
            )
    return {
        "scope": "four_cases_four_methods_16_arms",
        "cases": list(RECOVERY_CASES),
        "arm_count": 16,
        "all_recovery_arms_complete": all_complete,
        "completion_status": "complete" if all_complete else "incomplete",
        "controller_halted": summary.get("controller_halted"),
        "reported_completed_arms": summary.get("completed_arms"),
        "attempt_timing": verified_timing,
        "methods": methods,
        "paired_cost_rows": paired,
        "paired_cost_conclusion_scope": "observed_values_only_no_superiority_claim",
    }


def summarize(
    *,
    comparison_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    source_summary: Mapping[str, Any],
    recovery_summary: Mapping[str, Any],
    bindings: Mapping[str, str],
    recovery_attempt_timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic, diagnostic_costs = _diagnostic_summary(comparison_rows, cost_rows, step_rows)
    recovery = _recovery_summary(recovery_summary, diagnostic_costs, recovery_attempt_timing)
    return {
        "schema": "mechanism_resource_comparison_summary_v1",
        "status": "complete" if recovery["all_recovery_arms_complete"] else "incomplete",
        "input_bindings": dict(sorted(bindings.items())),
        "common_source_cost": _validate_source(source_summary),
        "diagnostic_primary": diagnostic,
        "recovery_primary": recovery,
        "measurement_metadata": {
            "sdk_wall": "simulator_owned_wall_s",
            "controller_cpu": "recorded_process_cpu_s_only",
            "owned_child_cpu_s": None,
            "provider_gpu_s": None,
            "actual_billing_usd": None,
            "energy": "device_wide_not_process_attributed",
            "unknown_values_are_not_zero": True,
        },
        "interpretation": {
            "descriptive_only": True,
            "superiority_claimed": False,
            "efficiency_improvement_claimed": False,
            "hold_low_cost_is_not_success": True,
            "normal_success": None,
            "normal_reason": "not_observed_in_primary_six_case_cohort",
            "tokenfactory_price_used": False,
        },
    }


def _binding(logical_id: str, path: Path) -> tuple[str, str]:
    return logical_id, _sha256(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-rows", type=Path, required=True)
    parser.add_argument("--cost-rows", type=Path, required=True)
    parser.add_argument("--step-rows", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--recovery-summary", type=Path, required=True)
    parser.add_argument("--recovery-attempt-timing", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "diagnostic_comparison_case_rows": args.comparison_rows,
        "diagnostic_cost_arm_rows": args.cost_rows,
        "diagnostic_controller_step_rows": args.step_rows,
        "common_source_summary": args.source_summary,
        "recovery_summary": args.recovery_summary,
    }
    if args.recovery_attempt_timing is not None:
        paths["recovery_attempt_timing"] = args.recovery_attempt_timing
    bindings = dict(_binding(logical_id, path) for logical_id, path in paths.items())
    result = summarize(
        comparison_rows=_read_jsonl(args.comparison_rows),
        cost_rows=_read_jsonl(args.cost_rows),
        step_rows=_read_jsonl(args.step_rows),
        source_summary=_read_json(args.source_summary),
        recovery_summary=_read_json(args.recovery_summary),
        recovery_attempt_timing=(
            _read_json(args.recovery_attempt_timing)
            if args.recovery_attempt_timing is not None
            else None
        ),
        bindings=bindings,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
