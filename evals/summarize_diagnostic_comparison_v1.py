"""종결된 진단 비교 실행을 경로 비노출 공개 집계로 변환한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evals import run_diagnostic_comparison_v1 as RUNNER
from rl_triage import resource_comparison as RESOURCE

SCORING_SCHEMA = "diagnostic_comparison_scoring_key_v1"
SUMMARY_SCHEMA = "diagnostic_comparison_public_summary_v1"
METHODS = RUNNER.METHODS
MODEL_METHODS = {"one_shot_llm", "iterative_agent"}
PENDING = {"not_started", "resume_required", "pending_result"}
FAILURE_EXACT = {"integrity_failure", "controller_transition_limit", "state_contract_failure"}
PUBLIC_FORBIDDEN = ("bench/private", "claude_reward", "\\Users\\", "/Users/", "/home/")


class SummaryContractError(ValueError):
    """입력 결합이나 공개 집계 계약이 깨졌을 때 발생한다."""


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise SummaryContractError(f"{label} cannot be a symlink")
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON: {token}")
            ),
        )
    except (OSError, ValueError) as error:
        raise SummaryContractError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SummaryContractError(f"{label} must be an object")
    canonical(value)
    return value


def _number(value: object, label: str, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SummaryContractError(f"{label} must be a finite non-negative number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SummaryContractError(f"{label} must be a finite non-negative number or null")
    return result


def _validate_scoring(value: dict[str, Any], plan: dict[str, Any], plan_sha: str) -> dict[str, dict]:
    required = {"schema", "source_anchor", "plan_sha256", "expected_arm_count", "cases"}
    if set(value) != required or value.get("schema") != SCORING_SCHEMA:
        raise SummaryContractError("scoring key fields differ")
    if value["source_anchor"] != plan["source_anchor"] or value["plan_sha256"] != plan_sha:
        raise SummaryContractError("scoring key source or plan binding differs")
    expected = len(plan["cases"]) * len(METHODS)
    if value["expected_arm_count"] != expected:
        raise SummaryContractError("scoring key arm denominator differs")
    rows = value["cases"]
    if not isinstance(rows, list):
        raise SummaryContractError("scoring cases must be a list")
    by_case: dict[str, dict] = {}
    for row in rows:
        fields = {"case_id", "case_kind", "injected_probes", "expected_probes"}
        if not isinstance(row, dict) or set(row) != fields:
            raise SummaryContractError("scoring case fields differ")
        case_id, kind = row["case_id"], row["case_kind"]
        injected, expected_probes = row["injected_probes"], row["expected_probes"]
        if (not isinstance(case_id, str) or not case_id or case_id in by_case
                or kind not in {"normal", "single", "compound"}
                or not isinstance(injected, list) or not isinstance(expected_probes, list)
                or any(not isinstance(item, str) or not item for item in injected + expected_probes)
                or len(injected) != len(set(injected)) or len(expected_probes) != len(set(expected_probes))
                or not set(expected_probes) <= set(injected)):
            raise SummaryContractError("scoring case content differs")
        if ((kind == "normal" and (injected or expected_probes))
                or (kind == "single" and len(injected) != 1)
                or (kind == "compound" and len(injected) < 2)):
            raise SummaryContractError("scoring case kind differs from injected probes")
        by_case[case_id] = row
    plan_cases = {row["case_id"] for row in plan["cases"]}
    if set(by_case) != plan_cases:
        raise SummaryContractError("scoring cases differ from plan")
    return by_case


def _status_group(status: object) -> str:
    if not isinstance(status, str) or not status:
        return "failure"
    if status in PENDING:
        return "pending"
    if status == "host_no_intervention":
        return "host_blocked"
    if status.startswith("hold_") or status == "model_hold":
        return "hold"
    if status in RUNNER.TERMINAL:
        return "terminal"
    if (status in FAILURE_EXACT or status.endswith("_failure")
            or status.startswith(("rejected", "blocked_"))):
        return "failure"
    return "incomplete"


def _observed_abnormal(arm: Mapping[str, Any]) -> set[str]:
    state = arm.get("state")
    if not isinstance(state, Mapping):
        return set()
    observed = state.get("observed")
    if not isinstance(observed, Mapping):
        view = state.get("view")
        observed = view.get("observed") if isinstance(view, Mapping) else None
    if not isinstance(observed, Mapping):
        return set()
    return {str(probe) for probe, outcome in observed.items() if outcome == "abnormal"}


def _diagnostic_outcome(arm: Mapping[str, Any], scoring: dict) -> str:
    group = _status_group(arm.get("status"))
    if group in {"pending", "failure", "incomplete"}:
        return group
    method = arm.get("method")
    observed = _observed_abnormal(arm)
    if scoring["case_kind"] == "normal":
        if group == "host_blocked":
            return "normal_host_blocked"
        if method in MODEL_METHODS and arm.get("status") == "model_hold" and not observed:
            state = arm.get("state")
            model = state.get("model") if isinstance(state, Mapping) else None
            if isinstance(model, Mapping) and model.get("model_status") == "valid":
                return "normal_model_success"
            return "hold"
        if group == "hold":
            return "hold"
        return "normal_other_terminal"
    if group == "hold":
        return "hold"
    injected, expected = set(scoring["injected_probes"]), set(scoring["expected_probes"])
    if observed - injected:
        return "wrong_confirmation"
    if arm.get("status") != "confirmed":
        return "no_confirmation"
    if scoring["case_kind"] == "compound" and observed & injected and not expected <= observed:
        return "partial_confirmation"
    if expected <= observed:
        return "correct_confirmation"
    return "no_confirmation"


def _cost_events(
    arm: Mapping[str, Any], seen: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resource = arm.get("resource_cost")
    if resource is None:
        return [], {"present": False, "pending_model_reservations": None,
                    "reserved_model_attempts": None}
    if not isinstance(resource, Mapping):
        raise SummaryContractError("arm resource_cost must be an object")
    events = resource.get("events")
    if not isinstance(events, list):
        raise SummaryContractError("resource events must be a list")
    clean: list[dict[str, Any]] = []
    local_event_ids: set[str] = set()
    for raw in events:
        if not isinstance(raw, dict) or not isinstance(raw.get("costs"), dict):
            raise SummaryContractError("resource event fields differ")
        phase, attempt, receipt = raw.get("phase_id"), raw.get("attempt_id"), raw.get("receipt_id")
        if not all(isinstance(value, str) and value for value in (phase, attempt, receipt)):
            raise SummaryContractError("resource event identity is missing")
        event_id = raw.get("event_id")
        if event_id is None:
            event_id = digest({"phase_id": phase, "attempt_id": attempt, "receipt_id": receipt})
        if not isinstance(event_id, str) or not event_id:
            raise SummaryContractError("resource event_id is invalid")
        model_api = raw.get("model_api")
        if model_api is not None and not isinstance(model_api, Mapping):
            raise SummaryContractError("resource event model_api differs")
        if isinstance(model_api, Mapping):
            for field in ("decision_manifest_sha256", "reservation_sha256"):
                value = model_api.get(field)
                if value is not None and (
                    not isinstance(value, str) or RUNNER._SHA.fullmatch(value) is None
                ):
                    raise SummaryContractError(f"resource event {field} differs")
        encoded = canonical(raw)
        if event_id in seen:
            if seen[event_id]["encoded"] != encoded:
                raise SummaryContractError(f"conflicting duplicate resource event: {event_id}")
            seen[event_id]["occurrences"] += 1
        costs = {}
        for key in RESOURCE.COST_KEYS:
            costs[key] = _number(raw["costs"].get(key), f"event.{key}")
        event = {
            "event_id": event_id,
            "status": raw.get("status"),
            "costs": costs,
            "model_api": (
                {
                    "model_requested": model_api.get("model_requested"),
                    "model_returned": model_api.get("model_returned"),
                    "finish_reason": model_api.get("finish_reason"),
                    "completion_attempt_returned": model_api.get("completion_attempt_returned"),
                    "result_class": model_api.get("result_class"),
                    "confirmed_http_response": model_api.get("confirmed_http_response"),
                    "global_index": model_api.get("global_index"),
                    "decision_manifest_sha256": model_api.get("decision_manifest_sha256"),
                    "reservation_sha256": model_api.get("reservation_sha256"),
                    "model_call_scope": model_api.get("model_call_scope"),
                    "hosted_provider_gpu_s": model_api.get("hosted_provider_gpu_s"),
                    "actual_billing_usd": model_api.get("actual_billing_usd"),
                    "reproducible_estimate_usd": model_api.get("reproducible_estimate_usd"),
                }
                if isinstance(model_api, Mapping)
                else None
            ),
        }
        if event_id not in seen:
            seen[event_id] = {"encoded": encoded, "event": event, "occurrences": 1}
        if event_id not in local_event_ids:
            clean.append(event)
            local_event_ids.add(event_id)
    model = resource.get("model_api_summary")
    if model is not None and not isinstance(model, Mapping):
        raise SummaryContractError("model API summary differs")
    model_fields = (
        "reserved_attempts", "completed_attempts", "normal_results", "infra_error_results",
        "failed_results", "unconfirmed_reservations", "model_calls", "known_model_calls",
        "confirmed_http_responses", "confirmed_no_http_responses",
        "unknown_http_response_count", "input_tokens", "output_tokens",
        "hosted_provider_gpu_s", "actual_billing_usd", "reproducible_estimate_usd",
    )
    model_values = (
        {field: model.get(field) for field in model_fields}
        if isinstance(model, Mapping)
        else {field: None for field in model_fields}
    )
    return clean, {
        "present": True,
        **model_values,
        "pending_model_reservations": model_values["unconfirmed_reservations"],
        "reserved_model_attempts": model_values["reserved_attempts"],
        "pending_reservation_source": (
            "reservation_without_response"
            if model_values["unconfirmed_reservations"]
            else None
        ),
    }


def _aggregate_events(events: list[dict[str, Any]], *, resource_present: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in RESOURCE.COST_KEYS:
        values = [event["costs"][key] for event in events]
        known = [value for value in values if value is not None]
        unknown = len(values) - len(known)
        if not events:
            value = None
        else:
            value = None if unknown else sum(known)
        result[key] = {"value": value, "known_sum": sum(known) if known else 0.0,
                       "unknown_event_count": unknown if events else (1 if resource_present else 0)}
    result["failed_event_count"] = sum(event["status"] == "failed" for event in events)
    result["pending_event_count"] = sum(event["status"] == "unknown" for event in events)
    result["unique_event_count"] = len(events)
    return result


def _actual_owned(
    resource: object, seen: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "owned_sdk_execution_count", "owned_wall_s", "natural_exit_count",
        "non_natural_exit_count", "failure_count", "incomplete_count",
    )
    if not isinstance(resource, Mapping):
        return {key: None for key in fields}
    rows = resource.get("request_costs")
    if not isinstance(rows, list):
        return {key: None for key in fields}
    owned_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("actual_owned"), Mapping):
            raise SummaryContractError("actual_owned row differs")
        identity = row.get("request_id") or row.get("phase_id")
        if not isinstance(identity, str) or not identity:
            raise SummaryContractError("actual_owned request identity is missing")
        encoded = canonical(row)
        if identity in seen:
            if seen[identity]["encoded"] != encoded:
                raise SummaryContractError(f"conflicting duplicate actual_owned request: {identity}")
            seen[identity]["occurrences"] += 1
        else:
            seen[identity] = {
                "encoded": encoded, "actual_owned": dict(row["actual_owned"]), "occurrences": 1,
            }
        owned_rows.append(row["actual_owned"])
    result: dict[str, Any] = {}
    for key in fields:
        values = [row.get(key) for row in owned_rows]
        if not values or any(value is None for value in values):
            result[key] = None
        elif any(isinstance(value, bool) or not isinstance(value, Real) or value < 0 for value in values):
            raise SummaryContractError(f"actual_owned {key} differs")
        else:
            result[key] = sum(float(value) for value in values if isinstance(value, Real))
    return result


def _device_scope(arm: Mapping[str, Any]) -> dict[str, Any]:
    sampling = arm.get("gpu_sampling")
    if not isinstance(sampling, Mapping):
        return {"available": False, "scope": None, "process_attribution": None,
                "provider_gpu_s": None, "power_or_energy": None}
    stop = sampling.get("stop")
    receipt = stop.get("receipt_ref") if isinstance(stop, Mapping) else None
    if receipt is None:
        return {"available": False, "scope": sampling.get("scope"),
                "process_attribution": None, "provider_gpu_s": None, "power_or_energy": None}
    _path, body = RUNNER._read_ref(receipt, "GPU sampler stop")
    if body.get("scope") != "device_wide":
        raise SummaryContractError("GPU sampler scope differs")
    return {
        "available": True,
        "scope": "whole_device_including_other_loads",
        "process_attribution": None,
        "provider_gpu_s": None,
        "power_or_energy": {
            "sample_count": body.get("sample_count"),
            "energy": body.get("energy"),
            "device_summary": body.get("device_summary"),
        },
    }


def _timing(arm: Mapping[str, Any], event_costs: dict[str, Any]) -> dict[str, Any]:
    timing = arm.get("timing")
    elapsed = None
    if isinstance(timing, Mapping):
        start = _number(timing.get("end_to_end_start_s"), "timing start")
        end = _number(timing.get("end_to_end_end_s"), "timing end")
        if start is not None and end is not None:
            if end < start:
                raise SummaryContractError("end-to-end timing is reversed")
            elapsed = end - start
    interval_sum = event_costs["execution_wall_s"]["value"]
    return {"end_to_end_elapsed_s": elapsed, "execution_interval_sum_s": interval_sum,
            "equality_required": False}


def _arm_row(
    case_id: str, method: str, arm: Mapping[str, Any] | None, scoring: dict,
    seen_events: dict[str, dict[str, Any]],
    seen_owned: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if arm is None:
        return {"case_id": case_id, "method": method, "present": False,
                "status": "missing", "status_group": "incomplete",
                "diagnostic_outcome": "incomplete", "resource": None,
                "recovery": {"executed": False, "outcome": None}}
    status = arm.get("status")
    events, model = _cost_events(arm, seen_events)
    event_costs = _aggregate_events(events, resource_present=model["present"])
    resource = arm.get("resource_cost")
    return {
        "case_id": case_id,
        "method": method,
        "present": True,
        "status": status,
        "status_group": _status_group(status),
        "diagnostic_outcome": _diagnostic_outcome(arm, scoring),
        "observed_abnormal_probes": sorted(_observed_abnormal(arm)),
        "resource": {
            "events": event_costs,
            "event_refs": [
                {"event_id": event["event_id"], "status": event["status"],
                 "model_api": event["model_api"]}
                for event in events
            ],
            "actual_owned": _actual_owned(resource, seen_owned),
            "model": model,
            "timing": _timing(arm, event_costs),
            "device": _device_scope(arm),
        },
        "recovery": {"executed": False, "outcome": None},
    }


def _method_aggregate(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    outcomes = sorted({row["diagnostic_outcome"] for row in rows})
    counts = {outcome: sum(row["diagnostic_outcome"] == outcome for row in rows)
              for outcome in outcomes}
    return {
        "denominator_cases": denominator,
        "present": sum(row["present"] for row in rows),
        "terminal": sum(row["status_group"] in {"terminal", "hold", "host_blocked"} for row in rows),
        "pending": sum(row["status_group"] == "pending" for row in rows),
        "failed": sum(row["status_group"] == "failure" for row in rows),
        "incomplete": sum(row["status_group"] == "incomplete" for row in rows),
        "outcomes": counts,
        "correct_confirmation_rate": (
            None if denominator == 0 else counts.get("correct_confirmation", 0) / denominator
        ),
        "recovery_rate": None,
    }


def _global_resource(
    seen_events: dict[str, dict[str, Any]], seen_owned: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unique = [entry["event"] for entry in seen_events.values()]
    totals = _aggregate_events(unique, resource_present=bool(unique))
    model_events = [event["model_api"] for event in unique if event["model_api"] is not None]
    owned_fields = (
        "owned_sdk_execution_count", "owned_wall_s", "natural_exit_count",
        "non_natural_exit_count", "failure_count", "incomplete_count",
    )
    owned_totals: dict[str, Any] = {}
    for field in owned_fields:
        values = [entry["actual_owned"].get(field) for entry in seen_owned.values()]
        owned_totals[field] = (
            None if not values or any(value is None for value in values) else sum(values)
        )
    return {
        "scope": "global_unique_event_set_not_multiplied_by_arm_references",
        "events": totals,
        "unique_event_count": len(unique),
        "shared_event_count": sum(entry["occurrences"] > 1 for entry in seen_events.values()),
        "repeated_reference_count": sum(entry["occurrences"] - 1 for entry in seen_events.values()),
        "actual_owned": {
            "scope": "global_unique_request_identity_not_multiplied_by_arm_references",
            "unique_request_count": len(seen_owned),
            "shared_request_count": sum(entry["occurrences"] > 1 for entry in seen_owned.values()),
            "repeated_reference_count": sum(
                entry["occurrences"] - 1 for entry in seen_owned.values()
            ),
            **owned_totals,
        },
        "model_api": {
            "reserved_attempts": len(model_events),
            "completion_attempt_returned": sum(
                event["completion_attempt_returned"] is True for event in model_events
            ),
            "completion_attempt_not_returned": sum(
                event["completion_attempt_returned"] is False for event in model_events
            ),
            "completion_attempt_unknown": sum(
                event["completion_attempt_returned"] is None for event in model_events
            ),
            "confirmed_http_response": sum(
                event["confirmed_http_response"] == 1 for event in model_events
            ),
            "confirmed_no_http_response": sum(
                event["confirmed_http_response"] == 0 for event in model_events
            ),
            "http_response_unknown": sum(
                event["confirmed_http_response"] is None for event in model_events
            ),
            "hosted_provider_gpu_s": None,
            "actual_billing_usd": None,
            "billing_scope": "provider_actual_billing_unavailable",
        },
    }


def build_summary(
    plan: dict[str, Any], plan_ref: dict, raw: dict[str, Any], raw_sha: str,
    scoring: dict[str, Any], scoring_sha: str, *, allow_fixture: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if raw.get("schema") != "diagnostic_comparison_result_v1" or raw.get("plan") != plan_ref:
        raise SummaryContractError("raw run plan binding differs")
    if raw.get("execution_class") != ("fixture" if allow_fixture else "registered"):
        raise SummaryContractError("raw run execution class differs")
    scoring_by_case = _validate_scoring(scoring, plan, plan_ref["sha256"])
    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, list):
        raise SummaryContractError("raw run arms must be a list")
    indexed: dict[tuple[str, str], dict] = {}
    for arm in raw_arms:
        if not isinstance(arm, dict):
            raise SummaryContractError("raw arm must be an object")
        case_id, method = arm.get("case_id"), arm.get("method")
        if (not isinstance(case_id, str) or not isinstance(method, str)
                or case_id not in scoring_by_case or method not in METHODS):
            raise SummaryContractError("raw arm case/method is missing, unknown, or duplicated")
        key = (case_id, method)
        if key in indexed:
            raise SummaryContractError("raw arm case/method is missing, unknown, or duplicated")
        indexed[key] = arm
    expected_arms = len(plan["cases"]) * len(METHODS)
    if len(indexed) > expected_arms:
        raise SummaryContractError("raw arm count exceeds preregistered denominator")
    seen_events: dict[str, dict[str, Any]] = {}
    seen_owned: dict[str, dict[str, Any]] = {}
    rows = [
        _arm_row(case["case_id"], method, indexed.get((case["case_id"], method)),
                 scoring_by_case[case["case_id"]], seen_events, seen_owned)
        for case in plan["cases"] for method in METHODS
    ]
    groups = {row["status_group"] for row in rows}
    complete = (raw.get("status") == "completed" and len(indexed) == expected_arms
                and not groups & {"pending", "incomplete"})
    status = "complete" if complete else "incomplete"
    methods = {method: _method_aggregate(
        [row for row in rows if row["method"] == method], len(plan["cases"]))
        for method in METHODS}
    case_pairs = [{
        "case_id": case["case_id"],
        "four_arm_present": all((case["case_id"], method) in indexed for method in METHODS),
        "four_arm_terminal": all(
            next(row for row in rows if row["case_id"] == case["case_id"]
                 and row["method"] == method)["status_group"]
            in {"terminal", "hold", "host_blocked", "failure"}
            for method in METHODS
        ),
    } for case in plan["cases"]]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "inputs": {"plan_sha256": plan_ref["sha256"], "raw_run_sha256": raw_sha,
                   "scoring_key_sha256": scoring_sha, "source_anchor": plan["source_anchor"]},
        "denominators": {"cases": len(plan["cases"]), "arms": expected_arms},
        "observed": {"arms_present": len(indexed),
                     "arms_terminal": sum(row["status_group"] in {
                         "terminal", "hold", "host_blocked", "failure"} for row in rows),
                     "arms_pending": sum(row["status_group"] == "pending" for row in rows),
                     "arms_missing_or_incomplete": sum(row["status_group"] == "incomplete" for row in rows)},
        "case_pairs": case_pairs,
        "methods": methods,
        "comparison": {"descriptive_only": True, "positive_effect_forced": False,
                       "superiority_claimed": False, "preferred_method": None},
        "recovery": {"executed": False, "diagnostic_success_used_as_recovery": False,
                     "outcome": None},
        "unknown_values_remain_null": True,
        "resource_accounting": _global_resource(seen_events, seen_owned),
        "event_deduplication": {"unique_events": len(seen_events),
                                "identity": "event_id_or_digest_of_phase_attempt_receipt",
                                "common_cost_counted_once": True},
    }
    wording = (
        f"등록된 {expected_arms}개 arm 중 {len(indexed)}개 기록을 확인했다. "
        f"종결 {summary['observed']['arms_terminal']}개, pending {summary['observed']['arms_pending']}개, "
        f"누락·미완료 {summary['observed']['arms_missing_or_incomplete']}개다. "
        "진단 비교는 같은 case의 네 방법을 기술 통계로만 나란히 두며, 회복 성과나 우월성을 주장하지 않는다."
    )
    return summary, rows, wording


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def write_outputs(output: Path, summary: dict, rows: list[dict], wording: str) -> dict:
    output = output.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    rows_path = output / "case-rows.jsonl"
    wording_path = output / "wording-candidate.md"
    _write_new(summary_path, canonical(summary) + b"\n")
    _write_new(rows_path, b"".join(canonical(row) + b"\n" for row in rows))
    _write_new(wording_path, (wording + "\n").encode("utf-8"))
    manifest = {
        "schema": "diagnostic_comparison_public_manifest_v1",
        "files": {
            path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
            for path in (summary_path, rows_path, wording_path)
        },
        "inputs": summary["inputs"],
        "contains_private_paths": False,
    }
    manifest_path = output / "public-manifest.json"
    _write_new(manifest_path, canonical(manifest) + b"\n")
    for path in (summary_path, rows_path, wording_path, manifest_path):
        text = path.read_text(encoding="utf-8")
        if any(token.casefold() in text.casefold() for token in PUBLIC_FORBIDDEN):
            raise SummaryContractError(f"public output contains a private path: {path.name}")
    return manifest


def summarize_files(
    plan_path: Path, raw_path: Path, scoring_path: Path, output: Path, anchor: str,
    *, repo_root: Path = ROOT,
) -> dict:
    repo = repo_root.resolve(strict=True)
    plan_resolved = plan_path.resolve(strict=True)
    try:
        relative = plan_resolved.relative_to(repo).as_posix()
    except ValueError as error:
        raise SummaryContractError("plan is outside repo") from error
    plan_ref = {"path": relative, "sha256": RUNNER._lf_sha(plan_resolved)}
    plan = RUNNER.load_plan(plan_ref, repo_root=repo, registration_anchor=anchor,
                            require_committed=True)
    raw, scoring = _read(raw_path, "raw run"), _read(scoring_path, "scoring key")
    if raw.get("registration_anchor") != anchor:
        raise SummaryContractError("raw run registration anchor differs")
    summary, rows, wording = build_summary(
        plan, plan_ref, raw, file_sha(raw_path.resolve(strict=True)), scoring,
        file_sha(scoring_path.resolve(strict=True)),
    )
    write_outputs(output, summary, rows, wording)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scoring-key", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = summarize_files(args.plan, args.run, args.scoring_key, args.output, args.anchor,
                                 repo_root=args.repo)
    except Exception as error:  # noqa: BLE001 - CLI는 입력을 수정하거나 재실행하지 않고 거부한다.
        print(canonical({"status": "rejected", "error_type": type(error).__name__,
                         "reason": str(error)}).decode("utf-8"))
        return 2
    print(canonical({"status": result["status"], "output": str(args.output.resolve())}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
