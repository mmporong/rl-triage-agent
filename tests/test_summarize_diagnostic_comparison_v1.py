from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import summarize_diagnostic_comparison_v1 as SUMMARY

RUNNER = SUMMARY.RUNNER
METHODS = SUMMARY.METHODS


def _event(name: str, *, unknown: bool = False, event_id: str | None = None) -> dict:
    costs: dict[str, int | float | None] = {key: 0 for key in SUMMARY.RESOURCE.COST_KEYS}
    costs.update({"gpu_wall_s": 1.5, "execution_wall_s": 1.5,
                  "cpu_s": None, "human_review_s": None, "simulator_steps": None})
    if unknown:
        costs.update({"model_calls": None, "input_tokens": None, "output_tokens": None,
                      "execution_wall_s": None})
    value = {
        "schema": "diagnostic_resource_cost_event_v1",
        "phase_id": f"phase:{name}",
        "attempt_id": f"attempt:{name}",
        "receipt_id": f"receipt:{name}",
        "status": "unknown" if unknown else "completed",
        "costs": costs,
    }
    if event_id is not None:
        value["event_id"] = event_id
    return value


def _arm(case: str, method: str, status: str, *, observed: dict | None = None,
         events: list[dict] | None = None, unconfirmed: int = 0) -> dict:
    return {
        "case_id": case,
        "method": method,
        "status": status,
        "state": {"status": status, "observed": observed or {}},
        "timing": {"end_to_end_start_s": 10.0, "end_to_end_end_s": 14.0},
        "resource_cost": {
            "schema": "diagnostic_arm_resource_cost_v1",
            "events": events if events is not None else [_event(f"{case}:{method}")],
            "request_costs": [{
                "schema": "diagnostic_request_resource_cost_v1",
                "phase_id": f"request:{case}:{method}",
                "request_id": f"request:{case}:{method}",
                "actual_owned": {
                    "owned_sdk_execution_count": 1,
                    "owned_wall_s": 1.5,
                    "natural_exit_count": 1,
                    "non_natural_exit_count": 0,
                    "failure_count": 0,
                    "incomplete_count": int(status == "resume_required"),
                },
            }],
            "model_api_summary": {
                "reserved_attempts": unconfirmed,
                "completed_attempts": 0,
                "normal_results": 0,
                "infra_error_results": 0,
                "failed_results": 0,
                "unconfirmed_reservations": unconfirmed,
                "model_calls": None if unconfirmed else 0,
                "known_model_calls": 0,
                "confirmed_http_responses": 0,
                "confirmed_no_http_responses": 0,
                "unknown_http_response_count": unconfirmed,
                "input_tokens": None if unconfirmed else 0,
                "output_tokens": None if unconfirmed else 0,
                "hosted_provider_gpu_s": None,
                "actual_billing_usd": None,
                "reproducible_estimate_usd": None,
            },
        },
        "gpu_sampling": {"start": None, "stop": None, "scope": None},
    }


def _plan(case_ids: list[str]) -> tuple[dict, dict]:
    plan = {
        "schema": RUNNER.SCHEMA,
        "source_anchor": "1" * 40,
        "source_closure": {"runner.py": "a" * 64},
        "methods": list(RUNNER.METHODS),
        "cases": [{"case_id": case, "rotation": list(RUNNER.METHODS)} for case in case_ids],
        "limits": {"max_checks": 3, "api_cap": 40, "watchdog_s": 660},
        "inputs": [{"role": f"protocol:{case}", "path": f"{case}.json", "sha256": "b" * 64}
                   for case in case_ids],
        "backend": RUNNER.ACTUAL_BACKEND,
        "authorization": {"actor": "registered-host", "probe_scope": ["p1", "p2", "p9"]},
    }
    return plan, {"path": "plan.json", "sha256": SUMMARY.digest(plan)}


def _scoring(plan: dict, plan_ref: dict, cases: list[dict]) -> dict:
    return {
        "schema": SUMMARY.SCORING_SCHEMA,
        "source_anchor": plan["source_anchor"],
        "plan_sha256": plan_ref["sha256"],
        "expected_arm_count": len(plan["cases"]) * 4,
        "cases": cases,
    }


def test_classifies_normal_host_model_and_compound_without_forcing_positive() -> None:
    plan, plan_ref = _plan(["normal", "compound"])
    scoring = _scoring(plan, plan_ref, [
        {"case_id": "normal", "case_kind": "normal", "injected_probes": [],
         "expected_probes": []},
        {"case_id": "compound", "case_kind": "compound", "injected_probes": ["p1", "p2"],
         "expected_probes": ["p1", "p2"]},
    ])
    arms = [
        _arm("normal", "fixed_sequence", "host_no_intervention"),
        _arm("normal", "static_rule", "none_supported"),
        _arm("normal", "one_shot_llm", "model_hold"),
        _arm("normal", "iterative_agent", "model_hold"),
        _arm("compound", "fixed_sequence", "confirmed", observed={"p1": "abnormal", "p2": "abnormal"}),
        _arm("compound", "static_rule", "confirmed", observed={"p1": "abnormal"}),
        _arm("compound", "one_shot_llm", "confirmed", observed={"p9": "abnormal"}),
        _arm("compound", "iterative_agent", "none_supported"),
    ]
    for arm in arms[2:4]:
        arm["state"]["model"] = {"model_status": "valid"}
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": arms}
    summary, rows, _wording = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    assert by_key[("normal", "fixed_sequence")]["diagnostic_outcome"] == "normal_host_blocked"
    assert by_key[("normal", "one_shot_llm")]["diagnostic_outcome"] == "normal_model_success"
    assert by_key[("compound", "static_rule")]["diagnostic_outcome"] == "partial_confirmation"
    assert by_key[("compound", "one_shot_llm")]["diagnostic_outcome"] == "wrong_confirmation"
    assert summary["comparison"] == {"descriptive_only": True, "positive_effect_forced": False,
                                      "superiority_claimed": False, "preferred_method": None}
    assert summary["recovery"]["outcome"] is None


@pytest.mark.parametrize("model_status", [None, "contract_invalid", "infrastructure_error"])
def test_unverified_or_invalid_normal_hold_is_not_model_normal_success(model_status: str | None) -> None:
    arm = _arm("normal", "one_shot_llm", "model_hold")
    if model_status is not None:
        arm["state"]["model"] = {"model_status": model_status}
    scoring = {"case_id": "normal", "case_kind": "normal", "injected_probes": [], "expected_probes": []}
    assert SUMMARY._diagnostic_outcome(arm, scoring) == "hold"


def test_missing_and_pending_keep_denominator_and_mark_incomplete() -> None:
    plan, plan_ref = _plan(["c1"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "single",
                                         "injected_probes": ["p1"], "expected_probes": ["p1"]}])
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": [
               _arm("c1", "fixed_sequence", "confirmed", observed={"p1": "abnormal"}),
               _arm("c1", "one_shot_llm", "resume_required",
                    events=[_event("pending", unknown=True)], unconfirmed=1),
           ]}
    summary, rows, wording = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    assert summary["status"] == "incomplete"
    assert summary["denominators"] == {"cases": 1, "arms": 4}
    assert summary["observed"] == {"arms_present": 2, "arms_terminal": 1,
                                    "arms_pending": 1, "arms_missing_or_incomplete": 2}
    pending = next(row for row in rows if row["method"] == "one_shot_llm")
    assert pending["resource"]["events"]["model_calls"]["value"] is None
    assert pending["resource"]["model"]["pending_reservation_source"] == "reservation_without_response"
    assert "등록된 4개 arm 중 2개" in wording


def test_duplicate_events_are_deduplicated_but_conflicts_rejected() -> None:
    plan, plan_ref = _plan(["c1"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "normal",
                                         "injected_probes": [], "expected_probes": []}])
    duplicate = _event("same", event_id="event-1")
    arms = [_arm("c1", method, "host_no_intervention", events=[duplicate]) for method in METHODS]
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": arms}
    summary, _rows, _ = SUMMARY.build_summary(plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    assert summary["event_deduplication"]["unique_events"] == 1
    conflicting = json.loads(json.dumps(raw))
    conflicting["arms"][1]["resource_cost"]["events"][0]["costs"]["gpu_wall_s"] = 9
    with pytest.raises(SUMMARY.SummaryContractError, match="conflicting duplicate"):
        SUMMARY.build_summary(plan, plan_ref, conflicting, "c" * 64, scoring, "d" * 64)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def test_actual_cli_validates_committed_sources_and_writes_path_free_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "summary@example.invalid")
    _git(repo, "config", "user.name", "Summary Test")
    code = repo / "runner.py"
    code.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "runner.py")
    _git(repo, "commit", "-m", "source")
    source_anchor = _git(repo, "rev-parse", "HEAD")
    protocol = repo / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8", newline="\n")
    plan = {
        "schema": RUNNER.SCHEMA, "source_anchor": source_anchor,
        "source_closure": {"runner.py": RUNNER._lf_sha(code)},
        "methods": list(METHODS),
        "cases": [{"case_id": "c1", "rotation": list(METHODS)}],
        "limits": {"max_checks": 3, "api_cap": 40, "watchdog_s": 660},
        "inputs": [{"role": "protocol", "path": "protocol.json",
                    "sha256": RUNNER._lf_sha(protocol)}],
        "backend": RUNNER.ACTUAL_BACKEND,
        "authorization": {"actor": "registered-host", "probe_scope": ["p1"]},
    }
    plan_path = repo / "plan.json"
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    _git(repo, "add", "plan.json", "protocol.json")
    _git(repo, "commit", "-m", "registration")
    anchor = _git(repo, "rev-parse", "HEAD")
    plan_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    arms = [_arm("c1", method, "host_no_intervention") for method in METHODS]
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": anchor, "execution_class": "registered",
           "status": "completed", "arms": arms}
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "normal",
                                         "injected_probes": [], "expected_probes": []}])
    raw_path, scoring_path = tmp_path / "raw.json", tmp_path / "scoring.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    scoring_path.write_text(json.dumps(scoring), encoding="utf-8")
    output = tmp_path / "public-output"
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(SUMMARY.__file__)), "--repo", str(repo),
         "--plan", str(plan_path), "--run", str(raw_path), "--scoring-key", str(scoring_path),
         "--anchor", anchor, "--output", str(output)],
        check=False, capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["denominators"]["arms"] == 4
    assert {path.name for path in output.iterdir()} == {
        "summary.json", "case-rows.jsonl", "public-manifest.json", "wording-candidate.md"}
    combined = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert str(tmp_path) not in combined and "bench/private" not in combined

    protocol.write_text('{"mutated":true}\n', encoding="utf-8")
    with pytest.raises(RUNNER.ComparisonContractError, match="plan input changed"):
        SUMMARY.summarize_files(plan_path, raw_path, scoring_path, tmp_path / "second", anchor,
                                repo_root=repo)


def test_scoring_key_and_raw_bindings_reject_mutation() -> None:
    plan, plan_ref = _plan(["c1"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "single",
                                         "injected_probes": ["p1"], "expected_probes": ["p1"]}])
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": []}
    bad = json.loads(json.dumps(scoring))
    bad["plan_sha256"] = "0" * 64
    with pytest.raises(SUMMARY.SummaryContractError, match="binding"):
        SUMMARY.build_summary(plan, plan_ref, raw, "c" * 64, bad, "d" * 64)
    raw["plan"] = {"path": "other.json", "sha256": plan_ref["sha256"]}
    with pytest.raises(SUMMARY.SummaryContractError, match="plan binding"):
        SUMMARY.build_summary(plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)


def test_forty_arm_denominator_distinguishes_terminal_failure_hold_and_pending() -> None:
    case_ids = [f"c{index:02d}" for index in range(10)]
    plan, plan_ref = _plan(case_ids)
    scoring = _scoring(plan, plan_ref, [
        {"case_id": case, "case_kind": "normal", "injected_probes": [],
         "expected_probes": []} for case in case_ids
    ])
    statuses = ("none_supported", "prepare_transport_failure", "hold_prepare_unknown",
                "resume_required")
    arms = [
        _arm(case, method, statuses[ordinal % 4])
        for case in case_ids for ordinal, method in enumerate(METHODS)
    ]
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": arms}
    summary, rows, _ = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    assert summary["denominators"] == {"cases": 10, "arms": 40}
    assert summary["observed"] == {"arms_present": 40, "arms_terminal": 30,
                                    "arms_pending": 10, "arms_missing_or_incomplete": 0}
    assert summary["status"] == "incomplete"
    assert {row["status_group"] for row in rows} == {"terminal", "failure", "hold", "pending"}


def test_device_measurement_stays_whole_device_and_elapsed_is_independent(tmp_path: Path) -> None:
    stop_path = tmp_path / "stop.json"
    stop = {"scope": "device_wide", "sample_count": 3,
            "energy": {"devices": {"gpu": {"energy_j": 12.0}}},
            "device_summary": {"gpu": {"scope": "whole_device_including_other_loads"}}}
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    arm = _arm("c1", "fixed_sequence", "host_no_intervention")
    arm["gpu_sampling"] = {"scope": "device_wide", "start": None,
                           "stop": {"scope": "device_wide", "receipt_ref": {
                               "path": str(stop_path), "sha256": SUMMARY.file_sha(stop_path)}}}
    plan, plan_ref = _plan(["c1"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "normal",
                                         "injected_probes": [], "expected_probes": []}])
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": [arm]}
    _summary, rows, _ = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    resource = rows[0]["resource"]
    assert resource["device"]["scope"] == "whole_device_including_other_loads"
    assert resource["device"]["process_attribution"] is None
    assert resource["device"]["provider_gpu_s"] is None
    assert resource["timing"] == {"end_to_end_elapsed_s": 4.0,
                                    "execution_interval_sum_s": 1.5,
                                    "equality_required": False}


def test_shared_source_event_is_counted_once_and_api_unknowns_are_preserved() -> None:
    shared = _event("source-collection", event_id="source-event")
    shared["costs"]["model_calls"] = 1
    shared["model_api"] = {
        "completion_attempt_returned": True,
        "result_class": "normal",
        "confirmed_http_response": 1,
        "global_index": 1,
        "decision_manifest_sha256": "1" * 64,
        "reservation_sha256": "2" * 64,
        "model_call_scope": "completion_attempt_returned",
        "hosted_provider_gpu_s": None,
        "actual_billing_usd": None,
        "reproducible_estimate_usd": None,
    }
    plan, plan_ref = _plan(["c1"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "c1", "case_kind": "normal",
                                         "injected_probes": [], "expected_probes": []}])
    arms = [_arm("c1", method, "model_hold", events=[shared]) for method in METHODS]
    shared_request = arms[0]["resource_cost"]["request_costs"][0]
    for arm in arms[1:]:
        arm["resource_cost"]["request_costs"] = [shared_request]
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": arms}
    summary, rows, _ = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    accounting = summary["resource_accounting"]
    assert accounting["unique_event_count"] == 1
    assert accounting["shared_event_count"] == 1
    assert accounting["repeated_reference_count"] == 3
    assert accounting["events"]["model_calls"]["value"] == 1
    assert accounting["actual_owned"]["unique_request_count"] == 1
    assert accounting["actual_owned"]["owned_sdk_execution_count"] == 1
    assert accounting["actual_owned"]["repeated_reference_count"] == 3
    assert accounting["model_api"] == {
        "reserved_attempts": 1,
        "completion_attempt_returned": 1,
        "completion_attempt_not_returned": 0,
        "completion_attempt_unknown": 0,
        "confirmed_http_response": 1,
        "confirmed_no_http_response": 0,
        "http_response_unknown": 0,
        "hosted_provider_gpu_s": None,
        "actual_billing_usd": None,
        "billing_scope": "provider_actual_billing_unavailable",
    }
    assert all(row["resource"]["events"]["model_calls"]["value"] == 1 for row in rows)


def test_excluded_hold_arms_remain_in_the_registered_denominator() -> None:
    plan, plan_ref = _plan(["planned-case"])
    scoring = _scoring(plan, plan_ref, [{"case_id": "planned-case", "case_kind": "normal",
                                         "injected_probes": [], "expected_probes": []}])
    raw = {"schema": "diagnostic_comparison_result_v1", "plan": plan_ref,
           "registration_anchor": "2" * 40, "execution_class": "registered",
           "status": "completed", "arms": [
               _arm("planned-case", method, "hold_excluded_by_registered_scope")
               for method in METHODS
           ]}
    summary, rows, _ = SUMMARY.build_summary(
        plan, plan_ref, raw, "c" * 64, scoring, "d" * 64)
    assert summary["denominators"] == {"cases": 1, "arms": 4}
    assert summary["observed"]["arms_terminal"] == 4
    assert summary["status"] == "complete"
    assert all(row["status_group"] == "hold" for row in rows)
