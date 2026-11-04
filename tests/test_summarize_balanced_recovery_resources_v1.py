from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import summarize_balanced_recovery_resources_v1 as summary
from rl_triage import investigation_session as io


def _plan(*, schema: str | None = None) -> dict:
    selected_schema = schema or summary.controller.PLAN_SCHEMA_V8
    profile = summary.PLAN_PROFILES[selected_schema]
    prior = 34 if selected_schema == "registered_recovery_comparison_plan_v10" else 33
    sdk_cap = 31 if selected_schema == "registered_recovery_comparison_plan_v10" else 32
    owned_cap = 63 if selected_schema == "registered_recovery_comparison_plan_v10" else 64
    remeasurement = {
        "profile_id": profile, "imports_allowed": False, "no_model_requery": True,
        "accounting": {"prior_whole_study_sdk_launches": prior,
                       "cumulative_gpu_sdk_launch_cap": 65},
        "known_primary_outcomes": {"path": "evals/results/old/summary.json",
                                   "sha256": "e" * 64},
    }
    if selected_schema == "registered_recovery_comparison_plan_v10":
        remeasurement["known_failed_remeasurement"] = {
            "path": "evals/results/failed/summary.json", "sha256": "f" * 64}
    methods = list(summary.controller.METHODS)
    cases = []
    for index, case_id in enumerate(summary.controller.SUPPORTED_CASES):
        cases.append({"case_id": case_id, "rotation": methods[index:] + methods[:index]})
    return {
        "schema": selected_schema,
        "cases": cases,
        "inputs": [{"role": "diagnostic_plan", "path": "bench/protocols/diagnostic.json",
                    "sha256": "d" * 64}],
        "limits": {"gpu_sdk_launch_cap": sdk_cap, "total_owned_launch_cap": owned_cap},
        "resource_remeasurement": remeasurement,
    }


def _manifest(plan: dict) -> dict:
    arms = []
    for case in plan["cases"]:
        for method in case["rotation"]:
            arms.append({"case_id": case["case_id"], "method": method,
                         "recovery_payload": None if (case["case_id"], method) == summary.EXPECTED_HOLD else {}})
    return {"arms": arms, "registration_anchor": "a" * 40,
            "plan": {}, "job": {}}


def _status(manifest: dict, plan: dict) -> dict:
    arms = []
    for row in manifest["arms"]:
        hold = (row["case_id"], row["method"]) == summary.EXPECTED_HOLD
        arms.append({"case_id": row["case_id"], "method": row["method"],
                     "outcome": "hold" if hold else "not_recovered",
                     "resource_cost": None if hold else {}})
    return {"completed": True, "halted": False, "completed_arms": 16, "total_arms": 16,
            "arms": arms, "limits": plan["limits"], "api_calls": 0}


def _timing(plan: dict) -> dict:
    transitions = []
    for sequence, (case_id, method) in enumerate(summary._expected_order(plan), start=1):
        transitions.extend([
            {"sequence": sequence * 2 - 1, "case_id": case_id, "method": method},
            {"sequence": sequence * 2, "case_id": case_id, "method": method},
        ])
    return {"transitions": transitions}


@pytest.mark.parametrize("mutation,match", [
    ("order", "arm re-entered|actual transition order"),
    ("hold", "HOLD"),
    ("import", "must not import"),
    ("incomplete", "not a complete"),
])
def test_batch_contract_rejects_order_hold_import_and_incomplete(mutation: str, match: str) -> None:
    plan = _plan()
    manifest = _manifest(plan)
    status = _status(manifest, plan)
    timing = _timing(plan)
    if mutation == "order":
        timing["transitions"][2], timing["transitions"][4] = (
            timing["transitions"][4], timing["transitions"][2])
    elif mutation == "hold":
        next(row for row in manifest["arms"] if row["recovery_payload"] is not None)["recovery_payload"] = None
    elif mutation == "import":
        status["arms"][0]["execution_origin"] = "verified_prior_effect"
    else:
        status["completed"] = False
    with pytest.raises(summary.ExportError, match=match):
        summary._validate_batch(manifest, plan, status, timing)


def test_timing_reentry_is_rejected() -> None:
    with pytest.raises(summary.ExportError, match="re-entered"):
        summary._observed_order({"transitions": [
            {"case_id": "case-05", "method": "a"},
            {"case_id": "case-05", "method": "b"},
            {"case_id": "case-05", "method": "a"},
        ]})


def _write(path: Path, body: dict) -> dict:
    path.write_text(json.dumps(body), encoding="utf-8")
    return io.reference(path)


def test_resource_aggregation_does_not_double_count_and_blocks_overlapping_energy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission = _write(tmp_path / "admission.json", {"request_id": "r", "effect_key": "e"})
    sampler_refs = []
    for index, start in enumerate(("2026-11-05T00:00:00+00:00", "2026-11-05T00:00:05+00:00")):
        sampler_refs.append(_write(tmp_path / f"sampler-{index}.json", {
            "schema": "periodic_device_sampling_stop_v1", "scope": "whole_device",
            "started_utc": start, "ended_utc": "2026-11-05T00:00:10+00:00",
            "device_summary": {"GPU-1": {"scope": "whole_device_including_other_loads",
                "metrics": {"memory_used_mib_peak": {"value": 5000.0}}}},
            "energy": {"devices": {"GPU-1": {"energy_wh": 1.0,
                "coverage_fraction": 1.0, "scope": "device_wide_estimate"}}},
            "process_gpu_seconds": None, "provider_gpu_seconds": None,
        }))
    raws = []
    for index, sampler_ref in enumerate(sampler_refs):
        raws.append({"phase_wall_s": 10.0, "locked_wall_s": 9.0, "owned_gpu_s": 8.0,
                    "lease_await_s": None, "simulator_sdk_launches": 2,
                    "cpu_checkpoint_launches": 1, "cpu_telemetry_launches": 1,
                    "total_owned_children": 4, "sampler_ref": sampler_ref})
    monkeypatch.setattr(summary.resources, "verify", lambda ref, **_kwargs: raws[ref["index"]])
    status = {"arms": [
        {"case_id": "case-05", "method": f"m{index}", "resource_cost": {
            "admission_ref": admission, "recovery_resource_ref": {"index": index},
            "sampler_ref": sampler_refs[index]}}
        for index in range(2)
    ]}
    arms, totals, _ = summary._resource_details(status)
    assert totals["new_sdk_launches"] == 4
    assert totals["total_owned_children"] == 8
    assert totals["device_energy_wh"] is None
    assert totals["device_energy_status"] == "unknown_overlap"
    assert arms[0]["phase_elapsed_s"] == 10.0
    assert arms[0]["locked_wall_s"] == 9.0


def test_terminal_sha_and_success_are_required(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    ref = _write(path, {"mode": "recovery_v14", "exit_code": 0, "natural_exit": True,
        "cleanup_verified": True, "supervisor_call_wall_s": 12.0, "config_sha256": "b" * 64})
    assert summary._terminal(path, ref["sha256"], expected_mode="recovery_v14")[
        "supervisor_call_wall_s"] == 12.0
    with pytest.raises(summary.ExportError, match="SHA"):
        summary._terminal(path, "0" * 64, expected_mode="recovery_v14")
    with pytest.raises(summary.ExportError, match="registered exit"):
        summary._terminal(path, ref["sha256"], expected_mode="recovery_v15")


def test_v9_namespaced_profile_is_accepted_without_changing_denominators() -> None:
    plan = _plan(schema="registered_recovery_comparison_plan_v9")
    manifest = _manifest(plan)
    status = _status(manifest, plan)
    summary._validate_batch(manifest, plan, status, _timing(plan))


def test_v10_manifest_retry_profile_preserves_denominators_and_uses_reduced_cap() -> None:
    plan = _plan(schema="registered_recovery_comparison_plan_v10")
    manifest = _manifest(plan)
    status = _status(manifest, plan)
    summary._validate_batch(manifest, plan, status, _timing(plan))
    assert plan["limits"]["gpu_sdk_launch_cap"] == 31
    assert plan["resource_remeasurement"]["accounting"]["prior_whole_study_sdk_launches"] == 34


def test_export_calls_production_base_and_preserves_measured_outcomes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    plan = _plan()
    plan_ref = _write(tmp_path / "plan.json", plan)
    job_ref = _write(tmp_path / "job.json", {"schema": "job"})
    manifest = _manifest(plan)
    manifest.update({"plan": plan_ref, "job": job_ref})
    status = _status(manifest, plan)
    status["arms"][0]["outcome"] = "recovered"
    (prepared / "final-controller-result.json").write_text(json.dumps(status), encoding="utf-8")
    terminal_ref = _write(tmp_path / "terminal.json", {
        "mode": "recovery_v14", "exit_code": 0, "natural_exit": True,
        "cleanup_verified": True, "supervisor_call_wall_s": 20.0, "config_sha256": "c" * 64})
    calls = {"base": 0}

    monkeypatch.setattr(summary.controller, "_load", lambda _root: manifest)
    monkeypatch.setattr(summary.controller, "verify_prepared_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(summary.controller, "status", lambda _root: status)
    monkeypatch.setattr(summary.timing_evidence, "read_timing", lambda _root: _timing(plan))
    monkeypatch.setattr(summary, "_resource_details", lambda _status: (
        [], {"measured_arms": 15, "new_sdk_launches": 30,
             "cpu_checkpoint_launches": 15, "cpu_telemetry_launches": 15,
             "total_owned_children": 60, "device_energy_wh": None,
             "device_energy_status": "unknown_missing",
             "nested_costs_are_not_added_to_outer_wall": True,
             "kernel_time_s": None, "provider_cost": None, "human_time_s": None,
             "billing_cost": None}, []))

    def fake_base(_prepared: Path, out: Path) -> dict:
        calls["base"] += 1
        (out / "private").mkdir(parents=True)
        (out / "private" / "source-refs.json").write_text(
            json.dumps({"refs": []}), encoding="utf-8")
        return {"arms": [
            {"recovery_outcome": "recovered" if index == 0 else (
                "hold" if index == 15 else "not_recovered")}
            for index in range(16)]}

    monkeypatch.setattr(summary.base, "summarize", fake_base)
    result = summary.export(prepared, tmp_path / "out",
        controller_sha=summary._lf_sha(summary.ROOT / "evals/run_registered_recovery_comparison_v1.py"),
        terminal=Path(terminal_ref["path"]), terminal_sha=terminal_ref["sha256"])
    assert calls["base"] == 1
    assert result["outcomes"]["recovered"] == 1
    assert result["outcomes"]["not_recovered"] == 14
    assert result["sdk_accounting"]["combined_launches"] == 63
    assert result["outer_supervisor"]["supervisor_call_wall_s"] == 20.0
