"""Join the sealed balanced prefix and its registered continuation once."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from evals import summarize_balanced_recovery_resources_v1 as E

ROOT = E.ROOT
PLAN_SCHEMA = "registered_recovery_comparison_plan_v11"
PROFILE = "balanced_resource_continuation_v1"
SEAL = "bench/protocols/registered_recovery_v16_prefix_20261105_v1.json"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _guard(value: Path | str) -> Path:
    path = Path(value)
    if "ri_codex2_tokenfactory_live_160" in str(path).replace("\\", "/").casefold().split("/"):
        raise E.ExportError("forbidden source subtree")
    return path


def _read(ref: dict, label: str) -> tuple[Path, dict]:
    _guard(ref["path"])
    return E._read_ref(ref, label)


def _prefix(manifest: dict, plan: dict, terminal: Path, terminal_sha: str) -> tuple[dict, dict]:
    seal_path = ROOT / SEAL
    if plan["source_closure"].get(SEAL) != E._lf_sha(seal_path):
        raise E.ExportError("prior timing seal is not source-bound")
    seal = json.loads(seal_path.read_bytes())
    effects = [a for a in manifest["arms"]
               if (a.get("continuation_import") or {}).get("kind") == "completed_effect"]
    if len(effects) != 7:
        raise E.ExportError("seven historical effects required")
    _, proof = _read(effects[0]["continuation_import"]["proof"], "completed prefix proof")
    old_manifest_path, old_manifest = _read(proof["refs"]["prepared_manifest"], "prior manifest")
    old_root = old_manifest_path.parent
    old_final_path = old_root / "final-controller-result.json"
    if (E.io.file_sha(old_manifest_path) != seal["prior_manifest_sha256"]
            or E.io.file_sha(old_final_path) != seal["prior_controller_result_sha256"]
            or old_manifest["plan"]["sha256"] != seal["prior_plan_sha256"]
            or plan["continuation"]["prior_plan_sha256"] != seal["prior_plan_sha256"]):
        raise E.ExportError("prior batch identity differs from timing seal")
    terminal = _guard(terminal).resolve(strict=True)
    if E.io.file_sha(terminal) != terminal_sha or terminal_sha != seal["prior_terminal_sha256"]:
        raise E.ExportError("prior terminal differs from timing seal")
    old_terminal = json.loads(terminal.read_bytes())
    if (old_terminal.get("mode") != "recovery_v16" or old_terminal.get("exit_code") != 1
            or old_terminal.get("natural_exit") is not True
            or old_terminal.get("cleanup_verified") is not True):
        raise E.ExportError("prior interrupted parent is not closed")
    timing = E.timing_evidence.read_timing(old_root)
    observed = {Path(r["path"]).name: r["sha256"] for r in timing["private_refs"]}
    if observed != seal["timing_file_sha256"]:
        raise E.ExportError("historical timing bytes changed")
    order = E._observed_order(timing)
    if [list(pair) for pair in order] != seal["order_prefix"] or order != E._expected_order(plan)[:8]:
        raise E.ExportError("historical prefix order differs")
    return timing, old_terminal


def _join_order(plan: dict, prior: dict, current: dict) -> list[tuple[str, str]]:
    expected = E._expected_order(plan)
    current_expected = [key for key in expected[7:] if key != E.EXPECTED_HOLD]
    if (E._observed_order(prior) != expected[:8]
            or E._observed_order(current) != current_expected):
        raise E.ExportError("continuation did not begin at the interrupted arm")
    return expected


def _validate(manifest: dict, plan: dict, status: dict) -> None:
    continuation = plan.get("continuation")
    if (plan.get("schema") != PLAN_SCHEMA
            or not isinstance(continuation, dict)
            or continuation.get("profile_id") != PROFILE
            or continuation.get("no_model_requery") is not True
            or status.get("completed") is not True or status.get("halted") is not False
            or status.get("completed_arms") != 16 or status.get("api_calls") != 0
            or status.get("limits") != plan.get("limits")):
        raise E.ExportError("registered continuation is not complete")
    imports = [r for r in manifest["arms"] if r.get("continuation_import") is not None]
    kinds = [r["continuation_import"]["kind"] for r in imports]
    if kinds.count("completed_effect") != 7 or kinds.count("completed_training") != 1 or len(kinds) != 8:
        raise E.ExportError("continuation imports differ")
    holds = [(r["case_id"], r["method"]) for r in manifest["arms"]
             if r.get("recovery_payload") is None and r.get("continuation_import") is None]
    if holds != [E.EXPECTED_HOLD]:
        raise E.ExportError("original diagnostic HOLD differs")


def _resources(status: dict) -> tuple[list[dict], dict, list[dict]]:
    rows: list[dict] = []
    refs: list[dict] = []
    windows: list[tuple[datetime, datetime]] = []
    energies: list[float | None] = []
    totals: dict[str, Any] = {"new_sdk_launches": 0, "historical_sdk_launches": 0,
              "cpu_checkpoint_launches": 0, "cpu_telemetry_launches": 0,
              "total_owned_children": 0, "historical_completed_effects": 0,
              "historical_completed_training": 0, "current_dispatches": 0}
    seen: set[str] = set()
    seen_samplers: set[str] = set()
    for arm in status["arms"]:
        projected = arm.get("resource_cost")
        row = {"case_id": arm["case_id"], "method": arm["method"], "segments": []}
        if projected is None:
            rows.append(row)
            continue
        imported = projected.get("schema") == "registered_recovery_imported_resource_v1"
        if imported and set(projected) != {
            "schema", "case_id", "method", "plan_sha256", "status", "execution_origin",
            "prior_effect_proof_ref", "prior_resource_ref", "prior_admission_ref",
            "historical_recovery_folder", "new_dispatch_costs", "historical_effect_costs",
            "receipt_ref",
        }:
            raise E.ExportError("imported resource projection differs")
        chunks = [("historical_completed_effect", projected["prior_resource_ref"])] if imported else [
            ("current_dispatch", projected["recovery_resource_ref"])]
        if not imported:
            _, raw = _read(projected["recovery_resource_ref"], "current resource")
            if raw["raw_receipt_ref"] is not None:
                _, receipt = _read(raw["raw_receipt_ref"], "current recovery receipt")
                reuse = receipt.get("training_reuse")
                if reuse is not None:
                    if (not isinstance(reuse, dict)
                            or reuse.get("new_training_launched") is not False
                            or not isinstance(reuse.get("prior_dispatch_resource_ref"), dict)):
                        raise E.ExportError("completed training reuse evidence differs")
                    chunks.insert(0, ("historical_completed_training", reuse["prior_dispatch_resource_ref"]))
        for kind, ref in chunks:
            if ref["sha256"] in seen:
                raise E.ExportError("resource evidence was counted twice")
            seen.add(ref["sha256"])
            _, body = _read(ref, "resource evidence")
            raw = E.resources.verify(ref, request_id=body["request_id"], effect_key=body["effect_key"])
            current = kind == "current_dispatch"
            if kind == "historical_completed_effect":
                totals["historical_completed_effects"] += 1
            elif kind == "historical_completed_training":
                totals["historical_completed_training"] += 1
            else:
                totals["current_dispatches"] += 1
            totals["new_sdk_launches" if current else "historical_sdk_launches"] += raw["simulator_sdk_launches"]
            if current:
                for key in ("cpu_checkpoint_launches", "cpu_telemetry_launches", "total_owned_children"):
                    totals[key] += raw[key]
            refs.append({"logical_id": f"{arm['case_id']}:{arm['method']}:{kind}", **ref})
            device = None
            if raw["sampler_ref"] is not None:
                if raw["sampler_ref"]["sha256"] in seen_samplers:
                    raise E.ExportError("device sampler was counted twice")
                seen_samplers.add(raw["sampler_ref"]["sha256"])
                _, sampler = _read(raw["sampler_ref"], "device sampler")
                device, window, energy = E._device_record(sampler)
                refs.append({"logical_id": f"{arm['case_id']}:{arm['method']}:{kind}:sampler", **raw["sampler_ref"]})
                if window is not None:
                    windows.append(window)
                energies.append(energy)
            else:
                energies.append(None)
            row["segments"].append({"scope": kind, "SDK": raw["simulator_sdk_launches"],
                "phase_wall_s": raw["phase_wall_s"], "nested_owned_gpu_s": raw["owned_gpu_s"],
                "device": device, "unmeasured_lease_wait_s": raw["lease_await_s"]})
        rows.append(row)
    overlap = any(a < d and c < b for i, (a, b) in enumerate(windows) for c, d in windows[i+1:])
    missing_energy = any(value is None for value in energies)
    energy_total = (None if overlap or missing_energy
                    else math.fsum(value for value in energies if value is not None))
    totals.update(combined_balanced_sdk_launches=totals["new_sdk_launches"] + totals["historical_sdk_launches"],
                  device_energy_wh=energy_total,
                  device_windows_overlap=overlap,
                  device_energy_status=("unknown_overlap" if overlap else
                                        "unknown_missing" if missing_energy else "complete"),
                  device_energy_scope="whole_device_recorded_windows_not_process",
                  kernel_time_s=None, provider_billing_usd=None, complete_child_CPU_s=None,
                  nested_costs_not_added_again=True)
    return rows, totals, refs


def _validate_accounting(plan: dict, totals: dict) -> int:
    prior = plan["continuation"]
    cumulative = prior["prior_whole_study_sdk_launches"] + totals["new_sdk_launches"]
    if (totals["historical_sdk_launches"] != 15 or totals["new_sdk_launches"] != 15
            or totals["historical_completed_effects"] != 7
            or totals["historical_completed_training"] != 1
            or totals["current_dispatches"] != 8
            or totals["cpu_checkpoint_launches"] != plan["limits"]["cpu_checkpoint_launch_cap"]
            or totals["cpu_telemetry_launches"] != plan["limits"]["cpu_telemetry_launch_cap"]
            or totals["new_sdk_launches"] > plan["limits"]["gpu_sdk_launch_cap"]
            or totals["total_owned_children"] > plan["limits"]["total_owned_launch_cap"]
            or cumulative > prior["cumulative_gpu_sdk_launch_cap"]):
        raise E.ExportError("actual continuation accounting differs")
    return cumulative


def export(prepared: Path, out: Path, *, controller_sha: str, terminal: Path, terminal_sha: str,
           prior_terminal: Path, prior_terminal_sha: str) -> dict[str, Any]:
    root, target = _guard(prepared).resolve(strict=True), _guard(out).resolve()
    if target.exists():
        raise FileExistsError("continuation export must be fresh")
    if E._lf_sha(ROOT / "evals/run_registered_recovery_comparison_v1.py") != controller_sha:
        raise E.ExportError("controller source differs")
    manifest = E.controller._load(root)
    if E.controller.verify_prepared_manifest(root, repo_root=ROOT,
            registration_anchor=manifest["registration_anchor"]) != manifest:
        raise E.ExportError("prepared continuation is not canonical")
    _, plan = _read(manifest["plan"], "registered continuation")
    status = E.controller.status(root)
    final_path = root / "final-controller-result.json"
    if json.loads(final_path.read_bytes()) != status:
        raise E.ExportError("final controller snapshot differs")
    _validate(manifest, plan, status)
    prior_timing, old_terminal = _prefix(manifest, plan, prior_terminal, prior_terminal_sha)
    current_timing = E.timing_evidence.read_timing(root)
    order = _join_order(plan, prior_timing, current_timing)
    current_terminal = E._terminal(_guard(terminal), terminal_sha, expected_mode="recovery_v17")
    current_raw = json.loads(_guard(terminal).read_bytes())
    arms, totals, refs = _resources(status)
    prior = plan["continuation"]
    cumulative = _validate_accounting(plan, totals)
    staging = target.parent / f".{target.name}.base-{uuid.uuid4().hex}"
    try:
        base = E.base.summarize(root, staging)
        base_private = json.loads(
            (staging / "private/source-refs.json").read_text(encoding="utf-8")
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    resume_gap = (_utc(current_raw["supervisor_call_started_utc"])
                  - _utc(old_terminal["finished_utc"])).total_seconds()
    if resume_gap < 0:
        raise E.ExportError("parent phases overlap")
    public = {"schema": "balanced_recovery_resource_continuation_summary_v1", "status": "complete",
        "development_remeasurement": True, "new_heldout_claim": False, "planned_arms": 16,
        "attempted_arms": 15, "diagnostic_HOLD": 1, "imported_completed_effects": 7,
        "imported_completed_training": 1, "fresh_effects": 7, "api_calls": 0,
        "actual_order": [list(p) for p in order], "actual_order_matches_registered": True,
        "order_scope": "logical_arm_order_including_original_unexecuted_diagnostic_HOLD",
        "observed_prior_transition_order": [list(p) for p in E._observed_order(prior_timing)],
        "observed_new_transition_order": [list(p) for p in E._observed_order(current_timing)],
        "original_diagnostic_HOLD_has_no_GPU_transition": True,
        "arms": arms, "resources": totals, "base_summary": base,
        "outcomes": {name: sum(a["recovery_outcome"] == name for a in base["arms"]) for name in E.base.OUTCOMES},
        "sdk_accounting": {"prior_recovery_study": prior["prior_whole_study_sdk_launches"],
            "new": totals["new_sdk_launches"], "cumulative": cumulative,
            "cap": prior["cumulative_gpu_sdk_launch_cap"], "historical_not_added_twice": True},
        "outer_supervisor": current_terminal,
        "elapsed": {"prior_supervisor_wall_s": old_terminal["supervisor_call_wall_s"],
            "new_supervisor_wall_s": current_raw["supervisor_call_wall_s"], "resume_gap_wall_s": resume_gap,
            "observed_calendar_span_s": (_utc(current_raw["finished_utc"])
                                           - _utc(old_terminal["supervisor_call_started_utc"])).total_seconds(),
            "gap_includes_preparation_not_added_again": True, "continuous_per_arm_E2E_s": None,
            "complete_campaign_wall_s": None},
        "sources": [{"logical_id": "prior_terminal", "sha256": prior_terminal_sha},
                    {"logical_id": "terminal", "sha256": terminal_sha},
                    {"logical_id": "registered_plan", "sha256": manifest["plan"]["sha256"]},
                    {"logical_id": "prior_timing_seal", "sha256": E.io.file_sha(ROOT / SEAL)}]}
    E._assert_public(public)
    target.mkdir(parents=True)
    (target / "public").mkdir()
    (target / "public/summary.json").write_bytes((json.dumps(public, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
    (target / "private").mkdir()
    timing_refs = [
        {"logical_id": f"prior-timing:{Path(ref['path']).name}", **ref}
        for ref in prior_timing["private_refs"]
    ]
    (target / "private/source-refs.json").write_bytes((json.dumps(
        {"refs": [*base_private["refs"], *refs, *timing_refs]},
        ensure_ascii=False, indent=2) + "\n").encode())
    return public


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("prepared", "out", "terminal", "prior-terminal"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("controller-sha", "terminal-sha", "prior-terminal-sha"):
        parser.add_argument("--"+name, required=True)
    args = vars(parser.parse_args())
    prepared, out = args.pop("prepared"), args.pop("out")
    result = export(prepared, out, **args)
    print(json.dumps({"status": result["status"], "outcomes": result["outcomes"],
                      "SDK": result["resources"]["new_sdk_launches"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
