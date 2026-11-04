"""Export a verified, path-free resource summary for the V14 balanced rerun."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench"), str(ROOT / "evals")]

from evals import registered_recovery_resources_v1 as resources
from evals import registered_recovery_timing_v1 as timing_evidence
from evals import run_registered_recovery_comparison_v1 as controller
from evals import summarize_registered_recovery_comparison_v1 as base
from rl_triage import investigation_session as io

SCHEMA = "balanced_recovery_resource_summary_v1"
PLAN_SCHEMA = controller.PLAN_SCHEMA_V8
PROFILE = controller.RESOURCE_REMEASUREMENT_PROFILE
EXPECTED_HOLD = ("case-07", "iterative_agent")


class ExportError(ValueError):
    """A registered input or measured result differs from the V14 contract."""


def _lf_sha(path: Path) -> str:
    return controller._lf_sha(path)


def _read_ref(ref: object, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ExportError(f"{label} ref fields differ")
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_file() or io.file_sha(path) != ref["sha256"]:
        raise ExportError(f"{label} ref changed")
    body = io.read_ref(ref)
    if not isinstance(body, dict):
        raise ExportError(f"{label} body differs")
    return path, body


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExportError(f"{label} differs")
    return float(value)


def _assert_public(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "path" in str(key).casefold():
                raise ExportError("public result contains a path field")
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if ("bench/private" in normalized.casefold() or PurePosixPath(value).is_absolute()
                or PureWindowsPath(value).is_absolute()):
            raise ExportError("public result contains a filesystem path")


def _expected_order(plan: dict[str, Any]) -> list[tuple[str, str]]:
    return [(case["case_id"], method) for case in plan["cases"] for method in case["rotation"]]


def _observed_order(timing: dict[str, Any]) -> list[tuple[str, str]]:
    compressed: list[tuple[str, str]] = []
    for row in timing["transitions"]:
        key = (row["case_id"], row["method"])
        if not compressed or compressed[-1] != key:
            if key in compressed:
                raise ExportError("an arm re-entered after another arm")
            compressed.append(key)
    return compressed


def _terminal(path: Path, expected_sha: str) -> dict[str, Any]:
    terminal = Path(path).resolve(strict=True)
    if io.file_sha(terminal) != expected_sha:
        raise ExportError("terminal receipt SHA differs")
    body = json.loads(terminal.read_text(encoding="utf-8"))
    required = {"mode", "exit_code", "natural_exit", "cleanup_verified",
                "supervisor_call_wall_s", "config_sha256"}
    if (not required <= set(body) or body["mode"] != "recovery_v14" or body["exit_code"] != 0
            or body["natural_exit"] is not True or body["cleanup_verified"] is not True):
        raise ExportError("terminal supervisor receipt is not a successful V14 exit")
    wall = _number(body["supervisor_call_wall_s"], "supervisor call wall")
    if wall < 0:
        raise ExportError("supervisor call wall differs")
    return {"sha256": expected_sha, "supervisor_call_wall_s": wall,
            "scope": "outer_supervisor_call_only_not_added_to_nested_owned_costs"}


def _device_record(sampler: dict[str, Any]) -> tuple[dict[str, Any], tuple[datetime, datetime] | None,
                                                       float | None]:
    if sampler.get("schema") != "periodic_device_sampling_stop_v1":
        raise ExportError("device sampler schema differs")
    device_summary = deepcopy(sampler.get("device_summary"))
    energy = deepcopy(sampler.get("energy"))
    if not isinstance(device_summary, dict) or not isinstance(energy, dict):
        raise ExportError("device sampler metrics differ")
    start, end = sampler.get("started_utc"), sampler.get("ended_utc")
    window = None
    if isinstance(start, str) and isinstance(end, str):
        parsed = (datetime.fromisoformat(start), datetime.fromisoformat(end))
        if parsed[0].tzinfo is None or parsed[1].tzinfo is None or parsed[1] < parsed[0]:
            raise ExportError("device sampler UTC window differs")
        window = parsed
    devices = energy.get("devices")
    energy_wh = None
    if isinstance(devices, dict) and devices:
        values = [row.get("energy_wh") for row in devices.values() if isinstance(row, dict)]
        if len(values) == len(devices) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                               and math.isfinite(v) and v >= 0 for v in values):
            energy_wh = sum(_number(value, "device energy") for value in values)
    return ({"scope": sampler.get("scope"), "device_summary": device_summary,
             "energy": energy, "process_gpu_seconds": sampler.get("process_gpu_seconds"),
             "provider_gpu_seconds": sampler.get("provider_gpu_seconds")}, window, energy_wh)


def _resource_details(status: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    arms: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    windows: list[tuple[datetime, datetime]] = []
    energies: list[float] = []
    sdk = cpu_checkpoint = cpu_telemetry = owned = 0
    for row in status["arms"]:
        key = f"{row['case_id']}:{row['method']}"
        projected = row.get("resource_cost")
        if projected is None:
            arms.append({"case_id": row["case_id"], "method": row["method"],
                         "resource_measured": False, "device": None})
            continue
        admission = io.read_ref(projected["admission_ref"])
        raw = resources.verify(projected["recovery_resource_ref"],
                               request_id=admission["request_id"], effect_key=admission["effect_key"])
        if projected.get("sampler_ref") != raw["sampler_ref"]:
            raise ExportError("projected and raw sampler refs differ")
        device = None
        if raw["sampler_ref"] is not None:
            _, sampler = _read_ref(raw["sampler_ref"], key + " sampler")
            device, window, energy = _device_record(sampler)
            private.append({"logical_id": key + ":device-sampler", **raw["sampler_ref"]})
            if window is not None:
                windows.append(window)
            if energy is not None:
                energies.append(energy)
        sdk += raw["simulator_sdk_launches"]
        cpu_checkpoint += raw["cpu_checkpoint_launches"]
        cpu_telemetry += raw["cpu_telemetry_launches"]
        owned += raw["total_owned_children"]
        arms.append({"case_id": row["case_id"], "method": row["method"],
                     "resource_measured": True, "device": device,
                     "phase_elapsed_s": raw["phase_wall_s"], "locked_wall_s": raw["locked_wall_s"],
                     "simulator_owned_wall_s": raw["owned_gpu_s"],
                     "lease_wait_s": raw["lease_await_s"],
                     "sdk_launches": raw["simulator_sdk_launches"],
                     "cpu_checkpoint_launches": raw["cpu_checkpoint_launches"],
                     "cpu_telemetry_launches": raw["cpu_telemetry_launches"],
                     "total_owned_children": raw["total_owned_children"]})
    overlap = any(start_b < end_a and start_a < end_b
                  for index, (start_a, end_a) in enumerate(windows)
                  for start_b, end_b in windows[index + 1:])
    measured = sum(row["resource_measured"] for row in arms)
    energy_total = sum(energies) if not overlap and len(energies) == measured else None
    totals = {"measured_arms": measured, "new_sdk_launches": sdk,
              "cpu_checkpoint_launches": cpu_checkpoint,
              "cpu_telemetry_launches": cpu_telemetry, "total_owned_children": owned,
              "device_energy_wh": energy_total,
              "device_energy_status": "unknown_overlap" if overlap else (
                  "complete" if energy_total is not None else "unknown_missing"),
              "nested_costs_are_not_added_to_outer_wall": True,
              "kernel_time_s": None, "provider_cost": None, "human_time_s": None,
              "billing_cost": None}
    return arms, totals, private


def _prep_refs(values: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    public, private = [], []
    for value in values:
        logical, separator, encoded = value.partition("=")
        path_text, marker, expected = encoded.rpartition("#")
        if not separator or not marker or not logical or not path_text:
            raise ExportError("preparation ref must be logical=path#sha256")
        path = Path(path_text).resolve(strict=True)
        if io.file_sha(path) != expected:
            raise ExportError("preparation ref changed")
        public.append({"logical_id": logical, "sha256": expected})
        private.append({"logical_id": logical, "path": str(path), "sha256": expected})
    return public, private


def _validate_batch(manifest: dict[str, Any], plan: dict[str, Any], status: dict[str, Any],
                    timing: dict[str, Any]) -> None:
    if (plan.get("schema") != PLAN_SCHEMA
            or plan.get("resource_remeasurement", {}).get("profile_id") != PROFILE
            or plan["resource_remeasurement"].get("imports_allowed") is not False
            or plan["resource_remeasurement"].get("no_model_requery") is not True):
        raise ExportError("balanced resource plan differs")
    if not status["completed"] or status["halted"] or status["completed_arms"] != 16:
        raise ExportError("balanced run is not a complete successful controller batch")
    if status["api_calls"] != 0 or status["limits"] != plan["limits"]:
        raise ExportError("balanced run limits differ")
    holds = [(row["case_id"], row["method"]) for row in manifest["arms"]
             if row["recovery_payload"] is None]
    if holds != [EXPECTED_HOLD]:
        raise ExportError("exact diagnostic HOLD arm differs")
    if sum(row.get("resource_cost") is not None for row in status["arms"]) != 15:
        raise ExportError("fresh attempted-arm denominator differs")
    if any(row.get("execution_origin") == "verified_prior_effect" for row in status["arms"]):
        raise ExportError("V14 must not import completed effects")
    if _observed_order(timing) != _expected_order(plan):
        raise ExportError("actual transition order differs from registered rotations")


def export(prepared: Path, out: Path, *, controller_sha: str, terminal: Path,
           terminal_sha: str, preparation_refs: list[str] | None = None) -> dict[str, Any]:
    root = Path(prepared).resolve(strict=True)
    target = Path(out).resolve()
    if target.exists():
        raise FileExistsError("balanced export output must be fresh")
    controller_path = ROOT / "evals" / "run_registered_recovery_comparison_v1.py"
    if _lf_sha(controller_path) != controller_sha:
        raise ExportError("controller source SHA differs")
    manifest = controller._load(root)
    canonical = controller.verify_prepared_manifest(
        root, repo_root=ROOT, registration_anchor=manifest["registration_anchor"])
    if canonical != manifest:
        raise ExportError("canonical prepared manifest differs")
    _, plan = _read_ref(manifest["plan"], "registered plan")
    status = controller.status(root)
    final_path = root / "final-controller-result.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final != status:
        raise ExportError("final controller snapshot differs from current status")
    timing = timing_evidence.read_timing(root)
    _validate_batch(manifest, plan, status, timing)
    terminal_public = _terminal(terminal, terminal_sha)
    staging = target.parent / f".{target.name}.base-{uuid.uuid4().hex}"
    try:
        base_public = base.summarize(root, staging)
        base_private = json.loads((staging / "private" / "source-refs.json").read_text(encoding="utf-8"))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    resource_arms, totals, private_resources = _resource_details(status)
    if totals["new_sdk_launches"] > 32 or totals["total_owned_children"] > 64:
        raise ExportError("measured launch count exceeds the registered cap")
    prior = plan["resource_remeasurement"]["accounting"]
    cumulative = (prior["prior_whole_study_sdk_launches"] + totals["new_sdk_launches"])
    if cumulative > prior["cumulative_gpu_sdk_launch_cap"]:
        raise ExportError("measured cumulative SDK count exceeds the registered cap")
    prep_public, prep_private = _prep_refs(preparation_refs or [])
    outcomes = {name: sum(row["recovery_outcome"] == name for row in base_public["arms"])
                for name in base.OUTCOMES}
    diagnostic_inputs = [row for row in plan["inputs"] if row.get("role") == "diagnostic_plan"]
    if len(diagnostic_inputs) != 1:
        raise ExportError("one original diagnostic plan source is required")
    known_primary = plan["resource_remeasurement"]["known_primary_outcomes"]
    public = {"schema": SCHEMA, "status": "complete", "planned_arms": 16,
              "attempted_arms": 15, "diagnostic_hold_arms": 1, "imported_arms": 0,
              "outcomes": outcomes, "arms": resource_arms, "resources": totals,
              "outer_supervisor": terminal_public,
              "preparation_measurements": prep_public or None,
              "sdk_accounting": {"prior_primary_and_post_launches": prior["prior_whole_study_sdk_launches"],
                                  "new_launches": totals["new_sdk_launches"],
                                  "combined_launches": cumulative,
                                  "combined_only_after_new_count_verification": True},
              "base_summary": base_public,
              "sources": [{"logical_id": "registered-controller", "sha256": controller_sha},
                          {"logical_id": "supervisor-terminal", "sha256": terminal_sha},
                          {"logical_id": "registered-plan-raw", "sha256": manifest["plan"]["sha256"]},
                          {"logical_id": "original-diagnostic-plan",
                           "sha256": diagnostic_inputs[0]["sha256"]},
                          {"logical_id": "known-primary-summary",
                           "sha256": known_primary["sha256"]},
                          {"logical_id": "registered-job", "sha256": manifest["job"]["sha256"]},
                          {"logical_id": "final-controller-result", "sha256": io.file_sha(final_path)}],
              "api_calls": 0, "unknown_values_are_not_zero": True}
    _assert_public(public)
    target.mkdir(parents=True)
    io.write_new(target / "public" / "summary.json", public)
    refs = list(base_private["refs"]) + private_resources + prep_private + [
        {"logical_id": "supervisor-terminal", "path": str(Path(terminal).resolve()), "sha256": terminal_sha},
        {"logical_id": "final-controller-result", "path": str(final_path), "sha256": io.file_sha(final_path)},
    ]
    io.write_new(target / "private" / "source-refs.json", {
        "schema": "balanced_recovery_resource_private_refs_v1", "prepared": str(root), "refs": refs})
    (target / "RESULT.md").write_text(
        "# Balanced recovery resource export\n\n"
        f"- Planned / attempted / HOLD: 16 / 15 / 1\n- New SDK launches: {totals['new_sdk_launches']}\n"
        f"- Outcomes: {json.dumps(outcomes, sort_keys=True)}\n",
        encoding="utf-8", newline="\n")
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--controller-sha", required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--terminal-sha", required=True)
    parser.add_argument("--preparation-ref", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = export(args.prepared, args.out, controller_sha=args.controller_sha,
                        terminal=args.terminal, terminal_sha=args.terminal_sha,
                        preparation_refs=args.preparation_ref)
    except Exception as exc:  # noqa: BLE001 - deterministic CLI rejection.
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error_type": type(exc).__name__, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema": SCHEMA, "status": "written",
                      "new_sdk_launches": result["resources"]["new_sdk_launches"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
