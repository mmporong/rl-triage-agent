"""Reverify and export a path-free public summary of the registered recovery comparison."""
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), *(str(ROOT / name) for name in ("src", "bench", "evals"))]

from evals import registered_recovery_resources_v1 as resource_evidence
from evals import registered_recovery_timing_v1 as recovery_timing
from evals import run_registered_recovery_comparison_v1 as controller
from rl_triage import investigation_recovery
from rl_triage import investigation_session as io

SCHEMA = "registered_recovery_comparison_public_summary_v1"
METHODS = controller.METHODS
CASES = controller.SUPPORTED_CASES
OUTCOMES = ("recovered", "not_recovered", "unknown", "hold", "unsupported",
            "normal_or_host_gate", "infrastructure_failure", "not_started")


class SummaryError(ValueError):
    """The prepared comparison or a public export invariant differs."""


def _write_new(path: Path, value: dict) -> None:
    io.write_new(path, value)


def _generic_ref(ref: object, logical_id: str, private: list[dict]) -> dict:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise SummaryError(f"{logical_id} ref fields differ")
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_file() or io.file_sha(path) != ref["sha256"]:
        raise SummaryError(f"{logical_id} ref changed")
    private.append({"logical_id": logical_id, "path": str(path), "sha256": ref["sha256"]})
    return {"logical_id": logical_id, "sha256": ref["sha256"]}


def _execution_wall(ref: object, logical_id: str, private: list[dict]) -> tuple[float | None, dict | None]:
    if ref is None:
        return None, None
    public = _generic_ref(ref, logical_id, private)
    if not isinstance(ref, dict):
        raise TypeError("execution reference must be an object")
    body = io.read_ref(ref)
    value = body.get("wall_s", body.get("execution_wall_s"))
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        return None, public
    return float(value), public


def _resource(row: dict | None, logical_prefix: str, private: list[dict]) -> tuple[dict, list[dict]]:
    if row is None:
        return {"status": "not_applicable", "phase_elapsed_s": None, "locked_wall_s": None,
                "simulator_owned_wall_s": None, "checkpoint_cpu_owned_wall_s": None,
                "telemetry_cpu_owned_wall_s": None,
                "lease_wait_s": None, "api_calls": 0, "device_energy": None,
                "owned_cost_complete": None}, []
    projected_ref = row.get("receipt_ref")
    public_refs = [_generic_ref(projected_ref, logical_prefix + ":projected-resource", private)]
    admission = io.read_ref(row["admission_ref"])
    raw = resource_evidence.verify(row["recovery_resource_ref"],
                                   request_id=admission["request_id"], effect_key=admission["effect_key"])
    public_refs.extend([
        _generic_ref(row["recovery_resource_ref"], logical_prefix + ":raw-resource", private),
        _generic_ref(row["admission_ref"], logical_prefix + ":lease-admission", private),
    ])
    checkpoint_wall, checkpoint_ref = _execution_wall(
        raw["execution_refs"]["checkpoint"], logical_prefix + ":checkpoint-execution", private)
    if checkpoint_ref is not None:
        public_refs.append(checkpoint_ref)
    telemetry_wall, telemetry_ref = _execution_wall(
        raw["execution_refs"].get("telemetry"), logical_prefix + ":telemetry-execution", private)
    if telemetry_ref is not None:
        public_refs.append(telemetry_ref)
    for name in ("training", "fixed"):
        _, execution_ref = _execution_wall(raw["execution_refs"][name],
                                           logical_prefix + f":{name}-execution", private)
        if execution_ref is not None:
            public_refs.append(execution_ref)
    device_energy = None
    if raw["sampler_ref"] is not None:
        public_refs.append(_generic_ref(raw["sampler_ref"], logical_prefix + ":device-sampler", private))
        sampler = io.read_ref(raw["sampler_ref"])
        device_energy = {"scope": sampler.get("scope"), "energy": deepcopy(sampler.get("energy")),
                         "process_gpu_seconds": sampler.get("process_gpu_seconds"),
                         "provider_gpu_seconds": sampler.get("provider_gpu_seconds")}
    return {"status": raw["status"], "phase_elapsed_s": float(raw["phase_wall_s"]),
            "locked_wall_s": float(raw["locked_wall_s"]),
            "simulator_owned_wall_s": raw["owned_gpu_s"],
            "checkpoint_cpu_owned_wall_s": checkpoint_wall,
            "telemetry_cpu_owned_wall_s": telemetry_wall,
            "lease_wait_s": raw["lease_await_s"], "api_calls": 0,
            "device_energy": device_energy,
            "owned_cost_complete": bool(raw["owned_gpu_s_complete"]
                                        and (raw["cpu_checkpoint_launches"] == 0
                                             or checkpoint_wall is not None)
                                        and (raw.get("cpu_telemetry_launches", 0) == 0
                                             or telemetry_wall is not None))}, public_refs


def _cost_aggregate(arms: list[dict], field: str) -> dict:
    attempted = [row for row in arms if row["recovery_attempted"]]
    values = [row["costs"][field] for row in attempted]
    known = [float(value) for value in values
             if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
    return {"known_sum": sum(known) if known else None, "known_arms": len(known),
            "unknown_arms": len(attempted) - len(known)}


def _method_summary(method: str, arms: list[dict]) -> dict:
    rows = [row for row in arms if row["method"] == method]
    outcomes = {name: sum(row["recovery_outcome"] == name for row in rows) for name in OUTCOMES}
    if sum(outcomes.values()) != len(rows):
        raise SummaryError("method outcomes do not cover the planned denominator")
    return {"method": method, "planned_arms": len(rows),
            "diagnostic_confirmed": sum(row["diagnostic_confirmed"] for row in rows),
            "recovery_attempted": sum(row["recovery_attempted"] for row in rows),
            "outcomes": outcomes,
            "costs": {field: _cost_aggregate(rows, field) for field in (
                "phase_elapsed_s", "locked_wall_s", "simulator_owned_wall_s",
                "checkpoint_cpu_owned_wall_s", "telemetry_cpu_owned_wall_s", "lease_wait_s")},
            "api_calls": sum(row["costs"]["api_calls"] for row in rows)}


def _assert_public(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            private_text = isinstance(child, str) and "bench/private" in child.replace("\\", "/").casefold()
            absolute = isinstance(child, str) and (PurePosixPath(child).is_absolute()
                                                   or PureWindowsPath(child).is_absolute())
            if "path" in str(key).casefold() or private_text or absolute:
                raise SummaryError("public summary contains a filesystem path")
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)


def _post_review(recorded: object, manifest: dict, recovery_folder: Path, request_id: str,
                 logical: str, private_refs: list[dict]) -> tuple[dict, dict]:
    if not isinstance(recorded, dict):
        return {"status": "not_recorded", "cpu_review_wall_s": None}, {}
    plan = io.read_ref(manifest["plan"])
    policies = [row for row in plan["inputs"] if row.get("role") == "recovery_review_policy"]
    if len(policies) != 1:
        raise SummaryError("one registered recovery review policy required")
    policy = policies[0]
    policy_path = ROOT / policy["path"]
    if controller._lf_sha(policy_path) != policy["sha256"]:
        raise SummaryError("recovery review policy changed")
    policy_ref = io.reference(policy_path)
    evidence = _generic_ref(policy_ref, logical + ":recovery-review-policy", private_refs)
    fresh = investigation_recovery.review(recovery_folder, request_id, {
        "schema": "intervention_host_gate_source_v1", "receipt": policy_ref})
    stable = {"schema", "decision", "behavior_outcome", "confirmation_scope",
              "confirmation_catalog_complete", "unobserved_components", "api_calls", "gpu_launches"}
    if any(recorded.get(key) != fresh.get(key) for key in stable):
        raise SummaryError("recorded post review differs from reverified review")
    wall = recorded.get("cpu_review_wall_s")
    if (isinstance(wall, bool) or not isinstance(wall, (int, float))
            or not math.isfinite(wall) or wall < 0):
        raise SummaryError("post review CPU wall differs")
    decision = fresh["decision"]
    return {"status": decision["status"], "reason_code": decision["reason_code"],
            "effective_action": decision["effective_action"],
            "behavior_outcome": fresh["behavior_outcome"],
            "confirmation_scope": fresh["confirmation_scope"],
            "confirmation_catalog_complete": fresh["confirmation_catalog_complete"],
            "unobserved_components": deepcopy(fresh["unobserved_components"]),
            "cpu_review_wall_s": float(wall), "api_calls": 0, "gpu_launches": 0}, evidence


def summarize(prepared: Path, out: Path) -> dict:
    root = Path(prepared).resolve(strict=True)
    target = Path(out).resolve()
    if target.exists():
        raise FileExistsError("summary output must be fresh")
    status = controller.status(root)
    manifest = controller._load(root)
    if status["total_arms"] != 16 or len(manifest["arms"]) != 16:
        raise SummaryError("production recovery summary requires the registered 16-arm denominator")
    private_refs: list[dict] = []
    arms = []
    halt = status.get("halt") if isinstance(status.get("halt"), dict) else None
    for source_arm, result in zip(manifest["arms"], status["arms"], strict=True):
        case_id, method = source_arm["case_id"], source_arm["method"]
        if case_id not in CASES or method not in METHODS:
            raise SummaryError("arm identity differs from the registered matrix")
        logical = f"{case_id}:{method}"
        refs = [_generic_ref(source_arm["diagnostic_manifest"], logical + ":diagnostic-manifest",
                             private_refs)]
        verified_diagnostic = controller._verified_diagnostic_status(Path(source_arm["session_path"]))
        diagnostic_status = verified_diagnostic.get("status")
        recorded_diagnostic = result.get("diagnostic_status")
        if recorded_diagnostic is not None and recorded_diagnostic != diagnostic_status:
            raise SummaryError("recorded diagnostic status differs from the reverified Phase A session")
        costs, resource_refs = _resource(result.get("resource_cost"), logical, private_refs)
        refs.extend(resource_refs)
        outcome = result.get("outcome", "not_started")
        if (halt is not None and halt.get("case_id") == case_id and halt.get("method") == method
                and halt.get("reason") == "infrastructure_or_integrity_failure"):
            outcome = "infrastructure_failure"
        recovery_status = None
        recovery_folder = source_arm.get("recovery_folder")
        if recovery_folder is not None and Path(recovery_folder).is_dir():
            recovery_status = investigation_recovery.status(Path(recovery_folder))
        consumed = bool(recovery_status is not None and any(
            row.get("state") in {"running", "done"} for row in recovery_status["requests"].values()))
        attempted = result.get("resource_cost") is not None or consumed
        if result.get("recovery_started") is True and not attempted:
            raise SummaryError("recorded recovery start lacks resource or consumed-ledger evidence")
        endpoint = None
        post_review = None
        if outcome in {"recovered", "not_recovered", "unknown"} and attempted:
            if recovery_status is None:
                raise SummaryError("attempted recovery folder is unavailable")
            done = [row for row in recovery_status["requests"].values() if row["state"] == "done"]
            if len(done) != 1:
                raise SummaryError("one completed recovery receipt required")
            ledger_receipt = done[0].get("receipt", {})
            if ledger_receipt.get("result_sha") is not None:
                post = investigation_recovery.verified_post_observation(Path(source_arm["recovery_folder"]),
                                                                        done[0]["request_id"])
                expected = {"normal": "recovered", "failed": "not_recovered",
                            "unknown": "unknown"}.get(post["behavior_outcome"])
                if expected != outcome:
                    raise SummaryError("controller outcome differs from reverified recovery receipt")
                fixed = _generic_ref(post["fixed_report"], logical + ":fixed-report", private_refs)
                checkpoint = _generic_ref(post["new_checkpoint"], logical + ":new-checkpoint", private_refs)
                refs.extend([fixed, checkpoint])
                if outcome in {"recovered", "not_recovered"}:
                    endpoint = {"fixed_report_sha256": fixed["sha256"],
                                "checkpoint_sha256": checkpoint["sha256"]}
                post_review, review_evidence = _post_review(result.get("post_review"), manifest,
                                                            Path(source_arm["recovery_folder"]),
                                                            done[0]["request_id"], logical, private_refs)
                if review_evidence:
                    refs.append(review_evidence)
            elif outcome != "unknown":
                raise SummaryError("recovery endpoint lacks a verified receipt")
        arms.append({"case_id": case_id, "method": method,
                     "diagnostic_status": diagnostic_status,
                     "diagnostic_confirmed": diagnostic_status == "confirmed",
                     "recovery_attempted": attempted, "recovery_outcome": outcome,
                     "endpoint": endpoint, "post_review": post_review,
                     "costs": costs, "evidence": refs})
    methods = [_method_summary(method, arms) for method in METHODS]
    timing = recovery_timing.read_timing(root)
    for ref in timing["private_refs"]:
        _generic_ref({"path": ref["path"], "sha256": ref["sha256"]},
                     f"timing:{ref['kind']}:{len(private_refs) + 1}", private_refs)
    public_timing = {key: deepcopy(value) for key, value in timing.items() if key != "private_refs"}
    recovered_by_case = []
    for case in CASES:
        recovered = [row for row in arms if row["case_id"] == case
                     and row["recovery_outcome"] == "recovered"]
        if len(recovered) >= 2:
            recovered_by_case.append({"case_id": case,
                                      "methods": [row["method"] for row in recovered],
                                      "endpoints": [row["endpoint"] for row in recovered]})
    public = {"schema": SCHEMA, "planned_cases": list(CASES), "planned_methods": list(METHODS),
              "planned_arms": 16, "completed_arms": status["completed_arms"],
              "controller_halted": bool(status["halted"]),
              "common_source_and_diagnostic_costs": None,
              "common_cost_note": "not present in the registered controller result; not repeated per arm",
              "methods": methods, "arms": arms,
              "timing": public_timing,
              "recovered_endpoint_comparison": recovered_by_case,
              "unknown_values_are_not_zero": True, "api_calls": 0}
    _assert_public(public)
    target.mkdir(parents=True)
    _write_new(target / "public" / "summary.json", public)
    _write_new(target / "private" / "source-refs.json", {
        "schema": "registered_recovery_comparison_private_refs_v1", "prepared": str(root),
        "refs": private_refs})
    (target / "RESULT.md").write_text(
        "# Registered recovery comparison summary\n\n"
        f"- Planned arms: 16\n- Controller completed arms: {status['completed_arms']}\n"
        "- Public paths: 0\n- Common source/diagnostic cost: null unless separately registered\n",
        encoding="utf-8", newline="\n")
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.prepared, args.out)
    except Exception as exc:  # noqa: BLE001 - emit one deterministic CLI rejection packet.
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error_type": type(exc).__name__, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema": SCHEMA, "status": "written", "planned_arms": result["planned_arms"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
