"""Export a reverified V7 recovery summary and path-free cost evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "evals/run_registered_recovery_comparison_v1.py").is_file():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench"), str(ROOT / "evals")]

from evals import registered_recovery_continuation_v4 as continuation_v4
from evals import run_registered_recovery_comparison_v1 as controller
from evals import summarize_registered_recovery_comparison_v1 as base
from rl_triage import investigation_recovery
from rl_triage import investigation_session as io

SCHEMA = "registered_recovery_comparison_public_export_v2"
PLAN_SCHEMA = controller.PLAN_SCHEMA_V7
EXPECTED_STUDY = "registered_recovery_comparison_s2664_s2667_20261105_v13"


class ExportError(ValueError):
    """A registered input or public export invariant differs."""


_VERIFIED_IMPORTS: dict[str, dict] = {}


def _sha(path: Path) -> str:
    return io.file_sha(Path(path).resolve(strict=True))


def _read(path: Path) -> dict[str, Any]:
    body = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ExportError("JSON root differs")
    return body


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ExportError(f"{name} SHA-256 differs")
    return value


def _nonnegative(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ExportError(f"{name} differs")
    return float(value)


def _assert_public(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "path" in str(key).casefold():
                raise ExportError("public export contains a path field")
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)
    elif isinstance(value, str):
        folded = value.replace("\\", "/").casefold()
        if ("bench/private" in folded or PurePosixPath(value).is_absolute()
                or PureWindowsPath(value).is_absolute()):
            raise ExportError("public export contains a filesystem path")


def _imported_resource_v7(row: dict, logical_prefix: str, private: list[dict], scope: dict,
                          plan_schema: str | None = None) -> tuple[dict, list[dict]]:
    if plan_schema != PLAN_SCHEMA:
        return _ORIGINAL_IMPORT(row, logical_prefix, private, scope, plan_schema)
    case_id, method = logical_prefix.split(":")
    if (scope.get("kind") != "completed_effect" or scope.get("case_id") != case_id
            or scope.get("method") != method
            or row["prior_effect_proof_ref"]["sha256"] != scope["proof_sha256"]):
        raise ExportError("completed effect differs from registered V7 import")
    proof_sha = row["prior_effect_proof_ref"]["sha256"]
    verified = _VERIFIED_IMPORTS.get(proof_sha)
    if verified is None:
        verified = continuation_v4.verify_completed_effect(
            row["prior_effect_proof_ref"], expected_import=scope, repo=ROOT)
    if (row["prior_resource_ref"] != verified["prior_resource_ref"]
            or row["prior_admission_ref"] != verified["prior_admission_ref"]
            or row["historical_recovery_folder"] != verified["recovery_folder"]):
        raise ExportError("completed effect historical references differ")
    historical = verified.get("historical_costs")
    current = verified.get("new_dispatch_costs")
    if (not isinstance(historical, dict) or historical.get("simulator_sdk_launches") != 2
            or not isinstance(current, dict) or current.get("sdk_launches") != 0):
        raise ExportError("completed effect resource accounting differs")
    historical_dispatch = _nonnegative(
        historical.get("owned_gpu_s"), "completed effect historical owned wall")
    historical_phase = _nonnegative(
        historical.get("phase_wall_s"), "completed effect historical phase wall")
    refs = [
        base._generic_ref(row["receipt_ref"], logical_prefix + ":projected-resource", private),
        base._generic_ref(verified["prior_resource_ref"], logical_prefix + ":historical-resource", private),
        base._generic_ref(verified["prior_admission_ref"], logical_prefix + ":historical-admission", private),
        base._generic_ref(row["prior_effect_proof_ref"],
                          logical_prefix + ":completed-effect-proof", private),
    ]
    costs = {
        "status": "complete",
        "execution_origin": "verified_prior_effect",
        "phase_elapsed_s": None,
        "locked_wall_s": None,
        "lease_wait_s": None,
        "simulator_owned_wall_s": historical_dispatch,
        "checkpoint_cpu_owned_wall_s": None,
        "telemetry_cpu_owned_wall_s": None,
        "current_dispatch_simulator_owned_wall_s": None,
        "current_dispatch_sdk_launches": 0,
        "reused_training_owned_wall_s": 0.0,
        "reused_training_sdk_launches": 0,
        "reused_completed_dispatch_owned_wall_s": historical_dispatch,
        "reused_completed_dispatch_sdk_launches": 2,
        "whole_effect_active_phase_s": historical_phase,
        "device_energy_scope": "historical_completed_dispatch_only",
        "device_energy": None,
        "reused_training_device_energy": None,
        "api_calls": 0,
        "owned_cost_complete": True,
    }
    return costs, refs


_ORIGINAL_IMPORT: Callable[..., tuple[dict, list[dict]]] = base._imported_resource


def _validate_registered_inputs(prepared: Path, controller_sha: str, terminal: Path,
                                terminal_sha: str, ledger: Path, ledger_sha: str) -> tuple[dict, dict]:
    root = Path(prepared).resolve(strict=True)
    manifest = controller._load(root)
    canonical = controller.verify_prepared_manifest(
        root, repo_root=ROOT, registration_anchor=manifest["registration_anchor"])
    if canonical != manifest:
        raise ExportError("canonical prepared manifest differs")
    plan = io.read_ref(manifest["plan"])
    if plan.get("schema") != PLAN_SCHEMA:
        raise ExportError("public V2 exporter requires the registered V7 plan")
    final = root / "final-controller-result.json"
    if _sha(final) != _digest(controller_sha, "controller result"):
        raise ExportError("controller result changed")
    final_body = _read(final)
    if (final_body.get("completed_arms"), final_body.get("total_arms"), final_body.get("api_calls")) != (16, 16, 0):
        raise ExportError("controller denominator or API count differs")
    terminal_body = _read(terminal)
    if _sha(terminal) != _digest(terminal_sha, "terminal"):
        raise ExportError("terminal changed")
    if not (terminal_body.get("exit_code") == 0 and terminal_body.get("natural_exit") is True
            and terminal_body.get("cleanup_verified") is True):
        raise ExportError("terminal completion differs")
    ledger_body = _read(ledger)
    if _sha(ledger) != _digest(ledger_sha, "cost ledger"):
        raise ExportError("cost ledger changed")
    if (ledger_body.get("schema") != "registered_recovery_whole_cost_join_v6"
            or ledger_body.get("study_status") != "complete"
            or ledger_body.get("study_id") != EXPECTED_STUDY
            or ledger_body.get("coverage") != {
                "import_complete": 8, "new_complete": 7, "new_pending": 0,
                "hold_complete": 1, "matrix_arms": 16, "terminal_outer_present": True,
            }):
        raise ExportError("cost ledger identity or coverage differs")
    refs = ledger_body.get("source_refs")
    if not isinstance(refs, list):
        raise ExportError("cost ledger source references differ")
    expected_refs = {"current_outer": terminal_sha, "v13_plan": manifest["plan"]["sha256"]}
    for role, digest in expected_refs.items():
        matches = [row for row in refs if row == {"role": role, "sha256": digest}]
        if len(matches) != 1:
            raise ExportError(f"cost ledger {role} binding differs")
    _assert_public(ledger_body)
    return final_body, ledger_body


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_historical_post(posts: dict[tuple[Path, str], dict], *, folder: Path,
                           binding: dict, verified: dict, done: dict) -> None:
    source_validation = verified.get("historical_source_validation")
    backend = verified.get("backend_receipt_validation")
    post = verified.get("post_observation")
    old_refs = verified.get("old_refs")
    if (not isinstance(source_validation, dict)
            or source_validation.get("source_anchor") != binding["source_anchor"]
            or source_validation.get("historical_source_authoritative") is not True):
        raise ExportError("historical executed source anchor differs")
    if (not isinstance(backend, dict) or backend.get("request_id") != done["request_id"]
            or done.get("receipt", {}).get("result_sha")
            != backend.get("receipt_ref", {}).get("sha256")):
        raise ExportError("historical request result SHA differs")
    if (not isinstance(post, dict) or not isinstance(old_refs, dict)
            or post.get("fixed_report") != old_refs.get("fixed_report")
            or post.get("new_checkpoint") != old_refs.get("new_checkpoint")):
        raise ExportError("historical post endpoint SHA differs")
    key = (folder, done["request_id"])
    if key in posts:
        raise ExportError("historical recovery request duplicated")
    posts[key] = {"post": deepcopy(post), "post_sha256": _canonical_sha(post),
                  "source_anchor": binding["source_anchor"]}


def _historical_post(posts: dict[tuple[Path, str], dict], folder: Path, request_id: str,
                     fallback: Callable[..., dict], *args: object, **kwargs: object) -> dict:
    resolved = Path(folder).resolve(strict=True)
    key = (resolved, request_id)
    if key in posts:
        cached = posts[key]
        post = cached["post"]
        if _canonical_sha(post) != cached["post_sha256"]:
            raise ExportError("cached historical post changed")
        return deepcopy(post)
    if resolved in {candidate[0] for candidate in posts}:
        raise ExportError("historical request identity differs")
    return fallback(folder, request_id, *args, **kwargs)


def _historical_posts(prepared: Path) -> dict[tuple[Path, str], dict]:
    root = Path(prepared).resolve(strict=True)
    manifest = controller._load(root)
    plan = io.read_ref(manifest["plan"])
    imports = {(row["case_id"], row["method"]): row
               for row in plan["continuation"]["stage_imports"]}
    posts: dict[tuple[Path, str], dict] = {}
    for arm in manifest["arms"]:
        binding = imports.get((arm["case_id"], arm["method"]))
        continuation = arm.get("continuation_import")
        if binding is None:
            continue
        if not isinstance(continuation, dict):
            raise ExportError("historical arm lacks continuation import")
        verified = continuation_v4.verify_completed_effect(
            continuation["proof"], expected_import=binding, repo=ROOT)
        proof_sha = continuation["proof"]["sha256"]
        if proof_sha in _VERIFIED_IMPORTS:
            raise ExportError("historical proof duplicated")
        _VERIFIED_IMPORTS[proof_sha] = verified
        folder = Path(verified["recovery_folder"]).resolve(strict=True)
        status = investigation_recovery.status(folder)
        done = [row for row in status["requests"].values() if row["state"] == "done"]
        if len(done) != 1:
            raise ExportError("historical recovery request denominator differs")
        _cache_historical_post(posts, folder=folder, binding=binding,
                               verified=verified, done=done[0])
    if len(posts) != 8:
        raise ExportError("historical post coverage differs")
    return posts


def _public_metrics(summary: dict, ledger: dict) -> dict:
    arms = summary.get("arms")
    if not isinstance(arms, list) or len(arms) != 16:
        raise ExportError("summary arm denominator differs")
    outcomes = {name: sum(row.get("recovery_outcome") == name for row in arms)
                for name in ("recovered", "not_recovered", "hold", "unknown")}
    imported = sum(row.get("execution_origin") == "verified_prior_effect" for row in arms)
    attempted = sum(row.get("recovery_attempted") is True for row in arms)
    fresh = attempted - imported
    if outcomes != {"recovered": 0, "not_recovered": 15, "hold": 1, "unknown": 0}:
        raise ExportError("recovery outcomes differ")
    if (imported, fresh, attempted) != (8, 7, 15):
        raise ExportError("import/fresh recovery denominator differs")
    methods = {row["method"]: row for row in summary.get("methods", [])}
    if set(methods) != set(controller.METHODS) or any(
            row["outcomes"]["recovered"] != 0 for row in methods.values()):
        raise ExportError("method recovery count differs")
    total_model = ledger["model_api_totals"]
    legacy_rows = [row for row in ledger.get("buckets", [])
                   if row.get("bucket_id") == "diagnostic.legacy_v3.known_events"]
    if len(legacy_rows) != 1 or not isinstance(legacy_rows[0].get("metrics"), dict):
        raise ExportError("legacy diagnostic model bucket differs")
    legacy = legacy_rows[0]["metrics"]
    diagnostic_calls = total_model["known_calls"] - legacy["model_calls"]
    diagnostic_input = total_model["known_input_tokens"] - legacy["input_tokens"]
    diagnostic_output = total_model["known_output_tokens"] - legacy["output_tokens"]
    values = (diagnostic_calls, diagnostic_input, diagnostic_output)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ExportError("diagnostic model usage differs")
    if diagnostic_calls != 21:
        raise ExportError("current diagnostic API count differs")
    metrics = {
        "schema": "registered_recovery_comparison_public_metrics_v1",
        "study_id": EXPECTED_STUDY,
        "recovery": {
            "planned_arms": 16, "attempted_arms": attempted,
            "imported_completed_arms": imported, "fresh_attempted_arms": fresh,
            "hold_arms": 1, "recovered_arms": 0, "not_recovered_arms": 15,
            "fresh_sdk_launches": 14, "api_calls": 0,
            "hold_is_not_a_saving_or_success": True,
        },
        "diagnostic_current": {
            "planned_arms": 24, "api_calls": int(diagnostic_calls),
            "input_tokens": int(diagnostic_input), "output_tokens": int(diagnostic_output),
            "derivation": "whole_model_totals_minus_separate_legacy_diagnostic",
        },
        "methods": [{"method": name, "recovered_arms": 0,
                     "not_recovered_arms": methods[name]["outcomes"]["not_recovered"],
                     "hold_arms": methods[name]["outcomes"]["hold"]}
                    for name in controller.METHODS],
        "interpretation": {
            "all_four_methods_recovered_zero": True,
            "historical_motor_import_order_is_observational_not_counterbalanced_performance": True,
            "cost_unknowns_preserved_in_cost_ledger": True,
        },
    }
    _assert_public(metrics)
    return metrics


def _arm_metrics(summary: dict) -> dict:
    rows = []
    for arm in summary["arms"]:
        rows.append({key: deepcopy(arm.get(key)) for key in (
            "case_id", "method", "execution_origin", "diagnostic_status",
            "diagnostic_confirmed", "recovery_attempted", "recovery_outcome",
            "endpoint", "post_review", "costs", "evidence")})
    body = {"schema": "registered_recovery_comparison_public_arm_metrics_v1", "arms": rows}
    _assert_public(body)
    return body


def export(prepared: Path, out: Path, *, controller_sha: str, terminal: Path,
           terminal_sha: str, ledger: Path, ledger_sha: str) -> dict:
    target = Path(out).resolve()
    if target.exists():
        raise FileExistsError("public export output must be fresh")
    final, ledger_body = _validate_registered_inputs(
        prepared, controller_sha, terminal, terminal_sha, ledger, ledger_sha)
    stage = target.with_name(target.name + ".building")
    if stage.exists():
        raise FileExistsError("public export staging output must be fresh")
    _VERIFIED_IMPORTS.clear()
    historical_posts = _historical_posts(prepared)
    original_post = investigation_recovery.verified_post_observation

    def verified_post(folder: Path, request_id: str, *args: object, **kwargs: object) -> dict:
        return _historical_post(historical_posts, folder, request_id, original_post, *args, **kwargs)

    base._imported_resource = _imported_resource_v7
    investigation_recovery.verified_post_observation = verified_post
    try:
        summary = base.summarize(prepared, stage)
    finally:
        base._imported_resource = _ORIGINAL_IMPORT
        investigation_recovery.verified_post_observation = original_post
    if summary.get("completed_arms") != 16 or summary.get("api_calls") != 0:
        raise ExportError("public summary completion differs")
    metrics = _public_metrics(summary, ledger_body)
    arm_metrics = _arm_metrics(summary)
    public = stage / "public"
    io.write_new(public / "arm-metrics.json", arm_metrics)
    io.write_new(public / "whole-cost-ledger.json", ledger_body)
    io.write_new(public / "metrics.json", metrics)
    sources = {
        "schema": "registered_recovery_comparison_public_sources_v1",
        "controller_result": {"role": "controller_result", "sha256": controller_sha},
        "terminal": {"role": "terminal", "sha256": terminal_sha},
        "cost_ledger": {"role": "cost_ledger", "sha256": ledger_sha},
        "plan": {"role": "plan", "sha256": controller._load(prepared)["plan"]["sha256"]},
    }
    _assert_public(sources)
    io.write_new(public / "sources.json", sources)
    manifest_files = []
    for path in sorted(public.iterdir()):
        manifest_files.append({"name": path.name, "sha256": _sha(path)})
    manifest = {"schema": SCHEMA, "study_id": EXPECTED_STUDY,
                "status": "complete", "files": manifest_files,
                "controller_completed_arms": final["completed_arms"]}
    io.write_new(public / "manifest.json", manifest)
    _assert_public(manifest)
    stage.replace(target)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--controller-result-sha256", required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--cost-ledger", required=True, type=Path)
    parser.add_argument("--cost-ledger-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = export(
            args.prepared, args.out,
            controller_sha=args.controller_result_sha256,
            terminal=args.terminal, terminal_sha=args.terminal_sha256,
            ledger=args.cost_ledger, ledger_sha=args.cost_ledger_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - deterministic rejection packet.
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error_type": type(exc).__name__, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "files": len(result["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
