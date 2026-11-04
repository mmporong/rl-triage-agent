"""Reverify and export the registered post-actuator confirmation without paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "evals/registered_post_actuator_confirmation_v5_core.py").is_file():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench"), str(ROOT / "evals")]

from evals import run_registered_post_actuator_confirmation_v5 as operation

SCHEMA = "post_actuator_confirmation_public_export_v1"
EXPECTED = {
    "observer_config": "1aad4482ce264e86602312da487cfab3159fb415cc21c60a40b3edccbf308a8b",
    "supervisor_config": "033bc3f9e5bae2916a1f0ecb89675018a4fdd3da0b45c1889a51d45a7fa7207e",
    "source_inputs": "11cd0bc13be4aec43cb31f5f73fde2bdeb33cdfbf0c0788ea4d8dc0d03cad26b",
    "outer_protocol": "b8727373a65f75cf3f3dab7cf6c355886093b4f53e1032abc0c17bf2cedae66c",
    "authority": "16da6c224ea830a68d69c686753d864b7bea38cc969ba175bb48d66b566993cf",
    "delegated_approval": "b7b626e60c50a68d17094295b668ed691ad30bcc96b910f21e93fc8ee46f1c92",
    "checkpoint": "3aaeabbc6a00f4fcb34fe13dff12fdadb557a031fc7ff581d6436b4452f5a3e7",
    "fixed_report": "8e553408c55053c64e97e83902f806d06586ee388ceb4a3a93be4a73355ee825",
    "recovery_receipt": "1774de2d8fed9c37cd736277ad40bc4b594b15787c927b972693996620f9353a",
    "primary_cost_raw": "b5fe390bd6e38a041ee46033800f39c646c53f0159063433e04656ae19c10428",
    "primary_cost_public": "a747bd0c719f67970a3dd2c7de1a709ab2044c17156feefd1b593b69a0d79ef1",
}


class ExportError(ValueError):
    """An actual input or public projection invariant differs."""


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).resolve(strict=True).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExportError("JSON root differs")
    return value


def _write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")


def _number(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ExportError(f"{name} differs")
    return float(value)


def _assert_public(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "path" in str(key).casefold():
                raise ExportError("public value contains a path field")
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)
    elif isinstance(value, str):
        folded = value.replace("\\", "/").casefold()
        if ("bench/private" in folded or PurePosixPath(value).is_absolute()
                or PureWindowsPath(value).is_absolute()):
            raise ExportError("public value contains a filesystem path")


def _ref(value: object, name: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ExportError(f"{name} reference fields differ")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    digest = str(value["sha256"])
    if _sha(path) != digest:
        raise ExportError(f"{name} reference changed")
    return path, digest


def _closure(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ExportError(f"{name} source closure differs")
    for relative, digest in value.items():
        if (not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts or not isinstance(digest, str)):
            raise ExportError(f"{name} source closure entry differs")
        path = (ROOT / relative).resolve(strict=True)
        if ROOT not in path.parents or _sha(path, lf=True) != digest:
            raise ExportError(f"{name} source closure changed: {relative}")


def _verify_lineage(observer: dict, source_path: Path, outer_path: Path,
                    authority_path: Path, isaaclab: Path) -> tuple[dict, dict, dict, dict]:
    if observer.get("schema") != "registered_post_observer_config_v1":
        raise ExportError("observer config schema differs")
    inputs = observer.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"source_inputs", "outer_protocol", "authority"}:
        raise ExportError("observer input roles differ")
    expected_paths = {"source_inputs": source_path, "outer_protocol": outer_path,
                      "authority": authority_path}
    for role, path in expected_paths.items():
        actual, digest = _ref(inputs[role], f"observer {role}")
        if actual != path or digest != EXPECTED[role]:
            raise ExportError(f"observer {role} identity differs")
    _closure(observer.get("source_sha256_lf"), "observer")
    source = operation._source(_read(source_path))
    outer = operation.validate_outer(_read(outer_path), repo_root=ROOT)
    authority = operation._authority(
        _read(authority_path), source_path, outer_path, outer
    )
    budget = operation._primary_budget(
        authority["primary_result"], authority["primary_manifest"], authority["primary_plan"],
        authority["primary_supervisor"], outer, repo_root=ROOT)
    if (source["checkpoint"]["sha256"] != EXPECTED["checkpoint"]
            or source["fixed_report"]["sha256"] != EXPECTED["fixed_report"]
            or source["recovery_receipt"]["sha256"] != EXPECTED["recovery_receipt"]
            or outer["delegated_approval"]["sha256"] != EXPECTED["delegated_approval"]
            or budget.get("actual_before_post") != 32
            or budget.get("remaining_before_post") != 1):
        raise ExportError("registered post lineage or budget differs")
    isaac = Path(isaaclab).resolve(strict=True)
    if not (isaac / "_isaac_sim/python.bat").is_file():
        raise ExportError("Isaac Lab launcher differs")
    return source, outer, authority, budget


def _verify_run(actual: Path, post_result_path: Path) -> dict[str, Any]:
    root = Path(actual).resolve(strict=True)
    observer_path = root / "observer-config.json"
    supervisor_path = root / "supervisor-config.json"
    launch_path = root / "launch.json"
    run = root / "run"
    if _sha(observer_path) != EXPECTED["observer_config"] or _sha(supervisor_path) != EXPECTED["supervisor_config"]:
        raise ExportError("actual observer/supervisor config changed")
    observer = _read(observer_path)
    supervisor = _read(supervisor_path)
    launch = _read(launch_path)
    _closure(supervisor.get("source_sha256_lf"), "supervisor")
    source_path, _ = _ref(observer["inputs"]["source_inputs"], "source inputs")
    outer_path, _ = _ref(observer["inputs"]["outer_protocol"], "outer protocol")
    authority_path, _ = _ref(observer["inputs"]["authority"], "authority")
    source, outer, authority, budget = _verify_lineage(
        observer, source_path, outer_path, authority_path, Path(observer["isaaclab"]))
    terminal_path = run / "supervisor-exit.json"
    owned_path = run / "owned-execution.json"
    receipt_path = run / "observation/observer-result.json"
    stop_path = run / "observation/device-sampling/collection-0001/stop.json"
    device_path = run / "post-device-metrics.json"
    meter_path = run / "post-parent-meter.json"
    terminal, owned, receipt = _read(terminal_path), _read(owned_path), _read(receipt_path)
    stop, device, meter = _read(stop_path), _read(device_path), _read(meter_path)
    post = _read(post_result_path)
    if (launch.get("config_sha256") != EXPECTED["supervisor_config"]
            or launch.get("pid") != terminal.get("supervisor_pid")
            or terminal.get("config_sha256") != EXPECTED["supervisor_config"]):
        raise ExportError("launcher/supervisor binding differs")
    owned_fields = {key: terminal.get(key) for key in owned}
    if (terminal.get("schema") != "registered_post_supervisor_terminal_v1"
            or terminal.get("exit_code") != 0 or terminal.get("natural_exit") is not True
            or terminal.get("cleanup_verified") is not True or owned_fields != owned):
        raise ExportError("owned terminal differs")
    post_ref_path, post_sha = _ref(receipt.get("post_result"), "observer post result")
    if (receipt.get("schema") != "registered_post_observer_receipt_v1"
            or receipt.get("status") != "complete" or receipt.get("post_status") != "complete"
            or receipt.get("sdk_launches_reported") != 1 or receipt.get("api_calls") != 0
            or post_ref_path != Path(post_result_path).resolve(strict=True)
            or post_sha != _sha(post_result_path)):
        raise ExportError("observer receipt differs")
    stop_ref_path, stop_sha = _ref(device.get("sampler_stop_ref"), "sampler stop")
    if (stop_ref_path != stop_path or stop.get("schema") != "periodic_device_sampling_stop_v1"
            or stop.get("lease_matches") is not True or stop_sha != _sha(stop_path)
            or device.get("schema") != "post_actuator_device_metrics_v1"
            or device.get("scope") != "device_wide_including_other_loads"
            or device.get("process_attribution") is not None):
        raise ExportError("device sampler evidence differs")
    if (meter.get("schema") != "post_actuator_parent_meter_v1"
            or meter.get("energy_wh") is not None or meter.get("peak_power_w") is not None
            or meter.get("child_process_cpu_s") is not None
            or meter.get("actual_billing_usd") is not None
            or meter.get("provider_gpu_cost_usd") is not None):
        raise ExportError("parent meter unknown fields differ")
    parent_wall = _number(meter.get("parent_full_wall_s"), "parent wall")
    owned_wall = _number(meter.get("owned_wall_s"), "owned wall")
    phase_wall = _number(meter.get("phase_wall_s"), "phase wall")
    if not (phase_wall <= owned_wall <= parent_wall):
        raise ExportError("post wall nesting differs")
    component = post.get("component_receipt")
    decision = post.get("supplementary_decision")
    if (post.get("schema") != "post_actuator_operational_result_v5"
            or post.get("status") != "complete" or post.get("condition") != "NONE"
            or post.get("sdk_launches") != 1 or post.get("outer_sdk_cap") != 1
            or post.get("checkpoint_association_only") is not True
            or post.get("producer_loads_checkpoint") is not False
            or post.get("trained_policy_validated") is not False
            or post.get("automatic_retry") is not False
            or not isinstance(component, Mapping) or not isinstance(decision, Mapping)):
        raise ExportError("post result fixed scope differs")
    for role, ref in component.get("source_refs", {}).items():
        _ref(ref, f"component {role}")
    if (component.get("schema") != "registered_post_actuator_confirmation_receipt_v1"
            or component.get("checkpoint", {}).get("sha256") != EXPECTED["checkpoint"]
            or component.get("actual_checkpoint_execution") is not False
            or component.get("trained_policy_validated") is not False
            or component.get("policy_free_runtime_config_readback") is not True
            or component.get("state_restored") is not True
            or component.get("measurement_present") is not True
            or component.get("outcome") != decision.get("mapping_outcome")
            or decision.get("schema") != "registered_post_actuator_supplementary_decision_v1"
            or decision.get("approval_granted") is not False
            or decision.get("authority_created") is not False
            or decision.get("api_calls") != 0 or decision.get("sdk_launches_consumed") != 1):
        raise ExportError("post component or supplementary decision differs")
    if post.get("budget") != {**budget, "remaining_after_post": 0}:
        raise ExportError("post budget result differs")
    return {"observer": observer, "source": source, "outer": outer, "authority": authority,
            "budget": budget, "launch": launch, "terminal": terminal, "owned": owned,
            "receipt": receipt, "stop": stop, "device": device, "meter": meter,
            "post": post, "refs": {
                "observer_config": {"sha256": _sha(observer_path)},
                "supervisor_config": {"sha256": _sha(supervisor_path)},
                "launch": {"sha256": _sha(launch_path)},
                "owned": {"sha256": _sha(owned_path)},
                "terminal": {"sha256": _sha(terminal_path)},
                "observer_receipt": {"sha256": _sha(receipt_path)},
                "sampler_stop": {"sha256": _sha(stop_path)},
                "device_metrics": {"sha256": _sha(device_path)},
                "parent_meter": {"sha256": _sha(meter_path)},
                "post_result": {"sha256": _sha(post_result_path)},
            }}


def _device_window(device: Mapping[str, Any]) -> dict[str, Any]:
    energy = device.get("energy")
    summaries = device.get("device_summary")
    if not isinstance(energy, Mapping) or not isinstance(summaries, Mapping):
        raise ExportError("device window evidence differs")
    devices = energy.get("devices")
    if not isinstance(devices, Mapping) or len(devices) != 1 or len(summaries) != 1:
        raise ExportError("device window denominator differs")
    key, row = next(iter(devices.items()))
    summary = summaries.get(key)
    if not isinstance(row, Mapping) or not isinstance(summary, Mapping):
        raise ExportError("device window row differs")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ExportError("device summary metrics differ")
    return {
        "scope": "sample_window_device_wide_including_other_loads",
        "sample_count": row.get("sample_count"),
        "window_s": row.get("window_s"),
        "covered_s": row.get("covered_s"),
        "coverage_fraction": row.get("coverage_fraction"),
        "gap_count": len(row.get("gaps", [])),
        "max_gap_s": row.get("max_gap_s"),
        "energy_wh": row.get("energy_wh"),
        "power_draw_w_peak": metrics.get("power_draw_w_peak", {}).get("value"),
        "memory_used_mib_peak": metrics.get("memory_used_mib_peak", {}).get("value"),
        "gpu_util_percent_sample_mean": metrics.get("gpu_util_percent_sample_mean", {}).get("value"),
        "process_attribution": None,
        "whole_parent_energy": None,
        "whole_parent_peak_power": None,
    }


def _public(verified: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    post = verified["post"]
    component = post["component_receipt"]
    decision = post["supplementary_decision"]
    meter = verified["meter"]
    window = _device_window(verified["device"])
    result = {
        "schema": "post_actuator_confirmation_public_result_v1",
        "status": "complete",
        "case_id": component["case_id"],
        "condition": post["condition"],
        "mapping_outcome": component["outcome"],
        "decision_status": decision["status"],
        "effective_action": decision["effective_action"],
        "approval_granted": False,
        "authority_created": False,
        "policy_free_runtime_config_readback": True,
        "state_restored": True,
        "checkpoint_association_only": True,
        "actual_checkpoint_execution": False,
        "trained_policy_validated": False,
        "new_recovery_claim": False,
        "interpretation": "stock actuator mapping confirmation; not trained-policy behavior validation",
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "fixed_report_sha256": EXPECTED["fixed_report"],
        "recovery_receipt_sha256": EXPECTED["recovery_receipt"],
        "component_receipt_sha256": component["receipt_sha256"],
    }
    metrics = {
        "schema": "post_actuator_confirmation_public_metrics_v1",
        "sdk": {"primary_cap": 32, "post_cap": 1, "whole_study_cap": 33,
                "post_sdk_launches": 1},
        "api_calls": 0,
        "parent": {
            "full_wall_s": meter["parent_full_wall_s"],
            "owned_wall_s": meter["owned_wall_s"],
            "phase_wall_s": meter["phase_wall_s"],
            "parent_process_cpu_s": meter["parent_process_cpu_s"],
            "child_process_cpu_s": None,
            "actual_billing_usd": None,
            "provider_gpu_cost_usd": None,
            "energy_wh": None,
            "peak_power_w": None,
        },
        "device_sample_window": window,
        "device_window_is_not_whole_parent_energy": True,
    }
    sources = {
        "schema": "post_actuator_confirmation_public_sources_v1",
        "source_inputs": {"role": "source_inputs", "sha256": EXPECTED["source_inputs"]},
        "outer_protocol": {"role": "outer_protocol", "sha256": EXPECTED["outer_protocol"]},
        "authority": {"role": "authority", "sha256": EXPECTED["authority"]},
        "delegated_approval": {"role": "delegated_approval", "sha256": EXPECTED["delegated_approval"]},
        "primary_cost_raw": {"role": "primary_cost_raw", "sha256": EXPECTED["primary_cost_raw"]},
        "primary_cost_public": {"role": "primary_cost_public", "sha256": EXPECTED["primary_cost_public"]},
        "actual": deepcopy(verified["refs"]),
        "component": {role: {"role": role, "sha256": ref["sha256"]}
                      for role, ref in component["source_refs"].items()},
        "operation": {role: {"role": role, "sha256": post[role]["sha256"]}
                      for role in ("intent", "consumption", "execution")},
    }
    for body in (result, metrics, sources):
        _assert_public(body)
    return result, metrics, sources


def export(actual: Path, post_result: Path, out: Path) -> dict:
    target = Path(out).resolve()
    if target.exists():
        raise FileExistsError("public post export must be fresh")
    verified = _verify_run(actual, post_result)
    result, metrics, sources = _public(verified)
    target.mkdir(parents=True)
    _write_new(target / "result.json", result)
    _write_new(target / "metrics.json", metrics)
    _write_new(target / "sources.json", sources)
    files = [{"name": path.name, "sha256": _sha(path)} for path in sorted(target.iterdir())]
    manifest = {"schema": SCHEMA, "status": "complete", "files": files}
    _assert_public(manifest)
    _write_new(target / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--post-result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        value = export(args.actual, args.post_result, args.out)
    except Exception as exc:  # noqa: BLE001 - deterministic rejection packet.
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error_type": type(exc).__name__, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema": SCHEMA, "status": value["status"],
                      "files": len(value["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
