"""Immutable resource evidence for one registered recovery dispatch."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rl_triage import investigation_session as io
from rl_triage.gpu_resource_sampler import PeriodicDeviceSampler

SCHEMA = "registered_recovery_resource_evidence_v1"
PROJECTED_SCHEMA = "registered_recovery_resource_receipt_v1"


def _read_ref(ref: object, label: str) -> dict:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{label} ref fields")
    body = io.read_ref(ref)
    if not isinstance(body, dict):
        raise TypeError(f"{label} body")
    return body


def _wall(body: dict) -> float | None:
    value = body.get("wall_s", body.get("execution_wall_s"))
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        return None
    return float(value)


class Measurement:
    """Own the sampler lifetime and seal all evidence available after one dispatch."""

    def __init__(self, folder: Path, request: dict, admission_ref: dict,
                 context: dict, *, sampler_factory: Callable[[Path], Any] | None = None) -> None:
        self.folder = Path(folder).resolve()
        self.request = request
        self.admission_ref = admission_ref
        self.context = context
        self.started = time.perf_counter()
        factory = sampler_factory or PeriodicDeviceSampler
        self.sampler = factory(self.folder / "device-sampling" / request["request_id"])
        self.sampler_start = self.sampler.start()

    def finish(self, *, raw: dict | None, error: BaseException | None) -> dict:
        sampler_ref = None
        sampler_error = None
        try:
            sampler_ref = self.sampler.stop()["receipt_ref"]
            sampler_body = io.read_ref(sampler_ref)
            if (sampler_body.get("schema") == "periodic_device_sampling_stop_v1"
                    and sampler_body.get("lease_matches") is not True):
                sampler_error = "LeaseMismatch"
        except Exception as exc:  # noqa: BLE001 - preserve execution evidence if sampling close fails.
            sampler_error = type(exc).__name__
        output = Path(self.context["output_root"])
        paths = {
            "training": output / "training" / "execution.json",
            "fixed": output / "fixed" / "execution.json",
            "checkpoint": output / "training" / "checkpoint_execution.json",
            "telemetry": output / "training" / "extract_execution.json",
        }
        execution_refs: dict[str, dict | None] = {}
        walls: dict[str, float | None] = {}
        for name, path in paths.items():
            if path.is_file():
                execution_ref = io.reference(path)
                execution_refs[name] = execution_ref
                walls[name] = _wall(io.read_ref(execution_ref))
            else:
                execution_refs[name] = None
                walls[name] = None
        simulator_launches = sum(execution_refs[name] is not None for name in ("training", "fixed"))
        cpu_launches = int(execution_refs["checkpoint"] is not None)
        telemetry_launches = int(execution_refs["telemetry"] is not None)
        launched_gpu_walls = [walls[name] for name in ("training", "fixed")
                              if execution_refs[name] is not None]
        gpu_complete = all(value is not None for value in launched_gpu_walls)
        owned_gpu_s = (sum(value for value in launched_gpu_walls if value is not None)
                       if launched_gpu_walls and gpu_complete else None)
        elapsed = max(0.0, time.perf_counter() - self.started)
        receipt = {
            "schema": SCHEMA, "request_id": self.request["request_id"],
            "effect_key": self.request["effect_key"],
            "status": "complete" if error is None and sampler_error is None else "failed",
            "phase_wall_s": elapsed, "lease_await_s": None, "locked_wall_s": elapsed,
            "owned_gpu_s": owned_gpu_s, "owned_gpu_s_complete": gpu_complete,
            "simulator_sdk_launches": simulator_launches,
            "cpu_checkpoint_launches": cpu_launches,
            "cpu_telemetry_launches": telemetry_launches,
            "total_owned_children": simulator_launches + cpu_launches + telemetry_launches,
            "admission_ref": self.admission_ref, "sampler_ref": sampler_ref,
            "sampler_error_type": sampler_error,
            "execution_refs": execution_refs,
            "raw_receipt_ref": raw.get("receipt_ref") if isinstance(raw, dict) else None,
            "error_type": type(error).__name__ if error is not None else None,
        }
        path = self.folder / "resources" / f"{self.request['request_id']}.json"
        io.write_new(path, receipt)
        return io.reference(path)


def verify(ref: object, *, request_id: str, effect_key: str) -> dict:
    body = _read_ref(ref, "recovery resource")
    fields = {"schema", "request_id", "effect_key", "status", "phase_wall_s", "lease_await_s",
              "locked_wall_s", "owned_gpu_s", "owned_gpu_s_complete", "simulator_sdk_launches",
              "cpu_checkpoint_launches", "cpu_telemetry_launches", "total_owned_children", "admission_ref", "sampler_ref",
              "sampler_error_type", "execution_refs", "raw_receipt_ref", "error_type"}
    if (set(body) != fields or body["schema"] != SCHEMA or body["request_id"] != request_id
            or body["effect_key"] != effect_key or body["status"] not in {"complete", "failed", "unknown"}):
        raise ValueError("recovery resource binding differs")
    for field in ("phase_wall_s", "locked_wall_s"):
        value = body[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("recovery resource wall differs")
    if body["lease_await_s"] is not None:
        raise ValueError("unmeasured lease wait must remain null")
    for field, maximum in (("simulator_sdk_launches", 2), ("cpu_checkpoint_launches", 1),
                           ("cpu_telemetry_launches", 1), ("total_owned_children", 4)):
        if type(body[field]) is not int or not 0 <= body[field] <= maximum:
            raise ValueError("recovery child count differs")
    if body["total_owned_children"] != (body["simulator_sdk_launches"] + body["cpu_checkpoint_launches"]
                                       + body["cpu_telemetry_launches"]):
        raise ValueError("recovery child count sum differs")
    _read_ref(body["admission_ref"], "lease admission")
    if body["sampler_ref"] is not None:
        _read_ref(body["sampler_ref"], "sampler stop")
    refs = body["execution_refs"]
    if not isinstance(refs, dict) or set(refs) != {"training", "fixed", "checkpoint", "telemetry"}:
        raise ValueError("execution refs differ")
    execution_bodies = {}
    for value in refs.values():
        if value is not None:
            execution_bodies[value["path"]] = _read_ref(value, "owned execution")
    expected_simulator = sum(refs[name] is not None for name in ("training", "fixed"))
    expected_cpu = int(refs["checkpoint"] is not None)
    expected_telemetry = int(refs["telemetry"] is not None)
    if (body["simulator_sdk_launches"] != expected_simulator
            or body["cpu_checkpoint_launches"] != expected_cpu
            or body["cpu_telemetry_launches"] != expected_telemetry):
        raise ValueError("execution refs and child counts differ")
    gpu_walls = [_wall(execution_bodies[refs[name]["path"]]) for name in ("training", "fixed")
                 if refs[name] is not None]
    complete = all(value is not None for value in gpu_walls)
    expected_gpu = (sum(value for value in gpu_walls if value is not None)
                    if gpu_walls and complete else None)
    if (type(body["owned_gpu_s_complete"]) is not bool or body["owned_gpu_s_complete"] is not complete
            or body["owned_gpu_s"] != expected_gpu):
        raise ValueError("owned GPU wall differs from raw executions")
    if body["raw_receipt_ref"] is not None:
        _read_ref(body["raw_receipt_ref"], "raw recovery receipt")
    return body


def project(ref: dict, destination: Path, *, case_id: str, method: str,
            plan_sha256: str, request_id: str, effect_key: str) -> dict:
    body = verify(ref, request_id=request_id, effect_key=effect_key)
    projected = {"schema": PROJECTED_SCHEMA, "case_id": case_id, "method": method,
                 "plan_sha256": plan_sha256, "status": body["status"],
                 "phase_wall_s": body["phase_wall_s"], "lease_await_s": None,
                 "locked_wall_s": body["locked_wall_s"], "owned_gpu_s": body["owned_gpu_s"],
                 "owned_gpu_s_complete": body["owned_gpu_s_complete"],
                 "simulator_sdk_launches": body["simulator_sdk_launches"],
                 "cpu_checkpoint_launches": body["cpu_checkpoint_launches"],
                 "cpu_telemetry_launches": body["cpu_telemetry_launches"],
                 "total_owned_children": body["total_owned_children"], "api_calls": 0,
                 "sampler_ref": body["sampler_ref"], "admission_ref": body["admission_ref"],
                 "recovery_resource_ref": ref, "execution_refs": body["execution_refs"],
                 "error_type": body["error_type"] or body["sampler_error_type"]}
    io.write_new(Path(destination), projected)
    return projected
