"""정책 실행 없이 stock Go2의 live DCMotor 토크-속도 곡선을 기록한다."""
from __future__ import annotations

import argparse
import inspect
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import sys
import time
import traceback
from types import CodeType
from datetime import datetime, timezone
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import runtime_readback_smoke as S  # noqa: E402
from rl_triage import actuator_envelope as E  # noqa: E402


TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
CONDITIONS = ("NONE", "F2", "F3", "F4")
EXPECTED_MATCH = {"NONE": True, "F2": False, "F3": True, "F4": True}
TARGET_METHODS = {
    "F2": ("isaaclab.actuators.actuator_pd", "DCMotor", "_clip_effort"),
    "F3": ("rsl_rl.modules.actor_critic", "ActorCritic", "act"),
    "F4": ("rsl_rl.runners", "OnPolicyRunner", "__init__"),
}
GO2_SOURCE = "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
ACTUATOR_SOURCE = "source/isaaclab/isaaclab/actuators/actuator_pd.py"
SDK_PYTHON_RELATIVE = "kit/python/kit.exe"
GO2_JOINT_NAMES = frozenset({
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
})
STATE_BUFFER_FIELDS = ("_joint_vel", "computed_effort", "applied_effort", "_zeros_effort")
STATE_KEYS = {"schema", "buffers", "common_step_counter", "sim_step_counter"}
BUFFER_KEYS = {"identity", "shape", "values"}
HELPER_KEYS = {
    "schema", "status", "curve_matches_expected", "invalid_reason", "requested_efforts",
    "measurements", "config_provenance", "limitations",
}
MEASUREMENT_KEYS = {
    "velocity_rad_s", "expected_min_effort", "expected_max_effort",
    "measured_min_effort", "measured_max_effort", "matches_expected",
}
PROVENANCE_KEYS = {
    "caller_asserted_source_sha256", "source_binding_verified", "expected_config",
    "velocity_grid_rad_s", "runtime_config_readback",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code": _canonical_code(value)}
    if value is None or value is Ellipsis or type(value) in (bool, int, str):
        if value is Ellipsis:
            return {"singleton": "Ellipsis"}
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("code object contains a non-finite float constant")
        return {"float_hex": value.hex()}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("code object contains a non-finite complex constant")
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if type(value) is slice:
        return {"slice": [_canonical_constant(value.start), _canonical_constant(value.stop),
                          _canonical_constant(value.step)]}
    if type(value) in (tuple, list):
        return {"sequence": [_canonical_constant(item) for item in value]}
    if type(value) is frozenset:
        rows = [_canonical_constant(item) for item in value]
        return {"frozenset": sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))}
    raise ValueError(f"unsupported code constant type: {type(value).__name__}")


def _canonical_code(code: CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount, "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount, "nlocals": code.co_nlocals,
        "flags": code.co_flags, "bytecode": code.co_code.hex(),
        "constants": [_canonical_constant(item) for item in code.co_consts],
        "names": list(code.co_names), "variables": list(code.co_varnames),
        "freevars": list(code.co_freevars), "cellvars": list(code.co_cellvars),
    }


def _code_sha(function_or_code: Any) -> str:
    code = function_or_code if isinstance(function_or_code, CodeType) else getattr(function_or_code, "__code__", None)
    if code is None:
        raise ValueError("target method has no Python code object")
    return S._canonical_sha(_canonical_code(code))


def runtime_binding(*, torch_module: Any | None = None,
                    isaaclab_root: Path | None = None) -> dict[str, Any]:
    def distribution(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unavailable"

    torch_module_version = None if torch_module is None else str(torch_module.__version__)
    executable = Path(sys.executable).resolve()
    if isaaclab_root is not None:
        root = (isaaclab_root / "_isaac_sim").resolve()
        try:
            executable_value = executable.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Python executable is outside the Isaac Lab checkout") from error
    else:
        executable_value = str(executable)
    version = sys.version_info
    return {
        "python": {"implementation": platform.python_implementation(),
                   "version": [version.major, version.minor, version.micro],
                   "cache_tag": sys.implementation.cache_tag,
                   "executable": executable_value},
        "distributions": {"torch": distribution("torch"), "rsl-rl-lib": distribution("rsl-rl-lib"),
                          "isaaclab": distribution("isaaclab")},
        "modules": {"torch": torch_module_version},
    }


def validate_isaacsim_raw_components(value: object, expected_version: str) -> bool:
    if (not isinstance(value, list) or len(value) != 8
            or any(type(item) is not str or not item for item in value)):
        return False
    release = expected_version.split(".")
    return (len(release) == 3 and value[0] == expected_version
            and value[2:5] == release and value[5] == "rc"
            and value[6].isdigit() and value[1] == f"rc.{value[6]}"
            and value[7].startswith("release."))


def _source_ref(function: Any) -> dict[str, Any]:
    source = inspect.getsourcefile(function)
    if not source:
        raise ValueError("target method source path is unavailable")
    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError("target method source is missing")
    return {"path": str(path), "sha256": S._sha256_file(path), "code_sha256": _code_sha(function)}


def _target_methods() -> dict[str, Any]:
    from isaaclab.actuators.actuator_pd import DCMotor
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.runners import OnPolicyRunner

    return {"F2": DCMotor._clip_effort, "F3": ActorCritic.act, "F4": OnPolicyRunner.__init__}


def _method_snapshot() -> dict[str, dict[str, Any]]:
    return {
        condition: {"identity": id(method), **_source_ref(method)}
        for condition, method in _target_methods().items()
    }


def validate_method_transition(before: object, after: object, condition: str,
                               contract: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected_targets = set(TARGET_METHODS)
    if (not isinstance(before, Mapping) or set(before) != expected_targets
            or not isinstance(after, Mapping) or set(after) != expected_targets
            or set(contract) != expected_targets):
        return ["method_target_set_mismatch"]
    row_keys = {"identity", "path", "sha256", "code_sha256"}
    for key in TARGET_METHODS:
        earlier, later = before[key], after[key]
        if (not isinstance(earlier, Mapping) or set(earlier) != row_keys
                or not isinstance(later, Mapping) or set(later) != row_keys):
            reasons.append(f"method_{key}_row_keys_mismatch")
            continue
        if any(type(row.get("identity")) is not int or row["identity"] <= 0
               or not isinstance(row.get("path"), str) or not row["path"]
               or not isinstance(row.get("sha256"), str)
               or not isinstance(row.get("code_sha256"), str)
               for row in (earlier, later)):
            reasons.append(f"method_{key}_row_value_invalid")
        expected_before = contract[key]["before"]
        expected_after = contract[key]["patched"] if key == condition else expected_before
        if ({"path": earlier.get("path"), "source_sha256": earlier.get("sha256"),
             "code_sha256": earlier.get("code_sha256")}
                != expected_before):
            reasons.append(f"method_{key}_before_binding_mismatch")
        if ({"path": later.get("path"), "source_sha256": later.get("sha256"),
             "code_sha256": later.get("code_sha256")}
                != expected_after):
            reasons.append(f"method_{key}_after_binding_mismatch")
        identity_changed = earlier.get("identity") != later.get("identity")
        code_changed = earlier.get("code_sha256") != later.get("code_sha256")
        source_changed = (earlier.get("path") != later.get("path")
                          or earlier.get("sha256") != later.get("sha256"))
        if key == condition:
            if not (identity_changed and code_changed and source_changed):
                reasons.append(f"method_{key}_patch_transition_mismatch")
        elif identity_changed or code_changed or source_changed or earlier != later:
            reasons.append(f"method_{key}_unexpected_cross_signal")
    return reasons


def _tensor_snapshot(actuator: Any, base: Any) -> dict[str, Any]:
    buffers = {}
    for name in STATE_BUFFER_FIELDS:
        if not hasattr(actuator, name):
            raise ValueError(f"missing actuator buffer {name}")
        value = getattr(actuator, name)
        try:
            detached = value.detach().cpu().clone()
            flat = detached.reshape(-1).tolist()
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid actuator buffer {name}") from error
        if list(getattr(value, "shape", ())) != [1, 12] or len(flat) != 12:
            raise ValueError(f"actuator buffer {name} must have shape [1, 12]")
        if any(type(item) not in (int, float) or isinstance(item, bool)
               or not math.isfinite(float(item)) for item in flat):
            raise ValueError(f"non-finite actuator buffer {name}")
        buffers[name] = {"identity": id(value), "shape": [1, 12],
                         "values": [float(item) for item in flat]}
    counters = (base.common_step_counter, base._sim_step_counter)
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("SDK counters must be non-negative integers")
    return {
        "schema": "actuator_envelope_state_v1",
        "buffers": buffers,
        "common_step_counter": base.common_step_counter,
        "sim_step_counter": base._sim_step_counter,
    }


def validate_state_snapshot(value: object) -> list[str]:
    reasons: list[str] = []
    if not isinstance(value, Mapping) or set(value) != STATE_KEYS:
        return ["state_schema_keys_mismatch"]
    if value.get("schema") != "actuator_envelope_state_v1":
        reasons.append("state_schema_mismatch")
    buffers = value.get("buffers")
    if not isinstance(buffers, Mapping) or set(buffers) != set(STATE_BUFFER_FIELDS):
        reasons.append("state_buffer_set_mismatch")
    else:
        for name in STATE_BUFFER_FIELDS:
            row = buffers[name]
            if not isinstance(row, Mapping) or set(row) != BUFFER_KEYS:
                reasons.append(f"state_{name}_keys_mismatch")
                continue
            if type(row.get("identity")) is not int or row["identity"] <= 0:
                reasons.append(f"state_{name}_identity_invalid")
            values = row.get("values")
            if (row.get("shape") != [1, 12] or not isinstance(values, list) or len(values) != 12
                    or any(type(item) not in (int, float) or isinstance(item, bool)
                           or not math.isfinite(float(item)) for item in values)):
                reasons.append(f"state_{name}_tensor_invalid")
    for name in ("common_step_counter", "sim_step_counter"):
        if type(value.get(name)) is not int or value[name] < 0:
            reasons.append(f"state_{name}_invalid")
    return reasons


def _restoration(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    checks = {name: before["buffers"][name] == after["buffers"][name]
              for name in STATE_BUFFER_FIELDS}
    checks["common_step_counter"] = before["common_step_counter"] == after["common_step_counter"]
    checks["sim_step_counter"] = before["sim_step_counter"] == after["sim_step_counter"]
    return {"checks": checks, "all_restored": all(checks.values())}


def validate_curve(result: object, expected_config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(result, Mapping):
        return ["helper_result_not_object"]
    if set(result) != HELPER_KEYS:
        reasons.append("helper_keyset_mismatch")
    if result.get("schema") != "actuator_envelope_readback_v1":
        reasons.append("helper_schema_mismatch")
    status = result.get("status")
    if status not in {"valid", "invalid"}:
        reasons.append("helper_status_invalid")
        return reasons
    if status == "invalid":
        if not isinstance(result.get("invalid_reason"), str) or not result["invalid_reason"]:
            reasons.append("invalid_reason_missing")
        if result.get("curve_matches_expected") is not None:
            reasons.append("invalid_curve_verdict_must_be_null")
        if result.get("limitations") != E.LIMITATIONS:
            reasons.append("helper_limitations_mismatch")
        return reasons
    if result.get("invalid_reason") is not None or type(result.get("curve_matches_expected")) is not bool:
        reasons.append("valid_verdict_malformed")
    requested = result.get("requested_efforts")
    magnitude = max(float(expected_config["effort_limit"]), float(expected_config["saturation_effort"])) * 1000.0
    if requested != {"positive": magnitude, "negative": -magnitude}:
        reasons.append("requested_effort_sign_or_magnitude_mismatch")
    provenance = result.get("config_provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("config_provenance_missing")
    else:
        if set(provenance) != PROVENANCE_KEYS:
            reasons.append("config_provenance_keyset_mismatch")
        if provenance.get("expected_config") != dict(expected_config):
            reasons.append("helper_expected_config_mismatch")
        if provenance.get("runtime_config_readback") != dict(expected_config):
            reasons.append("helper_runtime_config_mismatch")
        if provenance.get("velocity_grid_rad_s") != list(E.VELOCITY_GRID):
            reasons.append("helper_velocity_grid_binding_mismatch")
        if provenance.get("source_binding_verified") is not False:
            reasons.append("helper_must_not_claim_authenticated_source")
    if result.get("limitations") != E.LIMITATIONS:
        reasons.append("helper_limitations_mismatch")
    rows = result.get("measurements")
    if not isinstance(rows, list) or len(rows) != len(E.VELOCITY_GRID):
        reasons.append("measurement_row_count_mismatch")
        return reasons
    for index, velocity in enumerate(E.VELOCITY_GRID):
        row = rows[index]
        if not isinstance(row, Mapping) or set(row) != MEASUREMENT_KEYS:
            reasons.append(f"measurement_{index}_keyset_mismatch")
            continue
        if type(row.get("velocity_rad_s")) not in (int, float) or isinstance(row.get("velocity_rad_s"), bool) or row.get("velocity_rad_s") != velocity:
            reasons.append(f"measurement_{index}_velocity_mismatch")
            continue
        expected_minimum, expected_maximum = E._documented_limits(velocity, dict(expected_config))
        if (type(row.get("expected_min_effort")) not in (int, float)
                or type(row.get("expected_max_effort")) not in (int, float)
                or not math.isclose(float(row["expected_min_effort"]), expected_minimum,
                                    rel_tol=E.REL_TOL, abs_tol=E.ABS_TOL)
                or not math.isclose(float(row["expected_max_effort"]), expected_maximum,
                                    rel_tol=E.REL_TOL, abs_tol=E.ABS_TOL)):
            reasons.append(f"measurement_{index}_expected_limit_mismatch")
        row_match = True
        for key in ("measured_min_effort", "measured_max_effort"):
            values = row.get(key)
            if (not isinstance(values, list) or len(values) != 12
                    or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values)):
                reasons.append(f"measurement_{index}_{key}_invalid")
                row_match = False
                continue
            expected_value = expected_minimum if key == "measured_min_effort" else expected_maximum
            if not all(math.isclose(float(value), expected_value, rel_tol=E.REL_TOL, abs_tol=E.ABS_TOL)
                       for value in values):
                row_match = False
        if type(row.get("matches_expected")) is not bool or row["matches_expected"] is not row_match:
            reasons.append(f"measurement_{index}_matches_expected_mismatch")
    if isinstance(rows, list) and len(rows) == len(E.VELOCITY_GRID):
        aggregate = all(row.get("matches_expected") is True for row in rows if isinstance(row, Mapping))
        if result.get("curve_matches_expected") is not aggregate:
            reasons.append("curve_matches_expected_aggregate_mismatch")
    return reasons


def _runtime_config(actuator: Any) -> dict[str, float]:
    values = {
        "effort_limit": E._runtime_scalar(getattr(actuator, "effort_limit", None)),
        "saturation_effort": E._runtime_scalar(getattr(actuator, "_saturation_effort", None)),
        "velocity_limit": E._runtime_scalar(getattr(actuator, "velocity_limit", None)),
    }
    if any(value is None for value in values.values()):
        raise ValueError("live actuator config is not finite and uniform")
    return {key: float(value) for key, value in values.items()}


def run_sdk(validated: Mapping[str, Any], lifecycle: dict[str, Any] | None = None) -> tuple[dict[str, Any], Any]:
    import gymnasium as gym
    import torch
    from isaacsim.core.version import get_version
    from isaaclab.actuators.actuator_pd import DCMotor
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import hidden_faults

    protocol = validated["protocol"]
    condition = validated["manifest"]["condition"]
    version = list(get_version())
    if not validate_isaacsim_raw_components(version, protocol["isaacsim_version"]):
        raise ValueError("loaded Isaac Sim version differs from protocol")
    sim_version = ".".join(version[2:5])
    current_runtime = runtime_binding(
        torch_module=torch, isaaclab_root=validated["isaaclab_root"],
    )
    if current_runtime != protocol["runtime_binding"]:
        raise ValueError("loaded SDK Python or distribution versions differ from protocol")
    before_methods = _method_snapshot()
    hidden_faults._before_env(condition)
    after_methods = _method_snapshot()
    method_reasons = validate_method_transition(
        before_methods, after_methods, condition, protocol["method_binding_contract"],
    )
    if method_reasons:
        raise RuntimeError("fault method binding or cross-signal mismatch")
    changed = {key: key == condition for key in TARGET_METHODS}

    cfg = parse_env_cfg(TASK, device=protocol["device"], num_envs=protocol["num_envs"])
    cfg.seed = protocol["seed"]
    env = gym.make(TASK, cfg=cfg)
    if lifecycle is not None:
        lifecycle["env"] = env
    try:
        base = env.unwrapped
        robot = base.scene["robot"]
        actuator = robot.actuators["base_legs"]
        cls = type(actuator)
        if cls is not DCMotor:
            raise TypeError("base_legs is not the actual Isaac Lab DCMotor")
        joint_names = list(actuator.joint_names)
        shape = list(actuator._joint_vel.shape)
        if (len(joint_names) != 12 or any(type(name) is not str for name in joint_names)
                or set(joint_names) != GO2_JOINT_NAMES or shape != [1, 12]):
            raise ValueError("live Go2 actuator joint binding mismatch")
        runtime_config = _runtime_config(actuator)
        if runtime_config != protocol["expected_config"]:
            raise ValueError("live actuator config differs from AST-bound protocol config")
        before_state = _tensor_snapshot(actuator, base)
        helper = E.inspect_actuator_envelope(
            actuator,
            protocol["expected_config"],
            config_source_sha256=protocol["asset_binding"]["source_sha256"],
            tensor_module=torch,
        )
        after_state = _tensor_snapshot(actuator, base)
        restoration = _restoration(before_state, after_state)
        if not restoration["all_restored"]:
            raise RuntimeError("actuator or environment state changed during readback")
        validation_reasons = validate_curve(helper, protocol["expected_config"])
        measurement_status = "invalid" if helper.get("status") != "valid" or validation_reasons else "valid"
        curve_assessment = None if measurement_status == "invalid" else (
            "normal" if helper["curve_matches_expected"] else "curve_difference"
        )
        expected_match = protocol["development_expectations"][condition]
        protocol_status = (
            "invalid" if measurement_status == "invalid"
            else "expected" if helper["curve_matches_expected"] is expected_match
            else "unexpected"
        )
        class_source = _source_ref(cls._clip_effort)
        class_file = Path(inspect.getsourcefile(cls) or "").resolve()
        if not class_file.is_file():
            raise FileNotFoundError("DCMotor class source is missing")
        asset_path = (validated["isaaclab_root"] / GO2_SOURCE).resolve()
        if S._sha256_file(asset_path) != protocol["asset_binding"]["source_sha256"]:
            raise ValueError("fresh Go2 asset source hash mismatch")
        return {
            "sdk_versions": {"isaaclab_commit": ISAACLAB_COMMIT, "isaacsim": sim_version,
                             "isaacsim_raw_components": version,
                             "runtime_binding": current_runtime},
            "fault_method_bindings": {"before": before_methods, "after": after_methods, "changed": changed},
            "actuator_binding": {
                "class_name": cls.__name__, "class_module": cls.__module__,
                "clip_effort_source": class_source, "joint_names": joint_names,
                "class_source": {"path": str(class_file), "sha256": S._sha256_file(class_file)},
                "joint_count": len(joint_names), "joint_velocity_shape": shape,
                "device": str(actuator._joint_vel.device), "runtime_config": runtime_config,
                "asset_source": {"path": GO2_SOURCE, "sha256": S._sha256_file(asset_path)},
            },
            "state_before": before_state, "state_after": after_state, "state_restoration": restoration,
            "helper_result": helper, "measurement_validation_reasons": validation_reasons,
            "measurement_status": measurement_status, "curve_assessment": curve_assessment,
            "protocol_status": protocol_status,
        }, env
    except BaseException:
        raise


def _close_resources(env: Any, app: Any) -> None:
    first_error: BaseException | None = None
    if env is not None:
        try:
            env.close()
        except BaseException as error:
            first_error = error
    if app is not None:
        try:
            app.close()
        except BaseException as error:
            first_error = first_error or error
    if first_error is not None:
        raise first_error


def _launch_app(app_launcher_class: Any) -> Any:
    return app_launcher_class(headless=True, fast_shutdown=False).app


def _publish_success(path: Path, report: Mapping[str, Any], env: Any, app: Any) -> None:
    raw = (json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    _close_resources(env, app)
    S._publish_bytes_no_overwrite(path, raw)


def child_main(manifest: Path) -> int:
    import run_actuator_envelope_runtime as P

    validated = P.validate_manifest(manifest)
    identity = S.process_identity()
    from isaaclab.app import AppLauncher

    app = _launch_app(AppLauncher)
    lifecycle: dict[str, Any] = {}
    started = _utc_now()
    wall = time.perf_counter()
    try:
        result, env = run_sdk(validated, lifecycle)
        report = {
            "schema": "actuator_envelope_runtime_body_v1", "completed": True, **identity,
            "scope": validated["protocol"]["scope"], "task": TASK,
            "condition": validated["manifest"]["condition"],
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": S._sha256_file(validated["manifest_path"]),
            "project_commit_sha": validated["project_commit_sha"],
            "source_receipt": validated["source_receipt"],
            "request_lineage": validated["request_lineage"], **result,
            "policy_inference_used": False, "runner_created": False, "env_step_called": False,
            "cause_confirmed": False, "recovery_confirmed": False,
            "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True,
            "started_at": started, "finished_at": _utc_now(),
            "wall_elapsed_s": round(time.perf_counter() - wall, 6),
        }
        _publish_success(validated["paths"]["output_path"], report, env, app)
        lifecycle.clear()
        app = None
        print(json.dumps({"completed": True, "condition": report["condition"],
                          "measurement_status": report["measurement_status"],
                          "protocol_status": report["protocol_status"]}), flush=True)
    except BaseException:
        traceback.print_exc()
        try:
            _close_resources(lifecycle.get("env"), app)
        except BaseException:
            traceback.print_exc()
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    return child_main(args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
