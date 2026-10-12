"""동일 동결 정책의 3-arm 행동 평가를 frozen ``fixed_eval``로 실행하는 SDK 자식."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

import run_base_com_reset as B
import runtime_readback_smoke as S


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
FIXED_PROTOCOL = (ROOT / "bench/protocols/fixed_eval_v1.json").resolve()
ARMS = {"fault_before": "X1", "fault_sham": "X1", "callback_fixed": "X1_REF"}
EXPECTED_HORIZON = 1000
EXPECTED_DECIMATION = 4
EXPECTED_RESETS = 2
BODY_SCHEMA = "grounded_recovery_sdk_body_v1"
ISAACSIM_VERSION = "4.5.0"


class SdkExecutionError(Exception):
    """AppLauncher 뒤 실패를 앱 종료 전 부모 제어 흐름으로 넘긴다."""

    def __init__(self, cause: BaseException, app: object):
        super().__init__(str(cause))
        self.cause = cause
        self.app = app


def _loaded_isaacsim_version() -> str:
    from isaacsim.core.version import get_version

    components = get_version()
    if not isinstance(components, (list, tuple)) or len(components) < 5:
        raise ValueError("loaded Isaac Sim version components are unavailable")
    return ".".join(str(value) for value in components[2:5])


def _clock(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class FixedEvalInstrumentation:
    """기존 평가의 환경 생성·명시 reset·step 시계만 기록한다."""

    def __init__(self, *, horizon: int = EXPECTED_HORIZON, decimation: int = EXPECTED_DECIMATION):
        self.horizon = horizon
        self.decimation = decimation
        self.env_create_count = 0
        self.reset_calls = 0
        self.clock_rows: list[dict[str, int]] = []
        self.base = None

    def environment_created(self, env: object) -> object:
        self.env_create_count += 1
        if self.env_create_count != 1:
            raise RuntimeError("fixed evaluation created more than one environment")
        base = env.unwrapped
        self.base = base
        original_reset = base.reset

        def measured_reset(*args, **kwargs):
            self.reset_calls += 1
            if self.reset_calls > EXPECTED_RESETS:
                raise RuntimeError("fixed evaluation performed an extra explicit reset")
            return original_reset(*args, **kwargs)

        base.reset = measured_reset
        return env

    def measured_step(self, original_step, wrapper, actions):
        base = wrapper.unwrapped
        before_control = _clock(base.common_step_counter, "base.common_step_counter")
        before_physics = _clock(base._sim_step_counter, "base._sim_step_counter")
        result = original_step(wrapper, actions)
        after_control = _clock(base.common_step_counter, "base.common_step_counter")
        after_physics = _clock(base._sim_step_counter, "base._sim_step_counter")
        self.clock_rows.append({
            "control_before": before_control, "control_after": after_control,
            "physics_before": before_physics, "physics_after": after_physics,
        })
        if len(self.clock_rows) > self.horizon:
            raise RuntimeError("fixed evaluation exceeded its frozen horizon")
        return result

    def verify(self) -> dict:
        return validate_clock_trace(
            self.clock_rows, reset_calls=self.reset_calls, env_create_count=self.env_create_count,
            horizon=self.horizon, decimation=self.decimation,
        )


def validate_clock_trace(rows: object, *, reset_calls: object, env_create_count: object,
                         horizon: int = EXPECTED_HORIZON, decimation: int = EXPECTED_DECIMATION) -> dict:
    """0→horizon control, 0→horizon×decimation의 연속 시계 계약을 검사한다."""
    if not isinstance(rows, list) or len(rows) != horizon:
        raise ValueError("clock trace must contain exactly the frozen horizon")
    if reset_calls != EXPECTED_RESETS or isinstance(reset_calls, bool):
        raise ValueError("fixed evaluation must call explicit reset exactly twice")
    if env_create_count != 1 or isinstance(env_create_count, bool):
        raise ValueError("fixed evaluation must create exactly one environment")
    keys = {"control_before", "control_after", "physics_before", "physics_after"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("clock row schema differs from the fixed contract")
        expected = {
            "control_before": index, "control_after": index + 1,
            "physics_before": index * decimation, "physics_after": (index + 1) * decimation,
        }
        for key, value in expected.items():
            if _clock(row[key], f"clock_rows[{index}].{key}") != value:
                raise ValueError("SDK clocks are not continuous from the fixed origin")
    return {"rows": [dict(row) for row in rows], "decimation": decimation,
            "reset_calls": EXPECTED_RESETS}


def normalize_injection_readback(injection: object) -> dict:
    """SDK tuple 범위를 JSON 배열로 정규화한 뒤 원래 수치·타입 계약을 대조한다."""
    raw = json.dumps(injection, sort_keys=True, allow_nan=False)
    expected = json.dumps(B.EXPECTED_INJECTION_READBACK, sort_keys=True, allow_nan=False)
    if raw != expected:
        raise ValueError("injection readback differs from the fixed base COM contract")
    return json.loads(raw)


def validate_event_readback(injection: object, resolved: object) -> None:
    """X1/X1_REF 공통 상류 base_com reset 설정을 값으로 확인한다."""
    normalize_injection_readback(injection)
    if not isinstance(resolved, Mapping):
        raise ValueError("resolved base_com event readback is missing")
    if resolved.get("mode") != "reset":
        raise ValueError("resolved base_com event mode is not reset")
    params = resolved.get("params")
    com_range = params.get("com_range") if isinstance(params, Mapping) else None
    if not isinstance(com_range, Mapping) or set(com_range) != {"x", "y", "z"}:
        raise ValueError("resolved base_com range is missing")
    expected = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}
    for axis, limits in expected.items():
        actual = com_range[axis]
        if (not isinstance(actual, (list, tuple)) or len(actual) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in actual)
                or tuple(float(value) for value in actual) != limits):
            raise ValueError(f"resolved base_com {axis} range differs from the upstream default")


def _configuration_payload(base: object, injection: dict, class_to_dict) -> tuple[dict, dict]:
    resolved_cfg = base.event_manager.get_term_cfg("base_com")
    resolved = class_to_dict(resolved_cfg)
    validate_event_readback(injection, resolved)
    environment = class_to_dict(base.cfg)
    observations = class_to_dict(base.cfg.observations)
    original_asset = environment["events"]["base_com"]["params"]["asset_cfg"]
    target = validate_target_body_binding(base.scene["robot"].body_names, original_asset,
                                          resolved["params"]["asset_cfg"])
    return ({
        "environment": environment,
        "observation_configuration": observations,
        "injection_readback": injection,
        "resolved_event_readback": {
            "source": "base.event_manager.get_term_cfg('base_com')", "base_com": resolved,
        },
        "target_body_readback": target,
    }, observations)


def validate_target_body_binding(body_names: object, original_asset: object,
                                 resolved_asset: object) -> dict:
    """원본 selector, EventManager resolved ID, 실제 articulation body 목록을 하나로 결합한다."""
    if (not isinstance(body_names, list) or not body_names
            or any(not isinstance(name, str) or not name for name in body_names)
            or len(set(body_names)) != len(body_names) or body_names.count("base") != 1):
        raise ValueError("runtime robot body names must be unique strings with exactly one base")
    if not isinstance(original_asset, Mapping) or set(("name", "body_names", "body_ids")) - set(original_asset):
        raise ValueError("original base_com asset selector is missing")
    original_ids = original_asset["body_ids"]
    original_slice = (isinstance(original_ids, slice) and original_ids.start is None
                      and original_ids.stop is None and original_ids.step is None)
    if (original_asset["name"] != "robot" or original_asset["body_names"] != "base" or not original_slice):
        raise ValueError("original base_com selector differs from the stock unresolved selector")
    if not isinstance(resolved_asset, Mapping):
        raise ValueError("resolved base_com asset selector is missing")
    resolved_names, resolved_ids = resolved_asset.get("body_names"), resolved_asset.get("body_ids")
    if resolved_asset.get("name") != "robot" or resolved_names != ["base"]:
        raise ValueError("resolved base_com body name differs from base")
    if (not isinstance(resolved_ids, list) or len(resolved_ids) != 1
            or isinstance(resolved_ids[0], bool) or not isinstance(resolved_ids[0], int)
            or resolved_ids[0] < 0):
        raise ValueError("resolved base_com body_ids must contain one non-negative integer")
    base_index = body_names.index("base")
    if resolved_ids[0] != base_index:
        raise ValueError("resolved base_com body ID differs from the runtime base index")
    return {"source": "base.scene['robot'].body_names", "asset_name": "robot",
            "body_names": list(body_names), "base_body_index": base_index}


def _write_configuration(path: Path, base: object, injection: dict) -> tuple[str, str, dict, dict]:
    from isaaclab.utils import class_to_dict
    from isaaclab.utils.io import dump_yaml

    payload, _ = _configuration_payload(base, injection, class_to_dict)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.source.yaml")
    if temporary.exists() or path.exists():
        raise FileExistsError("configuration artifact already exists")
    try:
        dump_yaml(str(temporary), payload, sort_keys=True)
        raw = temporary.read_bytes()
        S._publish_bytes_no_overwrite(path, raw)
    finally:
        if temporary.exists():
            temporary.unlink()
    observation_sha = _observation_configuration_sha(path)
    stored = B._load_configuration_yaml(path)
    resolved, target = stored["resolved_event_readback"], stored["target_body_readback"]
    # SceneEntityCfg의 미지정 joint/tendon slice도 저장본과 같은 JSON 표현으로 결속한다.
    json.dumps({"resolved": resolved, "target": target}, allow_nan=False)
    return S._sha256_bytes(raw), observation_sha, resolved, target


def _observation_configuration_sha(path: Path) -> str:
    """실제로 발행한 YAML을 안전 loader로 다시 읽어 관측 설정 결합값을 만든다."""
    loaded = B._load_configuration_yaml(path)
    observations = loaded.get("observation_configuration")
    if not isinstance(observations, dict):
        raise ValueError("stored observation configuration is missing")
    return S._canonical_sha(observations)


def _write_json_new(path: Path, value: dict) -> None:
    raw = (json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    S._publish_bytes_no_overwrite(path, raw)


def _owned_paths(validated: dict) -> dict[str, Path]:
    paths = validated["paths"]
    report = Path(paths["report_path"])
    folder = report.parent
    owned = {
        "report": report, "body": Path(paths["body_path"]),
        "configuration": Path(paths["configuration_path"]),
        "pending": folder / "fixed_eval.pending.json", "jobs": folder / "jobs.json",
        "pending_tmp": folder / "fixed_eval.pending.tmp",
        "child_started": folder / "child_started.json", "child_failure": folder / "child_failure.json",
    }
    if any(path.parent.resolve() != folder.resolve() for path in owned.values()):
        raise ValueError("child artifacts must share one owned folder")
    return owned


def _verify_callback_identity(callback: str, external_faults) -> None:
    module_name, function_name, replacement = external_faults.SWAPS[callback]
    target = getattr(importlib.import_module(module_name), function_name)
    if target.__code__ is not replacement.__code__:
        raise ValueError("runtime callback code identity does not match the registered arm")


def _execute_fixed_eval(validated: dict, paths: dict[str, Path]) -> dict:
    import fixed_eval as F
    import gymnasium as gym
    import hidden_faults
    import external_faults
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    manifest = validated["manifest"]
    callback = manifest["callback"]
    hidden_faults._before_env(callback)
    _verify_callback_identity(callback, external_faults)
    tracker = FixedEvalInstrumentation()
    original_make = gym.make
    original_step = RslRlVecEnvWrapper.step
    runtime: dict[str, object] = {}

    def measured_make(*args, **kwargs):
        if tracker.env_create_count:
            raise RuntimeError("fixed evaluation attempted another environment creation")
        env = original_make(*args, **kwargs)
        try:
            env = tracker.environment_created(env)
            injection = normalize_injection_readback(hidden_faults._after_env(callback, env))
            config_sha, observation_sha, resolved, target = _write_configuration(
                validated["paths"]["configuration_path"], env.unwrapped, injection,
            )
            runtime.update({
                "configuration_sha256": config_sha,
                "observation_configuration_sha256": observation_sha,
                "injection_readback": injection,
                "resolved_event_readback": resolved,
                "target_body_readback": target,
            })
            return env
        except BaseException:
            env.close()
            raise

    def measured_step(wrapper, actions):
        return tracker.measured_step(original_step, wrapper, actions)

    gym.make = measured_make
    RslRlVecEnvWrapper.step = measured_step
    try:
        args = SimpleNamespace(
            protocol=validated["protocol"]["fixed_protocol_path"], jobs=paths["jobs"],
            device=validated["protocol"]["device"],
        )
        F.evaluate(args)
    finally:
        RslRlVecEnvWrapper.step = original_step
        gym.make = original_make
    clocks = tracker.verify()
    if set(runtime) != {
        "configuration_sha256", "observation_configuration_sha256", "injection_readback",
        "resolved_event_readback", "target_body_readback",
    }:
        raise RuntimeError("runtime configuration evidence was not collected")
    return {**runtime, "clocks": clocks, "env_create_count": tracker.env_create_count,
            "callback_code_identity_verified": True}


def run_sdk(validated: dict) -> tuple[dict, object]:
    """AppLauncher를 띄우고 frozen 평가를 한 번 실행한다. 성공 시 닫지 않은 app을 돌려준다."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        expected_sim = validated["protocol"].get("isaacsim_version")
        actual_sim = _loaded_isaacsim_version()
        if expected_sim != ISAACSIM_VERSION or actual_sim != expected_sim:
            raise ValueError("loaded Isaac Sim version does not match the fixed recovery protocol")
        paths = _owned_paths(validated)
        manifest = validated["manifest"]
        if manifest.get("arm") not in ARMS or manifest.get("callback") != ARMS.get(manifest.get("arm")):
            raise ValueError("arm and callback do not match the fixed three-arm contract")
        protocol = validated["protocol"]
        if protocol.get("device") != "cuda:0" or Path(protocol.get("fixed_protocol_path", "")).resolve() != FIXED_PROTOCOL:
            raise ValueError("child must use the fixed CUDA evaluation protocol")
        checkpoint = Path(manifest["checkpoint_path"])
        checkpoint_before = S._sha256_file(checkpoint)
        if checkpoint_before != manifest["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA differs before evaluation")
        if any(paths[key].exists() for key in (
                "pending", "pending_tmp", "report", "body", "jobs", "configuration", "child_failure")):
            raise FileExistsError("child output artifact already exists")
        jobs = {"action_scale": manifest["action_scale"], "jobs": [{
            "name": manifest["arm"], "checkpoint": str(checkpoint.resolve()), "output": str(paths["pending"].resolve()),
        }]}
        _write_json_new(paths["jobs"], jobs)
        result = _execute_fixed_eval(validated, paths)
        if S._sha256_file(checkpoint) != checkpoint_before:
            raise ValueError("checkpoint changed during evaluation")
        if not paths["pending"].is_file() or paths["report"].exists():
            raise RuntimeError("fixed evaluation staging report is missing or final report already exists")
        report_raw = paths["pending"].read_bytes()
        S._publish_bytes_no_overwrite(paths["report"], report_raw)
        return {**result, "isaacsim_version": actual_sim, "report_sha256": S._sha256_bytes(report_raw),
                "checkpoint_sha256": checkpoint_before}, app
    except BaseException as error:
        raise SdkExecutionError(error, app) from error


def child_main(manifest_path: Path) -> int:
    """부모 manifest를 재검증하고 body를 App 종료 전에 write-once로 기록한다."""
    import run_grounded_recovery as H

    validated = H.validate_manifest(manifest_path)
    paths = _owned_paths(validated)
    if any(paths[key].exists() for key in (
            "pending", "pending_tmp", "report", "body", "jobs", "child_started", "child_failure",
            "configuration")):
        raise FileExistsError("child artifact already exists; child runs are never resumed")
    identity = S.process_identity()
    H.write_new(paths["child_started"], {"schema": "grounded_recovery_child_started_v1", **identity,
                                         "manifest_sha256": validated["manifest_sha256"]})
    app = None
    try:
        result, app = run_sdk(validated)
        manifest = validated["manifest"]
        body = {
            "schema": BODY_SCHEMA, "completed": True,
            "arm": manifest["arm"], "callback": manifest["callback"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "isaacsim_version": result["isaacsim_version"],
            "configuration_sha256": result["configuration_sha256"],
            "observation_configuration_sha256": result["observation_configuration_sha256"],
            "callback_code_identity_verified": result["callback_code_identity_verified"],
            "injection_readback": result["injection_readback"],
            "resolved_event_readback": result["resolved_event_readback"],
            "target_body_readback": result["target_body_readback"],
            "clocks": result["clocks"], "env_create_count": result["env_create_count"],
            "process_identity": identity,
            "project_commit_sha": validated["project_commit_sha"],
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": validated["manifest_sha256"],
            "source_receipt": validated["source_receipt"],
            "report_sha256": result["report_sha256"],
            "cleanup_confirmed": False,
            "cleanup_confirmation_required_from_parent_receipt": True,
        }
        H.write_new(paths["body"], body)
    except SdkExecutionError as wrapped:
        app = wrapped.app
        H.write_new(paths["child_failure"], {"schema": "grounded_recovery_child_failure_v1",
                                             "error_type": type(wrapped.cause).__name__, **identity})
        traceback.print_exception(type(wrapped.cause), wrapped.cause, wrapped.cause.__traceback__)
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrapped.cause
    except BaseException as error:
        H.write_new(paths["child_failure"], {"schema": "grounded_recovery_child_failure_v1",
                                             "error_type": type(error).__name__, **identity})
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if app is not None:
            app.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    return child_main(args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
