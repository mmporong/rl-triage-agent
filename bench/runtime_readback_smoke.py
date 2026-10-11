"""고정된 stock Go2 실행에서 runtime readback API 결속만 확인한다.

이 스크립트는 진단 정확도, 정책 정상성, 원인, 회복을 판정하지 않는다. 부모 실행기가
검증한 단일 체크포인트를 20 control step 동안 실행하고, 완료된 ``env.step`` 반환값과
Isaac Lab SDK 상태를 :class:`RuntimeReadbackCollector`에 연결한 body report만 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
CHECKPOINT_SHA256 = "abc3223f0b8e4c14b93cb5db2f79b16d9cf63dff6dc142bc46822de89fb0a62f"
PROJECT_SOURCE_FILES = {
    "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py",
    "bench/runtime_readback_smoke.py",
    "src/rl_triage/runtime_observers.py",
    "src/rl_triage/grounded_evidence.py",
    "src/rl_triage/single_run.py",
    "src/rl_triage/triage_tools.py",
}
ISAACLAB_SOURCE_FILES = {
    "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
    "source/isaaclab/isaaclab/envs/manager_based_env.py",
    "source/isaaclab/isaaclab/assets/articulation/articulation_data.py",
    "source/isaaclab/isaaclab/managers/reward_manager.py",
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py",
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py",
    "source/isaaclab/isaaclab/app/app_launcher.py",
    "source/isaaclab/isaaclab/utils/io/yaml.py",
    "source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
}
REQUIRED_PROTOCOL_FIELDS = {
    "schema", "scope", "task", "seed", "num_envs", "completed_control_steps",
    "observed_env_indices", "device", "isaaclab_commit", "isaacsim_version",
    "checkpoint_sha256", "registry_sha256", "source_files", "isaaclab_source_files",
    "launcher_watchdog_s", "semantic_retries",
}
MANIFEST_FIELDS = {
    "schema", "protocol_path", "protocol_sha256", "checkpoint_path", "output_path",
    "configuration_path",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _relative_source_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must use a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label} must remain below its source root")
    path = (root / Path(*pure.parts)).resolve()
    resolved_root = root.resolve()
    if path == resolved_root or resolved_root not in path.parents:
        raise ValueError(f"{label} escapes its source root")
    return path


def _verify_source_map(root: Path, value: object, label: str, *, exact_paths: set[str] | None = None) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty path-to-SHA mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    if exact_paths is not None and set(value) != exact_paths:
        raise ValueError(f"{label} must bind exactly the fixed source file set")
    verified = {}
    for relative, expected in sorted(value.items()):
        digest = _require_sha(expected, f"{label}.{relative}")
        path = _relative_source_path(root, relative, f"{label}.{relative}")
        if not path.is_file():
            raise FileNotFoundError(f"bound source file is missing: {relative}")
        actual = _sha256_file(path)
        if actual != digest:
            raise ValueError(f"bound source SHA mismatch: {relative}")
        verified[relative] = actual
    return verified


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return completed.stdout.strip()


def _git_tracked_clean(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
        check=False, capture_output=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("git tracked-clean check failed")
    return completed.returncode == 0


def _windows_process_creation_filetime() -> int:
    if os.name != "nt":
        raise NotImplementedError("Windows process FILETIME is required by this SDK smoke contract")
    import ctypes
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    creation, exit_time, kernel_time, user_time = FileTime(), FileTime(), FileTime(), FileTime()
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(),
        ctypes.byref(creation), ctypes.byref(exit_time),
        ctypes.byref(kernel_time), ctypes.byref(user_time),
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
    return (int(creation.high) << 32) | int(creation.low)


def process_identity(*, creation_time_reader=None) -> dict:
    reader = _windows_process_creation_filetime if creation_time_reader is None else creation_time_reader
    creation_time = reader()
    if type(creation_time) is not int or creation_time <= 0:
        raise ValueError("process creation FILETIME must be a positive integer")
    return {
        "process_id": os.getpid(),
        "process_creation_time_filetime": creation_time,
    }


def _load_runtime_observer():
    from rl_triage.runtime_observers import REGISTRY_SHA256, RuntimeReadbackCollector

    return RuntimeReadbackCollector, REGISTRY_SHA256


def _validate_protocol(protocol: dict, *, repo_root: Path, isaaclab_root: Path) -> dict:
    missing = REQUIRED_PROTOCOL_FIELDS - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing required fields: {sorted(missing)}")
    expected = {
        "schema": "runtime_readback_smoke_protocol_v1",
        "scope": "actual_sdk_api_integration_not_diagnosis_or_recovery",
        "task": TASK,
        "seed": 20261011,
        "num_envs": 16,
        "completed_control_steps": 20,
        "observed_env_indices": [0, 1, 2, 3],
        "device": "cuda:0",
        "isaaclab_commit": ISAACLAB_COMMIT,
        "isaacsim_version": "4.5.0",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "launcher_watchdog_s": 360,
        "semantic_retries": 0,
    }
    for key, value in expected.items():
        if type(protocol.get(key)) is not type(value) or protocol.get(key) != value:
            raise ValueError(f"protocol field {key} does not match the fixed smoke contract")
    project_sources = _verify_source_map(
        repo_root, protocol["source_files"], "source_files", exact_paths=PROJECT_SOURCE_FILES,
    )
    sdk_sources = _verify_source_map(
        isaaclab_root, protocol["isaaclab_source_files"], "isaaclab_source_files",
        exact_paths=ISAACLAB_SOURCE_FILES,
    )
    if _git_head(isaaclab_root) != ISAACLAB_COMMIT:
        raise ValueError("Isaac Lab checkout does not match the pinned commit")
    if not _git_tracked_clean(isaaclab_root):
        raise ValueError("Isaac Lab checkout has tracked changes")
    _, registry_sha = _load_runtime_observer()
    if _require_sha(protocol["registry_sha256"], "registry_sha256") != registry_sha:
        raise ValueError("runtime readback registry SHA does not match the loaded source")
    task_binding = {
        "task": TASK,
        "isaaclab_commit": ISAACLAB_COMMIT,
        "isaaclab_source_files": sdk_sources,
    }
    return {
        "project_source_files": project_sources,
        "isaaclab_source_files": sdk_sources,
        "task_binding_sha256": _canonical_sha(task_binding),
        "task_binding_formula": "sha256(canonical_json({task,isaaclab_commit,isaaclab_source_files}))",
    }


def validate_manifest(manifest_path: Path, *, repo_root: Path = ROOT,
                      isaaclab_root: Path | None = None) -> dict:
    if not manifest_path.is_absolute():
        raise ValueError("manifest path must be absolute")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest must be an existing absolute file")
    private_root = (repo_root / "bench" / "private" / "runtime_readback").resolve()
    run_directory = manifest_path.parent
    if manifest_path.name != "manifest.json" or run_directory.parent != private_root:
        raise ValueError("manifest must be one tag below bench/private/runtime_readback")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_directory.name) is None:
        raise ValueError("runtime smoke tag contains unsupported characters")
    manifest = _read_json(manifest_path)
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("manifest must contain only the fixed runtime smoke fields")
    if manifest["schema"] != "runtime_readback_smoke_manifest_v1":
        raise ValueError("unsupported runtime smoke manifest schema")
    paths = {
        key: _absolute_path(manifest[key], key)
        for key in ("protocol_path", "checkpoint_path", "output_path", "configuration_path")
    }
    expected_protocol = (repo_root / "bench" / "protocols" / "runtime_readback_smoke_v1.json").resolve()
    if paths["protocol_path"].resolve() != expected_protocol:
        raise ValueError("manifest must bind the repository runtime readback smoke protocol")
    if paths["output_path"].parent.resolve() != run_directory or paths["output_path"].name != "body.json":
        raise ValueError("output_path must be body.json in the manifest tag directory")
    if (paths["configuration_path"].parent.resolve() != run_directory
            or paths["configuration_path"].name != "configuration.yaml"):
        raise ValueError("configuration_path must be configuration.yaml in the manifest tag directory")
    if paths["output_path"] == paths["configuration_path"]:
        raise ValueError("output_path and configuration_path must differ")
    if paths["configuration_path"].suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("configuration_path must be a YAML path")
    for key in ("protocol_path", "checkpoint_path"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key} must identify an existing file")
    for key in ("output_path", "configuration_path"):
        if paths[key].exists():
            raise FileExistsError(f"{key} already exists; smoke artifacts are never overwritten")
        if not paths[key].parent.is_dir():
            raise FileNotFoundError(f"{key} parent directory must already exist")
    protocol_sha = _require_sha(manifest["protocol_sha256"], "protocol_sha256")
    if _sha256_file(paths["protocol_path"]) != protocol_sha:
        raise ValueError("protocol SHA does not match the manifest")
    if _sha256_file(paths["checkpoint_path"]) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA does not match the fixed smoke protocol")
    protocol = _read_json(paths["protocol_path"])
    source_receipt = _validate_protocol(
        protocol, repo_root=repo_root,
        isaaclab_root=Path.cwd() if isaaclab_root is None else isaaclab_root,
    )
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "paths": paths,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "manifest_sha256": _sha256_file(manifest_path),
        "project_commit_sha": _git_head(repo_root),
        "source_receipt": source_receipt,
    }


def _to_list(value: object, label: str, expected_length: int) -> list:
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    operation = getattr(current, "tolist", None)
    if callable(operation):
        current = operation()
    if not isinstance(current, (list, tuple)) or len(current) != expected_length:
        raise ValueError(f"{label} must contain one value per environment")
    return list(current)


def _clock(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative SDK integer counter")
    return value


def _done(value: object, label: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{label} must be a boolean or binary integer")


def collect_window(env: object, policy, collector: object, protocol: Mapping[str, object]) -> dict:
    """Connect completed wrapper steps to the read-only collector without resetting the environment."""
    base = env.unwrapped
    num_envs = int(protocol["num_envs"])
    observed = tuple(protocol["observed_env_indices"])
    completed_steps = int(protocol["completed_control_steps"])
    decimation = _clock(base.cfg.decimation, "base.cfg.decimation")
    if decimation <= 0 or env.num_envs != num_envs or base.num_envs != num_envs:
        raise ValueError("runtime environment does not match the fixed protocol size")
    episode_ids = {index: 0 for index in observed}
    obs, _ = env.get_observations()
    clock_rows = []
    phase_counts = {"completed_control_no_reset": 0, "post_auto_reset": 0}
    for _ in range(completed_steps):
        control_before = _clock(base.common_step_counter, "base.common_step_counter")
        physics_before = _clock(base._sim_step_counter, "base._sim_step_counter")
        actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        control_after = _clock(base.common_step_counter, "base.common_step_counter")
        physics_after = _clock(base._sim_step_counter, "base._sim_step_counter")
        if control_after - control_before != 1:
            raise RuntimeError("SDK common_step_counter did not advance by one completed control step")
        if physics_after - physics_before != decimation:
            raise RuntimeError("SDK _sim_step_counter did not advance by cfg.decimation")
        reward_values = _to_list(rewards, "completed env.step rewards", num_envs)
        done_values = _to_list(dones, "RSL wrapper dones", num_envs)
        episode_metadata = {}
        completed_rewards = {}
        for index in observed:
            reset = _done(done_values[index], f"RSL wrapper done[{index}]")
            phase = "post_auto_reset" if reset else "completed_control_no_reset"
            episode_metadata[index] = {
                "episode_id": episode_ids[index], "reset_occurred": reset, "phase": phase,
            }
            completed_rewards[index] = reward_values[index]
            phase_counts[phase] += 1
        collector.observe_completed_control_sample(
            step_metadata={
                "control_step_index": control_after,
                "physics_step_index": physics_after,
            },
            episode_metadata=episode_metadata,
            completed_rewards=completed_rewards,
        )
        for index in observed:
            if _done(done_values[index], f"RSL wrapper done[{index}]"):
                episode_ids[index] += 1
        clock_rows.append({
            "control_before": control_before, "control_after": control_after,
            "physics_before": physics_before, "physics_after": physics_after,
            "control_delta": control_after - control_before,
            "physics_delta": physics_after - physics_before,
        })
    return {
        "clock_rows": clock_rows,
        "initial_control_step": clock_rows[0]["control_before"],
        "final_control_step": clock_rows[-1]["control_after"],
        "initial_physics_step": clock_rows[0]["physics_before"],
        "final_physics_step": clock_rows[-1]["physics_after"],
        "expected_control_delta": 1,
        "expected_physics_delta": decimation,
        "phase_counts_over_observed_environments": phase_counts,
        "episode_id_semantics": "completed_transition_episode; post_auto_reset state belongs to the next episode",
    }


def _publish_bytes_no_overwrite(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary artifact already exists: {temporary.name}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_configuration_snapshot(path: Path, env_cfg: object, agent_cfg: object,
                                  protocol: Mapping[str, object]) -> str:
    from isaaclab.utils import class_to_dict
    from isaaclab.utils.io import dump_yaml

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.yaml")
    if temporary.exists():
        raise FileExistsError(f"temporary configuration already exists: {temporary.name}")
    data = {
        "runtime_contract": {
            "task": protocol["task"], "seed": protocol["seed"],
            "num_envs": protocol["num_envs"], "device": protocol["device"],
            "completed_control_steps": protocol["completed_control_steps"],
            "observed_env_indices": protocol["observed_env_indices"],
        },
        "environment": class_to_dict(env_cfg),
        "agent": agent_cfg.to_dict(),
    }
    try:
        dump_yaml(str(temporary), data, sort_keys=True)
        raw = temporary.read_bytes()
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(raw)


def assess_smoke(readback: Mapping[str, object], clocks: Mapping[str, object],
                 protocol: Mapping[str, object]) -> dict:
    """Apply the protocol's API-integration gate without interpreting policy quality."""
    reasons = []
    expected_samples = int(protocol["completed_control_steps"])
    observed_envs = len(protocol["observed_env_indices"])
    expected_values = expected_samples * observed_envs
    scope = readback.get("scope")
    if not isinstance(scope, Mapping) or scope.get("completed_control_sample_count") != expected_samples:
        reasons.append("completed_control_sample_count_mismatch")
    clock_rows = clocks.get("clock_rows")
    if not isinstance(clock_rows, list) or len(clock_rows) != expected_samples:
        reasons.append("sdk_clock_row_count_mismatch")
    elif any(
        not isinstance(row, Mapping)
        or row.get("control_delta") != clocks.get("expected_control_delta")
        or row.get("physics_delta") != clocks.get("expected_physics_delta")
        for row in clock_rows
    ):
        reasons.append("sdk_clock_delta_mismatch")
    checks = readback.get("read_only_checks")
    expected_checks = {
        "runtime_com_state": {
            "com_position_z", "com_velocity_z", "body_com_local_position_max_abs",
        },
        "runtime_reward_window": {"reward_mean", "reward_min"},
        "runtime_contact_window": {"contact_force_max", "contact_event_fraction"},
    }
    if not isinstance(checks, Mapping) or set(checks) != set(expected_checks):
        reasons.append("registry_check_set_mismatch")
        checks = {}
    phase_counts = clocks.get("phase_counts_over_observed_environments")
    post_reset_count = (
        phase_counts.get("post_auto_reset")
        if isinstance(phase_counts, Mapping) and type(phase_counts.get("post_auto_reset")) is int
        else None
    )
    if post_reset_count is None or post_reset_count < 0 or post_reset_count > expected_values:
        reasons.append("post_auto_reset_count_invalid")
        post_reset_count = expected_values
    for check_id, measurement_ids in expected_checks.items():
        check = checks.get(check_id)
        measurements = check.get("measurements") if isinstance(check, Mapping) else None
        if not isinstance(measurements, Mapping) or set(measurements) != measurement_ids:
            reasons.append(f"{check_id}:measurement_set_mismatch")
            continue
        for measurement_id in sorted(measurement_ids):
            row = measurements[measurement_id]
            if not isinstance(row, Mapping) or row.get("expected_value_count") != expected_values:
                reasons.append(f"{measurement_id}:expected_count_mismatch")
                continue
            finite = row.get("finite_value_count")
            unavailable = row.get("unavailable_value_count")
            if type(finite) is not int or type(unavailable) is not int or finite + unavailable != expected_values:
                reasons.append(f"{measurement_id}:coverage_count_mismatch")
                continue
            reasons_by_kind = row.get("unavailable_reason_counts")
            if check_id == "runtime_contact_window":
                expected_finite = expected_values - post_reset_count
                expected_reasons = {} if post_reset_count == 0 else {"post_auto_reset_state": post_reset_count}
                if finite != expected_finite or unavailable != post_reset_count:
                    reasons.append(f"{measurement_id}:unexpected_contact_unavailable")
                if reasons_by_kind != expected_reasons:
                    reasons.append(f"{measurement_id}:contact_unavailable_reason_mismatch")
                if expected_finite <= 0:
                    reasons.append(f"{measurement_id}:no_non_reset_api_sample")
            elif finite != expected_values or unavailable != 0 or reasons_by_kind != {}:
                reasons.append(f"{measurement_id}:nonfinite_or_unavailable")
    return {
        "smoke_status": "pass" if not reasons else "invalid",
        "validation_reasons": reasons,
        "expected_completed_control_samples": expected_samples,
        "expected_value_count_per_measurement": expected_values,
        "post_auto_reset_value_count": post_reset_count,
        "known_post_auto_reset_contact_unavailable_is_preserved": True,
        "quality_or_recovery_judgment": False,
    }


def run_smoke(validated: dict) -> tuple[dict, object]:
    import gymnasium as gym
    import torch
    from isaacsim.core.version import get_version
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    protocol = validated["protocol"]
    paths = validated["paths"]
    version_components = get_version()
    actual_sim_version = ".".join(str(value) for value in version_components[2:5])
    if actual_sim_version != protocol["isaacsim_version"]:
        raise ValueError("loaded Isaac Sim version does not match the fixed protocol")
    env_cfg = parse_env_cfg(
        protocol["task"], device=protocol["device"], num_envs=protocol["num_envs"],
    )
    env_cfg.seed = protocol["seed"]
    agent_cfg = load_cfg_from_registry(protocol["task"], "rsl_rl_cfg_entry_point")
    agent_cfg.device = protocol["device"]
    configuration_sha = _write_configuration_snapshot(
        paths["configuration_path"], env_cfg, agent_cfg, protocol,
    )
    env = RslRlVecEnvWrapper(
        gym.make(protocol["task"], cfg=env_cfg), clip_actions=agent_cfg.clip_actions,
    )
    collector_class, _ = _load_runtime_observer()
    collector = collector_class(
        env.unwrapped,
        observed_env_indices=protocol["observed_env_indices"],
        max_completed_control_samples=protocol["completed_control_steps"],
        source_binding={
            "task_sha256": validated["source_receipt"]["task_binding_sha256"],
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "configuration_sha256": configuration_sha,
            "observer_sha256": protocol["source_files"]["src/rl_triage/runtime_observers.py"],
        },
    )
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(paths["checkpoint_path"]), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        with torch.no_grad():
            clocks = collect_window(env, policy, collector, protocol)
        readback = collector.aggregate()
        return {
            "configuration_sha256": configuration_sha,
            "isaacsim_version_components": list(version_components),
            "isaacsim_version": actual_sim_version,
            "clocks": clocks,
            "readback": readback,
        }, env
    except BaseException:
        env.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    validated = validate_manifest(args.manifest)
    identity = process_identity()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    started_at = _utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    env = None
    try:
        result, env = run_smoke(validated)
        readback = result["readback"]
        verdict = assess_smoke(readback, result["clocks"], validated["protocol"])
        report = {
            "schema": "runtime_readback_smoke_body_v1",
            "completed": True,
            **identity,
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": validated["manifest_sha256"],
            "project_commit_sha": validated["project_commit_sha"],
            "scope": validated["protocol"]["scope"],
            "task": TASK,
            "checkpoint": {
                "file": validated["paths"]["checkpoint_path"].name,
                "sha256": CHECKPOINT_SHA256,
            },
            "source_receipt": validated["source_receipt"],
            "configuration": {
                "file": validated["paths"]["configuration_path"].name,
                "sha256": result["configuration_sha256"],
            },
            "sdk_versions": {
                "isaaclab_commit": ISAACLAB_COMMIT,
                "isaacsim": result["isaacsim_version"],
                "isaacsim_raw_components": result["isaacsim_version_components"],
            },
            "measurement_status": readback["measurement_status"],
            "smoke_status": verdict["smoke_status"],
            "smoke_validation": verdict,
            "sdk_clock_receipt": result["clocks"],
            "collector_report": readback,
            "claims": {
                "api_integration_observed": True,
                "diagnosis_confirmed": False,
                "policy_health_confirmed": False,
                "cause_confirmed": False,
                "recovery_confirmed": False,
                "quality_comparison_performed": False,
            },
            "cleanup_confirmed": False,
            "cleanup_confirmation_required_from_parent_receipt": True,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_elapsed_s": round(time.perf_counter() - wall_start, 6),
            "process_cpu_s": round(time.process_time() - cpu_start, 6),
        }
        raw = (json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
        _publish_bytes_no_overwrite(validated["paths"]["output_path"], raw)
        print(json.dumps({
            "completed": True, "process_id": os.getpid(),
            "smoke_status": report["smoke_status"],
            "measurement_status": report["measurement_status"],
            "samples": readback["scope"]["completed_control_sample_count"],
        }), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
