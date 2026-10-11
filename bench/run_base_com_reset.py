"""고정된 X1 arm의 base COM reset readback을 자식 SDK 프로세스와 부모 영수증으로 기록한다."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Mapping

import record_evidence_media as M
import run_runtime_readback_smoke as P
import runtime_readback_smoke as S


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL = ROOT / "bench/protocols/base_com_reset_observation_v2.json"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ARMS = ("X1", "X1_REF")
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
PROJECT_SOURCE_FILES = {
    "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py",
    "bench/runtime_readback_smoke.py",
    "bench/run_base_com_reset.py",
    "bench/hidden_faults.py",
    "bench/external_faults.py",
    "src/rl_triage/reset_com_observer.py",
}
ISAACLAB_SOURCE_FILES = set(S.ISAACLAB_SOURCE_FILES) | {
    "source/isaaclab/isaaclab/envs/mdp/events.py",
    "source/isaaclab/isaaclab/assets/articulation/articulation.py",
    "source/isaaclab/isaaclab/managers/scene_entity_cfg.py",
}
RESULT_WORDING = {
    "pass": "고정한 arm과 SDK에서 base COM reset readback의 출처·coverage·종료를 확인했다. 원인 판정이나 회복 증거는 아니다.",
    "invalid": "실행·출처·reset coverage·종료 검증 중 하나가 실패했다. 원인 판정이나 회복 결과로 집계하지 않는다.",
}
EXCLUDED_CLAIMS = [
    "model_selected_measurement", "diagnosis_accuracy", "confirmed_cause",
    "policy_health", "policy_recovery", "gpu_savings",
]
EXPECTED_COM_RANGE = {
    "x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.01, 0.01],
}
EXPECTED_INJECTION_READBACK = {
    "changed": {
        "events.base_com": {"mode": "reset", "com_range": EXPECTED_COM_RANGE},
    },
    "checks": {"base_com_in_reset": True},
}
OBSERVER_MEASUREMENT_DEFINITIONS = {
    "base_com_max_abs_delta_xy": {"unit": "simulation_length_native"},
    "base_com_outside_range_fraction": {"unit": "fraction"},
}
OBSERVER_SAMPLE_FIELD_DEFINITIONS = {
    "delta_x": {"unit": "simulation_length_native", "scope": "reset_records.samples"},
    "delta_y": {"unit": "simulation_length_native", "scope": "reset_records.samples"},
}
OBSERVER_CALLER_PROVENANCE = {
    "baseline_capture_timing_claim": "after_environment_creation_before_first_reset",
    "baseline_capture_timing_verified_by_observer": False,
    "reset_completion_claim_source": "caller",
    "reset_completion_verified_by_observer": False,
}
EXECUTION_CONTRACT = {
    "arm_scope": "one_of_X1_or_X1_REF_per_process",
    "reset_call": "base._reset_idx(alternating_even_odd_env_ids)",
    "extra_env_reset": False,
    "policy_inference": False,
    "training": False,
    "cleanup": "parent_owned_windows_job_must_exit_naturally_and_be_empty",
}
FIXED_PROTOCOL_VALUES = {
    "schema": "base_com_reset_observation_protocol_v2",
    "scope": "actual_sdk_base_com_reset_mechanism_readback_not_diagnosis_or_recovery",
    "task": TASK,
    "seed": 20261012,
    "num_envs": 64,
    "reset_sequences": 20,
    "resets_per_env": 10,
    "reset_pattern": "alternating_even_odd",
    "device": "cuda:0",
    "allowed_xy_range": 0.05,
    "tolerance": 1.0e-4,
    "arms": list(ARMS),
    "isaaclab_commit": ISAACLAB_COMMIT,
    "isaacsim_version": "4.5.0",
    "launcher_watchdog_s": 360,
    "semantic_retries": 0,
    "model_selected": False,
    "gpu_admission_profile": "base_com_reset_readback_v2",
    "execution": EXECUTION_CONTRACT,
    "result_wording": RESULT_WORDING,
    "excluded_claims": EXCLUDED_CLAIMS,
    "source_binding_revision": 2,
}
PROTOCOL_FIELDS = set(FIXED_PROTOCOL_VALUES) | {
    "measurement_definitions_sha256", "source_files", "isaaclab_source_files",
}
MANIFEST_FIELDS = {
    "schema", "protocol_path", "protocol_sha256", "arm", "output_path", "configuration_path",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_observer():
    from rl_triage.reset_com_observer import BaseComResetObserver, MEASUREMENT_DEFINITIONS_SHA256

    return BaseComResetObserver, MEASUREMENT_DEFINITIONS_SHA256


def protocol_location(path: Path, repo_root: Path = ROOT) -> Path:
    resolved = path.resolve()
    expected = (repo_root / "bench/protocols/base_com_reset_observation_v2.json").resolve()
    if resolved != expected:
        raise ValueError("manifest must bind the fixed repository base COM reset protocol")
    return resolved


def validate_protocol(protocol: dict, *, repo_root: Path, isaaclab_root: Path) -> dict:
    if set(protocol) != PROTOCOL_FIELDS:
        raise ValueError("protocol fields differ from the fixed base COM reset contract")
    for key, expected in FIXED_PROTOCOL_VALUES.items():
        if type(protocol.get(key)) is not type(expected) or protocol.get(key) != expected:
            raise ValueError(f"protocol field {key} does not match the fixed contract")
    project_sources = S._verify_source_map(
        repo_root, protocol["source_files"], "source_files", exact_paths=PROJECT_SOURCE_FILES,
    )
    sdk_sources = S._verify_source_map(
        isaaclab_root, protocol["isaaclab_source_files"], "isaaclab_source_files",
        exact_paths=ISAACLAB_SOURCE_FILES,
    )
    if S._git_head(isaaclab_root) != ISAACLAB_COMMIT:
        raise ValueError("Isaac Lab checkout does not match the pinned commit")
    if not S._git_tracked_clean(isaaclab_root):
        raise ValueError("Isaac Lab checkout has tracked changes")
    _, measurement_sha = _load_observer()
    if S._require_sha(protocol["measurement_definitions_sha256"], "measurement_definitions_sha256") != measurement_sha:
        raise ValueError("base COM measurement definition SHA does not match the loaded observer")
    sdk_binding_sha = S._canonical_sha({
        "isaaclab_commit": ISAACLAB_COMMIT, "isaaclab_source_files": sdk_sources,
    })
    task_binding_sha = S._canonical_sha({
        "task": TASK, "isaaclab_commit": ISAACLAB_COMMIT,
        "isaaclab_source_files": sdk_sources,
    })
    return {
        "project_source_files": project_sources,
        "isaaclab_source_files": sdk_sources,
        "isaaclab_source_binding_sha256": sdk_binding_sha,
        "task_binding_sha256": task_binding_sha,
    }


def validate_manifest(manifest_path: Path, *, repo_root: Path = ROOT,
                      isaaclab_root: Path | None = None) -> dict:
    if not manifest_path.is_absolute():
        raise ValueError("manifest path must be absolute")
    manifest_path = manifest_path.resolve()
    private_root = (repo_root / "bench/private/base_com_reset").resolve()
    run_directory = manifest_path.parent
    if (not manifest_path.is_file() or manifest_path.name != "manifest.json"
            or run_directory.parent != private_root):
        raise ValueError("manifest must be one tag below bench/private/base_com_reset")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_directory.name) is None:
        raise ValueError("base COM reset tag contains unsupported characters")
    manifest = S._read_json(manifest_path)
    if set(manifest) != MANIFEST_FIELDS or manifest.get("schema") != "base_com_reset_manifest_v1":
        raise ValueError("unsupported base COM reset manifest contract")
    if manifest.get("arm") not in ARMS:
        raise ValueError("manifest arm must be X1 or X1_REF")
    paths = {
        key: S._absolute_path(manifest[key], key)
        for key in ("protocol_path", "output_path", "configuration_path")
    }
    protocol_location(paths["protocol_path"], repo_root)
    expected_names = {"output_path": "body.json", "configuration_path": "configuration.yaml"}
    for key, name in expected_names.items():
        if paths[key].parent.resolve() != run_directory or paths[key].name != name:
            raise ValueError(f"{key} must be {name} in the manifest tag directory")
        if paths[key].exists():
            raise FileExistsError(f"{key} already exists; artifacts are never overwritten")
    if not paths["protocol_path"].is_file():
        raise FileNotFoundError("protocol_path must identify an existing file")
    protocol_sha = S._require_sha(manifest["protocol_sha256"], "protocol_sha256")
    if S._sha256_file(paths["protocol_path"]) != protocol_sha:
        raise ValueError("protocol SHA does not match the manifest")
    protocol = S._read_json(paths["protocol_path"])
    source_receipt = validate_protocol(
        protocol, repo_root=repo_root,
        isaaclab_root=Path.cwd() if isaaclab_root is None else isaaclab_root,
    )
    return {
        "manifest_path": manifest_path, "manifest": manifest, "paths": paths,
        "protocol": protocol, "protocol_sha256": protocol_sha,
        "manifest_sha256": S._sha256_file(manifest_path),
        "project_commit_sha": S._git_head(repo_root), "source_receipt": source_receipt,
    }


def _finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _clock(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _cpu_list(value: object, label: str, expected_length: int) -> list:
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


def _optional_buffer(base: object, name: str, num_envs: int) -> list | None:
    value = getattr(base, name, None)
    return None if value is None else _cpu_list(value, name, num_envs)


def _reset_buffer_state(base: object, num_envs: int) -> tuple[bool, list | None]:
    present = hasattr(base, "reset_buf")
    return present, _optional_buffer(base, "reset_buf", num_envs) if present else None


def collect_resets(base: object, observer: object, torch_module: object,
                   protocol: Mapping[str, object]) -> dict:
    num_envs = int(protocol["num_envs"])
    sequences = int(protocol["reset_sequences"])
    if getattr(base, "num_envs", None) != num_envs:
        raise ValueError("runtime environment does not match the fixed protocol size")
    active = getattr(base.event_manager, "active_terms", None)
    active_reset_terms = active.get("reset") if isinstance(active, Mapping) else None
    if (not isinstance(active_reset_terms, (list, tuple))
            or "base_com" not in active_reset_terms
            or not all(type(term) is str for term in active_reset_terms)):
        raise ValueError("base_com is not active in the reset event manager")
    rows = []
    initial_reset_present, initial_reset_buffer = _reset_buffer_state(base, num_envs)
    initial = {
        "control_step": _clock(base.common_step_counter, "base.common_step_counter"),
        "physics_step": _clock(base._sim_step_counter, "base._sim_step_counter"),
        "episode_length": _optional_buffer(base, "episode_length_buf", num_envs),
        "reset_buffer_attribute_present": initial_reset_present,
        "reset_buffer": initial_reset_buffer,
    }
    for sequence in range(sequences):
        expected_ids = list(range(sequence % 2, num_envs, 2))
        env_ids = torch_module.arange(sequence % 2, num_envs, 2, device=base.device)
        sdk_reset_env_ids = _cpu_list(env_ids, "sdk_reset_env_ids", len(expected_ids))
        if sdk_reset_env_ids != expected_ids:
            raise ValueError("SDK reset IDs do not match the fixed alternating pattern")
        before_control = _clock(base.common_step_counter, "base.common_step_counter")
        before_physics = _clock(base._sim_step_counter, "base._sim_step_counter")
        episode_before = _optional_buffer(base, "episode_length_buf", num_envs)
        reset_present_before, reset_before = _reset_buffer_state(base, num_envs)
        base._reset_idx(env_ids)
        after_control = _clock(base.common_step_counter, "base.common_step_counter")
        after_physics = _clock(base._sim_step_counter, "base._sim_step_counter")
        reset_present_after, reset_after = _reset_buffer_state(base, num_envs)
        record = observer.record_completed_reset(
            reset_sequence=sequence, reset_env_ids=expected_ids,
        )
        rows.append({
            "reset_sequence": sequence, "expected_env_ids": expected_ids,
            "sdk_reset_env_ids": sdk_reset_env_ids,
            "observer_reset_env_ids": record.get("reset_env_ids"),
            "control_before": before_control, "control_after": after_control,
            "physics_before": before_physics, "physics_after": after_physics,
            "control_delta": after_control - before_control,
            "physics_delta": after_physics - before_physics,
            "episode_length_before": episode_before,
            "episode_length_after": _optional_buffer(base, "episode_length_buf", num_envs),
            "reset_buffer_attribute_present_before": reset_present_before,
            "reset_buffer_attribute_present_after": reset_present_after,
            "reset_buffer_before": reset_before,
            "reset_buffer_after": reset_after,
        })
    final_reset_present, final_reset_buffer = _reset_buffer_state(base, num_envs)
    return {
        "phase": "caller_completed_reset_readback",
        "base_com_active_reset_term": True,
        "active_reset_terms": list(active_reset_terms),
        "initial_sdk_state": initial,
        "reset_rows": rows,
        "final_sdk_state": {
            "control_step": _clock(base.common_step_counter, "base.common_step_counter"),
            "physics_step": _clock(base._sim_step_counter, "base._sim_step_counter"),
            "episode_length": _optional_buffer(base, "episode_length_buf", num_envs),
            "reset_buffer_attribute_present": final_reset_present,
            "reset_buffer": final_reset_buffer,
        },
    }


def _write_configuration_snapshot(path: Path, base: object, injection: dict,
                                  protocol: Mapping[str, object]) -> str:
    from isaaclab.utils import class_to_dict
    from isaaclab.utils.io import dump_yaml

    temporary = path.with_name(f".{path.name}.{os.getpid()}.source.yaml")
    if temporary.exists():
        raise FileExistsError("configuration temporary file already exists")
    data = {
        "runtime_contract": {
            key: protocol[key] for key in (
                "task", "seed", "num_envs", "reset_sequences", "resets_per_env",
                "reset_pattern", "device", "allowed_xy_range", "tolerance",
            )
        },
        "injection_readback": injection,
        "environment": class_to_dict(base.cfg),
    }
    try:
        dump_yaml(str(temporary), data, sort_keys=True)
        raw = temporary.read_bytes()
        S._publish_bytes_no_overwrite(path, raw)
    finally:
        if temporary.exists():
            temporary.unlink()
    return S._sha256_bytes(raw)


def _run_sdk(validated: dict) -> tuple[dict, object]:
    import gymnasium as gym
    import torch
    from isaacsim.core.version import get_version
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import hidden_faults

    protocol = validated["protocol"]
    arm = validated["manifest"]["arm"]
    versions = get_version()
    actual_sim = ".".join(str(value) for value in versions[2:5])
    if actual_sim != protocol["isaacsim_version"]:
        raise ValueError("loaded Isaac Sim version does not match the fixed protocol")
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    env_cfg = parse_env_cfg(protocol["task"], device=protocol["device"], num_envs=protocol["num_envs"])
    env_cfg.seed = protocol["seed"]
    hidden_faults._before_env(arm)
    env = gym.make(protocol["task"], cfg=env_cfg)
    try:
        injection = hidden_faults._after_env(arm, env)
        base = env.unwrapped
        configuration_sha = _write_configuration_snapshot(
            validated["paths"]["configuration_path"], base, injection, protocol,
        )
        observer_class, _ = _load_observer()
        observer = observer_class(
            base,
            source_binding={
                "task_sha256": validated["source_receipt"]["task_binding_sha256"],
                "configuration_sha256": configuration_sha,
                "isaaclab_sha256": validated["source_receipt"]["isaaclab_source_binding_sha256"],
                "observer_sha256": protocol["source_files"]["src/rl_triage/reset_com_observer.py"],
            },
            expected_reset_sequences=protocol["reset_sequences"],
            expected_resets_per_env=protocol["resets_per_env"],
            allowed_xy_range=protocol["allowed_xy_range"], tolerance=protocol["tolerance"],
        )
        phase = collect_resets(base, observer, torch, protocol)
        return {
            "configuration_sha256": configuration_sha,
            "injection_readback": injection,
            "isaacsim_version": actual_sim,
            "isaacsim_raw_components": list(versions),
            "phase_provenance": phase,
            "observer_report": observer.aggregate(),
        }, env
    except BaseException:
        env.close()
        raise


def _strict_number(value: object, label: str) -> float:
    if not _finite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _load_configuration_yaml(path: Path) -> dict:
    import yaml

    class ConfigurationLoader(yaml.SafeLoader):
        pass

    ConfigurationLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: loader.construct_sequence(node, deep=True),
    )
    ConfigurationLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        lambda loader, node: loader.construct_sequence(node, deep=True),
    )
    loader = ConfigurationLoader(path.read_text(encoding="utf-8"))
    try:
        loaded = loader.get_single_data()
    finally:
        loader.dispose()
    if not isinstance(loaded, dict):
        raise ValueError("configuration YAML root must be a mapping")
    return loaded


def verify_configuration_snapshot(path: Path, protocol: Mapping[str, object],
                                  injection_readback: object,
                                  observer_report: object) -> list[str]:
    reasons = []
    try:
        config = _load_configuration_yaml(path)
    except Exception:
        return ["configuration_yaml_invalid"]
    expected_runtime = {
        key: protocol[key] for key in (
            "task", "seed", "num_envs", "reset_sequences", "resets_per_env",
            "reset_pattern", "device", "allowed_xy_range", "tolerance",
        )
    }
    if config.get("runtime_contract") != expected_runtime:
        reasons.append("configuration_runtime_contract_mismatch")
    if injection_readback != EXPECTED_INJECTION_READBACK:
        reasons.append("body_injection_readback_mismatch")
    if config.get("injection_readback") != EXPECTED_INJECTION_READBACK:
        reasons.append("configuration_injection_readback_mismatch")
    try:
        term = config["environment"]["events"]["base_com"]
        params = term["params"]
        asset = params["asset_cfg"]
        body_ids = asset["body_ids"]
        body_binding = observer_report["body_binding"]
        observer_base_index = body_binding["base_body_index"]
        if (term["mode"] != "reset" or params["com_range"] != EXPECTED_COM_RANGE
                or asset["name"] != "robot" or asset["body_names"] != ["base"]
                or not isinstance(body_ids, list) or len(body_ids) != 1
                or type(body_ids[0]) is not int or body_ids[0] < 0
                or type(observer_base_index) is not int
                or body_ids[0] != observer_base_index):
            raise ValueError
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("configuration_base_com_contract_mismatch")
    return reasons


def _strict_int_buffer(value: object, num_envs: int) -> bool:
    return (isinstance(value, list) and len(value) == num_envs
            and all(type(item) is int and item >= 0 for item in value))


def _validate_baseline_snapshot(snapshot: object, num_envs: int) -> tuple[dict | None, list[str]]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != {"body_names", "shape", "coms"}:
        return None, ["baseline_snapshot_contract_mismatch"]
    names, shape, coms = snapshot.get("body_names"), snapshot.get("shape"), snapshot.get("coms")
    if (not isinstance(names, list) or not names
            or any(type(name) is not str or not name for name in names)
            or names.count("base") != 1
            or shape != [num_envs, len(names), 7]
            or not isinstance(coms, list) or len(coms) != num_envs):
        return None, ["baseline_snapshot_contract_mismatch"]
    try:
        for env_row in coms:
            if not isinstance(env_row, list) or len(env_row) != len(names):
                raise ValueError
            for body_row in env_row:
                if (not isinstance(body_row, list) or len(body_row) != 7
                        or not all(_finite(value) for value in body_row)):
                    raise ValueError
    except (TypeError, ValueError):
        return None, ["baseline_snapshot_contract_mismatch"]
    return {"body_names": names, "shape": shape, "coms": coms}, []


def verify_phase_provenance(phase: object, protocol: Mapping[str, object]) -> list[str]:
    if not isinstance(phase, Mapping):
        return ["phase_provenance_mismatch"]
    rows = phase.get("reset_rows")
    expected_phase_fields = {
        "phase", "base_com_active_reset_term", "active_reset_terms",
        "initial_sdk_state", "reset_rows", "final_sdk_state",
    }
    if (set(phase) != expected_phase_fields
            or phase.get("phase") != "caller_completed_reset_readback"
            or phase.get("base_com_active_reset_term") is not True
            or not isinstance(phase.get("active_reset_terms"), list)
            or "base_com" not in phase["active_reset_terms"]
            or not all(type(term) is str for term in phase["active_reset_terms"])
            or not isinstance(rows, list)
            or len(rows) != protocol["reset_sequences"]):
        return ["phase_provenance_mismatch"]
    num_envs = int(protocol["num_envs"])
    states = []
    for state_name in ("initial_sdk_state", "final_sdk_state"):
        state = phase.get(state_name)
        if (not isinstance(state, Mapping)
                or set(state) != {
                    "control_step", "physics_step", "episode_length",
                    "reset_buffer_attribute_present", "reset_buffer",
                }
                or type(state.get("control_step")) is not int or state["control_step"] < 0
                or type(state.get("physics_step")) is not int or state["physics_step"] < 0
                or not _strict_int_buffer(state.get("episode_length"), num_envs)
                or state.get("reset_buffer_attribute_present") is not False
                or state.get("reset_buffer") is not None):
            return ["phase_sdk_state_mismatch"]
        states.append(state)
    initial, final = states
    previous_episode = initial["episode_length"]
    expected_row_fields = {
        "reset_sequence", "expected_env_ids", "sdk_reset_env_ids", "observer_reset_env_ids",
        "control_before", "control_after", "physics_before", "physics_after",
        "control_delta", "physics_delta", "episode_length_before", "episode_length_after",
        "reset_buffer_attribute_present_before", "reset_buffer_attribute_present_after",
        "reset_buffer_before", "reset_buffer_after",
    }
    for sequence, row in enumerate(rows):
        expected_ids = list(range(sequence % 2, num_envs, 2))
        if (not isinstance(row, Mapping) or set(row) != expected_row_fields
                or row.get("reset_sequence") != sequence
                or row.get("expected_env_ids") != expected_ids
                or row.get("sdk_reset_env_ids") != expected_ids
                or row.get("observer_reset_env_ids") != expected_ids
                or row.get("control_before") != initial["control_step"]
                or row.get("control_after") != initial["control_step"]
                or row.get("control_delta") != 0
                or row.get("physics_before") != initial["physics_step"]
                or row.get("physics_after") != initial["physics_step"]
                or row.get("physics_delta") != 0
                or not _strict_int_buffer(row.get("episode_length_before"), num_envs)
                or not _strict_int_buffer(row.get("episode_length_after"), num_envs)
                or row.get("reset_buffer_attribute_present_before") is not False
                or row.get("reset_buffer_attribute_present_after") is not False
                or row.get("reset_buffer_before") is not None
                or row.get("reset_buffer_after") is not None
                or row["episode_length_before"] != previous_episode
                ):
            return ["phase_reset_sequence_mismatch"]
        selected = set(expected_ids)
        for env_id, (before, after) in enumerate(zip(
                row["episode_length_before"], row["episode_length_after"], strict=True)):
            if (env_id in selected and after != 0) or (env_id not in selected and after != before):
                return ["phase_reset_sequence_mismatch"]
        previous_episode = row["episode_length_after"]
    if (final["control_step"] != initial["control_step"]
            or final["physics_step"] != initial["physics_step"]
            or final["episode_length"] != previous_episode):
        return ["phase_final_state_mismatch"]
    return []


def verify_observer_report(report: object, protocol: Mapping[str, object]) -> list[str]:
    reasons = []
    if not isinstance(report, Mapping):
        return ["observer_report_not_object"]
    num_envs = int(protocol["num_envs"])
    sequences = int(protocol["reset_sequences"])
    resets_per_env = int(protocol["resets_per_env"])
    threshold = float(protocol["allowed_xy_range"]) + float(protocol["tolerance"])
    expected_fields = {
        "schema", "measurement_status", "source_binding", "body_binding",
        "measurement_definitions", "sample_field_definitions", "caller_provenance",
        "threshold", "coverage", "baseline_snapshot", "baseline_base_xy",
        "per_environment", "measurements", "reset_records", "cause_confirmed",
        "health_confirmed", "recovery_confirmed", "cleanup_confirmed", "model_selected",
    }
    if set(report) != expected_fields:
        reasons.append("observer_report_fields_mismatch")
    if report.get("schema") != "base_com_reset_observer_v1" or report.get("measurement_status") != "complete":
        reasons.append("observer_schema_or_status_mismatch")
    snapshot, snapshot_reasons = _validate_baseline_snapshot(
        report.get("baseline_snapshot"), num_envs,
    )
    reasons.extend(snapshot_reasons)
    baseline_map = {}
    if snapshot is not None:
        base_index = snapshot["body_names"].index("base")
        expected_body_binding = {
            "body_names": snapshot["body_names"], "base_body_index": base_index,
            "baseline_shape": snapshot["shape"],
            "read_source": "robot.root_physx_view.get_coms()",
            "cached_body_com_pose_b_used": False,
        }
        if report.get("body_binding") != expected_body_binding:
            reasons.append("observer_body_binding_mismatch")
        expected_baselines = [{
            "env_id": env_id,
            "x": snapshot["coms"][env_id][base_index][0],
            "y": snapshot["coms"][env_id][base_index][1],
        } for env_id in range(num_envs)]
        if report.get("baseline_base_xy") != expected_baselines:
            reasons.append("baseline_rows_snapshot_mismatch")
        baseline_map = {
            row["env_id"]: (float(row["x"]), float(row["y"])) for row in expected_baselines
        }
    if report.get("measurement_definitions") != OBSERVER_MEASUREMENT_DEFINITIONS:
        reasons.append("observer_measurement_definitions_mismatch")
    if report.get("sample_field_definitions") != OBSERVER_SAMPLE_FIELD_DEFINITIONS:
        reasons.append("observer_sample_field_definitions_mismatch")
    if report.get("caller_provenance") != OBSERVER_CALLER_PROVENANCE:
        reasons.append("observer_caller_provenance_mismatch")
    expected_threshold = {
        "allowed_xy_range": float(protocol["allowed_xy_range"]),
        "tolerance": float(protocol["tolerance"]),
        "outside_rule": "max_abs_delta_xy_strictly_greater_than_range_plus_tolerance",
    }
    if report.get("threshold") != expected_threshold:
        reasons.append("observer_threshold_mismatch")
    coverage = report.get("coverage")
    expected_coverage = {
        "expected_reset_sequence_count": sequences,
        "observed_reset_sequence_count": sequences,
        "expected_resets_per_env": resets_per_env,
        "expected_reset_measurement_count": num_envs * resets_per_env,
        "finite_reset_measurement_count": num_envs * resets_per_env,
        "unavailable_reset_sequences": [], "status": "complete",
    }
    if coverage != expected_coverage:
        reasons.append("observer_coverage_mismatch")
    records = report.get("reset_records")
    if not isinstance(records, list) or len(records) != sequences:
        reasons.append("reset_record_count_mismatch")
        records = []
    if len(baseline_map) != num_envs:
        reasons.append("baseline_count_mismatch")
    per_env_samples = {env_id: [] for env_id in range(num_envs)}
    try:
        for sequence, record in enumerate(records):
            expected_ids = list(range(sequence % 2, num_envs, 2))
            if (record.get("reset_sequence") != sequence or record.get("reset_env_ids") != expected_ids
                    or record.get("measurement_state") != "present"):
                raise ValueError
            samples = record.get("samples")
            if not isinstance(samples, list) or [sample.get("env_id") for sample in samples] != expected_ids:
                raise ValueError
            for sample in samples:
                env_id = sample["env_id"]
                bx, by = baseline_map[env_id]
                current_x = _strict_number(sample["current_x"], "sample.current_x")
                current_y = _strict_number(sample["current_y"], "sample.current_y")
                delta_x, delta_y = current_x - bx, current_y - by
                worst = max(abs(delta_x), abs(delta_y))
                if not all(_finite(value) for value in (delta_x, delta_y, worst)):
                    raise ValueError
                expected = {
                    "env_id": env_id, "baseline_x": bx, "baseline_y": by,
                    "current_x": current_x, "current_y": current_y,
                    "delta_x": delta_x, "delta_y": delta_y,
                    "max_abs_delta_xy": worst,
                    "outside_allowed_range": worst > threshold,
                }
                if sample != expected:
                    raise ValueError
                per_env_samples[env_id].append(sample)
    except (KeyError, TypeError, ValueError):
        reasons.append("reset_raw_reaggregation_failed")
        per_env_samples = {env_id: [] for env_id in range(num_envs)}
    expected_worst = []
    if all(len(rows) == resets_per_env for rows in per_env_samples.values()):
        expected_worst = [max(row["max_abs_delta_xy"] for row in per_env_samples[env_id])
                          for env_id in range(num_envs)]
    else:
        reasons.append("per_environment_reset_count_mismatch")
    per_environment = report.get("per_environment")
    if expected_worst:
        try:
            if not isinstance(per_environment, list) or len(per_environment) != num_envs:
                raise ValueError
            for env_id, row in enumerate(per_environment):
                bx, by = baseline_map[env_id]
                expected = {
                    "env_id": env_id, "baseline_x": bx, "baseline_y": by,
                    "declared_reset_count": resets_per_env,
                    "finite_reset_measurement_count": resets_per_env,
                    "worst_max_abs_delta_xy": expected_worst[env_id],
                    "measurement_state": "present",
                }
                if row != expected:
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            reasons.append("per_environment_summary_mismatch")
    measurements = report.get("measurements")
    try:
        maximum = measurements["base_com_max_abs_delta_xy"]
        fraction = measurements["base_com_outside_range_fraction"]
        outside = sum(value > threshold for value in expected_worst)
        expected_maximum = {
            "measurement_id": "base_com_max_abs_delta_xy",
            "unit": "simulation_length_native", "state": "present",
            "value": max(expected_worst), "coverage_status": "complete",
        }
        if maximum != expected_maximum:
            raise ValueError
        expected_fraction = {
            "measurement_id": "base_com_outside_range_fraction", "unit": "fraction",
            "state": "present", "value": outside / num_envs,
            "numerator": outside, "denominator": num_envs,
            "aggregation": "fraction_of_environments_whose_worst_max_abs_delta_xy_exceeds_threshold",
            "coverage_status": "complete",
        }
        if fraction != expected_fraction or set(measurements) != {
                "base_com_max_abs_delta_xy", "base_com_outside_range_fraction"}:
            raise ValueError
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        reasons.append("aggregate_measurement_mismatch")
    if any(report.get(key) is not False for key in (
            "cause_confirmed", "health_confirmed", "recovery_confirmed", "cleanup_confirmed", "model_selected")):
        reasons.append("forbidden_claim_flag")
    return reasons


def verify_body(body: dict, validated: dict, execution: dict, project_commit: str) -> list[str]:
    reasons = []
    protocol = validated["protocol"]
    expected = {
        "schema": "base_com_reset_body_v1", "completed": True,
        "scope": protocol["scope"], "task": TASK,
        "arm": validated["manifest"]["arm"],
        "protocol_sha256": validated["protocol_sha256"],
        "manifest_sha256": S._sha256_file(validated["manifest_path"]),
        "project_commit_sha": project_commit,
        "model_selected": False, "diagnosis_confirmed": False,
        "cause_confirmed": False, "recovery_confirmed": False,
        "cleanup_confirmed": False,
    }
    for key, value in expected.items():
        if type(body.get(key)) is not type(value) or body.get(key) != value:
            reasons.append(f"body_{key}_mismatch")
    identities = execution.get("owned_process_identities", {})
    pid, created = body.get("process_id"), body.get("process_creation_time_filetime")
    if type(pid) is not int or type(created) is not int or identities.get(str(pid)) != created:
        reasons.append("body_pid_creation_not_observed_in_owned_job")
    if body.get("source_receipt") != validated["source_receipt"]:
        reasons.append("body_source_receipt_mismatch")
    configuration_sha = S._sha256_file(validated["paths"]["configuration_path"])
    configuration_ref = body.get("configuration")
    if (not isinstance(configuration_ref, Mapping)
            or configuration_ref.get("sha256") != configuration_sha):
        reasons.append("body_configuration_sha_mismatch")
    reasons.extend(verify_configuration_snapshot(
        validated["paths"]["configuration_path"], protocol, body.get("injection_readback"),
        body.get("observer_report"),
    ))
    report = body.get("observer_report", {})
    bindings = report.get("source_binding", {}) if isinstance(report, Mapping) else {}
    snapshot = report.get("baseline_snapshot") if isinstance(report, Mapping) else None
    snapshot_sha = S._canonical_sha(snapshot) if isinstance(snapshot, Mapping) else None
    expected_bindings = {
        "task_sha256": validated["source_receipt"]["task_binding_sha256"],
        "configuration_sha256": configuration_sha,
        "isaaclab_sha256": validated["source_receipt"]["isaaclab_source_binding_sha256"],
        "observer_sha256": protocol["source_files"]["src/rl_triage/reset_com_observer.py"],
        "measurement_definitions_sha256": protocol["measurement_definitions_sha256"],
        "baseline_all_env_all_body_com_sha256": snapshot_sha,
    }
    if bindings != expected_bindings:
        reasons.append("observer_source_binding_mismatch")
    reasons.extend(verify_observer_report(report, protocol))
    reasons.extend(verify_phase_provenance(body.get("phase_provenance"), protocol))
    sdk_versions = body.get("sdk_versions")
    if (not isinstance(sdk_versions, Mapping)
            or sdk_versions.get("isaaclab_commit") != ISAACLAB_COMMIT
            or sdk_versions.get("isaacsim") != protocol["isaacsim_version"]):
        reasons.append("body_sdk_version_mismatch")
    return reasons


def child_main(manifest: Path) -> int:
    validated = validate_manifest(manifest)
    identity = S.process_identity()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    started_at = _utc_now()
    wall_start = time.perf_counter()
    try:
        result, env = _run_sdk(validated)
        report = {
            "schema": "base_com_reset_body_v1", "completed": True, **identity,
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": validated["manifest_sha256"],
            "project_commit_sha": validated["project_commit_sha"],
            "scope": validated["protocol"]["scope"], "task": TASK,
            "arm": validated["manifest"]["arm"],
            "source_receipt": validated["source_receipt"],
            "configuration": {
                "file": validated["paths"]["configuration_path"].name,
                "sha256": result["configuration_sha256"],
            },
            "injection_readback": result["injection_readback"],
            "sdk_versions": {
                "isaaclab_commit": ISAACLAB_COMMIT, "isaacsim": result["isaacsim_version"],
                "isaacsim_raw_components": result["isaacsim_raw_components"],
            },
            "phase_provenance": result["phase_provenance"],
            "observer_report": result["observer_report"],
            "model_selected": False, "diagnosis_confirmed": False,
            "cause_confirmed": False, "recovery_confirmed": False,
            "cleanup_confirmed": False,
            "cleanup_confirmation_required_from_parent_receipt": True,
            "started_at": started_at, "finished_at": _utc_now(),
            "wall_elapsed_s": round(time.perf_counter() - wall_start, 6),
        }
        raw = (json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
        S._publish_bytes_no_overwrite(validated["paths"]["output_path"], raw)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
    return 0


def run(tag: str, arm: str, isaaclab: Path, protocol_path: Path = PROTOCOL) -> dict:
    if os.name != "nt":
        raise RuntimeError("native Windows launcher required")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("tag must use lowercase letters, digits, underscores and hyphens")
    if arm not in ARMS:
        raise ValueError("arm must be X1 or X1_REF")
    isaaclab = isaaclab.resolve()
    protocol_path = protocol_location(protocol_path)
    protocol = S._read_json(protocol_path)
    source_receipt = validate_protocol(protocol, repo_root=ROOT, isaaclab_root=isaaclab)
    project_commit = P.verify_committed(
        ROOT, PROJECT_SOURCE_FILES | {protocol_path.relative_to(ROOT).as_posix()},
    )
    for argument in (str(isaaclab), str(ROOT)):
        if any(char in argument for char in '\r\n"%&^'):
            raise ValueError("path contains unsafe Windows batch characters")
    gpu = M.query_gpu(protocol["gpu_admission_profile"])
    if gpu["busy"]:
        raise RuntimeError("GPU workload or unclassified activity present; no process was stopped")
    folder = ROOT / "bench/private/base_com_reset" / tag
    if folder.exists():
        raise FileExistsError("use a new tag; an existing run is never overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    manifest = {
        "schema": "base_com_reset_manifest_v1",
        "protocol_path": str(protocol_path), "protocol_sha256": S._sha256_file(protocol_path),
        "arm": arm, "output_path": str(folder / "body.json"),
        "configuration_path": str(folder / "configuration.yaml"),
    }
    P.write_new(manifest_path, manifest)
    validated = validate_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab)
    if validated["source_receipt"] != source_receipt:
        raise ValueError("source changed after the first preflight")
    P.write_new(folder / "preflight.json", {
        "project_commit_sha": project_commit, "gpu": gpu, "source_receipt": source_receipt,
    })
    execution = P.run_owned(
        folder,
        [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/run_base_com_reset.py"),
         "--_child", "--manifest", str(manifest_path)],
        isaaclab, protocol["launcher_watchdog_s"],
    )
    reasons = []
    body_sha = None
    try:
        body = S._read_json(validated["paths"]["output_path"])
        body_sha = S._sha256_file(validated["paths"]["output_path"])
        reasons.extend(verify_body(body, validated, execution, project_commit))
    except Exception as error:
        reasons.append("body_validation_" + type(error).__name__)
    if not (execution["job_assigned_before_release"] and execution["natural_exit"]
            and execution["exit_code"] == 0 and execution["cleanup_verified"]
            and not execution["timed_out"] and execution["error_type"] is None):
        reasons.append("owned_execution_did_not_complete_naturally")
    gpu_after = None
    try:
        gpu_after = M.query_gpu(protocol["gpu_admission_profile"])
        owned = set(execution["owned_process_identities"])
        if any(str(row["pid"]) in owned for row in gpu_after["processes"]):
            reasons.append("owned_gpu_process_remaining")
    except Exception as error:
        reasons.append("post_gpu_query_" + type(error).__name__)
    receipt = {
        "schema": "base_com_reset_parent_v1", "scope": protocol["scope"],
        "arm": arm, "status": "invalid" if reasons else "pass",
        "validation_reasons": reasons, "project_commit_sha": project_commit,
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": S._sha256_file(manifest_path), "body_sha256": body_sha,
        "execution": execution, "gpu_before": gpu, "gpu_after": gpu_after,
        "model_selected": False, "diagnosis_or_recovery_evidence": False,
    }
    P.write_new(folder / "parent_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--_child", action="store_true")
    args = parser.parse_args()
    if args._child:
        if args.manifest is None or args.tag is not None or args.arm is not None:
            parser.error("child mode requires only --manifest")
        return child_main(args.manifest)
    if args.tag is None or args.arm is None or args.manifest is not None:
        parser.error("parent mode requires --tag and --arm")
    receipt = run(args.tag, args.arm, args.isaaclab)
    print(json.dumps({
        "status": receipt["status"], "arm": receipt["arm"],
        "validation_reasons": receipt["validation_reasons"],
    }, ensure_ascii=False))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
