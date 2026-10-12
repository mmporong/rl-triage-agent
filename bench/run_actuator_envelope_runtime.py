"""Actuator envelope 개발 검사를 사전등록하고 fresh owned SDK 프로세스로 실행한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import CodeType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "src"))

import actuator_envelope_runtime as B  # noqa: E402
import check_correction_cpu as CPU  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
from rl_triage import actuator_envelope as E  # noqa: E402


PROTOCOL_RELATIVE = "bench/protocols/actuator_envelope_runtime_v2.json"
PROTOCOL = ROOT / PROTOCOL_RELATIVE
PRIVATE_ROOT = ROOT / "bench/private/actuator_envelope_runtime"
PROJECT_SOURCE_FILES = {
    "bench/actuator_envelope_runtime.py", "bench/run_actuator_envelope_runtime.py",
    "bench/hidden_faults.py", "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py",
    "bench/check_correction_cpu.py", "src/rl_triage/actuator_envelope.py",
}
ISAACLAB_SOURCE_FILES = {
    B.ACTUATOR_SOURCE, B.GO2_SOURCE,
    "source/isaaclab/isaaclab/app/app_launcher.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
}
RSL_SOURCE_FILES = {"modules/actor_critic.py", "runners/on_policy_runner.py"}
GPU_PROFILE = "base_com_reset_readback_v2"
BODY_KEYS = {
    "schema", "completed", "process_id", "process_creation_time_filetime", "scope", "task",
    "condition", "protocol_sha256", "manifest_sha256", "project_commit_sha", "source_receipt",
    "request_lineage", "sdk_versions", "fault_method_bindings", "actuator_binding",
    "state_before", "state_after", "state_restoration", "helper_result",
    "measurement_validation_reasons", "measurement_status", "curve_assessment", "protocol_status",
    "policy_inference_used", "runner_created", "env_step_called", "cause_confirmed",
    "recovery_confirmed", "cleanup_confirmed", "cleanup_confirmation_required_from_parent_receipt",
    "started_at", "finished_at", "wall_elapsed_s",
}
ACTUATOR_BINDING_KEYS = {
    "class_name", "class_module", "clip_effort_source", "joint_names", "class_source",
    "joint_count", "joint_velocity_shape", "device", "runtime_config", "asset_source",
}
SOURCE_REF_KEYS = {"path", "sha256", "code_sha256"}
FILE_REF_KEYS = {"path", "sha256"}
EXPECTED_RUNTIME_BINDING = {
    "python": {"implementation": "CPython", "version": [3, 10, 15],
               "cache_tag": "cpython-310", "executable": B.SDK_PYTHON_RELATIVE},
    "distributions": {"torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "isaaclab": "0.41.3"},
    "modules": {"torch": "2.7.0+cu128"},
}
FIXED = {
    "schema": "actuator_envelope_runtime_protocol_v1",
    "scope": "actual_sdk_live_dcmotor_policy_free_registered_grid_readback",
    "task": B.TASK, "seed": 2027, "num_envs": 1, "device": "cuda:0",
    "conditions": list(B.CONDITIONS), "velocity_grid_rad_s": list(E.VELOCITY_GRID),
    "development_expectations": B.EXPECTED_MATCH,
    "isaaclab_commit": B.ISAACLAB_COMMIT, "isaacsim_version": "4.5.0",
    "launcher_watchdog_s": 360, "semantic_retries": 0,
    "gpu_admission_profile": GPU_PROFILE,
    "gpu_admission": deepcopy(M._gpu_admission_profile(GPU_PROFILE)),
    "runtime_binding": deepcopy(EXPECTED_RUNTIME_BINDING),
    "execution": {
        "fresh_owned_process_per_condition": True, "policy_inference": False,
        "runner_creation": False, "env_step": False, "helper_calls_per_process": 1,
        "app_launcher": {"headless": True, "fast_shutdown": False},
    },
    "interpretation": {
        "development_expectation_only": True, "cause_confirmed": False,
        "recovery_confirmed": False, "agreement_scope": "registered_input_grid_only",
    },
}


def _go2_binding(isaaclab: Path) -> dict[str, Any]:
    binding = CPU._go2_motor_binding(isaaclab)
    return {"source": binding["source"], "source_sha256": binding["source_sha256"],
            "values": binding["values"]}


def _source_map(root: Path, paths: set[str]) -> dict[str, str]:
    result = {}
    for relative in sorted(paths):
        path = S._relative_source_path(root, relative, "protocol source")
        if not path.is_file():
            raise FileNotFoundError(f"protocol source is missing: {relative}")
        result[relative] = S._sha256_file(path)
    return result


def _rsl_root(isaaclab: Path) -> Path:
    return (isaaclab / "_isaac_sim/kit/python/Lib/site-packages/rsl_rl").resolve()


def _find_code(code: CodeType, qualname: str) -> CodeType:
    current = code
    for segment in qualname.split("."):
        matches = [item for item in current.co_consts
                   if isinstance(item, CodeType) and item.co_name == segment]
        if len(matches) != 1:
            raise ValueError(f"source does not contain one {qualname} code path")
        current = matches[0]
    return current


def _compiled_method(path: Path, qualname: str) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    code = compile(raw, str(path), "exec", dont_inherit=True)
    return B._code_sha(_find_code(code, qualname))


def _method_contract(isaaclab: Path) -> dict[str, Any]:
    rsl = _rsl_root(isaaclab)
    hidden = ROOT / "bench/hidden_faults.py"
    specs = {
        "F2": (isaaclab / B.ACTUATOR_SOURCE, "DCMotor._clip_effort", "_before_env._clip_effort"),
        "F3": (rsl / "modules/actor_critic.py", "ActorCritic.act", "_before_env.act"),
        "F4": (rsl / "runners/on_policy_runner.py", "OnPolicyRunner.__init__", "_before_env.__init__"),
    }
    result = {}
    hidden_sha = S._sha256_file(hidden)
    for key, (original, qualname, replacement_qualname) in specs.items():
        result[key] = {
            "before": {"path": str(original.resolve()), "source_sha256": S._sha256_file(original),
                       "code_sha256": _compiled_method(original, qualname)},
            "patched": {"path": str(hidden.resolve()), "source_sha256": hidden_sha,
                        "code_sha256": _compiled_method(hidden, replacement_qualname)},
        }
    return result


def _validate_runtime_binding(runtime: object, isaaclab: Path) -> None:
    if (not isinstance(runtime, Mapping)
            or runtime != EXPECTED_RUNTIME_BINDING
            or any(type(runtime.get(key)) is not dict
                   for key in ("python", "distributions", "modules"))):
        raise ValueError("protocol was not prepared by the registered SDK Python runtime")
    sdk_python_root = (isaaclab / "_isaac_sim").resolve()
    executable = (sdk_python_root / runtime["python"]["executable"]).resolve()
    if not executable.is_file() or not executable.is_relative_to(sdk_python_root):
        raise ValueError("protocol Python executable is outside the registered SDK runtime")


def _validate_method_contract_static(value: object, protocol: Mapping[str, Any], isaaclab: Path) -> None:
    if not isinstance(value, Mapping) or set(value) != set(B.TARGET_METHODS):
        raise ValueError("method binding contract target set differs")
    source_sha = {
        "F2": protocol["isaaclab_source_files"][B.ACTUATOR_SOURCE],
        "F3": protocol["rsl_rl_source_files"]["modules/actor_critic.py"],
        "F4": protocol["rsl_rl_source_files"]["runners/on_policy_runner.py"],
    }
    rsl = _rsl_root(isaaclab)
    source_path = {
        "F2": str((isaaclab / B.ACTUATOR_SOURCE).resolve()),
        "F3": str((rsl / "modules/actor_critic.py").resolve()),
        "F4": str((rsl / "runners/on_policy_runner.py").resolve()),
    }
    hidden_path = str((ROOT / "bench/hidden_faults.py").resolve())
    hidden_sha = protocol["source_files"]["bench/hidden_faults.py"]
    for key in B.TARGET_METHODS:
        row = value[key]
        if (not isinstance(row, Mapping) or set(row) != {"before", "patched"}
                or row.get("before", {}).get("source_sha256") != source_sha[key]
                or row.get("patched", {}).get("source_sha256") != hidden_sha
                or row.get("before", {}).get("path") != source_path[key]
                or row.get("patched", {}).get("path") != hidden_path
                or any(not isinstance(row.get(phase), Mapping)
                       or set(row[phase]) != {"path", "source_sha256", "code_sha256"}
                       or not isinstance(row[phase].get("path"), str)
                       or re.fullmatch(r"[0-9a-f]{64}", row[phase].get("code_sha256", "")) is None
                       for phase in ("before", "patched"))):
            raise ValueError("method source or code binding is malformed")


def protocol_document(isaaclab: Path) -> dict[str, Any]:
    import torch

    isaaclab = isaaclab.resolve()
    if S._git_head(isaaclab) != B.ISAACLAB_COMMIT or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout must match the pinned clean commit")
    project_commit = P.verify_committed(ROOT, PROJECT_SOURCE_FILES)
    asset = _go2_binding(isaaclab)
    runtime = B.runtime_binding(torch_module=torch, isaaclab_root=isaaclab)
    _validate_runtime_binding(runtime, isaaclab)
    return {
        **deepcopy(FIXED), "prepared_source_commit_sha": project_commit,
        "expected_config": asset["values"], "asset_binding": asset,
        "runtime_binding": runtime,
        "method_binding_contract": _method_contract(isaaclab),
        "source_files": _source_map(ROOT, PROJECT_SOURCE_FILES),
        "isaaclab_source_files": _source_map(isaaclab, ISAACLAB_SOURCE_FILES),
        "rsl_rl_source_files": _source_map(_rsl_root(isaaclab), RSL_SOURCE_FILES),
    }


def prepare(path: Path, isaaclab: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.parent != (ROOT / "bench/protocols").resolve():
        raise ValueError("protocol must be created in bench/protocols")
    document = protocol_document(isaaclab)
    P.write_new(path, document)
    return {"state": "prepared", "protocol_path": str(path),
            "protocol_sha256": S._sha256_file(path), "gpu_queried": False,
            "next_step": "commit the protocol before execute"}


def validate_protocol(protocol: dict, *, repo_root: Path, isaaclab_root: Path) -> dict[str, Any]:
    expected_fields = set(FIXED) | {"prepared_source_commit_sha", "expected_config", "asset_binding",
                                      "method_binding_contract", "source_files",
                                      "isaaclab_source_files", "rsl_rl_source_files"}
    if set(protocol) != expected_fields:
        raise ValueError("protocol fields differ from fixed contract")
    for key, expected in FIXED.items():
        if type(protocol.get(key)) is not type(expected) or protocol.get(key) != expected:
            raise ValueError(f"protocol field {key} differs from fixed contract")
    project_commit = S._git_head(repo_root)
    prepared_commit = protocol["prepared_source_commit_sha"]
    if not isinstance(prepared_commit, str) or re.fullmatch(r"[0-9a-f]{40}", prepared_commit) is None:
        raise ValueError("protocol prepared source commit is invalid")
    try:
        ancestor = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", prepared_commit, project_commit],
            check=False, capture_output=True,
        )
    except OSError as error:
        raise ValueError("protocol prepared source commit cannot be verified") from error
    if ancestor.returncode != 0:
        raise ValueError("protocol prepared source commit is not in execution history")
    project = S._verify_source_map(repo_root, protocol["source_files"], "source_files",
                                   exact_paths=PROJECT_SOURCE_FILES)
    sdk = S._verify_source_map(isaaclab_root, protocol["isaaclab_source_files"],
                               "isaaclab_source_files", exact_paths=ISAACLAB_SOURCE_FILES)
    rsl = S._verify_source_map(_rsl_root(isaaclab_root), protocol["rsl_rl_source_files"],
                               "rsl_rl_source_files", exact_paths=RSL_SOURCE_FILES)
    if S._git_head(isaaclab_root) != B.ISAACLAB_COMMIT or not S._git_tracked_clean(isaaclab_root):
        raise ValueError("Isaac Lab checkout differs from pinned clean commit")
    fresh_asset = _go2_binding(isaaclab_root)
    if protocol["asset_binding"] != fresh_asset or protocol["expected_config"] != fresh_asset["values"]:
        raise ValueError("asset AST binding differs from protocol")
    _validate_method_contract_static(protocol["method_binding_contract"], protocol, isaaclab_root)
    runtime = protocol["runtime_binding"]
    _validate_runtime_binding(runtime, isaaclab_root)
    return {"project_source_files": project, "isaaclab_source_files": sdk,
            "rsl_rl_source_files": rsl,
            "asset_binding": fresh_asset, "project_commit_sha": project_commit,
            "prepared_source_commit_sha": prepared_commit}


def protocol_location(path: Path, repo_root: Path = ROOT) -> Path:
    resolved = path.resolve()
    if resolved != (repo_root / PROTOCOL_RELATIVE).resolve():
        raise ValueError("execute requires the fixed repository protocol")
    return resolved


def validate_manifest(path: Path, *, repo_root: Path = ROOT,
                      isaaclab_root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    private = (repo_root / "bench/private/actuator_envelope_runtime").resolve()
    condition_folder = path.parent
    tag_folder = condition_folder.parent
    if (not path.is_file() or path.name != "manifest.json" or tag_folder.parent != private
            or condition_folder.name not in B.CONDITIONS):
        raise ValueError("manifest location is outside the fixed private run layout")
    manifest = S._read_json(path)
    fields = {"schema", "protocol_path", "protocol_sha256", "condition", "output_path",
              "request_path", "consumption_path"}
    if set(manifest) != fields or manifest.get("schema") != "actuator_envelope_runtime_manifest_v1":
        raise ValueError("unsupported manifest contract")
    if manifest["condition"] != condition_folder.name:
        raise ValueError("manifest condition does not match its folder")
    protocol_path = S._absolute_path(manifest["protocol_path"], "protocol_path")
    protocol_location(protocol_path, repo_root)
    output = S._absolute_path(manifest["output_path"], "output_path")
    request = S._absolute_path(manifest["request_path"], "request_path")
    consumption = S._absolute_path(manifest["consumption_path"], "consumption_path")
    if output != condition_folder / "body.json" or request != tag_folder / "request.json" or consumption != tag_folder / "consumed.json":
        raise ValueError("manifest paths differ from fixed run layout")
    if output.exists():
        raise FileExistsError("body output already exists")
    protocol_sha = S._require_sha(manifest["protocol_sha256"], "protocol_sha256")
    if S._sha256_file(protocol_path) != protocol_sha:
        raise ValueError("protocol SHA mismatch")
    protocol = S._read_json(protocol_path)
    root = Path.cwd() if isaaclab_root is None else isaaclab_root.resolve()
    source = validate_protocol(protocol, repo_root=repo_root, isaaclab_root=root)
    request_value = S._read_json(request)
    consumption_value = S._read_json(consumption)
    request_core = {key: request_value.get(key) for key in (
        "tag", "protocol_path", "protocol_sha256", "project_commit_sha",
        "prepared_source_commit_sha", "conditions",
        "authorization_scope",
    )}
    expected_request_id = S._canonical_sha(request_core)
    if (request_value.get("schema") != "actuator_envelope_runtime_request_v1"
            or request_value.get("request_id") != expected_request_id
            or request_value.get("tag") != tag_folder.name
            or request_value.get("protocol_sha256") != protocol_sha
            or request_value.get("project_commit_sha") != S._git_head(repo_root)
            or request_value.get("prepared_source_commit_sha") != protocol["prepared_source_commit_sha"]
            or request_value.get("conditions") != list(B.CONDITIONS)):
        raise ValueError("request binding is invalid")
    expected_consumption = {
        "schema": "actuator_envelope_runtime_consumption_v1",
        "request_id": expected_request_id, "tag": tag_folder.name,
        "request_sha256": S._canonical_sha(request_value),
    }
    if consumption_value != expected_consumption:
        raise ValueError("request consumption binding is invalid")
    return {"manifest_path": path, "manifest": manifest, "protocol": protocol,
            "protocol_sha256": protocol_sha, "paths": {"output_path": output,
            "request_path": request, "consumption_path": consumption},
            "project_commit_sha": S._git_head(repo_root), "source_receipt": source,
            "request_lineage": {"request_id": expected_request_id,
                                "request_sha256": expected_consumption["request_sha256"],
                                "consumption_sha256": S._canonical_sha(consumption_value)},
            "isaaclab_root": root}


def _request(tag: str, protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    core = {"tag": tag, "protocol_path": str(protocol_path),
            "protocol_sha256": S._sha256_file(protocol_path),
            "project_commit_sha": S._git_head(ROOT),
            "prepared_source_commit_sha": protocol["prepared_source_commit_sha"],
            "conditions": list(B.CONDITIONS), "authorization_scope": "user_requested_development_inspection"}
    return {"schema": "actuator_envelope_runtime_request_v1", **core,
            "request_id": S._canonical_sha(core)}


def _consume(folder: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    consumed_root = PRIVATE_ROOT / "consumed"
    consumed_root.mkdir(parents=True, exist_ok=True)
    global_path = consumed_root / f"{request['request_id']}.json"
    value = {"schema": "actuator_envelope_runtime_consumption_v1",
             "request_id": request["request_id"], "tag": request["tag"],
             "request_sha256": S._canonical_sha(request)}
    P.write_new(global_path, value)
    try:
        P.write_new(folder / "consumed.json", value)
    except BaseException:
        global_path.unlink(missing_ok=True)
        raise
    return value


def verify_body(body: object, validated: Mapping[str, Any], execution: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(body, Mapping):
        return ["body_not_object"]
    if set(body) != BODY_KEYS:
        reasons.append("body_keyset_mismatch")
    protocol = validated["protocol"]
    condition = validated["manifest"]["condition"]
    expected = {"schema": "actuator_envelope_runtime_body_v1", "completed": True,
                "scope": protocol["scope"], "task": B.TASK, "condition": condition,
                "protocol_sha256": validated["protocol_sha256"],
                "manifest_sha256": S._sha256_file(validated["manifest_path"]),
                "project_commit_sha": validated["project_commit_sha"],
                "source_receipt": validated["source_receipt"],
                "request_lineage": validated["request_lineage"],
                "policy_inference_used": False, "runner_created": False,
                "env_step_called": False, "cleanup_confirmed": False,
                "cause_confirmed": False, "recovery_confirmed": False,
                "cleanup_confirmation_required_from_parent_receipt": True}
    for key, value in expected.items():
        if body.get(key) != value or type(body.get(key)) is not type(value):
            reasons.append(f"body_{key}_mismatch")
    identities = execution.get("owned_process_identities", {})
    if identities.get(str(body.get("process_id"))) != body.get("process_creation_time_filetime"):
        reasons.append("body_pid_creation_not_observed_in_owned_job")
    binding = body.get("actuator_binding")
    if not isinstance(binding, Mapping):
        reasons.append("actuator_binding_missing")
    else:
        if set(binding) != ACTUATOR_BINDING_KEYS:
            reasons.append("actuator_binding_keyset_mismatch")
        expected_binding = {"class_name": "DCMotor", "class_module": "isaaclab.actuators.actuator_pd",
                            "joint_count": 12, "joint_velocity_shape": [1, 12],
                            "runtime_config": protocol["expected_config"],
                            "asset_source": {"path": B.GO2_SOURCE,
                                             "sha256": protocol["asset_binding"]["source_sha256"]}}
        for key, value in expected_binding.items():
            if binding.get(key) != value:
                reasons.append(f"actuator_{key}_mismatch")
        if binding.get("device") != protocol["device"]:
            reasons.append("actuator_device_mismatch")
        names = binding.get("joint_names")
        if (not isinstance(names, list) or len(names) != 12
                or any(type(name) is not str for name in names)
                or set(names) != B.GO2_JOINT_NAMES):
            reasons.append("actuator_joint_names_invalid")
        source = binding.get("clip_effort_source")
        if not isinstance(source, Mapping) or set(source) != SOURCE_REF_KEYS:
            reasons.append("actuator_method_source_missing")
        else:
            try:
                source_path = Path(source["path"]).resolve()
                if S._sha256_file(source_path) != source.get("sha256"):
                    reasons.append("actuator_method_source_changed")
            except Exception:
                reasons.append("actuator_method_source_invalid")
        class_source = binding.get("class_source")
        expected_class_sha = protocol.get("isaaclab_source_files", {}).get(B.ACTUATOR_SOURCE)
        if (not isinstance(class_source, Mapping) or set(class_source) != FILE_REF_KEYS
                or class_source.get("sha256") != expected_class_sha):
            reasons.append("actuator_class_source_mismatch")
        else:
            try:
                if S._sha256_file(Path(class_source["path"])) != expected_class_sha:
                    reasons.append("actuator_class_source_changed")
            except Exception:
                reasons.append("actuator_class_source_invalid")
        asset_source = binding.get("asset_source")
        if not isinstance(asset_source, Mapping) or set(asset_source) != FILE_REF_KEYS:
            reasons.append("actuator_asset_source_keyset_mismatch")
    helper = body.get("helper_result")
    curve_reasons = B.validate_curve(helper, protocol["expected_config"])
    reasons.extend(curve_reasons)
    if body.get("measurement_validation_reasons") != curve_reasons:
        reasons.append("measurement_validation_receipt_mismatch")
    helper_valid = (isinstance(helper, Mapping) and helper.get("status") == "valid"
                    and not curve_reasons)
    expected_measurement = "valid" if helper_valid else "invalid"
    expected_assessment = None if not helper_valid else (
        "normal" if helper.get("curve_matches_expected") else "curve_difference")
    expected_protocol = "invalid" if not helper_valid else (
        "expected" if helper.get("curve_matches_expected") is protocol["development_expectations"][condition]
        else "unexpected")
    if body.get("measurement_status") != expected_measurement:
        reasons.append("measurement_status_mismatch")
    if body.get("curve_assessment") != expected_assessment:
        reasons.append("curve_assessment_mismatch")
    if body.get("protocol_status") != expected_protocol:
        reasons.append("protocol_status_mismatch")
    if not helper_valid:
        reasons.append("helper_measurement_invalid")
    provenance = helper.get("config_provenance", {}) if isinstance(helper, Mapping) else {}
    if provenance.get("caller_asserted_source_sha256") != protocol["asset_binding"]["source_sha256"]:
        reasons.append("helper_asset_source_assertion_mismatch")
    restoration = body.get("state_restoration")
    before_state, after_state = body.get("state_before"), body.get("state_after")
    reasons.extend("before_" + reason for reason in B.validate_state_snapshot(before_state))
    reasons.extend("after_" + reason for reason in B.validate_state_snapshot(after_state))
    if not B.validate_state_snapshot(before_state) and not B.validate_state_snapshot(after_state):
        expected_restoration = B._restoration(before_state, after_state)
        if restoration != expected_restoration:
            reasons.append("state_restoration_receipt_mismatch")
        if expected_restoration["all_restored"] is not True:
            reasons.append("state_not_restored")
    changes = body.get("fault_method_bindings", {}).get("changed")
    if (not isinstance(changes, Mapping) or set(changes) != set(B.TARGET_METHODS)
            or any(type(value) is not bool for value in changes.values())
            or changes != {key: key == condition for key in B.TARGET_METHODS}):
        reasons.append("fault_method_cross_signal")
    methods = body.get("fault_method_bindings", {})
    if not isinstance(methods, Mapping) or set(methods) != {"before", "after", "changed"}:
        reasons.append("fault_method_receipt_keyset_mismatch")
    before_methods, after_methods = methods.get("before"), methods.get("after")
    reasons.extend(B.validate_method_transition(
        before_methods, after_methods, condition, protocol["method_binding_contract"],
    ))
    if isinstance(before_methods, Mapping) and isinstance(after_methods, Mapping):
        for key in B.TARGET_METHODS:
            for phase, row in (("before", before_methods.get(key)), ("after", after_methods.get(key))):
                try:
                    if S._sha256_file(Path(row["path"])) != row.get("sha256"):
                        reasons.append(f"fault_method_{key}_{phase}_source_changed")
                except Exception:
                    reasons.append(f"fault_method_{key}_{phase}_source_invalid")
        clip_source = binding.get("clip_effort_source") if isinstance(binding, Mapping) else None
        f2_after = after_methods.get("F2")
        if (not isinstance(clip_source, Mapping) or not isinstance(f2_after, Mapping)
                or {key: clip_source.get(key) for key in ("path", "sha256", "code_sha256")}
                != {key: f2_after.get(key) for key in ("path", "sha256", "code_sha256")}):
            reasons.append("actuator_clip_method_binding_mismatch")
    sdk = body.get("sdk_versions", {})
    if (not isinstance(sdk, Mapping)
            or set(sdk) != {"isaaclab_commit", "isaacsim", "isaacsim_raw_components", "runtime_binding"}
            or sdk.get("isaaclab_commit") != B.ISAACLAB_COMMIT
            or sdk.get("isaacsim") != protocol["isaacsim_version"]
            or not B.validate_isaacsim_raw_components(
                sdk.get("isaacsim_raw_components"), protocol["isaacsim_version"])
            or sdk.get("runtime_binding") != protocol["runtime_binding"]):
        reasons.append("sdk_version_mismatch")
    for key in ("started_at", "finished_at"):
        if not isinstance(body.get(key), str) or not body[key]:
            reasons.append(f"body_{key}_invalid")
    elapsed = body.get("wall_elapsed_s")
    if type(elapsed) not in (int, float) or isinstance(elapsed, bool) or not math.isfinite(float(elapsed)) or elapsed < 0:
        reasons.append("body_wall_elapsed_s_invalid")
    return reasons


def _owned_execution_reasons(execution: Mapping[str, Any]) -> list[str]:
    if (execution.get("job_assigned_before_release") is True
            and execution.get("natural_exit") is True
            and execution.get("exit_code") == 0
            and execution.get("cleanup_verified") is True
            and execution.get("timed_out") is False
            and execution.get("error_type") is None):
        return []
    return ["owned_execution_did_not_complete_naturally"]


def _execution_log_reasons(folder: Path) -> list[str]:
    reasons = []
    for name in ("stdout.log", "stderr.log"):
        path = folder / name
        if not path.is_file():
            reasons.append(f"missing_{name}")
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"Windows fatal exception|Fatal Python error|Traceback \(most recent call last\)|\[Fatal\]", content):
            reasons.append(f"runtime_failure_in_{name}")
    return reasons


def execute(tag: str, isaaclab: Path, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("native Windows launcher required")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("invalid tag")
    isaaclab = isaaclab.resolve()
    protocol_path = protocol_location(protocol_path)
    protocol = S._read_json(protocol_path)
    source = validate_protocol(protocol, repo_root=ROOT, isaaclab_root=isaaclab)
    P.verify_committed(ROOT, PROJECT_SOURCE_FILES | {protocol_path.relative_to(ROOT).as_posix()})
    gpu = M.query_gpu(protocol["gpu_admission_profile"])
    if gpu["busy"]:
        return {"schema": "actuator_envelope_runtime_parent_v1", "status": "waiting_gpu_idle",
                "launch_count": 0, "request_consumed": False}
    folder = PRIVATE_ROOT / tag
    if folder.exists():
        raise FileExistsError("tag already exists; runtime evidence is never overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    request = _request(tag, protocol_path, protocol)
    P.write_new(folder / "request.json", request)
    consumption = _consume(folder, request)
    parents = []
    try:
        for condition in B.CONDITIONS:
            condition_folder = folder / condition
            condition_folder.mkdir()
            manifest_path = condition_folder / "manifest.json"
            manifest = {"schema": "actuator_envelope_runtime_manifest_v1",
                        "protocol_path": str(protocol_path), "protocol_sha256": S._sha256_file(protocol_path),
                        "condition": condition, "output_path": str(condition_folder / "body.json"),
                        "request_path": str(folder / "request.json"),
                        "consumption_path": str(folder / "consumed.json")}
            P.write_new(manifest_path, manifest)
            validated = validate_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab)
            if validated["source_receipt"] != source:
                raise ValueError("source changed after execute preflight")
            execution = P.run_owned(
                condition_folder,
                [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/actuator_envelope_runtime.py"),
                 "--manifest", str(manifest_path)], isaaclab, protocol["launcher_watchdog_s"],
            )
            reasons = []
            body = None
            body_sha = None
            try:
                body = S._read_json(validated["paths"]["output_path"])
                body_sha = S._sha256_file(validated["paths"]["output_path"])
                reasons.extend(verify_body(body, validated, execution))
            except Exception as error:
                reasons.append("body_validation_" + type(error).__name__)
            reasons.extend(_owned_execution_reasons(execution))
            reasons.extend(_execution_log_reasons(condition_folder))
            parents.append({"condition": condition, "status": "invalid" if reasons else "pass",
                            "validation_reasons": reasons, "body_sha256": body_sha,
                            "measurement_status": body.get("measurement_status") if isinstance(body, Mapping) else None,
                            "curve_assessment": body.get("curve_assessment") if isinstance(body, Mapping) else None,
                            "protocol_status": body.get("protocol_status") if isinstance(body, Mapping) else None,
                            "execution": execution})
            if reasons:
                break
    except BaseException as error:
        parents.append({"condition": B.CONDITIONS[len(parents)] if len(parents) < len(B.CONDITIONS) else None,
                        "status": "invalid", "validation_reasons": ["execution_" + type(error).__name__]})
    gpu_after = None
    try:
        gpu_after = M.query_gpu(protocol["gpu_admission_profile"])
        owned = {pid for row in parents for pid in row.get("execution", {}).get("owned_process_identities", {})}
        if any(str(row["pid"]) in owned for row in gpu_after["processes"]):
            parents.append({"condition": None, "status": "invalid",
                            "validation_reasons": ["owned_gpu_process_remaining"]})
    except Exception as error:
        parents.append({"condition": None, "status": "invalid",
                        "validation_reasons": ["post_gpu_query_" + type(error).__name__]})
    complete = len(parents) == len(B.CONDITIONS) and all(row["status"] == "pass" for row in parents)
    receipt = {"schema": "actuator_envelope_runtime_parent_v1",
               "status": "complete" if complete else "invalid", "tag": tag,
               "project_commit_sha": S._git_head(ROOT),
               "protocol_sha256": S._sha256_file(protocol_path),
               "request_id": request["request_id"], "consumption": consumption,
               "request_consumed": True, "launch_count": len([row for row in parents if row.get("condition")]),
               "conditions": parents, "gpu_before": gpu, "gpu_after": gpu_after,
               "cause_or_recovery_evidence": False}
    P.write_new(folder / "parent_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare(args.protocol, args.isaaclab)
        else:
            if not args.tag:
                parser.error("--execute requires --tag")
            result = execute(args.tag, args.isaaclab, args.protocol)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["status" if args.execute else "state"] in {"complete", "prepared"} else 3
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(json.dumps({"state": "rejected", "reason": type(error).__name__}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
