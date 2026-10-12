from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
for directory in (ROOT / "src", ROOT / "bench"):
    sys.path.insert(0, str(directory))
SPEC = importlib.util.spec_from_file_location(
    "run_actuator_envelope_runtime_test", ROOT / "bench/run_actuator_envelope_runtime.py",
)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


CONFIG = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}


def helper(matches=True):
    rows = []
    for velocity in R.E.VELOCITY_GRID:
        minimum, maximum = R.E._documented_limits(velocity, CONFIG)
        rows.append({"velocity_rad_s": velocity, "expected_min_effort": minimum,
                     "expected_max_effort": maximum, "measured_min_effort": [minimum] * 12,
                     "measured_max_effort": [maximum if matches else maximum - 1] * 12,
                     "matches_expected": matches})
    return {"schema": "actuator_envelope_readback_v1", "status": "valid",
            "curve_matches_expected": matches, "invalid_reason": None,
            "requested_efforts": {"positive": 23500.0, "negative": -23500.0},
            "measurements": rows, "config_provenance": {
                "expected_config": CONFIG, "runtime_config_readback": CONFIG,
                "velocity_grid_rad_s": list(R.E.VELOCITY_GRID), "source_binding_verified": False,
                "caller_asserted_source_sha256": "a" * 64,
            }, "limitations": dict(R.E.LIMITATIONS)}


def body_fixture(tmp_path, condition="NONE", matches=True):
    source = tmp_path / "actuator_pd.py"
    source.write_text("# source\n", encoding="utf-8")
    patch_source = tmp_path / "hidden_faults.py"
    patch_source.write_text("# patch source\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    runtime = deepcopy(R.EXPECTED_RUNTIME_BINDING)
    before_contract = {"path": str(source), "source_sha256": R.S._sha256_file(source),
                       "code_sha256": "0" * 64}
    patched_contract = {"path": str(patch_source), "source_sha256": R.S._sha256_file(patch_source),
                        "code_sha256": "1" * 64}
    method_contract = {key: {"before": deepcopy(before_contract), "patched": deepcopy(patched_contract)}
                       for key in R.B.TARGET_METHODS}
    protocol = {"scope": "scope", "expected_config": CONFIG, "device": "cuda:0",
                "development_expectations": deepcopy(R.B.EXPECTED_MATCH), "isaacsim_version": "4.5.0",
                "asset_binding": {"source_sha256": "a" * 64},
                "isaaclab_source_files": {R.B.ACTUATOR_SOURCE: R.S._sha256_file(source)},
                "method_binding_contract": method_contract, "runtime_binding": runtime}
    source_receipt = {"bound": True}
    validated = {"protocol": protocol, "protocol_sha256": "b" * 64,
                 "manifest": {"condition": condition}, "manifest_path": manifest,
                 "project_commit_sha": "c" * 40, "source_receipt": source_receipt}
    validated["request_lineage"] = {"request_id": "r", "request_sha256": "q", "consumption_sha256": "x"}
    snapshot = {"schema": "actuator_envelope_state_v1",
                "buffers": {name: {"identity": index + 9, "shape": [1, 12],
                                    "values": [0.0] * 12}
                            for index, name in enumerate(R.B.STATE_BUFFER_FIELDS)},
                "common_step_counter": 0, "sim_step_counter": 0}
    changed = {key: key == condition for key in R.B.TARGET_METHODS}
    assessment = "normal" if matches else "curve_difference"
    expected = protocol["development_expectations"][condition]
    before_methods = {key: {"identity": index + 1, "path": str(source),
                            "sha256": before_contract["source_sha256"],
                            "code_sha256": before_contract["code_sha256"]}
                      for index, key in enumerate(R.B.TARGET_METHODS)}
    after_methods = deepcopy(before_methods)
    if condition in R.B.TARGET_METHODS:
        after_methods[condition] = {"identity": 100, "path": str(patch_source),
                                    "sha256": patched_contract["source_sha256"],
                                    "code_sha256": patched_contract["code_sha256"]}
    clip_method = after_methods["F2"]
    body = {"schema": "actuator_envelope_runtime_body_v1", "completed": True,
            "scope": "scope", "task": R.B.TASK, "condition": condition,
            "protocol_sha256": "b" * 64, "manifest_sha256": R.S._sha256_file(manifest),
            "project_commit_sha": "c" * 40, "source_receipt": source_receipt,
            "request_lineage": validated["request_lineage"],
            "policy_inference_used": False, "runner_created": False, "env_step_called": False,
            "cause_confirmed": False, "recovery_confirmed": False,
            "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True,
            "process_id": 321, "process_creation_time_filetime": 999,
            "actuator_binding": {"class_name": "DCMotor",
                "class_module": "isaaclab.actuators.actuator_pd", "joint_count": 12,
                "joint_velocity_shape": [1, 12], "joint_names": sorted(R.B.GO2_JOINT_NAMES),
                "runtime_config": CONFIG, "device": "cuda:0",
                "class_source": {"path": str(source), "sha256": R.S._sha256_file(source)},
                "asset_source": {"path": R.B.GO2_SOURCE, "sha256": "a" * 64},
                "clip_effort_source": {key: clip_method[key]
                                       for key in ("path", "sha256", "code_sha256")}},
            "helper_result": helper(matches), "measurement_status": "valid",
            "measurement_validation_reasons": [],
            "curve_assessment": assessment, "protocol_status": "expected" if matches is expected else "unexpected",
            "state_before": snapshot, "state_after": deepcopy(snapshot),
            "state_restoration": R.B._restoration(snapshot, snapshot),
            "fault_method_bindings": {"changed": changed,
                "before": before_methods, "after": after_methods},
            "sdk_versions": {"isaaclab_commit": R.B.ISAACLAB_COMMIT, "isaacsim": "4.5.0",
                             "isaacsim_raw_components": [
                                 "4.5.0", "rc.36", "4", "5", "0", "rc", "36",
                                 "release.19112.f59b3005.gl",
                             ],
                             "runtime_binding": deepcopy(runtime)},
            "started_at": "2026-10-13T00:00:00+00:00",
            "finished_at": "2026-10-13T00:00:01+00:00", "wall_elapsed_s": 1.0}
    execution = {"owned_process_identities": {"321": 999}}
    return body, validated, execution


@pytest.mark.parametrize("condition,matches,assessment", [
    ("NONE", True, "normal"), ("F2", False, "curve_difference"),
    ("F3", True, "normal"), ("F4", True, "normal"),
])
def test_parent_accepts_actual_live_contract_shapes(condition, matches, assessment, tmp_path):
    body, validated, execution = body_fixture(tmp_path, condition, matches)
    assert body["curve_assessment"] == assessment
    assert R.verify_body(body, validated, execution) == []


def test_parent_rejects_mock_like_class_shape_source_and_cross_signal(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["actuator_binding"]["class_name"] = "FakeMotor"
    body["actuator_binding"]["joint_velocity_shape"] = [12]
    body["fault_method_bindings"]["changed"]["F3"] = True
    Path(body["actuator_binding"]["clip_effort_source"]["path"]).write_text("changed\n", encoding="utf-8")
    reasons = R.verify_body(body, validated, execution)
    assert "actuator_class_name_mismatch" in reasons
    assert "actuator_joint_velocity_shape_mismatch" in reasons
    assert "actuator_method_source_changed" in reasons
    assert "fault_method_cross_signal" in reasons


def test_parent_rejects_pid_reuse_state_change_and_masked_invalid(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["process_creation_time_filetime"] += 1
    body["state_after"]["buffers"]["_joint_vel"]["values"][0] = 1.0
    body["helper_result"] = {"schema": "actuator_envelope_readback_v1", "status": "invalid",
                             "curve_matches_expected": None, "invalid_reason": "runtime_config_mismatch",
                             "measurements": [], "requested_efforts": None}
    reasons = R.verify_body(body, validated, execution)
    assert "body_pid_creation_not_observed_in_owned_job" in reasons
    assert "state_not_restored" in reasons
    assert "state_restoration_receipt_mismatch" in reasons
    assert "measurement_status_mismatch" in reasons
    assert "curve_assessment_mismatch" in reasons


def test_parent_rejects_body_key_claim_and_runtime_tamper(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["extra"] = True
    body["cause_confirmed"] = True
    body["recovery_confirmed"] = True
    body["sdk_versions"]["runtime_binding"]["distributions"]["torch"] = "changed"
    reasons = R.verify_body(body, validated, execution)
    assert "body_keyset_mismatch" in reasons
    assert "body_cause_confirmed_mismatch" in reasons
    assert "body_recovery_confirmed_mismatch" in reasons
    assert "sdk_version_mismatch" in reasons


@pytest.mark.parametrize("mutation,expected_reason", [
    (lambda body: body["actuator_binding"].update(extra=True),
     "actuator_binding_keyset_mismatch"),
    (lambda body: body["actuator_binding"]["clip_effort_source"].update(extra=True),
     "actuator_method_source_missing"),
    (lambda body: body["actuator_binding"]["class_source"].update(extra=True),
     "actuator_class_source_mismatch"),
    (lambda body: body["actuator_binding"]["asset_source"].update(extra=True),
     "actuator_asset_source_keyset_mismatch"),
    (lambda body: body["actuator_binding"]["joint_names"].__setitem__(0, 1),
     "actuator_joint_names_invalid"),
    (lambda body: body["actuator_binding"]["joint_names"].__setitem__(0, "unknown_joint"),
     "actuator_joint_names_invalid"),
])
def test_parent_rejects_nested_binding_schema_and_joint_tamper(
        mutation, expected_reason, tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    mutation(body)
    assert expected_reason in R.verify_body(body, validated, execution)


@pytest.mark.parametrize("raw", [
    ["4.5.0", "rc.36", "4", "5", "0", "rc", "36"],
    ["4.5.0", "rc.36", 4, "5", "0", "rc", "36", "release.x"],
    ["4.5.1", "rc.36", "4", "5", "1", "rc", "36", "release.x"],
    ["4.5.0", "rc.35", "4", "5", "0", "rc", "36", "release.x"],
])
def test_parent_rejects_malformed_isaacsim_raw_components(raw, tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["sdk_versions"]["isaacsim_raw_components"] = raw
    assert "sdk_version_mismatch" in R.verify_body(body, validated, execution)


def test_parent_recomputes_restoration_instead_of_trusting_true_flag(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["state_after"]["buffers"]["applied_effort"]["identity"] += 1
    body["state_restoration"] = {"checks": {name: True for name in R.B.STATE_BUFFER_FIELDS}
                                 | {"common_step_counter": True, "sim_step_counter": True},
                                 "all_restored": True}
    reasons = R.verify_body(body, validated, execution)
    assert "state_restoration_receipt_mismatch" in reasons
    assert "state_not_restored" in reasons


def test_parent_rejects_non_target_identity_only_cross_signal(tmp_path):
    body, validated, execution = body_fixture(tmp_path, "F2", False)
    body["fault_method_bindings"]["after"]["F3"]["identity"] += 100
    reasons = R.verify_body(body, validated, execution)
    assert "method_F3_unexpected_cross_signal" in reasons


def test_timeout_or_fast_exit_cannot_be_a_valid_owned_execution():
    base = {"job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
            "cleanup_verified": True, "timed_out": False, "error_type": None}
    assert R._owned_execution_reasons(base) == []
    assert R._owned_execution_reasons({**base, "timed_out": True, "natural_exit": False}) == [
        "owned_execution_did_not_complete_naturally"
    ]
    assert R._owned_execution_reasons({**base, "cleanup_verified": False}) == [
        "owned_execution_did_not_complete_naturally"
    ]


def test_prepare_is_gpu_free_and_write_once(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    protocol_dir = repo / "bench/protocols"
    protocol_dir.mkdir(parents=True)
    monkeypatch.setattr(R, "ROOT", repo)
    monkeypatch.setattr(R, "PROTOCOL", protocol_dir / "actuator_envelope_runtime_v1.json")
    monkeypatch.setattr(R, "protocol_document", lambda _: {"schema": "prepared"})
    monkeypatch.setattr(R.M, "query_gpu", lambda *_: pytest.fail("prepare queried GPU"))
    path = protocol_dir / "actuator_envelope_runtime_v1.json"
    result = R.prepare(path, tmp_path / "isaac")
    assert result["state"] == "prepared" and result["gpu_queried"] is False
    with pytest.raises(FileExistsError):
        R.prepare(path, tmp_path / "isaac")


def test_gpu_profile_is_registered_in_existing_admission_api():
    profile = R.M._gpu_admission_profile(R.GPU_PROFILE)
    assert profile["policy"] == R.GPU_PROFILE
    assert R.FIXED["gpu_admission"] == profile


@pytest.mark.parametrize("section,field,wrong", [
    ("python", "implementation", "PyPy"),
    ("python", "version", [3, 10, 14]),
    ("python", "cache_tag", "cpython-311"),
    ("python", "executable", "kit/python/python.exe"),
    ("distributions", "torch", "2.7.0"),
    ("distributions", "rsl-rl-lib", "2.3.2"),
    ("distributions", "isaaclab", "0.41.2"),
    ("modules", "torch", "2.7.0"),
])
def test_prepare_runtime_binding_requires_each_exact_sdk_field(
        tmp_path, section, field, wrong):
    executable = tmp_path / "_isaac_sim" / R.B.SDK_PYTHON_RELATIVE
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    R._validate_runtime_binding(deepcopy(R.EXPECTED_RUNTIME_BINDING), tmp_path)
    runtime = deepcopy(R.EXPECTED_RUNTIME_BINDING)
    runtime[section][field] = wrong
    with pytest.raises(ValueError, match="SDK Python"):
        R._validate_runtime_binding(runtime, tmp_path)


def test_runtime_binding_requires_registered_sdk_relative_executable(tmp_path, monkeypatch):
    executable = tmp_path / "_isaac_sim" / R.B.SDK_PYTHON_RELATIVE
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setattr(R.B.sys, "executable", str(executable))
    versions = {"torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "isaaclab": "0.41.3"}
    monkeypatch.setattr(R.B.metadata, "version", versions.__getitem__)
    torch = type("Torch", (), {"__version__": "2.7.0+cu128"})
    runtime = R.B.runtime_binding(torch_module=torch, isaaclab_root=tmp_path)
    assert runtime["python"]["executable"] == R.B.SDK_PYTHON_RELATIVE
    assert runtime["distributions"] == versions
    assert runtime["modules"]["torch"] == "2.7.0+cu128"


def test_protocol_validation_rejects_stale_commit_and_source_tamper(tmp_path, monkeypatch):
    protocol = {**deepcopy(R.FIXED), "prepared_source_commit_sha": "not-a-commit",
                "expected_config": CONFIG,
                "asset_binding": {"source": R.B.GO2_SOURCE, "source_sha256": "b" * 64,
                                  "values": CONFIG},
                "method_binding_contract": {key: {} for key in R.B.TARGET_METHODS},
                "source_files": {name: "c" * 64 for name in R.PROJECT_SOURCE_FILES},
                "isaaclab_source_files": {name: "d" * 64 for name in R.ISAACLAB_SOURCE_FILES},
                "rsl_rl_source_files": {name: "e" * 64 for name in R.RSL_SOURCE_FILES}}
    monkeypatch.setattr(R.S, "_git_head", lambda _: "e" * 40)
    with pytest.raises(ValueError, match="invalid"):
        R.validate_protocol(protocol, repo_root=tmp_path, isaaclab_root=tmp_path)

    protocol["prepared_source_commit_sha"] = "a" * 40
    monkeypatch.setattr(R.S, "_git_head", lambda root: "a" * 40 if root.name == "repo" else R.B.ISAACLAB_COMMIT)
    monkeypatch.setattr(R, "__import__", __import__, raising=False)
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: type("C", (), {"returncode": 0})())
    monkeypatch.setattr(R.S, "_verify_source_map", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("SHA mismatch")))
    with pytest.raises(ValueError, match="SHA mismatch"):
        R.validate_protocol(protocol, repo_root=tmp_path / "repo", isaaclab_root=tmp_path / "isaac")


def test_gpu_busy_launches_zero_and_consumes_nothing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol = repo / "protocol.json"
    protocol.write_text(json.dumps({"gpu_admission_profile": R.GPU_PROFILE}), encoding="utf-8")
    monkeypatch.setattr(R, "protocol_location", lambda *_: protocol)
    monkeypatch.setattr(R.S, "_read_json", lambda *_: {"gpu_admission_profile": R.GPU_PROFILE})
    monkeypatch.setattr(R, "validate_protocol", lambda *_args, **_kwargs: {"source": True})
    monkeypatch.setattr(R.P, "verify_committed", lambda *_: "a" * 40)
    profiles = []
    monkeypatch.setattr(R.M, "query_gpu", lambda profile: profiles.append(profile) or {"busy": True})
    monkeypatch.setattr(R.P, "run_owned", lambda *_args, **_kwargs: pytest.fail("busy GPU launched"))
    monkeypatch.setattr(R, "PRIVATE_ROOT", tmp_path / "private")
    monkeypatch.setattr(R, "ROOT", repo)
    result = R.execute("tag", tmp_path / "isaac", protocol)
    assert result["status"] == "waiting_gpu_idle" and result["launch_count"] == 0
    assert result["request_consumed"] is False and not (tmp_path / "private/tag").exists()
    assert profiles == [R.GPU_PROFILE]


def test_protocol_location_accepts_default_and_preserves_old_protocol(tmp_path):
    assert R.protocol_location(R.PROTOCOL) == R.PROTOCOL.resolve()
    assert R.protocol_location(tmp_path / R.PROTOCOL_RELATIVE, tmp_path) == (tmp_path / R.PROTOCOL_RELATIVE).resolve()
    old = tmp_path / "bench/protocols/actuator_envelope_runtime_v1.json"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"original protocol")
    with pytest.raises(ValueError, match="fixed repository protocol"):
        R.protocol_location(old, tmp_path)
    assert old.read_bytes() == b"original protocol"


def test_request_consumption_is_global_write_once(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PRIVATE_ROOT", tmp_path / "private")
    first = tmp_path / "private/first"
    second = tmp_path / "private/second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    request = {"request_id": "a" * 64, "tag": "first"}
    R._consume(first, request)
    with pytest.raises(FileExistsError):
        R._consume(second, {**request, "tag": "second"})
    assert not (second / "consumed.json").exists()


@pytest.mark.parametrize("message", [
    "Windows fatal exception: access violation",
    "Fatal Python error: Segmentation fault",
    "Traceback (most recent call last):",
    "[Fatal] SDK shutdown failed",
])
def test_runtime_failure_log_cannot_be_hidden_by_successful_wrapper_exit(tmp_path, message):
    (tmp_path / "stdout.log").write_text("[Warning] benign SDK warning\n", encoding="utf-8")
    (tmp_path / "stderr.log").write_text("", encoding="utf-8")
    assert R._execution_log_reasons(tmp_path) == []
    (tmp_path / "stderr.log").write_text(message, encoding="utf-8")
    assert R._execution_log_reasons(tmp_path) == ["runtime_failure_in_stderr.log"]


def test_runtime_log_validation_requires_both_files(tmp_path):
    assert R._execution_log_reasons(tmp_path) == ["missing_stdout.log", "missing_stderr.log"]


def test_manifest_rejects_request_consumption_tamper(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    tag = repo / "bench/private/actuator_envelope_runtime/tag"
    condition_folder = tag / "NONE"
    condition_folder.mkdir(parents=True)
    protocol_path = repo / "bench/protocols/actuator_envelope_runtime_v1.json"
    protocol_path.parent.mkdir(parents=True)
    protocol = {"prepared_source_commit_sha": "a" * 40}
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(R, "protocol_location", lambda path, repo_root=repo: Path(path))
    monkeypatch.setattr(R, "validate_protocol", lambda *_args, **_kwargs: {"source": True})
    monkeypatch.setattr(R.S, "_git_head", lambda *_: "b" * 40)
    core = {"tag": "tag", "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": R.S._sha256_file(protocol_path), "project_commit_sha": "b" * 40,
            "prepared_source_commit_sha": "a" * 40, "conditions": list(R.B.CONDITIONS),
            "authorization_scope": "user_requested_development_inspection"}
    request = {"schema": "actuator_envelope_runtime_request_v1", **core,
               "request_id": R.S._canonical_sha(core)}
    consumption = {"schema": "actuator_envelope_runtime_consumption_v1",
                   "request_id": request["request_id"], "tag": "tag",
                   "request_sha256": R.S._canonical_sha(request)}
    (tag / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (tag / "consumed.json").write_text(json.dumps(consumption), encoding="utf-8")
    manifest = {"schema": "actuator_envelope_runtime_manifest_v1",
                "protocol_path": str(protocol_path.resolve()),
                "protocol_sha256": R.S._sha256_file(protocol_path), "condition": "NONE",
                "output_path": str((condition_folder / "body.json").resolve()),
                "request_path": str((tag / "request.json").resolve()),
                "consumption_path": str((tag / "consumed.json").resolve())}
    manifest_path = condition_folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = R.validate_manifest(manifest_path, repo_root=repo, isaaclab_root=tmp_path / "isaac")
    assert validated["request_lineage"]["request_id"] == request["request_id"]

    consumption["request_sha256"] = "0" * 64
    (tag / "consumed.json").write_text(json.dumps(consumption), encoding="utf-8")
    with pytest.raises(ValueError, match="consumption"):
        R.validate_manifest(manifest_path, repo_root=repo, isaaclab_root=tmp_path / "isaac")


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_owned_cpu_receipt_has_natural_exit_and_cleanup(tmp_path):
    execution = R.P.run_owned(tmp_path, [sys.executable, "-c", "pass"], tmp_path, 15)
    assert execution["job_assigned_before_release"]
    assert execution["natural_exit"] and execution["exit_code"] == 0
    assert execution["cleanup_verified"] and execution["owned_process_identities"]
