"""Actuator envelope 공개 export와 CPU 수치 replay 회귀."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))


def _module():
    spec = importlib.util.spec_from_file_location(
        "publish_actuator_envelope_under_test", ROOT / "evals/publish_actuator_envelope.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_body_with_marker(module, folder: Path, body: dict) -> str:
    body_path = folder / "body.json"
    _write(body_path, body)
    body_sha = module._sha(body_path)
    marker = {"condition": body["condition"], "manifest_sha256": body["manifest_sha256"],
              "body_sha256": body_sha}
    (folder / "stdout.log").write_text(
        module.B.SHUTDOWN_PREFIX + json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8",
    )
    return body_sha


def _helper(module, matches: bool) -> dict:
    config = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}
    rows = []
    for velocity in module.B.E.VELOCITY_GRID:
        minimum, maximum = module.B.E._documented_limits(velocity, config)
        rows.append({
            "velocity_rad_s": velocity, "expected_min_effort": minimum,
            "expected_max_effort": maximum, "measured_min_effort": [minimum] * 12,
            "measured_max_effort": [maximum] * 12, "matches_expected": matches,
        })
    if not matches:
        for row in rows:
            row["measured_max_effort"] = [row["expected_max_effort"] - 1.0] * 12
    return {
        "schema": "actuator_envelope_readback_v1", "status": "valid",
        "curve_matches_expected": matches, "invalid_reason": None,
        "requested_efforts": {"positive": 23500.0, "negative": -23500.0},
        "measurements": rows,
        "config_provenance": {
            "caller_asserted_source_sha256": "a" * 64, "source_binding_verified": False,
            "expected_config": config, "velocity_grid_rad_s": list(module.B.E.VELOCITY_GRID),
            "runtime_config_readback": config,
        },
        "limitations": deepcopy(module.B.E.LIMITATIONS),
    }


def _fixture(monkeypatch, tmp_path):
    module = _module()
    tag = "actuator_public_test"
    private = tmp_path / "private" / tag
    results = tmp_path / "results"
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir(parents=True)
    commit, protocol_sha = "5" * 40, "6" * 64
    source_receipt = {
        "project_source_files": {name: "1" * 64 for name in module.R.PROJECT_SOURCE_FILES},
        "isaaclab_source_files": {name: "2" * 64 for name in module.R.ISAACLAB_SOURCE_FILES},
        "rsl_rl_source_files": {name: "3" * 64 for name in module.R.RSL_SOURCE_FILES},
        "asset_binding": {}, "project_commit_sha": commit, "prepared_source_commit_sha": commit,
    }
    executions, bodies, rows = {}, {}, []
    request = {"tag": tag, "request_id": "r"}
    consumption = {"request_id": "r", "consumed": True}
    _write(private / "request.json", request)
    _write(private / "consumed.json", consumption)
    for condition in module.B.CONDITIONS:
        folder = private / condition
        manifest_path, body_path = folder / "manifest.json", folder / "body.json"
        _write(manifest_path, {"condition": condition})
        (folder / "stderr.log").write_text("", encoding="utf-8")
        matches = module.B.EXPECTED_MATCH[condition]
        state = {
            "schema": "actuator_envelope_state_v1",
            "buffers": {name: {"identity": index + 1, "shape": [1, 12], "values": [0.0] * 12}
                        for index, name in enumerate(module.B.STATE_BUFFER_FIELDS)},
            "common_step_counter": 0, "sim_step_counter": 0,
        }
        config = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}
        body = {
            "condition": condition, "manifest_sha256": module._sha(manifest_path),
            "helper_result": _helper(module, matches),
            "state_before": state, "state_after": deepcopy(state),
            "actuator_binding": {
                "class_name": "DCMotor", "class_module": "isaaclab.actuators.actuator_pd",
                "joint_names": sorted(module.B.GO2_JOINT_NAMES), "joint_count": 12,
                "joint_velocity_shape": [1, 12], "device": "cuda:0", "runtime_config": config,
                "class_source": {"sha256": "4" * 64},
                "clip_effort_source": {"sha256": "5" * 64, "code_sha256": "7" * 64},
                "asset_source": {"sha256": "a" * 64},
            },
            "source_receipt": source_receipt,
            "sdk_versions": {
                "isaaclab_commit": module.B.ISAACLAB_COMMIT, "isaacsim": "4.5.0",
                "isaacsim_raw_components": [
                    "4.5.0", "rc.36", "4", "5", "0", "rc", "36",
                    "release.19112.f59b3005.gl",
                ],
                "runtime_binding": deepcopy(module.R.EXPECTED_RUNTIME_BINDING),
            },
            "curve_assessment": "normal" if matches else "curve_difference",
            "protocol_status": "expected", "started_at": "2026-10-13T00:00:00+00:00",
            "finished_at": "2026-10-13T00:00:01+00:00", "wall_elapsed_s": 1.0,
        }
        body_sha = _write_body_with_marker(module, folder, body)
        execution = {
            "job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True,
            "owned_process_identities": {"123": 456}, "wall_s": 2.0,
            "wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification",
        }
        executions[condition], bodies[condition] = execution, body
        rows.append({
            "condition": condition, "status": "pass", "validation_reasons": [],
            "body_sha256": body_sha, "measurement_status": "valid",
            "curve_assessment": body["curve_assessment"], "protocol_status": "expected",
            "execution": execution,
        })
    parent = {
        "schema": "actuator_envelope_runtime_parent_v1", "status": "complete", "tag": tag,
        "project_commit_sha": commit, "protocol_sha256": protocol_sha, "request_id": "r",
        "consumption": consumption, "request_consumed": True, "launch_count": 4,
        "conditions": rows, "gpu_before": {}, "gpu_after": {},
        "cause_or_recovery_evidence": False,
    }
    _write(private / "parent_receipt.json", parent)
    calls = {"manifest": [], "body": [], "completed_output": []}

    def validate(path, *, repo_root, isaaclab_root, completed_output=False):
        condition = path.parent.name
        calls["manifest"].append(condition)
        calls["completed_output"].append(completed_output)
        if completed_output is not True or not (path.parent / "body.json").is_file():
            raise ValueError("completed output contract mismatch")
        return {"manifest": {"condition": condition}, "protocol_sha256": protocol_sha,
                "request_lineage": {"request_id": "r"},
                "paths": {"request_path": private / "request.json",
                          "consumption_path": private / "consumed.json"}}

    def verify(body, validated, execution):
        calls["body"].append(validated["manifest"]["condition"])
        return []

    monkeypatch.setattr(module, "PRIVATE_ROOT", private.parent)
    monkeypatch.setattr(module, "RESULTS", results)
    monkeypatch.setattr(module.R.S, "_git_head", lambda _: commit)
    monkeypatch.setattr(module.R, "validate_manifest", validate)
    monkeypatch.setattr(module.R, "verify_body", verify)
    return module, private, results, isaaclab, tag, calls


def _rehash(module, folder: Path) -> None:
    manifest_path = folder / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["result.json"] = module._sha(folder / "result.json")
    _write(manifest_path, manifest)


def test_export_revalidates_all_conditions_and_numeric_replay(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, calls = _fixture(monkeypatch, tmp_path)
    exported = module.export(tag, isaaclab)
    assert exported == {"status": "exported", "folder": f"evals/results/{tag}",
                        "replay_scope": module.REPLAY_SCOPE, "condition_count": 4}
    assert calls["manifest"] == list(module.B.CONDITIONS)
    assert calls["completed_output"] == [True] * len(module.B.CONDITIONS)
    assert calls["body"] == list(module.B.CONDITIONS)
    assert module.PUBLISHER_SOURCES == {"evals/publish_actuator_envelope.py"} | module.R.PROJECT_SOURCE_FILES
    public = results / tag
    raw = "".join(path.read_text(encoding="utf-8") for path in public.glob("*.json"))
    assert str(tmp_path) not in raw and "process_id" not in raw and "owned_process_identities" not in raw
    replayed = module.replay(public)
    assert replayed["status"] == "valid"
    assert replayed["conditions"]["F2"] == {
        "curve_assessment": "curve_difference", "protocol_status": "expected",
        "expectation_met": True,
    }


@pytest.mark.parametrize("mutation,error", [
    (lambda parent, private: parent.update(status="invalid"), "parent"),
    (lambda parent, private: parent.update(tag="another_tag"), "parent"),
    (lambda parent, private: parent["conditions"].pop(), "condition"),
    (lambda parent, private: (private / "F3" / "body.json").unlink(), "missing"),
    (lambda parent, private: parent["conditions"][0].update(body_sha256="0" * 64), "SHA"),
])
def test_export_rejects_invalid_missing_cross_tag_and_tamper(
        mutation, error, monkeypatch, tmp_path):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    parent_path = private / "parent_receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    mutation(parent, private)
    _write(parent_path, parent)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        module.export(tag, isaaclab)


def test_export_rejects_nonfinite_curve_and_fatal_log(monkeypatch, tmp_path):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    body_path = private / "NONE" / "body.json"
    body = json.loads(body_path.read_text(encoding="utf-8"))
    body["helper_result"]["measurements"][0]["measured_max_effort"][0] = float("nan")
    _write_body_with_marker(module, body_path.parent, body)
    parent_path = private / "parent_receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["conditions"][0]["body_sha256"] = module._sha(body_path)
    _write(parent_path, parent)
    with pytest.raises(ValueError, match="non-finite"):
        module.export(tag, isaaclab)

    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path / "fatal")
    (private / "NONE" / "stderr.log").write_text("Fatal Python error", encoding="utf-8")
    with pytest.raises(ValueError, match="log"):
        module.export(tag, isaaclab)


@pytest.mark.parametrize("relative,raw,error", [
    ("parent_receipt.json", '{"schema":"first","schema":"second"}', "duplicate"),
    ("parent_receipt.json", '{"value":NaN}', "non-finite"),
    ("NONE/body.json", '{"helper_result":{},"helper_result":{}}', "duplicate"),
    ("NONE/body.json", '{"value":Infinity}', "non-finite"),
])
def test_export_strict_json_rejects_private_duplicate_and_nonfinite(
        relative, raw, error, monkeypatch, tmp_path):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    (private / relative).write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        module.export(tag, isaaclab)


def test_export_is_write_once(monkeypatch, tmp_path):
    module, _, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    with pytest.raises(FileExistsError):
        module.export(tag, isaaclab)


def test_valid_unexpected_curve_is_published_without_masking(monkeypatch, tmp_path):
    module, private, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    body_path = private / "F3" / "body.json"
    body = json.loads(body_path.read_text(encoding="utf-8"))
    body["helper_result"] = _helper(module, False)
    body["curve_assessment"] = "curve_difference"
    body["protocol_status"] = "unexpected"
    _write_body_with_marker(module, body_path.parent, body)
    parent_path = private / "parent_receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["conditions"][2]["body_sha256"] = module._sha(body_path)
    parent["conditions"][2]["curve_assessment"] = "curve_difference"
    parent["conditions"][2]["protocol_status"] = "unexpected"
    _write(parent_path, parent)
    module.export(tag, isaaclab)
    assert module.replay(results / tag)["conditions"]["F3"] == {
        "curve_assessment": "curve_difference", "protocol_status": "unexpected",
        "expectation_met": False,
    }


@pytest.mark.parametrize("exit_code", [False, None, "0", 1])
def test_export_rejects_non_strict_zero_exit_code(exit_code, monkeypatch, tmp_path):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    parent_path = private / "parent_receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["conditions"][0]["execution"]["exit_code"] = exit_code
    _write(parent_path, parent)
    with pytest.raises(ValueError, match="owned execution"):
        module.export(tag, isaaclab)


@pytest.mark.parametrize("exit_code", [False, None, "0", 1])
def test_replay_rejects_non_strict_zero_exit_code(exit_code, monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    result_path = public / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["conditions"]["NONE"]["owned_execution"]["exit_code"] = exit_code
    _write(result_path, result)
    _rehash(module, public)
    with pytest.raises(ValueError, match="owned execution"):
        module.replay(public)


def test_replay_rejects_hash_missing_curve_and_restoration_tamper(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    result_path = public / "result.json"
    result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        module.replay(public)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["conditions"]["F2"]["curve_assessment"] = "normal"
    _write(result_path, result)
    _rehash(module, public)
    with pytest.raises(ValueError, match="curve verdict"):
        module.replay(public)

    result["conditions"]["F2"]["curve_assessment"] = "curve_difference"
    result["conditions"]["NONE"]["state_after"]["buffers"]["_joint_vel"]["values"][0] = 1.0
    _write(result_path, result)
    _rehash(module, public)
    with pytest.raises(ValueError, match="restoration"):
        module.replay(public)

    result_path.unlink()
    with pytest.raises(ValueError, match="file set"):
        module.replay(public)


def test_replay_rejects_forbidden_execution_identity_after_rehash(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    result_path = public / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["conditions"]["NONE"]["process_id"] = 123
    _write(result_path, result)
    _rehash(module, public)
    with pytest.raises(ValueError, match="not publishable"):
        module.replay(public)


@pytest.mark.parametrize("name,raw,error", [
    ("artifact_manifest.json", '{"schema":"first","schema":"second"}', "duplicate"),
    ("artifact_manifest.json", '{"value":-Infinity}', "non-finite"),
    ("result.json", '{"schema":"first","schema":"second"}', "duplicate"),
    ("result.json", '{"value":NaN}', "non-finite"),
])
def test_replay_strict_json_rejects_public_duplicate_and_nonfinite(
        name, raw, error, monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    (public / name).write_text(raw, encoding="utf-8")
    if name == "result.json":
        _rehash(module, public)
    with pytest.raises(ValueError, match=error):
        module.replay(public)


def test_replay_rejects_unexpected_empty_directory(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    (public / "unexpected").mkdir()
    with pytest.raises(ValueError, match="file set"):
        module.replay(public)


def test_replay_rejects_unexpected_directory_symlink_entry(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    linked = public / "linked_directory"
    linked.mkdir()
    path_type = type(linked)
    original = path_type.is_symlink
    monkeypatch.setattr(path_type, "is_symlink", lambda self: self == linked or original(self))
    with pytest.raises(ValueError, match="file set"):
        module.replay(public)


def test_replay_rejects_manifest_symlink(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    target = public / "artifact_manifest.json"
    path_type = type(target)
    original = path_type.is_symlink
    monkeypatch.setattr(path_type, "is_symlink", lambda self: self == target or original(self))
    with pytest.raises(ValueError, match="unsafe"):
        module.replay(public)


def test_replay_rejects_result_resolving_outside_folder(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    target = public / "result.json"
    external = tmp_path / "external_result.json"
    external.write_bytes(target.read_bytes())
    path_type = type(target)
    original = path_type.resolve

    def resolve(path, strict=False):
        if path == target:
            return original(external, strict=strict)
        return original(path, strict=strict)

    monkeypatch.setattr(path_type, "resolve", resolve)
    with pytest.raises(ValueError, match="unsafe"):
        module.replay(public)
