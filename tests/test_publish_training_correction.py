"""공개 교정 결과 export와 CPU 수치 replay 회귀."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

from rl_triage import grounded_recovery as C  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location(
        "publish_training_correction_under_test", ROOT / "evals/publish_training_correction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(module, name, *, lin=0.2, checkpoint="model_598.pt"):
    alive = 40_000
    rows = [{"command": list(command), "fall_rate": 0.0, "lin_vel_rmse_mps": lin,
             "yaw_rate_rmse_radps": 0.2, "num_envs": 40, "fall_count": 0,
             "alive_steps_sum": alive, "lin_error_sq_sum": lin * lin * alive,
             "yaw_error_sq_sum": 0.2 * 0.2 * alive} for command in C.GRID]
    versions = {"torch": "2.7.0", "gpu": "GPU", "isaaclab": "0.40.0",
                "isaaclab_tasks": "0.10.0", "isaaclab_rl": "0.10.0",
                "rsl-rl-lib": "2.3.3", "isaacsim": None}
    return {"protocol": C.PROTOCOL, "protocol_sha256_lf": module.H.sha(module.H.FIXED_EVAL, True),
            "task": C.TASK, "eval_seed": C.EVAL_SEED, "num_envs": C.NUM_ENVS,
            "horizon_steps": C.HORIZON_STEPS, "step_dt": C.STEP_DT,
            "action_scale": module.H.FIXED["action_scale"], "standstill": dict(C.STANDSTILL),
            "versions": versions, "name": name,
            "checkpoint": {"file": checkpoint, "sha256": "1" * 64},
            "metrics": {"fall_rate": 0.0, "mean_survival_s": alive * len(rows) / C.NUM_ENVS * C.STEP_DT,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": 0.2},
            "by_condition": rows, "elapsed_s": 10.0,
            "started_at": "2026-10-13T00:00:00+00:00",
            "finished_at": "2026-10-13T00:00:10+00:00"}


def _json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(monkeypatch, tmp_path):
    module = _module()
    private = tmp_path / "private/run"
    results = tmp_path / "results"
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    tag, case, commit = "public_test", "case_01", "a" * 40
    protocol_sha = module.H.sha(module.H.PROTOCOL, True)
    reports = {
        "clean_reference": _report(module, "clean_reference"),
        "fault_sham": _report(module, "fault_sham", lin=0.32),
        "corrected": _report(module, "corrected"),
    }
    originals = {
        "clean_reference": _report(module, "baseline_p0c_s2027", checkpoint="model_299.pt"),
        "fault": _report(module, "h02_s2027", lin=0.32, checkpoint="model_299.pt"),
    }
    runtime = reports["clean_reference"]["versions"]
    protocol = {"fixed_eval_sha256_lf": module.H.sha(module.H.FIXED_EVAL, True),
                "resume_inputs": {"bands": {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.1513,
                                                   "err_yaw_ratio_max": 1.1719, "fall_frac_max": 0.05},
                                  "runtime_versions": runtime, "summary_sha256_lf": "2" * 64,
                                  "runs": {
                                      "NONE": {"run": "baseline_p0c_s2027", "hashes": {"checkpoint": "3" * 64},
                                               "original_label": "healthy", "original_failed": []},
                                      "F1": {"run": "h02_s2027", "hashes": {"checkpoint": "4" * 64},
                                             "original_label": "unhealthy", "original_failed": ["err_xy_ratio"]},
                                  }}}
    request = {"tag": tag, "case": case, "source_commit_sha": commit,
               "protocol_sha256_lf": protocol_sha, "isaaclab_root": str(isaaclab.resolve())}
    _json(private / "request.json", request)
    original_paths = {}
    for key, report in originals.items():
        path = tmp_path / f"original_{key}.json"
        _json(path, report)
        original_paths[key] = path

    initials, receipts = {}, {}
    for arm_index, arm in enumerate(module.H.ARMS):
        training, evaluation = private / arm / "train", private / arm / "eval"
        train_wall = 10.0 + arm_index
        eval_wall = 2.5 + arm_index
        initial = {
            "policy_state": {"sha256": "5" * 64, "tensor_count": 1, "scalar_count": 0,
                             "tensors": [{"path": "$.actor.weight", "dtype": "float32",
                                          "shape": [2, 2], "content_sha256": "c" * 64}], "scalars": []},
            "optimizer_state": {"sha256": "6" * 64, "tensor_count": 1, "scalar_count": 1,
                                "tensors": [{"path": "$.state.0.exp_avg", "dtype": "float32",
                                             "shape": [2, 2], "content_sha256": "d" * 64}],
                                "scalars": [{"path": "$.param_groups.0.lr", "type": "float", "value": 0.001}]},
            "rng_state_before_learn": {"cpu_sha256": "7" * 64,
                                       "cuda_all_sha256": ["8" * 64], "cuda_device_count": 1},
            "runtime_versions": {"python": "3.10.15", "python_cache_tag": "cpython-310",
                                 "torch_distribution": "2.7.0", "isaaclab_distribution": "0.40.0",
                                 "rsl_rl_distribution": "2.3.3",
                                 "isaaclab_source_commit": module.H.FIXED["isaaclab_commit"]},
            "resume_checkpoint": {"path": str(tmp_path / "source/model_299.pt"),
                                  "sha256": "9" * 64, "load_optimizer": True},
            "resume_iteration": module.H.FIXED["resume_iteration"],
        }
        receipt = {
            "checkpoint": {"path": str(tmp_path / arm / "model_598.pt"), "sha256": "1" * 64},
            "configs": {"environment": {"path": str(tmp_path / arm / "params/env.yaml"), "sha256": "a" * 64},
                        "agent": {"path": str(tmp_path / arm / "params/agent.yaml"), "sha256": "b" * 64}},
            "resume_checkpoint": {"path": str(tmp_path / "source/model_299.pt"), "sha256": "9" * 64},
            "resume_iteration": module.H.FIXED["resume_iteration"],
            "final_iteration": module.H.FIXED["final_iteration"],
            "actual_control_steps": 7200, "aggregate_env_steps": 29_491_200,
            "budget": {"seed": 2027, "num_envs": 4096, "max_iterations": 300,
                       "num_steps_per_env": 24, "expected_control_steps": 7200},
            "observed": {"environment_seed": 2027, "environment_num_envs": 4096,
                         "action_scale": 0.25, "agent_seed": 2027,
                         "agent_max_iterations": 300, "runner_num_steps_per_env": 24},
            "runtime_versions": initial["runtime_versions"],
        }
        _json(training / "manifest.json", {"arm": arm})
        _json(training / "execution.json", {"complete": True, "wall_s": train_wall,
                                           "wall_scope": module.PARENT_WALL_SCOPE})
        _json(training / "initial.json", initial)
        _json(training / "receipt.json", receipt)
        _json(evaluation / "execution.json", {"complete": True, "wall_s": eval_wall,
                                             "wall_scope": module.PARENT_WALL_SCOPE})
        _json(evaluation / "report.json", reports[arm])
        _json(private / arm / "parent.json", {
            "status": "pass", "training_receipt_sha256": module.H.sha(training / "receipt.json"),
            "fixed_report_sha256": module.H.sha(evaluation / "report.json"),
            "train_parent_wall_s": train_wall, "eval_parent_wall_s": eval_wall})
        initials[arm], receipts[arm] = initial, receipt
    numeric = module.H.summarize(reports, initials, originals,
                                 protocol["resume_inputs"]["bands"], ["err_xy_ratio"])
    _json(private / "result.json", {"status": numeric["status"], "case": case,
                                    "source_commit_sha": commit})

    calls = {"manifest": [], "training": []}
    monkeypatch.setattr(module, "RESULTS", results)
    monkeypatch.setattr(module.H, "folder_for", lambda value: private)
    monkeypatch.setattr(module.H, "verify_protocol", lambda root: (protocol, commit))
    monkeypatch.setattr(module.H, "validate_manifest", lambda path: (
        calls["manifest"].append(path) or {"arm": path.parents[1].name, "case": case}, protocol))
    monkeypatch.setattr(module.H, "verify_training", lambda folder, arm, execution: (
        calls["training"].append(arm) or receipts[arm]))
    monkeypatch.setattr(module.H, "execution_valid", lambda row: row.get("complete") is True)
    monkeypatch.setattr(module.H, "source_paths", lambda root, fault: {
        "fixed_report": original_paths["clean_reference" if fault == "NONE" else "fault"]})
    return module, private, results, isaaclab, tag, calls


def test_export_and_numeric_replay_use_all_verified_arms(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, calls = _fixture(monkeypatch, tmp_path)
    exported = module.export(tag, isaaclab)
    assert exported["status"] == "restoration_observed_single_run"
    assert exported["folder"] == f"evals/results/{tag}"
    assert calls["training"] == list(module.H.ARMS) and len(calls["manifest"]) == 3
    public = results / tag
    combined = "".join(path.read_text(encoding="utf-8") for path in public.rglob("*.json"))
    assert str(tmp_path) not in combined and "process_id" not in combined
    replayed = module.replay(public)
    assert replayed["status"] == exported["status"]
    assert replayed["replay_scope"] == module.REPLAY_SCOPE
    assert replayed["parent_wall_scope"] == module.PARENT_WALL_SCOPE
    assert replayed["parent_wall_s_by_arm"]["clean_reference"] == {
        "train_s": 10.0, "fixed_evaluation_s": 2.5, "total_s": 12.5,
    }
    assert replayed["total_parent_wall_s"] == pytest.approx(43.5)
    public_initial = json.loads((public / "initial_states.json").read_text(encoding="utf-8"))
    assert public_initial["arms"]["corrected"]["policy_state"] == {
        "sha256": "5" * 64, "tensor_count": 1, "scalar_count": 0,
    }
    public_result = json.loads((public / "result.json").read_text(encoding="utf-8"))
    public_manifest = json.loads((public / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert public_result["source_commit_scope"] == module.SOURCE_COMMIT_SCOPE
    assert public_result["publisher_sources"] == public_manifest["publisher_sources"] == module._publisher_sources()
    assert set(public_result["publisher_sources"]) == module.PUBLISHER_SOURCES


def test_replay_rejects_hash_tamper_and_missing_artifact(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    report = public / "reports/corrected.json"
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        module.replay(public)
    report.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        module.replay(public)


def _rehash(public: Path, relative: str) -> None:
    manifest_path = public / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = hashlib.sha256((public / relative).read_bytes()).hexdigest()
    _json(manifest_path, manifest)


def test_replay_rejects_forbidden_field_even_after_manifest_rehash(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    relative = "training_evidence.json"
    evidence = json.loads((public / relative).read_text(encoding="utf-8"))
    evidence["arms"]["corrected"]["process_id"] = 1234
    _json(public / relative, evidence)
    _rehash(public, relative)
    with pytest.raises(ValueError, match="private execution field"):
        module.replay(public)


def test_replay_rejects_arm_schema_change_after_manifest_rehash(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    relative = "initial_states.json"
    initial = json.loads((public / relative).read_text(encoding="utf-8"))
    initial["arms"].pop("fault_sham")
    _json(public / relative, initial)
    _rehash(public, relative)
    with pytest.raises(ValueError, match="arm evidence schema"):
        module.replay(public)


def test_replay_rejects_current_publisher_source_change(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    changed = module._publisher_sources()
    changed["src/rl_triage/recovery.py"] = "0" * 64
    monkeypatch.setattr(module, "_publisher_sources", lambda: changed)
    with pytest.raises(ValueError, match="source LF-SHA"):
        module.replay(public)


def test_replay_rejects_manifest_result_source_binding_mismatch(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    relative = "result.json"
    result = json.loads((public / relative).read_text(encoding="utf-8"))
    result["publisher_sources"]["evals/run_training_correction.py"] = "0" * 64
    _json(public / relative, result)
    _rehash(public, relative)
    with pytest.raises(ValueError, match="result and manifest"):
        module.replay(public)


def test_replay_rejects_manifest_source_sha_mismatch(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    manifest_path = public / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publisher_sources"]["evals/publish_training_correction.py"] = "0" * 64
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source LF-SHA"):
        module.replay(public)


@pytest.mark.parametrize("arm_stage", ["train", "eval"])
@pytest.mark.parametrize("scope", [None, "gpu_kernel_time"])
def test_export_rejects_missing_or_changed_wall_scope(monkeypatch, tmp_path, arm_stage, scope):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    path = private / "corrected" / arm_stage / "execution.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    if scope is None:
        row.pop("wall_scope")
    else:
        row["wall_scope"] = scope
    _json(path, row)
    with pytest.raises(ValueError, match="parent wall scope"):
        module.export(tag, isaaclab)


def test_replay_rejects_invalid_public_parent_wall_after_rehash(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    module.export(tag, isaaclab)
    public = results / tag
    relative = "training_evidence.json"
    evidence = json.loads((public / relative).read_text(encoding="utf-8"))
    evidence["arms"]["corrected"]["parent_wall_s"]["train_s"] = math.nan
    _json(public / relative, evidence)
    _rehash(public, relative)
    with pytest.raises(ValueError, match="parent wall time is invalid"):
        module.replay(public)


@pytest.mark.parametrize("mutation", ["mismatch", "nan"])
def test_export_rejects_parent_and_execution_wall_time_disagreement(monkeypatch, tmp_path, mutation):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    parent_path = private / "fault_sham/parent.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["train_parent_wall_s"] = 99.0 if mutation == "mismatch" else math.nan
    _json(parent_path, parent)
    expected = "parent wall time" if mutation == "mismatch" else "non-finite JSON"
    with pytest.raises(ValueError, match=expected):
        module.export(tag, isaaclab)


def test_export_rejects_source_commit_change(monkeypatch, tmp_path):
    module, _, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(module.H, "verify_protocol", lambda root: ({}, "changed"))
    with pytest.raises(ValueError, match="current verified source"):
        module.export(tag, isaaclab)


def test_export_rejects_incomplete_or_invalid_experiment(monkeypatch, tmp_path):
    module, private, _, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    (private / "fault_sham/eval/execution.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.export(tag, isaaclab)


def test_export_rejects_path_leak_before_creating_output(monkeypatch, tmp_path):
    module, private, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    for arm in module.H.ARMS:
        initial_path = private / arm / "train/initial.json"
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        initial["runtime_versions"]["torch_distribution"] = "C:\\private\\torch"
        _json(initial_path, initial)
    with pytest.raises(ValueError, match="path leaked"):
        module.export(tag, isaaclab)
    assert not (results / tag).exists()


def test_export_never_overwrites_existing_public_folder(monkeypatch, tmp_path):
    module, _, results, isaaclab, tag, _ = _fixture(monkeypatch, tmp_path)
    (results / tag).mkdir(parents=True)
    marker = results / tag / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.export(tag, isaaclab)
    assert marker.read_text(encoding="utf-8") == "keep"
