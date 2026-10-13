"""v4 seed 결과의 공개 export·CPU 수치 replay·실험 커밋 고정 검증 회귀."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import publish_training_correction_v4 as PUB  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402

H = PUB.H
COMMIT = "a" * 40
SEED = 2026
LINS = {"clean_resume_reference": 0.2, "resume_corrected": 1.17,
        "resume_corrected_std_reset": 0.2, "restart_corrected": 0.2}
BANDS = {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.1513,
         "err_yaw_ratio_max": 1.1719, "fall_frac_max": 0.05}


def _report(name, *, lin=0.2, checkpoint="model_598.pt"):
    alive = 40_000
    rows = [{"command": list(command), "fall_rate": 0.0, "lin_vel_rmse_mps": lin,
             "yaw_rate_rmse_radps": 0.2, "num_envs": 40, "fall_count": 0,
             "alive_steps_sum": alive, "lin_error_sq_sum": lin * lin * alive,
             "yaw_error_sq_sum": 0.2 * 0.2 * alive} for command in C.GRID]
    versions = {"torch": "2.7.0+cu128", "gpu": "GPU", "isaaclab": "0.41.3", "isaaclab_tasks": "0.10.36",
                "isaaclab_rl": "0.1.8", "rsl-rl-lib": "2.3.3", "isaacsim": None}
    return {"protocol": C.PROTOCOL, "protocol_sha256_lf": H.sha(H.FIXED_EVAL, True),
            "task": C.TASK, "eval_seed": C.EVAL_SEED, "num_envs": C.NUM_ENVS,
            "horizon_steps": C.HORIZON_STEPS, "step_dt": C.STEP_DT,
            "action_scale": H.FIXED["action_scale"], "standstill": dict(C.STANDSTILL),
            "versions": versions, "name": name, "checkpoint": {"file": checkpoint, "sha256": "1" * 64},
            "metrics": {"fall_rate": 0.0, "mean_survival_s": alive * len(rows) / C.NUM_ENVS * C.STEP_DT,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": 0.2},
            "by_condition": rows, "elapsed_s": 10.0,
            "started_at": "2026-10-13T00:00:00+00:00", "finished_at": "2026-10-13T00:00:10+00:00"}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(char):
    return {"sha256": char * 64, "tensor_count": 1, "scalar_count": 0,
            "tensors": [{"path": "$.w", "dtype": "float32", "shape": [1], "content_sha256": "c" * 64}],
            "scalars": []}


def _arm_rows(tmp_path: Path, arm: str) -> tuple[dict, dict]:
    mode = H.ARM_SPECS[arm]["recovery_mode"]
    resumed = mode != "restart"
    final = H.FINAL_ITERATION[mode]
    source_ckpt = tmp_path / ("none" if arm == "clean_resume_reference" else "f1") / "model_299.pt"
    source_sha = ("8" if arm == "clean_resume_reference" else "9") * 64
    source_state = {"iteration": 299, "policy_state_sha256": "3" * 64, "optimizer_state_sha256": "4" * 64}
    versions = {"python": "3.10.15", "python_cache_tag": "cpython-310", "torch_distribution": "2.7.0+cu128",
                "isaaclab_distribution": "0.41.3", "rsl_rl_distribution": "2.3.3",
                "isaaclab_source_commit": H.FIXED["isaaclab_commit"]}
    std_reset = ({"schema": "action_std_reset_v2", "policy_state_before_reset": "3" * 64,
                  "optimizer_state_before_reset": "4" * 64, "before": [0.4], "after": [1.0]}
                 if mode == "resume_std_reset" else None)
    initial = {
        "recovery_mode": mode,
        "policy_state": _digest("3" if mode == "resume" and arm != "clean_resume_reference" else "5"),
        "optimizer_state": _digest("4" if arm == "resume_corrected" else "6"),
        "rng_state_before_learn": {"cpu_sha256": "7" * 64, "cuda_all_sha256": ["7" * 64], "cuda_device_count": 1},
        "runtime_versions": versions,
        "resume_checkpoint": {"path": str(source_ckpt), "sha256": source_sha, "load_optimizer": True} if resumed else None,
        "resume_iteration": 299 if resumed else None, "start_iteration": 299 if resumed else 0,
        "source_checkpoint_state": source_state if resumed else None, "std_reset": std_reset,
    }
    fault, correction, _ = H.selection(arm)
    receipt = {
        "recovery_mode": mode, "fault": fault, "correction_id": correction,
        "checkpoint": {"path": str(tmp_path / f"2026-10-13_00-00-00_tag_{arm}" / f"model_{final}.pt"),
                       "sha256": "1" * 64},
        "configs": {"environment": {"path": str(tmp_path / arm / "params/env.yaml"), "sha256": "a" * 64},
                    "agent": {"path": str(tmp_path / arm / "params/agent.yaml"), "sha256": "b" * 64}},
        "resume_checkpoint": {"path": str(source_ckpt), "sha256": source_sha} if resumed else None,
        "resume_iteration": 299 if resumed else None, "start_iteration": 299 if resumed else 0,
        "final_iteration": final, "checkpoint_load_calls": 1 if resumed else 0,
        "std_reset_applied": mode == "resume_std_reset",
        "resume_scope": {"policy_optimizer_restored": resumed, "full_environment_state_restored": False,
                         "rng_state_restored_from_checkpoint": False, "fresh_initialization": not resumed,
                         "action_std_reset": mode == "resume_std_reset"},
        "actual_control_steps": 7200, "aggregate_env_steps": 29_491_200,
        "budget": {"seed": SEED, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
                   "expected_control_steps": 7200},
        "observed": {"environment_seed": SEED, "environment_num_envs": 4096, "action_scale": 0.25,
                     "agent_seed": SEED, "agent_max_iterations": 300, "runner_num_steps_per_env": 24},
        "runtime_versions": versions,
    }
    return initial, receipt


def _fixture(monkeypatch, tmp_path, *, failed_arm=None, restart_lin=0.2):
    private, results, isaaclab = tmp_path / "private/run", tmp_path / "evals/results", tmp_path / "IsaacLab"
    isaaclab.mkdir()
    _json(private / "approval.json", {"actor": H.FIXED["approval_actor"]})
    tag = f"reward_v4_s{SEED}"
    lins = {**LINS, "restart_corrected": restart_lin}
    completed = list(H.ARMS[:H.ARMS.index(failed_arm)]) if failed_arm else list(H.ARMS)
    reports = {arm: _report(arm, lin=lins[arm],
                            checkpoint=f"model_{H.FINAL_ITERATION[H.ARM_SPECS[arm]['recovery_mode']]}.pt")
               for arm in completed}
    originals = {"clean_reference": _report("baseline_p0c_s2026", checkpoint="model_299.pt"),
                 "fault": _report("h02_s2026", lin=1.17, checkpoint="model_299.pt")}
    runs = {"NONE": {"run": "baseline_p0c_s2026", "hashes": {"checkpoint": "3" * 64}, "original_label": "healthy",
                     "original_failed": [], "standstill_ratio": 0.17},
            "F1": {"run": "h02_s2026", "hashes": {"checkpoint": "4" * 64}, "original_label": "unhealthy",
                   "original_failed": ["err_xy_ratio"], "standstill_ratio": 0.99}}
    protocol = {"fixed_eval_sha256_lf": H.sha(H.FIXED_EVAL, True),
                "resume_inputs": {"bands": BANDS, "runtime_versions": reports[completed[0]]["versions"],
                                  "summary_sha256_lf": "2" * 64, "init_noise_std": 1.0,
                                  "seeds": {str(SEED): {"runs": runs}}}}
    _json(private / "request.json", {"tag": tag, "seed": SEED, "source_commit_sha": COMMIT,
                                     "protocol_sha256_lf": H.sha(H.PROTOCOL, True),
                                     "isaaclab_root": str(isaaclab.resolve())})
    original_paths = {}
    for key, report in originals.items():
        original_paths[key] = tmp_path / f"original_{key}.json"
        _json(original_paths[key], report)
    initials, receipts = {}, {}
    for index, arm in enumerate(completed):
        training, evaluation = private / arm / "train", private / arm / "eval"
        initial, receipt = _arm_rows(tmp_path, arm)
        _json(training / "manifest.json", {"arm": arm})
        _json(private / arm / "consumed.json", {"arm": arm})
        _json(training / "correction.json", {"selected_correction": receipt["correction_id"]})
        _json(training / "execution.json", {"complete": True, "wall_s": 10.0 + index, "wall_scope": PUB.PARENT_WALL_SCOPE})
        _json(training / "initial.json", initial)
        _json(training / "receipt.json", receipt)
        _json(evaluation / "execution.json", {"complete": True, "wall_s": 2.5, "wall_scope": PUB.PARENT_WALL_SCOPE})
        _json(evaluation / "report.json", reports[arm])
        _json(private / arm / "parent.json", {
            "status": "pass", "training_receipt_sha256": H.sha(training / "receipt.json"),
            "fixed_report_sha256": H.sha(evaluation / "report.json"),
            "train_parent_wall_s": 10.0 + index, "eval_parent_wall_s": 2.5})
        initials[arm], receipts[arm] = initial, receipt
    if failed_arm:
        _json(private / "failure.json", {"arm": failed_arm, "stage": "training", "error_type": "ValueError",
                                         "detail": "C:/private/detail"})
    numeric = H.summarize(SEED, reports, initials, originals, BANDS, ["err_xy_ratio"], failed_arm)
    _json(private / "result.json", {**{key: numeric[key] for key in (
        "status", "seed", "labels", "label_reasons", "completed_arms", "failed_arm")},
        "tag": tag, "source_commit_sha": COMMIT, "protocol_sha256_lf": H.sha(H.PROTOCOL, True)})
    monkeypatch.setattr(PUB, "RESULTS", results)
    monkeypatch.setattr(H, "folder_for", lambda value: private)
    monkeypatch.setattr(H, "verify_protocol", lambda root: (protocol, COMMIT))
    monkeypatch.setattr(H, "validate_manifest", lambda path: ({"arm": path.parents[1].name, "seed": SEED}, protocol))
    monkeypatch.setattr(H, "verify_training", lambda folder, arm, execution: receipts[arm])
    monkeypatch.setattr(H, "execution_valid", lambda row: row.get("complete") is True)
    monkeypatch.setattr(H, "source_paths", lambda root, seed, fault: {
        "fixed_report": original_paths["clean_reference" if fault == "NONE" else "fault"]})
    return private, results, isaaclab, tag


def _rehash(public: Path, relative: str) -> None:
    manifest_path = public / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = hashlib.sha256((public / relative).read_bytes()).hexdigest()
    _json(manifest_path, manifest)


def test_export_and_replay_publish_four_arm_labels_without_private_paths(monkeypatch, tmp_path):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    exported = PUB.export(tag, isaaclab)
    assert exported["status"] == "complete" and exported["seed"] == SEED
    assert exported["labels"] == {"resume_corrected": "not_recovered", "resume_corrected_std_reset": "recovered",
                                  "restart_corrected": "recovered"}
    public = results / tag
    files = {path.relative_to(public).as_posix() for path in public.rglob("*.json")}
    assert files == PUB.public_files(list(H.ARMS)) | {"artifact_manifest.json"}
    combined = "".join(path.read_text(encoding="utf-8") for path in public.rglob("*.json"))
    assert str(tmp_path) not in combined and "process_id" not in combined and '"path"' not in combined
    replayed = PUB.replay(public)
    assert replayed["labels"] == exported["labels"]
    assert replayed["recovery_mode_rule"]["outcome"] == "hit"
    assert replayed["parent_wall_s_by_arm"]["clean_resume_reference"] == {
        "train_s": 10.0, "fixed_evaluation_s": 2.5, "total_s": 12.5}
    assert replayed["total_parent_wall_s"] == pytest.approx(4 * 12.5 + 6.0)
    initial = json.loads((public / "initial_states.json").read_text(encoding="utf-8"))["arms"]
    assert initial["restart_corrected"]["resume_checkpoint"] is None
    assert initial["resume_corrected"]["resume_checkpoint"] == {"file": "model_299.pt", "sha256": "9" * 64,
                                                                "load_optimizer": True}
    result = json.loads((public / "result.json").read_text(encoding="utf-8"))
    assert result["publisher_sources"] == PUB._publisher_sources()
    assert set(result["publisher_sources"]) == PUB.PUBLISHER_SOURCES


def test_partial_seed_publishes_only_completed_arms(monkeypatch, tmp_path):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path, failed_arm="restart_corrected")
    exported = PUB.export(tag, isaaclab)
    assert exported["status"] == "partial" and exported["labels"]["restart_corrected"] == "invalid"
    public = results / tag
    assert not (public / "reports/restart_corrected.json").exists()
    combined = "".join(path.read_text(encoding="utf-8") for path in public.rglob("*.json"))
    assert "C:/private/detail" not in combined
    replayed = PUB.replay(public)
    assert replayed["label_reasons"]["restart_corrected"] == "execution_failed"
    assert set(replayed["parent_wall_s_by_arm"]) == set(H.ARMS[:3])


@pytest.mark.parametrize("mutation", ["hash", "numeric_rehash", "forbidden_rehash", "arm_set_rehash", "extra_file"])
def test_replay_rejects_tampered_public_bundle(monkeypatch, tmp_path, mutation):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    PUB.export(tag, isaaclab)
    public = results / tag
    if mutation == "hash":
        path = public / "reports/restart_corrected.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expected = "hash"
    elif mutation == "numeric_rehash":
        report = json.loads((public / "reports/restart_corrected.json").read_text(encoding="utf-8"))
        report["metrics"]["lin_vel_rmse_mps"] = 1.17
        for row in report["by_condition"]:
            row["lin_vel_rmse_mps"], row["lin_error_sq_sum"] = 1.17, 1.17 * 1.17 * row["alive_steps_sum"]
        _json(public / "reports/restart_corrected.json", report)
        _rehash(public, "reports/restart_corrected.json")
        expected = "does not replay"
    elif mutation == "forbidden_rehash":
        evidence = json.loads((public / "training_evidence.json").read_text(encoding="utf-8"))
        evidence["arms"]["restart_corrected"]["process_id"] = 1
        _json(public / "training_evidence.json", evidence)
        _rehash(public, "training_evidence.json")
        expected = "private execution field"
    elif mutation == "arm_set_rehash":
        initial = json.loads((public / "initial_states.json").read_text(encoding="utf-8"))
        initial["arms"].pop("restart_corrected")
        _json(public / "initial_states.json", initial)
        _rehash(public, "initial_states.json")
        expected = "arm evidence schema"
    else:
        (public / "notes.json").write_text("{}", encoding="utf-8")
        expected = "file set"
    with pytest.raises(ValueError, match=expected):
        PUB.replay(public)


def test_replay_rejects_publisher_source_change(monkeypatch, tmp_path):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    PUB.export(tag, isaaclab)
    changed = {**PUB._publisher_sources(), "src/rl_triage/recovery.py": "0" * 64}
    monkeypatch.setattr(PUB, "_publisher_sources", lambda: changed)
    with pytest.raises(ValueError, match="source LF-SHA"):
        PUB.replay(results / tag)


@pytest.mark.parametrize("mutation", ["commit", "invalid_status", "failure_record", "private_labels", "wall_scope"])
def test_export_rejects_unbound_or_inconsistent_private_run(monkeypatch, tmp_path, mutation):
    private, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    if mutation == "commit":
        monkeypatch.setattr(H, "verify_protocol", lambda root: ({}, "b" * 40))
        expected = "verified experiment source"
    elif mutation == "invalid_status":
        result = json.loads((private / "result.json").read_text(encoding="utf-8"))
        _json(private / "result.json", {**result, "status": "invalid"})
        expected = "invalid seed result"
    elif mutation == "failure_record":
        _json(private / "failure.json", {"arm": "restart_corrected"})
        expected = "failure record"
    elif mutation == "private_labels":
        result = json.loads((private / "result.json").read_text(encoding="utf-8"))
        result["labels"]["restart_corrected"] = "not_recovered"
        _json(private / "result.json", result)
        expected = "private result differs"
    else:
        row = json.loads((private / "restart_corrected/train/execution.json").read_text(encoding="utf-8"))
        _json(private / "restart_corrected/train/execution.json", {**row, "wall_scope": "gpu_kernel_time"})
        expected = "parent wall scope"
    with pytest.raises(ValueError, match=expected):
        PUB.export(tag, isaaclab)
    assert not (results / tag).exists()


def test_export_never_overwrites_existing_public_folder(monkeypatch, tmp_path):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    (results / tag).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        PUB.export(tag, isaaclab)


def test_aggregate_replays_each_seed_bundle(monkeypatch, tmp_path):
    _, results, isaaclab, tag = _fixture(monkeypatch, tmp_path)
    PUB.export(tag, isaaclab)
    second = tmp_path / "second"
    second.mkdir()
    global SEED
    try:
        SEED = 2028
        _, results_two, isaaclab_two, tag_two = _fixture(monkeypatch, second, restart_lin=1.17)
        PUB.export(tag_two, isaaclab_two)
    finally:
        SEED = 2026
    summary = PUB.aggregate([results / tag, results_two / tag_two])
    assert summary["seeds"] == [2026, 2028] and summary["complete_seed_set"] is True
    assert summary["arms"]["restart_corrected"]["recovered"] == 1
    assert summary["recovery_mode_rule_outcomes"] == {"2026": "hit", "2028": "recommended_not_recovered"}
    assert summary["replay_scope"] == PUB.REPLAY_SCOPE


def test_costs_reject_non_finite_wall_time():
    row = {"scope": PUB.PARENT_WALL_SCOPE, "train_s": math.nan, "fixed_evaluation_s": 1.0, "total_s": 1.0}
    with pytest.raises(ValueError, match="invalid"):
        PUB._costs({"arms": {"clean_resume_reference": {"parent_wall_s": row}}}, ["clean_resume_reference"])


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                                "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def test_frozen_commit_accepts_unchanged_bound_files_after_head_moves(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "bound.txt").write_text("v1", encoding="utf-8")
    (repo / "other.txt").write_text("o1", encoding="utf-8")
    _git(repo, "add", "bound.txt", "other.txt")
    _git(repo, "commit", "-q", "-m", "experiment")
    experiment = _git(repo, "rev-parse", "HEAD")
    (repo / "other.txt").write_text("o2", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "unrelated")
    proxy = PUB._FrozenCommit(H.P, experiment)
    assert proxy.verify_committed(repo, {"bound.txt"}) == experiment
    _git(repo, "checkout", "-q", "-b", "side", experiment)
    (repo / "other.txt").write_text("side", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "side")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-")
    with pytest.raises(ValueError, match="not an ancestor"):
        PUB._FrozenCommit(H.P, side).verify_committed(repo, {"bound.txt"})
    (repo / "bound.txt").write_text("v2", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "bound changed")
    with pytest.raises(ValueError, match="bound source differs"):
        proxy.verify_committed(repo, {"bound.txt"})


def test_frozen_source_restores_module_and_rejects_malformed_commit():
    original = H.P
    with PUB.frozen_source(COMMIT):
        assert isinstance(H.P, PUB._FrozenCommit) and H.P.run_owned is original.run_owned
    assert H.P is original
    with pytest.raises(ValueError, match="malformed"):
        with PUB.frozen_source("HEAD"):
            pass
    assert H.P is original
