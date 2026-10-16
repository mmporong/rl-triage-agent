from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return load(ROOT / "evals/novel_actor_reset_confirmation.py", "novel_actor_reset_confirmation_test")


def test_protocol_freezes_novel_seeds_and_conditional_rule(runner):
    assert runner.FIXED["seeds"] == [2063, 2064]
    assert runner.FIXED["arms"] == ["continuation", "actor_last_layer_reset"]
    assert runner.FIXED["num_envs"] == 4096
    assert runner.FIXED["max_iterations"] == 300
    assert runner.FIXED["decision_rule"]["automatic_global_change"] is False
    assert runner.FIXED["prediction"] == {
        "continuation": ["not_recovered", "not_recovered"],
        "actor_last_layer_reset": ["recovered", "recovered"],
    }


def test_reset_seed_is_registered_and_deterministic(runner):
    assert runner.reset_seed(2063) == 122063
    assert runner.reset_seed(2064) == 122064
    with pytest.raises(ValueError):
        runner.reset_seed(2028)


def _base_intervention(runner, arm: str):
    before = "a" * 64
    optimizer = "b" * 64
    actor_before = {"actor.0": "c" * 64, "actor.2": "d" * 64}
    actor_after = dict(actor_before)
    if arm == "actor_last_layer_reset":
        actor_after["actor.2"] = "e" * 64
    return {
        "schema": "novel_actor_reset_intervention_v1", "arm": arm,
        "intervention": "none" if arm == "continuation" else arm,
        "reset_seed": None if arm == "continuation" else runner.reset_seed(2063),
        "target_modules": [] if arm == "continuation" else ["actor.2"],
        "target_parameter_count": 0 if arm == "continuation" else 2,
        "optimizer_param_indices": [] if arm == "continuation" else [7, 8],
        "optimizer_moment_indices_present_before": [] if arm == "continuation" else [7, 8],
        "optimizer_moments_present_after": None if arm == "continuation" else False,
        "parameter_identity_preserved": True, "policy_state_before": before,
        "policy_state_after": before if arm == "continuation" else "f" * 64,
        "optimizer_state_before": optimizer,
        "optimizer_state_expected_after": optimizer if arm == "continuation" else "1" * 64,
        "optimizer_state_after": optimizer if arm == "continuation" else "1" * 64,
        "optimizer_param_groups_before": "2" * 64, "optimizer_param_groups_after": "2" * 64,
        "critic_state_before": "3" * 64, "critic_state_after": "3" * 64,
        "std_state_before": "4" * 64, "std_state_after": "4" * 64,
        "actor_linear_states_before": actor_before, "actor_linear_states_after": actor_after,
    }


@pytest.mark.parametrize("arm", ["continuation", "actor_last_layer_reset"])
def test_intervention_contract_accepts_only_registered_state_change(runner, arm):
    manifest = {"arm": arm, "seed": 2063}
    row = _base_intervention(runner, arm)
    runner.validate_intervention(row, manifest, {
        "policy_state_sha256": row["policy_state_before"],
        "optimizer_state_sha256": row["optimizer_state_before"],
    })


def test_intervention_rejects_critic_or_earlier_actor_change(runner):
    row = _base_intervention(runner, "actor_last_layer_reset")
    row["critic_state_after"] = "9" * 64
    with pytest.raises(ValueError):
        runner.validate_intervention(row, {"arm": "actor_last_layer_reset", "seed": 2063}, {
            "policy_state_sha256": row["policy_state_before"],
            "optimizer_state_sha256": row["optimizer_state_before"],
        })
    row = _base_intervention(runner, "actor_last_layer_reset")
    row["actor_linear_states_after"]["actor.0"] = "9" * 64
    with pytest.raises(ValueError):
        runner.validate_intervention(row, {"arm": "actor_last_layer_reset", "seed": 2063}, {
            "policy_state_sha256": row["policy_state_before"],
            "optimizer_state_sha256": row["optimizer_state_before"],
        })


def test_manifest_pairs_both_arms_to_same_f1_checkpoint(tmp_path, monkeypatch, runner):
    folder = tmp_path / runner.TAG
    checkpoint = tmp_path / "logs/run/model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source = {"F1": {"checkpoint": str(checkpoint), "checkpoint_sha256": "a" * 64,
                     "environment": {"path": str(tmp_path / "env.yaml"), "sha256": "b" * 64},
                     "agent": {"path": str(tmp_path / "agent.yaml"), "sha256": "c" * 64}}}
    source_path = folder / "source/s2063/source_evidence.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    selection = {"seeds": {"2063": {"source_evidence_sha256": runner.sha(source_path),
                                      "checkpoint_sha256": "a" * 64}}}
    (folder / "source_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL", tmp_path / "protocol.json")
    request = {"source_commit_sha": "d" * 40, "protocol_sha256_lf": "e" * 64,
               "isaaclab_root": str(tmp_path / "IsaacLab")}
    plain = runner.manifest_for(folder, request, 2063, "continuation")
    reset = runner.manifest_for(folder, request, 2063, "actor_last_layer_reset")
    assert plain["resume_checkpoint_path"] == reset["resume_checkpoint_path"] == str(checkpoint.resolve())
    assert plain["resume_checkpoint_sha256"] == reset["resume_checkpoint_sha256"] == "a" * 64
    assert plain["final_iteration"] == reset["final_iteration"] == 598


def test_public_protocol_has_no_personal_absolute_path():
    path = ROOT / "bench/protocols/novel_actor_reset_confirmation_v1.json"
    if not path.is_file():
        pytest.skip("prepare 전")
    text = path.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text
    assert "/home/" not in text and "/Users/" not in text and "/mnt/c/Users/" not in text


def test_verify_cli_forwards_default_registered_tag_when_explicit(tmp_path, monkeypatch, runner, capsys):
    seen = []
    monkeypatch.setattr(runner, "verify", lambda isaaclab, tag=None: seen.append((isaaclab, tag)) or {"status": "ok"})
    assert runner.main(["verify", "--isaaclab", str(tmp_path), "--tag", runner.TAG]) == 0
    assert seen == [(tmp_path.resolve(), runner.TAG)]
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_label_rejects_undetermined_instead_of_not_recovered(monkeypatch, runner):
    monkeypatch.setattr(runner.C, "_behavior", lambda value: value)
    monkeypatch.setattr(runner.RC, "compare", lambda *_args: {})
    monkeypatch.setattr(runner.RC, "verdict", lambda *_args: {"label": "undetermined"})
    with pytest.raises(ValueError, match="완결되지 않았다"):
        runner._label({}, {}, {})


def test_label_treats_healthy_non_walk_as_not_recovered(monkeypatch, runner):
    monkeypatch.setattr(runner.C, "_behavior", lambda value: value)
    monkeypatch.setattr(runner.RC, "compare", lambda *_args: {})
    monkeypatch.setattr(runner.RC, "verdict", lambda *_args: {"label": "healthy"})
    monkeypatch.setattr(runner.V4, "walks", lambda _report: False)
    assert runner._label({}, {}, {})[0] == "not_recovered"


def test_result_document_rejects_each_provenance_mutation(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(runner, "sha", lambda *_args, **_kwargs: "f" * 64)
    protocol = {"prediction": {"continuation": ["not_recovered"] * 2,
                               "actor_last_layer_reset": ["recovered"] * 2},
                "result_wording": {"candidate": "candidate", "keep": "keep"}, "limits": ["limit"]}
    labels = {"continuation": {"2063": "not_recovered", "2064": "not_recovered"},
              "actor_last_layer_reset": {"2063": "recovered", "2064": "recovered"}}
    verdicts = {arm: {seed: {"label": "healthy"} for seed in rows} for arm, rows in labels.items()}
    (tmp_path / "evidence.json").write_text('{}\n', encoding="utf-8")
    expected = runner.result_document(tmp_path, protocol, "a" * 40, {"commit": "a" * 40}, labels, verdicts)
    assert runner.validate_result_document(expected, expected) == expected
    for key in expected:
        changed = dict(expected)
        changed[key] = "tampered"
        with pytest.raises(ValueError):
            runner.validate_result_document(changed, expected)


def test_source_evidence_rejects_launch_and_execution_hash_substitution(tmp_path, monkeypatch, runner):
    isaaclab = tmp_path / "IsaacLab"
    stage = tmp_path / "stage"
    run_name, command = runner._source_command(isaaclab, 2063, "F1")
    run_dir = isaaclab / "logs/rsl_rl/unitree_go2_flat" / ("2026-01-01_00-00-00_" + run_name)
    (run_dir / "params").mkdir(parents=True)
    checkpoint = run_dir / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "params/env.yaml").write_text("seed: 2063\nscene:\n  num_envs: 4096\n", encoding="utf-8")
    (run_dir / "params/agent.yaml").write_text("seed: 2063\nmax_iterations: 300\nnum_steps_per_env: 24\n", encoding="utf-8")
    stage.mkdir(parents=True)
    execution = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                 "error_type": None, "exit_code": 0, "cleanup_verified": True,
                 "owned_process_identities": {}}
    (stage / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    (stage / "launch.json").write_text(json.dumps({"args": command, "cwd": str(isaaclab)}), encoding="utf-8")
    (stage / "gpu_before.json").write_text(json.dumps({"admission_policy": runner.GPU_PROFILE,
                                                        "busy": False, "processes": []}), encoding="utf-8")
    (stage / "stdout.log").write_text('[hidden_fault] {"fault":"F1"}\n', encoding="utf-8")
    row = {"run_name": run_name, "run_dir": str(run_dir), "checkpoint": str(checkpoint),
           "checkpoint_sha256": runner.sha(checkpoint),
           "environment": {"path": str(run_dir / "params/env.yaml"), "sha256": runner.sha(run_dir / "params/env.yaml")},
           "agent": {"path": str(run_dir / "params/agent.yaml"), "sha256": runner.sha(run_dir / "params/agent.yaml")},
           "execution_sha256": runner.sha(stage / "execution.json")}
    assert runner.validate_source_evidence(isaaclab, stage, 2063, "F1", row) == row
    changed = dict(row, execution_sha256="0" * 64)
    with pytest.raises(ValueError, match="receipt/hash"):
        runner.validate_source_evidence(isaaclab, stage, 2063, "F1", changed)
    (stage / "launch.json").write_text(json.dumps({"args": command + ["--seed", "9999"],
                                                   "cwd": str(isaaclab)}), encoding="utf-8")
    with pytest.raises(ValueError, match="launch"):
        runner.validate_source_evidence(isaaclab, stage, 2063, "F1", row)


def test_freshness_contract_rejects_metadata_mutation(runner):
    row = {
        "schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": [2063, 2064],
        "checked_at_utc": "2026-10-16T00:00:00+00:00",
        "checked_roots": ["<repo>/bench/**/*.telemetry.json", "<repo>/bench/private/execution_records",
                          "<repo>/evals/results", "<repo>/evals/loops", "<isaaclab>/logs"],
        "checked_file_count": 42, "evidence": [],
        "scope": "현재 저장소의 실행 기록과 지정 Isaac Lab 로그에 한정한다",
    }
    assert runner.validate_freshness(row) == row
    for key in row:
        changed = dict(row)
        changed[key] = "tampered"
        with pytest.raises(ValueError):
            runner.validate_freshness(changed)


def test_request_and_consumption_are_exactly_bound(tmp_path, monkeypatch, runner):
    folder = tmp_path / runner.TAG
    folder.mkdir()
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    consumed = tmp_path / "consumed.json"
    monkeypatch.setattr(runner, "consumption_path", lambda _tag: consumed)
    anchor = {"commit": "a" * 40, "files": {}}
    monkeypatch.setattr(runner.ANCHOR, "verify_source_anchor", lambda *_args: anchor)
    request = {"tag": runner.TAG, "source_commit_sha": anchor["commit"], "source_anchor": anchor,
               "protocol_sha256_lf": runner.sha(protocol, True), "isaaclab_root": str(isaaclab.resolve())}
    request_path = folder / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    consumed.write_text(json.dumps({"tag": runner.TAG, "request_sha256": runner.sha(request_path)}), encoding="utf-8")
    assert runner.validate_run_request(folder, isaaclab) == (request, anchor)
    consumed.write_text(json.dumps({"tag": runner.TAG, "request_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="consumption"):
        runner.validate_run_request(folder, isaaclab)
    consumed.write_text(json.dumps({"tag": runner.TAG, "request_sha256": runner.sha(request_path)}), encoding="utf-8")
    request["isaaclab_root"] = str(tmp_path / "other")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="request/source"):
        runner.validate_run_request(folder, isaaclab)
