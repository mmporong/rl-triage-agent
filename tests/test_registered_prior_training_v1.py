from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals import registered_prior_training_v1 as P

SOURCE_CP = "527bcbaf2964073969f8fc7597ff48eb028d97ba420c56922776925efba9f123"
DIAGNOSTIC = "c" * 64
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path, dict]:
    repo = tmp_path / "repo"
    isaaclab = tmp_path / "IsaacLab"
    stage = tmp_path / "training"
    recovery = tmp_path / "recovery"
    repo.mkdir()
    isaaclab.mkdir()
    source_cp = tmp_path / "source-model_299.pt"
    source_env = tmp_path / "source-env.yaml"
    source_agent = tmp_path / "source-agent.yaml"
    output_cp = tmp_path / "model_598.pt"
    output_env = tmp_path / "output-env.yaml"
    output_agent = tmp_path / "output-agent.yaml"
    for path, raw in (
        (source_cp, b"source checkpoint"),
        (source_env, b"env: source\n"),
        (source_agent, b"agent: source\n"),
        (output_cp, b"output checkpoint"),
        (output_env, b"env: source\n"),
        (output_agent, b"agent: resumed\n"),
    ):
        path.write_bytes(raw)
    binding = {
        "schema": "registered_actuator_recovery_binding_v1",
        "source_commit_sha": P.SOURCE_ANCHOR,
        "task": TASK,
        "seed": 2664,
        "num_envs": 4096,
        "max_iterations": 300,
        "resume_iteration": 299,
        "resume_checkpoint": {"path": str(source_cp), "sha256": _sha(source_cp)},
        "resume_env_config": {"path": str(source_env), "sha256": _sha(source_env)},
        "resume_agent_config": {"path": str(source_agent), "sha256": _sha(source_agent)},
        "correction_source_contract": {"schema": "correction_source_contract_v1"},
        "action_scale": 0.25,
        "rsl_rl_version": "2.3.3",
        "isaaclab_commit": "d" * 40,
        "num_steps_per_env": 24,
    }
    manifest = {
        "run_name": "request01",
        "correction_id": P.CORRECTION,
        "protocol_sha256_lf": "e" * 64,
        "task": TASK,
    }
    _write(stage / "binding.json", binding)
    _write(stage / "manifest.json", manifest)
    initial = {
        "log_dir": str(tmp_path / "log"),
        "resume_checkpoint": {"sha256": SOURCE_CP, "load_optimizer": True},
        "resume_iteration": 299,
        "source_checkpoint_state": {"iteration": 299},
        "full_environment_state_restored": False,
        "rng_state_restored_from_checkpoint": False,
    }
    _write(stage / "initial.json", initial)
    correction = {
        "changed_targets": [P.CORRECTION],
        "restored_targets": [P.CORRECTION],
        "before_fault": {"restore_body_frame_reward": "r", "restore_configured_timeout": "t"},
        "after_correction": {"restore_body_frame_reward": "r", "restore_configured_timeout": "t"},
    }
    _write(stage / "correction.json", correction)
    receipt = {
        "schema": "training_correction_receipt_v1",
        "completed": True,
        "arm": "corrected",
        "fault": "F2",
        "correction_id": P.CORRECTION,
        "task": TASK,
        "manifest_sha256": _sha(stage / "manifest.json"),
        "initial_sha256": _sha(stage / "initial.json"),
        "correction_sha256": _sha(stage / "correction.json"),
        "checkpoint": {"path": str(output_cp), "sha256": _sha(output_cp)},
        "configs": {
            "environment": {"path": str(output_env), "sha256": _sha(output_env)},
            "agent": {"path": str(output_agent), "sha256": _sha(output_agent)},
        },
        "resume_scope": {
            "full_environment_state_restored": False,
            "policy_optimizer_restored": True,
            "rng_state_restored_from_checkpoint": False,
        },
        "budget": {"max_iterations": 300},
        "resume_iteration": 299,
        "final_iteration": 598,
        "actual_control_steps": 7200,
        "aggregate_env_steps": 29491200,
        "final_checkpoint_state": {"iteration": 598},
    }
    _write(stage / "receipt.json", receipt)
    execution = {
        "cleanup_verified": True,
        "error_type": None,
        "exit_code": 0,
        "job_assigned_before_release": True,
        "natural_exit": True,
        "timed_out": False,
        "wall_s": 12.0,
    }
    _write(stage / "execution.json", execution)
    _write(
        stage / "owned/launch.json",
        {
            "args": [
                str(isaaclab / "_isaac_sim/python.bat"),
                "-B",
                str(repo / "bench/registered_recovery_training_v1.py"),
                "--manifest",
                str(stage / "manifest.json"),
                "--binding",
                str(stage / "binding.json"),
            ],
            "cwd": str(isaaclab),
        },
    )
    for rel in P.REQUIRED_RECOVERY_FILES:
        path = recovery / rel
        if rel.endswith(".jsonl"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        else:
            _write(path, {})
    for rel in (
        "approvals/request01.json",
        "admissions/request01.json",
        "intents/request01.json",
        "resources/request01.json",
        "dispatch_errors/request01.json",
    ):
        _write(recovery / rel, {})

    monkeypatch.setattr(P.TRAIN, "validate_host_binding", lambda value, root: dict(value))
    monkeypatch.setattr(P.TRAIN, "manifest_for", lambda *args: manifest)
    monkeypatch.setattr(P.TRAIN.LEGACY, "validate_readback", lambda *args: None)
    monkeypatch.setattr(P.TRAIN._Host, "validate_training_configuration", lambda *args: None)
    monkeypatch.setattr(P, "_validate_historical_source", lambda *args: {"historical_equals_current": True})
    monkeypatch.setattr(
        P,
        "_validate_recovery",
        lambda *args, **kwargs: {
            "diagnostic_manifest_sha256": DIAGNOSTIC,
            "prior_dispatch_resource_ref": {"path": "resource.json", "sha256": "f" * 64},
        },
    )
    return repo, isaaclab, stage, recovery, binding


def _load(proof: dict, repo: Path, isaaclab: Path, **overrides: object) -> dict:
    kwargs = {
        "expected_source_checkpoint_sha256": SOURCE_CP,
        "expected_seed": 2664,
        "expected_correction_id": P.CORRECTION,
        "expected_task": TASK,
        "expected_diagnostic_manifest_sha256": DIAGNOSTIC,
        "repo": repo,
        "isaaclab": isaaclab,
    }
    kwargs.update(overrides)
    return P.load_completed_actuator_training(proof, **kwargs)  # type: ignore[arg-type]


def test_build_and_load_prior_training_without_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, isaaclab, stage, recovery, _ = _fixture(tmp_path, monkeypatch)
    proof = P.build_prior_training_proof(stage, recovery_folder=recovery, output=tmp_path / "proof.json")
    result = _load(proof, repo, isaaclab)
    assert result["new_training_launched"] is False
    assert result["training_stage"] == str(stage.resolve())
    assert result["training_result"]["arm"] == "CORRECTED"
    assert result["validation"]["fixed_evaluation_executed"] is False
    assert result["validation"]["recovery_label_available"] is False


@pytest.mark.parametrize("relative", ["receipt.json", "initial.json", "owned/launch.json"])
def test_changed_stage_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    repo, isaaclab, stage, recovery, _ = _fixture(tmp_path, monkeypatch)
    proof = P.build_prior_training_proof(stage, recovery_folder=recovery, output=tmp_path / "proof.json")
    with (stage / relative).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(P.PriorTrainingError, match="reference changed"):
        _load(proof, repo, isaaclab)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_source_checkpoint_sha256", "0" * 64),
        ("expected_seed", 1),
        ("expected_task", "different-task"),
        ("expected_correction_id", "different-correction"),
    ],
)
def test_cross_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    repo, isaaclab, stage, recovery, _ = _fixture(tmp_path, monkeypatch)
    proof = P.build_prior_training_proof(stage, recovery_folder=recovery, output=tmp_path / "proof.json")
    with pytest.raises(P.PriorTrainingError, match="expected source/seed/task/correction differs"):
        _load(proof, repo, isaaclab, **{field: value})


def test_changed_proof_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, isaaclab, stage, recovery, _ = _fixture(tmp_path, monkeypatch)
    proof = P.build_prior_training_proof(stage, recovery_folder=recovery, output=tmp_path / "proof.json")
    Path(proof["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(P.PriorTrainingError, match="proof reference changed"):
        _load(proof, repo, isaaclab)


def test_expected_binding_allows_only_source_commit_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, isaaclab, stage, recovery, binding = _fixture(tmp_path, monkeypatch)
    proof = P.build_prior_training_proof(stage, recovery_folder=recovery, output=tmp_path / "proof.json")
    expected = dict(binding)
    expected["source_commit_sha"] = "f" * 40
    result = _load(proof, repo, isaaclab, expected_binding=expected)
    assert result["validation"]["binding"]["provenance_only_source_commit_difference"] is True
    expected["seed"] = 9
    with pytest.raises(P.PriorTrainingError, match="operational training identity"):
        _load(proof, repo, isaaclab, expected_binding=expected)


def test_write_once_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, stage, recovery, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "proof.json"
    P.build_prior_training_proof(stage, recovery_folder=recovery, output=output)
    with pytest.raises(FileExistsError):
        P.build_prior_training_proof(stage, recovery_folder=recovery, output=output)

