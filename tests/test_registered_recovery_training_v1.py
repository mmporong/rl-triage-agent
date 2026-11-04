import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bench import registered_recovery_training_v1 as training


def test_standalone_child_cli_imports_outside_repo_and_pytest(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(training.__file__).resolve()), "--help"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout and "--binding" in result.stdout


def test_host_source_binding_preserves_contract_lf_normalization(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    targets = binding["correction_source_contract"]["targets"]
    for row in targets.values():
        path = tmp_path / row["source"]
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert training.validate_host_binding(binding, tmp_path) == binding
    changed_path = tmp_path / next(iter(targets.values()))["source"]
    changed_path.write_bytes(b"changed_source\r\n")
    with pytest.raises(ValueError, match="source changed"):
        training.validate_host_binding(binding, tmp_path)


def _ref(path: Path, body: bytes) -> dict:
    path.write_bytes(body)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(body).hexdigest()}


def _binding(tmp_path: Path) -> dict:
    targets = {}
    for correction_id, spec in training.ADAPTER.TARGETS.items():
        path = tmp_path / spec["source"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((correction_id + "\n").encode())
        targets[correction_id] = {**spec, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "clean_fingerprint": hashlib.sha256((correction_id + "-fp").encode()).hexdigest()}
    return {
        "schema": training.SCHEMA, "source_commit_sha": "a" * 40, "task": training.V1.TASK,
        "seed": 2552, "num_envs": 1024, "max_iterations": 300, "resume_iteration": 299,
        "resume_checkpoint": _ref(tmp_path / "model_299.pt", b"checkpoint"),
        "resume_env_config": _ref(tmp_path / "env.yaml", b"env"),
        "resume_agent_config": _ref(tmp_path / "agent.yaml", b"agent"),
        "correction_source_contract": {"schema": "correction_source_contract_v1",
                                       "runtime": {"implementation": "CPython", "version": [3, 10, 15],
                                                   "cache_tag": "cpython-310"}, "targets": targets},
        "action_scale": 0.25, "rsl_rl_version": "2.2.4", "isaaclab_commit": "b" * 40,
        "num_steps_per_env": 24,
    }


def test_cpu_binding_requires_exact_source_contract_and_resume_artifacts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding(tmp_path)
    monkeypatch.setattr(training.ADAPTER, "source_contract",
                        lambda _root: binding["correction_source_contract"])
    assert training.validate_binding(binding, tmp_path)["resume_iteration"] == 299
    Path(binding["resume_checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="resume_checkpoint changed"):
        training.validate_binding(binding, tmp_path)


def test_manifest_is_f2_resume_with_only_registered_correction(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding(tmp_path)
    monkeypatch.setattr(training.ADAPTER, "source_contract",
                        lambda _root: binding["correction_source_contract"])
    training.validate_binding(binding, tmp_path)
    manifest = training.manifest_for(tmp_path / "stage", binding, tmp_path, "request_1")
    assert manifest["fault"] == "F2"
    assert manifest["correction_id"] == "restore_motor_torque_speed"
    assert manifest["resume_checkpoint_sha256"] == binding["resume_checkpoint"]["sha256"]
    assert manifest["max_iterations"] == 300


def test_host_312_preserves_sdk_310_contract_and_builds_fixed_child_argv(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding(tmp_path)
    isaaclab = tmp_path
    launcher = isaaclab / "_isaac_sim/python.bat"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("@echo off\n", encoding="utf-8")
    stage = tmp_path / "stage"
    calls = []

    def owned(_owned_stage, args, cwd, watchdog):
        calls.append((args, cwd, watchdog))
        manifest_path = Path(args[args.index("--manifest") + 1])
        work = manifest_path.parent
        checkpoint = work / "model_598.pt"
        checkpoint.write_bytes(b"new-checkpoint")
        correction = {"schema": "correction_readback_v1", "selected_correction": training.CORRECTION}
        (work / "correction.json").write_text(json.dumps(correction), encoding="utf-8")
        (work / "receipt.json").write_text(json.dumps({
            "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        }), encoding="utf-8")
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                "error_type": None, "exit_code": 0, "cleanup_verified": True, "wall_s": 1.0}

    monkeypatch.setattr(training.LEGACY, "validate_readback", lambda *_args: None)
    result = training.train(stage, isaaclab, "CORRECTED", 2552, "request_1", binding, owned=owned)
    assert result["configuration"] == {"source_checkpoint_consumed": True, "optimizer_state_resumed": True}
    args, cwd, watchdog = calls[0]
    assert args[:3] == [str(launcher), "-B", str(Path(training.__file__).resolve())]
    assert cwd == isaaclab and watchdog == training.V1.WATCHDOG["train"]
    written = json.loads((stage / "binding.json").read_text(encoding="utf-8"))
    assert written["correction_source_contract"]["runtime"]["version"] == [3, 10, 15]
    assert written == binding
