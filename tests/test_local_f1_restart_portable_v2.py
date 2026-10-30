from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench"
sys.path.insert(0, str(BENCH))

BASE = importlib.import_module("local_f1_restart_base_v2")
TERMINAL = importlib.import_module("local_f1_restart_terminal_v2")
WORKER = importlib.import_module("local_f1_restart_worker_v2")
TRAINING = importlib.import_module("residual_training_v1")


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_ancestor_root_discovery() -> None:
    assert BASE.repo_root(Path(__file__)) == BASE.ROOT


def test_terminal_rejects_old_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"policy")
    output = write(tmp_path / "worker.json", {"completed": True, "arm": "CORRECTED", "seed": 3221,
                                               "iterations": 300,
                                               "checkpoint": {"path": str(checkpoint),
                                                              "sha256": BASE.sha(checkpoint)}})
    with pytest.raises(TERMINAL.TerminalEvidenceError):
        TERMINAL.training_evidence(output, {BASE.sha(checkpoint)})


def test_terminal_rejects_either_registered_old_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"policy")
    output = write(tmp_path / "worker.json", {"completed": True, "arm": "CORRECTED", "seed": 3221,
                                               "iterations": 300,
                                               "checkpoint": {"path": str(checkpoint),
                                                              "sha256": BASE.sha(checkpoint)}})
    for old in ({BASE.sha(checkpoint), "other"}, {"other", BASE.sha(checkpoint)}):
        with pytest.raises(TERMINAL.TerminalEvidenceError):
            TERMINAL.training_evidence(output, old)
    known = WORKER.old_checkpoint_shas({"none_checkpoint": {"sha256": "n" * 64}})
    assert known == {"n" * 64, WORKER.OLD_F1_SHA256}


def test_unhealthy_or_zero_none_reference_is_unknown() -> None:
    none = {"metrics": {"lin_vel_rmse_mps": 1.0, "yaw_rate_rmse_radps": 0.3,
                        "mean_survival_s": 20.0, "fall_rate": 0.0},
            "standstill": {"lin_vel_rmse_mps": 1.0}, "horizon_steps": 1000, "step_dt": 0.02}
    corrected = {"metrics": {"lin_vel_rmse_mps": 0.1, "yaw_rate_rmse_radps": 0.1,
                              "mean_survival_s": 20.0, "fall_rate": 0.0}}
    assert BASE.judge(none, corrected)["outcome"] == "unknown"
    none["metrics"]["lin_vel_rmse_mps"] = 0.0
    assert BASE.judge(none, corrected)["reason"] == "nonfinite_or_zero_reference"


def test_timeout_cleans_only_spawned_pid_and_writes_failure(tmp_path: Path) -> None:
    class Process:
        pid = 43210
        calls = 0

        def wait(self, timeout: int) -> int:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return -9

    process = Process()
    cleanup_calls = []

    def cleanup(argv, **kwargs):
        cleanup_calls.append(argv)
        return type("Completed", (), {"returncode": 0})()

    with pytest.raises(TimeoutError):
        WORKER.run(["child"], tmp_path / "stdout.txt", tmp_path / "stderr.txt", 1,
                   tmp_path / "failure.json", popen=lambda *args, **kwargs: process, cleanup=cleanup)
    assert cleanup_calls == [["taskkill", "/PID", "43210", "/T", "/F"]]
    assert json.loads((tmp_path / "failure.json").read_text())["cleanup_confirmed"] is True


def test_generic_worker_consumes_train_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=HERE) as folder:
        root = Path(folder)
        files = root / "inputs"
        none = files / "none.pt"
        none.parent.mkdir()
        none.write_bytes(b"none")
        fixed = BASE.ROOT / "bench/protocols/fixed_eval_v1.json"
        auth = {"none_checkpoint": BASE.binding(none), "fixed_eval": BASE.binding(fixed)}
        protocol = root / "registered.json"
        write(protocol, {"status": "registered", "registration_anchor": "a" * 40,
                         "timeouts_s": {"train": 6000, "evaluate": 900, "total": 7200, "cleanup": 30}})
        monkeypatch.setattr(WORKER, "authorization", lambda _: auth)
        bound = {name: "b" * 64 for name in TRAINING.MINIMUM_BOUND_SOURCES}
        monkeypatch.setattr(WORKER, "source_map", lambda: bound)
        monkeypatch.setattr(WORKER, "PROTOCOL", protocol)
        monkeypatch.setattr(WORKER.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
        train, evaluate = WORKER.build_jobs(root / "auth.json", root / "stage", BASE.ROOT, "a" * 40)
        assert WORKER.V3.validate_job(train)["seed"] == 3221
        manifest = WORKER.V2._residual_manifest(train, mode="train",
                                                manifest_path=Path(train["manifest_path"]),
                                                records_path=Path(train["records_path"]),
                                                readback_path=Path(train["readback_path"]))
        assert manifest["applied_corrections"] == ["restore_body_frame_reward"]
        assert evaluate["artifacts"]["CORRECTED"]["checkpoint"]["sha256"] == "PENDING"
