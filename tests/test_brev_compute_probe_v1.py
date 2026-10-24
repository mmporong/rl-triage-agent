"""CPU regressions for the Brev compute worker; no cloud success is simulated."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("brev_compute_probe_v1_test", ROOT / "bench/brev_compute_probe_v1.py")
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def test_metadata_writes_bounded_cpu_report_and_refuses_collision(tmp_path, monkeypatch):
    output = tmp_path / "metadata.json"
    monkeypatch.setattr(B, "_torch_metadata", lambda: {
        "imported": False, "cuda_available": False, "device_count": 0, "devices": [],
    })
    assert B.main(["--mode", "metadata", "--output", str(output), "--timeout-s", "30"]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "brev_compute_metadata_v1"
    assert report["completed"] is True and report["timeout_enforcement"] == "parent_controller"
    with pytest.raises(SystemExit) as error:
        B.main(["--mode", "metadata", "--output", str(output), "--timeout-s", "30"])
    assert error.value.code == 2


@pytest.mark.parametrize("iterations", [19, 21, 299, 301])
def test_train_cli_fixes_environment_and_iteration_budgets(tmp_path, iterations):
    with pytest.raises(SystemExit) as error:
        B.main([
            "--mode", "train", "--output", str(tmp_path / "out.json"),
            "--isaaclab", str(tmp_path), "--seed", "7", "--num-envs", "4096",
            "--iterations", str(iterations), "--run-name", "pilot-s7", "--timeout-s", "60",
        ])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        B.main([
            "--mode", "train", "--output", str(tmp_path / "out.json"),
            "--isaaclab", str(tmp_path), "--seed", "7", "--num-envs", "2048",
            "--iterations", "20", "--run-name", "pilot-s7", "--timeout-s", "60",
        ])
    assert error.value.code == 2


def _fake_training_modules(monkeypatch, tmp_path, *, learn_error=None, early_exit=None):
    events = []

    class FakeRunner:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.current_learning_iteration = 0
            self.num_steps_per_env = 24
            self.env = type("Env", (), {
                "num_envs": 4096,
                "unwrapped": type("BaseEnv", (), {
                    "spec": type("Spec", (), {"id": B.TASK})(),
                })(),
            })()

        def learn(self, num_learning_iterations, init_at_random_ep_len):
            events.append(("learn", num_learning_iterations, init_at_random_ep_len))
            if learn_error is not None:
                raise learn_error
            self.current_learning_iteration += num_learning_iterations - 1
            Path(self.log_dir).mkdir(parents=True)
            (Path(self.log_dir) / f"model_{num_learning_iterations - 1}.pt").write_bytes(b"checkpoint")
            return "learn-result"

    runners = ModuleType("rsl_rl.runners")
    setattr(runners, "OnPolicyRunner", FakeRunner)
    monkeypatch.setitem(sys.modules, "rsl_rl", ModuleType("rsl_rl"))
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    monkeypatch.setattr(B, "_torch_metadata", lambda: {"imported": True, "cuda_available": False,
                                                        "device_count": 0, "devices": []})

    def fake_run(path, run_name):
        events.append(("argv", list(sys.argv), Path.cwd(), Path(path).name, run_name))
        runner = FakeRunner(tmp_path / "logs" / "run")
        try:
            result = runner.learn(num_learning_iterations=20, init_at_random_ep_len=True)
            assert result == "learn-result"
        finally:
            if early_exit is not None:
                raise early_exit

    return fake_run, events


def _lab(tmp_path):
    lab = tmp_path / "IsaacLab"
    upstream = lab / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# test upstream\n", encoding="utf-8")
    return lab


def test_fake_runner_records_fixed_upstream_args_and_learn_evidence_before_early_exit(tmp_path, monkeypatch):
    lab, output = _lab(tmp_path), tmp_path / "train.json"
    fake_run, events = _fake_training_modules(monkeypatch, tmp_path, early_exit=SystemExit(0))
    with pytest.raises(SystemExit) as raised:
        B.run_train(output, lab, 7, 20, "pilot-s7", 60, run_path=fake_run)
    assert raised.value.code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["completed"] is True
    assert report["num_envs"] == report["observed_num_envs"] == 4096
    assert report["task"] == report["observed_task"] == B.TASK
    assert report["observed_iterations"] == 20
    assert report["aggregate_environment_steps"] == 20 * 24 * 4096
    assert report["aggregate_environment_steps_per_s"] > 0
    assert report["learn_arguments"] == {
        "num_learning_iterations": 20, "init_at_random_ep_len": True,
    }
    assert report["checkpoint_path"].endswith("model_19.pt")
    assert report["checkpoint_sha256_raw"] == "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    argv = events[0][1]
    assert argv[1:] == [
        "--task", B.TASK, "--seed", "7", "--num_envs", "4096",
        "--max_iterations", "20", "--headless", "--run_name", "pilot-s7",
    ]
    assert report["wall_scope"] == "worker_start_to_learn_return_before_upstream_app_close"
    assert report["learn_wall_scope"] == "OnPolicyRunner.learn_call_only"


def test_learn_failure_report_survives_false_zero_shutdown_and_has_no_raw_exception(tmp_path, monkeypatch):
    lab, output = _lab(tmp_path), tmp_path / "failed.json"
    secret = RuntimeError("api_key=hidden /home/private/person")
    fake_run, _ = _fake_training_modules(monkeypatch, tmp_path, learn_error=secret, early_exit=SystemExit(0))
    with pytest.raises(SystemExit) as raised:
        B.run_train(output, lab, 3, 20, "pilot-s3", 60, run_path=fake_run)
    assert raised.value.code == 0
    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["completed"] is False
    assert report["error"] == {"type": "RuntimeError", "stage": "learn"}
    assert "hidden" not in report_text and "/home/private" not in report_text
    assert B._read_completion(output) is False


def test_main_converts_false_zero_upstream_exit_to_failure(tmp_path, monkeypatch):
    output = tmp_path / "failed.json"

    def false_zero(path, lab, seed, iterations, run_name, timeout_s):
        B._failure(path, "train", timeout_s, RuntimeError("private"), "learn", 0.0)
        raise SystemExit(0)

    monkeypatch.setattr(B, "run_train", false_zero)
    code = B.main([
        "--mode", "train", "--output", str(output), "--isaaclab", str(tmp_path),
        "--seed", "3", "--num-envs", "4096", "--iterations", "20",
        "--run-name", "pilot-s3", "--timeout-s", "60",
    ])
    assert code == 1


def test_main_accepts_zero_early_exit_only_with_completed_evidence(tmp_path, monkeypatch):
    output = tmp_path / "completed.json"

    def completed_then_zero(path, lab, seed, iterations, run_name, timeout_s):
        report = B._base_report("train", timeout_s)
        report["completed"] = True
        B._write_new(path, report)
        raise SystemExit(0)

    monkeypatch.setattr(B, "run_train", completed_then_zero)
    code = B.main([
        "--mode", "train", "--output", str(output), "--isaaclab", str(tmp_path),
        "--seed", "3", "--num-envs", "4096", "--iterations", "20",
        "--run-name", "pilot-s3", "--timeout-s", "60",
    ])
    assert code == 0


def test_empty_stage_failure_is_written_before_app_close_and_cleanup(tmp_path, monkeypatch):
    output, events = tmp_path / "empty.json", []

    class SimulationCfg:
        def __init__(self, *, dt):
            assert dt == 0.01

    class SimulationContext:
        def __init__(self, config):
            events.append("context")

        def reset(self):
            events.append("reset")

        def step(self):
            raise RuntimeError("private failure")

        def clear_all_callbacks(self):
            events.append("callbacks")

        def clear_instance(self):
            events.append("instance")

    class App:
        def close(self):
            assert output.exists()
            events.append("close")

    app_module, sim_module = ModuleType("isaaclab.app"), ModuleType("isaaclab.sim")
    setattr(app_module, "AppLauncher", lambda **kwargs: type("Launcher", (), {"app": App()})())
    setattr(sim_module, "SimulationCfg", SimulationCfg)
    setattr(sim_module, "SimulationContext", SimulationContext)
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    with pytest.raises(RuntimeError, match="private failure"):
        B.run_empty_stage(output, 30)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["completed"] is False and report["error"]["stage"] == "empty_stage"
    assert events == ["context", "reset", "callbacks", "instance", "close"]
