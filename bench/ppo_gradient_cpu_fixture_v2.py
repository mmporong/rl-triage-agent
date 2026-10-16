"""Isaac Python의 실제 RSL learn으로 v4 완료 callback을 CPU에서 검사한다."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_cpu_fixture_v1 as OLD
import ppo_gradient_training_v4 as T
import isaac_training_guard_v1 as GUARD


def exercise(output: Path, arm: str, observed: bool, updates: int = 3, *, fail=False) -> dict:
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.env import VecEnv
    from rsl_rl.runners import OnPolicyRunner
    torch.set_num_threads(1)
    torch.manual_seed(17)

    class Environment(VecEnv):
        num_envs, num_actions, max_episode_length = 16, 2, 64
        device = torch.device("cpu")
        cfg = {}
        episode_length_buf = torch.zeros(16, dtype=torch.long)

        def get_observations(self):
            return torch.randn(16, 6), {"observations": {}}

        def reset(self):
            self.episode_length_buf.zero_()
            return self.get_observations()

        def step(self, actions):
            if fail:
                raise RuntimeError("cpu_fixture_learning_failure")
            obs = torch.randn(16, 6)
            return obs, obs[:, 0] + 0.3 * actions[:, 0], torch.zeros(16, dtype=torch.long), {"observations": {}}

    fixed = OLD.manifest(17, arm) | {"observer_enabled": observed, "max_iterations": updates,
                                    "runtime_updates_to_observe": updates}
    config = {"num_steps_per_env": 8, "save_interval": 50, "empirical_normalization": False,
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [16], "critic_hidden_dims": [16],
                   "activation": "elu", "init_noise_std": 0.7},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
            "clip_param": 0.2, "schedule": "adaptive", "desired_kl": 0.01, "entropy_coef": 0.01}}
    output.mkdir()
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        def sink(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        with T.training_hooks(PPO, OnPolicyRunner, fixed, output, sink) as state:
            with contextlib.redirect_stdout(io.StringIO()):
                runner = OnPolicyRunner(Environment(), config, str(output / "rsl_log"), device="cpu")
                runner.git_status_repos = []
                try:
                    runner.learn(num_learning_iterations=updates, init_at_random_ep_len=True)
                except RuntimeError:
                    if not fail:
                        raise
                finally:
                    if runner.writer is not None:
                        runner.writer.close()
            if fail:
                assert (output / "training_failure.json").is_file()
                assert not (output / "training_evidence.json").exists()
                return {"failure_preserved": True}
            assert (output / "training_evidence.json").is_file()
            T.before_close(output, fixed, state, stream)
    GUARD.validate_completion(output, T.Q.O.manifest_fingerprint(fixed))
    evidence = T.Q.read(output / "training_evidence.json")
    rows = T.Q.OLD.INSPECT.read_jsonl(output / "records.jsonl")
    return {"state": evidence["state"], "losses": evidence["losses"], "updates": evidence["updates"],
        "completion_before_close": evidence["recorded_before_app_close"],
        "inspection": T.G.inspect_records(rows, fixed) if observed else None}


def run() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU 회귀는 CUDA_VISIBLE_DEVICES=-1에서 실행한다")
    with tempfile.TemporaryDirectory(prefix="triage-learning-cpu-") as tmp:
        root = Path(tmp)
        off = exercise(root / "off", "NONE", False)
        on = exercise(root / "on", "NONE", True)
        assert off["state"] == on["state"] and off["losses"] == on["losses"]
        assert on["inspection"]["status"] == "normal"
        dry = exercise(root / "dry", "NONE", True, 1)
        assert dry["updates"] == 1 and dry["inspection"]["status"] == "normal"
        bad = exercise(root / "bad", "INVERSE_RATIO", True)
        assert bad["inspection"]["status"] == "abnormal"
        restored = exercise(root / "restored", "RESTORED", True)
        assert restored["state"] == off["state"] and restored["losses"] == off["losses"]
        failure = exercise(root / "failure", "NONE", True, fail=True)
    return {"status": "passed", "python": sys.version, "gpu_executions": 0,
        "simulation_app_created": False, "actual_rsl_runner": True, "synthetic_cpu_seed": 17,
        "off_on_exact": True, "restored_exact": True, "one_update_completed": True,
        "failure_preserved": failure["failure_preserved"], "normal": on, "inverse": bad, "dry": dry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    GUARD.durable_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "actual_rsl_runner", "off_on_exact", "one_update_completed", "gpu_executions")}))


if __name__ == "__main__":
    main()
