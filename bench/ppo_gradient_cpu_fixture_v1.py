"""실제 RSL-RL PPO의 작은 CPU 입력으로 계측 비간섭·계산 오류 검사를 확인한다."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import optimizer_observability as O
import ppo_ratio_mutation_v1 as MUT


def manifest(seed: int, arm: str) -> dict:
    return {"schema": O.MANIFEST_SCHEMA, "arm": arm, "seed": seed, "task": "synthetic_cpu_rollout",
        "num_envs": 16, "max_iterations": 3, "configured_learning_epochs": 5,
        "configured_mini_batches": 4, "expected_optimizer_steps_per_update": 20,
        "expected_minibatches_per_update": 20, "runtime_updates_to_observe": 3,
        "source_commit_sha": "0" * 40, "protocol_path": "cpu_development_fixture",
        "protocol_sha256_lf": "0" * 64,
        "source_sha256_lf": {"evals/ppo_gradient_invariant_v1.py": hashlib.sha256(
            (ROOT / "evals/ppo_gradient_invariant_v1.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
        "inspection_primitives": ["optimizer_step_count", "minibatch_count", "ppo_surrogate_gradient"],
        "gradient_invariant": {"schema": "ppo_surrogate_gradient_v1", "atol": 1e-6,
            "rtol": 1e-5, "scale": "batch_size", "clip_param": 0.2}}


def exercise(arm: str, observed: bool, seed: int = 17) -> dict:
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    import ppo_gradient_invariant_v1 as G
    import ppo_gradient_training_v1 as T
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(6, 6, 2, actor_hidden_dims=[16], critic_hidden_dims=[16],
                             activation="elu", init_noise_std=0.7)
    ppo = PPO(policy, num_learning_epochs=5, num_mini_batches=4, clip_param=0.2,
              schedule="adaptive", desired_kl=0.01, entropy_coef=0.01, device="cpu")
    ppo.init_storage("rl", 16, 8, [6], [6], [2])
    rows = []
    fixed = manifest(seed, arm) | {"observer_enabled": observed}
    finish, cleanup = T.attach(ppo, fixed, rows.append)
    try:
        for _ in range(3):
            with torch.inference_mode():
                for _step in range(8):
                    obs = torch.randn(16, 6, device="cpu")
                    actions = ppo.act(obs, obs)
                    assert actions is not None
                    reward = obs[:, 0] + 0.3 * actions[:, 0]
                    ppo.process_env_step(reward, torch.zeros(16, dtype=torch.long), {})
                ppo.compute_returns(torch.randn(16, 6, device="cpu"))
            ppo.update()
        evidence = finish()
    finally:
        cleanup()
    return {"arm": arm, "observed": observed, "readback": evidence["readback"], "state": evidence["state"],
            "losses": evidence["losses"], "wall_s": evidence["update_wall_s"], "records": rows,
            "inspection": G.inspect_records(rows, fixed) if observed else None,
            "restored_update": "update" not in ppo.__dict__,
            "restored_generator": "mini_batch_generator" not in ppo.storage.__dict__,
            "restored_log_prob": "get_actions_log_prob" not in ppo.policy.__dict__}


def run() -> dict:
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU fixture는 CUDA_VISIBLE_DEVICES=-1에서만 실행한다")
    runs = [exercise(arm, observed) for arm, observed in
            (("NONE", False), ("NONE", True), ("INVERSE_RATIO", False),
             ("INVERSE_RATIO", True), ("RESTORED", True))]
    normal_off, normal_on, bad_off, bad_on, restored = runs
    for off, on in ((normal_off, normal_on), (bad_off, bad_on)):
        if off["state"] != on["state"] or off["losses"] != on["losses"]:
            raise ValueError("CPU 계측 OFF/ON의 학습 상태 또는 난수·loss가 다르다")
    if restored["state"] != normal_off["state"] or restored["losses"] != normal_off["losses"]:
        raise ValueError("원래 PPO.update 복원 결과가 정상 대조와 다르다")
    expected = [(normal_on, "normal"), (bad_on, "abnormal"), (restored, "normal")]
    for item, status in expected:
        report = item["inspection"]
        if report["primitives"]["ppo_surrogate_gradient"]["status"] != status:
            raise ValueError("CPU 개발 fixture의 gradient 판정이 다르다")
        if any(report["primitives"][name]["status"] != "normal"
               for name in ("optimizer_step_count", "minibatch_count")):
            raise ValueError("계산 오류 fixture는 갱신 횟수를 바꾸지 않아야 한다")
    if not all(all(item[key] for key in ("restored_update", "restored_generator", "restored_log_prob"))
               for item in runs):
        raise ValueError("CPU fixture 임시 메서드가 남아 있다")
    return {"schema": "ppo_gradient_cpu_development_v1", "status": "passed", "device": "cpu",
        "torch_version": torch.__version__, "synthetic_seed": 17,
        "fixture_only": True, "production_attach_tested": True, "isaac_simulation_executed": False, "gpu_executions": 0,
        "normal_and_mutant_off_on_exact": True, "restored_matches_normal": True, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "device", "normal_and_mutant_off_on_exact",
                                                  "restored_matches_normal", "gpu_executions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
