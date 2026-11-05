# pyright: reportMissingImports=false
"""현재 RSL-RL policy 인터페이스로 고정 하행 과제 적응을 수행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import run_go2_stairs_agent_play_v1 as PLAY
import stairs_agent_repair_v1 as C


def configure_training(cfg, spec: dict):
    import importlib.util
    p = ROOT / "bench/run_stairs_playback_v2.py"
    if PLAY.sha(p, lf=True) != PLAY.LONG.BASE_SHA:
        raise ValueError("기존 계단 설정 소스 SHA 불일치")
    ms = importlib.util.spec_from_file_location("stairs_adaptation_configuration", p)
    if ms is None or ms.loader is None:
        raise ImportError(p)
    original = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(original)
    original.__dict__.update(FRAMES=1500, STEP_HEIGHT_M=0.05, STEP_WIDTH_M=0.3, COMMAND_X_M_S=0.25)
    original.configure_env(cfg, spec)
    # 학습은 접지·시간 종료를 유지하고, 장시간 관찰의 종료 해제를 쓰지 않는다.
    cfg.scene.num_envs = C.RECIPE["num_envs"]
    cfg.episode_length_s = 20
    if hasattr(cfg.curriculum, "terrain_levels"):
        cfg.curriculum.terrain_levels = None
    return cfg


def _model_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(args, app):
    import gymnasium as gym
    import isaaclab
    import isaaclab_tasks  # noqa: F401
    import lineage_lease_context_v2 as LEASE
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    spec = PLAY.read(args.spec)
    auth = PLAY.read(args.authorization)
    if auth["schema"] != "stairs_agent_adaptation_authorization_v1" or auth["recipe"] != C.RECIPE or auth["task"] != C.TASK:
        raise ValueError("등록한 적응 승인 계약 불일치")
    PLAY.load_base()._read_spec(args.spec)
    PLAY.load_base().verify_isaaclab_source(spec, isaaclab)
    LEASE.lease_context()
    out = args.out
    cfg = parse_env_cfg(spec["training_task"], device="cuda:0", num_envs=1024, use_fabric=True)
    configure_training(cfg, spec)
    if cfg.terminations.base_contact is None or cfg.terminations.time_out is None:
        raise ValueError("학습 종료 조건을 유지해야 한다")
    agent = load_cfg_from_registry(spec["training_task"], "rsl_rl_cfg_entry_point")
    agent.device = "cuda:0"
    agent.algorithm.learning_rate = C.RECIPE["learning_rate"]
    agent.max_iterations = C.RECIPE["updates"]
    agent.save_interval = C.RECIPE["updates"] + 1
    env = RslRlVecEnvWrapper(gym.make(spec["training_task"], cfg=cfg), clip_actions=agent.clip_actions)
    counts = {"updates": 0, "optimizer_steps": 0}
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out / "logs"), device=agent.device)
        runner.load(spec["checkpoint"]["path"], load_optimizer=False)
        before = _model_digest(runner.alg.policy)
        step, update = runner.alg.optimizer.step, runner.alg.update
        def measured_step(*a, **kw):
            result = step(*a, **kw)
            counts["optimizer_steps"] += 1
            return result
        def measured_update(*a, **kw):
            if counts["updates"] % 25 == 0:
                LEASE.lease_context()
            result = update(*a, **kw)
            counts["updates"] += 1
            with (out / "updates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**counts, "learning_rate": runner.alg.learning_rate,
                                    "mean_policy_std": float(runner.alg.policy.std.mean().item())})+"\n")
                f.flush()
            return result
        runner.alg.optimizer.step, runner.alg.update = measured_step, measured_update
        runner.learn(num_learning_iterations=C.RECIPE["updates"], init_at_random_ep_len=True)
        LEASE.lease_context()
        final = out / "final.pt"
        runner.save(str(final))
        after = _model_digest(runner.alg.policy)
        if counts != {"updates": 300, "optimizer_steps": 6000} or before == after:
            raise ValueError("실제 학습 갱신 수/가중치 변화 불일치")
        payload = torch.load(str(final), map_location="cpu", weights_only=False)
        if any(not torch.isfinite(v).all() for v in payload["model_state_dict"].values()):
            raise ValueError("최종 정책에 비유한 값이 있다")
        result = {"schema": "stairs_agent_training_receipt_v1", "status": "complete",
                  "pid": os.getpid(), "task": C.TASK, "recipe": C.RECIPE, **counts,
                  "source_checkpoint_sha256": spec["checkpoint"]["sha256"],
                  "checkpoint_sha256": C.sha(final), "policy_before": before, "policy_after": after,
                  "authorization_sha256": C.sha(args.authorization), "seed": spec["seed"],
                  "checkpoint": "final.pt", "policy_recovery_verified": False}
    finally:
        env.close()
    C.write_new(out / "training-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("spec", "authorization", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    auth = PLAY.read(args.authorization)
    if (auth.get("schema") != "stairs_agent_adaptation_authorization_v1"
            or auth.get("recipe") != C.RECIPE or auth.get("task") != C.TASK
            or auth.get("selected_by") != "model"):
        raise ValueError("GPU 시작 전 모델 제안·호스트 승인 계약 검증 실패")
    PLAY.load_base()._read_spec(args.spec)
    args.out.mkdir(parents=True, exist_ok=False)
    app = None
    try:
        from train_stairs_agent_adaptation_v2 import launcher_arguments
        app = AppLauncher(launcher_arguments(args)).app
        C.write_new(args.out/"app-startup.json", {"pid": os.getpid()})
        run(args, app)
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/fastShutdown", True)
        if settings.get_as_bool("/app/fastShutdown") is not True:
            raise ValueError("학습 증거 봉인 후 종료 설정 readback 실패")
        C.write_new(args.out/"app-close-entered.json", {"pid": os.getpid(), "fast_shutdown": True,
                    "wait_for_replicator": False, "training_result_sha256": C.sha(args.out/"training-result.json")})
        app.close(wait_for_replicator=False)
    except BaseException as error:  # noqa: BLE001 - SDK가 삼키는 예외도 실패 증거로 봉인한다.
        C.write_new(args.out/"training-failure.json", {"status": "invalid", "type": type(error).__name__,
                    "traceback": traceback.format_exc()})
        # SDK close가 예외를 삼키는 경로를 피하고 명시적인 실패 종료를 남긴다.
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
