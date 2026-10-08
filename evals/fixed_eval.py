"""P0-B2 고정 평가: 학습 실행의 체크포인트를 한 가지 조건(bench/protocols/fixed_eval_v1.json)에서 평가한다.

Isaac Sim 번들 파이썬에서 실행한다. 보통 evals/run_fixed_eval.py가 호출한다.
  <IsaacLab>\\_isaac_sim\\python.bat evals/fixed_eval.py --protocol bench/protocols/fixed_eval_v1.json --jobs <jobs.json> --headless
jobs.json: {"action_scale": 0.25, "jobs": [{"name": "<실행 이름>", "checkpoint": "<경로>", "output": "<경로>"}]}

- 정책 인터페이스(env.actions.joint_pos.scale)만 실행 설정을 따르고 나머지는 프로토콜 조건으로 고정한다.
  그래서 한 프로세스는 같은 action_scale의 실행만 평가한다.
- 실행마다 같은 eval seed로 전체 환경을 리셋하고, 환경마다 첫 에피소드만 센다.
- 명령은 매 스텝 격자 명령으로 덮어쓰고 관측을 다시 계산한다(명령 관리자가 재샘플해도 정책은 고정 명령만 본다).
- 출력 JSON에는 사용자 경로를 쓰지 않는다(실행 이름, 체크포인트 파일 이름과 SHA256만).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def command_grid(spec: dict) -> list[tuple[float, float, float]]:
    excluded = {tuple(float(v) for v in item) for item in spec.get("exclude", [])}
    return [(float(vx), float(vy), float(yaw)) for vx in spec["vx_mps"] for vy in spec["vy_mps"]
            for yaw in spec["yaw_rate_radps"] if (float(vx), float(vy), float(yaw)) not in excluded]


def standstill_rmse(grid: list[tuple[float, float, float]]) -> dict:
    """제자리에 서 있는 정책의 추종 오차(명령 크기의 RMS). 추종 오차를 해석하는 기준선이다."""
    return {"lin_vel_rmse_mps": math.sqrt(sum(vx * vx + vy * vy for vx, vy, _ in grid) / len(grid)),
            "yaw_rate_rmse_radps": math.sqrt(sum(yaw * yaw for _, _, yaw in grid) / len(grid))}


def _sha256(path: Path, lf: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def _versions() -> dict:
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # 버전 기록 실패는 평가를 막지 않는다
        out["torch_error"] = type(e).__name__
    from importlib import metadata
    for dist in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "isaacsim"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def evaluate(args) -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    proto = json.loads(args.protocol.read_text(encoding="utf-8"))
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    grid = command_grid(proto["command_grid"])
    n = len(grid) * int(proto["envs_per_condition"])
    env_cfg = parse_env_cfg(proto["task"], device=args.device, num_envs=n)
    env_cfg.seed = int(proto["eval_seed"])
    env_cfg.events.add_base_mass = None
    env_cfg.observations.policy.enable_corruption = False
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.actions.joint_pos.scale = float(jobs["action_scale"])
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    if abs(step_dt - proto["step_dt"]) > 1e-12 or env_cfg.episode_length_s != proto["episode_length_s"]:
        raise ValueError(f"평가 환경 step_dt={step_dt}, episode_length_s={env_cfg.episode_length_s}가 프로토콜과 다르다")
    agent_cfg = load_cfg_from_registry(proto["task"], "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    env = RslRlVecEnvWrapper(gym.make(proto["task"], cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    base = env.unwrapped
    device = base.device
    cond = torch.arange(n, device=device) % len(grid)
    fixed = torch.tensor([grid[i] for i in cond.tolist()], dtype=torch.float32, device=device)
    robot = base.scene["robot"]
    terms = base.termination_manager
    common = {"protocol": proto["name"], "protocol_sha256_lf": _sha256(args.protocol, lf=True),
              "task": proto["task"], "eval_seed": proto["eval_seed"], "num_envs": n,
              "horizon_steps": proto["horizon_steps"], "step_dt": step_dt, "action_scale": float(jobs["action_scale"]),
              "standstill": standstill_rmse(grid), "versions": _versions()}
    try:
        for job in jobs["jobs"]:
            t0 = time.time()
            started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ckpt = Path(job["checkpoint"])
            runner.load(str(ckpt), load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            # inference_mode로 만든 환경 버퍼는 다음 실행의 reset이 고칠 수 없어 no_grad를 쓴다.
            with torch.no_grad():
                base.reset(seed=int(proto["eval_seed"]))
                # 리셋이 명령 텐서를 새로 만들 수 있으므로 매 실행 다시 가져온다(v2.1.1은 같은 텐서를 제자리 수정).
                cmd_buf = base.command_manager.get_command("base_velocity")
                active = torch.ones(n, dtype=torch.bool, device=device)
                fell = torch.zeros(n, dtype=torch.bool, device=device)
                alive = torch.zeros(n, device=device)
                lin_sq = torch.zeros(n, device=device)
                yaw_sq = torch.zeros(n, device=device)
                for _ in range(int(proto["horizon_steps"])):
                    cmd_buf.copy_(fixed)
                    obs, _ = env.get_observations()
                    actions = policy(obs)
                    lin_err = torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2] - fixed[:, :2]), dim=1)
                    yaw_err = torch.square(robot.data.root_ang_vel_b[:, 2] - fixed[:, 2])
                    w = active.float()
                    lin_sq += lin_err * w
                    yaw_sq += yaw_err * w
                    alive += w
                    env.step(actions)
                    f = terms.get_term("base_contact").clone()
                    t = terms.get_term("time_out").clone()
                    fell |= f & active
                    active &= ~(f | t)
                if bool(active.any()):
                    raise RuntimeError("horizon이 끝났는데 첫 에피소드가 끝나지 않은 환경이 있다")
                def rmse(sq, mask):
                    # 오차는 종료 확인 전에 누적하므로 환경마다 살아 있는 스텝이 1개 이상이다. 방어로만 둔다.
                    steps = float(alive[mask].sum())
                    return math.sqrt(float(sq[mask].sum()) / steps) if steps > 0 else None

                by_condition = []
                for i, (vx, vy, yaw) in enumerate(grid):
                    m = cond == i
                    by_condition.append({"command": [vx, vy, yaw], "fall_rate": float(fell[m].float().mean()),
                                         "lin_vel_rmse_mps": rmse(lin_sq, m), "yaw_rate_rmse_radps": rmse(yaw_sq, m),
                                         "num_envs": int(m.sum()), "fall_count": int(fell[m].sum()),
                                         "alive_steps_sum": int(alive[m].sum()),
                                         "lin_error_sq_sum": float(lin_sq[m].sum()),
                                         "yaw_error_sq_sum": float(yaw_sq[m].sum())})
                metrics = {"fall_rate": float(fell.float().mean()),
                           "mean_survival_s": float(alive.mean()) * step_dt,
                           "lin_vel_rmse_mps": math.sqrt(float(lin_sq.sum() / alive.sum())),
                           "yaw_rate_rmse_radps": math.sqrt(float(yaw_sq.sum() / alive.sum()))}
            report = {**common, "name": job["name"], "checkpoint": {"file": ckpt.name, "sha256": _sha256(ckpt)},
                      "metrics": metrics, "by_condition": by_condition, "elapsed_s": round(time.time() - t0, 1)}
            report.update({"started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            tmp.replace(out)
            print(json.dumps({"name": job["name"], **metrics, "elapsed_s": report["elapsed_s"]}), flush=True)
    finally:
        env.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--jobs", required=True, type=Path)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app = AppLauncher(args).app
    try:
        evaluate(args)
    except BaseException:
        # Isaac Sim 4.5의 close()는 프로세스를 끝내 예외 출력이 사라지므로 먼저 찍는다.
        # 실패한 실행은 결과 파일이 없어서 run_fixed_eval.py가 센다.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
