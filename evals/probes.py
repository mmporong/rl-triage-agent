"""P1-A probe 측정: 실패 실행의 체크포인트를 그 실행과 같은 코드·설정(숨은 결함 포함)으로 짧게 돌려 계측한다.

Isaac Sim 번들 파이썬에서 실행한다(docs/P1-A-LOOP.md 2절, src/rl_triage/probe_loop.py PROBES).
  <IsaacLab>\\_isaac_sim\\python.bat evals/probes.py --jobs <jobs.json> --headless
jobs.json: {"fault": "NONE|F1..F6", "seed": 7, "jobs": [{"name": ..., "checkpoint": <경로>, "probes": [...], "output": <경로>}]}

- 결함은 프로세스 전체에 걸리므로 한 프로세스는 결함 하나의 실행만 잰다(bench/hidden_faults.py와 같은 주입).
- 출력은 원시 측정값이다. abnormal/normal 판정 임계값은 정상(NONE)·v1 dev로 따로 정해 커밋한다.
- 출력 JSON에는 사용자 경로를 쓰지 않는다(실행 이름, 체크포인트 파일 이름·SHA256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
from rl_triage.probe_loop import PROBES  # noqa: E402

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rollout(env, policy, steps, on_step):
    import torch
    obs, _ = env.get_observations()
    for _ in range(steps):
        with torch.no_grad():
            actions = policy.act(obs)
            info = {"actions": actions, "mean": policy.action_mean, "std": policy.action_std,
                    "value": policy.evaluate(obs).squeeze(-1)}
            obs, rew, dones, extras = env.step(actions)
            on_step(info, rew, dones, extras)


def measure(probe: str, env, runner, ckpt: Path) -> dict:
    import torch
    base = env.unwrapped
    policy = runner.alg.policy
    robot = base.scene["robot"]
    args = PROBES[probe]["args"]
    if probe == "P_noise":
        z = []
        _rollout(env, policy, args["rollout_steps"],
                 lambda i, r, d, x: z.append(((i["actions"] - i["mean"]) / i["std"]).flatten()))
        zz = torch.cat(z)
        return {"noise_ratio": float(zz.std()), "logged_std_mean": float(policy.action_std.mean())}
    if probe == "P_value":
        # dev 3차: 롤아웃 수익 기반 지표(설명 분산·상대 오차·상관)가 정상 실행에서도 흔들려(상관 -0.06)
        # critic이 학습되는지를 체크포인트로 본다. 첫 체크포인트 대비 마지막 체크포인트 파라미터 상대 변화.
        first = Path(ckpt).with_name(args["from"])
        sd0 = torch.load(str(first), map_location="cpu", weights_only=False)["model_state_dict"]
        sd1 = torch.load(str(ckpt), map_location="cpu", weights_only=False)["model_state_dict"]

        def change(prefix):
            keys = [k for k in sd0 if k.startswith(prefix)]
            if not keys:
                raise ValueError(f"체크포인트에 {prefix} 파라미터가 없다")
            d = sum(float((sd1[k].float() - sd0[k].float()).norm() ** 2) for k in keys) ** 0.5
            n = sum(float(sd0[k].float().norm() ** 2) for k in keys) ** 0.5
            return d / max(n, 1e-12)

        return {"critic_change": change("critic."), "actor_change": change("actor."), "from": first.name}
    if probe == "P_reward":
        rm = base.reward_manager
        original = rm.compute
        errs = {"track_lin_vel_xy_exp": [], "track_ang_vel_z_exp": []}

        def compute(dt):
            out = original(dt)
            cmd = base.command_manager.get_command("base_velocity")
            for name in errs:
                cfg = rm.get_term_cfg(name)
                std = cfg.params["std"]
                if name == "track_lin_vel_xy_exp":
                    e = torch.sum(torch.square(cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
                else:
                    e = torch.square(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])
                expected = cfg.weight * torch.exp(-e / std**2)
                logged = rm._step_reward[:, rm.active_terms.index(name)]
                errs[name].append(float((logged - expected).abs().mean() / expected.abs().mean().clamp_min(1e-8)))
            return out

        rm.compute = compute
        try:
            _rollout(env, policy, args["rollout_steps"], lambda *a: None)
        finally:
            rm.compute = original
        return {f"{k}_rel_error": sum(v) / len(v) for k, v in errs.items()}
    if probe == "P_torque":
        act = robot.actuators[next(iter(robot.actuators))]
        vl, el = torch.as_tensor(act.velocity_limit).flatten(), torch.as_tensor(act.effort_limit).flatten()
        if len(robot.actuators) != 1 or not (bool((vl == vl[0]).all()) and bool((el == el[0]).all())):
            raise ValueError("관절마다 한계가 같은 액추에이터 그룹 하나를 가정한다(Go2)")
        vlim, elim = float(vl[0]), float(el[0])
        sat, low = [0.0], [0.0]

        def on(i, r, d, x):
            slow = robot.data.joint_vel.abs() < 0.2 * vlim
            clipped = (robot.data.computed_torque - robot.data.applied_torque).abs() > 0.01 * elim
            sat[0] += float((slow & clipped).sum())
            low[0] += float(slow.sum())

        _rollout(env, policy, args["rollout_steps"], on)
        return {"low_speed_saturation": sat[0] / max(low[0], 1.0), "velocity_limit": vlim, "effort_limit": elim}
    if probe == "P_physics":
        # dev 3차: 접지 발 속도·마찰 사용률은 접촉 센서(접선력 없음)·몸체 속도 정의 때문에 정상 실행에서도 의미가 없었다.
        # 시뮬레이터에서 실제 물리 값을 읽어 설정과 무작위화 범위에 맞는지 본다.
        cfg = base.cfg
        mismatches = []
        mats = robot.root_physx_view.get_material_properties()
        sf, df = float(mats[..., 0].mean()), float(mats[..., 1].mean())
        pm = cfg.events.physics_material
        if pm is not None:
            for name, val, rng in (("static_friction", sf, pm.params["static_friction_range"]),
                                   ("dynamic_friction", df, pm.params["dynamic_friction_range"])):
                lo, hi = float(rng[0]), float(rng[1])
                if not (lo * 0.95 <= val <= hi * 1.05):
                    mismatches.append(f"{name} {val:.3f} not in [{lo}, {hi}]")
        delta = (robot.root_physx_view.get_masses() - robot.data.default_mass.to("cpu")).sum(dim=1)
        dmin, dmax = float(delta.min()), float(delta.max())
        am = cfg.events.add_base_mass
        lo, hi = (float(v) for v in am.params["mass_distribution_params"]) if am is not None else (0.0, 0.0)
        if dmin < lo - 0.01 or dmax > hi + 0.01:
            mismatches.append(f"mass delta [{dmin:.3f}, {dmax:.3f}] not in [{lo}, {hi}]")
        if abs(base.physics_dt - cfg.sim.dt) > 1e-9:
            mismatches.append(f"physics_dt {base.physics_dt} != cfg {cfg.sim.dt}")
        return {"mismatch_count": len(mismatches), "mismatches": mismatches, "static_friction_mean": sf,
                "dynamic_friction_mean": df, "mass_delta_min": dmin, "mass_delta_max": dmax,
                "physics_dt": base.physics_dt}
    if probe == "P_episode":
        count = torch.zeros(base.num_envs, device=base.device)
        ends = []

        def on(i, r, d, x):
            count.add_(1.0)
            timeout = base.termination_manager.get_term("time_out")
            if bool(timeout.any()):
                ends.extend(count[timeout].tolist())
            count[d.bool()] = 0.0

        _rollout(env, policy, args["rollout_steps"], on)
        ends.sort()
        median = ends[len(ends) // 2] if ends else None
        return {"timeout_count": len(ends), "median_timeout_step": median,
                "timeout_ratio": None if median is None else median / base.max_episode_length}
    raise ValueError(f"모르는 probe {probe!r}")


def run(args) -> None:
    import gymnasium as gym

    import hidden_faults
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    fault = jobs["fault"]
    num_envs = max(PROBES[p]["args"]["num_envs"] for j in jobs["jobs"] for p in j["probes"])
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=num_envs)
    env_cfg.seed = int(jobs["seed"])
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    hidden_faults._before_env(fault)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    hidden_faults._after_env(fault, env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        for job in jobs["jobs"]:
            ckpt = Path(job["checkpoint"])
            runner.load(str(ckpt), load_optimizer=False)
            results = {}
            for probe in job["probes"]:
                env.unwrapped.reset(seed=int(jobs["seed"]))
                t0 = time.time()
                try:
                    results[probe] = {"measurement": measure(probe, env, runner, ckpt), "exit": 0,
                                      "elapsed_s": round(time.time() - t0, 1)}
                except Exception as e:  # probe 하나의 실패는 불명으로 남기고 다음 probe로 간다
                    results[probe] = {"measurement": None, "exit": 1, "error": f"{type(e).__name__}: {e}"[:300],
                                      "elapsed_s": round(time.time() - t0, 1)}
            out = Path(job["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            report = {"name": job["name"], "checkpoint": {"file": ckpt.name, "sha256": _sha256(ckpt)},
                      "task": TASK, "seed": jobs["seed"], "num_envs": num_envs, "probes": results}
            out.with_suffix(".tmp").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            out.with_suffix(".tmp").replace(out)
            print(json.dumps({"name": job["name"], **{p: r["exit"] for p, r in results.items()}}), flush=True)
    finally:
        env.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True, type=Path)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
