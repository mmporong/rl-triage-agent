"""T1 smoke: 외부 원인 결함의 메커니즘이 v2.1.1·Isaac Sim 4.5에서 재현되는지 학습 전에 잰다(docs/T1-EXTERNAL-FAULTS.md 3절).

사용(Windows, Isaac Lab 폴더 밖에서): python bench/smoke_external.py --tag t1_smoke_<date>
  결함마다 Isaac 프로세스 하나(<IsaacLab>\\_isaac_sim\\python.bat bench/smoke_external.py --worker --fault <F> ...)를 띄운다.

- NONE·X1_REF·X1: reset을 반복해 base COM이 기본값에서 상류 범위(±5 cm) 밖으로 벗어나는지(Bug 1),
  멀리 옮긴 로봇이 reset 뒤 스폰 위치로 돌아오는지(Bug 2, set_coms 뒤 텔레포트 손실).
- NONE·X2: 뒤집혀 바닥에 닿은 base를 공중으로 옮긴 뒤 정책 주기(물리 4스텝)로 읽은 base 접촉력이 남는지.
- X3_REF·X3: 1 m/s로 움직이는 로봇에 push 이벤트를 걸면 속도가 더해지는지 표본값으로 덮이는지.
- 결과: evals/results/<tag>/<결함>.json과 summary.json(판정). 학습·정책은 쓰지 않는다. 결과는 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
SMOKES = {"NONE": ("com", "teleport", "contact"), "X1_REF": ("com", "teleport"), "X1": ("com", "teleport"),
          "X2": ("contact",), "X3_REF": ("push",), "X3": ("push",)}
COM_RANGE_XY = 0.05      # velocity EventCfg.base_com 기본 x·y 범위
CONTACT_THRESHOLD = 1.0  # Go2 flat terminations.base_contact 임계값(N)


def _step(base, n: int) -> None:
    for _ in range(n):
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(dt=base.physics_dt)


def _place(robot, origins, z: float, quat_wxyz, dx: float = 0.0) -> None:
    import torch
    state = robot.data.default_root_state.clone()
    state[:, :3] = origins
    state[:, 0] += dx
    state[:, 2] = z
    state[:, 3:7] = torch.tensor(quat_wxyz, device=state.device)
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))


def smoke_com(base, robot, default, resets: int = 20) -> dict:
    """짝수·홀수 env를 번갈아 reset한다(부분 reset으로 env_ids 색인도 거친다). env마다 reset 10회.

    default는 환경 생성 직후·첫 reset 전에 읽은 base COM이다(reset 모드 무작위화가 한 번도 걸리지 않은 값).
    """
    import torch
    bid = robot.find_bodies("base")[0][0]
    n = base.num_envs
    worst = torch.zeros(n)
    for k in range(resets):
        ids = torch.arange(k % 2, n, 2, device=base.device)
        base._reset_idx(ids)
        dev = (robot.root_physx_view.get_coms()[:, bid, :3] - default)[:, :2].abs().max(dim=1).values
        worst = torch.maximum(worst, dev)
    final = (robot.root_physx_view.get_coms()[:, bid, :3] - default)
    out = (worst > COM_RANGE_XY + 1e-4).float()
    return {"resets_per_env": resets // 2, "outside_range_frac": float(out.mean()), "worst_xy_max_m": float(worst.max()),
            "final_xy_rms_m": float(final[:, :2].pow(2).mean().sqrt()), "final_z_rms_m": float(final[:, 2].pow(2).mean().sqrt())}


def smoke_teleport(base, robot) -> dict:
    """로봇을 x로 3 m 옮긴 뒤 모두 reset하고 물리 4스텝 뒤 위치를 본다. reset_base 범위는 x·y ±0.5 m."""
    import torch
    origins = base.scene.env_origins
    _place(robot, origins, 0.4, (1.0, 0.0, 0.0, 0.0), dx=3.0)
    _step(base, 10)
    base._reset_idx(torch.arange(base.num_envs, device=base.device))
    _step(base, 4)
    dx = (robot.data.root_pos_w - origins)[:, 0]
    return {"lost_frac": float((dx > 2.0).float().mean()), "dx_mean_m": float(dx.mean()), "dx_max_m": float(dx.max())}


def smoke_contact(base, robot) -> dict:
    """뒤집어 바닥에 닿게 한 뒤 공중(1.5 m)으로 옮긴다. 마지막 읽기 직후 옮기므로 접촉이 끊긴 스텝에는 읽지 않는다."""
    import torch
    sensor = base.scene["contact_forces"]
    bid = sensor.find_bodies("base")[0][0]
    origins = base.scene.env_origins
    _place(robot, origins, 0.25, (0.0, 1.0, 0.0, 0.0))
    touched = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for _ in range(20):
        _step(base, 4)
        touched |= sensor.data.net_forces_w[:, bid].norm(dim=-1) > CONTACT_THRESHOLD
    _place(robot, origins, 1.5, (1.0, 0.0, 0.0, 0.0))
    air = []
    for _ in range(5):
        _step(base, 4)
        air.append(sensor.data.net_forces_w[:, bid].norm(dim=-1))
    air = torch.stack(air)  # (읽기 5회, env)
    stale = (air > CONTACT_THRESHOLD).all(dim=0) & touched
    z = (robot.data.root_pos_w - origins)[:, 2]
    return {"history_length": sensor.cfg.history_length, "touched_frac": float(touched.float().mean()),
            "stale_frac_of_touched": float(stale.sum() / touched.sum().clamp(min=1)),
            "air_force_mean_n": float(air.mean()), "base_height_after_m": float(z.mean())}


def smoke_push(base, robot) -> dict:
    """서 있는 로봇에 x 1 m/s를 주고 물리 1스텝 뒤 push 이벤트(interval, 모든 env)를 건다."""
    import torch
    _step(base, 20)
    v0 = torch.zeros(base.num_envs, 6, device=base.device)
    v0[:, 0] = 1.0
    robot.write_root_velocity_to_sim(v0)
    _step(base, 1)
    before = robot.data.root_vel_w[:, 0].clone()
    base.event_manager.apply(mode="interval", dt=100.0)
    _step(base, 1)
    after = robot.data.root_vel_w
    return {"vx_before_mean": float(before.mean()), "vx_after_mean": float(after[:, 0].mean()),
            "vx_after_min": float(after[:, 0].min()), "vx_after_max": float(after[:, 0].max()),
            "vz_after_abs_mean": float(after[:, 2].abs().mean())}


def worker(args) -> None:
    import gymnasium as gym
    import hidden_faults
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 7
    hidden_faults._before_env(args.fault)
    env = gym.make(TASK, cfg=env_cfg)
    readback = hidden_faults._after_env(args.fault, env)
    base = env.unwrapped
    robot = base.scene["robot"]
    default_com = robot.root_physx_view.get_coms()[:, robot.find_bodies("base")[0][0], :3].clone()
    t0 = time.time()
    try:
        result = {"fault": args.fault, "task": TASK, "num_envs": args.num_envs, "injected": readback}
        for name in SMOKES[args.fault]:
            env.reset(seed=7)
            result[name] = (smoke_com(base, robot, default_com) if name == "com" else
                            {"teleport": smoke_teleport, "contact": smoke_contact, "push": smoke_push}[name](base, robot))
        result["elapsed_s"] = round(time.time() - t0, 1)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("[smoke] " + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()


def judge(r: dict) -> dict:
    """docs/T1-EXTERNAL-FAULTS.md 3절 판정(결과를 보기 전에 고정)."""
    g = lambda f, k, m: r.get(f, {}).get(k, {}).get(m)  # noqa: E731
    bug1 = (g("X1", "com", "outside_range_frac") or 0) >= 0.5 and g("X1_REF", "com", "outside_range_frac") == 0.0
    bug2 = any((g(f, "teleport", "lost_frac") or 0) > 0.5 for f in ("X1_REF", "X1"))
    teleport_ok = (g("NONE", "teleport", "lost_frac") or 0) <= 0.05
    touched = all((g(f, "contact", "touched_frac") or 0) >= 0.5 for f in ("NONE", "X2"))
    stale = (g("X2", "contact", "stale_frac_of_touched") or 0) >= 0.5 and g("NONE", "contact", "stale_frac_of_touched") == 0.0
    ref_push, x3_push = g("X3_REF", "push", "vx_after_mean"), g("X3", "push", "vx_after_mean")
    overwrite = (ref_push is not None and x3_push is not None and abs(ref_push - 1.0) < 0.15 and abs(x3_push) < 0.15)
    x1 = ("pair" if bug1 and not bug2 else "stock_reference" if bug1 else "exclude") if teleport_ok else "inconclusive"
    return {"X1": {"bug1_accumulates": bug1, "bug2_teleport_lost": bug2, "none_teleport_ok": teleport_ok, "decision": x1},
            "X2": {"base_touched": touched, "stale_force": stale,
                   "decision": ("include" if stale else "exclude") if touched else "inconclusive"},
            "X3": {"overwrite": overwrite, "decision": "include" if overwrite else "exclude"}}


def driver(args) -> int:
    out_dir = ROOT / "evals" / "results" / args.tag
    if out_dir.exists():
        raise SystemExit(f"evals/results/{args.tag}가 이미 있다. 결과는 덮어쓰지 않는다")
    out_dir.mkdir(parents=True)
    log_dir = ROOT / "bench" / "private" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for fault in SMOKES:
        out = out_dir / f"{fault}.json"
        cmd = [str(args.isaaclab / "_isaac_sim" / "python.bat"), str(Path(__file__).resolve()), "--worker",
               "--fault", fault, "--out", str(out), "--num_envs", str(args.num_envs), "--headless"]
        with (log_dir / f"{args.tag}_{fault}.log").open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=args.isaaclab, stdout=fh, stderr=subprocess.STDOUT)
        # Isaac Sim 종료 코드는 믿지 않고 결과 파일로 판정한다(bench/run_case.ps1과 같은 이유).
        results[fault] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        print(f"{fault}: exit={proc.returncode} result={'ok' if results[fault] else 'missing'}", flush=True)
    summary = {"tag": args.tag, "decisions": judge({k: v for k, v in results.items() if v}),
               "missing": [k for k, v in results.items() if v is None]}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["decisions"], ensure_ascii=False), flush=True)
    return 1 if summary["missing"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--fault", choices=sorted(SMOKES))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    if "--worker" not in sys.argv:
        args = ap.parse_args()
        if not args.tag:
            raise SystemExit("--tag가 필요하다")
        return driver(args)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app = AppLauncher(args).app
    try:
        worker(args)
    except BaseException:
        traceback.print_exc()  # Isaac Sim close()가 예외를 삼키므로 먼저 찍는다
        sys.stdout.flush()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
