"""P0-B2 고정 평가 실행기. 실행 이름으로 Isaac Lab 로그의 체크포인트를 찾아 evals/fixed_eval.py를 호출한다.

사용(Windows, Isaac Sim 4.5 / Isaac Lab 2.1.1 로컬):
  python evals/run_fixed_eval.py --tag p0b2_fixed_eval_20261008 --runs all
  python evals/run_fixed_eval.py --tag <같은 tag> --runs baseline_s42 c01_s7 [--checkpoint model_49.pt]

- --runs all: bench/runs의 텔레메트리 실행(같은 run_dir_name은 하나)을 모두 평가한다.
- 결과: evals/results/<tag>/runs/<실행 이름>[__<체크포인트>].json. 이미 있는 결과는 건너뛰므로 중단 후 다시 실행할 수 있다.
- 작업 목록(체크포인트 절대경로 포함)과 Isaac 로그는 bench/private/fixed_eval/에만 둔다.
- GPU를 쓴다. 모델·네트워크는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

PROTOCOL = ROOT / "bench" / "protocols" / "fixed_eval_v1.json"
PRIVATE = ROOT / "bench" / "private" / "fixed_eval"


def telemetry_runs() -> list[str]:
    seen, names = set(), []
    for f in sorted((ROOT / "bench" / "runs").glob("*.telemetry.json")):
        rd = json.loads(f.read_text(encoding="utf-8"))["run_dir_name"]
        if rd not in seen:
            seen.add(rd)
            names.append(f.name.removesuffix(".telemetry.json"))
    return names


def find_run(log_root: Path, name: str) -> Path:
    dirs = sorted(d for d in log_root.iterdir() if d.is_dir() and re.fullmatch(rf"\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{re.escape(name)}", d.name))
    if len(dirs) != 1:
        raise SystemExit(f"{name}: 실행 폴더가 {len(dirs)}개다")
    return dirs[0]


def last_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"{run_dir.name}: 체크포인트가 없다")
    return ckpts[-1]


def action_scale(run_dir: Path) -> float:
    env = yaml.load((run_dir / "params" / "env.yaml").read_text(encoding="utf-8"), Loader=_TolerantLoader)
    return float(env["actions"]["joint_pos"]["scale"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="학습 실행 체크포인트를 고정 조건에서 평가한다(Isaac Sim, GPU)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="실행 이름들 또는 all")
    ap.add_argument("--checkpoints", nargs="+", help="model_<N>.pt 여러 개(기본: 마지막 체크포인트 하나)")
    ap.add_argument("--suffix", default="", help="결과 파일 이름 뒤에 붙일 표식(같은 체크포인트 반복 평가 등)")
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag) or not re.fullmatch(r"[A-Za-z0-9._-]*", args.suffix):
        raise SystemExit("--tag·--suffix는 영문·숫자·._- 만 쓴다")
    log_root = args.isaaclab / "logs" / "rsl_rl" / "unitree_go2_flat"
    names = telemetry_runs() if args.runs == ["all"] else args.runs
    out_dir = ROOT / "evals" / "results" / args.tag / "runs"
    groups: dict[float, list[dict]] = {}
    for name in names:
        run_dir = find_run(log_root, name)
        ckpts = [run_dir / c for c in args.checkpoints] if args.checkpoints else [last_checkpoint(run_dir)]
        for ckpt in ckpts:
            if not ckpt.is_file():
                raise SystemExit(f"{name}: {ckpt.name}이 없다")
            label = name + (f"__{ckpt.stem}" if args.checkpoints else "") + (f"__{args.suffix}" if args.suffix else "")
            out = out_dir / f"{label}.json"
            if out.exists():
                print(f"skip {label}")
                continue
            groups.setdefault(action_scale(run_dir), []).append({"name": label, "checkpoint": str(ckpt),
                                                                 "output": str(out)})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    python_bat = args.isaaclab / "_isaac_sim" / "python.bat"
    failed = 0
    for scale, jobs in sorted(groups.items()):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        jobs_file = PRIVATE / f"jobs_{args.tag}_{scale}_{stamp}.json"
        jobs_file.write_text(json.dumps({"action_scale": scale, "jobs": jobs}, indent=1), encoding="utf-8")
        log = PRIVATE / f"log_{args.tag}_{scale}_{stamp}.txt"
        cmd = [str(python_bat), str(ROOT / "evals" / "fixed_eval.py"), "--protocol", str(PROTOCOL),
               "--jobs", str(jobs_file), "--headless"]
        print(f"scale={scale} jobs={len(jobs)} log=bench/private/fixed_eval/{log.name}", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=args.isaaclab, stdout=fh, stderr=subprocess.STDOUT)
        done = sum(Path(j["output"]).exists() for j in jobs)
        print(f"  exit={proc.returncode} done={done}/{len(jobs)} wall={time.time() - t0:.0f}s", flush=True)
        failed += len(jobs) - done
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
