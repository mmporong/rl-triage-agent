"""P1-A probe 실행기(Windows, Isaac Lab 로컬 GPU). 실행마다 그 실행과 같은 결함 환경으로 evals/probes.py를 부른다.

사용:
  python evals/run_probes.py --tag p1a_probes_<date> --runs p0ccal_NONE_s7 p0ccal_F1_s7 ...   # 보정 실행: 결함은 이름에서
  python evals/run_probes.py --tag p1a_probes_<date> --runs baseline_p0c_s2026 h01_s2026 ...  # holdout: 비공개 대응에서
  python evals/run_probes.py --tag t1_probes_<date> --runs t1cal_X1_s7 ref_e01_s2026 e01_s2026 ...  # T1(외부 결함)

- 결함 주입은 프로세스 전체에 걸리므로 (결함, seed)마다 Isaac 프로세스 하나를 띄운다.
- 결과: evals/results/<tag>/probes/<실행 이름>.json(결함 정보 없음). 작업 목록·로그는 bench/private/probes/에만.
- holdout 실행의 결함 이름은 출력하지 않는다. 이미 있는 결과는 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
from run_fixed_eval import find_run, last_checkpoint  # noqa: E402
from rl_triage.probe_loop import PROBES  # noqa: E402
from t1_cases import KEY_PATH as T1_KEY_PATH  # noqa: E402
from t1_cases import fault_of_run  # noqa: E402

PRIVATE = ROOT / "bench" / "private" / "probes"
KEY_PATH = ROOT / "bench" / "private" / "answer_key_p0c.json"


def fault_of(name: str, key: dict | None, key_t1: dict | None = None) -> str:
    m = re.fullmatch(r"p0ccal_(NONE|F\d)b?_s\d+", name)
    if m:
        return m[1]
    m = re.fullmatch(r"t1cal_(X\d(?:_REF)?)_s\d+", name)
    if m:
        return m[1]
    m = re.fullmatch(r"((?:ref_)?e\d\d)_s\d+", name)
    if m:
        if key_t1 is None:
            raise SystemExit("T1 holdout 실행에는 bench/private/answer_key_t1.json이 필요하다")
        return fault_of_run(m[1], key_t1)
    if re.fullmatch(r"baseline_p0c_s\d+", name):
        return "NONE"
    m = re.fullmatch(r"(h\d\d)_s\d+", name)
    if m:
        if key is None:
            raise SystemExit("holdout 실행에는 bench/private/answer_key_p0c.json이 필요하다")
        return key["cases"][m[1]]["fault"]
    raise SystemExit(f"{name}: 결함 환경을 정할 수 없는 실행 이름")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실행마다 같은 결함 환경으로 probe를 잰다(Isaac Sim, GPU)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--probes", nargs="+", default=sorted(PROBES), choices=sorted(PROBES))
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise SystemExit("--tag는 영문·숫자·._- 만 쓴다")
    key = json.loads(KEY_PATH.read_text(encoding="utf-8")) if KEY_PATH.exists() else None
    key_t1 = json.loads(T1_KEY_PATH.read_text(encoding="utf-8")) if T1_KEY_PATH.exists() else None
    log_root = args.isaaclab / "logs" / "rsl_rl" / "unitree_go2_flat"
    out_dir = ROOT / "evals" / "results" / args.tag / "probes"
    groups: dict[tuple[str, int], list[dict]] = {}
    for name in args.runs:
        out = out_dir / f"{name}.json"
        if out.exists():
            print(f"skip {name}", flush=True)
            continue
        run_dir = find_run(log_root, name)
        seed = int(name.rsplit("_s", 1)[1])
        groups.setdefault((fault_of(name, key, key_t1), seed), []).append(
            {"name": name, "checkpoint": str(last_checkpoint(run_dir)), "probes": args.probes, "output": str(out)})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    failed = 0
    for i, ((fault, seed), jobs) in enumerate(sorted(groups.items())):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        jobs_file = PRIVATE / f"jobs_{args.tag}_{i}_{stamp}.json"
        jobs_file.write_text(json.dumps({"fault": fault, "seed": seed, "jobs": jobs}, indent=1), encoding="utf-8")
        log = PRIVATE / f"log_{args.tag}_{i}_{stamp}.txt"
        cmd = [str(args.isaaclab / "_isaac_sim" / "python.bat"), str(ROOT / "evals" / "probes.py"),
               "--jobs", str(jobs_file), "--headless"]
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=args.isaaclab, stdout=fh, stderr=subprocess.STDOUT)
        done = sum(Path(j["output"]).exists() for j in jobs)
        failed += len(jobs) - done
        # 그룹 번호와 실행 이름만 출력한다(결함 이름은 비공개 작업 목록에만 있다).
        print(f"group {i}: runs={[j['name'] for j in jobs]} exit={proc.returncode} done={done}/{len(jobs)} "
              f"wall={time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
