"""T1 학습 실행기(Windows, Isaac Lab 로컬 GPU). docs/T1-EXTERNAL-FAULTS.md 4~5절.

사용: python bench/run_t1.py cal [--faults X1 X2 X3] [--dry-run]   # dev seed 7 보정: t1cal_<결함>_s7(짝 기준 포함)
      python bench/run_t1.py holdout [--seeds 2026 2027 2028] [--dry-run]  # 비밀 대응(bench/private/answer_key_t1.json)

- 4096 env × 300회(P0-C와 같은 예산). 텔레메트리는 bench/runs_t1/, 실행 meta·로그는 bench/private/에 남는다.
- 보정 실행 이름에는 결함 이름이 들어간다(dev seed라 공개). holdout 실행 이름에는 사례 ID만 쓰고 결함 이름은 출력하지 않는다.
- X2의 짝 기준은 이미 있는 순정 실행(보정 p0ccal_NONE_s7, holdout baseline_p0c_s<seed>)을 쓰므로 다시 학습하지 않는다.
- 이미 텔레메트리가 있는 실행은 건너뛰므로 중단 후 다시 실행할 수 있다.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from t1_cases import (  # noqa: E402
    CANDIDATES,
    DEV_SEED,
    HOLDOUT_SEEDS,
    KEY_PATH,
    fault_of_run,
    plan,
)

OUT = ROOT / "bench" / "runs_t1"


def cal_plan(faults: list[str]) -> list[dict]:
    runs = []
    for f in faults:
        if f != "X2":
            runs.append({"case_id": f"t1cal_{f}_REF", "seed": DEV_SEED, "fault": f"{f}_REF"})
        runs.append({"case_id": f"t1cal_{f}", "seed": DEV_SEED, "fault": f})
    return runs


def train(case_id: str, seed: int, fault: str, num_envs: int, iters: int) -> bool:
    cmd = ["pwsh", "-NoProfile", "-File", str(ROOT / "bench" / "run_case.ps1"), "-CaseId", case_id,
           "-Seed", str(seed), "-NumEnvs", str(num_envs), "-MaxIterations", str(iters), "-Fault", fault,
           "-OutDir", str(OUT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode == 0 and (OUT / f"{case_id}_s{seed}.telemetry.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("cal", "holdout"))
    ap.add_argument("--faults", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(HOLDOUT_SEEDS))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.stage == "cal":
        todo = cal_plan(args.faults)
    else:
        key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
        todo = [{**r, "fault": fault_of_run(r["case_id"], key)} for r in plan(key) if r["seed"] in args.seeds]
    failed = 0
    for r in todo:
        name = f"{r['case_id']}_s{r['seed']}"
        if (OUT / f"{name}.telemetry.json").exists():
            print(f"skip {name}", flush=True)
            continue
        if args.dry_run:
            print(f"would train {name}", flush=True)
            continue
        ok = train(r["case_id"], r["seed"], r["fault"], args.num_envs, args.max_iterations)
        failed += not ok
        # holdout 출력에는 결함 이름이 없다. 실패 원인은 bench/private/logs를 본다.
        print(f"{'done' if ok else 'FAIL'} {name}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
