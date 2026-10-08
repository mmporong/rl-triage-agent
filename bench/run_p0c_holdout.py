"""P0-C holdout 학습 실행기(Windows, Isaac Lab 로컬 GPU). 비밀 대응을 읽어 실행마다 결함을 넣되, 결함 이름은 출력하지 않는다.

사용: python bench/run_p0c_holdout.py [--seeds 2026 2027 2028] [--dry-run]
- 기준 실행 baseline_p0c_s<seed>(TRIAGE_FAULT=NONE)와 h01~h06_s<seed>를 4096 env × 300회로 학습한다.
- 텔레메트리는 bench/runs_p0c/(결함 정보 없음), 실행 meta·로그는 bench/private/에 남는다.
- 이미 텔레메트리가 있는 실행은 건너뛰므로 중단 후 다시 실행할 수 있다.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from p0c_cases import HOLDOUT_SEEDS, KEY_PATH, plan  # noqa: E402

OUT = ROOT / "bench" / "runs_p0c"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(HOLDOUT_SEEDS))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    todo = [r for r in plan() if r["seed"] in args.seeds]
    failed = 0
    for r in todo:
        name = f"{r['case_id']}_s{r['seed']}"
        if (OUT / f"{name}.telemetry.json").exists():
            print(f"skip {name}", flush=True)
            continue
        fault = "NONE" if r["case_id"] == "baseline_p0c" else key["cases"][r["case_id"]]["fault"]
        cmd = ["pwsh", "-NoProfile", "-File", str(ROOT / "bench" / "run_case.ps1"), "-CaseId", r["case_id"],
               "-Seed", str(r["seed"]), "-NumEnvs", str(args.num_envs), "-MaxIterations", str(args.max_iterations),
               "-Fault", fault, "-OutDir", str(OUT)]
        if args.dry_run:
            print(f"would train {name}", flush=True)
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = proc.returncode == 0 and (OUT / f"{name}.telemetry.json").exists()
        failed += not ok
        # run_case.ps1 출력에는 결함 이름이 없다(경로·종료 코드·시간). 실패 원인은 bench/private/logs를 본다.
        print(f"{'done' if ok else 'FAIL'} {name} exit={proc.returncode}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
