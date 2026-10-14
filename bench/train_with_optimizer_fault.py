"""옵티마이저 외부 결함(bench/optimizer_faults.py) 하나 또는 NONE을 arm한 뒤 Isaac Lab의 RSL-RL train.py를 그대로 실행한다.

Isaac Lab 폴더에서 Isaac Sim 번들 파이썬으로 실행한다(evals/run_upstream_r1_smoke.py execute가 호출).
  set TRIAGE_OPTIMIZER_FAULT=R1 && _isaac_sim\\python.bat <repo>\\bench\\train_with_optimizer_fault.py --task ... --seed ...
bench/train_with_upstream_fault.py(U1)와 같은 구조이며, 정상 기준 학습도 NONE으로 같은 진입점을 지난다.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizer_faults  # noqa: E402

optimizer_faults.arm(os.environ.get("TRIAGE_OPTIMIZER_FAULT", ""))
upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not upstream.is_file():
    raise SystemExit(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
sys.path.insert(0, str(upstream.parent))
sys.argv = [str(upstream), *sys.argv[1:]]
runpy.run_path(str(upstream), run_name="__main__")
