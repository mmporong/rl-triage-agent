"""외부 결함(bench/upstream_faults.py) 하나를 arm한 뒤 Isaac Lab의 RSL-RL train.py를 그대로 실행한다.

Isaac Lab 폴더에서 Isaac Sim 번들 파이썬으로 실행한다(evals/run_upstream_smoke.py execute가 호출).
  set TRIAGE_UPSTREAM_FAULT=U1 && _isaac_sim\\python.bat <repo>\\bench\\train_with_upstream_fault.py --task ... --seed ...
P0-C의 bench/train_with_fault.py와 같은 구조이며, 결함 표만 다르다.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upstream_faults  # noqa: E402

upstream_faults.arm(os.environ.get("TRIAGE_UPSTREAM_FAULT", ""))
upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not upstream.is_file():
    raise SystemExit(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
sys.path.insert(0, str(upstream.parent))
sys.argv = [str(upstream), *sys.argv[1:]]
runpy.run_path(str(upstream), run_name="__main__")
