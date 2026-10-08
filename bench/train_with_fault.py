"""P0-C: 숨은 결함 하나(TRIAGE_FAULT)를 arm한 뒤 Isaac Lab의 RSL-RL train.py를 그대로 실행한다.

Isaac Lab 폴더에서 Isaac Sim 번들 파이썬으로 실행한다(bench/run_case.ps1 -Fault가 호출).
  set TRIAGE_FAULT=F1 && _isaac_sim\\python.bat <repo>\\bench\\train_with_fault.py --task ... --seed ...
기준 실행도 TRIAGE_FAULT=NONE으로 같은 경로를 거쳐 학습한다.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hidden_faults  # noqa: E402

hidden_faults.arm(os.environ.get("TRIAGE_FAULT", ""))
upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not upstream.is_file():
    raise SystemExit(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
sys.path.insert(0, str(upstream.parent))
sys.argv = [str(upstream), *sys.argv[1:]]
runpy.run_path(str(upstream), run_name="__main__")
