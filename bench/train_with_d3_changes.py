"""D3 변경 부분집합(bench/d3_changes.py)을 arm한 뒤 Isaac Lab의 RSL-RL train.py를 그대로 실행한다.

Isaac Lab 폴더에서 Isaac Sim 번들 파이썬으로 실행한다(evals/run_d3_harmless_smoke.py execute가 호출).
  set TRIAGE_D3_CHANGES=H1,H2 && _isaac_sim\\python.bat <repo>\\bench\\train_with_d3_changes.py --task ... --seed ...
변경 id는 환경 변수로만 넘긴다. 설정 변경의 hydra override는 이 프로세스 안에서 train.py 인자 뒤에 붙인다.
bench/train_with_upstream_fault.py와 같은 구조다.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d3_changes  # noqa: E402

overrides = d3_changes.arm(os.environ.get(d3_changes.ENV_VAR, ""))
upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not upstream.is_file():
    raise SystemExit(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
sys.path.insert(0, str(upstream.parent))
sys.argv = [str(upstream), *sys.argv[1:], *overrides]
runpy.run_path(str(upstream), run_name="__main__")
