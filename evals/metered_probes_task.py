"""task별 계측 probe 구동기: 동결 evals/probes.py의 TASK만 jobs의 task로 바꾸고 evals/metered_probes.py(CPU·env.step 계측
wrapper)를 그대로 부른다.

probes_task.py와 같은 방식이다. probes.py·metered_probes.py는 바꾸지 않는다. 측정 수식·인자·reset·결함 주입(jobs의 fault)·
계측 sidecar 형식이 같고, 보고서의 task 칸에 jobs의 task가 남는다.

  <IsaacLab>\\_isaac_sim\\python.bat evals/metered_probes_task.py --jobs <jobs.json> --headless --metrics <sidecar.json>
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metered_probes  # noqa: E402
import probes  # noqa: E402
from probes_task import task_from_argv  # noqa: E402

if __name__ == "__main__":
    probes.TASK = task_from_argv(sys.argv)
    raise SystemExit(metered_probes.main(sys.argv[1:]))
