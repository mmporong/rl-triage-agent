"""task별 probe 구동기: 동결 evals/probes.py의 TASK 상수만 jobs의 task로 바꾸고 그대로 부른다.

probes.py는 바꾸지 않는다. 측정 수식·인자·reset·결함 주입(jobs의 fault, hidden_faults)과 출력 형식이 같다. probes.run은
환경을 만들 때 모듈 전역 TASK를 읽으므로 main() 전에 바꾸면 그 task의 등록 학습 설정으로 잰다. 보고서의 task 칸에 그 값이 남는다.

  <IsaacLab>\\_isaac_sim\\python.bat evals/probes_task.py --jobs <jobs.json> --headless
  jobs.json: {"task": "<gym id>", "fault": "NONE", "seed": <seed>, "jobs": [{"name": ..., "checkpoint": ..., "probes": [...], "output": ...}]}
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes  # noqa: E402


def task_from_argv(argv: list[str]) -> str:
    jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
    task = jobs.get("task")
    if not isinstance(task, str) or not task.startswith("Isaac-Velocity-"):
        raise SystemExit(f"jobs의 task가 velocity task id가 아니다: {task!r}")
    return task


if __name__ == "__main__":
    probes.TASK = task_from_argv(sys.argv)
    raise SystemExit(probes.main())
