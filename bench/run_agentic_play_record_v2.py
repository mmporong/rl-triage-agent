"""agentic play 녹화 실행기 v2: RUN1을 ``JOB2.canonical_lease_scope()`` 안에서 실행하는 얇은 adapter.

- 부모: ``run_job``이 받은 인자를 그대로 RUN1.run_job에 넘긴다. 운영 Popen·GPU reader·identify·cleanup 판정은
  RUN1이 최초 인자로 하므로 바뀌지 않는다.
- 자식: scope 안에서 RUN1의 자식 launcher가 RUN5 entry를 RUN1로 바꾼 뒤, JOB2가 다시 이 파일로 바꾼다.
  이 파일의 ``main``도 같은 scope에 들어가므로 SDK·AABA 자식이 parent와 같은 canonical lease root로
  lease를 검증한다.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_play_job_v2 as JOB2
import run_agentic_play_record_v1 as RUN1

JOB1 = JOB2.JOB1
PRODUCTION = RUN1.PRODUCTION
FINAL_NAME = RUN1.FINAL_NAME


def run_job(job_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    with JOB2.canonical_lease_scope():
        return RUN1.run_job(job_path, **kwargs)


def child_main(flag: str, job_path: Path) -> int:
    with JOB2.canonical_lease_scope():
        return RUN1.child_main(flag, job_path)


def main(argv: Sequence[str] | None = None) -> int:
    with JOB2.canonical_lease_scope():
        return RUN1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
