"""임대한 Brev VM 안의 capture unit v2: UNIT1을 ``JOB2.canonical_lease_scope()`` 안에서 실행한다.

GPU 입장(UNIT1 admission의 nvidia-smi query) 전에 alias·canonical lease root 관측을
``<stage>-lease-root.json``으로 남긴다. lease·입장·job·smoke·녹화 절차는 UNIT1 그대로다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import agentic_play_job_v2 as JOB2
import run_agentic_capture_unit_v1 as UNIT1

OBSERVATION_SUFFIX = "-lease-root.json"


def observation_path(stage: Path) -> Path:
    stage = Path(stage).absolute()
    return stage.parent / (stage.name + OBSERVATION_SUFFIX)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if UNIT1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    path = observation_path(args.stage)
    with JOB2.canonical_lease_scope() as root:
        observation = JOB2.observe_lease_root()
        if observation["canonical"] != str(root):
            raise JOB2.LeaseError("관측한 canonical root가 scope root와 다르다")
        JOB2.write_new(path, observation)  # GPU 입장 전 기록(기존 파일이 있으면 거부)
        return UNIT1.execute(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["state-smoke", "record"], required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--provider-inventory", required=True, type=Path)
    parser.add_argument("--provider-observed-at", required=True)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--logical-id", required=True)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
