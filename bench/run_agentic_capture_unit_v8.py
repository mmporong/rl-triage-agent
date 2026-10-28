"""capture unit v8: UNIT7 F1 흐름을 JOB4·RUN6 AWS L40S 계약에 연결한다."""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import agentic_play_job_v4 as JOB4
import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v3 as V3
import run_agentic_capture_unit_v4 as V4
import run_agentic_capture_unit_v7 as V7
import run_agentic_play_record_v6 as RUN6

SOURCE_FILES = V7.SOURCE_FILES | JOB4.SOURCE_FILES | set(RUN6.SOURCE_FILES) | {
    "bench/run_agentic_capture_unit_v8.py", "tests/test_agentic_capture_unit_v8.py",
}
_LOCK = threading.RLock()
_BASE_RUN = V1.RUN


@contextmanager
def _run_scope():
    with _LOCK:
        if V1.RUN is not _BASE_RUN:
            raise RuntimeError("capture unit runner scope가 이미 교체됐다")
        V1.RUN = RUN6
        try:
            yield
        finally:
            V1.RUN = _BASE_RUN


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    with RUN6.aws_f1_record_scope():
        protocol = V3._protocol(args.protocol)
        if not SOURCE_FILES <= protocol["source_sha256_lf"].keys():
            raise ValueError("capture unit v8 source closure가 없다")
        with _run_scope():
            return V4.execute(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["state-smoke", "record"], required=True)
    for name in ("protocol", "provider-inventory", "isaaclab", "python", "stage"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--provider-observed-at", required=True)
    parser.add_argument("--logical-id", required=True)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
