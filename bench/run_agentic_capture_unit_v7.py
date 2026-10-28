"""capture unit v7: UNIT6 흐름(입장 v3·상태 smoke v4)에 F1 재생 계약 scope와 RUN5 runner를 연결한다.

UNIT6은 ``V1.RUN``을 RUN4로 바꾸므로 함께 쓰지 않고 UNIT4를 부른다. 부모 전체(protocol 검증·lease·입장·job
생성·녹화)가 ``RUN5.f1_record_scope()`` 안에서 실행되고, 자식은 RUN5 entry가 같은 scope를 다시 연다.
"""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import agentic_play_job_v3 as JOB3
import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v3 as V3
import run_agentic_capture_unit_v4 as V4
import run_agentic_capture_unit_v6 as V6
import run_agentic_play_record_v5 as RUN5

SOURCE_FILES = V6.SOURCE_FILES | JOB3.SOURCE_FILES | set(RUN5.SOURCE_FILES) | {
    "bench/run_agentic_capture_unit_v7.py", "tests/test_agentic_capture_unit_v7.py",
}
_LOCK = threading.RLock()
_BASE_RUN = V1.RUN


@contextmanager
def _run_scope():
    with _LOCK:
        if V1.RUN is not _BASE_RUN:
            raise RuntimeError("capture unit runner scope가 이미 교체됐다")
        V1.RUN = RUN5
        try:
            yield
        finally:
            V1.RUN = _BASE_RUN


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    with RUN5.f1_record_scope():
        protocol = V3._protocol(args.protocol)
        if not SOURCE_FILES <= protocol["source_sha256_lf"].keys():
            raise ValueError("capture unit v7 source closure가 없다")
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
