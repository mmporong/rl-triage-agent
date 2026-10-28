"""capture unit v6: UNIT4 흐름은 그대로 두고 녹화 runner를 렌더 sync를 포함한 RUN4로 연결한다.

RUN4는 RUN3의 GPU 관찰 원문 전달 scope를 포함한다. UNIT5는 ``V1.RUN``을 RUN3로 바꾸므로 함께 쓰지 않고
UNIT4를 부른다. source closure는 UNIT5 closure에 RUN4·UNIT6과 RUN4가 쓰는 readiness helper를 더한다.
"""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v3 as V3
import run_agentic_capture_unit_v4 as V4
import run_agentic_capture_unit_v5 as V5
import run_agentic_play_record_v4 as RUN4

SOURCE_FILES = V5.SOURCE_FILES | {
    "bench/run_agentic_play_record_v4.py", "tests/test_run_agentic_play_record_v4.py",
    "bench/run_agentic_capture_unit_v6.py", "tests/test_agentic_capture_unit_v6.py",
    "bench/vision_capture_readiness_v1.py", "tests/test_vision_capture_readiness_v1.py",
}
_LOCK = threading.RLock()
_BASE_RUN = V1.RUN


@contextmanager
def _run_scope():
    with _LOCK:
        if V1.RUN is not _BASE_RUN:
            raise RuntimeError("capture unit runner scope가 이미 교체됐다")
        V1.RUN = RUN4
        try:
            yield
        finally:
            V1.RUN = _BASE_RUN


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    protocol = V3._protocol(args.protocol)
    if not SOURCE_FILES <= protocol["source_sha256_lf"].keys():
        raise ValueError("capture unit v6 source closure가 없다")
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
