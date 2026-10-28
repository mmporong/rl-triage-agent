"""capture unit v5: UNIT4 흐름은 그대로 두고 녹화 runner만 GPU 관찰 원문 참조를 전달하는 RUN3로 연결한다."""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v3 as V3
import run_agentic_capture_unit_v4 as V4
import run_agentic_play_record_v3 as RUN3

SOURCE_FILES = V4.SOURCE_FILES | {
    "bench/run_agentic_play_record_v3.py", "tests/test_run_agentic_play_record_v3.py",
    "bench/run_agentic_capture_unit_v5.py", "tests/test_agentic_capture_unit_v5.py",
}
_LOCK = threading.RLock()
_BASE_RUN = V1.RUN


@contextmanager
def _run_scope():
    with _LOCK:
        if V1.RUN is not _BASE_RUN:
            raise RuntimeError("capture unit runner scope가 이미 교체됐다")
        V1.RUN = RUN3
        try:
            yield
        finally:
            V1.RUN = _BASE_RUN


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    protocol = V3._protocol(args.protocol)
    if not SOURCE_FILES <= protocol["source_sha256_lf"].keys():
        raise ValueError("capture unit v5 source closure가 없다")
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
