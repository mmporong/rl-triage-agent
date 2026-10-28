"""입장 검증은 v3를 유지하고 상태 smoke만 봉인된 SDK 종료 경로로 연결한다."""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v3 as V3

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = V3.SOURCE_FILES | {
    "bench/record_agentic_state_sidecar_v2.py", "tests/test_agentic_state_sidecar_v2.py",
    "bench/run_agentic_capture_unit_v4.py", "tests/test_agentic_capture_unit_v4.py",
    "bench/vision_shutdown_policy_v1.py", "tests/test_vision_shutdown_policy_v1.py",
}
_LOCK = threading.RLock()
_BASE_SMOKE = V1._state_smoke
_BASE_SUBPROCESS = V1.subprocess
OLD_ENTRY = ROOT / "bench/record_agentic_state_sidecar_v1.py"
NEW_ENTRY = ROOT / "bench/record_agentic_state_sidecar_v2.py"


class _SmokeSubprocess:
    def __getattr__(self, name):
        return getattr(_BASE_SUBPROCESS, name)

    def Popen(self, command, **kwargs):
        values = list(command)
        old = str(OLD_ENTRY)
        if values.count(old) != 1:
            raise ValueError("상태 smoke child entry가 정확히 하나가 아니다")
        index = values.index(old)
        if values[index + 1:index + 2] != ["state-dry-run"]:
            raise ValueError("상태 smoke 전용 child 인자가 아니다")
        values[index] = str(NEW_ENTRY)
        return _BASE_SUBPROCESS.Popen(values, **kwargs)


def _state_smoke(validated, stage, reader):
    import record_agentic_state_sidecar_v2 as SIDE2

    if V1.subprocess is not _BASE_SUBPROCESS:
        raise RuntimeError("상태 smoke subprocess scope가 이미 교체됐다")
    try:
        V1.subprocess = _SmokeSubprocess()
        try:
            result = _BASE_SMOKE(validated, stage, reader)
        finally:
            V1.subprocess = _BASE_SUBPROCESS
        shutdown = SIDE2.verify_shutdown_evidence(
            Path(stage) / "state-smoke", checkpoint_sha256=validated["checkpoint_sha256"],
            source_id=validated["job"]["job_id"] + "-state-smoke", frames=1)
        execution_path = V3.J._regular(Path(stage) / "state-smoke-execution.json", "state child execution")
        execution = V3.J._read(execution_path)
        if (execution["exit_code"] != 0 or execution["cleanup_verified"] is not True
                or shutdown["child_pid"] != execution["owned_identity"]["pid"]):
            raise ValueError("종료 증거의 child PID·exit·정리 결합이 다르다")
    except Exception as error:
        V3.J.write_new(Path(stage) / "state-shutdown-rejection.json", {
            "error_type": type(error).__name__, "error_message": str(error),
            "shutdown_markers": SIDE2.shutdown_markers(Path(stage) / "state-smoke"),
        })
        raise
    return {**result, "shutdown_evidence": shutdown}


@contextmanager
def _smoke_scope():
    with _LOCK:
        if V1._state_smoke is not _BASE_SMOKE:
            raise RuntimeError("상태 smoke scope가 이미 교체됐다")
        V1._state_smoke = _state_smoke
        try:
            yield
        finally:
            V1._state_smoke = _BASE_SMOKE


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    protocol = V3._protocol(args.protocol)
    if not SOURCE_FILES <= protocol["source_sha256_lf"].keys():
        raise ValueError("capture unit v4 source closure가 없다")
    with _smoke_scope():
        return V3.execute(args)


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
