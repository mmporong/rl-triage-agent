"""Fixed-eval v1에 bounded GPU admission wait를 연결하는 얇은 v2 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import lineage_fixed_eval_batch_v1 as V1
import lineage_gpu_wait_v1 as WAIT

SCHEMA = "lineage_fixed_eval_protocol_v2"
RESULT_SCHEMA = "lineage_fixed_eval_result_v2"
WHOLE_CONTROLLER_WALL_S = 3600
ADMISSION_WAIT_MAX_S = 600
SAFETY_MARGIN_S = 5
BASE_SOURCE_FILES = tuple(V1.SOURCE_FILES)
SOURCE_FILES = tuple(sorted(set(BASE_SOURCE_FILES) | {
    "evals/lineage_fixed_eval_batch_v2.py",
    "tests/test_lineage_fixed_eval_batch_v2.py",
    "bench/run_lineage_fixed_eval_v2.ps1",
    "evals/lineage_gpu_wait_v1.py",
    "tests/test_lineage_gpu_wait_v1.py",
}))
_V1_PREPARE_PROTOCOL = V1.prepare_protocol
_PATCH_LOCK = threading.RLock()


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _v1_surface(*, route_protocol: bool) -> Iterator[None]:
    """v1의 동적 protocol surface만 재진입 가능하게 교체하고 반드시 복원한다."""
    with _PATCH_LOCK:
        old = (V1.SCHEMA, V1.SOURCE_FILES, V1.prepare_protocol)
        V1.SCHEMA, V1.SOURCE_FILES = SCHEMA, SOURCE_FILES
        if route_protocol:
            V1.prepare_protocol = prepare_protocol
        try:
            yield
        finally:
            V1.SCHEMA, V1.SOURCE_FILES, V1.prepare_protocol = old


def prepare_protocol(lab: Path) -> dict:
    """v1 고정 평가 계약에 wait source closure와 포함 예산만 추가한다."""
    with _v1_surface(route_protocol=False):
        value = _V1_PREPARE_PROTOCOL(Path(lab).resolve())
    value = deepcopy(value)
    value["gpu_wait"] = {**WAIT.policy(), "included_in_whole_controller_wall_s": True,
                         "sdk_jobs_started_while_waiting": 0}
    value["budget"].update(
        admission_wait_max_s=ADMISSION_WAIT_MAX_S,
        owned_timeout_rule="min(v1_3600, remaining_whole_controller_s - safety_margin_s)",
        owned_timeout_safety_margin_s=SAFETY_MARGIN_S,
        gpu_budget_increase=0,
    )
    value["adapter"] = {
        "base": "lineage_fixed_eval_batch_v1",
        "worker_job_manifest_report_gate_schemas": "reused_without_change",
        "prepare_before_lease_execute_inside_lease": True,
        "global_patch": "RLock_reentrant_finally_restored",
    }
    if value["budget"]["whole_controller_wall_s"] != WHOLE_CONTROLLER_WALL_S:
        raise ValueError("v2 whole controller 상한은 기존 3600초여야 한다")
    if set(value["source_sha256_lf"]) != set(SOURCE_FILES):
        raise ValueError("v2 source closure가 확장 목록과 다르다")
    return value


def verify(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    """v2 protocol/current/blob/ancestor를 v1 검증기로 확인한다."""
    with _v1_surface(route_protocol=True):
        return V1.verify(Path(protocol_path), lab=Path(lab), anchor=anchor)


def prepare_inputs(source_root: Path, protocol_path: Path, anchor: str, out: Path,
                   lab: Path, isaac_python: Path) -> dict:
    """v1 worker job과 private manifest schema를 그대로 준비한다."""
    with _v1_surface(route_protocol=True):
        return V1.prepare_inputs(source_root, protocol_path, anchor, out, lab, isaac_python)


def execute(manifest_path: Path, *, runner: Callable,
            check_lease: Callable = V1.SOURCE.lease_context,
            waiter: Callable = WAIT.await_clear,
            clock: Callable = time.monotonic, sleep: Callable = time.sleep,
            safety_margin_s: float = SAFETY_MARGIN_S) -> dict:
    """wait를 포함한 3600초 안에서 v1 owned worker·gate 흐름을 단회 실행한다."""
    if (isinstance(safety_margin_s, bool) or not isinstance(safety_margin_s, (int, float))
            or not 0 < safety_margin_s < WHOLE_CONTROLLER_WALL_S - ADMISSION_WAIT_MAX_S):
        raise ValueError("owned timeout 안전 여유가 등록 범위를 벗어났다")
    manifest_path = Path(manifest_path).resolve()
    manifest = V1._read(manifest_path)
    root = Path(manifest.get("output_root", "")).resolve()
    v2_result, v2_failure = root / "v2-result.json", root / "v2-failure.json"
    if v2_result.exists() or v2_failure.exists():
        raise FileExistsError("v2 fixed-eval 결과 또는 실패 기록을 재사용하지 않는다")
    started = clock()
    wait_receipt: dict | None = None
    owned_timeout: float | None = None

    def remaining() -> float:
        value = WHOLE_CONTROLLER_WALL_S - (clock() - started)
        if value <= safety_margin_s:
            raise TimeoutError("GPU wait를 포함한 fixed-eval 전체 3600초 상한이 소진됐다")
        return value

    def admission(stage: Path) -> dict:
        nonlocal wait_receipt
        maximum = min(ADMISSION_WAIT_MAX_S, remaining() - safety_margin_s)
        if maximum <= 0:
            raise TimeoutError("GPU admission wait를 시작할 전체 controller 여유가 없다")
        wait_receipt = dict(waiter(stage, max_wait_s=maximum, interval_s=2,
                                   consecutive_clear=2, lease_check=check_lease,
                                   clock=clock, sleep=sleep))
        if wait_receipt.get("status") != "clear":
            raise WAIT.GpuWaitError("GPU wait clear receipt가 없다")
        receipt_path = Path(wait_receipt.get("receipt_path", ""))
        if (not receipt_path.is_file() or _sha(receipt_path) != wait_receipt.get("receipt_sha256")):
            raise WAIT.GpuWaitError("GPU wait receipt 경로 또는 SHA가 다르다")
        return wait_receipt

    def bounded_runner(stage: Path, argv: list[str], cwd: Path, timeout: float) -> dict:
        nonlocal owned_timeout
        owned_timeout = min(timeout, remaining() - safety_margin_s)
        if owned_timeout <= 0:
            raise TimeoutError("owned fixed-eval child를 시작할 전체 controller 여유가 없다")
        return dict(runner(stage, argv, cwd, owned_timeout))

    try:
        with _v1_surface(route_protocol=True):
            base = V1.execute(manifest_path, runner=bounded_runner,
                              check_lease=check_lease, check_gpu=admission)
        elapsed = clock() - started
        if not 0 <= elapsed <= WHOLE_CONTROLLER_WALL_S:
            raise TimeoutError("fixed-eval v2 전체 controller 3600초 상한을 넘었다")
        if wait_receipt is None or owned_timeout is None:
            raise RuntimeError("GPU wait 또는 bounded owned 실행 기록이 없다")
        result = {
            "schema": RESULT_SCHEMA, "status": "complete",
            "base_result_path": str(root / "result.json"),
            "base_result_sha256": _sha(root / "result.json"),
            "base_result": deepcopy(base), "gpu_wait": deepcopy(wait_receipt),
            "owned_timeout_s": owned_timeout, "whole_controller_wall_s": elapsed,
            "whole_controller_limit_s": WHOLE_CONTROLLER_WALL_S,
            "gpu_budget_increase": 0, "model_api_requests": 0,
        }
        _write_new(v2_result, result)
        return deepcopy(result)
    except BaseException as error:
        if root.is_dir() and not v2_failure.exists():
            _write_new(v2_failure, {
                "status": "invalid", "error_type": type(error).__name__, "error": str(error),
                "gpu_wait": deepcopy(wait_receipt), "owned_timeout_s": owned_timeout,
                "sdk_retry": False, "automatic_retry": False,
            })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "prepare-inputs", "execute"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.out is None:
            parser.error("prepare에는 --out이 필요하다")
        _write_new(args.out, prepare_protocol(args.lab))
        return 0
    if args.anchor is None or args.protocol is None:
        parser.error("verify/prepare-inputs/execute에는 --anchor와 --protocol이 필요하다")
    verify(args.protocol, lab=args.lab, anchor=args.anchor)
    if args.command == "verify":
        return 0
    if args.command == "prepare-inputs":
        if args.source_root is None or args.out is None or args.isaac_python is None:
            parser.error("prepare-inputs에는 --source-root --out --isaac-python이 필요하다")
        prepare_inputs(args.source_root, args.protocol, args.anchor, args.out, args.lab, args.isaac_python)
        return 0
    if args.manifest is None:
        parser.error("execute에는 --manifest가 필요하다")
    import run_runtime_readback_smoke as owned
    execute(args.manifest, runner=owned.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
