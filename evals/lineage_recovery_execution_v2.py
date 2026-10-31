"""Recovery v1의 owned stage 생성 누락만 보완하는 얇은 실행 어댑터."""
from __future__ import annotations

import argparse
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_execution_v1 as V1
import run_runtime_readback_smoke as OWNED

SCHEMA = "lineage_recovery_execution_protocol_v2"
SOURCE_FILES = tuple(sorted({
    *V1.SOURCE_FILES,
    "evals/lineage_recovery_execution_v2.py",
    "tests/test_lineage_recovery_execution_v2.py",
}))
_LOCK = RLock()
_LOCAL = threading.local()


@contextmanager
def _v1_scope():
    """V1 알고리즘을 새 schema/source closure로만 실행하고 복원한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            if V1.SCHEMA != SCHEMA or V1.SOURCE_FILES != SOURCE_FILES:
                raise RuntimeError("recovery v2 재진입 중 V1 결합이 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        if V1.SCHEMA != "lineage_recovery_execution_protocol_v1":
            raise RuntimeError("recovery V1 schema가 등록 원본이 아니다")
        original = (V1.SCHEMA, V1.SOURCE_FILES)
        V1.SCHEMA, V1.SOURCE_FILES = SCHEMA, SOURCE_FILES
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            V1.SCHEMA, V1.SOURCE_FILES = original


def _owned_runner(runner):
    """run_owned의 write-once launch 파일보다 먼저 새 owned 폴더를 만든다."""
    def run(folder: Path, args: list[str], cwd: Path, watchdog_s: float = 360) -> dict:
        owned = Path(folder).resolve()
        owned.mkdir(parents=True, exist_ok=False)
        return runner(owned, args, cwd, watchdog_s)

    return run


def prepare_protocol(batch_protocol: Path, fixed_protocol: Path = V1.FIXED_PROTOCOL) -> dict:
    with _v1_scope():
        return V1.prepare_protocol(batch_protocol, fixed_protocol)


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict:
    with _v1_scope():
        return V1.verify_protocol(protocol_path, anchor=anchor, lab=lab)


def prepare_inputs(protocol_path: Path, batch_manifest_path: Path, batch_result_path: Path,
                   target_record_path: Path, out: Path, *, anchor: str, lab: Path,
                   isaac_python: Path) -> dict:
    with _v1_scope():
        return V1.prepare_inputs(
            protocol_path, batch_manifest_path, batch_result_path, target_record_path, out,
            anchor=anchor, lab=lab, isaac_python=isaac_python,
        )


def smoke(manifest_path: Path, out: Path, *, runner=OWNED.run_owned,
          wait=V1.GPU_WAIT.await_clear) -> dict:
    with _v1_scope():
        return V1.smoke(manifest_path, out, runner=_owned_runner(runner), wait=wait)


def execute(manifest_path: Path, approval_path: Path, smoke_root: Path, out: Path, *,
            runner=OWNED.run_owned, wait=V1.GPU_WAIT.await_clear,
            collector=V1.RECOVERY_INPUTS.collect_recovery,
            clock=V1.time.monotonic) -> dict:
    with _v1_scope():
        return V1.execute(
            manifest_path, approval_path, smoke_root, out, runner=_owned_runner(runner),
            wait=wait, collector=collector, clock=clock,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    public = sub.add_parser("prepare-protocol")
    public.add_argument("--batch-protocol", type=Path, required=True)
    public.add_argument("--fixed-protocol", type=Path, default=V1.FIXED_PROTOCOL)
    public.add_argument("--out", type=Path, required=True)
    prep = sub.add_parser("prepare-inputs")
    for name in ("protocol", "batch-manifest", "batch-result", "target-record", "out", "lab",
                 "isaac-python"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--anchor", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--manifest", type=Path, required=True)
    smoke_parser.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--smoke-root", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        V1._write_new(args.out, prepare_protocol(args.batch_protocol, args.fixed_protocol))
    elif args.command == "prepare-inputs":
        prepare_inputs(args.protocol, args.batch_manifest, args.batch_result, args.target_record,
                       args.out, anchor=args.anchor, lab=args.lab, isaac_python=args.isaac_python)
    elif args.command == "smoke":
        smoke(args.manifest, args.out)
    else:
        execute(args.manifest, args.approval, args.smoke_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
