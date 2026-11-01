# pyright: reportMissingImports=false
"""lineage source batch v2를 준비·검증하고 공용 lease 안에서 실행한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import isaac_training_guard_v1 as GUARD
import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v2 as BATCH
import run_runtime_readback_smoke as OWNED
import runtime_observation_training_v1 as FIXED


def _seeds(value: str, count: int) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"seed는 쉼표로 구분한 {count}개여야 한다")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "fresh-seeds", "execute"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--case-seeds", default="2550,2551,2552,2553,2554,2555,2556,2557,2558,2559")
    parser.add_argument("--normal-reference-seeds", default="3550,3551")
    parser.add_argument("--smoke-seed", type=int, default=4550)
    args = parser.parse_args(argv)
    lab = args.lab.resolve()
    if args.command == "prepare":
        if args.output is None:
            parser.error("prepare에는 --output이 필요하다")
        value = BATCH.protocol_value(
            lab, case_seeds=_seeds(args.case_seeds, 10),
            normal_reference_seeds=_seeds(args.normal_reference_seeds, 2),
            smoke_seed=args.smoke_seed)
        GUARD.durable_new(args.output, value)
        return 0
    if args.protocol is None or not args.anchor:
        parser.error("verify/fresh-seeds/execute에는 --protocol과 --anchor가 필요하다")
    value = BATCH.verify(args.protocol, lab, args.anchor)
    if args.command == "verify":
        print(json.dumps({"status": "valid", "source_jobs": 20, "fixed26": 20, "api": 0}))
        return 0
    include_smoke = args.smoke_result is None
    freshness = BATCH.fresh(value, lab, include_smoke=include_smoke)
    if args.command == "fresh-seeds":
        if args.output:
            GUARD.durable_new(args.output, freshness)
        print(json.dumps(freshness, ensure_ascii=False))
        return 0
    if args.output is None or args.isaac_python is None or not args.isaac_python.is_file():
        parser.error("execute에는 새 --output과 실제 --isaac-python이 필요하다")
    smoke_reference = None
    if args.smoke_result is not None:
        if args.mode != "all":
            parser.error("--smoke-result는 all 모드에서만 쓴다")
        smoke_reference = BATCH.verify_smoke(args.smoke_result.resolve(), args.protocol, value,
                                             args.anchor)
    result = BATCH.run_sequence(
        args.output.resolve(), args.protocol.resolve(), value, args.anchor, lab,
        args.isaac_python.resolve(), mode=args.mode, runner=OWNED.run_owned,
        fixed_runner=FIXED.run_fixed, check_gpu=lambda stage: GPU.snapshot(stage),
        smoke_reference=smoke_reference, wait=FIXED.EXEC.GPU_WAIT.await_clear)
    print(json.dumps({"status": result["status"], "source_jobs": result["source_jobs_completed"],
                      "fixed26": result["fixed26_completed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
