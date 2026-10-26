"""v2 recovery worker에 학습 manifest 사전검사와 원오류 증거를 추가한다."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import brev_recovery_worker_v2 as V2

FAILURE_SCHEMA = "brev_recovery_worker_failure_v1"


def validate_job(value: Mapping[str, Any]) -> dict[str, Any]:
    """기존 job 계약과 300-update residual manifest 계약을 함께 검증한다."""
    job = V2.validate_job(value)
    if job["mode"] == "train" and job["iterations"] == 300:
        V2._residual_manifest(
            job,
            mode="train",
            manifest_path=Path(job["manifest_path"]),
            records_path=Path(job["records_path"]),
            readback_path=Path(job["readback_path"]),
        )
    return job


def _failure(error: BaseException) -> dict[str, Any]:
    return {
        "schema": FAILURE_SCHEMA,
        "completed": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def _write_failure(output: Path, error: BaseException) -> dict[str, Any]:
    evidence = _failure(error)
    sidecar = output.with_name("worker.failure.json")
    V2.write_new(sidecar, evidence)
    if not output.exists():
        V2.write_new(output, evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("worker output은 덮어쓰지 않는다")
        job = validate_job(json.loads(args.job.read_text(encoding="utf-8")))
        if job["mode"] == "train":
            V2.run_train(job, args.output)
        elif job["mode"] == "probe":
            V2.run_probe(job, args.output)
        else:
            V2.run_evaluate(job, args.output)
    except BaseException as error:  # noqa: BLE001 - app 종료 전 원오류를 write-once로 보존한다.
        if isinstance(error, SystemExit) and error.code in (0, None) and V2.completed(args.output):
            return 0
        evidence = _write_failure(args.output, error)
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
