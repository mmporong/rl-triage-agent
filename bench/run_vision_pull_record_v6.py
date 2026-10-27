"""새 자원 승인 검사를 부모·AABA·record 진입점에 적용하는 녹화 실행기."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import run_vision_pull_record_v5 as PREVIOUS
import vision_pull_record_job_v6 as JOB


@contextmanager
def _job_contract() -> Iterator[None]:
    previous_job = PREVIOUS.JOB
    previous_file = PREVIOUS.__file__
    PREVIOUS.JOB = JOB
    PREVIOUS.__file__ = __file__
    try:
        yield
    finally:
        PREVIOUS.JOB = previous_job
        PREVIOUS.__file__ = previous_file


def aaba_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.aaba_child(job_path, **kwargs)


def sdk_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.sdk_child(job_path, **kwargs)


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = PREVIOUS.OWNED._process_identity,
    cleanup: Callable[..., bool] = PREVIOUS.OWNED._safe_cleanup,
) -> dict[str, Any]:
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("pull record v6 실제 실행은 소유 Linux group에서 수행한다")
    with _job_contract():
        return PREVIOUS.run_job(
            job_path,
            popen=popen,
            gpu_reader=gpu_reader,
            identify=identify,
            cleanup=cleanup,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--aaba-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-authorization", type=Path)
    parser.add_argument("--resource-issuance", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id", choices=JOB.CHECKPOINT_IDS)
    args = parser.parse_args(argv)
    if args.sdk_child or args.aaba_child:
        if args.job is None:
            parser.error("child 실행에는 --job이 필요하다")
        return aaba_child(args.job) if args.aaba_child else sdk_child(args.job)
    if args.job is None:
        required = (
            args.protocol, args.resource_authorization, args.resource_issuance,
            args.resource_proof, args.resource_sha256, args.isaaclab, args.python,
            args.stage, args.checkpoint_logical_id,
        )
        if any(value is None for value in required):
            parser.error("job 생성 인자가 모두 필요하다")
        args.job = JOB.create_job(
            args.protocol, args.resource_authorization, args.resource_issuance,
            args.resource_proof, args.resource_sha256, args.isaaclab, args.python,
            args.stage, args.checkpoint_logical_id,
        )
    result = run_job(args.job)
    return 0 if result.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
