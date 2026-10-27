"""CORRECTED RUN3 계약을 RUN5 helper와 v2 자식 진입점까지 완성한다."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import run_vision_corrected_pull_record_v1 as V1
import run_vision_transferred_pull_record_v3 as RUN3
import vision_transferred_pull_job_v3 as JOB3

ROOT = V1.ROOT
PROTOCOL_SCHEMA = V1.PROTOCOL_SCHEMA
JOB_SCHEMA = V1.JOB_SCHEMA
RESOURCE_SCHEMA = V1.RESOURCE_SCHEMA
RESULT_SCHEMA = V1.RESULT_SCHEMA
FAILURE_SCHEMA = V1.FAILURE_SCHEMA
MODE = V1.MODE
LIMITS = V1.LIMITS
FIXED_FLAGS = V1.FIXED_FLAGS
CHECKPOINT_IDS = V1.CHECKPOINT_IDS
CATALOG = V1.CATALOG
COMMON, RECORD, SOURCE = V1.COMMON, V1.RECORD, V1.SOURCE
JOB_KEYS = V1.JOB_KEYS
REQUIRED_SOURCE_FILES = V1.REQUIRED_SOURCE_FILES | {
    "bench/run_vision_corrected_pull_record_v2.py",
    "evals/vision_corrected_native_handoff_v2.py",
    "tests/test_vision_corrected_pull_record_v2.py",
    "tests/test_vision_corrected_native_handoff_v2.py",
}

# RUN5가 JOB module에서 소비하는 helper를 명시적으로 제공한다.
_sha = V1._sha
_regular = JOB3._regular
write_new = JOB3.write_new
_read = V1._read
_verified_binding = V1._verified_binding


def __getattr__(name: str) -> Any:
    """동결 v1 계약과 JOB3 helper를 빠짐없이 전달한다."""
    if hasattr(V1, name):
        return getattr(V1, name)
    if hasattr(JOB3, name):
        return getattr(JOB3, name)
    raise AttributeError(f"{__name__!r} has no attribute {name!r}")


def validate_protocol(path: Path, *, repo: Path | None = None, now: datetime | None = None,
                      require_current: bool = True) -> dict[str, Any]:
    value = V1.validate_protocol(path, repo=repo, now=now, require_current=require_current)
    if not REQUIRED_SOURCE_FILES <= value["source_sha256_lf"].keys():
        raise ValueError("CORRECTED v2 source closure가 부족하다")
    return value


def validate_job(job_path: Path, *, now: datetime | None = None,
                 require_fresh: bool = True) -> dict[str, Any]:
    job = _read(job_path)
    protocol = job.get("protocol")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("path"), str):
        raise TypeError("CORRECTED v2 job protocol ref가 없다")
    validate_protocol(Path(protocol["path"]), now=now, require_current=require_fresh)
    return V1.validate_job(job_path, now=now, require_fresh=require_fresh)


def validate_resource_proof(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return V1.validate_resource_proof(*args, **kwargs)


def build_resource_proof(protocol_path: Path, **kwargs: Any) -> dict[str, Any]:
    validate_protocol(protocol_path, repo=kwargs.get("repo", ROOT))
    return V1.build_resource_proof(protocol_path, **kwargs)


def create_job(protocol_path: Path, resource_path: Path, resource_sha: str, isaaclab: Path,
               python: Path, stage: Path, checkpoint_logical_id: str) -> Path:
    validate_protocol(protocol_path)
    return V1.create_job(protocol_path, resource_path, resource_sha, isaaclab, python,
                         stage, checkpoint_logical_id)


def _child_launcher(popen: Callable[..., Any]) -> Callable[..., Any]:
    old = str(Path(RUN3.PREVIOUS.__file__).resolve())
    current = str(Path(__file__).resolve())

    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        rewritten = [current if value == old else value for value in command]
        if rewritten == list(command):
            raise ValueError("CORRECTED v2 자식 명령에 RUN5 진입점이 없다")
        return popen(rewritten, **kwargs)

    return launch


@contextmanager
def corrected_contract() -> Iterator[None]:
    old_job, old_launcher = RUN3.JOB, RUN3._child_launcher
    RUN3.JOB, RUN3._child_launcher = sys.modules[__name__], _child_launcher
    try:
        yield
    finally:
        RUN3.JOB, RUN3._child_launcher = old_job, old_launcher


def aaba_child(job_path: Path, **kwargs: Any) -> int:
    with corrected_contract():
        return RUN3.aaba_child(job_path, **kwargs)


def sdk_child(job_path: Path, **kwargs: Any) -> int:
    with corrected_contract():
        return RUN3.sdk_child(job_path, **kwargs)


def run_job(job_path: Path, *, popen: Callable[..., Any] = subprocess.Popen,
            gpu_reader: Callable[[], list[int]] | None = None,
            identify: Callable[[int], Mapping[str, Any]] = RUN3.PREVIOUS.OWNED._process_identity,
            cleanup: Callable[..., bool] = RUN3.PREVIOUS.OWNED._safe_cleanup) -> dict[str, Any]:
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("CORRECTED native record는 소유 Linux group에서 실행한다")
    with corrected_contract():
        return RUN3.run_job(job_path, popen=popen, gpu_reader=gpu_reader,
                            identify=identify, cleanup=cleanup)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--aaba-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id", choices=CHECKPOINT_IDS)
    args = parser.parse_args(argv)
    if args.sdk_child or args.aaba_child:
        if args.job is None:
            parser.error("child 실행에는 --job이 필요하다")
        return aaba_child(args.job) if args.aaba_child else sdk_child(args.job)
    if args.job is None:
        required = (args.protocol, args.resource_proof, args.resource_sha256, args.isaaclab,
                    args.python, args.stage, args.checkpoint_logical_id)
        if any(value is None for value in required):
            parser.error("job 생성 인자가 모두 필요하다")
        args.job = create_job(args.protocol, args.resource_proof, args.resource_sha256,
                              args.isaaclab, args.python, args.stage, args.checkpoint_logical_id)
    result = run_job(args.job)
    return 0 if result.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
