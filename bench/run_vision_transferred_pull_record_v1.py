"""전송 checkpoint를 새 recording resource에서 RUN5 안전 경계로 녹화한다."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import run_vision_pull_record_v5 as PREVIOUS
import vision_transferred_pull_job_v1 as JOB

POSTFLIGHT_RESERVE_S = float(JOB.LIMITS["postflight_reserve_s_per_policy"])


@contextmanager
def _job_contract() -> Iterator[None]:
    previous = PREVIOUS.JOB
    previous_reserve = PREVIOUS.POSTFLIGHT_RESERVE_S
    PREVIOUS.JOB = JOB
    PREVIOUS.POSTFLIGHT_RESERVE_S = POSTFLIGHT_RESERVE_S
    try:
        yield
    finally:
        PREVIOUS.JOB = previous
        PREVIOUS.POSTFLIGHT_RESERVE_S = previous_reserve


def aaba_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.aaba_child(job_path, **kwargs)


def sdk_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.sdk_child(job_path, **kwargs)


def _child_launcher(popen: Callable[..., Any]) -> Callable[..., Any]:
    old = str(Path(PREVIOUS.__file__).resolve())
    current = str(Path(__file__).resolve())

    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        rewritten = [current if value == old else value for value in command]
        if rewritten == list(command):
            raise ValueError("transferred pull 자식 명령에 RUN5 진입점이 없다")
        return popen(rewritten, **kwargs)

    return launch


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = PREVIOUS.OWNED._process_identity,
    cleanup: Callable[..., bool] = PREVIOUS.OWNED._safe_cleanup,
) -> dict[str, Any]:
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("transferred pull 실제 실행은 소유 Linux group에서 수행한다")
    with _job_contract():
        return PREVIOUS.run_job(
            job_path, popen=_child_launcher(popen), gpu_reader=gpu_reader,
            identify=identify, cleanup=cleanup,
        )


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
    parser.add_argument("--checkpoint-logical-id", choices=JOB.CHECKPOINT_IDS)
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
        args.job = JOB.create_job(
            args.protocol, args.resource_proof, args.resource_sha256,
            args.isaaclab, args.python, args.stage, args.checkpoint_logical_id,
        )
    result = run_job(args.job)
    return 0 if result.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
