"""IsaacReadTimes 등록 worker만 attested render gate에 고정하는 backend."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import vision_brev_render_backend_v2 as V2  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
WORKER_V1 = ROOT / "bench/vision_play_worker_v1.py"
WORKER_V2 = ROOT / "bench/vision_play_worker_v2.py"
SOURCE_FILES = frozenset(
    {
        "bench/vision_play_worker_v1.py",
        "bench/vision_play_worker_v2.py",
        "evals/vision_brev_render_backend_v3.py",
        "bench/run_vision_brev_annotated_v1.py",
    }
)


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verify_annotated_source_closure(
    protocol: Mapping[str, Any],
    repo: Path = ROOT,
) -> None:
    source = protocol.get("source_sha256_lf")
    if not isinstance(source, Mapping) or not SOURCE_FILES <= set(source):
        raise ValueError("annotated worker/backend/entry source 결합이 빠졌다")
    for relative in SOURCE_FILES:
        expected = source.get(relative)
        path = repo / relative
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or path.is_symlink()
            or _sha256_lf(path) != expected
        ):
            raise ValueError(f"annotated source LF SHA가 다르다: {relative}")


class PinnedVisionWorkerPopen:
    """동결 gate argv의 worker v1 경로만 v2로 바꾸고 나머지는 위임한다."""

    def __init__(
        self,
        delegate: Any,
        python: Path,
        worker_v1: Path = WORKER_V1,
        worker_v2: Path = WORKER_V2,
    ) -> None:
        self.delegate = delegate
        self.python = str(Path(python).absolute())
        self.worker_v1 = str(Path(worker_v1).absolute())
        self.worker_v2 = str(Path(worker_v2).absolute())

    def __call__(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, (str, bytes, os.PathLike)):
            argv: tuple[str, ...] | None = None
        elif isinstance(args, Sequence):
            argv = tuple(os.fspath(value) for value in args)
        else:
            argv = None

        expected_prefix = (
            self.python,
            "-I",
            "-S",
            "-c",
            V2.OWNED_WORKER_GATE,
            self.python,
            "-B",
            self.worker_v1,
            "--job",
        )
        if (
            argv is not None
            and len(argv) == 10
            and argv[:9] == expected_prefix
            and argv[9]
        ):
            translated = [*argv[:7], self.worker_v2, *argv[8:]]
            return self.delegate(translated, *positional, **kwargs)

        related = argv is not None and (
            self.worker_v1 in argv
            or self.worker_v2 in argv
            or V2.OWNED_WORKER_GATE in argv
        )
        if related:
            raise ValueError("render worker는 고정 owned-gate argv로만 실행할 수 있다")
        return self.delegate(args, *positional, **kwargs)


class AnnotatedVisionBrevBackend(V2.AttestedVisionBrevBackend):
    """v2 검증을 보존하면서 renderer worker 경로만 v2에 고정한다."""

    @classmethod
    def from_runtime(
        cls,
        lab: Path,
        python: Path,
        protocol: Mapping[str, Any],
        anchor: str,
        runtime_root: Path,
        training_plan: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> AnnotatedVisionBrevBackend:
        _verify_annotated_source_closure(protocol)
        delegate = kwargs.pop("popen", subprocess.Popen)
        pinned = PinnedVisionWorkerPopen(delegate, python)
        return cast(
            AnnotatedVisionBrevBackend,
            super().from_runtime(
                lab,
                python,
                protocol,
                anchor,
                runtime_root,
                training_plan,
                popen=pinned,
                **kwargs,
            ),
        )
