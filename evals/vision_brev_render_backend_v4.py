"""종료 검증 supervisor worker를 attested render gate에 고정하는 backend."""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import vision_brev_render_backend_v2 as V2  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v3 as V3  # pyright: ignore[reportMissingImports]
import vision_execution_evidence_v1 as EVIDENCE  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
WORKER_V3 = ROOT / "bench/vision_play_worker_v3.py"
SOURCE_FILES = frozenset({
    *V3.SOURCE_FILES,
    "bench/vision_capture_readiness_v1.py",
    "bench/vision_shutdown_policy_v1.py",
    "bench/vision_play_worker_v3.py",
    "evals/vision_brev_render_backend_v4.py",
    "bench/run_vision_brev_annotated_v2.py",
    "bench/vision_execution_evidence_v1.py",
    "bench/vision_fxaa_policy_v1.py",
    "tests/test_vision_shutdown_policy_v1.py",
})


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verify_source_closure(protocol: Mapping[str, Any], repo: Path = ROOT) -> None:
    source = protocol.get("source_sha256_lf")
    if not isinstance(source, Mapping) or not SOURCE_FILES <= set(source):
        raise ValueError("supervisor worker/backend/entry source 결합이 빠졌다")
    for relative in SOURCE_FILES:
        path, expected = repo / relative, source.get(relative)
        if (not isinstance(expected, str) or not path.is_file() or path.is_symlink()
                or _sha256_lf(path) != expected):
            raise ValueError(f"supervisor source LF SHA가 다르다: {relative}")


class PinnedVisionWorkerPopen:
    """기존 exact owned-gate argv의 worker v1 자리만 supervisor로 바꾼다."""

    def __init__(self, delegate: Any, python: Path, worker_v1: Path = V3.WORKER_V1,
                 worker_v3: Path = WORKER_V3) -> None:
        self.delegate = delegate
        self.python = str(Path(python).absolute())
        self.worker_v1 = str(Path(worker_v1).absolute())
        self.worker_v3 = str(Path(worker_v3).absolute())

    def __call__(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, (str, bytes, os.PathLike)):
            argv: tuple[str, ...] | None = None
        elif isinstance(args, Sequence):
            argv = tuple(os.fspath(value) for value in args)
        else:
            argv = None
        prefix = (self.python, "-I", "-S", "-c", V2.OWNED_WORKER_GATE,
                  self.python, "-B", self.worker_v1, "--job")
        if argv is not None and len(argv) == 10 and argv[:9] == prefix and argv[9]:
            return self.delegate([*argv[:7], self.worker_v3, *argv[8:]], *positional, **kwargs)
        related = argv is not None and (
            self.worker_v1 in argv or self.worker_v3 in argv or V2.OWNED_WORKER_GATE in argv
        )
        if related:
            raise ValueError("render supervisor는 고정 owned-gate argv로만 실행할 수 있다")
        return self.delegate(args, *positional, **kwargs)


class SupervisedVisionBrevBackend(V2.AttestedVisionBrevBackend):
    def _verify_mode_result(self, result_path: Path, job: Mapping[str, Any], job_sha: str) -> dict[str, Any]:
        result = super()._verify_mode_result(result_path, job, job_sha)
        stage = result_path.parent.parent
        EVIDENCE.verify_manifest(stage / EVIDENCE.MANIFEST_NAME, stage / "job.json", result_path)
        return result

    def _execution_auxiliary(self, result_path: Path, job: Mapping[str, Any]) -> dict[str, str]:
        stage = result_path.parent.parent
        manifest = stage / EVIDENCE.MANIFEST_NAME
        EVIDENCE.verify_manifest(manifest, stage / "job.json", result_path)
        return {"file": manifest.name, "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    @classmethod
    def from_runtime(cls, lab: Path, python: Path, protocol: Mapping[str, Any], anchor: str,
                     runtime_root: Path, training_plan: Sequence[Mapping[str, Any]],
                     **kwargs: Any) -> SupervisedVisionBrevBackend:
        _verify_source_closure(protocol)
        delegate = kwargs.pop("popen", subprocess.Popen)
        pinned = PinnedVisionWorkerPopen(delegate, python)
        return cast(SupervisedVisionBrevBackend, super().from_runtime(
            lab, python, protocol, anchor, runtime_root, training_plan, popen=pinned, **kwargs,
        ))
