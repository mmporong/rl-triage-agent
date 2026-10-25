"""Atomic Writer worker와 원문 증거 검증기를 고정하는 Brev backend."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import vision_brev_render_backend_v2 as V2  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v4 as V4  # pyright: ignore[reportMissingImports]
import vision_execution_evidence_v2 as EVIDENCE  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
WORKER_V4 = ROOT / "bench/vision_play_worker_v4.py"
SOURCE_FILES = frozenset(
    {
        *V4.SOURCE_FILES,
        "bench/vision_frame_sync_v1.py",
        "bench/vision_writer_payload_v1.py",
        "bench/vision_atomic_capture_v1.py",
        "bench/vision_atomic_smoke_v1.py",
        "bench/vision_execution_evidence_v2.py",
        "bench/vision_play_worker_v4.py",
        "evals/vision_brev_render_backend_v5.py",
        "bench/run_vision_brev_atomic_v1.py",
    }
)


def _verify_source_closure(protocol: Mapping[str, Any], repo: Path = ROOT) -> None:
    source = protocol.get("source_sha256_lf")
    if not isinstance(source, Mapping) or not SOURCE_FILES <= set(source):
        raise ValueError("atomic worker/backend/entry source 결합이 빠졌다")
    for relative in SOURCE_FILES:
        path, expected = repo / relative, source.get(relative)
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or path.is_symlink()
            or V4._sha256_lf(path) != expected
        ):
            raise ValueError(f"atomic source LF SHA가 다르다: {relative}")


class AtomicVisionBrevBackend(V4.SupervisedVisionBrevBackend):
    def _verify_mode_result(
        self, result_path: Path, job: Mapping[str, Any], job_sha: str
    ) -> dict[str, Any]:
        result = V2.AttestedVisionBrevBackend._verify_mode_result(
            self, result_path, job, job_sha
        )
        stage = result_path.parent.parent
        EVIDENCE.verify_manifest(
            stage / EVIDENCE.MANIFEST_NAME, stage / "job.json", result_path
        )
        return result

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
    ) -> AtomicVisionBrevBackend:
        _verify_source_closure(protocol)
        delegate = kwargs.pop("popen", subprocess.Popen)
        pinned = V4.PinnedVisionWorkerPopen(delegate, python, worker_v3=WORKER_V4)
        return cast(
            AtomicVisionBrevBackend,
            V2.AttestedVisionBrevBackend.from_runtime.__func__(
                cls,
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
