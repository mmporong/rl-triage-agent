"""Vision source 학습 child를 recovery worker v3에 명시적으로 결합한다."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import brev_recovery_backend_v4 as RECOVERY
import vision_brev_backend_v1 as V1

WORKER = ROOT / "bench/brev_recovery_worker_v3.py"


def worker_popen(
    popen: Callable[..., subprocess.Popen[Any]],
) -> Callable[..., subprocess.Popen[Any]]:
    """동결 backend의 worker argv 한 항목만 새 worker로 바꾼다."""

    def launch(argv: Sequence[str], *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        command = list(argv)
        if len(command) < 3 or Path(command[2]).resolve() != RECOVERY.WORKER.resolve():
            raise ValueError("recovery worker argv 결합이 다르다")
        command[2] = str(WORKER)
        return popen(command, *args, **kwargs)

    return launch


class VisionBrevTrainingBackendV2(V1.VisionBrevBackend):
    """기존 source·임대·cleanup 계약을 유지하는 worker-v3 판."""

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
    ) -> VisionBrevTrainingBackendV2:
        read_gpu = kwargs.pop("read_gpu", None)
        resource_proof = kwargs.pop("resource_proof", None)
        child_popen = kwargs.pop("popen", subprocess.Popen)
        if kwargs:
            raise TypeError(f"지원하지 않는 backend 인자: {sorted(kwargs)}")
        if read_gpu is None:
            if not isinstance(resource_proof, Mapping):
                raise ValueError("기본 원격 GPU reader에는 resource_proof가 필요하다")
            capture = {key: resource_proof[key] for key in V1.REMOTE.CAPTURE_KEYS}
            read_gpu = V1.REMOTE.StrictRemoteGpuReader(
                capture, Path(runtime_root) / "gpu-observations"
            )
        recovery = V1.SafeRecoveryBackend(
            lab,
            python,
            dict(protocol),
            anchor,
            read_gpu=read_gpu,
            popen=worker_popen(child_popen),
        )
        if isinstance(resource_proof, Mapping):
            recovery.vision_resource_proof = dict(resource_proof)
        return cls(recovery, runtime_root, training_plan)
