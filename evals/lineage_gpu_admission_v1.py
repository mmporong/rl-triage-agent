"""source 수집에서 Windows 설정과 Codex 표시 앱을 등록한 공유 GPU 입장."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import ppo_ratio_repair_execution_v3 as BASE

GPU_PROFILE = "desktop_shared_lineage_sources_v1"
EXTRA_DISPLAY_NAMES = ("applicationframehost.exe", "systemsettings.exe", "codex-computer-use.exe")
_LOCK = RLock()


def policy() -> dict:
    return {**BASE.GPU_ADMISSION, "policy": GPU_PROFILE,
            "allowed_display_process_names": sorted(
                set(BASE.GPU_ADMISSION["allowed_display_process_names"]) | set(EXTRA_DISPLAY_NAMES))}


@contextmanager
def _scope():
    media = BASE.V2.GPU
    with _LOCK:
        if GPU_PROFILE in media.GPU_ADMISSION_PROFILES:
            raise RuntimeError("source 수집 GPU 프로필이 이미 활성화돼 있다")
        original = BASE.query_gpu
        media.GPU_ADMISSION_PROFILES[GPU_PROFILE] = policy()
        BASE.query_gpu = lambda: media.query_gpu(GPU_PROFILE)
        try:
            yield
        finally:
            BASE.query_gpu = original
            media.GPU_ADMISSION_PROFILES.pop(GPU_PROFILE, None)


def query_gpu() -> dict:
    with _scope():
        return BASE.query_gpu()


def snapshot(stage: Path) -> dict:
    with _scope():
        return BASE.snapshot(stage, "before")
