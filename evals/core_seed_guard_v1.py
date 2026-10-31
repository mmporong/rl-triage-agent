"""중단된 외부 provider 작업 폴더를 열지 않는 RL core seed 검사."""
from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path

import preregistered_seed_guard as ORIGINAL

EXCLUDED_RELATIVE = Path(
    "bench/private/claude_reward_track_20261013/codex_main_20261025/"
    "vision_goal/ri_codex2_tokenfactory_live_160")
_LOCK = threading.RLock()


def assert_fresh_seeds(seeds: Iterable[int], isaaclab: Path, *,
                       repo: Path = ORIGINAL.ROOT) -> dict:
    """원래 실행기록 검사를 유지하고 제외 폴더는 열거 전 가지치기한다."""
    repo = Path(repo).resolve()
    excluded = repo / EXCLUDED_RELATIVE

    def files(root: Path):
        def fail(error):
            raise ValueError(f"seed 기록 열거 실패: {error.filename}") from error

        for directory, subdirs, filenames in os.walk(root, followlinks=False, onerror=fail):
            base = Path(directory)
            subdirs[:] = [name for name in subdirs if base / name != excluded]
            for name in [*subdirs, *filenames]:
                if (base / name).is_symlink():
                    raise ValueError(f"seed 검사 symlink 거부: {name}")
            yield base, filenames

    with _LOCK:
        old = ORIGINAL._files
        ORIGINAL._files = files
        try:
            result = ORIGINAL.assert_fresh_seeds(seeds, isaaclab, repo=repo)
        finally:
            ORIGINAL._files = old
    return {**result, "schema": "core_seed_freshness_v1",
            "excluded_before_enumeration": ["<repo>/" + EXCLUDED_RELATIVE.as_posix()],
            "scope": "현재 RL core 실행 기록·지정 Isaac Lab 로그에 한정하며 제외 provider 폴더·외부 보관본의 미사용을 보증하지 않는다"}
