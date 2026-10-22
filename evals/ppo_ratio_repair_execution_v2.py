"""PPO ratio backend v1에 GPU 공유 admission 프로필 하나만 더한 판.

v1 backend의 첫 학습 단계 GPU 사전 점검이 기존 화면 앱 허용 목록에 없는 Claude 데스크톱 앱(claude.exe)과
Edge(msedge.exe)를 거부했다. 두 앱은 이미 허용된 chrome.exe·msedgewebview2.exe와 같은 화면 맥락(C+G)이다.
이 판은 허용 화면 앱 이름 두 개만 더하고 계산 작업 차단·메모리 상한·엔진 종류 등 나머지 기준은 그대로 쓴다.
record_evidence_media.py와 v1 backend는 다른 프로토콜이 SHA로 묶고 있어 바꾸지 않는다. 프로필은 GPU 조회
호출 동안에만 등록하고, 동결 backend의 GPU 조회 함수도 이 판의 호출 동안에만 바꾼 뒤 되돌린다.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import ppo_ratio_repair_execution as V1  # noqa: E402
import r1_repair_execution as FROZEN  # noqa: E402
import record_evidence_media as GPU  # noqa: E402

TASK, FIXED_EVAL_PROTOCOL = V1.TASK, V1.FIXED_EVAL_PROTOCOL
sha, write_new, preflight = V1.sha, V1.write_new, V1.preflight
BASE_PROFILE = GPU.EXPECTED_GPU_ADMISSION["policy"]
GPU_PROFILE = "desktop_shared_recording_claude_v1"
EXTRA_DISPLAY_PROCESS_NAMES = ("claude.exe", "msedge.exe")
GPU_ADMISSION = {**GPU.EXPECTED_GPU_ADMISSION, "policy": GPU_PROFILE,
                 "allowed_display_process_names": sorted(GPU.DISPLAY_PROCESS_NAMES | set(EXTRA_DISPLAY_PROCESS_NAMES))}


def query_gpu() -> dict[str, Any]:
    profiles = GPU.GPU_ADMISSION_PROFILES
    if GPU_PROFILE in profiles:
        raise ValueError("GPU admission 프로필이 이미 등록돼 있다")
    profiles[GPU_PROFILE] = GPU_ADMISSION
    try:
        snapshot = GPU.query_gpu(GPU_PROFILE)
    finally:
        profiles.pop(GPU_PROFILE, None)
    if not isinstance(snapshot, dict) or snapshot.get("admission_policy") != GPU_PROFILE:
        raise ValueError("GPU admission receipt 형식 또는 profile이 다르다")
    if type(snapshot.get("busy")) is not bool:
        raise ValueError("GPU admission busy 필드 형식이 다르다")
    return snapshot


@contextmanager
def gpu_admission() -> Iterator[None]:
    """동결 backend의 GPU 사전·사후 점검을 이 판의 프로필로 읽게 하고 호출 뒤 원래 함수로 되돌린다."""
    original = FROZEN._query_gpu
    FROZEN._query_gpu = query_gpu
    try:
        yield
    finally:
        FROZEN._query_gpu = original


def train(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.train(*args, **kwargs)


def inspect(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.inspect(*args, **kwargs)


def fixed_evaluate(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.fixed_evaluate(*args, **kwargs)


def legacy_probe(*args, **kwargs) -> dict[str, Any]:
    with gpu_admission():
        return V1.legacy_probe(*args, **kwargs)
