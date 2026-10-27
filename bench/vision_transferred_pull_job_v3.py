"""v3 video 내용 결합 경로를 source closure에 추가한 전송 녹화 계약."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import vision_transferred_pull_job_v2 as PREVIOUS

PROTOCOL_SCHEMA = "vision_transferred_pull_protocol_v3"
JOB_SCHEMA = "vision_transferred_pull_job_v3"
REQUIRED_SOURCE_FILES = PREVIOUS.REQUIRED_SOURCE_FILES | {
    "bench/vision_pull_record_worker_v3.py",
    "bench/vision_transferred_pull_job_v3.py",
    "bench/run_vision_transferred_pull_record_v3.py",
    "tests/test_vision_pull_record_worker_v3.py",
    "tests/test_vision_transferred_pull_job_v3.py",
    "tests/test_run_vision_transferred_pull_record_v3.py",
}

COMMON = PREVIOUS.COMMON
SOURCE = PREVIOUS.SOURCE
RECORD = PREVIOUS.RECORD
RECORD_JOB = PREVIOUS.RECORD_JOB
ROOT = PREVIOUS.ROOT
MODE = PREVIOUS.MODE
RESOURCE_SCHEMA = PREVIOUS.RESOURCE_SCHEMA
RESULT_SCHEMA = PREVIOUS.RESULT_SCHEMA
FAILURE_SCHEMA = PREVIOUS.FAILURE_SCHEMA
CHECKPOINT_IDS = PREVIOUS.CHECKPOINT_IDS
JOB_KEYS = PREVIOUS.JOB_KEYS
SOURCE_PROTOCOL = PREVIOUS.SOURCE_PROTOCOL
BASE_PROTOCOL = PREVIOUS.BASE_PROTOCOL
TECHNICAL_LIMITS = PREVIOUS.TECHNICAL_LIMITS
LIMITS = PREVIOUS.LIMITS
FIXED_FLAGS = PREVIOUS.FIXED_FLAGS
PROTOCOL_KEYS = PREVIOUS.PROTOCOL_KEYS
TARGET_KEYS = PREVIOUS.TARGET_KEYS
RESOURCE_KEYS = PREVIOUS.RESOURCE_KEYS
FRESH_CAPTURE_S = PREVIOUS.FRESH_CAPTURE_S
_read = PREVIOUS._read
_sha = PREVIOUS._sha
_regular = PREVIOUS._regular
write_new = PREVIOUS.write_new


@contextmanager
def _v3_contract() -> Iterator[None]:
    previous_protocol = PREVIOUS.PROTOCOL_SCHEMA
    previous_job = PREVIOUS.JOB_SCHEMA
    previous_sources = PREVIOUS.REQUIRED_SOURCE_FILES
    PREVIOUS.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    PREVIOUS.JOB_SCHEMA = JOB_SCHEMA
    PREVIOUS.REQUIRED_SOURCE_FILES = REQUIRED_SOURCE_FILES
    try:
        yield
    finally:
        PREVIOUS.PROTOCOL_SCHEMA = previous_protocol
        PREVIOUS.JOB_SCHEMA = previous_job
        PREVIOUS.REQUIRED_SOURCE_FILES = previous_sources


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _v3_contract():
        return getattr(PREVIOUS, name)(*args, **kwargs)


def build_protocol(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call("build_protocol", *args, **kwargs)


def validate_protocol(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call("validate_protocol", *args, **kwargs)


def validate_resource_proof(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call("validate_resource_proof", *args, **kwargs)


def build_resource_proof(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call("build_resource_proof", *args, **kwargs)


def validate_job(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call("validate_job", *args, **kwargs)


def create_job(*args: Any, **kwargs: Any) -> Any:
    return _call("create_job", *args, **kwargs)
