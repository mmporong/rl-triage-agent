"""agentic play job v4: F1 계약을 AWS ``g6e.4xlarge`` L40S 자원까지 확장한다.

JOB3의 F1 등록·pair·lease 계약은 그대로 사용한다. 이 scope 안에서만 schema와
recording target 검사 함수를 v4로 바꾸며, 기존 L40S 이름 또는 정확한 AWS
``g6e.4xlarge``만 허용한다. provider·GPU 원문 결합은 상위 등록 준비기가 검증한다.
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import agentic_play_job_v1 as JOB1
import agentic_play_job_v2 as JOB2
import agentic_play_job_v3 as JOB3

PROTOCOL_SCHEMA = "agentic_play_protocol_v4"
JOB_SCHEMA = "agentic_play_job_v4"
AWS_L40S_INSTANCE_TYPE = "g6e.4xlarge"
AWS_PROVIDER = "aws"
AWS_GPU_NAME = "L40S"
AWS_GPU_COUNT = 1
AWS_VRAM_GB = 48
SOURCE_FILES = frozenset(set(JOB3.SOURCE_FILES) | {
    "bench/agentic_play_job_v4.py",
    "tests/test_agentic_play_job_v4.py",
})
PREDICTION = JOB3.PREDICTION
OUTCOMES = dict(JOB3.OUTCOMES)

_LOCK = threading.RLock()
_BASE: dict[str, Any] = {
    "PROTOCOL_SCHEMA": JOB3.PROTOCOL_SCHEMA,
    "JOB_SCHEMA": JOB3.JOB_SCHEMA,
    "_check_body": JOB1._check_body,
}
_STATE = {"depth": 0}


def validate_aws_l40s_quote(
    creation: Mapping[str, Any], quote_raw: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    """공개 생성 등록과 Brev 견적 원문이 정확한 단일 AWS L40S 행을 가리키는지 확인한다."""
    if hashlib.sha256(quote_raw).hexdigest() != expected_sha256:
        raise ValueError("Brev quote 원문 SHA가 다르다")
    expected = {
        "provider": AWS_PROVIDER,
        "instance_type": AWS_L40S_INSTANCE_TYPE,
        "gpu_name": AWS_GPU_NAME,
        "gpu_count": AWS_GPU_COUNT,
        "vram_gb": AWS_VRAM_GB,
    }
    if any(creation.get(name) != value for name, value in expected.items()):
        raise ValueError("생성 등록의 AWS L40S provider·type·GPU가 다르다")
    if creation.get("quote_stdout_sha256") != expected_sha256:
        raise ValueError("생성 등록의 quote 원문 SHA가 다르다")
    try:
        rows = json.loads(quote_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Brev quote 원문 JSON이 잘못됐다") from error
    if not isinstance(rows, list):
        raise ValueError("Brev quote 원문은 목록이어야 한다")
    matches = [row for row in rows if isinstance(row, dict) and row.get("type") == AWS_L40S_INSTANCE_TYPE]
    if len(matches) != 1:
        raise ValueError("Brev quote에 등록 type이 정확히 한 행이어야 한다")
    row = matches[0]
    quote = JOB1._positive_decimal(creation.get("quoted_usd_per_h"), "creation quote")
    if (row.get("provider") != AWS_PROVIDER or row.get("cloud") != AWS_PROVIDER
            or row.get("gpu_name") != AWS_GPU_NAME or row.get("gpu_count") != AWS_GPU_COUNT
            or row.get("vram_per_gpu_gb") != AWS_VRAM_GB or row.get("total_vram_gb") != AWS_VRAM_GB
            or JOB1._positive_decimal(row.get("price_per_hour"), "quote row price") != quote):
        raise ValueError("Brev quote 행의 provider·GPU·가격이 생성 등록과 다르다")
    return dict(row)


def _check_body(
    protocol: Mapping[str, Any], *, now: datetime, require_current: bool
) -> dict[str, Any]:
    """JOB1 본문 검사를 보존하고 recording target 허용 목록만 확장한다."""
    if (protocol["schema"] != PROTOCOL_SCHEMA
            or protocol["registered_before_measurement"] is not True
            or not isinstance(protocol["protocol_id"], str)
            or JOB1.COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
            or protocol["gpu_validation_status"] != "pending_live_gpu_execution"):
        raise ValueError("agentic play protocol 기본 계약이 다르다")
    sources = JOB1.COMMON._source_map(protocol["source_sha256_lf"])
    if not JOB1.REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("agentic play source closure가 부족하다")
    runtime = JOB1._runtime(protocol["runtime_source_sha256_lf"])
    target = JOB1._target(protocol["recording_target"], "recording target")
    instance_type = target["instance_type"].lower()
    if "l40s" not in instance_type and instance_type != AWS_L40S_INSTANCE_TYPE:
        raise ValueError("recording target이 등록된 L40S 계열이 아니다")
    resources = JOB1.COMMON._exact(protocol["resources"], {
        "compute_quote_usd_per_h", "batch_budget_usd", "deadline_utc", "max_owned_instances", "cost_scope",
    }, "recording resources")
    if resources["max_owned_instances"] != 1 or resources["cost_scope"] != JOB1.RESOURCE_SCOPE:
        raise ValueError("recording resource 상한이 다르다")
    quote = JOB1._positive_decimal(resources["compute_quote_usd_per_h"], "compute quote")
    budget = JOB1._positive_decimal(resources["batch_budget_usd"], "batch budget")
    required = (quote * Decimal(JOB1.TECHNICAL_LIMITS["max_policies"])
                * Decimal(JOB1.TECHNICAL_LIMITS["max_wall_s_per_policy"]) / Decimal(3600))
    if budget < required:
        raise ValueError("batch budget이 2 policy wall 상한보다 작다")
    registered = JOB1._utc(protocol["registered_at_utc"], "registered")
    deadline = JOB1._utc(resources["deadline_utc"], "deadline")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    current = now.astimezone(timezone.utc)  # noqa: UP017 - Isaac Python 3.10
    if registered > current or registered >= deadline or (require_current and current >= deadline):
        raise ValueError("protocol 등록·deadline이 유효하지 않다")
    checkpoints = JOB1.RECORD_JOB._checkpoints(protocol["checkpoints"])
    registrations = JOB1._registrations(protocol["playback_registrations"], checkpoints, sources)
    pairs = JOB1._pairs(protocol["pairs"], registrations)
    JOB1._sidecar(protocol["sidecar"])
    profile = JOB1.COMMON._exact(protocol["record_profile"], {"value", "sha256"}, "record profile")
    if (profile["value"] != JOB1.RECORD.fixed_profile()
            or profile["sha256"] != JOB1.RECORD.digest(JOB1.RECORD.fixed_profile())):
        raise ValueError("same Play record profile이 다르다")
    if protocol["limits"] != JOB1.TECHNICAL_LIMITS:
        raise ValueError("625 frame/50 fps/2 policy 기술 상한이 다르다")
    if protocol["interpretation"] != JOB1.INTERPRETATION:
        raise ValueError("F2·정상 영상 해석 범위가 등록값과 다르다")
    prereg = JOB1.COMMON._exact(
        protocol["preregistration"], {"prediction", "outcomes", "fixed_flags"}, "preregistration"
    )
    if (not isinstance(prereg["prediction"], str) or not prereg["prediction"].strip()
            or not isinstance(prereg["outcomes"], dict) or set(prereg["outcomes"]) != {
                "recording_complete", "capture_hold", "infrastructure_failure"}
            or prereg["fixed_flags"] != JOB1.FIXED_FLAGS):
        raise ValueError("agentic play 사전등록이 다르다")
    return {"sources": sources, "runtime": runtime, "target": target, "pairs": pairs}


@contextmanager
def aws_f1_play_scope() -> Iterator[None]:
    """JOB3 F1 scope 위에서 v4 schema와 target 검사만 교체하고 복원한다."""
    with JOB3.f1_play_scope(), _LOCK:
        if _STATE["depth"] == 0:
            if any(getattr(JOB1, name) is not value for name, value in _BASE.items()):
                raise RuntimeError("JOB3 계약 전역이 이미 교체됐다")
            JOB1.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
            JOB1.JOB_SCHEMA = JOB_SCHEMA
            JOB1._check_body = _check_body
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                for name, value in _BASE.items():
                    setattr(JOB1, name, value)


def scope_active() -> bool:
    return _STATE["depth"] > 0


def _scoped(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with aws_f1_play_scope():
            return function(*args, **kwargs)

    return wrapper


def build_protocol(*, prediction: str = PREDICTION, **kwargs: Any) -> dict[str, Any]:
    with aws_f1_play_scope(), JOB2.canonical_lease_scope():
        value = JOB1.build_protocol(**kwargs)
        value["preregistration"]["prediction"] = prediction
        value["preregistration"]["outcomes"] = dict(OUTCOMES)
        _check_body(value, now=JOB1._utc(value["registered_at_utc"], "registered"), require_current=False)
    return value


validate_protocol = _scoped(JOB2.validate_protocol)
validate_resource_proof = _scoped(JOB2.validate_resource_proof)
build_resource_proof = _scoped(JOB2.build_resource_proof)
validate_job = _scoped(JOB2.validate_job)
create_job = _scoped(JOB2.create_job)
lease_paths = _scoped(JOB2.lease_paths)
verify_lease = _scoped(JOB2.verify_lease)


@contextmanager
def resource_lease(protocol_path: Path, **kwargs: Any) -> Iterator[dict[str, str]]:
    with aws_f1_play_scope(), JOB2.resource_lease(protocol_path, **kwargs) as reference:
        yield reference


def __getattr__(name: str) -> Any:
    try:
        return getattr(JOB3, name)
    except AttributeError as error:
        raise AttributeError(f"agentic_play_job_v4에 {name}이 없다") from error
