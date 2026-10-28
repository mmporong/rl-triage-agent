"""JOB4가 기존 F1 계약을 보존하며 정확한 AWS L40S type만 추가 수용하는지 확인한다."""
from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v1 as JOB1  # pyright: ignore[reportMissingImports]
import agentic_play_job_v3 as JOB3  # pyright: ignore[reportMissingImports]
import agentic_play_job_v4 as JOB4  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as T1  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v3 as T3  # pyright: ignore[reportMissingImports]

AWS_TARGET = {
    "resource_id": "82ewdz0tm",
    "resource_name": "rl-triage-l40s-f1-20261028d",
    "instance_type": "g6e.4xlarge",
}


def test_aws_g6e_target_is_accepted_and_scope_restores_job3() -> None:
    original = JOB1._check_body
    value = JOB4.build_protocol(**T3._real_args(recording_target=AWS_TARGET, batch_budget_usd="3.00"))
    assert value["schema"] == JOB4.PROTOCOL_SCHEMA
    assert value["recording_target"] == AWS_TARGET
    assert value["interpretation"] == JOB3.INTERPRETATION
    assert JOB1._check_body is original and not JOB4.scope_active() and not JOB3.scope_active()


def test_existing_l40s_named_target_remains_accepted() -> None:
    value = JOB4.build_protocol(**T3._real_args(recording_target=T1.TARGET))
    assert value["schema"] == JOB4.PROTOCOL_SCHEMA


@pytest.mark.parametrize("instance_type", ["g6.4xlarge", "g6e.8xlarge", "aws-l4.1gpu", "a10g.large"])
def test_other_aws_or_gpu_types_are_rejected(instance_type: str) -> None:
    with pytest.raises(ValueError, match="등록된 L40S"):
        JOB4.build_protocol(**T3._real_args(recording_target=dict(AWS_TARGET, instance_type=instance_type)))


def test_job3_still_rejects_aws_machine_name_without_l40s_token() -> None:
    with pytest.raises(ValueError, match="L40S"):
        JOB3.build_protocol(**T3._real_args(recording_target=AWS_TARGET))


def test_nested_scope_changes_only_schema_and_check_then_restores() -> None:
    with JOB4.aws_f1_play_scope():
        with JOB4.aws_f1_play_scope():
            assert JOB1.PROTOCOL_SCHEMA == JOB4.PROTOCOL_SCHEMA
            assert JOB1.JOB_SCHEMA == JOB4.JOB_SCHEMA
            assert JOB1._check_body is JOB4._check_body
            assert JOB1.PLAYBACK is JOB3.PLAYBACK and JOB1._pairs is JOB3._pairs
        assert JOB4._STATE["depth"] == 1
    assert JOB1.PROTOCOL_SCHEMA == JOB1.PROTOCOL_SCHEMA == "agentic_play_protocol_v1"
    assert JOB1._check_body is JOB4._BASE["_check_body"]


def _quote() -> tuple[dict, bytes, str]:
    row = {"type": "g6e.4xlarge", "cloud": "aws", "provider": "aws", "gpu_name": "L40S",
           "gpu_count": 1, "vram_per_gpu_gb": 48, "total_vram_gb": 48, "price_per_hour": 3.605088}
    raw = json.dumps([row], separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    creation = {"provider": "aws", "instance_type": "g6e.4xlarge", "gpu_name": "L40S", "gpu_count": 1,
                "vram_gb": 48, "quoted_usd_per_h": "3.605088", "quote_stdout_sha256": digest}
    return creation, raw, digest


def test_quote_binding_accepts_exact_aws_single_l40s_raw_evidence() -> None:
    creation, raw, digest = _quote()
    assert JOB4.validate_aws_l40s_quote(creation, raw, expected_sha256=digest)["type"] == "g6e.4xlarge"


@pytest.mark.parametrize(("field", "value"), [
    ("provider", "nebius"), ("instance_type", "g6e.8xlarge"), ("gpu_name", "L4"),
    ("gpu_count", 2), ("vram_gb", 96),
])
def test_quote_binding_rejects_provider_type_or_gpu_spoof(field: str, value: object) -> None:
    creation, raw, digest = _quote()
    with pytest.raises(ValueError, match="AWS L40S"):
        JOB4.validate_aws_l40s_quote(dict(creation, **{field: value}), raw, expected_sha256=digest)


def test_quote_binding_rejects_raw_sha_or_row_mutation() -> None:
    creation, raw, digest = _quote()
    with pytest.raises(ValueError, match="원문 SHA"):
        JOB4.validate_aws_l40s_quote(creation, raw + b"\n", expected_sha256=digest)
    row = json.loads(raw)[0]
    row["gpu_count"] = 2
    changed = json.dumps([row], separators=(",", ":")).encode()
    changed_sha = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="provider·GPU·가격"):
        JOB4.validate_aws_l40s_quote(
            dict(creation, quote_stdout_sha256=changed_sha), changed, expected_sha256=changed_sha
        )
