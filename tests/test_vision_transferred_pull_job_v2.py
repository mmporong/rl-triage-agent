from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_transferred_pull_job_v2 as J  # pyright: ignore[reportMissingImports]


def test_source_closure_includes_all_v2_implementation_and_tests() -> None:
    assert {
        "bench/vision_pull_record_capture_v2.py",
        "bench/vision_pull_record_worker_v2.py",
        "bench/vision_transferred_pull_job_v2.py",
        "bench/run_vision_transferred_pull_record_v2.py",
        "tests/test_vision_pull_record_capture_v2.py",
        "tests/test_vision_pull_record_worker_v2.py",
        "tests/test_vision_transferred_pull_job_v2.py",
        "tests/test_run_vision_transferred_pull_record_v2.py",
    } <= J.REQUIRED_SOURCE_FILES


def test_wrapped_contract_uses_v2_schemas_and_restores_v1(monkeypatch) -> None:
    previous = (J.PREVIOUS.PROTOCOL_SCHEMA, J.PREVIOUS.JOB_SCHEMA, J.PREVIOUS.REQUIRED_SOURCE_FILES)

    def observe() -> dict[str, object]:
        return {
            "protocol": J.PREVIOUS.PROTOCOL_SCHEMA,
            "job": J.PREVIOUS.JOB_SCHEMA,
            "sources": J.PREVIOUS.REQUIRED_SOURCE_FILES,
        }

    monkeypatch.setattr(J.PREVIOUS, "build_protocol", observe)
    observed = J.build_protocol()
    assert observed == {"protocol": J.PROTOCOL_SCHEMA, "job": J.JOB_SCHEMA, "sources": J.REQUIRED_SOURCE_FILES}
    assert (J.PREVIOUS.PROTOCOL_SCHEMA, J.PREVIOUS.JOB_SCHEMA, J.PREVIOUS.REQUIRED_SOURCE_FILES) == previous
