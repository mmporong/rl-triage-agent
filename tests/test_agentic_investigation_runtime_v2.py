from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_investigation_runtime_v2 as M


def test_fresh_prefix_uses_actual_result_reference() -> None:
    reference = {"path": "/tmp/result.json", "sha256": "a" * 64}
    assert M.prefix_source({"schema": "fresh_frame_runtime_prefix_v1",
                            "actual_result": reference}) == ("fresh_actual_result", reference)


def test_legacy_prefix_keeps_call_capsule_reference() -> None:
    reference = {"path": "/tmp/capsule.json", "sha256": "b" * 64}
    assert M.prefix_source({"schema": "legacy", "capsule": reference}) == (
        "call_capsule", reference)


def test_fresh_prefix_never_accepts_missing_actual_result() -> None:
    try:
        M.prefix_source({"schema": "fresh_frame_runtime_prefix_v1"})
    except ValueError as error:
        assert "actual_result" in str(error)
    else:
        raise AssertionError("missing actual_result accepted")


def test_runtime_scope_restores_v1_runtime() -> None:
    original = M.V1.Runtime
    with M.runtime_scope():
        assert M.V1.Runtime is M.Runtime
    assert M.V1.Runtime is original
