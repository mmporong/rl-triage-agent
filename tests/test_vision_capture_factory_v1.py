from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import vision_play_worker_v1 as V1  # pyright: ignore[reportMissingImports]


def test_conflicting_capture_injection_rejected_before_sdk_imports():
    with pytest.raises(ValueError, match="함께"):
        V1.run_real({}, None, camera_capture=lambda *a: {},
                    camera_capture_factory=lambda *a: (lambda *b: {}, lambda: None))


def test_writer_cleanup_precedes_render_product_and_environment():
    events = []
    product = SimpleNamespace(destroy=lambda: events.append("product"))
    env = SimpleNamespace(close=lambda: events.append("environment"))
    V1._cleanup_capture_resources(None, product, [], env, None,
                                  capture_cleanup=lambda: events.append("writer"))
    assert events == ["writer", "product", "environment"]


def test_writer_cleanup_failure_still_closes_owned_resources():
    events = []
    error = RuntimeError("writer detach failed")

    def cleanup():
        events.append("writer")
        raise error

    with pytest.raises(RuntimeError) as caught:
        V1._cleanup_capture_resources(
            None, SimpleNamespace(destroy=lambda: events.append("product")), [],
            SimpleNamespace(close=lambda: events.append("environment")), None,
            capture_cleanup=cleanup,
        )
    assert caught.value is error
    assert events == ["writer", "product", "environment"]


def test_primary_capture_failure_survives_secondary_cleanup_failure(capsys):
    events = []

    def cleanup():
        raise RuntimeError("secondary cleanup")

    primary = ValueError("primary capture")
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            V1._cleanup_capture_resources(
                None, SimpleNamespace(destroy=lambda: events.append("product")), [],
                SimpleNamespace(close=lambda: events.append("environment")), sys.exc_info()[1],
                capture_cleanup=cleanup,
            )
    assert caught.value is primary
    assert events == ["product", "environment"]
    assert "secondary cleanup" in capsys.readouterr().err
