from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
V1 = importlib.import_module("vision_play_worker_v1")


class Annotator:
    def __init__(self, name: str, events: list[str], *, fail: bool = False) -> None:
        self.name, self.events, self.fail = name, events, fail

    def detach(self, render_product: object) -> None:
        del render_product
        self.events.append(f"detach:{self.name}")
        if self.fail:
            raise RuntimeError(f"detach failed:{self.name}")


class RenderProduct:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def destroy(self) -> None:
        self.events.append("destroy")


class Env:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("env-close")


def test_detach_failure_still_destroys_render_product_and_closes_env() -> None:
    events: list[str] = []
    annotators = [
        Annotator("first", events, fail=True),
        Annotator("second", events),
    ]
    with pytest.raises(RuntimeError, match="detach failed:first"):
        V1._cleanup_capture_resources(
            None, RenderProduct(events), annotators, Env(events), None
        )
    assert events == ["detach:first", "detach:second", "destroy", "env-close"]


def test_primary_capture_error_is_not_replaced_by_cleanup_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    primary = RuntimeError("primary capture failure")
    V1._cleanup_capture_resources(
        None,
        RenderProduct(events),
        [Annotator("first", events, fail=True)],
        Env(events),
        primary,
    )
    assert events == ["detach:first", "destroy", "env-close"]
    assert "detach failed:first" in capsys.readouterr().err
