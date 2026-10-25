from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
S = importlib.import_module("vision_replicator_startup_v1")


class Settings:
    def __init__(self, events: list[str], *, reject_orchestrator: bool = False) -> None:
        self.events = events
        self.reject_orchestrator = reject_orchestrator
        self.values: dict[str, bool] = {
            S.ORCHESTRATOR_ENABLED: False,
            S.CAPTURE_ON_PLAY: True,
        }

    def get(self, path: str) -> bool:
        self.events.append(f"get:{path}")
        return self.values[path]

    def set_bool(self, path: str, value: bool) -> None:
        self.events.append(f"set:{path}:{value}")
        if not (path == S.ORCHESTRATOR_ENABLED and self.reject_orchestrator):
            self.values[path] = value


class Orchestrator:
    def __init__(self, settings: Settings, events: list[str], *, reject_capture: bool = False) -> None:
        self.settings = settings
        self.events = events
        self.reject_capture = reject_capture

    def set_capture_on_play(self, value: bool) -> None:
        self.events.append(f"capture:{value}")
        if not self.reject_capture:
            self.settings.values[S.CAPTURE_ON_PLAY] = value


class Replicator:
    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator


def test_configures_and_reads_back_in_required_order() -> None:
    events: list[str] = []
    settings = Settings(events)
    rep = Replicator(Orchestrator(settings, events))

    result = S.configure_replicator_startup(settings=settings, replicator=rep)

    assert result == {
        "before": {"orchestrator_enabled": False, "capture_on_play": True},
        "after": {"orchestrator_enabled": True, "capture_on_play": False},
    }
    assert events == [
        f"get:{S.ORCHESTRATOR_ENABLED}",
        f"get:{S.CAPTURE_ON_PLAY}",
        f"set:{S.ORCHESTRATOR_ENABLED}:True",
        f"get:{S.ORCHESTRATOR_ENABLED}",
        "capture:False",
        f"get:{S.CAPTURE_ON_PLAY}",
    ]


def test_orchestrator_enable_readback_failure_stops_before_replicator() -> None:
    events: list[str] = []
    settings = Settings(events, reject_orchestrator=True)

    with pytest.raises(RuntimeError, match="enabled"):
        S.configure_replicator_startup(settings=settings, replicator=pytest.fail)

    assert not any(event.startswith("capture:") for event in events)


def test_capture_on_play_readback_failure_is_rejected() -> None:
    events: list[str] = []
    settings = Settings(events)
    rep = Replicator(Orchestrator(settings, events, reject_capture=True))

    with pytest.raises(RuntimeError, match="captureOnPlay"):
        S.configure_replicator_startup(settings=settings, replicator=rep)

    assert events[-2:] == ["capture:False", f"get:{S.CAPTURE_ON_PLAY}"]


def test_sdk_loaders_are_lazy_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "carb" not in S.__dict__ and "rep" not in S.__dict__
    events: list[str] = []
    settings = Settings(events)
    rep = Replicator(Orchestrator(settings, events))

    def load_settings() -> Settings:
        events.append("load:settings")
        return settings

    def load_replicator() -> Any:
        events.append("load:replicator")
        return rep

    monkeypatch.setattr(S, "_load_settings", load_settings)
    monkeypatch.setattr(S, "_load_replicator", load_replicator)

    result = S.configure_replicator_startup()

    assert result["after"] == {"orchestrator_enabled": True, "capture_on_play": False}
    assert events.index("load:settings") < events.index("load:replicator")
    assert events.index(f"get:{S.ORCHESTRATOR_ENABLED}", 2) < events.index("load:replicator")
