from __future__ import annotations

from datetime import datetime, timezone

import pytest


class _FixedFixtureDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 10, 27, 12, tzinfo=timezone.utc)
        return fixed.replace(tzinfo=None) if tz is None else fixed.astimezone(tz)


@pytest.fixture(autouse=True)
def _freeze_transferred_reuse_fixture_clock(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.path.name != "test_run_vision_transferred_reuse_v1.py":
        return

    import run_vision_transferred_reuse_v1 as reuse

    validator = reuse.JOB3.PREVIOUS.PREVIOUS
    if validator.__name__ != "vision_transferred_pull_job_v1":
        raise RuntimeError("transferred reuse validator chain이 달라졌다")
    monkeypatch.setattr(reuse, "datetime", _FixedFixtureDateTime)
    monkeypatch.setattr(validator, "datetime", _FixedFixtureDateTime)
