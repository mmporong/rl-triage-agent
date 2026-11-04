from __future__ import annotations

from pathlib import Path

import pytest

from rl_triage import investigate_v3 as V3


@pytest.mark.parametrize("command", ["status", "recovery-status", "recovery-post"])
def test_existing_commands_delegate_exactly_to_v2(monkeypatch, command: str) -> None:
    payload = {"opaque": object()}
    sentinel = {"sentinel": object()}
    calls = []
    monkeypatch.setattr(
        V3.v2,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or sentinel,
    )
    result = V3.run(
        command, Path("opaque"), payload, backend_name="backend", fixture=True,
    )
    assert result is sentinel
    assert calls == [((command, Path("opaque"), payload), {
        "backend_name": "backend", "fixture": True,
    })]


def test_post_status_requires_empty_payload_and_registered_backend(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(V3, "published_post_status", lambda path: calls.append(path) or {})
    with pytest.raises(ValueError, match="input must be empty"):
        V3.run("post-status", Path("result"), {"extra": True},
               backend_name=V3.v1.REGISTERED_BACKEND, fixture=False)
    with pytest.raises(ValueError, match="registered read-only backend"):
        V3.run("post-status", Path("result"), {},
               backend_name=V3.v1.REGISTERED_BACKEND, fixture=True)
    with pytest.raises(ValueError, match="registered read-only backend"):
        V3.run("post-status", Path("result"), {}, backend_name="other", fixture=False)
    assert calls == []


def test_post_status_calls_read_only_public_query(monkeypatch) -> None:
    sentinel = {"status": "complete"}
    calls = []
    monkeypatch.setattr(
        V3, "published_post_status", lambda path: calls.append(path) or sentinel,
    )
    result = V3.run(
        "post-status", Path("result"), {},
        backend_name=V3.v1.REGISTERED_BACKEND, fixture=False,
    )
    assert result is sentinel
    assert calls == [Path("result")]
