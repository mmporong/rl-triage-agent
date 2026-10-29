from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_fresh_selection_prefix_v2 as M


class Loop:
    def __init__(self, archived: dict, expected: dict):
        self.snapshot = archived
        self.expected = expected
        self.pending = None
        self.events: list[dict] = []
        self.extra_facts: dict = {}
        self.revealed: list[str] = []
        self.status = "active"
        self.remaining = 2
        self.goal = "goal"
        self.evidence: dict = {}

    def context(self) -> dict:
        return {"snapshot": self.snapshot, "goal": self.goal, "evidence": self.evidence,
                "revealed": self.revealed, "extra_facts": self.extra_facts,
                "decisions_left": self.remaining}


def test_scope_canonicalizes_only_byte_identical_ledger_path(tmp_path: Path) -> None:
    archived_ledger, live_ledger = tmp_path / "archive.jsonl", tmp_path / "live.jsonl"
    archived_ledger.write_bytes(b"")
    archived = {"binding": "same", "inputs": {"ledger_path": str(archived_ledger)}}
    live = {"binding": "same", "inputs": {"ledger_path": str(live_ledger)}}
    context = {"context_digest": "live-digest", "snapshot": live, "goal": "goal",
               "evidence": {}, "revealed": [], "extra_facts": {"telemetry:x": 1.0},
               "decisions_left": 2}
    prepared = {"context": context, "messages": [{"role": "user", "content": "x"}]}
    item = {"schema": M.SCHEMA, "prepared": {"path": "prepared"},
            "initial_host_preloaded_facts_sha256": "facts"}

    class Runtime:
        @staticmethod
        def _request(*_args):
            return {"old": True}

    runtime = SimpleNamespace(
        Runtime=Runtime,
        verify_prefix=lambda *_args, **_kwargs: ("old", "old"),
        read_ref=lambda _ref: prepared,
        _equivalent=lambda left, right: left["binding"] == right["binding"],
        J=SimpleNamespace(digest=lambda value: "facts" if value == context["extra_facts"] else "bad"),
    )
    loop = Loop(archived, context)
    original_context = loop.context
    loop.context = lambda: {**original_context(), "context_digest": "archive-digest"}
    original = vars(Runtime)["_request"]
    with M.scope(runtime, item):
        request = Runtime._request(loop, None, None)
        assert request == {"context": context, "messages": prepared["messages"]}
        assert loop.snapshot == archived
    assert vars(Runtime)["_request"] is original


def test_scope_does_not_rebind_non_equivalent_snapshot() -> None:
    prepared = {"context": {"snapshot": {"binding": "expected"}, "goal": "goal",
                             "evidence": {}, "revealed": [], "extra_facts": {},
                             "decisions_left": 2}, "messages": []}
    item = {"schema": M.SCHEMA, "prepared": {},
            "initial_host_preloaded_facts_sha256": "facts"}

    class Runtime:
        @staticmethod
        def _request(*_args):
            return {"old": True}

    runtime = SimpleNamespace(Runtime=Runtime, verify_prefix=lambda *_a, **_k: None,
                              read_ref=lambda _ref: prepared,
                              _equivalent=lambda _left, _right: False,
                              J=SimpleNamespace(digest=lambda _value: "facts"))
    loop = Loop({"binding": "other"}, prepared["context"])
    with M.scope(runtime, item):
        assert Runtime._request(loop, None, None) == {"old": True}
        assert loop.snapshot == {"binding": "other"}


def test_scope_rejects_archived_ledger_bytes_that_differ(tmp_path: Path) -> None:
    archived_ledger, live_ledger = tmp_path / "archive.jsonl", tmp_path / "live.jsonl"
    archived_ledger.write_text("archive\n", encoding="utf-8")
    live_ledger.write_text("live\n", encoding="utf-8")
    archived = {"binding": "same", "inputs": {"ledger_path": str(archived_ledger)}}
    live = {"binding": "same", "inputs": {"ledger_path": str(live_ledger)}}
    context = {"context_digest": "live-digest", "snapshot": live, "goal": "goal",
               "evidence": {}, "revealed": [], "extra_facts": {}, "decisions_left": 2}
    prepared = {"context": context, "messages": []}
    item = {"schema": M.SCHEMA, "prepared": {},
            "initial_host_preloaded_facts_sha256": "facts"}

    class Runtime:
        @staticmethod
        def _request(*_args):
            return {"old": True}

    runtime = SimpleNamespace(Runtime=Runtime, verify_prefix=lambda *_a, **_k: None,
                              read_ref=lambda _ref: prepared,
                              _equivalent=lambda _left, _right: True,
                              J=SimpleNamespace(digest=lambda _value: "facts"))
    loop = Loop(archived, context)
    original_context = loop.context
    loop.context = lambda: {**original_context(), "context_digest": "archive-digest"}
    with M.scope(runtime, item):
        try:
            Runtime._request(loop, None, None)
        except ValueError as error:
            assert "ledger bytes differ" in str(error)
        else:
            raise AssertionError("different archived/live ledger bytes accepted")
