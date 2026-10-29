"""Fresh API10 prefix를 Runtime의 보관 snapshot과 같은 계보로 재생한다."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import agentic_fresh_selection_prefix_v1 as V1

SCHEMA = V1.SCHEMA
verify = V1.verify


@contextmanager
def scope(runtime: Any, item: dict) -> Iterator[None]:
    """첫 요청에서 보관 snapshot을 API10이 본 동등한 live snapshot으로 되돌린다."""
    V1.require(item.get("schema") == SCHEMA, "fresh prefix scope item")
    prepared = V1._ref(runtime, item["prepared"])
    original = runtime.verify_prefix
    original_request_descriptor = vars(runtime.Runtime)["_request"]
    original_request = runtime.Runtime._request

    def request(loop: Any, bridge: Any, visual: dict | None) -> dict:
        expected = prepared["context"]
        if (bridge is None and visual is None and loop.pending is None and not loop.events
                and not loop.extra_facts and not loop.revealed
                and loop.status == "active" and loop.remaining == expected["decisions_left"]
                and runtime._equivalent(loop.snapshot, expected["snapshot"])
                and loop.goal == expected["goal"] and loop.evidence == expected["evidence"]):
            facts = deepcopy(expected["extra_facts"])
            V1.require(runtime.J.digest(facts) == item["initial_host_preloaded_facts_sha256"],
                       "fresh initial facts changed before host preload")
            loop.extra_facts = facts
            current = loop.context()
            current_ledger = Path(current["snapshot"]["inputs"]["ledger_path"])
            expected_ledger = Path(expected["snapshot"]["inputs"]["ledger_path"])
            expected_raw = expected_ledger.read_bytes() if expected_ledger.exists() else b""
            V1.require(current_ledger.is_file() and current_ledger.read_bytes() == expected_raw,
                       "fresh archived/live ledger bytes differ")
            canonical = deepcopy(current)
            canonical["snapshot"]["inputs"]["ledger_path"] = str(expected_ledger)
            canonical["context_digest"] = expected["context_digest"]
            V1.require(canonical == expected, "fresh archived/live context differs beyond ledger path")
            return {"context": deepcopy(expected), "messages": deepcopy(prepared["messages"])}
        return original_request(loop, bridge, visual)

    def bound(prefix_item: dict, request_value: dict, api_root: Path, *, repo: Path):
        if prefix_item.get("schema") == SCHEMA:
            return verify(prefix_item, request_value, api_root, runtime=runtime)
        return original(prefix_item, request_value, api_root, repo=repo)

    runtime.verify_prefix = bound
    runtime.Runtime._request = staticmethod(request)
    try:
        yield
    finally:
        runtime.verify_prefix = original
        runtime.Runtime._request = original_request_descriptor
