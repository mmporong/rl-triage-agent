"""Fresh prefix의 실제 결과 원장을 CALL capsule과 구분하는 Runtime 판."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import agentic_investigation_runtime_v1 as V1

SCHEMA = V1.SCHEMA
J, L, C, CALL = V1.J, V1.L, V1.C, V1.CALL
ROOT = V1.ROOT
HostCallbacks = V1.HostCallbacks
read_ref, verify_api, verify_sources = V1.read_ref, V1.verify_api, V1.verify_sources
same_or_new, completed, ref = V1.same_or_new, V1.completed, V1.ref
require, _adapter, _equivalent = V1.require, V1._adapter, V1._equivalent
verify_prefix, validate_prefix = V1.verify_prefix, V1.validate_prefix


def prefix_source(item: dict[str, Any]) -> tuple[str, dict[str, str]]:
    if item.get("schema") == "fresh_frame_runtime_prefix_v1":
        reference = item.get("actual_result")
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
                "fresh prefix actual_result reference")
        return "fresh_actual_result", reference
    reference = item.get("capsule")
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "CALL prefix capsule reference")
    return "call_capsule", reference


class Runtime(V1.Runtime):
    _request = staticmethod(V1.Runtime._request)

    def advance(self, *, complete: Callable[[list, dict], dict] | None = None,
                host: HostCallbacks | None = None, now_utc: Callable | None = None) -> dict:
        require(read_ref(self.manifest["registration"]) == self.reg
                and read_ref(self.protocol_ref) == self.protocol
                and J._read(self.folder / "manifest.json") == self.manifest
                and J._read(self.folder / "initial.json") == self.initial,
                "immutable runtime binding changed")
        verify_sources(self.protocol, self.anchor, repo=self.repo,
                       protocol_path=Path(self.protocol_ref["path"]))
        verify_api(self.reg["api"])
        bridge = _adapter(self.reg["evidence"])
        initial = self.initial["snapshot"]
        loop = (bridge.start(initial, goal=self.reg["goal"], max_decisions=self.reg["max_decisions"],
                             allow_fixture=self.reg["allow_fixture"]) if bridge else
                L.Investigation(initial, goal=self.reg["goal"], max_decisions=self.reg["max_decisions"],
                                allow_fixture=self.reg["allow_fixture"]))
        visual, live, events = None, self.initial["live_snapshot"], []
        prefix = self.reg["prefix"]
        paths = sorted((self.folder / "steps").glob("*"))
        require([p.name for p in paths] == [f"{i:04d}" for i in range(1, len(paths) + 1)],
                "step order")
        for ordinal in range(1, self.reg["max_decisions"] + 1):
            if loop.status != "active":
                return self._summary(loop, events)
            folder = self.folder / "steps" / f"{ordinal:04d}"
            request = self._request(loop, bridge, visual)
            same_or_new(folder / "request.json", {**request, "live_snapshot": live})
            from_prefix = ordinal <= len(prefix)
            if from_prefix:
                capsule, engine = self._prefix(prefix[ordinal - 1], request)
                source_kind, source_ref = prefix_source(prefix[ordinal - 1])
            else:
                ref_path = folder / "capsule-reference.json"
                if not ref_path.exists():
                    capsule_path = folder / "call/capsule.json"
                    reference = ref(capsule_path) if capsule_path.exists() else CALL.freeze(
                        request["context"], folder / "call", source_anchor=self.anchor,
                        protocol_sha256=self.protocol_ref["sha256"], messages=request["messages"])
                    same_or_new(ref_path, reference)
                source_kind, source_ref = "call_capsule", J._read(ref_path)
                _, capsule = CALL.prepared(source_ref)
                require(capsule["messages"] == request["messages"]
                        and capsule["source_anchor"] == self.anchor
                        and capsule["protocol_sha256"] == self.protocol_ref["sha256"],
                        "runtime capsule binding")
                if (folder / "call/journal/terminal.json").exists():
                    engine = completed(source_ref, self.api)
                else:
                    if complete is None:
                        return self._summary(loop, events, "blocked_model")
                    outcome = CALL.advance(source_ref, self.api, complete, api_cap=12, now_utc=now_utc)
                    if outcome["journal"]["status"] not in {"choice_ready", "terminal_hold"}:
                        return self._summary(loop, events, outcome["journal"]["status"])
                    engine = outcome["journal"]["engine_result"]
            content = engine.get("final_content")
            review = loop.propose(content if isinstance(content, str) else "")
            decision = review.get("decision") or {}
            receipt_facts = sorted(k for k in decision.get("fact_ids", [])
                                   if k.startswith("observation:"))
            observations = loop.snapshot["view"]["observations"]
            followup = None
            if observations:
                latest = observations[-1]
                fact_prefix = f"observation:{len(observations) - 1}:{latest['probe']}:"
                cited = [key for key in receipt_facts if key.startswith(fact_prefix)]
                sealed = next((e["host_judgment"] for e in reversed(events)
                               if (e["host_judgment"] or {}).get("event") == "probe"), None)
                followup = {"seal": {"path": sealed["path"], "sha256": sealed["sha256"]}
                            if sealed else None, "probe": latest["probe"],
                            "outcome": latest["outcome"], "fact_ids_cited": cited,
                            "cited": bool(cited)}
            event = {"ordinal": ordinal, "source_record_kind": source_kind,
                     "source_record": source_ref, "imported_prefix": from_prefix,
                     "model_status": engine.get("model_status", "unavailable"), "review": review,
                     "selector": "model", "receipt_fact_ids_cited": receipt_facts,
                     "latest_probe_followup": followup, "causal_effect_measured": False,
                     "host_judgment": None}
            if loop.pending is None:
                same_or_new(folder / "event.json", event)
                events.append(event)
                return self._summary(loop, events)
            action, target = decision["action"], decision["target"]
            completed_before = (folder / "event.json").exists()
            if action in {"read_evidence", "inspect_frames"}:
                if bridge is None:
                    raise ValueError("evidence adapter required")
                method = bridge.read_evidence if action == "read_evidence" else bridge.inspect_frames
                receipt = method(loop, target, visual_context=visual)
                same_or_new(folder / "evidence.json", receipt)
                visual = receipt["visual_context"]
            else:
                require(action in {"run_probe", "propose_correction"}, "unsupported host action")
                require(_equivalent(live, loop.snapshot), "live execution snapshot mismatch")
                intent = {"action": action, "target": target, "snapshot": live,
                          "review": review, "source_anchor": self.anchor,
                          "protocol": self.protocol_ref, "selector": "model",
                          "approval_granted": False}
                if action == "run_probe":
                    choice = loop.probe_choice()
                    intent.update(choice=choice, request_spec=C.request_spec(loop.snapshot, choice))
                envelope = self._operation(folder, intent, host)
                if envelope["status"] == "blocked_host":
                    return self._summary(loop, events, "blocked_host", envelope)
                if envelope["status"] == "hold":
                    event["host_judgment"] = "hold"
                    same_or_new(folder / "event.json", event)
                    return self._summary(loop, [*events, event], "host_hold")
                result = self._tool(loop, intent, envelope, folder)
                event["host_judgment"] = loop.events[-1]
                live = result["next_snapshot"] if result["next_snapshot"] is not None else live
                if action == "propose_correction" and loop.status == "active":
                    bridge = _adapter(result["evidence"])
                    snapshot, remaining = loop.snapshot, loop.remaining
                    loop = (bridge.start(snapshot, goal=self.reg["goal"], max_decisions=remaining,
                                         allow_fixture=self.reg["allow_fixture"]) if bridge else
                            L.Investigation(snapshot, goal=self.reg["goal"], max_decisions=remaining,
                                            allow_fixture=self.reg["allow_fixture"]))
                    visual = None
                elif action == "propose_correction":
                    require(result["evidence"] is None,
                            "terminal repair cannot register next evidence")
            same_or_new(folder / "event.json", event)
            events.append(event)
            if not completed_before and not from_prefix:
                return self._summary(loop, events)
        return self._summary(loop, events,
                             "decision_budget_exhausted" if loop.status == "active" else None)


@contextmanager
def runtime_scope():
    """기존 소비자가 참조하는 v1 모듈의 Runtime 이름만 새 판으로 좁게 교체한다."""
    original = V1.Runtime
    if original is not Runtime:
        V1.Runtime = Runtime
    try:
        yield
    finally:
        V1.Runtime = original
