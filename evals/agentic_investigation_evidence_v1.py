"""수용된 호스트 색인을 조사 행동·요청 이미지에 연결하는 CPU 어댑터."""
from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import agentic_evidence_index_v1 as E
import agentic_investigation_contract_v1 as A
import agentic_investigation_loop_v1 as L

REGISTRATION = "agentic_investigation_evidence_registration_v1"
VISUAL = "agentic_investigation_visual_context_v1"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _session(snapshot: dict, lineage: str) -> dict:
    checked = A.C.V.checked(snapshot)
    return {"binding_digest": A.C.L.digest(checked["binding"]), "source_lineage_id": lineage,
            "input_bindings": deepcopy(checked["inputs"]["packet"]["input_bindings"])}


def _aliases(index: dict, aliases: Any, session: dict) -> dict:
    _require(isinstance(aliases, dict) and bool(aliases), "empty alias registration")
    rows = {row["evidence_id"]: row for row in index["entries"]}
    for alias, value in aliases.items():
        _require(isinstance(alias, str) and re.fullmatch(r"evidence:(?:training|clip)-[0-9]{2,6}", alias)
                 is not None, "nonopaque evidence alias")
        _require(isinstance(value, dict) and set(value) == {"kind", "evidence_id", "frame_ids"},
                 "alias fields")
        _require(value["kind"] in {"training", "frames"} and value["evidence_id"] in rows, "unknown alias source")
        _require(alias.startswith("evidence:training-") == (value["kind"] == "training"), "alias kind/name mismatch")
        row = rows[value["evidence_id"]]
        bindings = session["input_bindings"]
        _require(row["binding"]["source_lineage_id"] == session["source_lineage_id"]
                 and row["binding"]["checkpoint_sha256"] == bindings["case"]["checkpoint_sha256"],
                 "alias case/source lineage mismatch")
        training = {item["role"]: item for item in row["training"]}
        _require(set(training) == {"case", "reference"}, "session training provenance unavailable")
        for role in ("case", "reference"):
            _require(training[role]["checkpoint_sha256"] == bindings[role]["checkpoint_sha256"]
                     and training[role]["telemetry"]["sha256"] == bindings[role]["telemetry_sha256"],
                     "alias checkpoint/telemetry mismatch")
        frames = value["frame_ids"]
        _require(isinstance(frames, list) and all(isinstance(frame, str) for frame in frames)
                 and len(set(frames)) == len(frames) and set(frames) <= set(row["frame_ids"]),
                 "unknown/duplicate registered frame")
        _require(bool(frames) if value["kind"] == "frames" else not frames, "alias kind/frame mismatch")
    return deepcopy(aliases)


def register(index: dict, roots: E.Roots, snapshot: dict, *, expected_index_sha256: str,
             source_lineage_id: str, aliases: dict) -> dict:
    """호스트가 고정할 alias·현재 case/reference·계보 등록 객체를 반환한다."""
    checked = E.verify_index(index, roots, expected_sha256=expected_index_sha256)
    session = _session(snapshot, source_lineage_id)
    return {"schema": REGISTRATION, "index_sha256": expected_index_sha256, "session": session,
            "aliases": _aliases(checked, aliases, session)}


class EvidenceAdapter:
    """한 등록 session의 조회만 완료한다. 승인·검사 실행·다음 epoch 이전은 수행하지 않는다."""

    def __init__(self, index: dict, roots: E.Roots, registration: dict, *,
                 expected_registration_sha256: str):
        _require(E.index_sha256(registration) == expected_registration_sha256
                 and set(registration) == {"schema", "index_sha256", "session", "aliases"}
                 and registration["schema"] == REGISTRATION, "registered alias SHA/schema mismatch")
        self.index, self.roots = deepcopy(index), dict(roots)
        self.registration = deepcopy(registration)
        self.registration_sha256 = expected_registration_sha256
        self._verify()

    def _verify(self, snapshot: dict | None = None) -> None:
        _require(E.index_sha256(self.registration) == self.registration_sha256, "alias registration changed")
        checked = E.verify_index(self.index, self.roots, expected_sha256=self.registration["index_sha256"])
        _aliases(checked, self.registration["aliases"], self.registration["session"])
        if snapshot is not None:
            _require(_session(snapshot, self.registration["session"]["source_lineage_id"])
                     == self.registration["session"], "stale session/epoch/checkpoint")

    def _payload(self, alias: str, cache: dict) -> dict:
        _require(alias in self.registration["aliases"], "unknown evidence alias")
        registered = self.registration["aliases"][alias]
        identity = registered["evidence_id"]
        if identity not in cache:
            view = E.model_view(self.index, identity, self.roots,
                                expected_sha256=self.registration["index_sha256"], view_kind="text_only")
            facts = json.loads(view["messages"][1]["content"][0]["text"])["evidence_facts"]
            spec = next(item for item in self.index["spec"]["entries"] if item["evidence_id"] == identity)
            path = E._ref(spec["contract"], self.roots)
            contract = E.V01.verify(path)
            facts.update(E.SELECT._frame_facts(contract))
            cache[identity] = A.public_facts(facts)
        all_facts = cache[identity]
        if registered["kind"] == "training":
            facts = {key: value for key, value in all_facts.items() if key.startswith("telemetry:")}
        else:
            frames = registered["frame_ids"]
            facts = {key: value for key, value in all_facts.items()
                     if any(key == f"vision:frame:{frame}" or key.startswith(f"vision:play:{frame}:")
                            for frame in frames)}
        return {"facts": facts, "frame_ids": list(registered["frame_ids"])}

    @staticmethod
    def _new(facts: dict, visible: dict) -> dict:
        _require(all(visible[key] == value for key, value in facts.items() if key in visible),
                 "same fact ID has a different payload")
        return {key: deepcopy(value) for key, value in facts.items() if key not in visible}

    def _visual(self, loop: L.Investigation, visual: dict | None, cache: dict) -> tuple[dict, list, list]:
        value = {"schema": VISUAL, "registration_sha256": self.registration_sha256,
                 "session": deepcopy(self.registration["session"]), "records": []} if visual is None else deepcopy(visual)
        _require(isinstance(value, dict) and set(value) == {"schema", "registration_sha256", "session", "records"}
                 and value["schema"] == VISUAL and value["registration_sha256"] == self.registration_sha256
                 and value["session"] == self.registration["session"] and isinstance(value["records"], list),
                 "unverified visual context")
        records = value["records"]
        _require([row["alias"] for row in records] == loop.revealed, "missing/reordered evidence context")
        visible = deepcopy(loop.snapshot["view"]["evidence_facts"])
        extra: dict = {}
        blocks, citations = [], []
        seen_frames: dict[str, dict] = {}
        for row in records:
            _require(set(row) == {"alias", "facts", "payload_sha256", "frame_refs"}, "visual record fields")
            payload = self._payload(row["alias"], cache)
            _require(row["facts"] == self._new(payload["facts"], visible)
                     and row["payload_sha256"] == E.index_sha256(payload), "same alias has a different payload")
            visible.update(row["facts"])
            extra.update(row["facts"])
            frame_refs = []
            for frame in payload["frame_ids"]:
                registered = self.registration["aliases"][row["alias"]]
                frame_view = E.frame_view(self.index, registered["evidence_id"], frame, self.roots,
                                         expected_sha256=self.registration["index_sha256"])
                content = frame_view["messages"][0]["content"]
                label = json.loads(content[0]["text"])
                fact_id = f"vision:frame:{frame}"
                _require(label == {"frame_id": frame, "timestamp_s": visible[fact_id]["timestamp_s"]},
                         "frame label/fact mismatch")
                frame_ref = {"frame_id": frame, "fact_id": fact_id, "timestamp_s": label["timestamp_s"],
                             "content_sha256": E.index_sha256({"content": content})}
                frame_refs.append(frame_ref)
                if frame not in seen_frames:
                    blocks.extend(deepcopy(content))
                    citations.append(frame_ref)
                    seen_frames[frame] = frame_ref
                else:
                    _require(seen_frames[frame] == frame_ref, "same frame ID has a different image")
            _require(row["frame_refs"] == frame_refs, "unverified frame context")
        _require(extra == loop.extra_facts, "loop facts differ from verified evidence")
        events = [row for row in loop.events if row["event"] == "evidence"]
        _require(events == [{"event": "evidence", "id": row["alias"],
                            "facts_digest": A.C.L.digest(row["facts"])} for row in records],
                 "evidence event/payload mismatch")
        return value, blocks, citations

    def _catalogue(self, loop: L.Investigation, cache: dict) -> dict:
        visible = {**loop.snapshot["view"]["evidence_facts"], **loop.extra_facts}
        catalogue: dict = {}
        union = deepcopy(visible)
        for alias, row in self.registration["aliases"].items():
            facts = self._payload(alias, cache)["facts"]
            self._new(facts, union)
            union.update(facts)
            if alias in loop.revealed or self._new(facts, visible):
                catalogue[alias] = row["kind"]
        return catalogue

    def start(self, snapshot: dict, *, goal: str, max_decisions: int = 16,
              allow_fixture: bool = False) -> L.Investigation:
        """새 fact를 줄 수 있는 자료만 광고한 기존 Investigation을 만든다."""
        self._verify(snapshot)
        loop = L.Investigation(snapshot, goal=goal, max_decisions=max_decisions, allow_fixture=allow_fixture)
        loop.evidence = self._catalogue(loop, {})
        return loop

    def _prepared(self, loop: L.Investigation, visual_context: dict | None) -> tuple[dict, dict, list, list]:
        self._verify(loop.snapshot)
        cache: dict = {}
        visual, blocks, citations = self._visual(loop, visual_context, cache)
        _require(loop.pending is None, "pending action has no next context")
        loop.evidence = self._catalogue(loop, cache)
        return loop.context(), visual, blocks, citations

    def prepare(self, loop: L.Investigation, *, visual_context: dict | None = None) -> dict:
        """기존 A.prepare/context를 사용하며 조회 이력·원문 결합을 재검증한다."""
        return self._prepared(loop, visual_context)[0]

    def messages(self, loop: L.Investigation, *, visual_context: dict | None = None) -> dict:
        """A.SYSTEM 한 개와 A.messages의 user text에 조회된 이미지 블록만 붙인다."""
        context, visual, blocks, citations = self._prepared(loop, visual_context)
        messages = A.messages(context)
        messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]}, *blocks]
        hidden = [self.registration["session"]["source_lineage_id"], *[str(path) for path in self.roots.values()]]
        for binding in self.registration["session"]["input_bindings"].values():
            hidden.extend(binding.values())
        E.TRANSFER.check_messages(messages, hidden)
        _require(messages[0] == {"role": "system", "content": A.SYSTEM}, "mixed system prompt")
        return {"schema": "agentic_investigation_evidence_messages_v1", "context": context,
                "messages": messages, "messages_sha256": E.index_sha256({"messages": messages}),
                "frame_refs": citations, "image_count": len(citations), "visual_context": visual}

    def _complete(self, loop: L.Investigation, alias: str, action: str, visual_context: dict | None) -> dict:
        self._verify(loop.snapshot)
        cache: dict = {}
        visual, _, _ = self._visual(loop, visual_context, cache)
        pending = loop.pending
        if pending is None:
            raise ValueError("no pending evidence action")
        context = A.checked(pending["context"])
        review = pending["review"]
        _require(review["status"] == "valid"
                 and A.assess(json.dumps(review["decision"]), context) == review
                 and review["decision"]["action"] == action and review["decision"]["target"] == alias,
                 "pending action/target mismatch")
        _require(A.C.L.digest(context["snapshot"]) == A.C.L.digest(loop.snapshot)
                 and context["extra_facts"] == loop.extra_facts and context["revealed"] == loop.revealed,
                 "stale pending context")
        _require(alias in self.registration["aliases"] and alias not in loop.revealed,
                 "unknown/already revealed alias")
        registered = self.registration["aliases"][alias]
        _require(registered["kind"] == ("frames" if action == "inspect_frames" else "training"),
                 "alias/action kind mismatch")
        payload = self._payload(alias, cache)
        new = self._new(payload["facts"], {**loop.snapshot["view"]["evidence_facts"], **loop.extra_facts})
        _require(bool(new), "evidence has no new facts")
        frame_refs = []
        for frame in payload["frame_ids"]:
            frame_view = E.frame_view(self.index, registered["evidence_id"], frame, self.roots,
                                     expected_sha256=self.registration["index_sha256"])
            content = frame_view["messages"][0]["content"]
            label = json.loads(content[0]["text"])
            _require(label == {"frame_id": frame, "timestamp_s": payload["facts"][f"vision:frame:{frame}"]["timestamp_s"]},
                     "frame label/fact mismatch")
            frame_refs.append({"frame_id": frame, "fact_id": f"vision:frame:{frame}",
                               "timestamp_s": label["timestamp_s"],
                               "content_sha256": E.index_sha256({"content": content})})
        visual["records"].append({"alias": alias, "facts": deepcopy(new),
                                  "payload_sha256": E.index_sha256(payload), "frame_refs": frame_refs})
        loop.complete_evidence(alias, new)
        loop.evidence = self._catalogue(loop, cache)
        return {"schema": "agentic_investigation_evidence_receipt_v1", "alias": alias, "action": action,
                "facts": deepcopy(new), "facts_sha256": E.index_sha256(new), "frame_refs": frame_refs,
                "image_count": len(frame_refs), "visual_context": visual,
                "request_context_digest": context["context_digest"], "approval_granted": False}

    def read_evidence(self, loop: L.Investigation, alias: str, *, visual_context: dict | None = None) -> dict:
        """현재 read_evidence 요청만 기존 complete_evidence로 완료한다."""
        return self._complete(loop, alias, "read_evidence", visual_context)

    def inspect_frames(self, loop: L.Investigation, alias: str, *, visual_context: dict | None = None) -> dict:
        """현재 inspect_frames 요청의 등록 frame 목록만 조회·결합한다."""
        return self._complete(loop, alias, "inspect_frames", visual_context)
