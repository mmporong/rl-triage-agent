"""기존 검사/교정 seal을 소비하는 CPU 조사 고리. API·승인·GPU 실행은 호스트 책임이다."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import agentic_investigation_contract_v1 as A

C = A.C


class Investigation:
    """한 고리에서 조회·검사·교정 제안을 순서대로 연결하고 미완료 행동을 재실행하지 않는다."""

    def __init__(self, snapshot: dict, *, goal: str, evidence: dict[str, str] | None = None,
                 max_decisions: int = 16, allow_fixture: bool = False):
        self.snapshot = C.V.checked(snapshot)
        if type(allow_fixture) is not bool:
            raise ValueError("fixture 허용은 명시적 bool이어야 한다")
        self.goal, self.evidence = goal, deepcopy(evidence or {})
        self.remaining = max_decisions
        self.allow_fixture = allow_fixture
        self.revealed: list[str] = []
        self.extra_facts: dict = {}
        self.pending: dict | None = None
        self.status = "active"
        self.events: list[dict] = []
        self.parent_repair: str | None = None
        A.prepare(self.snapshot, goal=goal, evidence=self.evidence, decisions_left=max_decisions)

    def context(self) -> dict:
        if self.status != "active" or self.pending is not None:
            raise ValueError("종료되었거나 미완료 행동이 있는 조사")
        return A.prepare(self.snapshot, goal=self.goal, evidence=self.evidence, revealed=self.revealed,
                         extra_facts=self.extra_facts, decisions_left=self.remaining)

    def propose(self, content: str) -> dict:
        context = self.context()
        if self.remaining == 0:
            self.status = "hold"
            self.events.append({"event": "hold", "reason": "decision_budget_exhausted"})
            return {"status": "hold", "approval_granted": False}
        review = A.assess(content, context)
        self.remaining -= 1
        self.events.append({"event": "decision", "review": deepcopy(review),
                            "content": content, "context_digest": context["context_digest"]})
        if review["status"] != "valid" or review["decision"]["action"] == "hold":
            self.status = "hold"
            return review
        self.pending = {"context": context, "review": review}
        return deepcopy(review)

    def _action(self, expected: str) -> dict:
        if self.pending is None or self.pending["review"]["decision"]["action"] != expected:
            raise ValueError("현재 미완료 행동과 결과 종류가 다르다")
        return self.pending

    def complete_evidence(self, evidence_id: str, facts: dict) -> None:
        if self.pending is None:
            raise ValueError("조회 요청이 없다")
        decision = self.pending["review"]["decision"]
        if decision["action"] not in {"read_evidence", "inspect_frames"} or decision["target"] != evidence_id:
            raise ValueError("조회 자료 ID와 요청이 다르다")
        safe = A.public_facts(facts)
        if not safe or set(safe) & (set(self.extra_facts) | set(self.snapshot["view"]["evidence_facts"])):
            raise ValueError("빈 조회 결과 또는 기존 fact 덮어쓰기")
        if decision["action"] == "inspect_frames" and not any(k.startswith("vision:frame:") for k in safe):
            raise ValueError("원본 frame 근거가 없는 영상 조회")
        self.extra_facts.update(safe)
        self.revealed.append(evidence_id)
        self.events.append({"event": "evidence", "id": evidence_id, "facts_digest": C.L.digest(safe)})
        self.pending = None

    def probe_choice(self) -> dict:
        pending = self._action("run_probe")
        return A.probe_choice(pending["context"], pending["review"])

    def complete_probe(self, seal_ref: dict, next_snapshot: dict) -> None:
        pending = self._action("run_probe")
        path = Path(seal_ref["path"])
        if C.digest_file(path) != seal_ref["sha256"]:
            raise ValueError("검사 seal SHA 불일치")
        raw = C.read(path)
        if not self.allow_fixture and raw.get("evidence_kind") != C.ACTUAL_EVIDENCE:
            raise ValueError("운영 조사에 CPU fixture를 사용할 수 없다")
        receipt = C.verify_probe(path, pending["context"]["snapshot"])
        updated = C.V.checked(next_snapshot)
        old = pending["context"]["snapshot"]
        expected = [*old["view"]["observations"],
                    {"probe": pending["review"]["decision"]["target"], "outcome": receipt["outcome"]}]
        if (updated["binding"] != old["binding"] or updated["inputs"] != old["inputs"]
                or updated["view"]["observations"] != expected):
            raise ValueError("다른 계보·중복·누락 검사 결과")
        self.snapshot, self.pending = updated, None
        self.events.append({"event": "probe", "path": str(path), "sha256": seal_ref["sha256"],
                            "probe": receipt["probe"], "outcome": receipt["outcome"]})
        if receipt["outcome"] == "unknown":
            self.status = "hold"

    def complete_repair(self, path: Path, *, ledger_path: Path, bands: dict,
                        next_snapshot: dict | None = None) -> None:
        pending = self._action("propose_correction")
        raw = C.read(path)
        if not self.allow_fixture and raw.get("authorization", {}).get("evidence_kind") != C.ACTUAL_EVIDENCE:
            raise ValueError("운영 조사에 CPU 교정 fixture를 사용할 수 없다")
        result = C.verify_repair(path, ledger_path=ledger_path, bands=bands)
        snapshot = pending["context"]["snapshot"]
        packet = snapshot["inputs"]["packet"]
        ledger = C.D1.CorrectionLedger(ledger_path, C.RC.correction_registry())
        request = ledger.requests()[raw["authorization"]["request_id"]]
        if (request["loop_state_digest"] != C.L.digest(snapshot)
                or result["applied_after"] != [*packet["applied_corrections"],
                                              pending["review"]["decision"]["target"]]
                or raw["input_artifact"]["checkpoint"]["sha256"] != packet["input_bindings"]["case"]["checkpoint_sha256"]
                or raw["reference_artifact"]["checkpoint"]["sha256"] != packet["input_bindings"]["reference"]["checkpoint_sha256"]):
            raise ValueError("교정 요청·원본·참조 checkpoint 결합 불일치")
        can_continue = C.next_epoch(path, ledger_path=ledger_path, bands=bands)
        if can_continue:
            if next_snapshot is None:
                raise ValueError("비회복 뒤 검증할 다음 epoch snapshot이 필요하다")
            updated = C.V.checked(next_snapshot)
            parent = updated["binding"]["parent"]
            next_packet = updated["inputs"]["packet"]
            if (next_packet["epoch"] != packet["epoch"] + 1
                    or updated["view"]["observations"]
                    or next_packet["applied_corrections"] != result["applied_after"]
                    or next_packet["input_bindings"]["case"]["checkpoint_sha256"]
                    != raw["output_artifact"]["checkpoint"]["sha256"]
                    or next_packet["input_bindings"]["reference"] != packet["input_bindings"]["reference"]
                    or parent["result_sha256"] != result["result_sha256"]
                    or updated["inputs"]["policy"] != snapshot["inputs"]["policy"]
                    or updated["inputs"]["run_id"] != snapshot["inputs"]["run_id"]):
                raise ValueError("다른 교정·정책·출력 checkpoint의 재진단")
            self.snapshot = updated
            self.extra_facts, self.revealed = {}, []
            self.evidence = {}  # 이전 checkpoint의 영상/자료 조회를 새 epoch에 재사용하지 않는다.
        else:
            if next_snapshot is not None:
                raise ValueError("회복 또는 최대 교정 뒤 불필요한 다음 고리")
            self.status = result["outcome"]
        self.parent_repair = str(path)
        self.pending = None
        self.events.append({"event": "repair", **result})

    def audit(self) -> dict:
        """CPU 고리 기록이며 실제 모델 호출/학습 효과의 증거로 쓰지 않는다."""
        return json.loads(json.dumps({"schema": "agentic_investigation_audit_v1",
            "status": self.status, "decisions_left": self.remaining, "pending": self.pending,
            "allow_fixture": self.allow_fixture, "events": self.events,
            "api_calls_made_by_this_module": 0, "gpu_jobs_started_by_this_module": 0,
            "provider_usage_not_collected": True}, ensure_ascii=False, allow_nan=False))
