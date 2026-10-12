"""E2 교정 제안·승인·소비·receipt 원장(사양 구성 2). probe_loop.Ledger와 같은 append-only 원칙을 따른다.

- effect key = (loop_state_digest, correction_id, recovery_mode). 진행 중이거나 끝난 같은 키는 다시 제안할 수 없다.
- 승인은 제안에 묶인 loop 상태·교정·회복 방식·protocol case/arm·registry digest와 일치할 때만 한 번 소비된다.
- 원장 파일이 깨지거나 사건 순서가 맞지 않으면 읽는 순간 LedgerError다.
"""
from __future__ import annotations

import json

from rl_triage import probe_loop as L
from rl_triage.correction_registry import HOLD, RECOVERY_MODES, REGISTRY, registry_digest

LedgerError = L.LedgerError
VERDICTS = ("recovered", "not_recovered", "unknown")
# 사건별로 직전 상태(None은 새 요청)와 새 상태.
_FROM = {"proposed": None, "approved": "proposed", "rejected": "proposed", "consumed": "approved",
         "receipt": "consumed"}


class CorrectionLedger(L.Ledger):
    def __init__(self, path, registry: dict | None = None):
        super().__init__(path)
        self._registry = REGISTRY if registry is None else registry
        self.registry_digest = registry_digest(self._registry)

    # 읽기: 손상·순서 역전 검사
    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        evs = []
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                raise LedgerError(f"원장 {n}행이 깨졌다: {e}") from e
            if not isinstance(ev, dict) or "event" not in ev or "at" not in ev:
                raise LedgerError(f"원장 {n}행에 event/at이 없다")
            evs.append(ev)
        self._verify(evs)
        return evs

    @staticmethod
    def _verify(evs: list[dict]) -> None:
        last: dict[str, str] = {}
        prev_at = ""
        for n, ev in enumerate(evs, 1):
            if ev["at"] < prev_at:
                raise LedgerError(f"원장 {n}행의 시각이 앞선 행보다 이르다")
            prev_at = ev["at"]
            kind, rid = ev["event"], ev.get("request_id")
            if kind == "held":
                continue
            if kind not in _FROM or not rid:
                raise LedgerError(f"원장 {n}행: 모르는 사건이거나 request_id가 없다({kind!r})")
            before = last.get(rid)
            if before != _FROM[kind]:
                raise LedgerError(f"원장 {n}행: {rid}의 {kind}가 {before!r} 다음에 올 수 없다(순서 역전 또는 중복)")
            last[rid] = "proposed" if kind == "proposed" else kind

    # 쓰기
    @staticmethod
    def effect_key(loop_state_digest: str, correction_id: str, recovery_mode: str) -> str:
        return L.digest({"loop_state": loop_state_digest, "correction": correction_id,
                         "mode": recovery_mode})[:16]

    def hold(self, loop_state_digest: str, mechanism: str, reason: str) -> dict:
        """지원 밖 확정 메커니즘이나 none_supported 기록. 요청은 만들지 않는다."""
        return self._append({"event": "held", "decision": HOLD, "loop_state_digest": loop_state_digest,
                             "mechanism": mechanism, "reason": reason})

    def propose(self, loop_state_digest: str, confirmed_mechanism: str, receipts_digest: str, correction_id: str,
                recovery_mode: str, protocol_case_arm: dict, budget_gpu_s: int, evidence: dict | None = None) -> dict:
        entry = self._registry.get(confirmed_mechanism)
        if entry is None:
            raise LedgerError(f"등록 밖 메커니즘 {confirmed_mechanism!r}: 제안하지 않고 HOLD")
        if correction_id != entry["correction_id"]:
            raise LedgerError(f"{confirmed_mechanism}의 등록 교정은 {entry['correction_id']}다({correction_id!r})")
        if recovery_mode not in RECOVERY_MODES or recovery_mode not in entry["recovery_modes"]:
            raise LedgerError(f"{confirmed_mechanism}에 등록되지 않은 회복 방식 {recovery_mode!r}")
        if protocol_case_arm != {"case": entry["protocol_case"], "arm": "corrected"}:
            raise LedgerError(f"protocol case/arm이 등록 표와 다르다: {protocol_case_arm!r}")
        if type(budget_gpu_s) is not int or budget_gpu_s < 0:
            raise LedgerError("GPU 예산은 0 이상의 정수여야 한다")
        ek = self.effect_key(loop_state_digest, correction_id, recovery_mode)
        same = [r for r in self.requests().values() if r.get("effect_key") == ek and r["state"] != "rejected"]
        if same:
            raise LedgerError(f"같은 교정 제안이 이미 있다: {[r['request_id'] for r in same]}")
        rid = L.digest({"effect_key": ek, "registry": self.registry_digest, "n": len(self.events())})[:12]
        return self._append({
            "event": "proposed", "request_id": rid, "loop_state_digest": loop_state_digest,
            "confirmed_mechanism": confirmed_mechanism, "receipts_digest": receipts_digest,
            "correction_id": correction_id, "recovery_mode": recovery_mode, "protocol_case_arm": protocol_case_arm,
            "registry_digest": self.registry_digest, "effect_key": ek, "budget_gpu_s": budget_gpu_s,
            "evidence": evidence or {}})

    def _bound(self, request_id: str, state: str) -> dict:
        req = self.requests().get(request_id)
        if req is None or req["state"] != state:
            raise LedgerError(f"{request_id}: {state} 상태의 요청이 아니다({req and req['state']})")
        if req.get("registry_digest") != self.registry_digest:
            raise LedgerError(f"{request_id}: 등록 표가 제안 때와 다르다")
        return req

    def approve(self, request_id: str, approver: str) -> dict:
        self._bound(request_id, "pending")
        return self._append({"event": "approved", "request_id": request_id, "approver": approver})

    def reject(self, request_id: str, approver: str, reason: str) -> dict:
        self._bound(request_id, "pending")
        return self._append({"event": "rejected", "request_id": request_id, "approver": approver, "reason": reason})

    def consume(self, request_id: str, loop_state_digest: str) -> dict:
        """실행 직전 호출. 승인된 그 요청이고 loop 상태가 승인 때와 같아야 하며 한 번만 된다."""
        req = self._bound(request_id, "approved")
        if req["loop_state_digest"] != loop_state_digest:
            raise LedgerError(f"{request_id}: 승인 뒤 loop 상태가 바뀌었다")
        return self._append({"event": "consumed", "request_id": request_id})

    def receipt(self, request_id: str, tag: str, verdict: str, result_sha: str | None, gpu_s: float | None = None,
                exit_code: int | None = 0, recovered_from: str | None = None, reason: str | None = None) -> dict:
        """실행 결과. 비정상 종료·예산 초과는 verdict를 unknown으로 낮춘다."""
        req = self.requests().get(request_id)
        if req is None or req["state"] != "running":
            raise LedgerError(f"{request_id}: 실행 중인 요청이 아니다")
        over = gpu_s is not None and gpu_s > req["budget_gpu_s"]
        if exit_code != 0 or over:
            verdict = "unknown"
        if verdict not in VERDICTS:
            raise LedgerError(f"모르는 verdict {verdict!r}")
        ev = {"event": "receipt", "request_id": request_id, "tag": tag, "verdict": verdict, "result_sha": result_sha,
              "gpu_s": gpu_s, "exit_code": exit_code, "over_budget": over}
        if recovered_from is not None:
            ev["recovered_from"] = recovered_from
        if reason is not None:
            ev["reason"] = reason
        return self._append(ev)

    def recover(self, request_id: str, artifact: dict | None) -> dict:
        """고아 요청을 재실행 없이 닫는다. 산출물({tag, verdict, result_sha, gpu_s, exit_code, source})이 없으면 unknown."""
        if request_id not in self.orphans():
            raise LedgerError(f"{request_id}: 고아 요청이 아니다")
        if artifact is None:
            return self.receipt(request_id, "", "unknown", None, 0.0, None, reason="orphan_without_artifact")
        return self.receipt(request_id, artifact["tag"], artifact["verdict"], artifact.get("result_sha"),
                            artifact.get("gpu_s", 0.0), artifact.get("exit_code", 0),
                            recovered_from=artifact["source"])
