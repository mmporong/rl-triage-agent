"""P1-A 실험 선택 고리의 핵심(docs/P1-A-LOOP.md). 표준 라이브러리만 쓰고 Isaac·모델을 부르지 않는다.

- PROBES: 공개 probe 목록. 범주마다 예상 결과(abnormal/normal/None=예상 없음)와 비용, canonical args.
- select_probe: 남은 가설들이 서로 다른 결과를 예측하는 probe 중 비용이 가장 작은 것(휴리스틱, 확률 없음).
- Ledger: 사전등록·승인·소비·receipt를 append-only JSONL로 남긴다. 승인은 사전등록 digest·probe·args에 묶이고
  한 번만 소비된다. 원장 파일만으로 상태를 다시 만든다(재시작).
- update: 관측이 예상과 반대인 가설을 기각한다. 불명(unknown) 관측은 아무것도 바꾸지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

MECHANISMS = ("reward", "actuator", "exploration", "optimizer", "physics", "termination")
OUTCOMES = ("abnormal", "normal", "unknown")

# 예상은 "그 범주가 원인이면 이 probe가 어떻게 나오나". None은 예상하지 않음(가르는 데 쓰지 않는다).
PROBES = {
    "P_noise": {"measures": "학습 경로 act() 샘플의 실제 표준편차 / 기록된 action_std", "cost_gpu_s": 60,
                "args": {"rollout_steps": 200, "num_envs": 256},
                "expect": {"exploration": "abnormal", "reward": "normal", "actuator": "normal",
                           "optimizer": "normal", "physics": "normal", "termination": "normal"}},
    "P_value": {"measures": "롤아웃 수익에 대한 critic 설명 분산", "cost_gpu_s": 90,
                "args": {"rollout_steps": 1000, "num_envs": 256},
                "expect": {"optimizer": "abnormal", "reward": "normal", "actuator": None,
                           "exploration": None, "physics": None, "termination": None}},
    "P_reward": {"measures": "상태에서 문서 정의대로 다시 계산한 추종 보상 항(track_lin_vel_xy_exp, track_ang_vel_z_exp)과 "
                             "보상 관리자 기록값의 상대 오차", "cost_gpu_s": 60,
                 "args": {"rollout_steps": 200, "num_envs": 256},
                 "expect": {"reward": "abnormal", "actuator": "normal", "exploration": "normal",
                            "optimizer": "normal", "physics": "normal", "termination": "normal"}},
    "P_torque": {"measures": "낮은 관절 속도에서 토크 포화 비율", "cost_gpu_s": 60,
                 "args": {"rollout_steps": 500, "num_envs": 256},
                 "expect": {"actuator": "abnormal", "reward": "normal", "exploration": "normal",
                            "optimizer": "normal", "physics": "normal", "termination": "normal"}},
    "P_slip": {"measures": "접지 중 발 수평 속도", "cost_gpu_s": 60,
               "args": {"rollout_steps": 500, "num_envs": 256},
               "expect": {"physics": "abnormal", "reward": "normal", "actuator": "normal",
                          "exploration": "normal", "optimizer": "normal", "termination": "normal"}},
    "P_episode": {"measures": "시간 제한 종료 스텝 / (episode_length_s / step_dt)", "cost_gpu_s": 60,
                  "args": {"rollout_steps": 1200, "num_envs": 64},
                  "expect": {"termination": "abnormal", "reward": "normal", "actuator": "normal",
                             "exploration": "normal", "optimizer": "normal", "physics": "normal"}},
}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def discriminates(probe: str, hypotheses: list[str]) -> bool:
    preds = {PROBES[probe]["expect"].get(h) for h in hypotheses}
    return {"abnormal", "normal"} <= preds


def select_probe(hypotheses: list[str], done: list[str] = ()) -> str | None:
    """남은 가설 중 둘 이상이 서로 다른 결과를 예측하는 probe를 비용·이름 순으로 고른다. 없으면 None."""
    cands = [p for p in PROBES if p not in done and discriminates(p, hypotheses)]
    return min(cands, key=lambda p: (PROBES[p]["cost_gpu_s"], p)) if cands else None


def next_probe(hypotheses: list[str], observed: dict[str, str]) -> str | None:
    """고리가 다음에 요청할 probe. 둘 이상 남으면 가르는 probe, 하나 남으면 아직 안 본 확정 probe."""
    if len(hypotheses) == 1:
        p = confirming_probe(hypotheses[0])
        return p if p is not None and p not in observed else None
    return select_probe(hypotheses, list(observed))


def confirming_probe(mechanism: str) -> str | None:
    """그 범주만 abnormal을 예상하는 probe(확정에 쓴다)."""
    for p, spec in PROBES.items():
        if spec["expect"].get(mechanism) == "abnormal" and all(
                v != "abnormal" for m, v in spec["expect"].items() if m != mechanism):
            return p
    return None


def update(hypotheses: list[str], probe: str, outcome: str) -> tuple[list[str], list[str]]:
    """(남은 가설, 기각된 가설). 예상과 반대 관측이면 기각, 불명이나 예상 없음이면 유지."""
    if outcome not in OUTCOMES:
        raise ValueError(f"모르는 관측 {outcome!r}")
    if outcome == "unknown":
        return list(hypotheses), []
    keep, dropped = [], []
    for h in hypotheses:
        pred = PROBES[probe]["expect"].get(h)
        (dropped if pred is not None and pred != outcome else keep).append(h)
    return keep, dropped


def status(hypotheses: list[str], observed: dict[str, str], budget_left: int) -> str:
    """confirmed / none_supported / unidentifiable / open."""
    if not hypotheses:
        return "none_supported"
    if len(hypotheses) == 1:
        p = confirming_probe(hypotheses[0])
        if p is not None and observed.get(p) == "abnormal":
            return "confirmed"
    if budget_left <= 0:
        return "unidentifiable"
    if len(hypotheses) > 1 and select_probe(hypotheses, list(observed)) is None:
        return "unidentifiable"
    return "open"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LedgerError(Exception):
    """원장 규칙 위반(승인 없는 실행, 중복 소비, digest 불일치 등)."""


class Ledger:
    """append-only JSONL 원장. 상태는 언제나 파일에서 다시 만든다."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append(self, ev: dict) -> dict:
        """잠금 파일(O_EXCL)을 잡은 동안만 한 줄을 쓴다. 승인 CLI와 실행기가 같은 원장에 동시에 써도 줄이 섞이지 않는다.
        상태 검사와 쓰기 사이의 경쟁은 막지 않으므로, 상태를 바꾸는 호출(approve·consume)은 한 프로세스에서 한다."""
        ev = {"at": _now(), **ev}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        for _ in range(200):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                time.sleep(0.05)
        else:
            raise LedgerError(f"원장 잠금을 10초 안에 얻지 못했다: {lock.name}")
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        finally:
            os.close(fd)
            os.unlink(lock)
        return ev

    def requests(self) -> dict[str, dict]:
        state: dict[str, dict] = {}
        for ev in self.events():
            rid = ev.get("request_id")
            if ev["event"] == "proposed":
                state[rid] = {**ev, "state": "pending"}
            elif rid in state:
                state[rid]["state"] = {"approved": "approved", "rejected": "rejected", "consumed": "running",
                                       "receipt": "done"}[ev["event"]]
                if ev["event"] == "receipt":
                    state[rid]["receipt"] = ev
        return state

    def propose(self, prereg: dict, probe: str, budget_gpu_s: int) -> dict:
        if probe not in PROBES:
            raise LedgerError(f"모르는 probe {probe!r}")
        args = PROBES[probe]["args"]
        rid = digest({"prereg": digest(prereg), "probe": probe, "args": args, "n": len(self.events())})[:12]
        return self._append({"event": "proposed", "request_id": rid, "prereg_digest": digest(prereg),
                             "probe": probe, "args_digest": digest(args), "budget_gpu_s": budget_gpu_s})

    def approve(self, request_id: str, approver: str) -> dict:
        req = self.requests().get(request_id)
        if req is None or req["state"] != "pending":
            raise LedgerError(f"{request_id}: 승인할 수 있는 대기 요청이 아니다({req and req['state']})")
        return self._append({"event": "approved", "request_id": request_id, "approver": approver})

    def reject(self, request_id: str, approver: str, reason: str) -> dict:
        req = self.requests().get(request_id)
        if req is None or req["state"] != "pending":
            raise LedgerError(f"{request_id}: 거절할 수 있는 대기 요청이 아니다")
        return self._append({"event": "rejected", "request_id": request_id, "approver": approver, "reason": reason})

    def consume(self, request_id: str, prereg: dict, probe: str) -> dict:
        """실행 직전 호출. 승인된 그 요청·그 사전등록·그 probe·그 args만, 한 번만 실행할 수 있다."""
        req = self.requests().get(request_id)
        if req is None or req["state"] != "approved":
            raise LedgerError(f"{request_id}: 승인되지 않았거나 이미 소비됐다")
        if req["prereg_digest"] != digest(prereg) or req["probe"] != probe or req["args_digest"] != digest(PROBES[probe]["args"]):
            raise LedgerError(f"{request_id}: 승인된 사전등록·probe·args와 다르다")
        return self._append({"event": "consumed", "request_id": request_id})

    def receipt(self, request_id: str, outcome: str, measurement: dict, gpu_s: float, exit_code: int | None) -> dict:
        """실행 결과. timeout·비정상 종료는 실패로 단정하지 않고 unknown으로 남긴다."""
        req = self.requests().get(request_id)
        if req is None or req["state"] != "running":
            raise LedgerError(f"{request_id}: 실행 중인 요청이 아니다")
        if exit_code != 0:
            outcome = "unknown"
        if outcome not in OUTCOMES:
            raise LedgerError(f"모르는 관측 {outcome!r}")
        return self._append({"event": "receipt", "request_id": request_id, "outcome": outcome,
                             "measurement": measurement, "gpu_s": gpu_s, "exit_code": exit_code})
