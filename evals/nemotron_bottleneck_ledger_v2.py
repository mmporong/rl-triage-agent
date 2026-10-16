"""네 병목 비교의 공통 요청 원장. 모든 생성 요청은 전송 전에 예약한다."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

API_CAP = 200
PRIOR_DISCOVERY_REQUESTS = 1  # 생성 비교 전 수행한 /v1/models GET도 전체 상한에 포함한다.
STAGE_CAPS = {"B1": 71, "B2": 52, "B3": 52, "B4": 24}
BASELINE = "nvidia/nemotron-3-super-120b-a12b"
CANDIDATE = "nvidia/nemotron-3.5-super-vl-120b-a12b"
MODELS = (BASELINE, CANDIDATE)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self, root: Path, experiment_binding: dict):
        self.root = Path(root)
        self.binding = experiment_binding

    @contextmanager
    def lock(self):
        lock = self.root / "ledger.lock"
        write_new(lock, {"pid": os.getpid(), "created_at": now()})
        try:
            yield
        finally:
            lock.unlink()

    def audit(self):
        rows = []
        reservations = sorted((self.root / "reservations").glob("*.json"))
        responses = {p.name: p for p in (self.root / "responses").glob("*.json")}
        if len(reservations) + PRIOR_DISCOVERY_REQUESTS > API_CAP:
            raise ValueError("공통 요청 상한을 넘은 원장")
        for index, path in enumerate(reservations, 1):
            row = read(path)
            if path.name != f"{index:04d}.json" or row.get("sequence") != index:
                raise ValueError("요청 원장 순서가 연속되지 않는다")
            if row.get("binding") != self.binding or row.get("stage") not in STAGE_CAPS:
                raise ValueError("요청 원장 결합 또는 단계가 다르다")
            if row.get("previous_sha256") != (digest(rows[-1]["reservation"]) if rows else None):
                raise ValueError("요청 원장 연결이 다르다")
            response_path = responses.pop(path.name, None)
            response = read(response_path) if response_path else None
            if response and (response.get("reservation_sha256") != digest(row)
                             or response.get("sequence") != index):
                raise ValueError("응답이 요청 예약과 결합되지 않는다")
            rows.append({"reservation": row, "response": response})
        if responses:
            raise ValueError("예약 없는 응답이 있다")
        labels = [(r["reservation"]["stage"], r["reservation"]["label"]) for r in rows]
        if len(labels) != len(set(labels)):
            raise ValueError("같은 요청 label이 중복됐다")
        for stage, cap in STAGE_CAPS.items():
            if sum(r["reservation"]["stage"] == stage for r in rows) > cap:
                raise ValueError("단계 요청 상한 초과")
        return rows

    def reserve(self, stage, label, request):
        if stage not in STAGE_CAPS or not isinstance(label, str) or not label:
            raise ValueError("요청 식별자가 잘못됐다")
        if request.get("model") not in MODELS:
            raise ValueError("등록되지 않은 모델")
        with self.lock():
            rows = self.audit()
            if any(r["response"] is None for r in rows):
                raise BudgetStop("완료 기록이 없는 예약이 있어 재전송하지 않는다")
            if any(r["reservation"]["stage"] == stage and r["reservation"]["label"] == label for r in rows):
                raise BudgetStop("이미 예약된 호출은 다시 보내지 않는다")
            if len(rows) + PRIOR_DISCOVERY_REQUESTS >= API_CAP or sum(r["reservation"]["stage"] == stage for r in rows) >= STAGE_CAPS[stage]:
                raise BudgetStop("사전등록 요청 상한 도달")
            row = {"schema": "nemotron_bottleneck_reservation_v2", "sequence": len(rows) + 1,
                   "binding": self.binding, "stage": stage, "label": label,
                   "model": request["model"], "request_sha256": digest(request),
                   "request_without_model_sha256": digest({k: v for k, v in request.items() if k != "model"}),
                   "previous_sha256": digest(rows[-1]["reservation"]) if rows else None,
                   "reserved_at": now()}
            write_new(self.root / "reservations" / f"{row['sequence']:04d}.json", row)
            return row

    def finish(self, reservation, result):
        row = {"schema": "nemotron_bottleneck_response_v2", "sequence": reservation["sequence"],
               "reservation_sha256": digest(reservation), "result": result, "finished_at": now()}
        write_new(self.root / "responses" / f"{reservation['sequence']:04d}.json", row)
        return row

    def call(self, stage, label, request, complete):
        reservation = self.reserve(stage, label, request)
        started = time.perf_counter()
        try:
            result = complete()
        except Exception as error:
            # 예외 본문과 헤더는 저장하지 않는다.
            result = {"error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
                      "infra_error": type(error).__name__ in {"APIConnectionError", "APITimeoutError",
                                                             "ConnectError", "ReadTimeout", "TimeoutError"}}
        result = {**result, "measured_wall_s": time.perf_counter() - started}
        self.finish(reservation, result)
        return result

    async def async_call(self, stage, label, request, complete):
        reservation = self.reserve(stage, label, request)
        started = time.perf_counter()
        try:
            result, value = await complete()
        except Exception as error:
            result = {"error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
                      "infra_error": type(error).__name__ in {"APIConnectionError", "APITimeoutError",
                                                             "ConnectError", "ReadTimeout", "TimeoutError"}}
            result["measured_wall_s"] = time.perf_counter() - started
            self.finish(reservation, result)
            raise
        result = {**result, "measured_wall_s": time.perf_counter() - started}
        self.finish(reservation, result)
        return result, value


def transport_status(raw, model):
    code = raw.get("status_code")
    if raw.get("error_type") or isinstance(code, int) and code >= 400:
        return "infra_error" if code == 429 or isinstance(code, int) and code >= 500 or raw.get("infra_error") else "call_error"
    if raw.get("provider_model") != model:
        return "call_error"
    if raw.get("finish_reason") == "length" or not (raw.get("content") or "").strip():
        return "infra_error"
    return "ok"


def costs(rows):
    results = [r["response"]["result"] for r in rows if r["response"] is not None]
    out = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [r.get(field) for r in results]
        known = [v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
        out[field] = {"known": len(known), "missing": len(values) - len(known), "sum_known": sum(known)}
    walls = [r.get("measured_wall_s") for r in results]
    walls = [v for v in walls if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0]
    out["wall_s"] = {"n": len(walls), "total": sum(walls), "mean": sum(walls) / len(walls) if walls else None}
    return out
