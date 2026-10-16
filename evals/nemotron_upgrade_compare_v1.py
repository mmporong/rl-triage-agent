"""Nemotron 3 Super와 Nemotron 3.5 Super VL의 N2 계약 유효성 개발 비교.

이미 본 N2 packet 11개를 모델마다 3회 단일 호출한다. 공개 계약의 메시지·검증기를
그대로 쓰며, 검증 피드백 재요청은 하지 않는다. API 실행 전 요청 예약을 새 파일로
남겨 중단된 호출을 다시 보내지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import feedback_model_call as F  # noqa: E402
import n2_pass_k as N2  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/nemotron_upgrade_compare_v1.json"
PRIVATE_BASE = ROOT / "bench/private/codex_support_20261015/task18/runner"
BASELINE = "nvidia/nemotron-3-super-120b-a12b"
CANDIDATE = "nvidia/nemotron-3.5-super-vl-120b-a12b"
MODELS = (BASELINE, CANDIDATE)
MODEL_SLUG = {BASELINE: "nemotron3", CANDIDATE: "nemotron35"}
RUNS = 3
BASE_CALLS = 66
API_CAP = 80
MAX_ATTEMPTS = 2
REGISTERED_TAG = "nemotron_upgrade_svl_20261016_v1"
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
REQUEST_COMMON = {"temperature": 0, "max_tokens": 16384, "stream": False,
                  "extra_body": {"reasoning_effort": "none"}}
SOURCE_FILES = {
    "evals/nemotron_upgrade_compare_v1.py",
    "tests/test_nemotron_upgrade_compare_v1.py",
    "evals/feedback_model_call.py",
    "evals/n2_pass_k.py",
    "evals/correction_model_call.py",
    "src/rl_triage/active_diagnosis.py",
    "src/rl_triage/probe_loop.py",
    "src/rl_triage/correction_registry.py",
    "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}


def lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_raw(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def sha_lf(path: Path) -> str:
    return sha_bytes(lf_bytes(Path(path).read_bytes()))


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def packet_rows() -> list[dict[str, Any]]:
    rows = []
    for set_name in N2.SET_ORDER:
        for kind in N2.INPUT_SETS[set_name]["kinds"]:
            path = N2.packet_path(set_name, kind)
            data = path.read_bytes()
            _messages, binding = F.first_messages(kind, data)
            rows.append({
                "packet_id": f"{set_name}_{kind}",
                "set_name": set_name,
                "kind": kind,
                "raw_sha256": sha_bytes(data),
                "messages_sha256": binding["messages_sha256"],
            })
    if len(rows) != 11 or Counter(row["kind"] for row in rows) != {"diagnosis": 6, "correction": 5}:
        raise ValueError("N2 packet 구성은 진단 6개·교정 5개여야 한다")
    return rows


def call_order() -> list[dict[str, Any]]:
    rows = []
    packets = packet_rows()
    for run in range(1, RUNS + 1):
        for packet_index, packet in enumerate(packets):
            first = (run + packet_index) % 2
            for model in (MODELS[first], MODELS[1 - first]):
                rows.append({"call": len(rows) + 1, "run": run, "packet_index": packet_index,
                             "model": model, **packet})
    if len(rows) != BASE_CALLS:
        raise ValueError("기본 호출 수가 66이 아니다")
    return rows


def protocol_dynamic() -> dict[str, Any]:
    return {
        "source_files": {name: sha_lf(ROOT / name) for name in sorted(SOURCE_FILES)},
        "inputs": [{key: row[key] for key in ("packet_id", "kind", "raw_sha256", "messages_sha256")}
                   for row in packet_rows()],
        "call_order_sha256": canonical_sha(call_order()),
    }


FIXED = {
    "schema": "nemotron_upgrade_compare_protocol_v1",
    "question": "새 Nemotron 후보가 기존 모델과 같은 N2 단일 호출 계약에서 사용할 조건을 충족하는가",
    "scope": "already_seen_n2_packets_development_measurement_not_accuracy_or_generalization",
    "official_model_evidence": {
        "candidate": CANDIDATE,
        "ga_date": "2026-10-15",
        "model_card": "https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b/modelcard",
        "hosted_api": "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer",
        "documented_parameters": {"reasoning_effort": ["none", "low", "high"],
                                    "reasoning_effort_none": "reasoning token disabled",
                                    "default_max_tokens": 16384, "maximum_max_tokens": 131072},
    },
    "models": {"baseline": BASELINE, "candidate": CANDIDATE},
    "request_common": REQUEST_COMMON,
    "runs_per_packet": RUNS,
    "registered_tag": REGISTERED_TAG,
    "base_calls": BASE_CALLS,
    "api_cap": API_CAP,
    "concurrency": 1,
    "order": "run 1..3 outer, fixed packet order inner, model order alternates by (run + packet_index) % 2",
    "retry_rule": "empty, length, 429, 5xx, connection, timeout only; same run at most once; contract invalid is not retried",
    "unsupported_rule": "unsupported 4xx stops remaining calls and produces incomplete_inconclusive; parameters are not changed",
    "reservation_rule": "write-once before API; any reservation without a completed response is an orphan and is never called again",
    "status_definitions": {
        "valid": "feedback_model_call.validate passes",
        "invalid": "contract validation rejects the response",
        "infra_error": "retryable transport/status/empty/length failure",
        "call_error": "non-retryable API call error",
    },
    "metrics": ["packet/model status counts", "decision distribution and pass^1, pass^2, pass^3",
                "diagnosis/correction valid counts and type mean pass^3", "cross-model decision agreement",
                "input/output/total tokens and wall seconds over all attempts"],
    "adoption_rule": {
        "complete": "all 66 base calls resolved, unresolved infra_error and call_error are zero",
        "candidate_valid": "candidate overall valid >= 32/33",
        "by_kind_valid": "candidate diagnosis and correction valid counts are each >= baseline",
        "diagnosis_continue": "candidate valid continue count >= baseline valid continue count on the same packets",
        "by_kind_pass3": "candidate diagnosis and correction mean pass^3 are each >= baseline and >= 0.8",
        "correction_rule_disagreement": "candidate correction disagreements <= baseline",
        "pass": "conditional_future_demo_candidate; existing defaults and prior protocols/results remain unchanged",
        "fail": "retain_current",
        "incomplete": "incomplete_inconclusive",
    },
    "wording": {
        "candidate": "새 모델은 사전등록 조건을 충족해 향후 시연 결정 단계의 조건부 후보로 기록한다. 기존 기본 모델은 이 측정으로 바꾸지 않는다.",
        "retain": "새 모델은 사전등록 조건을 모두 충족하지 않아 현재 모델을 유지한다.",
        "incomplete": "호출 오류·인프라 오류·미완료 예약이 남아 모델 선택을 판정하지 않는다.",
        "common": "이미 본 N2 packet의 개발 측정이며 정확도·일반화 근거가 아니다. 비용은 기록만 보고하며 우월성·절감 주장을 하지 않는다.",
    },
    "design_timing": "N2 packet과 기존 모델 결과를 본 뒤 만든 비교이며, 같은 입력과 공개 계약으로 새 모델의 조건부 사용 여부만 판정한다.",
}


def verify_protocol(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = read(PROTOCOL) if protocol is None else protocol
    expected = {**FIXED, **protocol_dynamic()}
    if protocol != expected:
        raise ValueError("프로토콜 고정값·소스·입력 결합이 현재 값과 다르다")
    if any("bench/private" in json.dumps(row) or re.search(r"[A-Za-z]:[/\\]", json.dumps(row))
           for row in protocol["inputs"]):
        raise ValueError("공개 프로토콜 입력에 비공개·절대 경로가 있다")
    return protocol


def prepare() -> dict[str, Any]:
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    protocol = {**FIXED, **protocol_dynamic()}
    write_new(PROTOCOL, protocol)
    verify_protocol(protocol)
    return {"status": "prepared", "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
            "packets": 11, "base_calls": BASE_CALLS, "api_cap": API_CAP}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def verify_commit_binding(commit: str) -> str:
    protocol = verify_protocol()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit은 40자리 SHA여야 한다")
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", commit, head, check=False).returncode != 0:
        raise ValueError("실험 commit이 현재 HEAD의 조상이 아니다")
    bound = sorted(SOURCE_FILES | {PROTOCOL.relative_to(ROOT).as_posix()})
    for relative in bound:
        if git("ls-files", "--error-unmatch", "--", relative, check=False).returncode != 0:
            raise ValueError(f"결합 파일이 추적되지 않는다: {relative}")
        if git("diff", "--quiet", "--", relative, check=False).returncode != 0:
            raise ValueError(f"결합 파일이 깨끗하지 않다: {relative}")
        blob = git("show", f"{commit}:{relative}", check=False)
        if blob.returncode != 0 or sha_bytes(lf_bytes(blob.stdout.encode("utf-8"))) != sha_lf(ROOT / relative):
            raise ValueError(f"실험 commit의 blob과 현재 파일이 다르다: {relative}")
        if relative in protocol["source_files"] and protocol["source_files"][relative] != sha_lf(ROOT / relative):
            raise ValueError(f"프로토콜 소스 SHA가 다르다: {relative}")
    return head


def run_dir(tag: str) -> Path:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("tag는 영문 소문자·숫자·밑줄·하이픈만 쓴다")
    if tag != REGISTERED_TAG:
        raise ValueError("이 프로토콜의 단일 등록 tag와 다르다")
    return PRIVATE_BASE / tag


def plan_path(tag: str) -> Path:
    return run_dir(tag) / "plan.json"


def reservation_path(tag: str, entry: dict[str, Any], attempt: int) -> Path:
    return run_dir(tag) / "reservations" / f"c{entry['call']:02d}_{MODEL_SLUG[entry['model']]}_a{attempt}.json"


def response_path(tag: str, entry: dict[str, Any], attempt: int) -> Path:
    return run_dir(tag) / "responses" / f"c{entry['call']:02d}_{MODEL_SLUG[entry['model']]}_a{attempt}.json"


def lock_path(tag: str) -> Path:
    return run_dir(tag) / "call.lock"


def load_plan(tag: str) -> dict[str, Any]:
    plan = read(plan_path(tag))
    fields = {"schema", "tag", "experiment_commit", "head_at_plan", "protocol_sha256_lf",
              "call_order_sha256", "created_at"}
    if set(plan) != fields or plan.get("schema") != "nemotron_upgrade_compare_plan_v1" or plan.get("tag") != tag:
        raise ValueError("실행 계획 형식이 다르다")
    if plan.get("protocol_sha256_lf") != sha_lf(PROTOCOL):
        raise ValueError("실행 계획의 프로토콜 결합이 다르다")
    if plan.get("call_order_sha256") != canonical_sha(call_order()):
        raise ValueError("실행 계획의 호출 순서 결합이 다르다")
    if (not re.fullmatch(r"[0-9a-f]{40}", plan.get("experiment_commit", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", plan.get("head_at_plan", ""))
            or not isinstance(plan.get("created_at"), str)):
        raise ValueError("실행 계획의 SHA·시각 형식이 다르다")
    verify_commit_binding(plan["experiment_commit"])
    return plan


def make_plan(commit: str, tag: str) -> dict[str, Any]:
    if tag != REGISTERED_TAG:
        raise ValueError("이 프로토콜의 단일 등록 tag와 다르다")
    if plan_path(tag).exists():
        raise FileExistsError(plan_path(tag))
    head = verify_commit_binding(commit)
    plan = {"schema": "nemotron_upgrade_compare_plan_v1", "tag": tag,
            "experiment_commit": commit, "head_at_plan": head,
            "protocol_sha256_lf": sha_lf(PROTOCOL), "call_order_sha256": canonical_sha(call_order()),
            "created_at": datetime.now(timezone.utc).isoformat()}
    write_new(plan_path(tag), plan)
    return {**plan, **remaining(tag)}


def verify_inputs() -> None:
    protocol = verify_protocol()
    registered = {row["packet_id"]: row for row in protocol["inputs"]}
    for row in packet_rows():
        if registered[row["packet_id"]] != {key: row[key] for key in registered[row["packet_id"]]}:
            raise ValueError(f"packet 또는 메시지 SHA가 바뀌었다: {row['packet_id']}")


def completed_response(tag: str, entry: dict[str, Any], attempt: int) -> dict[str, Any] | None:
    reservation, response = reservation_path(tag, entry, attempt), response_path(tag, entry, attempt)
    if not reservation.exists():
        if response.exists():
            raise ValueError("예약 없는 응답이 있다")
        return None
    if not response.exists():
        return {"status": "orphaned_reservation"}
    reserved, row = read(reservation), read(response)
    plan = read(plan_path(tag))
    if (plan.get("schema") != "nemotron_upgrade_compare_plan_v1" or plan.get("tag") != tag
            or plan.get("protocol_sha256_lf") != sha_lf(PROTOCOL)):
        raise ValueError("실행 계획 결합이 다르다")
    packet = next(item for item in packet_rows() if item["packet_id"] == entry["packet_id"])
    expected_reservation = {
        "schema": "nemotron_upgrade_compare_reservation_v1", "tag": tag,
        "experiment_commit": plan["experiment_commit"], "entry": entry, "attempt": attempt,
        "packet_sha256": packet["raw_sha256"], "messages_sha256": packet["messages_sha256"],
        "request": F.request_params(entry["model"]),
    }
    if any(reserved.get(key) != value for key, value in expected_reservation.items()):
        raise ValueError("예약 내용이 현재 계획·입력·요청과 다르다")
    if set(reserved) != set(expected_reservation) | {"reserved_at"}:
        raise ValueError("예약 필드가 고정 형식과 다르다")
    if (row.get("schema") != "nemotron_upgrade_compare_response_v1"
            or row.get("reservation_sha256") != sha_raw(reservation)
            or row.get("entry") != entry or row.get("attempt") != attempt):
        raise ValueError("예약과 응답 결합이 다르다")
    raw = row.get("response")
    if not isinstance(raw, dict):
        raise ValueError("원시 응답 기록이 없다")
    raw_fields = {"response_id", "provider_model", "finish_reason", "content", "reasoning_content",
                  "input_tokens", "output_tokens", "total_tokens", "wall_s", "error_type", "status_code", "infra_error"}
    if set(raw) != raw_fields:
        raise ValueError("원시 응답 필드가 고정 형식과 다르다")
    status_code = raw.get("status_code")
    if raw.get("error_type") or status_code is not None:
        retryable_status = status_code == 429 or isinstance(status_code, int) and status_code >= 500
        reviewed_status = "infra_error" if raw.get("infra_error") or retryable_status else "call_error"
        reviewed_verdict = None
    elif not (raw.get("content") or "").strip() or raw.get("finish_reason") == "length":
        reviewed_status, reviewed_verdict = "infra_error", None
    elif raw.get("provider_model") != entry["model"]:
        reviewed_status, reviewed_verdict = "call_error", None
    else:
        data = N2.packet_path(entry["set_name"], entry["kind"]).read_bytes()
        reviewed_verdict = F.validate(entry["kind"], raw.get("content"), json.loads(data))
        reviewed_status = reviewed_verdict["status"]
    if row.get("status") != reviewed_status or row.get("verdict") != reviewed_verdict:
        raise ValueError("저장 판정과 원시 응답 재검증 결과가 다르다")
    if not isinstance(row.get("measured_wall_s"), (int, float)) or row["measured_wall_s"] < 0:
        raise ValueError("벽시계 기록이 잘못됐다")
    return row


def final_response(tag: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    first = completed_response(tag, entry, 1)
    second_exists = reservation_path(tag, entry, 2).exists() or response_path(tag, entry, 2).exists()
    if first is None or first.get("status") == "orphaned_reservation":
        if second_exists:
            raise ValueError("첫 시도 완료 전에 두 번째 시도가 있다")
        return first
    if first.get("status") != "infra_error":
        if second_exists:
            raise ValueError("재시도 대상이 아닌 첫 응답 뒤 두 번째 시도가 있다")
        return first
    second = completed_response(tag, entry, 2)
    return second


def used_attempts(tag: str) -> int:
    folder = run_dir(tag) / "reservations"
    return len(list(folder.glob("*.json"))) if folder.exists() else 0


def remaining(tag: str, *, ignore_lock: bool = False) -> dict[str, Any]:
    if not plan_path(tag).exists():
        raise FileNotFoundError(plan_path(tag))
    entries, orphans, stopped = [], [], None
    for entry in call_order():
        first = completed_response(tag, entry, 1)
        if first and first.get("status") == "orphaned_reservation":
            orphans.append(entry["call"])
            break
        final = final_response(tag, entry)
        if final is not None:
            if final.get("status") == "orphaned_reservation":
                orphans.append(entry["call"])
                break
            if final.get("status") == "call_error":
                stopped = "unsupported_or_call_error"
                break
            continue
        attempt = 2 if first and first.get("status") == "infra_error" else 1
        entries.append((entry, attempt))
        break
    if lock_path(tag).exists() and not ignore_lock:
        status = "locked"
    elif orphans:
        status = "orphaned_reservations"
    elif stopped:
        status = "stopped_inconclusive"
    elif used_attempts(tag) >= API_CAP and entries:
        status = "budget_exhausted"
    elif entries:
        status = "ready"
    else:
        status = "complete"
    commands = [f'"$HOME/.venvs/rl-triage/bin/python" evals/nemotron_upgrade_compare_v1.py run '
                f'--tag {tag} --call {entry["call"]} --attempt {attempt}'
                for entry, attempt in entries]
    return {"status": status, "remaining": len(entries), "requests_used": used_attempts(tag),
            "orphaned_calls": orphans, "commands": commands}


@contextmanager
def call_lock(tag: str):
    path = lock_path(tag)
    write_new(path, {"schema": "nemotron_upgrade_compare_lock_v1", "pid": os.getpid(),
                     "created_at": datetime.now(timezone.utc).isoformat()})
    try:
        yield
    finally:
        if path.exists():
            path.unlink()


def execute_call(tag: str, call_number: int, attempt: int,
                 complete: Callable[[list[dict], dict], dict] = F.openai_complete) -> dict[str, Any]:
    plan = load_plan(tag)
    verify_inputs()
    entry = call_order()[call_number - 1] if 1 <= call_number <= BASE_CALLS else None
    if entry is None or attempt not in (1, 2):
        raise ValueError("call 또는 attempt 범위가 다르다")
    with call_lock(tag):
        state = remaining(tag, ignore_lock=True)
        expected = state["commands"][0] if state["commands"] else None
        if (state["status"] != "ready" or expected is None
                or f"--call {call_number} --attempt {attempt}" not in expected):
            raise ValueError("현재 고정 순서에서 실행할 수 없는 호출이다")
        if used_attempts(tag) >= API_CAP:
            raise ValueError("API 상한에 도달했다")
        packet = next(row for row in packet_rows() if row["packet_id"] == entry["packet_id"])
        path = N2.packet_path(entry["set_name"], entry["kind"])
        data = path.read_bytes()
        messages, _ = F.first_messages(entry["kind"], data)
        params = F.request_params(entry["model"])
        if params != {"model": entry["model"], **REQUEST_COMMON}:
            raise ValueError("공개 요청 설정이 사전등록과 다르다")
        reservation = {"schema": "nemotron_upgrade_compare_reservation_v1", "tag": tag,
                       "experiment_commit": plan["experiment_commit"], "entry": entry, "attempt": attempt,
                       "packet_sha256": packet["raw_sha256"], "messages_sha256": packet["messages_sha256"],
                       "request": params, "reserved_at": datetime.now(timezone.utc).isoformat()}
        reserved = reservation_path(tag, entry, attempt)
        write_new(reserved, reservation)
        started = time.perf_counter()
        result = complete(messages, params)
        wall = time.perf_counter() - started
        status = "call_error"
        verdict = None
        if result.get("error_type"):
            status = "infra_error" if result.get("infra_error") else "call_error"
        elif not (result.get("content") or "").strip() or result.get("finish_reason") == "length":
            status = "infra_error"
        elif result.get("provider_model") != entry["model"]:
            status = "call_error"
            result = {**result, "error_type": "provider_model_mismatch"}
        else:
            verdict = F.validate(entry["kind"], result.get("content"), json.loads(data))
            status = verdict["status"]
        response = {"schema": "nemotron_upgrade_compare_response_v1", "entry": entry, "attempt": attempt,
                    "reservation_sha256": sha_raw(reserved), "status": status,
                    "response": {key: result.get(key) for key in ("response_id", "provider_model", "finish_reason", "content",
                                                                        "reasoning_content", "input_tokens", "output_tokens",
                                                                        "total_tokens", "wall_s", "error_type", "status_code",
                                                                        "infra_error")},
                    "measured_wall_s": wall, "verdict": verdict,
                    "finished_at": datetime.now(timezone.utc).isoformat()}
        write_new(response_path(tag, entry, attempt), response)
    return {"status": status, "call": call_number, "attempt": attempt}


def decision_key(kind: str, verdict: dict[str, Any] | None) -> tuple | None:
    if not verdict or verdict.get("status") != "valid":
        return None
    return tuple(N2.decision_key(kind, verdict["decision"]))


def pass_hat(keys: list[tuple | None], k: int) -> float | None:
    return N2.pass_hat_k(keys, k)


def token_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [row["response"].get(field) for row in attempts]
        out[field] = {"known": sum(value is not None for value in values),
                      "missing": sum(value is None for value in values),
                      "sum_known": sum(value for value in values if value is not None)}
    walls = [row["response"].get("wall_s") if row["response"].get("wall_s") is not None else row["measured_wall_s"]
             for row in attempts]
    out["wall_s"] = {"count": len(walls), "total": sum(walls), "mean": sum(walls) / len(walls) if walls else None,
                     "median": statistics.median(walls) if walls else None}
    return out


def summarize(tag: str) -> dict[str, Any]:
    load_plan(tag)
    verify_inputs()
    rows, all_attempts = [], []
    for entry in call_order():
        attempts = []
        for attempt in (1, 2):
            record = completed_response(tag, entry, attempt)
            if record and record.get("status") != "orphaned_reservation":
                attempts.append(record)
                all_attempts.append(record)
        final = final_response(tag, entry)
        status = "missing" if final is None else final.get("status")
        key = decision_key(entry["kind"], final.get("verdict") if isinstance(final, dict) else None)
        agrees = None
        if key and entry["kind"] == "correction":
            agrees = bool(final["verdict"]["decision"].get("agrees_with_rule"))
        rows.append({**entry, "status": status, "decision_key": list(key) if key else None,
                     "agrees_with_rule": agrees, "attempts": len(attempts)})
    per_packet = []
    for packet in packet_rows():
        for model in MODELS:
            group = [row for row in rows if row["packet_id"] == packet["packet_id"] and row["model"] == model]
            keys = [tuple(row["decision_key"]) if row["decision_key"] else None for row in group]
            unresolved_group = any(row["status"] in {"missing", "orphaned_reservation", "infra_error", "call_error"}
                                   for row in group)
            per_packet.append({"packet_id": packet["packet_id"], "kind": packet["kind"], "model": model,
                               "statuses": dict(Counter(row["status"] for row in group)),
                               "decision_counts": {json.dumps(key, ensure_ascii=False): count for key, count in Counter(keys).items() if key},
                               "pass": {str(k): None if unresolved_group else pass_hat(keys, k) for k in (1, 2, 3)}})
    model_summary = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        by_kind = {}
        for kind in ("diagnosis", "correction"):
            subset = [row for row in model_rows if row["kind"] == kind]
            packet_pass = [row["pass"]["3"] for row in per_packet if row["model"] == model and row["kind"] == kind]
            actions = Counter(row["decision_key"][0] for row in subset if row["decision_key"])
            by_kind[kind] = {"valid": sum(row["status"] == "valid" for row in subset), "total": len(subset),
                             "mean_pass3": sum(packet_pass) / len(packet_pass) if all(v is not None for v in packet_pass) else None,
                             "action_counts": dict(actions),
                             "correction_rule_disagreements": sum(row["agrees_with_rule"] is False for row in subset)}
        model_summary[model] = {"statuses": dict(Counter(row["status"] for row in model_rows)), "by_kind": by_kind}
    agreements, comparable = 0, 0
    for run in range(1, RUNS + 1):
        for packet in packet_rows():
            pair = [row for row in rows if row["run"] == run and row["packet_id"] == packet["packet_id"]]
            if len(pair) == 2 and all(row["decision_key"] is not None for row in pair):
                comparable += 1
                agreements += pair[0]["decision_key"] == pair[1]["decision_key"]
    unresolved = [row for row in rows if row["status"] in {"missing", "orphaned_reservation", "infra_error", "call_error"}]
    baseline, candidate = model_summary[BASELINE], model_summary[CANDIDATE]
    complete = len(rows) == BASE_CALLS and not unresolved
    conditions = {
        "complete": complete,
        "candidate_valid_at_least_32": candidate["statuses"].get("valid", 0) >= 32,
        "kind_valid_not_below_baseline": all(candidate["by_kind"][kind]["valid"] >= baseline["by_kind"][kind]["valid"]
                                                   for kind in ("diagnosis", "correction")),
        "diagnosis_continue_not_below_baseline": candidate["by_kind"]["diagnosis"]["action_counts"].get("continue", 0)
                                                   >= baseline["by_kind"]["diagnosis"]["action_counts"].get("continue", 0),
        "kind_pass3_not_below_baseline_and_at_least_0_8": all(
            candidate["by_kind"][kind]["mean_pass3"] is not None
            and baseline["by_kind"][kind]["mean_pass3"] is not None
            and candidate["by_kind"][kind]["mean_pass3"] >= baseline["by_kind"][kind]["mean_pass3"]
            and candidate["by_kind"][kind]["mean_pass3"] >= 0.8 for kind in ("diagnosis", "correction")),
        "correction_disagreements_not_above_baseline": candidate["by_kind"]["correction"]["correction_rule_disagreements"]
                                                        <= baseline["by_kind"]["correction"]["correction_rule_disagreements"],
    }
    if not complete:
        outcome = "incomplete_inconclusive"
    elif all(conditions.values()):
        outcome = "conditional_future_demo_candidate"
    else:
        outcome = "retain_current"
    cost_by_model = {}
    for model in MODELS:
        model_attempts = [row for row in all_attempts if row["entry"]["model"] == model]
        cost_by_model[model] = {"all": token_summary(model_attempts), "by_kind": {
            kind: token_summary([row for row in model_attempts if row["entry"]["kind"] == kind])
            for kind in ("diagnosis", "correction")}}
    run_state = remaining(tag)
    terminal = run_state["status"] in {"complete", "orphaned_reservations", "stopped_inconclusive", "budget_exhausted"}
    report = {"schema": "nemotron_upgrade_compare_summary_v1", "tag": tag, "outcome": outcome,
              "terminal": terminal, "run_state": run_state["status"],
              "conditions": conditions, "model_summary": model_summary, "per_packet": per_packet,
              "cross_model_decision_agreement": {"agreements": agreements, "comparable": comparable,
                                                   "rate": agreements / comparable if comparable else None},
              "cost_all_attempts": token_summary(all_attempts), "cost_by_model": cost_by_model,
              "requests_reserved": used_attempts(tag), "response_count": len(all_attempts),
              "unresolved_reservations": run_state["orphaned_calls"], "attempts": len(all_attempts), "rows": rows,
              "wording": [FIXED["wording"]["candidate" if outcome == "conditional_future_demo_candidate" else
                                                   "retain" if outcome == "retain_current" else "incomplete"],
                          FIXED["wording"]["common"]]}
    output = run_dir(tag) / "summary.json"
    if not terminal:
        return report
    if output.exists():
        if read(output) != report:
            raise ValueError("기존 요약과 재계산 결과가 다르다")
    else:
        write_new(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--commit", required=True)
    plan_parser.add_argument("--tag", required=True)
    remaining_parser = sub.add_parser("remaining_commands")
    remaining_parser.add_argument("--tag", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--call", required=True, type=int)
    run_parser.add_argument("--attempt", required=True, type=int)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "verify":
        result = {"status": "verified", "protocol": verify_protocol()["schema"]}
    elif args.command == "plan":
        result = make_plan(args.commit, args.tag)
    elif args.command == "remaining_commands":
        load_plan(args.tag)
        result = remaining(args.tag)
    elif args.command == "run":
        result = execute_call(args.tag, args.call, args.attempt)
    else:
        result = summarize(args.tag)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
