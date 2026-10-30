"""Nemotron 3.5 동일 수치 A/B 비교의 호출 실행 고리. 실제 전송은 메인 PC의 별도 실행 등록에서만 연다.

P5 manifest·capsule을 다시 검증해 4계보×2모델 call slot을 write-once plan으로 고정한다. 실행은 slot마다
intent를 먼저 journal에 남기고 provider 원문·receipt·선택 결과를 따로 write-once로 쓴다. 재시도는 등록하지
않았으며 응답 없는 intent는 unknown_pending으로 HOLD한다. --prepare·--resume·--score는 키와 네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import feedback_model_call as F
import nemotron35_value_cases_v1 as P5

V, L = P5.V, P5.L
PROTOCOL = ROOT / "bench/protocols/nemotron35_value_compare_call_20261030_v1.json"
PROTOCOL_SCHEMA = "nemotron35_value_compare_call_protocol_v1"
PLAN_SCHEMA = "nemotron35_value_compare_plan_v1"
REGISTRATION_SCHEMA = "nemotron35_value_compare_registration_v1"
RECEIPT_SCHEMA = "nemotron35_value_compare_request_receipt_v1"
RESULT_SCHEMA = "nemotron35_value_compare_result_v1"
SCORE_SCHEMA = "nemotron35_value_compare_score_v1"
SOURCE_FILES = ("evals/nemotron35_value_compare_call_v1.py", "tests/test_nemotron35_value_compare_call_v1.py")
P5_FILES = ("evals/nemotron35_value_cases_v1.py", "tests/test_nemotron35_value_cases_v1.py",
            "bench/protocols/nemotron35_value_cases_20261030_v1.json", "docs/NEMOTRON35-VALUE-CASES.md")
ARMS = ("A", "B")
SYNTHETIC, REAL = "synthetic_cpu", "registered_real_run"
PROVIDER_KINDS = {SYNTHETIC: "synthetic_cpu_stub", REAL: "nvidia_hosted_api"}
UNAVAILABLE_HTTP = frozenset({404, 410})
GENESIS = "0" * 64
ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
REGISTRATION_FIELDS = frozenset({"schema", "kind", "protocol_sha256", "plan_digest", "p5_manifest_digest",
                                 "source_commit", "approval_granted", "approver", "request_cap", "logical_calls",
                                 "host"})
REGISTERED_PROVIDER = {"kind": "nvidia_hosted_api", "client": "openai_sdk_with_raw_response",
                       "base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY",
                       "timeout_s": 300, "max_retries": 0}
TOOL_CONTRACT = "selection_three_field_v1"

canonical, file_sha256, strict_json = P5.canonical, P5.file_sha256, P5.strict_json


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + f".{int(epoch % 1 * 1_000_000):06d}Z"


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_protocol(path: Path = PROTOCOL) -> dict:
    value = strict_json(Path(path).read_bytes())
    require(isinstance(value, dict), "사전등록이 JSON 객체가 아니다")
    return value


def verify_protocol(protocol: dict) -> None:
    """DRAFT 상태와 소스·P5 결합, 등록 params·예산을 확인한다. 실행 승인을 만들지 않는다."""
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "사전등록 schema가 다르다")
    require(protocol.get("status") == "DRAFT_not_executed" and protocol.get("executed") is False
            and protocol.get("approval_granted") is False, "미실행·미승인 DRAFT 사전등록만 받는다")
    for key, names in (("source_sha256_lf", SOURCE_FILES), ("p5_files_sha256_lf", P5_FILES)):
        bound = protocol.get(key)
        require(isinstance(bound, dict) and set(bound) == set(names), f"{key} 결합 목록이 다르다")
        for rel, expected in bound.items():
            require(file_sha256(ROOT / rel) == expected, f"사전등록 뒤 파일이 바뀌었다: {rel}")
    arms = protocol.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "A·B 외 조건은 실행하지 않는다")
    for spec in arms.values():
        require(isinstance(spec.get("model"), str) and spec["model"] in spec.get("accepted_provider_models", []),
                "모델·반환 모델 등록이 다르다")
    template = protocol.get("request_params")
    require(isinstance(template, dict) and "model" not in template
            and all(F.request_params(spec["model"]) == {"model": spec["model"], **template} for spec in arms.values()),
            "등록 params가 기존 호출기 params와 다르다")
    require(protocol.get("provider") == REGISTERED_PROVIDER, "provider·client 설정이 등록과 다르다")
    require(isinstance(protocol.get("tool_contract"), dict) and protocol["tool_contract"].get("name") == TOOL_CONTRACT,
            "도구 계약이 등록과 다르다")
    budget = protocol.get("budget")
    require(budget == {"logical_calls": 8, "request_cap": 24, "max_requests_per_slot": 1, "retries_registered": False},
            "8 논리 호출·24 요청 상한·재시도 없음 등록과 다르다")
    require(protocol.get("blocked_arms") == {"C": "blocked_new_collection_required"}, "C 차단 등록이 다르다")


def _slots(manifest: dict, protocol: dict) -> list[dict]:
    """계보 순서가 짝수면 A→B, 홀수면 B→A. 같은 계보의 두 slot은 같은 capsule·params를 쓴다."""
    slots = []
    for index, lineage in enumerate(manifest["lineages"]):
        for arm in ARMS if index % 2 == 0 else tuple(reversed(ARMS)):
            order = len(slots) + 1
            slots.append({"slot_id": f"{order:02d}-{lineage['lineage_id']}-{arm}", "order": order,
                          "lineage_id": lineage["lineage_id"], "lineage_index": index, "role": lineage["role"],
                          "arm": arm, "model": protocol["arms"][arm]["model"],
                          "accepted_provider_models": list(protocol["arms"][arm]["accepted_provider_models"]),
                          "capsule_file": f"p5/{lineage['capsule']['file']}",
                          "capsule_sha256": lineage["capsule"]["sha256"],
                          "params_sha256": L.digest(protocol["request_params"]), "max_requests": 1})
    return slots


def _p5_manifest(run: Path, protocol: dict) -> dict:
    require((run / "p5/manifest.json").is_file(), "P5 준비본이 없다. 먼저 --prepare를 실행한다")
    manifest = P5.check(P5.PROTOCOL, run / "p5/manifest.json")
    require(manifest["manifest_digest"] == protocol["p5"]["manifest_digest"]
            and manifest["protocol_digest"] == protocol["p5"]["protocol_digest"],
            "P5 manifest·protocol digest가 등록과 다르다")
    require(all(row["arms"]["C"]["status"] == protocol["blocked_arms"]["C"] for row in manifest["lineages"]),
            "C가 차단되지 않은 manifest다")
    return manifest


def validate_plan(plan: dict, manifest: dict, protocol: dict, protocol_sha: str) -> dict:
    """slot 수·요청 상한·중복·순서·계보·조건을 실행 전에 확인한다."""
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA, "plan 형식이 다르다")
    require(plan.get("plan_digest") == L.digest({k: v for k, v in plan.items() if k != "plan_digest"}),
            "plan이 바뀌었다")
    slots = plan.get("slots")
    budget = protocol["budget"]
    require(isinstance(slots, list) and len(slots) <= budget["logical_calls"], "논리 호출 8개를 넘는 계획이다")
    require(all(isinstance(slot, dict) and type(slot.get("max_requests")) is int for slot in slots)
            and sum(slot["max_requests"] for slot in slots) <= budget["request_cap"], "요청 상한 24를 넘는 계획이다")
    require(all(slot.get("max_requests") == budget["max_requests_per_slot"] for slot in slots), "등록되지 않은 slot 재시도다")
    require(len({slot.get("slot_id") for slot in slots}) == len(slots)
            and len({(slot.get("lineage_id"), slot.get("arm")) for slot in slots}) == len(slots), "중복 slot이다")
    require({slot.get("arm") for slot in slots} <= set(ARMS), "C 또는 등록 밖 조건을 실행하려는 계획이다")
    require({slot.get("lineage_id") for slot in slots} <= {row["lineage_id"] for row in manifest["lineages"]},
            "P5 밖 계보다")
    require(slots == _slots(manifest, protocol), "등록된 slot 순서·counterbalance·capsule·params와 다르다")
    expected = {"protocol_sha256": protocol_sha, "p5_manifest_digest": manifest["manifest_digest"],
                "p5_protocol_digest": manifest["protocol_digest"],
                "source_sha256": {**protocol["source_sha256_lf"], **protocol["p5_files_sha256_lf"]},
                "provider": protocol["provider"], "request_params": protocol["request_params"],
                "tool_contract": protocol["tool_contract"]["name"], "budget": budget,
                "blocked_arms": {row["lineage_id"]: row["arms"]["C"]["status"] for row in manifest["lineages"]}}
    require(all(plan.get(key) == value for key, value in expected.items()), "plan 결합이 사전등록·P5와 다르다")
    return plan


def prepare(run: Path, protocol_path: Path = PROTOCOL) -> dict:
    protocol = load_protocol(protocol_path)
    verify_protocol(protocol)
    run = Path(run)
    require(not run.exists(), "실행 폴더가 이미 있다")
    P5.prepare(P5.PROTOCOL, run / "p5")
    manifest = _p5_manifest(run, protocol)
    plan = {"schema": PLAN_SCHEMA, "status": "planned_no_calls", "protocol_sha256": file_sha256(Path(protocol_path)),
            "p5_manifest_digest": manifest["manifest_digest"], "p5_protocol_digest": manifest["protocol_digest"],
            "source_sha256": {**protocol["source_sha256_lf"], **protocol["p5_files_sha256_lf"]},
            "provider": deepcopy(protocol["provider"]), "request_params": deepcopy(protocol["request_params"]),
            "tool_contract": protocol["tool_contract"]["name"], "budget": deepcopy(protocol["budget"]),
            "blocked_arms": {row["lineage_id"]: row["arms"]["C"]["status"] for row in manifest["lineages"]},
            "slots": _slots(manifest, protocol)}
    plan["plan_digest"] = L.digest(plan)
    validate_plan(plan, manifest, protocol, plan["protocol_sha256"])
    _write_new(run / "plan.json", _json_bytes(plan))
    return plan


def load_plan(run: Path, protocol: dict, protocol_path: Path) -> tuple[dict, dict]:
    manifest = _p5_manifest(Path(run), protocol)
    require((Path(run) / "plan.json").is_file(), "plan이 없다. 먼저 --prepare를 실행한다")
    plan = strict_json((Path(run) / "plan.json").read_bytes())
    return validate_plan(plan, manifest, protocol, file_sha256(Path(protocol_path))), manifest


def _validators(run: Path, plan: dict, manifest: dict) -> dict:
    """capsule의 선택 화면으로 생산 세 필드 검증기에 넘길 packet을 만든다. 전송 bytes는 P5 capsule 그대로다."""
    lineages = {row["lineage_id"]: row for row in manifest["lineages"]}
    out = {}
    for slot in plan["slots"]:
        if slot["lineage_id"] in out:
            continue
        raw = (run / slot["capsule_file"]).read_bytes()
        require(raw_sha(raw) == slot["capsule_sha256"], "capsule bytes가 등록과 다르다")
        messages = strict_json(raw)
        view = strict_json(messages[1]["content"].encode("utf-8"))
        lineage = lineages[slot["lineage_id"]]
        require(L.digest(view) == lineage["view_digest"] and view["status"] == "open", "capsule 선택 화면이 P5와 다르다")
        packet = V.S.build_packet(
            epoch=1, applied_corrections=[], prior_receipt=None, facts=view["evidence_facts"],
            budget_left=view["budget_left"], execution_budget_s=view["execution_budget_s"],
            input_bindings={role: {"checkpoint_sha256": lineage["checkpoints"][role],
                                   "telemetry_sha256": lineage["telemetry_sha256"][role]}
                            for role in ("case", "reference")})
        out[slot["lineage_id"]] = {"messages": messages, "view": view, "packet": packet}
    return out


def validate_content(validator: dict, content: str) -> tuple[str, dict | None, list[str]]:
    """selection_comparison_view_v1.select와 같은 세 필드·근거·eligible 검사. 무효를 규칙 선택으로 바꾸지 않는다."""
    view = validator["view"]
    review = V.S.format_review(content, {**validator["packet"], "facts": view["evidence_facts"]})
    if review["status"] != "valid":
        return "semantic_invalid", None, list(review["codes"])
    decision = review["decision"]
    codes = [] if all(fact in view["evidence_facts"] for fact in decision["fact_ids"]) else ["S_FACT_UNKNOWN"]
    if decision["action"] == "continue" and decision["probe"] not in view["eligible_probes"]:
        codes.append("S_PROBE_INELIGIBLE")
    if codes:
        return "semantic_invalid", None, codes
    return "valid", {key: deepcopy(decision[key]) for key in ("action", "probe", "fact_ids")}, []


def _usage(body: dict) -> dict | None:
    usage = body.get("usage")
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    if not isinstance(usage, dict):
        return None
    values = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
              "total_tokens": usage.get("total_tokens")}
    if any(value is not None and (type(value) is not int or value < 0) for value in values.values()):
        return None
    if all(type(value) is int for value in values.values()) and (
            values["input_tokens"] + values["output_tokens"] != values["total_tokens"]):
        return None
    return values


def classify(slot: dict, validator: dict, http_status: int | None, raw: bytes | None,
             error_type: str | None) -> dict:
    """원문 바이트에서만 상태를 다시 계산한다. resume·score는 같은 함수로 재생한다."""
    out = {"status": "infrastructure_failed", "codes": [], "decision": None, "provider_model": None,
           "response_id": None, "finish_reason": None,
           "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None}}
    if http_status is None:
        return {**out, "codes": ["TRANSPORT_ERROR"]}
    if http_status in UNAVAILABLE_HTTP:
        return {**out, "status": "model_unavailable", "codes": [f"HTTP_{http_status}"]}
    if http_status != 200 or raw is None:
        return {**out, "codes": [f"HTTP_{http_status}"]}
    try:
        body = strict_json(raw)
        choice = body["choices"][0]
        content, finish = choice["message"]["content"], choice["finish_reason"]
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
        return {**out, "codes": ["RAW_CONTRACT"]}
    out.update(provider_model=body.get("model") if isinstance(body.get("model"), str) else None,
               response_id=body.get("id") if isinstance(body.get("id"), str) else None,
               finish_reason=finish if isinstance(finish, str) else None)
    usage = _usage(body)
    if usage is None:
        return {**out, "codes": ["USAGE_CONTRACT"]}
    out["usage"] = usage
    if out["provider_model"] not in slot["accepted_provider_models"]:
        return {**out, "codes": ["PROVIDER_MODEL_UNREGISTERED"]}
    if not isinstance(content, str) or not content.strip() or finish == "length":
        return {**out, "codes": ["EMPTY_OR_TRUNCATED"]}
    if finish != "stop":
        return {**out, "codes": ["FINISH_REASON"]}
    status, decision, codes = validate_content(validator, content)
    return {**out, "status": status, "decision": decision, "codes": codes}


def _registration(path: Path, protocol: dict, protocol_sha: str, plan: dict) -> dict:
    value = strict_json(Path(path).read_bytes())
    require(isinstance(value, dict) and set(value) == REGISTRATION_FIELDS
            and value["schema"] == REGISTRATION_SCHEMA and value["kind"] in PROVIDER_KINDS, "실행 등록 형식이 다르다")
    require(value["protocol_sha256"] == protocol_sha and value["plan_digest"] == plan["plan_digest"]
            and value["p5_manifest_digest"] == plan["p5_manifest_digest"], "실행 등록이 사전등록·plan과 다르다")
    commit = value["source_commit"]
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
            and value["approval_granted"] is True and isinstance(value["approver"], str) and value["approver"].strip(),
            "실행 등록의 commit·승인이 없다")
    require(value["request_cap"] == protocol["budget"]["request_cap"]
            and value["logical_calls"] == protocol["budget"]["logical_calls"], "실행 등록의 상한이 사전등록과 다르다")
    host = value["host"]
    require(isinstance(host, dict) and set(host) == {"key_env", "platform"}
            and host["key_env"] == protocol["provider"]["key_env"] and isinstance(host["platform"], str),
            "호스트 설정이 다르다")
    return value


def _read_journal(run: Path) -> list[dict]:
    path = run / "journal.jsonl"
    if not path.exists():
        return []
    events, previous = [], GENESIS
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        require(line.endswith(b"\n"), "journal 마지막 줄이 완결되지 않았다")
        event = strict_json(line)
        require(isinstance(event, dict) and event.get("seq") == index and event.get("prev") == previous,
                "journal 연결이 바뀌었다")
        events.append(event)
        previous = raw_sha(line)
    return events


def _append(run: Path, event: dict) -> None:
    events = _read_journal(run)
    path = run / "journal.jsonl"
    previous = raw_sha(path.read_bytes().splitlines(keepends=True)[-1]) if events else GENESIS
    line = json.dumps({**event, "seq": len(events), "prev": previous}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _paths(slot: dict) -> dict:
    stem = f"{slot['slot_id']}.attempt1"
    return {"raw": f"raw/{stem}.response", "receipt": f"requests/{stem}.json",
            "result": f"results/{slot['slot_id']}.json"}


def _request_digest(slot: dict, plan: dict, messages: list) -> str:
    return L.digest({"model": slot["model"], "messages": messages, "params": plan["request_params"]})


def _intent(slot: dict, plan: dict, messages: list, registration_sha: str, provider_kind: str) -> dict:
    return {"event": "intent", "slot_id": slot["slot_id"], "attempt": 1, "model": slot["model"],
            "capsule_sha256": slot["capsule_sha256"], "params_sha256": slot["params_sha256"],
            "protocol_sha256": plan["protocol_sha256"], "source_sha256": plan["source_sha256"],
            "plan_digest": plan["plan_digest"], "registration_sha256": registration_sha,
            "provider_kind": provider_kind, "request_digest": _request_digest(slot, plan, messages)}


def _receipt(slot: dict, intent: dict, timing: dict, http_status: int | None, error_type: str | None,
             raw: bytes | None, outcome: dict) -> dict:
    return {"schema": RECEIPT_SCHEMA, "logical_call_id": slot["slot_id"], "lineage_id": slot["lineage_id"],
            "arm": slot["arm"], **{key: intent[key] for key in (
                "attempt", "model", "capsule_sha256", "params_sha256", "protocol_sha256", "source_sha256",
                "plan_digest", "registration_sha256", "provider_kind", "request_digest")},
            "synthetic": intent["provider_kind"] == PROVIDER_KINDS[SYNTHETIC], **timing,
            "http_status": http_status, "error_type": error_type,
            "raw_sha256": None if raw is None else raw_sha(raw), "raw_bytes": None if raw is None else len(raw),
            **{key: outcome[key] for key in (
                "provider_model", "response_id", "finish_reason", "usage", "status", "codes")}}


def _result(slot: dict, receipt_sha: str, outcome: dict, raw_digest: str | None) -> dict:
    decision = outcome["decision"] or {"action": None, "probe": None, "fact_ids": None}
    return {"schema": RESULT_SCHEMA, "slot_id": slot["slot_id"], "lineage_id": slot["lineage_id"], "arm": slot["arm"],
            "status": outcome["status"], **decision, "codes": outcome["codes"],
            "receipt_sha256": receipt_sha, "raw_sha256": raw_digest, "rule_fallback_used": False}


def _timing_ok(receipt: dict) -> bool:
    start, end, latency = receipt.get("started_epoch"), receipt.get("finished_epoch"), receipt.get("latency_s")
    numbers = all(V.number(value) for value in (start, end, latency))
    return (numbers and 0 <= latency <= end - start + 1.0 and end >= start
            and receipt.get("started_at") == _iso(start) and receipt.get("finished_at") == _iso(end))


def run_state(run: Path, plan: dict, manifest: dict, validators: dict) -> dict:
    """journal·원문·receipt·결과를 다시 검증하고 slot별 상태를 낸다. provider를 부르지 않는다."""
    run = Path(run)
    slots = {slot["slot_id"]: slot for slot in plan["slots"]}
    registration = run / "registration.json"
    registration_sha = file_sha256(registration) if registration.exists() else None
    kind = strict_json(registration.read_bytes())["kind"] if registration.exists() else None
    state = {slot_id: {"status": "not_started"} for slot_id in slots}
    events = _read_journal(run)
    require(sum(event.get("event") == "intent" for event in events) <= plan["budget"]["request_cap"],
            "등록 요청 상한을 넘었다")
    for event in events:
        slot = slots.get(event.get("slot_id"))
        require(slot is not None, "계획 밖 slot 기록이다")
        current = state[slot["slot_id"]]
        if event.get("event") == "intent":
            require(current["status"] == "not_started", "같은 slot의 재시도는 등록되지 않았다")
            messages = validators[slot["lineage_id"]]["messages"]
            expected = _intent(slot, plan, messages, registration_sha, PROVIDER_KINDS.get(kind, ""))
            require({k: v for k, v in event.items() if k not in ("seq", "prev")} == expected,
                    "intent가 plan·capsule·등록과 다르다")
            state[slot["slot_id"]] = {"status": "unknown_pending", "intent": expected}
            continue
        require(event.get("event") == "response" and current["status"] == "unknown_pending", "intent 없는 응답 기록이다")
        paths = _paths(slot)
        receipt_raw = (run / paths["receipt"]).read_bytes()
        require(raw_sha(receipt_raw) == event.get("receipt_sha256"), "receipt가 journal과 다르다")
        receipt = strict_json(receipt_raw)
        raw = None
        if receipt.get("raw_sha256") is not None:
            raw = (run / paths["raw"]).read_bytes()
            require(raw_sha(raw) == receipt["raw_sha256"] == event.get("raw_sha256"), "provider 원문이 바뀌었다")
        else:
            require(not (run / paths["raw"]).exists() and event.get("raw_sha256") is None, "등록되지 않은 원문 파일이다")
        error_type = receipt.get("error_type")
        require(error_type is None or (isinstance(error_type, str) and ERROR_TYPE.fullmatch(error_type) is not None),
                "오류 종류 기록 형식이 다르다")
        http_status = receipt.get("http_status")
        require(http_status is None or (type(http_status) is int and 100 <= http_status <= 599), "HTTP 상태 기록 형식")
        outcome = classify(slot, validators[slot["lineage_id"]], http_status, raw, error_type)
        timing = {key: receipt.get(key) for key in ("started_at", "finished_at", "started_epoch", "finished_epoch",
                                                    "latency_s")}
        require(_timing_ok(receipt), "시각·latency 기록이 다르다")
        expected = _receipt(slot, current["intent"], timing, http_status, error_type, raw, outcome)
        require(receipt == expected, "receipt의 model·usage·상태가 원문·등록과 다르다")
        result_raw = (run / paths["result"]).read_bytes()
        require(raw_sha(result_raw) == event.get("result_sha256")
                and strict_json(result_raw) == _result(slot, event["receipt_sha256"], outcome, receipt["raw_sha256"]),
                "선택 결과가 원문 재생과 다르다")
        require(event.get("status") == outcome["status"], "journal 상태가 원문 재생과 다르다")
        state[slot["slot_id"]] = {"status": outcome["status"], "receipt": receipt, "outcome": outcome}
    for folder, key in (("raw", "raw"), ("requests", "receipt"), ("results", "result")):
        existing = {path.relative_to(run).as_posix() for path in (run / folder).glob("*")}
        allowed = {_paths(slots[slot_id])[key] for slot_id, row in state.items() if row["status"] != "not_started"}
        require(existing <= allowed, f"intent 없는 {folder} 파일이 있다")
    return state


def _summary(state: dict, plan: dict, provider_calls: int, status: str, reason: str | None = None) -> dict:
    return {"status": status, "reason": reason, "provider_calls_this_invocation": provider_calls,
            "requests_used": sum(row["status"] != "not_started" for row in state.values()),
            "request_cap": plan["budget"]["request_cap"],
            "slots": {slot_id: row["status"] for slot_id, row in state.items()}}


def _hold(reason: str) -> dict:
    return {"status": "hold", "reason": reason, "provider_calls_this_invocation": 0}


def _provider_kind(provider: object) -> str:
    kind = getattr(provider, "kind", None)
    require(kind in PROVIDER_KINDS.values(), "등록되지 않은 provider 종류다")
    return kind


def _reply(value: object) -> tuple[int | None, bytes | None, str | None]:
    if not isinstance(value, dict) or set(value) != {"status_code", "body", "error_type"}:
        return None, None, "ProviderReplyContractError"
    status, body, error = value["status_code"], value["body"], value["error_type"]
    if ((status is not None and (type(status) is not int or not 100 <= status <= 599))
            or (body is not None and not isinstance(body, bytes))
            or (error is not None and (not isinstance(error, str) or ERROR_TYPE.fullmatch(error) is None))):
        return None, None, "ProviderReplyContractError"
    return status, body, error


def execute(run: Path, registration_path: Path | None, provider: object, protocol_path: Path = PROTOCOL) -> dict:
    """별도 실행 등록과 호스트 키가 있을 때만 slot 순서대로 한 번씩 보낸다. 미완료 intent가 있으면 보내지 않는다."""
    if registration_path is None:
        return _hold("execution_registration_missing")
    protocol = load_protocol(protocol_path)
    verify_protocol(protocol)
    run, protocol_sha = Path(run), file_sha256(Path(protocol_path))
    plan, manifest = load_plan(run, protocol, protocol_path)
    registration = _registration(registration_path, protocol, protocol_sha, plan)
    kind = _provider_kind(provider)
    require(PROVIDER_KINDS[registration["kind"]] == kind, "실행 등록과 provider 종류가 다르다")
    if registration["kind"] == REAL:
        if registration["host"]["platform"] != sys.platform:
            return _hold("host_platform_mismatch")
        if not os.environ.get(registration["host"]["key_env"]):
            return _hold("host_key_missing")
    stored = run / "registration.json"
    if stored.exists():
        require(stored.read_bytes() == Path(registration_path).read_bytes(), "다른 실행 등록으로 이어갈 수 없다")
    else:
        require(not (run / "journal.jsonl").exists(), "등록 없이 시작된 journal이다")
        _write_new(stored, Path(registration_path).read_bytes())
    lock = run / "execute.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _hold("execute_lock_present")
    calls = 0
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
        os.close(handle)
        validators = _validators(run, plan, manifest)
        state = run_state(run, plan, manifest, validators)
        if any(row["status"] == "unknown_pending" for row in state.values()):
            return _summary(state, plan, calls, "hold", "unknown_pending_intent")
        registration_sha = file_sha256(stored)
        for slot in plan["slots"]:
            if state[slot["slot_id"]]["status"] != "not_started":
                continue
            require(sum(row["status"] != "not_started" for row in state.values()) < plan["budget"]["request_cap"],
                    "등록 요청 상한에 도달했다")
            messages = validators[slot["lineage_id"]]["messages"]
            intent = _intent(slot, plan, messages, registration_sha, kind)
            _append(run, intent)
            state[slot["slot_id"]] = {"status": "unknown_pending", "intent": intent}
            started, clock = time.time(), time.perf_counter()
            reply = provider({"slot_id": slot["slot_id"], "model": slot["model"], "messages": deepcopy(messages),
                              "params": deepcopy(plan["request_params"])})
            latency, finished = time.perf_counter() - clock, time.time()
            calls += 1
            http_status, raw, error_type = _reply(reply)
            paths = _paths(slot)
            if raw is not None:
                _write_new(run / paths["raw"], raw)
            outcome = classify(slot, validators[slot["lineage_id"]], http_status, raw, error_type)
            timing = {"started_at": _iso(started), "finished_at": _iso(finished), "started_epoch": started,
                      "finished_epoch": finished, "latency_s": latency}
            receipt_raw = _json_bytes(_receipt(slot, intent, timing, http_status, error_type, raw, outcome))
            _write_new(run / paths["receipt"], receipt_raw)
            raw_digest = None if raw is None else raw_sha(raw)
            result_raw = _json_bytes(_result(slot, raw_sha(receipt_raw), outcome, raw_digest))
            _write_new(run / paths["result"], result_raw)
            _append(run, {"event": "response", "slot_id": slot["slot_id"], "attempt": 1,
                          "receipt_sha256": raw_sha(receipt_raw), "result_sha256": raw_sha(result_raw),
                          "raw_sha256": None if raw is None else raw_sha(raw), "status": outcome["status"]})
            state[slot["slot_id"]] = {"status": outcome["status"]}
        return _summary(run_state(run, plan, manifest, validators), plan, calls, "completed")
    finally:
        lock.unlink(missing_ok=True)


def resume(run: Path, protocol_path: Path = PROTOCOL) -> dict:
    """완료 원문을 다시 검증해 재생할 뿐 새 요청을 만들지 않는다."""
    protocol = load_protocol(protocol_path)
    verify_protocol(protocol)
    plan, manifest = load_plan(Path(run), protocol, protocol_path)
    state = run_state(Path(run), plan, manifest, _validators(Path(run), plan, manifest))
    pending = any(row["status"] == "unknown_pending" for row in state.values())
    return _summary(state, plan, 0, "hold" if pending else "verified", "unknown_pending_intent" if pending else None)


def score_run(run: Path, protocol_path: Path = PROTOCOL) -> dict:
    """검증된 응답만 P5 choices로 바꿔 P5 score()에 넘긴다. 결과가 없는 조건은 pending_api로 남는다."""
    protocol = load_protocol(protocol_path)
    verify_protocol(protocol)
    run = Path(run)
    plan, manifest = load_plan(run, protocol, protocol_path)
    state = run_state(run, plan, manifest, _validators(run, plan, manifest))
    registration = run / "registration.json"
    kinds = {row["receipt"]["provider_kind"] for row in state.values() if "receipt" in row}
    real = registration.exists() and strict_json(registration.read_bytes())["kind"] == REAL
    require(kinds <= {PROVIDER_KINDS[REAL if real else SYNTHETIC]}, "합성 응답과 실제 응답을 섞을 수 없다")
    choices: dict = {}
    for slot in plan["slots"]:
        row = state[slot["slot_id"]]
        if row["status"] == "not_started":
            continue
        entry = {"status": row["status"], "capsule_sha256": slot["capsule_sha256"]}
        if row["status"] == "valid":
            entry.update(deepcopy(row["outcome"]["decision"]))
        choices.setdefault(slot["lineage_id"], {})[slot["arm"]] = entry
    value = {"schema": P5.CHOICES_SCHEMA, "manifest_digest": manifest["manifest_digest"],
             "measurement_kind": "registered_real_provider_run" if real and kinds else "synthetic_cpu",
             "choices": choices}
    result = P5.score(manifest, value, P5.load_protocol())
    journal = run / "journal.jsonl"
    head = raw_sha(journal.read_bytes()) if journal.exists() else GENESIS
    folder = run / "scores" / head[:16]
    for name, data in (("choices.json", _json_bytes(value)), ("score.json", _json_bytes(result))):
        if (folder / name).exists():
            require((folder / name).read_bytes() == data, "같은 journal의 점수 파일이 다르다")
        else:
            _write_new(folder / name, data)
    usage = [row["receipt"]["usage"] for row in state.values() if "receipt" in row]
    arm_status = {row["lineage_id"]: {} for row in manifest["lineages"]}
    for slot in plan["slots"]:
        status = state[slot["slot_id"]]["status"]
        arm_status[slot["lineage_id"]][slot["arm"]] = "pending_api" if status == "not_started" else status
    totals = {key: sum(item[key] for item in usage) if usage and all(type(item[key]) is int for item in usage)
              else None for key in ("input_tokens", "output_tokens", "total_tokens")}
    latency = [row["receipt"]["latency_s"] for row in state.values() if "receipt" in row]
    run_kind = "not_executed" if not kinds else (REAL if real else PROVIDER_KINDS[SYNTHETIC])
    return {"schema": SCORE_SCHEMA, "journal_head_sha256": head, "run_kind": run_kind,
            "measurement_kind": value["measurement_kind"],
            "arm_status": arm_status, "blocked_arms": plan["blocked_arms"],
            "axes": {name: axis["outcome"] for name, axis in result["axes"].items()}, "wording": result["wording"],
            "requests_used": len(latency), "request_cap": plan["budget"]["request_cap"], "usage_totals": totals,
            "latency_s_total": math.fsum(latency) if latency else None,
            "files": {"choices": (folder / "choices.json").relative_to(run).as_posix(),
                      "score": (folder / "score.json").relative_to(run).as_posix()},
            "selection_change_alone_counted": False, "claims": result["claims"]}


class NvidiaChatProvider:
    """기존 호출기와 같은 OpenAI 호환 client 설정으로 한 번 보낸다. 키 값·헤더는 반환·기록하지 않는다."""

    kind = PROVIDER_KINDS[REAL]

    def __init__(self, provider: dict):
        self.base_url, self.key_env = provider["base_url"], provider["key_env"]
        self.timeout_s, self.max_retries = provider["timeout_s"], provider["max_retries"]

    def __call__(self, request: dict) -> dict:
        import openai

        secret = os.environ.get(self.key_env)
        if not secret:
            return {"status_code": None, "body": None, "error_type": "KeyUnavailable"}
        client = openai.OpenAI(base_url=self.base_url, api_key=secret, timeout=self.timeout_s,
                               max_retries=self.max_retries)
        try:
            raw = client.chat.completions.with_raw_response.create(
                messages=request["messages"], model=request["model"], **request["params"])
            return {"status_code": raw.status_code, "body": raw.content, "error_type": None}
        except openai.APIStatusError as error:
            return {"status_code": error.status_code, "body": error.response.content,
                    "error_type": type(error).__name__}
        except openai.APIError as error:
            return {"status_code": None, "body": None, "error_type": type(error).__name__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--prepare", "--execute", "--resume", "--score"):
        mode.add_argument(flag, action="store_true")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            plan = prepare(args.run, args.protocol)
            result = {"status": "prepared", "plan_digest": plan["plan_digest"],
                      "p5_manifest_digest": plan["p5_manifest_digest"],
                      "slots": [slot["slot_id"] for slot in plan["slots"]], "blocked_arms": plan["blocked_arms"]}
        elif args.execute:
            protocol = load_protocol(args.protocol)
            result = execute(args.run, args.registration, NvidiaChatProvider(protocol["provider"]), args.protocol)
        elif args.resume:
            result = resume(args.run, args.protocol)
        else:
            result = score_run(args.run, args.protocol)
    except ValueError as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "hold" else 0


if __name__ == "__main__":
    raise SystemExit(main())
