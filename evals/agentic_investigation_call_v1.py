"""조사 행동의 동결 메시지·완료 응답 재생·전역 요청 상한을 기존 journal로 연결한다.

Provider는 호스트가 주입한다. 실제 endpoint 실행은 새 사전등록·소스 결합을 거친
별도 실행기가 담당하며 이 모듈 자체는 자격증명을 읽지 않는다.
"""
from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import agentic_investigation_contract_v1 as A
import grounded_vision_observer_v1 as O
import model_decision_journal_v1 as J

SCHEMA = "agentic_investigation_call_v1"
MAX_CALLS = 2
PARAMS = deepcopy(O.REQUEST_PARAMS)


def _messages(context: dict, supplied: list | None) -> list:
    expected = A.messages(context)
    if supplied is None:
        return expected
    if supplied == expected:
        return deepcopy(supplied)
    if (not isinstance(supplied, list) or len(supplied) != 2 or supplied[0] != expected[0]
            or set(supplied[1]) != {"role", "content"} or supplied[1]["role"] != "user"):
        raise ValueError("조회 결과가 조사 system/user 계약을 바꿨다")
    blocks = supplied[1]["content"]
    if (not isinstance(blocks, list) or not 1 <= len(blocks) <= 13 or len(blocks) % 2 != 1
            or blocks[0] != {"type": "text", "text": expected[1]["content"]}):
        raise ValueError("프레임 메시지가 현재 공개 facts와 다르다")
    frame_count = sum(k.startswith("vision:frame:") for k in context["model_view"]["evidence_facts"])
    if (len(blocks) - 1) // 2 > frame_count:
        raise ValueError("등록 frame 근거보다 많은 이미지")
    seen = set()
    for label, block in zip(blocks[1::2], blocks[2::2], strict=True):
        if not isinstance(label, dict) or set(label) != {"type", "text"} or label["type"] != "text":
            raise ValueError("프레임 식별자 텍스트 형식")
        try:
            citation = json.loads(label["text"], object_pairs_hook=A.V._unique_object)
            fact = context["model_view"]["evidence_facts"]["vision:frame:" + citation["frame_id"]]
            if (set(citation) != {"frame_id", "timestamp_s"}
                    or citation["timestamp_s"] != fact["timestamp_s"] or citation["frame_id"] in seen):
                raise ValueError("다른 시각 또는 중복 프레임")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("등록 프레임·시각 결합") from error
        seen.add(citation["frame_id"])
        if (not isinstance(block, dict) or set(block) != {"type", "image_url"}
                or block["type"] != "image_url" or not isinstance(block["image_url"], dict)
                or set(block["image_url"]) not in ({"url"}, {"url", "detail"})
                or block["image_url"].get("detail", "low") != "low"):
            raise ValueError("이미지 블록 형식")
        url = block["image_url"]["url"]
        if not isinstance(url, str) or not url.startswith("data:image/jpeg;base64,"):
            raise ValueError("등록된 JPEG data URL만 허용한다")
        try:
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
        except ValueError as error:
            raise ValueError("JPEG data URL 인코딩") from error
        if not raw.startswith(b"\xff\xd8"):
            raise ValueError("JPEG 원문 형식")
    return deepcopy(supplied)


def freeze(context: dict, folder: Path, *, source_anchor: str, protocol_sha256: str,
           messages: list | None = None) -> dict:
    """읽기 전용 history 사본으로 고정해 실제 고리가 진행된 뒤에도 응답을 재생한다."""
    value = A.checked(context)
    if not re.fullmatch(r"[0-9a-f]{40}", source_anchor) or not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256):
        raise ValueError("소스·사전등록 결합 형식")
    outgoing = _messages(value, messages)
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = value["snapshot"]
    inputs = snapshot["inputs"]
    original = Path(inputs["ledger_path"])
    if original.exists() and (original.is_symlink() or not original.is_file() or original.stat().st_nlink != 1):
        raise ValueError("원본 원장이 단일 regular file이 아니다")
    raw = original.read_bytes() if original.exists() else b""
    history = folder / "history.jsonl"
    with history.open("xb") as stream:
        stream.write(raw)
    frozen_snapshot = A.C.V.build_snapshot(inputs["packet"], inputs["run_id"], inputs["policy"],
        inputs["protocol_sha256"], history, parent=inputs["parent"], masked=inputs["masked"])
    frozen = A.prepare(frozen_snapshot, **{k: value[k] for k in (
        "goal", "evidence", "revealed", "extra_facts", "decisions_left")})
    if A.messages(frozen) != A.messages(value):
        raise ValueError("원장 동결이 공개 판단 입력을 바꿨다")
    capsule = {"schema": SCHEMA, "context": frozen, "messages": outgoing,
               "source_anchor": source_anchor, "protocol_sha256": protocol_sha256,
               "history_sha256": J.file_sha(history), "params": PARAMS,
               "image_origin_verification_scope": "registered_upstream_evidence_adapter"}
    path = folder / "capsule.json"
    J._write_new(path, capsule)
    return {"path": str(path), "sha256": J.file_sha(path)}


def prepared(reference: dict) -> tuple[Path, dict]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("동결 호출 참조 형식")
    path = Path(reference["path"])
    if path != path.resolve() or J.file_sha(path) != reference["sha256"]:
        raise ValueError("동결 호출 SHA 또는 경로")
    capsule = J._read(path)
    if (capsule.get("schema") != SCHEMA or capsule.get("params") != PARAMS
            or J.file_sha(path.parent / "history.jsonl") != capsule.get("history_sha256")):
        raise ValueError("동결 history·params 결합")
    A.checked(capsule["context"])
    _messages(capsule["context"], capsule["messages"])
    return path.parent, capsule


def engine(capsule: dict, complete: Callable[[list, dict], dict]) -> dict:
    """의미 무효는 HOLD. 기존 provider 분류의 인프라 오류만 한 번 재시도한다."""
    calls = []
    for attempt in range(MAX_CALLS):
        result = complete(deepcopy(capsule["messages"]), deepcopy(PARAMS))
        transport = O._transport(result)
        calls.append({"transport": transport, "result": result})
        if transport == "infra_error" and attempt + 1 < MAX_CALLS:
            continue
        if transport != "completed":
            return {"accepted": False, "stop_reason": "infra_error", "model_status": "unavailable",
                    "selection": {"action": "hold"}, "final_content": None, "calls": calls}
        review = A.assess(result.get("content"), capsule["context"])
        return {"accepted": review["status"] == "valid", "model_status": review["status"],
                "selection": review["decision"] or {"action": "hold"},
                "final_content": result.get("content"), "review": review, "calls": calls}
    raise AssertionError("닫히지 않은 호출 상한")


def advance(reference: dict, api_root: Path, complete: Callable[[list, dict], dict], *,
            api_cap: int = 12, now_utc: Callable[[], datetime] | None = None) -> dict:
    folder, capsule = prepared(reference)
    binding = {"run_id": "agentic-" + reference["sha256"][:24],
               "source_anchor": capsule["source_anchor"], "protocol_sha256": capsule["protocol_sha256"],
               "checkpoint_sha256": capsule["context"]["snapshot"]["inputs"]["packet"]
                                    ["input_bindings"]["case"]["checkpoint_sha256"],
               "snapshot_sha256": reference["sha256"],
               "initial_messages_sha256": J.digest(capsule["messages"])}
    journal = J.DecisionJournal(folder / "journal", api_root, binding, PARAMS, api_cap=api_cap)
    result = journal.advance(lambda completion: engine(capsule, completion),
        lambda messages, params: O._provider_result(complete(messages, params)),
        now_utc or (lambda: datetime.now(timezone.utc)))  # noqa: UP017 - Python 3.10
    return {"schema": "agentic_investigation_call_result_v1", "capsule_sha256": reference["sha256"],
            "journal": result, "approval_granted": False}
