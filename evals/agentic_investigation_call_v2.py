"""v1 history 동결·이미지 검증·journal을 사용하는 명시적 v2 메시지 어댑터."""
from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import agentic_investigation_call_v1 as V1
import agentic_investigation_contract_v2 as A

J, O = V1.J, V1.O
PARAMS = deepcopy(V1.PARAMS)
SCHEMA = "agentic_investigation_call_v2"


def params_for(context: dict) -> dict:
    return {**deepcopy(PARAMS), "response_format": A.response_format(context)}


def params_for_messages(messages: list) -> dict:
    """WSL에서 같은 공개 user view로 판단별 params를 다시 계산한다."""
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0] != {"role": "system", "content": A.SYSTEM}
            or messages[1].get("role") != "user"):
        raise ValueError("v2 provider 메시지 경계 불일치")
    content = messages[1]["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    view = json.loads(text, object_pairs_hook=A.V._unique_object)
    return {**deepcopy(PARAMS), "response_format": A.response_format_from_view(view)}


def from_v1(context: dict, messages: list) -> list:
    """수용된 v1 evidence adapter의 user text/이미지를 검증하고 v2 SYSTEM에 결합한다."""
    outgoing = V1._messages(A.checked(context), messages)
    outgoing[0] = {"role": "system", "content": A.SYSTEM}
    return _messages(context, outgoing)


def _messages(context: dict, supplied: list | None) -> list:
    outgoing = deepcopy(A.messages(context) if supplied is None else supplied)
    if not isinstance(outgoing, list) or not outgoing or outgoing[0] != {"role": "system", "content": A.SYSTEM}:
        raise ValueError("v2 system prompt 결합 불일치")
    mechanical = deepcopy(outgoing)
    mechanical[0] = V1.A.messages(context)[0]
    V1._messages(context, mechanical)
    content = outgoing[1]["content"]
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "image_url" and block["image_url"].get("detail") != "low":
                raise ValueError("v2 이미지 detail=low가 필요하다")
    return outgoing


def freeze(context: dict, folder: Path, *, source_anchor: str, protocol_sha256: str,
           messages: list | None = None) -> dict:
    outgoing = _messages(context, messages)
    mechanical = deepcopy(outgoing)
    mechanical[0] = V1.A.messages(context)[0]
    # v1 capsule은 history 검증용 호스트 산출물이다. Provider에는 아래 v2 messages만 전달한다.
    folder = Path(folder).resolve()
    base = V1.freeze(context, folder / "history-binding-v1", source_anchor=source_anchor,
                     protocol_sha256=protocol_sha256, messages=mechanical)
    _, frozen = V1.prepared(base)
    capsule = {"schema": SCHEMA, "base_reference": base, "context": frozen["context"],
               "messages": outgoing, "messages_sha256": J.digest(outgoing),
               "system_sha256": J.digest(A.SYSTEM), "source_anchor": source_anchor,
               "protocol_sha256": protocol_sha256, "params": params_for(frozen["context"])}
    capsule["params_sha256"] = J.digest(capsule["params"])
    path = folder / "capsule.json"
    J._write_new(path, capsule)
    return {"path": str(path), "sha256": J.file_sha(path)}


def prepared(reference: dict) -> tuple[Path, dict]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("v2 capsule ref 형식")
    path = Path(reference["path"])
    if path != path.resolve() or J.file_sha(path) != reference["sha256"]:
        raise ValueError("v2 capsule SHA/경로 불일치")
    capsule = J._read(path)
    expected_base = path.parent / "history-binding-v1/capsule.json"
    if Path(capsule["base_reference"]["path"]) != expected_base:
        raise ValueError("v2 history capsule 경로 불일치")
    _, base = V1.prepared(capsule["base_reference"])
    if (capsule.get("schema") != SCHEMA or capsule.get("params") != params_for(capsule["context"])
            or capsule["context"] != base["context"] or capsule["source_anchor"] != base["source_anchor"]
            or capsule["protocol_sha256"] != base["protocol_sha256"]
            or capsule["system_sha256"] != J.digest(A.SYSTEM)
            or capsule["messages_sha256"] != J.digest(capsule["messages"])
            or capsule["params_sha256"] != J.digest(capsule["params"])
            or capsule["params"] != params_for_messages(capsule["messages"])):
        raise ValueError("v2 history/context/system/params 결합 불일치")
    outgoing = _messages(capsule["context"], capsule["messages"])
    mechanical = deepcopy(outgoing)
    mechanical[0] = V1.A.messages(capsule["context"])[0]
    if mechanical != base["messages"]:
        raise ValueError("v2 공개 user 입력이 history capsule과 다르다")
    return path.parent, capsule


def _engine(capsule: dict, completion: Callable[[list, dict], dict]) -> dict:
    def registered_completion(messages: list, base_params: dict) -> dict:
        if base_params != PARAMS:
            raise ValueError("v1 engine base params가 바뀌었다")
        return completion(messages, deepcopy(capsule["params"]))

    result = V1.engine(capsule, registered_completion)
    if result["calls"][-1]["result"].get("status_code") == 400:
        result.update(stop_reason="schema_request_rejected", model_status="unavailable",
                      failure_category="infrastructure_request_rejection")
    return result


def advance(reference: dict, api_root: Path, complete: Callable[[list, dict], dict], *,
            api_cap: int = 12, now_utc: Callable[[], datetime] | None = None) -> dict:
    folder, capsule = prepared(reference)
    binding = {"run_id": "agentic-v2-" + reference["sha256"][:24],
               "source_anchor": capsule["source_anchor"], "protocol_sha256": capsule["protocol_sha256"],
               "checkpoint_sha256": capsule["context"]["snapshot"]["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"],
               "snapshot_sha256": reference["sha256"], "initial_messages_sha256": J.digest(capsule["messages"])}
    journal = J.DecisionJournal(folder / "journal", api_root, binding, capsule["params"], api_cap=api_cap)
    result = journal.advance(lambda completion: _engine(capsule, completion),
        lambda messages, params: O._provider_result(complete(messages, params)),
        now_utc or (lambda: datetime.now(timezone.utc)))  # noqa: UP017 - Python 3.10
    return {"schema": "agentic_investigation_call_result_v2", "capsule_sha256": reference["sha256"],
            "messages_sha256": capsule["messages_sha256"], "params_sha256": capsule["params_sha256"],
            "journal": result, "approval_granted": False}
