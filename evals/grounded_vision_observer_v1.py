"""근거 프레임을 요구하는 Nemotron 3.5 영상 관찰 계약과 재개 caller.

영상 관찰은 ``model_interpretation``이며 원인 확정·교정·회복 판정에 쓰지 않는다.
모델에는 V01의 공개 view만 보내고, 실행·checkpoint provenance는 원장 결합에만 남긴다.
이 모듈은 provider 함수를 주입받는 CPU caller만 제공한다. 실제 endpoint 실행기는
사전등록 protocol과 source 결합을 검증하는 별도 WSL worker에서 이 caller를 사용한다.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import feedback_model_call as PROVIDER
import model_decision_journal_v1 as JOURNAL
import nemotron_endpoint_ledger_v3 as ENDPOINT
import vision_evidence_contract_v1 as EVIDENCE

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ACCEPTED_PROVIDER_MODELS = frozenset(ENDPOINT.ALLOWED_PROVIDER_IDS[MODEL])
OBSERVATION_SCHEMA = "grounded_vision_observation_v1"
REQUEST_SCHEMA = "grounded_vision_request_v1"
STATES = frozenset({"yes", "no", "unclear"})
BEHAVIORS = ("walking", "in_place", "falling")
OUTPUT_FIELDS = {*BEHAVIORS, "evidence"}
MAX_CALLS = 3
DEFAULT_API_CAP = 6
REQUEST_PARAMS = {
    "model": MODEL,
    "temperature": 0,
    "max_tokens": 1024,
    "stream": False,
    "extra_body": {"reasoning_effort": "none"},
}

OUTPUT_INSTRUCTION = """Return one JSON object with exactly walking, in_place, falling, and evidence.
walking, in_place, and falling must each be one of: yes, no, unclear.
evidence must contain exactly walking, in_place, and falling. Each value is a nonempty list of
objects with exactly frame_id and timestamp_s copied from the supplied frame labels.
Use unclear when the supplied evidence does not support yes or no. Do not diagnose a fault,
declare recovery, choose a probe, propose a repair, add fields, or invent frames or timestamps.
Output JSON only."""

FEEDBACK = {
    "F_EMPTY": "Return one nonempty JSON object.",
    "F_JSON": "Return valid JSON with no prose or code fence.",
    "F_FIELDS": "Use exactly walking, in_place, falling, and evidence.",
    "F_STATE": "Each behavior must be yes, no, or unclear.",
    "F_EVIDENCE": "Evidence must have one nonempty citation list for each behavior.",
    "F_CITATION": "Each citation must contain exactly frame_id and timestamp_s.",
    "S_FRAME_UNKNOWN": "Copy only frame IDs and timestamps that exist in the supplied input.",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("F_FIELDS")
        result[key] = value
    return result


def _frame_map(contract_path: Path) -> dict[str, float]:
    contract = EVIDENCE.verify(contract_path)
    return {row["frame_id"]: float(row["timestamp_s"]) for row in contract["frames"]}


def request_view(contract_path: Path) -> dict[str, Any]:
    """V01 공개 view에 고정 출력 계약만 덧붙인다."""
    public = EVIDENCE.public_model_view(contract_path)
    messages = deepcopy(public["messages"])
    messages.append({"role": "user", "content": OUTPUT_INSTRUCTION})
    return {
        "schema": REQUEST_SCHEMA,
        "contract_sha256": public["contract_sha256"],
        "messages": messages,
        "evidence_accounting": deepcopy(public["evidence_accounting"]),
    }


def _request_snapshot(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA,
        "contract_sha256": view["contract_sha256"],
        "evidence_accounting": deepcopy(view["evidence_accounting"]),
    }


def validate_observation(contract_path: Path, content: object) -> dict[str, Any]:
    """모델 JSON과 실제 frame ID·시각을 엄격히 대조한다."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("F_EMPTY")
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _raw: (_ for _ in ()).throw(ValueError("F_JSON")),
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("F_JSON") from error
    if not isinstance(value, dict):
        raise ValueError("F_JSON")  # noqa: TRY004 - 모델 계약 오류 코드를 한 형식으로 반환한다.
    if set(value) != OUTPUT_FIELDS:
        raise ValueError("F_FIELDS")
    if any(type(value[name]) is not str or value[name] not in STATES for name in BEHAVIORS):
        raise ValueError("F_STATE")
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(BEHAVIORS):
        raise ValueError("F_EVIDENCE")
    frames = _frame_map(contract_path)
    checked: dict[str, list[dict[str, Any]]] = {}
    for behavior in BEHAVIORS:
        citations = evidence[behavior]
        if not isinstance(citations, list) or not citations:
            raise ValueError("F_EVIDENCE")
        rows = []
        seen = set()
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"frame_id", "timestamp_s"}:
                raise ValueError("F_CITATION")
            frame_id = citation["frame_id"]
            timestamp = citation["timestamp_s"]
            if (type(frame_id) is not str or type(timestamp) not in (int, float)
                    or not math.isfinite(timestamp)):
                raise ValueError("F_CITATION")
            if frame_id not in frames or float(timestamp) != frames[frame_id]:
                raise ValueError("S_FRAME_UNKNOWN")
            key = (frame_id, float(timestamp))
            if key in seen:
                raise ValueError("F_CITATION")
            seen.add(key)
            rows.append({"frame_id": frame_id, "timestamp_s": float(timestamp)})
        checked[behavior] = rows
    return {
        "walking": value["walking"],
        "in_place": value["in_place"],
        "falling": value["falling"],
        "evidence": checked,
    }


def _feedback(code: str) -> dict[str, str]:
    if code not in FEEDBACK:
        raise ValueError("등록되지 않은 관찰 검증 코드")
    return {
        "role": "user",
        "content": "\n".join((
            "The previous JSON was rejected.",
            f"- {code}: {FEEDBACK[code]}",
            "Return one corrected JSON object only.",
        )),
    }


def _provider_result(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {"error_type": "ProviderResultTypeError"}
    view = ENDPOINT.provider_view(raw, MODEL)
    allowed = {
        "response_id", "provider_model", "finish_reason", "content", "reasoning_content",
        "input_tokens", "output_tokens", "total_tokens", "wall_s", "error_type",
        "infra_error", "status_code",
    }
    safe = {key: deepcopy(value) for key, value in view.items() if key in allowed}
    bad = []
    for key in ("content", "reasoning_content", "response_id", "provider_model", "finish_reason"):
        if safe.get(key) is not None and not isinstance(safe[key], str):
            safe[key] = None
            bad.append(key)
    for key in ("error_type", "infra_error"):
        if safe.get(key) is not None and (not isinstance(safe[key], str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", safe[key]) is None):
            safe[key] = None
            bad.append(key)
    for key in ("input_tokens", "output_tokens", "total_tokens", "status_code"):
        value: Any = safe.get(key)
        if value is not None and (type(value) is not int or value < 0):
            safe[key] = None
            bad.append(key)
    wall_s: Any = safe.get("wall_s")
    if wall_s is not None and (type(wall_s) not in (int, float)
                               or not math.isfinite(wall_s)
                               or wall_s < 0):
        safe["wall_s"] = None
        bad.append("wall_s")
    if bad:
        safe.update(error_type="ProviderResponseContractError", infra_error=None,
                    response_contract_fields=sorted(bad))
    for key in ("provider_model_raw", "provider_identity_registered", "identity_policy"):
        if key in view:
            safe[key] = deepcopy(view[key])
    return safe


def default_complete(messages: list[dict], params: dict) -> dict[str, Any]:
    """기존 NVIDIA 환경의 SDK retry 0 provider 호출 경계."""
    return PROVIDER.openai_complete(messages, params)


def _safe_complete(
    complete: Callable[[list[dict], dict], dict],
    messages: list[dict],
) -> dict[str, Any]:
    try:
        raw = complete(deepcopy(messages), deepcopy(REQUEST_PARAMS))
    except JOURNAL.JournalError:
        raise
    except Exception as error:  # noqa: BLE001 - provider 경계에는 예외 class만 보존한다.
        raw = {"error_type": type(error).__name__}
    return _provider_result(raw)


def _transport(result: dict[str, Any]) -> str:
    if result.get("provider_model") is not None and result.get("provider_model") != MODEL:
        return "unknown"
    if result.get("infra_error"):
        return "infra_error"
    if result.get("error_type") or result.get("provider_model") != MODEL:
        return "unknown"
    content = result.get("content")
    if content is None or (isinstance(content, str) and not content.strip()) or result.get("finish_reason") == "length":
        return "infra_error"
    if not isinstance(content, str) or result.get("finish_reason") != "stop":
        return "unknown"
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = result.get(name)
        if value is not None and (type(value) is not int or value < 0):
            return "unknown"
    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    total_tokens = result.get("total_tokens")
    if (type(input_tokens) is int and type(output_tokens) is int and type(total_tokens) is int
            and input_tokens + output_tokens != total_tokens):
        return "unknown"
    wall_s: Any = result.get("wall_s")
    if type(wall_s) not in (int, float) or not math.isfinite(wall_s) or wall_s < 0:
        return "unknown"
    return "completed"


def engine(
    contract_path: Path,
    complete: Callable[[list[dict], dict], dict],
) -> dict[str, Any]:
    """인프라 재시도 한 번과 의미 피드백 한 번 뒤 유효 관찰 또는 HOLD를 반환한다."""
    base = request_view(contract_path)["messages"]
    sent = base
    calls = []
    infra_retried = False
    feedback_used = False
    while len(calls) < MAX_CALLS:
        result = _safe_complete(complete, sent)
        transport = _transport(result)
        row = {"attempt": len(calls) + 1, "transport": transport, "result": result}
        calls.append(row)
        if transport == "infra_error" and not infra_retried and len(calls) < MAX_CALLS:
            infra_retried = True
            continue
        if transport != "completed":
            return {
                "status": "terminal_hold",
                "reason": f"transport_{transport}",
                "observation": None,
                "provenance": _provenance(contract_path, calls),
                "calls": calls,
            }
        try:
            observation = validate_observation(contract_path, result.get("content"))
        except ValueError as error:
            code = str(error)
            row["validation"] = {"status": "invalid", "code": code}
            if not feedback_used and code in FEEDBACK and len(calls) < MAX_CALLS:
                feedback_used = True
                sent = [*base, {"role": "assistant", "content": result.get("content")}, _feedback(code)]
                continue
            return {
                "status": "terminal_hold",
                "reason": "observation_invalid",
                "observation": None,
                "provenance": _provenance(contract_path, calls),
                "calls": calls,
            }
        row["validation"] = {"status": "valid"}
        return {
            "status": "observation_ready",
            "reason": "grounded_observation_valid",
            "observation": observation,
            "provenance": _provenance(contract_path, calls),
            "calls": calls,
        }
    return {
        "status": "terminal_hold",
        "reason": "call_budget_exhausted",
        "observation": None,
        "provenance": _provenance(contract_path, calls),
        "calls": calls,
    }


def _provenance(contract_path: Path, calls: list[dict]) -> dict[str, Any]:
    raw_models = sorted({
        row["result"]["provider_model_raw"]
        for row in calls
        if isinstance(row.get("result", {}).get("provider_model_raw"), str)
    })
    return {
        "evidence_type": "model_interpretation",
        "used_for_decision": False,
        "contract_sha256": EVIDENCE.sha256(Path(contract_path)),
        "requested_model": MODEL,
        "provider_models_raw": raw_models,
        "provider_identity_policy": ENDPOINT.IDENTITY_POLICY,
    }


def checked_binding(contract_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """V01 계약과 caller 첫 요청에 결합된 DecisionJournal binding을 검사한다."""
    contract = EVIDENCE.verify(contract_path)
    view = request_view(contract_path)
    expected = deepcopy(binding)
    if expected.get("checkpoint_sha256") != contract["provenance"]["checkpoint"]["sha256"]:
        raise ValueError("binding checkpoint_sha256가 V01 계약과 다르다")
    if expected.get("snapshot_sha256") != JOURNAL.digest(_request_snapshot(view)):
        raise ValueError("binding snapshot_sha256가 관찰 request와 다르다")
    if expected.get("initial_messages_sha256") != JOURNAL.digest(view["messages"]):
        raise ValueError("binding initial_messages_sha256가 첫 messages와 다르다")
    JOURNAL._binding(expected)
    return expected


def run(
    folder: Path,
    contract_path: Path,
    binding: dict[str, Any],
    complete: Callable[[list[dict], dict], dict],
    *,
    api_root: Path | None = None,
    now_utc: Callable[[], datetime] | None = None,
    api_cap: int = DEFAULT_API_CAP,
) -> dict[str, Any]:
    """주입 provider로 관찰을 실행하며 완료 응답 재생·pending·상한 경계를 유지한다."""
    if not callable(complete):
        raise TypeError("complete provider가 필요하다")
    if type(api_cap) is not int or not 1 <= api_cap <= DEFAULT_API_CAP:
        raise ValueError("관찰 API 상한은 사전등록 최대 6회 안의 양의 정수여야 한다")
    root = Path(folder)
    view = request_view(contract_path)
    checked = checked_binding(contract_path, binding)
    journal = JOURNAL.DecisionJournal(
        root=root / "decision",
        api_root=Path(api_root) if api_root is not None else root / "api",
        binding=checked,
        params=REQUEST_PARAMS,
        retry_delay_s=30,
        api_cap=api_cap,
    )
    clock = now_utc if now_utc is not None else lambda: datetime.now(timezone.utc)  # noqa: UP017
    journal_result = journal.advance(
        lambda completion: engine(contract_path, completion),
        complete,
        clock,
    )
    if journal_result["engine_result"] is None:
        return journal_result
    engine_result = journal_result["engine_result"]
    status = "observation_ready" if engine_result.get("status") == "observation_ready" else journal_result["status"]
    return {
        "status": status,
        "engine_result": engine_result,
        "api": journal_result["api"],
        "request": {
            "schema": REQUEST_SCHEMA,
            "contract_sha256": view["contract_sha256"],
            "messages_sha256": JOURNAL.digest(view["messages"]),
            "params_sha256": JOURNAL.digest(REQUEST_PARAMS),
        },
    }


def binding_template(contract_path: Path, *, run_id: str, source_anchor: str,
                     protocol_sha256: str) -> dict[str, str]:
    """사전등록 controller가 protocol/source 값을 채워 봉인할 binding 형태를 만든다."""
    contract = EVIDENCE.verify(contract_path)
    view = request_view(contract_path)
    return {
        "run_id": run_id,
        "source_anchor": source_anchor,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": contract["provenance"]["checkpoint"]["sha256"],
        "snapshot_sha256": JOURNAL.digest(_request_snapshot(view)),
        "initial_messages_sha256": JOURNAL.digest(view["messages"]),
    }
