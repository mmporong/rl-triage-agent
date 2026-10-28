"""CPU provider fixture만 사용한 실제 v2 동결·journal 회귀."""
from __future__ import annotations

import base64
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"evals"), str(ROOT/"src")]
import agentic_investigation_call_v2 as CALL
import test_selection_comparison_controller_v1 as F


def result(content=None):
    return {"content": content or '{"action":"hold","target":null,"fact_ids":["config:action_scale"]}',
            "finish_reason": "stop", "provider_model": CALL.O.MODEL, "wall_s": 0.01}


def frozen(root, name="call"):
    context = CALL.A.prepare(F.snapshot(root, "case", "model_adaptive"), goal="자료 조사")
    return CALL.freeze(context, root/name, source_anchor="a"*40, protocol_sha256="b"*64)


def no_call(*_):
    pytest.fail("CPU 재생에서 provider 추가 호출")


def test_frozen_wire_uses_v2_and_replay_uses_no_provider(tmp_path):
    reference = frozen(tmp_path)
    sent = []

    def provider(messages, params):
        sent.append((messages, params))
        return result()

    first = CALL.advance(reference, tmp_path/"api", provider)
    assert CALL.advance(reference, tmp_path/"api", no_call) == first
    assert len(sent) == 1 and sent[0][0][0]["content"] == CALL.A.SYSTEM
    _, capsule = CALL.prepared(reference)
    assert sent[0][1] == capsule["params"] == CALL.params_for_messages(sent[0][0])
    assert first["params_sha256"] == CALL.J.digest(sent[0][1])
    assert CALL.J._read(Path(reference["path"]).parent/"journal/manifest.json")["params_sha256"] == first["params_sha256"]
    assert CALL.J._read(tmp_path/"api/reservations/0001.json")["params_sha256"] == first["params_sha256"]
    assert CALL.J.digest(sent[0][0]) == first["messages_sha256"]
    assert first["approval_granted"] is False
    assert CALL.V1.A.SYSTEM != CALL.A.SYSTEM


def test_semantic_invalid_stops_once_without_feedback(tmp_path):
    reference = frozen(tmp_path)
    sent = []

    def provider(*_):
        sent.append(1)
        return result('{"action":"read_evidence","target":null,"fact_ids":["config:action_scale"]}')

    answer = CALL.advance(reference, tmp_path/"api", provider)
    assert len(sent) == 1 and answer["journal"]["status"] == "terminal_hold"
    assert answer["journal"]["engine_result"]["model_status"] == "invalid"


def test_pending_and_infrastructure_retry_keep_original_journal_rules(tmp_path):
    reference = frozen(tmp_path)

    def crash(*_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        CALL.advance(reference, tmp_path/"api", crash)
    assert CALL.advance(reference, tmp_path/"api", no_call)["journal"]["status"] == "blocked_pending"
    second = frozen(tmp_path/"retry")
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
    calls = []

    def provider(*_):
        calls.append(1)
        return {"infra_error": "TimeoutError"} if len(calls) == 1 else result()

    assert CALL.advance(second, tmp_path/"api-retry", provider, now_utc=lambda: now)["journal"]["status"] == "paused_infra"
    final = CALL.advance(second, tmp_path/"api-retry", provider, now_utc=lambda: now+timedelta(seconds=31))
    assert final["journal"]["api"]["reserved"] == 2 and len(calls) == 2


def test_actual_shared_cap_has_only_nine_calls_after_three(tmp_path):
    api = tmp_path/"api"
    for number in range(3):
        old = CALL.V1.freeze(CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="CPU 기존 소비"),
            tmp_path/f"v1-{number}", source_anchor="a"*40, protocol_sha256="b"*64)
        CALL.V1.advance(old, api, lambda *_: result())
    sent = []
    for number in range(10):
        reference = frozen(tmp_path, f"v2-{number}")

        def provider(*_):
            sent.append(1)
            return result()

        answer = CALL.advance(reference, api, provider)
    assert len(sent) == 9 and answer["journal"]["api"] == {"reserved": 12, "pending": 0, "remaining": 0}
    assert answer["journal"]["engine_result"]["stop_reason"] == "budget_exhausted"


def test_image_metadata_detail_and_system_tampering_are_rejected(tmp_path):
    context = CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="영상 조사",
        extra_facts={"vision:frame:f01": {"evidence_type": "raw_frame", "timestamp_s": 1,
                                         "source_group": "visual_source_1"}})
    messages = CALL.V1.A.messages(context)
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
        {"type": "text", "text": '{"frame_id":"f01","timestamp_s":1}'},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+base64.b64encode(b"\xff\xd8cpu-fixture\xff\xd9").decode(), "detail": "low"}}]
    converted = CALL.from_v1(context, messages)
    reference = CALL.freeze(context, tmp_path/"visual", source_anchor="a"*40, protocol_sha256="b"*64, messages=converted)
    assert CALL.prepared(reference)[1]["messages"][1] == messages[1]
    for changed in ("system", "timestamp", "detail"):
        value = deepcopy(converted)
        if changed == "system":value[0]["content"] = CALL.V1.A.SYSTEM
        elif changed == "timestamp":value[1]["content"][1]["text"] = '{"frame_id":"f01","timestamp_s":2}'
        else:value[1]["content"][2]["image_url"]["detail"] = "high"
        with pytest.raises(ValueError):
            CALL.freeze(context, tmp_path/changed, source_anchor="a"*40, protocol_sha256="b"*64, messages=value)
    path = Path(reference["path"])
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="SHA"):
        CALL.advance(reference, tmp_path/"api", no_call)


def test_schema_params_tampering_is_rejected_even_with_new_raw_file_hash(tmp_path):
    reference = frozen(tmp_path)
    path = Path(reference["path"])
    capsule = CALL.J._read(path)
    capsule["params"]["response_format"]["json_schema"]["schema"]["properties"]["fact_ids"]["items"]["enum"].append("unknown:fact")
    capsule["params_sha256"] = CALL.J.digest(capsule["params"])
    path.write_bytes(CALL.J._bytes(capsule))
    changed = {"path": str(path), "sha256": CALL.J.file_sha(path)}
    with pytest.raises(ValueError, match="params"):
        CALL.advance(changed, tmp_path/"api", no_call)


def test_keyword_400_is_recorded_as_unavailable_infrastructure_spend_without_retry(tmp_path):
    reference = frozen(tmp_path)
    sent = []

    def provider(*_):
        sent.append(1)
        return {"error_type": "BadRequestError", "status_code": 400, "wall_s": 0.01}

    answer = CALL.advance(reference, tmp_path/"api", provider)
    assert len(sent) == 1 and answer["journal"]["api"]["reserved"] == 1
    engine = answer["journal"]["engine_result"]
    assert engine["model_status"] == "unavailable" and engine["failure_category"] == "infrastructure_request_rejection"
    assert engine["stop_reason"] == "schema_request_rejected"
    assert CALL.J._read(tmp_path/"api/responses/0001.json")["result"]["status_code"] == 400
    assert CALL.advance(reference, tmp_path/"api", no_call) == answer
