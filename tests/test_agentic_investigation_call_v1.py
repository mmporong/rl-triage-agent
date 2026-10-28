"""기존 journal을 사용하는 새 행동 호출기의 재시작·상한·입력 결합 회귀."""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import agentic_investigation_call_v1 as CALL
import test_selection_comparison_controller_v1 as F


def frozen(tmp_path, name="call", *, goal="학습 실패 조사"):
    context = CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal=goal)
    return CALL.freeze(context, tmp_path / name, source_anchor="a" * 40, protocol_sha256="b" * 64)


def result(content=None):
    content = content or json.dumps({"action": "run_probe", "target": "P_reward",
                                    "fact_ids": ["config:action_scale"]})
    return {"provider_model": CALL.O.MODEL, "finish_reason": "stop", "content": content,
            "input_tokens": 10, "output_tokens": 3, "total_tokens": 13, "wall_s": 0.01}


def no_call(*_):
    raise AssertionError("추가 provider 호출")


def test_terminal_replay_does_not_call_provider_and_does_not_approve(tmp_path):
    ref = frozen(tmp_path)
    sent = []

    def complete(messages, params):
        sent.append((messages, params))
        return result()

    first = CALL.advance(ref, tmp_path / "api", complete)
    replay = CALL.advance(ref, tmp_path / "api", no_call)
    assert first == replay and len(sent) == 1
    assert first["journal"]["status"] == "choice_ready"
    assert first["journal"]["api"] == {"reserved": 1, "pending": 0, "remaining": 11}
    assert first["approval_granted"] is False


def test_semantic_invalid_is_terminal_hold_without_retry(tmp_path):
    ref = frozen(tmp_path)
    sent = []

    def complete(*_):
        sent.append(1)
        return result('{"action":"run_shell","target":"bash","fact_ids":["config:action_scale"]}')

    answer = CALL.advance(ref, tmp_path / "api", complete)
    assert len(sent) == 1
    assert answer["journal"]["status"] == "terminal_hold"
    assert answer["journal"]["engine_result"]["model_status"] == "invalid"
    assert CALL.advance(ref, tmp_path / "api", no_call) == answer


def test_pending_after_crash_is_not_resent(tmp_path):
    ref = frozen(tmp_path)

    def crash(*_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        CALL.advance(ref, tmp_path / "api", crash)
    answer = CALL.advance(ref, tmp_path / "api", no_call)
    assert answer["journal"]["status"] == "blocked_pending"
    assert answer["journal"]["api"]["pending"] == 1


def test_infrastructure_retry_waits_and_consumes_shared_cap(tmp_path):
    ref = frozen(tmp_path)
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
    sent = []

    def complete(*_):
        sent.append(1)
        return {"infra_error": "TimeoutError"} if len(sent) == 1 else result()

    paused = CALL.advance(ref, tmp_path / "api", complete, now_utc=lambda: now)
    assert paused["journal"]["status"] == "paused_infra" and len(sent) == 1
    answer = CALL.advance(ref, tmp_path / "api", complete, now_utc=lambda: now + timedelta(seconds=31))
    assert answer["journal"]["status"] == "choice_ready" and len(sent) == 2
    assert answer["journal"]["api"]["reserved"] == 2


def test_budget_exhaustion_is_hold_without_new_request(tmp_path):
    first = frozen(tmp_path)
    CALL.advance(first, tmp_path / "api", lambda *_: result(), api_cap=1)
    second = frozen(tmp_path, "next", goal="다른 판단")
    answer = CALL.advance(second, tmp_path / "api", no_call, api_cap=1)
    assert answer["journal"]["status"] == "terminal_hold"
    assert answer["journal"]["api"]["reserved"] == 1


def test_frozen_history_is_independent_of_future_original_ledger(tmp_path):
    ref = frozen(tmp_path)
    (tmp_path / "case/epoch1").mkdir(parents=True, exist_ok=True)
    original = tmp_path / "case/epoch1/probes.ledger.jsonl"
    original.write_text('{"changed": true}\n', encoding="utf-8")
    answer = CALL.advance(ref, tmp_path / "api", lambda *_: result())
    assert answer["journal"]["status"] == "choice_ready"


def test_mutated_capsule_is_rejected_before_provider(tmp_path):
    ref = frozen(tmp_path)
    path = Path(ref["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        CALL.advance(ref, tmp_path / "api", no_call)


def test_visual_messages_keep_action_system_and_current_facts(tmp_path):
    context = CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="영상 조사",
        extra_facts={"vision:frame:f01": {"evidence_type": "raw_frame", "timestamp_s": 0,
                                        "source_group": "visual_source_1"}})
    messages = CALL.A.messages(context)
    url = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8fixture\xff\xd9").decode()
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
                               {"type": "text", "text": json.dumps({"frame_id": "f01", "timestamp_s": 0})},
                               {"type": "image_url", "image_url": {"url": url}}]
    ref = CALL.freeze(context, tmp_path / "visual", source_anchor="a" * 40,
                      protocol_sha256="b" * 64, messages=messages)
    sent = []

    def complete(messages, _):
        sent.append(messages)
        return result()

    CALL.advance(ref, tmp_path / "api", complete)
    assert sent[0][0]["content"] == CALL.A.SYSTEM
    assert sent[0][1]["content"][2]["image_url"]["url"] == url
    messages[0]["content"] = "legacy probe-only system"
    with pytest.raises(ValueError, match="system"):
        CALL.freeze(context, tmp_path / "wrong", source_anchor="a" * 40,
                    protocol_sha256="b" * 64, messages=messages)


def test_adapter_text_blocks_without_images_are_supported(tmp_path):
    context = CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="자료 조회")
    messages = CALL.A.messages(context)
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]}]
    ref = CALL.freeze(context, tmp_path / "text-block", source_anchor="a" * 40,
                      protocol_sha256="b" * 64, messages=messages)
    assert CALL.prepared(ref)[1]["messages"] == messages


def test_frame_timestamp_mismatch_is_rejected_before_freeze(tmp_path):
    context = CALL.A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="영상 조사",
        extra_facts={"vision:frame:f01": {"evidence_type": "raw_frame", "timestamp_s": 1,
                                        "source_group": "visual_source_1"}})
    messages = CALL.A.messages(context)
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
        {"type": "text", "text": '{"frame_id":"f01","timestamp_s":2}'},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9g="}}]
    with pytest.raises(ValueError, match="시각"):
        CALL.freeze(context, tmp_path / "mismatched", source_anchor="a" * 40,
                    protocol_sha256="b" * 64, messages=messages)
