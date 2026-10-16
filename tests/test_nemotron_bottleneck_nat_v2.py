import asyncio
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import nemotron_bottleneck_nat_v2 as N
import nemotron_bottleneck_ledger_v2 as L


def test_sse_collects_all_tool_argument_chunks():
    chunks = [
        {"model": L.CANDIDATE, "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "submit", "arguments": '{"x":'}}]}}]},
        {"model": L.CANDIDATE, "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'null}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
    ]
    raw = ("\n\n".join("data: " + json.dumps(c) for c in chunks) + "\n\ndata: [DONE]\n\n").encode()
    result = N.decode_wire(raw, 200)
    assert result["tool_calls"] == [{"name": "submit", "arguments": '{"x":null}'}]
    assert result["total_tokens"] == 14


def test_json_collects_message_tool_calls_and_usage():
    raw = json.dumps({"model": L.BASELINE, "choices": [{"message": {"content": None,
        "tool_calls": [{"function": {"name": "submit", "arguments": "{}"}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2,
                                                      "total_tokens": 9}}).encode()
    result = N.decode_wire(raw, 200)
    assert result["provider_model"] == L.BASELINE
    assert result["tool_calls"] == [{"name": "submit", "arguments": "{}"}]
    assert result["finish_reason"] == "tool_calls" and result["input_tokens"] == 7


def test_http_error_body_not_saved():
    result = N.decode_wire(b"account=secret", 404)
    assert result["status_code"] == 404
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(("choice", "error_type"), [
    ({"message": {"content": ""}, "finish_reason": "stop"}, "EmptyResponse"),
    ({"message": {"content": "partial"}, "finish_reason": "length"}, "LengthFinish"),
])
def test_empty_and_length_responses_are_marked_as_infrastructure(choice, error_type):
    result = N.decode_wire(json.dumps({"model": L.CANDIDATE, "choices": [choice]}).encode(), 200)
    assert result["error_type"] == error_type and result["infra_error"] is True


def test_transport_records_before_sending_and_retains_response(tmp_path, monkeypatch):
    import httpx
    book = L.Ledger(tmp_path, {"test": True})
    async def fake_send(client, request, *args, **kwargs):
        assert len(book.audit()) == 1
        return httpx.Response(200, request=request, json={"model": L.CANDIDATE,
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    async def run():
        async with httpx.AsyncClient() as client:
            with N.metered_http(book, "case", L.CANDIDATE):
                response = await client.post("https://integrate.api.nvidia.com/v1/chat/completions", json={
                    "model": L.CANDIDATE, "temperature": 0, "reasoning_effort": "none",
                    "messages": [{"role": "user", "content": "private-request-marker"}]})
                assert response.json()["model"] == L.CANDIDATE
    asyncio.run(run())
    assert book.audit()[0]["response"]["result"]["content"] == "done"
    assert "private-request-marker" not in "".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))


def test_unmetered_endpoint_and_sync_send_are_blocked(tmp_path):
    import httpx
    book = L.Ledger(tmp_path, {})
    with N.metered_http(book, "case", L.CANDIDATE):
        with pytest.raises(RuntimeError):
            httpx.Client().get("https://example.com")
        async def run():
            async with httpx.AsyncClient() as client:
                with pytest.raises(ValueError):
                    await client.post("https://example.com", json={})
        asyncio.run(run())
    assert book.audit() == []


def test_provider_model_mismatch_is_saved_before_workflow_stops(tmp_path, monkeypatch):
    import httpx
    book = L.Ledger(tmp_path, {})

    async def fake_send(client, request, *args, **kwargs):
        return httpx.Response(200, request=request, json={"model": L.BASELINE,
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    async def run():
        async with httpx.AsyncClient() as client:
            with N.metered_http(book, "case", L.CANDIDATE):
                with pytest.raises(ValueError, match="응답 모델"):
                    await client.post("https://integrate.api.nvidia.com/v1/chat/completions", json={
                        "model": L.CANDIDATE, "temperature": 0, "reasoning_effort": "none"})
    asyncio.run(run())
    saved = book.audit()[0]["response"]["result"]
    assert saved["error_type"] == "ProviderModelMismatch" and saved["infra_error"] is False


def _trace_row(tool, value=None, *, error=None, claimed_sha=None):
    text = None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {"tool": tool, "error": error, "result_head": text,
            "result_sha256": claimed_sha if claimed_sha is not None else (
                None if text is None else N.W1.text_sha(text))}


def test_tool_trace_classification_separates_contract_failure_from_limit_and_corruption():
    rows = [
        _trace_row("e4b2_check_decision", {"checked": True, "ok": True, "reasons": []}),
        _trace_row("e4b2_check_decision", {"checked": True, "ok": False, "reasons": ["bad"]}),
        _trace_row("e4b2_check_decision", {"checked": False, "ok": False, "reasons": ["limit"]}),
        _trace_row("e4b2_check_decision", {"checked": True, "ok": True}, claimed_sha="0" * 64),
        _trace_row("e4b2_check_decision", error="tool failed"),
        _trace_row("e4b2_submit_decision", error="submission failed"),
        _trace_row("e4b2_get_facts", error="read failed"),
    ]
    out = N.classify_tool_trace(rows)
    assert out["prechecks"] == {"valid": 1, "contract_invalid": 1, "limit_exhausted": 1,
                                "malformed": 1, "tool_error": 1}
    assert out["tool_errors"] == {"precheck": 1, "submission": 1, "other": 1}
    assert N.tool_argument_metrics(out["prechecks"], out["tool_errors"], True) == {
        "tool_arg_issue_count": 1, "tool_argument_status": "rejected_then_repaired",
        "uncertain_tool_error_count": 3}
    assert N.tool_argument_metrics(out["prechecks"], out["tool_errors"], False)[
        "tool_argument_status"] == "rejected_unrepaired"
    assert N.tool_argument_metrics({"contract_invalid": 0}, {"precheck": 0, "submission": 0, "other": 0},
                                   False, tool_reached=False) == {
        "tool_arg_issue_count": 0, "tool_argument_status": "not_reached",
        "uncertain_tool_error_count": 0}


def test_build_cases_exposes_only_logical_bindings_when_private_inputs_exist():
    expected = [N.ROOT / "bench/private/e2e_reward_demo/e2e_reward_s2030_20261013/model/correction.packet.json",
                N.ROOT / "bench/private/e4b_v3/inputs/h02_s2027/diagnosis.packet.json"]
    if not all(path.is_file() for path in expected):
        pytest.skip("비공개 결합 입력이 없는 공개 checkout")
    cases = N.build_cases()
    assert [(c["id"], c["kind"]) for c in cases] == [
        ("e4b_v2_s2030_correction", "correction"),
        ("e4b_v3_h02_s2027_diagnosis", "diagnosis")]
    assert all("packet" not in N.public_case(case) and "packet_path" not in N.public_case(case) for case in cases)
    assert all(len(case["binding"]["packet_sha256_raw"]) == 64 for case in cases)


def test_only_contract_verified_rejection_counts_as_submission_argument_issue(tmp_path):
    source = N.ROOT / "bench/private/e4b_v2/s2030/correction/run1_a1.submission.rejected.json"
    if not source.is_file():
        pytest.skip("비공개 과거 거절 기록이 없는 공개 checkout")
    case = next(row for row in N.build_cases() if row["kind"] == "correction")
    rejected = tmp_path / "submission.rejected.json"
    rejected.write_bytes(source.read_bytes())
    assert N.rejection_is_validation_error(case, rejected) is True

    valid = json.loads((N.ROOT / "bench/private/e2e_reward_demo/e2e_reward_s2030_20261013/model/"
                        "correction.call1.json").read_text(encoding="utf-8"))["content"]
    rejected.write_text(json.dumps({"tool": "e4b2_submit_proposal", "reasons": ["unrelated"],
                                    "submitted": json.loads(valid)}), encoding="utf-8")
    assert N.rejection_is_validation_error(case, rejected) is False


def _sse(model, *, tool=None, arguments=None, text=""):
    delta = {"role": "assistant", "content": text or None}
    if tool is not None:
        delta["tool_calls"] = [{"index": 0, "id": "offline-call", "type": "function",
                                "function": {"name": tool, "arguments": arguments}}]
    first = {"id": "offline", "object": "chat.completion.chunk", "created": 1, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    last = {"id": "offline", "object": "chat.completion.chunk", "created": 1, "model": model,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    return ("data: " + json.dumps(first) + "\n\ndata: " + json.dumps(last)
            + "\n\ndata: [DONE]\n\n").encode()


@pytest.mark.skipif(os.name != "posix", reason="NAT 1.9.0 WSL 검증")
def test_real_nat_workflow_accepts_both_models_with_identical_non_model_first_request(tmp_path, monkeypatch):
    """실제 NAT builder/SDK를 쓰되 HTTP 응답만 가짜 SSE로 바꾼다(API 호출 0)."""
    try:
        installed = version("nvidia-nat")
    except PackageNotFoundError:
        pytest.skip("NAT 미설치")
    if installed != "1.9.0":
        pytest.skip(f"등록 NAT 버전이 아님: {installed}")
    cases = N.build_cases()
    case = next(row for row in cases if row["kind"] == "correction")
    old_call = json.loads((N.ROOT / "bench/private/e2e_reward_demo/e2e_reward_s2030_20261013/model/"
                           "correction.call1.json").read_text(encoding="utf-8"))
    proposal = json.loads(old_call["content"])
    turns = [("e4b2_registry", "{}", ""), ("e4b2_skill_checklist", "{}", ""),
             ("e4b2_check_proposal", json.dumps(proposal), ""),
             ("e4b2_submit_proposal", json.dumps(proposal), ""), (None, None, "Submitted.")]
    counters = {model: 0 for model in L.MODELS}
    import httpx

    async def fake_send(client, request, *args, **kwargs):
        payload = json.loads(request.content)
        model = payload["model"]
        tool, arguments, text = turns[counters[model]]
        counters[model] += 1
        return httpx.Response(200, request=request, content=_sse(
            model, tool=tool, arguments=arguments, text=text),
            headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    book = L.Ledger(tmp_path / "ledger", {"test": "nat-workflow"})
    results = [N.run_case(case, model, tmp_path / f"run-{index}", book)
               for index, model in enumerate(L.MODELS)]
    assert all(row["valid"] for row in results), results
    assert counters == {L.BASELINE: 5, L.CANDIDATE: 5}
    assert results[0]["first_request"]["request_without_model_sha256"] == \
        results[1]["first_request"]["request_without_model_sha256"]
    assert all(row["first_request"]["request_count"] == 5 for row in results)
    assert len(book.audit()) == 10
    reviewed = [N.review_case(case, model, tmp_path / f"run-{index}", book)
                for index, model in enumerate(L.MODELS)]
    assert reviewed == results
    assert all(row["metered_complete"] and row["transport_complete"] for row in reviewed)

    result_path = tmp_path / "run-0/result.json"
    changed = json.loads(result_path.read_text(encoding="utf-8"))
    changed["valid"] = False
    result_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="재검증 불일치"):
        N.review_case(case, L.BASELINE, tmp_path / "run-0", book)
