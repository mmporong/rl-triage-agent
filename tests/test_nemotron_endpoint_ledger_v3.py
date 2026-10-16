import asyncio
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import nemotron_bottleneck_ledger_v2 as L
import nemotron_endpoint_ledger_v3 as V3
import nemotron_bottleneck_nat_v2 as N

assert V3.PREFIX == V3.PHASE_PREFIX == "v3::"
ACTUAL_PRIVATE_ALIAS = "private/nvidia/nemotron-3.5-super-120b-a12b"
WRONG_VL_PRIVATE_ALIAS = "private/nvidia/nemotron-3.5-super-vl-120b-a12b"


def request(model=L.CANDIDATE):
    return {"model": model, "messages": [{"role": "user", "content": "fixture"}]}


def alias_result(**extra):
    return {"provider_model": ACTUAL_PRIVATE_ALIAS, "content": "ok",
            "finish_reason": "stop", **extra}


def test_candidate_alias_is_only_normalized_in_view_and_raw_ledger_stays_unchanged(tmp_path):
    assert V3.PRIVATE_CANDIDATE_ALIAS == "private/nvidia/nemotron-3.5-super-120b-a12b"
    parent = L.Ledger(tmp_path, {"parent": "v2"})
    phase = V3.PhaseLedger(parent)
    viewed = phase.call("B1", "case_a1", request(), lambda: alias_result(
        error_type="ProviderModelMismatch", infra_error=False))

    assert viewed["provider_model"] == L.CANDIDATE
    assert viewed["provider_model_raw"] == V3.PRIVATE_CANDIDATE_ALIAS
    assert viewed["provider_identity_registered"] is True
    assert viewed["identity_policy"] == V3.IDENTITY_POLICY
    assert "error_type" not in viewed and "infra_error" not in viewed

    raw = parent.audit()[0]
    assert raw["reservation"]["label"] == "v3::case_a1"
    assert raw["response"]["result"]["provider_model"] == V3.PRIVATE_CANDIDATE_ALIAS
    assert raw["response"]["result"]["error_type"] == "ProviderModelMismatch"
    audit = phase.audit()[0]
    assert audit["reservation"]["label"] == "case_a1"
    assert audit["reservation"]["label_raw"] == "v3::case_a1"
    assert audit["response"]["result"] == viewed


def test_wrong_vl_private_alias_is_not_registered():
    view = V3.provider_view({"provider_model": WRONG_VL_PRIVATE_ALIAS, "content": "ok",
                             "finish_reason": "stop"}, L.CANDIDATE)
    assert view["provider_model"] == WRONG_VL_PRIVATE_ALIAS
    assert view["provider_identity_registered"] is False
    assert L.transport_status(view, L.CANDIDATE) == "call_error"


def test_unknown_alias_and_candidate_alias_for_old_model_are_not_reused(tmp_path):
    phase = V3.PhaseLedger(L.Ledger(tmp_path, {}))
    unknown = phase.call("B1", "unknown", request(), lambda: {
        "provider_model": "private/unregistered", "content": "ok", "finish_reason": "stop"})
    assert unknown["provider_model"] == "private/unregistered"
    assert unknown["provider_identity_registered"] is False
    assert L.transport_status(unknown, L.CANDIDATE) == "call_error"

    wrong_old = phase.call("B1", "old", request(L.BASELINE), alias_result)
    assert wrong_old["provider_model"] == V3.PRIVATE_CANDIDATE_ALIAS
    assert wrong_old["provider_identity_registered"] is False
    assert L.transport_status(wrong_old, L.BASELINE) == "call_error"


def test_registered_alias_does_not_hide_unrelated_errors():
    raw = alias_result(error_type="EmptyResponse", infra_error=True)
    view = V3.provider_view(raw, L.CANDIDATE)
    assert view["provider_model"] == L.CANDIDATE
    assert view["error_type"] == "EmptyResponse" and view["infra_error"] is True
    assert raw["provider_model"] == V3.PRIVATE_CANDIDATE_ALIAS


@pytest.mark.parametrize("model", L.MODELS)
def test_canonical_provider_ids_remain_exact_and_registered(model):
    raw = {"provider_model": model, "content": "ok", "finish_reason": "stop"}
    view = V3.provider_view(raw, model)
    assert view["provider_model"] == model and view["provider_model_raw"] == model
    assert view["provider_identity_registered"] is True
    assert raw == {"provider_model": model, "content": "ok", "finish_reason": "stop"}


def test_async_call_returns_normalized_view_but_persists_raw_response(tmp_path):
    parent = L.Ledger(tmp_path, {})
    phase = V3.PhaseLedger(parent)

    async def complete():
        return alias_result(error_type="ProviderModelMismatch", infra_error=False), "response-object"

    result, value = asyncio.run(phase.async_call("B3", "nat_http1", request(), complete))
    assert value == "response-object" and result["provider_model"] == L.CANDIDATE
    assert parent.audit()[0]["response"]["result"]["provider_model"] == V3.PRIVATE_CANDIDATE_ALIAS


def test_global_orphan_and_budget_are_inherited(tmp_path, monkeypatch):
    parent = L.Ledger(tmp_path, {})
    phase = V3.PhaseLedger(parent)
    parent.reserve("B1", "old_orphan", request())
    with pytest.raises(L.BudgetStop, match="완료 기록"):
        phase.call("B1", "new", request(), alias_result)

    budget_root = tmp_path / "budget"
    monkeypatch.setattr(L, "API_CAP", 2)
    monkeypatch.setattr(L, "PRIOR_DISCOVERY_REQUESTS", 0)
    monkeypatch.setattr(L, "STAGE_CAPS", {"B1": 2})
    parent = L.Ledger(budget_root, {})
    parent.call("B1", "v2_old", request(), alias_result)
    phase = V3.PhaseLedger(parent)
    phase.call("B1", "one", request(), alias_result)
    with pytest.raises(L.BudgetStop, match="상한"):
        phase.call("B1", "two", request(), alias_result)
    assert len(parent.audit()) == 2 and len(phase.audit()) == 1


def test_phase_prefix_prevents_duplicate_and_double_prefixed_labels(tmp_path):
    phase = V3.PhaseLedger(L.Ledger(tmp_path, {}))
    phase.call("B2", "same", request(), alias_result)
    with pytest.raises(L.BudgetStop, match="이미 예약"):
        phase.call("B2", "same", request(), alias_result)
    with pytest.raises(ValueError, match="label"):
        phase.call("B2", "v3::escape", request(), alias_result)


def _sse(*, tool=None, arguments=None, text=""):
    delta = {"role": "assistant", "content": text or None}
    if tool is not None:
        delta["tool_calls"] = [{"index": 0, "id": "offline-call", "type": "function",
                                "function": {"name": tool, "arguments": arguments}}]
    first = {"id": "offline", "object": "chat.completion.chunk", "created": 1,
             "model": ACTUAL_PRIVATE_ALIAS,
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    last = {"id": "offline", "object": "chat.completion.chunk", "created": 1,
            "model": ACTUAL_PRIVATE_ALIAS,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    return ("data: " + json.dumps(first) + "\n\ndata: " + json.dumps(last)
            + "\n\ndata: [DONE]\n\n").encode()


@pytest.mark.skipif(os.name != "posix", reason="NAT 1.9.0 WSL 검증")
def test_phase_ledger_keeps_no_vl_raw_alias_and_nat_review_matches(tmp_path, monkeypatch):
    """실제 NAT builder를 쓰고 HTTP만 가짜 SSE로 바꾼 v3 종단 회귀(API 0)."""
    try:
        installed = version("nvidia-nat")
    except PackageNotFoundError:
        pytest.skip("NAT 미설치")
    if installed != "1.9.0":
        pytest.skip(f"등록 NAT 버전이 아님: {installed}")
    case = next(row for row in N.build_cases() if row["kind"] == "correction")
    call = json.loads((N.ROOT / "bench/private/e2e_reward_demo/e2e_reward_s2030_20261013/model/"
                       "correction.call1.json").read_text(encoding="utf-8"))
    proposal = json.loads(call["content"])
    turns = [("e4b2_registry", "{}", ""), ("e4b2_skill_checklist", "{}", ""),
             ("e4b2_check_proposal", json.dumps(proposal), ""),
             ("e4b2_submit_proposal", json.dumps(proposal), ""), (None, None, "Submitted.")]
    counter = 0
    import httpx

    async def fake_send(client, request, *args, **kwargs):
        nonlocal counter
        assert json.loads(request.content)["model"] == L.CANDIDATE
        tool, arguments, text = turns[counter]
        counter += 1
        return httpx.Response(200, request=request, content=_sse(
            tool=tool, arguments=arguments, text=text), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    parent = L.Ledger(tmp_path / "ledger", {"test": "v3-nat"})
    phase = V3.PhaseLedger(parent)
    output = tmp_path / "run"
    result = N.run_case(case, L.CANDIDATE, output, phase)
    assert result["valid"] is True and counter == 5
    assert N.review_case(case, L.CANDIDATE, output, phase) == result
    raw = parent.audit()
    assert len(raw) == 5
    assert all(row["response"]["result"]["provider_model"] == ACTUAL_PRIVATE_ALIAS for row in raw)
    assert all(row["response"]["result"]["error_type"] == "ProviderModelMismatch" for row in raw)
