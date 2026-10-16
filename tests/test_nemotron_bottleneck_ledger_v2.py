"""공통 예산·예약·원장 결합 회귀 검사. 네트워크 호출은 하지 않는다."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import nemotron_bottleneck_ledger_v2 as L


def ledger(tmp_path):
    return L.Ledger(tmp_path, {"commit": "a" * 40, "protocol_sha256": "b" * 64})


def request():
    return {"model": L.CANDIDATE, "messages": [{"role": "user", "content": "fixture"}]}


def test_reserves_before_send_and_counts_each_retry(tmp_path):
    book = ledger(tmp_path)
    def complete():
        assert len(book.audit()) == 1
        assert book.audit()[0]["response"] is None
        return {"content": "{}"}
    book.call("B1", "case_a1", request(), complete)
    book.call("B1", "case_a2", request(), lambda: {"content": "{}"})
    assert len(book.audit()) == 2
    with pytest.raises(L.BudgetStop):
        book.call("B1", "case_a1", request(), lambda: pytest.fail("전송 금지"))


def test_orphan_blocks_all_stages(tmp_path):
    book = ledger(tmp_path)
    book.reserve("B1", "one", request())
    with pytest.raises(L.BudgetStop, match="완료 기록"):
        book.reserve("B4", "two", request())


def test_stage_budget_does_not_consume_later_stage_allocation(tmp_path, monkeypatch):
    book = ledger(tmp_path)
    monkeypatch.setattr(L, "STAGE_CAPS", {"B1": 1, "B4": 1})
    book.call("B1", "one", request(), lambda: {})
    with pytest.raises(L.BudgetStop):
        book.reserve("B1", "two", request())
    book.call("B4", "three", request(), lambda: {})
    assert len(book.audit()) == 2


def test_global_budget_and_chain_tamper_are_rejected(tmp_path, monkeypatch):
    book = ledger(tmp_path)
    monkeypatch.setattr(L, "API_CAP", 2)
    book.call("B1", "one", request(), lambda: {})
    with pytest.raises(L.BudgetStop):
        book.reserve("B2", "two", request())
    path = tmp_path / "reservations/0001.json"
    row = L.read(path)
    row["request_sha256"] = "0" * 64
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="응답"):
        book.audit()


def test_lock_and_unknown_model_fail_closed(tmp_path):
    book = ledger(tmp_path)
    with book.lock(), pytest.raises(FileExistsError):
        book.reserve("B1", "one", request())
    with pytest.raises(ValueError):
        book.reserve("B1", "one", {"model": "unregistered"})


def test_async_exception_records_type_without_sensitive_text(tmp_path):
    book = ledger(tmp_path)
    async def fail():
        raise TimeoutError("SECRET fixture must not be persisted")
    with pytest.raises(TimeoutError):
        asyncio.run(book.async_call("B3", "one", request(), fail))
    result = book.audit()[0]["response"]["result"]
    assert result["error_type"] == "TimeoutError"
    assert result["infra_error"] is True
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("raw,expected", [
    ({"error_type": "BadRequestError", "status_code": 400}, "call_error"),
    ({"error_type": "RateLimitError", "status_code": 429}, "infra_error"),
    ({"error_type": "InternalServerError", "status_code": 503}, "infra_error"),
    ({"provider_model": "wrong", "content": "{}"}, "call_error"),
    ({"provider_model": L.CANDIDATE, "content": "", "finish_reason": "stop"}, "infra_error"),
    ({"provider_model": L.CANDIDATE, "content": "{}", "finish_reason": "length"}, "infra_error"),
    ({"provider_model": L.CANDIDATE, "content": "{}", "finish_reason": "stop"}, "ok"),
])
def test_transport_error_precedes_empty_response(raw, expected):
    assert L.transport_status(raw, L.CANDIDATE) == expected
