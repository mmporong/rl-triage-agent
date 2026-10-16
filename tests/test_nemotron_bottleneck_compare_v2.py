"""해결 문구는 고정 기준을 통과한 경우에만 허용한다."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import nemotron_bottleneck_compare_v2 as C
import nemotron_bottleneck_ledger_v2 as L


def n2_rows(old_errors=3, new_errors=0):
    rows = []
    for model, errors in ((L.BASELINE, old_errors), (L.CANDIDATE, new_errors)):
        for index in range(11):
            kind = "diagnosis" if index < 6 else "correction"
            for repeat in range(1, 4):
                valid = index * 3 + repeat > errors
                rows.append({"entry": {"id": str(index), "model": model, "kind": kind, "repeat": repeat},
                             "status": "valid" if valid else "invalid",
                             "verdict": {"valid": valid, "decision_key": ["continue" if kind == "diagnosis" else "PROPOSE", "fixed"],
                                         "agrees_with_rule": True}})
    return rows


def test_b1_strict_improvement_and_no_hold_regression():
    rows = n2_rows()
    assert C.judge_text("B1", rows)["status"] == "solved"
    assert C.judge_text("B1", n2_rows(0, 0))["status"] == "inconclusive"
    assert C.judge_text("B1", n2_rows(3, 3))["status"] == "unresolved"
    for row in rows:
        if row["entry"]["model"] == L.CANDIDATE and row["entry"]["kind"] == "diagnosis":
            row["verdict"]["decision_key"][0] = "hold"
    assert C.judge_text("B1", rows)["status"] == "unresolved"


def test_incomplete_is_not_solved_and_usage_not_hidden():
    assert C.judge_text("B1", n2_rows()[:-1])["status"] == "inconclusive"
    rows = n2_rows()
    rows[-1]["status"] = "infra_error"
    assert C.judge_text("B1", rows)["status"] == "inconclusive"


def test_b2_citation_and_cause_guards_are_separate():
    rows = []
    for model in L.MODELS:
        for index in range(24):
            error = model == L.BASELINE and index < 8
            rows.append({"entry": {"id": f"d3_{index // 3}", "repeat": index % 3 + 1, "model": model}, "status": "invalid" if error else "valid",
                         "verdict": {"evidence_citation_error": error, "true_cause_top1": True,
                                     "valid_true_cause_top1": not error, "decision_key": ["cause"]}})
    assert C.judge_text("B2", rows)["status"] == "solved"
    rows[-1]["verdict"]["true_cause_top1"] = False
    assert C.judge_text("B2", rows)["status"] == "unresolved"


def test_readonly_review_never_reserves_or_calls(tmp_path):
    book = L.Ledger(tmp_path, {})
    entry = {"id": "fixture", "model": L.CANDIDATE, "kind": "diagnosis", "repeat": 1}
    case = {"messages": [], "request_params": {}}
    result = C.text_result(book, "B1", entry, case, SimpleNamespace(), allow_send=False)
    assert result["status"] == "budget_or_orphan"
    assert book.audit() == []


def test_semantic_invalid_not_retried_and_raw_revalidated(tmp_path):
    book = L.Ledger(tmp_path, {})
    entry = {"id": "fixture", "model": L.CANDIDATE, "kind": "diagnosis", "repeat": 1}
    case = {"messages": [], "request_params": {}}
    adapter = SimpleNamespace(validate_response=lambda c, text: {"valid": False, "content": text})
    result = C.text_result(book, "B1", entry, case, adapter, complete=lambda *_: {
        "provider_model": L.CANDIDATE, "content": "invalid", "finish_reason": "stop"})
    assert result["status"] == "invalid"
    assert len(book.audit()) == 1
    assert C.text_result(book, "B1", entry, case, adapter, allow_send=False) == result


def test_infra_retry_is_metered(tmp_path):
    book = L.Ledger(tmp_path, {})
    entry = {"id": "fixture", "model": L.CANDIDATE, "kind": "diagnosis", "repeat": 1}
    case = {"messages": [], "request_params": {}}
    adapter = SimpleNamespace(validate_response=lambda c, text: {"valid": True})
    calls = iter([{"status_code": 503, "error_type": "HTTPStatusError"},
                  {"provider_model": L.CANDIDATE, "content": "{}", "finish_reason": "stop"}])
    result = C.text_result(book, "B1", entry, case, adapter, complete=lambda *_: next(calls))
    assert result["status"] == "valid"
    assert len(book.audit()) == 2


def test_vision_success_requires_both_policy_groups():
    obs = []
    for label, count in (("normal", 7), ("failed", 11)):
        obs += [{"status": "valid", "verdict": {"policy_label": label, "label_alignment": True}} for _ in range(count)]
    rows = {"feasibility": {"status": "valid"}, "observations": obs, "graphs": []}
    assert C.judge_vision(rows)["status"] == "solved"
    bad = copy.deepcopy(rows)
    for row in bad["observations"][:2]:
        row["verdict"]["label_alignment"] = False
    # 합계16이어도 정상군5/7이면 미해결이다.
    assert C.judge_vision(bad)["status"] == "unresolved"


def test_budget_is_shared_and_old_v1_is_explicitly_unexecuted():
    assert sum(L.STAGE_CAPS.values()) + L.PRIOR_DISCOVERY_REQUESTS == L.API_CAP == 200
    assert "실행 없이 대체" in C.FIXED["v1_disposition"]


def test_invalid_image_contract_is_not_an_infrastructure_result():
    assert C.judge_vision({"feasibility": {"status": "invalid"}, "observations": [], "graphs": []})["status"] == "unresolved"


def test_readonly_review_rejects_duplicate_schedule(tmp_path):
    import pytest
    book = L.Ledger(tmp_path, {})
    case = {"id": "fixture", "kind": "diagnosis", "messages": [], "request_params": {}}
    rows = [{"entry": {"id": "not_registered", "repeat": 1, "model": L.CANDIDATE, "kind": "diagnosis"}}]
    with pytest.raises(ValueError, match="순서"):
        C.review_text_rows(book, "B1", rows, [case], SimpleNamespace())
    assert book.audit() == []


def test_nat_cannot_claim_success_without_reproducing_old_error():
    rows, audits = [], []
    for model in L.MODELS:
        for name in ("diagnosis", "correction"):
            rows.append({"entry": {"model": model, "id": name}, "result": {
                "valid": True, "trace_complete": True, "transport_complete": True, "metered_complete": True,
                "tool_arg_issue_count": 0,
                "first_request": {"request_without_model_sha256": name}}})
            audits.append({"reservation": {"stage": "B3", "model": model},
                           "response": {"result": {"tool_calls": [{"name": "fixture"}]}}})
    assert C.judge_nat(rows, audits)["status"] == "inconclusive"
    rows[0]["result"]["tool_arg_issue_count"] = 1
    assert C.judge_nat(rows, audits)["status"] == "solved"
    rows[-1]["result"]["trace_complete"] = False
    assert C.judge_nat(rows, audits)["status"] == "inconclusive"


def test_summary_refuses_missing_nat_artifacts(tmp_path, monkeypatch):
    import pytest
    cases = [{"id": "one", "kind": "diagnosis"}, {"id": "two", "kind": "correction"}]
    rows = [{"entry": entry, "result": {"valid": True}} for entry in C.entries("B3", cases)]
    monkeypatch.setattr(C, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(C, "ledger", lambda: L.Ledger(tmp_path / "ledger", {}))
    monkeypatch.setattr(C.B3, "build_cases", lambda: cases)
    L.write_new(tmp_path / "B3.json", rows)
    with pytest.raises(ValueError, match="원본 산출물"):
        C.summarize()
