# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import export_gradient_model_study_v1 as EXPORT
import gradient_hypothesis_contract_v1 as GH
import gradient_model_study_v1 as STUDY

ANCHOR = "a" * 40


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def provider(request: dict) -> dict:
    context = json.loads(request["messages"][-1]["content"])
    option = next((row for row in context["action_options"]
                   if row["target"] == GH.GRADIENT_PROBE), None)
    response = ({"option_id": option["option_id"], "fact_ids": option["required_fact_ids"]}
                if option else {"option_id": "hold", "fact_ids": []})
    return {"provider_model": STUDY.MODE.MODEL, "finish_reason": "stop",
            "content": json.dumps(response), "input_tokens": 10, "output_tokens": 5,
            "total_tokens": 15, "reasoning_tokens": 1}


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    plan_path = tmp_path / "plan.json"
    write(plan_path, STUDY.prepare_plan(ANCHOR))

    def registered(path: Path, anchor: str) -> dict:
        assert anchor == ANCHOR
        return STUDY.verify_plan(path)

    execution = tmp_path / "execution"
    STUDY.execute(plan_path, ANCHOR, execution, complete=provider,
                  verify_registered_fn=registered, require_platform=lambda: None)
    return plan_path, execution / "result.json"


def verify_fixture(path: Path, anchor: str) -> dict:
    assert anchor == ANCHOR
    return STUDY.verify_plan(path)


def test_export_replays_actual_structure_and_publishes_safe_projection(tmp_path: Path) -> None:
    plan, result = fixture(tmp_path)
    out = tmp_path / "public"
    summary = EXPORT.export(plan, ANCHOR, result, out, verify_registered_fn=verify_fixture)
    assert summary["status"] == "complete"
    assert summary["outcome"] == "confirmed_development_integration_candidate"
    assert summary["planned_cases"] == summary["complete_cases"] == 4
    assert summary["api_requests"] == summary["valid_model_calls"] == 7
    assert summary["correct_subtype_confirmations"] == 1
    assert summary["normal_wrong_confirmations"] == 0
    assert summary["model_holds"] == 3
    assert summary["total_tokens"] == 105
    assert summary["reasoning_tokens"] == 7
    assert summary["usage_observation"]["total_tokens"] == {
        "reported_calls": 7, "known_sum": 105, "complete": True}
    assert summary["claims"]["unseen"] is False
    g02 = json.loads((out / "cases/g02.json").read_text(encoding="utf-8"))
    assert g02["expected_subtype"] == GH.SURROGATE
    assert g02["correct_subtype"] is True
    assert g02["rules"]["count_only"]["outcome"] == "unresolved"
    assert g02["rules"]["count_plus_gradient"] == {
        "outcome": "confirmed", "confirmed_subtype": GH.SURROGATE}
    call = json.loads(next((out / "calls").glob("*.json")).read_text(encoding="utf-8"))
    assert set(call["final_json"]) == {"option_id", "fact_ids"}
    assert all(len(call[name]) == 64 for name in (
        "intent_sha256", "claim_sha256", "call_record_sha256",
        "request_sha256", "response_sha256"))
    assert "reasoning_content" not in json.dumps(call)
    manifest = json.loads((out / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_count"] == 13
    assert set(manifest["files"]) == {
        path.relative_to(out).as_posix() for path in out.rglob("*.json")
        if path.name != "artifact-manifest.json"}


def test_response_or_dialogue_tamper_is_rejected_before_output(tmp_path: Path) -> None:
    plan, result = fixture(tmp_path)
    call_path = next((result.parent / "calls").glob("*/result.json"))
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["response"]["content"] = json.dumps({"option_id": "hold", "fact_ids": []})
    write(call_path, call)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="response/dialogue"):
        EXPORT.export(plan, ANCHOR, result, out, verify_registered_fn=verify_fixture)
    assert not out.exists()


def test_incomplete_result_is_rejected_without_prefix(tmp_path: Path) -> None:
    plan, result = fixture(tmp_path)
    value = json.loads(result.read_text(encoding="utf-8"))
    value["status"] = "invalid"
    write(result, value)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="완료된 study result"):
        EXPORT.export(plan, ANCHOR, result, out, verify_registered_fn=verify_fixture)
    assert not out.exists()


def test_private_path_and_reasoning_body_never_enter_public_files(tmp_path: Path) -> None:
    plan, result = fixture(tmp_path)
    out = tmp_path / "public"
    EXPORT.export(plan, ANCHOR, result, out, verify_registered_fn=verify_fixture)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
    for forbidden in ("bench/private", "bench\\private", "reasoning_content",
                      "private_source", "C:\\Users\\", str(tmp_path)):
        assert forbidden not in combined
    provenance = json.loads((out / "source-provenance.json").read_text(encoding="utf-8"))
    assert provenance["private_paths_published"] is False
    assert provenance["protocol"]["path"] == "fixture/plan.json"


def test_missing_usage_is_null_with_observed_denominator_and_known_sum(tmp_path: Path) -> None:
    calls = 0

    def partial_usage(request: dict) -> dict:
        nonlocal calls
        calls += 1
        response = provider(request)
        if calls == 1:
            response.pop("reasoning_tokens")
        return response

    plan_path = tmp_path / "plan.json"
    write(plan_path, STUDY.prepare_plan(ANCHOR))
    execution = tmp_path / "execution"
    STUDY.execute(plan_path, ANCHOR, execution, complete=partial_usage,
                  verify_registered_fn=verify_fixture, require_platform=lambda: None)
    summary = EXPORT.export(plan_path, ANCHOR, execution / "result.json", tmp_path / "public",
                            verify_registered_fn=verify_fixture)
    assert summary["reasoning_tokens"] is None
    assert summary["usage_observation"]["reasoning_tokens"] == {
        "reported_calls": 6, "known_sum": 6, "complete": False}
    assert summary["total_tokens"] == 105


def test_invalid_model_response_exports_indeterminate_without_final_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write(plan_path, STUDY.prepare_plan(ANCHOR))
    execution = tmp_path / "execution"
    STUDY.execute(plan_path, ANCHOR, execution,
                  complete=lambda _request: {"error_type": "ProviderError", "finish_reason": None},
                  verify_registered_fn=verify_fixture, require_platform=lambda: None)
    out = tmp_path / "public"
    summary = EXPORT.export(plan_path, ANCHOR, execution / "result.json", out,
                            verify_registered_fn=verify_fixture)
    assert summary["outcome"] == "indeterminate"
    assert summary["valid_model_calls"] == 0
    assert summary["total_tokens"] is None
    assert summary["usage_observation"]["total_tokens"] == {
        "reported_calls": 0, "known_sum": 0, "complete": False}
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (out / "calls").glob("*.json")]
    assert len(rows) == 4
    assert all(row["final_json"] is None and row["model_valid"] is False
               and row["validation_codes"] == ["PROVIDER_UNAVAILABLE"] for row in rows)
