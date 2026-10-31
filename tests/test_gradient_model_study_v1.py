# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import gradient_hypothesis_contract_v1 as GH
import gradient_model_study_v1 as S

ANCHOR = "a" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _provider(request: dict) -> dict:
    context = json.loads(request["messages"][-1]["content"])
    options = {row["target"]: row for row in context["action_options"]}
    chosen = options.get(GH.GRADIENT_PROBE)
    value = ({"option_id": f"probe:{GH.GRADIENT_PROBE}",
              "fact_ids": chosen["required_fact_ids"]}
             if chosen is not None else {"option_id": "hold", "fact_ids": []})
    return {"provider_model": S.MODE.MODEL, "finish_reason": "stop",
            "content": json.dumps(value), "input_tokens": 10, "output_tokens": 5,
            "total_tokens": 15, "reasoning_tokens": 1}


def _run(tmp_path: Path, provider=_provider) -> tuple[dict, Path]:
    plan_path = tmp_path / "plan.json"
    _write(plan_path, S.prepare_plan(ANCHOR))

    def registered(path: Path, anchor: str) -> dict:
        assert anchor == ANCHOR
        return S.verify_plan(path)

    out = tmp_path / "execution"
    result = S.execute(plan_path, ANCHOR, out, complete=provider,
                       verify_registered_fn=registered, require_platform=lambda: None)
    return result, out


def test_prepare_plan_binds_fixed_public_inputs_and_enabled_low() -> None:
    plan = S.prepare_plan(ANCHOR)
    assert [row["case_id"] for row in plan["cases"]] == ["g01", "g02", "g03", "g04"]
    assert [(row["case"]["logical_name"], row["reference"]["logical_name"])
            for row in plan["cases"]] == [
                ("none_s2144_on", "none_s2144_on"),
                ("inverse_ratio_s2144_on", "none_s2144_on"),
                ("none_s2145_on", "none_s2144_on"),
                ("none_s2146_on", "none_s2144_on"),
            ]
    assert plan["request_params"] == {
        "model": S.MODE.MODEL, "temperature": 0, "max_tokens": 16384, "stream": False,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True},
                       "reasoning_effort": "low"},
    }
    assert plan["bounds"] == {
        "cases": 4, "model_decisions_per_case": 2, "actual_calls_per_decision": 1,
        "format_feedback_calls": 0, "api_requests": 8, "timeout_s": 360, "gpu_runs": 0,
    }
    assert all((ROOT / ref["path"]).is_file()
               and S.file_sha(ROOT / ref["path"]) == ref["sha256"]
               for row in plan["cases"] for role in ("case", "reference")
               for ref in row[role]["raw"].values())
    assert all(value is False for value in plan["claims"].values())


def test_actual_four_case_cpu_consumer_confirms_only_gradient_mismatch(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result["status"] == "complete"
    assert result["outcome"] == "confirmed_development_integration_candidate"
    assert result["api_requests"] == 7
    rows = {row["case_id"]: row for row in result["rows"]}
    assert rows["g02"]["model"]["confirmed_subtype"] == GH.SURROGATE
    assert all(rows[name]["model"]["decision_outcome"] == "HOLD"
               for name in ("g01", "g03", "g04"))
    assert all(row["rules"]["count_only"]["outcome"] == "unresolved"
               for row in rows.values())
    assert rows["g02"]["rules"]["count_plus_gradient"] == {
        "outcome": "confirmed", "confirmed_subtype": GH.SURROGATE}
    assert all(row["rules"]["count_plus_gradient"]["outcome"] == "unresolved"
               for name, row in rows.items() if name != "g02")
    intents = sorted((out / "calls").glob("*/intent.json"))
    claims = sorted((out / "calls").glob("*/claim.json"))
    assert len(intents) == len(claims) == 7
    assert all(json.loads(path.read_text(encoding="utf-8"))["request"]["extra_body"]
               ["reasoning_effort"] == "low" for path in intents)


def test_provider_failure_is_hold_and_aggregate_indeterminate(tmp_path: Path) -> None:
    result, _out = _run(tmp_path, provider=lambda _request: {
        "error_type": "ProviderError", "finish_reason": None})
    assert result["status"] == "complete"
    assert result["outcome"] == "indeterminate"
    assert result["api_requests"] == 4
    assert all(row["model"]["decision_outcome"] == "HOLD"
               and not row["model"]["all_model_calls_valid"] for row in result["rows"])


def test_plan_raw_or_source_tamper_is_rejected(tmp_path: Path) -> None:
    plan = S.prepare_plan(ANCHOR)
    plan["cases"][1]["case"]["raw"]["records.jsonl"]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    _write(path, plan)
    with pytest.raises(ValueError, match="plan이 현재 source/raw 입력과 다르다"):
        S.verify_plan(path)


def test_call_cap_is_checked_before_provider(tmp_path: Path) -> None:
    plan = S.prepare_plan(ANCHOR)
    counter = {"calls": 8, "protocol_sha256": "f" * 64}
    calls: list[dict] = []
    factory = S._complete_factory(tmp_path / "calls", plan, lambda: True,
                                  lambda request: calls.append(request) or {}, counter)
    invoke = factory("g01.gradient_model.d1", {"state_digest": "s"})
    with pytest.raises(ValueError, match="API cap 초과"):
        invoke([{"role": "user", "content": "x"}],
               S.MODE.CALL.F.request_params(S.MODE.CALL.MODEL))
    assert not calls
    assert not (tmp_path / "calls").exists()
