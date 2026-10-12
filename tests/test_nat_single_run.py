import asyncio
import hashlib
import json
import socket
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nat.builder.builder import Builder
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.plugins.langchain.tool_wrapper import langchain_tool_wrapper
from nat.runtime.loader import load_workflow
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from rl_triage import nat_single_run as N
from rl_triage import triage_tools as T


def _write_json(path, value):
    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _deny_external_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def deny_external(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host in {"127.0.0.1", "::1"}:
            return original_connect(self, address)
        raise AssertionError(f"external network attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", deny_external)

    def deny_external_ex(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host in {"127.0.0.1", "::1"}:
            return original_connect_ex(self, address)
        raise AssertionError(f"external network attempted through connect_ex: {address!r}")

    monkeypatch.setattr(socket.socket, "connect_ex", deny_external_ex)


def _grounded_active_args(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {
        "reference_status": "absent", "summary": {}, "series": {},
    })
    case = tmp_path / "cases" / "one"
    case.mkdir(parents=True)
    _write_json(case / "telemetry.json", {
        "summary": {
            "Train/mean_reward": {
                "n": 1, "first": 0.0, "last": 0.0, "min": 0.0, "max": 0.0,
                "mean_first_20pct": 0.0, "mean_last_20pct": 0.0,
            },
            "Custom/contact": {"n": 1, "last": None},
        },
        "series": {"Train/mean_reward": [0.0]},
    })
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    catalog = N.G.evidence_catalog("one")
    evidence_id = "telemetry.summary::Train/mean_reward::last"
    selection = N.G.evidence_catalog_details(
        "one", catalog["catalog_sha256"], [evidence_id],
    )["selection_binding"]
    hypotheses = [
        {"id": "reward_stall", "rank": 1, "narrative": "reward may remain stalled",
         "premises": [{"evidence_id": evidence_id, "relation": "eq",
                       "operand": {"value": 0.0, "unit": "telemetry_native"}}]},
        {"id": "contact_interaction", "rank": 2, "narrative": "contact may separate the cases",
         "premises": [{"evidence_id": evidence_id, "relation": "gte",
                       "operand": {"value": 0.0, "unit": "telemetry_native"}}]},
    ]
    next_check = {
        "registry_id": "runtime_reward_window",
        "hypothesis_predictions": [
            {"hypothesis_id": "reward_stall", "predictions": [
                {"measurement_id": "reward_mean", "relation": "lt",
                 "operand": {"value": 0.0, "unit": "reward_native"}},
                {"measurement_id": "reward_min", "relation": "lt",
                 "operand": {"value": -1.0, "unit": "reward_native"}},
            ]},
            {"hypothesis_id": "contact_interaction", "predictions": [
                {"measurement_id": "reward_mean", "relation": "gte",
                 "operand": {"value": 0.0, "unit": "reward_native"}},
                {"measurement_id": "reward_min", "relation": "gte",
                 "operand": {"value": -1.0, "unit": "reward_native"}},
            ]},
        ],
    }
    return {
        "case_id": "one", "catalog_sha256": catalog["catalog_sha256"],
        "selection_binding": selection, "hypotheses": hypotheses, "next_check": next_check,
    }


def test_grounded_active_registry_uses_empty_strict_nat_input(tmp_path, monkeypatch):
    trace_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace_path))
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.grounded_active_registry_fn(
                N.GroundedActiveRegistryConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo)
            assert tool.single_fn is not None
            schema = tool.input_schema.model_json_schema()
            assert schema["properties"] == {}
            assert schema["additionalProperties"] is False
            assert schema.get("required", []) == []
            request = tool.input_schema.model_validate({})
            return json.loads(await tool.single_fn(request))

    result = asyncio.run(invoke())
    assert result == N.A.active_registry_profile()
    assert result["host_execution"]["executable_check_ids"] == ["reset_base_com_stability"]
    assert result["host_execution"]["unsupported_check_result"] == "HOLD_without_execution"
    assert set(result["checks"]) - set(result["host_execution"]["executable_check_ids"]) == set(N.G.READ_ONLY_CHECKS)
    call = json.loads(trace_path.read_text(encoding="utf-8"))
    assert call["tool"] == "grounded_active_registry"
    assert call["args"] == {}
    with pytest.raises(ValidationError):
        N.GroundedActiveRegistryInput.model_validate({"unexpected": True})


def test_grounded_active_operand_schema_excludes_root_null_but_preserves_json_lists():
    schema = N.GroundedActiveOperand.model_json_schema()
    choices = schema["properties"]["value"]["anyOf"]
    assert all(item.get("type") != "null" and "$ref" not in item for item in choices)
    with pytest.raises(ValidationError):
        N.GroundedActiveOperand.model_validate({"value": None, "unit": "configuration_value"})
    for value in (0, 0.0, False, "", [], [None, 0, False]):
        parsed = N.GroundedActiveOperand.model_validate({"value": value, "unit": "configuration_value"})
        assert parsed.value == value
        assert type(parsed.value) is type(value)


def test_grounded_active_root_null_rejected_before_writer_or_model_execution(tmp_path, monkeypatch):
    args = _grounded_active_args(tmp_path, monkeypatch)
    args["hypotheses"][0]["premises"][0]["operand"]["value"] = None
    trace_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace_path))
    _deny_external_network(monkeypatch)
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.write_grounded_active_assessment_fn(
                N.WriteGroundedActiveAssessmentConfig(), builder) as tool:
            with pytest.raises(ValidationError):
                tool.input_schema.model_validate(args)

    asyncio.run(invoke())
    assert not trace_path.exists()
    assert not (tmp_path / "preregistrations/grounded_active_assessments/one.json").exists()


def test_grounded_active_workflow_loads_without_network(monkeypatch):
    config = Path(__file__).parents[1] / "configs" / "triage_single_run_grounded_active.yml"
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-test")
    _deny_external_network(monkeypatch)

    async def load():
        async with load_workflow(config, max_concurrency=1):
            pass

    asyncio.run(load())


def test_grounded_active_writer_exposes_validator_schema_and_preserves_wire(
        tmp_path, monkeypatch):
    args = _grounded_active_args(tmp_path, monkeypatch)
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.write_grounded_active_assessment_fn(
                N.WriteGroundedActiveAssessmentConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo)
            schema = tool.input_schema.model_json_schema()
            assert set(schema["properties"]) == {
                "case_id", "catalog_sha256", "selection_binding", "hypotheses", "next_check",
            }
            assert set(schema["$defs"]["GroundedActiveSelectionBinding"]["properties"]) == {
                "selection_id", "case_id", "catalog_sha256", "selected_evidence_ids", "receipt_sha256",
            }
            assert set(schema["$defs"]["GroundedActiveHypothesis"]["properties"]) == {
                "id", "rank", "narrative", "premises",
            }
            assert set(schema["$defs"]["GroundedActiveNextCheck"]["properties"]) == {
                "registry_id", "hypothesis_predictions",
            }
            request = tool.input_schema.model_validate(args)
            assert tool.single_fn is not None
            return json.loads(await tool.single_fn(request))

    result = asyncio.run(invoke())
    stored = json.loads((
        tmp_path / "preregistrations" / "grounded_active_assessments" / "one.json"
    ).read_text(encoding="utf-8"))
    assert result == {"saved": "preregistrations/grounded_active_assessments/one.json",
                      "assessment_sha256": N.A._sha(stored), **stored}
    invalid = deepcopy(args)
    invalid["hypotheses"][0]["label"] = "not in the validator contract"
    with pytest.raises(ValidationError):
        N.WriteGroundedActiveAssessmentInput.model_validate(invalid)


def test_grounded_active_writer_semantic_rejection_stops_nat_langchain_tool(
        tmp_path, monkeypatch):
    args = _grounded_active_args(tmp_path, monkeypatch)
    args["hypotheses"][0]["premises"][0]["operand"]["value"] = 1.0
    trace_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace_path))
    _deny_external_network(monkeypatch)
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.write_grounded_active_assessment_fn(
                N.WriteGroundedActiveAssessmentConfig(), builder) as info:
            function = LambdaFunction.from_info(
                config=N.WriteGroundedActiveAssessmentConfig(), info=info,
                instance_name="write_grounded_active_assessment",
            )
            adapter = langchain_tool_wrapper(
                "write_grounded_active_assessment", function, builder,
            )
            node = ToolNode([adapter], handle_tool_errors=False)
            graph = (
                StateGraph(MessagesState)
                .add_node("tools", node)
                .set_entry_point("tools")
                .set_finish_point("tools")
                .compile()
            )
            message = AIMessage(content="", tool_calls=[{
                "name": "write_grounded_active_assessment", "args": args,
                "id": "first-semantic-rejection", "type": "tool_call",
            }])
            with pytest.raises(ValueError, match="grounded active assessment rejected"):
                await graph.ainvoke({"messages": [message]})

    asyncio.run(invoke())
    calls = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1
    assert calls[0]["tool"] == "write_grounded_active_assessment"
    assert json.loads(calls[0]["result_head"])["error"].startswith("근거 관계가 참이 아닙니다")
    assert not (tmp_path / "preregistrations" / "grounded_active_assessments" / "one.json").exists()


def test_grounded_active_writer_unavailable_evidence_hold_stops_nat_langchain_tool(
        tmp_path, monkeypatch):
    args = _grounded_active_args(tmp_path, monkeypatch)
    unavailable_id = "telemetry.summary::Custom/contact::last"
    catalog = N.G.evidence_catalog("one")
    args["selection_binding"] = N.G.evidence_catalog_details(
        "one", catalog["catalog_sha256"], [unavailable_id],
    )["selection_binding"]
    for hypothesis in args["hypotheses"]:
        hypothesis["premises"] = [{
            "evidence_id": unavailable_id, "relation": "eq",
            "operand": {"value": 0.0, "unit": "telemetry_native"},
        }]
    trace_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace_path))
    _deny_external_network(monkeypatch)
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.write_grounded_active_assessment_fn(
                N.WriteGroundedActiveAssessmentConfig(), builder) as info:
            function = LambdaFunction.from_info(
                config=N.WriteGroundedActiveAssessmentConfig(), info=info,
                instance_name="write_grounded_active_assessment",
            )
            adapter = langchain_tool_wrapper(
                "write_grounded_active_assessment", function, builder,
            )
            graph = (
                StateGraph(MessagesState)
                .add_node("tools", ToolNode([adapter], handle_tool_errors=False))
                .set_entry_point("tools")
                .set_finish_point("tools")
                .compile()
            )
            message = AIMessage(content="", tool_calls=[{
                "name": "write_grounded_active_assessment", "args": args,
                "id": "unavailable-evidence-hold", "type": "tool_call",
            }])
            with pytest.raises(ValueError, match="grounded active assessment rejected"):
                await graph.ainvoke({"messages": [message]})

    asyncio.run(invoke())
    calls = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1
    assert calls[0]["tool"] == "write_grounded_active_assessment"
    assert json.loads(calls[0]["result_head"]) == {
        "status": "hold", "reason": "evidence_unavailable",
        "unavailable_evidence": [unavailable_id],
    }
    assert not (tmp_path / "preregistrations" / "grounded_active_assessments" / "one.json").exists()


@pytest.mark.parametrize("provided", [False, True])
def test_registered_context_and_assessment_tools_preserve_evidence_gate(tmp_path, monkeypatch, provided):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {
        "reference_status": "absent", "summary": {}, "series": {},
    })
    case = tmp_path / "cases" / "example"
    case.mkdir(parents=True)
    telemetry_sha = _write_json(case / "telemetry.json", {"summary": {}, "series": {}})
    configuration_sha: str | None = None
    if provided:
        configuration_sha = _write_json(case / "configuration.json", {
            "rewards": {"stand": {"weight": 10}}, "unselected": "not returned",
        })
        _write_json(case / "context.json", {
            "schema": "single_run_context_v1",
            "telemetry_sha256": telemetry_sha,
            "configuration_sha256": configuration_sha,
            "facts": [{"id": "stand_weight", "path": ["rewards", "stand", "weight"]}],
        })
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    trace_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace_path))
    assessment_path = tmp_path / "preregistrations" / "assessments" / "example.json"
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.single_run_context_fn(N.SingleRunContextConfig(), builder) as context_tool:
            assert isinstance(context_tool, FunctionInfo)
            assert context_tool.single_fn is not None
            context = json.loads(await context_tool.single_fn("example"))
        assert context["status"] == ("provided" if provided else "not_provided")
        assert "unselected" not in context
        args = {
            "case_id": "example", "hypotheses": ["competing explanations"],
            "evidence": "no observed success measurement", "limitations": "no healthy reference",
            "next_check": "measure task success",
        }
        async with N.write_single_assessment_fn(N.WriteSingleAssessmentConfig(), builder) as assessment_tool:
            assert isinstance(assessment_tool, FunctionInfo)
            assert assessment_tool.single_fn is not None
            if provided:
                args["fact_claims"] = [{"id": "stand_weight", "value": 0}]
                rejected = json.loads(await assessment_tool.single_fn(assessment_tool.input_schema(**args)))
                assert "error" in rejected
                assert not assessment_path.exists()
                args["fact_claims"] = [{"id": "stand_weight", "value": context["facts"][0]["value"]}]
            return json.loads(await assessment_tool.single_fn(assessment_tool.input_schema(**args)))

    saved = asyncio.run(invoke())
    assert saved["status"] == "hypotheses_only"
    stored = json.loads(assessment_path.read_text(encoding="utf-8"))
    assert stored == {key: value for key, value in saved.items() if key != "saved"}
    if provided:
        assert configuration_sha is not None
    assert stored["verified_fact_claims"] == ([{
        "id": "stand_weight", "path": ["rewards", "stand", "weight"], "value": 10,
        "kind": "configuration", "configuration_sha256": configuration_sha,
    }] if provided else [])
    calls = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [call["tool"] for call in calls] == (
        ["single_run_context", "write_single_assessment", "write_single_assessment"]
        if provided else ["single_run_context", "write_single_assessment"]
    )
