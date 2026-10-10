import asyncio
import hashlib
import json
from typing import cast
from unittest.mock import Mock

import pytest

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo

from rl_triage import nat_single_run as N
from rl_triage import triage_tools as T


def _write_json(path, value):
    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


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
