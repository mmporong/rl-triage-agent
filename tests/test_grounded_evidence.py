import asyncio
import hashlib
import json
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import yaml
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo

from rl_triage import grounded_evidence as G
from rl_triage import nat_single_run as N
from rl_triage import single_run as S
from rl_triage import triage_tools as T


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture()
def grounded_ws(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {
        "reference_status": "absent", "summary": {}, "series": {},
    })
    case = tmp_path / "cases" / "one"
    case.mkdir(parents=True)
    telemetry_sha = _write_json(case / "telemetry.json", {
        "summary": {
            "Train/mean_reward": {
                "n": 5, "first": -2.0, "last": 0.0, "min": -2.0, "max": 1.0,
                "mean_first_20pct": -2.0, "mean_last_20pct": 0.0,
            },
            "Custom/contact": {"n": 3, "first": None, "last": 0.0},
        },
        "series": {
            "Train/mean_reward": [-2.0, -1.0, 0.0, 1.0, 0.0],
            "Custom/contact": [0.0, 1.0, 0.0],
        },
    })
    configuration_sha = _write_json(case / "configuration.json", {
        "rewards": {"tracking": {"weight": 10}, "unselected": {"weight": 999}},
        "observations": {"contact_codes": [1]},
    })
    _write_json(case / "context.json", {
        "schema": "single_run_context_v1", "telemetry_sha256": telemetry_sha,
        "configuration_sha256": configuration_sha,
        "facts": [
            {"id": "tracking_weight", "path": ["rewards", "tracking", "weight"]},
            {"id": "contact_codes", "path": ["observations", "contact_codes"]},
        ],
    })
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    return tmp_path, case


def _premise(evidence_id: str, relation: str, value: object, unit: str):
    return {"evidence_id": evidence_id, "relation": relation,
            "operand": {"value": value, "unit": unit}}


def _hypotheses(evidence_id="telemetry.summary::Train/mean_reward::last"):
    return [
        {"id": "reward_stall", "rank": 1, "narrative": "reward may remain stalled",
         "premises": [_premise(evidence_id, "eq", 0.0, "telemetry_native")]},
        {"id": "contact_interaction", "rank": 2, "narrative": "contact may separate the cases",
         "premises": [_premise(evidence_id, "gte", 0.0, "telemetry_native")]},
    ]


def _next_check(discriminating=True):
    second_relation = "gte" if discriminating else "lte"
    return {
        "registry_id": "runtime_reward_window",
        "hypothesis_predictions": [
            {"hypothesis_id": "reward_stall", "predictions": [
                {"measurement_id": "reward_mean", "relation": "lt",
                 "operand": {"value": 0.0, "unit": "reward_native"}},
                {"measurement_id": "reward_min", "relation": "lt",
                 "operand": {"value": -1.0, "unit": "reward_native"}},
            ]},
            {"hypothesis_id": "contact_interaction", "predictions": [
                {"measurement_id": "reward_mean", "relation": second_relation,
                 "operand": {"value": 0.0, "unit": "reward_native"}},
                {"measurement_id": "reward_min", "relation": second_relation,
                 "operand": {"value": -1.0, "unit": "reward_native"}},
            ]},
        ],
    }


def _com_next_check(local_com_relation="gte"):
    return {
        "registry_id": "runtime_com_state",
        "hypothesis_predictions": [
            {"hypothesis_id": "reward_stall", "predictions": [
                {"measurement_id": "com_position_z", "relation": "gte",
                 "operand": {"value": 0.0, "unit": "simulation_length_native"}},
                {"measurement_id": "com_velocity_z", "relation": "lte",
                 "operand": {"value": 1.0, "unit": "simulation_velocity_native"}},
                {"measurement_id": "body_com_local_position_max_abs", "relation": "lt",
                 "operand": {"value": 0.5, "unit": "simulation_length_native"}},
            ]},
            {"hypothesis_id": "contact_interaction", "predictions": [
                {"measurement_id": "com_position_z", "relation": "gte",
                 "operand": {"value": 0.0, "unit": "simulation_length_native"}},
                {"measurement_id": "com_velocity_z", "relation": "lte",
                 "operand": {"value": 1.0, "unit": "simulation_velocity_native"}},
                {"measurement_id": "body_com_local_position_max_abs", "relation": local_com_relation,
                 "operand": {"value": 0.5, "unit": "simulation_length_native"}},
            ]},
        ],
    }


def test_full_telemetry_reader_needs_no_context_and_returns_raw_hash(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {"reference_status": "absent", "summary": {}, "series": {}})
    case = tmp_path / "cases" / "plain"
    case.mkdir(parents=True)
    value = {"summary": {"x": {"n": 2}}, "series": {"x": [0.0, 1.0]}}
    expected = _write_json(case / "telemetry.json", value)
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)

    full = S.read_full_telemetry("plain")
    assert full == {"case_id": "plain", "telemetry_sha256": expected, **value}
    catalog = G.evidence_catalog("plain")
    assert catalog["context_status"] == "not_provided"
    assert catalog["telemetry_sha256"] == expected


def test_catalog_preserves_present_null_missing_and_selected_facts(grounded_ws):
    _, case = grounded_ws
    catalog = G.evidence_catalog("one")
    rows = {item["evidence_id"]: item for item in catalog["evidence"]}
    assert rows["telemetry.summary::Train/mean_reward::last"]["state"] == "present"
    assert rows["telemetry.summary::Train/mean_reward::last"]["value"] == 0.0
    assert rows["telemetry.summary::Custom/contact::first"]["state"] == "null"
    assert rows["telemetry.summary::Custom/contact::mean_last_20pct"]["state"] == "missing"
    assert rows["configuration::tracking_weight"]["value"] == 10
    assert all("unselected" not in evidence_id for evidence_id in rows)
    assert catalog["telemetry_sha256"] == hashlib.sha256((case / "telemetry.json").read_bytes()).hexdigest()
    assert "body_com_local_position_max_abs" in catalog["read_only_checks"]["runtime_com_state"]["measurements"]


@pytest.mark.parametrize("bad_n", [True, 1.5])
def test_catalog_rejects_non_integer_or_bool_n(grounded_ws, bad_n):
    _, case = grounded_ws
    telemetry = json.loads((case / "telemetry.json").read_text(encoding="utf-8"))
    telemetry["summary"]["Train/mean_reward"]["n"] = bad_n
    _write_json(case / "telemetry.json", telemetry)
    with pytest.raises(ValueError, match="정수"):
        G.evidence_catalog("one")


def test_catalog_rejects_nonfinite_series(grounded_ws):
    _, case = grounded_ws
    raw = (case / "telemetry.json").read_text(encoding="utf-8").replace("-2.0", "NaN", 1)
    (case / "telemetry.json").write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="유한하지 않은"):
        G.evidence_catalog("one")


def test_grounded_assessment_saves_verified_relations_but_marks_prose_and_predictions_unverified(grounded_ws):
    workspace, _ = grounded_ws
    catalog = G.evidence_catalog("one")
    result = G.write_grounded_assessment("one", catalog["catalog_sha256"], _hypotheses(), _next_check())
    assert result["status"] == "hypotheses_only"
    assert result["premises_verified"] is True
    assert result["model_narrative_unverified"] is True
    assert result["next_check"]["prediction_unverified"] is True
    assert result["next_check"]["execution_status"] == "approval_pending"
    assert result["next_check"]["measurement_status"] == "not_observed"
    assert result["next_check"]["assessment_status"] == "hold"
    assert result["next_check"]["registered_not_executed"] is True
    stored = json.loads((workspace / "preregistrations" / "grounded_assessments" / "one.json").read_text(encoding="utf-8"))
    assert stored == {key: value for key, value in result.items() if key != "saved"}


@pytest.mark.parametrize("evidence_id", [
    "telemetry.summary::Custom/contact::first",
    "telemetry.summary::Custom/contact::mean_last_20pct",
])
def test_null_or_missing_premise_returns_hold_without_artifact(grounded_ws, evidence_id):
    workspace, _ = grounded_ws
    catalog = G.evidence_catalog("one")
    hypotheses = _hypotheses(evidence_id)
    result = G.write_grounded_assessment("one", catalog["catalog_sha256"], hypotheses, _next_check())
    assert result == {
        "status": "hold", "reason": "evidence_unavailable",
        "unavailable_evidence": [evidence_id],
    }
    assert not (workspace / "preregistrations").exists()


@pytest.mark.parametrize("mutation, message", [
    (lambda hypotheses, check: hypotheses[0].update(extra=True), "id, rank, narrative, premises"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0].update(extra=True), "evidence_id, relation, operand"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0].update(evidence_id="sampled::curve"), "없는 evidence id"),
    (lambda hypotheses, check: hypotheses[0]["premises"].append(hypotheses[0]["premises"][0].copy()), "중복 premise"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0].update(evidence_id=["bad"]), "문자열"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0]["operand"].update(unit="seconds"), "단위"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0]["operand"].update(value=True), "유한한 숫자"),
    (lambda hypotheses, check: hypotheses[0]["premises"][0].update(relation="lt", operand={"value": -1.0, "unit": "telemetry_native"}), "참이 아닙니다"),
    (lambda hypotheses, check: check.update(registry_id="unknown_check"), "등록되지 않은"),
])
def test_malformed_unknown_type_unit_and_false_relations_are_rejected(grounded_ws, mutation, message):
    catalog = G.evidence_catalog("one")
    hypotheses, check = _hypotheses(), _next_check()
    mutation(hypotheses, check)
    result = G.write_grounded_assessment("one", catalog["catalog_sha256"], hypotheses, check)
    assert message in result["error"]


def test_duplicate_ids_and_nondiscriminating_predictions_are_rejected(grounded_ws):
    catalog = G.evidence_catalog("one")
    hypotheses = _hypotheses()
    hypotheses[1]["id"] = hypotheses[0]["id"]
    assert "중복" in G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], hypotheses, _next_check())["error"]

    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], _hypotheses(), _next_check(discriminating=False))
    assert "배타적이지 않습니다" in result["error"]


def test_equal_predictions_are_allowed_when_one_common_measurement_is_exclusive(grounded_ws):
    catalog = G.evidence_catalog("one")
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], _hypotheses(), _com_next_check())
    assert result["status"] == "hypotheses_only"
    assert result["next_check"]["execution_status"] == "approval_pending"


def test_list_operands_reject_bool_number_alias_and_nested_values(grounded_ws):
    catalog = G.evidence_catalog("one")
    hypotheses = _hypotheses()
    hypotheses[0]["premises"] = [
        _premise("configuration::contact_codes", "eq", [True], "configuration_value")]
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], hypotheses, _next_check())
    assert "참이 아닙니다" in result["error"]

    hypotheses[0]["premises"][0]["operand"]["value"] = [[1]]
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], hypotheses, _next_check())
    assert "flat scalar list" in result["error"]


def test_source_mutation_and_overwrite_are_rejected(grounded_ws):
    _, case = grounded_ws
    catalog = G.evidence_catalog("one")
    (case / "telemetry.json").write_bytes((case / "telemetry.json").read_bytes() + b" ")
    changed = G.write_grounded_assessment("one", catalog["catalog_sha256"], _hypotheses(), _next_check())
    assert "catalog_sha256" in changed["error"] or "telemetry_sha256" in changed["error"]

    telemetry = json.loads((case / "telemetry.json").read_text(encoding="utf-8"))
    _write_json(case / "telemetry.json", telemetry)
    context = json.loads((case / "context.json").read_text(encoding="utf-8"))
    context["telemetry_sha256"] = hashlib.sha256((case / "telemetry.json").read_bytes()).hexdigest()
    _write_json(case / "context.json", context)
    fresh = G.evidence_catalog("one")
    assert "saved" in G.write_grounded_assessment("one", fresh["catalog_sha256"], _hypotheses(), _next_check())
    second = G.write_grounded_assessment("one", fresh["catalog_sha256"], _hypotheses(), _next_check())
    assert "덮어쓸 수 없습니다" in second["error"]


def test_grounded_nat_registration_and_config_exclude_free_code_tool(grounded_ws, monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "triage_single_run_grounded.yml").read_text(encoding="utf-8"))
    assert config["workflow"]["tool_names"] == ["grounded_evidence_catalog", "write_grounded_assessment"]
    assert "single_run_analysis" not in config["functions"]
    builder = cast(Builder, Mock(spec=Builder))
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace))

    async def invoke():
        async with N.grounded_evidence_catalog_fn(N.GroundedEvidenceCatalogConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            catalog = json.loads(await tool.single_fn("one"))
        async with N.write_grounded_assessment_fn(N.WriteGroundedAssessmentConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            args = tool.input_schema(
                case_id="one", catalog_sha256=catalog["catalog_sha256"],
                hypotheses=_hypotheses(), next_check=_next_check(),
            )
            saved = json.loads(await tool.single_fn(args))
        return catalog, saved

    catalog, saved = asyncio.run(invoke())
    assert catalog["context_status"] == "provided"
    assert saved["status"] == "hypotheses_only"
    calls = [json.loads(line)["tool"] for line in trace.read_text(encoding="utf-8").splitlines()]
    assert calls == ["grounded_evidence_catalog", "write_grounded_assessment"]
