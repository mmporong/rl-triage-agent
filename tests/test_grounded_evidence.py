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


def _selection_binding(catalog: dict, evidence_ids=None) -> dict:
    ids = evidence_ids or ["telemetry.summary::Train/mean_reward::last"]
    return G.evidence_catalog_details(
        "one", catalog["catalog_sha256"], ids,
    )["selection_binding"]


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
    selection = _selection_binding(catalog)
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, _hypotheses(), _next_check())
    assert result["status"] == "hypotheses_only"
    assert result["premises_verified"] is True
    assert result["model_narrative_unverified"] is True
    assert result["next_check"]["prediction_unverified"] is True
    assert result["next_check"]["execution_status"] == "approval_pending"
    assert result["next_check"]["measurement_status"] == "not_observed"
    assert result["next_check"]["assessment_status"] == "hold"
    assert result["next_check"]["registered_not_executed"] is True
    assert result["selection_binding"] == selection
    assert result["selected_evidence_ids"] == selection["selected_evidence_ids"]
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
    selection = _selection_binding(catalog, [evidence_id])
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, hypotheses, _next_check())
    assert result == {
        "status": "hold", "reason": "evidence_unavailable",
        "unavailable_evidence": [evidence_id],
    }
    assert not (workspace / "preregistrations" / "grounded_assessments").exists()


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
    selection = _selection_binding(catalog)
    hypotheses, check = _hypotheses(), _next_check()
    mutation(hypotheses, check)
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, hypotheses, check)
    assert message in result["error"]


def test_duplicate_ids_and_nondiscriminating_predictions_are_rejected(grounded_ws):
    catalog = G.evidence_catalog("one")
    selection = _selection_binding(catalog)
    hypotheses = _hypotheses()
    hypotheses[1]["id"] = hypotheses[0]["id"]
    assert "중복" in G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, hypotheses, _next_check())["error"]

    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection,
        _hypotheses(), _next_check(discriminating=False))
    assert "배타적이지 않습니다" in result["error"]


def test_equal_predictions_are_allowed_when_one_common_measurement_is_exclusive(grounded_ws):
    catalog = G.evidence_catalog("one")
    selection = _selection_binding(catalog)
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, _hypotheses(), _com_next_check())
    assert result["status"] == "hypotheses_only"
    assert result["next_check"]["execution_status"] == "approval_pending"


def test_list_operands_reject_bool_number_alias_and_nested_values(grounded_ws):
    catalog = G.evidence_catalog("one")
    hypotheses = _hypotheses()
    hypotheses[0]["premises"] = [
        _premise("configuration::contact_codes", "eq", [True], "configuration_value")]
    selection = _selection_binding(catalog, [
        "configuration::contact_codes", "telemetry.summary::Train/mean_reward::last",
    ])
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, hypotheses, _next_check())
    assert "참이 아닙니다" in result["error"]

    hypotheses[0]["premises"][0]["operand"]["value"] = [[1]]
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, hypotheses, _next_check())
    assert "flat scalar list" in result["error"]


def test_source_mutation_and_overwrite_are_rejected(grounded_ws):
    _, case = grounded_ws
    catalog = G.evidence_catalog("one")
    selection = _selection_binding(catalog)
    (case / "telemetry.json").write_bytes((case / "telemetry.json").read_bytes() + b" ")
    changed = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, _hypotheses(), _next_check())
    assert "catalog_sha256" in changed["error"] or "telemetry_sha256" in changed["error"]

    telemetry = json.loads((case / "telemetry.json").read_text(encoding="utf-8"))
    _write_json(case / "telemetry.json", telemetry)
    context = json.loads((case / "context.json").read_text(encoding="utf-8"))
    context["telemetry_sha256"] = hashlib.sha256((case / "telemetry.json").read_bytes()).hexdigest()
    _write_json(case / "context.json", context)
    fresh = G.evidence_catalog("one")
    fresh_selection = _selection_binding(fresh)
    assert "saved" in G.write_grounded_assessment(
        "one", fresh["catalog_sha256"], fresh_selection, _hypotheses(), _next_check())
    second = G.write_grounded_assessment(
        "one", fresh["catalog_sha256"], fresh_selection, _hypotheses(), _next_check())
    assert "덮어쓸 수 없습니다" in second["error"]


def _has_exact_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_exact_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_exact_key(item, key) for item in value)
    return False


def test_group_index_is_value_free_bounded_and_covers_every_evidence(grounded_ws):
    catalog = G.evidence_catalog("one")
    index = G.evidence_catalog_group_index("one")
    assert index["catalog_sha256"] == catalog["catalog_sha256"]
    assert index["total_evidence_count"] == len(catalog["evidence"])
    assert sum(group["evidence_count"] for group in index["groups"]) == len(catalog["evidence"])
    assert index["payload_utf8_bytes"] <= index["payload_byte_cap"] == 65_536
    assert index["values_included"] is False
    assert index["comprehensive_diagnosis_supported"] is False
    assert not _has_exact_key(index, "value")
    assert len({group["group_id"] for group in index["groups"]}) == index["group_count"]
    visited = []
    for group in index["groups"]:
        offset = 0
        group_ids = []
        while True:
            page = G.evidence_catalog_group_page(
                "one", catalog["catalog_sha256"], group["group_id"], offset=offset, limit=2,
            )
            group_ids.extend(item["evidence_id"] for item in page["evidence"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert len(group_ids) == group["evidence_count"]
        assert {"first": group_ids[0], "last": group_ids[-1]} == group["evidence_id_range"]
        visited.extend(group_ids)
    expected = sorted(item["evidence_id"] for item in catalog["evidence"])
    assert sorted(visited) == expected
    assert len(visited) == len(set(visited))


def test_group_index_fails_closed_when_payload_exceeds_cap(grounded_ws, monkeypatch):
    monkeypatch.setattr(G, "GROUP_INDEX_PAYLOAD_BYTE_CAP", 1)
    with pytest.raises(ValueError, match="byte 제한"):
        G.evidence_catalog_group_index("one")


def test_group_pages_have_no_missing_or_duplicate_ids_and_details_preserve_raw_rows(grounded_ws):
    catalog = G.evidence_catalog("one")
    index = G.evidence_catalog_group_index("one")
    target = next(group for group in index["groups"]
                  if group["kind"] == "telemetry_summary"
                  and group["path_prefix"] == ["Custom/contact"])
    offset = 0
    visited = []
    while True:
        page = G.evidence_catalog_group_page(
            "one", catalog["catalog_sha256"], target["group_id"], offset=offset, limit=2,
        )
        assert page["values_included"] is False
        assert page["catalog_coverage_status"] == "partial"
        assert not _has_exact_key(page, "value")
        visited.extend(item["evidence_id"] for item in page["evidence"])
        if page["next_offset"] is None:
            assert page["unvisited_after_count"] == 0
            break
        offset = page["next_offset"]

    expected = sorted(
        item["evidence_id"] for item in catalog["evidence"]
        if item["kind"] == "telemetry_summary" and item["source"]["tag"] == "Custom/contact"
    )
    assert visited == expected
    assert len(visited) == len(set(visited)) == target["evidence_count"]

    selected_ids = [
        "telemetry.summary::Custom/contact::first",
        "telemetry.summary::Custom/contact::last",
        "telemetry.summary::Custom/contact::mean_last_20pct",
    ]
    details = G.evidence_catalog_details("one", catalog["catalog_sha256"], selected_ids)
    assert details["selection_binding"]["case_id"] == "one"
    assert details["selection_binding"]["catalog_sha256"] == catalog["catalog_sha256"]
    assert details["selection_binding"]["selected_evidence_ids"] == sorted(selected_ids)
    rows = {item["evidence_id"]: item for item in details["evidence"]}
    full_rows = {item["evidence_id"]: item for item in catalog["evidence"]}
    assert rows == {evidence_id: full_rows[evidence_id] for evidence_id in sorted(selected_ids)}
    assert rows[selected_ids[0]]["state"] == "null" and rows[selected_ids[0]]["value"] is None
    assert rows[selected_ids[1]]["state"] == "present" and rows[selected_ids[1]]["value"] == 0.0
    assert rows[selected_ids[2]]["state"] == "missing" and rows[selected_ids[2]]["value"] is None


@pytest.mark.parametrize("kwargs, message", [
    ({"offset": True}, "offset"),
    ({"offset": -1}, "offset"),
    ({"offset": 7}, "offset"),
    ({"limit": True}, "limit"),
    ({"limit": 0}, "limit"),
    ({"limit": 65}, "limit"),
    ({"group_id": "grp_unknown"}, "없는 group"),
    ({"catalog_sha256": "0" * 64}, "catalog_sha256"),
])
def test_group_page_rejects_invalid_bounds_group_and_catalog_hash(grounded_ws, kwargs, message):
    catalog = G.evidence_catalog("one")
    index = G.evidence_catalog_group_index("one")
    group = next(item for item in index["groups"] if item["evidence_count"] == 7)
    arguments = {
        "case_id": "one", "catalog_sha256": catalog["catalog_sha256"],
        "group_id": group["group_id"], "offset": 0, "limit": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        G.evidence_catalog_group_page(**arguments)


@pytest.mark.parametrize("evidence_ids, message", [
    ([], "비어 있지 않은"),
    (["missing"] * 2, "중복"),
    (["missing"], "없는 evidence"),
    ([1], "문자열"),
    ([f"missing-{index}" for index in range(17)], "최대 16개"),
])
def test_details_reject_empty_duplicate_unknown_typed_and_excess_ids(grounded_ws, evidence_ids, message):
    catalog = G.evidence_catalog("one")
    with pytest.raises(ValueError, match=message):
        G.evidence_catalog_details("one", catalog["catalog_sha256"], evidence_ids)


def test_group_and_detail_reads_reject_source_mutation(grounded_ws):
    _, case = grounded_ws
    catalog = G.evidence_catalog("one")
    index = G.evidence_catalog_group_index("one")
    group_id = index["groups"][0]["group_id"]
    evidence_id = catalog["evidence"][0]["evidence_id"]
    (case / "telemetry.json").write_bytes((case / "telemetry.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="catalog_sha256|telemetry_sha256"):
        G.evidence_catalog_group_page("one", catalog["catalog_sha256"], group_id)
    with pytest.raises(ValueError, match="catalog_sha256|telemetry_sha256"):
        G.evidence_catalog_details("one", catalog["catalog_sha256"], [evidence_id])


def test_empty_fact_context_keeps_configuration_and_context_source_bindings(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {
        "reference_status": "absent", "summary": {}, "series": {},
    })
    case = tmp_path / "cases" / "empty"
    case.mkdir(parents=True)
    telemetry_sha = _write_json(case / "telemetry.json", {
        "summary": {"x": {"n": 1, "last": 0.0}}, "series": {"x": [0.0]},
    })
    configuration_sha = _write_json(case / "configuration.json", {"rewards": {}})
    context_sha = _write_json(case / "context.json", {
        "schema": "single_run_context_v1", "telemetry_sha256": telemetry_sha,
        "configuration_sha256": configuration_sha, "facts": [],
    })
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)

    catalog = G.evidence_catalog("empty")
    index = G.evidence_catalog_group_index("empty")
    assert catalog["configuration_sha256"] == configuration_sha
    assert catalog["context_sha256"] == context_sha
    assert index["source_bindings"]["configuration"] == {
        "status": "provided", "configuration_sha256": configuration_sha,
        "context_sha256": context_sha,
    }

    old_catalog_sha = catalog["catalog_sha256"]
    new_configuration_sha = _write_json(case / "configuration.json", {"rewards": {"changed": True}})
    context = json.loads((case / "context.json").read_text(encoding="utf-8"))
    context["configuration_sha256"] = new_configuration_sha
    _write_json(case / "context.json", context)
    with pytest.raises(ValueError, match="catalog_sha256"):
        G.evidence_catalog_details("empty", old_catalog_sha, ["telemetry.summary::x::last"])


def test_writer_requires_host_selection_and_rejects_unselected_premise_first(grounded_ws):
    workspace, _ = grounded_ws
    catalog = G.evidence_catalog("one")
    selected_id = "telemetry.summary::Custom/contact::last"
    selection = _selection_binding(catalog, [selected_id])
    malformed = [{"unexpected": True}]
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], {**selection, "selection_id": "forged"},
        malformed, {"unexpected": True},
    )
    assert "selection_id" in result["error"]

    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, _hypotheses(), _next_check(),
    )
    assert "selection에 포함되지 않은" in result["error"]
    assert not (workspace / "preregistrations" / "grounded_assessments").exists()


def test_writer_rejects_tampered_or_cross_catalog_selection_receipt(grounded_ws):
    workspace, case = grounded_ws
    catalog = G.evidence_catalog("one")
    selection = _selection_binding(catalog)
    receipt = (workspace / "preregistrations" / "evidence_selections"
               / f"one__{selection['selection_id']}.json")
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    stored["selected_evidence_ids"] = ["telemetry.summary::Custom/contact::last"]
    receipt.write_text(json.dumps(stored), encoding="utf-8")
    result = G.write_grounded_assessment(
        "one", catalog["catalog_sha256"], selection, _hypotheses(), _next_check(),
    )
    assert "receipt SHA-256" in result["error"]

    clean_selection = _selection_binding(catalog)
    telemetry = json.loads((case / "telemetry.json").read_text(encoding="utf-8"))
    telemetry["summary"]["Train/mean_reward"]["last"] = 1.0
    new_telemetry_sha = _write_json(case / "telemetry.json", telemetry)
    context = json.loads((case / "context.json").read_text(encoding="utf-8"))
    context["telemetry_sha256"] = new_telemetry_sha
    _write_json(case / "context.json", context)
    result = G.write_grounded_assessment(
        "one", clean_selection["catalog_sha256"], clean_selection,
        _hypotheses(), _next_check(),
    )
    assert "catalog_sha256" in result["error"]


def test_legacy_full_catalog_cpu_and_nat_apis_remain_available(grounded_ws):
    expected = G.evidence_catalog("one")
    builder = cast(Builder, Mock(spec=Builder))

    async def invoke():
        async with N.grounded_evidence_catalog_fn(N.GroundedEvidenceCatalogConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            return json.loads(await tool.single_fn("one"))

    assert asyncio.run(invoke()) == expected


def test_grounded_nat_registration_and_config_exclude_free_code_tool(grounded_ws, monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "triage_single_run_grounded.yml").read_text(encoding="utf-8"))
    assert config["workflow"]["tool_names"] == [
        "grounded_evidence_group_index", "grounded_evidence_group_page",
        "grounded_evidence_details", "write_grounded_assessment",
    ]
    assert "single_run_analysis" not in config["functions"]
    builder = cast(Builder, Mock(spec=Builder))
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TRIAGE_TRACE", str(trace))

    async def invoke():
        async with N.grounded_evidence_group_index_fn(N.GroundedEvidenceGroupIndexConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            index = json.loads(await tool.single_fn("one"))
        group = next(item for item in index["groups"]
                     if item["path_prefix"] == ["Train/mean_reward"])
        async with N.grounded_evidence_group_page_fn(N.GroundedEvidenceGroupPageConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            page_args = tool.input_schema(
                case_id="one", catalog_sha256=index["catalog_sha256"],
                group_id=group["group_id"], offset=0, limit=64,
            )
            page = json.loads(await tool.single_fn(page_args))
        async with N.grounded_evidence_details_fn(N.GroundedEvidenceDetailsConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            detail_args = tool.input_schema(
                case_id="one", catalog_sha256=index["catalog_sha256"],
                evidence_ids=["telemetry.summary::Train/mean_reward::last"],
            )
            details = json.loads(await tool.single_fn(detail_args))
        async with N.write_grounded_assessment_fn(N.WriteGroundedAssessmentConfig(), builder) as tool:
            assert isinstance(tool, FunctionInfo) and tool.single_fn is not None
            args = tool.input_schema(
                case_id="one", catalog_sha256=index["catalog_sha256"],
                selection_binding=details["selection_binding"],
                hypotheses=_hypotheses(), next_check=_next_check(),
            )
            saved = json.loads(await tool.single_fn(args))
        return index, page, details, saved

    index, page, details, saved = asyncio.run(invoke())
    assert index["source_bindings"]["configuration"]["status"] == "provided"
    assert page["values_included"] is False
    assert details["selected_evidence_count"] == 1
    assert saved["selected_evidence_ids"] == details["selection_binding"]["selected_evidence_ids"]
    assert saved["status"] == "hypotheses_only"
    calls = [json.loads(line)["tool"] for line in trace.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        "grounded_evidence_group_index", "grounded_evidence_group_page",
        "grounded_evidence_details", "write_grounded_assessment",
    ]
