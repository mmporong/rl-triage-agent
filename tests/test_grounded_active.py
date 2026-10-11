from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rl_triage import grounded_active_profile as A
from rl_triage import triage_tools as T


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def predictions():
    return [
        {"hypothesis_id": "h1", "predictions": [
            {"measurement_id": "base_com_max_abs_delta_xy", "relation": "lte",
             "operand": {"value": 0.0501, "unit": "simulation_length_native"}},
            {"measurement_id": "base_com_outside_range_fraction", "relation": "eq",
             "operand": {"value": 0.0, "unit": "fraction"}},
        ]},
        {"hypothesis_id": "h2", "predictions": [
            {"measurement_id": "base_com_max_abs_delta_xy", "relation": "gt",
             "operand": {"value": 0.0501, "unit": "simulation_length_native"}},
            {"measurement_id": "base_com_outside_range_fraction", "relation": "gt",
             "operand": {"value": 0.0, "unit": "fraction"}},
        ]},
    ]


@pytest.fixture
def active_ws(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(T, "WORKSPACE", workspace)
    case = workspace / "cases/one"
    sources = {}
    for name, value in {
        "telemetry": {"summary": {}, "series": {}},
        "configuration": {"task": "go2"},
        "context": {"status": "provided"},
    }.items():
        sha = write_json(case / f"{name}.json", value)
        sources[name] = {"path": f"cases/one/{name}.json", "sha256": sha}
    assessment = {
        "schema": "grounded_active_assessment_v1", "case_id": "one", "status": "hypotheses_only",
        "registry_id": A.ACTIVE_REGISTRY_ID, "registry_sha256": A._sha(A.ACTIVE_CHECKS),
        "selected_evidence_ids": ["telemetry:a"],
        "hypotheses": [{"id": "h1", "rank": 1}, {"id": "h2", "rank": 2}],
        "next_check": {"registry_id": "reset_base_com_stability",
                       "hypothesis_predictions": predictions()},
    }
    A._write_once("grounded_active_assessments", "one.json", assessment)
    manifest = {
        "schema": "grounded_active_case_manifest_v1", "case_id": "one",
        "public_case_alias": "public-one", "sources": sources,
        "assessment_sha256": A._sha(assessment), "task": "go2", "seed": 7,
        "protocol_sha256": "a" * 64, "source_commit_sha": "b" * 40,
        "private_executor_binding_sha256": "c" * 64,
    }
    binding = A.validate_case_manifest("one", manifest)
    return workspace, assessment, manifest, binding


def approved(active_ws):
    _, _, _, binding = active_ws
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"],
        "reset_base_com_stability", deepcopy(A.FIXED_EXECUTION_ARGUMENTS),
    )
    approval = A.approve_execution("one", request, "operator")
    consumed = A.consume_approval("one", approval)
    return binding, request, approval, consumed


def valid_receipt(binding, request, approval, consumed):
    return {
        "schema": "grounded_active_execution_receipt_v1", "case_id": "one",
        "effect_key": request["effect_key"],
        "case_source_binding_sha256": binding["case_source_binding_sha256"],
        "request_sha256": request["request_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "consumed_sha256": consumed["consumed_sha256"],
        "registry_id": "reset_base_com_stability",
        "check_definition_sha256": A._sha(A.ACTIVE_CHECKS["reset_base_com_stability"]),
        "parent_status": "pass", "parent_receipt_sha256": "d" * 64,
        "body_sha256": "e" * 64, "natural_exit": True, "cleanup_verified": True,
        "source_verified": True, "coverage_status": "complete",
        "measurements": [
            {"measurement_id": "base_com_max_abs_delta_xy",
             "unit": "simulation_length_native", "state": "present", "value": 0.09},
            {"measurement_id": "base_com_outside_range_fraction",
             "unit": "fraction", "state": "present", "value": 0.5},
        ],
        "model_selected": False, "diagnosis_confirmed": False, "cause_confirmed": False,
        "recovery_confirmed": False, "policy_recovery": False,
    }


def test_registry_is_separate_and_preserves_read_only_checks():
    profile = A.active_registry_profile()
    assert set(profile["checks"]) == {*A.G.READ_ONLY_CHECKS, "reset_base_com_stability"}
    assert profile["checks"]["reset_base_com_stability"]["safety"] == "approved_simulation_reset"
    assert A.G.READ_ONLY_CHECKS.keys().isdisjoint({"reset_base_com_stability"})
    next_check = {"registry_id": "reset_base_com_stability",
                  "hypothesis_predictions": predictions()}
    checked = A.G._checked_next_check(next_check, {"h1", "h2"}, A.ACTIVE_CHECKS)
    assert checked["safety"] == "approved_simulation_reset"
    with pytest.raises(ValueError, match="등록되지 않은"):
        A.G._checked_next_check(next_check, {"h1", "h2"})


def test_active_writer_uses_host_selection_and_active_registry(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(T, "WORKSPACE", workspace)
    catalog = {
        "catalog_sha256": "a" * 64, "telemetry_sha256": "b" * 64,
        "context_status": "provided",
    }
    checked = [
        {"id": "h1", "rank": 1, "narrative": "one", "premises": []},
        {"id": "h2", "rank": 2, "narrative": "two", "premises": []},
    ]
    monkeypatch.setattr(A.G, "_validated_selection",
                        lambda *_args: (catalog, ["telemetry:a"]))
    monkeypatch.setattr(A.G, "_checked_hypotheses",
                        lambda *_args: (checked, []))
    selection = {"selection_id": "sel_" + "1" * 32}
    result = A.write_grounded_active_assessment(
        "one", "a" * 64, selection, checked,
        {"registry_id": "reset_base_com_stability",
         "hypothesis_predictions": predictions()},
    )
    assert result["next_check"]["registry_id"] == "reset_base_com_stability"
    assert result["next_check"]["execution_status"] == "approval_pending"
    assert result["registry_sha256"] == A._sha(A.ACTIVE_CHECKS)
    assert A._load_bound("grounded_active_assessments", "one.json")["selection_binding"] == selection
    second = A.write_grounded_active_assessment(
        "one", "a" * 64, selection, checked,
        {"registry_id": "reset_base_com_stability",
         "hypothesis_predictions": predictions()},
    )
    assert "덮어쓸 수 없습니다" in second["error"]


def test_nat_config_exposes_no_approval_consume_or_execution_tool():
    config = yaml.safe_load(Path("configs/triage_single_run_grounded_active.yml").read_text(encoding="utf-8"))
    tools = config["workflow"]["tool_names"]
    assert "grounded_active_registry" in tools
    assert "write_grounded_active_assessment" in tools
    assert all(word not in " ".join(tools) for word in ("approve", "consume", "execute", "shell"))
    assert config["llms"]["nemotron"]["temperature"] == 0
    assert config["llms"]["nemotron"]["max_retries"] == 0
    assert config["llms"]["nemotron"]["request_timeout"] == 300
    assert config["workflow"]["truncation_retry"]["max_retries"] == 0


def test_manifest_binds_public_sources_but_keeps_case_and_probe_seeds_distinct(active_ws):
    _, _, manifest, binding = active_ws
    assert manifest["seed"] == 7
    assert A.FIXED_EXECUTION_ARGUMENTS["seed"] == 20261012
    assert len(binding["case_source_binding_sha256"]) == 64
    assert A.validate_case_manifest("one", manifest) == binding


def test_manifest_rejects_source_mutation_and_cross_case(active_ws):
    workspace, _, manifest, _ = active_ws
    (workspace / "cases/one/telemetry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA"):
        A.validate_case_manifest("one", manifest)
    with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
        A.validate_case_manifest("../one", manifest)


def test_request_requires_validated_source_and_fixed_arguments(active_ws):
    _, _, _, binding = active_ws
    with pytest.raises(FileNotFoundError):
        A.request_execution("one", "0" * 64, "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS)
    changed = {**A.FIXED_EXECUTION_ARGUMENTS, "seed": 7}
    with pytest.raises(ValueError, match="fixed execution"):
        A.request_execution("one", binding["case_source_binding_sha256"],
                            "reset_base_com_stability", changed)


def test_request_requires_the_model_selected_reset_check(active_ws):
    _, assessment, manifest, _ = active_ws
    changed = deepcopy(assessment)
    changed["next_check"]["registry_id"] = "runtime_reward_window"
    path = T.WORKSPACE / "preregistrations/grounded_active_assessments/one.json"
    path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    changed_manifest = deepcopy(manifest)
    changed_manifest["assessment_sha256"] = A._sha(changed)
    binding = A.validate_case_manifest("one", changed_manifest)
    with pytest.raises(ValueError, match="요청한 검사"):
        A.request_execution(
            "one", binding["case_source_binding_sha256"],
            "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
        )


def test_approval_is_write_once_and_consumed_before_side_effect(active_ws):
    _, request, approval, consumed = approved(active_ws)
    assert consumed["consumed_sha256"]
    stored = A._load_bound("active_consumed", f"one__{request['effect_key']}.json")
    assert stored["side_effect_started"] is False
    with pytest.raises(FileExistsError):
        A.consume_approval("one", approval)


def test_consume_revalidates_bound_case_sources(active_ws):
    workspace, _, _, binding = active_ws
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"],
        "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
    )
    approval = A.approve_execution("one", request, "operator")
    (workspace / "cases/one/context.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA"):
        A.consume_approval("one", approval)


def test_orphan_tampered_and_cross_source_approval_are_rejected(active_ws):
    _, _, _, binding = active_ws
    fake = {"effect_key": "0" * 64, "request_sha256": "1" * 64,
            "approval_sha256": "2" * 64,
            "case_source_binding_sha256": binding["case_source_binding_sha256"]}
    with pytest.raises(FileNotFoundError):
        A.consume_approval("one", fake)
    request = A.request_execution("one", binding["case_source_binding_sha256"],
                                  "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS)
    changed = {**request, "request_sha256": "3" * 64}
    with pytest.raises(ValueError):
        A.approve_execution("one", changed, "operator")


@pytest.mark.parametrize("field,value", [
    ("schema", "forged"),
    ("effect_key", "f" * 64),
    ("execution_limit", True),
])
def test_approve_rejects_semantically_tampered_request(active_ws, field, value):
    workspace, _, _, binding = active_ws
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"],
        "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
    )
    path = workspace / f"preregistrations/active_requests/one__{request['effect_key']}.json"
    document = A._load_bound("active_requests", f"one__{request['effect_key']}.json")
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    forged = {**request, "request_sha256": A._sha(document)}
    with pytest.raises(ValueError):
        A.approve_execution("one", forged, "operator")


@pytest.mark.parametrize("field,value", [
    ("schema", "forged"),
    ("case_id", "other"),
    ("effect_key", "f" * 64),
    ("approved", False),
    ("identity_attested", True),
])
def test_consume_rejects_semantically_tampered_approval(active_ws, field, value):
    workspace, _, _, binding = active_ws
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"],
        "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
    )
    approval = A.approve_execution("one", request, "operator")
    path = workspace / f"preregistrations/active_approvals/one__{request['effect_key']}.json"
    document = A._load_bound("active_approvals", f"one__{request['effect_key']}.json")
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    forged = {**approval, "approval_sha256": A._sha(document)}
    with pytest.raises(ValueError):
        A.consume_approval("one", forged)


def test_request_rejects_binding_whose_sources_and_file_changed_under_stale_binding_sha(active_ws):
    workspace, _, _, binding = active_ws
    context = workspace / "cases/one/context.json"
    context.write_text('{"status":"changed"}', encoding="utf-8")
    binding_path = workspace / (
        "preregistrations/active_case_bindings/"
        f"one__{binding['case_source_binding_sha256']}.json"
    )
    forged = deepcopy(binding)
    forged["sources"]["context"]["sha256"] = hashlib.sha256(context.read_bytes()).hexdigest()
    binding_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        A.request_execution(
            "one", binding["case_source_binding_sha256"],
            "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
        )


def test_atomic_publish_failure_leaves_no_partial_target(active_ws, monkeypatch):
    workspace, _, _, _ = active_ws
    monkeypatch.setattr(A.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("link failed")))
    with pytest.raises(OSError, match="link failed"):
        A._write_once("partial", "record.json", {"value": 1})
    directory = workspace / "preregistrations/partial"
    assert not (directory / "record.json").exists()
    assert list(directory.iterdir()) == []


def test_invalid_or_partial_receipt_masks_both_values(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    receipt = valid_receipt(binding, request, approval, consumed)
    receipt["coverage_status"] = "partial"
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"], receipt)
    assert normalized["execution_valid"] is False
    assert [row["value"] for row in normalized["measurements"]] == [None, None]
    assert normalized["diagnosis_confirmed"] is normalized["policy_recovery"] is False


def test_receipt_with_changed_source_binding_is_unavailable(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    changed_source = "f" * 64
    receipt = valid_receipt(binding, request, approval, consumed)
    receipt["case_source_binding_sha256"] = changed_source
    with pytest.raises(ValueError, match="consumed"):
        A.normalize_execution_receipt(
            "one", changed_source, request["effect_key"], receipt)


def test_negative_max_abs_measurement_is_unavailable(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    receipt = valid_receipt(binding, request, approval, consumed)
    receipt["measurements"][0]["value"] = -0.01
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"], receipt)
    assert normalized["execution_valid"] is False
    assert all(row["state"] == "unavailable" for row in normalized["measurements"])


def test_normalize_rejects_orphan_consumed_marker_without_original_request(active_ws):
    workspace, _, _, _ = active_ws
    binding, request, approval, consumed = approved(active_ws)
    request_path = workspace / f"preregistrations/active_requests/one__{request['effect_key']}.json"
    request_path.unlink()
    with pytest.raises(FileNotFoundError):
        A.normalize_execution_receipt(
            "one", binding["case_source_binding_sha256"], request["effect_key"],
            valid_receipt(binding, request, approval, consumed),
        )


@pytest.mark.parametrize("field,value", [
    ("schema", "forged"),
    ("side_effect_started", True),
])
def test_normalize_rejects_semantically_tampered_consumed_marker(
        active_ws, field, value):
    workspace, _, _, _ = active_ws
    binding, request, approval, consumed = approved(active_ws)
    path = workspace / f"preregistrations/active_consumed/one__{request['effect_key']}.json"
    document = A._load_bound("active_consumed", f"one__{request['effect_key']}.json")
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    receipt = valid_receipt(binding, request, approval, consumed)
    receipt["consumed_sha256"] = A._sha(document)
    with pytest.raises(ValueError, match="consumed"):
        A.normalize_execution_receipt(
            "one", binding["case_source_binding_sha256"], request["effect_key"], receipt)


def test_actual_and_masked_packets_differ_only_in_measurement_values(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"],
        valid_receipt(binding, request, approval, consumed))
    actual = A.build_post_measurement_packet("one", normalized)
    masked = A.build_post_measurement_packet("one", normalized, value_masked=True)
    actual_values = [row.pop("value") for row in actual["normalized_receipt"]["measurements"]]
    masked_values = [row.pop("value") for row in masked["normalized_receipt"]["measurements"]]
    assert actual == masked
    assert actual_values == [0.09, 0.5] and masked_values == [None, None]
    assert actual["normalized_receipt_sha256"] == normalized["normalized_receipt_sha256"]
    serialized = json.dumps(masked)
    for forbidden in ("private_executor", "arm", "private_path", "source_commit_sha"):
        assert forbidden not in serialized
    assert masked["budget_left"] == 0


def test_final_decision_requires_latest_exact_receipt_and_budget_zero(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"],
        valid_receipt(binding, request, approval, consumed))
    packet = A.build_post_measurement_packet("one", normalized)
    decision = {"schema": "grounded_active_final_decision_v1", "case_id": "one",
                "effect_key": request["effect_key"], "budget_left": 0, "status": "HOLD",
                "ranking": ["h2", "h1"],
                "receipt_citation": {"effect_key": request["effect_key"],
                                     "normalized_receipt_sha256": packet["normalized_receipt_sha256"]}}
    assert A.validate_final_decision(packet, decision) == decision
    decision["budget_left"] = 1
    with pytest.raises(ValueError, match="HOLD"):
        A.validate_final_decision(packet, decision)

    decision["budget_left"] = 0
    packet["selected_evidence_ids"] = ["forged"]
    with pytest.raises(ValueError, match="최신 assessment/receipt"):
        A.validate_final_decision(packet, decision)


def test_ref_or_mechanism_measurement_never_claims_policy_recovery(active_ws):
    binding, request, approval, consumed = approved(active_ws)
    receipt = valid_receipt(binding, request, approval, consumed)
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"], receipt)
    assert normalized["policy_recovery"] is False
    assert "arm" not in json.dumps(normalized)


def test_judgment_influence_requires_actual_change_and_masked_no_update(active_ws):
    _, assessment, _, binding = active_ws
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"], "reset_base_com_stability",
        A.FIXED_EXECUTION_ARGUMENTS,
    )
    approval = A.approve_execution("one", request, "operator")
    consumed = A.consume_approval("one", approval)
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"],
        valid_receipt(binding, request, approval, consumed))
    actual_packet = A.build_post_measurement_packet("one", normalized)
    masked_packet = A.build_post_measurement_packet("one", normalized, value_masked=True)

    def decision(packet, ranking):
        return {"schema": "grounded_active_final_decision_v1", "case_id": "one",
                "effect_key": request["effect_key"], "budget_left": 0, "status": "HOLD",
                "ranking": ranking,
                "receipt_citation": {"effect_key": request["effect_key"],
                                     "normalized_receipt_sha256": packet["normalized_receipt_sha256"]}}

    proof = A.build_judgment_influence_proof(
        assessment, actual_packet, decision(actual_packet, ["h2", "h1"]),
        masked_packet, decision(masked_packet, ["h1", "h2"]),
    )
    assert proof["judgment_influence_proven"] is True
    assert proof["correctness_proven"] is proof["superiority_proven"] is False
    assert proof["policy_recovery_proven"] is False
    assert A.compare_decision_rankings(
        assessment, decision(actual_packet, ["h2", "h1"]),
        decision(masked_packet, ["h1", "h2"]),
    ) == {
        "initial_top": "h1", "actual_top": "h2", "masked_top": "h1",
        "ranking_changed_with_values": True, "fixed_no_update": True,
    }


def test_judgment_proof_rejects_masked_packet_in_actual_role(active_ws):
    workspace, assessment, manifest, _ = active_ws
    eq_ne = deepcopy(assessment)
    for index, row in enumerate(eq_ne["next_check"]["hypothesis_predictions"]):
        relation = "eq" if index == 0 else "ne"
        for prediction in row["predictions"]:
            prediction["relation"] = relation
            prediction["operand"]["value"] = 999.0
    assessment_path = workspace / "preregistrations/grounded_active_assessments/one.json"
    assessment_path.write_text(json.dumps(eq_ne), encoding="utf-8")
    changed_manifest = deepcopy(manifest)
    changed_manifest["assessment_sha256"] = A._sha(eq_ne)
    binding = A.validate_case_manifest("one", changed_manifest)
    request = A.request_execution(
        "one", binding["case_source_binding_sha256"],
        "reset_base_com_stability", A.FIXED_EXECUTION_ARGUMENTS,
    )
    approval = A.approve_execution("one", request, "operator")
    consumed = A.consume_approval("one", approval)
    normalized = A.normalize_execution_receipt(
        "one", binding["case_source_binding_sha256"], request["effect_key"],
        valid_receipt(binding, request, approval, consumed),
    )
    masked = A.build_post_measurement_packet("one", normalized, value_masked=True)

    def decision(ranking):
        return {
            "schema": "grounded_active_final_decision_v1", "case_id": "one",
            "effect_key": request["effect_key"], "budget_left": 0, "status": "HOLD",
            "ranking": ranking,
            "receipt_citation": {
                "effect_key": request["effect_key"],
                "normalized_receipt_sha256": masked["normalized_receipt_sha256"],
            },
        }

    proof = A.build_judgment_influence_proof(
        eq_ne, masked, decision(["h2", "h1"]),
        masked, decision(["h1", "h2"]),
    )
    assert proof["judgment_influence_proven"] is False
