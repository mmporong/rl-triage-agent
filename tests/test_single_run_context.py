import hashlib
import json
import os

import pytest

from rl_triage import single_run as S
from rl_triage import triage_tools as T


TELEMETRY = {"summary": {"Train/mean_reward": {"n": 1}}, "series": {"Train/mean_reward": [1.0]}}
CONFIGURATION = {
    "rewards": {"stand": {"weight": 10}},
    "observations": {"contact": {"bodies": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]}},
}
FACTS = [
    {"id": "stand_weight", "path": ["rewards", "stand", "weight"]},
    {"id": "contact_bodies", "path": ["observations", "contact", "bodies"]},
]


def _write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_context(case, facts=FACTS, **overrides):
    telemetry_sha = hashlib.sha256((case / "telemetry.json").read_bytes()).hexdigest()
    configuration_sha = hashlib.sha256((case / "configuration.json").read_bytes()).hexdigest()
    manifest = {
        "schema": "single_run_context_v1",
        "telemetry_sha256": telemetry_sha,
        "configuration_sha256": configuration_sha,
        "facts": facts,
        **overrides,
    }
    _write_json(case / "context.json", manifest)
    return manifest


@pytest.fixture()
def context_ws(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_json(reference / "telemetry.json", {
        "reference_status": "absent", "summary": {}, "series": {},
    })
    case = tmp_path / "cases" / "one"
    case.mkdir(parents=True)
    _write_json(case / "telemetry.json", TELEMETRY)
    _write_json(case / "configuration.json", CONFIGURATION)
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    return tmp_path, case


def _assessment(claims=None):
    return S.write_assessment("one", ["competing hypotheses"], "measured reward=1", "single run only",
                              "measure contact timing", fact_claims=claims)


def test_missing_context_keeps_backward_compatible_assessment(context_ws):
    workspace, _ = context_ws
    assert S.get_context("one") == {"case_id": "one", "status": "not_provided", "facts": []}
    result = _assessment()
    assert result["context"] == {"status": "not_provided"}
    assert result["verified_fact_claims"] == []
    assert (workspace / "preregistrations" / "assessments" / "one.json").is_file()


def test_context_claim_presence_contract_rejects_without_artifact(context_ws):
    workspace, case = context_ws
    _write_context(case)
    missing_claims = _assessment()
    assert "하나 이상 인용" in missing_claims["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()

    (case / "context.json").unlink()
    claims_without_context = _assessment([{"id": "stand_weight", "value": 10}])
    assert "context가 없으면" in claims_without_context["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()


def test_selected_configuration_facts_reject_conflicts_and_save_matching_claims(context_ws):
    workspace, case = context_ws
    _write_context(case)
    context = S.get_context("one")
    assert [(fact["id"], fact["value"]) for fact in context["facts"]] == [
        ("stand_weight", 10),
        ("contact_bodies", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
    ]

    conflict = _assessment([{"id": "stand_weight", "value": 0}])
    assert "일치하지 않습니다" in conflict["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()
    absent = _assessment([{"id": "contact_bodies", "value": []}])
    assert "일치하지 않습니다" in absent["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()

    claims = [
        {"id": "stand_weight", "value": 10.0},
        {"id": "contact_bodies", "value": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]},
    ]
    saved = _assessment(claims)
    assert saved["status"] == "hypotheses_only"
    assert saved["context"]["status"] == "provided"
    assert [claim["id"] for claim in saved["verified_fact_claims"]] == ["stand_weight", "contact_bodies"]
    assert all(claim["configuration_sha256"] == context["configuration_sha256"]
               for claim in saved["verified_fact_claims"])


def test_context_hashes_are_rechecked_after_input_changes(context_ws):
    workspace, case = context_ws
    _write_context(case)
    (case / "telemetry.json").write_bytes((case / "telemetry.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="telemetry_sha256"):
        S.get_context("one")
    result = _assessment([{"id": "stand_weight", "value": 10}])
    assert "telemetry_sha256" in result["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()

    _write_json(case / "telemetry.json", TELEMETRY)
    _write_context(case)
    (case / "configuration.json").write_bytes((case / "configuration.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="configuration_sha256"):
        S.get_context("one")


def test_missing_selected_path_is_invalid_metadata(context_ws):
    _, case = context_ws
    _write_context(case, facts=[{"id": "missing", "path": ["rewards", "missing", "weight"]}])
    with pytest.raises(ValueError, match="없는 fact path"):
        S.get_context("one")


@pytest.mark.parametrize("root", ["truth", "answer_key"])
def test_sensitive_fact_roots_are_rejected(context_ws, root):
    _, case = context_ws
    _write_context(case, facts=[{"id": "forbidden", "path": [root]}])
    with pytest.raises(ValueError, match="허용되지 않은"):
        S.get_context("one")


@pytest.mark.parametrize("index", [-1, True])
def test_negative_and_bool_list_indices_are_rejected(context_ws, index):
    _, case = context_ws
    fact = {"id": "bad_index", "path": ["observations", "contact", "bodies", index]}
    _write_context(case, facts=[fact])
    with pytest.raises(ValueError, match="없는 fact path"):
        S.get_context("one")


def test_nested_selected_object_is_rejected(context_ws):
    _, case = context_ws
    _write_context(case, facts=[{"id": "nested", "path": ["rewards", "stand"]}])
    with pytest.raises(ValueError, match="scalar 또는 flat scalar list"):
        S.get_context("one")


@pytest.mark.parametrize("change, message", [
    ({"schema": "single_run_context_v2"}, "schema"),
    ({"truth": "answer"}, "최상위 키"),
])
def test_invalid_schema_or_truth_metadata_fails_closed(context_ws, change, message):
    _, case = context_ws
    _write_context(case, **change)
    with pytest.raises(ValueError, match=message):
        S.get_context("one")


def test_duplicate_fact_id_and_bool_numeric_type_trick_are_rejected(context_ws):
    workspace, case = context_ws
    _write_context(case, facts=[FACTS[0], {"id": "stand_weight", "path": FACTS[1]["path"]}])
    with pytest.raises(ValueError, match="중복 fact id"):
        S.get_context("one")

    _write_context(case)
    result = _assessment([{"id": "stand_weight", "value": True}])
    assert "일치하지 않습니다" in result["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()


@pytest.mark.parametrize("claims, message", [
    ([{"id": "stand_weight", "value": 10}, {"id": "stand_weight", "value": 10}], "중복"),
    ([{"id": "unknown", "value": 10}], "없는 fact id"),
])
def test_duplicate_or_unknown_claim_ids_are_rejected_without_artifact(context_ws, claims, message):
    workspace, case = context_ws
    _write_context(case)
    result = _assessment(claims)
    assert message in result["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()


def test_nonfinite_selected_value_is_rejected(context_ws):
    _, case = context_ws
    raw = b'{"rewards":{"stand":{"weight":NaN}}}'
    (case / "configuration.json").write_bytes(raw)
    _write_context(case, facts=[FACTS[0]])
    with pytest.raises(ValueError, match="유한하지 않은"):
        S.get_context("one")


def test_very_large_integer_claims_compare_exactly_without_float_conversion(context_ws):
    workspace, case = context_ws
    huge = 10 ** 400
    _write_json(case / "configuration.json", {"rewards": {"stand": {"weight": huge}}})
    _write_context(case, facts=[FACTS[0]])

    mismatch = _assessment([{"id": "stand_weight", "value": huge + 1}])
    assert "일치하지 않습니다" in mismatch["error"]
    assert not (workspace / "preregistrations" / "assessments" / "one.json").exists()

    saved = _assessment([{"id": "stand_weight", "value": huge}])
    assert saved["verified_fact_claims"][0]["value"] == huge


def _symlink_or_skip(target, link, directory=False):
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable on this host: {exc}")


def test_case_directory_symlink_is_rejected(context_ws):
    workspace, case = context_ws
    _symlink_or_skip(case, workspace / "cases" / "linked", directory=True)
    with pytest.raises(PermissionError, match="디렉터리 심볼릭 링크"):
        S.get_context("linked")


@pytest.mark.parametrize("name", ["telemetry.json", "context.json", "configuration.json"])
def test_case_input_symlinks_are_rejected(context_ws, name):
    workspace, case = context_ws
    _write_context(case)
    other = workspace / "cases" / "other"
    other.mkdir()
    target = other / name
    target.write_bytes((case / name).read_bytes())
    (case / name).unlink()
    _symlink_or_skip(target, case / name)
    with pytest.raises(PermissionError, match="심볼릭 링크"):
        S.get_context("one")
