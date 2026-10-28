"""v2는 prompt만 강화하며 실제 v1 오류의 의미를 유효로 바꾸지 않는다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"evals"), str(ROOT/"src")]
import agentic_investigation_contract_v2 as A
import test_selection_comparison_controller_v1 as F


def context(tmp_path, evidence=None):
    return A.prepare(F.snapshot(tmp_path, "case", "model_adaptive"), goal="학습 실패 조사", evidence=evidence)


@pytest.mark.parametrize("raw", [
    '{"action":"read_evidence","target":null,"fact_ids":["config:action_scale"]}',
    '{"action":"inspect_frames","target":null,"fact_ids":["config:action_scale"]}',
    '{"action":"read_evidence","target":"evidence:training-01","fact_ids":[]}',
    '{"action":"hold","target":null,"fact_ids":["unknown:fact"]}',
])
def test_invalid_contracts_stay_invalid_without_repair(tmp_path, raw):
    value = context(tmp_path, {"evidence:training-01": "training"})
    assert A.assess(raw, value) == A.A1.assess(raw, value)
    assert A.assess(raw, value)["status"] == "invalid"


def test_empty_action_targets_and_valid_hold_use_current_visible_facts(tmp_path):
    value = context(tmp_path)
    assert value["model_view"]["available_actions"]["read_evidence"] == []
    for action, target, expected in [("read_evidence", "evidence:training-01", "invalid"),
                                      ("hold", None, "valid"), ("run_probe", "P_reward", "valid")]:
        raw = json.dumps({"action": action, "target": target, "fact_ids": ["config:action_scale"]})
        assert A.assess(raw, value)["status"] == expected


def test_v2_messages_preserve_user_facts_and_exactly_one_new_system(tmp_path):
    value = context(tmp_path)
    old, new = A.A1.messages(value), A.messages(value)
    assert new[1:] == old[1:]
    assert new[0] == {"role": "system", "content": A.SYSTEM}
    assert new[0] != old[0] and len(new) == 2
    assert A.assess is A.A1.assess and A.prepare is A.A1.prepare


def test_registered_schema_has_only_current_enums_and_host_keeps_cross_field_validation(tmp_path):
    value = context(tmp_path)
    response = A.response_format(value)
    schema = response["json_schema"]["schema"]
    assert response["type"] == "json_schema"
    assert set(schema["required"]) == {"action", "target", "fact_ids"} and schema["additionalProperties"] is False
    assert "read_evidence" not in schema["properties"]["action"]["enum"]
    assert schema["properties"]["fact_ids"]["minItems"] == 1
    assert schema["properties"]["fact_ids"]["items"]["enum"] == sorted(value["model_view"]["evidence_facts"])
    assert None in schema["properties"]["target"]["enum"]
    assert not any(word in json.dumps(response) for word in ['"strict"','"oneOf"','"anyOf"','"uniqueItems"'])
    value = context(tmp_path, {"evidence:training-01": "training"})
    schema = A.response_format(value)["json_schema"]["schema"]
    assert "read_evidence" in schema["properties"]["action"]["enum"]
    assert "evidence:training-01" in schema["properties"]["target"]["enum"]
    # 단순 enum union은 action/target 종속이나 중복 인용의 호스트 검사를 대신하지 않는다.
    for raw in ['{"action":"read_evidence","target":null,"fact_ids":["config:action_scale"]}',
                '{"action":"hold","target":null,"fact_ids":["config:action_scale","config:action_scale"]}']:
        assert A.assess(raw, value)["status"] == "invalid"
