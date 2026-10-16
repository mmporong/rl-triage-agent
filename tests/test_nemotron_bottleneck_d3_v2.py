from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import nemotron_bottleneck_d3_v2 as B  # noqa: E402


def sample_packet() -> dict:
    return {
        "schema": "change_rank_packet_v1",
        "changes": [
            {"label": "change_1", "diff": "@@ -1 +1 @@\n-old = 1\n+new = 1"},
            {"label": "change_2", "diff": "@@ -4 +4 @@\n-old = 2\n+new = 2"},
        ],
        "facts": {"telemetry:case:x": 1.0},
        "probe_observations": [{"probe": "P_reward"}],
    }


def sample_case() -> dict:
    packet = sample_packet()
    messages = B.D3.messages(packet)
    return {"id": "C1_B4", "bottleneck": "B2", "kind": "d3", "messages": messages,
            "request_params": B.deepcopy(B.D3_CALL.PARAMS),
            "binding": {"case_id": "C1", "stage": "B4", "packet_sha256_raw": "1" * 64,
                        "messages_sha256": B.canonical_sha(messages), "contract": "d3_old_line_number_v1"},
            "packet": packet, "truth": {"top1_label": "change_2"}, "source_path": Path("private")}


def response(*, ranking=None, line=4) -> str:
    ranking = ranking or ["change_2", "change_1"]
    return json.dumps({"ranking": ranking, "reasons": [{"change": ranking[0],
        "facts": ["telemetry:case:x"], "probes": ["P_reward"], "diff_lines": [line]}],
        "uncertain": False})


def test_validate_response_separates_contract_citation_and_true_top1():
    case = sample_case()
    valid = B.validate_response(case, response())
    assert valid == {"valid": True, "errors": [], "decision_key": ["change_2", "change_1"],
                     "top1": "change_2", "ranking_parsed": True, "true_cause_top1": True,
                     "valid_true_cause_top1": True,
                     "evidence_citation_errors": [], "evidence_citation_error": False}
    invalid = B.validate_response(case, response(line=999))
    assert invalid["valid"] is False and invalid["errors"] == ["reason0_diff_lines"]
    assert invalid["evidence_citation_error"] is True
    assert invalid["decision_key"] == ["change_2", "change_1"] and invalid["true_cause_top1"] is True


def test_parse_behavior_matches_existing_contract_for_fences_and_broken_json():
    case = sample_case()
    fenced = B.validate_response(case, f"```json\n{response()}\n```")
    assert fenced["valid"] is True
    broken = B.validate_response(case, "not json")
    assert broken["valid"] is False and broken["decision_key"] is None
    assert broken["true_cause_top1"] is None and "keys" in broken["errors"]
    malformed = B.validate_response(case, json.dumps({"ranking": ["change_1", "change_2"],
        "reasons": [{"change": [], "facts": [], "probes": [], "diff_lines": []}], "uncertain": False}))
    assert malformed["valid"] is False and malformed["errors"] == ["validator_TypeError"]


def test_wrong_top1_is_measured_separately_from_contract_validity():
    judged = B.validate_response(sample_case(), response(ranking=["change_1", "change_2"], line=1))
    assert judged["valid"] is True and judged["true_cause_top1"] is False
    assert judged["evidence_citation_error"] is False


def test_public_case_excludes_prompt_packet_truth_and_paths():
    public = B.public_case(sample_case())
    assert set(public) == {"id", "bottleneck", "kind", "request_params", "binding"}
    text = json.dumps(public)
    assert "messages\"" not in text and "truth\"" not in text and "source_path" not in text and "bench/private" not in text
    assert "model" not in public["request_params"] and public["request_params"] == B.D3_CALL.PARAMS


def test_case_shape_rejects_wrong_bottleneck_or_missing_private_evaluation_data():
    case = sample_case()
    with pytest.raises(ValueError, match="형식"):
        B.validate_response({**case, "bottleneck": "B1"}, response())
    with pytest.raises(ValueError, match="metadata"):
        B.validate_response({key: value for key, value in case.items() if key != "truth"}, response())


def test_build_cases_reuses_frozen_messages_and_params_when_private_packets_exist():
    if not B.packet_path(B.PACKET_IDS[0]).is_file():
        pytest.skip("비공개 D3 packet이 없는 공개 checkout")
    cases = B.build_cases()
    assert [case["id"] for case in cases] == list(B.PACKET_IDS)
    assert len(cases) == 8 and len({case["binding"]["packet_sha256_raw"] for case in cases}) == 8
    for case in cases:
        raw = case["source_path"].read_bytes()
        packet = json.loads(raw)
        assert case["messages"] == B.D3.messages(packet)
        assert case["binding"]["messages_sha256"] == B.canonical_sha(B.D3.messages(packet))
        assert case["request_params"] == B.D3_CALL.PARAMS and "model" not in case["request_params"]
        assert case["binding"]["truth_sha256"] == B.canonical_sha(
            {"case_id": case["binding"]["case_id"], "top1_label": case["truth"]["top1_label"]})
        assert len(case["binding"]["historical_old_responses_sha256"]) == 64


def test_historical_old_contract_is_revalidated_as_8_invalid_of_24_when_available():
    if not B.HISTORICAL_CALLS.is_dir():
        pytest.skip("비공개 D3 기존 응답이 없는 공개 checkout")
    out = B.historical_summary()
    assert (out["responses"], out["valid"], out["invalid"]) == (24, 16, 8)
    assert out["evidence_citation_invalid"] == 8
    assert out["ranking_parsed"] == 24 and out["true_cause_top1_all_parsed"] == 21
    assert out["valid_true_cause_top1"] <= out["true_cause_top1_all_parsed"]
    assert sum(out["error_counts"].values()) >= out["invalid"]


def test_source_files_cover_adapter_contract_and_tests():
    assert B.SOURCE_FILES == {"evals/nemotron_bottleneck_d3_v2.py", "evals/d3_agent.py",
                              "evals/d3_model_call.py", "tests/test_nemotron_bottleneck_d3_v2.py"}
