from __future__ import annotations

# pyright: reportMissingImports=false
import base64
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import agentic_investigation_call_v2 as CALL
import agentic_investigation_contract_v2 as A2
import agentic_paired_evidence_v1 as P
import test_selection_comparison_controller_v1 as F

PLAY_FIELDS = (
    "timestamp_s",
    "command:linear_x_mps", "command:linear_y_mps", "command:yaw_rate_rps",
    "motion:linear_x_mps", "motion:linear_y_mps", "motion:yaw_rate_rps",
    "pose:base_height_m", "pose:up_axis_z",
)


def prepared_fixture(tmp_path: Path) -> Path:
    extra_facts = {}
    for number in range(1, 7):
        frame_id = f"f{number}"
        extra_facts[f"vision:frame:{frame_id}"] = {
            "evidence_type": "raw_frame", "timestamp_s": float(number),
            "source_group": "visual_source_1",
        }
        for offset, field in enumerate(PLAY_FIELDS):
            extra_facts[f"vision:play:{frame_id}:{field}"] = number + offset / 100
    context = A2.prepare(
        F.snapshot(tmp_path, "case", "model_adaptive"),
        goal="case와 reference의 공개 근거를 비교해 다음 등록 검사 하나를 선택한다.",
        extra_facts=extra_facts,
        decisions_left=2,
    )
    messages = A2.messages(context)
    blocks = [{"type": "text", "text": messages[1]["content"]}]
    for number in range(1, 7):
        frame_id = f"f{number}"
        blocks.extend([
            {"type": "text", "text": json.dumps({
                "frame_id": frame_id, "timestamp_s": float(number)}, separators=(",", ":"))},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(
                    b"\xff\xd8cpu-paired-fixture\xff\xd9").decode(),
                "detail": "low",
            }},
        ])
    messages[1]["content"] = blocks
    checked = A2.checked(context)
    assert CALL._messages(checked, messages) == messages
    params = CALL.params_for(checked)
    assert params == CALL.params_for_messages(messages)
    prepared = {
        "schema": "fresh_frame_decision_prepared_v1",
        "status": "ready_for_registered_calls",
        "approval_granted": False,
        "context": checked,
        "messages": messages,
        "params": params,
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared), encoding="utf-8")
    return path


def test_prepare_keeps_common_contract_and_removes_only_frame_exposure(tmp_path):
    result = P.prepare(prepared_fixture(tmp_path), tmp_path / "pair")
    assert result["status"] == "prepared_cpu_fixture_not_model_result"
    assert result["differences"]["snapshot_same"] is True
    assert result["differences"]["goal_same"] is True
    assert result["differences"]["eligible_probes_same"] is True
    assert result["differences"]["base_model_params_same"] is True
    assert len(result["differences"]["removed_raw_frame_fact_ids"]) == 6
    assert result["arms"]["logs_only"]["shape"]["image_blocks"] == 0
    assert result["arms"]["logs_plus_frames"]["shape"]["image_blocks"] == 6
    assert "data:image" not in (tmp_path / "pair/pair.json").read_text(encoding="utf-8")


def test_compare_is_inconclusive_without_both_actual_receipts(tmp_path):
    P.prepare(prepared_fixture(tmp_path), tmp_path / "pair")
    result = P.compare(tmp_path / "pair/pair.json", tmp_path / "comparison")
    assert result["status"] == result["contribution"] == "inconclusive"
    assert result["model_selection_change_is_success"] is False
    assert result["frame_citation_is_success"] is False


def test_compare_requires_complete_outcome_contract(tmp_path):
    P.prepare(prepared_fixture(tmp_path), tmp_path / "pair")
    pair = P.read(tmp_path / "pair/pair.json")
    refs = {}
    outcomes = {
        "model_valid": True, "hold": False, "selected_probe": "P_reward", "frame_fact_ids": [],
        "confirmed_cause": False, "false_confirmation": False, "normal_intervention": False,
        "probe_count": 1, "cost": {"api_calls": 1, "gpu_s": 60},
    }
    for arm in P.ARMS:
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({"schema": P.SCHEMA + "_arm_receipt", "arm": arm,
                                    "pair_binding_sha256": pair["binding_sha256"],
                                    "provider_input_sha256": pair["arms"][arm]["provider_input"]["sha256"],
                                    "status": "complete", "outcomes": outcomes}), encoding="utf-8")
        refs[arm] = {"path": str(path.resolve()), "sha256": P.file_sha(path)}
    result = P.compare(tmp_path / "pair/pair.json", tmp_path / "done",
                       refs["logs_only"], refs["logs_plus_frames"])
    assert result["status"] == result["contribution"] == "inconclusive"
    assert result["actual_outcomes_verified"] is False
    assert "self_reported_arm_outcomes" in result


def test_compare_rejects_receipt_from_another_pair(tmp_path):
    P.prepare(prepared_fixture(tmp_path), tmp_path / "pair")
    pair = P.read(tmp_path / "pair/pair.json")
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema": P.SCHEMA + "_arm_receipt", "arm": "logs_only",
                                "pair_binding_sha256": "0" * 64,
                                "provider_input_sha256": pair["arms"]["logs_only"]["provider_input"]["sha256"],
                                "status": "complete", "outcomes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="pair/provider binding"):
        P.compare(tmp_path / "pair/pair.json", tmp_path / "rejected",
                  {"path": str(path.resolve()), "sha256": P.file_sha(path)}, None)


def test_prepare_rejects_message_or_common_numeric_fact_tampering(tmp_path, monkeypatch):
    prepared_path = prepared_fixture(tmp_path)
    malformed = json.loads(prepared_path.read_text(encoding="utf-8"))
    malformed["messages"][1]["content"][2]["image_url"]["detail"] = "high"
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError):
        P.prepare(malformed_path, tmp_path / "malformed-pair")

    original_prepare = P.A2.prepare
    calls = 0

    def changed_common_fact(*args, **kwargs):
        nonlocal calls
        calls += 1
        current = deepcopy(kwargs)
        if calls == 2:
            current["extra_facts"]["vision:play:f1:timestamp_s"] += 1
        return original_prepare(*args, **current)

    monkeypatch.setattr(P.A2, "prepare", changed_common_fact)
    with pytest.raises(ValueError, match="paired contexts differ"):
        P.prepare(prepared_path, tmp_path / "changed-pair")
