"""비교 순서·동결 입력 대응·선택 변화 판정의 CPU 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import vision_selection_comparison_v1 as C


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs() -> list[dict]:
    rows = []
    for slot in C.design():
        policy = f"{slot['seed_group']}-{slot['source_role']}"
        for view in slot["view_order"]:
            rows.append({**{key: value for key, value in slot.items() if key != "view_order"},
                         "view_kind": view, "source_seed": 1 if slot["seed_group"] == "source-01" else 2,
                         "source_lineage_id": policy, "parent_lineage_id": None, "split": "comparison",
                         **{key: _sha(policy) for key in ("checkpoint_sha256", "video_sha256")},
                         **{key: _sha(slot["slot_id"]) for key in ("trace_sha256", "contract_sha256", "snapshot_sha256", "camera_pose_audit_sha256")},
                         **{key: _sha(slot["slot_id"] + view) for key in ("capsule_sha256", "messages_sha256")},
                         "call_id": f"choice-{len(rows) + 1:02d}",
                         "capsule_method_raw_sha256": {name: _sha(name) for name in C.CAPSULE_METHODS},
                         "method_protocol_sha256": "a" * 64, "source_protocol_sha256": "b" * 64})
    return rows


def _results() -> tuple[list[dict], dict, dict]:
    inputs = {(row["slot_id"], row["view_kind"]): row for row in _inputs()}
    rows = [{"slot_id": slot["slot_id"], "view_kind": view, "model_status": "valid", "action": "continue",
             "probe": "P_episode", "fact_ids": ["vision:play:f01:timestamp_s"],
             **{key: inputs[(slot["slot_id"], view)][key] for key in ("call_id", "capsule_sha256", "messages_sha256")},
             "terminal_sha256": _sha(slot["slot_id"] + view + "terminal")}
            for slot in C.design() for view in C.VIEWS]
    labels = {group: {role: {"checkpoint_sha256": _sha(group + "-" + role),
                            "fixed_eval_artifact_sha256": _sha(group + role + "eval"),
                            "label": "healthy" if role == "normal" else "unhealthy"} for role in C.ROLES}
              for group in C.SEED_GROUPS}
    receipts = {group: {"probe": "P_reward", "outcome": "abnormal", "checkpoint_sha256": _sha(group + "-registered-f1"),
                        "receipt_sha256": _sha(group + "receipt"), "authorization_sha256": _sha(group + "auth"),
                        "approval_consumed": True, "usable": True} for group in C.SEED_GROUPS}
    return rows, labels, receipts


def _report(rows: list[dict], labels: dict, receipts: dict) -> dict:
    return C.summarize(_inputs(), rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64,
                       source_labels=labels, probe_support=receipts)


def test_order_reverses_primary_contrast_within_every_seed_role_pair() -> None:
    slots = C.design()
    assert len(slots) == 8
    for group in C.SEED_GROUPS:
        for role in C.ROLES:
            pair = [row for row in slots if row["seed_group"] == group and row["source_role"] == role]
            assert sorted(row["view_order"].index("text_only") < row["view_order"].index("text_and_video")
                          for row in pair) == [False, True]
    for view in C.VIEWS:
        assert sorted(Counter(row["view_order"].index(view) for row in slots).values()) == [2, 3, 3]


def test_exact_input_trios_keep_source_and_window_binding() -> None:
    rows = _inputs()
    assert C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64) == rows
    for field in ("checkpoint_sha256", "contract_sha256", "snapshot_sha256", "video_sha256", "trace_sha256"):
        changed = deepcopy(rows)
        changed[0][field] = "e" * 64
        with pytest.raises(ValueError, match="slot|source|정책"):
            C.validate_inputs(changed, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)
    changed = deepcopy(rows)
    changed[0]["source_protocol_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="사전등록"):
        C.validate_inputs(changed, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)
    with pytest.raises(ValueError, match="24개"):
        C.validate_inputs(rows[:-1], method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)


def test_same_seed_group_and_distinct_group_cannot_be_relabelled() -> None:
    rows = _inputs()
    for row in rows:
        if row["source_role"] == "registered-f1":
            row["source_seed"] += 100
    with pytest.raises(ValueError, match="NONE/F1"):
        C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)
    rows = _inputs()
    for row in rows:
        row["source_seed"] = 1
    with pytest.raises(ValueError, match="같은 seed"):
        C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)


def test_raw_flip_requires_frame_citation_in_each_source_group_for_support() -> None:
    rows, labels, receipts = _results()
    for row in rows:
        if row["slot_id"] in {"input-03", "input-07"} and row["view_kind"] == "text_and_video":
            row["probe"] = "P_reward"
    raw = _report(rows, labels, receipts)
    assert raw["selection_outcome"] == "observation_only"
    assert all(group == {"H_raw": 1, "H_grounded": 0, "R": 0} for group in raw["counts_by_source_seed"].values())
    for row in rows:
        if row["probe"] == "P_reward":
            row["fact_ids"] = ["vision:frame:f02"]
    supported = _report(rows, labels, receipts)
    assert supported["selection_outcome"] == "selection_supported"
    assert supported["registered_fault_family_count"] == 1
    assert supported["registered_fault_checkpoint_count"] == supported["independent_source_seed_groups"] == 2
    assert supported["frame_citation_is_causal_confirmation"] is False


def test_incomplete_or_infrastructure_cannot_be_reported_as_regression() -> None:
    rows, labels, receipts = _results()
    target = next(row for row in rows if row["slot_id"] == "input-03" and row["view_kind"] == "text_only")
    target["probe"] = "P_reward"
    assert _report(rows, labels, receipts)["selection_outcome"] == "regression"
    assert _report(rows[:-1], labels, receipts)["selection_outcome"] == "inconclusive"
    combined = next(row for row in rows if row["slot_id"] == "input-03" and row["view_kind"] == "text_and_video")
    combined["model_status"] = "infrastructure_failed"
    report = _report(rows, labels, receipts)
    assert report["selection_outcome"] == "inconclusive"
    assert report["counts_by_source_seed"]["source-01"]["R"] == 0
    combined["model_status"] = "semantic_invalid"
    assert _report(rows, labels, receipts)["selection_outcome"] == "regression"
    assert _report(rows, labels, {})["selection_outcome"] == "inconclusive"
    combined.update(model_status="unknown_pending", terminal_sha256=None)
    assert _report(rows, labels, receipts)["selection_outcome"] == "inconclusive"


def test_normal_inputs_have_no_unique_positive_target_and_other_probes_are_not_errors() -> None:
    rows, labels, receipts = _results()
    report = _report(rows, labels, receipts)
    assert len(report["normal_choices"]) == 4
    assert report["other_probe_is_generic_misdiagnosis"] is False
    assert report["selection_outcome"] == "observation_only"
    labels["source-01"]["registered-f1"]["label"] = "healthy"
    assert _report(rows, labels, receipts)["selection_outcome"] == "inconclusive"


def test_wrong_capsule_evaluation_checkpoint_or_shared_receipt_is_rejected() -> None:
    rows, labels, receipts = _results()
    changed = deepcopy(rows)
    changed[0]["capsule_sha256"] = changed[1]["capsule_sha256"]
    with pytest.raises(ValueError, match="terminal.*capsule"):
        _report(changed, labels, receipts)
    changed = deepcopy(labels)
    changed["source-01"]["registered-f1"]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="고정 평가"):
        _report(rows, changed, receipts)
    changed = deepcopy(receipts)
    changed["source-01"]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="확인 검사"):
        _report(rows, labels, changed)
    changed = deepcopy(receipts)
    changed["source-02"]["receipt_sha256"] = changed["source-01"]["receipt_sha256"]
    with pytest.raises(ValueError, match="재사용"):
        _report(rows, labels, changed)
    changed_rows = deepcopy(rows)
    changed_rows[0]["terminal_sha256"] = changed_rows[1]["terminal_sha256"]
    with pytest.raises(ValueError, match="중복"):
        _report(changed_rows, labels, receipts)


def test_fake_frame_citation_does_not_count_as_grounded_flip() -> None:
    rows, labels, receipts = _results()
    for row in rows:
        if row["slot_id"] in {"input-03", "input-07"} and row["view_kind"] == "text_and_video":
            row.update(probe="P_reward", fact_ids=["vision:frame:f99"])
    report = _report(rows, labels, receipts)
    assert report["selection_outcome"] == "observation_only"
    assert all(row["H_raw"] == 1 and row["H_grounded"] == 0 for row in report["counts_by_source_seed"].values())


def test_camera_audit_and_capsule_raw_method_map_are_required() -> None:
    for field in ("camera_pose_audit_sha256", "capsule_method_raw_sha256"):
        rows = _inputs()
        del rows[0][field]
        with pytest.raises(ValueError, match="필수 필드"):
            C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)
    rows = _inputs()
    rows[1]["capsule_method_raw_sha256"][next(iter(C.CAPSULE_METHODS))] = "e" * 64
    with pytest.raises(ValueError, match="24입력"):
        C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)
    rows = _inputs()
    rows[0]["parent_lineage_id"] = ""
    with pytest.raises(ValueError, match="부모 source"):
        C.validate_inputs(rows, method_protocol_sha256="a" * 64, source_protocol_sha256="b" * 64)


def test_registered_method_matches_input_contract_and_frozen_runtime_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    method = json.loads((root / "bench/protocols/vision_selection_comparison_v1.json").read_text(encoding="utf-8"))
    assert set(method["execution_gate"]["input_manifest_fields"]) == C.INPUT_FIELDS
    assert method["inputs"]["slots"] == C.design()
    assert method["execution_gate"]["fixed_call_ids"] == [f"choice-{i:02d}" for i in range(1, 25)]
    assert sum(method["global_allocation"][key] for key in ("development", "comparison", "e2e", "heldout")) == 120
    for name, expected in method["source_sha256_lf"].items():
        assert hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, name
    for name in ("bench/protocols/selection_comparison_view_v1.json", "bench/protocols/fixed_eval_v1.json"):
        assert hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == method["evidence_sha256_lf"][name]
