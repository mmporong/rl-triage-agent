"""영상 검사 선택의 입력 대응과 사전등록 기술 통계를 계산하는 CPU 계약."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

VIEWS = ("text_only", "video_only", "text_and_video")
SEED_GROUPS = ("source-01", "source-02")
ROLES = ("normal", "registered-f1")
WINDOWS = ("window-01", "window-02")
_ORDER_INDEXES = ((0, 1, 2), (2, 1, 0), (1, 0, 2), (2, 0, 1),
                  (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0))
_TRIO_BINDINGS = (
    "source_seed", "source_lineage_id", "parent_lineage_id", "checkpoint_sha256",
    "video_sha256", "trace_sha256", "contract_sha256", "snapshot_sha256", "camera_pose_audit_sha256", "split",
)
_HASH_FIELDS = ("checkpoint_sha256", "video_sha256", "trace_sha256", "contract_sha256",
                "snapshot_sha256", "capsule_sha256", "messages_sha256", "camera_pose_audit_sha256")
CAPSULE_METHODS = frozenset({"evals/vision_selection_contract_v1.py", "evals/vision_selection_loop_v1.py",
                             "evals/vision_selection_call_v1.py"})
INPUT_FIELDS = frozenset({"slot_id", "seed_group", "source_role", "window", "view_kind", "call_id",
                          "source_seed", "source_lineage_id", "parent_lineage_id", "split",
                          "method_protocol_sha256", "source_protocol_sha256", "capsule_method_raw_sha256",
                          *_HASH_FIELDS})
_CHOICE_STATES = {"valid", "semantic_invalid", "infrastructure_failed", "unknown_pending"}


def design() -> list[dict[str, Any]]:
    """각 seed×role의 두 구간에서 text/combined 선후를 한 번씩 바꾼다."""
    slots = []
    for group in SEED_GROUPS:
        for role in ROLES:
            for window in WINDOWS:
                index = len(slots)
                slots.append({"slot_id": f"input-{index + 1:02d}", "seed_group": group,
                              "source_role": role, "window": window,
                              "view_order": [VIEWS[i] for i in _ORDER_INDEXES[index]]})
    return slots


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_inputs(rows: list[dict], *, method_protocol_sha256: str, source_protocol_sha256: str) -> list[dict]:
    """24개 입력의 순서·trio·seed·계보 결합을 검사한다. 원본 파일 검사는 실행기가 맡는다."""
    _require(_hash(method_protocol_sha256) and _hash(source_protocol_sha256), "사전등록 SHA 형식이 다르다")
    _require(isinstance(rows, list) and len(rows) == 24, "24개 동결 입력이 필요하다")
    source_seeds: dict[str, int] = {}
    lineage_splits: dict[str, str] = {}
    checkpoint_lineages: dict[str, str] = {}
    role_bindings: dict[tuple[str, str], tuple] = {}
    outputs = []
    offset = 0
    for slot in design():
        trio = rows[offset:offset + 3]
        offset += 3
        for trio_index, (row, view) in enumerate(zip(trio, slot["view_order"], strict=True)):
            _require(isinstance(row, dict), "입력 row가 객체가 아니다")
            _require(INPUT_FIELDS <= set(row), "입력 manifest 필수 필드가 없다")
            _require(all(row.get(key) == slot[key] for key in ("slot_id", "seed_group", "source_role", "window"))
                     and row.get("view_kind") == view, "고정 slot·view 순서가 다르다")
            _require(row.get("method_protocol_sha256") == method_protocol_sha256
                     and row.get("source_protocol_sha256") == source_protocol_sha256,
                     "입력의 방법·학습 사전등록 결합이 다르다")
            _require(all(_hash(row.get(key)) for key in _HASH_FIELDS), "입력 artifact SHA가 없다")
            methods = row["capsule_method_raw_sha256"]
            _require(isinstance(methods, dict) and set(methods) == CAPSULE_METHODS
                     and all(_hash(value) for value in methods.values()), "capsule method raw SHA 형식이 다르다")
            _require(methods == rows[0].get("capsule_method_raw_sha256"), "24입력의 capsule method raw SHA가 다르다")
            _require(row["call_id"] == f"choice-{len(outputs) + trio_index + 1:02d}", "고정 call ID 순서가 다르다")
            _require(type(row.get("source_seed")) is int and row["source_seed"] >= 0, "source seed 형식이 다르다")
            _require(isinstance(row.get("source_lineage_id"), str) and bool(row["source_lineage_id"]), "source 계보가 없다")
            parent = row.get("parent_lineage_id")
            _require(parent is None or (isinstance(parent, str) and bool(parent)
                                       and parent != row["source_lineage_id"]), "부모 source 계보가 잘못됐다")
            _require("parent_lineage_id" in row and row.get("split") == "comparison", "비교 입력 split 또는 부모 계보가 없다")
            source_seed = source_seeds.setdefault(slot["seed_group"], row["source_seed"])
            _require(source_seed == row["source_seed"], "같은 group의 NONE/F1 source seed가 다르다")
            split = lineage_splits.setdefault(row["source_lineage_id"], row["split"])
            _require(split == row["split"], "같은 계보의 split이 다르다")
            lineage = checkpoint_lineages.setdefault(row["checkpoint_sha256"], row["source_lineage_id"])
            _require(lineage == row["source_lineage_id"], "같은 체크포인트에 다른 계보를 붙였다")
        _require(all(tuple(row.get(key) for key in _TRIO_BINDINGS)
                     == tuple(trio[0].get(key) for key in _TRIO_BINDINGS) for row in trio),
                 "동일 slot의 세 view가 다른 base/checkpoint/영상/수치를 쓴다")
        _require(len({row["capsule_sha256"] for row in trio}) == 3
                 and len({row["messages_sha256"] for row in trio}) == 3, "세 조건의 capsule/messages가 중복이다")
        role_key = (slot["seed_group"], slot["source_role"])
        role_identity = tuple(trio[0][key] for key in ("source_seed", "source_lineage_id", "checkpoint_sha256", "video_sha256"))
        _require(role_bindings.setdefault(role_key, role_identity) == role_identity,
                 "같은 정책의 두 구간이 다른 source/checkpoint를 쓴다")
        outputs.extend(deepcopy(trio))
    _require(len(set(source_seeds.values())) == 2, "두 source seed group이 같은 seed다")
    _require(len({value[1] for value in role_bindings.values()}) == 4
             and len({value[2] for value in role_bindings.values()}) == 4,
             "정상·F1 두 쌍의 source 계보·체크포인트가 중복이다")
    return outputs


def summarize(inputs: list[dict], rows: list[dict], *, method_protocol_sha256: str,
              source_protocol_sha256: str, source_labels: dict, probe_support: dict) -> dict:
    """선택 변화와 frame 인용·실제 검사 지지를 분리한다. 입력 파일의 진실성 검증과 별개다."""
    bound_inputs = validate_inputs(inputs, method_protocol_sha256=method_protocol_sha256,
                                   source_protocol_sha256=source_protocol_sha256)
    input_index = {(row["slot_id"], row["view_kind"]): row for row in bound_inputs}
    checkpoints = {(row["seed_group"], row["source_role"]): row["checkpoint_sha256"] for row in bound_inputs}
    _require(isinstance(rows, list), "선택 결과는 목록이어야 한다")
    expected = {(slot["slot_id"], view) for slot in design() for view in VIEWS}
    indexed = {}
    terminal_hashes = set()
    for row in rows:
        key = (row.get("slot_id"), row.get("view_kind"))
        _require(key in expected and key not in indexed, "등록 밖 또는 중복 선택 결과다")
        _require(all(row.get(name) == input_index[key][name] for name in (
            "call_id", "capsule_sha256", "messages_sha256")), "선택 terminal과 등록 capsule/messages/call ID가 다르다")
        _require(row.get("model_status") in _CHOICE_STATES, "선택 결과 상태가 다르다")
        if row["model_status"] == "unknown_pending":
            _require(row.get("terminal_sha256") is None, "미완료 요청에 terminal SHA를 붙였다")
        else:
            _require(_hash(row.get("terminal_sha256")) and row["terminal_sha256"] not in terminal_hashes,
                     "선택 terminal artifact SHA가 없거나 중복이다")
            terminal_hashes.add(row["terminal_sha256"])
        if row["model_status"] == "valid":
            _require(row.get("action") in {"continue", "hold"}, "유효 선택 action이 다르다")
            _require(isinstance(row.get("fact_ids"), list) and bool(row["fact_ids"])
                     and all(isinstance(fact, str) for fact in row["fact_ids"]), "유효 선택의 근거 ID가 없다")
            _require((row["action"] == "hold" and row.get("probe") is None)
                     or (row["action"] == "continue" and isinstance(row.get("probe"), str)), "선택 probe가 다르다")
        indexed[key] = row
    counts = {group: {"H_raw": 0, "H_grounded": 0, "R": 0} for group in SEED_GROUPS}
    comparisons = []
    normal_choices = []

    def hit(row: dict) -> bool:
        return row["model_status"] == "valid" and row.get("action") == "continue" and row.get("probe") == "P_reward"

    for slot in design():
        selected = {view: indexed.get((slot["slot_id"], view)) for view in VIEWS}
        if slot["source_role"] == "normal":
            normal_choices.append({"slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
                                   "choices": selected})
            continue
        text, combined = selected["text_only"], selected["text_and_video"]
        measurable = text is not None and combined is not None and all(
            row["model_status"] in {"valid", "semantic_invalid"} for row in (text, combined)
        )
        h_raw = bool(measurable and text is not None and combined is not None and not hit(text) and hit(combined))
        h_grounded = bool(h_raw and combined is not None and any(
            fact in {f"vision:frame:f{i:02d}" for i in range(1, 7)} for fact in combined.get("fact_ids", [])
        ))
        r = bool(measurable and text is not None and combined is not None and hit(text) and not hit(combined))
        for key, value in (("H_raw", h_raw), ("H_grounded", h_grounded), ("R", r)):
            counts[slot["seed_group"]][key] += int(value)
        comparisons.append({"slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
                            "measurable": measurable, "H_raw": h_raw, "H_grounded": h_grounded, "R": r})
    terminal = set(indexed) == expected and all(
        row["model_status"] in {"valid", "semantic_invalid"} for row in indexed.values()
    )
    label_hashes = set()
    for group, labels in source_labels.items():
        _require(group in SEED_GROUPS and isinstance(labels, dict), "등록 밖 source label group이다")
        for role, artifact in labels.items():
            _require(role in ROLES and isinstance(artifact, dict), "등록 밖 source label role이다")
            _require(artifact.get("checkpoint_sha256") == checkpoints[(group, role)]
                     and _hash(artifact.get("fixed_eval_artifact_sha256")), "고정 평가 artifact/checkpoint가 등록 입력과 다르다")
            _require(artifact["fixed_eval_artifact_sha256"] not in label_hashes, "같은 고정 평가 artifact를 다른 정책에 재사용했다")
            label_hashes.add(artifact["fixed_eval_artifact_sha256"])
    labels_ok = set(source_labels) == set(SEED_GROUPS) and all(
        set(source_labels[group]) == set(ROLES)
        and source_labels[group]["normal"].get("label") == "healthy"
        and source_labels[group]["registered-f1"].get("label") == "unhealthy" for group in SEED_GROUPS)
    receipt_hashes, authorizations = set(), set()
    for group, receipt in probe_support.items():
        _require(group in SEED_GROUPS and isinstance(receipt, dict), "등록 밖 probe receipt group이다")
        _require(receipt.get("checkpoint_sha256") == checkpoints[(group, "registered-f1")]
                 and _hash(receipt.get("receipt_sha256")) and _hash(receipt.get("authorization_sha256")),
                 "확인 검사 receipt/checkpoint/authorization이 등록 입력과 다르다")
        _require(receipt["receipt_sha256"] not in receipt_hashes
                 and receipt["authorization_sha256"] not in authorizations, "확인 검사 receipt 또는 승인을 두 seed에 재사용했다")
        receipt_hashes.add(receipt["receipt_sha256"])
        authorizations.add(receipt["authorization_sha256"])
    support_ok = set(probe_support) == set(SEED_GROUPS) and all(
        probe_support[group].get("probe") == "P_reward"
        and probe_support[group].get("outcome") == "abnormal"
        and probe_support[group].get("approval_consumed") is True
        and probe_support[group].get("usable") is True for group in SEED_GROUPS
    )
    complete = terminal and labels_ok and support_ok
    if not complete:
        outcome = "inconclusive"
    elif any(row["R"] for row in counts.values()):
        outcome = "regression"
    elif all(row["H_grounded"] >= 1 for row in counts.values()):
        outcome = "selection_supported"
    else:
        outcome = "observation_only"
    view_counts = {
        view: {"states": dict(Counter(row["model_status"] for key, row in indexed.items() if key[1] == view)),
               "valid_hold": sum(row["model_status"] == "valid" and row.get("action") == "hold"
                                 for key, row in indexed.items() if key[1] == view),
               "P_reward_selections": sum(hit(row) for key, row in indexed.items() if key[1] == view)} for view in VIEWS
    }
    pairwise = {}
    for left, right in (("video_only", "text_only"), ("text_and_video", "video_only")):
        pairs = [(indexed.get((slot["slot_id"], left)), indexed.get((slot["slot_id"], right))) for slot in design()]
        measured = [(a, b) for a, b in pairs if a is not None and b is not None
                    and a["model_status"] == b["model_status"] == "valid"]
        pairwise[f"{left}_vs_{right}"] = {"valid_pairs": len(measured), "different_probe_or_hold": sum(
            (a.get("action"), a.get("probe")) != (b.get("action"), b.get("probe")) for a, b in measured),
            "descriptive_only": True}
    return {"schema": "vision_selection_comparison_summary_v1",
            "execution_status": "complete" if complete else "inconclusive", "selection_outcome": outcome,
            "prerequisites": {"all_choices_terminal": terminal, "source_labels": labels_ok, "actual_probe_support": support_ok},
            "registered_fault_family_count": 1, "registered_fault_checkpoint_count": 2,
            "independent_source_seed_groups": 2, "choice_status_counts": dict(Counter(row["model_status"] for row in rows)),
            "counts_by_source_seed": counts, "fault_windows": comparisons, "normal_choices": normal_choices,
            "view_counts": view_counts, "secondary_pairwise": pairwise,
            "bindings": {"method_protocol_sha256": method_protocol_sha256,
                         "source_protocol_sha256": source_protocol_sha256,
                         "terminal_sha256": [row["terminal_sha256"] for row in rows],
                         "fixed_eval_sha256": sorted(label_hashes), "probe_receipt_sha256": sorted(receipt_hashes)},
            "frame_citation_is_causal_confirmation": False, "other_probe_is_generic_misdiagnosis": False}
