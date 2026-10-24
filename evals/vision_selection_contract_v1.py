"""V01 영상 근거와 기존 검사 선택 상태를 결합하는 CPU 계약."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import selection_comparison_controller_v1 as CONTROLLER
import vision_evidence_contract_v1 as VISION

SCHEMA = "vision_selection_bundle_v1"
VIEW_SCHEMA = "vision_selection_view_v1"
VIEW_KINDS = frozenset({"text_only", "video_only", "text_and_video"})
OUTPUT_FIELDS = frozenset({"action", "probe", "fact_ids"})

SYSTEM = """Choose the next registered inspection from the supplied eligible_probes.
Return one JSON object with exactly action, probe, and fact_ids. action is continue or hold.
continue requires one eligible probe. hold requires probe=null. fact_ids must be a nonempty list
of distinct IDs visible in evidence_facts. Frames and same-Play measurements are selection context,
not confirmation of a cause. Do not invent values, probe arguments, expected outcomes, approval,
diagnosis, correction, or recovery. Output JSON only."""


class ChoiceError(ValueError):
    """모델 선택 계약 위반 코드."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return CONTROLLER.L.digest(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ChoiceError("F_JSON_DUPLICATE_KEY")
        value[key] = item
    return value


def _binding(value: Any) -> dict[str, str]:
    _require(isinstance(value, dict) and set(value) == {"contract_sha256", "source_lineage_id"},
             "영상 source binding 형식이 다르다")
    contract_sha = value["contract_sha256"]
    lineage = value["source_lineage_id"]
    _require(isinstance(contract_sha, str) and len(contract_sha) == 64
             and all(char in "0123456789abcdef" for char in contract_sha),
             "영상 contract SHA 형식이 다르다")
    _require(isinstance(lineage, str) and bool(lineage), "영상 source lineage가 없다")
    return {"contract_sha256": contract_sha, "source_lineage_id": lineage}


def _operational_base(snapshot: dict) -> tuple[dict, dict]:
    checked = CONTROLLER.V.checked(snapshot)
    view = checked["view"]
    case_checkpoint = checked["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"]
    base = {
        "context_digest": _digest(checked["binding"]),
        "registry_digest": CONTROLLER.S.M.registry_digest(CONTROLLER.RC.registry()),
        "case_checkpoint_sha256": case_checkpoint,
        "snapshot_digest": _digest(checked),
        "base_view_digest": checked["view_digest"],
        "policy": checked["inputs"]["policy"],
        "epoch": view["epoch"],
        "candidate_categories": deepcopy(view["candidate_categories"]),
        "remaining_candidates": deepcopy(view["remaining_candidates"]),
        "probe_catalog": [
            {key: deepcopy(row[key]) for key in ("probe", "measures", "cost_gpu_s")}
            for row in view["probe_catalog"]
        ],
        "eligible_probes": deepcopy(view["eligible_probes"]),
        "budget_left": view["budget_left"],
        "execution_budget_s": view["execution_budget_s"],
        "status": view["status"],
    }
    return checked, base


def _video_facts(contract: dict) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    samples = {row["timestamp_s"]: row for row in contract["synchronized_samples"]}
    for frame in contract["frames"]:
        frame_id = frame["frame_id"]
        timestamp = frame["timestamp_s"]
        facts[f"vision:frame:{frame_id}"] = {
            "evidence_type": "raw_frame",
            "timestamp_s": timestamp,
            "source_group": "visual_source_1",
        }
        sample = samples[timestamp]
        for group in ("command", "motion", "pose"):
            for name, value in sample[group].items():
                if value is not None:
                    _require(type(value) in (int, float) and math.isfinite(value),
                             "같은 Play 수치가 유한수가 아니다")
                facts[f"vision:play:{frame_id}:{group}:{name}"] = value
    return facts


def _build(snapshot: dict, contract_path: Path, source_binding: dict, view_kind: str) -> dict:
    _require(view_kind in VIEW_KINDS, "등록되지 않은 selection view")
    binding = _binding(source_binding)
    checked, base = _operational_base(snapshot)
    contract_file = Path(contract_path)
    contract = VISION.verify(contract_file)
    contract_sha = VISION.sha256(contract_file)
    provenance = contract["provenance"]
    _require(binding["contract_sha256"] == contract_sha, "등록 V01 contract SHA가 다르다")
    _require(binding["source_lineage_id"] == provenance["source_lineage_id"],
             "등록 source lineage가 다르다")
    case_checkpoint = base["case_checkpoint_sha256"]
    _require(provenance["checkpoint"]["sha256"] == case_checkpoint,
             "V01 checkpoint와 선택 case checkpoint가 다르다")

    facts: dict[str, Any] = {}
    if view_kind in {"text_only", "text_and_video"}:
        facts.update(deepcopy(checked["view"]["evidence_facts"]))
    if view_kind in {"video_only", "text_and_video"}:
        facts.update(_video_facts(contract))
    selection_view = {
        "schema": VIEW_SCHEMA,
        "view_kind": view_kind,
        "evidence_facts": facts,
        "candidate_categories": deepcopy(base["candidate_categories"]),
        "remaining_candidates": deepcopy(base["remaining_candidates"]),
        "probe_catalog": deepcopy(base["probe_catalog"]),
        "eligible_probes": deepcopy(base["eligible_probes"]),
        "budget_left": base["budget_left"],
        "execution_budget_s": base["execution_budget_s"],
        "status": base["status"],
        "evidence_accounting": {
            "raw_frame_source_count": 1 if view_kind != "text_only" else 0,
            "raw_frame_count": VISION.FRAME_COUNT if view_kind != "text_only" else 0,
            "model_interpretation_source_count": 0,
            "independent_confirmation_sources_added": 0,
        },
    }
    source = {
        "contract_sha256": contract_sha,
        "source_lineage_digest": _digest(provenance["source_lineage_id"]),
        "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        "video_sha256": provenance["video"]["sha256"],
        "frame_set_digest": _digest([
            {key: row[key] for key in ("frame_id", "timestamp_s", "sha256")}
            for row in contract["frames"]
        ]),
        "measurement_sha256": None if provenance["measurement"] is None
        else provenance["measurement"]["sha256"],
    }
    return {
        "schema": SCHEMA,
        "view_kind": view_kind,
        "operational_base": base,
        "operational_base_digest": _digest(base),
        "selection_view": selection_view,
        "selection_view_digest": _digest(selection_view),
        "source": source,
    }


def prepare(snapshot: dict, contract_path: Path, source_binding: dict, view_kind: str) -> dict:
    """검증된 기존 상태와 V01 자료에서 한 selection bundle을 만든다."""
    return _build(snapshot, Path(contract_path), source_binding, view_kind)


def verify(bundle: dict, snapshot: dict, contract_path: Path, source_binding: dict) -> dict:
    """현재 상태·V01 바이트로 bundle 전체를 다시 계산한다."""
    _require(isinstance(bundle, dict) and bundle.get("view_kind") in VIEW_KINDS,
             "selection bundle 형식이 다르다")
    expected = _build(snapshot, Path(contract_path), source_binding, bundle["view_kind"])
    _require(_canonical(bundle) == _canonical(expected), "selection bundle이 현재 입력과 다르다")
    return deepcopy(expected)


def messages(bundle: dict, snapshot: dict, contract_path: Path, source_binding: dict) -> list[dict]:
    """세 view별 실제 모델 payload를 만들며 provenance는 노출하지 않는다."""
    value = verify(bundle, snapshot, contract_path, source_binding)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _canonical(value["selection_view"]).decode("utf-8")},
    ]
    if value["view_kind"] != "text_only":
        public = VISION.public_model_view(Path(contract_path))
        raw = public["messages"][1]["content"]
        for index in range(2, len(raw), 2):
            label = json.loads(raw[index]["text"])
            fact_id = f"vision:frame:{label['frame_id']}"
            _require(fact_id in value["selection_view"]["evidence_facts"], "frame fact 결합이 없다")
            content.append({"type": "text", "text": _canonical({
                "fact_id": fact_id, "timestamp_s": label["timestamp_s"],
                "evidence_type": "raw_frame", "source_group": "visual_source_1",
            }).decode("utf-8")})
            content.append(deepcopy(raw[index + 1]))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def validate_choice(bundle: dict, content: object) -> dict:
    """모델의 세 필드 선택과 현재 view의 probe·fact ID만 허용한다."""
    if not isinstance(content, str) or not content.strip():
        raise ChoiceError("F_EMPTY")
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _raw: (_ for _ in ()).throw(ChoiceError("F_JSON")),
        )
    except json.JSONDecodeError as error:
        raise ChoiceError("F_JSON") from error
    if not isinstance(value, dict) or set(value) != OUTPUT_FIELDS:
        raise ChoiceError("F_FIELDS")
    action, probe, fact_ids = value["action"], value["probe"], value["fact_ids"]
    if type(action) is not str or action not in {"continue", "hold"}:
        raise ChoiceError("F_ACTION")
    if not isinstance(fact_ids, list) or not fact_ids or any(type(item) is not str for item in fact_ids):
        raise ChoiceError("F_FACTS")
    if len(set(fact_ids)) != len(fact_ids):
        raise ChoiceError("F_FACTS")
    visible = bundle["selection_view"]["evidence_facts"]
    if any(item not in visible for item in fact_ids):
        raise ChoiceError("S_FACT_UNKNOWN")
    if action == "continue":
        if type(probe) is not str or probe not in bundle["selection_view"]["eligible_probes"]:
            raise ChoiceError("S_PROBE_INELIGIBLE")
    elif probe is not None:
        raise ChoiceError("F_HOLD_PROBE")
    return {"action": action, "probe": probe, "fact_ids": list(fact_ids)}


def project_choice(bundle: dict, decision: dict, snapshot: dict,
                   contract_path: Path, source_binding: dict) -> dict:
    """검증된 시각 선택을 기존 host request_spec용 base choice로 투영한다."""
    value = verify(bundle, snapshot, contract_path, source_binding)
    _require(decision == validate_choice(value, _canonical(decision).decode("utf-8")),
             "검증되지 않은 모델 선택")
    base = value["operational_base"]
    return {
        "schema": "selection_controller_choice_v1",
        "context_digest": base["context_digest"],
        "view_digest": base["base_view_digest"],
        "policy": base["policy"],
        "selected_by": "model",
        "action": decision["action"],
        "probe": decision["probe"],
        "fact_ids": deepcopy(decision["fact_ids"]),
        "model_status": "valid",
        "codes": [],
        "reason": "model_valid",
        "used_for_decision": True,
        "approval_granted": False,
    }
