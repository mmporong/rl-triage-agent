"""첫 epoch의 공개 교정 이력 필드와 숨은 결함 정답 필드를 구분한다.

v1 수집 원문과 학습 source 결합은 보존한다. v1의 문자열 차단이 정상적인
빈 applied_corrections 필드까지 거부한 부분만 새 모델 입력 계약에서 고친다.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import prospective_diagnosis_inputs_v1 as V1

SCHEMA = "prospective_diagnosis_packet_v2"
checked = V1.checked


def checked_model_packet(packet: dict) -> dict:
    value = V1.BASE.SELECTION.validate_packet(packet)
    if (value.get("epoch") != 1 or value.get("prior_receipt") is not None
            or value.get("applied_corrections") != []):
        raise ValueError("새 source 비교 입력은 첫 epoch와 빈 공개 교정 이력만 허용한다")
    guarded = deepcopy(value)
    # 이 key는 동결 packet schema의 공개 host 상태다. 값은 위에서 빈 목록으로 고정한다.
    del guarded["applied_corrections"]
    V1._model_safe(guarded)
    return deepcopy(value)


def make_packet(case_path: Path, reference_path: Path, *, budget_left: int = 3,
                remaining_candidates: list[str] | None = None) -> dict:
    case, case_meta = checked(case_path)
    reference, reference_meta = checked(reference_path)
    base = V1.BASE.make_packet(
        Path(case_meta["base_collected"]["path"]),
        Path(reference_meta["base_collected"]["path"]),
        case_runtime_path=Path(case_meta["runtime_facts"]["case"]["path"]),
        reference_runtime_path=Path(reference_meta["runtime_facts"]["reference"]["path"]),
        budget_left=budget_left, remaining_candidates=remaining_candidates,
        applied_corrections=[])
    return {"schema": SCHEMA, "packet": checked_model_packet(base["packet"]),
            "source_binding": {"case": deepcopy(case["source_binding"]),
                               "reference": deepcopy(reference["source_binding"])},
            "scope": {"first_epoch_only": True, "approval": False, "execution_allowed": False,
                      "fault_label_exposed_to_model": False, "v1_collected_unchanged": True}}
