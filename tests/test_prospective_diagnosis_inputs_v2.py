from __future__ import annotations

import pytest

from evals import prospective_diagnosis_inputs_v2 as V2


def test_public_empty_correction_field_is_not_a_hidden_answer(monkeypatch) -> None:
    monkeypatch.setattr(V2.V1.BASE.SELECTION, "validate_packet", lambda x: x)
    packet = {"epoch": 1, "applied_corrections": [], "prior_receipt": None,
              "facts": {"runtime:case:optimizer_steps_min": 4}}
    with pytest.raises(ValueError, match="정답 필드"):
        V2.V1._model_safe(packet)
    assert V2.checked_model_packet(packet) == packet


@pytest.mark.parametrize("injection", [
    {"original_faults": []}, {"remaining_faults": ["F1"]}, {"category": "R1_ONLY"},
    {"facts": {"message": "F1"}}, {"artifact": "C:\\Users\\person\\checkpoint.pt"}])
def test_hidden_truth_and_physical_paths_remain_rejected(monkeypatch, injection) -> None:
    monkeypatch.setattr(V2.V1.BASE.SELECTION, "validate_packet", lambda x: x)
    packet = {"epoch": 1, "applied_corrections": [], "prior_receipt": None, **injection}
    with pytest.raises(ValueError):
        V2.checked_model_packet(packet)


def test_correction_and_prior_receipt_cannot_be_smuggled_through_exemption(monkeypatch) -> None:
    monkeypatch.setattr(V2.V1.BASE.SELECTION, "validate_packet", lambda x: x)
    with pytest.raises(ValueError, match="첫 epoch"):
        V2.checked_model_packet({"epoch": 1, "applied_corrections": ["restore"],
                                 "prior_receipt": None})
