# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import prospective_diagnosis_inputs_v1 as C


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, category: str = "R1_ONLY"):
    checkpoint = "c" * 64
    base_path = tmp_path / "base/collected.json"
    base_meta = tmp_path / "base/metadata.json"
    case_fact, ref_fact = tmp_path / "case-fact.json", tmp_path / "ref-fact.json"
    _write(base_path, {"base": True})
    _write(base_meta, {"base_meta": True})
    _write(case_fact, {"facts": "case"})
    _write(ref_fact, {"facts": "reference"})
    contract = C.TRAIN.CATEGORIES[category]
    base = {"source_binding": {"checkpoint_sha256": checkpoint,
                                "telemetry_sha256": "d" * 64,
                                "binding_digest": "e" * 64}}
    metadata = {"private_training": {"arm": contract["arm"],
                                      "original_faults": contract["original_faults"],
                                      "applied_corrections": []}}
    monkeypatch.setattr(C.BASE, "checked", lambda _path: (base, metadata))
    monkeypatch.setattr(C.FACT, "checked", lambda _path, **_kwargs: ({}, {}))
    training = {
        "collected": {"path": str(base_path), "sha256": C._sha(base_path)},
        "runtime_facts": {
            "case": {"path": str(case_fact), "sha256": C._sha(case_fact)},
            "reference": {"path": str(ref_fact), "sha256": C._sha(ref_fact)}},
        "checkpoint": {"path": str(tmp_path / "model.pt"), "sha256": checkpoint},
    }
    training_path = tmp_path / "training-result.json"
    _write(training_path, training)
    return training_path, base, metadata


def test_collect_keeps_fault_category_private_and_rechecks_runtime_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, _base, _metadata = _fixture(tmp_path, monkeypatch)
    public = C.collect(training, category="R1_ONLY", out=tmp_path / "out")
    assert public["scope"]["fault_label_exposed"] is False
    assert "R1" not in json.dumps(public)
    checked, private = C.checked(tmp_path / "out/collected.json")
    assert checked == public and private["category"] == "R1_ONLY"


def test_collect_rejects_false_category_and_cross_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, base, metadata = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="호출 category"):
        C.collect(training, category="NONE", out=tmp_path / "wrong-category")
    monkeypatch.setattr(C.BASE, "checked", lambda _path: (
        {"source_binding": {**base["source_binding"], "checkpoint_sha256": "f" * 64}}, metadata))
    with pytest.raises(ValueError, match="checkpoint"):
        C.collect(training, category="R1_ONLY", out=tmp_path / "wrong-checkpoint")


def test_checked_detects_training_result_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, _base, _metadata = _fixture(tmp_path, monkeypatch)
    C.collect(training, category="R1_ONLY", out=tmp_path / "out")
    _write(tmp_path / "extra.json", {})
    Path(training).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="training result 원문"):
        C.checked(tmp_path / "out/collected.json")


def test_packet_has_no_fault_label_or_physical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_training, case_base, case_metadata = _fixture(tmp_path / "case", monkeypatch)
    C.collect(case_training, category="R1_ONLY", out=tmp_path / "case-input")
    reference_training, reference_base, reference_metadata = _fixture(
        tmp_path / "reference", monkeypatch, category="NONE")
    C.collect(reference_training, category="NONE", out=tmp_path / "reference-input")
    monkeypatch.setattr(C.BASE, "checked", lambda path: (
        (case_base, case_metadata) if "case" in Path(path).parts
        else (reference_base, reference_metadata)))
    monkeypatch.setattr(C.BASE, "make_packet", lambda *_args, **_kwargs: {
        "packet": {"schema": "selection_packet_v1", "facts": {"metric:case:x": 1.0},
                   "remaining_candidates": ["reward", "optimizer_update_count"]},
        "private_metadata": {"paths": "private"}})
    result = C.make_packet(tmp_path / "case-input/collected.json",
                           tmp_path / "reference-input/collected.json")
    encoded = json.dumps(result["packet"])
    assert "R1" not in encoded and str(tmp_path) not in encoded
    assert result["scope"]["fault_label_exposed_to_model"] is False


def test_packet_rejects_label_or_path_leak(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fault 정답"):
        C._model_safe({"original_faults": ["R1"]})
    with pytest.raises(ValueError, match="물리 경로"):
        C._model_safe({"artifact": "C:/private/model.pt"})
