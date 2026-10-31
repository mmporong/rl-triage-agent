# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import prospective_diagnosis_training_v1 as C


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _protocol() -> dict:
    return C.prepare_protocol(
        [2401, 2402], ["NONE", "R1_ONLY"],
        {f"runtime/{index}.py": "a" * 64 for index in range(9)})


def test_protocol_requires_paired_none_and_r1_only_new_seeds() -> None:
    value = _protocol()
    assert value["seeds"] == [2401, 2402]
    assert value["training"] == {
        "task": C.TRAIN.RESIDUAL.TASK, "num_envs": 4096, "updates": 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "fresh_initialization": True,
        "checkpoint_loaded": False}
    assert value["category_contract"]["R1_ONLY"] == {
        "arm": "R1", "original_faults": ["R1"]}
    with pytest.raises(ValueError, match="NONE/R1_ONLY"):
        C.prepare_protocol([2401, 2402], ["NONE"], value["runtime_source_sha256_lf"])


def test_r1_only_manifest_is_genuine_r1_without_checkpoint_or_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol()
    _write(protocol_path, protocol)
    monkeypatch.setattr(C, "_relative", lambda _path: "bench/protocols/prospective.json")
    fixed = C.build_manifest(
        protocol_path, protocol, "a" * 40, stage=tmp_path / "stage", seed=2401,
        category="R1_ONLY", run_name="pd_r1_s2401")
    assert fixed["arm"] == "R1"
    assert fixed["original_faults"] == ["R1"] and fixed["applied_corrections"] == []
    assert fixed["source_checkpoint_sha256"] is None
    assert fixed["max_iterations"] == 300 and fixed["num_envs"] == 4096
    assert C.TRAIN.validate_manifest(fixed) == fixed
    cpu = C.TRAIN.RESIDUAL.cpu_check()
    assert cpu["status"] == "pass" and cpu["none_noop"] is True
    assert cpu["other_fault_preserved"] is True


def test_training_mode_rejects_checkpoint_and_unregistered_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol()
    _write(protocol_path, protocol)
    monkeypatch.setattr(C, "_relative", lambda _path: "bench/protocols/prospective.json")
    with pytest.raises(ValueError, match="checkpoint weight"):
        C.build_manifest(protocol_path, protocol, "a" * 40, stage=tmp_path / "stage",
                         seed=2401, category="R1_ONLY", run_name="bad",
                         source_checkpoint_sha256="b" * 64)
    with pytest.raises(ValueError, match="seed"):
        C.build_manifest(protocol_path, protocol, "a" * 40, stage=tmp_path / "stage",
                         seed=9999, category="R1_ONLY", run_name="bad")


def test_run_case_consumes_guarded_training_then_fixed26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fixed = {"arm": "R1", "seed": 2401, "original_faults": ["R1"],
             "applied_corrections": []}

    def training(stage: Path, manifest: dict, **_kwargs):
        calls.append("train")
        evidence = Path(stage) / "training/training_evidence.json"
        _write(evidence, {"checkpoint": {"sha256": "c" * 64}})
        return {"evidence": {"path": str(evidence), "sha256": C._sha(evidence)},
                "checkpoint": {"path": str(Path(stage) / "model_299.pt"),
                               "sha256": "c" * 64}}

    def fixed26(stage: Path, _evidence: dict, *, name: str, **_kwargs):
        calls.append("fixed")
        report = {"name": name, "conditions": 26}
        _write(Path(stage) / "corrected.json", report)
        return report, {"exit_code": 0}

    monkeypatch.setattr(C.RUNTIME, "run_training", training)
    monkeypatch.setattr(C.RUNTIME, "run_fixed", fixed26)
    result = C.run_case(tmp_path / "case", fixed, category="R1_ONLY",
                        lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                        runner=lambda *_args: None, wait=lambda *_args: None)
    assert calls == ["train", "fixed"]
    assert result["category"] == "R1_ONLY" and result["runtime_arm"] == "R1"
    assert result["report"]["conditions"] == 26
    assert (tmp_path / "case/training-result.json").is_file()


def test_run_case_rejects_category_manifest_mismatch_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C.RUNTIME, "run_training",
                        lambda *_args, **_kwargs: pytest.fail("runner must not be called"))
    with pytest.raises(ValueError, match="category와 manifest"):
        C.run_case(tmp_path / "case", {"arm": "NONE", "seed": 2401,
            "original_faults": [], "applied_corrections": []}, category="R1_ONLY",
            lab=tmp_path / "lab", isaac_python=Path(sys.executable),
            runner=lambda *_args: None, wait=lambda *_args: None)
