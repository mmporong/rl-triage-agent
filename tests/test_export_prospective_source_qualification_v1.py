# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from evals import export_prospective_source_qualification_v1 as E
from evals import run_prospective_reasoning_confirmation_v1 as R


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _owned(wall: float) -> dict[str, Any]:
    return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "timed_out": False, "error_type": None, "wall_s": wall,
            "wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification"}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, qualified: bool = True):
    monkeypatch.setattr(E, "ROOT", tmp_path)
    bands_path = tmp_path / "fixed-bands.json"
    _write(bands_path, {"bands": {"survival_ratio_min": 0.9}})
    monkeypatch.setattr(R.BASE.RUN, "BANDS", bands_path)
    monkeypatch.setattr(R.BASE.RUN, "BANDS_SHA256", E._sha(bands_path))
    design_path = tmp_path / "bench/protocols/design.json"
    source_protocol = tmp_path / "bench/protocols/source.json"
    _write(source_protocol, {"source": True})
    design = {"seeds": [2430, 2431], "categories": ["NONE", "F1", "R1_ONLY", "COMPOUND"],
              "source_protocol": {"path": source_protocol.relative_to(tmp_path).as_posix(),
                                  "sha256": E._sha(source_protocol)},
              "source_sha256_lf": {"evals/frozen.py": "a" * 64}}
    _write(design_path, design)
    rows = []
    for index, (seed, category) in enumerate(
            ((seed, category) for seed in design["seeds"] for category in design["categories"]), 1):
        checkpoint = f"{index:064x}"
        report = {"protocol": "fixed_eval_v1", "by_condition": [{}] * 26,
                  "checkpoint": {"sha256": checkpoint},
                  "metrics": {"fall_rate": 0.0, "mean_survival_s": 20.0,
                              "lin_vel_rmse_mps": 0.1 + index / 100,
                              "yaw_rate_rmse_radps": 0.2 + index / 100}}
        stage = tmp_path / "raw" / f"c{index:02d}"
        training_path = stage / "training-result.json"
        _write(training_path, {"checkpoint": {"sha256": checkpoint},
                               "execution": _owned(10.0 + index)})
        result_path = stage / "result.json"
        _write(result_path, {"schema": R.TRAIN.RESULT_SCHEMA, "status": "complete",
                             "seed": seed, "category": category,
                             "checkpoint": {"sha256": checkpoint},
                             "training": {"path": str(training_path),
                                          "sha256": E._sha(training_path)},
                             "fixed_execution": _owned(2.0 + index), "report": report,
                             "claims": {"recovery": False, "diagnosis": False,
                                        "model_advantage": False}})
        rows.append({"case_id": f"c{index:02d}", "seed": seed, "category": category,
                     "checkpoint": {"sha256": checkpoint}, "fixed_report": report,
                     "result": result_path})

    def gate(row: dict[str, Any], _reference: dict[str, Any], _bands: dict[str, Any]):
        if row["category"] == "NONE":
            return {"run": False, "reason": "healthy_gate", "source_outcome": "recovered",
                    "analysis_exclusion": None}
        if not qualified and row["case_id"] == "c02":
            return {"run": False, "reason": "healthy_gate", "source_outcome": "recovered",
                    "analysis_exclusion": "source_failure_absent"}
        return {"run": True, "reason": None, "source_outcome": "not_recovered"}

    expected, _ = E._expected_qualification(rows, gate, {"survival_ratio_min": 0.9})
    qualification_path = tmp_path / "raw/qualification.json"
    _write(qualification_path, expected)
    source_batch = tmp_path / "raw/source-batch.json"
    _write(source_batch, {"schema": R.SOURCE.BATCH_SCHEMA, "status": "complete",
                          "anchor": "b" * 40, "model_requests": 0, "repairs": 0,
                          "automatic_retry": False, "wall_s": 120.0,
                          "protocol": {"path": str(source_protocol),
                                       "sha256": E._sha(source_protocol)}})
    return {"design_path": design_path, "design": design, "source_batch": source_batch,
            "qualification": qualification_path, "rows": rows, "gate": gate}


def _export(tmp_path: Path, fixture: dict[str, Any], out: Path):
    return E.export(
        fixture["design_path"], "b" * 40, fixture["source_batch"],
        fixture["qualification"], out,
        verify_design_fn=lambda _path, _anchor: deepcopy(fixture["design"]),
        source_rows_fn=lambda _path, _design: deepcopy(fixture["rows"]),
        gate_fn=fixture["gate"])


def test_exports_qualified_eight_condition_evidence_without_private_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "public"
    summary = _export(tmp_path, fixture, out)
    assert summary["status"] == "qualified" and summary["comparison_outcome"] is None
    assert summary["healthy_reference_gates"] == 2
    assert summary["runnable_fault_gates"] == 6 and summary["api_requests"] == 0
    assert len(list((out / "conditions").glob("*.json"))) == 8
    first = json.loads((out / "conditions/c01.json").read_text(encoding="utf-8"))
    assert first["fixed_conditions"] == 26
    assert first["timing"]["training_owned_job"]["gpu_kernel_time"] is False
    encoded = "".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
    assert "bench/private" not in encoded and "C:\\Users\\" not in encoded


def test_exports_negative_qualification_as_indeterminate_without_success_promotion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch, qualified=False)
    summary = _export(tmp_path, fixture, tmp_path / "negative-public")
    assert summary["status"] == "source_not_qualified"
    assert summary["comparison_outcome"] == "indeterminate"
    assert summary["runnable_fault_gates"] == 5
    assert summary["claims"]["model_advantage"] is False
    assert summary["claims"]["recovery"] is False


def test_rejects_tampered_qualification_summary_or_gate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    value = json.loads(fixture["qualification"].read_text(encoding="utf-8"))
    value["status"] = "source_not_qualified"
    value["comparison_outcome"] = "indeterminate"
    _write(fixture["qualification"], value)
    with pytest.raises(ValueError, match="qualification"):
        _export(tmp_path, fixture, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_source_protocol_sha_tamper_before_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    batch = json.loads(fixture["source_batch"].read_text(encoding="utf-8"))
    batch["protocol"]["sha256"] = "0" * 64
    _write(fixture["source_batch"], batch)
    with pytest.raises(ValueError, match="registration"):
        _export(tmp_path, fixture, tmp_path / "out")


def test_rejects_checkpoint_sha_mismatch(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["rows"][0]["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity|checkpoint"):
        _export(tmp_path, fixture, tmp_path / "out")


def test_rejects_masked_normal_or_fault_category(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["rows"][1]["category"] = "NONE"
    with pytest.raises(ValueError, match="identity/category"):
        _export(tmp_path, fixture, tmp_path / "out")


def test_refuses_output_overwrite(tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _export(tmp_path, fixture, out)
    with pytest.raises(ValueError, match="fresh"):
        _export(tmp_path, fixture, out)


def test_default_contract_consumes_registered_verifier_rows_and_gate() -> None:
    defaults = E.export.__kwdefaults__
    assert defaults is not None
    assert defaults["verify_design_fn"] is E.R.verify_design
    assert defaults["source_rows_fn"] is E.R._source_rows
    assert defaults["gate_fn"] is E.R.BASE.RUN._gate
