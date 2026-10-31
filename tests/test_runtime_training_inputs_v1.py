# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "tests")]

import optimizer_runtime_facts_v1 as FACT
import runtime_training_inputs_v1 as C
import test_lineage_recovery_inputs_v1 as OLD_TEST


def _extract(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        Path(argv[-1]).write_text(json.dumps({"summary": {}}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok\r\n", stderr="")

    monkeypatch.setattr(C.subprocess, "run", run)


def _collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, arm: str = "COMPOUND"):
    completed = OLD_TEST._completed(tmp_path, monkeypatch, arm=arm)
    _extract(monkeypatch)
    out = tmp_path / "input"
    value = C.collect(
        completed["manifest_path"], completed["evidence_dir"], completed["owned"],
        lab=completed["lab"], isaac_python=Path(sys.executable), out=out)
    return completed, out, value


def test_collect_generic_compound_and_checked_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, out, value = _collect(tmp_path, monkeypatch)
    checked, metadata = C.checked(out / C.COLLECTED)
    assert checked == value
    assert checked["schema"] == C.SCHEMA
    assert checked["source_binding"]["purpose"] == C.PURPOSE
    assert checked["scope"] == {
        "activity": "cpu_training_input_materialization_only",
        "approval": False,
        "execution_allowed": False,
    }
    assert "original_faults" not in checked
    assert metadata["private_training"]["arm"] == "COMPOUND"
    assert Path(metadata["event_copy"]).read_bytes() == Path(completed["event"]).read_bytes()


def test_checked_rejects_raw_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _completed, out, _value = _collect(tmp_path, monkeypatch)
    metadata = json.loads((out / C.METADATA).read_text(encoding="utf-8"))
    observer = Path(metadata["raw_sources"]["observer_records"])
    observer.write_bytes(observer.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        C.checked(out / C.COLLECTED)


def test_packet_binds_runtime_checkpoint_and_uses_null_for_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_completed, case_out, _ = _collect(tmp_path / "case", monkeypatch)
    _reference_completed, reference_out, _ = _collect(tmp_path / "reference", monkeypatch, arm="NONE")
    case_checkpoint = Path(case_completed["checkpoint"])
    runtime_out = tmp_path / "runtime"
    FACT.prepare(
        case_completed["manifest_path"], case_completed["evidence_dir"], case_completed["owned"],
        role="case", lab=case_completed["lab"], expected_checkpoint_sha256=C._sha(case_checkpoint),
        out=runtime_out)
    result = C.make_packet(
        case_out / C.COLLECTED, reference_out / C.COLLECTED,
        case_runtime_path=runtime_out / FACT.ARTIFACT_NAME, reference_runtime_path=None)
    facts = result["packet"]["facts"]
    assert facts["runtime:case:available"] is True
    assert facts["runtime:case:expected_optimizer_steps"] == 20
    assert facts["runtime:reference:available"] is False
    assert facts["runtime:reference:optimizer_steps_min"] is None
    assert "original_faults" not in result["packet"]


def test_unavailable_runtime_is_null_not_zero() -> None:
    facts = FACT.unavailable_facts("reference")
    assert facts["runtime:reference:available"] is False
    assert all(value is None for key, value in facts.items() if not key.endswith(":available"))
