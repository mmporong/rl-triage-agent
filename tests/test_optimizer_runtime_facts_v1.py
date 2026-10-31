from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import optimizer_runtime_facts_v1 as F


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, values: list[int] | None = None,
) -> dict[str, object]:
    values = [4, 4, 4] if values is None else values
    source = tmp_path / "source"
    evidence_dir = source / "training"
    manifest_path = source / "manifest.json"
    owned_path = source / "owned-result.json"
    records_path = source / "observer-records.jsonl"
    readback_path = source / "readback.json"
    checkpoint_path = source / "model_299.pt"
    lab = tmp_path / "IsaacLab"
    lab.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint-v1")
    checkpoint_sha = F._sha(checkpoint_path)
    manifest = {
        "records_path": str(records_path),
        "readback_path": str(readback_path),
        "configured_learning_epochs": 5,
        "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
        "original_faults": ["SECRET_FAULT"],
        "remaining_faults": ["SECRET_REMAINING"],
    }
    rows = [
        {
            "schema": "optimizer_observability_record_v2",
            "update_index": index,
            "expected": {"optimizer_steps": 20, "minibatches": 20},
            "observed": {"optimizer_steps": value, "minibatches": value},
            "readback_label": "SECRET_LABEL",
            "status": "SECRET_STATUS",
            "error": None,
        }
        for index, value in enumerate(values)
    ]
    records_path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    evidence = {
        "residual_records": {"path": str(records_path), "sha256": F._sha(records_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
    }
    _json(manifest_path, manifest)
    _json(owned_path, {"natural_exit": True})
    _json(readback_path, {"probe": "SECRET_PROBE", "recommendation": "SECRET_REC"})
    for name, content in (
        (F.GUARD.EVIDENCE, evidence),
        (F.GUARD.RECORDS, {"completion": True}),
        (F.GUARD.CLOSE_READY, {"closed": True}),
    ):
        path = evidence_dir / name
        if name == F.GUARD.RECORDS:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content) + "\n", encoding="utf-8")
        else:
            _json(path, content)

    def checked(manifest_arg: Path, evidence_arg: Path, owned_arg: Path, lab_arg: Path):
        assert Path(manifest_arg).resolve() == manifest_path.resolve()
        assert Path(evidence_arg).resolve() == evidence_dir.resolve()
        assert Path(owned_arg).resolve() == owned_path.resolve()
        assert Path(lab_arg).resolve() == lab.resolve()
        current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_evidence = json.loads((evidence_dir / F.GUARD.EVIDENCE).read_text(encoding="utf-8"))
        return current_manifest, current_evidence, {}, {"records": rows, "readback": {}}

    monkeypatch.setattr(F.RECOVERY, "_check_training", checked)
    return {
        "manifest": manifest_path, "evidence_dir": evidence_dir, "owned": owned_path,
        "lab": lab, "checkpoint_sha": checkpoint_sha, "records": records_path,
        "readback": readback_path, "checkpoint": checkpoint_path,
    }


def _prepare(fixture: dict[str, object], tmp_path: Path, *, role: str = "case") -> Path:
    out = tmp_path / "sealed"
    F.prepare(
        fixture["manifest"], fixture["evidence_dir"], fixture["owned"], role=role,
        lab=fixture["lab"], expected_checkpoint_sha256=fixture["checkpoint_sha"], out=out)
    return out / F.ARTIFACT_NAME


def test_round_trip_has_exact_eight_numeric_facts_and_no_host_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before = {name: Path(fixture[name]).read_bytes() for name in (
        "manifest", "owned", "records", "readback", "checkpoint")}
    artifact_path = _prepare(fixture, tmp_path)
    envelope, facts = F.checked(
        artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])

    assert set(facts) == {f"runtime:case:{name}" for name in F.FACT_NAMES}
    assert facts == {
        "runtime:case:available": True,
        "runtime:case:updates_observed": 3,
        "runtime:case:expected_optimizer_steps": 20,
        "runtime:case:optimizer_steps_min": 4,
        "runtime:case:optimizer_steps_max": 4,
        "runtime:case:expected_minibatches": 20,
        "runtime:case:minibatches_min": 4,
        "runtime:case:minibatches_max": 4,
    }
    model_text = json.dumps(envelope["model_view"], sort_keys=True)
    assert not any(label in model_text for label in (
        "SECRET_FAULT", "SECRET_REMAINING", "SECRET_LABEL", "SECRET_STATUS",
        "SECRET_PROBE", "SECRET_REC", str(fixture["records"])))
    assert envelope["scope"]["approval"] is False
    assert envelope["scope"]["execution_allowed"] is False
    assert envelope["source_binding"]["source_contract"] == F.SOURCE_CONTRACT
    assert "COMPOUND" in envelope["source_binding"]["accepted_arms"]
    assert before == {name: Path(fixture[name]).read_bytes() for name in before}


def test_reference_and_unavailable_fact_shapes_are_fixed() -> None:
    facts = F.unavailable_facts("reference")
    assert set(facts) == {f"runtime:reference:{name}" for name in F.FACT_NAMES}
    assert facts["runtime:reference:available"] is False
    assert all(value is None for key, value in facts.items() if not key.endswith(":available"))
    assert set(F.FACT_IDS) == {
        f"runtime:{role}:{name}" for role in ("case", "reference") for name in F.FACT_NAMES}


def test_registered_arm_extension_is_scoped_and_restored_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = F.RECOVERY.ALLOWED_ARMS

    def fail(*_args: object) -> object:
        assert F.RECOVERY.ALLOWED_ARMS == set(F.RECOVERY.TRAIN.RESIDUAL.ARMS)
        assert "COMPOUND" in F.RECOVERY.ALLOWED_ARMS
        raise RuntimeError("stop")

    monkeypatch.setattr(F.RECOVERY, "_check_training", fail)
    with pytest.raises(RuntimeError, match="stop"):
        F._check_registered_training(Path("m"), Path("e"), Path("o"), Path("l"))
    assert F.RECOVERY.ALLOWED_ARMS is original


def test_zero_observation_is_available_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, values=[0, 0, 0])
    artifact_path = _prepare(fixture, tmp_path)
    _envelope, facts = F.checked(
        artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])
    assert facts["runtime:case:available"] is True
    assert facts["runtime:case:optimizer_steps_min"] == 0
    assert facts["runtime:case:minibatches_max"] == 0


@pytest.mark.parametrize("bad", [True, -1, float("nan")])
def test_bool_negative_and_nan_observed_counts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in Path(fixture["records"]).read_text().splitlines()]
    rows[1]["observed"]["optimizer_steps"] = bad
    Path(fixture["records"]).write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    evidence_path = Path(fixture["evidence_dir"]) / F.GUARD.EVIDENCE
    evidence = json.loads(evidence_path.read_text())
    evidence["residual_records"]["sha256"] = F._sha(Path(fixture["records"]))
    _json(evidence_path, evidence)
    with pytest.raises((TypeError, ValueError)):
        F.build(
            fixture["manifest"], fixture["evidence_dir"], fixture["owned"], role="case",
            lab=fixture["lab"], expected_checkpoint_sha256=fixture["checkpoint_sha"])


def test_missing_update_and_wrong_artifact_shape_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = Path(fixture["records"]).read_text().splitlines()
    Path(fixture["records"]).write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
    evidence_path = Path(fixture["evidence_dir"]) / F.GUARD.EVIDENCE
    evidence = json.loads(evidence_path.read_text())
    evidence["residual_records"]["sha256"] = F._sha(Path(fixture["records"]))
    _json(evidence_path, evidence)
    with pytest.raises(ValueError, match="0/1/2"):
        F.build(
            fixture["manifest"], fixture["evidence_dir"], fixture["owned"], role="case",
            lab=fixture["lab"], expected_checkpoint_sha256=fixture["checkpoint_sha"])

    fixture = _fixture(tmp_path / "second", monkeypatch)
    artifact_path = _prepare(fixture, tmp_path / "second")
    artifact = json.loads(artifact_path.read_text())
    artifact["extra"] = True
    _json(artifact_path, artifact)
    with pytest.raises(ValueError, match="artifact shape"):
        F.checked(artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])


def test_artifact_metadata_raw_bytes_and_checkpoint_binding_tampering_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    artifact_path = _prepare(fixture, tmp_path)
    artifact = json.loads(artifact_path.read_text())
    artifact["envelope"]["model_view"]["facts"]["runtime:case:optimizer_steps_min"] = 5
    _json(artifact_path, artifact)
    with pytest.raises(ValueError, match="envelope"):
        F.checked(artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])

    fixture = _fixture(tmp_path / "metadata", monkeypatch)
    artifact_path = _prepare(fixture, tmp_path / "metadata")
    (artifact_path.parent / F.METADATA_NAME).write_bytes(
        (artifact_path.parent / F.METADATA_NAME).read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata bytes"):
        F.checked(artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])

    fixture = _fixture(tmp_path / "raw", monkeypatch)
    artifact_path = _prepare(fixture, tmp_path / "raw")
    Path(fixture["readback"]).write_bytes(Path(fixture["readback"]).read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata"):
        F.checked(artifact_path, expected_checkpoint_sha256=fixture["checkpoint_sha"])

    fixture = _fixture(tmp_path / "checkpoint", monkeypatch)
    artifact_path = _prepare(fixture, tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        F.checked(artifact_path, expected_checkpoint_sha256="f" * 64)
