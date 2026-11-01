# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_lineage_source_collection_v2 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def fixture(tmp_path: Path, *, points: int = 300) -> tuple[dict, dict, dict, dict, dict]:
    source_root = tmp_path / "private" / "batch"
    protocol = save(tmp_path / "private/protocol.json", {"schema": "source-v2"})
    result = save(source_root / "result.json", {"schema": "result", "wall_s": 1234.5})
    jobs, sources, private_sources, public_sources = [], {}, {}, {}
    fields = M.INPUT.SELECT.V.TELEMETRY_FIELDS
    for index in range(1, 21):
        job_id, alias = f"private-job-{index:02d}", f"source-{index:02d}"
        jobs.append({"job_id": job_id, "arm": "NONE", "seed": 1000 + index,
                     "role": "reference"})
        stage = source_root / job_id
        checkpoint = save(tmp_path / f"private/checkpoints/{job_id}.pt", {"cp": index})
        telemetry = save(tmp_path / f"private/index/{job_id}/telemetry.json", {
            "run_dir_name": f"C:\\Users\\secret\\{job_id}",
            "summary": {field: {"n": points, "first": index + field_index / 1000,
                                 "last": index + (points - 1) + field_index / 1000,
                                 "mean_last_20pct": index + field_index}
                        for field_index, field in enumerate(fields)},
            "series": {field: [index + step + field_index / 1000 for step in range(points)]
                       for field_index, field in enumerate(fields)}})
        event = tmp_path / f"private/events/{job_id}.tfevents"
        event.parent.mkdir(parents=True, exist_ok=True)
        event.write_bytes(f"event-{index}".encode())
        receipt = save(tmp_path / f"private/index/{job_id}/extract.json", {"returncode": 0})
        manifest = save(stage / "manifest.json", {"job_id": job_id})
        evidence = save(stage / "training/evidence.json", {"checkpoint": str(checkpoint)})
        records = save(stage / "training/records.jsonl", {"update": 300})
        save(stage / "owned-result.json", {
            "exit_code": 0, "wall_s": 10.0 + index, "natural_exit": True,
            "cleanup_verified": True, "job_assigned_before_release": True,
            "timed_out": False, "error_type": None,
            "private_path": str(stage)})
        fixed = save(stage / "fixed-result.json", {
            "report": {"name": job_id, "metrics": {"score": index + 0.25},
                       "checkpoint_path": str(checkpoint)},
            "execution": {"exit_code": 0, "wall_s": 2.0 + index,
                          "natural_exit": True, "cleanup_verified": True,
                          "job_assigned_before_release": True, "timed_out": False,
                          "error_type": None, "log_dir": str(stage)}})
        binding = {
            "checkpoint_sha256": M.sha(checkpoint), "telemetry_sha256": M.sha(telemetry),
            "manifest_sha256": M.sha(manifest), "evidence_sha256": M.sha(evidence),
            "records_sha256": M.sha(records), "fixed_result_sha256": M.sha(fixed)}
        material = {
            "event": {"path": str(event), "sha256": M.sha(event)},
            "event_copy": {"path": str(event), "sha256": M.sha(event)},
            "telemetry": {"path": str(telemetry), "sha256": M.sha(telemetry)},
            "receipt": {"path": str(receipt), "sha256": M.sha(receipt)}}
        sources[job_id] = {
            "stage": stage, "checkpoint": {"path": str(checkpoint),
                                              "sha256": M.sha(checkpoint)},
            "report": {"name": job_id}}
        private_sources[job_id] = {"public_id": alias, "binding": binding,
                                   "material": material}
        public_sources[alias] = binding
    arms = ["NONE", "F1", "F2", "F5", "COMPOUND"] * 2
    cases = [{"case_id": f"case-{index:02d}", "source_job": jobs[index - 1]["job_id"],
              "reference_job": jobs[index + 9]["job_id"], "arm": arms[index - 1],
              "label": "eligible" if index <= 8 else "out_of_support",
              "eligible_after_dedup": index <= 8,
              "reference_relation": "different_seed_normal_reference"}
             for index in range(1, 11)]
    context = {
        "protocol": {"training_jobs": jobs, "cases": cases},
        "result": {"wall_s": 1234.5, "source_jobs_completed": 20,
                   "fixed26_completed": 20, "model_api": 0, "repairs": 0,
                   "runtime_facts_fabricated": False},
        "sources": sources, "cases": cases,
        "duplicate_checkpoint_groups": []}
    metadata = {
        "schema": M.INPUT.METADATA_SCHEMA, "fixture": False,
        "anchor": "a" * 40, "protocol": M.INPUT._ref(protocol),
        "source_result": M.INPUT._ref(result), "sources": private_sources,
        "cases": cases, "duplicate_checkpoint_groups": []}
    public = {"schema": M.INPUT.SCHEMA, "fixture": False, "planned_cases": 10,
              "source_count": 20, "sources": public_sources}
    index_path = save(tmp_path / "private/index/collected.json", public)
    inputs = {"source_root": source_root, "protocol": protocol, "index": index_path,
              "lab": tmp_path / "lab", "anchor": "a" * 40}
    validators = {
        "validate_batch": lambda *_args, **_kwargs: deepcopy(context),
        "checked": lambda _path: (deepcopy(public), deepcopy(metadata))}
    return inputs, validators, context, public, metadata


def test_fixture_export_is_write_once_anonymous_and_preserves_series_values(
        tmp_path: Path) -> None:
    inputs, validators, _context, _public, _metadata = fixture(tmp_path)
    out = tmp_path / "public/tag"
    summary = M.export(
        inputs["source_root"], inputs["protocol"], inputs["index"],
        lab=inputs["lab"], anchor=inputs["anchor"], out=out,
        allow_fixture=True, validators=validators)
    assert summary["scope"] == "development_fixture" and summary["actual"] is False
    assert summary["counts"] == {
        "planned_cases": 10, "valid_after_dedup": 8, "excluded": 2,
        "anonymous_sources": 20, "training_sdk_completed": 20,
        "fixed26_sdk_completed": 20, "actual_model_api_calls": 0, "repairs": 0}
    curves = M.read(out / "sources/source-01/training-curves.json")
    field = M.INPUT.SELECT.V.TELEMETRY_FIELDS[3]
    assert curves["series"][field][17] == 18.003
    assert curves["summary"][field]["mean_last_20pct"] == 4
    fixed = M.read(out / "sources/source-01/fixed26.json")
    assert fixed["report"]["name"] == "source-01"
    assert fixed["report"]["metrics"]["score"] == 1.25
    with pytest.raises(ValueError, match="fresh"):
        M.export(inputs["source_root"], inputs["protocol"], inputs["index"],
                 lab=inputs["lab"], anchor=inputs["anchor"], out=out,
                 allow_fixture=True, validators=validators)


def test_public_inventory_sha_and_privacy_have_no_raw_paths_or_credentials(
        tmp_path: Path) -> None:
    inputs, validators, *_ = fixture(tmp_path)
    out = tmp_path / "public/tag"
    M.export(inputs["source_root"], inputs["protocol"], inputs["index"],
             lab=inputs["lab"], anchor=inputs["anchor"], out=out,
             allow_fixture=True, validators=validators)
    manifest = M.verify_export(
        out, index_path=inputs["index"], allow_fixture=True,
        index_checker=validators["checked"])
    assert manifest["public_files"] == list(M.PUBLIC_FILES)
    assert len(manifest["files"]) == len(M.PUBLIC_FILES) - 1 == 62
    encoded = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
    assert "private-job" not in encoded and "C:\\Users" not in encoded
    assert str(tmp_path) not in encoded and "api_key" not in encoded.lower()
    provenance = M.read(out / "sources/source-01/provenance.json")
    assert provenance["checkpoint_copied"] is provenance["event_copied"] is False
    assert set(provenance["sha256_only"]) == {
        "checkpoint", "event", "event_copy", "telemetry", "extract_receipt",
        "manifest", "training_evidence", "training_records", "fixed_result", "owned_result"}


def test_current_index_sha_drift_is_rejected(tmp_path: Path) -> None:
    inputs, validators, *_ = fixture(tmp_path)
    out = tmp_path / "public/tag"
    M.export(inputs["source_root"], inputs["protocol"], inputs["index"],
             lab=inputs["lab"], anchor=inputs["anchor"], out=out,
             allow_fixture=True, validators=validators)
    save(inputs["index"], {"changed": True})
    with pytest.raises(ValueError, match="index SHA"):
        M.verify_export(out, index_path=inputs["index"], allow_fixture=True,
                        index_checker=validators["checked"])


def test_series_must_be_original_300_points(tmp_path: Path) -> None:
    inputs, validators, *_ = fixture(tmp_path, points=299)
    with pytest.raises(ValueError, match="300-point"):
        M.export(inputs["source_root"], inputs["protocol"], inputs["index"],
                 lab=inputs["lab"], anchor=inputs["anchor"], out=tmp_path / "public/tag",
                 allow_fixture=True, validators=validators)


def test_fixture_requires_explicit_validators_and_never_claims_actual(tmp_path: Path) -> None:
    inputs, _validators, *_ = fixture(tmp_path)
    with pytest.raises(ValueError, match="명시 validator"):
        M.export(inputs["source_root"], inputs["protocol"], inputs["index"],
                 lab=inputs["lab"], anchor=inputs["anchor"], out=tmp_path / "public/tag",
                 allow_fixture=True)


def test_actual_export_requires_new_evals_results_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evals/results"):
        M.export(tmp_path / "source", tmp_path / "protocol", tmp_path / "index",
                 lab=tmp_path / "lab", anchor="a" * 40, out=tmp_path / "outside")


def test_cli_passes_actual_source_protocol_index_lab_anchor_and_out(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(M, "export", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs) or {})
    paths = [tmp_path / name for name in ("source", "protocol", "index", "lab", "out")]
    assert M.main([
        "--source-root", str(paths[0]), "--protocol", str(paths[1]),
        "--index", str(paths[2]), "--lab", str(paths[3]),
        "--anchor", "a" * 40, "--out", str(paths[4])]) == 0
    assert captured["args"] == tuple(paths[:3])
    assert captured["kwargs"] == {"lab": paths[3], "anchor": "a" * 40, "out": paths[4]}
