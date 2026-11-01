# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_source_inputs_v2 as M


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _telemetry(base: float) -> dict:
    return {"summary": {
        field: {"n": index + 1, "mean_last_20pct": base + index}
        for index, field in enumerate(M.SELECT.V.TELEMETRY_FIELDS)
    }}


def _jobs() -> tuple[list[dict], list[dict]]:
    cases, jobs, _smoke = M.BATCH.jobs(list(range(2550, 2560)), [3550, 3551], 4550)
    return cases, jobs


def _source(tmp_path: Path, job: dict, index: int) -> dict:
    stage = tmp_path / "batch" / job["job_id"]
    checkpoint = tmp_path / "logs" / job["job_id"] / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{index}".encode())
    for name, value in (
        ("manifest.json", {"job": job["job_id"]}),
        (f"training/{M.BATCH.GUARD.EVIDENCE}", {"evidence": job["job_id"]}),
        (f"training/{M.BATCH.GUARD.RECORDS}", {"record": job["job_id"]}),
        ("fixed-result.json", {"fixed": job["job_id"]}),
    ):
        _write(stage / name, value)
    return {
        "stage": stage,
        "manifest": {"job_id": job["job_id"], "arm": job["arm"], "seed": job["seed"]},
        "evidence": {"log_dir": str(checkpoint.parent), "checkpoint": {
            "path": str(checkpoint), "sha256": M._sha(checkpoint),
            "log_dir": str(checkpoint.parent)}, "ppo_readback": {
                "configured": {"learning_epochs": 5, "mini_batches": 4},
                "effective": {"learning_epochs": 1 if job["arm"] == "COMPOUND" else 5,
                              "mini_batches": 4}}},
        "checkpoint": {"path": str(checkpoint), "sha256": M._sha(checkpoint),
                       "log_dir": str(checkpoint.parent)},
        "report": {"name": job["job_id"]},
        "effective_fault_metadata": {"registered_arm": job["arm"], "applicable": True},
    }


def _context(tmp_path: Path) -> dict:
    cases, jobs = _jobs()
    sources = {job["job_id"]: _source(tmp_path, job, index) for index, job in enumerate(jobs)}
    classified = [{"case_id": case["case_id"], "source_job": case["source_job"],
                   "reference_job": case["reference_job"], "arm": case["arm"],
                   "seed": case["seed"], "label": "eligible",
                   "fixed_verdict": {"label": "unhealthy"}}
                  for case in cases]
    return {"protocol": {"cases": cases, "training_jobs": jobs}, "result": {},
            "sources": sources, "cases": classified, "duplicate_checkpoint_groups": []}


def test_collect_and_packet_keep_private_truth_out_of_model_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    protocol = tmp_path / "protocol.json"
    result = tmp_path / "batch/result.json"
    _write(protocol, {"schema": M.BATCH.SCHEMA})
    _write(result, {"schema": M.BATCH.RESULT_SCHEMA})
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.setattr(M, "validate_batch", lambda *_args, **_kwargs: context)

    def materialize(stage: Path, evidence: dict, out: Path, isaac_python: Path) -> dict:
        del evidence, isaac_python
        out.mkdir(parents=True)
        event = out / "source.events.out.tfevents.raw"
        telemetry = out / "telemetry.json"
        receipt = out / "extract.receipt.json"
        event.write_bytes(stage.name.encode())
        _write(telemetry, _telemetry(float(len(stage.name))))
        _write(receipt, {"returncode": 0})
        return {"stage": str(stage), "event": M._ref(event), "event_copy": M._ref(event),
                "telemetry": M._ref(telemetry), "receipt": M._ref(receipt)}

    monkeypatch.setattr(M, "_materialize_telemetry", materialize)
    public = M.collect_batch(tmp_path / "batch", protocol, lab=tmp_path, anchor="a" * 40,
                             isaac_python=python, out=tmp_path / "inputs")
    encoded_public = json.dumps(public, ensure_ascii=False)
    assert set(public["sources"]) == {f"source-{index:02d}" for index in range(1, 21)}
    assert not any(job["job_id"] in encoded_public for job in context["protocol"]["training_jobs"])
    assert "unhealthy" not in encoded_public
    assert "C:\\" not in encoded_public and str(tmp_path) not in encoded_public

    packet = M.make_case_packet(tmp_path / "inputs/collected.json", "case-03")
    M.SELECT.V.S.validate_packet(packet["packet"])
    facts = packet["packet"]["facts"]
    assert len([name for name in facts if name.startswith("telemetry:")]) == 48
    for role in ("case", "reference"):
        assert facts[f"runtime:{role}:available"] is False
        assert all(facts[f"runtime:{role}:{name}"] is None
                   for name in M.RUNTIME.FACT_NAMES if name != "available")
    model_json = json.dumps(packet["packet"], ensure_ascii=False)
    assert "F1" not in model_json and "unhealthy" not in model_json and str(tmp_path) not in model_json
    assert packet["private_metadata"]["classification"]["arm"] == "F1"
    assert packet["scope"]["runtime_counts_fabricated"] is False


def test_classification_independent_normal_f5_and_not_applicable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M.GATE.RESIDUAL.D1.V.R1.U, "walks", lambda _report: True)
    monkeypatch.setattr(M.GATE.RESIDUAL, "recovery", lambda *_args: {
        "verdict": {"label": "healthy"}})
    normal = {"case_id": "case-01", "arm": "NONE", "seed": 2550,
              "source_job": "normal-a", "reference_job": "normal-b",
              "reference_relation": "different_seed_normal_reference"}
    row = M._classify(normal, {}, {}, {}, {"applicable": True}, "a" * 64, "b" * 64)
    assert row["label"] == "eligible" and row["reason"].startswith("independent_normal")
    assert row["source_job"] != row["reference_job"]
    assert M._classify(normal, {}, {}, {}, {"applicable": True}, "a" * 64, "a" * 64)[
        "label"] == "duplicate_checkpoint"

    monkeypatch.setattr(M.GATE.RESIDUAL, "recovery", lambda *_args: {
        "verdict": {"label": "unhealthy"}})
    f5 = {**normal, "case_id": "case-09", "arm": "F5", "source_job": "f5"}
    row = M._classify(f5, {}, {}, {}, {"applicable": True}, "c" * 64, "b" * 64)
    assert row["label"] == "out_of_support" and row["correction_support"] == "unsupported"
    assert "low_friction" in row["reason"]
    assert M._classify(f5, {}, {}, {}, {"applicable": False}, "c" * 64, "b" * 64)[
        "label"] == "not_applicable"


def test_validate_batch_reconsumes_twenty_records_and_fixed26(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases, jobs = _jobs()
    protocol = {"schema": M.BATCH.SCHEMA, "cases": cases, "training_jobs": jobs,
                "source_sha256_lf": {M.GATE.FIXED_PROTOCOL: M.GATE.FIXED_PROTOCOL_SHA256_LF}}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    root = tmp_path / "batch"
    records = []
    evidence_by_job = {}
    manifest_by_job = {}
    for index, job in enumerate(jobs):
        stage = root / job["job_id"]
        checkpoint = tmp_path / "logs" / job["job_id"] / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"cp-{index}".encode())
        manifest = {"job_id": job["job_id"], "arm": job["arm"], "seed": job["seed"]}
        evidence = {"log_dir": str(checkpoint.parent), "checkpoint": {
            "path": str(checkpoint), "sha256": M._sha(checkpoint), "log_dir": str(checkpoint.parent)},
            "ppo_readback": {
                "configured": {"learning_epochs": 5, "mini_batches": 4},
                "effective": {"learning_epochs": 1 if job["arm"] == "COMPOUND" else 5,
                              "mini_batches": 4}}}
        manifest_by_job[job["job_id"]] = manifest
        evidence_by_job[job["job_id"]] = evidence
        _write(stage / "manifest.json", manifest)
        owned = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                 "job_assigned_before_release": True, "timed_out": False, "error_type": None}
        _write(stage / "owned-result.json", owned)
        _write(stage / f"training/{M.BATCH.GUARD.EVIDENCE}", evidence)
        (stage / f"training/{M.BATCH.GUARD.RECORDS}").write_text("{}\n", encoding="utf-8")
        fixed = {"report": {"name": job["job_id"]}, "execution": owned}
        _write(stage / "fixed-result.json", fixed)
        records.append({
            "job_id": job["job_id"], "purpose": "source", "arm": job["arm"],
            "seed": job["seed"], "role": job["role"], "checkpoint": evidence["checkpoint"],
            "runtime_facts": None,
            "evidence_sha256": M._sha(stage / f"training/{M.BATCH.GUARD.EVIDENCE}"),
            "records_sha256": M._sha(stage / f"training/{M.BATCH.GUARD.RECORDS}"),
            "fixed26": {"conditions": 26, "execution": owned,
                        "result_sha256": M._sha(stage / "fixed-result.json")},
        })
    result = {"schema": M.BATCH.RESULT_SCHEMA, "status": "complete", "mode": "all",
              "source_jobs_completed": 20, "fixed26_completed": 20, "records": records,
              "runtime_facts_fabricated": False, "model_api": 0, "repairs": 0}
    _write(root / "result.json", result)
    _write(root / "intent.json", {"anchor": "a" * 40,
                                    "protocol_sha256_lf": M._sha(protocol_path, lf=True)})
    monkeypatch.setattr(M.BATCH, "verify", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(M.BATCH, "manifest", lambda _p, _v, row, _a, purpose: manifest_by_job[row["job_id"]])
    monkeypatch.setattr(M.BATCH.CHILD, "validate_manifest", lambda value: value)
    monkeypatch.setattr(M.BATCH, "completed", lambda stage, _manifest, _owned: evidence_by_job[stage.name])
    monkeypatch.setattr(M, "_validate_report", lambda report, _sha: report)
    monkeypatch.setattr(M.GATE, "_bands", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(M, "_classify", lambda case, *_args: {
        "case_id": case["case_id"], "source_job": case["source_job"],
        "reference_job": case["reference_job"], "label": "eligible"})
    validated = M.validate_batch(root, protocol_path, lab=tmp_path, anchor="a" * 40)
    assert len(validated["sources"]) == 20 and len(validated["cases"]) == 10
    assert all(item["evidence"]["checkpoint"]["sha256"]
               == evidence_by_job[job_id]["checkpoint"]["sha256"]
               for job_id, item in validated["sources"].items())


def test_extract_uses_cpu_environment_and_no_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "log"
    log.mkdir()
    (log / "events.out.tfevents.fixture").write_bytes(b"event")
    python = tmp_path / "python.exe"
    python.touch()
    calls = 0

    def run(argv, **kwargs):
        nonlocal calls
        calls += 1
        _write(Path(argv[-1]), _telemetry(1.0))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["timeout"] == M.SELECT.EXTRACT_TIMEOUT_S
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(M.subprocess, "run", run)
    result = M._materialize_telemetry(tmp_path / "stage", {"log_dir": str(log)},
                                      tmp_path / "out", python)
    assert calls == 1 and result["telemetry"]["sha256"] == M._sha(tmp_path / "out/telemetry.json")
