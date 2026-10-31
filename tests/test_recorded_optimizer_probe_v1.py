# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import recorded_optimizer_probe_v1 as R


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _facts(role: str, observed: int = 4) -> dict[str, bool | int]:
    prefix = f"runtime:{role}:"
    return {
        prefix + "available": True, prefix + "updates_observed": 3,
        prefix + "expected_optimizer_steps": 20,
        prefix + "optimizer_steps_min": observed, prefix + "optimizer_steps_max": observed,
        prefix + "expected_minibatches": 20,
        prefix + "minibatches_min": observed, prefix + "minibatches_max": observed,
    }


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str = "case") -> tuple[Path, Path]:
    checkpoint = "a" * 64
    manifest, evidence, owned = (tmp_path / "manifest.json", tmp_path / "evidence",
                                  tmp_path / "owned.json")
    evidence.mkdir()
    for path in (manifest, owned, evidence / "training-evidence.json",
                 evidence / "guard-records.jsonl", evidence / "records.jsonl",
                 evidence / "checkpoint.pt"):
        path.write_text(path.name, encoding="utf-8")
    collected = tmp_path / "collected.json"
    _write(collected, {"schema": "fixture"})
    runtime_dir = tmp_path / "runtime"
    runtime = runtime_dir / "artifact.json"
    metadata_path = runtime_dir / R.FACT.METADATA_NAME
    raw_sha = {"manifest": "1" * 64, "owned_result": "2" * 64,
               "training_evidence": "3" * 64, "guard_records": "4" * 64,
               "close_marker": "5" * 64, "observer_records": "6" * 64,
               "readback": "7" * 64, "checkpoint": "8" * 64}
    metadata = {
        "role": role,
        "sources": {"manifest": str(manifest), "owned_result": str(owned),
                    "training_evidence": str(evidence / "training-evidence.json"),
                    "guard_records": str(evidence / "guard-records.jsonl"),
                    "observer_records": str(evidence / "records.jsonl"),
                    "checkpoint": str(evidence / "checkpoint.pt")},
        "raw_source_sha256": raw_sha,
    }
    metadata["sources"].update({
        "close_marker": str(evidence / "close.marker"),
        "readback": str(evidence / "readback.json"),
    })
    for name in ("close.marker", "readback.json"):
        (evidence / name).write_text(name, encoding="utf-8")
    metadata["raw_source_sha256"] = {
        name: R.sha(Path(path)) for name, path in metadata["sources"].items()
    }
    _write(metadata_path, metadata)
    _write(runtime, {"envelope": {"source_binding": {"checkpoint_sha256": checkpoint}},
                     "private_metadata_ref": {"file": R.FACT.METADATA_NAME,
                                              "sha256": R.sha(metadata_path)}})
    monkeypatch.setattr(R.INPUT, "checked", lambda path: (
        {"source_binding": {"checkpoint_sha256": checkpoint,
                            "telemetry_sha256": "b" * 64, "binding_digest": "c" * 64}},
        {"manifest": str(manifest), "evidence_dir": str(evidence),
         "owned_result": str(owned), "private_training": {"seed": 2311, "arm": "FIRST"}},
    ))
    monkeypatch.setattr(R.FACT, "checked", lambda path, **kwargs: (
        {"source_binding": {"checkpoint_sha256": checkpoint, "binding_digest": "d" * 64}},
        _facts(role),
    ))
    return collected, runtime


def test_approved_cpu_reuse_flow_is_one_time_and_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected, runtime = _inputs(tmp_path, monkeypatch)
    ledger = tmp_path / "probe.ledger.jsonl"
    plan_dir = tmp_path / "plan"
    plan = R.prepare(collected, runtime, role="case", request_id="req-1", params=R.PARAMS,
                     ledger_path=ledger, evidence_kind="actual_training_reuse", out=plan_dir)
    auth_path = tmp_path / "authorization.json"
    R.authorize(plan_dir / "plan.json", approved_by="host-test", out=auth_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = R.execute(plan_dir / "plan.json", auth_path, out=receipt_path)

    assert receipt["probe"] == R.PROBE
    assert receipt["measurement"]["updates_observed"] == 3
    assert receipt["measurement"]["expected_optimizer_steps"] == 20
    assert receipt["measurement"]["optimizer_steps_min"] == 4
    assert receipt["measurement"]["step_count_matches_config"] is False
    assert receipt["outcome"] == "abnormal"
    assert receipt["source_binding"]["seed"] == 2311
    assert receipt["scope"]["new_training"] is False
    assert receipt["evidence_kind"] == "actual_training_reuse"
    assert R.checked_receipt(plan_dir / "plan.json", receipt_path) == receipt
    assert [row["event"] for row in R._events(ledger, "req-1")] == [
        "proposed", "approved", "consumed", "receipt"]
    with pytest.raises(ValueError, match="consume 전"):
        R.execute(plan_dir / "plan.json", auth_path, out=tmp_path / "second.json")
    assert plan["scope"]["new_observation"] is False


def test_flag_or_unapproved_request_cannot_create_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected, runtime = _inputs(tmp_path, monkeypatch)
    ledger = tmp_path / "probe.ledger.jsonl"
    plan_dir = tmp_path / "plan"
    R.prepare(collected, runtime, role="case", request_id="req-2", params=R.PARAMS,
              ledger_path=ledger, evidence_kind="development_fixture", out=plan_dir)
    fake_auth = tmp_path / "fake.json"
    _write(fake_auth, {"approved": True})
    with pytest.raises(ValueError):
        R.execute(plan_dir / "plan.json", fake_auth, out=tmp_path / "receipt.json")
    assert [row["event"] for row in R._events(ledger, "req-2")] == ["proposed"]


def test_raw_observer_bytes_are_rechecked_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected, runtime = _inputs(tmp_path, monkeypatch)
    plan_dir = tmp_path / "plan"
    R.prepare(collected, runtime, role="case", request_id="req-raw", params=R.PARAMS,
              ledger_path=tmp_path / "ledger", evidence_kind="development_fixture",
              out=plan_dir)
    (tmp_path / "evidence/records.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="raw source bytes"):
        R.authorize(plan_dir / "plan.json", approved_by="host", out=tmp_path / "auth.json")


def test_params_source_and_ledger_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected, runtime = _inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="params"):
        R.prepare(collected, runtime, role="case", request_id="req-3",
                  params={**R.PARAMS, "updates": [0]}, ledger_path=tmp_path / "ledger-a",
                  evidence_kind="development_fixture", out=tmp_path / "plan-a")

    ledger = tmp_path / "ledger-b"
    plan_dir = tmp_path / "plan-b"
    R.prepare(collected, runtime, role="case", request_id="req-4", params=R.PARAMS,
              ledger_path=ledger, evidence_kind="development_fixture", out=plan_dir)
    runtime.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        R.authorize(plan_dir / "plan.json", approved_by="host", out=tmp_path / "auth.json")

    rows = ledger.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["params_digest"] = "0" * 64
    ledger.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        R._ledger(ledger)
