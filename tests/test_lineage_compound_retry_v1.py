from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_retry_v1 as R


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_protocol_registers_two_rows_and_fresh_nine_request_queue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old_nim = json.loads((ROOT / "bench/protocols/lineage_nim_transport_20261031_v1.json")
                         .read_text(encoding="utf-8"))
    old_nim["plan"] = R.transport_plan()
    nim_path = tmp_path / "lineage_compound_nim_20261031_v1.json"
    _write(nim_path, old_nim)
    original_relative = R._relative
    monkeypatch.setattr(R, "_relative", lambda path: (
        "bench/protocols/lineage_compound_nim_20261031_v1.json"
        if Path(path).resolve() == nim_path.resolve() else original_relative(path)
    ))
    protocol = R.prepare_protocol(
        ROOT / "bench/protocols/lineage_comparison_20261031_v1.json",
        nim_path,
        ROOT / "bench/protocols/lineage_probe_failure_evidence_20261031_v3.json",
    )

    assert protocol["schema"] == R.SCHEMA
    assert protocol["target"] == {
        "case_id": "case-07",
        "seed": 2307,
        "arm": "COMPOUND",
        "policies": ["registry_next", "model_adaptive"],
        "execution_plan": list(R.PLAN),
    }
    assert protocol["plan"] == R.transport_plan()
    assert protocol["plan"]["global_request_cap"] == 9
    assert protocol["retry_contract"]["automatic_retry"] is False
    assert set(protocol["driver_source_sha256_lf"]) == set(R.BOUND_SOURCE_FILES)


def test_transport_initialize_uses_actual_plan_schema_and_enforces_global_cap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = tmp_path / "protocol.json"
    _write(protocol, {"placeholder": True})
    monkeypatch.setattr(R.T, "ROOT", tmp_path)
    monkeypatch.setattr(
        R.T,
        "_verify_sources",
        lambda path, anchor: {"plan": R.transport_plan()},
    )
    queue = tmp_path / "queue"
    session = R.T.initialize(queue, protocol, "a" * 40, R.transport_plan())
    assert session["plan"]["global_request_cap"] == 9

    messages = [{"role": "user", "content": "registered diagnostic choice"}]
    for slot in session["plan"]["logical_slots"]:
        for _ in range(3):
            ref = R.T.enqueue(queue, slot["slot_id"], messages, session["params"], "1" * 64, "2" * 64)
            _write(queue / "responses" / f"{ref['request_id']}.json", {})
    with pytest.raises(RuntimeError, match="global request cap"):
        R.T.enqueue(queue, "case-07.model.1", messages, session["params"], "1" * 64, "2" * 64)


def _original_row(tmp_path: Path, policy: str) -> dict:
    run = tmp_path / policy
    step = run / "probes/step_01"
    probe = "P_reward" if policy == "model_adaptive" else "P_value"
    auth = step / "authorization.json"
    _write(auth, {"probe": probe, "policy": policy})
    reasons = ["unapproved_gpu_process:139312", "unverified_process_identity:139312"]
    for role in ("reference", "case"):
        _write(step / role / "gpu_before.json", {
            "compute_process_count": 0,
            "busy_reasons": reasons,
            "processes": [{"pid": 139312, "name": "svchost.exe", "type": "C+G"}],
        })
    receipts = {
        role: {"status": "unknown", "measurement": None, "execution": {},
               "error_type": "RuntimeError", "role": role}
        for role in ("reference", "case")
    }
    result = {
        "end_state": "HOLD", "hold_reason": "probe_outcome_unknown", "decision_count": 1,
        "steps": [{"outcome": "unknown", "probe": probe, "authorization_path": str(auth),
                   "paired_receipts": receipts}],
        "requests": 1 if policy == "model_adaptive" else 0,
        "tokens": {}, "wall": {}, "selection_actor": [], "model_validity": [],
        "model_observation_use": [],
    }
    result_path = run / "result.json"
    _write(result_path, result)
    return {"case_id": "case-07", "arm": "COMPOUND", "policy": policy,
            "result": result, "result_path": str(result_path), "result_sha256": _sha(result_path)}


def test_original_failure_requires_absent_receipts_and_same_svchost_cause(tmp_path: Path) -> None:
    row = _original_row(tmp_path, "registry_next")
    evidence = R._original_failure(row)
    assert evidence["failure_cause"]["process"] == "svchost.exe"
    assert all(value["full_receipt_absent"] and value["owned_execution_absent"]
               for value in evidence["roles"].values())

    authorization = Path(row["result"]["steps"][0]["authorization_path"])
    _write(authorization.parent / "case/probe_receipt.json", {"status": "complete"})
    with pytest.raises(ValueError, match="full receipt"):
        R._original_failure(row)


def test_lease_read_scope_retries_only_canonical_metadata_and_restores_reader(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = tmp_path / "metadata.json"
    other = tmp_path / "other.json"
    original = R.LEASE.B.read
    calls = {"count": 0}

    def flaky(path: Path) -> dict:
        calls["count"] += 1
        if Path(path).resolve() == canonical.resolve() and calls["count"] == 1:
            raise PermissionError(13, "busy", str(canonical))
        if Path(path).resolve() == other.resolve():
            raise PermissionError(13, "busy", str(other))
        return {"fresh": True}

    monkeypatch.setattr(R.LEASE, "canonical_path", lambda: canonical)
    monkeypatch.setattr(R.LEASE, "_READ", flaky)
    monkeypatch.setattr(R.LEASE, "_SLEEP", lambda _seconds: None)
    evidence = tmp_path / "lease-evidence.json"
    with R.lease_read_scope(evidence):
        assert R.LEASE.B.read(canonical) == {"fresh": True}
        with pytest.raises(PermissionError):
            R.LEASE.B.read(other)
    assert R.LEASE.B.read is original
    recorded = json.loads(evidence.read_text(encoding="utf-8"))
    assert recorded["status"] == "complete"
    assert recorded["permission_error_attempts"] == 2
    assert recorded["fresh_read_success"] is True
    assert recorded["cached_metadata_accepted"] is False


def _execute_manifest(tmp_path: Path) -> dict:
    for name in ("case.json", "reference.json", "gate.json", "probe.json"):
        _write(tmp_path / name, {"name": name})
    queue = tmp_path / "nim-rpc-v1"
    queue.mkdir()
    return {
        "case": {"case_id": "case-07", "arm": "COMPOUND", "source_job": "case-job",
                 "reference_job": "reference-job"},
        "collected": {
            "case-job": {"path": str(tmp_path / "case.json")},
            "reference-job": {"path": str(tmp_path / "reference.json")},
        },
        "lab": str(tmp_path / "lab"), "execution_anchor": "a" * 40,
        "probe_protocol": {"path": str(tmp_path / "probe.json")},
        "gate": {"path": str(tmp_path / "gate.json"), "sha256": _sha(tmp_path / "gate.json")},
        "transport_root": str(queue), "execution_plan": list(R.PLAN),
        "fresh_public_view": {"digest": "3" * 64},
        "original_failures": {policy: {"policy": policy} for policy in R.POLICIES},
    }


def test_execute_uses_f3_default_contract_without_executor_injection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {})
    manifest = _execute_manifest(tmp_path)
    protocol = {"component_protocols": {
        "probe_failure_evidence": {"path": "bench/protocols/failure.json", "sha256_lf": "4" * 64}
    }}
    monkeypatch.setattr(R, "_load_manifest", lambda _path: (manifest, protocol))
    monkeypatch.setattr(R, "lease_read_scope", lambda _path: _null_scope())
    monkeypatch.setattr(R.T, "close", lambda *_args: {"status": "closed"})
    monkeypatch.setattr(R.BATCH, "_classification", lambda _case, _result: {"confirmed": False})
    calls: list[dict] = []

    def runner(out: Path, *_args: object, **kwargs: object) -> dict:
        calls.append(dict(kwargs))
        result = {"hold_reason": None, "end_state": "confirmed", "confirmed_subtype": "reward",
                  "evidence_scope": "actual_backend"}
        _write(Path(out) / "result.json", result)
        return result

    result = R.execute(manifest_path, epoch_runner=runner)
    assert result["evidence_scope"] == "cpu_fixture"
    assert len(calls) == 2
    assert all("probe_executor" not in call and "complete_factory" not in call for call in calls)
    assert all(call["max_decisions"] == 3 and call["execution_budget_s"] == 1200 for call in calls)
    assert calls[1]["slot_ids"] == ("case-07.model.1", "case-07.model.2", "case-07.model.3")


class _null_scope:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def test_execute_unknown_or_baseexception_stops_before_second_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("unknown", "base"):
        root = tmp_path / mode
        root.mkdir()
        manifest_path = root / "manifest.json"
        _write(manifest_path, {})
        manifest = _execute_manifest(root)
        protocol = {"component_protocols": {
            "probe_failure_evidence": {"path": "bench/protocols/failure.json", "sha256_lf": "4" * 64}
        }}
        monkeypatch.setattr(R, "_load_manifest", lambda _path, m=manifest, p=protocol: (m, p))
        monkeypatch.setattr(R, "lease_read_scope", lambda _path: _null_scope())
        monkeypatch.setattr(R.T, "close", lambda *_args: {"status": "closed"})
        calls: list[str] = []

        def runner(out: Path, *_args: object, _mode: str = mode,
                   _calls: list[str] = calls, **kwargs: object) -> dict:
            _calls.append(str(kwargs["run_id"]))
            if _mode == "base":
                raise KeyboardInterrupt("gpu busy")
            result = {"hold_reason": "probe_outcome_unknown", "end_state": "HOLD",
                      "evidence_scope": "actual_backend"}
            _write(Path(out) / "result.json", result)
            return result

        expected = KeyboardInterrupt if mode == "base" else RuntimeError
        with pytest.raises(expected):
            R.execute(manifest_path, epoch_runner=runner)
        assert len(calls) == 1
        failure = json.loads((root / "execution/failed.json").read_text(encoding="utf-8"))
        assert failure["completed_rows"] == (0 if mode == "base" else 1)
        assert failure["automatic_retry"] is False


def test_execute_rejects_already_started_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {})
    manifest = _execute_manifest(tmp_path)
    protocol = {"component_protocols": {
        "probe_failure_evidence": {"path": "bench/protocols/failure.json", "sha256_lf": "4" * 64}
    }}
    monkeypatch.setattr(R, "_load_manifest", lambda _path: (manifest, protocol))
    (tmp_path / "execution").mkdir()
    with pytest.raises(RuntimeError, match="resume_not_supported"):
        R.execute(manifest_path)
