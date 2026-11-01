# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_lineage_same_play_inputs_v1 as M


def setup(tmp, monkeypatch):
    config_path = tmp / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    order = [f"source-{i:02d}" for i in range(1, 21)]
    cases = [{"case_id": f"case-{i+1:02d}", "source_job": f"job-{i*2+1}", "reference_job": f"job-{i*2+2}"}
             for i in range(10)]
    metadata = {"cases": cases, "sources": {f"job-{i+1}": {"public_id": source_id} for i, source_id in enumerate(order)}}
    config = {"index": {"path": str(tmp / "index.json")}, "lab": str(tmp), "isaac_python": "not-executed"}
    protocol = {"source_order": order, "smoke_source_id": order[0], "sdk_cap": 19,
                "execution_config_sha256": M.B._sha(config_path), "wall_cap_s": 45000,
                "capture_protocols": {source_id: {"path": source_id} for source_id in order},
                "pair_protocols": {case["case_id"]: {"path": case["case_id"]} for case in cases}}
    captures = {source_id: {"status": "hold_unsupported" if source_id in order[-2:] else "registered_before_local_gpu"}
                for source_id in order}
    monkeypatch.setattr(M, "preflight", lambda *_: (protocol, config, captures))
    monkeypatch.setattr(M.LEASE, "lease_context", lambda: {"owner": M.LEASE.OWNER, "leaseId": "fixture"})
    monkeypatch.setattr(M.INPUT, "checked", lambda *_: ({}, metadata))
    monkeypatch.setattr(M.GPU, "snapshot", lambda *_: {})
    monkeypatch.setattr(M.CAPTURE, "export_artifact", lambda path, _index, _lab, stage, **_: {
        "status": "hold_unsupported" if stage is None else "available", "protocol": str(path),
        "stage": str(stage) if stage is not None else None})
    def pair(_index, _case, _capture, _reference, output, **_kwargs):
        M.B._write_new(output / "result.json", {"fixture": True})
    monkeypatch.setattr(M.PAIR, "prepare_pair", pair)
    monkeypatch.setattr(M.PAIR, "load_pair", lambda *_: {"fixture": True})
    return config_path


def test_capture_sequence_requires_smoke_and_preserves_unsupported_without_replacement(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    observed = []
    monkeypatch.setattr(M.CAPTURE, "execute", lambda *_args, **kwargs: observed.append((_args[4].name, kwargs["mode"])))
    out = tmp_path / "run"
    result = M.execute(tmp_path / "not-live-protocol", "a" * 40, config, out)
    assert observed[0] == ("smoke", "smoke")
    assert len(observed) == 19 and all(mode == "full" for _name, mode in observed[1:])
    assert not any(name in {"source-19", "source-20"} for name, _mode in observed)
    assert result["source_artifacts"] == 20 and len(result["pairs"]) == 10
    assert result["api"] == 0 and result["recovery"] is False
    for source_id in ("source-19", "source-20"):
        held = M.B._read(out / "artifacts" / f"{source_id}.json")
        assert held["status"] == "hold_unsupported" and held["stage"] is None


def test_first_capture_failure_records_reserved_intent_and_never_retries(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    calls = []
    def failed(*args, **kwargs):
        calls.append(kwargs["mode"])
        raise RuntimeError("controlled SDK failure")
    monkeypatch.setattr(M.CAPTURE, "execute", failed)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="SDK failure"):
        M.execute(tmp_path / "not-live-protocol", "a" * 40, config, out)
    assert calls == ["smoke"]
    failure = M.B._read(out / "failure.json")
    assert failure["sdk_intents"] == 1 and failure["automatic_retry"] is False
    assert not (out / "pairs").exists()


def test_changed_lease_stops_before_any_sdk_or_gpu_observation(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    leases = iter([{"owner": M.LEASE.OWNER, "leaseId": "first"}, {"owner": M.LEASE.OWNER, "leaseId": "changed"}])
    monkeypatch.setattr(M.LEASE, "lease_context", lambda: next(leases))
    monkeypatch.setattr(M.GPU, "snapshot", lambda *_: pytest.fail("no GPU read after lease mismatch"))
    monkeypatch.setattr(M.CAPTURE, "execute", lambda *_args, **_kwargs: pytest.fail("no SDK after lease mismatch"))
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="lease boundary"):
        M.execute(tmp_path / "not-live-protocol", "a" * 40, config, out)
    assert M.B._read(out / "failure.json")["sdk_intents"] == 0
