"""완료 source→공통 selection view CPU 어댑터 회귀."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import lineage_selection_inputs_v1 as M
import selection_comparison_controller_v1 as C


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _manifest(job: str, arm: str, seed: int) -> dict:
    hashes = {name: "1" * 64 for name in M.CHILD.SOURCE_FILES}
    return M.CHILD.validate_manifest({
        "schema": M.CHILD.SCHEMA,
        "purpose": "source",
        "job_id": job,
        "case_id": "case-01",
        "arm": arm,
        "seed": seed,
        "run_name": f"lineage_{job}",
        "task": M.CHILD.TASK,
        "num_envs": 4096,
        "max_iterations": 300,
        "source_commit": "2" * 40,
        "protocol_relative": "bench/protocols/fixture.json",
        "protocol_sha256_lf": "3" * 64,
        "source_sha256_lf": hashes,
        "runtime_source_sha256_lf": {"runtime.py": "4" * 64},
    })


def _telemetry(base: float = 1.0, *, extra: bool = False) -> dict:
    summary = {
        field: {"n": index + 1, "first": 0.0, "last": base + index,
                "min": 0.0, "max": base + index, "mean_first_20pct": 0.0,
                "mean_last_20pct": base + index}
        for index, field in enumerate(M.V.TELEMETRY_FIELDS)
    }
    if extra:
        summary["secret/F3/path/C:/private"] = {"n": 300, "mean_last_20pct": 999.0}
    return {"run_dir_name": "private_fault_seed", "summary": summary, "series": {}}


def _collect(tmp_path: Path, name: str, arm: str, seed: int, base: float = 1.0,
             run_observer=None) -> Path:
    stage = tmp_path / f"stage-{name}"
    out = tmp_path / f"collected-{name}"
    log_dir = tmp_path / f"logs-{name}"
    checkpoint = log_dir / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{name}".encode())
    event = log_dir / "events.out.tfevents.fixture"
    event.write_bytes(f"event-{name}".encode())
    manifest = _manifest(name, arm, seed)
    _write(stage / "manifest.json", manifest)
    _write(stage / "owned-result.json", {"fixture": True})
    evidence = {
        "log_dir": str(log_dir), "arm": arm, "seed": seed,
        "checkpoint": {"path": str(checkpoint), "sha256": M._sha(checkpoint), "log_dir": str(log_dir)},
    }
    _write(stage / "training/training_evidence.json", evidence)
    accepted = {"job_id": name, "purpose": "source", "arm": arm, "seed": seed,
                "checkpoint": evidence["checkpoint"],
                "evidence_sha256": M._sha(stage / "training/training_evidence.json")}
    _write(stage / "accepted.json", accepted)
    python = tmp_path / "isaac-python.exe"
    python.touch(exist_ok=True)

    def fake_run(argv, **kwargs):
        if run_observer is not None:
            run_observer(kwargs)
        _write(Path(argv[-1]), _telemetry(base, extra=True))
        return subprocess.CompletedProcess(argv, 0, "wrote telemetry\n", "")

    with patch.object(M.B, "completed", return_value=evidence), patch.object(M.subprocess, "run", fake_run):
        result = M.collect_source(stage, out, python)
    assert "training" in result["scope"]["excluded"]
    assert event.read_bytes() == f"event-{name}".encode()
    return out / "collected.json"


def _packet(tmp_path: Path) -> dict:
    case = _collect(tmp_path, "f3-s2301", "F3", 2301, 10.0)
    reference = _collect(tmp_path, "none-s2301", "NONE", 2301, 1.0)
    return M.make_selection_packet(case, reference)


def test_fake_collector_packet_whitelist_and_private_coverage(tmp_path):
    observed = {}
    case = _collect(tmp_path, "f3-s2301", "F3", 2301, 10.0, observed.update)
    reference = _collect(tmp_path, "none-s2301", "NONE", 2301, 1.0)
    result = M.make_selection_packet(case, reference)
    packet = result["packet"]
    field = M.V.TELEMETRY_FIELDS[0]
    assert packet["facts"][f"telemetry:case:{field}"] == 10.0
    assert packet["facts"][f"telemetry:reference:{field}"] == 1.0
    public = json.dumps(packet, ensure_ascii=False)
    for secret in ("f3-s2301", "none-s2301", "private_fault_seed", "secret/F3/path", "2301"):
        assert secret not in public
    assert "config:action_scale" not in packet["facts"]
    assert observed["timeout"] == 120
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["env"]["PYTHONUTF8"] == observed["env"]["PYTHONUNBUFFERED"] == "1"
    assert observed["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    coverage = result["private_metadata"]["case"]["coverage"][field]
    assert coverage == {"n": 1, "observed_scalar_index_max": 0}
    assert "not a 300-step claim" in result["private_metadata"]["coverage_semantics"]
    assert "approval" in result["scope"]["excluded"]


def test_extract_timeout_is_sealed_without_retry(tmp_path):
    calls = 0

    def timeout(_argv, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["timeout"] == M.EXTRACT_TIMEOUT_S
        raise subprocess.TimeoutExpired("extract", kwargs["timeout"], output="partial out", stderr="partial err")

    stage = tmp_path / "timeout-stage"
    log_dir = tmp_path / "timeout-log"
    log_dir.mkdir()
    checkpoint = log_dir / "model.pt"
    checkpoint.write_bytes(b"cp")
    (log_dir / "events.out.tfevents.fixture").write_bytes(b"event")
    manifest = _manifest("timeout", "NONE", 2304)
    evidence = {"log_dir": str(log_dir), "arm": "NONE", "seed": 2304,
                "checkpoint": {"path": str(checkpoint), "sha256": M._sha(checkpoint)}}
    _write(stage / "manifest.json", manifest)
    _write(stage / "owned-result.json", {})
    _write(stage / "training/training_evidence.json", evidence)
    _write(stage / "accepted.json", {"job_id": "timeout", "purpose": "source", "arm": "NONE",
                                      "seed": 2304, "checkpoint": evidence["checkpoint"],
                                      "evidence_sha256": M._sha(stage / "training/training_evidence.json")})
    python = tmp_path / "python.exe"
    python.touch()
    out = tmp_path / "timeout-out"
    with patch.object(M.B, "completed", return_value=evidence), patch.object(M.subprocess, "run", timeout), \
            pytest.raises(subprocess.TimeoutExpired):
        M.collect_source(stage, out, python)
    receipt = M._json(out / "extract.receipt.json")
    assert calls == 1
    assert receipt["cpu_only"] is True and receipt["timeout_s"] == 120
    assert "env" not in receipt and "environment" not in receipt
    assert receipt["returncode"] is None and receipt["error_type"] == "TimeoutExpired"
    assert (out / "extract.stdout.txt").read_text(encoding="utf-8") == "partial out"
    assert (out / "extract.stderr.txt").read_text(encoding="utf-8") == "partial err"
    assert not (out / "collected.json").exists()


def test_tamper_ambiguous_events_and_bad_registered_types_fail_closed(tmp_path):
    case = _collect(tmp_path, "case", "F1", 2302)
    reference = _collect(tmp_path, "reference", "NONE", 2302)
    telemetry = Path(M._json(case)["private_source_map"]["telemetry"])
    value = M._json(telemetry)
    value["summary"][M.V.TELEMETRY_FIELDS[0]]["mean_last_20pct"] = math.nan
    telemetry.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry_sha256"):
        M.make_selection_packet(case, reference)

    stage = tmp_path / "ambiguous"
    log_dir = tmp_path / "ambiguous-log"
    log_dir.mkdir()
    for index in range(2):
        (log_dir / f"events.out.tfevents.{index}").write_bytes(b"event")
    manifest = _manifest("ambiguous", "NONE", 2303)
    checkpoint = log_dir / "model.pt"
    checkpoint.write_bytes(b"cp")
    evidence = {"log_dir": str(log_dir), "arm": "NONE", "seed": 2303,
                "checkpoint": {"path": str(checkpoint), "sha256": M._sha(checkpoint)}}
    _write(stage / "manifest.json", manifest)
    _write(stage / "owned-result.json", {})
    _write(stage / "training/training_evidence.json", evidence)
    _write(stage / "accepted.json", {"job_id": "ambiguous", "purpose": "source", "arm": "NONE",
                                      "seed": 2303, "checkpoint": evidence["checkpoint"],
                                      "evidence_sha256": M._sha(stage / "training/training_evidence.json")})
    python = tmp_path / "python.exe"
    python.touch()
    with patch.object(M.B, "completed", return_value=evidence), pytest.raises(ValueError, match="정확히 하나"):
        M.collect_source(stage, tmp_path / "ambiguous-out", python)


def test_old_telemetry_unique_numeric_values_and_bad_type_projection():
    path = ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json"
    telemetry = M._json(path)
    facts, coverage = M._telemetry_projection("case", telemetry)
    assert facts["telemetry:case:Train/mean_reward"] == pytest.approx(33.46911201477051)
    assert coverage["Train/mean_reward"] == {"n": 300, "observed_scalar_index_max": 299}
    broken = deepcopy(telemetry)
    broken["summary"]["Train/mean_reward"]["mean_last_20pct"] = "33.4"
    with pytest.raises(ValueError, match="유한 수치"):
        M._telemetry_projection("case", broken)


def test_three_snapshots_share_view_rules_and_model_fixture_reach_request_spec(tmp_path):
    envelope = _packet(tmp_path)
    result = M.build_snapshots(envelope, "lineage_fixture", tmp_path / "snapshots")
    snapshots = {policy: M._json(Path(path)) for policy, path in result["snapshots"].items()}
    assert len({snapshot["view_digest"] for snapshot in snapshots.values()}) == 1
    assert M.V.messages(snapshots["registry_next"]) == M.V.messages(snapshots["model_adaptive"])
    for policy in ("registry_next", "fixed_eligible"):
        choice = C.choose(snapshots[policy], policy)
        spec = C.request_spec(snapshots[policy], choice)
        assert choice["probe"] in snapshots[policy]["view"]["eligible_probes"]
        assert spec["approval_granted"] is False
        assert not Path(snapshots[policy]["inputs"]["ledger_path"]).exists()

    adaptive = snapshots["model_adaptive"]
    view = C.dialogue_input(adaptive)["view"]
    probe = view["eligible_probes"][0]
    fact_id = next(iter(view["evidence_facts"]))
    response = {"provider_model": C.CALL.MODEL, "finish_reason": "stop",
                "content": json.dumps({"action": "continue", "probe": probe, "fact_ids": [fact_id]}),
                "input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "wall_s": 0.01}
    dialogue = C.replay_dialogue(adaptive, "production", [response])
    choice = C.choose(adaptive, "model_adaptive", dialogue)
    spec = C.request_spec(adaptive, choice)
    assert choice["selected_by"] == "model" and spec["probe"] == probe
    assert spec["approval_granted"] is False and adaptive["execution_authorized"] is False
