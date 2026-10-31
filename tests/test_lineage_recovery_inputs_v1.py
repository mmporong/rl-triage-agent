from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "tests")]

import lineage_recovery_inputs_v1 as C
import lineage_recovery_training_v1 as T
import test_lineage_recovery_training_v1 as WRAPPER_TEST


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = "BOTH",
    mode: str = "train",
) -> dict[str, Path | dict]:
    manifest_path, lab, evidence_dir, manifest, *_ = WRAPPER_TEST._runtime(
        tmp_path, monkeypatch, arm=arm, mode=mode)
    T.execute(manifest_path, lab, evidence_dir)
    evidence = json.loads((evidence_dir / T.GUARD.EVIDENCE).read_text(encoding="utf-8"))
    log_dir = Path(evidence["log_dir"])
    (log_dir / "params").mkdir()
    (log_dir / "params/agent.yaml").write_text(
        "seed: 235\nmax_iterations: 300\nalgorithm:\n"
        "  num_learning_epochs: 5\n  num_mini_batches: 4\n",
        encoding="utf-8",
    )
    event = log_dir / "events.out.tfevents.fixture"
    event.write_bytes(b"tensorboard\r\nraw\r\n")
    owned = tmp_path / "owned-result.json"
    _write_json(owned, {
        "exit_code": 0, "natural_exit": True, "timed_out": False,
        "error_type": None, "cleanup_verified": True,
        "job_assigned_before_release": True,
    })
    return {"manifest_path": manifest_path, "manifest": manifest, "lab": lab,
            "evidence_dir": evidence_dir, "owned": owned, "event": event,
            "checkpoint": Path(evidence["checkpoint"]["path"])}


def _successful_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["timeout"] == C.INPUT.EXTRACT_TIMEOUT_S
        _write_json(Path(argv[-1]), {"summary": {"reward": {"n": 300}}})
        return SimpleNamespace(returncode=0, stdout="extract ok\r\n", stderr="")

    monkeypatch.setattr(C.INPUT.subprocess, "run", run)


def _collect(
    completed: dict[str, Path | dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict]:
    _successful_extract(monkeypatch)
    out = tmp_path / "collected"
    value = C.collect_recovery(
        completed["manifest_path"], completed["evidence_dir"], completed["owned"],
        out, Path(sys.executable), lab=completed["lab"],
    )
    return out, value


def test_collect_uses_actual_wrapper_contract_and_checked_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    original_event = completed["event"].read_bytes()
    out, value = _collect(completed, tmp_path, monkeypatch)

    checked, telemetry = C.checked_recovery(out / "collected.json")
    assert checked == value and telemetry["summary"]["reward"]["n"] == 300
    assert value["schema"] == "lineage_recovery_collected_v1"
    assert value["training"]["training_kind"] == "cumulative_correction"
    assert value["training"]["remaining_faults"] == []
    assert value["training"]["fresh_initialization"] is True
    assert value["training"]["checkpoint_loaded"] is False
    assert value["training"]["source_checkpoint_role"] == "provenance_metadata_only_not_loaded"
    assert value["scope"]["approval"] is False
    assert value["scope"]["execution_allowed"] is False
    assert Path(value["private_source_map"]["event_copy"]).read_bytes() == original_event
    assert b"\r\n" in original_event
    receipt = json.loads((out / "extract.receipt.json").read_text(encoding="utf-8"))
    assert receipt["cpu_only"] is True and receipt["returncode"] == 0


def test_none_is_healthy_reference_and_not_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch, arm="NONE")
    _, value = _collect(completed, tmp_path, monkeypatch)
    assert value["training"]["training_kind"] == "healthy_reference"
    assert value["training"]["original_faults"] == []
    assert value["training"]["applied_corrections"] == []


def test_inspection_three_is_rejected_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch, mode="inspection")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("extractor must not run")

    monkeypatch.setattr(C.INPUT.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="300 update train"):
        C.collect_recovery(
            completed["manifest_path"], completed["evidence_dir"], completed["owned"],
            tmp_path / "out", Path(sys.executable), lab=completed["lab"])
    assert called is False and not (tmp_path / "out").exists()


@pytest.mark.parametrize(("target", "message"), [
    ("checkpoint", "checkpoint"),
    ("observer_records", "원문"),
    ("training_evidence", "결합"),
])
def test_checked_rejects_checkpoint_evidence_and_observer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, message: str,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    out, value = _collect(completed, tmp_path, monkeypatch)
    path = Path(value["private_source_map"][target])
    if target == "training_evidence":
        body = json.loads(path.read_text(encoding="utf-8"))
        body["unexpected"] = True
        _write_json(path, body)
    elif target == "observer_records":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["unexpected"] = True
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=message):
        C.checked_recovery(out / "collected.json")


@pytest.mark.parametrize(("field", "value"), [
    ("checkpoint_loaded", True),
    ("fresh_initialization", False),
    ("seed", 236),
])
def test_seed_load_and_freshness_mismatch_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    evidence_path = Path(completed["evidence_dir"]) / T.GUARD.EVIDENCE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[field] = value
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="close marker 결합|fresh|checkpoint load|seed"):
        C.collect_recovery(
            completed["manifest_path"], completed["evidence_dir"], completed["owned"],
            tmp_path / "out", Path(sys.executable), lab=completed["lab"])


def test_checked_rejects_swapped_checkpoint_telemetry_and_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    out, _value = _collect(completed, tmp_path, monkeypatch)
    collected_path = out / "collected.json"
    swapped = json.loads(collected_path.read_text(encoding="utf-8"))
    source = swapped["private_source_map"]
    source["checkpoint"], source["telemetry"] = source["telemetry"], source["checkpoint"]
    collected_path.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint 경로|checkpoint bytes"):
        C.checked_recovery(collected_path)


def test_checked_rejects_source_event_and_telemetry_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    out, _value = _collect(completed, tmp_path, monkeypatch)
    collected_path = out / "collected.json"
    swapped = json.loads(collected_path.read_text(encoding="utf-8"))
    source = swapped["private_source_map"]
    source["source_event"], source["telemetry"] = source["telemetry"], source["source_event"]
    collected_path.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(ValueError, match="source_event 경로"):
        C.checked_recovery(collected_path)


def test_extraction_failure_preserves_raw_receipt_and_logs_without_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    out = tmp_path / "failed"

    def fail(*_args, **_kwargs):
        error = subprocess.TimeoutExpired("extract", 120, output=b"partial\r\n", stderr=b"timeout\r\n")
        raise error

    monkeypatch.setattr(C.INPUT.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        C.collect_recovery(
            completed["manifest_path"], completed["evidence_dir"], completed["owned"],
            out, Path(sys.executable), lab=completed["lab"])
    receipt = json.loads((out / "extract.receipt.json").read_text(encoding="utf-8"))
    assert receipt["error_type"] == "TimeoutExpired" and receipt["returncode"] is None
    assert (out / "extract.stdout.txt").read_bytes() == b"partial\r\n"
    assert (out / "extract.stderr.txt").read_bytes() == b"timeout\r\n"
    assert not (out / "collected.json").exists()


def test_owned_result_failure_blocks_before_output_and_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _completed(tmp_path, monkeypatch)
    owned = json.loads(Path(completed["owned"]).read_text(encoding="utf-8"))
    owned["natural_exit"] = False
    _write_json(Path(completed["owned"]), owned)
    monkeypatch.setattr(C.INPUT.subprocess, "run", lambda *_a, **_k: pytest.fail("SDK called"))
    with pytest.raises(ValueError, match="owned"):
        C.collect_recovery(
            completed["manifest_path"], completed["evidence_dir"], completed["owned"],
            tmp_path / "out", Path(sys.executable), lab=completed["lab"])
    assert not (tmp_path / "out").exists()
