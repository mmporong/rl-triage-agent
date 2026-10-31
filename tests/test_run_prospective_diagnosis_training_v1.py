# pyright: reportMissingImports=false
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import run_prospective_diagnosis_training_v1 as C


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _protocol(categories: list[str] | None = None) -> dict:
    return C.TRAIN.prepare_protocol(
        [2401, 2402], categories or ["NONE", "R1_ONLY"],
        {f"runtime/{index}.py": "a" * 64 for index in range(9)})


def test_prepare_protocol_cli_is_utf8_safe_under_non_utf8_parent(tmp_path: Path) -> None:
    seeds, categories, runtime, out = (tmp_path / name for name in
        ("seeds.json", "categories.json", "runtime.json", "protocol.json"))
    _write(seeds, [2401, 2402])
    _write(categories, ["NONE", "R1_ONLY"])
    _write(runtime, {f"runtime/{index}.py": "a" * 64 for index in range(9)})
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "0"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "bench/run_prospective_diagnosis_training_v1.py"),
         "prepare-protocol", "--seeds", str(seeds), "--categories", str(categories),
         "--runtime-pins", str(runtime), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="strict", env=environment,
        check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == C.TRAIN.SCHEMA


def test_preflight_consumes_every_registered_manifest_and_seed_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    isaac = tmp_path / "python.bat"
    isaac.write_text("", encoding="utf-8")
    monkeypatch.setattr(C.TRAIN, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(C.SEEDS, "assert_fresh_seeds",
                        lambda seeds, _lab: {"status": "fresh", "seeds": seeds})
    monkeypatch.setattr(C.TRAIN, "_relative", lambda _path: "bench/protocols/prospective.json")
    result = C.preflight(protocol_path, "a" * 40, tmp_path / "lab", isaac,
                         tmp_path / "preflight")
    assert result["status"] == "ready" and len(result["planned"]) == 4
    r1 = [row["manifest"] for row in result["planned"] if row["category"] == "R1_ONLY"]
    assert len(r1) == 2 and all(row["arm"] == "R1" and row["original_faults"] == ["R1"]
                                and row["applied_corrections"] == []
                                and row["source_checkpoint_sha256"] is None for row in r1)
    assert all(C.TRAIN.TRAIN.validate_manifest(row["manifest"]) for row in result["planned"])


def _install_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     *, fail_label: str | None = None) -> SimpleNamespace:
    protocol = _protocol()
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    isaac = tmp_path / "python.bat"
    isaac.write_text("", encoding="utf-8")
    calls = SimpleNamespace(smoke=[], cases=[], lease=0, fresh=0)
    monkeypatch.setattr(C.TRAIN, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(C.SEEDS, "assert_fresh_seeds",
                        lambda *_args: setattr(calls, "fresh", calls.fresh + 1) or {"fresh": True})
    monkeypatch.setattr(C.LEASE, "lease_context",
                        lambda *_args: setattr(calls, "lease", calls.lease + 1)
                        or {"owner": C.OWNER, "lease_id": "lease-1"})
    monkeypatch.setattr(C.TRAIN, "build_manifest", lambda _p, _v, _a, **kwargs: {
        "seed": kwargs["seed"], "category": kwargs["category"],
        "arm": C.TRAIN.CATEGORIES[kwargs["category"]]["arm"],
        "original_faults": C.TRAIN.CATEGORIES[kwargs["category"]]["original_faults"],
        "applied_corrections": [], "source_checkpoint_sha256": kwargs.get("source_checkpoint_sha256")})

    def smoke(stage: Path, fixed: dict, **_kwargs):
        calls.smoke.append((Path(stage), fixed))
        return {"checkpoint": {"sha256": "s" * 64}, "mode": "inspection"}

    def case(stage: Path, fixed: dict, *, category: str, **_kwargs):
        label = Path(stage).name
        calls.cases.append(label)
        if fail_label == label:
            raise RuntimeError("registered failure")
        Path(stage).mkdir(parents=True)
        result = {"checkpoint": {"sha256": (label.encode().hex() + "0" * 64)[:64]}}
        _write(Path(stage) / "result.json", result)
        return result

    monkeypatch.setattr(C.TRAIN.RUNTIME, "run_training", smoke)
    monkeypatch.setattr(C.TRAIN, "run_case", case)
    def collect(training_path: Path, *, category: str, out: Path):
        assert Path(training_path).name == "training-result.json"
        Path(out).mkdir(parents=True)
        value = {"source_binding": {"checkpoint_sha256": "c" * 64,
                                    "telemetry_sha256": "d" * 64}}
        _write(Path(out) / C.INPUT.COLLECTED, value)
        return value
    monkeypatch.setattr(C.INPUT, "collect", collect)
    return SimpleNamespace(protocol=protocol_path, isaac=isaac, calls=calls)


def test_execute_runs_smoke_then_all_cases_once_under_same_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _install_execute(tmp_path, monkeypatch)
    result = C.execute(setup.protocol, "a" * 40, tmp_path / "lab", setup.isaac,
                       tmp_path / "out", "b" * 64,
                       runner=lambda *_args: None, wait=lambda *_args: None)
    assert result["status"] == "complete"
    assert len(setup.calls.smoke) == 1
    assert setup.calls.cases == [
        "s2401-none", "s2401-r1_only", "s2402-none", "s2402-r1_only"]
    assert setup.calls.fresh == 1 and setup.calls.lease >= 6
    assert len(result["stages"]) == 5 and result["automatic_retry"] is False
    assert all("input" in row for row in result["stages"][1:])


def test_execute_fail_stops_without_retrying_later_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _install_execute(tmp_path, monkeypatch, fail_label="s2401-r1_only")
    with pytest.raises(RuntimeError, match="registered failure"):
        C.execute(setup.protocol, "a" * 40, tmp_path / "lab", setup.isaac,
                  tmp_path / "out", "b" * 64,
                  runner=lambda *_args: None, wait=lambda *_args: None)
    assert setup.calls.cases == ["s2401-none", "s2401-r1_only"]
    failure = C._read(tmp_path / "out/failure.json")
    assert failure["status"] == "invalid" and failure["automatic_retry"] is False


def test_execute_rejects_missing_checkpoint_and_wrong_lease_before_gpu_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _install_execute(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="checkpoint SHA"):
        C.execute(setup.protocol, "a" * 40, tmp_path / "lab", setup.isaac,
                  tmp_path / "missing", "bad",
                  runner=lambda *_args: None, wait=lambda *_args: None)
    assert setup.calls.smoke == [] and setup.calls.cases == []
    monkeypatch.setattr(C.LEASE, "lease_context",
                        lambda *_args: {"owner": "other", "lease_id": "wrong"})
    with pytest.raises(ValueError, match="Use-GpuLease"):
        C.execute(setup.protocol, "a" * 40, tmp_path / "lab", setup.isaac,
                  tmp_path / "wrong-lease", "b" * 64,
                  runner=lambda *_args: None, wait=lambda *_args: None)
    assert setup.calls.smoke == [] and setup.calls.cases == []


def test_preflight_rejects_job_plan_above_registered_batch_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = C.TRAIN.prepare_protocol(
        [2401, 2402, 2403], ["NONE", "F1", "R1_ONLY", "COMPOUND"],
        {f"runtime/{index}.py": "a" * 64 for index in range(9)})
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    isaac = tmp_path / "python.bat"
    isaac.write_text("", encoding="utf-8")
    monkeypatch.setattr(C.TRAIN, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(C.SEEDS, "assert_fresh_seeds", lambda *_args: {"fresh": True})
    monkeypatch.setattr(C.TRAIN, "_relative", lambda _path: "bench/protocols/prospective.json")
    with pytest.raises(ValueError, match="GPU job 수"):
        C.preflight(protocol_path, "a" * 40, tmp_path / "lab", isaac,
                    tmp_path / "preflight")
    assert not (tmp_path / "preflight").exists()
