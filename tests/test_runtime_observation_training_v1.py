# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "tests")]

import runtime_observation_training_v1 as C


def _protocol() -> dict:
    return {
        "source_sha256_lf": {name: "a" * 64 for name in C.TRAIN.BOUND_SOURCES},
        "runtime_source_sha256_lf": {f"runtime/{index}.py": "b" * 64 for index in range(9)},
    }


def _manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, arm: str = "COMPOUND",
    mode: str = "train", checkpoint: str | None = None,
) -> dict:
    monkeypatch.setattr(C, "_registered_sources", lambda *_args: None)
    monkeypatch.setattr(C, "_relative", lambda _path: "bench/protocols/runtime.json")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    faults = [] if arm == "NONE" else ["F1", "R1"]
    corrections = []
    if arm == "FIRST":
        corrections = ["restore_body_frame_reward"]
    if arm == "BOTH":
        corrections = ["restore_body_frame_reward", "restore_configured_ppo_epochs"]
    return C.build_manifest(
        protocol, _protocol(), "a" * 40, stage=tmp_path / "stage", seed=2311, arm=arm,
        original_faults=faults, applied_corrections=corrections,
        source_checkpoint_sha256=checkpoint, run_name=f"runtime_{arm.lower()}", mode=mode)


@pytest.mark.parametrize("arm", ["NONE", "COMPOUND"])
def test_source_train_requires_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str,
) -> None:
    value = _manifest(tmp_path, monkeypatch, arm=arm)
    assert value["max_iterations"] == 300 and value["source_checkpoint_sha256"] is None
    with pytest.raises(ValueError, match="source checkpoint"):
        _manifest(tmp_path, monkeypatch, arm=arm, checkpoint="c" * 64)


@pytest.mark.parametrize("arm", ["FIRST", "BOTH"])
def test_correction_train_requires_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str,
) -> None:
    with pytest.raises(ValueError, match="source checkpoint"):
        _manifest(tmp_path, monkeypatch, arm=arm)
    value = _manifest(tmp_path, monkeypatch, arm=arm, checkpoint="c" * 64)
    assert value["source_checkpoint_sha256"] == "c" * 64


def test_smoke_is_none_three_updates_and_requires_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _manifest(tmp_path, monkeypatch, arm="NONE", mode="inspection", checkpoint="c" * 64)
    assert value["max_iterations"] == 3
    with pytest.raises(ValueError, match="inspection smoke"):
        _manifest(tmp_path, monkeypatch, arm="COMPOUND", mode="inspection", checkpoint="c" * 64)


def test_run_training_uses_owned_adapter_and_materializes_both_fact_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _manifest(tmp_path, monkeypatch)
    stage = tmp_path / "actual"
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: list[Path] = []

    def runner(folder: Path, _argv, _cwd, _watchdog):
        assert folder.is_dir()
        observed.append(folder)
        return {"exit_code": 0, "natural_exit": True}

    def execute(target, manifest, outer, protocol, *, runner, wait, watchdog):
        assert protocol == {} and outer["isaac_python"] == str(Path(sys.executable).resolve())
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (target / "training").mkdir()
        (target / "training/training_evidence.json").write_text("{}", encoding="utf-8")
        execution = runner(target / "owned", [], tmp_path, watchdog)
        (target / "owned-result.json").write_text(json.dumps(execution), encoding="utf-8")
        return execution, {"checkpoint": {"path": str(checkpoint), "sha256": C._sha(checkpoint)}}

    def collect(_manifest, _evidence, _owned, *, lab, isaac_python, out):
        out.mkdir(parents=True)
        (out / C.INPUT.COLLECTED).write_text("{}", encoding="utf-8")

    roles: list[str] = []

    def facts(_manifest, _evidence, _owned, *, role, lab, expected_checkpoint_sha256, out):
        roles.append(role)
        out.mkdir(parents=True)
        (out / C.FACT.ARTIFACT_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(C.TRAIN, "verify_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(C.EXEC, "_run_training", execute)
    monkeypatch.setattr(C.INPUT, "collect", collect)
    monkeypatch.setattr(C.FACT, "prepare", facts)
    result = C.run_training(
        stage, fixed, lab=tmp_path, isaac_python=Path(sys.executable), runner=runner,
        wait=lambda *_args, **_kwargs: None)
    assert observed == [stage / "owned"] and roles == ["case", "reference"]
    assert result["collected"]["sha256"] == C._sha(stage / "collected/collected.json")
    assert set(result["runtime_facts"]) == {"case", "reference"}


def test_run_fixed_uses_owned_adapter_and_unique_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []

    def raw_runner(folder: Path, *_args):
        assert folder.is_dir()
        observed.append(folder)
        return {"exit_code": 0}

    def evaluate(stage, row, outer, evidence, *, runner, wait):
        execution = runner(stage / "owned", [], tmp_path, 10)
        return {"name": row["case_id"], "checkpoint": evidence["checkpoint"]}, execution

    monkeypatch.setattr(C.EXEC, "_fixed_evaluate", evaluate)
    report, receipt = C.run_fixed(
        tmp_path / "fixed", {"checkpoint": {"sha256": "c" * 64}}, name="s2311_compound",
        lab=tmp_path, isaac_python=Path(sys.executable), runner=raw_runner,
        wait=lambda *_args, **_kwargs: None)
    assert report["name"] == "s2311_compound" and receipt["exit_code"] == 0
    assert observed == [tmp_path / "fixed/owned"]
