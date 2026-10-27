from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import vision_transferred_input_acceptance_v2 as M


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def training_fixture(tmp_path: Path, *, value: float = 0.2) -> dict:
    """검증 경계만 시험하는 작은 단위 테스트 원문이다."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(str(value).encode())
    telemetry = {"summary": {name: {"n": 300, "mean_last_20pct": value}
                              for name in M.SELECT.CONTROLLER.V.TELEMETRY_FIELDS},
                 "series": {name: [value] * 300 for name in M.SELECT.CONTROLLER.V.TELEMETRY_FIELDS}}
    write(tmp_path / "telemetry.json", telemetry)
    manifest = {"arm": "NONE", "seed": 1, "max_iterations": 300,
                "source_commit_sha": "a" * 40, "protocol_sha256_lf": "b" * 64}
    readback = {"schema": "unit_test_readback"}
    write(tmp_path / "training.manifest.json", manifest)
    write(tmp_path / "training.readback.json", readback)
    artifact = {"arm": "NONE", "seed": 1, "checkpoint": {"path": "remote/model", "sha256": M.V01.sha256(checkpoint)},
                "telemetry": {"path": "remote/telemetry", "sha256": M.V01.sha256(tmp_path / "telemetry.json"),
                              "summary": telemetry["summary"]}, "manifest": manifest, "readback": readback,
                "manifest_sha256": M.V01.sha256(tmp_path / "training.manifest.json")}
    worker = {**artifact, "completed": True, "iterations": 300,
              "readback_sha256": M.V01.sha256(tmp_path / "training.readback.json")}
    write(tmp_path / "worker.json", worker)
    write(tmp_path / "execution.json", {"completed": True, "cleanup_verified": True, "exit_code": 0})
    write(tmp_path / "adapter_artifact.json", artifact)
    raw = tmp_path / "raw.json"
    write(raw, {"kind": "train", "job_id": "unit-test", "artifact": artifact})
    return {"stage": tmp_path, "raw_action": raw, "checkpoint": checkpoint,
            "public": {"job_id": "unit-test", "arm": "NONE", "seed": 1, "iterations": 300,
                       "checkpoint_sha256": M.V01.sha256(checkpoint), "raw_sha256": M.V01.sha256(raw)},
            "summary": {"source_commit": "a" * 40, "source_protocol": {"sha256_lf": "b" * 64}}}


def test_same_training_source_is_repeatable_and_read_only(tmp_path: Path) -> None:
    kwargs = training_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    first = M.verify_training(**kwargs)
    assert M.verify_training(**kwargs) == first
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert first["telemetry_sha256"] == M.V01.sha256(tmp_path / "telemetry.json")


@pytest.mark.parametrize("target", ["checkpoint.bin", "telemetry.json", "training.readback.json", "raw.json"])
def test_changed_training_sources_are_rejected(tmp_path: Path, target: str) -> None:
    kwargs = training_fixture(tmp_path)
    path = tmp_path / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        M.verify_training(**kwargs)


def test_telemetry_from_another_policy_is_rejected(tmp_path: Path) -> None:
    case = training_fixture(tmp_path / "case", value=0.2)
    other = training_fixture(tmp_path / "other", value=0.4)
    (case["stage"] / "telemetry.json").write_bytes((other["stage"] / "telemetry.json").read_bytes())
    with pytest.raises(ValueError, match="telemetry"):
        M.verify_training(**case)


def test_source_change_during_read_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = training_fixture(tmp_path)
    read = M.V01.read_object

    def change_after_read(path: Path) -> dict:
        value = read(path)
        if path.name == "execution.json":
            path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(M.V01, "read_object", change_after_read)
    with pytest.raises(ValueError, match="재조회"):
        M.verify_training(**kwargs)


def test_training_snapshot_uses_file_sha_and_registered_summary_facts(tmp_path: Path) -> None:
    case = M.verify_training(**training_fixture(tmp_path / "case", value=0.2))
    baseline = M.verify_training(**training_fixture(tmp_path / "reference", value=0.4))
    trace = {"checkpoint_sha256": case["checkpoint_sha256"], "profile": {"action_scale": 0.25}}
    old = M.BASE.snapshot(trace, trace, tmp_path / "old-ledger.jsonl")
    snapshot = M.training_snapshot(case, baseline, old, tmp_path / "new-ledger.jsonl")
    packet = snapshot["inputs"]["packet"]
    assert packet["input_bindings"]["case"]["telemetry_sha256"] == case["telemetry_sha256"]
    assert packet["input_bindings"]["reference"]["telemetry_sha256"] == baseline["telemetry_sha256"]
    assert len(packet["facts"]) == 49
    assert packet["facts"]["telemetry:case:Train/mean_reward"] == 0.2
    assert packet["facts"]["telemetry:reference:Train/mean_reward"] == 0.4
    assert packet["budget_left"] == old["inputs"]["packet"]["budget_left"]
    assert packet["execution_budget_s"] == old["inputs"]["packet"]["execution_budget_s"]
    assert snapshot["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["checkpoint", "lineage", "split"])
def test_cross_training_and_video_bindings_are_rejected(mutation: str) -> None:
    contract = {"provenance": {"checkpoint": {"sha256": "a" * 64}, "source_lineage_id": "same-lineage"}}
    window = {"source_lineage_id": "same-lineage", "split": "development", "contract": {"sha256": "b" * 64}}
    training = {"checkpoint_sha256": "a" * 64}
    if mutation == "checkpoint":
        training["checkpoint_sha256"] = "c" * 64
    elif mutation == "lineage":
        contract["provenance"]["source_lineage_id"] = "other-lineage"
    else:
        window["split"] = "holdout"
    with pytest.raises(ValueError):
        M.contract_binding(contract, window, training)


@pytest.mark.parametrize("payload", [
    {"seed": 2220}, {"fault": "F1"}, {"run_name": "private-run"}, {"answer_key": "secret"},
    {"source_role": "normal"}, {"label": "walking"}, {"source_lineage_id": "private-lineage"},
])
def test_labels_and_source_metadata_are_rejected(payload: dict) -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]}]
    with pytest.raises(ValueError):
        M.check_messages(messages, [])


def test_output_outside_owned_tree_is_rejected_before_source_reads(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        M.prepare(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "other", owned_root=tmp_path / "owned")
