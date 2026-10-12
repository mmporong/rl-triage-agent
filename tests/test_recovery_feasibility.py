"""별도 개발 실험의 출처·중복 실행·기존 에이전트 승인 경계 검증."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import run_recovery_feasibility as F  # noqa: E402


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "PRIVATE", tmp_path / "private")
    entry = {"checkpoint_path": str(tmp_path / "model.pt"), "action_scale": 0.25}
    document = {"schema": "recovery_feasibility_request_v1", "bundle_key": "b" * 64,
                "source_commit_sha": "a" * 40, "protocol_sha256": "c" * 64,
                "entry": entry, "model_calls": 0, "agent_entry_or_approval_evidence": False}
    monkeypatch.setattr(F, "request_document", lambda _root: (deepcopy(document), {"bound": True}))
    monkeypatch.setattr(F.H.M, "query_gpu", lambda *_args: {"busy": False, "processes": []})
    return document, tmp_path / "IsaacLab"


def test_uncommitted_source_fails_before_gpu_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "PRIVATE", tmp_path / "private")

    def deny(_isaac):
        raise ValueError("uncommitted source")

    monkeypatch.setattr(F, "verify_source", deny)
    monkeypatch.setattr(F.H.M, "query_gpu", lambda *_: pytest.fail("GPU query before source gate"))
    with pytest.raises(ValueError, match="uncommitted"):
        F.execute("example", tmp_path)
    assert not F.PRIVATE.exists()


def test_busy_gpu_has_no_consumption_or_launch(experiment, monkeypatch):
    _, isaaclab = experiment
    monkeypatch.setattr(F.H.M, "query_gpu", lambda *_args: {"busy": True})
    monkeypatch.setattr(F.H.P, "run_owned", lambda *_args: pytest.fail("GPU launched"))
    assert F.execute("example", isaaclab)["sdk_jobs_launched"] == 0
    assert not F.PRIVATE.exists()


def test_failed_first_arm_preserves_execution_and_blocks_all_retries(experiment, monkeypatch):
    document, isaaclab = experiment
    monkeypatch.setattr(F, "validate_manifest", lambda *_args: {})
    launches = []

    def launch(folder, *_args):
        launches.append(folder.name)
        assert (folder / "effect_consumed.json").is_file()
        assert (F.PRIVATE / "consumed" / f"{document['bundle_key']}.json").is_file()
        F.H.write_new(folder / "launch.json", {"mock": True})
        return {"exit_code": 1, "cleanup_verified": True, "wall_s": 0.1}

    monkeypatch.setattr(F.H.P, "run_owned", launch)
    result = F.execute("example", isaaclab)
    assert result["status"] == "invalid"
    assert launches == ["fault_before"]
    assert result["sdk_jobs_launched"] == 1
    assert result["execution_provenance_verified"] is False
    assert result["agent_entry_or_approval_evidence"] is False
    assert result["judgment_influence_proven"] is False
    assert (F.PRIVATE / "example/arms/fault_before/execution.json").is_file()
    for tag in ("example", "another_tag"):
        with pytest.raises(FileExistsError, match="consumed"):
            F.execute(tag, isaaclab)
    assert launches == ["fault_before"]


def test_feasibility_manifest_cannot_enter_agent_recovery(experiment):
    _, isaaclab = experiment
    path = (F.PRIVATE / "example/arms/fault_before/manifest.json").resolve()
    with pytest.raises(ValueError, match="outside"):
        F.H.validate_manifest(path, isaaclab)


def test_manifest_rejects_callback_or_consumption_tampering(experiment, monkeypatch):
    document, isaaclab = experiment
    folder = F.folder_for("example")
    arm_folder = folder / "arms/fault_before"
    F.H.write_new(folder / "request.json", document)
    consumed = {"bundle_key": document["bundle_key"],
                "request_sha256": F.H.W.file_sha(folder / "request.json"), "execution_limit": 3}
    F.H.write_new(F.PRIVATE / "consumed" / f"{document['bundle_key']}.json", consumed)
    F.H.write_new(folder / "bundle_consumed.json", consumed)
    F.H.write_new(arm_folder / "effect_consumed.json", {**consumed, "arm": "fault_before"})
    manifest = F.make_manifest(folder, document, "fault_before")
    manifest["callback"] = "X1_REF"
    path = arm_folder / "manifest.json"
    F.H.write_new(path, manifest)
    with pytest.raises(ValueError, match="binding changed"):
        F.validate_manifest(path.resolve(), isaaclab)


def test_incomplete_triplet_cannot_be_effect_evidence(experiment):
    _, isaaclab = experiment
    with pytest.raises(FileNotFoundError):
        F.verify_triplet(F.folder_for("missing"), isaaclab)


@pytest.mark.parametrize("artifact", ["execution.json", "launch.json", "stdout.log", "stderr.log",
                                     "gpu_before.json", "gpu_after.json"])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_missing_or_changed_raw_record_rejects_effect_evidence(experiment, monkeypatch, artifact, mutation):
    _, isaaclab = experiment
    folder = F.folder_for("example")
    monkeypatch.setattr(F, "verify_arm", lambda *_: ({}, {"configuration_sha256": "a",
                                                        "observation_configuration_sha256": "b"}))
    for arm in F.H.C.ARMS:
        arm_folder = folder / "arms" / arm
        for name in F.ARM_ARTIFACTS:
            F.H.write_new(arm_folder / name, {"value": name})
        parent = {"status": "pass", "scope": F.FIXED["scope"], "execution": {},
                  "gpu_after": {"value": "gpu_after.json"},
                  "artifact_sha256": {name: F.H.W.file_sha(arm_folder / name) for name in F.ARM_ARTIFACTS}}
        F.H.write_new(arm_folder / "parent_receipt.json", parent)
    path = folder / "arms/fault_before" / artifact
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"{}")
    with pytest.raises((FileNotFoundError, ValueError)):
        F.verify_triplet(folder, isaaclab)


@pytest.mark.parametrize("tag", ["../escape", "X", "", "a/b", "a\\b"])
def test_tag_cannot_escape_private_results(tag):
    with pytest.raises(ValueError, match="tag"):
        F.folder_for(tag)
