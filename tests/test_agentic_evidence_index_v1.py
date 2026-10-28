"""호스트 색인과 모델 공개 경계의 CPU 회귀. 합성 계약 fixture와 실제 공개 모터 자료를 구분한다."""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "tests")]
import agentic_evidence_index_v1 as E
from test_vision_selection_contract_v1 import setup_case


def ref(root: Path, path: Path, name: str = "fixture") -> dict:
    return {"root": name, "path": path.relative_to(root).as_posix(), "sha256": E.V01.sha256(path)}


def write(path: Path, value: dict) -> None:
    path.write_bytes(E.V01.canonical(value))


@pytest.fixture
def case(tmp_path: Path) -> tuple[dict, dict]:
    contract, binding, snapshot = setup_case(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    write(snapshot_path, snapshot)
    body = E.V01.verify(contract)
    entry = {"evidence_id": "evidence-01", "split": "development", "already_seen": True,
             "binding": {"source_lineage_id": binding["source_lineage_id"],
                         "checkpoint_sha256": body["provenance"]["checkpoint"]["sha256"]},
             "contract": ref(tmp_path, contract), "snapshot": ref(tmp_path, snapshot_path),
             "training": None, "same_play_trace": None, "window": None}
    return {"schema": E.SPEC_SCHEMA, "entries": [entry]}, {"fixture": tmp_path}


def test_replay_reuses_v03_messages_and_frame_pair_without_metadata(case: tuple) -> None:
    spec, roots = case
    before = {p: E.V01.sha256(p) for p in roots["fixture"].rglob("*") if p.is_file()}
    index = E.build_index(spec, roots)
    digest = E.index_sha256(index)
    entry = spec["entries"][0]
    contract = E._ref(entry["contract"], roots)
    snapshot = E._read(E._ref(entry["snapshot"], roots))
    binding = {"contract_sha256": entry["contract"]["sha256"],
               "source_lineage_id": entry["binding"]["source_lineage_id"]}
    for kind in sorted(E.SELECT.VIEW_KINDS):
        bundle = E.SELECT.prepare(snapshot, contract, binding, kind)
        model = E.model_view(index, "evidence-01", roots, expected_sha256=digest, view_kind=kind)
        assert model["messages"] == E.SELECT.messages(bundle, snapshot, contract, binding)
        text = E.TRANSFER.MATERIAL._model_text(model["messages"])
        for hidden in (str(roots["fixture"]), "lineage-contract-1", "play-contract-1", "checkpoint_sha256",
                       "healthy", "recovered", "F1", "source_role", "arm", "filename"):
            assert hidden.lower() not in text
    frame = E.frame_view(index, "evidence-01", "f02", roots, expected_sha256=digest)
    raw = E.V01.public_model_view(contract)["messages"][1]["content"]
    assert frame["messages"][0]["content"] == raw[4:6]
    assert frame["independent_confirmation_sources_added"] == 0
    assert index["entries"][0]["behavior_ground_truth"] == "unavailable"
    assert before == {p: E.V01.sha256(p) for p in roots["fixture"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("relative", ["../snapshot.json", "/snapshot.json", "C:/snapshot.json",
                                      "x/../../snapshot.json", "contract\\contract.json",
                                      "contract//contract.json", "contract/./contract.json",
                                      "contract/contract.json:stream"])
def test_path_traversal_is_rejected(case: tuple, relative: str) -> None:
    spec, roots = case
    spec["entries"][0]["contract"]["path"] = relative
    with pytest.raises(ValueError, match="unsafe relative path"):
        E.build_index(spec, roots)


def test_raw_sha_missing_artifact_and_registered_index_tamper(case: tuple) -> None:
    spec, roots = case
    index = E.build_index(spec, roots)
    digest = E.index_sha256(index)
    changed = deepcopy(index)
    changed["entries"][0]["behavior_ground_truth"] = "healthy"
    with pytest.raises(ValueError, match="registered index SHA"):
        E.verify_index(changed, roots, expected_sha256=digest)
    with pytest.raises(ValueError, match="index/current evidence"):
        E.verify_index(changed, roots, expected_sha256=E.index_sha256(changed))
    entry = spec["entries"][0]
    path = E._ref(entry["snapshot"], roots)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw SHA256"):
        E.verify_index(index, roots, expected_sha256=digest)
    entry["snapshot"]["sha256"] = E.V01.sha256(path)
    E.build_index(spec, roots)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        E.build_index(spec, roots)


@pytest.mark.parametrize("field,value", [("source_lineage_id", "other-source"),
                                         ("checkpoint_sha256", "a" * 64)])
def test_mixed_source_checkpoint_rejected(case: tuple, field: str, value: str) -> None:
    spec, roots = case
    spec["entries"][0]["binding"][field] = value
    with pytest.raises(ValueError, match="source/checkpoint mismatch"):
        E.build_index(spec, roots)


def test_snapshot_checkpoint_and_unregistered_training_rejected(case: tuple) -> None:
    spec, roots = case
    entry = spec["entries"][0]
    path = E._ref(entry["snapshot"], roots)
    snapshot = E._read(path)
    snapshot["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "a" * 64
    write(path, snapshot)
    entry["snapshot"] = ref(roots["fixture"], path)
    with pytest.raises(ValueError):
        E.build_index(spec, roots)


def test_training_artifact_checkpoint_telemetry_and_numeric_facts_are_bound(case: tuple) -> None:
    spec, roots = case
    root = roots["fixture"]
    entry = spec["entries"][0]
    snapshot_path = E._ref(entry["snapshot"], roots)
    old = E._read(snapshot_path)
    facts = {"config:action_scale": 0.25}
    bindings, training = {}, {}
    for role in ("case", "reference"):
        summary = {metric: {"mean_last_20pct": 0.5} for metric in E.SELECT.CONTROLLER.V.TELEMETRY_FIELDS}
        telemetry = root / f"{role}-training.json"
        write(telemetry, {"summary": summary})
        checkpoint = root / "contract/artifacts/checkpoint.bin"
        cp_sha = E.V01.sha256(checkpoint)
        artifact = root / f"{role}-artifact.json"
        # 정답·arm 메타데이터는 호스트 artifact에만 있고 모델 입력에는 복사되지 않는다.
        write(artifact, {"arm": "F1", "fixed_eval_label": "healthy", "checkpoint": {"sha256": cp_sha},
                         "telemetry": {"sha256": E.V01.sha256(telemetry), "summary": summary}})
        training[role] = {"artifact": ref(root, artifact), "telemetry": ref(root, telemetry),
                          "checkpoint": ref(root, checkpoint)}
        bindings[role] = {"checkpoint_sha256": cp_sha, "telemetry_sha256": E.V01.sha256(telemetry)}
        facts.update({f"telemetry:{role}:{metric}": 0.5 for metric in summary})
    packet = E.SELECT.CONTROLLER.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                facts=facts, input_bindings=bindings, execution_budget_s=1200)
    snapshot = E.SELECT.CONTROLLER.V.build_snapshot(packet, old["inputs"]["run_id"], "model_adaptive",
                old["inputs"]["protocol_sha256"], root / "training-ledger.jsonl")
    write(snapshot_path, snapshot)
    entry["snapshot"] = ref(root, snapshot_path)
    entry["training"] = training
    index = E.build_index(spec, roots)
    assert len(index["entries"][0]["training"]) == 2
    view = E.model_view(index, "evidence-01", roots, expected_sha256=E.index_sha256(index))
    assert "healthy" not in E.TRANSFER.MATERIAL._model_text(view["messages"])
    entry["training"] = None
    with pytest.raises(ValueError, match="training provenance required"):
        E.build_index(spec, roots)
    entry["training"] = training
    artifact = E._ref(training["case"]["artifact"], roots)
    body = E._read(artifact)
    body["checkpoint"]["sha256"] = "1" * 64
    write(artifact, body)
    training["case"]["artifact"] = ref(root, artifact)
    with pytest.raises(ValueError, match="training checkpoint mismatch"):
        E.build_index(spec, roots)


def test_duplicate_id_seen_evaluation_and_unknown_frame_rejected(case: tuple) -> None:
    spec, roots = case
    index = E.build_index(spec, roots)
    for identity, frame in (("evidence-01", "f99"), ("evidence-99", "f01")):
        with pytest.raises(ValueError, match="unknown"):
            E.frame_view(index, identity, frame, roots, expected_sha256=E.index_sha256(index))
    spec["entries"].append(deepcopy(spec["entries"][0]))
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        E.build_index(spec, roots)
    spec["entries"].pop()
    spec["entries"][0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="invalid split"):
        E.build_index(spec, roots)


@pytest.mark.parametrize("poison", ["healthy", "recovered", "F1", "NONE", "restore_body_frame_reward",
                                    "/home/user/private", "bench/private/secret", "policy.pt"])
def test_model_export_leak_is_rejected(case: tuple, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    spec, roots = case
    index = E.build_index(spec, roots)
    original = E.SELECT.messages

    def unsafe(*args: object) -> list:
        messages = original(*args)
        messages[1]["content"].append({"type": "text", "text": poison})
        return messages

    monkeypatch.setattr(E.SELECT, "messages", unsafe)
    with pytest.raises(ValueError):
        E.model_view(index, "evidence-01", roots, expected_sha256=E.index_sha256(index))


def test_missing_snapshot_keeps_video_only_and_missing_ground_truth(case: tuple) -> None:
    spec, roots = case
    spec["entries"][0]["snapshot"] = None
    index = E.build_index(spec, roots)
    digest = E.index_sha256(index)
    assert index["entries"][0]["available_views"] == ["video_only"]
    view = E.model_view(index, "evidence-01", roots, expected_sha256=digest, view_kind="video_only")
    assert sum(part["type"] == "image_url" for part in view["messages"][1]["content"]) == 6
    with pytest.raises(ValueError, match="snapshot unavailable"):
        E.model_view(index, "evidence-01", roots, expected_sha256=digest)


def test_hardlinked_artifact_and_duplicate_json_rejected(case: tuple) -> None:
    spec, roots = case
    path = E._ref(spec["entries"][0]["snapshot"], roots)
    twin = path.with_name("linked.json")
    os.link(path, twin)
    with pytest.raises(ValueError, match="single regular file"):
        E.build_index(spec, roots)
    twin.unlink()
    path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    spec["entries"][0]["snapshot"]["sha256"] = E.V01.sha256(path)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        E.build_index(spec, roots)


def test_snapshot_cannot_read_an_unregistered_ledger_root(case: tuple) -> None:
    spec, roots = case
    entry = spec["entries"][0]
    path = E._ref(entry["snapshot"], roots)
    snapshot = E._read(path)
    snapshot["inputs"]["ledger_path"] = str(roots["fixture"].parent / "outside.jsonl")
    write(path, snapshot)
    entry["snapshot"] = ref(roots["fixture"], path)
    with pytest.raises(ValueError, match="ledger outside registered roots"):
        E.build_index(spec, roots)


def test_real_public_motor_clips_are_development_with_missing_same_play_truth() -> None:
    path = ROOT / "docs/media/go2-correction-motor_resume_dev_20261013/source_manifest.json"
    inventory = E.inventory_media(ref(ROOT, path, "repo"), {"repo": ROOT})
    assert len(inventory["clips"]) == 3
    assert inventory["split"] == "development" and inventory["already_seen"]
    assert all(not row["model_view_available"] and row["same_play_state"] == "unavailable"
               for row in inventory["clips"])
    assert inventory["fixed_evaluation_is_same_play_ground_truth"] is False


def test_media_sha_mismatch_and_missing_clip_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    video = tmp_path / "play.mp4"
    video.write_bytes(b"cpu-sha-fixture-only")
    manifest = {"schema": "training_correction_media_artifact_v2", "clips": [
        {"id": "clip-1", "role": "host-only", "file": video.name, "bytes": video.stat().st_size,
         "sha256": "0" * 64, "checkpoint_sha256": "1" * 64}]}
    write(path, manifest)
    with pytest.raises(ValueError, match="media raw SHA"):
        E.inventory_media(ref(tmp_path, path), {"fixture": tmp_path})
    video.unlink()
    with pytest.raises(FileNotFoundError):
        E.inventory_media(ref(tmp_path, path), {"fixture": tmp_path})
