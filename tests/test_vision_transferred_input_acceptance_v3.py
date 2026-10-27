from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import vision_transferred_input_acceptance_v3 as M


def _policy(seed: int, suffix: str) -> dict:
    logical = f"source-s{seed}-{suffix}"
    return {"logical_id": logical, "payloads": [
        {"window": window, "view": view}
        for window in (1, 2) for view in M.VIEWS
    ]}


def _pair(seed: int) -> dict:
    return {"seed": seed, "protocol": {}, "capture_receipt": {}, "training_receipt": {},
            "mechanical_verification": {"source_refs": True, "capture_artifacts": True,
                "raw_rgb": True, "actual_pts": True, "training_archive": True,
                "telemetry": True},
            "policies": [_policy(seed, "none"), _policy(seed, "f1")]}


def test_protocol_adapter_uses_each_real_catalog_without_seed_alias() -> None:
    protocol_2220, rows_2220 = M._protocol(
        ROOT / "bench/protocols/vision_transferred_pull_record_20261027b_v3.json", 2220)
    protocol_2221, rows_2221 = M._protocol(
        ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json", 2221)
    assert set(rows_2220) == {"source-s2220-none", "source-s2220-f1"}
    assert set(rows_2221) == {"source-s2221-none", "source-s2221-f1"}
    assert protocol_2220["protocol_id"] != protocol_2221["protocol_id"]


def test_four_policies_make_24_payload_attestation_without_labels(tmp_path: Path) -> None:
    result = M.attest([_pair(2220), _pair(2221)], tmp_path / "receipt.json")
    assert result["policy_count"] == 4
    assert result["payload_count"] == 24
    assert result["complete_four_policy_set"] is True
    assert all(result["mechanical_verification"].values())
    scope = result["use_scope_attestation"]
    assert scope["scopes"] == list(M.USE_SCOPE)
    assert scope["behavior_labels_created"] is False
    assert scope["behavior_ground_truth_assigned"] is False
    assert scope["model_use_authorized"] is False
    assert scope["used_for_decision"] is False
    assert scope["final_approval"] == "leader_pending"


def test_duplicate_seed_pair_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="중복 seed"):
        M.attest([_pair(2220), _pair(2220)], tmp_path / "receipt.json")


def test_capture_copy_gate_uses_seed2221_catalog_and_manifest_root(tmp_path: Path) -> None:
    state = tmp_path / "state-c"
    state.mkdir()
    files = []
    for name in ("policies/source-s2221-none/job.json", "policies/source-s2221-f1/job.json"):
        path = state / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": M.V01.sha256(path)})
    manifest_path = tmp_path / "capture-manifest.json"
    manifest_path.write_text(json.dumps({"root_basename": state.name, "files": files}), encoding="utf-8")
    gate_path = tmp_path / "copy-verification.json"
    gate_path.write_text(json.dumps({"all_match": True, "files_verified": 2,
        "manifest_sha256": M.V01.sha256(manifest_path), "policies": [
            {"logical_id": f"source-s2221-{role}", "raw_receipts": 625, "npz_frames": 625,
             "completed": True, "cleanup_verified": True, "capture_approved": False,
             "model_ready": False} for role in ("none", "f1")]}), encoding="utf-8")
    _protocol, catalog = M._protocol(
        ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json", 2221)
    result = M._capture_copy_gate(gate_path, manifest_path, state.resolve(), catalog)
    assert result["files_verified"] == 2
    assert result["all_match"] is True


def test_archive_material_discovers_seed2221_checkpoint_and_training_root(tmp_path: Path) -> None:
    _protocol_body, catalog = M._protocol(
        ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json", 2221)
    archive = tmp_path / "archive"
    tree = archive / "tree"
    rows = []
    for logical, checkpoint in catalog.items():
        path = tree / "source" / checkpoint["isaaclab_relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(logical.encode())
        checkpoint["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": path.relative_to(tree).as_posix(), "bytes": path.stat().st_size,
                     "sha256": checkpoint["sha256"]})
    source = tree / "source/runtime"
    source.mkdir(parents=True)
    summary = {"source_commit": "a" * 40, "source_protocol": {"sha256_lf": "b" * 64}}
    (source / "manifest.json").write_text(json.dumps({"source_anchor": summary["source_commit"],
        "protocol_sha256_lf": summary["source_protocol"]["sha256_lf"]}), encoding="utf-8")
    (source / "training_plan.json").write_text(json.dumps({"rows": [
        {"job_id": f"vision-s2221-{role}"} for role in ("none", "f1")]}), encoding="utf-8")
    (source / "training.ledger.jsonl").write_text("{}\n", encoding="utf-8")
    for name in ("manifest.json", "training_plan.json", "training.ledger.jsonl"):
        path = source / name
        rows.append({"path": path.relative_to(tree).as_posix(), "bytes": path.stat().st_size,
                     "sha256": M.V01.sha256(path)})
    archive.mkdir(exist_ok=True)
    (archive / "inventory.json").write_text(json.dumps({"files": rows}), encoding="utf-8")
    found_root, checkpoints = M._archive_material(archive, summary, catalog)
    assert found_root == source.resolve()
    assert set(checkpoints) == {"source-s2221-none", "source-s2221-f1"}


def test_prepare_pair_routes_seed2221_protocol_through_capture_training_and_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json"
    archive, state, output = tmp_path / "archive", tmp_path / "state", tmp_path / "output"
    archive.mkdir(); state.mkdir()
    for name in ("summary.json", "copy.json", "capture-manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(M, "_capture_copy_gate", lambda *args: {"all_match": True})
    monkeypatch.setattr(M.V2, "verify_archive", lambda *args: {"all_match": True, "files_verified": 437})
    monkeypatch.setattr(M, "_archive_material", lambda _archive, _summary, catalog: (
        tmp_path / "source", {logical: tmp_path / f"{logical}.pt" for logical in catalog}))
    def capture(_state, _protocol, catalog, _checkpoints, _transfer, target, *, seed):
        target.mkdir()
        (target / "receipt.json").write_text("{}", encoding="utf-8")
        calls.append(("capture", seed, tuple(catalog)))
        return {}
    def training(_source, _archive, _summary, _capture, target, *, owned_root):
        target.mkdir()
        (target / "receipt.json").write_text("{}", encoding="utf-8")
        calls.append(("training", owned_root))
        return {}
    monkeypatch.setattr(M, "_prepare_capture", capture)
    monkeypatch.setattr(M.V2, "prepare", training)
    monkeypatch.setattr(M, "accept_pair", lambda _protocol, _capture, _training, **kwargs: {
        "seed": kwargs["seed"], "policies": [_policy(2221, "none"), _policy(2221, "f1")]})
    result = M.prepare_pair(archive, state, protocol, tmp_path / "summary.json", tmp_path / "copy.json",
                            tmp_path / "capture-manifest.json", output, seed=2221, owned_root=tmp_path)
    assert result["policy_count"] == 2 and result["payload_count"] == 12
    assert calls[0] == ("capture", 2221, ("source-s2221-none", "source-s2221-f1"))
    assert calls[1][0] == "training"


def test_generic_pair_uses_protocol_catalog_and_437_file_training_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json"
    capture_path = tmp_path / "capture.json"
    training_path = tmp_path / "training.json"
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}", encoding="utf-8")
    ref = {"path": str(dummy), "sha256": M.V01.sha256(dummy)}
    capture = {"cpu_preparation": "PASS", "source_seed": 2221,
               "policies": [{"logical_id": "source-s2221-none"}, {"logical_id": "source-s2221-f1"}]}
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    training = {"cpu_preparation": "PASS", "behavior_labels_created": False,
                "model_use_authorized": False, "used_for_decision": False,
                "source_inputs": {"capture": {"path": str(capture_path),
                                                  "sha256": M.V01.sha256(capture_path)}},
                "archive_verification": {"all_match": True, "files_verified": 437,
                                           "transfer": ref, "inventory": ref},
                "shared_training_sources": {"manifest.json": ref},
                "policies": [{"logical_id": "source-s2221-none"},
                             {"logical_id": "source-s2221-f1"}]}
    training_path.write_text(json.dumps(training), encoding="utf-8")

    monkeypatch.setattr(M, "_capture_policy", lambda row, checkpoint, _profile, **_kwargs: {
        "logical_id": row["logical_id"], "checkpoint_sha256": checkpoint["sha256"]})
    monkeypatch.setattr(M, "_training_policy", lambda row, capture_row: {
        "logical_id": row["logical_id"], "checkpoint_sha256": capture_row["checkpoint_sha256"],
        "payloads": [{"window": window, "view": view} for window in (1, 2) for view in M.VIEWS]})
    result = M.accept_pair(protocol, capture_path, training_path, seed=2221, verify_raw_rgb=False)
    assert [row["logical_id"] for row in result["policies"]] == [
        "source-s2221-none", "source-s2221-f1"]
    assert sum(len(row["payloads"]) for row in result["policies"]) == 12
    assert result["mechanical_verification"]["raw_rgb"] is False
    assert result["mechanical_verification"]["telemetry"] is True


@pytest.mark.parametrize("mutation", ["label", "model", "archive"])
def test_training_authorization_and_archive_fail_closed(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json"
    capture_path = tmp_path / "capture.json"
    training_path = tmp_path / "training.json"
    capture_path.write_text(json.dumps({"cpu_preparation": "PASS", "source_seed": 2221,
        "policies": []}), encoding="utf-8")
    body = {"cpu_preparation": "PASS", "behavior_labels_created": False,
            "model_use_authorized": False, "used_for_decision": False,
            "source_inputs": {"capture": {"path": str(capture_path),
                "sha256": M.V01.sha256(capture_path)}}, "archive_verification": {"all_match": True,
                "files_verified": 437, "transfer": {}, "inventory": {}},
            "shared_training_sources": {}, "policies": []}
    if mutation == "label":
        body["behavior_labels_created"] = True
    elif mutation == "model":
        body["model_use_authorized"] = True
    else:
        body["archive_verification"]["files_verified"] = 436
    training_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        M.accept_pair(protocol, capture_path, training_path, seed=2221, verify_raw_rgb=False)
