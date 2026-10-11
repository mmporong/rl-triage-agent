import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence_media", ROOT / "docs" / "demo" / "build_evidence_media.py")
media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = media
SPEC.loader.exec_module(media)


def test_frozen_public_snapshot_bindings_are_valid():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    assert evidence.audit["valid"] is True
    assert evidence.selected["case"] == "h01_s2027"
    assert evidence.primary_count == 0
    assert evidence.selected["result"]["status"] == "hold"


def test_source_manifest_tamper_is_rejected(tmp_path):
    snapshot = json.loads(media.DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["sources"][0]["sha256_lf"] = "0" * 64
    tampered = tmp_path / "snapshot.json"
    tampered.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(media.EvidenceError, match="source LF SHA-256 불일치"):
        media.verify_evidence(ROOT, tampered, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)


def test_receipt_digest_tamper_is_rejected():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    receipt = dict(evidence.selected["result"]["observed"][0])
    receipt["receipt_digest"] = "0" * 64
    result = {**evidence.selected["result"], "observed": [receipt]}
    with pytest.raises(media.EvidenceError, match="receipt digest 불일치"):
        media._verify_receipts(ROOT, evidence.snapshot["tag"], evidence.selected["case"], result, evidence.source_hashes)


def test_wrong_existing_commit_cannot_claim_frozen_sources():
    wrong_commit = "43fa1da2b9e9a6c02aec0023de091051d9f47146"
    with pytest.raises(media.EvidenceError, match="동결 커밋 code source 불일치"):
        media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL, wrong_commit)


def test_receipt_cannot_substitute_an_unrelated_registered_source_hash():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    receipt = dict(evidence.selected["result"]["observed"][0])
    receipt["report_sha256_lf"] = evidence.source_hashes["bench/protocols/active_diagnosis_v2.json"]
    result = {**evidence.selected["result"], "observed": [receipt]}
    with pytest.raises(media.EvidenceError, match="report/execution 결속 불일치"):
        media._verify_receipts(ROOT, evidence.snapshot["tag"], evidence.selected["case"], result, evidence.source_hashes)


def test_public_artifact_tamper_is_rejected(tmp_path):
    artifact = tmp_path / "card.png"
    artifact.write_bytes(b"verified card")
    digest = media.hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"artifacts": {
        "card.png": {"sha256": digest, "bytes": artifact.stat().st_size}
    }}), encoding="utf-8")
    media.verify_artifact_manifest(tmp_path, {"card.png"})
    artifact.write_bytes(b"tampered card")
    with pytest.raises(media.EvidenceError, match="artifact hash/size 불일치"):
        media.verify_artifact_manifest(tmp_path, {"card.png"})


def test_first_verified_primary_wins_over_registered_fallback():
    snapshot = {"registered_order": ["first", "second"], "cases": [
        {"case": "first", "state": "verified", "proof": False},
        {"case": "second", "state": "verified", "proof": True},
    ]}
    audit = {"cases": [
        {"case": "first", "valid": True, "proof_changed": False},
        {"case": "second", "valid": True, "proof_changed": True},
    ]}
    assert media.select_scene(snapshot, audit)["case"] == "second"


def test_no_primary_uses_first_registered_case_only():
    snapshot = {"registered_order": ["registered-first", "later"], "cases": [
        {"case": "later", "state": "verified", "proof": False},
        {"case": "registered-first", "state": "verified", "proof": False},
    ]}
    audit = {"cases": [
        {"case": "registered-first", "valid": True, "proof_changed": False},
        {"case": "later", "valid": True, "proof_changed": False},
    ]}
    assert media.select_scene(snapshot, audit)["case"] == "registered-first"
