from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_recovery_continuation_v4")


def git(repo: Path, *args: str, env=None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", env=env).stdout.strip()


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir()
    git(repo, "init", "-q")
    parent_path = save(repo / "bench/protocols/parent.json", {"schema": M.V2.SCHEMA})
    git(repo, "add", "--", "bench/protocols/parent.json")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "parent")
    parent = git(repo, "rev-parse", "HEAD")
    save(repo / "bench/protocols/continuation.json", {"schema": M.SCHEMA})
    git(repo, "add", "--", "bench/protocols/continuation.json")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "registration")
    registration = git(repo, "rev-parse", "HEAD")
    return repo, parent_path, parent, registration


def validation_inputs(tmp_path: Path, repo: Path, parent_path: Path, parent: str):
    manifest_path = save(tmp_path / "manifest.json", {})
    approval_path = save(tmp_path / "approval.json", {})
    failed = tmp_path / "failed"; failed.mkdir()
    smoke = tmp_path / "smoke"; smoke.mkdir()
    manifest = {"anchor": parent, "lab": str(tmp_path / "lab"),
                "protocol": {"path": str(parent_path.resolve()), "sha256": sha(parent_path)},
                "rows": []}
    smoke_ref = {"path": str(smoke.resolve()), "result_sha256": "a" * 64}
    save(failed / "intent.json", {
        "manifest": M.V3._ref(manifest_path), "approval": M.V3._ref(approval_path),
        "smoke": smoke_ref, "automatic_retry": False, "model_calls": 0})
    protocol = {"parent_protocol": {
        "path": parent_path.resolve().relative_to(repo.resolve()).as_posix(),
        "sha256_lf": M.V3._sha(parent_path, lf=True)}}
    return manifest_path, approval_path, failed, smoke, manifest, smoke_ref, protocol


def patches(tmp_path: Path, repo: Path, manifest: dict, smoke_ref: dict, protocol: dict):
    return (
        patch.object(M.V3, "ROOT", repo), patch.object(M, "ROOT", repo),
        patch.object(M.V3, "verify_protocol", return_value=protocol),
        patch.object(M.V1, "_load_manifest", return_value=(manifest, {})),
        patch.object(M.V1, "verify_smoke", return_value=smoke_ref),
        patch.object(M.V1, "_approval", return_value={"ok": True}),
        patch.object(M.V3, "_failure_boundary", return_value=({"status": "invalid"}, {})),
        patch.object(M.V3, "_completed_training", return_value={"ok": True}),
    )


def test_different_parent_and_registration_anchors_are_required_and_allowed(tmp_path):
    repo, parent_path, parent, registration = history(tmp_path)
    inputs = validation_inputs(tmp_path, repo, parent_path, parent)
    manifest_path, approval, failed, smoke, manifest, smoke_ref, protocol = inputs
    contexts = patches(tmp_path, repo, manifest, smoke_ref, protocol)
    for context in contexts: context.start()
    try:
        result = M._validate_sources(tmp_path / "v4.json", manifest_path, approval, smoke, failed,
                                     anchor=registration, lab=tmp_path / "lab")
    finally:
        for context in reversed(contexts): context.stop()
    assert result[0]["anchor"] == parent and parent != registration


def test_non_ancestor_registration_is_rejected(tmp_path):
    repo, parent_path, parent, _registration = history(tmp_path)
    tree = git(repo, "rev-parse", f"{parent}^{{tree}}")
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    proc = subprocess.run(["git", "-C", str(repo), "commit-tree", tree], input="unrelated\n",
                          text=True, capture_output=True, check=True, env=env)
    unrelated = proc.stdout.strip()
    inputs = validation_inputs(tmp_path, repo, parent_path, parent)
    manifest_path, approval, failed, smoke, manifest, smoke_ref, protocol = inputs
    contexts = patches(tmp_path, repo, manifest, smoke_ref, protocol)
    for context in contexts: context.start()
    try:
        with pytest.raises(ValueError, match="Git anchor"):
            M._validate_sources(tmp_path / "v4.json", manifest_path, approval, smoke, failed,
                                anchor=unrelated, lab=tmp_path / "lab")
    finally:
        for context in reversed(contexts): context.stop()


def test_parent_protocol_blob_change_after_parent_anchor_is_rejected(tmp_path):
    repo, parent_path, parent, _registration = history(tmp_path)
    parent_path.write_text(json.dumps({"schema": M.V2.SCHEMA, "changed": True}), encoding="utf-8")
    git(repo, "add", "--", "bench/protocols/parent.json")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "changed parent")
    registration = git(repo, "rev-parse", "HEAD")
    inputs = validation_inputs(tmp_path, repo, parent_path, parent)
    manifest_path, approval, failed, smoke, manifest, smoke_ref, protocol = inputs
    protocol["parent_protocol"]["sha256_lf"] = M.V3._sha(parent_path, lf=True)
    contexts = patches(tmp_path, repo, manifest, smoke_ref, protocol)
    for context in contexts: context.start()
    try:
        with pytest.raises(ValueError, match="두 anchor"):
            M._validate_sources(tmp_path / "v4.json", manifest_path, approval, smoke, failed,
                                anchor=registration, lab=tmp_path / "lab")
    finally:
        for context in reversed(contexts): context.stop()


def test_v3_old_equality_contract_rejects_required_two_anchor_shape(tmp_path):
    with patch.object(M.V3, "verify_protocol", return_value={"parent_protocol": {"sha256_lf": "a" * 64}}), \
            patch.object(M.V1, "_load_manifest", return_value=({
                "anchor": "1" * 40, "lab": str(tmp_path), "protocol": {"path": str(tmp_path / "p")}}, {})), \
            pytest.raises(ValueError, match="anchor/lab"):
        M._ORIGINAL[4](tmp_path / "v3", tmp_path / "manifest", tmp_path / "approval",
                       tmp_path / "smoke", tmp_path / "failed", anchor="2" * 40, lab=tmp_path)


def test_v4_scope_reentrant_and_restores_v3_after_error():
    before = (M.V3.SCHEMA, M.V3.MANIFEST_SCHEMA, M.V3.RESULT_SCHEMA,
              M.V3.SOURCE_FILES, M.V3._validate_sources)
    with pytest.raises(RuntimeError, match="stop"), M._v3_scope():
        assert (M.V3.SCHEMA, M.V3.MANIFEST_SCHEMA, M.V3.RESULT_SCHEMA) == (
            M.SCHEMA, M.MANIFEST_SCHEMA, M.RESULT_SCHEMA)
        with M._v3_scope():
            assert M.V3._validate_sources is M._validate_sources
        raise RuntimeError("stop")
    assert (M.V3.SCHEMA, M.V3.MANIFEST_SCHEMA, M.V3.RESULT_SCHEMA,
            M.V3.SOURCE_FILES, M.V3._validate_sources) == before
