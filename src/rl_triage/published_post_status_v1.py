"""커밋된 공개 post-actuator 결과를 부수효과 없이 조회한다."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT_REL = "evals/results"
RESULT_FOLDER = "post_actuator_confirmation_s2664_20261105_v1"
MANIFEST_SCHEMA = "post_actuator_confirmation_public_export_v1"
FORBIDDEN = "ri_codex2_tokenfactory_live_160"
SHA = re.compile(r"[0-9a-f]{64}\Z")
DATA_FILES = ("metrics.json", "result.json", "sources.json")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_object(value: bytes, label: str) -> dict[str, Any]:
    parsed = json.loads(value.decode("utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must contain an object")
    return parsed


def _lexical_public_folder(raw: os.PathLike[str] | str, repo_root: Path) -> Path:
    text = os.fspath(raw)
    normalized = text.replace("\\", "/")
    pure = PurePath(text)
    if (not text or FORBIDDEN in normalized.casefold() or ".." in pure.parts):
        raise ValueError("published post folder is forbidden")
    candidate = Path(text) if pure.is_absolute() else repo_root / Path(text)
    lexical = Path(os.path.abspath(candidate))
    expected = repo_root / PUBLIC_ROOT_REL / RESULT_FOLDER
    if lexical != expected:
        raise ValueError("published post folder differs")
    return lexical


def _git_blob(repo_root: Path, relative: str) -> tuple[str, bytes]:
    anchor = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", anchor) is None:
        raise ValueError("published post HEAD differs")
    subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", relative],
        check=True, capture_output=True,
    )
    blob = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{anchor}:{relative}"],
        check=True, capture_output=True,
    ).stdout
    return anchor, blob


def _assert_public(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() == "path":
                raise ValueError("published post data contains a path")
            _assert_public(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public(item)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        if (FORBIDDEN in normalized or "bench/private" in normalized
                or re.search(r"(^|[^a-z])[a-z]:/", normalized)
                or normalized.startswith(("/home/", "/mnt/", "/users/"))):
            raise ValueError("published post data contains a private path")


def published_post_status(folder: os.PathLike[str] | str, *, repo_root: Path = ROOT
                          ) -> dict[str, Any]:
    repo_root = Path(os.path.abspath(repo_root))
    lexical = _lexical_public_folder(folder, repo_root)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical or not resolved.is_dir():
        raise ValueError("published post folder escapes its registered location")
    entries = {path.name for path in resolved.iterdir()}
    if entries != {"manifest.json", *DATA_FILES}:
        raise ValueError("published post folder contents differ")
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("published post manifest differs")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_object(manifest_bytes, "published manifest")
    rows = manifest.get("files")
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete"
            or not isinstance(rows, list) or len(rows) != len(DATA_FILES)):
        raise ValueError("published post manifest contract differs")
    expected_rows: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("published post manifest file list differs")
        name, digest = row.get("name"), row.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise TypeError("published post manifest file list differs")
        expected_rows[name] = digest
    if (set(expected_rows) != set(DATA_FILES)
            or any(SHA.fullmatch(expected_rows[name]) is None for name in DATA_FILES)):
        raise ValueError("published post manifest file list differs")
    loaded: dict[str, dict[str, Any]] = {}
    file_refs: dict[str, dict[str, str]] = {}
    anchor: str | None = None
    for name in DATA_FILES:
        path = resolved / name
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True).parent != resolved:
            raise ValueError(f"published post file escapes: {name}")
        raw = path.read_bytes()
        if _sha_bytes(raw) != expected_rows[name]:
            raise ValueError(f"published post file hash differs: {name}")
        relative = path.relative_to(repo_root).as_posix()
        current_anchor, blob = _git_blob(repo_root, relative)
        if raw != blob or (anchor is not None and anchor != current_anchor):
            raise ValueError(f"published post file is not the committed HEAD blob: {name}")
        anchor = current_anchor
        loaded[name] = _read_object(raw, name)
        _assert_public(loaded[name])
        file_refs[name] = {"path": relative, "sha256": expected_rows[name]}
    manifest_relative = manifest_path.relative_to(repo_root).as_posix()
    manifest_anchor, manifest_blob = _git_blob(repo_root, manifest_relative)
    if manifest_blob != manifest_bytes or anchor != manifest_anchor:
        raise ValueError("published post manifest is not the committed HEAD blob")
    result, metrics, sources = loaded["result.json"], loaded["metrics.json"], loaded["sources.json"]
    if (result.get("schema") != "post_actuator_confirmation_public_result_v1"
            or result.get("status") != "complete" or result.get("mapping_outcome") != "normal"
            or result.get("effective_action") != "reference_review"
            or result.get("trained_policy_validated") is not False
            or result.get("new_recovery_claim") is not False):
        raise ValueError("published post result differs")
    if (metrics.get("schema") != "post_actuator_confirmation_public_metrics_v1"
            or metrics.get("api_calls") != 0
            or metrics.get("sdk") != {
                "post_cap": 1, "post_sdk_launches": 1,
                "primary_cap": 32, "whole_study_cap": 33,
            }):
        raise ValueError("published post metrics differ")
    if sources.get("schema") != "post_actuator_confirmation_public_sources_v1":
        raise ValueError("published post sources differ")
    return {
        "schema": "published_post_status_v1",
        "status": "complete",
        "evidence_mode": "committed_public_result_replay",
        "fresh_measurement": False,
        "new_execution": False,
        "query_cost": {"sdk_launches": 0, "api_calls": 0},
        "stored_execution_cost": {"sdk_launches": 1, "api_calls": 0},
        "stored_result": deepcopy(result),
        "stored_metrics": deepcopy(metrics),
        "stored_sources_sha256": expected_rows["sources.json"],
        "public_files": file_refs,
        "manifest": {"path": manifest_relative, "sha256": _sha_bytes(manifest_bytes)},
        "git_anchor": anchor,
    }
