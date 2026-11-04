from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from rl_triage.published_post_status_v1 import RESULT_FOLDER, published_post_status


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _data() -> dict[str, dict]:
    return {
        "result.json": {
            "schema": "post_actuator_confirmation_public_result_v1",
            "status": "complete",
            "mapping_outcome": "normal",
            "effective_action": "reference_review",
            "trained_policy_validated": False,
            "new_recovery_claim": False,
        },
        "metrics.json": {
            "schema": "post_actuator_confirmation_public_metrics_v1",
            "api_calls": 0,
            "sdk": {
                "post_cap": 1,
                "post_sdk_launches": 1,
                "primary_cap": 32,
                "whole_study_cap": 33,
            },
        },
        "sources.json": {
            "schema": "post_actuator_confirmation_public_sources_v1",
            "source_inputs": {"sha256": "1" * 64},
        },
    }


def _commit(repo: Path) -> Path:
    folder = repo / "evals/results" / RESULT_FOLDER
    folder.mkdir(parents=True)
    data = _data()
    rows = []
    for name in ("metrics.json", "result.json", "sources.json"):
        raw = _bytes(data[name])
        (folder / name).write_bytes(raw)
        rows.append({"name": name, "sha256": _sha(raw)})
    (folder / "manifest.json").write_bytes(_bytes({
        "schema": "post_actuator_confirmation_public_export_v1",
        "status": "complete",
        "files": rows,
    }))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "evals/results"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return folder


def _rewrite_manifest(folder: Path) -> None:
    rows = [
        {"name": name, "sha256": _sha((folder / name).read_bytes())}
        for name in ("metrics.json", "result.json", "sources.json")
    ]
    (folder / "manifest.json").write_bytes(_bytes({
        "schema": "post_actuator_confirmation_public_export_v1",
        "status": "complete",
        "files": rows,
    }))


def test_committed_public_result_replays_without_new_cost(tmp_path: Path) -> None:
    folder = _commit(tmp_path / "repo")
    result = published_post_status(folder, repo_root=tmp_path / "repo")
    assert result["evidence_mode"] == "committed_public_result_replay"
    assert result["fresh_measurement"] is result["new_execution"] is False
    assert result["query_cost"] == {"sdk_launches": 0, "api_calls": 0}
    assert result["stored_execution_cost"] == {"sdk_launches": 1, "api_calls": 0}
    assert result["stored_result"]["effective_action"] == "reference_review"


def test_uncommitted_self_consistent_export_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    folder = _commit(repo)
    result = json.loads((folder / "result.json").read_text())
    result["extra"] = "uncommitted"
    (folder / "result.json").write_bytes(_bytes(result))
    _rewrite_manifest(folder)
    with pytest.raises(ValueError, match="committed HEAD blob"):
        published_post_status(folder, repo_root=repo)


def test_file_hash_mutation_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    folder = _commit(repo)
    (folder / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash differs"):
        published_post_status(folder, repo_root=repo)


def test_extra_untracked_file_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    folder = _commit(repo)
    (folder / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="folder contents differ"):
        published_post_status(folder, repo_root=repo)


@pytest.mark.parametrize("value", ["../post", "evals/results/other", "ri_codex2_tokenfactory_live_160"])
def test_other_folder_or_path_escape_is_rejected_before_io(tmp_path: Path, value: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="published post folder"):
        published_post_status(value, repo_root=repo)


def test_symlink_folder_escape_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = _commit(tmp_path / "target-repo")
    link = repo / "evals/results" / RESULT_FOLDER
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"symlink unavailable: {error}")
        quoted_link = str(link).replace("'", "''")
        quoted_target = str(target).replace("'", "''")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             f"New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_target}' | Out-Null"],
            check=True, capture_output=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    with pytest.raises(ValueError, match="escapes"):
        published_post_status(link, repo_root=repo)


def test_public_data_with_path_key_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    folder = _commit(repo)
    sources = json.loads((folder / "sources.json").read_text())
    sources["private"] = {"path": "C:/Users/example/private.json"}
    (folder / "sources.json").write_bytes(_bytes(sources))
    _rewrite_manifest(folder)
    subprocess.run(["git", "-C", str(repo), "add", "evals/results"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "second"], check=True)
    with pytest.raises(ValueError, match="contains a path"):
        published_post_status(folder, repo_root=repo)
