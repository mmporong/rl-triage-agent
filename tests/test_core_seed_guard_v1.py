from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import core_seed_guard_v1 as G


def layout(tmp_path):
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    for path in (repo / "bench", repo / "evals", lab / "logs"):
        path.mkdir(parents=True)
    return repo, lab


def test_excluded_invalid_record_is_not_opened_and_scope_is_explicit(tmp_path):
    repo, lab = layout(tmp_path)
    excluded = repo / G.EXCLUDED_RELATIVE
    excluded.mkdir(parents=True)
    (excluded / "result.json").write_text("not JSON", encoding="utf-8")
    old = G.ORIGINAL._files
    result = G.assert_fresh_seeds([2311], lab, repo=repo)
    assert result["status"] == "fresh"
    assert result["excluded_before_enumeration"]
    assert G.ORIGINAL._files is old


def test_core_used_seed_rejects_and_restores_original(tmp_path):
    repo, lab = layout(tmp_path)
    path = repo / "bench/private/core/result.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"seed":2311}', encoding="utf-8")
    old = G.ORIGINAL._files
    with pytest.raises(G.ORIGINAL.SeedReuseError):
        G.assert_fresh_seeds([2311], lab, repo=repo)
    assert G.ORIGINAL._files is old


def test_malformed_core_record_remains_fail_closed(tmp_path):
    repo, lab = layout(tmp_path)
    path = repo / "bench/private/core/result.json"
    path.parent.mkdir(parents=True)
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="읽기 실패"):
        G.assert_fresh_seeds([2311], lab, repo=repo)


def test_isaac_agent_yaml_used_seed_rejects(tmp_path):
    repo, lab = layout(tmp_path)
    path = lab / "logs/one/params/agent.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("seed: 2311\n", encoding="utf-8")
    with pytest.raises(G.ORIGINAL.SeedReuseError):
        G.assert_fresh_seeds([2311], lab, repo=repo)
