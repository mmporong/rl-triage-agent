"""선택 비교의 실행 전 고정 범위와 commit 봉인 회귀."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import selection_comparison_demo_v1 as D  # noqa: E402


def test_registered_plan_has_twenty_slots_and_separate_source_costs():
    protocol = D.fixed_protocol()
    plan = protocol["api"]["slot_plan"]
    assert len(plan["slots"]) == 20 and sum(row["max_requests"] for row in plan["slots"]) == 60
    assert len(set(row["slot_id"] for row in plan["slots"])) == 20
    assert protocol["training"]["checkpoint_loaded"] is False
    assert protocol["budgets"]["full_training_runs_max"] == 2 + 4 * 2
    assert protocol["budgets"]["fixed_evaluation_stages_max"] == 1 + 4 * 2
    assert protocol["selection"]["correction_model_calls"] == 0
    assert protocol["exposure"]["used_for_decision"] is False
    assert not any(value in json.dumps(protocol) for value in ("C:\\\\Users", "LIMMM", "bench/private"))


def test_repo_imports_of_bound_sources_are_also_sealed():
    missing = set()
    for name in sorted(D.CODE):
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *(node.module + "." + item.name for item in node.names)]
            else:
                continue
            for module in modules:
                for directory in ("evals", "bench", "src"):
                    stem = ROOT / directory / module.replace(".", "/")
                    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                        relative = candidate.relative_to(ROOT).as_posix()
                        if candidate.is_file() and relative not in D.CODE:
                            missing.add(relative)
    assert not missing, sorted(missing)


def test_verify_repository_rejects_index_or_worktree_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"value = 1\n")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(b"{}\n")
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "PROTOCOL", protocol_path)
    monkeypatch.setattr(D, "CODE", {"source.py"})
    monkeypatch.setattr(D, "EVIDENCE", set())
    monkeypatch.setattr(D, "git", lambda *args: "head")
    data = {"source.py": source.read_bytes(), "protocol.json": protocol_path.read_bytes()}
    calls = []

    def run(args, **kwargs):
        version = args[-1]
        calls.append(version)
        return SimpleNamespace(stdout=data[version.split(":", 1)[1]])

    monkeypatch.setattr(D.subprocess, "run", run)
    protocol = {"source_sha256_lf": {"source.py": D.sha(source, True)}, "evidence_sha256_lf": {}}
    D.verify_repository(protocol, "anchor", D.sha(protocol_path, True))
    assert ":source.py" in calls and "HEAD:source.py" in calls and "anchor:source.py" in calls
    data["source.py"] = b"value = 2\n"
    with pytest.raises(ValueError, match="commit/index"):
        D.verify_repository(protocol, "anchor", D.sha(protocol_path, True))
    source.write_bytes(b"value = 3\n")
    with pytest.raises(ValueError, match="worktree"):
        D.verify_repository(protocol, "anchor", D.sha(protocol_path, True))


def test_binding_keeps_cumulative_faults_and_new_initialization(tmp_path, monkeypatch):
    protocol = {"source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    path = tmp_path / "protocol.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "PROTOCOL", path)
    corrections = [D.CONTROL.RC.CORRECTIONS["reward"]]
    value = D.binding(protocol, "a" * 40, faults=["F1", "R1"], applied=corrections,
                      checkpoint="b" * 64)
    assert value["original_faults"] == ["F1", "R1"]
    assert value["applied_corrections"] == corrections
    assert value["source_checkpoint_sha256"] == "b" * 64
    assert value["max_iterations"] == 300 and value["protocol_path"] == "protocol.json"
