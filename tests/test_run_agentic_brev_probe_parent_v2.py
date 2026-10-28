"""합성 CPU fixture로 Parent2의 JOB4 scope·복원·source closure를 검증한다."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]
import run_agentic_brev_probe_parent_v2 as P2
from test_agentic_brev_probe_parent_v1 import NOW, Services, world


def test_parent2_scope_is_reentrant_and_restores_frozen_parent1() -> None:
    original_job, original_sources = P2.V1.JOB2, P2.V1.SOURCES
    with P2.aws_parent_scope():
        assert P2.scope_active() and P2.V1.JOB2 is P2.JOB4 and P2.V1.SOURCES == P2.SOURCES
        with P2.aws_parent_scope():
            assert P2.V1.JOB2 is P2.JOB4 and P2._STATE["depth"] == 2
        assert P2._STATE["depth"] == 1
    assert not P2.scope_active() and P2.V1.JOB2 is original_job and P2.V1.SOURCES is original_sources


def test_parent2_delegates_prepare_and_run_only_inside_aws_scope(monkeypatch) -> None:
    seen = []

    def fake(*args, **kwargs):
        seen.append((P2.V1.JOB2, P2.V1.SOURCES, args, kwargs))
        return {"status": "synthetic_cpu"}

    monkeypatch.setattr(P2.V1, "prepare", fake)
    assert P2.prepare("request", allow_fixture=True) == {"status": "synthetic_cpu"}
    monkeypatch.setattr(P2.V1, "run", fake)
    assert P2.run("request", execute=False) == {"status": "synthetic_cpu"}
    assert all(job is P2.JOB4 and sources == P2.SOURCES for job, sources, _args, _kwargs in seen)
    assert P2.V1.JOB2.__name__ == "agentic_play_job_v2"


def test_parent2_closure_contains_job4_and_actual_capture_entries() -> None:
    required = {
        P2.SOURCE, P2.TEST_SOURCE, "bench/agentic_play_job_v4.py", "tests/test_agentic_play_job_v4.py",
        "bench/run_agentic_play_record_v6.py", "tests/test_run_agentic_play_record_v6.py",
        "bench/run_agentic_capture_unit_v8.py", "tests/test_agentic_capture_unit_v8.py",
    }
    assert required <= P2.SOURCES


def test_same_name_foreign_job_module_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(P2.V1, "JOB2", SimpleNamespace(__name__=P2._BASE_JOB.__name__))
    with pytest.raises(RuntimeError, match="이미 교체"), P2.aws_parent_scope():
        pass


def test_parent2_runs_complete_two_arm_parent_lifetime_with_synthetic_cpu_fixture(tmp_path) -> None:
    with P2.aws_parent_scope():
        fixture = world(tmp_path / "synthetic-parent2")
    services = Services(fixture)
    result = P2.run(fixture["request_ref"], execute=True, allow_fixture=True,
                    home=fixture["home"], now=NOW, services=services)
    assert result["status"] == "complete"
    assert [row["operation_id"] for row in result["completed"]] == ["1" * 64, "2" * 64]
    assert services.calls == ["NONE", "F1"] and services.acquires == services.releases == 1
    assert not services.active and not P2.scope_active()

