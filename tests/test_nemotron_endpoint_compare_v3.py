"""새 endpoint 판이 기존 소비와 판정 기준을 유지하는지 검사한다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import nemotron_endpoint_compare_v3 as V3
import nemotron_bottleneck_ledger_v2 as L


def test_criteria_unchanged_and_identity_not_claimed():
    assert V3.FIXED["comparison_rules"] == V3.C.FIXED
    assert V3.FIXED["response_identity"]["official_alias_mapping_confirmed"] is False
    assert V3.FIXED["budget"]["remaining_generation_max"] == 197
    assert V3.FIXED["budget"]["prior_models_get"] + V3.FIXED["budget"]["prior_v2_generation"] + V3.FIXED["budget"]["additional_synthetic_image"] == 3


def test_parent_plan_cannot_point_at_another_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(V3.C, "RUN_ROOT", tmp_path)
    L.write_new(tmp_path / "plan.json", {"experiment_commit": "a"*40})
    with pytest.raises(ValueError, match="이전 원장"):
        V3.parent_book()


def test_new_stage_cannot_skip_predecessors(tmp_path, monkeypatch):
    monkeypatch.setattr(V3, "RUN_ROOT", tmp_path / "phase")
    monkeypatch.setattr(V3.C, "RUN_ROOT", tmp_path / "parent")
    def phase_under_lock():
        assert (tmp_path / "parent/run.lock").is_file()
        return object()
    monkeypatch.setattr(V3, "phase_book", phase_under_lock)
    with pytest.raises(ValueError, match="이전 단계"):
        V3.run("B2")
    assert not (tmp_path / "parent/run.lock").exists()


def test_summary_reads_under_shared_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(V3.C, "RUN_ROOT", tmp_path)
    def summarize_under_lock():
        assert (tmp_path / "run.lock").is_file()
        return {"verified": True}
    monkeypatch.setattr(V3, "summarize_locked", summarize_under_lock)
    assert V3.summarize() == {"verified": True}
    assert not (tmp_path / "run.lock").exists()


def test_snapshot_requires_original_consumption_and_image_match(tmp_path, monkeypatch):
    monkeypatch.setattr(V3.C, "dynamic", lambda: {"inputs": {}, "historical": {}})
    monkeypatch.setattr(V3, "parent_book", lambda: type("Book", (), {"audit": lambda _: []})())
    with pytest.raises(ValueError, match="이전 요청"):
        V3.dynamic()


def parent_rows():
    return [{"reservation": {"stage": "B1", "label": "first", "model": L.CANDIDATE}, "response": {}},
            {"reservation": {"stage": "B4", "label": "supplemental_image_identity_check_v1", "model": L.CANDIDATE}, "response": {}}]


def test_parent_prefix_remains_fixed_when_registered_phase_grows(monkeypatch):
    registered = ("B1", "v3::case_a1", L.CANDIDATE)
    monkeypatch.setattr(V3, "allowed_reservations", lambda: {registered})
    before = V3.validate_parent_rows(parent_rows())
    extra = {"reservation": dict(zip(("stage", "label", "model"), registered)), "response": {}}
    assert V3.validate_parent_rows(parent_rows() + [extra]) == before
    assert before["parent_generation_count_at_design"] == 2
    assert before["parent_tail_reservation_sha256"] == L.digest(parent_rows()[-1]["reservation"])


@pytest.mark.parametrize("label", ["outside", "v3::unregistered"])
def test_extra_unregistered_requests_rejected(monkeypatch, label):
    monkeypatch.setattr(V3, "allowed_reservations", lambda: set())
    extra = {"reservation": {"stage": "B1", "label": label, "model": L.CANDIDATE}, "response": {}}
    with pytest.raises(ValueError, match="등록 일정 밖"):
        V3.validate_parent_rows(parent_rows() + [extra])


def test_new_plan_refuses_existing_phase_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(V3.C, "RUN_ROOT", tmp_path)
    def verify_under_shared_lock(_):
        assert (tmp_path / "run.lock").is_file()
    monkeypatch.setattr(V3, "verify", verify_under_shared_lock)
    monkeypatch.setattr(V3, "parent_book", lambda: type("Book", (), {"audit": lambda _: [1, 2, 3]})())
    with pytest.raises(ValueError, match="계획 전에"):
        V3.plan("a"*40)
    assert not (tmp_path / "run.lock").exists()
