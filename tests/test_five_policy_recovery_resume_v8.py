# pyright: reportMissingImports=false
from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_five_policy_recovery_resume_v8 as M

PRIVATE = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025"
           "/vision_goal/ri_full_completion_251/replan_369/recovery_actual_414")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_source_closure_adds_only_reuser_v2_and_v8() -> None:
    assert len(M.SOURCE_FILES) == 36
    assert set(M.V7.SOURCE_FILES) < set(M.SOURCE_FILES)
    assert set(M.SOURCE_FILES) - set(M.V7.SOURCE_FILES) == {
        "evals/recovery_recorded_pair_reuse_v2.py",
        "tests/test_recovery_recorded_pair_reuse_v2.py",
        "bench/run_five_policy_recovery_resume_v8.py",
        "tests/test_five_policy_recovery_resume_v8.py",
    }


def test_actual_followup_failure_separates_reuse_from_cancelled_intent() -> None:
    base = PRIVATE / "receipt-followup-retry-v1"
    value = M._failure_state(
        PRIVATE / "receipt-reuse-retry-v1/execute-v1",
        base / "old-failure-inventory-v1.json",
        base / "closed-lease-v1.json")
    assert value["inventory_digest"] == (
        "3f18f91e480e7718f727a500b83efa738c36c2218501c21809984caa501c30c2")
    assert value["wall_us"] == 951_372_786
    assert value["api_calls"] == value["new_sdk_started"] == 0
    assert value["logical_pairs_reused"] == 1
    assert value["logical_roles_reused"] == 2
    assert value["followup_intents_cancelled_before_backend"] == 1
    assert value["followup_roles_reserved_cancelled"] == 2
    assert value["prior_physical_roles_reused"] == 2
    assert value["new_physical_role_cap"] == 30
    assert value["physical_cumulative_cap"] == 32


def test_failure_inventory_rejects_changed_reuse_receipt(tmp_path: Path) -> None:
    phase = PRIVATE / "receipt-reuse-retry-v1/execute-v1"
    base = PRIVATE / "receipt-followup-retry-v1"
    changed = tmp_path / "changed"
    shutil.copytree(phase, changed)
    (changed / M.REUSE_RECEIPT_RELATIVE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        M._failure_state(changed, base / "old-failure-inventory-v1.json",
                         base / "closed-lease-v1.json")


def test_build_preserves_caps_and_does_not_count_replay_as_new_sdk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_protocol = save(tmp_path / "parent.json", {"parent": True})
    parent_plan = save(tmp_path / "parent-plan.json", {"parent": True})
    inventory = save(tmp_path / "inventory.json", {"inventory": True})
    lease = save(tmp_path / "lease.json", {"lease": True})
    phase = tmp_path / "phase"
    phase.mkdir()
    monkeypatch.setattr(M.V7, "verify_protocol", lambda *_args, **_kwargs: {
        "cumulative_wall_us": M.EXPECTED_PARENT_WALL_US})
    failure = {
        "inventory_digest": "a" * 64, "inventory_sha256": "b" * 64,
        "closed_lease_sha256": "c" * 64, "lease_id": "lease",
        "wall_us": M.EXPECTED_FAILURE_WALL_US,
        "api_calls": 0, "new_sdk_started": 0, "new_sdk_completed": 0,
        "logical_pairs_reused": 1, "logical_roles_reused": 2,
        "followup_intents_cancelled_before_backend": 1,
        "followup_roles_reserved_cancelled": 2,
        "completed_reused_pair": {"path": "private/result", "sha256": "d" * 64},
        "reuse_receipt": {"path": "private/receipt", "sha256": "e" * 64},
        "original_physical_pair": {"path": "private/original", "sha256": "f" * 64},
        "prior_physical_roles_reused": 2,
        "new_physical_role_cap": 30, "physical_cumulative_cap": 32,
    }
    monkeypatch.setattr(M, "_failure_state", lambda *_args: failure)
    monkeypatch.setattr(M.C, "_relative", lambda path: f"bench/protocols/{Path(path).name}")
    private_out, public_out = tmp_path / "private.json", tmp_path / "public.json"
    value = M.build_protocol(
        parent_protocol=parent_protocol, parent_anchor="b" * 40,
        parent_private_plan=parent_plan, old_phase=phase,
        old_inventory=inventory, old_closed_lease=lease,
        private_out=private_out, out=public_out)
    assert value["prior"]["new_sdk_started"] == 0
    assert value["prior"]["logical_roles_reused"] == 2
    assert value["prior"]["followup_roles_reserved_cancelled"] == 2
    assert value["prior"]["cumulative_wall_us"] == 3_833_489_176
    assert value["budget"] == {
        "api_cap": 32, "sdk_cap": 32, "new_physical_sdk_cap": 30,
        "prior_physical_sdk_reused": 2,
        "cumulative_parent_wall_cap_s": 43200,
        "remaining_parent_wall_s": 39366.510824,
        "per_arm_pair_cap": 3, "targets": 4}
    assert str(tmp_path) not in public_out.read_text(encoding="utf-8")


def install_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    refs = {name: save(tmp_path / f"{name}.json", {"name": name})
            for name in ("second", "host", "compat", "cache", "inner",
                         "result", "context", "choice")}
    plan = {"remaining_wall_s": 39366.510824}
    v7 = {"prior_failure": {
        "completed_pair": str(refs["result"]),
        "original_context": str(refs["context"]),
        "original_choice": str(refs["choice"])}}
    v6 = {"parent_anchor": "e" * 40}
    v5 = {"host_protocol": str(refs["host"]),
          "compatibility_protocol": str(refs["compat"]),
          "cache_protocol": str(refs["cache"]),
          "inner_loop_protocol": str(refs["inner"]), "inner_anchor": "f" * 40}
    sources = [{"case_id": "case-03", "policy": "fixed_eligible"}]
    monkeypatch.setattr(M, "verify_protocol", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(M, "_parents",
                        lambda _plan: (v7, v6, v5, refs["second"], sources))
    monkeypatch.setattr(M.C, "_bound", lambda value, _label: Path(value))

    @contextmanager
    def scope(name: str, yielded=None):
        events.append("enter:" + name)
        try:
            yield yielded
        finally:
            events.append("exit:" + name)

    monkeypatch.setattr(M.V5.V3.CACHE, "proof_scope", lambda **_kwargs: scope("cache"))
    monkeypatch.setattr(M.V6.LEGACY, "source_validation_scope", lambda: scope("legacy"))
    monkeypatch.setattr(M.V5.BIND, "epoch2_snapshot_scope",
                        lambda _sources: scope("parents", {"case-03:fixed_eligible": {}}))
    monkeypatch.setattr(M.V5.PROBE_BOUNDARY, "probe_boundary_scope",
                        lambda *_args, **_kwargs: scope("boundary"))
    monkeypatch.setattr(M.V7.ROUTE, "receipt_snapshot_routes_scope",
                        lambda *_args, **_kwargs: scope("routes"))
    monkeypatch.setattr(M.REUSER2, "recorded_pair_reuse_scope",
                        lambda **_kwargs: scope("reuse2"))
    monkeypatch.setattr(M.V5.V4, "remaining_wall_scope", lambda _wall: scope("wall"))
    return events


def test_execute_closes_reuser_entry_cache_before_runner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = install_runtime(monkeypatch, tmp_path)

    def run(*_args, **kwargs):
        events.append("run")
        assert events == [
            "enter:cache", "exit:cache", "enter:legacy", "enter:cache",
            "enter:parents", "exit:cache", "enter:boundary", "enter:routes",
            "enter:cache", "enter:reuse2", "exit:cache", "enter:wall", "run"]
        assert kwargs["host_anchor"] == kwargs["compatibility_anchor"] == "e" * 40
        assert kwargs["cache_anchor"] == "e" * 40
        return {"status": "complete"}

    monkeypatch.setattr(M.V5.V3, "run_second_operational", run)
    result = M.execute(
        protocol=tmp_path / "protocol", anchor="a" * 40,
        private_plan=tmp_path / "plan", transport_root=tmp_path / "transport",
        lab=tmp_path / "lab", isaac_python=tmp_path / "python",
        out=tmp_path / "out", approver=M.C.RUN.APPROVER)
    assert result["status"] == "complete"
    assert events[-6:] == ["exit:wall", "exit:reuse2", "exit:routes",
                           "exit:boundary", "exit:parents", "exit:legacy"]


def test_export_keeps_routes_reuser2_and_separate_export_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = install_runtime(monkeypatch, tmp_path)

    def export(*_args, **_kwargs):
        events.append("export")
        assert events[-2:] == ["enter:cache", "export"]
        return {"status": "complete"}

    monkeypatch.setattr(M.V5.EXPORT, "export", export)
    result = M.export(
        protocol=tmp_path / "protocol", anchor="a" * 40,
        private_plan=tmp_path / "plan", first_result=tmp_path / "first",
        second_result=tmp_path / "second", lab=tmp_path / "lab",
        out=tmp_path / "out")
    assert result["status"] == "complete"
    assert "enter:wall" not in events
    assert events.count("enter:cache") == events.count("exit:cache") == 4
    assert events[-6:] == ["exit:cache", "exit:reuse2", "exit:routes",
                           "exit:boundary", "exit:parents", "exit:legacy"]
