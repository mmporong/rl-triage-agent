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

import run_five_policy_recovery_resume_v6 as M

PRIVATE = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025"
           "/vision_goal/ri_full_completion_251/replan_369/recovery_actual_414")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_source_closure_adds_only_legacy_isolation_and_v6() -> None:
    assert len(M.SOURCE_FILES) == 26
    assert set(M.V5.SOURCE_FILES) < set(M.SOURCE_FILES)
    assert set(M.SOURCE_FILES) - set(M.V5.SOURCE_FILES) == {
        "evals/recovery_legacy_validation_isolation_v1.py",
        "tests/test_recovery_legacy_validation_isolation_v1.py",
        "bench/run_five_policy_recovery_resume_v6.py",
        "tests/test_five_policy_recovery_resume_v6.py",
    }


def test_actual_legacy_failure_is_exact_pre_backend_cancellation() -> None:
    base = PRIVATE / "legacy-validation-retry-v1"
    value = M._failure_state(
        PRIVATE / "probe-boundary-retry-v1/execute-v1",
        base / "old-failure-inventory-v1.json",
        base / "closed-lease-v1.json")
    assert value == {
        "inventory_digest": "dbbb3bcbdac78a8cbad60d5c514bc40a5118b61ab3a89ae2d9966228327f1429",
        "inventory_sha256": M.C._sha(base / "old-failure-inventory-v1.json"),
        "closed_lease_sha256": M.C._sha(base / "closed-lease-v1.json"),
        "lease_id": "4c456072-2b71-492f-aec9-9be9ee9865b9",
        "wall_us": 736_432_658,
        "api_calls": 0, "sdk_started": 0,
        "paired_intents_cancelled_before_backend": 1,
        "sdk_roles_reserved_cancelled": 2,
    }


def test_failure_inventory_rejects_any_added_or_changed_file(tmp_path: Path) -> None:
    phase = PRIVATE / "probe-boundary-retry-v1/execute-v1"
    inventory = PRIVATE / "legacy-validation-retry-v1/old-failure-inventory-v1.json"
    lease = PRIVATE / "legacy-validation-retry-v1/closed-lease-v1.json"
    changed = tmp_path / "changed"
    shutil.copytree(phase, changed)
    (changed / "actual-v1/parent-error.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        M._failure_state(changed, inventory, lease)
    added = tmp_path / "added"
    shutil.copytree(phase, added)
    (added / "actual-v1/jobs.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        M._failure_state(added, inventory, lease)


def test_build_preserves_science_and_sums_third_failed_wall(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_protocol = save(tmp_path / "parent.json", {"parent": True})
    parent_plan = save(tmp_path / "parent-plan.json", {"parent": True})
    inventory = save(tmp_path / "inventory.json", {"inventory": True})
    lease = save(tmp_path / "lease.json", {"lease": True})
    phase = tmp_path / "old-phase"
    phase.mkdir()
    monkeypatch.setattr(M.V5, "verify_protocol", lambda *_args, **_kwargs: {
        "cumulative_wall_us": M.EXPECTED_PARENT_WALL_US})
    failure = {
        "inventory_digest": "a" * 64, "inventory_sha256": "b" * 64,
        "closed_lease_sha256": "c" * 64, "lease_id": "lease",
        "wall_us": M.EXPECTED_FAILURE_WALL_US, "api_calls": 0,
        "sdk_started": 0, "paired_intents_cancelled_before_backend": 1,
        "sdk_roles_reserved_cancelled": 2,
    }
    monkeypatch.setattr(M, "_failure_state", lambda *_args: failure)
    monkeypatch.setattr(M.C, "_relative", lambda path: f"bench/protocols/{Path(path).name}")
    private_out, public_out = tmp_path / "private.json", tmp_path / "public.json"
    value = M.build_protocol(
        parent_protocol=parent_protocol, parent_anchor="e" * 40,
        parent_private_plan=parent_plan, old_phase=phase,
        old_inventory=inventory, old_closed_lease=lease,
        private_out=private_out, out=public_out)
    assert value["prior"]["cumulative_wall_us"] == 1_761_960_669
    assert value["prior"]["total_paired_intents_cancelled_before_backend"] == 2
    assert value["prior"]["total_sdk_roles_reserved_cancelled"] == 4
    assert value["budget"] == {
        "api_cap": 32, "sdk_cap": 32, "cumulative_parent_wall_cap_s": 43200,
        "remaining_parent_wall_s": 41438.039331,
        "per_arm_pair_cap": 3, "targets": 4}
    assert str(tmp_path) not in public_out.read_text(encoding="utf-8")
    assert json.loads(private_out.read_text(encoding="utf-8"))["old_phase"] == str(
        phase.resolve())


def install_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    refs = {name: save(tmp_path / f"{name}.json", {"name": name})
            for name in ("parent", "parent-plan", "second", "host", "compat", "cache")}
    plan = {"parent_protocol": str(refs["parent"]),
            "parent_anchor": "e" * 40,
            "parent_private_plan": str(refs["parent-plan"]),
            "remaining_wall_s": 41438.039331}
    parent = {"second_plan": str(refs["second"]),
              "host_protocol": str(refs["host"]),
              "compatibility_protocol": str(refs["compat"]),
              "cache_protocol": str(refs["cache"])}
    monkeypatch.setattr(M, "verify_protocol", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(M.V5, "verify_protocol", lambda *_args, **_kwargs: parent)
    monkeypatch.setattr(M.V5.V4, "registered_sources", lambda _path: [{"id": "source"}])
    monkeypatch.setattr(M.C, "_bound", lambda value, _label: Path(value))

    @contextmanager
    def scope(name: str):
        events.append("enter:" + name)
        try:
            yield
        finally:
            events.append("exit:" + name)

    monkeypatch.setattr(M.LEGACY, "source_validation_scope", lambda: scope("legacy"))

    def old_scopes(_plan, _sources, stack):
        events.append("v5-scopes")
        stack.enter_context(scope("parent-boundary"))

    monkeypatch.setattr(M.V5, "_scopes", old_scopes)
    monkeypatch.setattr(M.V5.V4, "remaining_wall_scope", lambda _wall: scope("wall"))
    monkeypatch.setattr(M.V5.V3.CACHE, "proof_scope", lambda **_kwargs: scope("cache"))
    return events, refs


def test_execute_wraps_real_v3_call_with_legacy_then_v5_scopes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _refs = install_runtime(monkeypatch, tmp_path)

    def run(*_args, **kwargs):
        events.append("run")
        assert events[:6] == ["enter:cache", "exit:cache", "enter:legacy",
                              "v5-scopes", "enter:parent-boundary", "enter:wall"]
        assert kwargs["host_anchor"] == kwargs["compatibility_anchor"] == "e" * 40
        assert kwargs["cache_anchor"] == "e" * 40
        return {"status": "complete"}

    monkeypatch.setattr(M.V5.V3, "run_second_operational", run)
    result = M.execute(
        protocol=tmp_path / "protocol", anchor="f" * 40,
        private_plan=tmp_path / "plan", transport_root=tmp_path / "transport",
        lab=tmp_path / "lab", isaac_python=tmp_path / "python",
        out=tmp_path / "out", approver=M.C.RUN.APPROVER)
    assert result["status"] == "complete"
    assert events[-3:] == ["exit:wall", "exit:parent-boundary", "exit:legacy"]


def test_export_uses_legacy_v5_and_non_nested_cache_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _refs = install_runtime(monkeypatch, tmp_path)

    def export(*_args, **_kwargs):
        events.append("export")
        assert events[-2:] == ["enter:cache", "export"]
        return {"status": "complete"}

    monkeypatch.setattr(M.V5.EXPORT, "export", export)
    result = M.export(
        protocol=tmp_path / "protocol", anchor="f" * 40,
        private_plan=tmp_path / "plan", first_result=tmp_path / "first",
        second_result=tmp_path / "second", lab=tmp_path / "lab",
        out=tmp_path / "out")
    assert result["status"] == "complete"
    assert "enter:wall" not in events
    assert events[:2] == ["enter:cache", "exit:cache"]
    assert events.count("enter:cache") == events.count("exit:cache") == 2
    assert events[-3:] == ["exit:cache", "exit:parent-boundary", "exit:legacy"]
