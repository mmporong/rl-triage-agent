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

import run_five_policy_recovery_resume_v5 as M

PRIVATE = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025"
           "/vision_goal/ri_full_completion_251/replan_369/recovery_actual_414")
CPU_PROVENANCE = PRIVATE / "probe-boundary-retry-v1/POST_FAILURE_CPU_LEDGER_v2.json"


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_source_closure_is_exact_and_contains_boundary_export_contract() -> None:
    assert len(M.SOURCE_FILES) == 22
    assert set(M.V4.SOURCE_FILES) < set(M.SOURCE_FILES)
    assert set(M.PROBE_BOUNDARY.SOURCE_FILES) <= set(M.SOURCE_FILES)
    assert {
        "bench/run_five_policy_recovery_resume_v5.py",
        "tests/test_five_policy_recovery_resume_v5.py",
        "evals/export_five_policy_recovery_v2.py",
        "tests/test_export_five_policy_recovery_v2.py",
    } <= set(M.SOURCE_FILES)


def test_actual_second_failure_is_exact_pre_backend_cancellation() -> None:
    phase = PRIVATE / "second-parent-retry-v1/execute-v1"
    inventory = PRIVATE / "probe-boundary-retry-v1/old-failure-inventory-v1.json"
    closed = PRIVATE / "probe-boundary-retry-v1/closed-lease-v1.json"
    value = M._second_failure_state(
        phase, inventory, closed,
        registration_anchor="a38a08c6c2586b6c66102651c6693c6f66ea99a7",
        parent_anchor="64085a626504e85d70bec25c981d50836d2d3517",
        cpu_provenance=CPU_PROVENANCE)
    assert set(value["inventory"]) == M.SECOND_FAILURE_FILES
    assert value["wall_us"] == 647_737_403
    cpu = value["post_failure_cpu_provenance"]
    assert cpu["status"] == "post_failure_cpu_fixture_pending_not_production_execution"
    assert cpu["ledger"] == {
        "path": M.POST_FAILURE_CPU_LEDGER,
        "sha256": "be01c5c963be552f06281e4f23adc03f739d2fb646b8d50c407b85c25dd33f05",
        "size_bytes": 963}
    assert cpu["events"] == ["proposed", "approved", "consumed"]
    assert cpu["sdk_started"] == cpu["api_requests"] == 0
    lease = M.C._read(closed)
    assert M.math.ceil((M.C._old_elapsed(lease) + 0.000001) * 1_000_000) == value["wall_us"]
    assert value["api_calls"] == value["sdk_started"] == 0
    assert value["paired_intents"] == 1
    assert value["sdk_roles_reserved_cancelled"] == 2


def test_post_failure_cpu_ledger_requires_explicit_provenance() -> None:
    source = PRIVATE / "second-parent-retry-v1/execute-v1"
    with pytest.raises(ValueError, match="등록된 CPU 원장"):
        M._inventory(source)


def test_post_failure_cpu_ledger_must_be_exact_registered_file(tmp_path: Path) -> None:
    source = PRIVATE / "second-parent-retry-v1/execute-v1"
    evidence = M._cpu_provenance(CPU_PROVENANCE, source)
    assert set(M._inventory(source, permitted_extra=evidence["ledger"])) == M.SECOND_FAILURE_FILES
    changed = tmp_path / "changed"
    shutil.copytree(source, changed)
    ledger = changed / M.POST_FAILURE_CPU_LEDGER
    ledger.write_text("event\n", encoding="utf-8")
    with pytest.raises(ValueError, match="등록값 또는 내용"):
        M._inventory(changed, permitted_extra=evidence["ledger"])

    renamed = tmp_path / "renamed"
    shutil.copytree(source, renamed)
    ledger = renamed / M.POST_FAILURE_CPU_LEDGER
    ledger.rename(ledger.with_name("other.ledger.jsonl"))
    with pytest.raises(ValueError, match="등록된 CPU 원장"):
        M._inventory(renamed, permitted_extra=evidence["ledger"])


def test_build_keeps_private_paths_out_of_public_and_sums_two_failures(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    components = {}
    for name in ("second", "host", "compat", "cache", "loop"):
        components[name] = save(tmp_path / f"{name}.json", {"name": name})
    old_protocol = save(tmp_path / "old-protocol.json", {"old": True})
    old_plan = save(tmp_path / "old-plan.json", {"old": True})
    old_inventory = save(tmp_path / "old-inventory.json", {"old": True})
    old_lease = save(tmp_path / "old-lease.json", {"old": True})
    phase = tmp_path / "private-old-phase"
    phase.mkdir()
    monkeypatch.setattr(M.V4, "verify_protocol", lambda *_args, **_kwargs: {
        "old_wall_upper_bound_s": 377.790608,
        "inventory_digest": "a" * 64,
        "lease_id": "first-lease",
        "parent_anchor": "f" * 40,
    })
    monkeypatch.setattr(M, "_second_failure_state", lambda *_args, **_kwargs: {
        "inventory": {"x": {"sha256": "b" * 64, "size_bytes": 1}},
        "inventory_digest": "c" * 64,
        "inventory_sha256": "d" * 64,
        "lease_id": "second-lease",
        "closed_lease_sha256": "e" * 64,
        "wall_us": 647_737_403,
        "api_calls": 0, "sdk_started": 0,
        "paired_intents": 1, "sdk_roles_reserved_cancelled": 2,
        "post_failure_cpu_provenance": {
            "status": "post_failure_cpu_fixture_pending_not_production_execution",
            "provenance_sha256": "9" * 64,
            "ledger": {"path": M.POST_FAILURE_CPU_LEDGER,
                       "sha256": "8" * 64, "size_bytes": 963},
            "events": ["proposed", "approved", "consumed"],
            "request_id": "request", "prereg_digest": "7" * 64,
            "cpu_result_sha256": "6" * 64, "context_sha256": "5" * 64,
            "choice_sha256": "4" * 64, "authorization_sha256": "3" * 64,
            "cpu_harness_sha256": "2" * 64,
            "sdk_started": 0, "api_requests": 0,
        },
    })
    monkeypatch.setattr(M.C, "_relative", lambda path: f"bench/protocols/{Path(path).name}")
    private_out, public_out = tmp_path / "private-plan.json", tmp_path / "public.json"
    value = M.build_protocol(
        old_protocol=old_protocol, old_anchor="a" * 40, old_private_plan=old_plan,
        old_phase=phase, old_inventory=old_inventory, old_closed_lease=old_lease,
        second_plan=components["second"], host_protocol=components["host"],
        compatibility_protocol=components["compat"], cache_protocol=components["cache"],
        inner_loop_protocol=components["loop"], inner_anchor="b" * 40,
        private_out=private_out, out=public_out)
    assert value["prior"]["cumulative_wall_us"] == 1_025_528_011
    assert value["prior"]["second_closed_lease_sha256"] == "e" * 64
    assert value["prior"]["post_failure_cpu_provenance"] == (
        json.loads(private_out.read_text(encoding="utf-8"))["prior_second"]
        ["post_failure_cpu_provenance"])
    assert value["budget"]["remaining_parent_wall_s"] == 42174.471989
    assert value["budget"]["api_cap"] == value["budget"]["sdk_cap"] == 32
    assert value["budget"]["per_arm_pair_cap"] == 3
    assert value["budget"]["targets"] == 4
    assert value["components"]["second_plan"] == {
        "sha256": M.C._sha(components["second"])}
    assert "path" not in value["components"]["second_plan"]
    assert str(tmp_path) not in public_out.read_text(encoding="utf-8")
    assert json.loads(private_out.read_text(encoding="utf-8"))["old_phase"] == str(phase.resolve())


def install_scopes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    files = {}
    for name in ("second", "host", "compat", "cache", "loop"):
        files[name] = save(tmp_path / f"{name}.json", {"name": name})
    plan = {
        "second_plan": str(files["second"]), "host_protocol": str(files["host"]),
        "compatibility_protocol": str(files["compat"]),
        "cache_protocol": str(files["cache"]),
        "inner_loop_protocol": str(files["loop"]), "inner_anchor": "b" * 40,
        "remaining_wall_s": 42174.471989,
    }
    sources = [{"case_id": "case-03", "policy": "fixed_eligible"}]
    monkeypatch.setattr(M, "verify_protocol", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(M.V4, "registered_sources", lambda _path: sources)
    monkeypatch.setattr(M.C, "_bound", lambda value, _label: Path(value))

    @contextmanager
    def scope(name: str, yielded=None):
        events.append("enter:" + name)
        try:
            yield yielded
        finally:
            events.append("exit:" + name)

    monkeypatch.setattr(M.V3.CACHE, "proof_scope", lambda **_kwargs: scope("cache", {}))
    monkeypatch.setattr(M.BIND, "epoch2_snapshot_scope",
                        lambda _sources: scope("parents", {"target": {}}))
    monkeypatch.setattr(M.PROBE_BOUNDARY, "probe_boundary_scope",
                        lambda *_args, **_kwargs: scope("boundary", {}))
    monkeypatch.setattr(M.V4, "remaining_wall_scope", lambda _value: scope("wall"))
    return events, files, plan


def test_execute_scope_order_closes_outer_cache_before_v3_nested_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _files, _plan = install_scopes(monkeypatch, tmp_path)

    def run(*_args, **_kwargs):
        events.append("run")
        assert events[:4] == ["enter:cache", "enter:parents", "exit:cache", "enter:boundary"]
        assert "enter:wall" in events
        return {"status": "complete"}

    monkeypatch.setattr(M.V3, "run_second_operational", run)
    result = M.execute(
        protocol=tmp_path / "protocol", anchor="c" * 40,
        private_plan=tmp_path / "plan", transport_root=tmp_path / "transport",
        lab=tmp_path / "lab", isaac_python=tmp_path / "python",
        out=tmp_path / "out", approver=M.C.RUN.APPROVER)
    assert result["status"] == "complete"
    assert events[-3:] == ["exit:wall", "exit:boundary", "exit:parents"]


def test_export_uses_same_parent_and_probe_boundary_scopes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _files, _plan = install_scopes(monkeypatch, tmp_path)

    def export(*_args, **_kwargs):
        events.append("export")
        assert events[:4] == ["enter:cache", "enter:parents", "exit:cache", "enter:boundary"]
        assert events[-2] == "enter:cache"
        return {"status": "complete"}

    monkeypatch.setattr(M.EXPORT, "export", export)
    result = M.export(
        protocol=tmp_path / "protocol", anchor="c" * 40,
        private_plan=tmp_path / "plan", first_result=tmp_path / "first",
        second_result=tmp_path / "second-result", lab=tmp_path / "lab",
        out=tmp_path / "public")
    assert result["status"] == "complete"
    assert "enter:wall" not in events
    assert events[-3:] == ["exit:cache", "exit:boundary", "exit:parents"]


def test_cli_failure_records_error_and_traceback(tmp_path: Path) -> None:
    out = tmp_path / "failed"
    try:
        raise ValueError("registered failure")
    except ValueError as error:
        M._record_failure(out, "run", error)
    value = json.loads((out / "parent-error.json").read_text(encoding="utf-8"))
    assert value["error"] == "registered failure"
    assert value["automatic_retry"] is False
    assert "ValueError" in (out / "parent-traceback.txt").read_text(encoding="utf-8")
