from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_five_policy_recovery_resume_v2 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return M._ref(path)


def rows(tmp_path: Path) -> list[dict]:
    result = []
    seeds = [4552, 4553, 4556, 4557]
    for (case_id, policy), seed in zip(M.RUN.TARGETS, seeds, strict=True):
        candidate = save(tmp_path / f"{case_id}_{policy}.json", {
            "case_id": case_id, "policy": policy, "status": "validated_candidate"})
        result.append({"case_id": case_id, "policy": policy, "candidate": ref(candidate),
                       "status": "validated_candidate", "hold_reason": None,
                       "registered_seed": seed})
    return result


def test_old_elapsed_uses_closed_lease_timestamps() -> None:
    lease = {"acquiredUtc": "2026-11-01T23:38:32.0386329Z",
             "completedUtc": "2026-11-02T00:03:31.3547212Z"}
    assert M._old_elapsed(lease) == 1499.316089


def test_run_first_reuses_prior_and_executes_only_remaining_three(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = save(tmp_path / "first-protocol.json", {})
    plan_path = save(tmp_path / "plan.json", {
        "schema": M.RUN.FIRST_PLAN_SCHEMA, "anchor": M.ORIGINAL_FIRST_ANCHOR,
        "recovery_protocol": ref(protocol), "targets": rows(tmp_path)})
    epoch = save(tmp_path / "old-epoch.json", {})
    old_result_path = save(tmp_path / "old-result.json", {
        "schema": M.FIRST.RESULT_SCHEMA, "status": "complete", "outcome": "not_recovered",
        "epoch2_input": ref(epoch), "execution_kind": "verified_prior_roles_finalize"})
    overlay_path = save(tmp_path / "overlay.json", {})
    continuation_plan = save(tmp_path / "continuation-plan.json", {})
    overlay = {
        "prior": {"lease_id": "old-lease"},
        "budgets": {"new_sdk_execution_cap": 6, "remaining_wall_s": 12_000.0,
                    "old_lease_elapsed_s": 1499.0},
    }
    private = {"original_plan": ref(plan_path), "finalized_result": ref(old_result_path)}
    monkeypatch.setattr(M.FIRST, "checked_candidate", lambda path: M._read(path))
    monkeypatch.setattr(M.RUN, "_approval", lambda *_a, **_k: {})
    approvals = {}
    for row in M._read(plan_path)["targets"][1:]:
        approvals[f"{row['case_id']}:{row['policy']}"] = save(
            tmp_path / f"approval-{row['registered_seed']}.json", {})
    calls = []
    def execute(candidate_path, _protocol, _approval, *, out, registered_seed, **_kwargs):
        calls.append((Path(candidate_path).name, registered_seed))
        epoch_path = save(out / "epoch2-input.json", {"seed": registered_seed})
        save(out / "result.json", {"outcome": "not_recovered",
                                     "epoch2_input": ref(epoch_path)})
        return {"outcome": "not_recovered", "epoch2_input": ref(epoch_path)}
    lease = lambda: {"leaseId": "new-lease", "owner": M.RUN.LEASE.OWNER}
    result = M.run_first(
        plan_path, overlay_path, continuation_plan, approvals,
        execution_anchor="b" * 40, lab=tmp_path,
        isaac_python=tmp_path / "python", out=tmp_path / "out", runner=lambda *_: {},
        wait=lambda *_: None, lease_reader=lease, candidate_executor=execute,
        protocol_verifier=lambda *_a, **_k: {"protocol": overlay, "plan": private},
        use_host_binding=False)
    assert [row["execution_kind"] for row in result["records"]] == [
        "verified_prior_roles_finalize", "new_registered_execution",
        "new_registered_execution", "new_registered_execution"]
    assert result["records"][0]["lease_id"] == "old-lease"
    assert {row["lease_id"] for row in result["records"][1:]} == {"new-lease"}
    assert result["lease_id"] == "new-lease" and len(calls) == 3
    assert result["old_sdk_executions"] == 2 and result["new_sdk_executions"] == 6
    assert result["total_sdk_execution_upper_bound"] == 8 and result["api_calls"] == 0


def test_build_continuation_accounts_old_lease_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_protocol = save(tmp_path / "first-protocol.json", {})
    host_protocol = save(tmp_path / "host-protocol.json", {})
    plan = save(tmp_path / "plan.json", {
        "schema": M.RUN.FIRST_PLAN_SCHEMA, "anchor": M.ORIGINAL_FIRST_ANCHOR,
        "recovery_protocol": ref(first_protocol), "targets": rows(tmp_path)})
    finalized = save(tmp_path / "finalized.json", {
        "schema": M.FIRST.RESULT_SCHEMA,
        "execution_kind": "verified_prior_roles_finalize"})
    lease = save(tmp_path / "closed-lease.json", {
        "leaseId": "old", "acquiredUtc": "2026-11-01T23:38:32.0386329Z",
        "completedUtc": "2026-11-02T00:03:31.3547212Z"})
    inventory = save(tmp_path / "inventory.json", {"files": 49})
    old_phase = tmp_path / "old-phase"
    old_phase.mkdir()
    monkeypatch.setattr(M, "_source_map", lambda: {"parent.py": "1" * 64})
    monkeypatch.setattr(M, "_relative", lambda path: Path(path).name)
    protocol = M.build_continuation_protocol(
        plan_path=plan, first_protocol=first_protocol,
        original_host_protocol=host_protocol, old_failed_phase=old_phase,
        finalized_result=finalized, old_closed_lease=lease,
        inventory_path=inventory, continuation_plan_out=tmp_path / "private-plan.json",
        out=tmp_path / "continuation.json")
    assert protocol["budgets"] == {
        "api_calls": 0, "first_wall_s": 14_400.0,
        "old_lease_elapsed_s": 1499.316089,
        "remaining_wall_s": 12_900.683911,
        "old_sdk_executions": 2, "new_sdk_execution_cap": 6,
        "total_sdk_execution_cap": 8}
    assert protocol["prior"]["lease_id"] == "old"
    assert protocol["original"]["host_anchor"] == M.ORIGINAL_HOST_ANCHOR
    assert len(protocol["targets"]) == 4


def test_run_first_rejects_changed_new_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = save(tmp_path / "first-protocol.json", {})
    plan_path = save(tmp_path / "plan.json", {
        "schema": M.RUN.FIRST_PLAN_SCHEMA, "anchor": M.ORIGINAL_FIRST_ANCHOR,
        "recovery_protocol": ref(protocol), "targets": rows(tmp_path)})
    old_result = save(tmp_path / "old-result.json", {
        "schema": M.FIRST.RESULT_SCHEMA, "status": "complete", "outcome": "recovered",
        "epoch2_input": None, "execution_kind": "verified_prior_roles_finalize"})
    overlay_path = save(tmp_path / "overlay.json", {})
    continuation_plan = save(tmp_path / "continuation-plan.json", {})
    overlay = {"prior": {"lease_id": "old"},
               "budgets": {"new_sdk_execution_cap": 6, "remaining_wall_s": 12_000.0,
                           "old_lease_elapsed_s": 1.0}}
    private = {"original_plan": ref(plan_path), "finalized_result": ref(old_result)}
    monkeypatch.setattr(M.FIRST, "checked_candidate", lambda path: M._read(path))
    monkeypatch.setattr(M.RUN, "_approval", lambda *_a, **_k: {})
    approval = save(tmp_path / "approval.json", {})
    leases = iter([{"leaseId": "new", "owner": M.RUN.LEASE.OWNER},
                   {"leaseId": "changed", "owner": M.RUN.LEASE.OWNER}])
    def execute(*_args, out, **_kwargs):
        save(out / "result.json", {"outcome": "recovered"})
        return {"outcome": "recovered", "epoch2_input": None}
    with pytest.raises(ValueError, match="lease"):
        M.run_first(
            plan_path, overlay_path, continuation_plan,
            {f"{M.RUN.TARGETS[1][0]}:{M.RUN.TARGETS[1][1]}": approval},
            execution_anchor="b" * 40, lab=tmp_path, isaac_python=tmp_path / "python",
            out=tmp_path / "out", runner=lambda *_: {}, wait=lambda *_: None,
            lease_reader=lambda: next(leases), candidate_executor=execute,
            protocol_verifier=lambda *_a, **_k: {"protocol": overlay, "plan": private},
            use_host_binding=False)


def test_prepare_second_accepts_v2_without_v1_projection(tmp_path: Path) -> None:
    epoch = save(tmp_path / "epoch.json", {})
    leaf = save(tmp_path / "leaf.json", {"outcome": "not_recovered",
                                          "epoch2_input": ref(epoch)})
    records = []
    for index, (case_id, policy) in enumerate(M.RUN.TARGETS):
        records.append({"case_id": case_id, "policy": policy,
                        "outcome": "not_recovered" if index == 0 else "recovered",
                        "result": ref(leaf) if index == 0 else None,
                        "epoch2_input": ref(epoch) if index == 0 else None})
    first = save(tmp_path / "first-v2.json", {
        "schema": M.FIRST_CONTINUATION_RESULT_SCHEMA, "status": "complete",
        "records": records})
    loop = save(tmp_path / "loop.json", {})
    source = {"case_id": "case-03", "policy": "fixed_eligible"}
    registered = {"source_digests": {"case-03:fixed_eligible": M.SECOND.digest(source)},
                  "second_seeds": {"case-03:fixed_eligible": 5552}}
    plan = M.prepare_second(
        first, loop_protocol=loop, anchor="c" * 40, lab=tmp_path,
        out=tmp_path / "second", source_preparer=lambda *_a, **_k: source,
        source_checker=lambda value: value,
        protocol_verifier=lambda *_a, **_k: registered)
    assert plan["schema"] == M.RUN.SECOND_PLAN_SCHEMA
    assert plan["first_result"] == ref(first) and plan["registered_count"] == 1
    assert plan["records"][0]["registered_seed"] == 5552


def test_prepare_second_rejects_leaf_epoch_mismatch(tmp_path: Path) -> None:
    epoch = save(tmp_path / "epoch.json", {})
    other = save(tmp_path / "other.json", {})
    leaf = save(tmp_path / "leaf.json", {"outcome": "not_recovered",
                                          "epoch2_input": ref(other)})
    records = [{"case_id": c, "policy": p,
                "outcome": "not_recovered" if i == 0 else "recovered",
                "result": ref(leaf) if i == 0 else None,
                "epoch2_input": ref(epoch) if i == 0 else None}
               for i, (c, p) in enumerate(M.RUN.TARGETS)]
    first = save(tmp_path / "first-v2.json", {
        "schema": M.FIRST_CONTINUATION_RESULT_SCHEMA, "status": "complete",
        "records": records})
    loop = save(tmp_path / "loop.json", {})
    with pytest.raises(ValueError, match="receipt/epoch2"):
        M.prepare_second(first, loop_protocol=loop, anchor="c" * 40, lab=tmp_path,
                         out=tmp_path / "second",
                         protocol_verifier=lambda *_a, **_k: {})


def test_second_operational_requires_same_host_and_compat_anchor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor"):
        M.run_second_operational(
            tmp_path / "plan.json", host_protocol=tmp_path / "host.json",
            host_anchor="a" * 40, compatibility_protocol=tmp_path / "compat.json",
            compatibility_anchor="b" * 40, transport_root=tmp_path / "transport",
            lab=tmp_path, isaac_python=tmp_path / "python", out=tmp_path / "out",
            approver=M.RUN.APPROVER)
