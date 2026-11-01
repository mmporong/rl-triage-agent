# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_five_policy_recovery_loop_v1 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def candidate(case: str, policy: str, status: str) -> dict:
    return {"schema": M.FIRST.SCHEMA, "case_id": case, "policy": policy,
            "status": status, "hold_reason": "model_hold" if status == "HOLD" else None}


def lease(lease_id: str = "lease-1") -> dict:
    return {"leaseId": lease_id, "owner": M.LEASE.OWNER}


def complete_second_rows(first: dict, tmp_path: Path) -> list[dict]:
    rows = [first]
    existing = {(first["case_id"], first["policy"])}
    for case, policy in M.TARGETS:
        if (case, policy) not in existing:
            rows.append({"case_id": case, "policy": policy,
                         "status": "not_in_second_subset", "source": None})
    return rows


def test_prepare_first_keeps_all_four_targets_and_hold(tmp_path: Path) -> None:
    seeds = {f"{case}:{policy}": 7000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    recovery = save(tmp_path / "recovery.json", {"recovery_seeds": seeds})
    inputs = [save(tmp_path / name, {}) for name in ("batch.json", "five.json", "config.json")]
    def prepare(_batch, *, case_id, policy, **_kwargs):
        return candidate(case_id, policy, "HOLD" if case_id == "case-03" else "validated_candidate")
    plan = M.prepare_first(inputs[0], five_policy_protocol=inputs[1], config=inputs[2],
                           recovery_protocol=recovery, anchor="a" * 40, out=tmp_path / "plan",
                           candidate_preparer=prepare, candidate_checker=lambda _path: {})
    assert len(plan["targets"]) == 4
    assert [row["status"] for row in plan["targets"]].count("HOLD") == 2
    assert plan["automatic_epoch2"] is False and plan["max_sdk_executions"] == 8


def test_run_first_checks_common_lease_and_preserves_hold(tmp_path: Path, monkeypatch) -> None:
    protocol = save(tmp_path / "recovery.json", {})
    config = save(tmp_path / "config.json", {})
    rows = []
    approvals = {}
    candidates = {}
    for index, (case, policy) in enumerate(M.TARGETS):
        status = "HOLD" if index else "validated_candidate"
        path = save(tmp_path / f"{case}-{policy}.json", candidate(case, policy, status))
        rows.append({"case_id": case, "policy": policy, "candidate": M._ref(path),
                     "status": status, "hold_reason": None, "registered_seed": 8000 + index})
        candidates[str(path.resolve())] = candidate(case, policy, status)
        if status != "HOLD":
            approval = save(tmp_path / "approval.json", {
                "schema": M.FIRST.APPROVAL_SCHEMA, "decision": "approve", "approver": "operator",
                "candidate": M._ref(path), "protocol": M._ref(protocol), "anchor": "a" * 40,
                "registered_seed": 8000 + index, "lease_id": "lease-1"})
            approvals[f"{case}:{policy}"] = approval
    plan_path = save(tmp_path / "plan.json", {"schema": M.FIRST_PLAN_SCHEMA,
        "recovery_protocol": M._ref(protocol), "config": M._ref(config), "anchor": "a" * 40,
        "targets": rows, "automatic_epoch2": False})
    monkeypatch.setattr(M.FIRST, "checked_candidate", lambda path: candidates[str(path.resolve())])
    def execute(*_args, out, **_kwargs):
        save(out / "result.json", {"outcome": "not_recovered"})
        epoch = save(out / "epoch2-input.json", {})
        return {"outcome": "not_recovered", "epoch2_input": M._ref(epoch)}
    result = M.run_first(plan_path, approvals, lab=tmp_path, isaac_python=tmp_path / "python",
                         out=tmp_path / "out", runner=lambda *_: {}, wait=lambda *_: None,
                         lease_reader=lease, candidate_executor=execute)
    assert result["executed_targets"] == 1 and result["sdk_execution_upper_bound"] == 2
    assert result["second_phase_requires_new_registered_protocol"] is True
    assert [row["outcome"] for row in result["records"]].count("HOLD") == 3


def test_run_first_rejects_lease_change(tmp_path: Path, monkeypatch) -> None:
    protocol = save(tmp_path / "recovery.json", {})
    config = save(tmp_path / "config.json", {})
    path = save(tmp_path / "candidate.json", candidate("case-03", "fixed_eligible", "validated_candidate"))
    approval = save(tmp_path / "approval.json", {
        "schema": M.FIRST.APPROVAL_SCHEMA, "decision": "approve", "approver": "operator",
        "candidate": M._ref(path), "protocol": M._ref(protocol), "anchor": "a" * 40,
        "registered_seed": 8, "lease_id": "lease-1"})
    rows = [{"case_id": "case-03", "policy": "fixed_eligible", "candidate": M._ref(path),
             "status": "validated_candidate", "hold_reason": None, "registered_seed": 8}]
    rows += [{"case_id": c, "policy": p, "candidate": M._ref(save(
        tmp_path / f"{c}-{p}.json", candidate(c, p, "HOLD"))), "status": "HOLD",
        "hold_reason": "hold", "registered_seed": 9} for c, p in M.TARGETS[1:]]
    plan = save(tmp_path / "plan.json", {"schema": M.FIRST_PLAN_SCHEMA,
        "recovery_protocol": M._ref(protocol), "config": M._ref(config), "anchor": "a" * 40,
        "targets": rows, "automatic_epoch2": False})
    monkeypatch.setattr(M.FIRST, "checked_candidate", lambda p: M._read(p))
    leases = iter([lease("lease-1"), lease("lease-2")])
    def execute(*_args, out, **_kwargs):
        save(out / "result.json", {"outcome": "recovered"})
        return {"outcome": "recovered", "epoch2_input": None}
    with pytest.raises(ValueError, match="lease"):
        M.run_first(plan, {"case-03:fixed_eligible": approval}, lab=tmp_path,
                    isaac_python=tmp_path / "python", out=tmp_path / "out",
                    runner=lambda *_: {}, wait=lambda *_: None,
                    lease_reader=lambda: next(leases), candidate_executor=execute)


def test_prepare_second_keeps_all_rows_and_only_registers_not_recovered(
        tmp_path: Path) -> None:
    epoch = save(tmp_path / "epoch.json", {})
    records = [{"case_id": "case-03", "policy": "fixed_eligible",
                "outcome": "not_recovered", "epoch2_input": M._ref(epoch)}]
    records += [{"case_id": c, "policy": p, "outcome": "HOLD", "epoch2_input": None}
                for c, p in M.TARGETS[1:]]
    first = save(tmp_path / "first.json", {"schema": M.FIRST_RESULT_SCHEMA,
                                           "status": "complete", "records": records})
    loop = save(tmp_path / "loop.json", {})
    source = {"case_id": "case-03", "policy": "fixed_eligible"}
    protocol = {"source_digests": {"case-03:fixed_eligible": M.SECOND.digest(source)},
                "second_seeds": {"case-03:fixed_eligible": 9001}}
    plan = M.prepare_second(first, loop_protocol=loop, anchor="a" * 40, lab=tmp_path,
                            out=tmp_path / "second", source_preparer=lambda *_a, **_k: source,
                            source_checker=lambda value: value,
                            protocol_verifier=lambda *_a, **_k: protocol)
    assert plan["registered_count"] == 1 and len(plan["records"]) == 4
    assert plan["probe_sdk_cap"] == 24 and plan["total_sdk_cap"] == 32


def test_run_second_holds_candidate_and_counts_probe_pairs(tmp_path: Path, monkeypatch) -> None:
    protocol = save(tmp_path / "loop.json", {})
    source_path = save(tmp_path / "source.json", {"case_id": "case-03", "policy": "logs_only"})
    plan = save(tmp_path / "plan.json", {"schema": M.SECOND_PLAN_SCHEMA, "api_cap": 32,
        "loop_protocol": M._ref(protocol), "anchor": "a" * 40, "records": complete_second_rows(
            {"case_id": "case-03", "policy": "logs_only", "status": "registered",
             "source": M._ref(source_path), "registered_seed": 9001}, tmp_path)})
    monkeypatch.setattr(M.SECOND, "checked_source", lambda value: value)
    monkeypatch.setattr(M.TRANSPORT, "initialize", lambda *_a, **_k: {})
    monkeypatch.setattr(M.TRANSPORT, "status", lambda _root: {"reservations_charged": 1})
    def diagnosis(*_args, output, **_kwargs):
        result = {"steps": [{"probe": "P_reward"}]}
        save(output / "result.json", result)
        return result
    hold = {"status": "HOLD", "hold_reason": "model_hold"}
    result = M.run_second(plan, {}, transport_root=tmp_path / "transport", lab=tmp_path,
                          isaac_python=tmp_path / "python", out=tmp_path / "out",
                          send=lambda *_: {}, execute_probe=lambda *_: {},
                          busy_reader=lambda *_: {}, runner=lambda *_: {}, wait=lambda *_: None,
                          lease_reader=lease,
                          diagnosis_runner=diagnosis,
                          candidate_preparer=lambda *_a, **_k: hold,
                          candidate_executor=lambda *_a, **_k: pytest.fail("HOLD execution"))
    assert result["probe_pairs"] == 1 and result["probe_sdk_upper_bound"] == 2
    assert result["executed_second_targets"] == 0 and result["records"][0]["outcome"] == "HOLD"


def test_run_second_rejects_source_changed_after_plan(tmp_path: Path, monkeypatch) -> None:
    protocol = save(tmp_path / "loop.json", {})
    source = save(tmp_path / "source.json", {"case_id": "case-03", "policy": "logs_only"})
    plan = save(tmp_path / "plan.json", {"schema": M.SECOND_PLAN_SCHEMA, "api_cap": 32,
        "loop_protocol": M._ref(protocol), "anchor": "a" * 40, "records": complete_second_rows(
            {"case_id": "case-03", "policy": "logs_only", "status": "registered",
             "source": M._ref(source), "registered_seed": 9001}, tmp_path)})
    save(source, {"case_id": "case-03", "policy": "logs_only", "changed": True})
    monkeypatch.setattr(M.TRANSPORT, "initialize", lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="SHA"):
        M.run_second(plan, {}, transport_root=tmp_path / "transport", lab=tmp_path,
                     isaac_python=tmp_path / "python", out=tmp_path / "out",
                     send=lambda *_: {}, execute_probe=lambda *_: {}, busy_reader=lambda *_: {},
                     runner=lambda *_: {}, wait=lambda *_: None,
                     lease_reader=lease)


def test_run_second_passes_registered_seed_and_paths_to_executor(
        tmp_path: Path, monkeypatch) -> None:
    protocol = save(tmp_path / "loop.json", {})
    source = save(tmp_path / "source.json", {"case_id": "case-07", "policy": "fixed_eligible"})
    plan = save(tmp_path / "plan.json", {"schema": M.SECOND_PLAN_SCHEMA, "api_cap": 32,
        "loop_protocol": M._ref(protocol), "anchor": "a" * 40, "records": complete_second_rows(
            {"case_id": "case-07", "policy": "fixed_eligible", "status": "registered",
             "source": M._ref(source), "registered_seed": 9123}, tmp_path)})
    monkeypatch.setattr(M.SECOND, "checked_source", lambda value: value)
    monkeypatch.setattr(M.TRANSPORT, "initialize", lambda *_a, **_k: {})
    monkeypatch.setattr(M.TRANSPORT, "status", lambda _root: {"reservations_charged": 0})
    monkeypatch.setattr(M, "_approval", lambda *_a, **_k: {})
    approval = save(tmp_path / "approval.json", {})
    def diagnosis(*_args, output, **_kwargs):
        result = {"steps": []}
        save(output / "result.json", result)
        return result
    seen = {}
    def execute(candidate_path, protocol_path, approval_path, **kwargs):
        seen.update(candidate_path=candidate_path, protocol_path=protocol_path,
                    approval_path=approval_path, **kwargs)
        save(kwargs["out"] / "result.json", {"outcome": "recovered"})
        return {"outcome": "recovered"}
    result = M.run_second(
        plan, {"case-07:fixed_eligible": approval}, transport_root=tmp_path / "transport",
        lab=tmp_path, isaac_python=tmp_path / "python", out=tmp_path / "out",
        send=lambda *_: {}, execute_probe=lambda *_: {}, busy_reader=lambda *_: {},
        runner=lambda *_: {}, wait=lambda *_: None,
        lease_reader=lease, diagnosis_runner=diagnosis,
        candidate_preparer=lambda *_a, **_k: {
            "status": "validated_candidate", "second_correction": "restore_x"},
        candidate_executor=execute)
    assert seen["registered_seed"] == 9123 and seen["anchor"] == "a" * 40
    assert seen["protocol_path"] == protocol.resolve() and seen["approval_path"] == approval.resolve()
    assert result["executed_second_targets"] == 1


def test_host_approval_factory_binds_candidate_seed_protocol_and_lease(tmp_path: Path) -> None:
    protocol = save(tmp_path / "protocol.json", {})
    candidate_path = save(tmp_path / "candidate.json", {})
    approval = M._approval_for(
        candidate_path, protocol, schema="approval-v1", anchor="a" * 40, seed=9910,
        lease=lease(), approvals=None, key="case-03:fixed_eligible",
        approver=M.APPROVER, root=tmp_path / "run")
    value = M._approval(approval, schema="approval-v1", candidate=candidate_path,
                        protocol=protocol, anchor="a" * 40, seed=9910,
                        lease_id="lease-1")
    assert value["generated_by"] == "host_operator_boundary"
    assert value["model_generated"] is False and value["protocol_generated"] is False
    with pytest.raises(FileExistsError):
        M._approval_for(candidate_path, protocol, schema="approval-v1", anchor="a" * 40,
                        seed=9910, lease=lease(), approvals=None,
                        key="case-03:fixed_eligible", approver=M.APPROVER,
                        root=tmp_path / "run")


@pytest.mark.parametrize("command", ["run-first", "run-second"])
def test_run_cli_uses_operational_entry_and_records_failure(
        tmp_path: Path, monkeypatch, command: str) -> None:
    seen = {}
    def fail(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        raise RuntimeError("bounded failure")
    if command == "run-first":
        monkeypatch.setattr(M, "run_first_operational", fail)
        argv = [command, "--plan", str(tmp_path / "plan.json"), "--lab", str(tmp_path),
                "--isaac-python", str(tmp_path / "python"), "--out", str(tmp_path / "out"),
                "--approver", M.APPROVER]
    else:
        monkeypatch.setattr(M, "run_second_operational", fail)
        argv = [command, "--plan", str(tmp_path / "plan.json"),
                "--transport-root", str(tmp_path / "transport"), "--lab", str(tmp_path),
                "--isaac-python", str(tmp_path / "python"), "--out", str(tmp_path / "out"),
                "--approver", M.APPROVER]
    with pytest.raises(RuntimeError, match="bounded failure"):
        M.main(argv)
    failure = M._read(tmp_path / "out" / "parent-error.json")
    assert seen["kwargs"]["approver"] == M.APPROVER
    assert failure["command"] == command and failure["automatic_retry"] is False


def test_operational_first_resolves_real_common_lease_and_owned_signatures(tmp_path, monkeypatch):
    seen = {}
    def first(plan, approvals, **kwargs):
        seen.update(plan=plan, approvals=approvals, **kwargs)
        return {"scope": "cpu_fixture_wiring"}
    monkeypatch.setattr(M, "run_first", first)
    result = M.run_first_operational(tmp_path / "plan.json", lab=tmp_path,
                                     isaac_python=tmp_path / "python.bat", out=tmp_path / "out",
                                     approver=M.APPROVER)
    assert result["scope"] == "cpu_fixture_wiring"
    assert seen["approvals"] is None and seen["lease_reader"] is M.LEASE.lease_context
    assert seen["runner"] is M.OWNED.run_owned and seen["wait"] is M.WAIT._lease_wait
    import inspect
    inspect.signature(seen["lease_reader"]).bind()
    inspect.signature(seen["runner"]).bind(tmp_path, ["not-executed"], tmp_path, 1.0)
    inspect.signature(seen["wait"]).bind(tmp_path, max_wait_s=1.0)
