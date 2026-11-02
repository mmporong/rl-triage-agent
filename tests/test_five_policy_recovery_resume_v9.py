# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_five_policy_recovery_resume_v9 as M

PRIVATE = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025"
           "/vision_goal/ri_full_completion_251/replan_369/recovery_actual_414")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_source_closure_adds_training_boundary_and_v9_only() -> None:
    assert len(M.SOURCE_FILES) == 40
    assert set(M.V8.SOURCE_FILES) < set(M.SOURCE_FILES)
    assert set(M.SOURCE_FILES) - set(M.V8.SOURCE_FILES) == {
        "evals/recovery_epoch2_training_boundary_v1.py",
        "tests/test_recovery_epoch2_training_boundary_v1.py",
        "bench/run_five_policy_recovery_resume_v9.py",
        "tests/test_five_policy_recovery_resume_v9.py",
    }


def test_actual_training_failure_seals_902_files_and_costs() -> None:
    base = PRIVATE / "training-source-retry-v1"
    value = M._failure_state(
        PRIVATE / "receipt-followup-retry-v1/execute-v1",
        base / "old-failure-inventory-v1.json", base / "closed-lease-v1.json")
    assert value["inventory_digest"] == M.EXPECTED_INVENTORY_DIGEST
    assert value["old_wall_us"] == 6_950_256_152
    assert value["cumulative_wall_us"] == 10_783_745_328
    assert value["remaining_wall_s"] == 32_416.254672
    assert value["prior_api_calls"] == value["prior_nim_calls"] == 6
    assert value["prior_physical_roles"] == 24
    assert value["new_training_physical_cap"] == 2
    assert len(value["records"]) == 4


def test_failure_state_rejects_changed_inventory(tmp_path: Path) -> None:
    base = PRIVATE / "training-source-retry-v1"
    inventory = json.loads((base / "old-failure-inventory-v1.json").read_text(
        encoding="utf-8"))
    inventory["cumulative_physical_roles"] = 22
    changed = save(tmp_path / "inventory.json", inventory)
    with pytest.raises(ValueError, match="API/SDK"):
        M._failure_state(PRIVATE / "receipt-followup-retry-v1/execute-v1",
                         changed, base / "closed-lease-v1.json")


def failure_fixture(tmp_path: Path) -> dict:
    records = []
    for case, policy, status in (
        ("case-03", "fixed_eligible", "HOLD"),
        ("case-03", "logs_only", "HOLD"),
        ("case-07", "fixed_eligible", "HOLD"),
        ("case-07", "logs_only", "validated_candidate"),
    ):
        key = f"{case}_{policy}"
        candidate = save(tmp_path / f"{key}-candidate.json", {"candidate": key})
        diagnosis = save(tmp_path / f"{key}-diagnosis.json", {"diagnosis": key})
        records.append({
            "key": key, "candidate": M.C._ref(candidate),
            "diagnosis": M.C._ref(diagnosis), "candidate_status": status,
            "confirmed_family": "optimizer_update_count" if status != "HOLD" else None,
            "second_correction": (
                "restore_configured_ppo_epochs" if status != "HOLD" else None),
            "end_state": "confirmed" if status != "HOLD" else "HOLD",
            "probes": [],
        })
    return {
        "inventory_digest": "a" * 64, "inventory_sha256": "b" * 64,
        "closed_lease_sha256": "c" * 64, "lease_id": "old-lease",
        "records": records, "old_wall_us": M.EXPECTED_FAILURE_WALL_US,
        "cumulative_wall_us": M.EXPECTED_CUMULATIVE_WALL_US,
        "remaining_wall_s": 32_416.254672, "prior_api_calls": 6,
        "prior_physical_roles": 24, "prior_nim_calls": 6,
        "new_training_physical_cap": 2, "new_api_cap": 0,
    }


def test_build_binds_training_component_without_private_public_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = save(tmp_path / "parent.json", {"parent": True})
    parent_plan = save(tmp_path / "parent-plan.json", {"parent": True})
    second_plan = save(tmp_path / "second-plan.json", {
        "loop_protocol": M.C._ref(save(tmp_path / "loop.json", {"loop": True})),
        "anchor": "d" * 40})
    training = save(tmp_path / "training.json", {"training": True})
    inventory = save(tmp_path / "inventory.json", {})
    lease = save(tmp_path / "lease.json", {})
    phase = tmp_path / "phase"
    phase.mkdir()
    monkeypatch.setattr(M.V8, "verify_protocol", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(M.V8, "_parents",
                        lambda _parent: ({}, {}, {}, second_plan, []))
    monkeypatch.setattr(M.TRAIN_BOUNDARY, "verify_protocol",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(M, "_failure_state", lambda *_args: failure_fixture(tmp_path))
    monkeypatch.setattr(M.C, "_relative", lambda path: f"bench/protocols/{Path(path).name}")
    public = tmp_path / "public.json"
    value = M.build_protocol(
        parent_protocol=parent, parent_anchor="e" * 40,
        parent_private_plan=parent_plan, old_phase=phase,
        old_inventory=inventory, old_closed_lease=lease,
        training_protocol=training, training_anchor="f" * 40,
        private_out=tmp_path / "private.json", out=public)
    assert value["budget"]["cumulative_sdk_upper_bound"] == 26
    assert value["budget"]["new_api_calls"] == 0
    assert value["preserved_diagnosis_count"] == 4
    assert value["training_component"]["anchor"] == "f" * 40
    assert str(tmp_path) not in public.read_text(encoding="utf-8")


def install_execute_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    failure = failure_fixture(tmp_path)
    loop = save(tmp_path / "loop.json", {"loop": True})
    rows = []
    candidates: dict[Path, dict] = {}
    for index, old in enumerate(failure["records"]):
        case, policy = old["key"].split("_", 1)
        source = save(tmp_path / f"source-{index}.json", {"source": index})
        row = {
            "case_id": case, "policy": policy, "status": "registered",
            "source": M.C._ref(source),
            "registered_seed": M.TARGET_SEED if index == 3 else 5000 + index,
        }
        rows.append(row)
        candidate_path = M.C._bound(old["candidate"], "candidate")
        candidates[candidate_path] = {
            "case_id": case, "policy": policy, "status": old["candidate_status"],
            "inputs": {"source": row["source"], "diagnosis_result": old["diagnosis"]},
        }
    second_plan = save(tmp_path / "second-plan.json", {
        "schema": M.RUN.SECOND_PLAN_SCHEMA, "loop_protocol": M.C._ref(loop),
        "anchor": "1" * 40, "records": rows})
    training = save(tmp_path / "training.json", {"training": True})
    plan = {
        "parent_protocol": M.C._ref(save(tmp_path / "parent.json", {})),
        "parent_private_plan": M.C._ref(save(tmp_path / "parent-private.json", {})),
        "parent_anchor": "2" * 40, "training_protocol": M.C._ref(training),
        "training_anchor": "3" * 40, "prior_failure": failure,
    }
    monkeypatch.setattr(M, "verify_protocol", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(M.V8, "verify_protocol", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(M.V8, "_parents",
                        lambda _parent: ({}, {}, {}, second_plan, []))
    monkeypatch.setattr(M.V8, "_scopes", lambda *_args: events.append("scopes"))
    monkeypatch.setattr(M.SECOND, "checked_second_candidate",
                        lambda path: deepcopy(candidates[Path(path).resolve()]))

    cache_depth = {"value": 0}

    @contextmanager
    def cache(**_kwargs):
        cache_depth["value"] += 1
        events.append("cache+")
        try:
            yield
        finally:
            cache_depth["value"] -= 1
            events.append("cache-")

    @contextmanager
    def plain(name: str):
        events.append(name + "+")
        try:
            yield
        finally:
            events.append(name + "-")

    @contextmanager
    def training_scope(**_kwargs):
        assert cache_depth["value"] == 0
        events.append("training+")
        try:
            yield
        finally:
            events.append("training-")

    monkeypatch.setattr(M.CACHE, "proof_scope", cache)
    monkeypatch.setattr(M.V5.V4, "remaining_wall_scope", lambda _wall: plain("wall"))
    monkeypatch.setattr(M.PARENT.HOST, "retry_lease_binding", lambda: plain("host"))
    monkeypatch.setattr(M.PARENT.OUTCOME, "recovery_outcome_scope", lambda: plain("outcome"))
    monkeypatch.setattr(M.TRAIN_BOUNDARY, "training_boundary_scope", training_scope)
    lease = {"leaseId": "new-lease", "owner": "owner"}
    monkeypatch.setattr(M.RUN, "_lease", lambda _reader: lease)
    monkeypatch.setattr(M.RUN.LEASE, "lease_context", lambda: lease)
    monkeypatch.setattr(M.RUN, "_same_lease", lambda *_args: None)

    def approval_for(*_args, **kwargs):
        events.append("approval")
        return save(kwargs["root"] / "approval.json", {"approved": True})

    monkeypatch.setattr(M.RUN, "_approval_for", approval_for)
    monkeypatch.setattr(M.RUN, "_approval", lambda *_args, **_kwargs: None)

    def execute(candidate, protocol, approval, **kwargs):
        assert events[-1] == "training+"
        assert kwargs["runner"] is M.RUN.OWNED.run_owned
        assert kwargs["wait"] is M.RUN.WAIT._lease_wait
        assert kwargs["registered_seed"] == M.TARGET_SEED
        events.append("execute")
        save(kwargs["out"] / "result.json", {"result": True})
        return {"outcome": "not_recovered"}

    monkeypatch.setattr(M.SECOND, "execute_second_candidate", execute)
    return events, plan


def test_execute_rechecks_four_candidates_and_runs_only_seed5557_in_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _plan = install_execute_fixture(monkeypatch, tmp_path)
    result = M.execute(
        protocol=tmp_path / "protocol", anchor="4" * 40,
        private_plan=tmp_path / "private", lab=tmp_path / "lab",
        isaac_python=tmp_path / "python", out=tmp_path / "out",
        approver=M.RUN.APPROVER)
    assert [row["outcome"] for row in result["records"]] == [
        "HOLD", "HOLD", "HOLD", "not_recovered"]
    assert result["schema"] == M.RUN.SECOND_RESULT_SCHEMA
    assert result["total_sdk_upper_bound"] == 26
    assert result["new_api_calls"] == result["model_or_probe_reruns"] == 0
    assert events.count("execute") == events.count("training+") == 1
    assert events.index("cache-") < events.index("training+")


def test_execute_exhausted_wall_refuses_before_approval(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, plan = install_execute_fixture(monkeypatch, tmp_path)
    plan["prior_failure"]["remaining_wall_s"] = 0
    with pytest.raises(ValueError, match="wall 예약 부족"):
        M.execute(protocol=tmp_path / "protocol", anchor="4" * 40,
                  private_plan=tmp_path / "private", lab=tmp_path / "lab",
                  isaac_python=tmp_path / "python", out=tmp_path / "out",
                  approver=M.RUN.APPROVER)
    assert "approval" not in events
    assert "execute" not in events


def test_checked_records_rejects_a_second_runnable_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _events, plan = install_execute_fixture(monkeypatch, tmp_path)
    second_plan = tmp_path / "second-plan.json"
    bad = deepcopy(plan["prior_failure"])
    bad["records"][0]["candidate_status"] = "validated_candidate"
    candidates = {}
    for old in bad["records"]:
        candidate = M.C._bound(old["candidate"], "candidate")
        candidates[candidate] = {
            "case_id": "case-03" if "case-03" in old["key"] else "case-07",
            "policy": "fixed_eligible" if "fixed_eligible" in old["key"] else "logs_only",
            "status": old["candidate_status"], "inputs": {}}
    original = M.SECOND.checked_second_candidate

    def changed(path: Path):
        value = deepcopy(original(path))
        if Path(path).resolve() == M.C._bound(bad["records"][0]["candidate"], "candidate"):
            value["status"] = "validated_candidate"
        return value

    monkeypatch.setattr(M.SECOND, "checked_second_candidate", changed)
    with pytest.raises(ValueError, match="유일 미실행"):
        M._checked_records(bad, second_plan)

