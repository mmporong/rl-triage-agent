# pyright: reportMissingImports=false
"""five-policy 중단 배치 continuation의 불변 inventory와 경계 회귀."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import resume_lineage_five_policy_v1 as M

ARMS = list(M.DRIVER.ARMS)


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def completed(path: Path, case_id: str, arm: str) -> None:
    write(path / case_id / arm / "result.json", {
        "schema": M.DRIVER.RESULT_SCHEMA, "scope": "cpu_fixture",
        "case_id": case_id, "arm": arm, "end_state": "HOLD",
        "scoring": {"correct_any_family": False},
    })


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    previous = tmp_path / "previous"
    previous.mkdir(parents=True)
    protocol_path = write(tmp_path / "protocol.json", {"fixture": True})
    config_path = write(tmp_path / "config.json", {"fixture": True})
    arm_order = {f"case-{number:02d}": list(ARMS) for number in range(1, 11)}
    arm_order["case-04"] = [
        "registry_next", "logs_only", "logs_plus_frames", "fixed_eligible",
        "optimizer_first_registered_rule",
    ]
    protocol = {
        "case_order": [f"case-{number:02d}" for number in range(1, 11)],
        "arm_order": arm_order,
        "source_digests": {f"case-{number:02d}": str(number % 10) * 64
                           for number in range(1, 11)},
        "sources_lf": {}, "tool_protocols": {},
    }
    sources = {}
    for case_id in protocol["case_order"]:
        source = {
            "case_id": case_id, "index": {"path": "fixture", "sha256": "a" * 64},
            "source_binding": {"checkpoint_sha256": "b" * 64},
            "qualification": {"arm": "F1", "diagnosis_eligible": True},
            "packet": {"input_bindings": {
                "case": {"checkpoint_sha256": "b" * 64},
                "reference": {"checkpoint_sha256": "c" * 64},
            }},
            "backend_packet": {},
        }
        sources[case_id] = {"source": source, "inputs": {}, "registered_input_roots": ()}
    ordered = M._ordered(protocol)
    assert ordered[18] == (M.PARTIAL_CASE, M.PARTIAL_ARM)
    for case_id, arm in ordered[:18]:
        completed(previous, case_id, arm)
    partial = previous / M.PARTIAL_CASE / M.PARTIAL_ARM
    first_snapshot = {"view_digest": "c" * 64,
                      "view": {"observations": [], "remaining_candidates": ["reward"]}}
    second_snapshot = {"view_digest": "d" * 64,
                       "view": {"observations": [{"probe": "P_episode", "outcome": "normal"}],
                                "remaining_candidates": ["reward"]}}
    first_choice = {"selected_by": "fixed_order", "fact_ids": [], "probe": "P_episode"}
    second_choice = {"selected_by": "fixed_order", "fact_ids": [], "probe": "P_noise"}
    write(partial / "step01/context.json", {"snapshot": first_snapshot})
    write(partial / "step01/choice.json", first_choice)
    write(partial / "step01/request-spec.json", {"probe": "P_episode"})
    write(partial / "step01/probe/result.json", {
        "probe": "P_episode", "outcome": "normal", "owned_child_wall_s": 2.5,
    })
    write(partial / "step02/context.json", {"snapshot": second_snapshot})
    write(partial / "step02/choice.json", second_choice)
    write(partial / "step02/request-spec.json", {"probe": "P_noise"})
    write(partial / "step02/busy-gate.json", {
        "decision": "continue", "dispatch_granted": False, "automatic_retry": False,
    })
    ledger = [
        {"event": event, "request_id": "one" if index < 4 else "two",
         "at": f"2026-11-01T00:0{index}:00+00:00"}
        for index, event in enumerate(
            ["proposed", "approved", "consumed", "receipt", "proposed", "approved", "consumed"]
        )
    ]
    (partial / "probes.ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger) + "\n", encoding="utf-8"
    )
    write(previous / "paired-intents/0018.json", {
        "case_id": M.PARTIAL_CASE, "arm": M.PARTIAL_ARM,
        "source_digest": M.B.digest(sources[M.PARTIAL_CASE]["source"]),
        "choice_digest": M.B.digest(second_choice),
        "snapshot_digest": M.B.digest(second_snapshot),
        "sdk_roles_reserved": 2, "automatic_retry": False,
    })
    for number in range(1, 18):
        write(previous / f"paired-intents/{number:04d}.json", {"fixture": number})
    write(previous / "failure.json", {
        "status": "invalid", "error_type": "PermissionError", "automatic_retry": False,
    })
    write(previous / "lease-initial.json", {
        "acquiredUtc": "2026-11-01T00:00:00+00:00", "leaseId": "old",
        "owner": M.LEASE.OWNER, "wrapperPid": 10, "childPid": 11,
    })
    stopped_lease = write(tmp_path / "lease-after-failure.json", {
        "acquiredUtc": "2026-11-01T00:00:00+00:00",
        "completedUtc": "2026-11-01T00:10:00+00:00", "leaseId": "old",
        "owner": M.LEASE.OWNER, "wrapperPid": 10, "childPid": 11,
        "state": "failed", "exitCode": 1,
    })
    (previous / "nim").mkdir()
    monkeypatch.setattr(M.BASE, "preflight", lambda *_args: (protocol, {"fixture": True}, sources))
    monkeypatch.setattr(M.NIM, "status", lambda _path: {
        "api_cap": 72, "reservations_charged": 7, "remaining": 65,
    })
    monkeypatch.setattr(M.DRIVER, "_score", lambda qualification, confirmed: {
        "correct_any_family": False, "wrong_confirmation": False,
        "compound_partial": False, "full_cause_confirmation": False,
        "repair_unsupported": False,
    })
    return {"previous": previous, "protocol_path": protocol_path,
            "config_path": config_path, "stopped_lease": stopped_lease,
            "protocol": protocol, "sources": sources}


def prepare_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, dict]:
    values = fixture(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan = M.prepare(values["protocol_path"], "a" * 40, values["config_path"],
                     values["previous"], values["stopped_lease"], plan_path,
                     allow_fixture=True)
    return plan, plan_path, values


def test_prepare_binds_completed_partial_pending_and_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _plan_path, values = prepare_one(tmp_path, monkeypatch)
    assert len(plan["completed_rows"]) == 18
    assert len(plan["pending_rows"]) == 31
    assert plan["partial_result"]["hold_reason"] == "lease_metadata_read_failed"
    assert plan["partial_result"]["infra_unknown"] == 1
    assert len(plan["partial_result"]["steps"]) == 1
    assert plan["partial_result"]["parent_wall_s"] == 360.0
    assert plan["partial_result"]["parent_wall_is_observed_lower_bound"] is True
    assert plan["partial_result"]["parent_wall_comparable_for_policy_cost"] is False
    assert plan["old_parent_wall_s"] == 600.0
    assert plan["old_model_reservations"] == 7
    assert plan["old_paired_intents"] == 18
    assert plan["requests_replayed"] == plan["model_api_calls"] == plan["gpu_executions"] == 0
    assert not (values["previous"] / "result.json").exists()


def test_prepared_inventory_rejects_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, plan_path, values = prepare_one(tmp_path, monkeypatch)
    (values["previous"] / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        M._verify_plan(plan_path)


def test_prepare_rejects_already_finished_unknown_pending_and_duplicate_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = fixture(tmp_path, monkeypatch)
    write(values["previous"] / "result.json", {"status": "complete"})
    with pytest.raises(ValueError, match="이미 완결"):
        M.prepare(values["protocol_path"], "a" * 40, values["config_path"],
                  values["previous"], values["stopped_lease"],
                  tmp_path / "finished.json", allow_fixture=True)

    values = fixture(tmp_path / "unknown", monkeypatch)
    (values["previous"] / "case-04/optimizer_first_registered_rule").mkdir(parents=True)
    with pytest.raises(ValueError, match="알 수 없는 부분"):
        M.prepare(values["protocol_path"], "a" * 40, values["config_path"],
                  values["previous"], values["stopped_lease"],
                  tmp_path / "unknown.json", allow_fixture=True)

    values = fixture(tmp_path / "duplicate", monkeypatch)
    protocol = deepcopy(values["protocol"])
    protocol["arm_order"]["case-01"][1] = protocol["arm_order"]["case-01"][0]
    monkeypatch.setattr(M.BASE, "preflight",
                        lambda *_args: (protocol, {"fixture": True}, values["sources"]))
    with pytest.raises(ValueError, match="중복"):
        M.prepare(values["protocol_path"], "a" * 40, values["config_path"],
                  values["previous"], values["stopped_lease"],
                  tmp_path / "duplicate.json", allow_fixture=True)


def test_prepare_rejects_unknown_pending_ledger_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = fixture(tmp_path, monkeypatch)
    ledger = values["previous"] / M.PARTIAL_CASE / M.PARTIAL_ARM / "probes.ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "receipt", "request_id": "two",
                                 "at": "2026-11-01T00:07:00+00:00"}) + "\n")
    with pytest.raises(ValueError, match=r"완료1\+미완료1"):
        M.prepare(values["protocol_path"], "a" * 40, values["config_path"],
                  values["previous"], values["stopped_lease"],
                  tmp_path / "bad-ledger.json", allow_fixture=True)


def test_public_protocol_binds_runner_lease_v2_and_previous_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _plan_path, _values = prepare_one(tmp_path, monkeypatch)
    protocol = M.continuation_protocol_value(plan)
    assert protocol["previous_inventory_sha256"] == plan["previous_inventory_sha256"]
    assert protocol["prepared_plan_digest"] == M.B.digest(dict(plan))
    assert protocol["old_model_reservations"] == 7
    assert protocol["old_paired_intents"] == 18
    assert set(protocol["sources_lf"]) == set(M.SOURCE_FILES)
    assert set(M.LEASE_V2.SOURCE_FILES) < set(M.SOURCE_FILES)
    assert protocol["automatic_infra_retry"] == protocol["request_replay"] == 0
    path = write(tmp_path / "continuation.json", protocol)
    assert M._verify_continuation_protocol(path, "b" * 40, plan, allow_fixture=True) == protocol
    changed = deepcopy(protocol)
    changed["pending_rows"] = 30
    write(path, changed)
    with pytest.raises(ValueError, match="prepared plan"):
        M._verify_continuation_protocol(path, "b" * 40, plan, allow_fixture=True)
    write(path, protocol)
    changed_plan = deepcopy(plan)
    changed_plan["caps"]["paired_intents"] -= 1
    with pytest.raises(ValueError, match="prepared plan"):
        M._verify_continuation_protocol(
            path, "b" * 40, changed_plan, allow_fixture=True
        )


def test_production_lease_boundary_is_existing_v2_module() -> None:
    assert M._LEASE is M.LEASE_V2
    assert M._LEASE.lease_context.__defaults__ == (None,)
    assert M.LEASE_V2._LEASE_CONTEXT is M.LEASE.lease_context


def test_comparison_status_preserves_original_missing_source_branch() -> None:
    assert M._comparison_status(0) == "eligible_for_registered_summary"
    assert M._comparison_status(1) == "inconclusive_input_unavailable"
    with pytest.raises(ValueError, match="unavailable"):
        M._comparison_status(-1)
