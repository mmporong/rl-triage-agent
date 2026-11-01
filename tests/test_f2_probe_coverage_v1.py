from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_f2_probe_coverage_v1", ROOT / "bench/run_f2_probe_coverage_v1.py"
)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def file_ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source(case_id: str, seed: int) -> dict:
    return {
        "case_id": case_id,
        "qualification": {"arm": "F2"},
        "packet": {"input_bindings": {
            "case": {"checkpoint_sha256": f"case-{seed}", "telemetry_sha256": "tc"},
            "reference": {"checkpoint_sha256": f"ref-{seed}", "telemetry_sha256": "tr"},
        }},
        "backend_packet": {"fixture": True},
    }


def checked(role: str, seed: int):
    arm = "F2" if role == "case" else "NONE"
    checkpoint = f"case-{seed}" if role == "case" else f"ref-{seed}"
    return ({"binding": {"checkpoint_sha256": checkpoint}},
            {"manifest": {"arm": arm, "seed": seed}})


def test_target_material_rejects_wrong_label_checkpoint_and_role(tmp_path: Path):
    inputs = {"case": tmp_path / "case.json", "reference": tmp_path / "ref.json"}
    good = source("case-05", 2554)

    def checker(path: Path):
        return checked("case" if path == inputs["case"] else "reference", 2554)

    assert M._target_material(good, inputs, case_id="case-05", checker=checker)[
        "case_checkpoint_sha256"] == "case-2554"
    bad_label = deepcopy(good)
    bad_label["qualification"]["arm"] = "F1"
    with pytest.raises(ValueError, match="label"):
        M._target_material(bad_label, inputs, case_id="case-05", checker=checker)
    bad_cp = deepcopy(good)
    bad_cp["packet"]["input_bindings"]["reference"]["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint"):
        M._target_material(bad_cp, inputs, case_id="case-05", checker=checker)

    def wrong_role(path: Path):
        return checked("case", 2554)

    with pytest.raises(ValueError, match="role/seed"):
        M._target_material(good, inputs, case_id="case-05", checker=wrong_role)


def test_real_snapshot_oracle_choice_and_host_frozen_request_are_compatible(tmp_path: Path):
    bindings = {
        "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
        "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
    }
    packet = M.DRIVER.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=bindings,
        budget_left=3, execution_budget_s=1200)
    snapshot = M.DRIVER.C.V.build_snapshot(
        packet, "f2_oracle_compatibility", M.HOST_STORAGE_POLICY,
        M.DRIVER.C.V.file_digest(M.DRIVER.C.V.PROTOCOL), tmp_path / "oracle.ledger.jsonl")
    choice = M.oracle_choice(snapshot)

    assert snapshot["inputs"]["policy"] == M.HOST_STORAGE_POLICY
    assert choice["policy"] == M.HOST_STORAGE_POLICY
    assert choice["probe"] == M.PROBE
    assert choice["selected_by"] == "registered_rule"
    assert M.ORACLE_POLICY not in {snapshot["inputs"]["policy"], choice["policy"]}
    request = M.DRIVER.C.request_spec(snapshot, choice)
    frozen = M.HOST._HOST.C._frozen_request_spec(snapshot, choice)
    assert frozen == request
    assert request["probe"] == M.PROBE


def _execute_fixture(tmp_path: Path, outcomes: list[str], *, historical: bool = False,
                     consumed_pairs: int = 0, consumed_wall: float = 0.0):
    plan_path = tmp_path / "plan.json"
    detail = write(tmp_path / "detail.json", {"fixture": True})
    targets = []
    prepared = {}
    for index, (case_id, item) in enumerate(M.TARGETS.items()):
        row = {
            "case_id": case_id,
            "source_digest": M.B.digest(source(case_id, item["seed"])),
            "action": "reuse" if historical else "execute_one_pair",
            "historical": ({"detail": file_ref(detail), "step_index": index,
                            "verified": {"probe": M.PROBE, "outcome": outcomes[index]}}
                           if historical else None),
        }
        targets.append(row)
        prepared[case_id] = {
            "source": source(case_id, item["seed"]),
            "inputs": {"case": tmp_path / f"{case_id}-case.json",
                       "reference": tmp_path / f"{case_id}-ref.json"},
        }
    plan = {
        "schema": M.PLAN_SCHEMA,
        "semantic_anchor": "a" * 40,
        "targets": targets,
        "original_budget": {"paired_cap": 150, "sdk_role_cap": 300,
                            "wall_cap_s": 144000, "paired_consumed": consumed_pairs,
                            "sdk_roles_consumed": consumed_pairs * 2,
                            "wall_consumed_s": consumed_wall},
    }
    write(plan_path, plan)
    protocol = {"private_plan_sha256": M.sha(plan_path),
                "private_plan_digest": M.digest(plan),
                "budget": {"pair_cap": 2, "sdk_role_cap": 4, "api_cap": 0,
                           "wall_cap_s": 3600, "automatic_retry": False}}
    material = {"prepared": prepared,
                "protocol": {"tool_protocols": {
                    "paired_probe": {"path": "probe.json"},
                    "view": {"path": "view.json"}}}}
    calls = {"execute": 0, "verify": 0, "history": 0}

    def validator(*_args, **_kwargs):
        return protocol, plan, material

    def pair_executor(*_args, **kwargs):
        calls["execute"] += 1
        kwargs["verify_context"]()
        result_path = write(kwargs["out"] / "result.json", {"fixture": True})
        return {"result_ref": file_ref(result_path)}

    def pair_verifier(*_args):
        index = calls["verify"]
        calls["verify"] += 1
        return {"probe": M.PROBE, "retry_count": 0, "outcome": outcomes[index]}

    def history(*_args):
        index = calls["history"]
        calls["history"] += 1
        return {"probe": M.PROBE, "outcome": outcomes[index]}

    leases = lambda: {"owner": M.RUN.LEASE.OWNER, "leaseId": "lease-fixture"}
    return plan_path, validator, pair_executor, pair_verifier, history, leases, calls


@pytest.mark.parametrize(("outcomes", "coverage"), [
    (["normal", "normal"], ["signature_not_observed", "signature_not_observed"]),
    (["abnormal", "abnormal"], ["confirmed_signature", "confirmed_signature"]),
    (["unknown", "unknown"], ["unknown_HOLD", "unknown_HOLD"]),
])
def test_execute_classifies_outcomes_and_never_resends(tmp_path: Path, monkeypatch,
                                                        outcomes, coverage):
    fixture = _execute_fixture(tmp_path, outcomes)
    plan_path, validator, pair_executor, pair_verifier, _history, leases, calls = fixture
    monkeypatch.setattr(M.DRIVER.C.V, "build_snapshot", lambda *_a, **_k: {"fixture": True})
    monkeypatch.setattr(M, "oracle_choice", lambda _snapshot: {"fixture": True})
    result = M.execute(tmp_path / "protocol.json", "a" * 40, plan_path,
                       lab=tmp_path, out=tmp_path / "out", pair_executor=pair_executor,
                       pair_verifier=pair_verifier, lease_reader=leases, validator=validator)
    assert [row["coverage"] for row in result["targets"]] == coverage
    assert calls == {"execute": 2, "verify": 2, "history": 0}
    assert result["new_pairs"] == 2 and result["sdk_roles_reserved"] == 4
    assert result["api_calls"] == result["model_calls"] == 0
    assert result["selection_performance_credit"] == 0


def test_existing_receipts_are_reused_without_sdk_callback(tmp_path: Path, monkeypatch):
    fixture = _execute_fixture(tmp_path, ["abnormal", "normal"], historical=True)
    plan_path, validator, pair_executor, pair_verifier, history, leases, calls = fixture
    monkeypatch.setattr(M.AUDIT, "inspect_step", history)
    result = M.execute(tmp_path / "protocol.json", "a" * 40, plan_path,
                       lab=tmp_path, out=tmp_path / "out", pair_executor=pair_executor,
                       pair_verifier=pair_verifier, lease_reader=leases, validator=validator)
    assert calls == {"execute": 0, "verify": 0, "history": 2}
    assert result["new_pairs"] == result["sdk_roles_reserved"] == 0
    assert [row["execution"] for row in result["targets"]] == [
        "reused_existing_receipt", "reused_existing_receipt"]


def test_original_pair_sdk_cap_fails_before_callback(tmp_path: Path, monkeypatch):
    fixture = _execute_fixture(tmp_path, ["normal", "normal"], consumed_pairs=150)
    plan_path, validator, pair_executor, pair_verifier, _history, leases, calls = fixture
    monkeypatch.setattr(M.DRIVER.C.V, "build_snapshot", lambda *_a, **_k: {"fixture": True})
    monkeypatch.setattr(M, "oracle_choice", lambda _snapshot: {"fixture": True})
    with pytest.raises(ValueError, match="pair/SDK cap"):
        M.execute(tmp_path / "protocol.json", "a" * 40, plan_path,
                  lab=tmp_path, out=tmp_path / "out", pair_executor=pair_executor,
                  pair_verifier=pair_verifier, lease_reader=leases, validator=validator)
    assert calls["execute"] == 0


def test_wrong_lease_owner_and_exhausted_wall_fail_closed(tmp_path: Path, monkeypatch):
    fixture = _execute_fixture(tmp_path, ["normal", "normal"])
    plan_path, validator, pair_executor, pair_verifier, _history, _leases, calls = fixture
    wrong = lambda: {"owner": "other", "leaseId": "lease-fixture"}
    with pytest.raises(ValueError, match="common lease"):
        M.execute(tmp_path / "protocol.json", "a" * 40, plan_path,
                  lab=tmp_path, out=tmp_path / "wrong-owner", pair_executor=pair_executor,
                  pair_verifier=pair_verifier, lease_reader=wrong, validator=validator)
    assert calls["execute"] == 0

    fixture = _execute_fixture(tmp_path / "wall", ["normal", "normal"],
                               consumed_wall=144001)
    plan_path, validator, pair_executor, pair_verifier, _history, leases, calls = fixture
    monkeypatch.setattr(M.DRIVER.C.V, "build_snapshot", lambda *_a, **_k: {"fixture": True})
    monkeypatch.setattr(M, "oracle_choice", lambda _snapshot: {"fixture": True})
    with pytest.raises(ValueError, match="wall cap"):
        M.execute(tmp_path / "wall/protocol.json", "a" * 40, plan_path,
                  lab=tmp_path, out=tmp_path / "wall/out", pair_executor=pair_executor,
                  pair_verifier=pair_verifier, lease_reader=leases, validator=validator)
    assert calls["execute"] == 0


def test_plan_tamper_is_rejected_at_owned_boundary(tmp_path: Path, monkeypatch):
    fixture = _execute_fixture(tmp_path, ["normal", "normal"])
    plan_path, validator, _pair_executor, pair_verifier, _history, leases, calls = fixture
    monkeypatch.setattr(M.DRIVER.C.V, "build_snapshot", lambda *_a, **_k: {"fixture": True})
    monkeypatch.setattr(M, "oracle_choice", lambda _snapshot: {"fixture": True})

    def tampering_executor(*_args, **kwargs):
        calls["execute"] += 1
        write(plan_path, {"tampered": True})
        kwargs["verify_context"]()

    with pytest.raises(ValueError, match="plan 변경"):
        M.execute(tmp_path / "protocol.json", "a" * 40, plan_path,
                  lab=tmp_path, out=tmp_path / "out", pair_executor=tampering_executor,
                  pair_verifier=pair_verifier, lease_reader=leases, validator=validator)
    assert calls["execute"] == 1


def test_prepare_requires_complete_fifty_and_preserves_source_closure(tmp_path: Path):
    protocol_path = write(tmp_path / "r3.json", {"fixture": True})
    config_path = write(tmp_path / "config.json", {"fixture": True})
    batch_path = tmp_path / "batch.json"
    semantic = "a" * 40
    protocol = {"paired_cap": 150, "sdk_role_cap": 300, "wall_cap_s": 144000}
    config = {"source_index": {"path": "private", "sha256": "index-sha"}}
    prepared = {}
    paths = {}
    for case_id, item in M.TARGETS.items():
        paths[case_id] = {
            "case": write(tmp_path / f"{case_id}-case.json", {}),
            "reference": write(tmp_path / f"{case_id}-ref.json", {}),
        }
        prepared[case_id] = {"source": source(case_id, item["seed"]),
                             "inputs": paths[case_id]}

    rows = [{"case_id": "other"} for _ in range(50)]
    batch = {"schema": "lineage_five_policy_batch_result_v1", "status": "complete",
             "planned_cases": 10, "planned_rows": 50, "rows": rows,
             "execution_protocol": M.B.ref(protocol_path),
             "execution_config": M.B.ref(config_path), "execution_anchor": semantic,
             "repair": False, "recovery": False, "paired_intents": 0,
             "sdk_roles_reserved": 0, "parent_wall_s": 0}
    write(batch_path, batch)

    def preflight(*_args):
        return protocol, config, prepared

    def input_checker(path: Path):
        for case_id, roles in paths.items():
            for role, expected in roles.items():
                if path == expected:
                    return checked(role, M.TARGETS[case_id]["seed"])
        raise AssertionError(path)

    before = {name: M.sha(ROOT / name, lf=True) for name in M.SOURCE_FILES}
    plan = M.prepare(batch_path, protocol_path, config_path, semantic, tmp_path / "plan.json",
                     preflight=preflight, input_checker=input_checker,
                     inspect=lambda *_a: pytest.fail("history callback"))
    after = {name: M.sha(ROOT / name, lf=True) for name in M.SOURCE_FILES}
    assert before == after
    assert [row["action"] for row in plan["targets"]] == [
        "execute_one_pair", "execute_one_pair"]
    assert plan["oracle_budget"] == {"pair_cap": 2, "sdk_role_cap": 4,
                                     "api_cap": 0, "wall_cap_s": 3600,
                                     "automatic_retry": False}
    overlay = M.build_protocol_value(tmp_path / "plan.json")
    assert overlay["source_files_lf"] == before
    assert overlay["oracle_policy"]["selection_performance_credit"] == 0

    bad = deepcopy(batch)
    bad["rows"] = bad["rows"][:-1]
    bad["planned_rows"] = 49
    write(batch_path, bad)
    with pytest.raises(ValueError, match="complete original R3 batch50"):
        M.prepare(batch_path, protocol_path, config_path, semantic, tmp_path / "bad-plan.json",
                  preflight=preflight, input_checker=input_checker)


def test_build_protocol_rejects_target_substitution(tmp_path: Path):
    plan = {"schema": M.PLAN_SCHEMA,
            "targets": [{"case_id": "case-06", "action": "reuse"},
                        {"case_id": "case-05", "action": "reuse"}]}
    plan_path = write(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="private oracle plan"):
        M.build_protocol_value(plan_path)
