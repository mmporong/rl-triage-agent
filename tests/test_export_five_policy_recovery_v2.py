from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src", "tests")]

import export_five_policy_recovery_v2 as E
import recovery_outcome_contract_v1 as OUTCOME
from test_export_five_policy_recovery_v1 import make_fixture


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _v2_fixture(
    tmp_path: Path, *, not_recovered: bool = False, with_second: bool = False,
) -> tuple[Path, Path | None, dict]:
    first_path, second_path, base = make_fixture(
        tmp_path, not_recovered=not_recovered, with_second=with_second)
    first = E._read(first_path)
    plan_path = E._bound(first["plan"], "plan")
    plan = E._read(plan_path)
    old_lease = "old-closed-lease"
    new_lease = "new-phase-lease"
    old_lease_path = save(tmp_path / "old-lease.json", {"leaseId": old_lease})
    inventory_path = save(tmp_path / "old-inventory.json", {"preserved": True})
    for index, row in enumerate(first["records"]):
        lease = old_lease if index == 0 else new_lease
        row["lease_id"] = lease
        row["execution_kind"] = (
            E.PRIOR_EXECUTION_KIND if index == 0
            else E.NEW_EXECUTION_KIND
        )
        if row.get("result") is None:
            continue
        result_path = E._bound(row["result"], "result")
        result = E._read(result_path)
        receipt_path = E._bound(result["receipt"], "receipt")
        receipt = E._read(receipt_path)
        approval_path = E._bound(receipt["approval"], "approval")
        approval = E._read(approval_path)
        approval["lease_id"] = lease
        save(approval_path, approval)
        receipt["lease_id"] = lease
        receipt["approval"] = E.FIRST.ref(approval_path)
        if index == 0:
            failure_path = save(tmp_path / "old-target/failure.json", {"status": "invalid"})
            receipt["continuation_provenance"] = {
                "kind": E.PRIOR_EXECUTION_KIND,
                "original_failure": E.FIRST.ref(failure_path),
                "original_approval": receipt["approval"],
                "original_closed_lease": E.FIRST.ref(old_lease_path),
                "original_raw_inventory": E.FIRST.ref(inventory_path),
                "abi_normalization": {
                    "source": E.FIRST.ref(ROOT / "evals/recovery_outcome_contract_v1.py"),
                    "canonical_recovery_preserved": True,
                    "inner_outcome_added_from_outer": True,
                },
            }
        save(receipt_path, receipt)
        result["receipt"] = E.FIRST.ref(receipt_path)
        save(result_path, result)
        row["result"] = E.FIRST.ref(result_path)
        if row.get("epoch2_input") is not None:
            epoch_path = E._bound(row["epoch2_input"], "epoch")
            epoch = E._read(epoch_path)
            epoch["prior_receipt"] = E.FIRST.ref(receipt_path)
            save(epoch_path, epoch)
            row["epoch2_input"] = E.FIRST.ref(epoch_path)
    host_protocol_path = save(tmp_path / "first-host-protocol.json", {})
    private_plan = {
        "schema": E.CONTINUATION.FIRST_CONTINUATION_PLAN_SCHEMA,
        "original_plan": E.FIRST.ref(plan_path),
        "original_first_protocol": plan["recovery_protocol"],
        "original_host_protocol": E.FIRST.ref(host_protocol_path),
        "old_failed_phase": str(tmp_path / "old-phase"),
        "old_raw_inventory": E.FIRST.ref(inventory_path),
        "finalized_result": first["records"][0]["result"],
        "old_closed_lease": E.FIRST.ref(old_lease_path),
    }
    private_plan_path = save(tmp_path / "continuation-private-plan.json", private_plan)
    phase_value = {
        "schema": E.CONTINUATION.FIRST_CONTINUATION_PROTOCOL_SCHEMA,
        "private_plan_sha256": E._sha(private_plan_path),
        "original": {"anchor": plan["anchor"]},
        "prior": {"case_id": first["records"][0]["case_id"],
                  "policy": first["records"][0]["policy"],
                  "lease_id": old_lease},
    }
    phase = save(tmp_path / "continuation-protocol.json", phase_value)
    first.update(
        schema=E.CONTINUATION.FIRST_CONTINUATION_RESULT_SCHEMA,
        phase_protocol=E.FIRST.ref(phase),
        continuation_plan=E.FIRST.ref(private_plan_path),
        execution_anchor="b" * 40,
        anchor=plan["anchor"], lease_id=new_lease,
        old_sdk_executions=2, new_sdk_executions=6,
        total_sdk_execution_upper_bound=8, api_calls=0,
        prior_reuse_count=1, new_executed_targets=3,
        old_wall_s=2.0, new_parent_wall_s=3.0, total_parent_wall_s=5.0,
        second_phase_requires_new_registered_protocol=not_recovered,
    )
    save(first_path, first)
    if second_path is not None:
        second = E._read(second_path)
        second_plan_path = E._bound(second["plan"], "second plan")
        second_plan = E._read(second_plan_path)
        second_plan["first_result"] = E.FIRST.ref(first_path)
        save(second_plan_path, second_plan)
        second["plan"] = E.FIRST.ref(second_plan_path)
        save(second_path, second)
    material = {"protocol": phase_value, "plan": private_plan}
    verifiers = dict(base)
    verifiers["continuation_protocol"] = lambda *_args, **_kwargs: deepcopy(material)
    verifiers["second_compatibility_protocol"] = lambda *_args: {}
    return first_path, second_path, verifiers


def test_v2_parent_exports_v1_public_schemas_and_no_private_path(tmp_path: Path) -> None:
    first, _second, verifiers = _v2_fixture(tmp_path)
    out = tmp_path / "public"
    E.export(first, out, lab=tmp_path, allow_fixture=True, verifiers=verifiers)
    assert E._read(out / "summary.json")["schema"] == E.SUMMARY_SCHEMA
    assert E._read(out / "cases.json")["schema"] == E.CASES_SCHEMA
    provenance = E._read(out / "provenance.json")
    assert provenance["first_result_schema"] == E.CONTINUATION.FIRST_CONTINUATION_RESULT_SCHEMA
    assert provenance["continuation"]["prior_gpu_reexecution"] is False
    assert E._read(out / "manifest.json")["schema"] == E.MANIFEST_SCHEMA
    text = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.json"))
    assert str(tmp_path) not in text and "bench/private" not in text


def test_default_callbacks_use_public_private_final_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuation = tmp_path / "continuation.json"
    private_plan = tmp_path / "private-plan.json"
    second_compat = tmp_path / "second-compat.json"
    second_plan = tmp_path / "second-plan.json"
    host_protocol = tmp_path / "host.json"
    seen: dict[str, object] = {}

    def verify_continuation(
        path: Path, *, continuation_plan_path: Path,
        execution_anchor: str, lab: Path,
    ) -> dict:
        seen["continuation"] = (
            path, continuation_plan_path, execution_anchor, lab)
        return {"protocol": {"public": True}, "plan": {"private": True}}

    def verify_second(
        path: Path, *, anchor: str, second_plan_path: Path,
        host_protocol: Path,
    ) -> dict:
        seen["second"] = (path, anchor, second_plan_path, host_protocol)
        return {"second_plan_sha256": "a" * 64}

    monkeypatch.setattr(E.CONTINUATION, "verify_continuation_protocol", verify_continuation)
    monkeypatch.setattr(
        E.CONTINUATION, "verify_second_compatibility_protocol", verify_second)
    verify = E._default_verifiers()
    material = verify["continuation_protocol"](
        continuation, private_plan, "b" * 40, tmp_path)
    second = verify["second_compatibility_protocol"](
        second_compat, "c" * 40, second_plan, host_protocol)

    assert material == {"protocol": {"public": True}, "plan": {"private": True}}
    assert second == {"second_plan_sha256": "a" * 64}
    assert seen["continuation"] == (
        continuation, private_plan, "b" * 40, tmp_path)
    assert seen["second"] == (
        second_compat, "c" * 40, second_plan, host_protocol)


def test_real_canonical_recovery_is_normalized_during_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second, verifiers = _v2_fixture(tmp_path)
    bands = json.loads((ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json")
                       .read_text(encoding="utf-8"))["bands"]
    reference = json.loads((ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_none.json")
                           .read_text(encoding="utf-8"))
    report = json.loads((ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_f1.json")
                        .read_text(encoding="utf-8"))
    original = E._training

    def checked(*args, **kwargs):
        verdict = OUTCOME.GATE.RESIDUAL.recovery(report, reference, bands)["verdict"]
        assert verdict["label"] == "unhealthy"
        assert verdict["outcome"] == "not_recovered"
        return original(*args, **kwargs)

    monkeypatch.setattr(E, "_training", checked)
    E.export(first, tmp_path / "public", lab=tmp_path,
             allow_fixture=True, verifiers=verifiers)
    assert OUTCOME.GATE.RESIDUAL.recovery is OUTCOME._CANONICAL_RECOVERY


def test_second_result_is_fully_validated_inside_normalized_scope(tmp_path: Path) -> None:
    first, second, verifiers = _v2_fixture(
        tmp_path, not_recovered=True, with_second=True)
    assert second is not None
    second_parent = E._read(second)
    second_plan_path = E._bound(second_parent["plan"], "second plan")
    compatibility = save(tmp_path / "second-compatibility.json", {})
    second_host = save(tmp_path / "second-host-protocol.json", {})
    verifiers["second_compatibility_protocol"] = lambda *_args, **_kwargs: {
        "second_plan_sha256": E._sha(second_plan_path),
    }
    E.export(first, tmp_path / "public", lab=tmp_path, second_result=second,
             second_compatibility_protocol=compatibility,
             second_compatibility_anchor="c" * 40,
             second_host_protocol=second_host,
             allow_fixture=True, verifiers=verifiers)
    rows = E._read(tmp_path / "public/cases.json")["rows"]
    target = next(row for row in rows if row["case_id"] == "case-03"
                  and row["policy"] == "fixed_eligible")
    assert target["first_outcome"] == "not_recovered"
    assert target["second_outcome"] == "HOLD"


def test_wrong_old_result_reuse_is_rejected(tmp_path: Path) -> None:
    first_path, _second, verifiers = _v2_fixture(tmp_path)
    first = E._read(first_path)
    prior = next(row for row in first["records"]
                 if row["execution_kind"] == E.PRIOR_EXECUTION_KIND)
    other = next(row for row in first["records"] if row.get("result") is not None
                 and row is not prior)
    prior["result"] = other["result"]
    save(first_path, first)
    with pytest.raises(ValueError, match="prior closed result"):
        E.export(first_path, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=verifiers)


def test_per_row_lease_mismatch_is_rejected(tmp_path: Path) -> None:
    first_path, _second, verifiers = _v2_fixture(tmp_path)
    first = E._read(first_path)
    row = next(row for row in first["records"] if row.get("result") is not None
               and row["execution_kind"] == E.NEW_EXECUTION_KIND)
    row["lease_id"] = "wrong-row-lease"
    save(first_path, first)
    with pytest.raises(ValueError, match="new 행 lease|행별 lease"):
        E.export(first_path, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=verifiers)


def test_checkpoint_receipt_swap_is_rejected(tmp_path: Path) -> None:
    first_path, _second, verifiers = _v2_fixture(tmp_path)
    first = E._read(first_path)
    executed = [row for row in first["records"] if row.get("result") is not None]
    assert len(executed) == 2
    executed[0]["result"], executed[1]["result"] = executed[1]["result"], executed[0]["result"]
    material = verifiers["continuation_protocol"](None)
    prior = next(row for row in first["records"]
                 if row["execution_kind"] == E.PRIOR_EXECUTION_KIND)
    material["plan"]["finalized_result"] = prior["result"]
    verifiers["continuation_protocol"] = lambda *_args, **_kwargs: deepcopy(material)
    save(first_path, first)
    with pytest.raises(ValueError, match="first result/receipt"):
        E.export(first_path, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=verifiers)


def test_missing_first_row_is_rejected(tmp_path: Path) -> None:
    first_path, _second, verifiers = _v2_fixture(tmp_path)
    first = E._read(first_path)
    first["records"].pop()
    save(first_path, first)
    with pytest.raises(ValueError, match="first exact4"):
        E.export(first_path, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=verifiers)


def test_v1_parent_schema_is_not_accepted_as_continuation(tmp_path: Path) -> None:
    first_path, _second, verifiers = _v2_fixture(tmp_path)
    first = E._read(first_path)
    first["schema"] = E.PARENT.FIRST_RESULT_SCHEMA
    save(first_path, first)
    with pytest.raises(ValueError, match="continuation parent"):
        E.export(first_path, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=verifiers)
