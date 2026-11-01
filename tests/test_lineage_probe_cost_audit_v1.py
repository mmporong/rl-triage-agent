# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_probe_cost_audit_v1 as A


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(A.json_bytes(value))
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": A.sha(path)}


def pair_fixture(tmp_path: Path, *, walls: dict[str, float] | None = None,
                 missing: set[str] = frozenset(), mismatch: str | None = None) -> Path:
    walls = walls or {"reference": 2.0, "case": 3.0}
    probe, request_id = "P_reward", "backend-request-1"
    plans: dict[str, dict] = {}
    plan_paths: dict[str, Path] = {}
    checkpoints = {"case": "1" * 64, "reference": "2" * 64}
    for role in A.ROLES:
        plan = {"role": role, "probe": probe,
                "checkpoint": {"sha256": checkpoints[role]}}
        plan_paths[role] = save(tmp_path / role / "plan.json", plan)
        plans[role] = ref(plan_paths[role])
    authorization = {
        "request_id": request_id, "request_spec": {
            "probe": probe, "checkpoint_sha256": checkpoints["case"],
            "reference_checkpoint_sha256": checkpoints["reference"],
        },
        "plan_sha256": {role: A.sha(path) for role, path in plan_paths.items()},
    }
    auth_path = save(tmp_path / "authorization.json", authorization)
    receipts: dict[str, dict | None] = {}
    outer: dict[str, dict | None] = {}
    for role in A.ROLES:
        if role in missing:
            receipts[role] = None
            outer[role] = {"status": "unknown", "error_type": "MissingReceipt"}
            continue
        execution = {
            "job_assigned_before_release": True, "natural_exit": True,
            "timed_out": False, "error_type": None, "exit_code": 0,
            "cleanup_verified": True, "owned_process_identities": {"redacted": 1},
            "wall_s": walls[role], "wall_scope": A.WALL_SCOPE,
        }
        owned = deepcopy(execution)
        if mismatch == role:
            owned["wall_s"] = walls[role] + 1.0
        owned_path = save(tmp_path / role / "owned-result.json", owned)
        claim_body = {
            "role": role, "request_id": request_id,
            "plan_sha256": A.sha(plan_paths[role]),
            "checkpoint_sha256": checkpoints[role],
        }
        claim_path = save(tmp_path / role / "claim.json", claim_body)
        receipt = {
            "status": "complete", "role": role, "probe": probe,
            "plan": plans[role], "checkpoint_sha256": checkpoints[role],
            "authorization": {**ref(auth_path), "request_id": request_id},
            "claim": {**ref(claim_path), "body": claim_body},
            "raw_refs": {"owned_result": ref(owned_path)}, "execution": execution,
        }
        receipt_path = save(tmp_path / role / "probe_receipt.json", receipt)
        receipts[role], outer[role] = ref(receipt_path), None
    unknown = [role for role in A.ROLES if role in missing]
    known = sum(walls[role] for role in A.ROLES if role not in missing)
    pair = {
        "schema": A.BASE_HOST.SCHEMA, "status": "unknown_hold" if unknown else "complete",
        "evidence_kind": "actual_runtime_observation", "retry_count": 0,
        "probe": probe, "outcome": "unknown" if unknown else "abnormal",
        "authorization": ref(auth_path), "backend_request_id": request_id,
        "plans": plans, "receipts": receipts, "outer_unknown_receipts": outer,
        "parent_wall_s": 7.0,
        "owned_child_wall_s": None if unknown else known,
        "owned_child_known_wall_s": known,
        "owned_child_unknown_roles": unknown,
    }
    return save(tmp_path / "result.json", pair)


def universe() -> list[dict]:
    return [{
        "schema": A.ROW_SCHEMA, "case_id": f"case-{case:02d}", "policy": policy,
        "end_state": "HOLD", "execution_state": "no_probe_executed", "steps": [],
        "row_parent_wall_s": 1.0, "source_digest": "3" * 64,
    } for case in range(1, 11) for policy in A.DRIVER.ARMS]


def interruption_fixture(tmp_path: Path, *, tamper_owned: bool = False) -> dict:
    pair_path = pair_fixture(tmp_path, walls={"reference": 18.0, "case": 2.0})
    pair = A.read(pair_path)
    reference_receipt = A.read(Path(pair["receipts"]["reference"]["path"]))
    case_receipt_path = Path(pair["receipts"]["case"]["path"])
    case_receipt = A.read(case_receipt_path)
    case_owned = Path(case_receipt["raw_refs"]["owned_result"]["path"])
    case_claim = Path(case_receipt["claim"]["path"])
    case_receipt_path.unlink()
    case_owned.unlink()
    case_claim.unlink()
    if tamper_owned:
        owned = Path(reference_receipt["raw_refs"]["owned_result"]["path"])
        value = A.read(owned)
        value["wall_s"] = 19.0
        owned.write_bytes(A.json_bytes(value))
    return {
        "schema": "lineage_five_policy_partial_role_interruption_v1",
        "stage": "paired_probe_role_execution", "case_id": "case-04",
        "arm": "fixed_eligible", "step": 2, "probe": "P_reward",
        "pair_result": None, "authorization": pair["authorization"],
        "roles": {
            "reference": {
                "status": "complete", "plan": pair["plans"]["reference"],
                "receipt": pair["receipts"]["reference"],
                "owned_result": reference_receipt["raw_refs"]["owned_result"],
                "claim": {"path": reference_receipt["claim"]["path"],
                          "sha256": reference_receipt["claim"]["sha256"]},
            },
            "case": {
                "status": "not_executed", "plan": pair["plans"]["case"],
                "receipt": None, "owned_result": None, "claim": None,
                "reason": "lease_read_permission_error_before_case_execute_approval",
            },
        },
    }


def test_inspect_pair_reconstructs_two_owned_roles_and_preserves_reported_unknown(
        tmp_path: Path) -> None:
    path = pair_fixture(tmp_path)
    raw = A.read(path)
    raw["owned_child_wall_s"] = None
    raw["owned_child_known_wall_s"] = 0
    raw["owned_child_unknown_roles"] = ["reference", "case"]
    save(path.with_suffix(".tmp"), raw)
    path.unlink()
    path.with_suffix(".tmp").rename(path)
    result = A.inspect_pair(path)
    assert result["reported"] == {
        "owned_child_wall_s": None, "owned_child_known_wall_s": 0,
        "owned_child_unknown_roles": ["reference", "case"],
    }
    assert result["reconstructed"]["owned_wall_s"] == 5.0
    assert result["reconstructed"]["known_owned_wall_s"] == 5.0
    assert result["reconstructed"]["unknown_roles"] == []
    assert all(row["timing_source"] == "receipt_reconstruction"
               for row in result["reconstructed"]["roles"])


def test_actual_zero_wall_is_distinct_from_missing_unknown(tmp_path: Path) -> None:
    result = A.inspect_pair(pair_fixture(
        tmp_path, walls={"reference": 4.0, "case": 0.0}, missing={"reference"}))
    reconstructed = result["reconstructed"]
    assert reconstructed["owned_wall_s"] is None
    assert reconstructed["known_owned_wall_s"] == 0.0
    assert reconstructed["unknown_roles"] == ["reference"]
    case = next(row for row in reconstructed["roles"] if row["role"] == "case")
    assert case["status"] == "complete" and case["wall_s"] == 0.0


def test_mismatched_owned_and_swapped_receipt_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inline execution"):
        A.inspect_pair(pair_fixture(tmp_path / "mismatch", mismatch="case"))
    path = pair_fixture(tmp_path / "swapped")
    pair = A.read(path)
    pair["receipts"]["case"], pair["receipts"]["reference"] = (
        pair["receipts"]["reference"], pair["receipts"]["case"])
    save(path.with_suffix(".tmp"), pair)
    path.unlink()
    path.with_suffix(".tmp").rename(path)
    with pytest.raises(ValueError, match="receipt/plan/role"):
        A.inspect_pair(path)


def test_duplicate_receipt_is_counted_once_and_all_50_rows_are_preserved() -> None:
    rows = universe()
    role = {"role": "case", "status": "complete", "wall_s": 2.5,
            "receipt_sha256": "a" * 64}
    step = {"step": 1, "pair": {"pair_parent_wall_s": 3.0,
            "reconstructed": {"roles": [role]}}}
    rows[0]["steps"] = [deepcopy(step)]
    rows[0]["execution_state"] = "paired_probe_executed"
    rows[1]["steps"] = [deepcopy(step)]
    rows[1]["execution_state"] = "paired_probe_executed"
    result = A._aggregate(rows)
    assert len(result["rows"]) == 50 and len(result["duplicates"]) == 1
    assert result["totals"]["sdk_roles_counted_once"] == 1
    assert result["totals"]["known_owned_wall_s"] == 2.5
    assert result["totals"]["zero_probe_rows"] == 48
    with pytest.raises(ValueError, match="complete"):
        A._aggregate(rows[:-1])
    accounting = A._intent_accounting(
        18, 17, timed_roles=35, orphan_roles=1, explicit_not_executed_roles=1)
    assert accounting["pair_intents_without_pair_result"] == 1
    assert accounting["verified_orphan_sdk_roles"] == 1
    assert accounting["verified_timed_sdk_roles"] == 35
    assert accounting["explicit_not_executed_roles"] == 1
    assert accounting["unresolved_reserved_roles"] == 0
    assert accounting["whole_pair_sdk_zero_inferred"] is False


def test_orphan_reference_is_timed_once_and_case_zero_requires_explicit_evidence(
        tmp_path: Path) -> None:
    orphan = A.inspect_orphan_role(interruption_fixture(tmp_path / "valid"))
    assert orphan["policy_diagnosis_step"] is False
    assert orphan["orphan_role_cost"]["role"] == "reference"
    assert orphan["orphan_role_cost"]["wall_s"] == 18.0
    assert orphan["unexecuted_role"]["role"] == "case"
    assert orphan["unexecuted_role"]["wall_s"] is None
    rows = universe()
    rows[0]["partial_role_interruptions"] = [orphan]
    duplicate_role = deepcopy(orphan["orphan_role_cost"])
    rows[1]["steps"] = [{"step": 1, "pair": {
        "pair_parent_wall_s": 1.0, "reconstructed": {"roles": [duplicate_role]}}}]
    result = A._aggregate(rows)
    assert len(result["duplicates"]) == 1
    assert result["totals"]["sdk_roles_counted_once"] == 1
    assert result["totals"]["orphan_roles_counted_once"] == 1
    assert result["totals"]["known_owned_wall_s"] == 18.0
    assert result["totals"]["explicit_not_executed_roles"] == 1
    with pytest.raises(ValueError, match="owned-result"):
        A.inspect_orphan_role(interruption_fixture(tmp_path / "tampered", tamper_owned=True))


def test_inspect_step_reuses_host_verify_pair(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair_path = pair_fixture(tmp_path / "pair")
    pair = A.read(pair_path)
    pair["public_choice"] = {"action": "continue"}
    save(pair_path.with_suffix(".tmp"), pair)
    pair_path.unlink()
    pair_path.with_suffix(".tmp").rename(pair_path)
    detail = {"steps": [{"step": 1, "probe": "P_reward", "outcome": "abnormal",
                          "probe_result": ref(pair_path), "owned_child_wall_s": 5.0}]}
    detail_path = save(tmp_path / "detail.json", detail)
    calls = {"count": 0}
    monkeypatch.setattr(A, "_historical_snapshot", lambda *_args: {"snapshot": True})
    monkeypatch.setattr(A, "_historical_protocol_scope", lambda *_args: nullcontext())

    def verified(path: Path, snapshot: dict, choice: dict) -> dict:
        calls["count"] += 1
        assert Path(path) == pair_path and snapshot == {"snapshot": True}
        assert choice == {"action": "continue"}
        return {"result_ref": ref(pair_path)}

    monkeypatch.setattr(A.HOST, "verify_pair", verified)
    result = A.inspect_step(detail_path, {"packet": {}}, 0, tmp_path / "temp")
    assert calls["count"] == 1 and result["pair"]["reconstructed"]["owned_wall_s"] == 5.0


def test_verify_public_tag_checks_canonical_inventory_privacy_and_50_rows(
        tmp_path: Path) -> None:
    out = tmp_path / "tag"
    out.mkdir()
    aggregate = A._aggregate(universe())
    summary = {"schema": A.SUMMARY_SCHEMA, "timing_source": "receipt_reconstruction",
               "interpretation_limits": {"missing_wall_filled_with_zero": False}}
    cases = {"schema": A.CASES_SCHEMA, "rows": aggregate["rows"]}
    provenance = {"schema": A.PROVENANCE_SCHEMA, "private_paths_included": False}
    for name, value in (("summary.json", summary), ("cases.json", cases),
                        ("provenance.json", provenance)):
        A.write_new(out / name, value)
    manifest = {"schema": A.MANIFEST_SCHEMA, "status": "complete",
                "public_files": list(A.PUBLIC_FILES), "overwrite_allowed": False,
                "files": {name: {"sha256": A.sha(out / name),
                                  "size_bytes": (out / name).stat().st_size}
                          for name in A.PUBLIC_FILES if name != "manifest.json"}}
    A.write_new(out / "manifest.json", manifest)
    assert A.verify_export(out) == manifest
    summary["leak"] = r"Z:\private\raw.json"
    (out / "summary.json").write_bytes(A.json_bytes(summary))
    manifest["files"]["summary.json"] = {"sha256": A.sha(out / "summary.json"),
                                          "size_bytes": (out / "summary.json").stat().st_size}
    (out / "manifest.json").write_bytes(A.json_bytes(manifest))
    with pytest.raises(ValueError, match="개인·private"):
        A.verify_export(out)
