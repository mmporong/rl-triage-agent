# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]

import lineage_probe_cost_audit_v2 as A

V1_TEST_SPEC = importlib.util.spec_from_file_location(
    "cost_audit_v1_fixtures", ROOT / "tests/test_lineage_probe_cost_audit_v1.py")
assert V1_TEST_SPEC and V1_TEST_SPEC.loader
F = importlib.util.module_from_spec(V1_TEST_SPEC)
V1_TEST_SPEC.loader.exec_module(F)


def _packet() -> dict:
    bindings = {
        "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
        "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
    }
    return A.V1.BASE_HOST.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=bindings,
        budget_left=3, execution_budget_s=1200)


def _snapshot(packet: dict, ledger: Path) -> dict:
    C = A.V1.BASE_HOST.C
    return C.V.build_snapshot(packet, "audit_v2_test", "registry_next",
                              C.V.file_digest(C.V.PROTOCOL), ledger)


def _ledger_with_future_pending(tmp_path: Path):
    C = A.V1.BASE_HOST.C
    ledger_path = tmp_path / "probes.ledger.jsonl"
    packet = _packet()
    snapshot = _snapshot(packet, ledger_path)
    choice = C.choose(snapshot, "registry_next")
    spec = C.request_spec(snapshot, choice)
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "cpu-test")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    ledger.receipt(request["request_id"], "normal", {}, 1.0, 0)
    proof = A.V1.BASE_HOST._prefix_proof(ledger_path)

    future_snapshot = _snapshot(packet, ledger_path)
    future_choice = C.choose(future_snapshot, "registry_next")
    future_spec = C.request_spec(future_snapshot, future_choice)
    future = ledger.propose(
        future_spec["prereg"], future_spec["probe"], future_spec["budget_gpu_s"])
    ledger.approve(future["request_id"], "cpu-test")
    ledger.consume(future["request_id"], future_spec["prereg"], future_spec["probe"])
    return ledger_path, packet, snapshot, spec, request["request_id"], proof, future["request_id"]


def _done(args, *, prefix_proof=None):
    ledger_path, packet, snapshot, spec, request_id, proof, _future = args
    return A.HOST_IMPL._done(
        ledger_path, snapshot["binding"], packet, request_id, spec,
        prior=[], outcome="normal", allow_suffix=True,
        prefix_proof=proof if prefix_proof is None else prefix_proof)


def test_completed_prefix_accepts_valid_future_pending_and_v1_still_rejects(tmp_path: Path):
    args = _ledger_with_future_pending(tmp_path)
    with pytest.raises(ValueError, match="ledger not single done request"):
        _done(args)
    original = A.HOST_IMPL._done
    with A._historical_done_prefix_scope(tmp_path / "scope") as evidence:
        checked = _done(args)
        assert checked["path"] == str(args[0].resolve())
    assert A.HOST_IMPL._done is original
    public = next(row for row in evidence if row["ledger_kind"] == "public")
    assert public["prefix_event_count"] == 4
    assert public["suffix_event_count"] == 3
    assert public["same_request_suffix_events"] == 0
    assert public["full_pending_request_sha256"] is not None


def test_current_incomplete_mutated_prefix_illegal_suffix_and_wrong_cp_are_rejected(
        tmp_path: Path):
    C = A.V1.BASE_HOST.C
    packet = _packet()
    ledger_path = tmp_path / "incomplete.jsonl"
    snapshot = _snapshot(packet, ledger_path)
    choice = C.choose(snapshot, "registry_next")
    spec = C.request_spec(snapshot, choice)
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "cpu-test")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    proof = A.V1.BASE_HOST._prefix_proof(ledger_path)
    with (
        A._historical_done_prefix_scope(tmp_path / "incomplete-scope"),
        pytest.raises(ValueError, match="ledger not single done request"),
    ):
        A.HOST_IMPL._done(
            ledger_path, snapshot["binding"], packet, request["request_id"], spec,
            prior=[], outcome="normal", allow_suffix=True, prefix_proof=proof)

    args = _ledger_with_future_pending(tmp_path / "valid")
    bad_proof = deepcopy(args[5]); bad_proof["sha256"] = "0" * 64
    with (
        A._historical_done_prefix_scope(tmp_path / "bad-prefix"),
        pytest.raises(ValueError, match="prefix SHA"),
    ):
        _done(args, prefix_proof=bad_proof)

    wrong = deepcopy(args[3])
    wrong["prereg"] = deepcopy(wrong["prereg"])
    wrong["prereg"]["checkpoint_sha256"] = "9" * 64
    with (
        A._historical_done_prefix_scope(tmp_path / "wrong-cp"),
        pytest.raises(ValueError, match="ledger not single done request"),
    ):
        A.HOST_IMPL._done(
            args[0], args[2]["binding"], args[1], args[4], wrong,
            prior=[], outcome="normal", allow_suffix=True, prefix_proof=args[5])

    with (
        A._historical_done_prefix_scope(tmp_path / "backend-suffix"),
        pytest.raises(ValueError, match="backend ledger has illegal suffix"),
    ):
        A.HOST_IMPL._done(
            args[0], args[2]["binding"], args[1], args[4], args[3],
            prior=[], outcome="normal", allow_suffix=False, prefix_proof=args[5])

    raw = args[0].read_bytes()
    args[0].write_bytes(raw + json.dumps({
        "event": "approved", "request_id": args[4], "approver": "illegal"
    }).encode("utf-8") + b"\n")
    with (
        A._historical_done_prefix_scope(tmp_path / "illegal-suffix"),
        pytest.raises((ValueError, A.V1.BASE_HOST.C.L.LedgerError)),
    ):
        _done(args)


def test_scope_restores_host_even_when_verification_raises(tmp_path: Path):
    original = A.HOST_IMPL._done
    with (
        pytest.raises(RuntimeError, match="fixture"),
        A._historical_done_prefix_scope(tmp_path / "scope"),
    ):
        assert A.HOST_IMPL._done is not original
        raise RuntimeError("fixture")
    assert A.HOST_IMPL._done is original


def test_wrong_receipt_orphan_18_032_dedup_and_no_zero_fill(tmp_path: Path):
    swapped = F.pair_fixture(tmp_path / "swapped")
    pair = A.V1.read(swapped)
    pair["receipts"]["case"], pair["receipts"]["reference"] = (
        pair["receipts"]["reference"], pair["receipts"]["case"])
    swapped.write_bytes(A.V1.json_bytes(pair))
    with pytest.raises(ValueError, match="receipt/plan/role"):
        A.V1.inspect_pair(swapped)

    interruption = F.interruption_fixture(tmp_path / "orphan")
    receipt_path = Path(interruption["roles"]["reference"]["receipt"]["path"])
    receipt = A.V1.read(receipt_path)
    owned_path = Path(receipt["raw_refs"]["owned_result"]["path"])
    owned = A.V1.read(owned_path); owned["wall_s"] = 18.032
    owned_path.write_bytes(A.V1.json_bytes(owned))
    receipt["execution"]["wall_s"] = 18.032
    receipt["raw_refs"]["owned_result"]["sha256"] = A.V1.sha(owned_path)
    receipt_path.write_bytes(A.V1.json_bytes(receipt))
    interruption["roles"]["reference"]["receipt"]["sha256"] = A.V1.sha(receipt_path)
    interruption["roles"]["reference"]["owned_result"]["sha256"] = A.V1.sha(owned_path)
    orphan = A.V1.inspect_orphan_role(interruption)
    assert orphan["orphan_role_cost"]["wall_s"] == 18.032
    assert orphan["unexecuted_role"]["wall_s"] is None

    rows = F.universe()
    rows[0]["partial_role_interruptions"] = [orphan]
    duplicate = deepcopy(orphan["orphan_role_cost"])
    rows[1]["steps"] = [{"step": 1, "pair": {
        "pair_parent_wall_s": 1.0, "reconstructed": {"roles": [duplicate]}}}]
    aggregate = A.V1._aggregate(rows)
    assert aggregate["totals"]["known_owned_wall_s"] == 18.032
    assert aggregate["totals"]["orphan_roles_counted_once"] == 1
    assert aggregate["totals"]["explicit_not_executed_roles"] == 1
    accounting = A.V1._intent_accounting(
        18, 17, timed_roles=35, orphan_roles=1, explicit_not_executed_roles=1)
    assert accounting["missing_wall_filled_with_zero"] is False
    assert accounting["unresolved_reserved_roles"] == 0


def test_v1_four_file_consumer_v2_provenance_and_fresh_output_guards(tmp_path: Path):
    rows = F.universe()
    aggregate = A.V1._aggregate(rows)
    summary = {
        "schema": A.V1.SUMMARY_SCHEMA, "timing_source": "receipt_reconstruction",
        "interpretation_limits": {"missing_wall_filled_with_zero": False},
    }
    cases = {"schema": A.V1.CASES_SCHEMA, "rows": aggregate["rows"]}
    provenance = {
        "schema": A.V1.PROVENANCE_SCHEMA,
        "historical_prefix_validation": A._prefix_provenance(aggregate["rows"]),
        "private_paths_included": False,
    }
    out = tmp_path / "tag"; out.mkdir()
    for name, value in (("summary.json", summary), ("cases.json", cases),
                        ("provenance.json", provenance)):
        A.V1.write_new(out / name, value)
    manifest = {
        "schema": A.V1.MANIFEST_SCHEMA, "status": "complete",
        "public_files": list(A.V1.PUBLIC_FILES), "overwrite_allowed": False,
        "files": {name: {"sha256": A.V1.sha(out / name),
                          "size_bytes": (out / name).stat().st_size}
                  for name in A.V1.PUBLIC_FILES if name != "manifest.json"},
    }
    A.V1.write_new(out / "manifest.json", manifest)
    assert A.verify_export(out) == manifest
    assert A.main(["verify", "--out", str(out)]) == 0

    existing = tmp_path / "existing"; existing.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        A.export(tmp_path / "batch", tmp_path / "protocol", tmp_path / "config",
                 "a" * 40, existing)
    with pytest.raises(ValueError, match="evals/results"):
        A.export(tmp_path / "batch", tmp_path / "protocol", tmp_path / "config",
                 "a" * 40, tmp_path / "new")


def test_actual_case04_step1_v1_fails_v2_passes_without_mutation(tmp_path: Path):
    base = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025/"
            "vision_goal/ri_full_completion_251/replan_369")
    batch_path = base / "diagnosis_continuation_403/run-v1/result.json"
    config_path = (base / "registration_388/five_type_diagnosis_s2550_s2559_20261101_v1/"
                   "execution-config.json")
    protocol_path = ROOT / "bench/protocols/five_type_diagnosis_s2550_s2559_20261101_v1_five_policy.json"
    batch = A.V1.read(batch_path)
    row = next(item for item in batch["rows"]
               if item["case_id"] == "case-04" and item["arm"] == "fixed_eligible")
    detail_path = A.V1.SUMMARY._bound_file(row["result"], "case04 fixed result")
    detail = A.V1.read(detail_path)
    pair_path = A.V1._bound(detail["steps"][0]["probe_result"], "case04 step1 pair")
    pair = A.V1.read(pair_path)
    public_ledger = Path(pair["public_ledger"]["path"]).resolve(strict=True)
    config = A.V1.read(config_path)
    source_path = A.V1.SUMMARY._bound_file(
        config["cases"]["case-04"]["source"], "case04 source")
    protected = [batch_path, config_path, protocol_path, source_path,
                 detail_path, pair_path, public_ledger]
    before = {str(path): A.V1.sha(path) for path in protected}
    source = A.V1.DRIVER.checked_source(A.V1.read(source_path))
    assert source["case_id"] == "case-04"
    with pytest.raises(ValueError, match="ledger not single done request"):
        A.V1.inspect_step(detail_path, source, 0, tmp_path / "v1")
    checked = A.inspect_step(detail_path, source, 0, tmp_path / "v2")
    public = next(row for row in checked["historical_prefix"]
                  if row["ledger_kind"] == "public")
    assert checked["step"] == 1 and public["prefix_event_count"] == 4
    assert public["suffix_event_count"] == 3
    assert before == {str(path): A.V1.sha(path) for path in protected}
