# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_recovery_same_play_capture_v1 as R


def save(path: Path, value: object) -> Path:
    R.B._write_new(path, value)
    return path


def ref(path: Path) -> dict[str, str]:
    return R.B._ref(path)


def recovery_result(tmp: Path, name: str, checkpoint: Path) -> Path:
    receipt = save(tmp / f"{name}-receipt.json", {
        "status": "recovered", "training": {"checkpoint": ref(checkpoint)}})
    return save(tmp / f"{name}-result.json", {
        "status": "complete", "outcome": "recovered", "receipt": ref(receipt)})


def actual_export_fixture(tmp: Path) -> tuple[Path, Path, Path, Path, Path]:
    original03, original07 = "3" * 64, "7" * 64
    cp03, cp07 = tmp / "cp03.pt", tmp / "cp07.pt"
    cp03.write_bytes(b"first-final")
    cp07.write_bytes(b"second-final")
    result03 = recovery_result(tmp, "first03", cp03)
    result07 = recovery_result(tmp, "second07", cp07)
    first = save(tmp / "first.json", {
        "schema": R.EXPORT.PARENT.FIRST_RESULT_SCHEMA, "status": "complete",
        "records": [
            {"case_id": "case-03", "policy": "logs_only", "outcome": "recovered",
             "registered_seed": 4553, "result": ref(result03)},
            {"case_id": "case-07", "policy": "logs_only", "outcome": "not_recovered",
             "registered_seed": 4557, "result": None},
        ]})
    second = save(tmp / "second.json", {
        "schema": R.EXPORT.PARENT.SECOND_RESULT_SCHEMA, "status": "complete",
        "records": [
            {"case_id": "case-03", "policy": "logs_only", "outcome": "not_in_second_subset",
             "status": "not_in_second_subset"},
            {"case_id": "case-07", "policy": "logs_only", "outcome": "recovered",
             "status": "registered", "registered_seed": 5557, "result": ref(result07)},
        ]})
    exported = tmp / "export"
    exported.mkdir()
    summary = save(exported / "summary.json", {
        "schema": R.EXPORT.SUMMARY_SCHEMA, "scope": "actual_backend"})
    cases = save(exported / "cases.json", {"schema": R.EXPORT.CASES_SCHEMA, "rows": [
        {"case_id": "case-03", "policy": "logs_only", "source_arm": "F1",
         "original_source_checkpoint_sha256": original03, "first_outcome": "recovered",
         "final_outcome": "recovered", "first_new_checkpoint_sha256": R.B._sha(cp03),
         "first_training": {"original_source_checkpoint_sha256": original03}},
        {"case_id": "case-07", "policy": "logs_only", "source_arm": "COMPOUND",
         "original_source_checkpoint_sha256": original07, "first_outcome": "not_recovered",
         "final_outcome": "recovered", "first_new_checkpoint_sha256": "1" * 64,
         "first_training": {"original_source_checkpoint_sha256": original07},
         "second_training": {"new_checkpoint_sha256": R.B._sha(cp07)}},
    ]})
    provenance = save(exported / "provenance.json", {
        "schema": R.EXPORT.PROVENANCE_SCHEMA,
        "first_result": {"sha256": R.B._sha(first)},
        "second_result": {"sha256": R.B._sha(second)},
        "first_protocol": {"path": "bench/protocols/five_type_recovery_first_s4552_s4557_20261101_v1.json",
                           "sha256": R.B._sha(ROOT / "bench/protocols/five_type_recovery_first_s4552_s4557_20261101_v1.json")},
        "conditional_design": {"path": "bench/protocols/five_policy_recovery_conditional_s2552_s2556_20261101_v1.json",
                               "sha256": R.B._sha(ROOT / "bench/protocols/five_policy_recovery_conditional_s2552_s2556_20261101_v1.json")}})
    files = {p.name: {"sha256": R.B._sha(p), "size_bytes": p.stat().st_size}
             for p in (summary, cases, provenance)}
    save(exported / "manifest.json", {"schema": R.EXPORT.MANIFEST_SCHEMA,
         "status": "complete", "files": files})
    return first, second, exported, cp03, cp07


def design() -> dict:
    return {"target_order": list(R.CAPTURE.TARGET_KEYS), "budget": {"sdk_cap": 3},
            "recovery_inputs": {
                "first_protocol": "bench/protocols/five_type_recovery_first_s4552_s4557_20261101_v1.json",
                "prior_conditional_design": "bench/protocols/five_policy_recovery_conditional_s2552_s2556_20261101_v1.json"},
            "targets": {
                "case-03:logs_only": {"case_id": "case-03", "policy": "logs_only",
                    "source_arm": "F1", "source_pair": {"reference": "source-05", "case": "source-06"},
                    "first_seed": 4553, "second_seed": 5553},
                "case-07:logs_only": {"case_id": "case-07", "policy": "logs_only",
                    "source_arm": "COMPOUND", "source_pair": {"reference": "source-13", "case": "source-14"},
                    "first_seed": 4557, "second_seed": 5557}}}


def test_actual_export_shape_selects_first_and_valid_second_final_without_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, exported, cp03, cp07 = actual_export_fixture(tmp_path)
    monkeypatch.setattr(R.CAPTURE, "validate_design", lambda *_: design())
    rows = R.validated_targets(tmp_path / "design.json", {
        "first_result": ref(first), "second_result": ref(second), "recovery_export": str(exported)})
    assert [(r["key"], r["final_epoch"], r["registered_seed"]) for r in rows] == [
        ("case-03:logs_only", "first", 4553), ("case-07:logs_only", "second", 5557)]
    assert Path(rows[0]["checkpoint_path"]) == cp03.resolve()
    assert Path(rows[1]["checkpoint_path"]) == cp07.resolve()


def test_hold_target_is_not_replaced_by_other_success(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    first, second, exported, _cp03, _cp07 = actual_export_fixture(tmp_path)
    second_value = R.B._read(second)
    second_value["records"][1].update(outcome="HOLD", status="registered", result=None)
    second.unlink()
    save(second, second_value)
    cases_path = exported / "cases.json"
    cases = R.B._read(cases_path)
    cases["rows"][1]["final_outcome"] = "HOLD"
    cases_path.unlink(); save(cases_path, cases)
    provenance_path = exported / "provenance.json"
    provenance = R.B._read(provenance_path)
    provenance["second_result"]["sha256"] = R.B._sha(second)
    provenance_path.unlink(); save(provenance_path, provenance)
    manifest_path = exported / "manifest.json"
    manifest = R.B._read(manifest_path)
    for path in (cases_path, provenance_path):
        manifest["files"][path.name] = {"sha256": R.B._sha(path), "size_bytes": path.stat().st_size}
    manifest_path.unlink(); save(manifest_path, manifest)
    monkeypatch.setattr(R.CAPTURE, "validate_design", lambda *_: design())
    rows = R.validated_targets(tmp_path / "design.json", {
        "first_result": ref(first), "second_result": ref(second), "recovery_export": str(exported)})
    assert rows[1]["status"] == "HOLD" and rows[1]["replacement_target"] is False
    assert rows[0]["status"] == "registered_recovered"


def test_export_source_mutation_is_rejected_before_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, exported, _cp03, _cp07 = actual_export_fixture(tmp_path)
    cases = exported / "cases.json"
    cases.write_text(cases.read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(R.CAPTURE, "validate_design", lambda *_: design())
    with pytest.raises(ValueError, match="cases.json"):
        R.validated_targets(tmp_path / "design.json", {
            "first_result": ref(first), "second_result": ref(second), "recovery_export": str(exported)})


def test_execute_reserves_smoke_once_then_each_recovered_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, config = tmp_path / "protocol.json", tmp_path / "config.json"
    protocol.write_text("{}", encoding="utf-8"); config.write_text("{}", encoding="utf-8")
    targets = [
        {"key": "case-03:logs_only", "case_id": "case-03", "status": "registered_recovered"},
        {"key": "case-07:logs_only", "case_id": "case-07", "status": "registered_recovered"},
    ]
    d = {"budget": {"sdk_cap": 3, "wall_cap_s": 9000}}
    cfg = {"design": ref(protocol), "isaac_python": str(tmp_path / "python.bat"), "input_pairs": {}}
    monkeypatch.setattr(R, "preflight", lambda *_: (d, cfg, targets))
    monkeypatch.setattr(R.LEASE, "lease_context", lambda: {"owner": R.LEASE.OWNER, "leaseId": "lease"})
    monkeypatch.setattr(R.GPU, "snapshot", lambda *_: {})
    observed = []
    monkeypatch.setattr(R.CAPTURE, "execute", lambda _p, t, *_a, **kw: observed.append((t["key"], kw["mode"])))
    result = R.execute(protocol, config, "a" * 40, "b" * 40, tmp_path / "out")
    assert observed == [("case-03:logs_only", "smoke"),
                        ("case-03:logs_only", "full"), ("case-07:logs_only", "full")]
    assert result["sdk_intents"] == 3 and result["fault_reference_recaptures"] == 0


def test_zero_success_is_hold_without_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, config = tmp_path / "protocol.json", tmp_path / "config.json"
    protocol.write_text("{}", encoding="utf-8"); config.write_text("{}", encoding="utf-8")
    targets = [{"key": key, "case_id": key[:7], "status": "HOLD"} for key in R.CAPTURE.TARGET_KEYS]
    monkeypatch.setattr(R, "preflight", lambda *_: ({"budget": {"sdk_cap": 3, "wall_cap_s": 9000}},
                                                     {"design": ref(protocol), "input_pairs": {}}, targets))
    monkeypatch.setattr(R.LEASE, "lease_context", lambda: {"owner": R.LEASE.OWNER, "leaseId": "lease"})
    monkeypatch.setattr(R.CAPTURE, "execute", lambda *_a, **_k: pytest.fail("SDK must be zero"))
    result = R.execute(protocol, config, "a" * 40, "b" * 40, tmp_path / "out")
    assert result["status"] == "HOLD" and result["sdk_intents"] == 0


def test_duplicate_sdk_reservation_is_rejected(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, config = tmp_path / "protocol.json", tmp_path / "config.json"
    protocol.write_text("{}", encoding="utf-8"); config.write_text("{}", encoding="utf-8")
    row = {"key": "case-03:logs_only", "case_id": "case-03", "status": "registered_recovered"}
    d = {"budget": {"sdk_cap": 3, "wall_cap_s": 9000}}
    cfg = {"design": ref(protocol), "isaac_python": str(tmp_path / "python.bat"), "input_pairs": {}}
    monkeypatch.setattr(R, "preflight", lambda *_: (d, cfg, [row, dict(row)]))
    monkeypatch.setattr(R.LEASE, "lease_context", lambda: {"owner": R.LEASE.OWNER, "leaseId": "lease"})
    monkeypatch.setattr(R.GPU, "snapshot", lambda *_: {})
    monkeypatch.setattr(R.CAPTURE, "execute", lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="중복"):
        R.execute(protocol, config, "a" * 40, "b" * 40, tmp_path / "out")
    failure = R.B._read(tmp_path / "out" / "failure.json")
    assert failure["sdk_intents"] == 2 and failure["automatic_retry"] is False
