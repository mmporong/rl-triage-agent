from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

import evals.combination_contract_v1 as C
import evals.combination_record_fixtures_v1 as F

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def public_copy(tmp_path: Path) -> Path:
    target = tmp_path / "bench/fixtures/combination_contract_v1"
    target.parent.mkdir(parents=True)
    shutil.copytree(F.OUTPUT, target)
    return target


def test_public_bundle_has_a_fixed_canonical_digest_without_private_sources():
    inventory = {p.relative_to(F.OUTPUT).as_posix(): F.sha(p) for p in F.OUTPUT.rglob("*.json")}
    assert C.L.digest(inventory) == F.PUBLIC_SHA256
    assert len(inventory) == 11


def test_five_history_states_stop_at_the_first_blocking_gate():
    checked = F.verify_public()
    assert {case: row["status"] for case, row in checked["cases"].items()} == F.EXPECTED
    assert checked["cases"]["anymal_f2_not_applicable"]["trace"] == ["applicability"]
    assert checked["cases"]["anymal_f6_source_failure_absent"]["trace"] == [
        "applicability", "injection", "source"]
    assert checked["cases"]["go2_u1_legacy_scope_insufficient"]["trace"] == [
        "applicability", "injection", "source", "scope"]
    assert checked["cases"]["go2_u1_extended_ready"]["trace"] == list(C.ORDER)
    assert checked["cases"]["go2_f1_ready"]["trace"] == list(C.ORDER)


def test_public_fixture_omits_original_locations_and_marks_development_replay():
    forbidden = ("bench/private/", "codex_support_20261015", "C:\\Users\\", "/home/", "/mnt/")
    for path in F.OUTPUT.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not any(word in text for word in forbidden)
    for case in F.CASES:
        fixture = read(F.OUTPUT / f"cases/{case}.json")
        assert fixture["development_replay"] is True
        assert fixture["historical_ready_is_contract_development_state"] is case.endswith("_ready")
        assert fixture["manifest"]["evidence"][0]["path"].startswith(
            "bench/fixtures/combination_contract_v1/records/")


def test_public_records_preserve_original_lf_fingerprints_without_paths():
    for case in F.CASES:
        record = read(F.OUTPUT / f"records/{case}.json")
        assert record["measurement_kind"] == "post_result_contract_development_replay"
        assert record["source_fingerprints"]
        assert all(name in F.SOURCES and value == F.SOURCES[name][1]
                   for name, value in record["source_fingerprints"].items())


def test_public_record_sha_tampering_is_rejected(tmp_path: Path):
    output = public_copy(tmp_path)
    record = output / "records/go2_f1_ready.json"
    record.write_text(record.read_text(encoding="utf-8").replace(
        '"case": "go2_f1_ready"', '"case": "go2_f1_changed"'), encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="SHA 결합"):
        F.verify_public(root=tmp_path, output=output)


def test_manifest_evidence_sha_tampering_is_rejected():
    fixture = read(F.OUTPUT / "cases/go2_u1_extended_ready.json")
    manifest = deepcopy(fixture["manifest"])
    manifest["evidence"][0]["sha256_lf"] = "0" * 64
    checked = C.evaluate(manifest)
    assert checked["status"] == "invalid"
    assert checked["reason"].startswith("contract_evidence_invalid:")


def test_source_sha_tampering_is_rejected_before_normalization(tmp_path: Path, monkeypatch):
    relative, expected = F.SOURCES["adapter"]
    source = ROOT / relative
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes() + b"\n# tampered\n")
    monkeypatch.setattr(F, "SOURCES", {"adapter": (relative, expected)})
    with pytest.raises(ValueError, match="원본 SHA 결합이 다르다"):
        F.verify_sources(tmp_path)


def test_build_is_write_once(tmp_path: Path, monkeypatch):
    generated = {p.relative_to(F.OUTPUT).as_posix(): p.read_bytes() for p in F.OUTPUT.rglob("*.json")}
    monkeypatch.setattr(F, "payloads", lambda root: generated)
    output = tmp_path / "fixtures"
    result = F.build(output=output)
    assert result == {"status": "built", "files": 11, "cases": 5}
    with pytest.raises(FileExistsError, match="이미 있다"):
        F.build(output=output)


def test_f1_none_noop_is_explicitly_a_cpu_adapter_development_replay():
    record = read(F.OUTPUT / "records/go2_f1_ready.json")
    replay = record["payload"]["none_noop_development_replay"]
    assert replay["status"] == "passed" and replay["changed_targets"] == replay["restored_targets"] == []
    assert replay["before"] == replay["after"]
    assert {"adapter", "adapter_test", "f1_none_noop", "f1_none_noop_runner"}.issubset(record["source_fingerprints"])


def test_manifest_fact_change_with_unchanged_evidence_sha_is_invalid():
    fixture = read(F.OUTPUT / "cases/go2_f1_ready.json")
    m = fixture["manifest"]
    m["stages"]["injection"]["after"]["track_lin_vel_xy_exp"] = "f" * 64
    assert C.evaluate(m)["status"] == "invalid"


def test_record_and_manifest_sha_simultaneous_change_is_rejected(tmp_path):
    output = public_copy(tmp_path)
    path = output / "records/go2_f1_ready.json"
    record = read(path)
    record["stages"]["injection"]["after"]["track_lin_vel_xy_exp"] = "f" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    casepath = output / "cases/go2_f1_ready.json"
    fixture = read(casepath)
    fixture["manifest"]["stages"] = record["stages"]
    fixture["manifest"]["evidence"][0]["sha256_lf"] = F.sha(path)
    casepath.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA"):
        F.verify_public(tmp_path, output)


def test_historical_blocked_cases_have_zero_requests_and_ready_has_complete_targets(tmp_path):
    for case in F.CASES:
        m = read(F.OUTPUT / f"cases/{case}.json")["manifest"]
        mechanism = C.FAULTS[m["fault"]][0]
        state = {"status": "confirmed", "hypotheses": [mechanism], "observed": {"P_reward": "abnormal"},
                 "selection_actor": "registered_rule"}
        ledger = C.CorrectionLedger(tmp_path / f"{case}.jsonl", registry=C.registry())
        result = C.propose(m, state, ledger)
        if F.EXPECTED[case] == "ready":
            assert result["event"] == "proposed"
            assert result["evidence"]["restoration_targets"] == sorted(C.FAULTS[m["fault"]][1])
        else:
            assert result["event"] == "held" and ledger.requests() == {}
