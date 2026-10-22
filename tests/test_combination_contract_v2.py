"""적용성 계약 v2(모터 P_torque·종료 P_episode 범위 관문)의 CPU 수용 검사."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "tests"):
    sys.path.insert(0, str(folder))
import combination_contract_v1 as K1  # noqa: E402
import combination_contract_v2 as K  # noqa: E402
import combination_record_fixtures_v1 as R1  # noqa: E402
import combination_record_fixtures_v2 as R  # noqa: E402
from test_combination_contract_v1 import make_case, refresh_evidence  # noqa: E402

# 범주별 확정 측정: (abnormal 측정, normal 측정, 기준값). 문턱은 동결 probe_loop.THRESHOLDS.
SCOPES = {"F2": ("P_torque", {"low_speed_saturation": 0.08}, {"low_speed_saturation": 0.005},
                 {"low_speed_saturation": 0.0005}),
          "F6": ("P_episode", {"timeout_ratio": 0.4}, {"timeout_ratio": 0.99}, None)}
MODES = {"F2": "restart", "F6": "resume"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_manifest(case: str) -> dict:
    return read(R.OUTPUT / f"cases/{case}.json")["manifest"]


def v2_case(tmp_path, fault, *, abnormal=True):
    m = make_case(tmp_path, fault=fault)
    probe, bad, good, reference = SCOPES[fault]
    m["stages"]["scope"] = {"probe": probe, "measurement": bad if abnormal else good, "reference": reference}
    m["stages"]["recovery"] = {"mode": MODES[fault]}
    refresh_evidence(m, tmp_path)
    return m


def state_for(manifest, actor="registered_rule"):
    probe = K.confirming_probe(manifest)
    return {"status": "confirmed", "hypotheses": [K.FAULTS[manifest["fault"]][0]],
            "observed": {probe: "abnormal"}, "selection_actor": actor}


def test_public_v2_bundle_digest_and_both_motor_records_are_ready():
    inventory = {p.relative_to(R.OUTPUT).as_posix(): R1.sha(p) for p in R.OUTPUT.rglob("*.json")}
    assert K.L.digest(inventory) == R.PUBLIC_SHA256 and len(inventory) == 5
    checked = R.verify_public()
    assert checked["status"] == "valid"
    assert {case: row["status"] for case, row in checked["cases"].items()} == R.EXPECTED
    assert all(row["trace"] == list(K.ORDER) for row in checked["cases"].values())


def test_v1_history_fixtures_keep_the_same_state_under_v2():
    for case, expected in R1.EXPECTED.items():
        manifest = read(R1.OUTPUT / f"cases/{case}.json")["manifest"]
        old, new = K1.evaluate(deepcopy(manifest)), K.evaluate(deepcopy(manifest))
        assert new["status"] == old["status"] == expected
        assert new["trace"] == old["trace"] and new["contract"] == K.VERSION


@pytest.mark.parametrize("case", sorted(R.CASES))
def test_motor_records_are_rejected_by_v1_and_ready_with_registered_targets_under_v2(case):
    manifest = fixture_manifest(case)
    old = K1.evaluate(deepcopy(manifest))
    assert old["status"] == "invalid" and old["reason"] == "scope"
    assert old["trace"][-1]["details"]["reason"] == "scope_verification_unregistered"
    new = K.evaluate(deepcopy(manifest))
    assert new["status"] == "ready" and new["recovery_mode"] == "restart"
    assert new["active_targets"] == new["restoration_targets"] == [R.TARGET]
    scope = new["trace"][K.ORDER.index("scope")]["details"]
    assert scope["probe_scope"]["classification"] == "abnormal"
    assert scope["scope_basis"] == "registered_targets_cover_active_fault_targets"


def test_public_v2_fixture_omits_private_locations_and_marks_development_replay():
    forbidden = ("bench/private/", "claude_reward_track", "C:\\Users\\", "/home/", "/mnt/")
    for path in R.OUTPUT.rglob("*.json"):
        assert not any(word in path.read_text(encoding="utf-8") for word in forbidden)
    for case in R.CASES:
        fixture = read(R.OUTPUT / f"cases/{case}.json")
        assert fixture["development_replay"] is True
        assert fixture["manifest"]["evidence"][0]["path"].startswith(R.PUBLIC_PREFIX + "/records/")
        record = read(R.OUTPUT / f"records/{case}.json")
        assert all(record["source_fingerprints"][name] == R.SOURCES[name][1] for name in record["source_fingerprints"])
        replay = record["payload"]["none_noop_development_replay"]
        assert replay["status"] == "passed" and replay["before"] == replay["after"]


@pytest.mark.parametrize("fault", sorted(SCOPES))
def test_synthetic_confirmed_scope_is_ready_and_a_normal_measurement_is_invalid(tmp_path, fault):
    ready = K.evaluate(v2_case(tmp_path, fault), tmp_path)
    assert ready["status"] == "ready" and [row["stage"] for row in ready["trace"]] == list(K.ORDER)
    assert ready["recovery_mode"] == MODES[fault]
    normal = K.evaluate(v2_case(tmp_path, fault, abnormal=False), tmp_path)
    assert normal["status"] == "invalid" and normal["reason"] == "scope"
    assert normal["trace"][-1]["details"]["reason"] == "confirming_measurement_not_abnormal"


def assert_scope_blocks(manifest, root):
    result = K.evaluate(manifest, root)
    assert result["status"] == "invalid" and result["reason"] == "scope"


@pytest.mark.parametrize("fault", sorted(SCOPES))
def test_disabling_the_v2_scope_gate_breaks_acceptance(tmp_path, monkeypatch, fault):
    manifest = v2_case(tmp_path, fault, abnormal=False)
    assert_scope_blocks(manifest, tmp_path)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    assert K.propose(manifest, state_for(manifest), ledger, root=tmp_path)["event"] == "held"
    assert ledger.requests() == {}
    monkeypatch.setitem(K.GATES, "scope", lambda m, row: (None, {"mutation": "gate_disabled"}))
    with pytest.raises(AssertionError):
        assert_scope_blocks(manifest, tmp_path)
    assert K.evaluate(manifest, tmp_path)["status"] == "ready"


def test_active_target_outside_the_registered_correction_is_scope_insufficient(tmp_path, monkeypatch):
    extra = "DCMotor._extra_target"
    monkeypatch.setitem(K.FAULTS, "F2", ("actuator", ("DCMotor._clip_effort", extra)))
    # 주입 단계는 결함 표적 둘을 모두 읽었고, 등록 교정은 그중 하나만 되돌린다.
    result = K.evaluate(v2_case(tmp_path, "F2"), tmp_path)
    assert result["status"] == "scope_insufficient" and result["trace"][-1]["details"]["uncovered_targets"] == [extra]


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(probe="P_reward"),
    lambda s: s.update(probe="P_episode"),
    lambda s: s.pop("reference"),
    lambda s: s.update(extra=1),
    lambda s: s.update(measurement=[0.08]),
    lambda s: s.update(reference=0.0005),
    lambda s: s.update(measurement={"low_speed_saturation": "high"}),
])
def test_malformed_motor_scope_never_creates_request(tmp_path, mutation):
    m = v2_case(tmp_path, "F2")
    mutation(m["stages"]["scope"])
    refresh_evidence(m, tmp_path)
    result = K.evaluate(m, tmp_path)
    assert result["status"] == "invalid"
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    assert K.propose(m, state_for(m), ledger, root=tmp_path)["event"] == "held" and ledger.requests() == {}


@pytest.mark.parametrize("fault", sorted(SCOPES))
def test_confirmed_state_must_name_the_category_probe(tmp_path, fault):
    m = v2_case(tmp_path, fault)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    wrong = {**state_for(m), "observed": {"P_reward": "abnormal"}}
    assert K.propose(m, wrong, ledger, root=tmp_path)["reason"] == "diagnosis_unconfirmed"
    assert ledger.requests() == {}
    proposed = K.propose(m, state_for(m, "model"), ledger, root=tmp_path)
    assert proposed["event"] == "proposed" and proposed["evidence"]["selection_actor"] == "model"


def test_motor_request_needs_approval_and_consumes_once(tmp_path):
    m = v2_case(tmp_path, "F2")
    state = state_for(m)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    rid = K.propose(m, state, ledger, root=tmp_path)["request_id"]
    with pytest.raises(K.LedgerError):
        K.consume(m, state, ledger, rid, root=tmp_path)
    ledger.approve(rid, "operator")
    K.consume(m, state, ledger, rid, root=tmp_path)
    with pytest.raises(K.LedgerError):
        K.consume(m, state, ledger, rid, root=tmp_path)
    assert [event["event"] for event in ledger.events()] == ["proposed", "approved", "consumed"]


def test_record_tampering_is_rejected(tmp_path):
    output = tmp_path / R.PUBLIC_PREFIX
    output.parent.mkdir(parents=True)
    shutil.copytree(R.OUTPUT, output)
    path = output / "records/go2_f2_ready_s2046.json"
    path.write_text(path.read_text(encoding="utf-8").replace("go2_f2_ready_s2046", "go2_f2_changed"),
                    encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="SHA 결합"):
        R.verify_public(root=tmp_path, output=output)


def test_source_sha_tampering_is_rejected_before_normalization(tmp_path, monkeypatch):
    relative, expected = R.SOURCES["f2_none_noop_runner"]
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / relative).read_bytes() + b"\n# tampered\n")
    monkeypatch.setattr(R, "SOURCES", {"f2_none_noop_runner": (relative, expected)})
    with pytest.raises(ValueError, match="원본 SHA 결합이 다르다"):
        R.verify_sources(tmp_path)


def test_build_is_write_once(tmp_path, monkeypatch):
    generated = {p.relative_to(R.OUTPUT).as_posix(): p.read_bytes() for p in R.OUTPUT.rglob("*.json")}
    monkeypatch.setattr(R, "payloads", lambda root: generated)
    output = tmp_path / "fixtures"
    assert R.build(output=output) == {"status": "built", "files": 5, "cases": 2}
    with pytest.raises(FileExistsError, match="이미 있다"):
        R.build(output=output)
