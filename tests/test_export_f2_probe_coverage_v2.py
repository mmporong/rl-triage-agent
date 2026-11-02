# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_f2_probe_coverage_v2 as A


def _packet(*, checkpoint: str = "1") -> dict:
    bindings = {
        "case": {"checkpoint_sha256": checkpoint * 64, "telemetry_sha256": "2" * 64},
        "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
    }
    return A.V1.RUN.DRIVER.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=bindings,
        budget_left=3, execution_budget_s=1200)


def _completed_pair(tmp_path: Path, *, case_id: str = "case-05"):
    ledger = tmp_path / case_id / "oracle.ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"")
    packet = _packet()
    snapshot = A.V1.RUN.DRIVER.C.V.build_snapshot(
        packet, f"f2_oracle_{case_id.replace('-', '_')}",
        A.V1.RUN.HOST_STORAGE_POLICY,
        A.V1.RUN.DRIVER.C.V.file_digest(A.V1.RUN.DRIVER.C.V.PROTOCOL), ledger)
    choice = A.V1.RUN.oracle_choice(snapshot)
    spec = A.V1.RUN.DRIVER.C.request_spec(snapshot, choice)
    registry = A.V1.RUN.DRIVER.C.S.M.RegistryLedger(
        ledger, A.V1.RUN.DRIVER.C.RC.registry())
    request = registry.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    registry.approve(request["request_id"], "cpu-test")
    registry.consume(request["request_id"], spec["prereg"], spec["probe"])
    registry.receipt(request["request_id"], "abnormal", {}, 1.0, 0)
    frozen = deepcopy(snapshot)
    frozen["inputs"]["ledger_path"] = str(ledger.resolve())
    pair_path = tmp_path / case_id / "pair" / "result.json"
    pair_path.parent.mkdir(parents=True)
    pair_path.write_text(json.dumps({
        "public_ledger": {"path": str(ledger.resolve())},
        "public_snapshot_sha256": A.V1.RUN.B.digest(frozen),
        "public_choice": choice,
    }, sort_keys=True), encoding="utf-8")
    return pair_path, ledger, {"packet": packet}, frozen, choice


def test_completed_ledger_v1_is_stale_and_v2_reconstructs_exact_snapshot(tmp_path: Path):
    pair_path, ledger, source, frozen, choice = _completed_pair(tmp_path)
    with pytest.raises(ValueError, match="변조 또는 오래된 snapshot"):
        A.V1._frozen_snapshot(pair_path, source, "case-05", tmp_path / "v1")
    rebuilt, selected = A._frozen_snapshot(pair_path, source, "case-05", tmp_path / "v2")
    assert rebuilt == frozen and selected == choice
    assert rebuilt["inputs"]["ledger_path"] == str(ledger.resolve())
    assert A.V1.RUN.B.digest(rebuilt) == A.V1.read(pair_path)["public_snapshot_sha256"]


@pytest.mark.parametrize("tamper", ["choice", "digest", "packet"])
def test_choice_digest_and_packet_tamper_are_rejected(tmp_path: Path, tamper: str):
    pair_path, _ledger, source, _frozen, _choice = _completed_pair(tmp_path)
    if tamper in {"choice", "digest"}:
        pair = A.V1.read(pair_path)
        if tamper == "choice":
            pair["public_choice"]["probe"] = "P_reward"
        else:
            pair["public_snapshot_sha256"] = "0" * 64
        pair_path.write_text(json.dumps(pair, sort_keys=True), encoding="utf-8")
    else:
        source = {"packet": _packet(checkpoint="9")}
    with pytest.raises(ValueError, match="historical choice|snapshot digest"):
        A._frozen_snapshot(pair_path, source, "case-05", tmp_path / "verify")


def test_v1_build_export_is_scoped_and_v2_provenance_binds_old_anchor(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original = A.V1._frozen_snapshot
    anchor = "e" * 40
    calls = {"count": 0}

    def fake(*_args, **_kwargs):
        calls["count"] += 1
        assert A.V1._frozen_snapshot is A._frozen_snapshot
        return {"schema": A.V1.SCHEMA, "status": "verified", "evidence": {},
                "targets": [], "new_cost": {}, "claims": {}}

    monkeypatch.setattr(A.V1, "build_export", fake)
    value = A.build_export(tmp_path / "result", tmp_path / "protocol", anchor,
                           tmp_path / "plan", lab=tmp_path)
    assert calls["count"] == 1 and A.V1._frozen_snapshot is original
    assert value["validator"] == {
        "schema": A.SCHEMA,
        "historical_reconstruction": "validate_choice_before_exact_ledger_retarget",
        "v2_source_sha256_lf": A._lf_sha(ROOT / A.SOURCE_FILES[0]),
        "frozen_v1_verifier_sha256_lf": A._lf_sha(ROOT / A.SOURCE_FILES[2]),
        "frozen_v1_schema": A.V1.SCHEMA,
        "frozen_v1_source_anchor": anchor,
        "frozen_v1_source_closure_preserved": True,
    }

    def raises(*_args, **_kwargs):
        assert A.V1._frozen_snapshot is A._frozen_snapshot
        raise RuntimeError("fixture")

    monkeypatch.setattr(A.V1, "build_export", raises)
    with pytest.raises(RuntimeError, match="fixture"):
        A.build_export(tmp_path / "result", tmp_path / "protocol", anchor,
                       tmp_path / "plan", lab=tmp_path)
    assert A.V1._frozen_snapshot is original


def test_actual_case05_snapshot_v1_fails_v2_passes_without_mutation(tmp_path: Path):
    base = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025/"
            "vision_goal/ri_full_completion_251/replan_369")
    production = base / "probe_coverage_control_410/production-v1"
    plan_path = production / "plan.json"
    result_path = production / "actual-v1/result.json"
    pair_path = production / "actual-v1/case-05/pair/result.json"
    plan = A.V1.read(plan_path)
    config_path = A.V1._bound(plan["execution_config"], "execution config")
    config = A.V1.read(config_path)
    source_path = A.V1.RUN.B._bound_file(config["cases"]["case-05"]["source"])
    source = A.V1.RUN.DRIVER.checked_source(A.V1.read(source_path))
    pair = A.V1.read(pair_path)
    ledger = Path(pair["public_ledger"]["path"]).resolve(strict=True)
    protected = [plan_path, result_path, config_path, source_path, pair_path, ledger]
    before = {str(path): A.V1.sha(path) for path in protected}
    with pytest.raises(ValueError, match="변조 또는 오래된 snapshot"):
        A.V1._frozen_snapshot(pair_path, source, "case-05", tmp_path / "v1")
    snapshot, choice = A._frozen_snapshot(pair_path, source, "case-05", tmp_path / "v2")
    assert choice == pair["public_choice"]
    assert A.V1.RUN.B.digest(snapshot) == pair["public_snapshot_sha256"]
    assert snapshot["inputs"]["ledger_path"] == str(ledger)
    assert before == {str(path): A.V1.sha(path) for path in protected}
