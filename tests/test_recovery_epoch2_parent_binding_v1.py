# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import recovery_epoch2_parent_binding_v1 as M
import test_local_pair_four_arm_v1 as F


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    base = F.source(tmp_path / "base")
    values: list[dict] = []
    by_input: dict[Path, dict] = {}
    for index, (case_id, arm) in enumerate(SECOND.TARGETS, 1):
        epoch2 = save(tmp_path / f"{case_id}-{arm}/epoch2.json", {"target": f"{case_id}:{arm}"})
        receipt = save(tmp_path / f"{case_id}-{arm}/first-receipt.json", {"receipt": index})
        receipt_sha = SECOND.sha(receipt)
        checkpoint = f"{index:x}" * 64
        correction = "restore_body_frame_reward"
        prior = {"verdict": "not_recovered", "correction_id": correction,
                 "result_sha256": receipt_sha, "checkpoint_sha256": checkpoint}
        bindings = deepcopy(base["packet"]["input_bindings"])
        bindings["case"]["checkpoint_sha256"] = checkpoint
        public = SECOND.DRIVER.C.V.S.build_packet(
            epoch=2, applied_corrections=[correction], prior_receipt=prior,
            facts=base["packet"]["facts"], input_bindings=bindings, budget_left=3)
        backend = SECOND.DRIVER.C.V.S.build_packet(
            epoch=2, applied_corrections=[correction], prior_receipt=prior,
            facts=base["backend_packet"]["facts"], input_bindings=bindings, budget_left=3)
        source = {
            "schema": SECOND.SOURCE_SCHEMA, "case_id": case_id, "policy": arm,
            "packet": public, "backend_packet": backend,
            "qualification": {"arm": "COMPOUND", "label": "eligible", "diagnosis_eligible": True},
            "source_binding": {
                "epoch2_input_sha256": SECOND.sha(epoch2),
                "first_receipt_sha256": receipt_sha,
                "first_component_sha256_lf": "a" * 64,
                "case_checkpoint_sha256": checkpoint,
                "reference_checkpoint_sha256": bindings["reference"]["checkpoint_sha256"],
            },
            "index": deepcopy(base["prepared"]),
            "inputs": {"epoch2_input": SECOND.ref(epoch2), "first_receipt": SECOND.ref(receipt)},
            "lab": str(tmp_path), "first_correction": correction,
            "original_faults": ["F1", "R1"], "remaining_faults": ["R1"],
            "providers": None, "visual_evidence": None,
            "approval_granted": False, "api_authorized": False,
        }
        values.append(source)
        by_input[epoch2.resolve()] = source

    def rebuild(path: Path, **_kwargs: object) -> dict:
        return deepcopy(by_input[Path(path).resolve()])

    monkeypatch.setattr(SECOND, "_prepare_source_value", rebuild)
    return values


def test_scope_runs_real_driver_snapshot_and_revalidates_history(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registered = sources(tmp_path, monkeypatch)
    original = SECOND._DRIVER.C
    observed_steps = 0
    with M.epoch2_snapshot_scope(registered) as parents:
        for source in registered:
            arm = source["policy"]
            run_id = f"r4_{source['case_id'].replace('-', '_')}_{arm}"
            protocol = save(tmp_path / f"{run_id}-protocol.json", {"fixture": True})
            result = SECOND._DRIVER.run_arm(
                source, arm=arm, run_id=run_id, output=tmp_path / run_id,
                protocol_path=protocol, execution_anchor="a" * 40,
                complete=F.valid_model, execute_probe=F.fake_probe,
                verify_probe=F.verify, allow_fixture=True)
            observed_steps += len(result["steps"])
            context = json.loads((tmp_path / run_id / "step01/context.json").read_text("utf-8"))
            snapshot = context["snapshot"]
            assert snapshot["inputs"]["parent"] == parents[f"{source['case_id']}:{arm}"]
            final = SECOND._DRIVER.C.V.build_snapshot(
                source["packet"], run_id, SECOND._DRIVER._storage_policy(arm),
                SECOND.DRIVER.C.V.file_digest(SECOND.DRIVER.C.V.PROTOCOL),
                tmp_path / run_id / "probes.ledger.jsonl")
            assert SECOND._DRIVER.C.V.checked(final) == final
            bound_parent = parents[f"{source['case_id']}:{arm}"]
            positional = SECOND._DRIVER.C.V.build_snapshot(
                source["packet"], run_id, SECOND._DRIVER._storage_policy(arm),
                SECOND.DRIVER.C.V.file_digest(SECOND.DRIVER.C.V.PROTOCOL),
                tmp_path / run_id / "probes.ledger.jsonl", bound_parent, False)
            keyword = SECOND._DRIVER.C.V.build_snapshot(
                source["packet"], run_id, SECOND._DRIVER._storage_policy(arm),
                SECOND.DRIVER.C.V.file_digest(SECOND.DRIVER.C.V.PROTOCOL),
                tmp_path / run_id / "probes.ledger.jsonl",
                parent=bound_parent, masked=False)
            assert positional == keyword == final
    assert observed_steps > 0
    assert SECOND._DRIVER.C is original


def test_parent_binding_rejects_cross_target_checkpoint_prior_and_run_id(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registered = sources(tmp_path, monkeypatch)
    source = registered[0]
    arm = source["policy"]
    run_id = f"r4_{source['case_id'].replace('-', '_')}_{arm}"
    policy = SECOND._DRIVER._storage_policy(arm)
    parent = M.parent_binding(source, run_id, policy)
    assert parent["checkpoint_sha256"] == source["packet"]["input_bindings"]["case"]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="run_id"):
        M.parent_binding(source, run_id + "_other", policy)
    with pytest.raises(ValueError, match="storage policy"):
        M.parent_binding(source, run_id, SECOND._DRIVER._storage_policy(registered[1]["policy"]))

    changed = deepcopy(source)
    changed["packet"]["prior_receipt"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="원문|checkpoint"):
        M.epoch2_snapshot_scope([changed]).__enter__()
    changed = deepcopy(source)
    changed["inputs"]["first_receipt"] = deepcopy(registered[1]["inputs"]["first_receipt"])
    with pytest.raises(ValueError, match="원문|receipt"):
        M.epoch2_snapshot_scope([changed]).__enter__()


def test_scope_rejects_packet_swap_and_restores_after_external_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registered = sources(tmp_path, monkeypatch)
    source = registered[0]
    original = SECOND._DRIVER.C
    with M.epoch2_snapshot_scope(registered), pytest.raises(ValueError, match="packet|run_id"):
        SECOND._DRIVER.C.V.build_snapshot(
            registered[1]["packet"],
            f"r4_{source['case_id'].replace('-', '_')}_{source['policy']}",
            SECOND._DRIVER._storage_policy(source["policy"]),
            SECOND.DRIVER.C.V.file_digest(SECOND.DRIVER.C.V.PROTOCOL),
            tmp_path / "swapped-ledger.jsonl")
    assert SECOND._DRIVER.C is original

    scope = M.epoch2_snapshot_scope([source])
    scope.__enter__()
    SECOND._DRIVER.__dict__["C"] = ModuleSentinel = object()
    with pytest.raises(RuntimeError, match="외부"):
        scope.__exit__(None, None, None)
    assert SECOND._DRIVER.C is original and ModuleSentinel is not original

