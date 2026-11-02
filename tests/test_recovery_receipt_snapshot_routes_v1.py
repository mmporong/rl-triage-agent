# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import recovery_epoch2_parent_binding_v1 as BIND
import recovery_epoch2_probe_boundary_v1 as BOUNDARY
import recovery_receipt_snapshot_routes_v1 as M
import test_local_pair_probe_host_v1 as HOST_FIXTURE
import test_recovery_epoch2_probe_boundary_v1 as PROBE_FIXTURE


def active_scopes(source: dict, parents: dict):
    return BOUNDARY.backend_parent_scope([source], parents=parents)


def test_public_backend_candidate_and_final_snapshot_routes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _rows = PROBE_FIXTURE.epoch2_source(tmp_path, monkeypatch)
    target = "case-03:fixed_eligible"
    with (
        BIND.epoch2_snapshot_scope([source]) as parents,
        active_scopes(source, parents) as backend,
        M.receipt_snapshot_routes_scope([source], parents=parents) as routes,
    ):
        view = M.FIRST.DRIVER.C.V
        protocol_sha = view.file_digest(view.PROTOCOL)
        run_id = routes[target]["run_id"]
        policy = routes[target]["policy"]
        public = M.SECOND._HOST.C.V.build_snapshot(
            source["packet"], run_id, policy, protocol_sha,
            tmp_path / "public-ledger.jsonl")
        assert public["inputs"]["parent"] == parents[target]
        assert view.checked(public) == public

        backend_row = backend[target]
        with BOUNDARY.RV.view_scope(BOUNDARY.RV.PROTOCOL, None, fixture=True):
            backend_snapshot = M.SECOND._HOST.C.V.build_snapshot(
                source["backend_packet"], backend_row["backend_run_id"],
                backend_row["backend_policy"],
                BOUNDARY.RV.file_digest(BOUNDARY.RV.PROTOCOL),
                tmp_path / "backend-ledger.jsonl")
        assert backend_snapshot["inputs"]["parent"] == backend_row["backend_parent"]

        result = {"run_id": run_id, "host_storage_policy": policy}
        candidate = M.FIRST._snapshot(
            source["packet"], result, tmp_path / "candidate-ledger.jsonl", [])
        final = M.FIRST.DRIVER.C.V.build_snapshot(
            source["packet"], run_id, policy, protocol_sha,
            tmp_path / "candidate-ledger.jsonl")
        assert candidate == final
        assert candidate["inputs"]["parent"] == parents[target]


def test_epoch1_snapshot_and_driver_build_keep_original_route(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _rows = PROBE_FIXTURE.epoch2_source(tmp_path, monkeypatch)
    epoch1 = HOST_FIXTURE.packet(False)
    result = {"run_id": "epoch1_fixture", "host_storage_policy": "fixed_eligible"}
    with (
        BIND.epoch2_snapshot_scope([source]) as parents,
        active_scopes(source, parents),
        M.receipt_snapshot_routes_scope([source], parents=parents),
    ):
        snapshot = M.FIRST._snapshot(
            epoch1, result, tmp_path / "epoch1-ledger.jsonl", [])
        assert snapshot["inputs"]["parent"] is None
        rebuilt = M.FIRST.DRIVER.C.V.build_snapshot(**snapshot["inputs"])
        assert rebuilt == snapshot


def test_routes_reject_cross_arm_parent_checkpoint_and_unknown_epoch2(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _rows = PROBE_FIXTURE.epoch2_source(tmp_path, monkeypatch)
    target = "case-03:fixed_eligible"
    with (
        BIND.epoch2_snapshot_scope([source]) as parents,
        active_scopes(source, parents),
        M.receipt_snapshot_routes_scope([source], parents=parents),
    ):
        view = M.FIRST.DRIVER.C.V
        protocol_sha = view.file_digest(view.PROTOCOL)
        with pytest.raises(ValueError, match="run_id"):
            view.build_snapshot(
                source["packet"], "r4_case_03_logs_only", "fixed_eligible",
                protocol_sha, tmp_path / "wrong-run.jsonl")
        with pytest.raises(ValueError, match="policy"):
            view.build_snapshot(
                source["packet"], "r4_case_03_fixed_eligible", "model_adaptive",
                protocol_sha, tmp_path / "wrong-policy.jsonl")
        with pytest.raises(ValueError, match="parent"):
            view.build_snapshot(
                source["packet"], "r4_case_03_fixed_eligible", "fixed_eligible",
                protocol_sha, tmp_path / "wrong-parent.jsonl",
                parent={**parents[target], "checkpoint_sha256": "f" * 64})
        unknown = deepcopy(source["packet"])
        unknown["budget_left"] = 2
        with pytest.raises(ValueError, match="미등록 epoch-2"):
            view.build_snapshot(
                unknown, "r4_case_03_fixed_eligible", "fixed_eligible",
                protocol_sha, tmp_path / "unknown.jsonl")

    wrong = deepcopy(source)
    wrong["packet"]["prior_receipt"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        M._registrations([wrong], {target: {
            **parents[target], "checkpoint_sha256": "f" * 64,
        }})


def test_scope_restores_identities_and_detects_external_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _rows = PROBE_FIXTURE.epoch2_source(tmp_path, monkeypatch)
    original = (M.FIRST.DRIVER.C, M.SECOND._HOST.C, M.FIRST._snapshot)
    with (
        BIND.epoch2_snapshot_scope([source]) as parents,
        active_scopes(source, parents),
    ):
        outer = (M.FIRST.DRIVER.C, M.SECOND._HOST.C, M.FIRST._snapshot)
        scope = M.receipt_snapshot_routes_scope([source], parents=parents)
        scope.__enter__()
        M.FIRST.__dict__["_snapshot"] = lambda *_args, **_kwargs: {}
        with pytest.raises(RuntimeError, match="외부"):
            scope.__exit__(None, None, None)
        assert (M.FIRST.DRIVER.C, M.SECOND._HOST.C, M.FIRST._snapshot) == outer
    assert (M.FIRST.DRIVER.C, M.SECOND._HOST.C, M.FIRST._snapshot) == original
