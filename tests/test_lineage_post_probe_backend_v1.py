from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import lineage_post_probe_backend_v1 as M

MODULE = Path(M.__file__ or "")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def base_protocol() -> dict:
    return {"source_sha256_lf": {
                name: "1" * 64 for name in M.P1.RESIDUAL.MINIMUM_BOUND_SOURCES},
            "runtime_source_sha256_lf": {"runtime.py": "2" * 64},
            "watchdogs_s": {"legacy": 600, "optimizer": 900},
            "isaac_python_relative": "_isaac_sim/python.bat"}


def fixture_source(tmp_path: Path, *, remaining: list[str]) -> tuple[Path, dict]:
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    post = save(tmp_path / "observation.json", {"fixture": True})
    correction = "restore_body_frame_reward"
    value = {
        "source_binding": {"checkpoint_sha256": M.sha(checkpoint),
                           "telemetry_sha256": "3" * 64,
                           "repair_receipt_sha256": "4" * 64},
        "private_metadata": {
            "sources": {"repair_receipt": str(tmp_path / "receipt.json"),
                        "lab": str(tmp_path / "lab")},
            "host_only": {"original_faults": ["F1", "R1"], "remaining_faults": remaining,
                          "applied_correction_ids": [correction],
                          "readback": str(tmp_path / "readback.json")},
        },
    }
    receipt = {"seed": 2307, "new_checkpoint": {"path": str(checkpoint),
                "sha256": M.sha(checkpoint)}, "collected_recovery": {
                    "source_binding": {"manifest_sha256": "5" * 64}}}
    save(tmp_path / "receipt.json", receipt)
    return post, value


def prepared(tmp_path: Path, probe: str, *, remaining: list[str]) -> tuple[dict, Path]:
    post_path, post = fixture_source(tmp_path, remaining=remaining)
    lab = tmp_path / "lab"
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").touch()
    stage = tmp_path / "stage"
    with patch.object(M, "verify_protocol", return_value={
            "component_probe_protocol": {
                "path": "evals/lineage_post_probe_backend_v1.py",
                "sha256_lf": M.sha(MODULE, lf=True)}}), \
            patch.object(M.P1, "verify_protocol", return_value=base_protocol()), \
            patch.object(M.POST, "checked_observation", return_value=(post, {})), \
            patch.object(M, "_admit", return_value={"status": "clear"}):
        plan = M.prepare_probe(post_path, probe, stage=stage, lab=lab,
                               protocol_path=MODULE, anchor="a" * 40)
    return plan, stage / "plan.json"


def test_repaired_f1_is_not_reintroduced_for_legacy_probe(tmp_path):
    plan, _path = prepared(tmp_path, "P_reward", remaining=["R1"])
    jobs = M.read(Path(plan["raw_files"]["jobs"]))
    assert jobs["fault"] == "NONE"
    assert plan["projection"]["already_repaired_fault_reintroduced"] is False
    assert plan["private_lineage"]["remaining_faults"] == ["R1"]
    assert plan["scope"]["source_purpose_fabricated"] is False


def test_remaining_f1_uses_f1_legacy_projection(tmp_path):
    plan, _path = prepared(tmp_path, "P_reward", remaining=["F1"])
    assert M.read(Path(plan["raw_files"]["jobs"]))["fault"] == "F1"


def optimizer_rows(manifest: dict, observed: int) -> list[dict]:
    observer = M.P1.RESIDUAL.observer_manifest(manifest)
    rows = []
    for index in range(3):
        rows.append({"schema": M.P1.OPTIMIZER.SCHEMA,
            "manifest_sha256": M.P1.OPTIMIZER.manifest_fingerprint(observer),
            "binding": M.P1.OPTIMIZER.manifest_binding(observer), "update_index": index,
            "expected": {"optimizer_steps": 20, "minibatches": 20},
            "runtime_configuration": {"learning_epochs": 5, "mini_batches": 4},
            "observed": {"optimizer_steps": observed, "minibatches": observed,
                "adam_step_before": {"p": index * observed},
                "adam_step_after": {"p": (index + 1) * observed},
                "adam_step_delta": {"p": observed}, "gradient_present_steps": {"p": observed},
                "pre_step_gradient_norm": [1.0] * observed,
                "learning_rates": [[0.001]] * observed,
                "approx_kl": None, "clip_fraction": None},
            "unavailable": {"approx_kl": "fixture", "clip_fraction": "fixture"},
            "error": None})
    return rows


def test_optimizer_measures_remaining_r1_as_four_vs_twenty(tmp_path):
    plan, _path = prepared(tmp_path, M.OPTIMIZER_PROBE, remaining=["R1"])
    manifest = M.read(Path(plan["raw_files"]["manifest"]))
    Path(plan["raw_files"]["records"]).write_text(
        "".join(json.dumps(row) + "\n" for row in optimizer_rows(manifest, 4)),
        encoding="utf-8")
    save(Path(plan["raw_files"]["readback"]), {
        "observer_manifest": M.P1.RESIDUAL.observer_manifest(manifest)})
    measured = M._optimizer_measurement(plan)
    assert measured["status"] == "abnormal"
    assert measured["records"] == 3
    assert manifest["arm"] == "FIRST"
    assert manifest["original_faults"] == ["F1", "R1"]
    assert manifest["applied_corrections"] == ["restore_body_frame_reward"]


@pytest.mark.parametrize("mutation", ["checkpoint", "telemetry", "receipt"])
def test_checkpoint_receipt_or_telemetry_swap_is_rejected(tmp_path, mutation):
    plan, plan_path = prepared(tmp_path, "P_episode", remaining=["R1"])
    changed = deepcopy(plan)
    if mutation == "checkpoint":
        changed["checkpoint"]["sha256"] = "f" * 64
    elif mutation == "telemetry":
        changed["telemetry_sha256"] = "e" * 64
    else:
        changed["private_lineage"]["prior_valid_not_recovered_receipt_sha256"] = "d" * 64
    save(plan_path, changed)
    wrapper = {"component_probe_protocol": {
        "path": "evals/lineage_post_probe_backend_v1.py",
        "sha256_lf": M.sha(MODULE, lf=True)}}
    source = {
        "kind": plan["source_kind"], "source_arm": plan["source_arm"], "seed": plan["seed"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "checkpoint": plan["checkpoint"], "telemetry_sha256": plan["telemetry_sha256"],
        "original_faults": plan["private_lineage"]["original_faults"],
        "remaining_faults": plan["private_lineage"]["remaining_faults"],
        "applied_corrections": plan["private_lineage"]["applied_corrections"],
        "readback": plan["private_lineage"]["readback"],
        "prior_not_recovered_receipt_sha256": "4" * 64}
    with patch.object(M, "verify_protocol", return_value=wrapper), \
            patch.object(M.P1, "verify_protocol", return_value=base_protocol()), \
            patch.object(M, "_case_source", return_value=({}, {}, {
                **source})), pytest.raises(ValueError, match="lineage"):
        M._verify_plan(plan_path)


def test_reference_requires_none_input(tmp_path):
    lab = tmp_path / "lab"
    collected = save(tmp_path / "reference.json", {})
    with patch.object(M.P1, "_source", return_value=({}, {"arm": "F1"}, {})), \
            pytest.raises(ValueError, match="NONE"):
        M._reference_source(collected, lab)


def test_backend_scope_restores_original_functions_and_receipt_schema():
    originals = (M.P1._verify_plan, M.P1._verify_auxiliary_inputs,
                 M.P1._legacy_measurement, M.P1.RECEIPT_SCHEMA)
    with M._backend_scope():
        assert M.P1._verify_plan is M._verify_plan
        assert M.P1.RECEIPT_SCHEMA == M.RECEIPT_SCHEMA
    assert originals == (M.P1._verify_plan, M.P1._verify_auxiliary_inputs,
                         M.P1._legacy_measurement, M.P1.RECEIPT_SCHEMA)


def test_execute_uses_real_owned_folder_and_restores_scope(tmp_path):
    plan, plan_path = prepared(tmp_path, "P_reward", remaining=["R1"])
    auth = save(tmp_path / "authorization.json", {})
    called = {"owned_folder": False}

    def fake_execute(path, authorization, *, owned, check_lease, check_gpu):
        assert path == plan_path.resolve() and authorization == auth
        folder = Path(plan["stage"]) / "owned"
        folder.mkdir()
        called["owned_folder"] = folder.is_dir()
        return {"schema": M.RECEIPT_SCHEMA, "status": "complete"}

    original = M.P1._verify_plan
    with patch.object(M, "_verify_plan", return_value=plan), \
            patch.object(M, "_admit", return_value={"status": "clear"}), \
            patch.object(M.P1, "execute_probe", side_effect=fake_execute):
        receipt = M.execute_probe(plan_path, auth, owned=lambda *_args: {})
    assert receipt["status"] == "complete" and called["owned_folder"] is True
    assert M.P1._verify_plan is original


def test_failure_sidecar_keeps_filename_traceback_errno(tmp_path):
    try:
        raise FileNotFoundError(2, "missing", "checkpoint.pt")
    except FileNotFoundError as error:
        M._failure(tmp_path / "failure.json", "execute", error)
    value = M.read(tmp_path / "failure.json")
    assert value["filename"] == "checkpoint.pt" and value["errno"] == 2
    assert "FileNotFoundError" in value["traceback"]
    assert value["post_block_roles_started"] == 0


def test_existing_lease_scope_is_reused_without_nested_context(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(M.P1.B, "lease_context", lambda: calls.append("base") or {
        "leaseId": "0" * 36, "owner": M.P1.B.OWNER, "state": "running"})
    monkeypatch.setattr(M.RETRY, "lease_read_scope",
                        lambda _path: (_ for _ in ()).throw(AssertionError("nested scope")))
    M.LEASE._LOCAL.events = []
    try:
        reader = M._lease_reader(tmp_path)
        assert reader()["leaseId"] == "0" * 36
    finally:
        M.LEASE._LOCAL.events = None
    assert calls == ["base"]


def test_standalone_lease_read_opens_evidence_scope(tmp_path, monkeypatch):
    entered = []

    class Scope:
        def __enter__(self):
            entered.append("enter")

        def __exit__(self, *_args):
            entered.append("exit")

    monkeypatch.setattr(M.RETRY, "lease_read_scope", lambda path: entered.append(path) or Scope())
    monkeypatch.setattr(M.P1.B, "lease_context", lambda: {"leaseId": "1" * 36})
    M.LEASE._LOCAL.events = None
    result = M._lease_reader(tmp_path)()
    assert result["leaseId"] == "1" * 36
    assert entered[1:] == ["enter", "exit"]
    assert Path(entered[0]).name == "0001.json"
