from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_first_recovery_v1 as M


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": M._sha(path)}


def retry_row(tmp_path: Path, policy: str, *, confirmed: bool = True) -> dict:
    result = {
        "status": "complete", "end_state": "confirmed" if confirmed else "HOLD",
        "hold_reason": None if confirmed else "diagnosis_invalid",
        "repair_supported": confirmed, "correction_executed": False,
        "training_executed": False, "epoch2_registered": False,
    }
    path = write(tmp_path / f"{policy}.json", result)
    return {"order": 1 if policy == "registry_next" else 2, "case_id": M.TARGET_CASE,
            "policy": policy, "arm": "COMPOUND", "fresh_view_digest": "a" * 64,
            "result": result, "result_path": str(path.resolve()),
            "result_sha256": M._sha(path), "original_failure": {},
            "classification": {"confirmed": confirmed, "correct_confirmation": confirmed,
                               "compound_partial": confirmed}}


def manifest_rows(status: str = "HOLD") -> list[dict]:
    return [
        {"order": index, "case_id": M.TARGET_CASE, "policy": policy,
         "arm": "COMPOUND", "seed": M.TARGET_SEED, "status": status,
         "hold_reason": "diagnosis_invalid" if status == "HOLD" else None,
         "candidate": None, "fresh_view_digest": "a" * 64}
        for index, policy in enumerate(M.POLICIES, 1)
    ]


def test_prepare_protocol_fixes_first_only_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recovery_path = ROOT / "bench/protocols/lineage_recovery_execution_20261031_v2.json"
    recovery = M._read(recovery_path)
    retry = {
        "schema": M.RETRY.SCHEMA,
        "component_protocols": {"comparison": recovery["component_protocols"]["comparison_batch"]},
        "runtime_source_sha256_lf": recovery["runtime_source_sha256_lf"],
    }
    retry_path = write(tmp_path / "retry.json", retry)
    original_relative = M._relative
    monkeypatch.setattr(
        M, "_relative",
        lambda path: "bench/protocols/retry-fixture.json"
        if Path(path).resolve() == retry_path.resolve() else original_relative(path),
    )
    protocol = M.prepare_protocol(retry_path, recovery_path)
    assert protocol["target"] == {"case_id": "case-07", "seed": 2307,
                                  "arm": "COMPOUND", "policies": list(M.POLICIES)}
    assert protocol["scope"]["first_correction_only"] is True
    assert protocol["scope"]["second_epoch"] is False
    assert protocol["training"] == {
        "mode": "fresh_restart", "iterations": 300, "seed": 2307,
        "source_checkpoint_role": "provenance_only_never_loaded",
        "arm": "FIRST", "model_calls": 0,
    }
    assert protocol["predictions"]["single_component_correction"] == (
        "not_recovered_due_to_remaining_fault"
    )
    assert len(protocol["runtime_source_sha256_lf"]) == 9


@pytest.mark.parametrize("confirmed,expected", [(True, True), (False, False)])
def test_eligible_retry_row_requires_supported_compound_confirmation(
        tmp_path: Path, confirmed: bool, expected: bool) -> None:
    assert M._eligible_retry_row(retry_row(tmp_path, "registry_next", confirmed=confirmed)) is expected


def test_checked_retry_result_binds_raw_rows_and_backend(tmp_path: Path) -> None:
    rows = [retry_row(tmp_path, policy) for policy in M.POLICIES]
    result = {"schema": M.RETRY.RESULT_SCHEMA, "status": "complete",
              "evidence_scope": "actual_backend", "rows": rows,
              "original_results_immutable": True, "correction_executed": False,
              "training_executed": False, "epoch2_registered": False}
    path = write(tmp_path / "result.json", result)
    checked = M._checked_retry_result(path, {"fresh_public_view": {"digest": "a" * 64}})
    assert checked == result
    rows[0]["result_sha256"] = "0" * 64
    write(tmp_path / "bad.json", {**result, "rows": rows})
    with pytest.raises(ValueError, match="raw SHA"):
        M._checked_retry_result(tmp_path / "bad.json", {"fresh_public_view": {"digest": "a" * 64}})


def test_lease_wait_uses_fresh_evidence_directory_for_every_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def lease(path: Path) -> dict:
        calls.append(Path(path))
        return {"leaseId": "0" * 36, "owner": "owner", "state": "running"}

    def await_clear(stage: Path, **kwargs) -> dict:
        assert Path(stage) == tmp_path
        assert kwargs["lease_check"]()["state"] == "running"
        assert kwargs["lease_check"]()["state"] == "running"
        return {"status": "clear"}

    monkeypatch.setattr(M.LEASE, "lease_context", lease)
    monkeypatch.setattr(M.V1.GPU_WAIT, "await_clear", await_clear)
    assert M._lease_wait(tmp_path) == {"status": "clear"}
    assert calls == [tmp_path / "lease-reads/0001", tmp_path / "lease-reads/0002"]


def test_smoke_wraps_runner_with_existing_owned_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = write(tmp_path / "protocol.json", {})
    manifest = {"protocol": ref(protocol_path), "anchor": "a" * 40,
                "rows": manifest_rows(), "lab": str(tmp_path),
                "isaac_python": sys.executable}
    fixed = {"max_iterations": 3}
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M.V1, "_training_manifest", lambda *_a, **_k: fixed)
    monkeypatch.setattr(M.V1.GUARD, "EVIDENCE", "training-evidence.json")

    runner_calls: list[Path] = []

    def runner(folder: Path, _args: list[str], _cwd: Path, _watchdog: float) -> dict:
        assert folder.is_dir()
        runner_calls.append(folder)
        return {"exit_code": 0}

    def run_training(stage: Path, _fixed: dict, _manifest: dict, _protocol: dict, **kwargs):
        execution = kwargs["runner"](stage / "owned", [], tmp_path, 300)
        training = stage / "training"
        training.mkdir(parents=True)
        write(stage / "manifest.json", fixed)
        write(training / "training-evidence.json", {"checkpoint": {"sha256": "b" * 64}})
        write(stage / "owned-result.json", execution)
        return {**execution, "wall_s": 1.0}, {"checkpoint": {"sha256": "b" * 64}}

    monkeypatch.setattr(M.V1, "_run_training", run_training)
    result = M.smoke(tmp_path / "manifest.json", tmp_path / "smoke-root", runner=runner)
    assert result["status"] == "complete"
    assert result["seed"] == 2307
    assert runner_calls == [tmp_path / "smoke-root/smoke/owned"]


def test_execute_rejects_missing_operator_approval_before_runner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {"status": "validated_candidate"})
    rows = manifest_rows("candidate")
    for row in rows:
        row["candidate"] = ref(candidate)
    manifest = {"rows": rows}
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M, "verify_smoke", lambda *_a: {"path": "smoke", "result_sha256": "a" * 64})
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": []})
    with pytest.raises(ValueError, match="approval entries"):
        M.execute(tmp_path / "manifest.json", approval, tmp_path / "smoke", tmp_path / "out",
                  runner=lambda *_a, **_k: pytest.fail("runner must not start"))
    assert not (tmp_path / "out").exists()


def test_execute_holds_unconfirmed_rows_without_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"rows": manifest_rows()}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M, "verify_smoke", lambda *_a: {"path": "smoke", "result_sha256": "a" * 64})
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": []})
    clock = iter((0.0, 0.1))
    result = M.execute(
        manifest_path, approval, tmp_path / "smoke", tmp_path / "out",
        runner=lambda *_a, **_k: pytest.fail("runner must not start"), clock=lambda: next(clock),
    )
    assert result["status"] == "complete"
    assert [row["status"] for row in result["receipts"]] == ["HOLD", "HOLD"]
    assert all(row["sdk_jobs"] == 0 and row["followup_epoch_allowed"] is False
               for row in result["receipts"])


def test_first_candidate_failure_starts_no_next_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {"status": "validated_candidate"})
    rows = manifest_rows("candidate")
    entries = []
    for row in rows:
        row["candidate"] = ref(candidate)
        entries.append({"case_id": row["case_id"], "policy": row["policy"],
                        "candidate_sha256": row["candidate"]["sha256"], "decision": "approve"})
    manifest = {"rows": rows, "protocol": {"sha256": "a" * 64}}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M, "verify_smoke", lambda *_a: {"path": "smoke", "result_sha256": "a" * 64})
    calls = 0

    def fail_candidate(_row: dict, _manifest: dict) -> dict:
        nonlocal calls
        calls += 1
        raise ValueError("candidate evidence invalid")

    monkeypatch.setattr(M.V1, "_candidate", fail_candidate)
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": entries})
    with pytest.raises(ValueError, match="candidate evidence invalid"):
        M.execute(manifest_path, approval, tmp_path / "smoke", tmp_path / "out",
                  runner=lambda *_a, **_k: pytest.fail("runner must not start"), clock=lambda: 0.0)
    assert calls == 1
    failure = M._read(tmp_path / "out/failure.json")
    assert failure["completed_receipts"] == []


def test_v1_training_manifest_keeps_compound_first_and_source_checkpoint_provenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {"correction": "restore_body_frame_reward"})
    protocol = {"source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    row = {"case_id": "case-07", "policy": "registry_next", "arm": "COMPOUND",
           "seed": 2307, "candidate": ref(candidate),
           "source_checkpoint_sha256": "1" * 64,
           "reference_checkpoint_sha256": "2" * 64}
    monkeypatch.setattr(M.V1.TRAIN, "validate_manifest", lambda value: value)
    monkeypatch.setattr(M.V1, "_relative", lambda _path: "bench/protocols/example.json")
    monkeypatch.setattr(M.V1, "_sha", lambda _path, **_kwargs: "3" * 64)
    value = M.V1._training_manifest(tmp_path / "protocol.json", protocol, "a" * 40, row,
                                      tmp_path / "train")
    assert value["arm"] == "FIRST"
    assert value["original_faults"] == ["F1", "R1"]
    assert value["applied_corrections"] == ["restore_body_frame_reward"]
    assert value["max_iterations"] == 300
    assert value["source_checkpoint_sha256"] == "1" * 64


def test_cli_exposes_all_four_stages() -> None:
    source = (ROOT / "evals/lineage_compound_first_recovery_v1.py").read_text(encoding="utf-8")
    for command in ("prepare-protocol", "prepare-inputs", "smoke", "execute"):
        assert f'add_parser("{command}")' in source
