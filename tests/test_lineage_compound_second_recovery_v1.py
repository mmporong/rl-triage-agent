from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_second_recovery_v1 as M


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": M._sha(path)}


def row(status: str = "HOLD") -> dict:
    return {"case_id": "case-07", "policy": "registry_next", "arm": "COMPOUND",
            "seed": 2307, "status": status,
            "hold_reason": "diagnosis_invalid" if status == "HOLD" else None,
            "candidate": None, "applied_corrections_before": [],
            "applied_corrections_after": []}


def test_prepare_protocol_registers_reused_smoke_and_both_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recovery_path = ROOT / "bench/protocols/lineage_recovery_execution_20261031_v2.json"
    recovery = M._read(recovery_path)
    post_path = write(tmp_path / "post.json", {"schema": M.E2.PROTOCOL_SCHEMA})
    first = {"schema": M.FIRST.SCHEMA,
             "component_protocols": {
                 "recovery_v2": {"path": M._relative(recovery_path),
                                 "sha256_lf": M._sha(recovery_path, lf=True)},
                 "fixed_evaluation": recovery["component_protocols"]["fixed_evaluation"],
                 "bands": recovery["component_protocols"]["bands"]},
             "runtime_source_sha256_lf": recovery["runtime_source_sha256_lf"]}
    first_path = write(tmp_path / "first.json", first)
    original = M._relative

    def relative(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved == post_path.resolve():
            return "bench/protocols/post-fixture.json"
        if resolved == first_path.resolve():
            return "bench/protocols/first-fixture.json"
        return original(path)

    monkeypatch.setattr(M, "_relative", relative)
    protocol = M.prepare_protocol(post_path, recovery_path, first_path)
    assert protocol["training"] == {
        "mode": "fresh_restart", "arm": "BOTH", "iterations": 300, "seed": 2307,
        "source_checkpoint_role": "provenance_only_never_loaded",
        "fault_strength_change": False, "model_calls": 0}
    assert protocol["smoke"] == {
        "strategy": "reuse_and_reverify_first_controller_smoke",
        "same_runtime_and_training_path_required": True, "new_smoke_updates": 0}
    assert protocol["scope"]["max_distinct_corrections"] == 2
    assert len(protocol["runtime_source_sha256_lf"]) == 9


@pytest.mark.parametrize("state,expected", [("confirmed", True), ("HOLD", False)])
def test_epoch2_eligibility_requires_actual_supported_confirmation(state: str, expected: bool) -> None:
    value = {"schema": M.E2.SCHEMA, "status": "complete", "evidence_scope": "actual_backend",
             "epoch": 2, "epoch2_registered": True, "end_state": state,
             "hold_reason": None if state == "confirmed" else "probe_outcome_unknown",
             "repair_supported": state == "confirmed", "correction_executed": False,
             "training_executed": False}
    assert M._eligible_epoch2(value) is expected


def test_prior_recovered_receipt_is_rejected_before_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = write(tmp_path / "receipt.json", {"status": "recovered"})
    ledger = write(tmp_path / "ledger.jsonl", {})
    post = {"private_metadata": {"sources": {
        "repair_receipt": str(receipt), "correction_ledger": str(ledger)}}}
    monkeypatch.setattr(M.POST, "checked_observation", lambda _path: (post, {}))
    with pytest.raises(ValueError, match="not_recovered"):
        M._checked_first_inputs(
            tmp_path / "post.json", tmp_path / "first-manifest.json", tmp_path / "smoke")


def test_training_manifest_is_both_fresh300_and_clears_residual_faults(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {
        "applied_corrections_before": ["restore_body_frame_reward"],
        "applied_corrections_after": [
            "restore_body_frame_reward", "restore_configured_ppo_epochs"],
    })
    protocol = {"source_sha256_lf": {"x": "a" * 64},
                "runtime_source_sha256_lf": {"y": "b" * 64}}
    value = {**row("candidate"), "candidate": ref(candidate),
             "source_checkpoint_sha256": "c" * 64,
             "applied_corrections_before": ["restore_body_frame_reward"],
             "applied_corrections_after": [
                 "restore_body_frame_reward", "restore_configured_ppo_epochs"]}
    monkeypatch.setattr(M.V1.TRAIN, "validate_manifest", lambda item: item)
    monkeypatch.setattr(M, "_relative", lambda _path: "bench/protocols/second.json")
    monkeypatch.setattr(M, "_sha", lambda _path, **_kwargs: "d" * 64)
    fixed = M._training_manifest(
        tmp_path / "protocol.json", protocol, "e" * 40, value, tmp_path / "train")
    assert fixed["arm"] == "BOTH"
    assert fixed["max_iterations"] == 300
    assert fixed["original_faults"] == ["F1", "R1"]
    assert fixed["applied_corrections"] == [
        "restore_body_frame_reward", "restore_configured_ppo_epochs"]
    assert fixed["source_checkpoint_sha256"] == "c" * 64


def test_training_manifest_rejects_duplicate_second_correction(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {
        "applied_corrections_before": ["restore_body_frame_reward"],
        "applied_corrections_after": ["restore_body_frame_reward", "restore_body_frame_reward"]})
    value = {**row("candidate"), "candidate": ref(candidate),
             "source_checkpoint_sha256": "c" * 64,
             "applied_corrections_before": ["restore_body_frame_reward"],
             "applied_corrections_after": ["restore_body_frame_reward", "restore_body_frame_reward"]}
    monkeypatch.setattr(M, "_relative", lambda _path: "bench/protocols/second.json")
    with pytest.raises(ValueError, match="두 교정 계보"):
        M._training_manifest(tmp_path / "protocol.json", {
            "source_sha256_lf": {}, "runtime_source_sha256_lf": {}},
            "e" * 40, value, tmp_path / "train")


def test_lease_wait_writes_each_read_to_fresh_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def lease(path: Path) -> dict:
        calls.append(Path(path))
        return {"leaseId": "0" * 36, "owner": "owner", "state": "running"}

    def await_clear(_stage: Path, **kwargs) -> dict:
        kwargs["lease_check"]()
        kwargs["lease_check"]()
        return {"status": "clear"}

    monkeypatch.setattr(M.LEASE, "lease_context", lease)
    monkeypatch.setattr(M.V1.GPU_WAIT, "await_clear", await_clear)
    assert M._lease_wait(tmp_path) == {"status": "clear"}
    assert calls == [tmp_path / "lease-reads/0001", tmp_path / "lease-reads/0002"]


def test_execute_requires_separate_operator_approval_before_any_runner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {"status": "validated_candidate"})
    item = {**row("candidate"), "candidate": ref(candidate)}
    manifest = {"row": item}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": []})
    with pytest.raises(ValueError, match="approval entries"):
        M.execute(manifest_path, approval, tmp_path / "out",
                  runner=lambda *_a, **_k: pytest.fail("runner must not start"))
    assert not (tmp_path / "out").exists()


def test_execute_holds_unconfirmed_epoch2_without_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"row": row(), "first_smoke": {
        "path": str(tmp_path / "smoke"), "result_sha256": "a" * 64}}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": []})
    clock = iter((0.0, 0.1))
    result = M.execute(
        manifest_path, approval, tmp_path / "out",
        runner=lambda *_a, **_k: pytest.fail("runner must not start"), clock=lambda: next(clock))
    assert result["status"] == "complete"
    assert result["receipts"][0]["status"] == "HOLD"
    assert result["receipts"][0]["sdk_jobs"] == 0


def test_candidate_validation_failure_starts_no_owned_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = write(tmp_path / "candidate.json", {"status": "validated_candidate"})
    item = {**row("candidate"), "candidate": ref(candidate)}
    manifest = {"row": item, "protocol": {"sha256": "a" * 64}, "first_smoke": {
        "path": str(tmp_path / "smoke"), "result_sha256": "a" * 64}}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M, "_checked_candidate",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("candidate invalid")))
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": [{
            "case_id": "case-07", "policy": "registry_next",
            "candidate_sha256": item["candidate"]["sha256"], "decision": "approve"}]})
    with pytest.raises(ValueError, match="candidate invalid"):
        M.execute(manifest_path, approval, tmp_path / "out",
                  runner=lambda *_a, **_k: pytest.fail("runner must not start"), clock=lambda: 0.0)
    assert M._read(tmp_path / "out/failure.json")["completed_receipts"] == []


def test_positive_path_precreates_owned_folder_and_emits_second_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_value = {
        "component": "optimizer_update_count", "correction": "restore_configured_ppo_epochs",
        "inspection_evidence": {},
        "applied_corrections_before": ["restore_body_frame_reward"],
        "applied_corrections_after": [
            "restore_body_frame_reward", "restore_configured_ppo_epochs"]}
    candidate_path = write(tmp_path / "candidate.json", candidate_value)
    prior = write(tmp_path / "prior.json", {"status": "not_recovered"})
    first_ledger = write(tmp_path / "first-ledger.jsonl", {})
    reference = write(tmp_path / "reference.json", {
        "checkpoint": {"sha256": "2" * 64}})
    protocol_path = write(tmp_path / "protocol.json", {})
    item = {**row("candidate"), "candidate": ref(candidate_path),
            "source_checkpoint_sha256": "1" * 64,
            "reference_checkpoint_sha256": "2" * 64,
            "reference_report": ref(reference), "prior_receipt": ref(prior),
            "first_ledger": ref(first_ledger),
            "applied_corrections_before": ["restore_body_frame_reward"],
            "applied_corrections_after": [
                "restore_body_frame_reward", "restore_configured_ppo_epochs"]}
    manifest = {"row": item, "protocol": ref(protocol_path),
                "first_smoke": {"path": str(tmp_path / "smoke"),
                                "result_sha256": "a" * 64},
                "execution_anchor": "e" * 40, "lab": str(tmp_path),
                "isaac_python": sys.executable}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(M, "_load_manifest", lambda _path: (manifest, {}))
    monkeypatch.setattr(M, "_checked_candidate", lambda *_a, **_k: candidate_value)
    monkeypatch.setattr(M, "_training_manifest", lambda *_a, **_k: {})
    runner_calls: list[Path] = []

    def runner(folder: Path, _args: list[str], _cwd: Path, _watchdog: float) -> dict:
        assert folder.is_dir()
        runner_calls.append(folder)
        return {"exit_code": 0}

    def run_training(stage: Path, _fixed: dict, _manifest: dict, _protocol: dict, **kwargs):
        stage.mkdir(parents=True)
        execution = kwargs["runner"](stage / "owned", [], tmp_path, 1800)
        return {**execution, "wall_s": 1.0}, {
            "checkpoint": {"path": str(tmp_path / "model.pt"), "sha256": "3" * 64},
            "remaining_faults": [], "applied_corrections": item["applied_corrections_after"]}

    corrected = {"checkpoint": {"sha256": "3" * 64}}

    def fixed_evaluate(stage: Path, *_args, **_kwargs):
        stage.mkdir(parents=True)
        write(stage / "corrected.json", corrected)
        return corrected, {"wall_s": 1.0}

    monkeypatch.setattr(M.V1, "_run_training", run_training)
    monkeypatch.setattr(M.V1, "_fixed_evaluate", fixed_evaluate)
    monkeypatch.setattr(M.V1.RECOVERY_INPUTS, "checked_recovery", lambda _path: None)
    monkeypatch.setattr(M.V1.C.D1, "recovery",
                        lambda *_a: {"outcome": "recovered", "reasons": []})
    approval = write(tmp_path / "approval.json", {
        "schema": M.V1.APPROVAL_SCHEMA, "approver": "operator", "entries": [{
            "case_id": "case-07", "policy": "registry_next",
            "candidate_sha256": item["candidate"]["sha256"], "decision": "approve"}]})

    def collect(*_args, **_kwargs):
        return {"training": {}}

    result = M.execute(
        manifest_path, approval, tmp_path / "out", runner=runner,
        collector=collect, clock=lambda: 0.0)
    assert runner_calls == [tmp_path / "out/run/train/owned"]
    receipt = M._read(tmp_path / "out/receipt.json")
    assert result["status"] == "complete" and receipt["status"] == "recovered"
    assert receipt["scope"] == "second_registered_correction_only"
    assert receipt["applied_corrections_after"] == item["applied_corrections_after"]
    assert receipt["remaining_faults"] == [] and receipt["third_epoch_registered"] is False


def test_cli_has_protocol_inputs_and_execute_commands() -> None:
    source = (ROOT / "evals/lineage_compound_second_recovery_v1.py").read_text(encoding="utf-8")
    for command in ("prepare-protocol", "prepare-inputs", "execute"):
        assert f'add_parser("{command}")' in source
    for option in ("post-diagnosis-protocol", "first-recovery-protocol",
                   "diagnosis-anchor", "first-smoke-root"):
        assert option in source
