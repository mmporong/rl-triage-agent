"""개발판 provenance·CALL·native API의 CPU 회귀. Git/GPU/provider 실행 없음."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import zlib
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "tests", "src")]
import test_vision_selection_contract_v1 as F
import vision_transferred_e2e_v2 as E


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def loose(repo: Path, kind: str, value: bytes) -> str:
    raw = f"{kind} {len(value)}".encode() + b"\0" + value
    sha = hashlib.sha1(raw).hexdigest()
    path = repo / ".git/objects" / sha[:2] / sha[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(raw))
    return sha


def test_object_evidence_verifies_real_hash_and_tree_without_git(tmp_path: Path) -> None:
    blob = loose(tmp_path, "blob", b"registered data")
    tree = loose(tmp_path, "tree", b"100644 proof.json\0" + bytes.fromhex(blob))
    commit = loose(tmp_path, "commit", f"tree {tree}\nauthor Unit <unit@example.invalid> 1 +0000\ncommitter Unit <unit@example.invalid> 1 +0000\n\nfixture\n".encode())
    store = E.ObjectEvidence(tmp_path)
    assert store.file(commit, "proof.json") == (blob, b"registered data")
    assert store.commit(commit)["timestamp"] == 1
    with pytest.raises(FileNotFoundError): store.file(commit, "absent.json")
    path = tmp_path / ".git/objects" / blob[:2] / blob[2:]
    path.write_bytes(zlib.compress(b"blob 7\0changed"))
    with pytest.raises(ValueError, match="hash"):
        E.ObjectEvidence(tmp_path).object(blob)


def test_packed_delta_computes_copy_and_insert_and_rejects_bounds() -> None:
    # 6-byte base -> 7-byte result: copy 'abc', insert 'XY', copy 'ef'.
    assert E.ObjectEvidence.delta(b"abcdef", bytes([6, 7, 0x90, 3, 2]) + b"XY" + bytes([0x91, 4, 2])) == b"abcXYef"
    with pytest.raises(ValueError): E.ObjectEvidence.delta(b"a", bytes([1, 2, 0x91, 5, 2]))


def test_development_policy_is_explicit_and_keeps_original_targets() -> None:
    method = E.validate_method()
    assert method["development_scope"] == E.DEVELOPMENT
    assert method["original_before_comparison_policy"] == "attempt1_v1_preserved_not_satisfied_not_waived"
    assert method["targets"] == list(E.CONTRACT.TARGETS)
    assert method["limits"]["e2e_api_requests"] == 32
    assert "before_comparison" not in inspect.signature(E.prepare_blueprint).parameters
    assert method["model_relay_design"]["copy_credentials_to_Brev"] is False


@pytest.mark.parametrize("field", ["development_scope", "execution_registration", "targets", "request_params", "training"])
def test_method_cannot_silently_relabel_as_prospective_or_change_target(tmp_path: Path, field: str) -> None:
    value = E.ADAPTER._read(E.METHOD)
    value[field] = None
    path = tmp_path / "method.json"
    write(path, value)
    with pytest.raises(ValueError): E.validate_method(path)


def test_owned_planned_manifest_equals_existing_call_consumer(tmp_path: Path) -> None:
    contract, binding, snapshot = F.setup_case(tmp_path)
    capsule = E.CALL.freeze(snapshot, contract, binding, "text_and_video", tmp_path / "capsule")
    eid = "unit-development-v2"
    calls = [{"call_id": "e2e-source-01", "run_id": f"vision-selection-{eid}-e2e-source-01", "capsule": capsule}]
    planned = E.planned_manifest(tmp_path, eid, "a" * 40, calls)
    reference = E.CALL.initialize(tmp_path, eid, "e2e", "a" * 40, E.ADAPTER.file_sha(E.METHOD), calls)
    assert E.ADAPTER._read(Path(reference["path"])) == planned
    assert planned["api_cap"] == 32 and not (Path(reference["path"]).parent / "api").exists()
    changed = Path(capsule["path"]).parent / "messages.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): E.planned_manifest(tmp_path, eid, "a" * 40, calls)


def native_correction(tmp_path: Path, seed: int = 2220) -> tuple[dict, Path, E.CorrectionLedger]:
    # 승인 export 경계를 위한 단위 원문. GPU 학습 provenance 통과 주장에 사용하지 않는다.
    runner, _ = E.native_components()
    tmp_path.mkdir(parents=True, exist_ok=True)
    catalog = runner.CATALOG[f"corrected-s{seed}"]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(f"unit-corrected-checkpoint-{seed}".encode())
    target = {"source_job_id": f"vision-s{seed}-f1", "corrected_job_id": f"vision-s{seed}-corrected",
        "source_seed": seed, "checkpoint_sha256": catalog["source_f1_checkpoint_sha256"]}
    correction = {"schema": "vision_selection_correction_result_v1", "status": "complete", "group": "source-01" if seed == 2220 else "source-02",
        "job_id": target["corrected_job_id"], "checkpoint_sha256": E.ADAPTER.file_sha(checkpoint),
        "training": {"arm": "CORRECTED", "seed": seed, "source_checkpoint_sha256": target["checkpoint_sha256"],
                     "checkpoint": {"path": str(checkpoint), "sha256": E.ADAPTER.file_sha(checkpoint)}},
        "training_evidence": {"unit_fixture": True}, "evaluation": {"unit_fixture": True},
        "fixed_eval": {"none": "healthy", "corrected": "healthy"}, "gpu_s": 1.0, "compute_usd": 0.0, "exit_code": 0}
    result = tmp_path / "correction.json"
    write(result, correction)
    ledger = E.CorrectionLedger(tmp_path / "correction.ledger.jsonl")
    event = ledger.propose("a" * 64, "reward", "b" * 64, "restore_body_frame_reward", "restart",
        {"case": "case_01", "arm": "corrected"}, 3600, evidence={
            "source_job_id": target["source_job_id"], "source_checkpoint_sha256": target["checkpoint_sha256"]})
    ledger.approve(event["request_id"], "unit")
    ledger.consume(event["request_id"], "a" * 64)
    ledger.receipt(event["request_id"], "source-01", "recovered", E.ADAPTER.digest(correction), 1.0, 0)
    return target, result, ledger


def test_native_approval_export_is_accepted_by_actual_binding_consumer(tmp_path: Path) -> None:
    target, result, ledger = native_correction(tmp_path)
    approval = tmp_path / "approval.json"
    value = E.record_approval(ledger, target, result, approval)
    runner, _ = E.native_components()
    verified = runner._verified_binding("corrected-s2220", result, approval)
    assert verified["request_id"] == value["request_id"]
    assert value["correction_result_sha256"] == E.ADAPTER.file_sha(result)
    assert E.record_approval(ledger, target, result, approval) == value
    changed = E.ADAPTER._read(result)
    changed["checkpoint_sha256"] = "f" * 64
    write(result, changed)
    with pytest.raises(ValueError): E.record_approval(ledger, target, result, approval)


def test_native_keyword_api_binds_real_service_signature() -> None:
    _, native = E.native_components()
    payload = {"resource_proof": "fresh.json", "resource_sha256": "a" * 64,
        "isaaclab": "IsaacLab", "python": "python", "stage": "new-stage", "receipt_output": "new-receipt"}
    kwargs = E.record_kwargs(payload, "corrected-s2220", Path("correction.json"), Path("approval.json"))
    bound = inspect.signature(native.NativeCorrectedRecordService.record).bind(object(), **kwargs)
    assert bound.arguments["logical_id"] == "corrected-s2220"
    assert bound.arguments["normal_reference"] is None
    assert "from_runtime" not in E.validate_method()["native_record_interface"]


def test_real_native_v01_compute_and_observer_input_fixture(tmp_path: Path) -> None:
    contract, _binding = F.prepared_case(tmp_path)
    spec = tmp_path / "source/prepare.json"
    original = E.ADAPTER._read(spec)
    checkpoint = E.INPUTS.V01.verify(contract)["provenance"]["checkpoint"]["sha256"]
    output = tmp_path / "native-receipt"
    output.mkdir()
    rows = E.compute_native_contracts({"spec_1": spec, "spec_2": spec,
        "video": Path(original["video"]["path"])}, output, checkpoint)
    receipt = {"schema": "vision_corrected_native_record_receipt_v1", "checkpoint_sha256": checkpoint,
        "v01_contracts": rows, "capture_approved": False, "model_ready": False, "used_for_decision": False}
    selected = E.native_contract(receipt, checkpoint)
    binding = E.OBSERVER.binding_template(selected, run_id="unit-native-post", source_anchor="a" * 40, protocol_sha256="b" * 64)
    assert E.OBSERVER.checked_binding(selected, binding)
    assert [r["window"] for r in rows] == [1, 2]
    receipt["model_ready"] = True
    with pytest.raises(ValueError): E.native_contract(receipt, checkpoint)


def test_fresh_E2E_probe_namespace_does_not_accept_comparison_root(tmp_path: Path) -> None:
    protocol = E.development_probe(ROOT, "unit-dev-e2e")
    assert protocol["purpose"].startswith("fresh_E2E_confirmation")
    assert E.probe_stage(tmp_path, protocol, "source-01") != tmp_path / ".triage-runtime/vision-transferred-probe-v1/source-01"
    with pytest.raises(ValueError):
        E.probe_stage(tmp_path, {"execution_root": ".triage-runtime/vision-transferred-probe-v1"}, "source-01")


def test_failed_consumption_stays_unknown_without_retry(tmp_path: Path) -> None:
    _target, _result, ledger = native_correction(tmp_path)
    rid = next(iter(ledger.requests()))
    with pytest.raises(ValueError, match="새 승인"):
        E.consumed_action(ledger, rid, "source-01", lambda: {"status": "complete"}, tmp_path / "fail.json")


def relay_case(tmp_path: Path) -> tuple[E.RelayComplete, Path, Path]:
    spool, ledger = tmp_path / "spool", tmp_path / "local-ledger"
    spool.mkdir(); ledger.mkdir()
    config = {"schema": "vision_transferred_e2e_relay_config_v2", "spool": str(spool), "job_id": "unit-e2e-model",
        "source": {"path": "evals/vision_model_spool_v1.py", "sha256_lf": E.ADAPTER.file_sha(ROOT / "evals/vision_model_spool_v1.py", lf=True)},
        "timeout_s": 0, "poll_interval_s": 0.25}
    return E.RelayComplete(config, tmp_path / "sequence"), spool, ledger


def test_actual_spool_callback_increments_sequence_and_preserves_provider_dict(tmp_path: Path) -> None:
    relay, spool, ledger = relay_case(tmp_path)
    S = relay.spool
    for sequence in (1, 2):
        messages = [{"role": "user", "content": f"unit-{sequence}"}]
        with pytest.raises(S.SpoolPendingError):
            S.remote_complete(messages, E.OBSERVER.REQUEST_PARAMS, spool=spool, job_id="unit-e2e-model", sequence=sequence, timeout_s=0)
        request = spool / f"requests/unit-e2e-model/{sequence:04d}.json"
        response = spool / f"responses/unit-e2e-model/{sequence:04d}.json"
        provider = {"unit_fixture": True, "all_fields_preserved": [sequence, None]}
        S.process_one(request, response, ledger, lambda *_, bound=provider: bound)
        assert relay(messages, deepcopy(E.OBSERVER.REQUEST_PARAMS)) == provider
    assert [p.name for p in (tmp_path / "sequence/allocations").glob("*.json")] == ["0001.json", "0002.json"]


def test_relay_does_not_coerce_different_proposal_params(tmp_path: Path) -> None:
    relay, _spool, _ledger = relay_case(tmp_path)
    wrong = {**E.OBSERVER.REQUEST_PARAMS, "temperature": 0.5}
    with pytest.raises(ValueError, match="proposal params"):
        relay([], wrong)
    assert not (tmp_path / "sequence/allocations").exists()


def test_spool_pending_keeps_journal_reservation_without_resend(tmp_path: Path) -> None:
    contract, binding, snapshot = F.setup_case(tmp_path)
    capsule = E.CALL.freeze(snapshot, contract, binding, "text_and_video", tmp_path / "capsule")
    eid = "unit-relay-pending-v2"
    reference = E.CALL.initialize(tmp_path, eid, "e2e", "a" * 40, "b" * 64,
        [{"call_id": "initial", "run_id": f"vision-selection-{eid}-initial", "capsule": capsule}])
    relay, spool, _ledger = relay_case(tmp_path)
    with pytest.raises(E.RelayPending): E.CALL.run(reference, "initial", relay)
    resumed = E.CALL.run(reference, "initial", relay)["journal"]
    assert resumed["status"] == "blocked_pending" and resumed["api"]["pending"] == 1
    assert len(list((spool / "requests/unit-e2e-model").glob("*.json"))) == 1
    assert len(list((tmp_path / "sequence/allocations").glob("*.json"))) == 1


def private_native_bound(tmp_path: Path) -> tuple[Path, Path]:
    runner, _ = E.native_components()
    template = ROOT / "bench/protocols/vision_corrected_pull_record_template_v1.json"
    design = runner.validate_template(template)
    rows = []
    for seed in (2220, 2221):
        directory = tmp_path / f"private-seed-{seed}"
        target, result, ledger = native_correction(directory, seed)
        approval = directory / "approval.json"
        E.record_approval(ledger, target, result, approval)
        rows.append(runner._verified_binding(f"corrected-s{seed}", result, approval))
    bound = {"schema": runner.PROTOCOL_SCHEMA, "protocol_id": design["protocol_id"],
        "registered_at_utc": "2026-10-27T11:00:00+00:00", "source_commit": "a" * 40,
        "source_sha256_lf": {}, "source_lineage": design["source_lineage"],
        "recording_target": {"resource_id": "unit-resource", "resource_name": "unit", "instance_type": "unit"},
        "resources": {"compute_quote_usd_per_h": "2.09616", "batch_budget_usd": "1", "deadline_utc": "2026-10-28T00:00:00Z",
                      "max_owned_instances": 1, "cost_scope": "new_corrected_native_recording_only"},
        "record_profile": design["record_profile"], "limits": design["limits"], "fixed_flags": design["fixed_flags"],
        "checkpoints": rows, "post_observer": design["post_observer"], "gpu_validation_status": "pending_live_gpu_execution"}
    path = tmp_path / "bench/private/raw-bound.json"
    write(path, bound)
    return path, template


def commit_registration_fixture(repo: Path, relative: str, seal: dict) -> tuple[dict, str]:
    path = repo / relative
    write(path, seal)
    blob = loose(repo, "blob", path.read_bytes())
    leaf = loose(repo, "tree", b"100644 native.registration.json\0" + bytes.fromhex(blob))
    protocols = loose(repo, "tree", b"40000 protocols\0" + bytes.fromhex(leaf))
    root = loose(repo, "tree", b"40000 bench\0" + bytes.fromhex(protocols))
    commit = loose(repo, "commit", f"tree {root}\nauthor Unit <unit@example.invalid> 1 +0000\ncommitter Unit <unit@example.invalid> 1 +0000\n\nregistration fixture\n".encode())
    return {"path": relative, "sha256": E.ADAPTER.file_sha(path)}, commit


def test_public_native_seal_binds_private_raw_without_committing_private_file(tmp_path: Path) -> None:
    raw, template = private_native_bound(tmp_path)
    seal = E.native_registration(raw, template)
    assert seal["schema"] == "vision_corrected_record_registration_v1"
    assert seal["bound_protocol_sha256"] == E.ADAPTER.file_sha(raw)
    assert [r["logical_id"] for r in seal["checkpoints"]] == ["corrected-s2220", "corrected-s2221"]
    E.public_registration(seal)
    reference, anchor = commit_registration_fixture(tmp_path, "bench/protocols/native.registration.json", seal)
    assert E.verify_native_registration(tmp_path, reference, anchor, raw, template) == seal
    with pytest.raises(FileNotFoundError): E.ObjectEvidence(tmp_path).file(anchor, "bench/private/raw-bound.json")
    changed = E.ADAPTER._read(raw)
    changed["resources"]["batch_budget_usd"] = "2"
    write(raw, changed)
    with pytest.raises(ValueError, match="공개 seal"):
        E.verify_native_registration(tmp_path, reference, anchor, raw, template)


def test_registration_file_change_cannot_be_hidden_by_new_reference_sha(tmp_path: Path) -> None:
    raw, template = private_native_bound(tmp_path)
    seal = E.native_registration(raw, template)
    reference, anchor = commit_registration_fixture(tmp_path, "bench/protocols/native.registration.json", seal)
    seal["checkpoints"][0]["checkpoint_sha256"] = "f" * 64
    write(tmp_path / reference["path"], seal)
    reference["sha256"] = E.ADAPTER.file_sha(tmp_path / reference["path"])
    with pytest.raises(ValueError, match="commit blob"):
        E.verify_native_registration(tmp_path, reference, anchor, raw, template)


@pytest.mark.parametrize("private", ["/home/test-user/a", "/home/another-user/b", r"C:\Users\unit\a", "bench/private/run/file.json"])
def test_public_registration_rejects_private_and_user_paths(private: str) -> None:
    with pytest.raises(ValueError, match="private 경로"):
        E.public_registration({"schema": "vision_corrected_record_registration_v1", "leak": private})
