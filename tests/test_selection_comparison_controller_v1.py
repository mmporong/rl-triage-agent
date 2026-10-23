"""선택 비교 controller의 CPU fixture 상태 전이와 변조 거부 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import selection_comparison_controller_v1 as C  # noqa: E402

VIEW_SHA = C.V.file_digest(C.V.PROTOCOL)
FACTS = {"config:action_scale": 0.25}
BINDINGS = {"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64}}
BANDS = C.read(ROOT / C.D1.BANDS)["bands"]
REPORTS = ROOT / "evals/results/residual_success_public_20261017/evaluations"


def packet(epoch=1, applied=None, prior=None):
    bindings = deepcopy(BINDINGS)
    if epoch == 2:
        assert isinstance(prior, dict)
        bindings["case"]["checkpoint_sha256"] = prior["checkpoint_sha256"]
    return C.S.build_packet(epoch=epoch, applied_corrections=list(applied or []), prior_receipt=prior,
                            facts=FACTS, input_bindings=bindings, execution_budget_s=1200)


def snapshot(root, arm, policy, *, epoch=1, applied=None, prior=None):
    ledger = root / arm / f"epoch{epoch}/probes.ledger.jsonl"
    run_id = f"controller_{arm.lower()}"
    if epoch == 1:
        parent = None
    else:
        assert isinstance(prior, dict)
        parent = {"run_id": run_id, "policy": policy, "epoch": 1, **prior}
    return C.V.build_snapshot(packet(epoch, applied, prior), run_id, policy, VIEW_SHA, ledger, parent)


def response(view, *, invalid=False, hold=False, infra=False):
    if infra:
        return {"infra_error": "offline_fixture", "content": None, "wall_s": 0.01}
    body = ({"wrong": True} if invalid else
            {"action": "hold", "probe": None, "fact_ids": [next(iter(view["evidence_facts"]))]} if hold else
            {"action": "continue", "probe": view["eligible_probes"][0],
             "fact_ids": [next(iter(view["evidence_facts"]))]})
    return {"provider_model": C.CALL.MODEL, "finish_reason": "stop", "content": json.dumps(body),
            "input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "wall_s": 0.01}


def model_dialogue(value, *, invalid=False, hold=False, infra=False):
    view = C.dialogue_input(value)["view"]
    rows = [response(view, invalid=invalid, hold=hold, infra=infra)]
    if invalid:
        rows.append(response(view, invalid=True))
    if infra:
        rows.append(response(view, infra=True))
    return C.replay_dialogue(value, "production", rows)


def measurement(probe, outcome):
    if probe == "P_episode":
        return {"timeout_ratio": 0.1 if outcome == "abnormal" else 1.0}
    if probe == "P_noise":
        return {"noise_ratio": 4.0 if outcome == "abnormal" else 1.0}
    if probe == "P_reward":
        value = 0.1 if outcome == "abnormal" else 0.0
        return {"track_lin_vel_xy_exp_rel_error": value, "track_ang_vel_z_exp_rel_error": value}
    if probe == "P_torque":
        return {"low_speed_saturation": 0.1 if outcome == "abnormal" else 0.01}
    if probe == "P_physics":
        return {"mismatch_count": 1 if outcome == "abnormal" else 0}
    if probe == "P_value":
        return {"critic_change": 0.01 if outcome == "abnormal" else 1.0}
    if probe == C.RC.NEW_PROBE:
        return {"status": outcome}
    raise AssertionError(probe)


def progress(root, arm, policy, component, *, epoch=1, applied=None, prior=None):
    seals = []
    for index in range(8):
        current = snapshot(root, arm, policy, epoch=epoch, applied=applied, prior=prior)
        if current["view"]["status"] != "open":
            return current, seals
        dialogue = model_dialogue(current) if C.dialogue_input(current)["needs_model"] else None
        choice = C.choose(current, policy, dialogue)
        probe = choice["probe"]
        outcome = C.RC.registry()["probes"][probe]["expect"][component]
        assert outcome in {"normal", "abnormal"}
        folder = root / "raw" / arm / f"e{epoch}_{index}"
        folder.mkdir(parents=True)
        case, reference = folder / "case.json", folder / "reference.json"
        case.write_text(json.dumps(measurement(probe, outcome)), encoding="utf-8")
        reference.write_text(json.dumps(measurement(probe, "normal")), encoding="utf-8")
        seals.append(C.record_probe(root, arm, epoch, current, choice, case, reference, "cpu-operator", 1.0))
    raise AssertionError("fixture did not close")


def artifact(root, arm, policy, epoch, corrections, content=b"checkpoint", source=None):
    path = root / "checkpoints" / arm / policy / f"epoch{epoch}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    value = {"schema": "selection_controller_artifact_v1", "evidence_kind": C.FIXTURE_EVIDENCE,
             "actual_ready": False, "arm": arm, "policy": policy,
             "epoch": epoch, "corrections": list(corrections), "training_mode": "new_initialization",
             "source_checkpoint_sha256": source,
             "checkpoint": {"path": str(path.resolve()), "sha256": C.digest_file(path), "identity": ""}}
    value["checkpoint"]["identity"] = C.checkpoint_identity(value)
    return value


def report(root, source_name, target, artifact_value):
    value = C.read(REPORTS / source_name)
    for key in ("schema", "seed", "arm", "source_evaluation_report_sha256",
                "source_training_artifact_sha256", "recovery_outcome"):
        value.pop(key)
    value["name"] = target
    value.update(elapsed_s=1.0, started_at="2026-10-17T00:00:00+00:00",
                 finished_at="2026-10-17T00:00:01+00:00")
    value["checkpoint"] = {"file": Path(artifact_value["checkpoint"]["path"]).name,
                           "sha256": artifact_value["checkpoint"]["sha256"]}
    path = root / "reports" / f"{target}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def repair_epoch(root, arm, policy, epoch, applied, current, reference, reference_report,
                 component, report_name, *, parent=None, prior=None):
    final, seals = progress(root, arm, policy, component, epoch=epoch, applied=applied, prior=prior)
    auth = C.authorize_repair(root / arm / "corrections.ledger.jsonl", arm, final, applied,
                              seals, "cpu-operator", 3600)
    output = artifact(root, arm, policy, epoch, [*applied, auth["correction_id"]],
                      f"{arm}-{epoch}".encode(), current["checkpoint"]["sha256"])
    output_report = report(root, report_name, f"{arm}_{policy}_e{epoch}", output)
    seal_path = root / arm / f"epoch{epoch}/repair.json"
    C.seal_repair(seal_path, ledger_path=root / arm / "corrections.ledger.jsonl", authorization=auth,
                  arm=arm, policy=policy, epoch=epoch, applied_before=applied, input_artifact=current,
                  output_artifact=output, output_report_path=output_report, reference_artifact=reference,
                  reference_report_path=reference_report, bands=BANDS, gpu_s=30.0, parent_path=parent)
    return seal_path, output, C.verify_repair(seal_path, ledger_path=root / arm / "corrections.ledger.jsonl", bands=BANDS)


def test_four_policies_receive_the_same_public_view_and_rules_use_no_dialogue(tmp_path):
    views = []
    for policy in C.POLICIES:
        value = snapshot(tmp_path, policy, policy)
        views.append(C.dialogue_input(value)["view"])
        if policy in {"registry_next", "fixed_eligible"}:
            assert C.dialogue_input(value)["needs_model"] is False
            assert C.choose(value, policy)["selected_by"] in {"registered_rule", "fixed_order"}
    assert all({k: v for k, v in view.items() if k != "policy"} ==
               {k: v for k, v in views[0].items() if k != "policy"} for view in views)


def test_invalid_infra_and_hold_never_call_rule_fallback(tmp_path):
    value = snapshot(tmp_path, "arm", "model_adaptive")
    with patch.object(C.V, "select", side_effect=AssertionError("fallback")), \
            patch.object(C.V, "rule_probe", side_effect=AssertionError("fallback")):
        for kwargs in ({"invalid": True}, {"infra": True}, {"hold": True}):
            selected = C.choose(value, "model_adaptive", model_dialogue(value, **kwargs))
            assert selected["action"] == "hold" and selected["probe"] is None


def test_pending_unknown_closed_and_shadow_do_not_authorize(tmp_path):
    value = snapshot(tmp_path, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    with pytest.raises(ValueError, match="shadow"):
        C.request_spec(value, choice, condition="exposure_full")
    spec = C.request_spec(value, choice)
    ledger = C.S.M.RegistryLedger(tmp_path / "arm/epoch1/probes.ledger.jsonl", C.RC.registry())
    ledger.propose(spec["prereg"], spec["probe"], 1200)
    pending = snapshot(tmp_path, "arm", "registry_next")
    assert pending["view"]["status"] == "pending" and C.choose(pending, "registry_next")["action"] == "hold"

    unknown_root = tmp_path / "unknown"
    value = snapshot(unknown_root, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    spec = C.request_spec(value, choice)
    ledger = C.S.M.RegistryLedger(unknown_root / "arm/epoch1/probes.ledger.jsonl", C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "cpu")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    ledger.receipt(request["request_id"], "unknown", {}, 0.0, 0)
    unidentifiable = snapshot(unknown_root, "arm", "registry_next")
    assert unidentifiable["view"]["status"] == "unidentifiable"
    assert C.choose(unidentifiable, "registry_next")["action"] == "hold"


def test_dialogue_replay_rejects_stale_view_and_mutated_core(tmp_path):
    value = snapshot(tmp_path, "arm", "model_adaptive")
    saved = model_dialogue(value)
    changed = deepcopy(saved)
    changed["engine"]["selection"]["probe"] = "P_value"
    with pytest.raises(ValueError):
        C.verify_dialogue(value, "production", changed)
    other = snapshot(tmp_path, "other", "model_adaptive")
    with pytest.raises(ValueError, match="stale"):
        C.verify_dialogue(other, "production", saved)


def test_raw_measurement_outcome_and_budget_tampering_are_rejected(tmp_path):
    value = snapshot(tmp_path, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    folder = tmp_path / "raw"
    folder.mkdir()
    case, ref = folder / "case.json", folder / "ref.json"
    expected = C.RC.registry()["probes"][choice["probe"]]["expect"]["reward"]
    case.write_text(json.dumps(measurement(choice["probe"], expected)), encoding="utf-8")
    ref.write_text(json.dumps(measurement(choice["probe"], "normal")), encoding="utf-8")
    seal = C.record_probe(tmp_path, "arm", 1, value, choice, case, ref, "cpu", 1.0)
    bad_seal = C.read(Path(seal["path"]))
    original_auth = C.read(Path(bad_seal["authorization_path"]))
    bad_auth = deepcopy(original_auth)
    bad_auth["choice"]["approval_granted"] = True
    bad_auth_path = Path(bad_seal["authorization_path"]).with_name("authorization_99.json")
    bad_auth_path.write_text(json.dumps(bad_auth), encoding="utf-8")
    bad_seal["authorization_path"] = str(bad_auth_path.resolve())
    bad_seal["authorization_sha256"] = C.digest_file(bad_auth_path)
    bad_path = Path(seal["path"]).with_name("bad_choice.json")
    bad_path.write_text(json.dumps(bad_seal), encoding="utf-8")
    with pytest.raises(ValueError, match="미승인"):
        C.verify_probe(bad_path, value)
    case.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        C.verify_probe(Path(seal["path"]), value)

    root = tmp_path / "over"
    value = snapshot(root, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    (root / "raw").mkdir(parents=True)
    case, ref = root / "raw/case.json", root / "raw/ref.json"
    case.write_text(json.dumps(measurement(choice["probe"], "normal")), encoding="utf-8")
    ref.write_text(json.dumps(measurement(choice["probe"], "normal")), encoding="utf-8")
    seal = C.record_probe(root, "arm", 1, value, choice, case, ref, "cpu", 1201.0)
    with pytest.raises(ValueError, match="receipt"):
        C.verify_probe(Path(seal["path"]), value)


def _probe_plan(value, choice, stage, run_name="probe_case", reference_path=None):
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_path or stage / "planned_reference.json").resolve()
    if not reference_path.exists():
        reference_path.write_text("{}", encoding="utf-8")
    return {"stage": str(Path(stage).resolve()), "run_name": run_name,
            "checkpoint_sha256": value["inputs"]["packet"]["input_bindings"]["case"]
            ["checkpoint_sha256"], "seed": 2122,
            "args": C.request_spec(value, choice)["args"], "gpu_contract": "residual_v1",
            "reference": {
                "evidence_path": str(reference_path),
                "evidence_sha256": C.digest_file(reference_path),
                "checkpoint_sha256": value["inputs"]["packet"]["input_bindings"]["reference"]
                ["checkpoint_sha256"], "stage": str(reference_path.parent),
                "run_name": "probe_reference", "seed": 2122, "gpu_contract": "residual_v1",
            }}


def test_probe_authorization_precedes_execution_and_is_single_use(tmp_path):
    value = snapshot(tmp_path, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    folder = tmp_path / "raw"
    folder.mkdir()
    case, ref = folder / "case.json", folder / "ref.json"
    case.write_text(json.dumps(measurement(choice["probe"], "normal")), encoding="utf-8")
    ref.write_text(json.dumps(measurement(choice["probe"], "normal")), encoding="utf-8")
    plan = _probe_plan(value, choice, folder, reference_path=ref)
    with pytest.raises(ValueError, match="execution plan"):
        C.authorize_probe(tmp_path / "wrong", "arm", 1, value, choice, "cpu",
                          evidence_kind=C.FIXTURE_EVIDENCE,
                          execution_plan={**plan, "checkpoint_sha256": "f" * 64})
    wrong_reference = deepcopy(plan)
    wrong_reference["reference"]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="reference execution plan"):
        C.authorize_probe(tmp_path / "wrong_reference", "arm", 1, value, choice, "cpu",
                          evidence_kind=C.FIXTURE_EVIDENCE, execution_plan=wrong_reference)
    auth = C.authorize_probe(tmp_path, "arm", 1, value, choice, "cpu",
                             evidence_kind=C.FIXTURE_EVIDENCE, execution_plan=plan)
    assert C.validate_probe_authorization(auth, value)["execution_plan"] == plan

    auth_path = Path(auth["path"])
    original_bytes = auth_path.read_bytes()
    changed = C.read(auth_path)
    changed["execution_plan"]["run_name"] = "changed"
    auth_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable SHA"):
        C.validate_probe_authorization(auth, value)
    auth_path.write_bytes(original_bytes)

    other_path = tmp_path / "other/epoch1/authorization_01.json"
    other_path.parent.mkdir(parents=True)
    other_path.write_bytes(original_bytes)
    cross = {**auth, "path": str(other_path), "sha256": C.digest_file(other_path)}
    with pytest.raises(ValueError, match="arm/epoch"):
        C.validate_probe_authorization(cross, value)

    ledger_path = tmp_path / "arm/epoch1/probes.ledger.jsonl"
    ledger_rows = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("\n".join(ledger_rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="승인·소비"):
        C.seal_probe(auth, value, case, ref, 1.0)
    ledger_path.write_text("\n".join(ledger_rows) + "\n", encoding="utf-8")

    alternate = folder / "alternate_reference.json"
    alternate.write_bytes(ref.read_bytes())
    with pytest.raises(ValueError, match="reference evidence file"):
        C.seal_probe(auth, value, case, alternate, 1.0)
    seal = C.seal_probe(auth, value, case, ref, 1.0)
    assert C.verify_probe(Path(seal["path"]), value)["outcome"] == "normal"
    with pytest.raises((ValueError, C.L.LedgerError)):
        C.seal_probe(auth, value, case, ref, 1.0)
    with pytest.raises(ValueError, match="authorization"):
        C.record_probe(tmp_path / "actual", "arm", 1, value, choice, case, ref, "cpu", 1.0,
                       evidence_kind=C.ACTUAL_EVIDENCE)


def test_failed_probe_closes_unknown_with_positive_or_null_cost(tmp_path):
    for index, gpu_s in enumerate((None, 0.25)):
        root = tmp_path / str(index)
        value = snapshot(root, "arm", "registry_next")
        choice = C.choose(value, "registry_next")
        auth = C.authorize_probe(root, "arm", 1, value, choice, "cpu",
                                 evidence_kind=C.ACTUAL_EVIDENCE,
                                 execution_plan=_probe_plan(value, choice, root))
        seal = C.fail_probe(auth, value, gpu_s=gpu_s, exit_code=None, reason="backend_failed")
        verified = C.verify_probe(Path(seal["path"]), value)
        assert verified["outcome"] == "unknown"

    root = tmp_path / "zero"
    value = snapshot(root, "arm", "registry_next")
    choice = C.choose(value, "registry_next")
    auth = C.authorize_probe(root, "arm", 1, value, choice, "cpu",
                             evidence_kind=C.ACTUAL_EVIDENCE,
                             execution_plan=_probe_plan(value, choice, root))
    with pytest.raises(ValueError, match="실측 비용"):
        C.fail_probe(auth, value, gpu_s=0.0, exit_code=1, reason="backend_failed")


def test_missing_approval_reconsume_and_duplicate_correction_are_rejected(tmp_path):
    value = snapshot(tmp_path, "arm", "registry_next")
    choice, spec = C.choose(value, "registry_next"), None
    spec = C.request_spec(value, choice)
    prereg = spec["prereg"]
    ledger = C.S.M.RegistryLedger(tmp_path / "ledger.jsonl", C.RC.registry())
    request = ledger.propose(prereg, choice["probe"], 1200)
    with pytest.raises(C.L.LedgerError):
        ledger.consume(request["request_id"], prereg, choice["probe"])
    ledger.approve(request["request_id"], "cpu")
    ledger.consume(request["request_id"], prereg, choice["probe"])
    with pytest.raises(C.L.LedgerError):
        ledger.consume(request["request_id"], prereg, choice["probe"])


def test_compound_fixture_requires_two_distinct_repairs_and_recovers(tmp_path):
    arm, policy = "compound", "registry_next"
    reference = artifact(tmp_path, "NONE", "shared_reference", 0, [], b"none")
    reference_report = report(tmp_path, "s2120_none.json", "reference", reference)
    initial = artifact(tmp_path, arm, policy, 0, [], b"compound")
    first_path, first_artifact, first = repair_epoch(
        tmp_path, arm, policy, 1, [], initial, reference, reference_report,
        "reward", "s2120_first.json")
    assert first_artifact["source_checkpoint_sha256"] == initial["checkpoint"]["sha256"]
    assert first["outcome"] == "not_recovered" and C.next_epoch(
        first_path, ledger_path=tmp_path / arm / "corrections.ledger.jsonl", bands=BANDS)
    prior = {"verdict": first["outcome"], "correction_id": first["applied_after"][0],
             "result_sha256": first["result_sha256"],
             "checkpoint_sha256": first_artifact["checkpoint"]["sha256"]}
    second_path, second_artifact, second = repair_epoch(
        tmp_path, arm, policy, 2, first["applied_after"], first_artifact, reference, reference_report,
        C.RC.COUNT, "s2120_both.json", parent=first_path, prior=prior)
    assert second_artifact["source_checkpoint_sha256"] == first_artifact["checkpoint"]["sha256"]
    assert second["outcome"] == "recovered" and len(second["applied_after"]) == 2
    assert C.next_epoch(None, ledger_path=tmp_path / arm / "corrections.ledger.jsonl",
                        bands=BANDS, source_outcome="source_failure_absent") is False
    assert C.next_epoch(None, ledger_path=tmp_path / arm / "corrections.ledger.jsonl",
                        bands=BANDS, source_outcome="recovered_after_one") is False
    assert Path(second_path).is_file()


@pytest.mark.parametrize("target", ["parent_sha", "cross_arm", "checkpoint", "report", "outcome"])
def test_repair_tampering_and_cross_arm_parent_are_rejected(tmp_path, target):
    arm, policy = "compound", "registry_next"
    reference = artifact(tmp_path, "NONE", "shared_reference", 0, [], b"none")
    reference_report = report(tmp_path, "s2120_none.json", "reference", reference)
    initial = artifact(tmp_path, arm, policy, 0, [], b"compound")
    first_path, first_artifact, first = repair_epoch(
        tmp_path, arm, policy, 1, [], initial, reference, reference_report,
        "reward", "s2120_first.json")
    if target == "checkpoint":
        Path(first_artifact["checkpoint"]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError):
            C.verify_repair(first_path, ledger_path=tmp_path / arm / "corrections.ledger.jsonl", bands=BANDS)
        return
    if target == "report":
        path = Path(C.read(first_path)["output_report_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            C.verify_repair(first_path, ledger_path=tmp_path / arm / "corrections.ledger.jsonl", bands=BANDS)
        return
    if target == "outcome":
        value = C.read(first_path)
        value["recovery"]["outcome"] = "recovered"
        Path(first_path).write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError):
            C.verify_repair(first_path, ledger_path=tmp_path / arm / "corrections.ledger.jsonl", bands=BANDS)
        return
    prior = {"verdict": first["outcome"], "correction_id": first["applied_after"][0],
             "result_sha256": first["result_sha256"],
             "checkpoint_sha256": first_artifact["checkpoint"]["sha256"]}
    final, seals = progress(tmp_path, arm, policy, C.RC.COUNT, epoch=2,
                            applied=first["applied_after"], prior=prior)
    auth = C.authorize_repair(tmp_path / arm / "corrections.ledger.jsonl", arm, final,
                              first["applied_after"], seals, "cpu", 3600)
    output = artifact(tmp_path, arm, policy, 2, [*first["applied_after"], auth["correction_id"]],
                      b"both", first_artifact["checkpoint"]["sha256"])
    output_report = report(tmp_path, "s2120_both.json", "both", output)
    if target == "parent_sha":
        fake_parent = tmp_path / "fake_parent.json"
        fake_parent.write_text(first_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        parent = fake_parent
    else:
        other = artifact(tmp_path, "other", policy, 1, first["applied_after"], b"compound-1",
                         initial["checkpoint"]["sha256"])
        first_artifact = other
        parent = first_path
    with pytest.raises(ValueError):
        C.seal_repair(tmp_path / arm / "epoch2/bad.json", ledger_path=tmp_path / arm / "corrections.ledger.jsonl",
                      authorization=auth, arm=arm, policy=policy, epoch=2, applied_before=first["applied_after"],
                      input_artifact=first_artifact, output_artifact=output, output_report_path=output_report,
                      reference_artifact=reference, reference_report_path=reference_report, bands=BANDS,
                      gpu_s=30.0, parent_path=parent)


def test_more_than_two_or_duplicate_corrections_are_rejected():
    with pytest.raises(ValueError):
        C.RC.validate_applied([C.RC.REWARD, C.RC.REWARD])
    with pytest.raises(ValueError):
        C.RC.validate_applied([C.RC.REWARD, C.RC.EPOCHS, C.RC.REWARD])


def test_fixed_report_uses_external_protocol_hash_and_fixture_cannot_be_actual(tmp_path):
    value = artifact(tmp_path, "arm", "registry_next", 0, [], b"source")
    path = report(tmp_path, "s2120_none.json", "fixed", value)
    raw = C.read(path)
    raw["protocol_sha256_lf"] = "f" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        C.fixed_report(path, value)

    forged = deepcopy(value)
    forged["evidence_kind"] = C.ACTUAL_EVIDENCE
    forged["actual_ready"] = True
    forged["backend"] = {}
    forged["checkpoint"]["identity"] = C.checkpoint_identity(forged)
    with pytest.raises((TypeError, ValueError)):
        C.validate_artifact(forged, arm="arm", policy="registry_next", epoch=0,
                            corrections=[], source_checkpoint_sha256=None,
                            expected_kind=C.ACTUAL_EVIDENCE)


def test_checkpoint_uses_raw_bytes_while_source_hash_uses_lf_normalization(tmp_path):
    checkpoint = tmp_path / "binary.pt"
    checkpoint.write_bytes(b"tensor-prefix\r\ntensor-suffix")
    assert C.digest_file(checkpoint) != C.digest_lf_file(checkpoint)
    value = artifact(tmp_path, "arm", "registry_next", 0, [], checkpoint.read_bytes())
    assert C.validate_artifact(
        value, arm="arm", policy="registry_next", epoch=0, corrections=[],
        source_checkpoint_sha256=None, expected_kind=C.FIXTURE_EVIDENCE)["checkpoint"]["sha256"] \
        == C.digest_file(Path(value["checkpoint"]["path"]))
    forged = deepcopy(value)
    forged["checkpoint"]["sha256"] = C.digest_lf_file(Path(value["checkpoint"]["path"]))
    forged["checkpoint"]["identity"] = C.checkpoint_identity(forged)
    with pytest.raises(ValueError, match="checkpoint"):
        C.validate_artifact(forged, arm="arm", policy="registry_next", epoch=0, corrections=[],
                            source_checkpoint_sha256=None, expected_kind=C.FIXTURE_EVIDENCE)


def _execution():
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True}


def _gpu(utilization=0):
    return {"busy": False, "busy_reasons": [], "admission_policy": C.BACKEND.V1.GPU_PROFILE,
            "gpu_exclusive": False, "performance_measurement": False,
            "utilization_percent": utilization, "compute_process_count": 0, "memory_used_mib": 100,
            "processes": [], "active_engines": []}


def _apps_receipt(kind, *, utilization=0, accepted=True):
    profile = C.GPU_CHECK.EXECUTION_V3.GPU_PROFILE
    active = utilization > 0
    clear = {"busy": False, "busy_reasons": [], "admission_policy": profile,
             "gpu_exclusive": False, "performance_measurement": False,
             "utilization_percent": utilization, "memory_used_mib": 100,
             "compute_process_count": 0,
             "processes": ([{"pid": 5, "name": "chrome.exe", "type": "C+G",
                             "known_workload": False}] if active else []),
             "active_engines": ([{
                 "name": "pid_5_luid_0x0_phys_0_eng_0_engtype_Copy",
                 "utilization_percent": utilization, "process_name": "chrome.exe"}] if active else [])}
    if kind == "initial":
        samples, identities, reason = [clear], [], "initial_clear"
    elif kind == "resampled":
        pid = 1320
        transient = {**clear, "busy": True,
            "busy_reasons": [f"unapproved_active_engine:{pid}:3D",
                             f"unverified_active_engine_process:{pid}"],
            "utilization_percent": 0, "active_engines": [{
                "name": f"pid_{pid}_luid_0x0_phys_0_eng_0_engtype_3D",
                "utilization_percent": 1, "process_name": "csrss.exe"}]}
        identity = {"ProcessId": pid, "CreationDate": "2026-10-23T00:00:00.0000000Z", "SessionId": 1}
        samples, identities, reason = [transient, {**clear, "utilization_percent": 0},
                                       {**clear, "utilization_percent": 0}], [identity] * 3, \
                                      "two_consecutive_strict_clear"
    else:
        pid = 77556
        rejected = {**clear, "busy": True,
            "busy_reasons": [f"unapproved_gpu_process:{pid}", f"unapproved_active_engine:{pid}:3D",
                             f"unverified_active_engine_process:{pid}"],
            "compute_process_count": 1,
            "processes": [{"pid": pid, "name": "kit.exe", "type": "C", "known_workload": True}],
            "active_engines": [{"name": f"pid_{pid}_x_phys_0_eng_0_engtype_3D",
                                "utilization_percent": 12, "process_name": "kit.exe"}]}
        samples, identities, reason = [rejected], [], "strict_busy_not_registered_transient"
    bounded = {"attempted": len(samples) > 1, "accepted": accepted, "reason": reason,
               "interval_s": 2, "max_samples": 3, "sample_count": len(samples),
               "required_consecutive_clear": 2, "identities": identities}
    return {**samples[-1], "bounded_resample": bounded, "samples": samples}


def test_gpu_contract_dispatch_accepts_registered_raw_shapes_and_rejects_kit(tmp_path):
    stage = tmp_path / "gpu"
    stage.mkdir()
    for name in ("gpu_before.json", "gpu_after.json"):
        (stage / name).write_text(json.dumps(_apps_receipt("initial", utilization=19)), encoding="utf-8")
    result = C._gpu_evidence(stage, "apps_bounded_v1")
    assert result["gpu_before.json.status"] == result["gpu_after.json.status"] == "initial_clear"

    (stage / "gpu_after.json").write_text(json.dumps(_apps_receipt("resampled")), encoding="utf-8")
    assert C._gpu_evidence(stage, "apps_bounded_v1")["gpu_after.json.status"] == "resampled_clear"
    with pytest.raises(ValueError, match="등록되지 않은"):
        C._gpu_evidence(stage, "unknown")
    (stage / "gpu_after.json").write_text(json.dumps(_apps_receipt("kit", accepted=False)), encoding="utf-8")
    with pytest.raises(ValueError):
        C._gpu_evidence(stage, "apps_bounded_v1")
    (stage / "gpu_after.json").write_text(json.dumps(_apps_receipt("kit", accepted=True)), encoding="utf-8")
    with pytest.raises(ValueError):
        C._gpu_evidence(stage, "apps_bounded_v1")


def test_actual_legacy_probe_revalidates_jobs_report_execution_and_gpu(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"policy")
    raw_artifact = {"arm": "FIRST", "seed": 2122, "run_name": "training_compound",
                    "source_checkpoint_sha256": None, "remaining_faults": ["F1", "R1"],
                    "checkpoint": {"path": str(checkpoint), "sha256": C.digest_file(checkpoint)}}
    with patch.object(C.BACKEND, "validate_artifact", return_value={"valid": True}):
        normalized = C.adapt_actual_artifact(
            raw_artifact, {"bound": True}, backend_arm="FIRST", seed=2122, arm="compound",
            policy="registry_next", epoch=0, corrections=[], source_checkpoint_sha256=None)
        stage = (tmp_path / "legacy").resolve()
        stage.mkdir()
        report_path = stage / "report.json"
        jobs = {"task": C.BACKEND.TASK, "fault": "F1", "seed": 2122, "jobs": [{
            "name": "probe_compound", "checkpoint": str(checkpoint), "probes": ["P_reward"],
            "output": str(report_path)}]}
        raw_report = {"name": "probe_compound", "task": C.BACKEND.TASK, "seed": 2122,
                      "checkpoint": {"sha256": C.digest_file(checkpoint)},
                      "probes": {"P_reward": {"exit": 0, "measurement": measurement("P_reward", "abnormal")}}}
        (stage / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        report_path.write_text(json.dumps(raw_report), encoding="utf-8")
        (stage / "execution.json").write_text(json.dumps(_execution()), encoding="utf-8")
        (stage / "gpu_before.json").write_text(json.dumps(_gpu(17)), encoding="utf-8")
        (stage / "gpu_after.json").write_text(json.dumps(_gpu(23)), encoding="utf-8")
        raw = {"report": raw_report, "report_path": str(report_path), "probes": ["P_reward"],
               "execution": _execution()}
        evidence = C.actual_legacy_probe_evidence(stage, raw, normalized, "P_reward", "probe_compound")
        assert C.validate_actual_probe_evidence(evidence) == measurement("P_reward", "abnormal")
        plan = {"stage": str(stage), "run_name": "probe_compound",
                "checkpoint_sha256": normalized["checkpoint"]["sha256"], "seed": 2122,
                "args": C.RC.registry()["probes"]["P_reward"]["args"],
                "gpu_contract": "residual_v1"}
        C._validate_evidence_plan(evidence, plan)
        with pytest.raises(ValueError, match="execution plan"):
            C._validate_evidence_plan(evidence, {**plan, "run_name": "other"})
        evidence_path = tmp_path / "shared_reference_evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        reference_plan = {"reference": {
            "evidence_path": str(evidence_path.resolve()),
            "evidence_sha256": C.digest_file(evidence_path),
            "checkpoint_sha256": normalized["checkpoint"]["sha256"],
            "stage": str(stage), "run_name": "probe_compound", "seed": 2122,
            "gpu_contract": "residual_v1"}}
        C._validate_reference_binding(evidence_path, evidence, reference_plan, C.ACTUAL_EVIDENCE)
        swapped = tmp_path / "swapped_reference_evidence.json"
        swapped.write_bytes(evidence_path.read_bytes())
        with pytest.raises(ValueError, match="reference evidence file"):
            C._validate_reference_binding(swapped, evidence, reference_plan, C.ACTUAL_EVIDENCE)
        wrong_contract = deepcopy(evidence)
        wrong_contract["gpu_contract"] = "unknown"
        with pytest.raises(ValueError, match="등록되지 않은"):
            C.validate_actual_probe_evidence(wrong_contract)
        gpu_before = C.read(stage / "gpu_before.json")
        (stage / "gpu_before.json").write_text(
            json.dumps({**gpu_before, "bounded_resample": {"accepted": True}}), encoding="utf-8")
        with pytest.raises(ValueError, match="bounded"):
            C.validate_actual_probe_evidence(evidence)
        (stage / "gpu_before.json").write_text(json.dumps(gpu_before), encoding="utf-8")
        jobs["jobs"][0]["name"] = "other_probe"
        (stage / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        with pytest.raises(ValueError):
            C.validate_actual_probe_evidence(evidence)

        probes = sorted(C.L.PROBES)
        jobs["jobs"][0].update(name="probe_all", probes=probes)
        raw_report["name"] = "probe_all"
        raw_report["probes"] = {
            item: {"exit": 0, "measurement": measurement(item, "normal")} for item in probes}
        (stage / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        report_path.write_text(json.dumps(raw_report), encoding="utf-8")
        raw.update(report=raw_report, probes=probes)
        shared = [C.actual_legacy_probe_evidence(
            stage, raw, normalized, item, "probe_all", expected_probes=probes) for item in probes]
        assert len({json.dumps(item["files"], sort_keys=True) for item in shared}) == 1

        with pytest.raises(ValueError, match="선택"):
            C.actual_legacy_probe_evidence(
                stage, {**raw, "probes": probes[:-1]}, normalized, probes[0], "probe_all",
                expected_probes=probes)
        with pytest.raises(ValueError, match="선택"):
            C.actual_legacy_probe_evidence(
                stage, raw, normalized, probes[0], "probe_all", expected_probes=[*probes, "extra"])
        with pytest.raises(ValueError, match="선택"):
            C.actual_legacy_probe_evidence(
                stage, {**raw, "probes": list(reversed(probes))}, normalized, probes[0], "probe_all",
                expected_probes=list(reversed(probes)))
        missing_target = deepcopy(raw_report)
        missing_target["probes"].pop(probes[0])
        report_path.write_text(json.dumps(missing_target), encoding="utf-8")
        with pytest.raises(ValueError, match="report binding"):
            C.actual_legacy_probe_evidence(
                stage, {**raw, "report": missing_target}, normalized, probes[0], "probe_all",
                expected_probes=probes)


def test_actual_optimizer_probe_binds_manifest_records_readback_and_gpu(tmp_path):
    stage = (tmp_path / "optimizer").resolve()
    stage.mkdir()
    run_dir = (tmp_path / "run").resolve()
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "params/agent.yaml").write_text(
        "seed: 2122\nmax_iterations: 3\nalgorithm:\n  num_learning_epochs: 5\n  num_mini_batches: 4\n",
        encoding="utf-8")
    checkpoint = run_dir / "model_2.pt"
    checkpoint.write_bytes(b"inspect")
    paths = {name: stage / name for name in (
        "manifest.json", "records.jsonl", "readback.json", "inspection.json", "execution.json",
        "checkpoint.json", "checkpoint_execution.json")}
    manifest = {"source_checkpoint_sha256": "a" * 64, "original_faults": ["F1", "R1"],
                "applied_corrections": [C.RC.REWARD]}
    readback = {"verified": True}
    report = {"status": "abnormal", "rows": 3}
    paths["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["records.jsonl"].write_text("{}\n{}\n{}\n", encoding="utf-8")
    paths["readback.json"].write_text(json.dumps(readback), encoding="utf-8")
    paths["inspection.json"].write_text(json.dumps(report), encoding="utf-8")
    paths["execution.json"].write_text(json.dumps(_execution()), encoding="utf-8")
    first_checkpoint = run_dir / "model_0.pt"
    first_checkpoint.write_bytes(b"first")
    checkpoint_receipt = {"device": "cpu",
        "first": {"path": str(first_checkpoint), "sha256": C.digest_file(first_checkpoint), "iteration": 0},
        "final": {"path": str(checkpoint), "sha256": C.digest_file(checkpoint), "iteration": 2}}
    paths["checkpoint.json"].write_text(json.dumps(checkpoint_receipt), encoding="utf-8")
    paths["checkpoint_execution.json"].write_text(json.dumps(_execution()), encoding="utf-8")
    for name in ("gpu_before.json", "gpu_after.json"):
        (stage / name).write_text(json.dumps(_gpu()), encoding="utf-8")
    raw = {"run_name": "inspect_compound", "arm": "FIRST", "seed": 2122,
           "source_checkpoint_sha256": "a" * 64, "checkpoint_loaded": False,
           "original_faults": ["F1", "R1"], "applied_corrections": [C.RC.REWARD],
           "remaining_faults": ["R1"],
           "configuration": {"seed": 2122, "max_iterations": 3,
                             "num_learning_epochs": 5, "num_mini_batches": 4},
           "run_dir": str(run_dir), "manifest_path": str(paths["manifest.json"]),
           "records_path": str(paths["records.jsonl"]),
           "readback_path": str(paths["readback.json"]), "report_path": str(paths["inspection.json"]),
           "inspection": {"status": "abnormal", "optimizer_steps": [4, 4, 4],
                          "minibatches": [4, 4, 4]},
           "measurement": None, "report": report, "readback": readback, "execution": _execution(),
           "checkpoint": {"path": str(checkpoint), "sha256": C.digest_file(checkpoint)},
           "checkpoint_evidence": checkpoint_receipt, "checkpoint_execution": _execution()}
    measured = {"status": "abnormal", "optimizer_steps": [4, 4, 4],
                "minibatches": [4, 4, 4], "report": report}
    raw["measurement"] = measured
    with patch.object(C.BACKEND.WRAPPER, "validate_manifest", return_value=manifest), \
            patch.object(C.BACKEND, "_manifest", return_value=manifest), \
            patch.object(C.BACKEND, "_verify_repository_provenance"), \
            patch.object(C.BACKEND, "_inspection", return_value=measured), \
            patch.object(C.BACKEND, "_validate_readback", return_value=readback):
        evidence = C.actual_optimizer_probe_evidence(
            stage, raw, {"binding": True}, backend_arm="FIRST", seed=2122,
            source_checkpoint_sha256="a" * 64)
        assert C.validate_actual_probe_evidence(evidence) == {"status": "abnormal"}
        bad = deepcopy(evidence)
        bad["raw"]["arm"] = "NONE"
        with pytest.raises(ValueError, match="raw"):
            C.validate_actual_probe_evidence(bad)
        paths["checkpoint_execution.json"].unlink()
        with pytest.raises(ValueError, match="원본"):
            C.validate_actual_probe_evidence(evidence)
        paths["checkpoint_execution.json"].write_text(json.dumps(_execution()), encoding="utf-8")
        paths["records.jsonl"].write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="provenance"):
            C.validate_actual_probe_evidence(evidence)


def test_subprocess_smoke_uses_utf8_and_protocol_has_no_user_path(tmp_path):
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "evals/selection_comparison_controller_v1.py"), "check"],
        cwd=ROOT, env=env, check=False, capture_output=True, text=True, encoding="utf-8", errors="strict",
    )
    assert completed.returncode == 0, completed.stderr
    raw = C.PROTOCOL.read_text(encoding="utf-8")
    assert "C:\\Users" not in raw and "/home/" not in raw and "LIMMM" not in raw
