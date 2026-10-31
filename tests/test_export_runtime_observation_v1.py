# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import export_runtime_observation_v1 as E


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def fixture(tmp_path: Path) -> dict[str, Path]:
    execution = tmp_path / "execution"
    nim = tmp_path / "nim"
    inputs = save(tmp_path / "inputs.json", {"schema": "runtime_observation_development_inputs_v1"})
    prereg = {
        "schema": E.DEV.SCHEMA,
        "purpose": "posthoc_same_input_next_probe_development_comparison",
        "conditions": list(E.CONDITIONS), "order": list(E.CONDITIONS),
        "input_sha256": E.sha(inputs),
        "scope": {"approval": False, "gpu_execution": False, "new_seed": False,
                  "old_measurement_overwrite": False, "posthoc_design": True},
        "source_sha256_lf": {"evals/source.py": "a" * 64},
    }
    prereg_path = save(tmp_path / "prereg.json", prereg)
    anchor = "5" * 40
    save(nim / "session.json", {"registration_anchor": anchor})
    for name in ("atomic-transport-v2.json", "lineage-transport-v3.json",
                 "lineage-transport-v4.json"):
        save(nim / name, {"anchor": anchor})
    summary_rows = []
    for index, condition in enumerate(E.CONDITIONS):
        expected = E.EXPECTED_CHOICES[condition]
        facts = ({"telemetry:case:Episode_Termination/time_out": 4.2,
                  "telemetry:reference:Episode_Termination/time_out": 4.3}
                 if condition == "telemetry_only" else {
                     **E.EXPECTED_RUNTIME_FACTS,
                     "telemetry:case:Episode_Termination/time_out": 4.2,
                     "telemetry:reference:Episode_Termination/time_out": 4.3})
        view = {"evidence_facts": facts}
        messages = [{"role": "system", "content": "Select one probe. Output JSON only."},
                    {"role": "user", "content": json.dumps(view, sort_keys=True)}]
        request_id = f"{condition}-1-fixture"
        request = {
            "schema": "lineage_nim_transport_request_v1", "slot_id": condition,
            "ordinal": 1, "request_id": request_id, "messages": messages,
            "context_digest": f"context-{index}", "view_digest": f"view-{index}",
        }
        request_path = save(nim / "requests" / f"{request_id}.json", request)
        dispatch = {
            "schema": "lineage_nim_transport_dispatch_v1", "request_id": request_id,
            "request_sha256": E.sha(request_path),
        }
        dispatch_path = save(nim / "dispatches" / f"{request_id}.json", dispatch)
        raw_result = {
            "content": json.dumps({"action": "continue", "probe": expected["probe"],
                                   "fact_ids": expected["fact_ids"]}),
            "finish_reason": "stop", "reasoning_content": None,
            "input_tokens": 100 + index, "output_tokens": 10 + index,
            "total_tokens": 110 + index * 2, "wall_s": 0.5 + index,
            "provider_model": "private/nvidia/model", "response_id": f"response-{index}",
        }
        response = {
            "schema": "lineage_nim_transport_response_v1", "request_id": request_id,
            "request_sha256": E.sha(request_path), "dispatch_sha256": E.sha(dispatch_path),
            "result": raw_result,
        }
        response_path = save(nim / "responses" / f"{request_id}.json", response)
        save(nim / "response-sha" / f"{request_id}.sha256", {
            "request_id": request_id, "response_sha256": E.sha(response_path)})
        snapshot = {"schema": "fixture", "condition": condition}
        snapshot_path = save(execution / condition / "snapshot.json", snapshot)
        choice = {"action": "continue", "probe": expected["probe"],
                  "fact_ids": expected["fact_ids"], "model_status": "valid",
                  "context_digest": f"context-{index}"}
        condition_result = {
            "condition": condition, "approval": False, "probe_executed": False,
            "choice": choice,
            "dialogue_input": {"messages": messages, "messages_digest": f"messages-{index}",
                               "view_digest": f"view-{index}",
                               "snapshot_digest": E.DEV.C.L.digest(snapshot)},
            "engine": {"requests": 1, "calls": [{"transport": "completed",
                                                    "messages_digest": f"messages-{index}",
                                                    "result": raw_result}]},
        }
        save(execution / condition / "result.json", condition_result)
        assert snapshot_path.is_file()
        summary_rows.append({"condition": condition, "choice": choice, "requests": 1})
    save(execution / "result.json", {
        "schema": "runtime_observation_development_result_v1", "status": "complete",
        "input_sha256": E.sha(inputs), "protocol_sha256": E.sha(prereg_path),
        "runtime_observation_used": True, "changed_next_probe": True,
        "conditions": summary_rows, "scope": prereg["scope"],
    })
    return {"execution": execution, "nim": nim, "inputs": inputs, "prereg": prereg_path}


def run_export(paths: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(E.DEV, "checked_inputs", lambda _path: {
        "schema": "runtime_observation_development_inputs_v1"})
    out, private = tmp_path / "public", tmp_path / "private"
    value = E.export_development(
        paths["execution"], paths["nim"], paths["prereg"], paths["inputs"],
        out=out, private_out=private)
    return value, out, private


def test_export_preserves_eight_safe_api_raw_files_and_scoped_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = fixture(tmp_path)
    value, out, private = run_export(paths, tmp_path, monkeypatch)
    assert value["status"] == "complete" and value["scope"]["development_pairs"] == 1
    assert value["comparison"]["runtime_choice_cited_observed_four"] is False
    assert value["conditions"][0]["input_tokens"] == 100
    assert value["conditions"][1]["output_tokens"] == 11
    assert len(list((out / "api").iterdir())) == 8
    assert len(list(out.rglob("*.*"))) == 11
    assert (out / "preregistration.json").read_bytes() == paths["prereg"].read_bytes()
    provenance = E.read(out / "provenance.json")
    assert provenance["registered_protocol"]["sha256_lf"] == E.sha_lf(paths["prereg"])
    assert (private / "metadata.json").is_file()
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            assert "bench/private" not in text and "\\users\\" not in text
    request = next((out / "api").glob("telemetry_and_runtime.request.json"))
    assert "runtime:case:optimizer_steps_min" in request.read_text(encoding="utf-8")
    assert "runtime:case:optimizer_steps_min" not in value["conditions"][1]["cited_fact_ids"]


def test_raw_response_tamper_and_missing_response_fail_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = fixture(tmp_path)
    response = next((paths["nim"] / "responses").glob("telemetry_only-*.json"))
    response.write_bytes(response.read_bytes() + b" ")
    monkeypatch.setattr(E.DEV, "checked_inputs", lambda _path: {
        "schema": "runtime_observation_development_inputs_v1"})
    with pytest.raises(ValueError, match="response SHA"):
        E.export_development(paths["execution"], paths["nim"], paths["prereg"], paths["inputs"],
                             out=tmp_path / "public", private_out=tmp_path / "private")
    assert not (tmp_path / "public").exists()

    paths = fixture(tmp_path / "missing")
    next((paths["nim"] / "responses").glob("telemetry_only-*.json")).unlink()
    with pytest.raises(ValueError, match="response 원문"):
        E.export_development(paths["execution"], paths["nim"], paths["prereg"], paths["inputs"],
                             out=tmp_path / "public2", private_out=tmp_path / "private2")
    assert not (tmp_path / "public2").exists()


def test_unknown_choice_label_and_negative_metrics_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = fixture(tmp_path)
    condition_path = paths["execution"] / "telemetry_only/result.json"
    condition = E.read(condition_path)
    condition["choice"]["probe"] = "P_unknown"
    save(condition_path, condition)
    monkeypatch.setattr(E.DEV, "checked_inputs", lambda _path: {
        "schema": "runtime_observation_development_inputs_v1"})
    with pytest.raises(ValueError):
        E.export_development(paths["execution"], paths["nim"], paths["prereg"], paths["inputs"],
                             out=tmp_path / "public", private_out=tmp_path / "private")

    paths = fixture(tmp_path / "negative")
    response_path = next((paths["nim"] / "responses").glob("telemetry_only-*.json"))
    response = E.read(response_path)
    response["result"]["wall_s"] = -0.1
    save(response_path, response)
    sha_path = next((paths["nim"] / "response-sha").glob("telemetry_only-*.sha256"))
    sidecar = E.read(sha_path)
    sidecar["response_sha256"] = E.sha(response_path)
    save(sha_path, sidecar)
    condition_path = paths["execution"] / "telemetry_only/result.json"
    condition = E.read(condition_path)
    condition["engine"]["calls"][0]["result"]["wall_s"] = -0.1
    save(condition_path, condition)
    with pytest.raises(ValueError, match="wall_s"):
        E.export_development(paths["execution"], paths["nim"], paths["prereg"], paths["inputs"],
                             out=tmp_path / "public2", private_out=tmp_path / "private2")


def test_prospective_missing_or_incomplete_never_creates_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="아직 없다"):
        E.export_prospective(tmp_path / "missing.json", out=tmp_path / "public",
                             private_out=tmp_path / "private")
    assert not (tmp_path / "public").exists() and not (tmp_path / "private").exists()
    result = save(tmp_path / "result.json", {
        "schema": "runtime_observation_recovery_result_v1", "status": "unknown"})
    with pytest.raises(ValueError, match="complete"):
        E.export_prospective(result, out=tmp_path / "public", private_out=tmp_path / "private")


def prospective_fixture(tmp_path: Path, *, diagnosis: bool = False) -> Path:
    root = tmp_path / "fresh-execution-v2"
    healthy = {"survival_ratio": True, "fall_frac": True,
               "err_xy_ratio": True, "err_yaw_ratio": True}
    unhealthy = {**healthy, "err_yaw_ratio": False}
    stages = []
    for stage, arm, seed, updates in (("smoke", "NONE", E.R.SMOKE_SEED, 3),
                                      ("none", "NONE", E.R.SEED, 300),
                                      ("compound", "COMPOUND", E.R.SEED, 300)):
        manifest = save(root / stage / "train/manifest.json", {
            "mode": "inspection" if stage == "smoke" else "train", "arm": arm,
            "seed": seed, "max_iterations": updates, "source_commit_sha": "4" * 40,
        })
        training = save(root / stage / "training-result.json", {
            "manifest": {"path": str(manifest), "sha256": E.sha(manifest)},
            "checkpoint": {"sha256": str(len(stage)) * 64},
        })
        stages.append({"stage": stage, "training": {"path": str(training),
                                                      "sha256": E.sha(training)}})
        save(root / stage / "fixed/corrected.json", {"stage": stage})
    if diagnosis:
        snapshot = save(root / "diagnosis-e1/final-snapshot.json", {"schema": "snapshot"})
        save(root / "diagnosis-e1/result.json", {
            "schema": "runtime_observation_diagnosis_v1", "status": "complete",
            "epoch": 1, "end_state": "HOLD", "hold_reason": "probe_outcome_unknown",
            "steps": [], "requests": 1,
            "runtime_fact_citations": [{"decision": 1, "fact_ids": [], "count": 0}],
            "final_snapshot": {"path": str(snapshot), "sha256": E.sha(snapshot)},
        })
    return save(root / "result.json", {
        "schema": "runtime_observation_recovery_result_v1", "status": "complete",
        "seed": E.R.SEED, "outcome": "HOLD", "reason": "diagnosis_unconfirmed",
        "wall_s": 3.5, "stages": stages, "repairs": [],
        "normal_control": {"host_gate": True, "model_called": False,
                           "probes": 0, "corrections": 0,
                           "recovery": {"outcome": "recovered", "verdict": {
                               "label": "healthy", "checks": healthy}}},
        "source_recovery": {"outcome": "not_recovered", "verdict": {
            "label": "unhealthy", "checks": unhealthy}},
        "claims": {"new_fault_family": False, "model_version_superiority": False,
                   "video_effect": False, "tokenfactory": False},
    })


def patch_prospective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "_smoke_checked", lambda training: (
        E.read(training["manifest"]["path"]), {}))
    monkeypatch.setattr(E.R, "_training_checked", lambda training: (
        {}, {"private_training": {"seed": E.R.SEED}}))
    monkeypatch.setattr(E.R, "_check_report", lambda path, _training: {
        "tag": path.parent.parent.name,
        "by_condition": [{}] * 26,
        "metrics": {"fall_rate": 0.0, "mean_survival_s": 20.0,
                    "lin_vel_rmse_mps": 0.1, "yaw_rate_rmse_radps": 0.1},
    })
    monkeypatch.setattr(E.R, "load_verified_snapshot", lambda _value: {"view": {}})
    monkeypatch.setattr(E.RV, "view_scope", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(E.R.C.D1, "recovery", lambda report, _reference, _bands: {
        "outcome": "recovered" if report["tag"] == "none" else "not_recovered",
        "verdict": {"label": "healthy" if report["tag"] == "none" else "unhealthy",
                    "checks": {"survival_ratio": True, "fall_frac": True,
                               "err_xy_ratio": True,
                               "err_yaw_ratio": report["tag"] == "none"}},
    })


def test_prospective_actual_contract_preserves_hold_unknown_and_fixed26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = prospective_fixture(tmp_path, diagnosis=True)
    patch_prospective(monkeypatch)
    public = E.export_prospective(result, out=tmp_path / "public",
                                  private_out=tmp_path / "private")
    assert public["outcome"] == "HOLD" and public["recovery_count"] == 0
    assert public["epochs"][0]["end_state"] == "HOLD"
    assert public["epochs"][0]["hold_reason"] == "probe_outcome_unknown"
    assert all(item["conditions"] == 26 for item in public["fixed26"].values())
    assert public["scope"]["execution_allowed"] is False
    metadata = E.read(tmp_path / "private/metadata.json")
    assert len(metadata["immutable_git_cache"]) == 2
    assert all(item["execution_allowed"] is False
               for item in metadata["immutable_git_cache"])
    assert len(list((tmp_path / "public/fixed").glob("*.json"))) == 3


def test_prospective_tamper_and_non_300_training_fail_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = prospective_fixture(tmp_path)
    patch_prospective(monkeypatch)
    value = E.read(result)
    value["stages"][1]["training"]["sha256"] = "0" * 64
    save(result, value)
    with pytest.raises(ValueError, match="SHA"):
        E.export_prospective(result, out=tmp_path / "public", private_out=tmp_path / "private")
    assert not (tmp_path / "public").exists()

    result = prospective_fixture(tmp_path / "negative")
    patch_prospective(monkeypatch)
    manifest_path = result.parent / "none/train/manifest.json"
    manifest = E.read(manifest_path)
    manifest["max_iterations"] = -300
    save(manifest_path, manifest)
    training_path = result.parent / "none/training-result.json"
    training = E.read(training_path)
    training["manifest"]["sha256"] = E.sha(manifest_path)
    save(training_path, training)
    summary = E.read(result)
    summary["stages"][1]["training"]["sha256"] = E.sha(training_path)
    save(result, summary)
    with pytest.raises(ValueError, match="300-update"):
        E.export_prospective(result, out=tmp_path / "public2", private_out=tmp_path / "private2")
    assert not (tmp_path / "public2").exists()


def test_path_safety_rejects_private_or_windows_paths() -> None:
    with pytest.raises(ValueError, match="private 경로"):
        E._safe_text(b'{"path":"bench/private/secret.json"}', "private.json")
    with pytest.raises(ValueError, match="Windows"):
        E._safe_text(b'{"path":"C:\\\\Users\\\\name\\\\secret.json"}', "windows.json")


def combined_fixture(tmp_path: Path) -> dict[str, object]:
    origin = tmp_path / "origin"
    continuation = tmp_path / "continuation"
    queue = tmp_path / "queue"
    origin_queue = tmp_path / "origin-queue"
    inputs_path = save(tmp_path / "inputs.json", {"schema": E.CONT.INPUT_SCHEMA})
    origin_protocol = save(tmp_path / "protocols/origin.json", {"schema": "origin"})
    continuation_protocol = save(
        tmp_path / "protocols/continuation.json", {"schema": "continuation"})
    failure = save(origin / "failure.json", {"status": "invalid", "wall_s": 10.0})
    save(origin / "freshness-before-execution.json", {
        "schema": "core_seed_freshness_v1", "status": "fresh",
        "seeds": [E.R.SEED, E.R.SMOKE_SEED], "evidence": []})
    diagnosis1 = save(origin / "diagnosis-e1/result.json", {"epoch": 1})
    trainings = {}
    for stage in ("none", "compound", "repair1"):
        runtime_refs = {}
        for role in ("case", "reference"):
            artifact = save(origin / f"{stage}/runtime-facts/{role}/artifact.json", {
                "schema": "optimizer_runtime_facts_v1",
                "model_view": {"facts": {f"runtime:{role}:available": True}}})
            runtime_refs[role] = {"path": str(artifact), "sha256": E.sha(artifact)}
        trainings[stage] = save(origin / f"{stage}/training-result.json", {
            "checkpoint": {"sha256": str(len(stage)) * 64},
            "runtime_facts": runtime_refs})
        save(origin / f"{stage}/fixed/corrected.json", {"tag": stage})
    first_receipt = save(origin / "repair1/receipt.json", {
        "schema": E.R.RECEIPT_SCHEMA, "epoch": 1, "applied_after": ["restore_reward"],
        "recovery": {"outcome": "not_recovered"},
        "diagnosis": {"path": str(diagnosis1), "sha256": E.sha(diagnosis1)},
    })
    save(origin / "corrections.ledger.jsonl", {"event": "fixture"})
    diagnosis2 = save(continuation / "diagnosis-e2/result.json", {"epoch": 2})
    result = save(continuation / "result.json", {
        "schema": E.CONT.RESULT_SCHEMA, "status": "complete", "seed": E.R.SEED,
        "origin_status": "valid_first_not_recovered",
        "origin_inputs": {"path": str(inputs_path), "sha256": E.sha(inputs_path)},
        "outcome": "HOLD", "continuation_status": "HOLD",
        "reason": "probe_outcome_unknown", "wall_s": 1.0,
        "diagnosis": {"path": str(diagnosis2), "sha256": E.sha(diagnosis2)},
        "second_repair_training": 0, "stages": [], "repairs": [], "receipt": None,
    })
    def add_api(target: Path, request_id: str, slot: str) -> None:
        request = save(target / "requests" / f"{request_id}.json", {
            "schema": "lineage_nim_transport_request_v1", "request_id": request_id,
            "slot_id": slot, "messages": [{"role": "user", "content": "safe"}],
        })
        dispatch = save(target / "dispatches" / f"{request_id}.json", {
            "schema": "lineage_nim_transport_dispatch_v1", "request_id": request_id,
            "request_sha256": E.sha(request),
        })
        response = save(target / "responses" / f"{request_id}.json", {
            "schema": "lineage_nim_transport_response_v1", "request_id": request_id,
            "request_sha256": E.sha(request), "dispatch_sha256": E.sha(dispatch),
            "result": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                       "wall_s": 0.5, "provider_model": "provider/model"},
        })
        save(target / "response-sha" / f"{request_id}.sha256", {
            "request_id": request_id, "response_sha256": E.sha(response)})

    add_api(origin_queue, "epoch1.decision1-1-fixture", "epoch1.decision1")
    add_api(queue, "epoch2.decision1-1-fixture", "epoch2.decision1")
    save(origin_queue / "session.json", {"registration_anchor": "a" * 40})
    save(queue / "session.json", {"registration_anchor": "b" * 40})
    origin_value = {
        "receipt": {"path": str(first_receipt), "sha256": E.sha(first_receipt)},
        "reference": {"path": str(trainings["none"]), "sha256": E.sha(trainings["none"])},
        "compound": {"path": str(trainings["compound"]), "sha256": E.sha(trainings["compound"])},
        "first": {"path": str(trainings["repair1"]), "sha256": E.sha(trainings["repair1"])},
        "phase": {"status": "invalid", "artifact": {"path": str(failure),
                                                         "sha256": E.sha(failure)}},
        "queue": {"root": str(origin_queue)}, "origin_requests": 1,
    }
    inputs = {"origin": origin_value, "origin_root": str(origin),
              "origin_anchor": "a" * 40, "continuation_anchor": "b" * 40,
              "origin_protocol": {"path": str(origin_protocol),
                                  "sha256": E.sha(origin_protocol)},
              "continuation_protocol": {"path": str(continuation_protocol),
                                        "sha256": E.sha(continuation_protocol)}}
    return {"inputs_path": inputs_path, "result": result, "queue": queue,
            "inputs": inputs, "first_receipt": first_receipt}


def patch_combined(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, object]) -> None:
    monkeypatch.setattr(E.CONT, "verify_origin", lambda _path: fixture["inputs"])
    monkeypatch.setattr(E.RV, "view_scope", lambda *_args, **_kwargs: nullcontext())
    checks = {"survival_ratio": True, "fall_frac": True,
              "err_xy_ratio": True, "err_yaw_ratio": True}

    def report(path: Path, _training: dict) -> dict:
        return {"tag": path.parent.parent.name, "by_condition": [{}] * 26,
                "metrics": {"fall_rate": 0.0, "mean_survival_s": 20.0,
                            "lin_vel_rmse_mps": 0.1, "yaw_rate_rmse_radps": 0.1}}

    monkeypatch.setattr(E.R, "_check_report", report)
    monkeypatch.setattr(E.R.TRAIN.FACT, "checked", lambda *_args, **_kwargs: ({}, {}))

    def recovery(value: dict, _reference: dict, _bands: dict) -> dict:
        tag = value["tag"]
        label = "healthy" if tag == "none" else "unhealthy"
        return {"outcome": "recovered" if tag == "none" else "not_recovered",
                "verdict": {"label": label, "checks": checks}}

    monkeypatch.setattr(E.R.C.D1, "recovery", recovery)
    monkeypatch.setattr(E.R, "verify_repair", lambda *_args, **_kwargs: {
        "recovery": recovery({"tag": "repair1"}, {}, {}),
        "output_training": {"checkpoint": {"sha256": "9" * 64}}})

    def diagnosis(path: Path, applied: list[str]) -> dict:
        epoch = 1 if "diagnosis-e1" in str(path) else 2
        if epoch == 1:
            applied.append("restore_reward")
        return {"epoch": epoch, "end_state": "confirmed" if epoch == 1 else "HOLD",
                "hold_reason": None if epoch == 1 else "probe_outcome_unknown",
                "probes": [{"probe": "P_reward", "outcome": "abnormal"}],
                "runtime_fact_citations": [], "model_requests": 1}

    monkeypatch.setattr(E, "_paired_diagnosis", diagnosis)
    monkeypatch.setattr(E.CONT.TRANSPORT, "status", lambda _queue: {
        "closed": {"reason": "done"}, "requests": 1, "dispatches": 1, "responses": 1,
        "pending": 0, "unknown": 0, "failures": 0})


def test_combined_export_preserves_invalid_origin_and_legitimate_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = combined_fixture(tmp_path)
    patch_combined(monkeypatch, fixture)
    out, private = tmp_path / "public", tmp_path / "private"
    public = E.export_combined(
        fixture["inputs_path"], fixture["result"], fixture["queue"], out, private)
    assert public["origin_run"] == {"status": "invalid", "queue_closed": True,
                                     "valid_first_receipt": True,
                                     "first_outcome": "not_recovered"}
    assert public["continuation"] == {"status": "HOLD", "outcome": "HOLD",
                                       "reason": "probe_outcome_unknown"}
    assert public["scope"]["recovery_count"] == 0
    assert len(list((out / "raw/api").glob("*.json"))) == 8
    assert public["total_provider_request_count"] == 2
    assert set(public["provider_by_phase"]) == {"origin_epoch1", "continuation_epoch2"}
    assert len(list((out / "raw/fixed").glob("*.json"))) == 3
    assert len(list((out / "raw/runtime").glob("*.json"))) == 6
    assert (private / "metadata.json").is_file()
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            assert "bench/private" not in text and "\\users\\" not in text


def test_combined_rejects_origin_status_rewrite_and_missing_result_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = combined_fixture(tmp_path)
    cast_inputs = fixture["inputs"]
    assert isinstance(cast_inputs, dict)
    cast_inputs["origin"]["phase"]["status"] = "complete"
    patch_combined(monkeypatch, fixture)
    with pytest.raises(ValueError, match="origin 전체 invalid"):
        E.export_combined(fixture["inputs_path"], fixture["result"], fixture["queue"],
                          tmp_path / "public", tmp_path / "private")
    assert not (tmp_path / "public").exists()
    with pytest.raises(ValueError, match="아직 없다"):
        E.export_combined(fixture["inputs_path"], tmp_path / "missing.json", fixture["queue"],
                          tmp_path / "public2", tmp_path / "private2")
    assert not (tmp_path / "public2").exists()


def test_optimizer_measurement_projection_uses_sealed_update_rows(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    rows = [{"update_index": index,
             "expected": {"optimizer_steps": 20, "minibatches": 20},
             "observed": {"optimizer_steps": value, "minibatches": value}}
            for index, value in enumerate((4, 4, 4))]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    plan = save(tmp_path / "plan.json", {"raw_files": {"records": str(records)}})
    receipt = save(tmp_path / "receipt.json", {
        "raw_refs": {"records": {"path": str(records), "sha256": E.sha(records)}}})
    projected = E._measurement_projection("P_optimizer_updates", {
        "manifest": {"expected_optimizer_steps_per_update": 20,
                     "expected_minibatches_per_update": 20},
        "records": 3,
    }, plan, receipt)
    assert projected == {
        "expected_optimizer_steps": 20, "expected_minibatches": 20, "records": 3,
        "updates": [{"update_index": index, "expected_optimizer_steps": 20,
                     "observed_optimizer_steps": 4, "expected_minibatches": 20,
                     "observed_minibatches": 4} for index in range(3)],
    }
