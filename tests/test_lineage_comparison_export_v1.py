from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "tests")]

import lineage_comparison_export_v1 as X
import test_lineage_comparison_batch_v1 as BT


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _transport(root: Path, requests: int, anchor: str,
               monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    for name in ("requests", "dispatches", "responses", "response-sha", "failures", "worker"):
        (root / name).mkdir(parents=True, exist_ok=True)
    plan = {"case_ids": [row["case_id"] for row in BT.cases()],
            "logical_slots": [{"slot_id": f"{row['case_id']}.model.{decision}",
                               "case_id": row["case_id"]}
                              for row in BT.cases() for decision in range(1, 4)],
            "global_request_cap": 90, "per_slot_max": 3}
    session = {"registration_anchor": anchor, "plan": plan}
    _write(root / "session.json", session)
    monkeypatch.setattr(X.T, "_session", lambda _root: session)
    closed = {"schema": "lineage_nim_transport_closed_v1", "reason": "batch complete",
              "closed_at": "2026-10-31T00:00:00+00:00"}
    _write(root / "closed.json", closed)
    for index in range(1, requests + 1):
        request_id = f"request-{index:02d}"
        request = {"request_id": request_id, "messages": [{"role": "assistant",
                    "content": "numeric public view"}], "provider": "private/nvidia"}
        request_path = _write(root / "requests" / f"{request_id}.json", request)
        dispatch = {"schema": "lineage_nim_transport_dispatch_v1", "request_id": request_id,
                    "request_sha256": X._sha(request_path),
                    "claimed_at": "2026-10-31T00:00:00+00:00"}
        dispatch_path = _write(root / "dispatches" / f"{request_id}.json", dispatch)
        response = {"schema": X.T.RESPONSE_SCHEMA, "request_id": request_id,
                    "request_sha256": X._sha(request_path),
                    "dispatch_sha256": X._sha(dispatch_path),
                    "result": {"provider_model": "private/nvidia", "content": "{}"}}
        response_path = _write(root / "responses" / f"{request_id}.json", response)
        _write(root / "response-sha" / f"{request_id}.sha256",
               {"request_id": request_id, "response_sha256": X._sha(response_path)})
    status = X.T.status(root)
    start = {"schema": "lineage_nim_transport_worker_start_v1",
             "evidence_scope": "actual_wsl", "registration_anchor": anchor,
             "session_sha256": X._sha(root / "session.json")}
    start_path = _write(root / "worker/start.json", start)
    result = {"schema": "lineage_nim_transport_worker_result_v1", "status": "closed",
              "evidence_scope": "actual_wsl", "posts": requests,
              "start_sha256": X._sha(start_path), "queue_status": status}
    _write(root / "worker/result.json", result)
    return session, status


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo = tmp_path / "repo"
    output = tmp_path / "private-batch"
    execution = output / "execution"
    transport = tmp_path / "private-transport"
    protocol_paths = {}
    for name in ("batch", "transport", "probe"):
        protocol_paths[name] = _write(repo / f"bench/protocols/{name}.json", {"name": name})
    cases = BT.cases()
    plan = X.BATCH.latin_plan(cases)
    protocol = {"cases": cases, "latin_execution_plan": plan}
    collected = {}
    for case in cases:
        for job in (case["source_job"], case["reference_job"]):
            collected[job] = {"job_id": job, "seed": case["seed"],
                              "checkpoint_sha256": (job.encode().hex() + "0" * 64)[:64],
                              "telemetry_sha256": (job.encode().hex() + "1" * 64)[:64],
                              "path": str(tmp_path / f"{job}.json")}
    gates = {case["case_id"]: {"label": "eligible"} for case in cases}
    rows = []
    for planned in plan:
        case = next(item for item in cases if item["case_id"] == planned["case_id"])
        epoch = BT.fake_result(case["case_id"], planned["policy"])
        epoch.update({"status": "complete", "evidence_scope": "actual_backend",
                      "case_id": case["case_id"], "policy": planned["policy"],
                      "epoch2_registered": False, "decision_count": 0, "steps": [],
                      "probe_order": [], "remaining_candidates": [],
                      "hold_reason": None, "model_observation_use": [], "selection_actor": [],
                      "source_binding": {
                          "case": {key: collected[case["source_job"]][key]
                                   for key in ("job_id", "seed", "checkpoint_sha256", "telemetry_sha256")},
                          "reference": {key: collected[case["reference_job"]][key]
                                        for key in ("job_id", "seed", "checkpoint_sha256", "telemetry_sha256")}}})
        if planned["order"] == 7:
            epoch["decision_count"] = 1
            epoch["probe_order"] = [X.BATCH.E.C.RC.NEW_PROBE]
            pair = {}
            for role, status in (("case", "abnormal"), ("reference", "normal")):
                stage = execution / "fixture-probes" / role
                plan_path = _write(stage / "plan.json", {"role": role})
                report = _write(stage / "report.json", {"status": status})
                receipt = {"role": role, "status": "complete",
                           "probe": X.BATCH.E.C.RC.NEW_PROBE,
                           "measurement": {"status": status},
                           "plan": {"path": str(plan_path)},
                           "raw_refs": {"report": {"path": str(report),
                                                      "sha256": X._sha(report)}}}
                _write(stage / "probe_receipt.json", receipt)
                pair[role] = receipt
            epoch["steps"] = [{"decision": 1, "probe": X.BATCH.E.C.RC.NEW_PROBE,
                               "outcome": "abnormal", "paired_receipts": pair}]
        result_path = _write(execution / "runs" /
                             f"{planned['order']:02d}_{case['case_id']}_{planned['policy']}" /
                             "result.json", epoch)
        row = {**deepcopy(planned), "arm": case["arm"], "gate_label": "eligible",
               "fresh_view_digest": f"view-{case['case_id']}", "result": epoch,
               "result_path": str(result_path), "result_sha256": X._sha(result_path),
               "classification": X.BATCH._classification(case, epoch)}
        _write(execution / "rows" / f"{planned['order']:02d}.json", row)
        rows.append(row)
    batch_result = {"schema": X.BATCH.RESULT_SCHEMA, "status": "complete",
                    "evidence_scope": "actual_backend", "rows": rows,
                    "summary": X.BATCH._summary(rows), "correction_executed": False,
                    "training_executed": False, "epoch2_registered": False}
    result_path = _write(execution / "result.json", batch_result)
    anchor = "a" * 40
    manifest = {"output_root": str(output), "execution_plan": plan,
                "transport_root": str(transport), "transport_plan": {},
                "execution_anchor": anchor, "source_collection_anchor": "b" * 40,
                "source_result_sha256": "1" * 64, "collected_index_sha256": "2" * 64,
                "gate_sha256": "3" * 64, "protocol_path": str(protocol_paths["batch"]),
                "protocol_sha256": X._sha(protocol_paths["batch"]),
                "transport_protocol": str(protocol_paths["transport"]),
                "transport_protocol_sha256_lf": X._sha(protocol_paths["transport"]),
                "probe_protocol": str(protocol_paths["probe"]),
                "probe_protocol_sha256_lf": X._sha(protocol_paths["probe"])}
    session, _status = _transport(transport, 10, anchor, monkeypatch)
    manifest["transport_plan"] = session["plan"]
    manifest_path = _write(output / "manifest.json", manifest)
    source_summary = {"status": "complete", "planned_cases": 10, "valid_pairs": 10,
                      "excluded_total": 0, "diagnosis_executed": False,
                      "correction_executed": False, "recovery_claim": False,
                      "label_counts": {"eligible": 8, "out_of_support": 2},
                      "source_result_sha256": "1" * 64, "collected_index_sha256": "2" * 64,
                      "source_collection_anchor": "b" * 40,
                      "continuation_and_eval_anchor": anchor,
                      "evaluation_result_sha256": "5" * 64,
                      "development_coupling": {"heldout_claim": False},
                      "source_protocol": "bench/protocols/source.json",
                      "source_protocol_sha256_lf": "4" * 64}
    _write(repo / X.SOURCE_SUMMARY_RELATIVE, source_summary)
    monkeypatch.setattr(X.BATCH, "_validate_manifest",
                        lambda _path: (manifest, protocol, collected, gates))
    return {"repo": repo, "output": output, "manifest": manifest,
            "manifest_path": manifest_path, "result_path": result_path,
            "result": batch_result, "transport": transport, "session": session}


def test_completed_export_preserves_public_numbers_raw_sha_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "public"
    manifest = X.export_completed(fixture["manifest_path"], fixture["result_path"],
                                  fixture["transport"], out, repo=fixture["repo"])
    assert X.verify_export(out) == manifest
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["comparison"] == fixture["result"]["summary"]
    assert summary["scope"]["heldout"] is False
    assert summary["scope"]["general_advantage_claim"] is False
    assert summary["scope"]["correction_executed"] is False
    assert len(list((out / "rows").glob("*.json"))) == 30
    assert len(list((out / "nim/requests").glob("*.json"))) == 10
    observation = json.loads((out / "observations/07-01-case.json").read_text(encoding="utf-8"))
    assert observation["schema"] == X.OBSERVATION_SCHEMA
    assert observation["measurement"] == {"status": "abnormal"}
    assert manifest["raw_artifact_git_note"].endswith("-text handling before commit")
    public_text = "".join(path.read_text(encoding="utf-8") for path in out.rglob("*.*"))
    assert "private/nvidia" in public_text
    assert str(tmp_path) not in public_text


def test_probe_role_unknown_preserves_hold_denominator_and_known_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = deepcopy(fixture["result"])
    row = result["rows"][13]
    assert row["order"] == 14 and row["policy"] == "model_adaptive"
    epoch = row["result"]
    paired = {}
    for role, status, measurement, error_type, wall_s in (
        ("case", "complete", {"episode_len_mean": 19.0}, None, 0.7),
        ("reference", "unknown", None, "PermissionError", 0.8),
    ):
        stage = fixture["output"] / "execution/fixture-unknown" / role
        plan_path = _write(stage / "plan.json", {"role": role})
        report = _write(stage / "report.json", {"role": role, "status": status})
        receipt = {
            "role": role, "status": status, "probe": "P_episode",
            "measurement": measurement, "error_type": error_type,
            "plan": {"path": str(plan_path)},
            "execution": {"exit_code": 0, "natural_exit": True, "timed_out": False,
                          "error_type": None, "cleanup_verified": True,
                          "job_assigned_before_release": True, "wall_s": wall_s},
            "raw_refs": {"report": {"path": str(report), "sha256": X._sha(report)}},
        }
        _write(stage / "probe_receipt.json", receipt)
        paired[role] = receipt
    epoch.update({"end_state": "HOLD", "hold_reason": "probe_outcome_unknown",
                  "confirmed_subtype": None, "decision_count": 1,
                  "probe_order": ["P_episode"],
                  "steps": [{"decision": 1, "probe": "P_episode", "outcome": "unknown",
                             "paired_receipts": paired}],
                  "wall": {**epoch["wall"], "probe_child_known_s": 1.5,
                           "probe_child_unknown_roles": [], "probe_parent_known_s": 1.7}})
    raw_result = Path(row["result_path"]); _write(raw_result, epoch)
    row["result_sha256"] = X._sha(raw_result)
    row["classification"] = X.BATCH._classification(BT.cases()[4], epoch)
    _write(fixture["output"] / "execution/rows/14.json", row)
    result["summary"] = X.BATCH._summary(result["rows"])
    _write(fixture["result_path"], result)

    out = tmp_path / "public"
    exported = X.export_completed(fixture["manifest_path"], fixture["result_path"],
                                  fixture["transport"], out, repo=fixture["repo"])
    assert X.verify_export(out) == exported
    public_row = json.loads((out / "rows/14.json").read_text(encoding="utf-8"))
    reference = json.loads((out / "observations/14-01-reference.json").read_text(encoding="utf-8"))
    assert public_row["end_state"] == "HOLD"
    assert public_row["hold_reason"] == "probe_outcome_unknown"
    assert public_row["wall"]["probe_child_known_s"] == 1.5
    assert public_row["wall"]["probe_parent_known_s"] == 1.7
    assert reference["status"] == "unknown" and reference["measurement"] is None
    assert reference["error_type"] == "PermissionError"
    assert reference["execution"]["wall_s"] == 0.8
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["comparison"]
    assert summary == result["summary"]
    assert summary["denominator_case_policy_pairs"] == 30
    assert summary["by_policy"]["model_adaptive"]["paired_backend_unknown"] >= 1


def test_parent_exception_stub_is_exported_only_as_unknown_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = deepcopy(fixture["result"])
    row = result["rows"][17]
    assert row["order"] == 18 and row["policy"] == "fixed_eligible"
    epoch = row["result"]
    paired = {
        role: {"status": "unknown", "measurement": None, "execution": {},
               "error_type": "RuntimeError", "role": role}
        for role in ("case", "reference")
    }
    epoch.update({"end_state": "HOLD", "hold_reason": "probe_outcome_unknown",
                  "confirmed_subtype": None, "decision_count": 1,
                  "probe_order": ["P_noise"],
                  "steps": [{"decision": 1, "probe": "P_noise", "outcome": "unknown",
                             "paired_receipts": paired}],
                  "wall": {**epoch["wall"], "probe_child_known_s": 0,
                           "probe_child_unknown_roles": ["reference", "case"]}})
    raw_result = Path(row["result_path"]); _write(raw_result, epoch)
    row["result_sha256"] = X._sha(raw_result)
    row["classification"] = X.BATCH._classification(BT.cases()[5], epoch)
    _write(fixture["output"] / "execution/rows/18.json", row)
    result["summary"] = X.BATCH._summary(result["rows"])
    _write(fixture["result_path"], result)

    out = tmp_path / "public"
    exported = X.export_completed(fixture["manifest_path"], fixture["result_path"],
                                  fixture["transport"], out, repo=fixture["repo"])
    assert X.verify_export(out) == exported
    public_row = json.loads((out / "rows/18.json").read_text(encoding="utf-8"))
    case = json.loads((out / "observations/18-01-case.json").read_text(encoding="utf-8"))
    assert public_row["end_state"] == "HOLD"
    assert public_row["hold_reason"] == "probe_outcome_unknown"
    assert case == {
        "decision": 1,
        "error_type": "RuntimeError",
        "evidence_scope": "parent_exception_without_probe_receipt",
        "execution": {},
        "measurement": None,
        "order": 18,
        "probe": None,
        "projection_rule": "numeric/string values preserved; path values become basename+raw_text_sha256",
        "raw_refs": {},
        "role": "case",
        "schema": X.OBSERVATION_SCHEMA,
        "status": "unknown",
    }
    provenance = public_row["provenance"]["probe_receipts"]
    assert all(item["receipt_sha256"] is None and item["receipt_missing"] is True
               and item["raw_refs"] == {} for item in provenance)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["comparison"] \
        == result["summary"]


@pytest.mark.parametrize("mutation", ["measurement", "execution", "extra"])
def test_malformed_parent_exception_stub_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = deepcopy(fixture["result"])
    row = result["rows"][17]
    epoch = row["result"]
    stub = {"status": "unknown", "measurement": None, "execution": {},
            "error_type": "RuntimeError", "role": "case"}
    if mutation == "measurement":
        stub["measurement"] = {"status": "normal"}
    elif mutation == "execution":
        stub["execution"] = {"exit_code": 0}
    else:
        stub["unexpected"] = True
    peer = {"status": "unknown", "measurement": None, "execution": {},
            "error_type": "RuntimeError", "role": "reference"}
    epoch.update({"end_state": "HOLD", "hold_reason": "probe_outcome_unknown",
                  "decision_count": 1, "probe_order": ["P_noise"],
                  "steps": [{"decision": 1, "probe": "P_noise", "outcome": "unknown",
                             "paired_receipts": {"case": stub, "reference": peer}}]})
    raw_result = Path(row["result_path"]); _write(raw_result, epoch)
    row["result_sha256"] = X._sha(raw_result)
    row["classification"] = X.BATCH._classification(BT.cases()[5], epoch)
    _write(fixture["output"] / "execution/rows/18.json", row)
    result["summary"] = X.BATCH._summary(result["rows"])
    _write(fixture["result_path"], result)
    with pytest.raises(ValueError, match="stub|결합"):
        X.export_completed(fixture["manifest_path"], fixture["result_path"],
                           fixture["transport"], tmp_path / "public", repo=fixture["repo"])
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("mutation", ["partial", "open", "unknown"])
def test_partial_open_or_unknown_never_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if mutation == "partial":
        result = deepcopy(fixture["result"]); result["status"] = "running"
        _write(fixture["result_path"], result)
    elif mutation == "open":
        (fixture["transport"] / "closed.json").unlink()
    else:
        response = next((fixture["transport"] / "responses").glob("*.json"))
        response.unlink()
        next((fixture["transport"] / "response-sha").glob("*.sha256")).unlink()
    with pytest.raises(ValueError):
        X.export_completed(fixture["manifest_path"], fixture["result_path"],
                           fixture["transport"], tmp_path / "public", repo=fixture["repo"])
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("mutation", ["source", "view", "role", "response"])
def test_source_view_role_and_response_tamper_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if mutation in {"source", "view", "role"}:
        result = deepcopy(fixture["result"])
        row = result["rows"][6 if mutation == "role" else 0]
        if mutation == "source":
            row["result"]["source_binding"]["case"]["telemetry_sha256"] = "f" * 64
        elif mutation == "view":
            row["fresh_view_digest"] = "changed-view"
        else:
            row["result"]["steps"][0]["paired_receipts"]["case"]["role"] = "reference"
        raw_result = Path(row["result_path"]); _write(raw_result, row["result"])
        row["result_sha256"] = X._sha(raw_result)
        _write(fixture["output"] / "execution/rows" / f"{row['order']:02d}.json", row)
        result["summary"] = X.BATCH._summary(result["rows"])
        _write(fixture["result_path"], result)
    else:
        receipt = next((fixture["transport"] / "response-sha").glob("*.sha256"))
        value = json.loads(receipt.read_text(encoding="utf-8")); value["response_sha256"] = "f" * 64
        _write(receipt, value)
    with pytest.raises(ValueError):
        X.export_completed(fixture["manifest_path"], fixture["result_path"],
                           fixture["transport"], tmp_path / "public", repo=fixture["repo"])
    assert not (tmp_path / "public").exists()


def test_private_path_or_key_in_raw_queue_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    request = next((fixture["transport"] / "requests").glob("*.json"))
    value = json.loads(request.read_text(encoding="utf-8"))
    value["messages"][0]["content"] = "private local path C:\\Users\\person"
    _write(request, value)
    dispatch = fixture["transport"] / "dispatches" / request.name
    dispatch_value = json.loads(dispatch.read_text(encoding="utf-8"))
    dispatch_value["request_sha256"] = X._sha(request); _write(dispatch, dispatch_value)
    response = fixture["transport"] / "responses" / request.name
    response_value = json.loads(response.read_text(encoding="utf-8"))
    response_value["request_sha256"] = X._sha(request)
    response_value["dispatch_sha256"] = X._sha(dispatch); _write(response, response_value)
    sha_path = fixture["transport"] / "response-sha" / request.with_suffix(".sha256").name
    _write(sha_path, {"request_id": request.stem, "response_sha256": X._sha(response)})
    status = X.T.status(fixture["transport"])
    worker = fixture["transport"] / "worker/result.json"
    worker_value = json.loads(worker.read_text(encoding="utf-8")); worker_value["queue_status"] = status
    _write(worker, worker_value)
    with pytest.raises(ValueError, match="절대경로|credential"):
        X.export_completed(fixture["manifest_path"], fixture["result_path"],
                           fixture["transport"], tmp_path / "public", repo=fixture["repo"])
    assert not (tmp_path / "public").exists()


def test_existing_output_is_rejected_without_touching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "public"; out.mkdir(); sentinel = out / "keep.txt"; sentinel.write_text("keep")
    with pytest.raises(ValueError, match="fresh"):
        X.export_completed(fixture["manifest_path"], fixture["result_path"],
                           fixture["transport"], out, repo=fixture["repo"])
    assert sentinel.read_text() == "keep"
