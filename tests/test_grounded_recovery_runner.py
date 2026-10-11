"""Grounded recovery 부모의 승인 소비, source 결합, SDK 산출물 검증 회귀."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))
import run_grounded_recovery as H  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def request_fixture(monkeypatch, tmp_path):
    private = tmp_path / "private"
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, {"fixed": True})
    monkeypatch.setattr(H, "PRIVATE_ROOT", private)
    monkeypatch.setattr(H, "PROTOCOL", protocol_path)
    source = {"source": "bound"}
    protocol = {"device": "cuda:0"}
    commit = "a" * 40
    entry = {"checkpoint_path": str(tmp_path / "model.pt"), "action_scale": 0.25,
             "judgment_influence": {"judgment_influence_proven": False}}
    monkeypatch.setattr(H, "verified_protocol", lambda _isaac: (protocol, source, commit))
    monkeypatch.setattr(H, "validate_entry", lambda _tag, _isaac: deepcopy(entry))
    core = {"grounded_tag": "grounded", "source_commit_sha": commit,
            "protocol_sha256": H.W.file_sha(protocol_path), "entry": entry,
            "effects": [{"arm": arm, "callback": H.CALLBACKS[arm]} for arm in H.C.ARMS],
            "execution_limit": 3}
    document = {"schema": "grounded_recovery_request_v1", "bundle_key": H.A._sha(core), **core}
    folder = private / "tag"
    write_json(folder / "request.json", document)
    return folder, document, protocol, source, entry


def test_request_approval_and_current_source_files_are_bound(monkeypatch, tmp_path):
    folder, document, protocol, source, entry = request_fixture(monkeypatch, tmp_path)
    loaded = H.load_request("tag", tmp_path / "IsaacLab")
    assert loaded[:2] == (folder, document) and loaded[2:] == (protocol, source)
    approval = H.approve("tag", "operator", tmp_path / "IsaacLab")
    assert H.validate_approval(folder, document) == approval

    changed = deepcopy(document)
    changed["entry"] = {**entry, "action_scale": 0.5}
    write_json(folder / "request.json", changed)
    with pytest.raises(ValueError, match="binding"):
        H.load_request("tag", tmp_path / "IsaacLab")

    write_json(folder / "request.json", document)
    tampered = deepcopy(approval)
    tampered["request_sha256"] = "f" * 64
    write_json(folder / "approval.json", tampered)
    with pytest.raises(ValueError, match="approval"):
        H.validate_approval(folder, document)

    write_json(folder / "approval.json", approval)
    monkeypatch.setattr(H, "validate_entry", lambda _tag, _isaac: {**entry, "action_scale": 0.75})
    with pytest.raises(ValueError, match="source|decision"):
        H.load_request("tag", tmp_path / "IsaacLab")


def execute_fixture(monkeypatch, tmp_path, *, key="b" * 64):
    private = tmp_path / "private"
    folder = private / "tag"
    folder.mkdir(parents=True)
    monkeypatch.setattr(H, "PRIVATE_ROOT", private)
    document = {"bundle_key": key, "source_commit_sha": "a" * 40,
                "protocol_sha256": "c" * 64,
                "entry": {"checkpoint_path": str(tmp_path / "model.pt"),
                          "judgment_influence": {"judgment_influence_proven": False}}}
    monkeypatch.setattr(H, "load_request", lambda _tag, _isaac: (folder, document, {}, {}))
    consumed = {"schema": "consumed", "bundle_key": key}
    monkeypatch.setattr(H, "bundle_consumed", lambda *_args: deepcopy(consumed))
    return private, folder, document, consumed


def test_gpu_busy_consumes_nothing_and_launches_nothing(monkeypatch, tmp_path):
    private, folder, _, _ = execute_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(H.M, "query_gpu", lambda *_args: {"busy": True})
    monkeypatch.setattr(H, "run_effect", lambda *_args: pytest.fail("busy GPU launched an effect"))
    with pytest.raises(RuntimeError, match="approval not consumed"):
        H.execute("tag", tmp_path / "IsaacLab")
    assert not (folder / "bundle_consumed.json").exists()
    assert not (private / "consumed").exists()


def test_same_bundle_global_consume_blocks_another_tag(monkeypatch, tmp_path):
    private, first, document, consumed = execute_fixture(monkeypatch, tmp_path)
    second = private / "other"
    second.mkdir()
    monkeypatch.setattr(H.M, "query_gpu", lambda *_args: {"busy": False, "processes": []})
    folders = {"tag": first, "other": second}
    monkeypatch.setattr(H, "load_request", lambda tag, _isaac: (folders[tag], document, {}, {}))
    monkeypatch.setattr(H, "bundle_consumed", lambda *_args: deepcopy(consumed))
    launches = []

    def fail_first(folder, _document, arm, _isaac):
        launches.append((folder.name, arm))
        (folder / "arms").mkdir(exist_ok=True)
        raise RuntimeError("stop")

    monkeypatch.setattr(H, "run_effect", fail_first)
    first_result = H.execute("tag", tmp_path / "IsaacLab")
    assert first_result["status"] == "invalid" and launches == [("tag", "fault_before")]
    with pytest.raises(FileExistsError):
        H.execute("other", tmp_path / "IsaacLab")
    assert launches == [("tag", "fault_before")]


def test_bundle_and_effect_consumption_precede_owned_launch(monkeypatch, tmp_path):
    private, folder, document, consumed = execute_fixture(monkeypatch, tmp_path)
    write_json(folder / "request.json", {"request": True})
    write_json(folder / "approval.json", {"approval": True})
    write_json(private / "consumed" / f"{document['bundle_key']}.json", consumed)
    write_json(folder / "bundle_consumed.json", consumed)
    monkeypatch.setattr(H, "effect_consumed", lambda *_args: {"effect": "bound"})
    monkeypatch.setattr(H, "make_manifest", lambda *_args: {"manifest": "bound"})
    monkeypatch.setattr(H, "validate_manifest", lambda path, isaac: {"manifest": {}})
    monkeypatch.setattr(H.M, "query_gpu", lambda *_args: {"busy": False, "processes": []})

    def owned(arm_folder, *_args):
        assert H.read(private / "consumed" / f"{document['bundle_key']}.json") == consumed
        assert H.read(folder / "bundle_consumed.json") == consumed
        assert H.read(arm_folder / "effect_consumed.json") == {"effect": "bound"}
        assert (arm_folder / "manifest.json").is_file()
        return {"job_assigned_before_release": True, "natural_exit": False, "exit_code": 1,
                "cleanup_verified": True, "timed_out": False, "error_type": "failed",
                "owned_process_identities": {}, "wall_s": 0.1}

    monkeypatch.setattr(H.P, "run_owned", owned)
    parent, report, body = H.run_effect(folder, document, "fault_before", tmp_path / "IsaacLab")
    assert parent["status"] == "invalid" and report is body is None


def test_first_arm_failure_stops_triplet_and_consumed_bundle_never_reruns(monkeypatch, tmp_path):
    _, folder, _, _ = execute_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(H.M, "query_gpu", lambda *_args: {"busy": False, "processes": []})
    launches = []

    def fail(folder, document, arm, isaac):
        launches.append(arm)
        (folder / "arms" / arm).mkdir(parents=True)
        return ({"status": "invalid", "execution": {"wall_s": 0.1}}, None, None)

    monkeypatch.setattr(H, "run_effect", fail)
    result = H.execute("tag", tmp_path / "IsaacLab")
    assert result["status"] == "invalid" and launches == ["fault_before"]
    assert result["effect_order"] == ["fault_before"]
    with pytest.raises(FileExistsError, match="never rerun"):
        H.execute("tag", tmp_path / "IsaacLab")
    assert launches == ["fault_before"]


def fixed_report(name: str, *, lin=0.2, yaw=0.2) -> dict:
    alive = 40000
    protocol_sha = H.R.sha_lf(H.FIXED_EVAL)
    rows = [{"command": list(command), "fall_rate": 0.0,
             "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw,
             "num_envs": 40, "fall_count": 0, "alive_steps_sum": alive,
             "lin_error_sq_sum": lin * lin * alive,
             "yaw_error_sq_sum": yaw * yaw * alive} for command in H.C.GRID]
    report = {"protocol": H.C.PROTOCOL, "protocol_sha256_lf": protocol_sha, "task": H.C.TASK,
              "eval_seed": 2026, "num_envs": 1040, "horizon_steps": 1000, "step_dt": 0.02,
              "action_scale": 0.25, "standstill": dict(H.C.STANDSTILL),
              "versions": {"torch": "2.7", "gpu": "GPU", "isaaclab": "x", "isaaclab_tasks": "x",
                           "isaaclab_rl": "x", "rsl-rl-lib": "2.3.3", "isaacsim": None},
              "name": name, "checkpoint": {"file": "model_299.pt", "sha256": H.CHECKPOINT_SHA256},
              "metrics": {"fall_rate": 0.0, "mean_survival_s": 20.0,
                          "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw},
              "by_condition": rows, "elapsed_s": 10.0,
              "started_at": "2026-10-12T00:00:00+00:00", "finished_at": "2026-10-12T00:00:10+00:00"}
    H.C.validate_fixed_report(report, {"name": name, "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": H.CHECKPOINT_SHA256, "action_scale": 0.25,
        "protocol_sha256_lf": protocol_sha})
    return report


def sdk_fixture(tmp_path, arm="fault_before", *, lin=0.2, yaw=0.2, validated=None):
    folder = tmp_path / arm
    folder.mkdir(parents=True, exist_ok=True)
    report = fixed_report(arm, lin=lin, yaw=yaw)
    write_json(folder / "fixed_report.json", report)
    resolved_event = {"mode": "reset", "params": {"com_range": {
        "x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.01, 0.01]},
        "asset_cfg": {"name": "robot", "body_names": ["base"], "body_ids": [1]}}}
    original_event = {"mode": "reset", "params": {"com_range": {
        "x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.01, 0.01]},
        "asset_cfg": {"name": "robot", "body_names": "base", "body_ids": slice(None)}}}
    target = {"source": "base.scene['robot'].body_names", "asset_name": "robot",
              "body_names": ["trunk", "base", "foot"], "base_body_index": 1}
    observations = {"policy": {"enable_corruption": False}}
    environment = {"seed": 2026, "scene": {"num_envs": 1040}, "decimation": 4,
                   "sim": {"dt": 0.005, "device": "cuda:0"}, "episode_length_s": 20.0,
                   "actions": {"joint_pos": {"scale": 0.25}},
                   "commands": {"base_velocity": {"heading_command": False, "rel_standing_envs": 0.0,
                                                    "resampling_time_range": [1e9, 1e9]}},
                   "events": {"add_base_mass": None, "base_com": original_event},
                   "observations": observations}
    config = {"environment": environment, "observation_configuration": observations,
              "injection_readback": deepcopy(H.B.EXPECTED_INJECTION_READBACK),
              "resolved_event_readback": {"source": "base.event_manager.get_term_cfg('base_com')",
                                          "base_com": resolved_event},
              "target_body_readback": target}
    (folder / "configuration.yaml").write_text(yaml.dump(config, sort_keys=True), encoding="utf-8")
    identity = {"process_id": 123, "process_creation_time_filetime": 456}
    callback = H.CALLBACKS[arm]
    manifest = {"arm": arm, "callback": callback, "checkpoint_path": str(tmp_path / "model_299.pt"),
                "action_scale": 0.25}
    if validated is None:
        validated = {"manifest": manifest,
                     "paths": {"report_path": folder / "fixed_report.json", "body_path": folder / "body.json",
                               "configuration_path": folder / "configuration.yaml"},
                     "project_commit_sha": "a" * 40, "protocol_sha256": "c" * 64,
                     "manifest_sha256": "d" * 64, "source_receipt": {"source": "bound"}}
    else:
        manifest = validated["manifest"]
    clocks = {"rows": [{"control_before": i, "control_after": i + 1,
                         "physics_before": i * 4, "physics_after": (i + 1) * 4} for i in range(1000)],
              "decimation": 4, "reset_calls": 2}
    body = {"schema": H.D.BODY_SCHEMA, "completed": True, "arm": arm, "callback": callback,
            "checkpoint_sha256": H.CHECKPOINT_SHA256,
            "isaacsim_version": "4.5.0",
            "configuration_sha256": H.W.file_sha(folder / "configuration.yaml"),
            "observation_configuration_sha256": H.A._sha(observations),
            "callback_code_identity_verified": True,
            "injection_readback": config["injection_readback"],
            "resolved_event_readback": config["resolved_event_readback"], "clocks": clocks,
            "target_body_readback": target,
            "env_create_count": 1, "process_identity": identity,
            "project_commit_sha": validated["project_commit_sha"],
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": validated["manifest_sha256"], "source_receipt": validated["source_receipt"],
            "report_sha256": H.W.file_sha(folder / "fixed_report.json"),
            "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True}
    write_json(folder / "body.json", body)
    write_json(folder / "child_started.json", {"schema": "grounded_recovery_child_started_v1", **identity,
                                                "manifest_sha256": validated["manifest_sha256"]})
    execution = {"owned_process_identities": {"123": 456}}
    assert H.verify_body(body, validated, execution)["name"] == arm
    return body, validated, execution, config


@pytest.mark.parametrize("tamper", ["body", "report", "configuration", "clock", "owned_pid", "isaacsim",
                                      "target_name", "target_index"])
def test_sdk_body_report_configuration_clock_and_owned_identity_tamper_rejected(tmp_path, tamper):
    body, validated, execution, config = sdk_fixture(tmp_path)
    changed = deepcopy(body)
    if tamper == "body":
        changed["checkpoint_sha256"] = "0" * 64
    elif tamper == "report":
        report = H.read(validated["paths"]["report_path"])
        report["metrics"]["lin_vel_rmse_mps"] = 9.0
        write_json(validated["paths"]["report_path"], report)
        changed["report_sha256"] = H.W.file_sha(validated["paths"]["report_path"])
    elif tamper == "configuration":
        config["environment"]["scene"]["num_envs"] = 1039
        validated["paths"]["configuration_path"].write_text(yaml.dump(config), encoding="utf-8")
        changed["configuration_sha256"] = H.W.file_sha(validated["paths"]["configuration_path"])
    elif tamper == "clock":
        changed["clocks"]["rows"][500]["physics_after"] += 1
    elif tamper == "isaacsim":
        changed["isaacsim_version"] = "4.6.0"
    elif tamper == "target_name":
        changed["target_body_readback"]["body_names"][1] = "not_base"
    elif tamper == "target_index":
        changed["target_body_readback"]["base_body_index"] = True
    else:
        execution["owned_process_identities"] = {"123": 999}
    with pytest.raises(ValueError):
        H.verify_body(changed, validated, execution)


def fresh_record_fixture(tmp_path, *, started=1, finished=1, max_tokens=None):
    folder = tmp_path / "fresh"
    folder.mkdir(exist_ok=True)
    packet = {"schema": "grounded_active_post_packet_v1"}
    write_json(folder / "actual.packet.json", packet)
    record = {"schema": "grounded_active_model_call_v1", "valid": True, "stage": "actual",
              "packet_sha256": H.W.file_sha(folder / "actual.packet.json"), "model": H.W.MODEL,
              "temperature": 0, "semantic_retries": 0, "transport_retries": 0,
              "client_cache_used": False, "saved_response_cached": False,
              "messages_sha256": H.W.canonical_sha(H.W.post_messages(packet)),
              "llm_calls_started": started, "llm_calls_finished": finished,
              "provider_response_id": "response", "provider_model": H.W.MODEL,
              "max_output_tokens": H.W.POST_MAX_TOKENS if max_tokens is None else max_tokens,
              "reasoning_effort": H.W.POST_REASONING_EFFORT, "provider_cache_status": "not_attested",
              "decision": {}}
    write_json(folder / "actual.model.json", record)
    outcome = {"valid": True, "worker_cleanup_verified": True, "stage": "actual",
               "model_record_sha256": H.W.file_sha(folder / "actual.model.json"),
               "launch": {"returncode": 0, "timed_out": False}}
    write_json(folder / "actual.outcome.json", outcome)
    return folder, packet


@pytest.mark.parametrize("started,finished,max_tokens", [
    (True, True, None), (1, 0, None), (1, 1, 1),
])
def test_fresh_record_rejects_bool_incomplete_or_wrong_budget(tmp_path, started, finished, max_tokens):
    folder, packet = fresh_record_fixture(tmp_path, started=started, finished=finished, max_tokens=max_tokens)
    with pytest.raises(ValueError, match="fresh actual"):
        H.fresh_record(folder, "actual", packet)


def entry_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "ROOT", tmp_path)
    folder = tmp_path / "grounded"
    folder.mkdir()
    case, effect = "case_public", "e" * 64
    metadata_path = tmp_path / "source.meta.json"
    write_json(metadata_path, {"source": True})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = run_dir / "env.yaml"
    config_path.write_text("actions:\n  joint_pos:\n    scale: 0.25\n", encoding="utf-8")
    spec = {"arm": "X1", "source_seed": 7, "case_id": case, "metadata_path": str(metadata_path),
            "source_commit_sha": "a" * 40}
    binding = {"case_source_binding_sha256": "b" * 64}
    monkeypatch.setattr(H.R, "load_run", lambda _tag: (folder, spec, binding))

    predictions = [
        {"measurement_id": "base_com_max_abs_delta_xy", "relation": "gt",
         "operand": {"value": 0.0501}},
        {"measurement_id": "base_com_outside_range_fraction", "relation": "gt",
         "operand": {"value": 0.0}},
    ]
    assessment = {"next_check": {"registry_id": "reset_base_com_stability",
                                  "hypothesis_predictions": [
                                      {"hypothesis_id": "h1", "predictions": predictions},
                                      {"hypothesis_id": "h2", "predictions": []}]},
                  "hypotheses": [{"id": "h1", "rank": 1}, {"id": "h2", "rank": 2}],
                  "selected_evidence_ids": []}
    receipt = {"execution_valid": True, "case_source_binding_sha256": binding["case_source_binding_sha256"],
               "measurements": [
        {"measurement_id": "base_com_max_abs_delta_xy", "value": 0.09},
        {"measurement_id": "base_com_outside_range_fraction", "value": 0.5}],
    }
    consumed = {"effect_key": effect, "request_sha256": "1" * 64, "approval_sha256": "2" * 64,
                "case_source_binding_sha256": binding["case_source_binding_sha256"]}
    write_json(folder / "consumed_binding.json", {**consumed, "consumed_sha256": H.A._sha(consumed)})

    sdk = tmp_path / "bench/private/base_com_reset" / "ga_grounded"
    for name in ("manifest.json", "body.json", "parent_receipt.json"):
        write_json(sdk / name, {"name": name})
    body = {"observer_report": {"measurements": {
        "base_com_max_abs_delta_xy": {"value": 0.09},
        "base_com_outside_range_fraction": {"value": 0.5}}}}
    write_json(sdk / "body.json", body)
    parent = {"status": "pass", "arm": "X1", "project_commit_sha": spec["source_commit_sha"],
              "body_sha256": H.W.file_sha(sdk / "body.json"),
              "manifest_sha256": H.W.file_sha(sdk / "manifest.json"),
              "protocol_sha256": H.W.file_sha(H.B.PROTOCOL), "execution": {}}
    write_json(sdk / "parent_receipt.json", parent)
    monkeypatch.setattr(H.B, "validate_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(H.B, "verify_body", lambda *_args: [])
    monkeypatch.setattr(H.R, "original_source_metadata", lambda _path: (
        {"num_envs": 4096, "max_iterations": 300, "run_dir": str(run_dir)}, None, config_path))
    original_file_sha = H.W.file_sha
    monkeypatch.setattr(H.W, "file_sha", lambda path: H.CHECKPOINT_SHA256
                        if Path(path) == checkpoint else original_file_sha(Path(path)))
    monkeypatch.setattr(H.A, "_validate_consumed_document", lambda *_args: None)
    monkeypatch.setattr(H.G, "evidence_catalog", lambda _case: {"catalog_sha256": "3" * 64})
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("workflow: true\n", encoding="utf-8")
    monkeypatch.setattr(H.W, "CONFIG", workflow)

    def load_bound(directory, filename, *args):
        if directory == "grounded_active_assessments":
            return assessment
        if directory == "active_receipts":
            return receipt
        if directory == "active_consumed":
            return consumed
        raise AssertionError((directory, filename))

    monkeypatch.setattr(H.A, "_load_bound", load_bound)
    packet = {"schema": "grounded_active_post_packet_v1", "case_id": case, "effect_key": effect,
              "budget_left": 0, "hypotheses": [{"hypothesis_id": "h1"}, {"hypothesis_id": "h2"}],
              "selected_evidence_ids": [], "selected_evidence": [], "check_definition": {},
              "normalized_receipt": receipt, "normalized_receipt_sha256": H.A._sha(receipt)}
    monkeypatch.setattr(H.A, "_post_packet", lambda _case, _effect, _receipt, masked: {
        **deepcopy(packet), "normalized_receipt": {**deepcopy(receipt), "measurements": [
            {**row, "value": None if masked else row["value"]} for row in receipt["measurements"]]}})
    decision = {"schema": "grounded_active_final_decision_v1", "case_id": case, "effect_key": effect,
                "budget_left": 0, "status": "HOLD", "ranking": ["h1", "h2"],
                "receipt_citation": {"effect_key": effect,
                                     "normalized_receipt_sha256": H.A._sha(receipt)}}
    records = {"initial": {"assessment_sha256": H.A._sha(assessment)},
               "actual": {"decision": decision, "provider_response_id": "actual", "provider_model": "model"}}

    def fresh(_folder, stage, _packet):
        if stage == "masked":
            raise ValueError("masked pair intentionally absent")
        return deepcopy(records[stage])

    monkeypatch.setattr(H, "fresh_record", fresh)
    for name in ("host_spec.json", "case_binding.json", "initial.packet.json", "initial.model.json",
                 "initial.outcome.json", "actual.packet.json", "actual.model.json", "actual.outcome.json"):
        write_json(folder / name, {"artifact": name})
    return assessment, receipt, body, decision, records


def test_validate_entry_allows_correct_initial_hypothesis_without_rank_change(monkeypatch, tmp_path):
    _, _, _, decision, _ = entry_fixture(monkeypatch, tmp_path)
    entry = H.validate_entry("grounded", tmp_path / "IsaacLab")
    assert entry["actual_top"] == decision["ranking"][0] == "h1"
    assert entry["mechanism_values"] == {"base_com_max_abs_delta_xy": 0.09,
                                         "base_com_outside_range_fraction": 0.5}
    assert entry["judgment_influence"]["judgment_influence_proven"] is False
    assert entry["model_selected_intervention"] is False


@pytest.mark.parametrize("tamper", ["latest_citation", "initial_check", "actual_value_binding", "prediction"])
def test_validate_entry_rejects_selection_citation_prediction_or_value_binding(monkeypatch, tmp_path, tamper):
    assessment, receipt, body, decision, records = entry_fixture(monkeypatch, tmp_path)
    if tamper == "latest_citation":
        decision["receipt_citation"]["normalized_receipt_sha256"] = "0" * 64
    elif tamper == "initial_check":
        assessment["next_check"]["registry_id"] = "reward_curve_summary"
        records["initial"]["assessment_sha256"] = H.A._sha(assessment)
    elif tamper == "actual_value_binding":
        body["observer_report"]["measurements"]["base_com_max_abs_delta_xy"]["value"] = 0.08
        write_json(tmp_path / "bench/private/base_com_reset/ga_grounded/body.json", body)
        parent_path = tmp_path / "bench/private/base_com_reset/ga_grounded/parent_receipt.json"
        parent = H.read(parent_path)
        parent["body_sha256"] = H.W.file_sha(tmp_path / "bench/private/base_com_reset/ga_grounded/body.json")
        write_json(parent_path, parent)
    else:
        receipt["measurements"][0]["value"] = 0.04
    with pytest.raises(ValueError):
        H.validate_entry("grounded", tmp_path / "IsaacLab")


def integration_fixture(monkeypatch, tmp_path, metrics=None, *, invalid_arm=None, gpu_busy_call=None):
    metrics = metrics or {"fault_before": 0.32, "fault_sham": 0.32, "callback_fixed": 0.2}
    private = tmp_path / "private"
    folder = private / "tag"
    folder.mkdir(parents=True)
    monkeypatch.setattr(H, "PRIVATE_ROOT", private)
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"frozen")
    document = {"bundle_key": "b" * 64, "source_commit_sha": "a" * 40,
                "protocol_sha256": "c" * 64,
                "entry": {"checkpoint_path": str(checkpoint), "action_scale": 0.25,
                          "judgment_influence": {"judgment_influence_proven": False}}}
    write_json(folder / "request.json", {"request": "bound"})
    monkeypatch.setattr(H, "load_request", lambda _tag, _isaac: (folder, document, {}, {}))
    monkeypatch.setattr(H, "bundle_consumed", lambda *_args: {"schema": "consumed",
                                                               "bundle_key": document["bundle_key"]})
    source = {"host_source_gate": "synthetic_cpu_fixture"}

    def validate(manifest_path, _isaaclab):
        manifest = H.read(manifest_path)
        arm_folder = manifest_path.parent
        return {"manifest": manifest,
                "paths": {"report_path": arm_folder / "fixed_report.json",
                          "body_path": arm_folder / "body.json",
                          "configuration_path": arm_folder / "configuration.yaml"},
                "protocol": {"device": "cuda:0"}, "protocol_sha256": document["protocol_sha256"],
                "manifest_sha256": H.W.file_sha(manifest_path),
                "project_commit_sha": document["source_commit_sha"], "source_receipt": source}

    monkeypatch.setattr(H, "validate_manifest", validate)
    launches = []

    def owned(arm_folder, *_args):
        arm = arm_folder.name
        launches.append(arm)
        write_json(arm_folder / "launch.json", {"cpu_fixture": True})
        if arm == invalid_arm:
            return {"job_assigned_before_release": True, "natural_exit": False, "exit_code": 1,
                    "cleanup_verified": True, "timed_out": False, "error_type": "synthetic_failure",
                    "owned_process_identities": {}, "wall_s": 0.1}
        validated = validate((arm_folder / "manifest.json").resolve(), tmp_path / "IsaacLab")
        body, _, _, _ = sdk_fixture(arm_folder.parent, arm, lin=metrics[arm], validated=validated)
        identity = body["process_identity"]
        return {"job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
                "cleanup_verified": True, "timed_out": False, "error_type": None,
                "owned_process_identities": {str(identity["process_id"]):
                                             identity["process_creation_time_filetime"]},
                "wall_s": 0.1}

    monkeypatch.setattr(H.P, "run_owned", owned)
    gpu_calls = []

    def gpu(_profile):
        gpu_calls.append(len(gpu_calls) + 1)
        return {"busy": len(gpu_calls) == gpu_busy_call, "processes": []}

    monkeypatch.setattr(H.M, "query_gpu", gpu)
    return folder, launches, gpu_calls


@pytest.mark.parametrize("expected,metrics", [
    ("recovery_observed", {"fault_before": 0.32, "fault_sham": 0.32, "callback_fixed": 0.2}),
    ("no_failure_observed", {"fault_before": 0.2, "fault_sham": 0.2, "callback_fixed": 0.2}),
    ("no_recovery", {"fault_before": 1.0, "fault_sham": 1.0, "callback_fixed": 0.7}),
    ("control_invalid", {"fault_before": 0.32, "fault_sham": 0.36, "callback_fixed": 0.2}),
])
def test_execute_full_saved_triplet_statuses_and_raw_artifact_inventory(
        monkeypatch, tmp_path, expected, metrics):
    folder, launches, _ = integration_fixture(monkeypatch, tmp_path, metrics)
    result = H.execute("tag", tmp_path / "IsaacLab")
    assert launches == list(H.C.ARMS) and result["status"] == expected
    assert result["execution_provenance_verified"] is True
    assert result["summary_scope"] == "numeric_conditions_only"
    assert result["summary"]["execution_provenance_verified"] is False
    assert result["recovery_proven"] is (expected == "recovery_observed")
    assert result["result_wording"] == H.FIXED["result_wording"][expected]
    assert result["cpu_verification_is_recovery_evidence"] is False
    assert result["effect_order"] == list(H.C.ARMS)
    assert set(result["arm_artifacts"]) == set(H.C.ARMS)
    for arm in H.C.ARMS:
        arm_folder = folder / "arms" / arm
        inventory = result["arm_artifacts"][arm]
        expected_files = {"parent_receipt_sha256": "parent_receipt.json", "body_sha256": "body.json",
                          "report_sha256": "fixed_report.json", "configuration_sha256": "configuration.yaml",
                          "manifest_sha256": "manifest.json"}
        assert inventory["parent_status"] == "pass" and set(inventory) == set(expected_files) | {"parent_status"}
        assert all(inventory[key] == H.W.file_sha(arm_folder / name)
                   for key, name in expected_files.items())
    assert result["sdk_jobs_launched"] == 3


@pytest.mark.parametrize("artifact", [
    "parent_receipt.json", "body.json", "fixed_report.json", "configuration.yaml", "manifest.json",
])
def test_saved_triplet_tamper_makes_final_invalid_before_summary(monkeypatch, tmp_path, artifact):
    folder, _, _ = integration_fixture(monkeypatch, tmp_path)
    original_run_effect = H.run_effect
    count = 0

    def tampering_run_effect(*args):
        nonlocal count
        result = original_run_effect(*args)
        count += 1
        if count == 3:
            target = folder / "arms/fault_before" / artifact
            target.write_bytes(target.read_bytes() + b"\n# tampered")
        return result

    monkeypatch.setattr(H, "run_effect", tampering_run_effect)
    monkeypatch.setattr(H.C, "summarize_triplet", lambda *_args: pytest.fail("tampered triplet reached summary"))
    result = H.execute("tag", tmp_path / "IsaacLab")
    assert count == 3 and result["status"] == "invalid" and not result["recovery_proven"]
    assert result["execution_provenance_verified"] is False
    assert len(result["validation_reasons"]) == 1
    assert result["validation_reasons"][0].startswith("triplet_")
    assert result["cpu_verification_is_recovery_evidence"] is False


def test_middle_effect_invalid_stops_callback_fixed(monkeypatch, tmp_path):
    _, launches, _ = integration_fixture(monkeypatch, tmp_path, invalid_arm="fault_sham")
    result = H.execute("tag", tmp_path / "IsaacLab")
    assert launches == ["fault_before", "fault_sham"]
    assert result["status"] == "invalid" and not result["recovery_proven"]
    assert result["execution_provenance_verified"] is False
    assert result["effect_order"] == ["fault_before", "fault_sham"]


def test_middle_gpu_admission_busy_stops_remaining_effects(monkeypatch, tmp_path):
    # query 1=전체 사전 gate, 2=fault_before 사전, 3=fault_before 종료, 4=fault_sham 사전.
    _, launches, gpu_calls = integration_fixture(monkeypatch, tmp_path, gpu_busy_call=4)
    result = H.execute("tag", tmp_path / "IsaacLab")
    assert launches == ["fault_before"] and gpu_calls == [1, 2, 3, 4]
    assert result["status"] == "invalid" and not result["recovery_proven"]
    assert result["execution_provenance_verified"] is False
    assert result["effect_order"] == ["fault_before"]
