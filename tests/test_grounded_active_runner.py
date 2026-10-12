import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(folder))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


W = load_module("grounded_active_model_call_runner_test", "evals/grounded_active_model_call.py")
R = load_module("run_grounded_active_runner_test", "evals/run_grounded_active.py")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def deny_external_network(monkeypatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host in {"127.0.0.1", "::1"}:
            return original_connect(self, address)
        raise AssertionError(f"external network attempted: {address!r}")

    def guarded_connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host in {"127.0.0.1", "::1"}:
            return original_connect_ex(self, address)
        raise AssertionError(f"external network attempted through connect_ex: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.mark.parametrize("events", [
    [
        {"type": "LLM_START", "id": "a"},
        {"type": "LLM_END", "id": "a", "usage": None},
    ],
    [
        {"type": "LLM_START", "id": "a"},
        {"type": "LLM_END", "id": "a", "usage": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }},
    ],
    [
        {"type": "LLM_START", "id": "a"},
        {"type": "LLM_END", "id": "foreign", "usage": {
            "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
        }},
    ],
])
def test_nat_usage_missing_zero_or_unmatched_end_is_partial_not_zero_cost(events):
    usage = W.summarize_nat_usage(events)
    assert usage["usage_coverage"] == "partial"
    assert usage["input_tokens"] is usage["output_tokens"] is usage["total_tokens"] is None
    assert usage["usage_source"] == "nat_1_9_explicit_langchain_callback_llm_end"


def test_nat_usage_rejects_duplicate_call_uuid_instead_of_collapsing_it():
    events = [
        {"type": "LLM_START", "id": "same"},
        {"type": "LLM_START", "id": "same"},
        {"type": "LLM_END", "id": "same", "usage": {
            "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
        }},
    ]
    usage = W.summarize_nat_usage(events)
    assert usage["llm_calls_started"] == 2
    assert usage["llm_call_count_coverage"] == "incomplete"
    assert usage["usage_coverage"] == "partial"
    assert usage["total_tokens"] is None


def test_nat_usage_distinguishes_not_started_from_executed_without_events():
    not_started = W.summarize_nat_usage([], execution_entered=False)
    assert not_started["llm_calls_started"] == not_started["llm_calls_finished"] == 0
    assert not_started["llm_call_count_coverage"] == "not_started"

    unavailable = W.summarize_nat_usage([], execution_entered=True)
    assert unavailable["llm_calls_started"] is unavailable["llm_calls_finished"] is None
    assert unavailable["llm_call_count_coverage"] == "unavailable"
    assert unavailable["usage_coverage"] == "partial"


def test_nat_usage_accepts_only_paired_unique_callback_ids_as_complete():
    events = [
        {"type": "LLM_START", "id": "one", "usage": None},
        {"type": "LLM_END", "id": "one", "usage": {
            "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14,
        }},
    ]
    usage = W.summarize_nat_usage(events, execution_entered=True)
    assert usage["llm_calls_started"] == usage["llm_calls_finished"] == 1
    assert usage["llm_call_count_coverage"] == usage["usage_coverage"] == "complete"
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 3, 14)


@pytest.mark.parametrize("attack", ["end_before_start", "events_before_execution"])
def test_nat_event_artifact_rejects_impossible_order_or_execution_flag(tmp_path, attack):
    events = [
        {"type": "LLM_START", "id": "one", "usage": None},
        {"type": "LLM_END", "id": "one", "usage": {
            "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14,
        }},
    ]
    entered = attack != "events_before_execution"
    if attack == "end_before_start":
        events.reverse()
    output = tmp_path / "initial.model.json"
    artifact = output.with_suffix(".events.json")
    write_json(artifact, {"schema": "grounded_active_nat_events_v1", "packet_sha256": "a" * 64,
                          "workflow_execution_entered": entered, "events": events})
    record = {"packet_sha256": "a" * 64, "llm_events_sha256": W.file_sha(artifact),
              **W.summarize_nat_usage(events, entered)}
    assert record["llm_call_count_coverage"] == "incomplete"
    assert record["total_tokens"] is None
    with pytest.raises(ValueError, match="order or execution entry"):
        W.validate_nat_event_record(output, record)


def test_nat_initial_real_callback_flow_records_one_offline_stream_call(
        monkeypatch, tmp_path):
    deny_external_network(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI

    provider_calls = []

    async def offline_stream(_self, *args, **kwargs):
        provider_calls.append((args, kwargs))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="offline-ok", id="offline-response",
                usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                chunk_position="last",
            ),
            generation_info={"finish_reason": "stop"},
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", offline_stream)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "tools.jsonl"
    trace_path.touch()
    events = []
    progress = {"workflow_execution_entered": False}

    result = asyncio.run(W.nat_initial(
        {"case_id": "offline-case"}, workspace, trace_path, events, progress,
    ))

    assert result == "offline-ok"
    assert len(provider_calls) == 1
    assert progress == {"workflow_execution_entered": True}
    assert [event["type"] for event in events] == ["LLM_START", "LLM_END"]
    assert all(set(event) == {"type", "id", "usage"} for event in events)
    assert events[0]["id"] == events[1]["id"]
    usage = W.summarize_nat_usage(events, execution_entered=True)
    assert usage["llm_calls_started"] == usage["llm_calls_finished"] == 1
    assert usage["llm_call_count_coverage"] == usage["usage_coverage"] == "complete"
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 3, 14)


@pytest.mark.parametrize("rejection", ["error", "hold"])
def test_nat_initial_semantic_tool_rejection_stops_before_second_provider_call(
        monkeypatch, tmp_path, rejection):
    deny_external_network(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI
    if rejection == "hold":
        from rl_triage import grounded_active_profile as A
        monkeypatch.setattr(A, "write_grounded_active_assessment", lambda **_args: {
            "status": "hold", "reason": "evidence_unavailable",
        })

    writer_args = {
        "case_id": "missing-case", "catalog_sha256": "a" * 64,
        "selection_binding": {
            "selection_id": "sel_" + "1" * 32, "case_id": "missing-case",
            "catalog_sha256": "a" * 64, "selected_evidence_ids": ["telemetry:x"],
            "receipt_sha256": "b" * 64,
        },
        "hypotheses": [
            {"id": "h1", "rank": 1, "narrative": "first", "premises": [{
                "evidence_id": "telemetry:x", "relation": "eq",
                "operand": {"value": 0.0, "unit": "telemetry_native"},
            }]},
            {"id": "h2", "rank": 2, "narrative": "second", "premises": [{
                "evidence_id": "telemetry:x", "relation": "gte",
                "operand": {"value": 0.0, "unit": "telemetry_native"},
            }]},
        ],
        "next_check": {
            "registry_id": "runtime_reward_window", "hypothesis_predictions": [
                {"hypothesis_id": "h1", "predictions": [
                    {"measurement_id": "reward_mean", "relation": "lt",
                     "operand": {"value": 0.0, "unit": "reward_native"}},
                    {"measurement_id": "reward_min", "relation": "lt",
                     "operand": {"value": -1.0, "unit": "reward_native"}},
                ]},
                {"hypothesis_id": "h2", "predictions": [
                    {"measurement_id": "reward_mean", "relation": "gte",
                     "operand": {"value": 0.0, "unit": "reward_native"}},
                    {"measurement_id": "reward_min", "relation": "gte",
                     "operand": {"value": -1.0, "unit": "reward_native"}},
                ]},
            ],
        },
    }
    provider_calls = []

    async def rejected_writer_stream(_self, *args, **kwargs):
        provider_calls.append((args, kwargs))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", id="offline-tool-call",
                tool_call_chunks=[{
                    "name": "write_grounded_active_assessment",
                    "args": json.dumps(writer_args), "id": "writer-1", "index": 0,
                    "type": "tool_call_chunk",
                }],
                usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                chunk_position="last",
            ),
            generation_info={"finish_reason": "tool_calls"},
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", rejected_writer_stream)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "tools.jsonl"
    trace_path.touch()
    events = []
    progress = {"workflow_execution_entered": False}

    with pytest.raises(ValueError, match="grounded active assessment rejected"):
        asyncio.run(W.nat_initial(
            {"case_id": "missing-case"}, workspace, trace_path, events, progress,
        ))

    assert len(provider_calls) == 1
    assert progress == {"workflow_execution_entered": True}
    assert [event["type"] for event in events] == ["LLM_START", "LLM_END"]
    assert all(set(event) == {"type", "id", "usage"} for event in events)
    assert events[0]["id"] == events[1]["id"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["tool"] == "write_grounded_active_assessment"
    returned = json.loads(rows[0]["result_head"])
    assert ("error" in returned) if rejection == "error" else returned["status"] == "hold"
    assert not (workspace / "preregistrations/grounded_active_assessments/missing-case.json").exists()


def trace(path: Path, tools: list[tuple[str, str]]) -> None:
    path.write_text("".join(json.dumps({"tool": tool, "result_head": result}) + "\n"
                            for tool, result in tools), encoding="utf-8")


def valid_trace_rows() -> list[tuple[str, str]]:
    return [
        ("grounded_active_registry", "{}"),
        ("grounded_evidence_group_index", "{}"),
        ("grounded_evidence_group_page", "{}"),
        ("grounded_evidence_details", "{}"),
        ("write_grounded_active_assessment", "{}"),
    ]


def test_tool_trace_enforces_budget_and_semantic_rejection(tmp_path):
    valid_path = tmp_path / "valid.jsonl"
    trace(valid_path, valid_trace_rows())
    valid = W.validate_tool_trace(valid_path)
    assert valid["budget_valid"] is True
    assert valid["semantic_rejection_observed"] is False

    over_path = tmp_path / "over.jsonl"
    trace(over_path, valid_trace_rows() + [
        ("grounded_evidence_group_page", "{}"),
        ("grounded_evidence_group_page", "{}"),
    ])
    assert W.validate_tool_trace(over_path)["budget_valid"] is False

    rejected_path = tmp_path / "rejected.jsonl"
    rows = valid_trace_rows()
    rows[-1] = (rows[-1][0], '{"error":"semantic rejection"}')
    trace(rejected_path, rows)
    rejected = W.validate_tool_trace(rejected_path)
    assert rejected["semantic_rejection_observed"] is True


def test_active_workflow_stops_on_tool_semantic_error_instead_of_retrying():
    config = yaml.safe_load(
        (ROOT / "configs/triage_single_run_grounded_active.yml").read_text(encoding="utf-8"))
    assert config["workflow"]["handle_tool_errors"] is False
    assert config["workflow"]["truncation_retry"]["max_retries"] == 0


def test_worker_setup_failure_records_stage_and_code_location_without_exception_content(
        monkeypatch, tmp_path):
    from rl_triage import grounded_evidence as G, triage_tools as T

    config = tmp_path / "workflow.yml"
    config.write_text("offline: true\n", encoding="utf-8")
    monkeypatch.setattr(W, "CONFIG", config)
    monkeypatch.setattr(W, "os", SimpleNamespace(
        name="posix", environ={"NVIDIA_API_KEY": "offline-unused-placeholder"}))
    monkeypatch.setattr(W, "version", lambda _name: "1.9.0")
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    monkeypatch.setattr(G, "evidence_catalog", lambda _case: {"catalog_sha256": "c" * 64})
    invocations = []

    async def workflow_setup_failure(*_args):
        invocations.append("setup")
        raise ValueError("sensitive-exception-content")

    monkeypatch.setattr(W, "nat_initial", workflow_setup_failure)
    packet_path, output = tmp_path / "initial.packet.json", tmp_path / "initial.model.json"
    write_json(packet_path, {
        "schema": "grounded_active_initial_packet_v1", "case_id": "case",
        "catalog_sha256": "c" * 64,
        "workflow_config_sha256_lf": hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    })
    assert W.call("initial", packet_path, tmp_path, output) == 2
    raw = output.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert invocations == ["setup"]
    assert record["valid"] is False and record["error_type"] == "ValueError"
    assert record["error_stage"] == "initial_workflow"
    assert record["llm_calls_started"] == record["llm_calls_finished"] == 0
    assert record["error_frames"][-1]["function"] == "workflow_setup_failure"
    assert all(frame["line"] > 0 and frame["file"] == Path(frame["file"]).name
               for frame in record["error_frames"])
    assert "sensitive-exception-content" not in raw
    assert "offline-unused-placeholder" not in raw and str(tmp_path) not in raw


def packet(value: float | None) -> dict:
    return {
        "schema": "grounded_active_post_packet_v1", "case_id": "case_public",
        "effect_key": "e" * 64, "budget_left": 0,
        "normalized_receipt_sha256": "r" * 64,
        "hypotheses": [{"hypothesis_id": "h1", "preregistered_predictions": []}],
        "selected_evidence_ids": [], "selected_evidence": [], "check_definition": {},
        "normalized_receipt": {"measurements": [{
            "measurement_id": "base_com_max_abs_delta_xy",
            "unit": "simulation_length_native", "state": "present", "value": value,
        }]},
    }


def test_actual_and_masked_post_messages_share_instructions_and_only_mask_values():
    actual, masked = packet(0.09), packet(None)
    actual_messages, masked_messages = W.post_messages(actual), W.post_messages(masked)
    assert actual_messages[0] == masked_messages[0]
    actual_payload = json.loads(actual_messages[1]["content"])
    masked_payload = json.loads(masked_messages[1]["content"])
    actual_value = actual_payload["normalized_receipt"]["measurements"][0].pop("value")
    masked_value = masked_payload["normalized_receipt"]["measurements"][0].pop("value")
    assert actual_payload == masked_payload
    assert actual_value == 0.09 and masked_value is None
    assert actual_payload["normalized_receipt_sha256"] == "r" * 64


def assessment() -> dict:
    return {
        "hypotheses": [{"id": "h1", "rank": 1}, {"id": "h2", "rank": 2}],
        "next_check": {"hypothesis_predictions": [
            {"hypothesis_id": "h1", "predictions": []},
            {"hypothesis_id": "h2", "predictions": []},
        ]},
    }


def base_execute(monkeypatch, tmp_path: Path):
    folder = tmp_path / "run"
    folder.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = {
        "case_id": "case_public", "arm": "X1", "source_commit_sha": "a" * 40,
        "workspace": str(workspace), "source_seed": 7, "probe_seed": 20261012,
        "source_relationship": "separate_reset_mechanism_replay_not_original_training_run",
        "source_task_and_seed_attestation": "original_training_metadata_and_configuration_not_original_code",
    }
    binding = {"case_source_binding_sha256": "b" * 64}
    approval = {
        "effect_key": "e" * 64, "request_sha256": "1" * 64,
        "approval_sha256": "2" * 64, "case_source_binding_sha256": "b" * 64,
    }
    write_json(folder / "approval_binding.json", approval)
    monkeypatch.setattr(R, "load_run", lambda _tag: (folder, spec, binding))
    monkeypatch.setattr(R.B.M, "query_gpu", lambda _profile: {"busy": False})
    return folder, workspace, spec, binding, approval


def consumed_record() -> dict:
    return {
        "effect_key": "e" * 64, "request_sha256": "1" * 64,
        "approval_sha256": "2" * 64, "case_source_binding_sha256": "b" * 64,
        "consumed_sha256": "3" * 64,
    }


def consumed_active_ledger(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "active-workspace"
    case_id = "case_public"
    case = workspace / "cases" / case_id
    sources = {}
    for name, value in {
        "telemetry": {"summary": {}, "series": {}},
        "configuration": {"task": "go2"},
        "context": {"status": "provided"},
    }.items():
        path = case / f"{name}.json"
        write_json(path, value)
        sources[name] = {"path": f"cases/{case_id}/{name}.json",
                         "sha256": W.file_sha(path)}
    monkeypatch.setattr(R.T, "WORKSPACE", workspace)
    assessment_document = {
        "schema": "grounded_active_assessment_v1", "case_id": case_id,
        "registry_id": R.A.ACTIVE_REGISTRY_ID,
        "registry_sha256": R.A._sha(R.A.ACTIVE_CHECKS),
        "next_check": {"registry_id": "reset_base_com_stability"},
    }
    R.A._write_once("grounded_active_assessments", f"{case_id}.json", assessment_document)
    manifest = {
        "schema": "grounded_active_case_manifest_v1", "case_id": case_id,
        "public_case_alias": case_id, "sources": sources,
        "assessment_sha256": R.A._sha(assessment_document), "task": R.B.TASK, "seed": 7,
        "protocol_sha256": "a" * 64, "source_commit_sha": "b" * 40,
        "private_executor_binding_sha256": "c" * 64,
    }
    binding = R.A.validate_case_manifest(case_id, manifest)
    request = R.A.request_execution(
        case_id, binding["case_source_binding_sha256"],
        "reset_base_com_stability", deepcopy(R.A.FIXED_EXECUTION_ARGUMENTS),
    )
    approval = R.A.approve_execution(case_id, request, "operator")
    R.A.consume_approval(case_id, approval)
    return workspace, case_id, binding, approval


def invalid_normalized() -> dict:
    return {
        "execution_valid": False, "normalized_receipt_sha256": "4" * 64,
        "measurements": [
            {"measurement_id": "base_com_max_abs_delta_xy", "value": None},
            {"measurement_id": "base_com_outside_range_fraction", "value": None},
        ],
    }


def test_existing_consumed_ledger_without_result_blocks_sdk_reexecution(monkeypatch, tmp_path):
    workspace, case_id, binding, approval = consumed_active_ledger(monkeypatch, tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    write_json(folder / "approval_binding.json", approval)
    spec = {
        "case_id": case_id, "arm": "X1", "source_commit_sha": "b" * 40,
        "workspace": str(workspace), "source_seed": 7, "probe_seed": 20261012,
        "source_relationship": "separate_reset_mechanism_replay_not_original_training_run",
        "source_task_and_seed_attestation": "original_training_metadata_and_configuration_not_original_code",
    }
    monkeypatch.setattr(R, "load_run", lambda _tag: (folder, spec, binding))
    monkeypatch.setattr(R.B.M, "query_gpu", lambda _profile: {"busy": False})
    monkeypatch.setattr(R.B, "run", lambda *_args: pytest.fail("소비된 effect SDK 재실행"))
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        R.execute("one", tmp_path / "IsaacLab")


def test_execute_consumes_before_sdk_and_invalid_sdk_never_calls_post_model(monkeypatch, tmp_path):
    folder, _, _, _, _ = base_execute(monkeypatch, tmp_path)
    order = []
    monkeypatch.setattr(R.A, "consume_approval", lambda *_args: order.append("consume") or consumed_record())

    def failed_sdk(*_args):
        order.append("sdk")
        raise RuntimeError("invalid SDK result")

    monkeypatch.setattr(R.B, "run", failed_sdk)
    monkeypatch.setattr(R.A, "normalize_execution_receipt", lambda *_args: invalid_normalized())
    monkeypatch.setattr(R.A, "_load_bound", lambda *_args: assessment())
    monkeypatch.setattr(R, "fresh_model", lambda *_args: pytest.fail("invalid SDK 뒤 post RPC 호출"))
    result = R.execute("one", tmp_path / "IsaacLab")
    assert order == ["consume", "sdk"]
    assert result["execution_valid"] is False and result["post_model_calls"] == 0
    assert (folder / "consumed_binding.json").is_file()


def test_saved_sdk_parent_receipt_must_equal_returned_receipt_before_post_model(
        monkeypatch, tmp_path):
    folder, _, spec, _, _ = base_execute(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R.A, "consume_approval", lambda *_args: consumed_record())

    def mismatched_sdk(tag, arm, _isaaclab):
        sdk = tmp_path / "bench/private/base_com_reset" / tag
        write_json(sdk / "parent_receipt.json", {"status": "different"})
        return {
            "status": "pass", "arm": arm, "project_commit_sha": spec["source_commit_sha"],
            "protocol_sha256": "p" * 64, "body_sha256": "d" * 64,
            "manifest_sha256": "m" * 64,
            "execution": {"natural_exit": True, "cleanup_verified": True},
        }

    monkeypatch.setattr(R.B, "run", mismatched_sdk)
    monkeypatch.setattr(R.A, "normalize_execution_receipt", lambda *_args: invalid_normalized())
    monkeypatch.setattr(R.A, "_load_bound", lambda *_args: assessment())
    monkeypatch.setattr(R, "fresh_model", lambda *_args: pytest.fail("parent mismatch 뒤 post RPC 호출"))
    result = R.execute("one", tmp_path / "IsaacLab")
    failure = json.loads((folder / "execution_failure.json").read_text(encoding="utf-8"))
    assert failure == {"error_type": "ValueError", "side_effect_retry": False}
    assert result["execution_valid"] is False and result["post_model_calls"] == 0


def test_finished_run_refuses_reexecution_before_gpu_or_consume(monkeypatch, tmp_path):
    folder, _, _, _, _ = base_execute(monkeypatch, tmp_path)
    write_json(folder / "result.json", {"status": "HOLD"})
    monkeypatch.setattr(R.B.M, "query_gpu", lambda *_args: pytest.fail("재실행 GPU gate 호출"))
    monkeypatch.setattr(R.A, "consume_approval", lambda *_args: pytest.fail("재승인 소비"))
    with pytest.raises(FileExistsError, match="never overwritten"):
        R.execute("one", tmp_path / "IsaacLab")


@pytest.mark.parametrize("pair_case", [
    "duplicate_id", "different_model", "empty_model", "incomplete_finished",
])
def test_invalid_post_provider_pairs_are_not_accepted_or_given_extra_gpu_run(
        monkeypatch, tmp_path, pair_case):
    folder, _, spec, _, _ = base_execute(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R.A, "consume_approval", lambda *_args: consumed_record())

    def sdk_run(tag, arm, _isaaclab):
        sdk = tmp_path / "bench/private/base_com_reset" / tag
        write_json(sdk / "manifest.json", {"ok": True})
        body = {"observer_report": {"measurements": {
            "base_com_max_abs_delta_xy": {"value": 0.09},
            "base_com_outside_range_fraction": {"value": 0.5},
        }}}
        write_json(sdk / "body.json", body)
        parent = {
            "status": "pass", "arm": arm, "project_commit_sha": spec["source_commit_sha"],
            "protocol_sha256": W.file_sha(R.B.PROTOCOL),
            "body_sha256": W.file_sha(sdk / "body.json"),
            "manifest_sha256": W.file_sha(sdk / "manifest.json"),
            "execution": {"natural_exit": True, "cleanup_verified": True},
        }
        write_json(sdk / "parent_receipt.json", parent)
        return parent

    monkeypatch.setattr(R.B, "run", sdk_run)
    monkeypatch.setattr(R.B, "validate_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(R.B, "verify_body", lambda *_args: [])
    normalized = {
        "execution_valid": True, "normalized_receipt_sha256": "4" * 64,
        "measurements": [
            {"measurement_id": "base_com_max_abs_delta_xy", "value": 0.09},
            {"measurement_id": "base_com_outside_range_fraction", "value": 0.5},
        ],
    }
    monkeypatch.setattr(R.A, "normalize_execution_receipt", lambda *_args: normalized)
    monkeypatch.setattr(R.A, "_load_bound", lambda *_args: assessment())
    monkeypatch.setattr(
        R.A, "build_post_measurement_packet",
        lambda _case, _receipt, value_masked=False: packet(None if value_masked else 0.09),
    )
    calls = []

    def model(_folder, stage, sent, _workspace):
        calls.append((stage, sent["normalized_receipt"]["measurements"][0]["value"]))
        identity = "same" if pair_case == "duplicate_id" else stage
        provider_model = "model-a"
        if pair_case == "different_model" and stage == "masked":
            provider_model = "model-b"
        if pair_case == "empty_model" and stage == "actual":
            provider_model = ""
        finished = 0 if pair_case == "incomplete_finished" and stage == "actual" else 1
        return {"outcome": {"valid": True, "worker_cleanup_verified": True},
                "record": {"provider_response_id": identity, "provider_model": provider_model,
                           "llm_calls_started": 1, "llm_calls_finished": finished,
                           "max_output_tokens": W.POST_MAX_TOKENS,
                           "reasoning_effort": W.POST_REASONING_EFFORT,
                           "provider_cache_status": "not_attested", "decision": {}}}

    monkeypatch.setattr(R, "fresh_model", model)
    monkeypatch.setattr(R.A, "build_judgment_influence_proof",
                        lambda *_args: pytest.fail("중복 provider ID pair proof 호출"))
    result = R.execute("one", tmp_path / "IsaacLab")
    assert len(calls) == 2 and {value for _, value in calls} == {0.09, None}
    assert result["valid_post_model_pair"] is False
    assert result["judgment_influence"]["reason"] == "invalid_model_pair"
    assert result["controls"]["additional_gpu_runs"] == 0
    assert result["controls"]["receipt_sha256"] == R.A._sha({
        key: value for key, value in normalized.items() if key != "normalized_receipt_sha256"
    })


def test_timed_out_model_without_owned_pid_proof_is_not_cleanup_verified(monkeypatch, tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(R, "wsl_path", lambda path: str(path.resolve()))

    def timeout(*_args, **_kwargs):
        raise R.subprocess.TimeoutExpired(cmd="worker", timeout=1)

    monkeypatch.setattr(R.subprocess, "run", timeout)
    result = R.fresh_model(folder, "actual", packet(0.09), workspace)
    assert result["record"] is None
    assert result["outcome"]["valid"] is False
    assert result["outcome"]["worker_cleanup_verified"] is False
    assert result["outcome"]["launch"]["timed_out"] is True


@pytest.mark.parametrize("field,value", [
    ("max_output_tokens", 1),
    ("reasoning_effort", "high"),
])
def test_fresh_model_rejects_wrong_post_budget_or_reasoning(
        monkeypatch, tmp_path, field, value):
    folder = tmp_path / "model"
    folder.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sent = packet(0.09)
    monkeypatch.setattr(R, "wsl_path", lambda path: str(path.resolve()))

    def completed(*_args, **_kwargs):
        packet_path = folder / "actual.packet.json"
        record = {
            "schema": "grounded_active_model_call_v1", "stage": "actual", "valid": True,
            "packet_sha256": W.file_sha(packet_path), "model": W.MODEL, "temperature": 0,
            "semantic_retries": 0, "transport_retries": 0, "client_cache_used": False,
            "saved_response_cached": False, "provider_cache_status": "not_attested",
            "messages_sha256": W.canonical_sha(W.post_messages(sent)),
            "provider_response_id": "response-a", "provider_model": W.MODEL,
            "llm_calls_started": 1, "llm_calls_finished": 1,
            "max_output_tokens": W.POST_MAX_TOKENS,
            "reasoning_effort": W.POST_REASONING_EFFORT,
        }
        record[field] = value
        write_json(folder / "actual.model.json", record)
        return R.subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(R.subprocess, "run", completed)
    result = R.fresh_model(folder, "actual", sent, workspace)
    assert result["record"][field] == value
    assert result["outcome"]["valid"] is False


@pytest.mark.parametrize("mutation", [
    None, "too_many_calls", "budget_summary", "semantic_summary", "trace_tamper", "budget_overrun",
])
def test_fresh_initial_parent_reaggregates_events_and_tool_trace_before_accepting(
        monkeypatch, tmp_path, mutation):
    folder, workspace = tmp_path / "model", tmp_path / "workspace"
    folder.mkdir()
    workspace.mkdir()
    sent = {"schema": "grounded_active_initial_packet_v1", "case_id": "case_public"}
    monkeypatch.setattr(R, "wsl_path", lambda path: str(path.resolve()))

    def completed(*_args, **_kwargs):
        output = folder / "initial.model.json"
        events = []
        for index in range(8 if mutation == "too_many_calls" else 1):
            events.extend([
                {"type": "LLM_START", "id": str(index), "usage": None},
                {"type": "LLM_END", "id": str(index), "usage": {
                    "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14,
                }},
            ])
        event_path = output.with_suffix(".events.json")
        packet_sha = W.file_sha(folder / "initial.packet.json")
        write_json(event_path, {"schema": "grounded_active_nat_events_v1", "packet_sha256": packet_sha,
                               "workflow_execution_entered": True, "events": events})
        trace_path = output.with_suffix(".trace.jsonl")
        rows = valid_trace_rows()
        if mutation == "budget_overrun":
            rows.extend([("grounded_evidence_group_page", "{}") for _ in range(2)])
        trace(trace_path, rows)
        record = {"schema": "grounded_active_model_call_v1", "stage": "initial", "valid": True,
                  "packet_sha256": packet_sha, "model": W.MODEL, "temperature": 0,
                  "semantic_retries": 0, "transport_retries": 0, "client_cache_used": False,
                  "saved_response_cached": False, "llm_events_sha256": W.file_sha(event_path),
                  **W.summarize_nat_usage(events, True), **W.validate_tool_trace(trace_path)}
        if mutation == "budget_summary":
            record["budget_valid"] = False
        elif mutation == "semantic_summary":
            record["semantic_rejection_observed"] = True
        elif mutation == "trace_tamper":
            trace(trace_path, rows + [("grounded_evidence_group_page", "{}")])
        write_json(output, record)
        return R.subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(R.subprocess, "run", completed)
    result = R.fresh_model(folder, "initial", sent, workspace)
    assert result["outcome"]["valid"] is (mutation is None)
    assert result["outcome"]["record_error"] == (
        None if mutation is None else "invalid_initial_execution_artifact")


def model_cost_record(input_tokens, output_tokens, *, valid=True):
    total = None if input_tokens is None or output_tokens is None else input_tokens + output_tokens
    return {
        "valid": valid, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total, "wall_s": 2.0, "cpu_s": 0.2,
        "llm_calls_started": 1, "llm_calls_finished": 1,
    }


def test_cost_summary_keeps_tokens_null_when_stage_or_nat_usage_is_missing(tmp_path):
    folder = tmp_path / "cost"
    folder.mkdir()
    write_json(folder / "initial.model.json", model_cost_record(None, None))
    actual = {"record": model_cost_record(10, 2)}
    missing_stage = R.cost_summary(folder, {"actual": actual}, {"execution": {"wall_s": 12.5}})
    assert missing_stage["usage_coverage"] == "partial"
    assert missing_stage["input_tokens"] is missing_stage["total_tokens"] is None
    assert missing_stage["model_stages"]["masked"]["record_status"] == "absent_or_invalid"
    assert missing_stage["sdk_parent_wall_s"] == 12.5
    assert missing_stage["sdk_gpu_time_s"] is None
    assert missing_stage["gpu_time_status"] == "not_measured"


def test_cost_summary_aggregates_tokens_only_when_all_three_stages_are_complete(tmp_path):
    folder = tmp_path / "cost"
    folder.mkdir()
    write_json(folder / "initial.model.json", model_cost_record(100, 20))
    calls = {
        "actual": {"record": model_cost_record(30, 4)},
        "masked": {"record": model_cost_record(31, 5)},
    }
    cost = R.cost_summary(folder, calls, {"execution": {"wall_s": 9.0}})
    assert cost["usage_coverage"] == "complete"
    assert (cost["input_tokens"], cost["output_tokens"], cost["total_tokens"]) == (161, 29, 190)
    assert cost["model_stages"]["initial"]["wall_s"] == 2.0
    assert cost["model_stages"]["actual"]["cpu_s"] == 0.2
    assert cost["sdk_parent_wall_s"] == 9.0 and cost["sdk_gpu_time_s"] is None
    assert cost["controls_additional_model_calls"] == cost["controls_additional_gpu_runs"] == 0


def lineage_fixture(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    case_id = "case_public"
    case = workspace / "cases" / case_id
    case.mkdir(parents=True)
    metadata_path = tmp_path / "source.meta.json"
    telemetry = tmp_path / "source.telemetry.json"
    configuration = tmp_path / "env.yaml"
    metadata = {"task": R.B.TASK, "seed": 7, "num_envs": 4096, "fault": "X1"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    telemetry.write_text('{"summary":{},"series":{}}', encoding="utf-8")
    source = {"seed": 7, "scene": {"num_envs": 4096}, "events": {}}
    configuration.write_text("seed: 7\nscene:\n  num_envs: 4096\nevents: {}\n", encoding="utf-8")
    derived, facts = R.neutral_configuration(metadata, source)
    (case / "telemetry.json").write_bytes(telemetry.read_bytes())
    write_json(case / "configuration.json", derived)
    write_json(case / "context.json", {"facts": facts})
    monkeypatch.setattr(R, "original_source_metadata",
                        lambda _path: (metadata, telemetry, configuration))
    lineage = {
        "schema": "grounded_active_input_lineage_v1", "case_id": case_id,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": W.file_sha(metadata_path),
        "original_telemetry_path": str(telemetry),
        "original_telemetry_sha256": W.file_sha(telemetry),
        "original_configuration_path": str(configuration),
        "original_configuration_sha256": W.file_sha(configuration),
        "derived_configuration_sha256": W.file_sha(case / "configuration.json"),
        "case_input_context_sha256": W.file_sha(case / "context.json"),
        "input_source_seed": 7, "probe_seed": 20261012,
        "original_source_code_attestation": "not_recorded_by_legacy_training_harness",
        "source_relationship": R.FIXED["case_and_probe_relationship"],
        "policy_recovery_verified": False,
    }
    write_json(workspace / "host_input_lineage.json", lineage)
    return workspace, case_id, metadata_path, lineage, telemetry, configuration


@pytest.mark.parametrize("field,value", [
    ("input_source_seed", 20261012),
    ("probe_seed", 7),
])
def test_input_lineage_keeps_source_seed_distinct_from_probe_seed_and_rejects_tamper(
        monkeypatch, tmp_path, field, value):
    workspace, case_id, metadata_path, lineage, _, _ = lineage_fixture(monkeypatch, tmp_path)
    assert R.validate_input_lineage(workspace, case_id, metadata_path)["seed"] == 7
    changed = deepcopy(lineage)
    changed[field] = value
    write_json(workspace / "host_input_lineage.json", changed)
    with pytest.raises(ValueError, match="seed|lineage"):
        R.validate_input_lineage(workspace, case_id, metadata_path)


@pytest.mark.parametrize("source_name", ["telemetry", "configuration"])
def test_input_lineage_rejects_original_telemetry_or_configuration_mutation(
        monkeypatch, tmp_path, source_name):
    workspace, case_id, metadata_path, _, telemetry, configuration = lineage_fixture(
        monkeypatch, tmp_path)
    target = telemetry if source_name == "telemetry" else configuration
    target.write_bytes(target.read_bytes() + b"\n# changed")
    with pytest.raises(ValueError, match="source|binding"):
        R.validate_input_lineage(workspace, case_id, metadata_path)
