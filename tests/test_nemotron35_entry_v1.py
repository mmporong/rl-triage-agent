import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from rl_triage import nemotron35_entry_v1 as E

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "nvapi-synthetic-cpu-test-key-0123456789"
CHECKPOINT_MARKER = b"SYNTHETIC-CPU-CHECKPOINT-BODY"
RETURNED_MODEL = "private/nvidia/nemotron-3.5-super-120b-a12b"
EVIDENCE_ID = "telemetry.summary::Train/mean_reward::last"
# synthetic_cpu fixture: 실제 학습 로그가 아니다.
TELEMETRY = {
    "run_dir_name": "synthetic_cpu_run",
    "summary": {"Train/mean_reward": {"n": 3, "first": 1.0, "last": 0.5, "min": 0.5, "max": 1.0,
                                      "mean_first_20pct": 1.0, "mean_last_20pct": 0.5}},
    "series": {"Train/mean_reward": [1.0, 0.7, 0.5]},
}
TASK_CONTEXT = {"schema": "nemotron35_task_context_v1", "task": "Isaac-Velocity-Rough-H1-v0",
                "observed_problem": "synthetic_cpu: reward drops late in training"}


def _write(path, value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path, telemetry=TELEMETRY, task=TASK_CONTEXT):
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "telemetry.json", telemetry)
    _write(run / "model_99.pt", CHECKPOINT_MARKER)
    _write(tmp_path / "task.json", task)
    return run, tmp_path / "task.json"


def _args(run, task, out, mode="--prepare-only"):
    return ["--run-dir", str(run), "--task-context", str(task), "--out", str(out), mode]


def _deny_external_network(monkeypatch):
    # Windows asyncio의 self-pipe는 loopback socketpair를 쓰므로 loopback만 허용한다.
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original):
        def connect(self, address):
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return original(self, address)
            raise AssertionError(f"external network attempted: {address!r}")
        return connect

    monkeypatch.setattr(socket.socket, "connect", guarded(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(original_connect_ex))


def _env_without_key():
    env = {key: value for key, value in os.environ.items() if key != E.KEY_ENV}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    return env


def _all_bytes(folder):
    return b"".join(path.read_bytes() for path in folder.rglob("*") if path.is_file())


def test_help_and_import_from_other_cwd_load_no_api_or_runtime(tmp_path):
    help_run = subprocess.run([sys.executable, "-m", "rl_triage.nemotron35_entry_v1", "--help"],
                              cwd=tmp_path, env=_env_without_key(), capture_output=True, text=True, timeout=60,
                              check=False)
    assert help_run.returncode == 0
    assert "--prepare-only" in help_run.stdout and "--execute" in help_run.stdout
    probe = ("import json, sys, rl_triage.nemotron35_entry_v1; "
             "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & "
             "{'nat', 'openai', 'langchain', 'langchain_openai', 'httpx', 'torch', 'isaaclab', 'omni'})))")
    imported = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=_env_without_key(),
                              capture_output=True, text=True, timeout=60, check=False)
    assert imported.returncode == 0 and json.loads(imported.stdout) == []


def test_prepare_only_from_other_cwd_accepts_minimal_input(tmp_path):
    _inputs(tmp_path)
    work = tmp_path / "elsewhere"
    work.mkdir()
    completed = subprocess.run(
        [sys.executable, "-m", "rl_triage.nemotron35_entry_v1",
         *_args(Path("../run"), Path("../task.json"), Path("out"))],
        cwd=work, env=_env_without_key(), capture_output=True, text=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {"status": "prepared", "out": str((work / "out").resolve()), "ready_for_execute": False,
                       "dependencies_ok": True, "config_ok": True, "credential_present": False,
                       "model_calls": 0}
    out = work / "out"
    reference = json.loads((out / "workspace/reference/telemetry.json").read_text(encoding="utf-8"))
    assert reference == E.ABSENT_REFERENCE
    assert sorted(path.name for path in (out / "workspace/cases/user_run").iterdir()) == ["telemetry.json"]
    report = json.loads((out / "entry_report.json").read_text(encoding="utf-8"))
    assert report["unused_run_dir_entries"] == ["model_99.pt"]
    assert report["config"]["nat_load_config"] == "ok" and report["config"]["nat_model_name"] == E.MODEL
    assert report["check_execution"] == {"performed": False, "gpu_runs": 0}
    assert report["claims"]["healthy_reference"] == "absent" and report["claims"]["cause_confirmed"] is False
    packet = json.loads((out / "entry_packet.json").read_text(encoding="utf-8"))
    assert TASK_CONTEXT["observed_problem"] in packet["user_message"]
    assert packet["inputs"]["telemetry.json"]["sha256"] == hashlib.sha256(
        (tmp_path / "run/telemetry.json").read_bytes()).hexdigest()
    assert CHECKPOINT_MARKER not in _all_bytes(out)


def test_prepare_only_reports_key_presence_without_value_or_network(tmp_path, monkeypatch, capsys):
    run, task = _inputs(tmp_path)
    monkeypatch.setenv(E.KEY_ENV, FAKE_KEY)
    _deny_external_network(monkeypatch)
    assert E.main(_args(run, task, tmp_path / "out")) == 0
    printed = capsys.readouterr()
    summary = json.loads(printed.out)
    assert summary["ready_for_execute"] is True and summary["model_calls"] == 0
    report = json.loads((tmp_path / "out/entry_report.json").read_text(encoding="utf-8"))
    assert report["credential"] == {"env_var": E.KEY_ENV, "present": True, "value_recorded": False}
    assert FAKE_KEY not in printed.out + printed.err
    assert FAKE_KEY.encode() not in _all_bytes(tmp_path / "out")


def test_prepare_binds_selected_configuration_through_existing_context_contract(tmp_path, monkeypatch):
    run, task = _inputs(tmp_path)
    configuration = {"rewards": {"track_lin_vel_xy_exp": {"weight": 1.5}}}
    context = {"schema": "single_run_context_v1",
               "telemetry_sha256": hashlib.sha256((run / "telemetry.json").read_bytes()).hexdigest(),
               "configuration_sha256": _write(run / "configuration.json", configuration),
               "facts": [{"id": "track_weight", "path": ["rewards", "track_lin_vel_xy_exp", "weight"]}]}
    _write(run / "context.json", context)
    monkeypatch.delenv(E.KEY_ENV, raising=False)
    prepared = E.prepare(run, task, tmp_path / "out", "prepare_only")
    assert prepared["report"]["catalog"]["context_status"] == "provided"
    assert set(prepared["packet"]["inputs"]) == {"telemetry.json", "configuration.json", "context.json"}


@pytest.mark.parametrize("telemetry, message", [
    (b"{not json", "Expecting"),
    (b'{"summary": {}, "summary": {}, "series": {}}', "중복 JSON 키"),
    (b'{"summary": {"a": {"n": 1, "last": NaN}}, "series": {}}', "유한하지 않은"),
    ({"summary": [], "series": {}}, "summary와 series"),
    ({"summary": {"Train/mean_reward": {"n": 1, "last": 1.0}}, "series": {"Train/mean_reward": ["x"]}}, "series"),
])
def test_damaged_telemetry_is_rejected_without_leaving_output(tmp_path, capsys, telemetry, message):
    run, task = _inputs(tmp_path, telemetry=telemetry)
    assert E.main(_args(run, task, tmp_path / "out")) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rejected" and message in result["error"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run", "task.json"]


def test_missing_telemetry_points_to_existing_extractor(tmp_path, capsys):
    run, task = _inputs(tmp_path)
    (run / "telemetry.json").unlink()
    (run / "events.out.tfevents.1.synthetic").write_bytes(b"")
    assert E.main(_args(run, task, tmp_path / "out")) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "telemetry.json" in error and "bench/extract_tb.py" in error


def test_context_without_configuration_is_rejected(tmp_path, capsys):
    run, task = _inputs(tmp_path)
    _write(run / "context.json", {"schema": "single_run_context_v1", "telemetry_sha256": "0" * 64,
                                  "configuration_sha256": "0" * 64, "facts": []})
    assert E.main(_args(run, task, tmp_path / "out")) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("task", [
    {key: value for key, value in TASK_CONTEXT.items() if key != "observed_problem"},
    {**TASK_CONTEXT, "healthy_reference": "baseline_run"},
    {**TASK_CONTEXT, "answer_key": "reward"},
    {**TASK_CONTEXT, "schema": "other"},
    {**TASK_CONTEXT, "task": "../escape"},
    {**TASK_CONTEXT, "observed_problem": "  "},
    {**TASK_CONTEXT, "observed_problem": "x" * (E.OBSERVED_PROBLEM_MAX_CHARS + 1)},
])
def test_task_context_contract_rejects_missing_or_invented_fields(tmp_path, capsys, task):
    run, task_path = _inputs(tmp_path, task=task)
    assert E.main(_args(run, task_path, tmp_path / "out")) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    assert not (tmp_path / "out").exists()


def test_output_paths_cannot_overlap_or_overwrite(tmp_path, capsys):
    run, task = _inputs(tmp_path)
    assert E.main(_args(run, task, run / "out")) == 2
    assert E.main(_args(run, task, run)) == 2
    (tmp_path / "taken").mkdir()
    assert E.main(_args(run, task, tmp_path / "taken")) == 2
    assert E.main(_args(tmp_path / "absent", task, tmp_path / "out")) == 2
    assert all(json.loads(line)["status"] == "rejected" for line in capsys.readouterr().out.splitlines())
    assert sorted(path.name for path in run.iterdir()) == ["model_99.pt", "telemetry.json"]


def test_symlinked_inputs_cannot_escape_run_dir(tmp_path, capsys):
    run, task = _inputs(tmp_path)
    outside = tmp_path / "outside.json"
    _write(outside, TELEMETRY)
    (run / "telemetry.json").unlink()
    try:
        (run / "telemetry.json").symlink_to(outside)
        (tmp_path / "task_link.json").symlink_to(task)
    except OSError:
        pytest.skip("이 OS 계정은 심볼릭 링크를 만들 수 없다")
    assert E.main(_args(run, task, tmp_path / "out")) == 2
    (run / "telemetry.json").unlink()
    _write(run / "telemetry.json", TELEMETRY)
    assert E.main(_args(run, tmp_path / "task_link.json", tmp_path / "out")) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("old, new", [
    ("    max_retries: 0\n", "    max_retries: 0\n    stop: ['</done>']\n"),
    ("    max_retries: 0\n", "    max_retries: 0\n    frequency_penalty: 0.1\n"),
    ("do_auto_retry: false", "do_auto_retry: true"),
    ("nemotron-3.5-super-vl-120b-a12b", "nemotron-3-super-120b-a12b"),
])
def test_config_contract_rejects_unregistered_request_settings(tmp_path, old, new):
    text = E.CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.yml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    result = E.check_config(path)
    assert result["ok"] is False and "llm_settings_differ_from_registered_contract" in result["problems"]
    assert E.check_config()["ok"] is True


def _chunk(delta=None, finish=None, usage=None):
    row = {"id": "stub", "object": "chat.completion.chunk", "created": 0, "model": RETURNED_MODEL,
           "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        row["usage"] = usage
    return row


def _sse(rows):
    return "".join(f"data: {json.dumps(row)}\n\n" for row in rows) + "data: [DONE]\n\n"


class _ScriptedNemotron:
    """local stub transport: 실제 도구 결과를 읽어 정해진 도구 순서를 재생한다. 모델 판단이 아니다."""

    def __init__(self, status=200):
        self.status, self.requests = status, []

    async def send(self, client, request, *args, **kwargs):
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "method": request.method,
                              "authorization": request.headers.get("authorization"),
                              "raw": request.content, "body": body})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "synthetic"}, request=request)
        results = [json.loads(message["content"]) for message in body["messages"] if message["role"] == "tool"]
        step = len(results)
        usage = _chunk(usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
        if step == 5:
            rows = [_chunk({"role": "assistant", "content": "Registered."}), _chunk(finish="stop"), usage]
        else:
            name, arguments = self._next_call(step, results)
            call = {"index": 0, "id": f"call_{step}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
            rows = [_chunk({"role": "assistant", "content": None, "tool_calls": [call]}),
                    _chunk(finish="tool_calls"), usage]
        return httpx.Response(200, content=_sse(rows).encode(),
                              headers={"content-type": "text/event-stream"}, request=request)

    @staticmethod
    def _next_call(step, results):
        if step == 0:
            return "grounded_active_registry", {}
        if step == 1:
            return "grounded_evidence_group_index", {"case_id": E.CASE_ID}
        catalog = results[1]["catalog_sha256"]
        if step == 2:
            return "grounded_evidence_group_page", {"case_id": E.CASE_ID, "catalog_sha256": catalog,
                                                    "group_id": results[1]["groups"][0]["group_id"]}
        if step == 3:
            return "grounded_evidence_details", {"case_id": E.CASE_ID, "catalog_sha256": catalog,
                                                 "evidence_ids": [EVIDENCE_ID]}
        premise = {"evidence_id": EVIDENCE_ID, "relation": "eq", "operand": {"value": 0.5, "unit": "telemetry_native"}}

        def predict(relation):
            return [{"measurement_id": name, "relation": relation, "operand": {"value": 0.0, "unit": "reward_native"}}
                    for name in ("reward_mean", "reward_min")]

        return "write_grounded_active_assessment", {
            "case_id": E.CASE_ID, "catalog_sha256": catalog, "selection_binding": results[3]["selection_binding"],
            "hypotheses": [{"id": "late_reward_drop", "rank": 1, "narrative": "synthetic", "premises": [premise]},
                           {"id": "other", "rank": 2, "narrative": "synthetic", "premises": [premise]}],
            "next_check": {"registry_id": "runtime_reward_window", "hypothesis_predictions": [
                {"hypothesis_id": "late_reward_drop", "predictions": predict("lt")},
                {"hypothesis_id": "other", "predictions": predict("gte")}]},
        }


def _stub_transport(monkeypatch, model):
    async def send(client, request, *args, **kwargs):
        return await model.send(client, request, *args, **kwargs)

    def reject_sync(*args, **kwargs):
        raise AssertionError("synchronous HTTP is not part of the NAT path")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", reject_sync)
    _deny_external_network(monkeypatch)


def test_execute_serializes_documented_request_within_one_session(tmp_path, monkeypatch, capsys):
    run, task = _inputs(tmp_path)
    model = _ScriptedNemotron()
    _stub_transport(monkeypatch, model)
    monkeypatch.setenv(E.KEY_ENV, FAKE_KEY)
    assert E.main(_args(run, task, tmp_path / "out", "--execute")) == 0
    printed = capsys.readouterr()
    summary = json.loads(printed.out)
    assert summary["status"] == "hypotheses_only" and summary["model_calls"] == 6
    assert summary["next_check"] == {"registry_id": "runtime_reward_window",
                                     "registered_status": "host_execution_unsupported"}
    assert summary["check_execution"] == "HOLD"

    assert len(model.requests) == 6
    for request in model.requests:
        body = request["body"]
        assert request["method"] == "POST" and request["url"] == E.BASE_URL + "/chat/completions"
        assert request["authorization"] == f"Bearer {FAKE_KEY}"
        assert set(body) - E.DOCUMENTED_REQUEST_FIELDS <= E.UNDOCUMENTED_OBSERVED_FIELDS
        assert "stop" not in body
        assert (body["model"], body["temperature"], body["reasoning_effort"]) == (E.MODEL, 0, "none")
        assert body["max_completion_tokens"] == 16384 and body["stream"] is True
        assert {tool["function"]["name"] for tool in body["tools"]} == E.TOOL_NAMES
        assert FAKE_KEY.encode() not in request["raw"] and CHECKPOINT_MARKER not in request["raw"]
        assert str(run).encode() not in request["raw"]
    first_user = [message for message in model.requests[0]["body"]["messages"] if message["role"] == "user"]
    assert first_user == [{"role": "user", "content": json.loads((tmp_path / "out/entry_packet.json").read_text(
        encoding="utf-8"))["user_message"]}]

    record = json.loads((tmp_path / "out/execution/record.json").read_text(encoding="utf-8"))
    assert record["workflow_sessions"] == 1 and record["llm_calls_started"] == record["llm_calls_finished"] == 6
    assert record["semantic_retries"] == record["transport_retries"] == 0
    assert record["tools"]["budget_valid"] is True and record["roles"] == E.ROLES
    assert record["claims"] == E.CLAIMS and record["provider_model"] == "not_recorded"
    assessment = json.loads((tmp_path / "out/workspace/preregistrations/grounded_active_assessments/user_run.json")
                            .read_text(encoding="utf-8"))
    assert assessment["status"] == "hypotheses_only"
    assert FAKE_KEY not in printed.out + printed.err
    assert FAKE_KEY.encode() not in _all_bytes(tmp_path / "out")


def test_execute_provider_error_stops_after_one_request(tmp_path, monkeypatch, capsys):
    run, task = _inputs(tmp_path)
    model = _ScriptedNemotron(status=500)
    _stub_transport(monkeypatch, model)
    monkeypatch.setenv(E.KEY_ENV, FAKE_KEY)
    assert E.main(_args(run, task, tmp_path / "out", "--execute")) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "HOLD" and summary["reason"] == "workflow_failed"
    assert len(model.requests) == 1
    record = json.loads((tmp_path / "out/execution/record.json").read_text(encoding="utf-8"))
    assert record["workflow_sessions"] == 1 and record["error_stage"] == "workflow"
    assert record["assessment_sha256"] is None and record["check_execution"]["performed"] is False
    assert not (tmp_path / "out/workspace/preregistrations/grounded_active_assessments").exists()


def test_execute_without_key_makes_no_request(tmp_path, monkeypatch, capsys):
    run, task = _inputs(tmp_path)
    model = _ScriptedNemotron()
    _stub_transport(monkeypatch, model)
    monkeypatch.delenv(E.KEY_ENV, raising=False)
    assert E.main(_args(run, task, tmp_path / "out", "--execute")) == 1
    summary = json.loads(capsys.readouterr().out)
    assert (summary["status"], summary["reason"], summary["model_calls"]) == ("HOLD", "credential_unavailable", 0)
    assert model.requests == []
    record = json.loads((tmp_path / "out/execution/record.json").read_text(encoding="utf-8"))
    assert record["workflow_sessions"] == 0
