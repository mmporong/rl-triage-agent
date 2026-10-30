from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
T = importlib.import_module("lineage_nim_transport_v1")
C = importlib.import_module("selection_comparison_controller_v1")

VIEW_SHA = C.V.file_digest(C.V.PROTOCOL)


def _plan(cap: int = 6) -> dict:
    return {"case_ids": ["CASE-A", "CASE-B"],
            "logical_slots": [{"slot_id": "case-a-production", "case_id": "CASE-A"},
                              {"slot_id": "case-b-production", "case_id": "CASE-B"}],
            "global_request_cap": cap, "per_slot_max": 3}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _lf_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    plan = _plan()
    monkeypatch.setattr(T, "ROOT", tmp_path)
    protocol = tmp_path / "bench/protocols/lineage_nim_transport_v1.json"
    protocol.parent.mkdir(parents=True)
    value = {"schema": T.SCHEMA, "model": T.CALL2.MODEL,
             "accepted_provider_models": sorted(T.CALL2.ACCEPTED_PROVIDER_MODELS),
             "params": T.F.request_params(T.CALL2.MODEL), "operator_approved": True,
             "source_sha256_lf": {}, "plan": plan}
    protocol.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(T, "_verify_sources", lambda path, anchor: (
        value if Path(path) == protocol and anchor == "a" * 40 else (_ for _ in ()).throw(ValueError())))
    root = tmp_path / "queue"
    T.initialize(root, protocol, "a" * 40, plan)
    return {"root": root, "protocol": protocol, "plan": plan}


def _snapshot(root: Path) -> dict:
    packet = C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25},
        input_bindings={"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
                        "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64}},
        execution_budget_s=1200,
    )
    return C.V.build_snapshot(packet, "nim_transport_case", "model_adaptive", VIEW_SHA,
                              root / "host-ledger.jsonl", None)


def _result(view: dict, *, invalid: bool = False) -> dict:
    content = ({"wrong": True} if invalid else
               {"action": "continue", "probe": view["eligible_probes"][0],
                "fact_ids": [next(iter(view["evidence_facts"]))]})
    return {"response_id": "fixture-response", "provider_model": T.CALL2.MODEL,
            "finish_reason": "stop", "content": json.dumps(content),
            "reasoning_content": None, "input_tokens": 2, "output_tokens": 1,
            "total_tokens": 3, "wall_s": 0.01}


def _wait_ref(root: Path) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        paths = list((root / "requests").glob("*.json"))
        if paths:
            path = paths[-1]
            return {"request_id": path.stem, "relative_path": f"requests/{path.name}",
                    "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        time.sleep(0.01)
    raise AssertionError("request intent가 생성되지 않았다")


def _wait_path(path: Path) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"기대 파일이 생성되지 않았다: {path.name}")


def test_call2_engine_file_rpc_and_controller_choose_are_consistent(transport: dict) -> None:
    root = transport["root"]
    snapshot = _snapshot(root)
    inputs = C.dialogue_input(snapshot, "production")
    context = C.L.digest(snapshot["binding"])
    callback = T.file_completion(root, "case-a-production", context, inputs["view_digest"], 2)
    output: dict = {}

    def host() -> None:
        output["engine"] = T.CALL2.engine(snapshot, inputs["view"], inputs["messages"],
                                          "production", callback)

    thread = threading.Thread(target=host)
    thread.start()
    ref = _wait_ref(root)
    T.serve_one(root, ref, complete=lambda messages, params: (
        _result(inputs["view"]) if messages == inputs["messages"]
        and params == T.F.request_params(T.CALL2.MODEL) else (_ for _ in ()).throw(AssertionError())))
    thread.join(2)
    assert not thread.is_alive()
    expected = C.replay_dialogue(snapshot, "production", [_result(inputs["view"])])
    assert output["engine"] == expected["engine"]
    choice = C.choose(snapshot, "model_adaptive", expected)
    assert choice["action"] == "continue" and choice["selected_by"] == "model"


def test_pending_intent_blocks_duplicate_or_automatic_retransmission(transport: dict) -> None:
    root = transport["root"]
    messages = [{"role": "user", "content": "one"}]
    params = T.F.request_params(T.CALL2.MODEL)
    T.enqueue(root, "case-a-production", messages, params, "1" * 64, "2" * 64)
    with pytest.raises(RuntimeError, match="미완료 intent"):
        T.enqueue(root, "case-a-production", messages, params, "1" * 64, "2" * 64)


def test_slot_and_global_caps_count_intents_before_provider_completion(
        transport: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    root = transport["root"]
    params = T.F.request_params(T.CALL2.MODEL)
    for ordinal in range(3):
        ref = T.enqueue(root, "case-a-production", [{"role": "user", "content": str(ordinal)}],
                        params, "1" * 64, "2" * 64)
        T.serve_one(root, ref, complete=lambda *_args: _result({
            "eligible_probes": ["unused"], "evidence_facts": {"fact": 1}}))
    with pytest.raises(RuntimeError, match="최대 3회"):
        T.enqueue(root, "case-a-production", [{"role": "user", "content": "four"}],
                  params, "1" * 64, "2" * 64)

    protocol_value = json.loads(transport["protocol"].read_text(encoding="utf-8"))
    limited = _plan(cap=1)
    protocol_value["plan"] = limited
    transport["protocol"].write_text(json.dumps(protocol_value), encoding="utf-8")
    monkeypatch.setattr(T, "_verify_sources", lambda _path, _anchor: protocol_value)
    limited_root = root.parent / "limited-queue"
    T.initialize(limited_root, transport["protocol"], "a" * 40, limited)
    first = T.enqueue(limited_root, "case-a-production", [{"role": "user", "content": "one"}],
                      params, "1" * 64, "2" * 64)
    T.serve_one(limited_root, first, complete=lambda *_args: _result({
        "eligible_probes": ["unused"], "evidence_facts": {"fact": 1}}))
    with pytest.raises(RuntimeError, match="global request cap"):
        T.enqueue(limited_root, "case-b-production", [{"role": "user", "content": "two"}],
                  params, "1" * 64, "2" * 64)


def test_request_context_tamper_and_path_escape_are_rejected(transport: dict) -> None:
    root = transport["root"]
    ref = T.enqueue(root, "case-a-production", [{"role": "user", "content": "one"}],
                    T.F.request_params(T.CALL2.MODEL), "1" * 64, "2" * 64)
    path = root / ref["relative_path"]
    body = json.loads(path.read_text(encoding="utf-8"))
    body["context_digest"] = "3" * 64
    path.write_text(json.dumps(body), encoding="utf-8")
    changed = {**ref, "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="canonical context/view"):
        T.serve_one(root, changed, complete=lambda *_: {})
    with pytest.raises(ValueError, match="경로"):
        T.serve_one(root, {**changed, "relative_path": "../outside.json"}, complete=lambda *_: {})


def test_wrong_params_model_duplicate_id_and_response_tamper_are_rejected(transport: dict) -> None:
    root = transport["root"]
    params = T.F.request_params(T.CALL2.MODEL)
    with pytest.raises(ValueError, match="params"):
        T.enqueue(root, "case-a-production", [{"role": "user", "content": "x"}],
                  {**params, "model": "other/model"}, "1" * 64, "2" * 64)

    callback = T.file_completion(root, "case-a-production", "1" * 64, "2" * 64, 1)
    errors: list[BaseException] = []
    thread = threading.Thread(target=lambda: _capture(errors, callback,
                                                       [{"role": "user", "content": "x"}], params))
    thread.start()
    ref = _wait_ref(root)
    response = {"schema": T.RESPONSE_SCHEMA, "request_id": ref["request_id"],
                "request_sha256": ref["request_sha256"], "result": _result({
                    "eligible_probes": ["unused"], "evidence_facts": {"fact": 1}}),
                "finished_at": "2026-10-30T00:00:00+00:00", "transport_wall_s": 0.01}
    T._write_new(root / "responses" / f"{ref['request_id']}.json", response)
    T._write_new(root / "response-sha" / f"{ref['request_id']}.sha256",
                 {"request_id": ref["request_id"], "response_sha256": "0" * 64})
    thread.join(2)
    assert errors and isinstance(errors[0], ValueError)

    duplicate = root / "requests" / f"{ref['request_id']}.json"
    with pytest.raises(FileExistsError):
        T._write_new(duplicate, json.loads(duplicate.read_text(encoding="utf-8")))


def _capture(errors: list[BaseException], callback, messages: list[dict], params: dict) -> None:
    try:
        callback(messages, params)
    except ValueError as error:
        errors.append(error)


def test_timeout_preserves_pending_and_worker_failure_records_only_error_type(transport: dict) -> None:
    root = transport["root"]
    callback = T.file_completion(root, "case-a-production", "1" * 64, "2" * 64, 0.03)
    with pytest.raises(TimeoutError, match="pending intent"):
        callback([{"role": "user", "content": "timeout"}], T.F.request_params(T.CALL2.MODEL))
    assert len(list((root / "requests").glob("*.json"))) == 1
    assert not list((root / "responses").glob("*.json"))


def test_dispatch_claim_survives_complete_failure_and_blocks_second_post(transport: dict) -> None:
    root = transport["root"]
    ref = T.enqueue(root, "case-a-production", [{"role": "user", "content": "one"}],
                    T.F.request_params(T.CALL2.MODEL), "1" * 64, "2" * 64)
    calls = 0

    def failed(*_args):
        nonlocal calls
        calls += 1
        raise RuntimeError("fixture provider ambiguity")

    with pytest.raises(RuntimeError, match="ambiguity"):
        T.serve_one(root, ref, complete=failed)
    assert calls == 1
    with pytest.raises(RuntimeError, match="dispatch claim/failure"):
        T.serve_one(root, ref, complete=lambda *_args: (_ for _ in ()).throw(AssertionError()))
    assert calls == 1
    measured = T.status(root)
    assert measured["requests"] == measured["dispatches"] == measured["unknown"] == 1
    assert measured["responses"] == measured["pending"] == 0
    assert measured["failures"] == 1 and measured["rows"][0]["failure"] is True


def test_preexisting_claim_and_closed_session_block_new_dispatch_or_enqueue(transport: dict) -> None:
    root = transport["root"]
    params = T.F.request_params(T.CALL2.MODEL)
    ref = T.enqueue(root, "case-a-production", [{"role": "user", "content": "claim"}],
                    params, "1" * 64, "2" * 64)
    T._write_new(root / "dispatches" / f"{ref['request_id']}.json", {
        "schema": "lineage_nim_transport_dispatch_v1", "request_id": ref["request_id"],
        "request_sha256": ref["request_sha256"], "claimed_at": "2026-10-31T00:00:00+00:00",
    })
    calls = []
    with pytest.raises(RuntimeError, match="dispatch claim"):
        T.serve_one(root, ref, complete=lambda *_args: calls.append(True))
    assert calls == [] and T.status(root)["unknown"] == 1

    closed = T.close(root, "operator closed ambiguous dispatch")
    assert closed["reason"] == "operator closed ambiguous dispatch"
    with pytest.raises(RuntimeError, match="종료된"):
        T.enqueue(root, "case-b-production", [{"role": "user", "content": "new"}],
                  params, "1" * 64, "2" * 64)
    with pytest.raises(RuntimeError, match="종료된"):
        T.serve_one(root, ref, complete=lambda *_args: calls.append(True))
    assert calls == []
    measured = T.status(root)
    assert measured["closed"] == closed and measured["unknown"] == 1
    with pytest.raises(FileExistsError, match="이미 종료"):
        T.close(root, "again")


def test_plan_requires_explicit_unique_cases_slots_and_no_default_cap(
        transport: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = deepcopy(transport["plan"])
    bad["logical_slots"][1]["slot_id"] = bad["logical_slots"][0]["slot_id"]
    with pytest.raises(ValueError, match="중복"):
        T._plan(bad)
    missing = deepcopy(transport["plan"])
    missing.pop("global_request_cap")
    with pytest.raises(ValueError, match="필드"):
        T._plan(missing)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WSL|NVIDIA_API_KEY"):
        T._actual_wsl_ready()


def test_real_git_registration_anchor_has_no_protocol_self_reference(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "config", "core.autocrlf", "false")
    for name in T.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    _git(repo, "add", *T.SOURCE_FILES)
    _git(repo, "commit", "-m", "source code A")
    code_anchor = _git(repo, "rev-parse", "HEAD")

    plan = _plan()
    protocol = repo / "bench/protocols/lineage_nim_transport_v1.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema": T.SCHEMA, "model": T.CALL2.MODEL,
             "accepted_provider_models": sorted(T.CALL2.ACCEPTED_PROVIDER_MODELS),
             "params": T.F.request_params(T.CALL2.MODEL), "operator_approved": True,
             "source_sha256_lf": {name: _lf_sha(repo / name) for name in T.SOURCE_FILES},
             "plan": plan}
    protocol.write_text(json.dumps(value), encoding="utf-8")
    _git(repo, "add", "bench/protocols/lineage_nim_transport_v1.json")
    _git(repo, "commit", "-m", "registration protocol B")
    registration_anchor = _git(repo, "rev-parse", "HEAD")
    assert registration_anchor != code_anchor and "source_commit" not in value

    monkeypatch.setattr(T, "ROOT", repo)
    first = T.initialize(tmp_path / "queue-one", protocol, registration_anchor, plan)
    assert first["registration_anchor"] == registration_anchor

    (repo / "unrelated.txt").write_text("later", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-m", "unrelated descendant")
    second = T.initialize(tmp_path / "queue-two", protocol, registration_anchor, plan)
    assert second["registration_anchor"] == registration_anchor

    changed = repo / T.SOURCE_FILES[0]
    changed.write_text(changed.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="현재 source SHA"):
        T.initialize(tmp_path / "queue-three", protocol, registration_anchor, plan)


def test_finite_worker_serves_one_request_then_waits_for_parent_close(transport: dict) -> None:
    root = transport["root"]
    ref = T.enqueue(root, "case-a-production", [{"role": "user", "content": "worker"}],
                    T.F.request_params(T.CALL2.MODEL), "1" * 64, "2" * 64)
    output: dict = {}

    def worker() -> None:
        output["result"] = T.run_worker(root, max_wall_s=2, poll_s=0.01,
                                        complete=lambda *_args: _result({
                                            "eligible_probes": ["unused"],
                                            "evidence_facts": {"fact": 1}}))

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_path(root / "responses" / f"{ref['request_id']}.json")
    assert thread.is_alive(), "empty queue를 부모 조사 완료로 추정하면 안 된다"
    T.close(root, "parent completed all planned cases")
    thread.join(2)
    assert not thread.is_alive()
    result = output["result"]
    assert result["status"] == "closed" and result["posts"] == 1
    assert result["evidence_scope"] == "cpu_fixture"
    assert result["queue_status"]["responses"] == 1
    assert (root / "worker/start.json").is_file() and (root / "worker/result.json").is_file()


def test_worker_dispatch_failure_never_resends_and_preserves_failure(transport: dict) -> None:
    root = transport["root"]
    T.enqueue(root, "case-a-production", [{"role": "user", "content": "fail"}],
              T.F.request_params(T.CALL2.MODEL), "1" * 64, "2" * 64)
    calls = 0

    def failed(*_args):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider fixture failed")

    with pytest.raises(RuntimeError, match="provider fixture failed"):
        T.run_worker(root, max_wall_s=1, poll_s=0.01, complete=failed)
    assert calls == 1 and (root / "worker/failure.json").is_file()
    assert T.status(root)["unknown"] == 1
    with pytest.raises((FileExistsError, RuntimeError)):
        T.run_worker(root, max_wall_s=1, poll_s=0.01,
                     complete=lambda *_args: (_ for _ in ()).throw(AssertionError()))
    assert calls == 1


def test_empty_worker_stops_at_deadline_without_deleting_queue_state(transport: dict) -> None:
    root = transport["root"]
    result = T.run_worker(root, max_wall_s=0.03, poll_s=0.01, complete=lambda *_args: {})
    assert result["status"] == "deadline" and result["posts"] == 0
    assert result["queue_status"]["requests"] == 0
    assert (root / "worker/result.json").is_file()
    assert not (root / "worker/failure.json").exists()


def test_duplicate_worker_is_rejected_by_live_worker_lock(transport: dict) -> None:
    root = transport["root"]
    output: dict = {}

    def first() -> None:
        output["result"] = T.run_worker(root, max_wall_s=2, poll_s=0.01, complete=lambda *_args: {})

    thread = threading.Thread(target=first)
    thread.start()
    _wait_path(root / "worker/start.json")
    with pytest.raises(RuntimeError, match="병렬 transport"):
        T.run_worker(root, max_wall_s=1, poll_s=0.01, complete=lambda *_args: {})
    T.close(root, "stop duplicate worker fixture")
    thread.join(2)
    assert not thread.is_alive() and output["result"]["status"] == "closed"
