from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_fresh_selection_prefix_v1 as M


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> dict:
    path.write_bytes(encoded(value))
    return {"path": str(path), "sha256": file_sha(path)}


def fixture(tmp_path: Path):
    choice = {"probe": "P_reward", "selected_by": "model",
              "fact_ids": [f"telemetry:fact:{index}" for index in range(6)]}
    decision = {"action": "run_probe", "target": "P_reward", "fact_ids": choice["fact_ids"]}
    context = {"snapshot": {"view": {"observations": []}}, "goal": "g", "evidence": {},
               "revealed": [], "extra_facts": {}, "decisions_left": 2}
    prepared = {"context": context, "messages": [{"role": "user", "content": "x"}]}
    result = {"api": {"reserved": 10, "pending": 0, "remaining": 2}, "engine_result": {
        "review": {"review": {"decision": decision}, "controller_choice": choice}}}
    prepared_ref = write(tmp_path / "prepared.json", prepared)
    source_ref = write(tmp_path / "source.json", {"source": True})
    result_ref = write(tmp_path / "result.json", result)
    projection_ref = write(tmp_path / "projection.json", {
        "status": "PASS", "projected_prepared_sha256": prepared_ref["sha256"],
        "source_prepared_sha256": source_ref["sha256"], "same_model_messages": True,
        "same_model_view": True, "same_prior_history": True,
    })
    config = {"prepared_linux": prepared_ref, "source_prepared": source_ref,
              "prepared_projection": projection_ref, "actual_call_result": result_ref}
    config_ref = write(tmp_path / "config.json", config)
    run_id = "fresh-test"
    binding = {"run_id": run_id, "initial_messages_sha256": digest(prepared["messages"])}
    params = {"model": "nemotron-test"}
    manifest_ref = write(tmp_path / "journal-manifest.json", {
        "schema": "model_decision_journal_v1", "api_cap": 12, "binding": binding,
        "binding_sha256": digest(binding), "params_sha256": digest(params),
    })
    api = tmp_path / "api"
    (api / "reservations").mkdir(parents=True)
    (api / "responses").mkdir()
    (api / "receipts").mkdir()
    write(api / "manifest.json", {"schema": "model_decision_api_v1", "api_cap": 12})
    reservation = {
        "schema": "model_decision_reservation_v1", "index": 10, "ordinal": 1,
        "run_id": run_id, "binding_sha256": digest(binding),
        "manifest_sha256": manifest_ref["sha256"],
        "messages_sha256": digest(prepared["messages"]), "params_sha256": digest(params),
    }
    reservation_ref = write(api / "reservations/0010.json", reservation)
    response = {"schema": "model_decision_response_v1", "status": "completed",
                "reservation_sha256": reservation_ref["sha256"],
                "result": {"content": json.dumps(decision)}}
    response_ref = write(api / "responses/0010.json", response)
    write(api / "receipts/0010.json", {
        "reservation_sha256": reservation_ref["sha256"],
        "response_sha256": response_ref["sha256"],
    })
    event_ref = write(tmp_path / "event.json", {
        "schema": "model_decision_event_v1", "global_index": 10,
        "manifest_sha256": manifest_ref["sha256"],
        "reservation_sha256": reservation_ref["sha256"],
        "response_sha256": response_ref["sha256"],
    })
    terminal_ref = write(tmp_path / "terminal.json", {
        "schema": "model_decision_terminal_v1", "status": "choice_ready",
        "completed_calls": 1, "manifest_sha256": manifest_ref["sha256"],
        "engine_result": result["engine_result"],
    })
    item = {
        "schema": M.SCHEMA, "selection_config": config_ref, "prepared": prepared_ref,
        "actual_result": result_ref,
        "model_journal": {"manifest": manifest_ref, "event": event_ref,
                          "terminal": terminal_ref},
        "operator": "codex_main_user_delegated",
        "old_transport_role": "resource_and_host_envelope_only",
        "api_global_index": 10, "api_calls_replayed": 0,
        "initial_host_preloaded_facts_sha256": digest(context["extra_facts"]),
    }
    runtime = SimpleNamespace(
        read_ref=lambda ref: json.loads(Path(ref["path"]).read_text(encoding="utf-8")),
        verify_prefix=lambda *args, **kwargs: ("old", "old"),
        J=SimpleNamespace(file_sha=file_sha, digest=digest),
    )
    return choice, decision, prepared, item, runtime, api


def patch_contract(monkeypatch, choice: dict, decision: dict) -> None:
    monkeypatch.setattr(M.A2, "checked", lambda context: context)
    monkeypatch.setattr(M.A2, "assess", lambda content, context: {"decision": decision})
    monkeypatch.setattr(M.A2, "probe_choice", lambda context, review: choice)


def test_fresh_prefix_recomputes_choice_without_api_replay(tmp_path: Path, monkeypatch) -> None:
    choice, decision, prepared, item, runtime, api = fixture(tmp_path)
    patch_contract(monkeypatch, choice, decision)
    capsule, engine = M.verify(item, prepared, api, runtime=runtime)
    assert capsule["api_calls_replayed"] == 0
    assert json.loads(engine["final_content"])["target"] == "P_reward"


def test_other_probe_or_tampered_api_blocks_before_dispatch(tmp_path: Path, monkeypatch) -> None:
    choice, decision, prepared, item, runtime, api = fixture(tmp_path)
    patch_contract(monkeypatch, choice, decision)
    response_path = api / "responses/0010.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["result"]["content"] = json.dumps({**decision, "target": "P_value"})
    response_path.write_bytes(encoded(response))
    try:
        M.verify(item, prepared, api, runtime=runtime)
    except ValueError as error:
        assert "raw reservation" in str(error)
    else:
        raise AssertionError("tampered API10 response accepted")


def test_scope_delegates_old_prefix_and_restores_descriptor(tmp_path: Path) -> None:
    _choice, _decision, _prepared, item, runtime, _api = fixture(tmp_path)

    class Runtime:
        @staticmethod
        def _request(*args):
            return {"old": True}

    runtime.Runtime = Runtime
    runtime._equivalent = lambda left, right: left == right
    original_verify = runtime.verify_prefix
    original_descriptor = vars(Runtime)["_request"]
    with M.scope(runtime, item):
        assert runtime.verify_prefix({"schema": "old"}, {}, tmp_path, repo=tmp_path) == ("old", "old")
    assert runtime.verify_prefix is original_verify
    assert vars(Runtime)["_request"] is original_descriptor
