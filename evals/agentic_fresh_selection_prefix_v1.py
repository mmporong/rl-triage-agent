"""Fresh-frame API 판단을 기존 investigation runtime의 첫 prefix로 검증한다.

옛 CALL.v2 capsule로 변환하지 않는다. 실제 fresh-frame 입력·journal·A2 판단을
다시 검증하고, 같은 runtime 요청에서만 이미 완료된 판단을 한 번 재생한다.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import agentic_investigation_contract_v2 as A2

SCHEMA = "fresh_frame_runtime_prefix_v1"


def require(ok: Any, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _ref(runtime: Any, reference: dict) -> dict:
    return runtime.read_ref(reference)


def verify(item: dict, request: dict, api_root: Path, *, runtime: Any) -> tuple[dict, dict]:
    """API를 호출하거나 원장을 쓰지 않고 fresh 판단과 현재 요청을 결합한다."""
    require(item.get("schema") == SCHEMA and set(item) == {
        "schema", "selection_config", "prepared", "actual_result", "model_journal",
        "operator", "old_transport_role", "api_global_index", "api_calls_replayed",
        "initial_host_preloaded_facts_sha256",
    }, "fresh prefix fields")
    require(item["operator"] == "codex_main_user_delegated"
            and item["old_transport_role"] == "resource_and_host_envelope_only"
            and item["api_global_index"] == 10 and item["api_calls_replayed"] == 0,
            "fresh prefix operator/transport/API role")
    config = _ref(runtime, item["selection_config"])
    prepared = _ref(runtime, item["prepared"])
    actual = _ref(runtime, item["actual_result"])
    require(prepared == _ref(runtime, config["prepared_linux"]),
            "fresh prepared/config binding")
    require(actual == _ref(runtime, config["actual_call_result"]),
            "fresh result/config binding")
    projection = _ref(runtime, config["prepared_projection"])
    require(projection.get("status") == "PASS"
            and projection.get("projected_prepared_sha256") == config["prepared_linux"]["sha256"]
            and projection.get("source_prepared_sha256") == config["source_prepared"]["sha256"]
            and projection.get("same_model_messages") is True
            and projection.get("same_model_view") is True
            and projection.get("same_prior_history") is True,
            "official Linux prepared projection binding")
    require(request.get("context") == prepared.get("context")
            and request.get("messages") == prepared.get("messages"),
            "fresh model input differs from runtime request")
    require(item["initial_host_preloaded_facts_sha256"]
            == runtime.J.digest(prepared["context"]["extra_facts"]),
            "fresh initial host-preloaded facts binding")

    refs = item["model_journal"]
    require(set(refs) == {"manifest", "event", "terminal"}, "fresh journal refs")
    manifest = _ref(runtime, refs["manifest"])
    event = _ref(runtime, refs["event"])
    terminal = _ref(runtime, refs["terminal"])
    require(manifest.get("schema") == "model_decision_journal_v1"
            and manifest.get("api_cap") == 12
            and event.get("schema") == "model_decision_event_v1"
            and event.get("global_index") == 10
            and event.get("manifest_sha256") == refs["manifest"]["sha256"]
            and all(isinstance(event.get(key), str) and len(event[key]) == 64
                    for key in ("reservation_sha256", "response_sha256"))
            and terminal.get("schema") == "model_decision_terminal_v1"
            and terminal.get("status") == "choice_ready"
            and terminal.get("completed_calls") == 1
            and terminal.get("manifest_sha256") == refs["manifest"]["sha256"],
            "fresh API10 journal chain")
    engine = actual.get("engine_result")
    require(engine == terminal.get("engine_result")
            and actual.get("api") == {"reserved": 10, "pending": 0, "remaining": 2},
            "fresh result/journal/API state")
    decision = engine["review"]["review"]["decision"]
    choice = engine["review"]["controller_choice"]
    reservation_path = Path(api_root) / "reservations/0010.json"
    response_path = Path(api_root) / "responses/0010.json"
    receipt_path = Path(api_root) / "receipts/0010.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(runtime.J.file_sha(reservation_path) == event["reservation_sha256"]
            and runtime.J.file_sha(response_path) == event["response_sha256"]
            and receipt.get("reservation_sha256") == event["reservation_sha256"]
            and receipt.get("response_sha256") == event["response_sha256"],
            "fresh API10 raw reservation/response/receipt SHA")
    require(reservation.get("schema") == "model_decision_reservation_v1"
            and reservation.get("index") == 10 and reservation.get("ordinal") == 1
            and reservation.get("run_id") == manifest["binding"]["run_id"]
            and reservation.get("binding_sha256") == manifest["binding_sha256"]
            and reservation.get("manifest_sha256") == refs["manifest"]["sha256"]
            and reservation.get("messages_sha256") == runtime.J.digest(prepared["messages"])
            == manifest["binding"]["initial_messages_sha256"]
            and reservation.get("params_sha256") == manifest["params_sha256"],
            "fresh API10 exact request binding")
    content = response.get("result", {}).get("content")
    require(response.get("schema") == "model_decision_response_v1"
            and response.get("status") == "completed"
            and response.get("reservation_sha256") == event["reservation_sha256"]
            and isinstance(content, str), "fresh API10 response binding")
    context = A2.checked(prepared["context"])
    review = A2.assess(content, context)
    recomputed_choice = A2.probe_choice(context, review)
    require(choice == recomputed_choice and review["decision"] == decision and decision == {
        "action": "run_probe", "target": "P_reward", "fact_ids": choice["fact_ids"]},
        "A2 assess/probe_choice differs from fresh journal")
    require(len(choice["fact_ids"]) == 6
            and all(value.startswith("telemetry:") for value in choice["fact_ids"]),
            "fresh P_reward requires six telemetry facts")

    api_manifest = json.loads((Path(api_root) / "manifest.json").read_text(encoding="utf-8"))
    require(api_manifest == {"schema": "model_decision_api_v1", "api_cap": 12},
            "shared API cap changed")
    capsule = {
        "schema": SCHEMA + "_validated",
        "selection_config": deepcopy(item["selection_config"]),
        "prepared": deepcopy(item["prepared"]),
        "actual_result": deepcopy(item["actual_result"]),
        "model_journal": deepcopy(item["model_journal"]),
        "api_global_index": 10,
        "api_calls_replayed": 0,
    }
    replay = deepcopy(engine)
    replay["final_content"] = json.dumps(decision, ensure_ascii=False, sort_keys=True)
    replay["fresh_prefix"] = capsule
    return capsule, replay


@contextmanager
def scope(runtime: Any, item: dict) -> Iterator[None]:
    """첫 요청을 실제 fresh 입력으로 고정하고 기존 runtime 전이는 보존한다."""
    require(item.get("schema") == SCHEMA, "fresh prefix scope item")
    prepared = _ref(runtime, item["prepared"])
    original = runtime.verify_prefix
    original_request_descriptor = vars(runtime.Runtime)["_request"]
    original_request = runtime.Runtime._request

    def request(loop: Any, bridge: Any, visual: dict | None) -> dict:
        expected = prepared["context"]
        if (bridge is None and visual is None and loop.pending is None and not loop.events
                and not loop.extra_facts and not loop.revealed
                and loop.status == "active" and loop.remaining == expected["decisions_left"]
                and runtime._equivalent(loop.snapshot, expected["snapshot"])
                and loop.goal == expected["goal"] and loop.evidence == expected["evidence"]):
            facts = deepcopy(expected["extra_facts"])
            require(runtime.J.digest(facts) == item["initial_host_preloaded_facts_sha256"],
                    "fresh initial facts changed before host preload")
            loop.extra_facts = facts
            current = loop.context()
            require(current == expected, "fresh host-preloaded context differs")
            return {"context": deepcopy(current), "messages": deepcopy(prepared["messages"])}
        return original_request(loop, bridge, visual)

    def bound(item: dict, request: dict, api_root: Path, *, repo: Path):
        if item.get("schema") == SCHEMA:
            return verify(item, request, api_root, runtime=runtime)
        return original(item, request, api_root, repo=repo)

    runtime.verify_prefix = bound
    runtime.Runtime._request = staticmethod(request)
    try:
        yield
    finally:
        runtime.verify_prefix = original
        runtime.Runtime._request = original_request_descriptor
