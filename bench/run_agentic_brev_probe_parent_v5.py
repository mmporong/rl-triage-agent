"""Parent4 worker lifetime 위에서 API10 fresh selection source를 검증한다."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import agentic_investigation_contract_v2 as A2
import run_agentic_brev_probe_parent_v4 as BASE

V1 = BASE.V2.V1

SCHEMA = "agentic_brev_probe_pair_parent_v5"
SOURCES = BASE.SOURCES | {
    "bench/run_agentic_brev_probe_parent_v5.py",
    "tests/test_run_agentic_brev_probe_parent_v5.py",
}


def fresh_selection(execution: dict[str, Any], resolver: Any) -> tuple[dict, dict, dict]:
    source = execution.get("selection_source")
    V1.require(isinstance(source, dict) and source.get("schema") == "fresh_selection_source_v1",
               "fresh selection source schema")
    required = {"schema", "prepared", "actual_result", "projection", "manifest", "event",
                "terminal", "reservation", "response", "receipt", "api_global_index"}
    V1.require(set(source) == required and source["api_global_index"] == 10,
               "fresh selection source fields/index")
    values = {name: resolver.read(source[name]) for name in required - {"schema", "api_global_index"}}
    prepared, actual = values["prepared"], values["actual_result"]
    projection, manifest = values["projection"], values["manifest"]
    event, terminal = values["event"], values["terminal"]
    reservation, response, receipt = values["reservation"], values["response"], values["receipt"]
    V1.require(projection.get("status") == "PASS"
               and projection.get("projected_prepared_sha256") == source["prepared"]["sha256"]
               and projection.get("same_model_messages") is True
               and projection.get("same_model_view") is True
               and projection.get("same_prior_history") is True,
               "fresh prepared projection")
    V1.require(manifest.get("schema") == "model_decision_journal_v1"
               and event.get("schema") == "model_decision_event_v1"
               and event.get("global_index") == 10
               and event.get("manifest_sha256") == source["manifest"]["sha256"]
               and terminal.get("schema") == "model_decision_terminal_v1"
               and terminal.get("status") == "choice_ready"
               and terminal.get("manifest_sha256") == source["manifest"]["sha256"]
               and event.get("reservation_sha256") == source["reservation"]["sha256"]
               and event.get("response_sha256") == source["response"]["sha256"]
               and receipt.get("reservation_sha256") == source["reservation"]["sha256"]
               and receipt.get("response_sha256") == source["response"]["sha256"],
               "fresh API10 journal/raw chain")
    binding = manifest.get("binding", {})
    V1.require(manifest.get("binding_sha256") == V1.E.digest(binding)
               and binding.get("snapshot_sha256") == V1.E.digest(prepared["context"]["snapshot"])
               and binding.get("initial_messages_sha256") == V1.E.digest(prepared["messages"])
               and manifest.get("params_sha256") == V1.E.digest(manifest.get("params")),
               "fresh manifest binding")
    V1.require(reservation.get("schema") == "model_decision_reservation_v1"
               and reservation.get("index") == response.get("index") == 10
               and reservation.get("ordinal") == event.get("ordinal") == 1
               and reservation.get("binding_sha256") == manifest["binding_sha256"]
               and reservation.get("messages_sha256") == V1.E.digest(prepared["messages"])
               and reservation.get("manifest_sha256") == source["manifest"]["sha256"]
               and reservation.get("params_sha256") == manifest["params_sha256"]
               and reservation.get("run_id") == binding.get("run_id")
               and response.get("schema") == "model_decision_response_v1"
               and response.get("status") == "completed"
               and response.get("reservation_sha256") == source["reservation"]["sha256"],
               "fresh API10 exact request/response")
    content = response.get("result", {}).get("content")
    V1.require(isinstance(content, str), "fresh API10 content")
    context = A2.checked(prepared["context"])
    review = A2.assess(content, context)
    choice = A2.probe_choice(context, review)
    engine = actual.get("engine_result")
    V1.require(engine == terminal.get("engine_result")
               and terminal.get("completed_calls") == 1
               and terminal.get("calls_sha256") == V1.E.digest([response["result"]])
               and engine["review"]["controller_choice"] == choice
               and review["decision"] == {"action": "run_probe", "target": "P_reward",
                                           "fact_ids": choice["fact_ids"]},
               "fresh A2 reassessment/actual result")
    snapshot = deepcopy(context["snapshot"])
    spec = V1.C._frozen_request_spec(snapshot, choice)
    V1.require(spec["probe"] == "P_reward", "fresh request spec")
    return snapshot, spec, {"schema": "fresh_selection_transport_v1",
                            "source_sha256": V1.E.digest(source),
                            "api_global_index": 10, "reassessed": True}


@contextmanager
def fresh_parent_scope():
    with BASE.registered_worker_scope():
        original_selection, original_schema = V1._selection, V1.SCHEMA
        V1._selection, V1.SCHEMA = fresh_selection, SCHEMA
        try:
            yield
        finally:
            V1._selection, V1.SCHEMA = original_selection, original_schema


def prepare(*args: Any, **kwargs: Any) -> dict:
    with fresh_parent_scope():
        return V1.prepare(*args, **kwargs)


def lookup(*args: Any, **kwargs: Any) -> dict:
    with fresh_parent_scope():
        return V1.lookup(*args, **kwargs)


def run(*args: Any, **kwargs: Any) -> dict:
    with fresh_parent_scope():
        return V1.run(*args, **kwargs)


def verify_authority(*args: Any, **kwargs: Any) -> dict:
    with fresh_parent_scope():
        return V1.verify_authority(*args, **kwargs)


def _verify_completion(*args: Any, **kwargs: Any) -> dict:
    with fresh_parent_scope():
        return V1._verify_completion(*args, **kwargs)


def main() -> None:
    with fresh_parent_scope():
        V1.main()


def __getattr__(name: str):
    return getattr(BASE, name)


if __name__ == "__main__":
    main()
