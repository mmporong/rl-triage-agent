# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import nemotron_reasoning_options_study_v1 as V1
from evals import nemotron_reasoning_options_study_v2 as V2


def _write(path: Path, value: object, *, pretty: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2 if pretty else None,
                               separators=None if pretty else (",", ":")), encoding="utf-8")
    return V1.file_sha(path)


def _corpus(tmp_path: Path, state: dict) -> Path:
    entries = []
    for index in range(1, 9):
        case_id = f"c{index:02d}"
        messages = V1.O2.messages(state)
        state_path = tmp_path / "states" / f"{case_id}.json"
        messages_path = tmp_path / "messages" / f"{case_id}.json"
        entries.append({
            "case_id": case_id,
            "host_gate": ({"run": False, "reason": "healthy_gate"}
                          if index in (1, 5) else {"run": True, "reason": None}),
            "state": {"path": f"states/{case_id}.json", "sha256": _write(state_path, state)},
            "messages": {"path": f"messages/{case_id}.json",
                         "sha256": _write(messages_path, messages, pretty=index == 1)},
            "messages_digest": V1.O2.O1._digest(messages),
        })
    corpus = tmp_path / "corpus.json"
    _write(corpus, {"schema": V1.CORPUS_SCHEMA, "source_anchor": "6" * 40,
                    "source_batch_sha256": "7" * 64, "source_protocol_sha256_lf": "8" * 64,
                    "same_source_measurements": True, "post_exposure_development": True,
                    "approval_granted": False, "GPU_executed": False, "entries": entries}, pretty=True)
    return corpus


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    state, _path, _ledger = cast(Any, FIXTURE.selector).__wrapped__(tmp_path, monkeypatch)
    return _corpus(tmp_path / "corpus-root", state)


def _plan(corpus: Path) -> Path:
    path = corpus.parent / "plan-v2.json"
    V1._write_new(path, V2.prepare_plan(corpus))
    return path


def _valid_result() -> dict:
    return {"provider_model": V1.MODEL, "finish_reason": "stop",
            "content": '{"option_id":"hold","fact_ids":[]}',
            "reasoning_content": None, "reasoning": None,
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
            "reasoning_tokens": None, "reasoning_tokens_status": "unknown", "wall_s": 0.01}


def test_plan_separates_raw_file_sha_from_equal_canonical_digest(corpus):
    plan = V2.prepare_plan(corpus)
    first, second_case = plan["calls"][0], plan["calls"][4]
    assert first["messages_file_sha256"] != second_case["messages_file_sha256"]
    assert first["messages_digest"] == second_case["messages_digest"]
    assert first["messages_file_sha256"] != first["messages_digest"]
    assert "messages_sha256" not in first
    assert plan["prior_invalid"] == {
        "protocol": {"path": V2.V1_PROTOCOL, "sha256_lf": V1.lf_sha(V1.ROOT / V2.V1_PROTOCOL)},
        "result_schema": V1.RESULT_SCHEMA, "status": "invalid", "api_requests": 0,
        "intent_files": 0, "failure_stage": "before_first_dispatch",
        "cause": "raw_messages_file_sha_was_compared_to_canonical_messages_digest",
        "semantic_retry": False}
    assert V2.verify_plan(_plan(corpus), corpus) == plan


def test_actual_execute_path_reaches_all_32_calls_with_pretty_messages(corpus, tmp_path):
    requests = []

    def complete(request):
        requests.append(deepcopy(request))
        return _valid_result()

    result = V2.execute(
        _plan(corpus), corpus, "6" * 40, tmp_path / "run", complete=complete,
        registration_check=lambda p, c, _a: V2.verify_plan(p, c))
    assert result["schema"] == V2.RESULT_SCHEMA and result["request_count"] == 32
    assert len(requests) == 32 and all(row["valid"] for row in result["calls"])
    intent = json.loads((tmp_path / "run/calls/01/intent.json").read_text(encoding="utf-8"))
    assert intent["messages_file_sha256"] != intent["messages_digest"]
    assert intent["messages_digest"] == V1.O2.O1._digest(intent["request"]["messages"])
    assert result["prior_invalid_api_requests"] == 0


def test_v2_rejects_raw_or_canonical_binding_tamper(corpus):
    path = _plan(corpus)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["calls"][0]["messages_digest"] = "0" * 64
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="v2 plan"):
        V2.verify_plan(path, corpus)


@pytest.mark.parametrize("command", ["prepare-plan", "execute"])
def test_cli_help_has_no_api_side_effect(monkeypatch, capsys, command):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        V2.main([command, "--help"])
    assert stop.value.code == 0 and "--out" in capsys.readouterr().out
