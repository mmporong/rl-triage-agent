# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import nemotron_reasoning_options_study_v1 as S


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")), encoding="utf-8")
    return S.file_sha(path)


def _corpus(tmp_path: Path, state: dict) -> Path:
    entries = []
    for index in range(1, 9):
        case_id = f"c{index:02d}"
        case_state = deepcopy(state)
        messages = S.O2.messages(case_state)
        state_path = tmp_path / "states" / f"{case_id}.json"
        messages_path = tmp_path / "messages" / f"{case_id}.json"
        entries.append({
            "case_id": case_id,
            "host_gate": ({"run": False, "reason": "healthy_gate"}
                          if index > 6 else {"run": True, "reason": "registered_fault"}),
            "state": {"path": f"states/{case_id}.json", "sha256": _write(state_path, case_state)},
            "messages": {"path": f"messages/{case_id}.json", "sha256": _write(messages_path, messages)},
            "messages_digest": S.O2.O1._digest(messages),
        })
    corpus = tmp_path / "corpus.json"
    _write(corpus, {"schema": S.CORPUS_SCHEMA, "source_anchor": "6" * 40,
                    "source_batch_sha256": "7" * 64, "source_protocol_sha256_lf": "8" * 64,
                    "same_source_measurements": True, "post_exposure_development": True,
                    "approval_granted": False, "GPU_executed": False, "entries": entries})
    return corpus


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    state, _path, _ledger = cast(Any, FIXTURE.selector).__wrapped__(tmp_path, monkeypatch)
    return _corpus(tmp_path / "corpus-root", state)


def _plan(corpus: Path) -> Path:
    path = corpus.parent / "plan.json"
    S._write_new(path, S.prepare_plan(corpus))
    return path


def _valid_result(content: str = '{"option_id":"hold","fact_ids":[]}') -> dict:
    return {"provider_model": S.MODEL, "finish_reason": "stop", "content": content,
            "reasoning_content": "private chain", "reasoning": None,
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
            "reasoning_tokens": 2, "reasoning_tokens_status": "reported", "wall_s": 0.01}


def test_plan_has_same_model_messages_and_exact_abba_cap(corpus):
    plan = S.prepare_plan(corpus)
    assert len(plan["calls"]) == S.API_CAP == 32
    assert [row["mode"] for row in plan["calls"][:4]] == [
        "legacy_none", "enabled_low", "enabled_low", "legacy_none"]
    assert [row["replicate"] for row in plan["calls"][:4]] == [1, 1, 2, 2]
    assert sum(row["evaluation_role"] == "normal_model_only" for row in plan["calls"]) == 8
    assert plan["status"] == "registered_only_after_commit"
    assert plan["analysis"]["prediction"]
    assert plan["analysis"]["adoption_rule"]["complete_calls"] == 32
    assert plan["analysis"]["adoption_rule"]["normal_valid_probe_proposals_not_above_legacy"] is True
    none = S.build_request([{"role": "user", "content": "x"}], "legacy_none")
    low = S.build_request([{"role": "user", "content": "x"}], "enabled_low")
    assert {key: none[key] for key in ("model", "messages", "temperature", "max_tokens", "stream")} == {
        key: low[key] for key in ("model", "messages", "temperature", "max_tokens", "stream")}
    assert none["extra_body"] == {"reasoning_effort": "none"}
    assert low["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True},
                                 "reasoning_effort": "low"}
    assert S.verify_plan(_plan(corpus), corpus) == plan


def test_execute_records_32_single_calls_and_public_projection_removes_reasoning(
        corpus, tmp_path, monkeypatch):
    requests = []

    def complete(request):
        requests.append(deepcopy(request))
        return _valid_result()

    out = tmp_path / "run"
    plan = _plan(corpus)
    monkeypatch.setattr(S, "verify_registered", lambda p, c, _a: S.verify_plan(p, c))
    result = S.execute(plan, corpus, "6" * 40, out, complete=complete)
    assert result["status"] == "complete" and result["request_count"] == 32
    assert len(requests) == 32
    assert [row["extra_body"] for row in requests[:4]] == [
        {"reasoning_effort": "none"},
        {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"},
        {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"},
        {"reasoning_effort": "none"},
    ]
    private = json.loads((out / "calls/01/private.json").read_text(encoding="utf-8"))
    public = json.loads((out / "calls/01/public.json").read_text(encoding="utf-8"))
    assert private["response"]["reasoning_content"] == "private chain"
    assert "reasoning_content" not in public["response"]
    assert public["response"]["reasoning_content_presence"] == "present"
    assert all(row["action"] == "hold" and row["valid"] for row in result["calls"])
    assert result["claims"]["model_or_reasoning_advantage"] is False


def test_enabled_low_error_is_preserved_and_stops_without_fallback(corpus, tmp_path, monkeypatch):
    requests = []

    def complete(request):
        requests.append(request)
        if len(requests) == 2:
            return {"error_type": "BadRequestError", "status_code": 400, "wall_s": 0.01}
        return _valid_result()

    out = tmp_path / "unsupported"
    monkeypatch.setattr(S, "verify_registered", lambda p, c, _a: S.verify_plan(p, c))
    with pytest.raises(RuntimeError, match="fallback 없이 중단"):
        S.execute(_plan(corpus), corpus, "6" * 40, out, complete=complete)
    assert len(requests) == 2
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_calls"] == 2 and failure["automatic_retry"] is False
    assert failure["fallback"] is False
    raw = json.loads((out / "calls/02/private.json").read_text(encoding="utf-8"))
    assert raw["response"]["error_type"] == "BadRequestError"


def test_length_and_unreported_reasoning_tokens_are_hold_not_zero(corpus, tmp_path, monkeypatch):
    def complete(_request):
        return {"provider_model": S.MODEL, "finish_reason": "length", "content": "",
                "wall_s": 0.01}

    monkeypatch.setattr(S, "verify_registered", lambda p, c, _a: S.verify_plan(p, c))
    result = S.execute(_plan(corpus), corpus, "6" * 40, tmp_path / "length", complete=complete)
    assert all(not row["valid"] and row["action"] == "HOLD" for row in result["calls"])
    assert all(row["reasoning_tokens"] is None
               and row["reasoning_tokens_status"] == "unknown" for row in result["calls"])


def test_corpus_rejects_fault_truth_cross_file_and_plan_tampering(corpus, tmp_path):
    plan_path = _plan(corpus)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["calls"][0]["mode"] = "enabled_low"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan"):
        S.verify_plan(plan_path, corpus)

    value = json.loads(corpus.read_text(encoding="utf-8"))
    value["entries"][0]["state"]["path"] = "../outside.json"
    corpus.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="상대 ref"):
        S.prepare_plan(corpus)


def test_plan_change_after_first_response_blocks_second_request(corpus, tmp_path, monkeypatch):
    plan_path = _plan(corpus)
    calls = []

    def complete(request):
        calls.append(request)
        changed = json.loads(plan_path.read_text(encoding="utf-8"))
        changed["calls"][1]["mode"] = "legacy_none"
        plan_path.write_text(json.dumps(changed), encoding="utf-8")
        return _valid_result()

    monkeypatch.setattr(S, "verify_registered", lambda p, c, _a: S.verify_plan(p, c))
    out = tmp_path / "changed-during-call"
    with pytest.raises(ValueError, match="plan"):
        S.execute(plan_path, corpus, "6" * 40, out, complete=complete)
    assert len(calls) == 1
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_calls"] == 1
    assert (out / "calls/01/private.json").is_file()


@pytest.mark.parametrize("command", ["prepare-plan", "execute"])
def test_cli_help_does_not_read_api_key(monkeypatch, capsys, command):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        S.main([command, "--help"])
    assert stop.value.code == 0
    assert "--out" in capsys.readouterr().out

