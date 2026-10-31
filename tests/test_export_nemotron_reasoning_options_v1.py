# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import export_nemotron_reasoning_options_v1 as X


def write(path: Path, value: object, *, pretty: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2 if pretty else None,
                               separators=None if pretty else (",", ":")), encoding="utf-8")
    return path


def planned(index: int = 1, *, case: str = "c01", mode: str = "legacy_none") -> dict:
    return {"call_index": index, "case_id": case, "evaluation_role": "normal_model_only",
            "abba_position": 1, "mode": mode, "replicate": 1,
            "messages_file_sha256": "a" * 64, "messages_digest": "b" * 64}


def test_pretty_raw_sha_and_canonical_digest_are_distinct_and_both_required(tmp_path: Path):
    messages = [{"role": "system", "content": "contract"},
                {"role": "user", "content": "{}"}]
    path = write(tmp_path / "messages.json", messages, pretty=True)
    raw, canonical = X._sha(path), X.V1.O2.O1._digest(messages)
    assert raw != canonical
    X._bind_messages(path, messages, raw, canonical)
    with pytest.raises(ValueError, match="raw file SHA"):
        X._bind_messages(path, messages, canonical, canonical)
    with pytest.raises(ValueError, match="canonical"):
        X._bind_messages(path, messages, raw, "0" * 64)


def test_complete_result_rejects_missing_and_plan_row_swap():
    calls = []
    for index in range(1, 33):
        case = f"c{(index - 1) // 4 + 1:02d}"
        mode = ("legacy_none", "enabled_low", "enabled_low", "legacy_none")[(index - 1) % 4]
        row = planned(index, case=case, mode=mode)
        row["replicate"] = 1 if (index - 1) % 4 < 2 else 2
        row["evaluation_role"] = "normal_model_only" if case in {"c01", "c05"} else "fault_development"
        calls.append(row)
    result_rows = [{key: row[key] for key in (
        "call_index", "case_id", "mode", "replicate", "evaluation_role")} for row in calls]
    value = {"schema": X.V2.RESULT_SCHEMA, "status": "complete", "calls": result_rows,
             "request_count": 32, "api_cap": 32}
    assert X._result_rows(value, {"calls": calls}) == result_rows
    value["calls"] = result_rows[:-1]
    with pytest.raises(ValueError, match="32행"):
        X._result_rows(value, {"calls": calls})
    value["calls"] = result_rows
    value["calls"][0], value["calls"][1] = value["calls"][1], value["calls"][0]
    with pytest.raises(ValueError, match="결합"):
        X._result_rows(value, {"calls": calls})


def test_call_rejects_request_or_case_binding_swap(tmp_path: Path, monkeypatch):
    messages = [{"role": "user", "content": "visible"}]
    request = X.V1.build_request(messages, "legacy_none")
    plan = planned()
    plan["messages_file_sha256"] = "1" * 64
    plan["messages_digest"] = X.V1.O2.O1._digest(messages)
    response = {"provider_model": X.V1.MODEL, "finish_reason": "stop",
                "content": '{"option_id":"hold","fact_ids":[]}',
                "reasoning_content": "private trace", "reasoning": None,
                "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                "reasoning_tokens": 0, "reasoning_tokens_status": "reported", "wall_s": 0.1}
    review = {"status": "valid", "response_text": response["content"]}
    monkeypatch.setattr(X.V1, "_review", lambda result, state: (review, "hold"))
    root = tmp_path / "call"
    intent = {"schema": "nemotron_reasoning_options_intent_v2", "planned": plan,
              "request": request, "messages_file_sha256": plan["messages_file_sha256"],
              "messages_digest": plan["messages_digest"],
              "request_sha256": X.V1.digest(request)}
    private = {"schema": X.V2.CALL_SCHEMA, "planned": plan,
               "request_sha256": X.V1.digest(request), "response": response,
               "response_field_names": sorted(response), "review": review, "model_action": "hold",
               "normal_model_probe_proposed": False, "actual_system_gate_executed": False,
               "approval": False, "repair": False, "recovery": False}
    write(root / "intent.json", intent)
    write(root / "private.json", private)
    write(root / "public.json", X.V1._public_call(private))
    row = {"call_index": 1, "case_id": "c01", "mode": "legacy_none", "replicate": 1,
           "evaluation_role": "normal_model_only", "valid": True, "action": "hold",
           "finish_reason": "stop", "provider_model": X.V1.MODEL, "input_tokens": 10,
           "output_tokens": 4, "total_tokens": 14, "reasoning_tokens": 0,
           "reasoning_tokens_status": "reported", "wall_s": 0.1,
           "private_sha256": X._sha(root / "private.json"),
           "public_sha256": X._sha(root / "public.json")}
    public, _proof = X._validate_call(root, plan, row, {"messages": messages, "state": {}})
    assert "reasoning_content" not in public["response"]
    intent["request"]["max_tokens"] = 1
    write(root / "tampered-intent.json", intent)
    (root / "intent.json").write_bytes((root / "tampered-intent.json").read_bytes())
    with pytest.raises(ValueError, match="intent"):
        X._validate_call(root, plan, row, {"messages": messages, "state": {}})


def test_registered_adoption_rule_is_action_only_and_not_diagnosis_benefit():
    rows = []
    for case_index in range(1, 9):
        role = "normal_model_only" if case_index in (1, 5) else "fault_development"
        for replicate in (1, 2):
            rows.extend([
                {"case_id": f"c{case_index:02d}", "replicate": replicate,
                 "evaluation_role": role, "mode": "legacy_none", "valid": True,
                 "action": "hold", "finish_reason": "stop", "reasoning_tokens": 0,
                 "wall_s": 1.0, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                {"case_id": f"c{case_index:02d}", "replicate": replicate,
                 "evaluation_role": role, "mode": "enabled_low", "valid": True,
                 "action": ("read_evidence" if role == "fault_development" else "hold"),
                 "finish_reason": "stop", "reasoning_tokens": 1, "wall_s": 2.0,
                 "input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            ])
    metrics = X._metrics(rows)
    rule = {"complete_calls": 32, "enabled_low_fault_valid_at_least": "10/12",
            "distinct_fault_cases_with_valid_read_or_probe_at_least": 2}
    result = X._adoption({"analysis": {"adoption_rule": rule}}, metrics)
    assert result["action_candidate"] is True
    assert result["diagnosis_benefit"] is False
    assert result["actual_probe_confirmation_performed"] is False


def test_script_and_module_help_bootstrap() -> None:
    script = X.ROOT / "evals/export_nemotron_reasoning_options_v1.py"
    for command in ([sys.executable, "-B", str(script), "--help"],
                    [sys.executable, "-B", "-m", "evals.export_nemotron_reasoning_options_v1",
                     "--help"]):
        result = subprocess.run(command, cwd=X.ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False)
        assert result.returncode == 0, result.stderr
        assert "--result" in result.stdout and "--corpus" in result.stdout
