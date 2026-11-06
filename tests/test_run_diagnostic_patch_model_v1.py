# pyright: reportMissingImports=false
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evals"), str(ROOT / "bench")]
import diagnostic_patch_bridge_v1 as B
import diagnostic_patch_prompts_v1 as M
import ppo_patch_cases_v1 as C
import run_diagnostic_patch_model_v1 as R

from evals import ppo_patch_feedback_v1 as F


def request():
    context = B.diagnosis_context(C.SOURCES["literal_override"], F.validation_plan()[0], case_id="case-a")
    req = {"schema": "diagnostic_patch_request_v1", "kind": "diagnosis", "case_id": "case-a", "model": "test-model",
           "context": context, "context_sha256": B.digest(context), "messages": M.messages("diagnosis", context), "chain": {}}
    protocol = {"model": "test-model", "CPU_configurations": F.validation_plan(),
                "cases": [{"case_id": "case-a", "mode": "new_adapter", "source": context["source"]}]}
    return protocol, req


def test_actual_prompt_request_is_reconstructed_from_registered_root(tmp_path):
    p, r = request()
    assert R.validate_request(p, r, tmp_path) == (r["context"], r["messages"])


@pytest.mark.parametrize("field", ["source", "messages", "context_sha256", "case_id"])
def test_request_substitution_is_rejected(tmp_path, field):
    p, r = request()
    if field == "source":
        r["context"]["source"] = C.SOURCES["healthy"]
        r["context_sha256"] = B.digest(r["context"])
        r["messages"] = M.messages("diagnosis", r["context"])
    else:
        r[field] = "tampered"
    with pytest.raises(ValueError):
        R.validate_request(p, r, tmp_path)


def test_phase_allows_infrastructure_retry_but_blocks_semantic_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    reqpath = root / "request.json"
    p, req = request()
    p.update(campaign_id="test", API_cap=3)
    reqpath.write_text(json.dumps(req), encoding="utf-8")
    out = root / "response01.json"
    assert R.reserve(p, req, reqpath, root, out) == 1
    with pytest.raises(ValueError, match="봉인된 인프라"):
        R.reserve(p, req, reqpath, root, root / "response02.json")
    out.write_text(json.dumps({"provider_raw": {"content": "{}", "finish_reason": "stop"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="의미 결과"):
        R.reserve(p, req, reqpath, root, root / "response02.json")
    out.write_text(json.dumps({"provider_raw": {"infra_error": True}}), encoding="utf-8")
    assert R.reserve(p, req, reqpath, root, root / "response02.json") == 2


def test_native_response_cannot_be_rewritten_to_another_proposal():
    _, req = request()
    choice = {"action": "inspect", "probe": B.PROBE, "source_sha256": req["context"]["source_sha256"],
              "fact_ids": ["configured.epochs"], "rationale": "check actual work"}
    response = {"status": "valid", "context_sha256": req["context_sha256"],
                "provider_raw": {"content": json.dumps(choice)}, "proposal": copy.deepcopy(choice)}
    assert R.native_proposal(response, req, kind="diagnosis") == choice
    response["proposal"]["rationale"] = "rewritten"
    with pytest.raises(ValueError, match="원문"):
        R.native_proposal(response, req, kind="diagnosis")
