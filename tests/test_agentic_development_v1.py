"""실제 어댑터·journal을 사용하는 개발 실행 경로와 예약 보존 회귀."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import run_agentic_development_v1 as D
from test_agentic_investigation_evidence_v1 import case  # noqa: F401 - 공유된 실제 계약 fixture


@pytest.fixture
def plan(case):  # noqa: F811 - pytest가 공유 fixture를 주입한다.
    path = case["root"] / "index.json"
    D.J._write_new(path, case["index"])
    return {"schema": D.SCHEMA, "roots": {k: str(v) for k, v in case["roots"].items()},
            "index": {"path": str(path), "sha256": D.J.file_sha(path)},
            "index_canonical_sha256": D.E.index_sha256(case["index"]),
            "output": str(case["root"] / "development"), "source_anchor": "a" * 40,
            "protocol": {"path": "unused", "sha256": "b" * 64}, "goal": "등록 자료로 다음 검사를 고른다",
            "max_decisions": 4, "api_cap": 12,
            "cases": [{"id": "sample_01", "evidence_id": "evidence-01",
                       "initial_facts": "config_only", "frame_ids": ["f01", "f02"]}]}


def response(action, target, fact="config:action_scale"):
    return {"provider_model": D.CALL.O.MODEL, "finish_reason": "stop", "wall_s": 0.01,
            "content": json.dumps({"action": action, "target": target, "fact_ids": [fact]})}


def test_driver_replays_training_frames_probe_without_recalling_provider(plan):
    sent = []

    def provider(messages, _):
        sent.append(messages)
        if len(sent) == 1:
            return response("read_evidence", "evidence:training-01")
        if len(sent) == 2:
            return response("inspect_frames", "evidence:clip-01")
        return response("run_probe", "P_reward", "vision:frame:f01")

    result = D.run_case(plan, plan["cases"][0], provider)
    assert result["status"] == "probe_proposed_unexecuted" and len(sent) == 3
    assert [row["image_count"] for row in result["decisions"]] == [0, 0, 2]
    assert result["api"]["reserved"] == 3 and result["approval_granted"] is False
    replay = D.run_case(plan, plan["cases"][0], lambda *_: pytest.fail("완료 요청 재전송"))
    assert replay == result


def test_driver_semantic_invalid_is_hold_not_rule_fallback(plan):
    calls = []

    def provider(*_):
        calls.append(1)
        return response("run_shell", "bash")

    result = D.run_case(plan, plan["cases"][0], provider)
    assert result["status"] == "hold" and len(calls) == 1
    assert result["decisions"][0]["model_status"] == "invalid"
    assert not (Path(plan["output"]) / "sample_01/probe-request.json").exists()


def test_driver_pending_request_survives_parent_restart(plan):
    def provider(*_):
        raise D.ProviderPending("자식 응답 미확정")

    with pytest.raises(D.ProviderPending):
        D.run_case(plan, plan["cases"][0], provider)
    replay = D.run_case(plan, plan["cases"][0], lambda *_: pytest.fail("미확정 요청 재전송"))
    assert replay["status"] == "blocked_pending" and replay["api"]["pending"] == 1


def test_index_tamper_is_rejected_before_provider(plan):
    Path(plan["index"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="색인"):
        D.run_case(plan, plan["cases"][0], lambda *_: pytest.fail("변조 뒤 provider"))


def test_driver_budget_hold_is_unavailable_not_model_invalid(plan):
    plan["api_cap"] = 1  # 하위 고리의 예산 소진 경계. 운영 execute는 등록된12만 허용한다.
    calls = []

    def provider(*_):
        calls.append(1)
        return response("read_evidence", "evidence:training-01")

    result = D.run_case(plan, plan["cases"][0], provider)
    assert result["status"] == "hold" and len(calls) == 1
    assert result["decisions"][-1]["model_status"] == "unavailable"


def test_source_blob_mismatch_is_rejected():
    anchor = __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout.strip()
    with pytest.raises(ValueError, match="blob"):
        D.verify_sources({"sources_lf": {"evals/agentic_investigation_loop_v1.py": "0" * 64}}, anchor)


@pytest.mark.parametrize("changed", ["pending", "allocation", "index"])
def test_execution_checks_phase_budget_and_bound_input_before_provider(plan, changed):
    root = Path(plan["output"]).parent
    audit = root / "audit.json"
    D.J._write_new(audit, {"known_pending": int(changed == "pending")})
    plan["budget"] = {"original_core_allocation_cap": 120, "new_phase_cap": 12,
                      "combined_allocation_cap": 133 if changed == "allocation" else 132,
                      "prior_audit": {"path": str(audit), "sha256": D.J.file_sha(audit)}}
    protocol = {"cases": plan["cases"], "goal": plan["goal"],
                "input_index_raw_sha256": "0" * 64 if changed == "index" else plan["index"]["sha256"],
                "input_index_canonical_sha256": plan["index_canonical_sha256"],
                "budget": {"combined_allocation_cap": 132}}
    pp = root / "protocol.json"
    D.J._write_new(pp, protocol)
    plan["protocol"] = {"path": str(pp), "sha256": D.J.file_sha(pp)}
    path = root / "plan.json"
    D.J._write_new(path, plan)
    with pytest.raises(ValueError, match="변경|배정"):
        D.execute(path, lambda *_: pytest.fail("계약 위반 뒤 provider"))
