"""스킬 유무 A/B 사전등록·WSL worker·호출 계획·요약의 결합과 분모 규칙."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import skill_ab as AB  # noqa: E402
import skill_ab_call as W  # noqa: E402
import skill_ab_compare as C  # noqa: E402

PROMPT_IDS = sorted(f"{s}:{v}" for s in C.SCENARIOS for v in AB.VARIANTS)
GOOD = json.dumps({"run_assessment": "not_successful", "run_status": "exploratory", "attributed_change": None,
                   "steps": [{"action": a, "detail": "d"} for a in (
                       "define_success_metric", "inspect_task_metric", "audit_reward_terms", "check_reward_geometry",
                       "deterministic_rollout", "compare_obs_action_dims", "classify_environment_contract_change",
                       "control_run", "one_variable_ablation", "revert_contract_change", "retrain")],
                   "checkpoint_selection_basis": ["task_metric", "rollout_behavior"], "next_run_changes": ["one"],
                   "explanation": "x"})
BAD = json.dumps({"run_assessment": "successful", "run_status": "trusted", "attributed_change": "reward",
                  "steps": [{"action": a, "detail": "d"} for a in (
                      "change_ppo_hyperparameters", "declare_checkpoint_corrupt", "patch_runner_ignore_shapes", "retrain")],
                  "checkpoint_selection_basis": ["total_reward"], "next_run_changes": ["a", "b"], "explanation": "x"})


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__("provider said no")
        self.status_code = status_code


def fake_client(content="{}", finish="stop", error: Exception | None = None):
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(id="resp-1", model=W.MODEL,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return (lambda: client), sent


@pytest.fixture(scope="module")
def prompts():
    return AB.build_prompts(C.CASE)


@pytest.fixture
def prompts_file(tmp_path, prompts):
    path = tmp_path / "prompts.json"
    path.write_text(C.dump(prompts), encoding="utf-8")
    return path


@pytest.fixture
def credential(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-placeholder")


# --- 프로토콜 ---
def test_call_order_covers_every_prompt_once_per_repeat_with_shuffled_order():
    order = C.call_order()
    assert order == C.FIXED["call_order"] and [e["call"] for e in order] == list(range(1, 31))
    by_repeat = [[e["prompt_id"] for e in order if e["repeat"] == r] for r in range(1, C.REPEATS + 1)]
    assert all(sorted(ids) == PROMPT_IDS for ids in by_repeat)
    assert len({tuple(ids) for ids in by_repeat}) > 1


def test_protocol_fixes_model_request_rules_and_prompt_hashes():
    dynamic = C.dynamic_fields()
    protocol = {**copy.deepcopy(C.FIXED), **dynamic}
    C.check_protocol(protocol, dynamic)
    assert protocol["request"] == {"model": "nvidia/nemotron-3-super-120b-a12b", "temperature": 0, "max_tokens": 16384,
                                   "stream": False, "extra_body": {"reasoning_effort": "none"}}
    assert (protocol["repeats"], protocol["calls"], sorted(protocol["prompts"])) == (5, 30, PROMPT_IDS)
    for scenario in C.SCENARIOS:
        pair = [protocol["prompts"][f"{scenario}:{v}"] for v in AB.VARIANTS]
        assert pair[0]["user_sha256"] == pair[1]["user_sha256"] and pair[0]["system_sha256"] != pair[1]["system_sha256"]
    pinned = AB.parse_evaluations((ROOT / AB.SKILL_DIR / AB.GRADER_FILE).read_text(encoding="utf-8"))
    assert protocol["scoring"]["rules"]["failure_modes"] == {s["id"]: s["failure_modes"] for s in pinned}
    assert protocol["scoring"]["rules"]["expected"] == {s["id"]: s["expected"] for s in pinned}
    assert set(protocol["source_files"]) == C.CODE and AB.GRADER_FILE not in protocol["skill_files_in_prompt"]
    for key, value in (("repeats", 3), ("temperature", 0), ("wording", {})):
        with pytest.raises(ValueError):
            C.check_protocol({**protocol, key: value}, dynamic)
    changed = copy.deepcopy(protocol)
    changed["prompts"]["S1:with_skill"]["system_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound"):
        C.check_protocol(changed, dynamic)


def test_prepared_protocol_file_matches_code_when_present():
    if not C.PROTOCOL.is_file():
        pytest.skip("protocol not prepared yet")
    C.check_protocol(C.read(C.PROTOCOL), C.dynamic_fields())


def test_fixed_wording_has_no_superiority_claim_and_checker_catches_one():
    for text in (C.FIXED["wording"]["result"], C.FIXED["wording"]["common"], C.FIXED["wording"]["incomplete"]):
        assert C.publish_violations(text) == []
    assert C.publish_violations("스킬을 넣으면 계획이 개선된다.")
    assert C.publish_violations("스킬 포함 조건이 우위다.")
    assert C.publish_violations({"note": "C:\\Users\\someone\\calls"})


# --- WSL worker ---
def test_worker_records_one_answer_with_registered_request(tmp_path, prompts_file, prompts, credential):
    factory, sent = fake_client(content=GOOD)
    out = tmp_path / "calls/r1_S1-with_skill_a1.json"
    assert W.call(prompts_file, "S1:with_skill", 1, 1, out, client_factory=factory, require_posix=False) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    row = next(r for r in prompts["prompts"] if r["prompt_id"] == "S1:with_skill")
    assert sent == [{"messages": [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
                     **W.REQUEST}]
    assert (record["status"], record["system_sha256"], record["user_sha256"]) == ("answered", row["system_sha256"],
                                                                                  row["user_sha256"])
    assert record["prompts_sha256"] == C.sha_raw(prompts_file) and record["content"] == GOOD
    assert "test-placeholder" not in out.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        W.call(prompts_file, "S1:with_skill", 1, 1, out, client_factory=factory, require_posix=False)


@pytest.mark.parametrize(("kwargs", "status", "retryable"), [
    ({"content": ""}, "infra_error", True),
    ({"content": GOOD, "finish": "length"}, "infra_error", True),
    ({"error": APITimeoutError()}, "infra_error", True),
    ({"error": APIStatusError(503)}, "infra_error", True),
    ({"error": APIStatusError(429)}, "infra_error", True),
    ({"error": APIStatusError(400)}, "call_error", False),
])
def test_worker_separates_infra_errors_from_other_call_errors(tmp_path, prompts_file, credential, kwargs, status, retryable):
    factory, _ = fake_client(**kwargs)
    out = tmp_path / "r.json"
    assert W.call(prompts_file, "S2:without_skill", 1, 1, out, client_factory=factory, require_posix=False) == 2
    record = json.loads(out.read_text(encoding="utf-8"))
    assert (record["status"], record["retryable"]) == (status, retryable)
    assert "provider said no" not in out.read_text(encoding="utf-8")


def test_worker_refuses_without_writing_when_environment_or_input_is_wrong(tmp_path, prompts_file, monkeypatch):
    factory, sent = fake_client(content=GOOD)
    out = tmp_path / "r.json"
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        W.call(prompts_file, "S1:with_skill", 1, 1, out, client_factory=factory, require_posix=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-placeholder")
    monkeypatch.setattr(W.os, "name", "nt")
    with pytest.raises(RuntimeError, match="WSL"):
        W.call(prompts_file, "S1:with_skill", 1, 1, out, client_factory=factory)
    monkeypatch.undo()
    monkeypatch.setenv("NVIDIA_API_KEY", "test-placeholder")
    with pytest.raises(ValueError):
        W.call(prompts_file, "S9:with_skill", 1, 1, out, client_factory=factory, require_posix=False)
    with pytest.raises(ValueError):
        W.call(prompts_file, "S1:with_skill", 1, 3, out, client_factory=factory, require_posix=False)
    assert not out.exists() and sent == []


def test_worker_reads_credentials_only_from_the_environment():
    source = (ROOT / "evals/skill_ab_call.py").read_text(encoding="utf-8")
    assert ".config" not in source and "nvidia/env" not in source
    assert "import yaml" not in source and "rl_triage" not in source


# --- 계획과 요약 ---
@pytest.fixture
def workspace(tmp_path, monkeypatch):
    protocol = {**copy.deepcopy(C.FIXED), **C.dynamic_fields()}
    protocol_path = tmp_path / "skill_ab_v1.json"
    protocol_path.write_text(C.dump(protocol), encoding="utf-8")
    monkeypatch.setattr(C, "PROTOCOL", protocol_path)
    monkeypatch.setattr(C, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(C, "RESULT", tmp_path / "results/skill_ab_v1/summary.json")
    monkeypatch.setattr(C, "verify_protocol", lambda: (protocol, "c" * 40))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-placeholder")
    return protocol


def run_call(entry: dict, attempt: int = 1, **kwargs) -> None:
    content = GOOD if entry["prompt_id"].endswith(":with_skill") else BAD
    factory, _ = fake_client(**({"content": content} | kwargs))
    W.call(C.PRIVATE / "prompts.json", entry["prompt_id"], entry["repeat"], attempt,
           C.record_path(entry["repeat"], entry["prompt_id"], attempt), client_factory=factory, require_posix=False)


def test_plan_lists_registered_order_then_only_remaining_and_retries(workspace):
    first = C.plan()
    order = workspace["call_order"]
    assert (first["calls"], first["remaining"], first["retries_pending"]) == (30, 30, 0)
    assert all(f"--prompt-id {e['prompt_id']} --repeat {e['repeat']} --attempt 1 " in cmd
               for e, cmd in zip(order, first["commands"]))
    assert all(cmd.count('. "$HOME/.config/nvidia/env"') == 1 for cmd in first["commands"])
    assert (C.PRIVATE / "prompts.json").is_file() and (C.PRIVATE / "binding.json").is_file()
    run_call(order[0])
    run_call(order[1], error=APITimeoutError())
    second = C.plan()
    assert (second["remaining"], second["retries_pending"]) == (29, 1)
    assert "--attempt 2" in second["commands"][0] and second["commands"][0].endswith(
        f"{C.record_path(order[1]['repeat'], order[1]['prompt_id'], 2).name}'")
    (C.PRIVATE / "prompts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts.json"):
        C.plan()


def test_summary_keeps_unscorable_items_in_denominators_and_writes_public_counts(workspace):
    C.plan()
    order = workspace["call_order"]
    unscorable = next(e for e in order if e["prompt_id"] == "S1:without_skill")
    retried = next(e for e in order if e["prompt_id"] == "S3:with_skill")
    for entry in order:
        if entry is unscorable:
            run_call(entry, content="I think the run is fine.")
        elif entry is retried:
            run_call(entry, error=APITimeoutError())
        else:
            run_call(entry)
    assert C.summarize()["status"] == "incomplete" and not C.RESULT.exists()
    run_call(retried, attempt=2)
    assert C.summarize()["status"] == "written"
    result = json.loads(C.RESULT.read_text(encoding="utf-8"))
    with_skill, without = result["variants"]["with_skill"], result["variants"]["without_skill"]
    assert (with_skill["answered"], with_skill["scorable"], with_skill["infra_retries"]) == (15, 15, 1)
    assert with_skill["failure_modes"] == {"items": 30, "rule_judged": 30, "hits": 0, "human_grading_required": 0}
    assert with_skill["expected"] == {"items": 50, "rule_judged": 50, "hits": 50, "human_grading_required": 0}
    assert (without["answered"], without["scorable"], without["unscorable_reasons"]) == (15, 14, {"not_json": 1})
    assert without["failure_modes"] == {"items": 30, "rule_judged": 28, "hits": 28, "human_grading_required": 2}
    assert without["expected"] == {"items": 50, "rule_judged": 46, "hits": 5, "human_grading_required": 4}
    assert without["explanation_items_human_grading_required"] == 30
    assert result["failure_hit_difference_with_minus_without"] == -28
    assert "0/30건" in result["wording"] and "28/28건" in result["wording"] and "-28건" in result["wording"]
    assert C.publish_violations(result) == [] and result["human_grading"]["status"] == "pending"
    assert len(result["rows"]) == 30 and all("content" not in row for row in result["rows"])
    with pytest.raises(FileExistsError):
        C.summarize()


def test_summary_refuses_call_errors_and_tampered_or_out_of_order_records(workspace):
    C.plan()
    order = workspace["call_order"]
    for entry in order[1:]:
        run_call(entry)
    run_call(order[0], error=APIStatusError(401))
    out = C.summarize()
    assert (out["status"], out["call_errors"]) == ("call_error", [1]) and not C.RESULT.exists()
    run_call(order[1], attempt=2)
    with pytest.raises(ValueError, match="재시도"):
        C.summarize()
    C.record_path(order[1]["repeat"], order[1]["prompt_id"], 2).unlink()
    path = C.record_path(order[2]["repeat"], order[2]["prompt_id"], 1)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["system_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="등록 프롬프트"):
        C.summarize()
