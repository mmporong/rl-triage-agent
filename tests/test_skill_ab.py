"""스킬 포함·미포함 프롬프트 쌍의 결정성·누설 경계와 규칙 채점."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import build_replay_page as PAGE  # noqa: E402
import skill_ab as AB  # noqa: E402

PREREG = ROOT / "bench/private/claude_reward_track_20261013/skill_ab_prereg.md"


def _response(**changes) -> str:
    value = {"run_assessment": "not_successful", "run_status": "undetermined", "attributed_change": None,
             "steps": [{"action": "inspect_task_metric", "detail": "x"}], "checkpoint_selection_basis": ["task_metric"],
             "next_run_changes": ["one"], "explanation": "x"}
    value.update(changes)
    return json.dumps(value)


def _steps(*actions):
    return [{"action": action, "detail": "d"} for action in actions]


def test_prompt_pairs_are_deterministic_and_differ_only_by_skill_block():
    first, second = AB.build_prompts("h02_s2027"), AB.build_prompts("h02_s2027")
    assert first == second and len(first["prompts"]) == 6
    by_id = {row["prompt_id"]: row for row in first["prompts"]}
    skill_text = (ROOT / AB.SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for scenario in ("S1", "S2", "S3"):
        plain, skilled = by_id[f"{scenario}:without_skill"], by_id[f"{scenario}:with_skill"]
        assert plain["user"] == skilled["user"] and skilled["system"].startswith(plain["system"])
        assert skill_text in skilled["system"] and skill_text not in plain["system"]
    assert set(first["skill_files_in_prompt"]) == {"SKILL.md", "reference.md", "examples.md"}


def test_grader_text_and_case_answer_never_enter_prompts():
    prompts = AB.build_prompts("h02_s2027")
    joined = "\n".join(row["system"] + row["user"] for row in prompts["prompts"])
    for leaked in ("Known failure modes", "Declares the run successful from reward alone", "Scenario 1: Reward Hacking",
                   "h02", "F1", "P_reward", "restore_", "body frame"):
        assert leaked not in joined, leaked


def test_rules_are_bound_to_the_pinned_evaluation_text(tmp_path):
    scenarios = AB.parse_evaluations((ROOT / AB.SKILL_DIR / AB.GRADER_FILE).read_text(encoding="utf-8"))
    AB.check_rules_match(scenarios)
    scenarios[0]["failure_modes"][0] = "Declares success."
    with pytest.raises(ValueError, match="failure rules differ"):
        AB.check_rules_match(scenarios)


@pytest.mark.parametrize(("scenario", "changes", "hits"), [
    ("S1", {"run_assessment": "successful", "steps": _steps("change_ppo_hyperparameters", "audit_reward_terms")},
     [True, True]),
    ("S1", {"steps": _steps("audit_reward_terms", "change_ppo_hyperparameters")}, [False, False]),
    ("S2", {"steps": _steps("declare_checkpoint_corrupt", "compare_obs_action_dims", "patch_runner_ignore_shapes")},
     [True, True]),
    ("S2", {"steps": _steps("compare_obs_action_dims", "declare_checkpoint_corrupt")}, [False, False]),
    ("S3", {"attributed_change": "reward", "checkpoint_selection_basis": ["total_reward"]}, [True, True]),
    ("S3", {"attributed_change": "reward", "steps": _steps("control_run"),
            "checkpoint_selection_basis": ["total_reward", "task_metric"]}, [False, False]),
])
def test_failure_mode_rules(scenario, changes, hits):
    result = AB.score_response(scenario, _response(**changes))
    assert result["scorable"] is True and list(result["failure_modes"].values()) == hits


def test_expected_behavior_rules_for_a_complete_plan():
    s1 = AB.score_response("S1", _response(steps=_steps("inspect_task_metric", "audit_reward_terms", "check_reward_geometry",
                                                        "deterministic_rollout", "change_ppo_hyperparameters")))
    assert all(s1["expected"].values())
    s3 = AB.score_response("S3", _response(run_status="exploratory",
                                           steps=_steps("define_success_metric", "one_variable_ablation", "retrain")))
    assert all(s3["expected"].values())
    late = AB.score_response("S3", _response(run_status="exploratory", steps=_steps("retrain", "define_success_metric")))
    assert list(late["expected"].values()) == [True, False, False]


@pytest.mark.parametrize(("text", "reason"), [
    ("I think the run is fine.", "not_json"),
    (json.dumps({"run_assessment": "successful"}), "keys_differ"),
    (_response(steps=_steps("tune_everything")), "schema_differs"),
])
def test_unscorable_responses_are_counted_not_guessed(text, reason):
    result = AB.score_response("S1", text)
    assert (result["scorable"], result["unscorable_reason"], result["failure_modes"]) == (False, reason, None)
    assert result["human_grading_required"] == list(AB.HUMAN_GRADING)


def test_fenced_json_is_accepted():
    assert AB.score_response("S2", "```json\n" + _response() + "\n```")["scorable"] is True


def test_score_summary_keeps_unscorable_in_denominator_and_rejects_duplicates():
    prompts = AB.build_prompts("h02_s2027")
    responses = [{"prompt_id": "S1:with_skill", "repeat": 0, "text": _response(run_assessment="successful")},
                 {"prompt_id": "S1:without_skill", "repeat": 0, "text": "not json"}]
    result = AB.score(prompts, responses)
    assert result["summary"]["with_skill"]["failure_mode_hits"] == 1
    assert result["summary"]["without_skill"] == {"responses": 1, "scorable": 0, "unscorable": 1, "failure_mode_hits": 0,
                                                  "failure_mode_checks": 0, "expected_hits": 0, "expected_checks": 0}
    with pytest.raises(ValueError, match="duplicated"):
        AB.score(prompts, responses + [responses[0]])
    with pytest.raises(ValueError, match="unknown"):
        AB.score(prompts, [{"prompt_id": "S9:with_skill", "repeat": 0, "text": "{}"}])


def test_harness_and_prereg_text_avoid_forbidden_claims():
    prompts = AB.build_prompts("h02_s2027")
    texts = [prompts["scope"], AB.score(prompts, [])["note"]]
    if PREREG.is_file():
        prereg = PREREG.read_text(encoding="utf-8")
        assert "직접" not in prereg
        texts += [line for line in prereg.splitlines() if not line.startswith("- 쓰지 않는 문장")]
    assert not [problem for text in texts for problem in PAGE.claim_violations(text)]
    # JSON 직렬화의 "\n" 이스케이프가 드라이브 문자로 오인되지 않도록 원문 문자열마다 검사한다.
    for row in prompts["prompts"]:
        assert not PAGE.path_violations(row["system"]) and not PAGE.path_violations(row["user"])
