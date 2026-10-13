"""공식 Isaac Lab 디버깅 스킬 텍스트를 넣은 프롬프트와 뺀 프롬프트를 같은 고정 입력으로 만들고 응답을 규칙으로 채점한다.

  python evals/skill_ab.py build --case h02_s2027 --output <prompts.json>
  python evals/skill_ab.py score --prompts <prompts.json> --responses <responses.jsonl> --output <scores.json>

이 파일은 모델을 호출하지 않는다. 응답 파일 한 줄은 {"prompt_id": "S1:with_skill", "repeat": 0, "text": "<모델 원문>"}이다.
채점 기준은 스킬의 evaluations.md(시나리오 3개와 Known failure modes)이며, 이 파일은 프롬프트에 넣지 않는다.
응답의 계획 단계 enum으로 판정 가능한 항목만 규칙으로 채점하고 설명의 타당성은 사람 채점 필요로 남긴다.
채점은 모델이 진술한 계획이며 실행한 행동이 아니다. 결과는 개수와 분모만 보고하고 우위 주장에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rl_triage.skill_report import SKILL_DIR, sha256  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

SCHEMA_PROMPTS = "skill_ab_prompts_v1"
SCHEMA_SCORES = "skill_ab_scores_v1"
SKILL_FILES_IN_PROMPT = ("SKILL.md", "reference.md", "examples.md")
GRADER_FILE = "evaluations.md"
VARIANTS = ("without_skill", "with_skill")
STEP_ACTIONS = (
    "inspect_task_metric", "audit_reward_terms", "check_reward_geometry", "check_observations",
    "deterministic_rollout", "state_trace", "change_ppo_hyperparameters", "compare_obs_action_dims",
    "classify_environment_contract_change", "patch_runner_ignore_shapes", "declare_checkpoint_corrupt",
    "revert_contract_change", "retrain", "control_run", "one_variable_ablation", "define_success_metric",
    "select_checkpoint", "other",
)
ASSESSMENTS = ("successful", "not_successful", "undetermined")
RUN_STATUSES = ("trusted", "exploratory", "undetermined")
SELECTION_BASES = ("total_reward", "task_metric", "rollout_behavior", "stability")
RESPONSE_KEYS = {"run_assessment", "run_status", "attributed_change", "steps", "checkpoint_selection_basis",
                 "next_run_changes", "explanation"}
HUMAN_GRADING = ("explanation is consistent with the case packet", "step details are specific and correct")
SYSTEM_BASE = """You assist an engineer who debugs Isaac Lab reinforcement learning training.
Read the fixed case packet and answer the engineer's question.
Answer with one JSON object and nothing else. Use exactly these keys:
- "run_assessment": one of {assessments}
- "run_status": one of {statuses}
- "attributed_change": null, or the single change you attribute the behavior to
- "steps": an ordered list of the next actions, each {{"action": one of {actions}, "detail": short text}}
- "checkpoint_selection_basis": a list using only {bases}
- "next_run_changes": the list of variables the next training run would change
- "explanation": short text
""".format(assessments=json.dumps(ASSESSMENTS), statuses=json.dumps(RUN_STATUSES),
           actions=json.dumps(STEP_ACTIONS), bases=json.dumps(SELECTION_BASES))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 채점 기준

def parse_evaluations(text: str) -> list[dict]:
    scenarios = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        title = block.splitlines()[0].strip()
        query = re.search(r'^Query: "(.+)"\s*$', block, re.M)
        expected = re.search(r"Expected behavior:\s*\n((?:- .+\n?)+)", block)
        failures = re.search(r"Known failure modes:\s*\n((?:- .+\n?)+)", block)
        if not (query and expected and failures):
            raise ValueError(f"evaluation scenario is incomplete: {title}")
        bullets = (lambda body: [line[2:].strip() for line in body.strip().splitlines()])
        scenarios.append({"id": f"S{len(scenarios) + 1}", "title": title, "query": query[1],
                          "expected": bullets(expected[1]), "failure_modes": bullets(failures[1])})
    if len(scenarios) != 3:
        raise ValueError("pinned evaluations.md must contain three scenarios")
    return scenarios


def _actions(response: dict) -> list[str]:
    return [step["action"] for step in response["steps"]]


def _before(response: dict, first: str, others: tuple[str, ...]) -> bool:
    """first가 나오고, 그 앞에 others 중 어느 것도 없으면 True."""
    actions = _actions(response)
    if first not in actions:
        return False
    return not any(action in others for action in actions[:actions.index(first)])


def _present(response: dict, *names: str) -> bool:
    return any(name in _actions(response) for name in names)


# 실패 모드 규칙: True = 실패 모드에 해당. 문구는 evaluations.md 원문과 같아야 한다.
FAILURE_RULES: dict[str, list[tuple[str, Callable[[dict], bool]]]] = {
    "S1": [
        ("Declares the run successful from reward alone.",
         lambda r: r["run_assessment"] == "successful"),
        ("Changes PPO hyperparameters before checking reward and observation semantics.",
         lambda r: _before(r, "change_ppo_hyperparameters", ("audit_reward_terms", "check_reward_geometry", "check_observations"))),
    ],
    "S2": [
        ("Patches the runner to ignore shape mismatches.",
         lambda r: _present(r, "patch_runner_ignore_shapes")),
        ("Assumes the checkpoint is corrupt before checking environment dimensions.",
         lambda r: _before(r, "declare_checkpoint_corrupt", ("compare_obs_action_dims",))),
    ],
    "S3": [
        ("Attributes behavior to one change without controls.",
         lambda r: r["attributed_change"] is not None and not _present(r, "control_run", "one_variable_ablation")),
        ("Chooses checkpoints only from total reward.",
         lambda r: r["checkpoint_selection_basis"] == ["total_reward"]),
    ],
}
# 기대 행동 규칙: True = 진술한 계획에 해당 단계가 있음. 규칙으로 판정할 수 없는 기대 행동은 None.
EXPECTED_RULES: dict[str, list[tuple[str, Callable[[dict], bool] | None]]] = {
    "S1": [
        ("Separates reward curves from task success metrics.", lambda r: _present(r, "inspect_task_metric")),
        ("Audits per-term rewards against task phases.", lambda r: _present(r, "audit_reward_terms")),
        ("Checks success reward, termination, and metric geometry.", lambda r: _present(r, "check_reward_geometry")),
        ("Recommends deterministic rollout or state traces before retuning PPO.",
         lambda r: _present(r, "deterministic_rollout", "state_trace")
         and not _before(r, "change_ppo_hyperparameters", ("deterministic_rollout", "state_trace"))),
    ],
    "S2": [
        ("Compares current observation/action spaces with the checkpoint training config.",
         lambda r: _present(r, "compare_obs_action_dims")),
        ("Treats shape mismatch as an environment-contract issue.",
         lambda r: _present(r, "classify_environment_contract_change")),
        ("Recommends retraining or reverting unintended contract changes.",
         lambda r: _present(r, "retrain", "revert_contract_change")),
    ],
    "S3": [
        ("Marks the run as exploratory because multiple variables changed.", lambda r: r["run_status"] == "exploratory"),
        ("Recommends control runs or one-variable ablations.", lambda r: _present(r, "control_run", "one_variable_ablation")),
        ("Defines success metrics before launching the next run.",
         lambda r: _present(r, "define_success_metric") and not _before(r, "retrain", ("define_success_metric",))),
    ],
}


def check_rules_match(scenarios: list[dict]) -> None:
    for scenario in scenarios:
        if [text for text, _ in FAILURE_RULES[scenario["id"]]] != scenario["failure_modes"]:
            raise ValueError(f"failure rules differ from the pinned evaluations.md: {scenario['id']}")
        if [text for text, _ in EXPECTED_RULES[scenario["id"]]] != scenario["expected"]:
            raise ValueError(f"expected-behavior rules differ from the pinned evaluations.md: {scenario['id']}")


# ---------------------------------------------------------------- 고정 입력

def case_packet(case: str, root: Path = ROOT) -> dict:
    """공개 기록에서 만든 사례 사실. 결함 정체·진단 결과·교정은 넣지 않는다."""
    seed = re.fullmatch(r".+_s(\d{4})", case)[1]
    tele = json.loads((root / f"bench/runs_p0c/{case}.telemetry.json").read_text(encoding="utf-8"))
    report = json.loads((root / f"evals/results/p0c_fixed_eval_20261009/runs/{case}.json").read_text(encoding="utf-8"))
    env = yaml.load((root / f"bench/reference/params/p0c_seed{seed}/env.yaml").read_text(encoding="utf-8"), Loader=_TolerantLoader)
    summary = tele["summary"]

    def first_last(tag: str) -> dict:
        return {"first_20pct_mean": round(summary[tag]["mean_first_20pct"], 4),
                "last_20pct_mean": round(summary[tag]["mean_last_20pct"], 4)}

    policy = env["observations"]["policy"]
    return {
        "task": report["task"], "rl_library": f"rsl_rl {report['versions']['rsl-rl-lib']}",
        "training": {"iterations": summary["Train/mean_reward"]["n"], "num_envs": env["scene"]["num_envs"], "seed": env["seed"],
                     "mean_reward": first_last("Train/mean_reward"),
                     "mean_episode_length_steps": first_last("Train/mean_episode_length"),
                     "logged_linear_tracking_error": first_last("Metrics/base_velocity/error_vel_xy"),
                     "linear_tracking_reward_term": first_last("Episode_Reward/track_lin_vel_xy_exp")},
        "fixed_evaluation": {"commands": 26, "envs": report["num_envs"], "steps": report["horizon_steps"],
                             "linear_velocity_rmse_mps": round(report["metrics"]["lin_vel_rmse_mps"], 4),
                             "standing_still_linear_velocity_rmse_mps": round(report["standstill"]["lin_vel_rmse_mps"], 4),
                             "yaw_rate_rmse_radps": round(report["metrics"]["yaw_rate_rmse_radps"], 4),
                             "fall_rate": round(report["metrics"]["fall_rate"], 4)},
        "environment_contract": {
            "observation_terms": [name for name, term in policy.items() if isinstance(term, dict) and "func" in term],
            "action": "joint position targets for all joints, scale 0.25",
            "reward_terms": {name: term["weight"] for name, term in env["rewards"].items() if isinstance(term, dict) and term.get("weight")},
            "termination_terms": list(env["terminations"]),
        },
    }


def build_prompts(case: str, root: Path = ROOT) -> dict:
    folder = root / SKILL_DIR
    scenarios = parse_evaluations((folder / GRADER_FILE).read_text(encoding="utf-8"))
    check_rules_match(scenarios)
    skill_block = "\n\n".join(
        f"===== {name} =====\n{(folder / name).read_text(encoding='utf-8')}" for name in SKILL_FILES_IN_PROMPT)
    systems = {"without_skill": SYSTEM_BASE,
               "with_skill": SYSTEM_BASE + "\nReference material (Isaac Lab skill isaaclab-debugging-rl-training):\n\n" + skill_block}
    packet = case_packet(case, root)
    packet_text = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True)
    prompts = []
    for scenario in scenarios:
        user = f"Case packet (fixed):\n{packet_text}\n\nQuestion:\n{scenario['query']}"
        for variant in VARIANTS:
            prompts.append({"prompt_id": f"{scenario['id']}:{variant}", "scenario": scenario["id"], "variant": variant,
                            "system": systems[variant], "user": user,
                            "system_sha256": _sha_text(systems[variant]), "user_sha256": _sha_text(user)})
    return {
        "schema": SCHEMA_PROMPTS, "case": case, "model_calls": 0,
        "packet": packet, "packet_sha256": _sha_text(packet_text),
        "skill_files_in_prompt": {name: sha256(folder / name) for name in SKILL_FILES_IN_PROMPT},
        "grader_file_not_in_prompt": {GRADER_FILE: sha256(folder / GRADER_FILE)},
        "scenarios": [{key: value for key, value in scenario.items()} for scenario in scenarios],
        "prompts": prompts,
        "scope": "스킬 텍스트 포함 여부만 다른 프롬프트 쌍이다. 진술한 계획을 채점하며 실행 행동이나 우위의 근거가 아니다.",
    }


# ---------------------------------------------------------------- 채점

def parse_response(text: str) -> tuple[dict | None, str | None]:
    """응답에서 JSON 객체 하나를 꺼내 스키마를 확인한다. 실패하면 채점 불가 사유를 돌려준다."""
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
    if fenced:
        body = fenced[1]
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None, "not_json"
    if not isinstance(value, dict) or set(value) != RESPONSE_KEYS:
        return None, "keys_differ"
    steps = value["steps"]
    if (value["run_assessment"] not in ASSESSMENTS or value["run_status"] not in RUN_STATUSES
            or not (value["attributed_change"] is None or isinstance(value["attributed_change"], str))
            or not isinstance(steps, list) or not steps
            or any(not isinstance(step, dict) or step.get("action") not in STEP_ACTIONS for step in steps)
            or not isinstance(value["checkpoint_selection_basis"], list)
            or any(item not in SELECTION_BASES for item in value["checkpoint_selection_basis"])
            or not isinstance(value["next_run_changes"], list) or not isinstance(value["explanation"], str)):
        return None, "schema_differs"
    return value, None


def score_response(scenario: str, text: str) -> dict:
    parsed, reason = parse_response(text)
    if parsed is None:
        return {"scorable": False, "unscorable_reason": reason, "failure_modes": None, "expected": None,
                "human_grading_required": list(HUMAN_GRADING)}
    return {"scorable": True, "unscorable_reason": None,
            "failure_modes": {name: rule(parsed) for name, rule in FAILURE_RULES[scenario]},
            "expected": {name: (rule(parsed) if rule else None) for name, rule in EXPECTED_RULES[scenario]},
            "next_run_change_count": len(parsed["next_run_changes"]),
            "human_grading_required": list(HUMAN_GRADING)}


def score(prompts: dict, responses: list[dict]) -> dict:
    by_id = {row["prompt_id"]: row for row in prompts["prompts"]}
    rows, seen = [], set()
    for response in responses:
        key = (response["prompt_id"], response["repeat"])
        if response["prompt_id"] not in by_id or key in seen:
            raise ValueError(f"unknown or duplicated response: {key}")
        seen.add(key)
        prompt = by_id[response["prompt_id"]]
        rows.append({"prompt_id": prompt["prompt_id"], "scenario": prompt["scenario"], "variant": prompt["variant"],
                     "repeat": response["repeat"], "response_sha256": _sha_text(response["text"]),
                     **score_response(prompt["scenario"], response["text"])})
    summary = {}
    for variant in VARIANTS:
        mine = [row for row in rows if row["variant"] == variant]
        scorable = [row for row in mine if row["scorable"]]
        summary[variant] = {
            "responses": len(mine), "scorable": len(scorable), "unscorable": len(mine) - len(scorable),
            "failure_mode_hits": sum(value for row in scorable for value in row["failure_modes"].values()),
            "failure_mode_checks": sum(len(row["failure_modes"]) for row in scorable),
            "expected_hits": sum(bool(value) for row in scorable for value in row["expected"].values()),
            "expected_checks": sum(len(row["expected"]) for row in scorable),
        }
    return {"schema": SCHEMA_SCORES, "case": prompts["case"], "packet_sha256": prompts["packet_sha256"],
            "rows": rows, "summary": summary,
            "note": "개수와 분모만 보고한다. 표본이 작고 진술한 계획의 채점이라 스킬의 효과나 우위의 근거로 쓰지 않는다."}


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--case", default="h02_s2027")
    build_parser.add_argument("--output", type=Path, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--prompts", type=Path, required=True)
    score_parser.add_argument("--responses", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_prompts(args.case)
        _write_new(args.output, result)
        print(json.dumps({"prompts": len(result["prompts"]), "packet_sha256": result["packet_sha256"]}))
    else:
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        responses = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(prompts, responses)
        _write_new(args.output, result)
        print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
