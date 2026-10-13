"""스킬 유무 A/B(skill_ab_v1)의 사전등록·호출 계획·기록 재검증·공개 요약.

모델을 호출하지 않는다. 호출은 WSL에서 evals/skill_ab_call.py로 하고, 이 파일은 프롬프트·채점 규칙 결합,
호출 순서, 재시도 판정, 지표 계산만 맡는다. 결과는 개수와 분모의 기술이며 스킬의 효과·우위를 주장하지 않는다.

  python evals/skill_ab_compare.py prepare     # 프로토콜 작성. 커밋한 뒤 호출한다
  python evals/skill_ab_compare.py verify      # 프로토콜과 결합 파일이 커밋돼 있고 바뀌지 않았는지
  python evals/skill_ab_compare.py plan        # 남은 WSL 호출 명령(등록 순서)과 인프라 재시도
  python evals/skill_ab_compare.py summarize   # 30회가 끝나면 evals/results/skill_ab_v1/summary.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import skill_ab as AB  # noqa: E402
import skill_ab_call as W  # noqa: E402
from rl_triage.skill_report import SKILL_DIR  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/skill_ab_v1.json"
PRIVATE = ROOT / "bench/private/skill_ab_v1"
RESULT = ROOT / "evals/results/skill_ab_v1/summary.json"
CASE = "h02_s2027"
SCENARIOS = ("S1", "S2", "S3")
REPEATS = 5
ORDER_SEED = 20261013
CODE = {"evals/skill_ab.py", "evals/skill_ab_call.py", "evals/skill_ab_compare.py",
        *(f"{SKILL_DIR.as_posix()}/{name}" for name in (*AB.SKILL_FILES_IN_PROMPT, AB.GRADER_FILE))}
RESULT_WORDS_NEED_NEGATION = ("우위", "개선", "향상", "높였", "효과", "낫")


def call_order() -> list[dict]:
    """반복마다 6개 프롬프트의 순서를 고정 seed로 섞는다. 프로토콜에 목록 그대로 남긴다."""
    ids = [f"{scenario}:{variant}" for scenario in SCENARIOS for variant in AB.VARIANTS]
    order: list[dict] = []
    for repeat in range(1, REPEATS + 1):
        shuffled = list(ids)
        random.Random(ORDER_SEED + repeat).shuffle(shuffled)
        order += [{"call": len(order) + i, "repeat": repeat, "prompt_id": prompt_id}
                  for i, prompt_id in enumerate(shuffled, start=1)]
    return order


def rule_texts() -> dict:
    return {"failure_modes": {s: [text for text, _ in AB.FAILURE_RULES[s]] for s in SCENARIOS},
            "expected": {s: [text for text, _ in AB.EXPECTED_RULES[s]] for s in SCENARIOS}}


FIXED: dict[str, Any] = {
    "schema": "skill_ab_protocol_v1",
    "question": ("같은 사례 사실과 스킬 evaluations.md의 질의 3개를 줄 때, 시스템 프롬프트에 공식 Isaac Lab 디버깅 스킬 텍스트"
                 "(SKILL.md·reference.md·examples.md)를 넣은 경우와 뺀 경우 모델이 진술한 다음 단계 계획이 스킬의 Known failure "
                 "modes와 Expected behavior 규칙에 각각 몇 번 해당하는가"),
    "scope": "seen_case_fixed_packet_stated_plan_counts_not_skill_effect_superiority_or_generalization",
    "case": CASE,
    "model": W.MODEL,
    "request": deepcopy(W.REQUEST),
    "top_p": "보내지 않음(제공자 기본값)",
    "repeats": REPEATS,
    "calls": REPEATS * len(SCENARIOS) * len(AB.VARIANTS),
    "order_seed": ORDER_SEED,
    "call_order": call_order(),
    "attempts": ("호출마다 첫 시도. 인프라 오류(연결·시간 초과·429·5xx·빈 응답·잘린 응답)일 때만 같은 호출로 1회 더. "
                 "응답 내용 때문에 재시도하지 않는다"),
    "status_definitions": {
        "answered": "응답 본문이 있고 잘리지 않았다. 채점 가능 여부와 무관하게 응답 분모에 들어간다",
        "infra_error": "두 번째 시도까지 인프라 오류. 첫 시도만 인프라 오류이고 두 번째 시도가 없으면 미완료로 두고 요약하지 않는다",
        "call_error": "인프라 외 호출 오류(인증·요청 형식 등). 하나라도 있으면 요약하지 않고 실행 환경을 고친 뒤 새 버전으로 다시 한다",
    },
    "scoring": {
        "code": "evals/skill_ab.py",
        "grader": f"{SKILL_DIR.as_posix()}/{AB.GRADER_FILE}(프롬프트에 넣지 않음)",
        "rules": rule_texts(),
        "rule_texts_sha256": W.canonical_sha(rule_texts()),
        "scorable": "응답이 JSON 객체 하나이고 키·enum이 응답 형식과 같다(```json 울타리 허용)",
        "human_grading_items": list(AB.HUMAN_GRADING),
        "automatic": "규칙으로 판정 가능한 항목만 자동 채점한다. 결과를 본 뒤 규칙을 바꾸지 않는다",
    },
    "denominators": {
        "calls": "조건마다 15회(시나리오 3 × 반복 5). 인프라 오류로 끝난 호출은 응답이 아니므로 응답 분모에서 빼고 따로 적는다",
        "unscorable": "채점 불가 응답 수 / 그 조건의 응답 수",
        "failure_modes": ("응답 × 시나리오별 실패 모드 2개. 채점 가능한 응답의 항목은 규칙 판정, 채점 불가 응답의 항목은 "
                          "사람 채점 필요로 같은 분모에 남긴다"),
        "expected": ("응답 × 시나리오별 기대 행동 수(S1 4, S2 3, S3 3). 규칙 판정과 사람 채점 필요를 같은 방식으로 나눈다"),
        "human_grading": "응답 × 설명 항목 2개. 조건을 가린 채 사람이 채점하며 이 요약에서는 pending이다",
    },
    "metrics": ["조건·시나리오별 응답·채점 가능·채점 불가 수와 사유", "실패 모드 해당 수(규칙 판정)와 사람 채점 필요 수",
                "기대 행동 포함 수(규칙 판정)와 사람 채점 필요 수", "스킬 포함 − 미포함 실패 모드 해당 건수 차이",
                "입력·출력 토큰과 호출 wall 시간(나란히 기록)", "인프라 오류 재시도 수"],
    "wording": {
        "result": ("같은 고정 입력에서 스킬 포함 프롬프트 응답의 알려진 실패 모드 해당은 규칙 판정 {with_failure_hits}/"
                   "{with_failure_judged}건, 스킬 미포함 프롬프트 응답은 {without_failure_hits}/{without_failure_judged}건이다"
                   "(포함 − 미포함 = {failure_hit_difference:+d}건). 기대 행동 포함은 규칙 판정 {with_expected_hits}/"
                   "{with_expected_judged}건과 {without_expected_hits}/{without_expected_judged}건이다. 채점 불가 응답은 "
                   "{with_unscorable}개와 {without_unscorable}개이며, 그 응답의 규칙 항목과 모든 응답의 설명 항목은 사람 채점 "
                   "필요로 분모에 남겼다."),
        "common": ("이미 본 한 사례의 고정 입력을 반복 호출해 모델이 진술한 계획을 센 개발 측정이다. 실행한 행동이나 진단의 "
                   "맞고 틀림은 재지 않았다. 스킬 예시가 채점 기준과 겹치고 두 조건이 같은 행동 목록을 받으므로, 건수 차이는 "
                   "기록일 뿐 스킬의 효과·우위·일반화의 근거가 아니다."),
        "incomplete": "등록한 호출이 모두 끝나지 않아 요약하지 않았다.",
    },
    "forbidden_claims": ["스킬이 성능을 높였다", "스킬을 넣으면 더 정확하다", "다른 방법과의 우열", "일반화", "절감", "최초"],
    "results_path": RESULT.relative_to(ROOT).as_posix(),
}
DYNAMIC = {"prompts", "packet_sha256", "skill_files_in_prompt", "grader_file_not_in_prompt", "source_files"}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(dump(value))


# --- 프로토콜 ---
def dynamic_fields(prompts: dict | None = None) -> dict:
    prompts = prompts or AB.build_prompts(CASE)
    folder = ROOT / SKILL_DIR
    return {
        "prompts": {row["prompt_id"]: {"system_sha256": row["system_sha256"], "user_sha256": row["user_sha256"]}
                    for row in prompts["prompts"]},
        "packet_sha256": prompts["packet_sha256"],
        "skill_files_in_prompt": {name: sha_lf(folder / name) for name in AB.SKILL_FILES_IN_PROMPT},
        "grader_file_not_in_prompt": {AB.GRADER_FILE: sha_lf(folder / AB.GRADER_FILE)},
        "source_files": {name: sha_lf(ROOT / name) for name in sorted(CODE)},
    }


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[key] != value for key, value in FIXED.items()):
        raise ValueError("fixed comparison contract changed")
    if any(protocol[key] != value for key, value in dynamic.items()):
        raise ValueError("bound prompts or sources changed")


def prepare() -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_calls", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    return protocol, P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix()})


# --- 호출 기록 ---
def record_path(repeat: int, prompt_id: str, attempt: int) -> Path:
    return PRIVATE / "calls" / f"r{repeat}_{prompt_id.replace(':', '-')}_a{attempt}.json"


def verify_record(record: dict, entry: dict, attempt: int, protocol: dict, prompts_sha256: str) -> None:
    expected = {"schema": W.RECORD_SCHEMA, "prompt_id": entry["prompt_id"], "repeat": entry["repeat"], "attempt": attempt,
                "prompts_sha256": prompts_sha256, "request": protocol["request"],
                **protocol["prompts"][entry["prompt_id"]]}
    name = record_path(entry["repeat"], entry["prompt_id"], attempt).name
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name}: 호출 기록이 등록 프롬프트·설정과 다르다")
    status = record.get("status")
    if status not in ("answered", "infra_error", "call_error"):
        raise ValueError(f"{name}: 모르는 호출 상태")
    answered = bool((record.get("content") or "").strip()) and record.get("finish_reason") != "length" \
        and record.get("error_type") is None
    if (status == "answered") != answered or (status == "infra_error") != bool(record.get("retryable")):
        raise ValueError(f"{name}: 호출 상태가 기록 내용과 맞지 않는다")


def load_call(entry: dict, protocol: dict, prompts_sha256: str) -> dict:
    """호출의 최종 시도. 첫 시도가 인프라 오류일 때만 두 번째 시도를 본다."""
    paths = [record_path(entry["repeat"], entry["prompt_id"], a) for a in range(1, W.MAX_ATTEMPTS + 1)]
    present = [p.is_file() for p in paths]
    if present != sorted(present, reverse=True):
        raise ValueError(f"call {entry['call']}: 시도 번호가 이어지지 않는다")
    records = [read(p) for p, ok in zip(paths, present) if ok]
    for attempt, record in enumerate(records, start=1):
        verify_record(record, entry, attempt, protocol, prompts_sha256)
        if attempt < len(records) and record["status"] != "infra_error":
            raise ValueError(f"call {entry['call']}: 인프라 오류가 아닌 시도 뒤에 재시도 기록이 있다")
    if not records:
        return {**entry, "status": "missing", "next_attempt": 1}
    if records[-1]["status"] == "infra_error" and len(records) < W.MAX_ATTEMPTS:
        return {**entry, "status": "missing", "next_attempt": len(records) + 1}
    return {**entry, "status": records[-1]["status"], "attempts": len(records), "record": records[-1]}


def bind_prompts(protocol: dict) -> str:
    """처음 plan에서 프롬프트 파일과 결합 기록을 만들고, 이후에는 같은지 확인한다. 프롬프트 파일 SHA를 돌려준다."""
    prompts = AB.build_prompts(CASE)
    if dynamic_fields(prompts)["prompts"] != protocol["prompts"]:
        raise ValueError("다시 만든 프롬프트가 프로토콜과 다르다")
    path, binding_path = PRIVATE / "prompts.json", PRIVATE / "binding.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != dump(prompts):
            raise ValueError("prompts.json이 결합 뒤 바뀌었다")
    else:
        write_new(path, prompts)
    binding = {"protocol_sha256_lf": sha_lf(PROTOCOL), "prompts_sha256": sha_raw(path)}
    if binding_path.exists():
        if {key: read(binding_path).get(key) for key in binding} != binding:
            raise ValueError("프로토콜이나 프롬프트 파일이 결합 뒤 바뀌었다")
    else:
        write_new(binding_path, binding)
    return binding["prompts_sha256"]


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def command(entry: dict, attempt: int) -> str:
    return ("wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"$HOME/.venvs/rl-triage/bin/python\" "
            f"{wsl(ROOT / 'evals/skill_ab_call.py')} --prompts {wsl(PRIVATE / 'prompts.json')} "
            f"--prompt-id {entry['prompt_id']} --repeat {entry['repeat']} --attempt {attempt} "
            f"--output {wsl(record_path(entry['repeat'], entry['prompt_id'], attempt))}'")


def plan() -> dict:
    protocol, commit = verify_protocol()
    prompts_sha = bind_prompts(protocol)
    calls = [load_call(entry, protocol, prompts_sha) for entry in protocol["call_order"]]
    pending = [c for c in calls if c["status"] == "missing"]
    return {"source_commit_sha": commit, "calls": len(calls), "remaining": len(pending),
            "retries_pending": sum(1 for c in pending if c["next_attempt"] > 1),
            "call_errors": [c["call"] for c in calls if c["status"] == "call_error"],
            "commands": [command(c, c["next_attempt"]) for c in pending],
            "retry_rule": "명령을 순서대로 실행한 뒤 plan을 다시 실행한다. 인프라 오류였던 호출만 두 번째 시도 명령이 나온다"}


# --- 지표 ---
def _stats(values: list) -> dict | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return {"n": len(numbers), "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers)}


def _row(call: dict, prompts: dict) -> dict:
    scenario, variant = call["prompt_id"].split(":")
    record = call.get("record") or {}
    row = {"call": call["call"], "repeat": call["repeat"], "prompt_id": call["prompt_id"], "scenario": scenario,
           "variant": variant, "status": call["status"], "attempts": call.get("attempts"),
           "input_tokens": record.get("input_tokens"), "output_tokens": record.get("output_tokens"),
           "wall_s": record.get("execution_wall_s")}
    if call["status"] != "answered":
        return row
    text = record["content"]
    scored = AB.score(prompts, [{"prompt_id": call["prompt_id"], "repeat": call["repeat"], "text": text}])["rows"][0]
    parsed, _ = AB.parse_response(text)
    plan_fields = None if parsed is None else {
        "actions": [step["action"] for step in parsed["steps"]], "run_assessment": parsed["run_assessment"],
        "run_status": parsed["run_status"], "attributed_change_present": parsed["attributed_change"] is not None,
        "checkpoint_selection_basis": parsed["checkpoint_selection_basis"],
        "next_run_change_count": len(parsed["next_run_changes"])}
    return {**row, "response_sha256": scored["response_sha256"], "scorable": scored["scorable"],
            "unscorable_reason": scored["unscorable_reason"], "failure_modes": scored["failure_modes"],
            "expected": scored["expected"], "plan": plan_fields}


def _counts(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["status"] == "answered"]
    scorable = [r for r in answered if r["scorable"]]
    unscorable = [r for r in answered if not r["scorable"]]
    failure_judged = sum(len(r["failure_modes"]) for r in scorable)
    expected_judged = sum(sum(v is not None for v in r["expected"].values()) for r in scorable)
    expected_human = (sum(sum(v is None for v in r["expected"].values()) for r in scorable)
                      + sum(len(AB.EXPECTED_RULES[r["scenario"]]) for r in unscorable))
    failure_human = sum(len(AB.FAILURE_RULES[r["scenario"]]) for r in unscorable)
    return {
        "calls": len(rows), "answered": len(answered),
        "infra_error": sum(r["status"] == "infra_error" for r in rows),
        "infra_retries": sum((r["attempts"] or 1) - 1 for r in rows if r["attempts"]),
        "scorable": len(scorable), "unscorable": len(unscorable),
        "unscorable_reasons": {reason: sum(r["unscorable_reason"] == reason for r in unscorable)
                               for reason in sorted({r["unscorable_reason"] for r in unscorable})},
        "failure_modes": {"items": failure_judged + failure_human, "rule_judged": failure_judged,
                          "hits": sum(v for r in scorable for v in r["failure_modes"].values()),
                          "human_grading_required": failure_human},
        "expected": {"items": expected_judged + expected_human, "rule_judged": expected_judged,
                     "hits": sum(bool(v) for r in scorable for v in r["expected"].values()),
                     "human_grading_required": expected_human},
        "explanation_items_human_grading_required": len(answered) * len(AB.HUMAN_GRADING),
        **{key: _stats([r[key] for r in answered]) for key in ("input_tokens", "output_tokens", "wall_s")},
    }


def wording_values(variants: dict) -> dict:
    out = {}
    for variant, prefix in (("with_skill", "with"), ("without_skill", "without")):
        counts = variants[variant]
        out.update({f"{prefix}_failure_hits": counts["failure_modes"]["hits"],
                    f"{prefix}_failure_judged": counts["failure_modes"]["rule_judged"],
                    f"{prefix}_expected_hits": counts["expected"]["hits"],
                    f"{prefix}_expected_judged": counts["expected"]["rule_judged"],
                    f"{prefix}_unscorable": counts["unscorable"]})
    out["failure_hit_difference"] = out["with_failure_hits"] - out["without_failure_hits"]
    return out


def publish_violations(value: Any) -> list[str]:
    import build_replay_page as PAGE
    texts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            texts.append(node)
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(key)
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)
    problems = [p for text in texts for p in PAGE.claim_violations(text) + PAGE.path_violations(text)]
    for text in texts:
        for sentence in text.replace("\n", ".").split("."):
            if any(w in sentence for w in RESULT_WORDS_NEED_NEGATION) and not any(n in sentence for n in PAGE.NEGATIONS):
                problems.append(sentence.strip())
    return problems


def summarize() -> dict:
    protocol, commit = verify_protocol()
    prompts_sha = bind_prompts(protocol)
    prompts = read(PRIVATE / "prompts.json")
    calls = [load_call(entry, protocol, prompts_sha) for entry in protocol["call_order"]]
    missing = [c["call"] for c in calls if c["status"] == "missing"]
    errors = [c["call"] for c in calls if c["status"] == "call_error"]
    if missing or errors:
        return {"status": "incomplete" if missing else "call_error", "missing_calls": missing, "call_errors": errors,
                "wording": FIXED["wording"]["incomplete"]}
    rows = [_row(call, prompts) for call in calls]
    variants = {variant: _counts([r for r in rows if r["variant"] == variant]) for variant in AB.VARIANTS}
    scenarios = {scenario: {variant: _counts([r for r in rows if r["variant"] == variant and r["scenario"] == scenario])
                            for variant in AB.VARIANTS} for scenario in SCENARIOS}
    values = wording_values(variants)
    result = {"schema": "skill_ab_result_v1", "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit,
              "case": CASE, "model": protocol["model"], "request": protocol["request"], "prompts_sha256": prompts_sha,
              "packet_sha256": protocol["packet_sha256"], "variants": variants, "scenarios": scenarios,
              "failure_hit_difference_with_minus_without": values["failure_hit_difference"],
              "human_grading": {"status": "pending", "items": list(AB.HUMAN_GRADING), "blind_to_condition": True},
              "wording": FIXED["wording"]["result"].format(**values), "common_wording": FIXED["wording"]["common"],
              "rows": rows}
    problems = publish_violations(result)
    if problems:
        raise ValueError(f"공개 요약에 금지 문구나 경로가 있다: {problems[:5]}")
    write_new(RESULT, result)
    return {"status": "written", "path": FIXED["results_path"], "wording": result["wording"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="스킬 유무 A/B 사전등록·계획·요약")
    parser.add_argument("command", choices=("prepare", "verify", "plan", "summarize"))
    args = parser.parse_args(argv)
    commands = {"prepare": prepare, "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol()[1]},
                "plan": plan, "summarize": summarize}
    try:
        out = commands[args.command]()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError:
        print("거부: 프로토콜이나 결합 파일이 아직 커밋되지 않았다", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
