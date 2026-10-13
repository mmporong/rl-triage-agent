"""D3 B4·B5 계약: 변경 순위를 묻는 모델 단일 호출의 입력 packet, 메시지, 출력 스키마, 검증기.

- 입력: 중립 라벨(change_1..change_8, 사례 순서)과 Isaac Lab 소스 unified diff 원문, 실패 실행·같은 seed NONE 실행의
  텔레메트리 사실(run_active_diagnosis.input_packet과 같은 id: telemetry:<case|reference>:<tag> = 마지막 20% 평균),
  B4만 동결 고리의 probe 관측. B5 packet은 B4 packet에서 probe 관측만 비운 것이다.
- 변경 id·범주·역할은 packet에 넣지 않는다. 메시지 전체와 diff마다 단서 canary(bench/d3_diff.canary)가 0건이어야 만든다.
- 출력: ranking(라벨 8개 순열), reasons(상위 변경의 근거: 사실 id·probe·diff 새 줄 번호), uncertain(기록만 하고 순서에 쓰지 않는다).
- 검증 실패는 의미 재시도 없이 model_invalid다. 재시도는 인프라 오류(빈 응답·잘림·연결·429/5xx)에만 한 번.
표준 라이브러리만 쓴다(WSL 호출 스크립트 evals/d3_model_call.py도 이 모듈의 messages를 쓴다).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

SCHEMA = "change_rank_packet_v1"
LABEL_RE = r"change_[1-9][0-9]*"
PROBE_TEXT = {
    "P_noise": "actual action-sampling standard deviation divided by the logged action_std",
    "P_value": "relative change of critic parameters from the first to the last checkpoint",
    "P_reward": "relative error between tracking reward terms recomputed from the state by their documented definition "
                "and the values logged by the reward manager",
    "P_torque": "fraction of joint torques at saturation at low joint speed",
    "P_physics": "number of runtime physics values (friction, mass, physics dt) that differ from the configuration",
    "P_episode": "time-limit termination step divided by episode_length_s / step_dt",
}
SYSTEM = (
    "You review source changes for a legged-robot reinforcement-learning run. Unitree Go2, flat terrain, velocity-command "
    "tracking, RSL-RL PPO, 4096 environments, 300 iterations, Isaac Lab v2.1.1. The run failed: compared with a run of the "
    "same seed without the changes, its policy tracks velocity commands worse or falls more. Several source changes were "
    "applied together before training; each is shown as a unified diff of Isaac Lab source under a neutral label. "
    "'facts' are training telemetry values (mean over the last 20% of iterations) of the failed run ('case') and of the "
    "same-seed run without the changes ('reference'). 'probe_observations', which may be empty, are short measurements "
    "on the failed policy with the outcome of a fixed threshold rule. Rank every change from most to least likely to "
    "have caused the failure. Return only one JSON object with keys: 'ranking' (all labels, most likely first, each "
    "exactly once), 'reasons' (a list of objects with 'change' (a label), 'facts' (fact ids you used), 'probes' (probe "
    "names you used) and 'diff_lines' (line numbers on the new side of that change's diff)), and 'uncertain' (true or "
    "false). The first reason must be about the first label in 'ranking' and cite at least one diff line. Cite only fact "
    "ids and probe names that appear in the input.")


def labels(k: int) -> list[str]:
    return [f"change_{i + 1}" for i in range(k)]


def telemetry_facts(case_summary: dict, reference_summary: dict) -> dict[str, float]:
    """run_active_diagnosis.input_packet과 같은 사실 id와 값(텔레메트리 summary의 mean_last_20pct)."""
    facts = {}
    for prefix, summary in (("case", case_summary), ("reference", reference_summary)):
        for tag, values in sorted(summary.items()):
            if isinstance(values, dict) and "mean_last_20pct" in values:
                facts[f"telemetry:{prefix}:{tag}"] = values["mean_last_20pct"]
    return facts


def probe_observations(steps: list[dict], measurements: dict[str, dict], reference: dict[str, dict],
                       thresholds: dict) -> list[dict]:
    """고리 순서대로 probe 이름·설명·판정 규칙·판정·측정값·같은 seed NONE 측정값."""
    return [{"probe": s["probe"], "measures": PROBE_TEXT[s["probe"]], "rule": list(thresholds[s["probe"]]),
             "outcome": s["outcome"], "measurement": measurements.get(s["probe"]),
             "reference_measurement": reference.get(s["probe"])} for s in steps]


def build_packet(diffs: list[str], facts: dict, observations: list[dict]) -> dict:
    """diffs는 사례 순서의 diff 원문. 라벨은 그 순서로 붙인다."""
    return {"schema": SCHEMA, "changes": [{"label": label, "diff": diff} for label, diff in zip(labels(len(diffs)), diffs)],
            "facts": facts, "probe_observations": observations}


def messages(packet: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False, sort_keys=True)}]


def clues(packet: dict) -> list[str]:
    """diff마다(추가 줄 주석 포함)와 메시지 전체의 단서. 빈 목록이어야 packet을 쓴다."""
    bench = Path(__file__).resolve().parents[1] / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import d3_diff as R
    found = [f"{c['label']}:{label}" for c in packet["changes"] for label in R.canary(c["diff"])]
    return found + [f"messages:{label}" for label in R.canary(json.dumps(messages(packet), ensure_ascii=False))]


def new_lines(diff: str) -> set[int]:
    """unified diff 각 hunk의 새 쪽 줄 번호 범위."""
    lines = set()
    for start, count in re.findall(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        lines.update(range(int(start), int(start) + (int(count) if count else 1)))
    return lines


def parse(content: str):
    text = (content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    try:
        return json.loads(fenced[1] if fenced else text)
    except json.JSONDecodeError:
        return None


def validate(output, packet: dict) -> list[str]:
    """출력 계약 위반 목록. 빈 목록이면 유효하다."""
    if not isinstance(output, dict) or set(output) != {"ranking", "reasons", "uncertain"}:
        return ["keys"]
    names = [c["label"] for c in packet["changes"]]
    ranking, reasons = output["ranking"], output["reasons"]
    problems = []
    if not isinstance(ranking, list) or sorted(map(str, ranking)) != sorted(names) or len(ranking) != len(names):
        problems.append("ranking_not_a_permutation")
    if not isinstance(output["uncertain"], bool):
        problems.append("uncertain_not_bool")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(r, dict) for r in reasons):
        return problems + ["reasons"]
    observed = {o["probe"] for o in packet["probe_observations"]}
    diffs = {c["label"]: new_lines(c["diff"]) for c in packet["changes"]}
    for n, reason in enumerate(reasons):
        if set(reason) != {"change", "facts", "probes", "diff_lines"} or reason["change"] not in diffs:
            problems.append(f"reason{n}_shape")
            continue
        if not isinstance(reason["facts"], list) or any(f not in packet["facts"] for f in reason["facts"]):
            problems.append(f"reason{n}_facts")
        if not isinstance(reason["probes"], list) or any(p not in observed for p in reason["probes"]):
            problems.append(f"reason{n}_probes")
        lines = reason["diff_lines"]
        if (not isinstance(lines, list) or any(type(x) is not int for x in lines)
                or any(x not in diffs[reason["change"]] for x in lines)):
            problems.append(f"reason{n}_diff_lines")
    if not problems and (reasons[0]["change"] != ranking[0] or not reasons[0]["diff_lines"]):
        problems.append("top_reason")
    return problems


def review(record: dict, packet_bytes: bytes, model: str) -> dict:
    """호출 기록 하나 → valid / invalid / infra_error / call_error. packet·모델이 다르면 거부(ValueError)."""
    if not isinstance(record, dict) or record.get("schema") != "d3_model_call_v1":
        raise ValueError("d3_model_call_v1 기록이 아니다")
    if record.get("packet_sha256") != hashlib.sha256(packet_bytes).hexdigest():
        raise ValueError("기록이 이 packet의 호출이 아니다")
    if (record.get("request") or {}).get("model") != model or (record.get("request") or {}).get("temperature") != 0:
        raise ValueError("등록한 모델·temperature 0의 호출이 아니다")
    usage = {k: record.get(k) for k in ("input_tokens", "output_tokens", "total_tokens", "execution_wall_s")}
    if record.get("infra_error"):
        return {"status": "infra_error", "problems": [record["infra_error"]], "ranking": None, **usage}
    if record.get("exit_code") != 0 or record.get("error_type"):
        return {"status": "call_error", "problems": [record.get("error_type") or "exit_code"], "ranking": None, **usage}
    output = parse(record.get("content"))
    problems = validate(output, json.loads(packet_bytes))
    return {"status": "invalid" if problems else "valid", "problems": problems,
            "ranking": None if problems else output["ranking"], "uncertain": None if problems else output["uncertain"],
            "reasons": None if problems else output["reasons"], **usage}
