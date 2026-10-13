"""E3 모델 교정 제안: Isaac Lab 공식 debug-rl-training skill 체크리스트에 근거한 packet·프롬프트·검증과 호출 기록.

- packet은 진단 확정 뒤의 사실만 담는다(정답·결함 정의 없음). packet 생성·검증은 모델·네트워크를 호출하지 않는다.
- 모델 출력 {correction_id, recovery_mode, cited_facts, checklist, hold_reason}을 검증해 거부 사유를 남기고 예외를 던지지 않는다.
- 결정적 규칙(recommend_recovery_mode)과의 일치는 기록만 하고 우위를 주장하지 않는다. 승인은 사람이 correct.py approve로만 한다.
- call은 WSL의 기존 NVIDIA 자격증명 환경에서 실행한다(active_model_call.py와 같은 방식, 코드는 키 파일을 읽지 않는다).
- agrees_with_rule은 모델이 규칙 결과(facts.rule_mode)를 본 상태의 일치다. 독립 검증이나 성능 근거로 쓰지 않는다.

순서: correct.py evidence → correction_model_call.py(WSL) → correct.py propose --model-proposal → 사람 approve.
같은 교정 제안이 원장에 이미 있으면 다시 propose할 수 없으므로 모델 기록은 첫 propose에 붙인다.

  python evals/correction_model_call.py --evidence <packet.json> --output <record.json> --model <model>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_registry import HOLD, REGISTRY  # noqa: E402

SKILL_DIR = ROOT / "bench" / "reference" / "nvidia_skills" / "isaaclab-debugging-rl-training"
SKILL_COMMIT = "5c3ca250b400b9038a03d46494c1812c6b48dab4"
SKILL_SHA256 = "9c6edf7d7dab455eb3d8e20fa0f101815c48e8853bb82a7ebd1beec2b4420cb0"  # LF 정규화 SKILL.md
SKILL_URL = f"https://github.com/isaac-sim/IsaacLab/blob/{SKILL_COMMIT}/skills/user/debug-rl-training/SKILL.md"
PACKET_SCHEMA = "correction_model_packet_v1"
RECORD_SCHEMA = "correction_model_call_v1"
REVIEW_SCHEMA = "correction_model_review_v1"
KEYS = ("correction_id", "recovery_mode", "cited_facts", "checklist", "hold_reason")
UNMET = "unmet"
MAX_PROPOSAL_CHARS = 4096  # 원장에 남기는 모델 출력 원문의 상한(승인자 화면에 그대로 보인다)
COST_SOURCE = "evals/results/reward_resume_dev3_20261013/training_evidence.json"

SYSTEM = (
    "RL 학습 실패의 원인 메커니즘이 승인된 probe로 이미 확정됐다. 너는 등록 교정 하나와 회복 방식 하나를 제안하거나, "
    "근거가 부족하면 보류(HOLD)한다. 판단 기준은 NVIDIA Isaac Lab debug-rl-training skill의 검증 체크리스트다. "
    "규칙: (1) correction_id와 recovery_mode는 registry의 확정 메커니즘 항목 값만 쓴다. "
    "(2) cited_facts와 checklist 값은 facts의 키만 인용한다. 없는 사실을 만들지 않는다. "
    "(3) checklist는 skill_checklist의 모든 번호에 대해 그 항목을 충족하는 facts 키 하나, 충족 근거가 없으면 \"unmet\"을 쓴다. "
    "(4) 보류하면 hold_reason에 이유를 쓰고 correction_id·recovery_mode는 null, 제안하면 hold_reason은 null이다. "
    "출력은 JSON 객체 하나 {\"correction_id\": str|null, \"recovery_mode\": str|null, \"cited_facts\": [str...], "
    "\"checklist\": {\"1\": str, ...}, \"hold_reason\": str|null}뿐이다.")


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def skill_checklist(skill_dir: Path = SKILL_DIR) -> dict[str, str]:
    """고정한 SKILL.md의 SHA256을 확인하고 Validation 절의 번호 항목 6개를 원문 그대로 반환한다."""
    data = (Path(skill_dir) / "SKILL.md").read_bytes()
    got = _lf_sha256(data)
    if got != SKILL_SHA256:
        raise ValueError(f"고정한 skill 원문과 SHA256이 다르다: {got}")
    text = data.decode("utf-8").replace("\r\n", "\n")
    section = re.search(r"^## Validation\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    items = re.findall(r"^(\d+)\. (.+)$", section.group(1), re.M) if section else []
    if [n for n, _ in items] != [str(i) for i in range(1, 7)]:
        raise ValueError("skill Validation 체크리스트가 1~6 번호 항목이 아니다")
    return {n: t.strip() for n, t in items}


def cost_basis(path: Path = ROOT / COST_SOURCE) -> dict:
    """v3 보상 사례 세 arm의 실측 parent wall(초, 0.1 s 반올림). 추가 PPO update는 회복 방식과 무관하게 300회."""
    arms = json.loads(Path(path).read_text(encoding="utf-8"))["arms"]
    return {"train_300_updates_s": {a: round(v["parent_wall_s"]["train_s"], 1) for a, v in sorted(arms.items())},
            "fixed_eval_s": {a: round(v["parent_wall_s"]["fixed_evaluation_s"], 1) for a, v in sorted(arms.items())}}


def packet_digest(packet: dict) -> str:
    return L.digest(packet)


def build_packet(*, case: str, loop_state_digest: str, receipts_digest: str, mechanism: str, confirming: dict,
                 observed: dict, rule: dict, source_report: dict, source_report_sha256: str | None,
                 checkpoint_sha256: str | None, registry: dict | None = None) -> dict:
    """확정 진단 뒤 모델에게 줄 근거. confirming = 확정 probe의 {probe, request_id, outcome, measurement}."""
    reg = REGISTRY if registry is None else registry
    entry = reg.get(mechanism)
    probe = confirming["probe"]
    cost = cost_basis()
    facts = {
        "confirmed_mechanism": mechanism,
        "confirming_probe": probe,
        "confirming_probe_measures": L.PROBES[probe]["measures"],
        "confirming_probe_outcome": confirming["outcome"],
        "confirming_probe_measurement": confirming["measurement"],
        "probe_observations": observed,
        "eval_protocol": source_report.get("protocol"),
        "eval_task": source_report.get("task"),
        "eval_metrics": source_report.get("metrics"),
        "eval_standstill": source_report.get("standstill"),
        "standstill_ratio": rule.get("standstill_ratio"),
        "rule_mode": rule["mode"],
        "rule": rule["rule"],
        "registered_correction": entry["correction_id"] if entry else None,
        "registered_recovery_modes": list(entry["recovery_modes"]) if entry else [],
        "cost_train_300_updates_parent_wall_s": cost["train_300_updates_s"],
        "cost_fixed_eval_parent_wall_s": cost["fixed_eval_s"],
        "checkpoint_sha256": checkpoint_sha256,
        "source_report_sha256": source_report_sha256,
    }
    table = {m: {"correction_id": e["correction_id"], "recovery_modes": list(e["recovery_modes"])} for m, e in reg.items()}
    return {"schema": PACKET_SCHEMA, "case": case, "loop_state_digest": loop_state_digest,
            "receipts_digest": receipts_digest, "confirming_request_id": confirming["request_id"],
            "mechanism": mechanism, "facts": facts, "registry": table,
            "rule_result": {"correction_id": rule["correction_id"], "mode": rule["mode"]},
            "checklist": skill_checklist(), "cost_basis_source": COST_SOURCE,
            "skill_source": {"url": SKILL_URL, "commit": SKILL_COMMIT, "sha256": SKILL_SHA256}}


def build_messages(packet: dict) -> list[dict]:
    """chat 형식 메시지. 케이스 이름·원장 digest는 넣지 않는다."""
    user = {"confirmed_mechanism": packet["mechanism"], "facts": packet["facts"], "registry": packet["registry"],
            "skill_checklist": packet["checklist"], "skill_source": packet["skill_source"]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)}]


def parse_content(content) -> tuple[dict | None, str | None]:
    """응답 문자열 → (JSON 객체, None) 또는 (None, 사유). ```json 울타리만 벗긴다."""
    if not isinstance(content, str) or not content.strip():
        return None, "빈 응답"
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON 파손: {e.msg}"
    if not isinstance(obj, dict):
        return None, "JSON 객체가 아니다"
    return obj, None


def _rejected(reasons: list[str]) -> dict:
    return {"accepted": False, "decision": None, "reasons": reasons, "agrees_with_rule": False, "unmet_checklist": []}


def validate_model_proposal(proposal, packet: dict, registry: dict | None = None) -> dict:
    """{accepted, decision(propose|HOLD|None), reasons, agrees_with_rule, unmet_checklist}. 예외를 던지지 않는다."""
    reg = REGISTRY if registry is None else registry
    if not isinstance(proposal, dict) or set(proposal) != set(KEYS):
        return _rejected([f"제안은 {'·'.join(KEYS)} 다섯 키여야 한다"])
    cid, mode, cited, checklist, hold = (proposal[k] for k in KEYS)
    facts = packet.get("facts") or {}
    reasons: list[str] = []
    is_hold = isinstance(hold, str) and bool(hold.strip())
    if hold is not None and not is_hold:
        reasons.append("hold_reason은 null이거나 비지 않은 문자열이어야 한다")
    if is_hold:
        if cid is not None or mode is not None:
            reasons.append("HOLD 제안은 correction_id·recovery_mode가 null이어야 한다")
    elif not isinstance(cid, str) or not isinstance(mode, str):
        reasons.append("correction_id와 recovery_mode는 문자열이어야 한다")
    else:
        entry = reg.get(packet.get("mechanism"))
        if entry is None:
            reasons.append("확정 메커니즘이 등록 표에 없다")
        else:
            if cid not in {e["correction_id"] for e in reg.values()}:
                reasons.append(f"등록 표에 없는 교정 {cid!r}")
            elif cid != entry["correction_id"]:
                reasons.append(f"확정 메커니즘의 교정이 아니다({cid!r})")
            if mode not in entry["recovery_modes"]:
                reasons.append(f"등록되지 않은 회복 방식 {mode!r}")
    if not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited):
        reasons.append("근거 인용(cited_facts)이 비었거나 형식이 틀리다")
    else:
        missing = [c for c in cited if c not in facts]
        if missing:
            reasons.append(f"제안 근거에 없는 인용 {missing}")
    expected = list(packet.get("checklist") or {})
    unmet: list[str] = []
    if not isinstance(checklist, dict) or set(checklist) != set(expected):
        reasons.append(f"체크리스트는 {expected} 항목을 모두 가져야 한다")
    else:
        for k in expected:
            v = checklist[k]
            if v == UNMET:
                unmet.append(k)
            elif not isinstance(v, str) or v not in facts:
                reasons.append(f"체크리스트 {k}의 근거 {v!r}가 facts에 없다")
    if reasons:
        return _rejected(reasons)
    rule = packet.get("rule_result") or {}
    agrees = (rule.get("mode") == HOLD) if is_hold else (cid == rule.get("correction_id") and mode == rule.get("mode"))
    return {"accepted": True, "decision": HOLD if is_hold else "propose", "reasons": [], "agrees_with_rule": agrees,
            "unmet_checklist": unmet}


def review_call_record(record: dict, packet: dict, record_sha256: str | None = None,
                       registry: dict | None = None) -> dict:
    """호출 기록을 현재 packet에 묶어 검증한다. 형식·packet 불일치·비정상 종료는 ValueError(원장에 쓰지 않는다)."""
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise ValueError("교정 모델 호출 기록 형식이 아니다")
    if record.get("exit_code") != 0 or record.get("infra_error") or record.get("error_type"):
        raise ValueError("모델 호출이 정상 종료되지 않았다(인프라 오류는 같은 packet으로 재시도)")
    if record.get("packet_digest") != packet_digest(packet):
        raise ValueError("호출 기록의 packet이 현재 근거와 다르다(loop 상태·report·skill이 바뀌었으면 다시 호출)")
    if record.get("messages_sha256") != L.digest(build_messages(packet)) or record.get("skill_sha256") != SKILL_SHA256:
        raise ValueError("호출 기록의 프롬프트나 skill이 현재 코드와 다르다(프롬프트를 바꿨으면 다시 호출)")
    proposal, error = parse_content(record.get("content"))
    verdict = _rejected([error]) if error else validate_model_proposal(proposal, packet, registry)
    review = {"schema": REVIEW_SCHEMA, "record_sha256": record_sha256, "packet_digest": record["packet_digest"],
              "messages_sha256": record["messages_sha256"], "model": (record.get("request") or {}).get("model"),
              "provider_model": record.get("provider_model"), "response_id": record.get("response_id"),
              "rule_visible_to_model": True, "proposal": proposal, **verdict}
    if proposal is not None and len(json.dumps(proposal, ensure_ascii=False)) > MAX_PROPOSAL_CHARS:
        review.update(proposal=None, proposal_omitted="too_large")
    return review


def call(evidence_path: Path, output: Path, model: str) -> int:
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError

    output = Path(output)
    if output.exists():  # 호출한 뒤 기록을 못 쓰는 일이 없도록 먼저 막는다
        raise FileExistsError(output)
    data = Path(evidence_path).read_bytes()
    packet = json.loads(data)
    if packet.get("schema") != PACKET_SCHEMA or packet.get("checklist") != skill_checklist():
        raise ValueError("packet 형식이나 skill 체크리스트가 고정 원문과 다르다")
    messages = build_messages(packet)
    started = datetime.now(timezone.utc).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    params = {"model": model, "temperature": 0, "max_tokens": 16384, "stream": False,
              "extra_body": {"reasoning_effort": "none"}}
    record = {"schema": RECORD_SCHEMA, "packet_sha256": hashlib.sha256(data).hexdigest(),
              "packet_digest": packet_digest(packet), "skill_sha256": SKILL_SHA256,
              "messages_sha256": L.digest(messages), "started_at": started, "request": params,
              "response_id": None, "provider_model": None, "finish_reason": None, "content": None,
              "reasoning_content": None, "input_tokens": None, "output_tokens": None, "total_tokens": None}
    exit_code = 0
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                        api_key=os.environ["NVIDIA_API_KEY"], timeout=300, max_retries=0)
        response = client.chat.completions.create(messages=messages, **params)
        record["response_id"] = response.id
        record["provider_model"] = response.model
        if response.choices:
            choice = response.choices[0]
            record.update(content=choice.message.content, finish_reason=choice.finish_reason,
                          reasoning_content=getattr(choice.message, "reasoning_content", None))
        if response.usage is not None:
            record.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens)
        if not (record.get("content") or "").strip() or record["finish_reason"] == "length":
            record["infra_error"] = "empty_or_truncated_response"
            record["retryable"] = True
            exit_code = 2
    except Exception as exc:
        # 예외 문자열에는 요청 헤더·자격증명이 섞일 수 있으므로 종류만 보존한다.
        retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        record["error_type"] = type(exc).__name__
        record["retryable"] = retryable
        if retryable:
            record["infra_error"] = type(exc).__name__
        exit_code = 2
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), execution_wall_s=time.perf_counter() - wall,
                  cpu_s=time.process_time() - cpu, exit_code=exit_code)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"  # 직렬화 실패로 반쪽 파일을 남기지 않는다
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3 교정 제안 모델 호출 한 번을 기록한다")
    parser.add_argument("--evidence", type=Path, required=True, help="correct.py evidence가 만든 packet JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    pid_path = args.output.with_suffix(".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with pid_path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        code = call(args.evidence, args.output, args.model)
    finally:
        pid_path.unlink(missing_ok=True)
    raise SystemExit(code)
