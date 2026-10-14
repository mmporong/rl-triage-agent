"""N2 검증 피드백 1회 재요청 arm: N2와 같은 11개 packet × 5회 단일 호출(출력 형식 없음)에서, 기존 계약 검증기가 1차 출력을
거절하면 문제 코드와 고정 규칙 설명을 붙여 같은 대화로 한 번만 다시 요청한다. 사전등록·실행 계획·실행·요약을 한 파일에서 한다.

- 1차 호출은 N2 단일 호출과 같다(메시지·모델·temperature 0·reasoning_effort none·max_tokens). 호출기는
  evals/feedback_model_call.py다(기존 active_model_call·correction_model_call·N2·구조화 파일은 바꾸지 않는다).
- 피드백은 검증기 거절 메시지를 등록 문제 코드로 바꾼 고정 문장이다. 정답·카탈로그 값·사실 id·값·등록 교정이나 프롬프트 밖
  힌트를 넣지 않는다. 요약기는 기록의 피드백 문장이 코드에서 다시 만든 문장과 같은지 확인한다.
- 판정은 기존 검증기로 1차·최종을 다시 계산한다. 재요청 뒤에도 무효면 무효다.
- "개선"은 구조화 arm과 같은 사전등록 기준(최종 무효 ≤ 1/55, 진단·교정 평균 pass^4 ≥ 0.8)을 넘을 때만 쓴다. 넘더라도
  이미 본 packet의 개발 측정이며 시연 결정 단계에 넣으려면 새 사전등록이 필요하다고 적는다.

  python evals/n2_feedback.py prepare|verify|plan|run|summarize
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import feedback_model_call as F  # noqa: E402
import n2_pass_k as N  # noqa: E402  입력 세트·packet 경로·결정 키·pass^k 정의를 그대로 쓴다
from rl_triage import probe_loop as L  # noqa: E402

CODE_ROOT = ROOT
PROTOCOL = ROOT / "bench/protocols/n2_feedback_v1.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/n2_feedback"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "n2_feedback_v1"
N2_SUMMARY = "evals/results/n2_pass_k_v1/summary.json"
MODEL = N.MODEL
KINDS = N.KINDS
RUNS = N.RUNS
K_VALUES = N.K_VALUES
API_CAP = 120
MAX_ATTEMPTS = 2
MAX_REQUESTS_PER_RUN = 2
CALL_TIMEOUT_S = 900
USE_THRESHOLD = N.USE_THRESHOLD
IMPROVE = {"max_invalid_total": 1, "min_mean_pass4_each_kind": 0.8}
SCRIPT = "evals/feedback_model_call.py"
CODE = N.CODE | {"evals/n2_feedback.py", "evals/feedback_model_call.py"}
FIXED = {
    "schema": "n2_feedback_protocol_v1",
    "question": ("N2와 같은 packet·같은 단일 호출에서 계약 검증기가 거절한 출력에 문제 코드와 고정 규칙 설명을 붙여 한 번만 다시 요청하면, "
                 "최종 계약 위반과 결정 일치(pass^k)가 어떻게 되는가"),
    "scope": "validator_feedback_single_retry_arm_on_already_seen_packets_development_measurement_not_accuracy_superiority_generalization_or_savings",
    "model": MODEL, "temperature": 0, "reasoning_effort": "none", "max_tokens": F.MAX_TOKENS,
    "first_call": {"same_as": "N2 단일 호출(active_model_call·correction_model_call과 같은 메시지 함수·요청 설정)",
                   "output_format": "없음(response_format 쓰지 않음)"},
    "retry": {"when": "1차 출력을 기존 계약 검증기(validate_decision, validate_model_proposal)가 거절했을 때만, 한 번",
              "messages": "1차 메시지 + 1차 응답(assistant) + 피드백(user). 요청 설정은 1차와 같다",
              "feedback_rule": ("검증기 거절 메시지를 등록 문제 코드로 바꾸고 코드마다 고정 설명(프롬프트 계약 규칙을 다시 적은 문장)을 붙인다. "
                                "정답·카탈로그 값·사실 id·값·등록 교정 같은 packet별 내용과 프롬프트 밖 힌트는 넣지 않는다"),
              "codes": {"diagnosis": [c for _, c in F.DIAGNOSIS_CODES] + ["D_OTHER"],
                        "correction": sorted({c for _, c in F.CORRECTION_CODES}) + ["C_OTHER"]},
              "texts": {"diagnosis": F.DIAGNOSIS_TEXT, "correction": F.CORRECTION_TEXT},
              "frame": {kind: list(parts) for kind, parts in F.FEEDBACK_HEAD.items()},
              "final": "재요청 결과를 같은 검증기로 다시 판정한다. 재요청 뒤에도 무효면 무효"},
    "input_sets": "N2 프로토콜(bench/protocols/n2_pass_k_v1.json)의 input_sets 11개 packet·종류 그대로",
    "runs_per_packet": RUNS, "k_values": list(K_VALUES),
    "order": "N2와 같다: run 번호 1~5를 바깥 반복, 세트 순서·종류를 안쪽 반복",
    "attempts": ("run 번호마다 첫 시도. 1차나 재요청에서 인프라 오류(연결·시간 초과·429·5xx·빈 응답·잘림)가 나면 그 run을 처음부터 "
                 "1회 더(attempt 2)"),
    "api_budget": {"cap_requests": API_CAP, "planned_first_requests": len(N.schedule()), "max_retry_requests": len(N.schedule()),
                   "counting": ("기록의 requests 합(1차 1 + 재요청 0~1, 인프라 재시도 포함). 다음 run에 남은 요청이 2 미만이면 남은 실행은 "
                                "budget_exhausted")},
    "metrics": ["1차 유효율", "재요청 횟수와 문제 코드", "최종 유효율과 최종 무효 코드", "최종 결정 키 pass^k·분포",
                "재요청이 결정 키를 바꿨는지(1차가 무효라 비교 불가로 표기. 1차 출력에서 결정 키를 읽을 수 있으면 참고로 같음·다름)",
                "N2 단일 호출 packet별 결정과의 일치", "토큰·벽시계(1차, 재요청 추가분)"],
    "pass_k": "N2와 같다: 최종 결정 기준 Σ_g C(c_g,k)/C(n,k)",
    "comparison": {"baseline": N2_SUMMARY, "note": "N2 단일 호출(이미 본 결과)과 packet별로 나란히 둔다. 1차 호출은 같은 조건의 새 표본이다"},
    "improvement_rule": {**IMPROVE, "rule": ("최종 무효가 55회 중 1 이하이고 진단·교정 각각 packet 평균 pass^4가 0.8 이상일 때만 '개선'이라고 "
                                             "쓴다. 아니면 개선을 말하지 않고 수치만 적는다")},
    "use_rule": {"threshold_pass4": USE_THRESHOLD, "rule": "N2와 같다: 종류별 packet 평균 pass^4 ≥ 0.8이면 usable, 아니면 reconsider"},
    "wording": {
        "improved": ("검증 피드백 1회 재요청 arm이 사전등록 기준(최종 무효 ≤ 1/55, 진단·교정 평균 pass^4 ≥ 0.8)을 넘었다: 1차 유효 "
                     "{first_valid}/55, 재요청 {retries}회, 최종 무효 {invalid}/55, 평균 pass^4 진단 {p4d}·교정 {p4c}. 같은 packet의 "
                     "N2 단일 호출(무효 6/55)보다 개선됐다. 이미 본 packet의 개발 측정이며, 시연 결정 단계에 넣으려면 새 사전등록이 필요하다."),
        "not_improved": ("검증 피드백 1회 재요청 arm은 사전등록 개선 기준(최종 무효 ≤ 1/55, 진단·교정 평균 pass^4 ≥ 0.8)을 넘지 못했다: "
                         "1차 유효 {first_valid}/55, 재요청 {retries}회, 최종 무효 {invalid}/55, 평균 pass^4 진단 {p4d}·교정 {p4c}. "
                         "개선을 주장하지 않는다. 이미 본 packet의 개발 측정이다."),
        "insufficient": "인프라 오류·예산 소진으로 일부 packet의 pass^4를 계산하지 못해 개선 판정을 하지 않는다.",
        "common": ("이미 본 packet을 같은 모델·temperature 0으로 반복하고, 계약 검증기가 거절한 출력에만 고정 피드백으로 한 번 다시 요청한 개발 "
                   "측정이다. 재요청은 요청 수와 토큰을 늘린다. 반복 일관성·유효율은 정확도의 근거가 아니다. N2 결과는 이미 본 값이고 시점이 "
                   "다른 반복이다. 비용은 토큰·벽시계 기록이며 절감 주장이 아니다."),
    },
    "forbidden_claims": ["LLM이 규칙보다 우월하다", "GPU·토큰·시간 절감", "일반화", "정확도", "최초",
                         "사전등록 기준 없이 '개선'", "재요청이 진단 내용(순위·probe)을 좋게 했다",
                         "기준을 넘었다고 시연 결정 단계에 바로 넣는다"],
}
DYNAMIC = {"source_files", "inputs"}


def read(path: Path):
    return N.read(path)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def public_inputs() -> list[str]:
    return sorted(set(N.public_inputs()) | {N2_SUMMARY, rel(N.PROTOCOL)})


def dynamic_fields() -> dict:
    return {"source_files": {name: N.sha_lf(CODE_ROOT / name) for name in sorted(CODE)},
            "inputs": {"public": {name: N.sha_lf(ROOT / name) for name in public_inputs()},
                       "private": {f"{s}/{k}": {"packet_sha256": N.sha_raw(N.packet_path(s, k))}
                                   for s in N.SET_ORDER for k in N.INPUT_SETS[s]["kinds"]}}}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed feedback-arm contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources or inputs changed")


def prepare() -> dict:
    N.write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_any_call", "path": rel(PROTOCOL), "cap_requests": API_CAP}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    try:
        return protocol, P.verify_committed(ROOT, CODE | set(public_inputs()) | {rel(PROTOCOL)})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


# --- 실행 ---
def run_path(set_name: str, kind: str, run: int, attempt: int) -> Path:
    N.packet_path(set_name, kind)  # 등록 확인
    return PRIVATE / "runs" / set_name / kind / f"run{run}_a{attempt}.json"


def call_argv(set_name: str, kind: str, run: int, attempt: int) -> list[str]:
    """WSL에서 자격증명 파일을 source한 셸로 피드백 호출기를 부른다. 키 값은 이 프로세스가 보지 않는다."""
    shell = ('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
             f"{N.wsl(CODE_ROOT / SCRIPT)} --kind {kind} --packet {N.wsl(N.packet_path(set_name, kind))} "
             f"--output {N.wsl(run_path(set_name, kind, run, attempt))} --model {MODEL}")
    return ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", shell]


def requests_used() -> int:
    if not (PRIVATE / "runs").exists():
        return 0
    return sum(int(read(p).get("requests") or 1) for p in (PRIVATE / "runs").glob("*/*/run*_a*.json"))


def _is_infra(record: dict) -> bool:
    return bool(record.get("infra_error"))


def final_record(set_name: str, kind: str, run: int) -> tuple[Path | None, dict | None]:
    present = [(run_path(set_name, kind, run, a), a) for a in range(1, MAX_ATTEMPTS + 1)
               if run_path(set_name, kind, run, a).exists()]
    if [a for _, a in present] != list(range(1, len(present) + 1)):
        raise ValueError(f"{set_name} {kind} run{run}: 시도 번호가 이어지지 않는다")
    if not present:
        return None, None
    path, attempt = present[-1]
    record = read(path)
    if attempt < MAX_ATTEMPTS and _is_infra(record):
        return None, None
    return path, record


def plan() -> dict:
    _, commit = verify_protocol()
    return {"source_commit_sha": commit, "protocol_sha256_lf": N.sha_lf(PROTOCOL), "planned_first_requests": len(N.schedule()),
            "cap_requests": API_CAP, "requests_used": requests_used(),
            "first_commands": [" ".join(call_argv(s, k, r, 1)) for s, k, r in N.schedule()[:2]]}


def execute(runner=None) -> dict:
    """등록 순서대로 남은 실행을 부른다. 끝난 실행은 건너뛰고(재개 가능), 남은 요청이 run 하나의 최대(2)보다 적으면 멈춘다."""
    _, commit = verify_protocol()
    runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, timeout=CALL_TIMEOUT_S).returncode)
    log_path = PRIVATE / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done, stopped = 0, None
    for set_name, kind, run in N.schedule():
        for attempt in range(1, MAX_ATTEMPTS + 1):
            path = run_path(set_name, kind, run, attempt)
            if path.exists():
                if attempt < MAX_ATTEMPTS and _is_infra(read(path)):
                    continue
                break
            if API_CAP - requests_used() < MAX_REQUESTS_PER_RUN:
                stopped = "budget_exhausted"
                break
            started = time.perf_counter()
            try:
                code = runner(call_argv(set_name, kind, run, attempt))
            except subprocess.TimeoutExpired:
                code = "timeout"
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(json.dumps({"set": set_name, "kind": kind, "run": run, "attempt": attempt, "exit": code,
                                      "parent_wall_s": round(time.perf_counter() - started, 2), "commit": commit,
                                      "at": datetime.now(timezone.utc).isoformat()}) + "\n")
            if not path.exists():
                stopped = f"no_record:{set_name}/{kind}/run{run}"
                break
            done += 1
            if not _is_infra(read(path)):
                break
        if stopped:
            break
    return {"runs_made": done, "requests_used": requests_used(), "cap_requests": API_CAP, "stopped": stopped,
            "remaining": sum(1 for s, k, r in N.schedule() if final_record(s, k, r)[0] is None)}


# --- 기록 재검증 ---
def raw_key(kind: str, content) -> list | None:
    """계약상 무효여도 JSON 객체에서 결정 키 자리를 읽을 수 있으면 그 값(참고용, 결정 아님). 못 읽으면 None.
    본문은 검증기와 같은 방식으로 꺼낸다(진단은 diagnosis_body, 교정은 parse_content)."""
    import correction_model_call as CM
    from rl_triage.correction_registry import HOLD
    if kind == "diagnosis":
        try:
            obj = json.loads(F.diagnosis_body(content))
        except json.JSONDecodeError:
            return None
    else:
        obj, error = CM.parse_content(content)
        if error:
            return None
    try:
        if kind == "diagnosis":
            return [obj["action"], obj["ranking"][0]]
        hold = isinstance(obj.get("hold_reason"), str) and bool(obj["hold_reason"].strip())
        return [HOLD if hold else "propose", obj.get("correction_id"), obj.get("recovery_mode")]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def review(kind: str, record: dict, packet: Path) -> dict:
    """기록을 다시 검증한다. 1차·최종 판정을 기존 검증기로 다시 계산하고, 피드백 문장이 코드에서 만든 고정 문장인지 확인한다."""
    if record.get("schema") != F.RECORD_SCHEMA or record.get("kind") != kind:
        raise ValueError("피드백 호출 기록 형식이 아니다")
    data = Path(packet).read_bytes()
    messages, binding = F.first_messages(kind, data)
    if any(record.get(k) != v for k, v in binding.items()):
        raise ValueError("호출 기록의 packet·메시지가 등록 입력과 다르다")
    if record.get("request") != F.request_params(MODEL):
        raise ValueError("등록 요청 설정과 다르다")
    calls = record.get("calls") or []
    if record.get("infra_error"):
        return {"status": "infra_error"}
    if record.get("error_type"):
        return {"status": "call_error"}
    if not calls:
        raise ValueError("호출 기록이 비어 있다")
    packet_data = json.loads(data)
    first = F.validate(kind, calls[0]["content"], packet_data)
    out = {"first_status": first["status"], "first_codes": first["codes"], "first_reasons": first["reasons"],
           "retried": len(calls) > 1, "requests": record.get("requests")}
    if first["status"] == "valid":
        if len(calls) != 1 or record.get("feedback") is not None:
            raise ValueError("1차가 유효한데 재요청 기록이 있다")
        final = first
    else:
        if len(calls) != 2:
            raise ValueError("1차가 무효인데 재요청이 한 번이 아니다")
        expected = F.feedback_text(kind, first["codes"])
        sent = F.retry_messages(messages, calls[0]["content"], kind, first["codes"])
        if (record["feedback"]["text"] != expected or record["feedback"]["codes"] != first["codes"]
                or record["feedback"]["messages_sha256"] != L.digest(sent)):
            raise ValueError("피드백 문장이 등록 코드에서 만든 고정 문장과 다르다")
        final = F.validate(kind, calls[1]["content"], packet_data)
        out["raw_first_key"] = raw_key(kind, calls[0]["content"])
    if (record.get("final") or {}).get("status") != final["status"]:
        raise ValueError("기록의 최종 판정이 검증기 재계산과 다르다")
    out.update(status=final["status"], final_codes=final["codes"], final_reasons=final["reasons"])
    if final["status"] == "valid":
        out["key"] = list(N.decision_key(kind, final["decision"]))
        if kind == "diagnosis":
            out["probe"] = final["decision"].get("probe")
    return out


def feedback_runs(set_name: str, kind: str) -> list[dict]:
    packet = N.packet_path(set_name, kind)
    exhausted = API_CAP - requests_used() < MAX_REQUESTS_PER_RUN
    runs = []
    for run in range(1, RUNS + 1):
        path, record = final_record(set_name, kind, run)
        if record is None:
            runs.append({"run": run, "status": "budget_exhausted" if exhausted else "missing"})
            continue
        out = review(kind, record, packet)
        calls = record.get("calls") or []
        first, second = (calls + [{}, {}])[:2]
        row = {"run": run, **out, "record_sha256": N.sha_raw(path),
               "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
               "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
               "wall_s": sum(c.get("wall_s") or 0 for c in calls), "llm_calls": len(calls),
               "first_tokens": (first.get("input_tokens") or 0) + (first.get("output_tokens") or 0),
               "retry_tokens": (second.get("input_tokens") or 0) + (second.get("output_tokens") or 0),
               "first_wall_s": first.get("wall_s"), "retry_wall_s": second.get("wall_s")}
        if row.get("retried"):
            # 1차가 무효라 결정 비교는 불가. 1차 JSON에서 키 자리를 읽을 수 있으면 최종 키와 같은지만 참고로 둔다.
            raw = row.get("raw_first_key")
            row["key_change"] = {"comparable": False, "raw_first_key": raw, "raw_first_key_same": (
                None if raw is None or row["status"] != "valid" else raw == row["key"])}
        runs.append(row)
    return runs


def judge(totals: dict, means: dict, complete: bool) -> tuple[str, list[str]]:
    """사전등록 개선 판정과 문구."""
    if not complete or any(v is None for v in means.values()):
        return "insufficient", [FIXED["wording"]["insufficient"]]
    invalid = totals.get("invalid", 0)
    improved = invalid <= IMPROVE["max_invalid_total"] and all(v >= IMPROVE["min_mean_pass4_each_kind"] for v in means.values())
    decision = "improved" if improved else "not_improved"
    return decision, [FIXED["wording"][decision].format(
        first_valid=totals.get("first_valid", 0), retries=totals.get("retried", 0), invalid=invalid,
        p4d=f"{means['diagnosis']:.2f}", p4c=f"{means['correction']:.2f}")]


def summarize(write: bool = True) -> dict:
    protocol, commit = verify_protocol()
    n2 = read(ROOT / N2_SUMMARY)
    sets, by_kind, totals = {}, {k: {} for k in KINDS}, Counter()
    for set_name in N.SET_ORDER:
        spec = N.INPUT_SETS[set_name]
        sets[set_name] = {"case": spec["case"], "seed": spec["seed"], "mechanism": spec["mechanism"], "kinds": {}}
        for kind in spec["kinds"]:
            if N.sha_raw(N.packet_path(set_name, kind)) != protocol["inputs"]["private"][f"{set_name}/{kind}"]["packet_sha256"]:
                raise ValueError(f"{set_name} {kind}: packet이 사전등록 SHA와 다르다")
            runs = feedback_runs(set_name, kind)
            summary = N.repeat_summary(kind, runs, spec["mechanism"], None)
            n2_decisions = n2["sets"][set_name]["kinds"][kind]["single"]["decisions"]
            modal = next(iter(n2_decisions)) if len(n2_decisions) == 1 else None
            judged = [r for r in runs if r["status"] in ("valid", "invalid", "call_error")]
            first_keys = [r.get("key") if not r.get("retried") and r["status"] == "valid" else None for r in judged]
            changes = [r["key_change"] for r in judged if r.get("key_change")]
            summary.update(
                first_valid=sum(1 for r in judged if r.get("first_status") == "valid"),
                retried=sum(1 for r in judged if r.get("retried")),
                recovered_by_retry=sum(1 for r in judged if r.get("retried") and r["status"] == "valid"),
                first_codes=dict(Counter(c for r in judged for c in r.get("first_codes") or [])),
                final_codes=dict(Counter(c for r in judged if r["status"] == "invalid" for c in r.get("final_codes") or [])),
                key_change={"not_comparable": len(changes),
                            "raw_first_key_same": sum(1 for c in changes if c["raw_first_key_same"] is True),
                            "raw_first_key_differs": sum(1 for c in changes if c["raw_first_key_same"] is False)},
                first_only_pass_k={str(k): N.pass_hat_k(first_keys, k) for k in K_VALUES},
                same_as_n2_decision=None if modal is None else sum(
                    1 for r in judged if r["status"] == "valid" and " · ".join(map(str, r["key"])) == modal),
                first_tokens_total=sum(r.get("first_tokens") or 0 for r in judged),
                retry_tokens_total=sum(r.get("retry_tokens") or 0 for r in judged),
                retry_wall_s_total=sum(r.get("retry_wall_s") or 0 for r in judged))
            n2_single = n2["sets"][set_name]["kinds"][kind]["single"]
            sets[set_name]["kinds"][kind] = {"feedback": summary, "n2_single": {
                key: n2_single[key] for key in ("counts", "n", "decisions", "pass_k", "pass_k_valid")}}
            by_kind[kind][set_name] = summary
            totals.update(first_valid=summary["first_valid"], retried=summary["retried"],
                          recovered_by_retry=summary["recovered_by_retry"])
            for r in runs:
                totals[r["status"]] += 1
    complete = all(s["counts"]["missing"] == 0 and s["counts"]["budget_exhausted"] == 0
                   for kinds in by_kind.values() for s in kinds.values())
    exhausted = API_CAP - requests_used() < MAX_REQUESTS_PER_RUN
    means = {}
    for kind, rows in by_kind.items():
        pass4 = [s["pass_k"]["4"] for s in rows.values()]
        means[kind] = None if any(v is None for v in pass4) else sum(pass4) / len(pass4)
    decision, wording = judge(totals, means, complete)
    use = {kind: N.outcome_wording("피드백 재요청", N.KIND_KO[kind], rows) for kind, rows in by_kind.items()}
    result = {"schema": "n2_feedback_result_v1", "protocol_sha256_lf": N.sha_lf(PROTOCOL), "source_commit_sha": commit,
              "complete": complete, "budget": {"requests_used": requests_used(), "cap_requests": API_CAP, "exhausted": exhausted},
              "totals": dict(sorted(totals.items())), "mean_pass4": means, "improvement": {"decision": decision, **IMPROVE},
              "wording": wording, "use": use, "sets": sets, "common_wording": FIXED["wording"]["common"],
              "n2_baseline": {"path": N2_SUMMARY, "sha256_lf": N.sha_lf(ROOT / N2_SUMMARY)},
              "feedback_policy_sha256": hashlib.sha256(
                  json.dumps(FIXED["retry"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    if write and (complete or exhausted):
        N.write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N2 검증 피드백 1회 재요청 arm")
    ap.add_argument("command", choices=("prepare", "verify", "plan", "run", "summarize"))
    args = ap.parse_args(argv)
    commands = {"prepare": prepare, "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol()[1]},
                "plan": plan, "run": execute, "summarize": summarize}
    try:
        out = commands[args.command]()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
