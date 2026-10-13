"""N2 구조화 출력 arm: N2와 같은 11개 packet × 5회 단일 호출에 출력 형식(response_format json_schema strict)만 바꿔
계약 위반 출력과 결정 일치(pass^k)를 다시 잰다. 사전등록·실행 계획·실행·요약을 한 파일에서 한다.

- 바꾸는 변수는 출력 형식 하나다. 모델·temperature 0·reasoning_effort none·max_tokens·메시지(프롬프트 본문)는 N2 단일
  호출과 같다. 호출기는 evals/structured_model_call.py다(기존 active_model_call·correction_model_call·N2 파일은 바꾸지 않는다).
- schema는 구조만 정한다(필드·필수·형·추가 필드 금지). 의미 규칙은 넣지 않고 기존 계약 검증기가 그대로 센다.
- 무효는 두 갈래로 나눠 센다: 문법(JSON 파손·객체 아님)과 계약(검증기 거절: 빈 claim, 카탈로그 밖 expected_outcomes 등).
- 비교 기준은 N2 단일 호출 결과(evals/results/n2_pass_k_v1/summary.json, 이미 본 값)다. "개선"이라는 말은 사전등록
  기준(무효 ≤ 1/55이고 두 종류 평균 pass^4 ≥ 0.8)을 넘을 때만 쓴다.
- 이미 본 packet의 개발 측정이다. 정확도·우월성·일반화·절감의 근거가 아니다.

  python evals/n2_structured.py prepare|verify|plan|run|summarize
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import n2_pass_k as N  # noqa: E402  입력 세트·packet 경로·결정 키·pass^k 정의를 그대로 쓴다
import structured_model_call as S  # noqa: E402

CODE_ROOT = ROOT
PROTOCOL = ROOT / "bench/protocols/n2_structured_v1.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/n2_structured"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "n2_structured_v1"
N2_SUMMARY = "evals/results/n2_pass_k_v1/summary.json"
MODEL = N.MODEL
KINDS = N.KINDS
RUNS = N.RUNS
K_VALUES = N.K_VALUES
API_CAP = 60
MAX_ATTEMPTS = 2
CALL_TIMEOUT_S = 420
USE_THRESHOLD = N.USE_THRESHOLD
IMPROVE = {"max_invalid_total": 1, "min_mean_pass4_each_kind": 0.8}
SCRIPT = "evals/structured_model_call.py"
CODE = N.CODE | {"evals/n2_structured.py", "evals/structured_model_call.py"}
FIXED = {
    "schema": "n2_structured_protocol_v1",
    "question": ("N2와 같은 packet·같은 프롬프트에 출력 형식(response_format json_schema strict)만 더하면 계약 위반 출력(문법·계약)과 "
                 "결정 일치(pass^k)가 어떻게 되는가"),
    "scope": "structured_output_arm_on_already_seen_packets_development_measurement_not_accuracy_superiority_generalization_or_savings",
    "model": MODEL, "temperature": 0, "reasoning_effort": "none", "max_tokens": 16384,
    "variable": {"changed": "response_format: json_schema(strict) 하나",
                 "same": ["모델", "temperature 0", "reasoning_effort none", "max_tokens 16384",
                          "메시지(active_diagnosis.decision_messages, correction_model_call.build_messages)",
                          "packet(N2 프로토콜이 SHA로 묶은 11개)", "계약 검증기(validate_decision, validate_model_proposal)"]},
    "schemas": S.SCHEMAS,
    "schema_policy": ("구조만: 계약의 필드 이름·필수 여부·형(널 허용 포함)·추가 필드 금지. enum·최소 개수·packet별 규칙(순열, 카탈로그 "
                      "기대 결과, 사실 id·값, 등록 교정)은 넣지 않는다. 그 의미 위반은 기존 검증기가 센다"),
    "feasibility": {
        "official_docs": ("NIM LLM 문서(1.13)는 nvext.guided_json으로 JSON schema를 주라고 권하고 response_format json_schema는 다루지 "
                          "않는다. 호스팅 모델 API 참조(nvidia-nemotron-3-super-120b-a12b-infer)는 요청 필드로 response_format·nvext를 "
                          "나열하지 않는다. 그래서 지원 여부를 합성 최소 입력 호출로 확인했다"),
        "calls": 2,
        "finding": ("평가 packet이 아닌 합성 입력 두 번에서 서버가 strict json_schema를 강제했다: 프롬프트가 다른 JSON({\"answer\": "
                    "\"hello\"})을 요구했는데 출력이 schema를 따랐다. 널 허용·객체 배열·문자열 맵 schema도 따랐다. 값은 의미 없는 "
                    "문자열이 나올 수 있었다(형식만 보장)"),
    },
    "input_sets": "N2 프로토콜(bench/protocols/n2_pass_k_v1.json)의 input_sets 11개 packet·종류 그대로",
    "runs_per_packet": RUNS, "k_values": list(K_VALUES),
    "order": "N2와 같다: run 번호 1~5를 바깥 반복, 세트 순서·종류를 안쪽 반복",
    "attempts": "run 번호마다 첫 시도. 인프라 오류(연결·시간 초과·429·5xx·빈 응답·잘림)일 때만 같은 run 번호로 1회 더",
    "api_budget": {"cap_requests": API_CAP, "planned_requests": len(N.schedule()),
                   "counting": "기록 파일 하나가 요청 하나(인프라 재시도 포함). 상한에 닿으면 남은 실행은 budget_exhausted",
                   "feasibility_requests_outside_cap": 2},
    "invalid_categories": {"syntax": "응답 본문(```json 울타리 제거 뒤)이 JSON 객체로 읽히지 않는다",
                           "contract": "JSON 객체로 읽히지만 계약 검증기가 거절했다(빈 claim, 카탈로그 밖 expected_outcomes, 필드 불일치 등)"},
    "pass_k": "N2와 같다: Σ_g C(c_g,k)/C(n,k), n은 인프라 오류를 뺀 실행 수, c_g는 결정 키가 g인 유효 실행 수",
    "comparison": {"baseline": N2_SUMMARY,
                   "note": ("N2 단일 호출(이미 본 결과)과 packet별로 나란히 둔다. 시점이 다른 반복이며 같은 packet·프롬프트다. 유효 결정이 "
                            "N2의 packet별 결정과 같은지도 센다")},
    "improvement_rule": {**IMPROVE, "rule": ("구조화 arm의 무효(문법+계약)가 55회 중 1 이하이고, 진단·교정 각각 packet 평균 pass^4가 0.8 "
                                             "이상일 때만 '개선'이라고 쓴다. 아니면 개선을 말하지 않고 수치만 적는다. 문법 무효가 0이면 "
                                             "그 사실은 따로 적을 수 있다")},
    "use_rule": {"threshold_pass4": USE_THRESHOLD, "rule": "N2와 같다: 종류별 packet 평균 pass^4 ≥ 0.8이면 usable, 아니면 reconsider"},
    "wording": {
        "improved": ("출력 형식만 바꾼 구조화 arm이 사전등록 기준(무효 ≤ 1/55, 진단·교정 평균 pass^4 ≥ 0.8)을 넘었다: 무효 {invalid}/55, "
                     "평균 pass^4 진단 {p4d}·교정 {p4c}. 같은 packet의 N2 단일 호출(무효 6/55)보다 개선됐다."),
        "not_improved": ("출력 형식만 바꾼 구조화 arm은 사전등록 개선 기준(무효 ≤ 1/55, 진단·교정 평균 pass^4 ≥ 0.8)을 넘지 못했다: 무효 "
                         "{invalid}/55(문법 {syntax}, 계약 {contract}), 평균 pass^4 진단 {p4d}·교정 {p4c}. 개선을 주장하지 않는다."),
        "syntax_zero": "구조화 arm에서 문법 무효(JSON 파손)는 0이었다. 계약 위반은 따로 센다.",
        "insufficient": "인프라 오류·예산 소진으로 일부 packet의 pass^4를 계산하지 못해 개선 판정을 하지 않는다.",
        "common": ("이미 본 packet을 같은 모델·temperature 0·같은 프롬프트로 반복하고 출력 형식만 바꾼 개발 측정이다. 반복 일관성·유효율은 "
                   "정확도의 근거가 아니다. N2 결과는 이미 본 값이고 시점이 다른 반복이다. 비용은 토큰·벽시계 기록이며 절감 주장이 아니다."),
    },
    "forbidden_claims": ["LLM이 규칙보다 우월하다", "GPU·토큰·시간 절감", "일반화", "정확도", "최초",
                         "사전등록 기준 없이 '개선'", "구조화 출력이 진단 내용(순위·probe)을 좋게 했다"],
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
        raise ValueError("fixed structured-arm contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources or inputs changed")


def prepare() -> dict:
    N.write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_any_call", "path": rel(PROTOCOL), "planned_requests": len(N.schedule())}


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
    """WSL에서 자격증명 파일을 source한 셸로 구조화 호출기를 부른다. 키 값은 이 프로세스가 보지 않는다."""
    shell = ('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
             f"{N.wsl(CODE_ROOT / SCRIPT)} --kind {kind} --packet {N.wsl(N.packet_path(set_name, kind))} "
             f"--output {N.wsl(run_path(set_name, kind, run, attempt))} --model {MODEL}")
    return ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", shell]


def requests_used() -> int:
    return sum(1 for _ in (PRIVATE / "runs").glob("*/*/run*_a*.json")) if (PRIVATE / "runs").exists() else 0


def plan() -> dict:
    protocol, commit = verify_protocol()
    return {"source_commit_sha": commit, "protocol_sha256_lf": N.sha_lf(PROTOCOL), "planned_requests": len(N.schedule()),
            "cap_requests": API_CAP, "requests_used": requests_used(),
            "first_commands": [" ".join(call_argv(s, k, r, 1)) for s, k, r in N.schedule()[:2]]}


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


def execute(runner=None) -> dict:
    """등록 순서대로 남은 실행을 부른다. 끝난 실행은 건너뛰고(재개 가능), 상한에 닿으면 멈춘다."""
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
            if requests_used() >= API_CAP:
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
    return {"calls_made": done, "requests_used": requests_used(), "cap_requests": API_CAP, "stopped": stopped,
            "remaining": sum(1 for s, k, r in N.schedule() if final_record(s, k, r)[0] is None)}


# --- 기록 재검증 ---
def _body(content) -> str:
    text = (content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    return fenced.group(1) if fenced else text


def invalid_category(content) -> str:
    """무효 출력의 갈래: 본문이 JSON 객체로 읽히지 않으면 syntax, 읽히면 contract."""
    try:
        return "contract" if isinstance(json.loads(_body(content)), dict) else "syntax"
    except json.JSONDecodeError:
        return "syntax"


def review(kind: str, record: dict, packet: Path) -> dict:
    """구조화 호출 기록 → {status, decision?, category?}. 기존 검토 함수에 기록 형식 이름만 바꿔 넘긴다(내용 검증은 그대로)."""
    if record.get("schema") != S.RECORD_SCHEMA or record.get("kind") != kind:
        raise ValueError("구조화 호출 기록 형식이 아니다")
    if record.get("response_format_sha256") != S.response_format_sha256(kind) or (
            record.get("request") or {}).get("response_format") != S.response_format(kind):
        raise ValueError("등록 출력 형식으로 부른 호출이 아니다")
    if record.get("packet_sha256") != N.sha_raw(packet):
        raise ValueError("호출 기록의 packet이 등록 packet과 다르다")
    import correction_model_call as CM
    legacy = "active_model_call_v1" if kind == "diagnosis" else CM.RECORD_SCHEMA
    review_out = N.review_single(kind, {**record, "schema": legacy}, packet)
    if review_out["status"] == "invalid":
        review_out = {**review_out, "category": invalid_category(record.get("content"))}
    return review_out


def structured_runs(set_name: str, kind: str) -> list[dict]:
    packet = N.packet_path(set_name, kind)
    exhausted = requests_used() >= API_CAP
    runs = []
    for run in range(1, RUNS + 1):
        path, record = final_record(set_name, kind, run)
        if record is None:
            runs.append({"run": run, "status": "budget_exhausted" if exhausted else "missing"})
            continue
        out = review(kind, record, packet)
        row = {"run": run, "status": out["status"], "input_tokens": record.get("input_tokens"),
               "output_tokens": record.get("output_tokens"), "wall_s": record.get("execution_wall_s"), "llm_calls": 1,
               "record_sha256": N.sha_raw(path)}
        if out["status"] == "valid":
            row["key"] = list(N.decision_key(kind, out["decision"]))
            if kind == "diagnosis":
                row["probe"] = out["decision"].get("probe")
        elif out["status"] == "invalid":
            row["category"] = out["category"]
            row["reason"] = (out.get("reason") or "")[:200]
        runs.append(row)
    return runs


def n2_modal_decision(n2: dict, set_name: str, kind: str) -> str | None:
    """N2 단일 호출의 유효 결정이 하나뿐이면 그 결정('a · b' 표기). 없거나 여러 개면 None."""
    decisions = n2["sets"][set_name]["kinds"][kind]["single"]["decisions"]
    return next(iter(decisions)) if len(decisions) == 1 else None


def judge(totals: dict, means: dict, complete: bool) -> tuple[str, list[str]]:
    """사전등록 개선 판정과 문구. 개선은 무효 ≤ 1/55이고 진단·교정 평균 pass^4가 모두 0.8 이상일 때만."""
    if not complete or any(v is None for v in means.values()):
        return "insufficient", [FIXED["wording"]["insufficient"]]
    invalid = totals.get("invalid", 0)
    improved = invalid <= IMPROVE["max_invalid_total"] and all(v >= IMPROVE["min_mean_pass4_each_kind"] for v in means.values())
    decision = "improved" if improved else "not_improved"
    wording = [FIXED["wording"][decision].format(
        invalid=invalid, syntax=totals.get("invalid_syntax", 0), contract=totals.get("invalid_contract", 0),
        p4d=f"{means['diagnosis']:.2f}", p4c=f"{means['correction']:.2f}")]
    if totals.get("invalid_syntax", 0) == 0:
        wording.append(FIXED["wording"]["syntax_zero"])
    return decision, wording


def summarize(write: bool = True) -> dict:
    protocol, commit = verify_protocol()
    n2 = read(ROOT / N2_SUMMARY)
    sets, by_kind = {}, {k: {} for k in KINDS}
    totals = Counter()
    for set_name in N.SET_ORDER:
        spec = N.INPUT_SETS[set_name]
        sets[set_name] = {"case": spec["case"], "seed": spec["seed"], "mechanism": spec["mechanism"], "kinds": {}}
        for kind in spec["kinds"]:
            if N.sha_raw(N.packet_path(set_name, kind)) != protocol["inputs"]["private"][f"{set_name}/{kind}"]["packet_sha256"]:
                raise ValueError(f"{set_name} {kind}: packet이 사전등록 SHA와 다르다")
            runs = structured_runs(set_name, kind)
            summary = N.repeat_summary(kind, runs, spec["mechanism"], None)
            modal = n2_modal_decision(n2, set_name, kind)
            valid_keys = [" · ".join(map(str, r["key"])) for r in runs if r["status"] == "valid"]
            summary.update(
                invalid_categories=dict(Counter(r["category"] for r in runs if r["status"] == "invalid")),
                invalid_reasons=[r["reason"] for r in runs if r["status"] == "invalid"],
                same_as_n2_decision=None if modal is None else sum(1 for k in valid_keys if k == modal))
            n2_single = n2["sets"][set_name]["kinds"][kind]["single"]
            sets[set_name]["kinds"][kind] = {
                "structured": summary,
                "n2_single": {key: n2_single[key] for key in ("counts", "n", "decisions", "pass_k", "pass_k_valid")}}
            by_kind[kind][set_name] = summary
            for r in runs:
                totals[r["status"]] += 1
                if r["status"] == "invalid":
                    totals[f"invalid_{r['category']}"] += 1
    complete = all(s["counts"]["missing"] == 0 and s["counts"]["budget_exhausted"] == 0
                   for kinds in by_kind.values() for s in kinds.values())
    exhausted = requests_used() >= API_CAP
    means = {}
    for kind, rows in by_kind.items():
        pass4 = [s["pass_k"]["4"] for s in rows.values()]
        means[kind] = None if any(v is None for v in pass4) else sum(pass4) / len(pass4)
    decision, wording = judge(totals, means, complete)
    use = {kind: N.outcome_wording("구조화 출력", N.KIND_KO[kind], rows) for kind, rows in by_kind.items()}
    result = {"schema": "n2_structured_result_v1", "protocol_sha256_lf": N.sha_lf(PROTOCOL), "source_commit_sha": commit,
              "complete": complete, "budget": {"requests_used": requests_used(), "cap_requests": API_CAP, "exhausted": exhausted},
              "totals": dict(sorted(totals.items())), "mean_pass4": means, "improvement": {"decision": decision, **IMPROVE},
              "wording": wording, "use": use, "sets": sets, "common_wording": FIXED["wording"]["common"],
              "n2_baseline": {"path": N2_SUMMARY, "sha256_lf": N.sha_lf(ROOT / N2_SUMMARY)}}
    if write and (complete or exhausted):
        N.write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N2 구조화 출력 arm")
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
