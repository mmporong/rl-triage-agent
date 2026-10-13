"""E4b v2: 고친 NAT 도구 호출 경로(nat_decision_call_v2)를 E4 시연 고정 packet으로 다시 재는 사전등록·요약기.

GPU와 모델을 쓰지 않는다. NAT 실행은 WSL에서 evals/nat_decision_call_v2.py로 하고, 이 파일은 입력 결합·실행 명령·
기록 재검증·지표만 맡는다. v1(evals/e4b_compare.py)은 v1 사전등록 결과의 소스 결합 때문에 바꾸지 않았다.
v1과 다른 점:
- 입력 세트 두 개를 프로토콜에 SHA로 묶는다. 채택 판정은 시연 v2(seed 2030) 세트로만 한다. seed 2029 세트는
  v1에서 실패를 보고 고친 바로 그 입력이라 수정의 개발 확인으로 따로 보고하고 판정에 쓰지 않는다.
- 기록 재검증에 v2 설정·v1/v2 도구 모듈 SHA와 제출 전 점검 호출 수(도구 기록에서 다시 센 값)를 더했다.
- v1 결과(do_not_adopt)와 분모를 합치지 않는다. 수정 뒤 같은 종류의 입력을 다시 잰 개발 측정이다.

  python evals/e4b_compare_v2.py prepare|verify
  python evals/e4b_compare_v2.py plan          # WSL 실행 명령(세트×종류×5)
  python evals/e4b_compare_v2.py summarize
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

PROTOCOL = ROOT / "bench/protocols/e4b_nat_compare_v2.json"
PRIVATE = ROOT / "bench/private/e4b_v2"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "e4b_nat_compare_v2"
E4_PRIVATE = ROOT / "bench/private/e2e_reward_demo"
INPUT_SETS = {"s2030": "e2e_reward_s2030_20261013", "s2029": "e2e_reward_s2029_20261013"}
ADOPTION_SET = "s2030"
KINDS = ("diagnosis", "correction")
RUNS_PER_KIND = 5
MAX_ATTEMPTS_PER_RUN = 2  # 인프라 오류일 때만 같은 run 번호로 한 번 더
PRECHECK_LIMIT = 2
MODEL = "nvidia/nemotron-3-super-120b-a12b"
CODE = {
    "src/rl_triage/nat_demo_tools_v2.py", "evals/nat_decision_call_v2.py", "evals/e4b_compare_v2.py",
    "configs/nat_demo_v2_diagnosis.yml", "configs/nat_demo_v2_correction.yml",
    "src/rl_triage/nat_demo_tools.py", "evals/nat_decision_call.py",
    "src/rl_triage/active_diagnosis.py", "src/rl_triage/probe_loop.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_registry.py", "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
ADOPTION = {"min_non_infra_runs": 4, "min_valid_runs": 4}
FIXED = {
    "schema": "e4b_nat_compare_protocol_v2",
    "question": ("v1 실패 형태에 맞춰 도구 설명·형식 규칙·제출 전 계약 점검 도구를 고친 NAT 도구 호출 워크플로가 시연 v2 고정 "
                 "입력에서 계약상 유효한 결정을 얼마나 내는가, 단일 호출 결정과 얼마나 같은가"),
    "scope": "fixed_packet_repeated_nat_decision_after_fix_development_measurement_not_accuracy_superiority_or_savings",
    "model": MODEL, "nat_version": "1.9.0", "temperature": 0,
    "input_sets": {
        "s2030": {"tag": INPUT_SETS["s2030"], "role": "adoption",
                  "note": "시연 v2 결정 입력. v1 NAT 비교에 쓰지 않았다. 채택 판정은 이 세트로만 한다"},
        "s2029": {"tag": INPUT_SETS["s2029"], "role": "separate_denominator_not_for_adoption",
                  "note": "v1 NAT 비교(do_not_adopt)의 입력이자 수정 근거가 된 입력. 수정의 개발 확인으로만 따로 보고한다"},
    },
    "runs_per_kind": RUNS_PER_KIND,
    "attempts": "run 번호마다 첫 시도. 인프라 오류(제공자 연결·시간 초과·429·5xx·빈 응답)일 때만 같은 run 번호로 1회 더",
    "status_definitions": {
        "valid": ("제출 도구가 계약 검증을 통과한 결정 하나를 저장했고 worker가 같은 검증기로 다시 통과시켰다. "
                  "제출 뒤 마지막 응답 단계의 오류는 post_submission_error로 따로 기록하고 결정을 버리지 않는다"),
        "invalid": "제출 거절(두 번째 제출 포함, 제출 파일이 있어도 거절 기록이 있으면 무효), 미제출, 반복 한도, 그 밖의 의미 오류",
        "infra_error": "두 번째 시도까지 인프라 오류. 첫 시도만 인프라 오류이고 두 번째 시도가 없으면 미완료로 두고 판정하지 않는다",
    },
    "changes_from_v1": [
        "도구 입력 스키마 필드 설명: 제안 시 hold_reason은 JSON null(문자열 \"None\" 아님), 체크리스트 칸마다 사실 id 하나 "
        "또는 unmet, expected_outcomes는 상위 3가설 키만, 허용 값 목록",
        "도구 규칙(시스템 프롬프트 뒤에 붙는 부분)에 형식 규칙과 점검→제출 순서를 적었다. 판단 계약 프롬프트는 v1과 같다",
        f"제출 전 계약 점검 도구(e4b2_check_decision·e4b2_check_proposal): 제출 도구와 같은 공개 계약 검증기로 결과만 "
        f"돌려주고 저장하지 않는다. 실행당 {PRECHECK_LIMIT}회까지 검증하고 넘으면 검증 없이 거부 메시지를 돌려준다",
        "max_iterations 8 → 12(점검 2회와 여유). 모델·temperature·reasoning_effort·max_retries·사용자 메시지는 v1과 같다",
        "도구 이름 e4b_* → e4b2_*",
    ],
    "v1_failures_fixed_for": ("v1 결과[사실]: 교정 5/5가 hold_reason에 문자열 \"None\", 4/5가 체크리스트 칸에 쉼표로 이은 "
                              "사실 id, 1건이 체크리스트에 교정 id, 진단 2/5가 expected_outcomes에 6개 메커니즘 전부"),
    "comparison_conditions": {
        "same": ["모델·temperature 0·reasoning_effort none(extra_body)", "같은 packet", "같은 판단 계약 프롬프트와 검증기"],
        "different": ["NAT는 사실·카탈로그·등록 표·체크리스트를 도구로 조회하는 여러 턴, 단일 호출은 packet 전체를 한 번에 받는다",
                      "NAT 시스템 프롬프트는 계약 프롬프트 뒤에 도구 사용 규칙을 붙인다",
                      f"NAT는 제출 전 계약 점검 도구로 실행당 최대 {PRECHECK_LIMIT}회 사유를 받아 고칠 수 있다. 단일 호출 경로에는 이 기능이 없다",
                      "NAT 기록은 결정 제출 도구의 저장 파일에서, 단일 호출은 응답 본문에서 결정을 읽는다"],
    },
    "metrics": ["세트·종류별 유효·무효·인프라 실행 수", "단일 호출 결정과의 일치(진단: action·1위·상위 3 집합·probe, 교정: 결정·교정·"
                "회복 방식·미충족 체크리스트)", "점검 호출 수와 점검을 쓴 실행 수", "도구 호출 순서와 수", "LLM 호출 수",
                "입력·출력 토큰", "wall 시간", "단일 호출의 같은 비용 항목(나란히 기록)"],
    "adoption_rule": {**ADOPTION, "set": ADOPTION_SET,
                      "rule": ("다음 seed 시연에 NAT 결정을 쓰려면 seed 2030 세트에서 두 종류 모두 인프라 오류를 뺀 실행이 4 이상이고 "
                               "그중 유효 결정이 4 이상이어야 한다. 수치 조건은 v1과 같다. 일치율은 채택 조건에 쓰지 않는다"
                               "(단일 호출이 정답이 아니다). 요약은 두 세트 실행이 모두 끝난 뒤에만 쓴다")},
    "wording": {
        "adopt": ("고친 NAT 도구 호출 워크플로가 시연 v2 고정 입력에서 사전등록 조건만큼 유효한 결정을 냈다. 다음 seed 시연의 "
                  "결정 단계에 쓸 수 있다. 정확도·우위의 근거가 아니다."),
        "do_not_adopt": ("고친 NAT 도구 호출 워크플로의 유효 결정이 시연 v2 고정 입력에서 사전등록 조건에 못 미쳤다. 다음 seed "
                         "시연은 단일 호출 결정을 유지한다."),
        "s2029": ("seed 2029 세트는 v1에서 실패를 보고 고친 바로 그 입력이다. 수정의 개발 확인으로만 보고하고 채택 판정에 "
                  "쓰지 않으며 seed 2030 세트와 분모를 합치지 않는다."),
        "common": ("E4 시연 고정 입력을 반복 호출한 개발 측정이다. v1 실패를 보고 도구 설명·형식 규칙·제출 전 계약 점검 도구"
                   f"(실행당 최대 {PRECHECK_LIMIT}회, 단일 호출 경로에는 없는 기능)를 더한 뒤 다시 쟀다. v1 결과(do_not_adopt)와 "
                   "분모를 합치지 않는다. NAT는 근거를 도구로 나눠 조회하고 점검으로 고칠 기회가 있으므로 일치율·비용 차이에는 "
                   "이 방식 차이가 섞여 있다. 일반화·정확도·절감을 주장하지 않는다."),
    },
    "forbidden_claims": ["NAT가 더 정확하다", "모든 결정이 NAT로 이뤄졌다", "토큰·시간 절감", "일반화",
                         "v1과 v2를 합친 유효율", "점검 도구가 없는 단일 호출과 같은 조건의 비교"],
}
DYNAMIC = {"source_files", "inputs"}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


# --- 입력 결합 ---
def _latest_diagnosis_call(model_dir: Path) -> Path | None:
    calls = sorted((p for p in model_dir.glob("diagnosis.call*.json") if p.name[14:-5].isdigit()),
                   key=lambda p: int(p.name[14:-5]))
    return calls[-1] if calls else None


def bind_inputs(tag: str) -> dict:
    """E4 산출물에서 두 종류의 packet과 단일 호출 기록을 찾아 SHA를 대조한다. 하나라도 없으면 거부한다."""
    import correction_model_call as CM
    model_dir = E4_PRIVATE / tag / "model"
    packet, call = model_dir / "diagnosis.packet.json", _latest_diagnosis_call(model_dir) if model_dir.exists() else None
    if not packet.is_file() or call is None:
        raise ValueError(f"{tag}: 진단 packet 또는 단일 호출 기록이 없다")
    if read(call).get("schema") != "active_model_call_v1" or read(call).get("packet_sha256") != sha_raw(packet):
        raise ValueError(f"{tag}: 진단 단일 호출 기록이 진단 packet과 다르다")
    out = {"diagnosis": {"packet": rel(packet), "packet_sha256": sha_raw(packet),
                         "single_call": rel(call), "single_call_sha256": sha_raw(call)}}
    packet, call = model_dir / "correction.packet.json", model_dir / "correction.call1.json"
    if not packet.is_file() or not call.is_file():
        raise ValueError(f"{tag}: 교정 packet 또는 단일 호출 기록이 없다")
    if read(call).get("schema") != CM.RECORD_SCHEMA or read(call).get("packet_digest") != CM.packet_digest(read(packet)):
        raise ValueError(f"{tag}: 교정 단일 호출 기록이 교정 packet과 다르다")
    out["correction"] = {"packet": rel(packet), "packet_sha256": sha_raw(packet),
                         "single_call": rel(call), "single_call_sha256": sha_raw(call)}
    return out


# --- 프로토콜 ---
def dynamic_fields() -> dict:
    return {"source_files": {name: sha_lf(ROOT / name) for name in sorted(CODE)},
            "inputs": {name: bind_inputs(tag) for name, tag in INPUT_SETS.items()}}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed comparison contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources or inputs changed")


def prepare() -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_runs", "path": rel(PROTOCOL)}


def verify_protocol() -> tuple[dict, str]:
    import subprocess
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    try:  # 추적되지 않은 결합 파일도 CLI가 거부로 보고하게 한다
        return protocol, P.verify_committed(ROOT, CODE | {rel(PROTOCOL)})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


def run_path(set_name: str, kind: str, run: int, attempt: int) -> Path:
    if set_name not in INPUT_SETS:
        raise ValueError(f"모르는 입력 세트 {set_name}")
    return PRIVATE / set_name / kind / f"run{run}_a{attempt}.json"


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def plan() -> dict:
    protocol, commit = verify_protocol()
    commands = []
    for set_name in INPUT_SETS:
        for kind in KINDS:
            packet = ROOT / protocol["inputs"][set_name][kind]["packet"]
            for run in range(1, RUNS_PER_KIND + 1):
                commands.append(
                    "wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"$HOME/.venvs/rl-triage/bin/python\" "
                    f"{wsl(ROOT / 'evals/nat_decision_call_v2.py')} --kind {kind} --packet {wsl(packet)} "
                    f"--output {wsl(run_path(set_name, kind, run, 1))}'")
    return {"source_commit_sha": commit, "protocol_sha256_lf": sha_lf(PROTOCOL), "commands": commands,
            "retry_rule": "인프라 오류 기록(status=infra_error)이면 같은 명령의 출력만 run<i>_a2.json으로 바꿔 1회 더 실행"}


# --- 기록 재검증과 지표 ---
def single_call_decision(kind: str, inputs: dict) -> dict | None:
    """단일 호출 기록을 같은 계약으로 다시 검증한 결정. 무효면 None."""
    record, packet = read(ROOT / inputs[kind]["single_call"]), read(ROOT / inputs[kind]["packet"])
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        text = record.get("content") or ""
        if text.startswith("```json") and text.rstrip().endswith("```"):
            text = text[7:text.rfind("```")].strip()
        try:
            return D.validate_decision(json.loads(text), packet)
        except (ValueError, TypeError):
            return None
    import correction_model_call as CM
    proposal, error = CM.parse_content(record.get("content"))
    if error:
        return None
    verdict = CM.validate_model_proposal(proposal, packet)
    return {"proposal": proposal, **verdict} if verdict["accepted"] else None


def agreement(kind: str, nat: dict, single: dict | None) -> dict | None:
    if single is None:
        return None
    if kind == "diagnosis":
        return {"action": nat["action"] == single["action"], "top1": nat["ranking"][0] == single["ranking"][0],
                "top3_set": set(nat["ranking"][:3]) == set(single["ranking"][:3]), "probe": nat["probe"] == single["probe"]}
    return {"decision": nat["decision"] == single["decision"],
            "correction_id": nat["proposal"]["correction_id"] == single["proposal"]["correction_id"],
            "recovery_mode": nat["proposal"]["recovery_mode"] == single["proposal"]["recovery_mode"],
            "unmet_checklist": nat["unmet_checklist"] == single["unmet_checklist"]}


def verify_record(path: Path, record: dict, kind: str, inputs: dict, protocol: dict) -> None:
    """NAT v2 기록의 결합값과, 사건·도구 기록·궤적·거절 파일에서 다시 계산한 값(점검 호출 수 포함)을 대조한다."""
    import nat_decision_call as W1
    import nat_decision_call_v2 as W2
    expected = {"schema": W2.RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0, "nat_version": "1.9.0",
                "semantic_retries": 0, "precheck_limit": PRECHECK_LIMIT, "packet_sha256": inputs[kind]["packet_sha256"],
                "config_sha256_lf": protocol["source_files"][f"configs/nat_demo_v2_{kind}.yml"],
                "tools_module_sha256_lf": protocol["source_files"]["src/rl_triage/nat_demo_tools_v2.py"],
                "base_tools_module_sha256_lf": protocol["source_files"]["src/rl_triage/nat_demo_tools.py"],
                "system_prompt_sha256": W1.text_sha(W2.system_prompt(kind)),
                "user_message_sha256": W1.text_sha(W1.user_message(kind, read(ROOT / inputs[kind]["packet"])))}
    if any(record.get(k) != v for k, v in expected.items()) or record.get("status") not in ("valid", "invalid", "infra_error"):
        raise ValueError(f"{path.name}: NAT 기록이 등록 입력·설정·프롬프트와 다르다")
    events = path.with_suffix(".events.json")
    if record.get("events_sha256") != sha_raw(events):
        raise ValueError(f"{path.name}: 사건 기록이 바뀌었다")
    summary = W1.summarize_events(read(events)["events"])
    if any(record.get(k) != v for k, v in summary.items()):
        raise ValueError(f"{path.name}: 기록의 호출 수·토큰·도구 순서가 사건 파일과 다르다")
    trace = path.with_suffix(".tools.jsonl")
    if record.get("tools_trace_sha256") != sha_raw(trace):
        raise ValueError(f"{path.name}: 도구 호출 기록이 바뀌었다")
    if record.get("precheck_calls") != W2.precheck_calls(trace):
        raise ValueError(f"{path.name}: 점검 호출 수가 도구 기록과 다르다")
    if record.get("atif_sha256") is not None and record["atif_sha256"] != sha_raw(path.with_suffix(".atif.json")):
        raise ValueError(f"{path.name}: ATIF 궤적이 바뀌었다")
    rejected = path.with_suffix(".submission.rejected.json")
    if rejected.exists() != ("rejection_sha256" in record) or (rejected.exists() and record["rejection_sha256"] != sha_raw(rejected)):
        raise ValueError(f"{path.name}: 거절 기록 결합이 맞지 않는다")
    if record["status"] == "valid" and rejected.exists():
        raise ValueError(f"{path.name}: 거절 기록이 있는데 유효로 기록됐다")


def load_run(set_name: str, kind: str, run: int, inputs: dict, protocol: dict) -> dict:
    """run 번호의 최종 시도. 첫 시도가 인프라 오류일 때만 두 번째 시도를 본다. 재시도가 남았으면 missing."""
    import nat_decision_call as W1
    attempts = [run_path(set_name, kind, run, a) for a in range(1, MAX_ATTEMPTS_PER_RUN + 1)]
    records = [(i + 1, p, read(p)) for i, p in enumerate(attempts) if p.is_file()]
    if [a for a, _, _ in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"{set_name} {kind} run{run}: 시도 번호가 이어지지 않는다")
    for attempt, path, record in records:
        verify_record(path, record, kind, inputs, protocol)
        if attempt < len(records) and record["status"] != "infra_error":
            raise ValueError(f"{set_name} {kind} run{run}: 인프라 오류가 아닌 시도 뒤에 재시도 기록이 있다")
    if not records:
        return {"run": run, "status": "missing"}
    attempt, path, record = records[-1]
    if record["status"] == "infra_error" and attempt < MAX_ATTEMPTS_PER_RUN:
        return {"run": run, "status": "missing", "pending": "infra_retry"}
    if record["status"] == "valid":
        submission = path.with_suffix(".submission.json")
        if sha_raw(submission) != record["submission_sha256"]:
            raise ValueError(f"{path.name}: 제출 파일이 바뀌었다")
        if W1.validate_submission(kind, read(ROOT / inputs[kind]["packet"]), read(submission)) != record["decision"]:
            raise ValueError(f"{path.name}: 제출이 다시 검증되지 않는다")
    return {"run": run, "attempt": attempt, "status": record["status"], "decision": record["decision"],
            "error_type": record.get("error_type"), "precheck_calls": record.get("precheck_calls"),
            "tool_sequence": record.get("tool_sequence"), "llm_calls": record.get("llm_calls"),
            "input_tokens": record.get("input_tokens"), "output_tokens": record.get("output_tokens"),
            "wall_s": record.get("wall_s"), "atif_sha256": record.get("atif_sha256")}


def _stats(values: list) -> dict | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return {"n": len(numbers), "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers)}


def kind_summary(set_name: str, kind: str, inputs: dict, protocol: dict) -> dict:
    runs = [load_run(set_name, kind, run, inputs, protocol) for run in range(1, RUNS_PER_KIND + 1)]
    single = single_call_decision(kind, inputs)
    valid = [r for r in runs if r["status"] == "valid"]
    counts = {status: sum(1 for r in runs if r["status"] == status) for status in ("valid", "invalid", "infra_error", "missing")}
    matches = [agreement(kind, r["decision"], single) for r in valid]
    agree = None if single is None or not matches else {key: sum(1 for m in matches if m[key]) for key in matches[0]}
    done = [r for r in runs if r["status"] != "missing"]
    sequences = [" > ".join(r["tool_sequence"] or []) for r in done]
    record = read(ROOT / inputs[kind]["single_call"])
    return {"counts": counts, "non_infra_runs": counts["valid"] + counts["invalid"],
            "single_call_valid": single is not None, "agreement_with_single_call": agree,
            "agreement_denominator": len(valid) if single is not None else 0,
            "runs_using_precheck": sum(1 for r in done if (r.get("precheck_calls") or 0) > 0),
            "tool_sequences": {seq: sequences.count(seq) for seq in sorted(set(sequences))},
            **{key: _stats([r.get(key) for r in runs])
               for key in ("precheck_calls", "llm_calls", "input_tokens", "output_tokens", "wall_s")},
            "single_call_costs": {k: record.get(k) for k in ("input_tokens", "output_tokens", "execution_wall_s")},
            "runs": runs, "complete": counts["missing"] == 0}


def meets_rule(summary: dict) -> bool:
    return all(s["non_infra_runs"] >= ADOPTION["min_non_infra_runs"] and s["counts"]["valid"] >= ADOPTION["min_valid_runs"]
               for s in summary.values())


def summarize() -> dict:
    protocol, commit = verify_protocol()
    sets = {name: {kind: kind_summary(name, kind, protocol["inputs"][name], protocol) for kind in KINDS}
            for name in INPUT_SETS}
    complete = all(s["complete"] for kinds in sets.values() for s in kinds.values())
    decision = None if not complete else ("adopt" if meets_rule(sets[ADOPTION_SET]) else "do_not_adopt")
    result = {"schema": "e4b_nat_compare_result_v2", "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit,
              "sets": {name: {"tag": INPUT_SETS[name], "role": FIXED["input_sets"][name]["role"], "kinds": kinds,
                              "meets_numeric_rule": meets_rule(kinds) if complete else None}
                       for name, kinds in sets.items()},
              "adoption": {"decision": decision, "set": ADOPTION_SET, "rule": FIXED["adoption_rule"]["rule"],
                           "reason": None if complete else "두 세트 실행이 모두 끝나지 않았다"},
              "wording": None if decision is None else FIXED["wording"][decision],
              "s2029_wording": FIXED["wording"]["s2029"], "common_wording": FIXED["wording"]["common"]}
    if decision is not None:
        write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4b v2 NAT 결정 재측정")
    ap.add_argument("command", choices=("prepare", "verify", "plan", "summarize"))
    args = ap.parse_args(argv)
    commands = {"prepare": prepare, "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol()[1]},
                "plan": plan, "summarize": summarize}
    try:
        out = commands[args.command]()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
