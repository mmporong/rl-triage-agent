"""E4b v3: null 없는 교정 제출 스키마의 NAT 경로를 새 입력으로 다시 재는 사전등록·요약기.

GPU와 모델을 쓰지 않는다. NAT 실행은 WSL에서 evals/nat_decision_call_v3.py로 하고, 이 파일은 입력 결합·실행 명령·
기록 재검증·지표만 맡는다. v1·v2 비교기는 사전등록 결과의 소스 결합 때문에 바꾸지 않았다. v2와 다른 점:
- 채택 판정 입력은 E4b v1·v2에 쓰지 않은 h02_s2027(보상 결함, 고리 confirmed)의 진단·교정 packet이다. 공개 기록
  (bench/runs_p0c 텔레메트리, evals/loops/h02_s2027, P0-C 고정 평가)에서 결정적으로 만들어 SHA로 묶고, verify 때마다
  같은 원천에서 다시 만들어 대조한다. 이미 본 seed(2027)이며 설계자가 본 사례라 일반화 근거가 아니다.
- 개발 분모(교정만): v2에서 실패한 s2030·s2029 교정 packet, 그리고 등록 밖 메커니즘(exploration)이 확정된 h05_s2027
  교정 packet(보류 경로 확인). 채택 판정에 쓰지 않고 채택 세트와 분모를 합치지 않는다.
- 진단 결정은 v2 도구·설정 그대로다(v2에서 5/5 유효). 교정만 null 없는 v3 제출 스키마다.

  python evals/e4b_compare_v3.py prepare|verify|plan|summarize
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

CODE_ROOT = ROOT  # 코드 파일의 위치. 입력·결과 위치(ROOT)와 따로 둬 테스트가 ROOT만 임시 폴더로 바꿀 수 있다
PROTOCOL = ROOT / "bench/protocols/e4b_nat_compare_v3.json"
PRIVATE = ROOT / "bench/private/e4b_v3"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "e4b_nat_compare_v3"
E4_PRIVATE = ROOT / "bench/private/e2e_reward_demo"
P0C_RUNS = "evals/results/p0c_fixed_eval_20261009/runs"
KINDS = ("diagnosis", "correction")
INPUT_SETS = {
    "h02_s2027": {"role": "adoption", "kinds": list(KINDS), "source": "public", "case": "h02_s2027",
                  "note": "E4b v1·v2에 쓰지 않은 입력. 보상 결함 고리 confirmed. 채택 판정은 이 세트로만 한다"},
    "s2030": {"role": "development", "kinds": ["correction"], "source": "e2e", "tag": "e2e_reward_s2030_20261013",
              "note": "v2 교정 0/5 입력. 수정의 개발 확인"},
    "s2029": {"role": "development", "kinds": ["correction"], "source": "e2e", "tag": "e2e_reward_s2029_20261013",
              "note": "v1·v2 교정 0/5 입력. 수정의 개발 확인"},
    "h05_s2027": {"role": "development_hold_branch", "kinds": ["correction"], "source": "public", "case": "h05_s2027",
                  "note": "등록 밖 메커니즘(exploration) 확정 고리. 결정적 규칙은 HOLD이며 hold 경로를 확인한다"},
}
ADOPTION_SET = "h02_s2027"
RUNS_PER_KIND = 5
MAX_ATTEMPTS_PER_RUN = 2  # 인프라 오류일 때만 같은 run 번호로 한 번 더
PRECHECK_LIMIT = 2
MODEL = "nvidia/nemotron-3-super-120b-a12b"
DIAGNOSIS_BUILD = {"max_probes": 4, "probe_execution_budget_s": 180}  # 종단 시연 packet 단계와 같은 값
CODE = {
    "src/rl_triage/nat_demo_tools_v3.py", "evals/nat_decision_call_v3.py", "evals/e4b_compare_v3.py",
    "configs/nat_demo_v3_correction.yml", "configs/nat_demo_v2_diagnosis.yml",
    "src/rl_triage/nat_demo_tools_v2.py", "evals/nat_decision_call_v2.py",
    "src/rl_triage/nat_demo_tools.py", "evals/nat_decision_call.py",
    "src/rl_triage/active_diagnosis.py", "src/rl_triage/probe_loop.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_registry.py", "src/rl_triage/correction_ledger.py", "evals/correct.py",
    "evals/run_active_diagnosis.py", "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
INPUT_SOURCES = {
    "bench/runs_p0c/h02_s2027.telemetry.json", "bench/runs_p0c/baseline_p0c_s2027.telemetry.json",
    f"{P0C_RUNS}/h02_s2027.json", f"{P0C_RUNS}/h05_s2027.json",
    *(f"evals/loops/{case}/{name}" for case in ("h02_s2027", "h05_s2027")
      for name in ("prereg.json", "state.json", "ledger.jsonl", "reference.json")),
}
ADOPTION = {"min_non_infra_runs": 4, "min_valid_runs": 4}
FIXED = {
    "schema": "e4b_nat_compare_protocol_v3",
    "question": ("null 없는 교정 제출 스키마로 고친 NAT 도구 호출 워크플로가 E4b v1·v2에 쓰지 않은 입력에서 계약상 유효한 "
                 "결정을 얼마나 내는가"),
    "scope": "fixed_packet_repeated_nat_decision_after_fix_development_measurement_not_accuracy_superiority_or_savings",
    "model": MODEL, "nat_version": "1.9.0", "temperature": 0,
    "input_sets": INPUT_SETS, "adoption_set": ADOPTION_SET, "diagnosis_packet_build": DIAGNOSIS_BUILD,
    "runs_per_kind": RUNS_PER_KIND,
    "attempts": "run 번호마다 첫 시도. 인프라 오류(제공자 연결·시간 초과·429·5xx·빈 응답)일 때만 같은 run 번호로 1회 더",
    "status_definitions": {
        "valid": ("제출 도구가 계약 검증을 통과한 결정 하나를 저장했고 worker가 같은 검증기로 다시 통과시켰다. "
                  "제출 뒤 마지막 응답 단계의 오류는 post_submission_error로 따로 기록하고 결정을 버리지 않는다"),
        "invalid": "제출 거절(두 번째 제출 포함, 제출 파일이 있어도 거절 기록이 있으면 무효), 미제출, 반복 한도, 그 밖의 의미 오류",
        "infra_error": "두 번째 시도까지 인프라 오류. 첫 시도만 인프라 오류이고 두 번째 시도가 없으면 미완료로 두고 판정하지 않는다",
    },
    "v2_failure_evidence": [
        "[사실] v2 교정 제출 20회(s2030·s2029 각 5회 × 점검·제출 4호출 중 교정 인자 20개) 중 19개의 hold_reason이 문자열 \"None\"이었다",
        "[사실] 같은 호출의 ATIF 궤적(파싱된 LLM 도구 호출 인자)과 도구 기록(도구가 받은 인자)이 호출마다 같았다(null 1회 포함)",
        "[사실] langchain은 도구 인자 문자열을 json.loads로 파싱하므로 null은 None이 되고 \"None\"으로 바뀌지 않는다",
        "[사실] 같은 계약의 단일 호출(응답 본문 JSON)은 hold_reason에 null을 썼다",
        "[추론] \"None\"은 제공자 쪽 도구 호출 인자 생성·파싱 단계에서 생긴다. 원시 응답을 남기지 않아 모델과 제공자 파서를 가를 수 없다",
    ],
    "changes_from_v2": [
        "교정 제출·점검 도구 e4b3_*: decision(propose|hold) 판별 칸, 쓰지 않는 칸은 빈 문자열, null 칸 없음",
        "어댑터가 공개 계약 형태(hold_reason null 등)로 바꾼다. 판별 칸과 맞지 않는 칸은 고치지 않고 사유를 돌려준다",
        "교정 도구 규칙(시스템 프롬프트 뒤 부분)을 새 스키마에 맞췄다. 진단 도구·설정·프롬프트는 v2와 같다",
        "채택 입력을 E4b v1·v2에 쓰지 않은 h02_s2027 packet으로 바꿨다",
    ],
    "precheck": {"limit": PRECHECK_LIMIT, "tools": {"diagnosis": "e4b2_check_decision", "correction": "e4b3_check_proposal"},
                 "note": "단일 호출 경로에는 없는 기능이다"},
    "comparison_conditions": {
        "same": ["모델·temperature 0·reasoning_effort none(extra_body)", "같은 판단 계약 프롬프트와 검증기"],
        "different": ["NAT는 사실·카탈로그·등록 표·체크리스트를 도구로 조회하는 여러 턴, 단일 호출은 packet 전체를 한 번에 받는다",
                      f"NAT는 제출 전 계약 점검 도구로 실행당 최대 {PRECHECK_LIMIT}회 사유를 받아 고칠 수 있다",
                      "NAT 교정은 null 없는 도구 스키마를 어댑터가 계약 형태로 바꾼다. 단일 호출은 계약 JSON을 그대로 쓴다"],
    },
    "metrics": ["세트·종류별 유효·무효·인프라 실행 수", "단일 호출 기록이 있는 세트의 단일 호출 결정과의 일치",
                "점검 호출 수와 점검을 쓴 실행 수", "도구 호출 순서", "LLM 호출 수", "입력·출력 토큰", "wall 시간"],
    "adoption_rule": {**ADOPTION, "set": ADOPTION_SET,
                      "rule": ("다음 seed 시연에 NAT 결정을 쓰려면 h02_s2027 세트에서 두 종류 모두 인프라 오류를 뺀 실행이 4 이상이고 "
                               "그중 유효 결정이 4 이상이어야 한다. 수치 조건은 v1·v2와 같다. 일치율은 채택 조건에 쓰지 않는다. "
                               "요약은 모든 세트 실행이 끝난 뒤에만 쓴다")},
    "wording": {
        "adopt": ("null 없는 교정 제출 스키마로 고친 NAT 도구 호출 워크플로가 E4b v1·v2에 쓰지 않은 입력에서 사전등록 조건만큼 "
                  "유효한 결정을 냈다. 다음 seed 시연의 결정 단계에 쓸 수 있다. 정확도·우위의 근거가 아니다."),
        "do_not_adopt": ("고친 NAT 도구 호출 워크플로의 유효 결정이 새 입력에서 사전등록 조건에 못 미쳤다. 다음 seed 시연은 단일 "
                         "호출 결정을 유지한다."),
        "development": ("s2030·s2029·h05_s2027 세트는 수정의 개발 확인이다. 채택 판정에 쓰지 않고 채택 세트·v1·v2 결과와 "
                        "분모를 합치지 않는다."),
        "common": ("고정 입력을 반복 호출한 개발 측정이다. 채택 입력 h02_s2027은 이미 본 seed의 공개 기록에서 만든 packet이라 "
                   "설계자가 본 사례다. NAT는 근거를 도구로 나눠 조회하고 점검으로 고칠 기회가 있으므로 단일 호출과 조건이 다르다. "
                   "일반화·정확도·절감을 주장하지 않는다."),
    },
    "forbidden_claims": ["NAT가 더 정확하다", "모든 결정이 NAT로 이뤄졌다", "토큰·시간 절감", "일반화",
                         "v1·v2·v3를 합친 유효율", "점검 도구가 없는 단일 호출과 같은 조건의 비교"],
}
DYNAMIC = {"source_files", "input_sources", "inputs"}


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


def code_rel(path: Path) -> str:
    """코드 파일의 저장소 기준 상대 경로(입력·결과 위치인 ROOT와 무관하다)."""
    return Path(path).resolve().relative_to(CODE_ROOT.resolve()).as_posix()


# --- 입력 ---
def public_packet(set_name: str, kind: str) -> dict:
    """공개 기록에서 결정적으로 만든 packet. 진단은 종단 시연 packet 단계, 교정은 correct.py evidence와 같은 경로다."""
    case = INPUT_SETS[set_name]["case"]
    if kind == "diagnosis":
        import run_active_diagnosis as RA
        sources = {"telemetry": {"path": f"bench/runs_p0c/{case}.telemetry.json"},
                   "reference_telemetry": {"path": f"bench/runs_p0c/baseline_p0c_{case.split('_')[1]}.telemetry.json"}}
        return RA.input_packet({"sources": sources}, DIAGNOSIS_BUILD)
    import correct as CR
    report_path = ROOT / P0C_RUNS / f"{case}.json"
    return CR.evidence_packet(case, read(report_path), ROOT / "evals/loops", sha_lf(report_path))


def input_path(set_name: str, kind: str) -> Path:
    return PRIVATE / "inputs" / set_name / f"{kind}.packet.json"


def _latest_diagnosis_call(model_dir: Path) -> Path | None:
    calls = sorted((p for p in model_dir.glob("diagnosis.call*.json") if p.name[14:-5].isdigit()),
                   key=lambda p: int(p.name[14:-5]))
    return calls[-1] if calls else None


def bind_set(set_name: str) -> dict:
    """세트의 packet(과 단일 호출 기록이 있으면 그것)을 찾아 SHA를 대조한다. 공개 세트는 다시 만들어 대조한다."""
    spec = INPUT_SETS[set_name]
    out = {}
    for kind in spec["kinds"]:
        if spec["source"] == "public":
            path = input_path(set_name, kind)
            rebuilt = json.loads(json.dumps(public_packet(set_name, kind)))  # 저장 형식과 같게(튜플 → 리스트)
            if not path.is_file() or read(path) != rebuilt:
                raise ValueError(f"{set_name} {kind}: 공개 기록에서 다시 만든 packet과 저장 packet이 다르다(prepare 먼저)")
            out[kind] = {"packet": rel(path), "packet_sha256": sha_raw(path), "single_call": None, "single_call_sha256": None}
            continue
        import correction_model_call as CM
        model_dir = E4_PRIVATE / spec["tag"] / "model"
        packet = model_dir / f"{kind}.packet.json"
        call = _latest_diagnosis_call(model_dir) if kind == "diagnosis" else model_dir / "correction.call1.json"
        if not packet.is_file() or call is None or not call.is_file():
            raise ValueError(f"{set_name} {kind}: packet 또는 단일 호출 기록이 없다")
        record = read(call)
        bound = (record.get("packet_sha256") == sha_raw(packet) if kind == "diagnosis"
                 else record.get("packet_digest") == CM.packet_digest(read(packet)))
        if not bound:
            raise ValueError(f"{set_name} {kind}: 단일 호출 기록이 packet과 다르다")
        out[kind] = {"packet": rel(packet), "packet_sha256": sha_raw(packet),
                     "single_call": rel(call), "single_call_sha256": sha_raw(call)}
    return out


# --- 프로토콜 ---
def dynamic_fields() -> dict:
    return {"source_files": {name: sha_lf(CODE_ROOT / name) for name in sorted(CODE)},
            "input_sources": {name: sha_lf(ROOT / name) for name in sorted(INPUT_SOURCES)},
            "inputs": {name: bind_set(name) for name in INPUT_SETS}}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed comparison contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources or inputs changed")


def prepare() -> dict:
    for set_name, spec in INPUT_SETS.items():
        if spec["source"] == "public":
            for kind in spec["kinds"]:
                write_new(input_path(set_name, kind), public_packet(set_name, kind))
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_runs", "path": rel(PROTOCOL)}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    try:  # 추적되지 않은 결합 파일도 CLI가 거부로 보고하게 한다
        return protocol, P.verify_committed(ROOT, CODE | INPUT_SOURCES | {rel(PROTOCOL)})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


def run_path(set_name: str, kind: str, run: int, attempt: int) -> Path:
    if set_name not in INPUT_SETS or kind not in INPUT_SETS[set_name]["kinds"]:
        raise ValueError(f"등록되지 않은 세트·종류 {set_name} {kind}")
    return PRIVATE / set_name / kind / f"run{run}_a{attempt}.json"


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def plan() -> dict:
    protocol, commit = verify_protocol()
    commands = []
    for set_name, spec in INPUT_SETS.items():
        for kind in spec["kinds"]:
            packet = ROOT / protocol["inputs"][set_name][kind]["packet"]
            for run in range(1, RUNS_PER_KIND + 1):
                commands.append(
                    "wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"$HOME/.venvs/rl-triage/bin/python\" "
                    f"{wsl(ROOT / 'evals/nat_decision_call_v3.py')} --kind {kind} --packet {wsl(packet)} "
                    f"--output {wsl(run_path(set_name, kind, run, 1))}'")
    return {"source_commit_sha": commit, "protocol_sha256_lf": sha_lf(PROTOCOL), "commands": commands,
            "retry_rule": "인프라 오류 기록(status=infra_error)이면 같은 명령의 출력만 run<i>_a2.json으로 바꿔 1회 더 실행"}


# --- 기록 재검증과 지표 ---
def single_call_decision(kind: str, inputs: dict) -> dict | None:
    """단일 호출 기록을 같은 계약으로 다시 검증한 결정. 기록이 없거나 무효면 None."""
    if inputs[kind]["single_call"] is None:
        return None
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
    """NAT v3 기록의 결합값과, 사건·도구 기록·궤적·거절 파일에서 다시 계산한 값(점검 호출 수 포함)을 대조한다."""
    import nat_decision_call as W1
    import nat_decision_call_v3 as W3
    source = protocol["source_files"]
    expected = {"schema": W3.RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0, "nat_version": "1.9.0",
                "semantic_retries": 0, "precheck_limit": PRECHECK_LIMIT, "packet_sha256": inputs[kind]["packet_sha256"],
                "config_sha256_lf": source[code_rel(W3.CONFIGS[kind])],
                "tools_modules_sha256_lf": {name: source[code_rel(path)] for name, path in W3.TOOLS_MODULES.items()},
                "system_prompt_sha256": W1.text_sha(W3.system_prompt(kind)),
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
    if record.get("precheck_calls") != W3.precheck_calls(kind, trace):
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
    labels = [r["decision"]["action" if kind == "diagnosis" else "decision"] for r in valid]  # continue·hold / propose·HOLD
    single_costs = None
    if inputs[kind]["single_call"] is not None:
        record = read(ROOT / inputs[kind]["single_call"])
        single_costs = {k: record.get(k) for k in ("input_tokens", "output_tokens", "execution_wall_s")}
    return {"counts": counts, "non_infra_runs": counts["valid"] + counts["invalid"],
            "single_call_available": inputs[kind]["single_call"] is not None, "single_call_valid": single is not None,
            "agreement_with_single_call": agree, "agreement_denominator": len(valid) if single is not None else 0,
            "decisions": {label: labels.count(label) for label in sorted(set(labels))},
            "runs_using_precheck": sum(1 for r in done if (r.get("precheck_calls") or 0) > 0),
            "tool_sequences": {seq: sequences.count(seq) for seq in sorted(set(sequences))},
            **{key: _stats([r.get(key) for r in runs])
               for key in ("precheck_calls", "llm_calls", "input_tokens", "output_tokens", "wall_s")},
            "single_call_costs": single_costs, "runs": runs, "complete": counts["missing"] == 0}


def meets_rule(summary: dict) -> bool:
    return all(s["non_infra_runs"] >= ADOPTION["min_non_infra_runs"] and s["counts"]["valid"] >= ADOPTION["min_valid_runs"]
               for s in summary.values())


def summarize() -> dict:
    protocol, commit = verify_protocol()
    sets = {name: {kind: kind_summary(name, kind, protocol["inputs"][name], protocol) for kind in spec["kinds"]}
            for name, spec in INPUT_SETS.items()}
    complete = all(s["complete"] for kinds in sets.values() for s in kinds.values())
    decision = None if not complete else ("adopt" if meets_rule(sets[ADOPTION_SET]) else "do_not_adopt")
    result = {"schema": "e4b_nat_compare_result_v3", "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit,
              "sets": {name: {"role": INPUT_SETS[name]["role"], "kinds": kinds,
                              "meets_numeric_rule": meets_rule(kinds) if complete else None}
                       for name, kinds in sets.items()},
              "adoption": {"decision": decision, "set": ADOPTION_SET, "rule": FIXED["adoption_rule"]["rule"],
                           "reason": None if complete else "모든 세트 실행이 끝나지 않았다"},
              "wording": None if decision is None else FIXED["wording"][decision],
              "development_wording": FIXED["wording"]["development"], "common_wording": FIXED["wording"]["common"]}
    if decision is not None:
        write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4b v3 NAT 결정 재측정(null 없는 교정 제출)")
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
