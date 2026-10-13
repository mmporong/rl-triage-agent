"""E4b v4: 진단 실수 표기 고정·표기 일치 치환과 packet 등록 표 교정 검증의 NAT 경로를 아직 쓰지 않은 시연 입력으로 재는
사전등록·입력 결합·요약기.

GPU와 모델을 쓰지 않는다. NAT 실행은 WSL에서 evals/nat_decision_call_v4.py로 하고, 이 파일은 입력 결합·실행 명령·
기록 재검증·지표만 맡는다. v1~v3 비교기는 사전등록 결과의 소스 결합 때문에 바꾸지 않았다. v3와 다른 점:
- 채택 입력은 이 프로토콜을 커밋한 뒤 시작한 시연 v3 F2 첫 사례(seed 2031)의 진단·교정 packet이다. 사전등록 때는 packet이
  없으므로 고르는 규칙만 고정하고, packet이 생긴 뒤 bind가 SHA를 한 번만 기록한다(bench/private/e4b_v4/adoption_inputs.json).
  첫 사례의 실험 커밋에 이 프로토콜이 같은 내용으로 있어야 묶는다. 시연이 packet 단계 전에 끝나 생기지 않은 종류가
  있으면 판정은 no_input이다.
- 진단·교정 모두 v4 도구다. 진단은 실수 사실을 유효숫자 6자리 문자열로 보여 주고 보여 준 표기와 같은 claim만 저장 값으로
  바꿔 공개 검증기로 검증한다. 교정은 v3 null 없는 스키마를 쓰고 packet에 실린 등록 표로 검증한다.
- 개발 분모: v3 채택 입력 h02_s2027의 진단 packet(v3 진단 무효 3/5의 입력). 채택 판정에 쓰지 않는다.

  python evals/e4b_compare_v4.py prepare|verify|bind|plan|summarize
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
PROTOCOL = ROOT / "bench/protocols/e4b_nat_compare_v4.json"
PRIVATE = ROOT / "bench/private/e4b_v4"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "e4b_nat_compare_v4"
DEMO_PRIVATE = ROOT / "bench/private/e2e_demo_v3"
KINDS = ("diagnosis", "correction")
INPUT_SETS = {
    "f2_s2031": {"role": "adoption", "kinds": list(KINDS), "source": "demo_v3", "variant": "f2", "seed": 2031,
                 "case": "h06_s2031", "mechanism": "actuator", "decision_path": "single",
                 "demo_protocol": "bench/protocols/e2e_demo_v3_f2.json",
                 "selection": ("bench/private/e2e_demo_v3/<tag>/steps/00_begin.json의 variant가 f2인 시연 중 begin 시각(at)이 "
                               "가장 이른 것 하나. 그 시연의 실험 커밋에 이 프로토콜이 같은 내용으로 있어야 하고(사전등록 뒤 시작), "
                               "결정 경로는 single이어야 한다"),
                 "packets": {"diagnosis": "model/diagnosis.packet.json(packet 단계 기록의 SHA와 같아야 한다)",
                             "correction": ("model/correction.packet.json(사례 h06_s2031·메커니즘 actuator·등록 표가 시연 프로토콜의 "
                                            "registry와 같아야 한다)")},
                 "ready": ("종류마다 packet과 그 단일 호출 기록(진단: loop_start 기록의 호출, 교정: correction.call1.json)이 있거나, "
                           "시연 종료 기록(terminal.json)이 있어야 묶는다. 종료 기록이 있고 packet이 없으면 그 종류는 not_reached"),
                 "note": "E4b v1~v3에 쓰지 않은 입력. 사전등록 뒤 생기는 새 seed packet. 채택 판정은 이 세트로만 한다"},
    "h02_s2027": {"role": "development", "kinds": ["diagnosis"], "source": "public", "case": "h02_s2027",
                  "note": "v3 채택 입력(진단 유효 2/5, 무효 3회 모두 실수 claim 불일치). 수정의 개발 확인"},
}
ADOPTION_SET = "f2_s2031"
RUNS_PER_KIND = 5
MAX_ATTEMPTS_PER_RUN = 2  # 인프라 오류일 때만 같은 run 번호로 한 번 더
PRECHECK_LIMIT = 2
DISPLAY_DIGITS = 6
MODEL = "nvidia/nemotron-3-super-120b-a12b"
DIAGNOSIS_BUILD = {"max_probes": 4, "probe_execution_budget_s": 180}  # v3 공개 packet과 같은 값
CODE = {
    "src/rl_triage/nat_demo_tools_v4.py", "evals/nat_decision_call_v4.py", "evals/e4b_compare_v4.py",
    "configs/nat_demo_v4_diagnosis.yml", "configs/nat_demo_v4_correction.yml",
    "src/rl_triage/nat_demo_tools_v3.py", "evals/nat_decision_call_v3.py",
    "src/rl_triage/nat_demo_tools_v2.py", "evals/nat_decision_call_v2.py",
    "src/rl_triage/nat_demo_tools.py", "evals/nat_decision_call.py",
    "src/rl_triage/active_diagnosis.py", "src/rl_triage/probe_loop.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_registry.py", "evals/run_active_diagnosis.py",
    "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
INPUT_SOURCES = {"bench/runs_p0c/h02_s2027.telemetry.json", "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}
ADOPTION = {"min_non_infra_runs": 4, "min_valid_runs": 4}
FIXED = {
    "schema": "e4b_nat_compare_protocol_v4",
    "question": ("실수 사실 표기를 고정한 진단 도구와 packet 등록 표로 검증하는 교정 도구의 NAT 워크플로가 사전등록 뒤 생긴 "
                 "새 seed 시연 입력에서 계약상 유효한 결정을 얼마나 내는가"),
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
    "v3_failure_evidence": [
        "[사실] v3 채택 입력 h02_s2027의 진단 무효 3회(run1·2·4)는 모두 제출 거절 사유가 claim "
        "'telemetry:reference:Policy/mean_noise_std'의 값 불일치였다",
        "[사실] 세 실행 모두 점검 호출 3회(한도 거절 1회 포함)와 제출에서 그 claim 값이 0.3697160104910533이었고 packet 사실 값은 "
        "0.36971601049105324였다(다른 double)",
        "[사실] 유효 2회(run3·5)는 17자리 실수 claim 4개를 사실 값과 같은 double로 인용했다",
        "[사실] 공개 검증기 active_diagnosis.validate_decision은 claim 값을 사실 값과 == 로 비교한다(_same_value)",
        "[추론] 17자리 실수 표기를 옮기는 과정에서 마지막 자리가 바뀌었다. 원시 응답이 없어 모델과 제공자 파서를 가르지 못한다",
    ],
    "claim_mapping": {
        "choice": "display_exact_mapping",
        "display": (f"e4b4_get_facts는 유한 실수 사실을 유효숫자 {DISPLAY_DIGITS}자리 문자열(format '.{DISPLAY_DIGITS}g')로 "
                    "보여 준다. 정수·문자열·불리언은 그대로 보여 준다"),
        "mapping": ("claim id의 저장 사실이 유한 실수이고 claim 값이 보여 준 문자열과 같거나 그 문자열을 float로 읽은 수와 같을 때만 "
                    "저장 값으로 바꾼다. 나머지 claim은 그대로 두고 공개 검증기(수정 없음)가 == 로 판정한다"),
        "rationale": [
            ("공개 검증기의 의미(모델이 받은 근거 값을 그대로 인용한다)가 유지된다. 사실마다 받아들이는 값은 모델에게 보여 준 표기 "
             "하나(문자열 또는 같은 수)뿐이고, 보여 주지 않은 근사값은 통과하지 못한다"),
            ("허용 오차는 보여 주지 않은 근사값도 통과시켜 '그대로 인용'을 '가까운 값'으로 바꾼다. h02_s2027 packet의 0이 아닌 사실 "
             "값은 4.3e-4~6.8e4라 절대·상대 오차 방식과 크기를 새로 정해야 하고, 그 값은 v3 실패를 본 뒤 정하는 값이 된다"),
            "[추론] 문자열 값은 제공자 쪽 도구 인자 생성·파싱의 실수 재표기 영향을 받지 않는다",
        ],
        "rejected_alternative": "허용 오차 사전등록(어댑터가 오차 안의 claim을 사실 값으로 바꾼다)",
        "cost": "모델이 보는 실수 정밀도가 유효숫자 6자리로 준다. 그 영향(순위·probe 선택)은 이 측정 범위 밖이다",
    },
    "changes_from_v3": [
        "진단 사실 읽기 도구 e4b4_get_facts: 유한 실수를 유효숫자 6자리 문자열로 보여 준다",
        ("진단 점검·제출 도구 e4b4_*: 보여 준 표기와 정확히 같은 claim만 저장 값으로 바꾼 뒤 공개 검증기로 검증한다. 제출 파일에는 "
         "바꾼 결정(decision)과 모델 인자(tool_args)를 함께 남긴다"),
        "진단 도구 규칙(시스템 프롬프트 뒤 부분)에 보여 준 표기 그대로 인용하는 규칙을 더했다. 진단 도구 목록은 v2의 조회 도구를 그대로 쓴다",
        ("교정 점검·제출 도구 e4b4_*: v3 null 없는 스키마·어댑터 그대로, 검증 등록 표를 packet에 실린 등록 표로 한다. worker 제출 "
         "재검증도 같은 표로 한다. 기본 등록 표 packet에서는 v3와 같다"),
        "채택 입력을 사전등록 뒤 생기는 시연 v3 F2 첫 사례 packet으로 바꿨다",
    ],
    "precheck": {"limit": PRECHECK_LIMIT, "tools": {"diagnosis": "e4b4_check_decision", "correction": "e4b4_check_proposal"},
                 "note": "단일 호출 경로에는 없는 기능이다"},
    "comparison_conditions": {
        "same": ["모델·temperature 0·reasoning_effort none(extra_body)", "같은 판단 계약 프롬프트와 검증기"],
        "different": ["NAT는 사실·카탈로그·등록 표·체크리스트를 도구로 조회하는 여러 턴, 단일 호출은 packet 전체를 한 번에 받는다",
                      f"NAT는 제출 전 계약 점검 도구로 실행당 최대 {PRECHECK_LIMIT}회 사유를 받아 고칠 수 있다",
                      "NAT 진단은 실수 사실을 6자리 문자열로 받고 표기 일치 claim이 저장 값으로 바뀐다. 단일 호출은 원 실수를 받는다",
                      "NAT 교정은 null 없는 도구 스키마를 어댑터가 계약 형태로 바꾼다. 단일 호출은 계약 JSON을 그대로 쓴다"],
    },
    "metrics": ["세트·종류별 유효·무효·인프라 실행 수", "단일 호출 기록이 있는 세트의 단일 호출 결정과의 일치",
                "점검 호출 수와 점검을 쓴 실행 수", "도구 호출 순서", "LLM 호출 수", "입력·출력 토큰", "wall 시간"],
    "adoption_rule": {**ADOPTION, "set": ADOPTION_SET,
                      "rule": ("다음 seed 시연에 NAT 결정을 쓰려면 f2_s2031 세트에서 두 종류 모두 인프라 오류를 뺀 실행이 4 이상이고 "
                               "그중 유효 결정이 4 이상이어야 한다. 수치 조건은 v1~v3과 같다. 시연이 packet 단계 전에 끝나 한 종류라도 "
                               "입력이 생기지 않으면 no_input이고 채택하지 않는다. 일치율은 채택 조건에 쓰지 않는다. 요약은 모든 세트 "
                               "실행이 끝난 뒤에만 쓴다")},
    "wording": {
        "adopt": ("실수 표기를 고정한 NAT 도구 호출 워크플로가 사전등록 뒤 생긴 새 seed 시연 입력에서 사전등록 조건만큼 유효한 결정을 "
                  "냈다. 다음 seed 시연의 결정 단계에 쓸 수 있다. 정확도·우위의 근거가 아니다."),
        "do_not_adopt": ("고친 NAT 도구 호출 워크플로의 유효 결정이 새 seed 시연 입력에서 사전등록 조건에 못 미쳤다. 다음 seed 시연은 "
                         "단일 호출 결정을 유지한다."),
        "no_input": ("시연 v3 F2 첫 사례가 packet 단계 전에 끝나 채택 입력의 일부가 생기지 않았다. 채택 판정을 하지 못했고 다음 seed "
                     "시연은 단일 호출 결정을 유지한다."),
        "development": ("h02_s2027 세트는 수정의 개발 확인이다. 채택 판정에 쓰지 않고 채택 세트·v1~v3 결과와 분모를 합치지 않는다."),
        "common": ("고정 입력을 반복 호출한 개발 측정이다. 결함과 probe는 같은 설계자가 만들었고, NAT는 근거를 도구로 나눠 조회하고 "
                   "점검으로 고칠 기회가 있으므로 단일 호출과 조건이 다르다. 일반화·정확도·절감을 주장하지 않는다."),
    },
    "forbidden_claims": ["NAT가 더 정확하다", "모든 결정이 NAT로 이뤄졌다", "토큰·시간 절감", "일반화",
                         "v1~v4를 합친 유효율", "점검 도구가 없는 단일 호출과 같은 조건의 비교",
                         "표기 고정이 진단 내용(순위·probe)을 개선했다"],
}
DYNAMIC = {"source_files", "input_sources", "development_inputs"}


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


# --- 개발 입력(공개 기록) ---
def public_packet(set_name: str) -> dict:
    """공개 기록에서 결정적으로 만든 진단 packet(v3 공개 packet과 같은 경로·값)."""
    import run_active_diagnosis as RA
    case = INPUT_SETS[set_name]["case"]
    sources = {"telemetry": {"path": f"bench/runs_p0c/{case}.telemetry.json"},
               "reference_telemetry": {"path": f"bench/runs_p0c/baseline_p0c_{case.split('_')[1]}.telemetry.json"}}
    return RA.input_packet({"sources": sources}, DIAGNOSIS_BUILD)


def input_path(set_name: str, kind: str) -> Path:
    return PRIVATE / "inputs" / set_name / f"{kind}.packet.json"


def bind_public(set_name: str) -> dict:
    path = input_path(set_name, "diagnosis")
    rebuilt = json.loads(json.dumps(public_packet(set_name)))  # 저장 형식과 같게(튜플 → 리스트)
    if not path.is_file() or read(path) != rebuilt:
        raise ValueError(f"{set_name}: 공개 기록에서 다시 만든 packet과 저장 packet이 다르다(prepare 먼저)")
    return {"diagnosis": {"status": "bound", "packet": rel(path), "packet_sha256": sha_raw(path),
                          "single_call": None, "single_call_sha256": None}}


# --- 프로토콜 ---
def dynamic_fields() -> dict:
    return {"source_files": {name: sha_lf(CODE_ROOT / name) for name in sorted(CODE)},
            "input_sources": {name: sha_lf(ROOT / name) for name in sorted(INPUT_SOURCES)},
            "development_inputs": {name: bind_public(name) for name, spec in INPUT_SETS.items() if spec["source"] == "public"}}


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
            write_new(input_path(set_name, "diagnosis"), public_packet(set_name))
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_demo_begin", "path": rel(PROTOCOL)}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    try:  # 추적되지 않은 결합 파일도 CLI가 거부로 보고하게 한다
        return protocol, P.verify_committed(ROOT, CODE | INPUT_SOURCES | {rel(PROTOCOL)})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


# --- 채택 입력(시연 v3 F2 첫 사례) ---
def adoption_inputs_path() -> Path:
    return PRIVATE / "adoption_inputs.json"


def committed_sha_lf(commit: str, relative: str) -> str | None:
    """커밋에 있는 파일의 LF 정규화 SHA. 없으면 None."""
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"], capture_output=True)
    return hashlib.sha256(shown.stdout.replace(b"\r\n", b"\n")).hexdigest() if shown.returncode == 0 else None


def first_demo() -> Path | None:
    """begin 기록의 변형이 등록 변형인 시연 중 begin 시각이 가장 이른 시연 폴더."""
    spec = INPUT_SETS[ADOPTION_SET]
    begins = []
    for path in DEMO_PRIVATE.glob("*/steps/00_begin.json"):
        record = read(path)
        if record.get("variant") == spec["variant"]:
            begins.append((record["at"], path.parent.parent.name, path.parent.parent))
    return min(begins)[2] if begins else None


def _steps(folder: Path) -> dict[str, Path]:
    return {p.name.split("_", 1)[1][:-5]: p for p in (folder / "steps").glob("*.json")}


def _latest_diagnosis_call(model_dir: Path) -> Path | None:
    """단일 호출 기록(diagnosis.call<n>.json)만 찾는다. 채택 세트는 결정 경로 single만 묶는다."""
    calls = sorted((p for p in model_dir.glob("diagnosis.call*.json") if p.name[14:-5].isdigit()),
                   key=lambda p: int(p.name[14:-5]))
    return calls[-1] if calls else None


def _diagnosis_binding(folder: Path, steps: dict, terminal: dict | None) -> dict:
    packet = folder / "model" / "diagnosis.packet.json"
    if not packet.is_file():
        if terminal is None:
            raise ValueError("진단 packet이 아직 없다(시연 진행 중)")
        return {"status": "not_reached", "terminal_outcome": terminal["outcome"]}
    if "packet" not in steps or read(steps["packet"])["packet"]["sha256"] != sha_raw(packet):
        raise ValueError("진단 packet이 시연 packet 단계 기록과 다르다")
    if "loop_start" not in steps and terminal is None:
        raise ValueError("진단 단일 호출이 아직 끝나지 않았다(시연 진행 중)")
    call = _latest_diagnosis_call(folder / "model")
    if call is not None and read(call).get("packet_sha256") != sha_raw(packet):
        raise ValueError("진단 단일 호출 기록이 packet과 다르다")
    if "loop_start" in steps:
        used = read(steps["loop_start"])["call"]
        if call is None or used["record"] != rel(call) or used["record_sha256"] != sha_raw(call):
            raise ValueError("loop_start가 쓴 진단 호출이 마지막 진단 호출 기록과 다르다")
    return {"status": "bound", "packet": rel(packet), "packet_sha256": sha_raw(packet),
            "single_call": None if call is None else rel(call), "single_call_sha256": None if call is None else sha_raw(call)}


def _correction_binding(folder: Path, terminal: dict | None, demo_protocol: dict) -> dict:
    import correction_model_call as CM
    spec = INPUT_SETS[ADOPTION_SET]
    packet_path = folder / "model" / "correction.packet.json"
    if not packet_path.is_file():
        if terminal is None:
            raise ValueError("교정 packet이 아직 없다(시연 진행 중)")
        return {"status": "not_reached", "terminal_outcome": terminal["outcome"]}
    packet = read(packet_path)
    table = {m: {"correction_id": e["correction_id"], "recovery_modes": list(e["recovery_modes"])}
             for m, e in demo_protocol["registry"].items()}
    if (packet.get("schema") != CM.PACKET_SCHEMA or packet.get("case") != spec["case"]
            or packet.get("mechanism") != spec["mechanism"] or packet.get("registry") != table):
        raise ValueError("교정 packet의 형식·사례·메커니즘·등록 표가 등록 값과 다르다")
    call = folder / "model" / "correction.call1.json"
    if not call.is_file():
        if terminal is None:
            raise ValueError("교정 단일 호출이 아직 없다(시연 진행 중)")
        call = None
    elif read(call).get("packet_digest") != CM.packet_digest(packet):
        raise ValueError("교정 단일 호출 기록이 packet과 다르다")
    return {"status": "bound", "packet": rel(packet_path), "packet_sha256": sha_raw(packet_path),
            "single_call": None if call is None else rel(call), "single_call_sha256": None if call is None else sha_raw(call)}


def adoption_binding() -> dict:
    """첫 F2 시연의 packet을 찾아 결합값을 만든다. 아직 묶을 수 없으면 ValueError(사유)."""
    spec = INPUT_SETS[ADOPTION_SET]
    folder = first_demo()
    if folder is None:
        raise ValueError("시연 v3 F2 사례가 아직 시작되지 않았다")
    begin_path = folder / "steps" / "00_begin.json"
    begin = read(begin_path)
    if begin.get("decision_path") != spec["decision_path"]:
        raise ValueError("첫 F2 시연의 결정 경로가 등록 값(single)과 다르다")
    commit = begin.get("source_commit_sha")
    if not isinstance(commit, str) or committed_sha_lf(commit, rel(PROTOCOL)) != sha_lf(PROTOCOL):
        raise ValueError("첫 F2 시연의 실험 커밋에 이 프로토콜이 같은 내용으로 없다(사전등록 뒤 시작한 시연이 아니다)")
    demo_protocol_path = ROOT / spec["demo_protocol"]
    if sha_lf(demo_protocol_path) != begin.get("protocol_sha256_lf"):
        raise ValueError("시연 프로토콜이 begin 기록과 다르다")
    demo_protocol = read(demo_protocol_path)
    if demo_protocol.get("variant") != spec["variant"] or demo_protocol.get("seed") != spec["seed"]:
        raise ValueError("시연 프로토콜의 변형·seed가 등록 값과 다르다")
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").is_file() else None
    steps = _steps(folder)
    return {"schema": "e4b_v4_adoption_inputs", "set": ADOPTION_SET, "tag": folder.name, "begin": rel(begin_path),
            "begin_sha256": sha_raw(begin_path), "experiment_commit": commit,
            "demo_protocol_sha256_lf": begin["protocol_sha256_lf"],
            "kinds": {"diagnosis": _diagnosis_binding(folder, steps, terminal),
                      "correction": _correction_binding(folder, terminal, demo_protocol)}}


def bind() -> dict:
    verify_protocol()
    binding = adoption_binding()
    write_new(adoption_inputs_path(), {**binding, "bound_at": datetime.now(timezone.utc).isoformat()})
    return {"status": "bound", "path": rel(adoption_inputs_path()), "tag": binding["tag"],
            "kinds": {kind: entry["status"] for kind, entry in binding["kinds"].items()}}


def adoption_inputs() -> dict | None:
    """묶인 채택 입력. 아직 bind 전이면 None. 묶인 뒤 원천이 바뀌었으면 ValueError."""
    path = adoption_inputs_path()
    if not path.is_file():
        return None
    stored = read(path)
    stored.pop("bound_at", None)
    if stored != adoption_binding():
        raise ValueError("채택 입력 결합이 시연 산출물과 다르다")
    return stored


def set_inputs(protocol: dict, adoption: dict | None) -> dict:
    """세트별 종류 → 결합값. 채택 세트는 bind 뒤에만 있다."""
    out = dict(protocol["development_inputs"])
    if adoption is not None:
        out[ADOPTION_SET] = adoption["kinds"]
    return out


def run_path(set_name: str, kind: str, run: int, attempt: int) -> Path:
    if set_name not in INPUT_SETS or kind not in INPUT_SETS[set_name]["kinds"]:
        raise ValueError(f"등록되지 않은 세트·종류 {set_name} {kind}")
    return PRIVATE / set_name / kind / f"run{run}_a{attempt}.json"


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def plan() -> dict:
    protocol, commit = verify_protocol()
    adoption = adoption_inputs()
    commands, not_reached = [], {}
    for set_name, kinds in set_inputs(protocol, adoption).items():
        for kind, entry in kinds.items():
            if entry["status"] != "bound":
                not_reached[f"{set_name}/{kind}"] = entry["terminal_outcome"]
                continue
            for run in range(1, RUNS_PER_KIND + 1):
                commands.append(
                    "wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"$HOME/.venvs/rl-triage/bin/python\" "
                    f"{wsl(ROOT / 'evals/nat_decision_call_v4.py')} --kind {kind} --packet {wsl(ROOT / entry['packet'])} "
                    f"--output {wsl(run_path(set_name, kind, run, 1))}'")
    return {"source_commit_sha": commit, "protocol_sha256_lf": sha_lf(PROTOCOL), "commands": commands,
            "adoption_inputs": "unbound: 시연 v3 F2 첫 사례의 packet이 생긴 뒤 bind" if adoption is None else adoption["tag"],
            "not_reached": not_reached,
            "retry_rule": "인프라 오류 기록(status=infra_error)이면 같은 명령의 출력만 run<i>_a2.json으로 바꿔 1회 더 실행"}


# --- 기록 재검증과 지표 ---
def single_call_decision(kind: str, entry: dict) -> dict | None:
    """단일 호출 기록을 같은 계약으로 다시 검증한 결정. 기록이 없거나 무효면 None. 교정은 packet 등록 표로 검증한다."""
    if entry["single_call"] is None:
        return None
    record, packet = read(ROOT / entry["single_call"]), read(ROOT / entry["packet"])
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
    verdict = CM.validate_model_proposal(proposal, packet, packet["registry"])
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


def verify_record(path: Path, record: dict, kind: str, entry: dict, protocol: dict) -> None:
    """NAT v4 기록의 결합값과, 사건·도구 기록·궤적·거절 파일에서 다시 계산한 값(점검 호출 수 포함)을 대조한다."""
    import nat_decision_call as W1
    import nat_decision_call_v4 as W4
    source = protocol["source_files"]
    expected = {"schema": W4.RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0, "nat_version": "1.9.0",
                "semantic_retries": 0, "precheck_limit": PRECHECK_LIMIT, "packet_sha256": entry["packet_sha256"],
                "config_sha256_lf": source[code_rel(W4.CONFIGS[kind])],
                "tools_modules_sha256_lf": {name: source[code_rel(path)] for name, path in W4.TOOLS_MODULES.items()},
                "system_prompt_sha256": W1.text_sha(W4.system_prompt(kind)),
                "user_message_sha256": W1.text_sha(W1.user_message(kind, read(ROOT / entry["packet"])))}
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
    if record.get("precheck_calls") != W4.precheck_calls(kind, trace):
        raise ValueError(f"{path.name}: 점검 호출 수가 도구 기록과 다르다")
    if record.get("atif_sha256") is not None and record["atif_sha256"] != sha_raw(path.with_suffix(".atif.json")):
        raise ValueError(f"{path.name}: ATIF 궤적이 바뀌었다")
    rejected = path.with_suffix(".submission.rejected.json")
    if rejected.exists() != ("rejection_sha256" in record) or (rejected.exists() and record["rejection_sha256"] != sha_raw(rejected)):
        raise ValueError(f"{path.name}: 거절 기록 결합이 맞지 않는다")
    if record["status"] == "valid" and rejected.exists():
        raise ValueError(f"{path.name}: 거절 기록이 있는데 유효로 기록됐다")


def load_run(set_name: str, kind: str, run: int, entry: dict, protocol: dict) -> dict:
    """run 번호의 최종 시도. 첫 시도가 인프라 오류일 때만 두 번째 시도를 본다. 재시도가 남았으면 missing."""
    import nat_decision_call_v4 as W4
    attempts = [run_path(set_name, kind, run, a) for a in range(1, MAX_ATTEMPTS_PER_RUN + 1)]
    records = [(i + 1, p, read(p)) for i, p in enumerate(attempts) if p.is_file()]
    if [a for a, _, _ in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"{set_name} {kind} run{run}: 시도 번호가 이어지지 않는다")
    for attempt, path, record in records:
        verify_record(path, record, kind, entry, protocol)
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
        if W4.validate_submission(kind, read(ROOT / entry["packet"]), read(submission)) != record["decision"]:
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


def kind_summary(set_name: str, kind: str, entry: dict, protocol: dict) -> dict:
    if entry["status"] != "bound":
        return {"not_reached": True, "terminal_outcome": entry["terminal_outcome"], "complete": True}
    runs = [load_run(set_name, kind, run, entry, protocol) for run in range(1, RUNS_PER_KIND + 1)]
    single = single_call_decision(kind, entry)
    valid = [r for r in runs if r["status"] == "valid"]
    counts = {status: sum(1 for r in runs if r["status"] == status) for status in ("valid", "invalid", "infra_error", "missing")}
    matches = [agreement(kind, r["decision"], single) for r in valid]
    agree = None if single is None or not matches else {key: sum(1 for m in matches if m[key]) for key in matches[0]}
    done = [r for r in runs if r["status"] != "missing"]
    sequences = [" > ".join(r["tool_sequence"] or []) for r in done]
    labels = [r["decision"]["action" if kind == "diagnosis" else "decision"] for r in valid]  # continue·hold / propose·HOLD
    single_costs = None
    if entry["single_call"] is not None:
        record = read(ROOT / entry["single_call"])
        single_costs = {k: record.get(k) for k in ("input_tokens", "output_tokens", "execution_wall_s")}
    return {"not_reached": False, "counts": counts, "non_infra_runs": counts["valid"] + counts["invalid"],
            "single_call_available": entry["single_call"] is not None, "single_call_valid": single is not None,
            "agreement_with_single_call": agree, "agreement_denominator": len(valid) if single is not None else 0,
            "decisions": {label: labels.count(label) for label in sorted(set(labels))},
            "runs_using_precheck": sum(1 for r in done if (r.get("precheck_calls") or 0) > 0),
            "tool_sequences": {seq: sequences.count(seq) for seq in sorted(set(sequences))},
            **{key: _stats([r.get(key) for r in runs])
               for key in ("precheck_calls", "llm_calls", "input_tokens", "output_tokens", "wall_s")},
            "single_call_costs": single_costs, "runs": runs, "complete": counts["missing"] == 0}


def meets_rule(summary: dict) -> bool:
    return all(not s["not_reached"] and s["non_infra_runs"] >= ADOPTION["min_non_infra_runs"]
               and s["counts"]["valid"] >= ADOPTION["min_valid_runs"] for s in summary.values())


def summarize() -> dict:
    protocol, commit = verify_protocol()
    adoption = adoption_inputs()
    inputs = set_inputs(protocol, adoption)
    sets = {name: {kind: kind_summary(name, kind, entry, protocol) for kind, entry in kinds.items()}
            for name, kinds in inputs.items()}
    complete = adoption is not None and all(s["complete"] for kinds in sets.values() for s in kinds.values())
    decision = None
    if complete:
        if any(s["not_reached"] for s in sets[ADOPTION_SET].values()):
            decision = "no_input"
        else:
            decision = "adopt" if meets_rule(sets[ADOPTION_SET]) else "do_not_adopt"
    reason = None if complete else ("채택 입력이 아직 묶이지 않았다(bind 전)" if adoption is None else "모든 세트 실행이 끝나지 않았다")
    result = {"schema": "e4b_nat_compare_result_v4", "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit,
              "adoption_input": None if adoption is None else {
                  k: adoption[k] for k in ("tag", "begin_sha256", "experiment_commit", "demo_protocol_sha256_lf")} | {
                  "kinds": {kind: {k: v for k, v in entry.items() if k in ("status", "terminal_outcome", "packet_sha256",
                                                                          "single_call_sha256")}
                            for kind, entry in adoption["kinds"].items()}},
              "sets": {name: {"role": INPUT_SETS[name]["role"], "kinds": kinds,
                              "meets_numeric_rule": meets_rule(kinds) if complete else None}
                       for name, kinds in sets.items()},
              "adoption": {"decision": decision, "set": ADOPTION_SET, "rule": FIXED["adoption_rule"]["rule"], "reason": reason},
              "wording": None if decision is None else FIXED["wording"][decision],
              "development_wording": FIXED["wording"]["development"], "common_wording": FIXED["wording"]["common"]}
    if decision is not None:
        write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4b v4 NAT 결정 재측정(실수 표기 고정·packet 등록 표 교정 검증)")
    ap.add_argument("command", choices=("prepare", "verify", "bind", "plan", "summarize"))
    args = ap.parse_args(argv)
    commands = {"prepare": prepare, "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol()[1]},
                "bind": bind, "plan": plan, "summarize": summarize}
    try:
        out = commands[args.command]()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
