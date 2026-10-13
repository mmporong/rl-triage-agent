"""D3 실험 실행기: 변경 8개(무해 7 + 해로운 1)가 섞인 실패 학습에서 해로운 변경을 확정하기까지의 확인 재학습 수를 다섯 방법으로 비교한다.

사례 = (seed, 해로운 변경, 사전등록 위치). 개발 사례 D0(이미 본 seed, 집계 제외)와 본실험 C1~C3(새 seed 2035~2037,
U1·F1·F2). 재학습은 (seed, 적용 부분집합) 캐시를 방법끼리 함께 쓰고, 방법별 비용은 그 방법이 혼자 돌았을 때의 질의로 센다
(evals/d3_methods.py). 결과 문구는 evals/d3_summary.py의 네 가지다. 프로토콜은 bench/protocols/d3_experiment_v1.json.

  python evals/run_d3_experiment.py prepare | verify
  python evals/run_d3_experiment.py begin --case D0                   # 실험 커밋 기록·전제 확인
  python evals/run_d3_experiment.py setup --case D0 --arm none        # GPU: NONE 학습·추출·고정 평가
  python evals/run_d3_experiment.py setup --case D0 --arm failure     # GPU: 변경 8개 전부(실패 실행), healthy면 case_invalid
  python evals/run_d3_experiment.py reference --case D0               # GPU: NONE 정책 probe 6종(고리 판정 기준)
  python evals/run_d3_experiment.py loop --case D0                    # GPU: 실패 정책에 고정 순서 고리(B3·B4 공유 관측)
  python evals/run_d3_experiment.py packet --case D0                  # B4·B5 packet과 WSL 호출 명령
  python evals/run_d3_experiment.py rank --case D0 --method B4 --record <호출 기록>
  python evals/run_d3_experiment.py plan --case D0                    # 방법들이 다음에 필요한 부분집합 → 원장 요청
  python evals/run_d3_experiment.py approve --case D0 --request <id> [<id> ...] --approver <이름>
  python evals/run_d3_experiment.py retrain --case D0 --request <id>  # GPU: 승인 소비 → 학습·평가·판정 → 캐시
  python evals/run_d3_experiment.py recover --case D0 --request <id>  # 결과 없이 끝난 요청을 닫기(재실행 없음)
  python evals/run_d3_experiment.py determinism --case D0 [--rerun]   # D0만: 같은 부분집합 체크포인트 SHA 대조
  python evals/run_d3_experiment.py close --case D0
  python evals/run_d3_experiment.py summarize --tag d3_experiment_summary_<날짜>
  python evals/run_d3_experiment.py status --case D0

학습은 bench/train_with_d3_changes.py, probe는 bench/d3_isaac.py(동결 evals/probes.py), 고정 평가는 evals/fixed_eval.py다.
소유 Job·watchdog·GPU 입장·결합 경로 재확인은 evals/run_d3_harmless_smoke.py·run_upstream_smoke.py와 같다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import d3_agent as AG  # noqa: E402
import d3_changes as D  # noqa: E402
import d3_diff as R  # noqa: E402
import d3_methods as MT  # noqa: E402
import d3_summary as SU  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger, LedgerError  # noqa: E402
import run_d3_harmless_smoke as H  # noqa: E402

sha, read, write_new, load_params, finished, find_run_dir, _owned = (
    H.sha, H.read, H.write_new, H.load_params, H.finished, H.find_run_dir, H._owned)

PROTOCOL = ROOT / "bench/protocols/d3_experiment_v1.json"
PRIVATE = ROOT / "bench/private/d3_experiment"
RESULTS = ROOT / "evals/results/d3_experiment_v1"
SUMMARIES = ROOT / "evals/results"
RUNS_DIR = ROOT / "bench/runs_d3"
P0C = ROOT / "evals/results/p0c_fixed_eval_20261009"
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS, MAX_ITERATIONS, ACTION_SCALE, FINAL_CHECKPOINT = 4096, 300, 0.25, "model_299.pt"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
GPU_PROFILE = "base_com_reset_readback_v2"
WATCHDOG_S = {"training": 2700, "telemetry_extract": 300, "fixed_evaluation": 360, "probes": 600}
EVAL_GPU_WAIT_S, EVAL_GPU_POLL_S = 1800, 60
RETRAIN_BUDGET_GPU_S = 3600
MAX_EXECUTIONS = 2  # 같은 부분집합·probe·호출은 인프라 원인일 때만 한 번 더
APPROVAL = "사용자가 위임한 운영자 승인 또는 사람 승인(승인자 문자열 기록, 신원 인증 아님)"
SUMMARY_TAG_RE = r"d3_experiment_summary_[0-9]{8}[a-z0-9_]{0,10}"
CASES = {
    "D0": {"seed": 2026, "harmful": "U1", "aggregate": False,
           "role": "개발 사례. 실행기·캐시·결정성 확인용이며 집계하지 않는다. 무해성 smoke와 같은 seed라 결정성 대조에 그 체크포인트를 쓸 수 있다"},
    "C1": {"seed": 2035, "harmful": "U1", "aggregate": True, "role": "본실험"},
    "C2": {"seed": 2036, "harmful": "F1", "aggregate": True, "role": "본실험"},
    "C3": {"seed": 2037, "harmful": "F2", "aggregate": True, "role": "본실험"},
}
CODE = ("bench/d3_changes.py", "bench/d3_diff.py", "bench/train_with_d3_changes.py", "bench/d3_isaac.py",
        "evals/d3_methods.py", "evals/d3_summary.py", "evals/d3_agent.py", "evals/d3_model_call.py",
        "evals/run_d3_experiment.py", "evals/run_d3_harmless_smoke.py", "bench/protocols/d3_harmless_smoke_v1.json",
        "bench/upstream_faults.py", "bench/hidden_faults.py", "evals/run_upstream_smoke.py", "evals/probes.py",
        "src/rl_triage/probe_loop.py", "src/rl_triage/correction_ledger.py", "src/rl_triage/correction_registry.py",
        "bench/extract_tb.py", "evals/fixed_eval.py", "bench/protocols/fixed_eval_v1.json",
        "src/rl_triage/recovery.py", "src/rl_triage/grounded_recovery.py", "bench/record_evidence_media.py",
        "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py")


class GPUBusy(Exception):
    pass


def rel(path: Path) -> str:
    """저장소 안 경로는 저장소 기준 상대 경로로 적는다(실제 실행 산출물은 모두 저장소 안이다)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


# --- 사례 정의 ---
def position(case: str) -> int:
    """해로운 변경의 1부터 센 위치. 문자열 seed 난수로 결과를 보기 전에 정한다."""
    return random.Random(f"d3_experiment_v1:{case}").randint(1, len(D.HARMLESS) + 1)


def case_spec(case: str) -> dict:
    if case not in CASES:
        raise ValueError(f"모르는 사례 {case!r}: {sorted(CASES)}")
    meta = CASES[case]
    order = list(D.HARMLESS)
    order.insert(position(case) - 1, meta["harmful"])
    return {**meta, "case": case, "position": position(case), "harmful_index": position(case) - 1, "order": order,
            "labels": dict(zip(AG.labels(len(order)), order)), "categories": [D.CHANGES[c]["category"] for c in order]}


def ids_of(order: list[str], mask: int) -> tuple[str, ...]:
    chosen = {order[i] for i in range(len(order)) if mask >> i & 1}
    return tuple(cid for cid in D.CHANGES if cid in chosen)


def spec_of(ids) -> str:
    return ",".join(ids) or "NONE"


def run_name(case: str, mask: int, attempt: int = 1, rep: bool = False) -> str:
    seed = CASES[case]["seed"]
    return f"d3v1_{case.lower()}_s{seed}_m{mask:02x}" + ("_rep" if rep else "") + ("" if attempt == 1 else f"_a{attempt}")


def predicted_retrains(case: str) -> dict:
    """상호작용이 없고 고리가 해로운 변경의 범주를 확정한다는 전제에서 B1~B3의 확인 재학습 수(설계안 5절)."""
    spec = case_spec(case)
    k, h = len(spec["order"]), spec["harmful_index"]
    oracle = {mask: "unhealthy" if mask >> h & 1 else "healthy" for mask in range(1 << k)}
    out = {}
    for method in ("B1", "B2", "B3"):
        row = MT.run_method(method, k, oracle, categories=spec["categories"],
                            loop_conclusion=D.CHANGES[spec["harmful"]]["category"])
        out[method] = len(row["queries"])
    return out


FIXED = {
    "schema": "d3_experiment_protocol_v1",
    "question": "실패한 학습 하나에 변경 8개가 섞여 있고 그중 하나만 해로울 때, 해로운 변경을 확정하기까지 확인 재학습이 몇 번 "
                "필요한가. 같은 재학습 상한에서 이분 탐색·하나씩 되돌리기·probe 범주 규칙·모델 순위(probe 있음·없음)를 비교한다",
    "scope": "Go2 flat 4096 env × 300회, k = 8, 사전등록 사례 3건(+집계하지 않는 개발 사례 1건). 결함·무해 변경·probe는 같은 "
             "설계자가 만들었고(U1만 외부 원인 결함의 의미 이식), 진단 정확도·일반화·실제 프로젝트에서의 절감을 재지 않는다",
    "changes": {cid: H.catalog_entry(cid) for cid in (*D.HARMLESS, *D.HARMFUL)},
    "cases": {case: {key: value for key, value in case_spec(case).items() if key != "labels"} for case in CASES},
    "position_rule": "random.Random('d3_experiment_v1:<사례>').randint(1, 8). 무해 7개는 카탈로그 순서, 해로운 변경을 그 위치에 넣는다",
    "label_rule": "모델 입력에는 사례 순서(cases.order)의 라벨 change_1..change_8과 diff 원문만 넣는다. 변경 id·범주·역할은 넣지 않는다",
    "confirmation_rule": "변경 c 하나만 되돌린 재학습(전체 − {c})이 healthy일 때만 c를 확정한다(모든 방법 공통). 이 관측 없이 이름만 "
                         "대면 확정이 아니다",
    "cap": MT.R,
    "methods": {
        "B1": "이분 탐색: 후보의 앞쪽 절반을 되돌리고(나머지는 적용) healthy면 그 절반, 아니면 나머지 절반. 하나 남으면 단일 되돌리기로 확정",
        "B2": "하나씩 되돌리기: 사례 순서대로 단일 되돌리기",
        "B3": "probe 범주 규칙: 동결 고리(probe_loop, 모든 범주 가설, FIXED_ORDER, 같은 판정 임계값)로 메커니즘 m을 확정하면 범주가 m인 "
              "변경을 사례 순서로 먼저, 나머지를 사례 순서로 단일 되돌리기. 고리가 확정하지 못하면 B1",
        "B4": "모델 순위: B3와 같은 텔레메트리 사실·probe 관측과 변경 diff 원문으로 모델 단일 호출이 낸 순위대로 단일 되돌리기",
        "B5": "모델 순위(probe 없음): B4에서 probe 관측만 뺀 절제",
    },
    "costs": {"main": "확정까지 그 방법이 혼자 돌았을 때 질의한 서로 다른 확인 재학습 수. unresolved·model_invalid는 R=8로 센다",
              "secondary": ["확인 재학습의 부모 벽시계 합(학습+추출+고정 평가). B3·B4에는 기준 probe와 고리 probe 벽시계를 더한다",
                            "잘못된 확정 수, unresolved 수", "B4·B5 모델 호출 토큰·지연"],
              "shared": "NONE·실패 실행 학습과 고정 평가는 모든 방법이 같이 쓰므로 방법 비용에 넣지 않고 따로 보고한다"},
    "cache": {"key": "(프로토콜, seed, 적용 변경 부분집합)",
              "reuse": "같은 키는 한 번만 학습한다. 여러 방법이 같은 부분집합을 물으면 그 결과를 같이 쓴다. 실제 GPU는 질의 합집합만큼 든다",
              "attempts": "부분집합마다 실행은 최대 2회이고 두 번째는 첫 실행이 인프라 원인으로 답을 못 냈을 때만이다",
              "determinism": "개발 사례 D0의 '해로운 변경만 되돌린' 부분집합(무해 7개 적용) 체크포인트를 무해성 smoke s2026 체크포인트와 "
                             "대조하고, 다르거나 smoke 결과가 없으면 같은 부분집합을 한 번 더 학습해 대조한다. 같지 않으면 캐시를 쓰지 "
                             "않는 새 프로토콜 판이 필요하고 본실험을 시작하지 않는다"},
    "ledger": {"format": "src/rl_triage/correction_ledger.CorrectionLedger의 사건 형식·순서 검사(proposed→approved→consumed→receipt)를 "
                         "물려받은 재학습 요청 원장. 효과 키는 (사례, 적용 부분집합)",
               "approval": APPROVAL, "rule": "모든 방법의 확인 재학습은 같은 원장·같은 승인 방식을 거친다. 승인 대기 시간은 지표에서 뺀다"},
    "verdict": "재학습 정책을 같은 사례 NONE 정책과 같은 고정 평가(fixed_eval_v1)로 비교하고 P0-C summary.json의 p0c 밴드로 "
               "healthy/unhealthy를 정한다. undetermined나 유효 조건 실패는 unknown이며 방법은 그 자리에서 unresolved다",
    "validity_checks": {"training": ["학습 종료 코드 0", "model_299.pt", "[d3_changes] 확인 줄 ok(그 부분집합의 id·교체·결합·env 설정)",
                                     "params 차이가 그 부분집합 override와 run_name뿐(NONE 제외)"],
                        "evaluation": ["fixed_eval_v1 계약·체크포인트 SHA·action scale 검증", "평가 런타임이 같은 사례 NONE과 같음",
                                       "NONE 정책이 걷는다(정지 대비 오차 비 0.5 미만, 낙상 0.05 이하)"],
                        "causes": "infrastructure(실행기·watchdog·GPU·소스 변경) / training_process / injection / comparability / evaluation. "
                                  "infrastructure만 재실행 대상이다"},
    "case_validity": {"case_invalid": "실패 실행(변경 8개 전부)이 NONE 대비 unhealthy가 아니거나, NONE·실패 실행·기준 probe가 유효하게 "
                                      "끝나지 않음",
                      "interaction_observed": "해로운 변경이 없는 부분집합이 unhealthy이거나 있는 부분집합이 healthy. 그 사례는 재학습을 "
                                              "멈추고 방법 비교 집계에서 뺀 채 따로 보고한다"},
    "probes": {"rule": "기준은 같은 사례 NONE 정책의 probe 6종 측정. 고리는 실패 정책(변경 8개 전부를 건 probe 환경)에 FIXED_ORDER로 "
                       "하나씩 재고 probe_loop.classify·update·status를 그대로 쓴다. 다 재면 probe_loop.simulate('fixed')와 같다",
               "injection": "bench/d3_isaac.py: 코드 교체와 env 설정 override를 첫 make 직전에 같은 값으로 넣고 확인 줄을 남긴다. "
                            "기준 probe 6종 실패나 probe 하나의 인프라 실패는 한 번 다시 잰다",
               "approval": "probe는 방법의 선택이 아니라 B3·B4가 함께 쓰는 고정 관측이라 원장 승인 없이 운영자 명령으로 잰다"},
    "model": {"id": MODEL, "call": "evals/d3_model_call.py(WSL, temperature 0, max_tokens 16384, reasoning_effort none, 사례·방법마다 "
                                   "단일 호출)", "contract": "evals/d3_agent.py: 라벨·diff·텔레메트리 사실(B4는 probe 관측 포함), 출력 "
                                                             "ranking·reasons·uncertain, 검증 실패는 의미 재시도 없이 model_invalid",
              "retry": f"인프라 오류(빈 응답·잘림·연결·429/5xx)만 사례·방법마다 최대 {MAX_EXECUTIONS}회 호출",
              "uncertain": "보류 여부(uncertain)는 기록만 하고 순서에 쓰지 않는다",
              "nat": "NAT 도구 호출 경로는 이 판에서 쓰지 않는다(새 도구·계약이 필요하다)"},
    "combination": {"decision": "U1은 bench/d3_changes.py 카탈로그(bench/upstream_faults.SWAPS·swap_code)로 무해 변경과 같은 "
                                "프로세스에서 건다. 학습은 bench/train_with_d3_changes.py, probe는 bench/d3_isaac.py",
                    "not_used": "sub_2의 bench/upstream_isaac.py·upstream_runs.py는 결함 하나 단위 진입점이라 쓰지도 고치지도 않는다"},
    "smoke_dependency": {"rule": "begin은 무해성 smoke(bench/protocols/d3_harmless_smoke_v1.json) 요약이 pass이고 그 변경 목록이 "
                                 "지금 카탈로그 HARMLESS와 같을 때만 진행한다",
                         "fail_branch": "smoke가 fail·invalid면 이 프로토콜로는 시작하지 않는다. 원인 분리(변경별·범주별 smoke)나 예비 후보 "
                                        "교체를 새 smoke 판으로 사전등록해 pass를 얻은 뒤, 바뀐 HARMLESS로 이 실험 프로토콜도 새 판으로 "
                                        "다시 만든다(위치·예측이 바뀐다)"},
    "main_case_preconditions": ["D0 결정성 대조가 same", "seed 2035~2037이 저장소·Isaac Lab 로그·학습 params에서 쓰인 적 없음(begin이 기록)"],
    "predictions": {
        "B1": "사례마다 3 또는 4", "B2": "사례마다 해로운 변경의 위치 p", "B3": "고리가 해로운 변경의 범주를 확정하면 그 범주 안 순번",
        "B4": "3사례 중 2사례 이상에서 1위가 해로운 변경(재학습 1회), 나머지도 B3 이하",
        "B5": "B4와 같거나 1회 많다",
        "computed": {case: predicted_retrains(case) for case in CASES},
        "loop": {"U1": "P_episode·P_noise normal, P_reward abnormal → reward 확정(probe 3개). U1 종단 시연 seed 2040(4e13082)에서도 "
                       "P_reward 한 번으로 reward가 확정됐다",
                 "F1": "U1과 같은 경로로 reward 확정(probe 3개)",
                 "F2": "P_reward normal로 reward 기각 → P_torque abnormal → actuator 확정(probe 4개)",
                 "note": "동결 P_reward는 기대값을 보상 관리자의 그 항 가중치·std로 다시 계산하므로 무해 변경 H2(가중치)는 양쪽에 같이 "
                         "들어가 판정을 바꾸지 않는다[코드 읽기, 실행 확인 아님]"},
        "reason": "[추론] U1·F1·F2의 diff는 속도 신호를 0이나 다른 좌표계로 바꾸거나 토크 곡선 계수를 바꾸고, 무해 변경은 수식을 보존하거나 "
                  "작은 수치만 바꾼다. 다만 무해 변경 H2(가중치)·H4(토크 한계)를 1위로 둘 위험이 있다"},
    "outcomes": SU.WORDING,
    "advantage_conditions": ["① 유효 사례에서 B4 확인 재학습 합 < B1·B2·B3 합의 최소", "② 어느 유효 사례에서도 B4 > B3 아님",
                             "③ B4 잘못된 확정 0, unresolved 0", "④ 유효 사례 3개"],
    "forbidden_claims": list(SU.FORBIDDEN_CLAIMS),
    "execution": {"watchdog_s": WATCHDOG_S, "gpu_admission_profile": GPU_PROFILE, "retrain_budget_gpu_s": RETRAIN_BUDGET_GPU_S,
                  "gpu_admission": "GPU 단계마다 시작 전 GPU가 사용 중이면 아무것도 쓰지 않고 실행하지 않는다(종료 코드 3). 고정 평가 전에는 "
                                   f"최대 {EVAL_GPU_WAIT_S}초 기다린다",
                  "source_binding": "begin이 기록한 HEAD가 그 사례의 실험 커밋이다. 이후 단계는 프로토콜·결합 코드가 커밋돼 있고 실험 커밋과 "
                                    "현재 HEAD 사이에 결합 경로 diff가 없을 때만 진행한다. 실행 중 결합 경로 커밋은 그 실행을 invalid로 만든다"},
    "gpu_estimate": {"unit": "부모 프로세스 벽시계(초)",
                     "per_training_with_eval_s": [558, 1001],
                     "basis": "무해성 smoke gpu_estimate와 같은 근거(시연 교정 재학습 508.8~541.1 s, U1 smoke 590.8 s, 고정 평가 48.4~67.5 s, "
                              "경합 상한 873.6 s)",
                     "per_case": "NONE·실패 실행 2회 + 기준 probe 약 90 s + 고리 probe 1~4회(회당 약 20~32 s) + 확인 재학습 합집합 5~9회",
                     "development_extra": "D0 결정성 재학습 0~1회",
                     "total_hours": "개발 사례 약 1.2~2.6시간, 본실험 3사례 약 3.3~7.5시간(설계안 8절 추정을 같은 근거로 다시 계산)"},
}
DYNAMIC = {"source_files", "isaaclab_sources", "rendered_diffs", "bands", "bands_summary_sha256_lf", "smoke_protocol_sha256_lf"}


# --- 프로토콜 ---
def isaaclab_files() -> list[str]:
    files = {t["file"] for cid in D.CHANGES for t in D.code_targets(cid)}
    files |= {D.CHANGES[cid]["file"] for cid in D.CHANGES if D.CHANGES[cid]["kind"] == "config"}
    return sorted(files | {"scripts/reinforcement_learning/rsl_rl/train.py"})


def dynamic_fields(isaaclab: Path) -> dict:
    summary = P0C / "summary.json"
    return {"source_files": {name: sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": {f: sha(Path(isaaclab) / f, True) for f in isaaclab_files()},
            "rendered_diffs": H.rendered_diffs(isaaclab), "bands": read(summary)["bands"],
            "bands_summary_sha256_lf": sha(summary, True), "smoke_protocol_sha256_lf": sha(H.PROTOCOL, True)}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in json.loads(json.dumps(FIXED)).items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab 소스·diff·기준이 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_begin", "path": rel(PROTOCOL)}


def bound_paths() -> list[str]:
    return [*CODE, rel(PROTOCOL)]


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    try:
        commit = verify_committed(ROOT, set(bound_paths()))
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 프로토콜이 커밋되지 않았다") from error
    return protocol, commit


def bound_sources_same(start: str, now: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", start, now, "--", *bound_paths()],
                          capture_output=True).returncode == 0


# --- 사례 폴더·단계 기록 ---
def folder(case: str) -> Path:
    case_spec(case)
    return PRIVATE / sha(PROTOCOL, True)[:16] / case


def step_path(case: str, name: str) -> Path:
    return folder(case) / "steps" / f"{name}.json"


def step(case: str, name: str) -> dict | None:
    path = step_path(case, name)
    return read(path) if path.is_file() else None


def terminal(case: str) -> dict | None:
    path = folder(case) / "terminal.json"
    return read(path) if path.is_file() else None


def _end(case: str, outcome: str, detail: dict) -> dict:
    record = {"outcome": outcome, "detail": detail, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_new(folder(case) / "terminal.json", record)
    return record


def _check_open(case: str, *needed: str) -> None:
    if terminal(case):
        raise ValueError(f"{case}는 이미 끝났다: {terminal(case)['outcome']}")
    missing = [name for name in needed if step(case, name) is None]
    if missing:
        raise ValueError(f"{case}: 먼저 {missing}")


def _verify_case(case: str, isaaclab: Path) -> tuple[dict, str]:
    """begin의 실험 커밋 기준으로 프로토콜·결합 경로를 다시 확인한다."""
    protocol, head = verify(isaaclab)
    commit = step(case, "begin")["source_commit_sha"]
    if head != commit and not bound_sources_same(commit, head):
        raise ValueError("실험 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")
    return protocol, commit


def _write_step(case: str, name: str, record: dict) -> dict:
    write_new(step_path(case, name), {"step": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                      "protocol_sha256_lf": sha(PROTOCOL, True), **record})
    return read(step_path(case, name))


# --- 전제 ---
def smoke_pass() -> dict:
    """무해성 smoke 요약이 pass이고 변경 목록이 지금 HARMLESS와 같아야 한다."""
    marker = H.PRIVATE / "consumed" / f"{sha(H.PROTOCOL, True)[:16]}_summary.json"
    if not marker.is_file():
        raise ValueError("무해성 smoke 요약이 없다(smoke를 먼저 끝낸다)")
    tag = read(marker)["tag"]
    path = H.RESULTS / tag / "summary.json"
    summary = read(path)
    if summary.get("outcome") != "pass" or summary.get("changes") != list(D.HARMLESS):
        raise ValueError(f"무해성 smoke가 pass가 아니거나 변경 목록이 다르다: {summary.get('outcome')}")
    return {"tag": tag, "summary_sha256_lf": sha(path, True), "outcome": "pass"}


def seed_unused_evidence(isaaclab: Path, seed: int) -> dict:
    log_root = Path(isaaclab) / "logs/rsl_rl/unitree_go2_flat"
    roots = [RUNS_DIR, ROOT / "bench/runs_p0c", ROOT / "bench/runs_upstream", ROOT / "bench/private", ROOT / "evals/loops", log_root]
    hits = sorted(p.name for root in roots if root.exists() for p in root.iterdir() if f"_s{seed}" in p.name)
    for agent in (log_root.glob("*/params/agent.yaml") if log_root.exists() else ()):
        match = re.search(r"^seed: (\d+)\s*$", agent.read_text(encoding="utf-8", errors="replace"), re.M)
        if match and int(match[1]) == seed:
            hits.append(f"{agent.parent.parent.name}: seed {seed}")
    return {"seed": seed, "checked_roots": [str(r.relative_to(ROOT)) if r.is_relative_to(ROOT) else "isaaclab_logs"
                                            for r in roots], "hits": hits}


def begin(case: str, isaaclab: Path) -> dict:
    if step(case, "begin") is not None:
        raise FileExistsError(f"{case} begin은 이미 기록됐다")
    protocol, commit = verify(isaaclab)
    smoke = smoke_pass()
    spec = case_spec(case)
    record = {"source_commit_sha": commit, "case": case, "smoke": smoke, "seed_evidence": None, "determinism": None}
    if spec["aggregate"]:
        determinism_record = step("D0", "determinism")
        if not determinism_record or determinism_record.get("same") is not True:
            raise ValueError("본실험은 D0 결정성 대조가 same일 때만 시작한다")
        evidence = seed_unused_evidence(isaaclab, spec["seed"])
        if evidence["hits"]:
            raise ValueError(f"seed {spec['seed']}가 이미 쓰였다: {evidence['hits'][:5]}")
        record.update(seed_evidence=evidence, determinism={"same": True, "source": determinism_record["source"]})
    return _write_step(case, "begin", record)


# --- 재학습 원장 ---
class RetrainLedger(CorrectionLedger):
    """교정 원장의 사건 형식·순서 검사·승인을 그대로 쓰고, 제안 대상을 (사례, 적용 부분집합)으로 바꾼다."""

    VERDICTS = ("healthy", "unhealthy", "unknown")

    def __init__(self, path: Path, protocol_sha: str, case: str):
        super().__init__(path, registry={"d3_retrain": {"protocol_sha256_lf": protocol_sha, "case": case}})
        self.case = case

    def effect(self, ids) -> str:
        return L.digest({"case": self.case, "ids": list(ids)})[:16]

    def propose(self, mask: int, ids, requested_by: list[str], budget_gpu_s: int = RETRAIN_BUDGET_GPU_S,
                repeat_of: str | None = None) -> dict:
        key = self.effect(ids)
        same = [r for r in self.requests().values() if r.get("effect_key") == key]
        live = [r["request_id"] for r in same if r["state"] in ("pending", "approved", "running")]
        if live:
            raise LedgerError(f"같은 부분집합 재학습이 이미 진행 중이다: {live}")
        rid = L.digest({"effect_key": key, "registry": self.registry_digest, "n": len(self.events())})[:12]
        return self._append({"event": "proposed", "request_id": rid, "effect_key": key, "mask": mask, "ids": list(ids),
                             "requested_by": requested_by, "registry_digest": self.registry_digest,
                             "budget_gpu_s": budget_gpu_s, "repeat_of": repeat_of})

    def consume(self, request_id: str, mask: int) -> dict:
        req = self._bound(request_id, "approved")
        if req["mask"] != mask:
            raise LedgerError(f"{request_id}: 승인된 부분집합과 다르다")
        return self._append({"event": "consumed", "request_id": request_id})

    def receipt(self, request_id: str, answer: str, cause: str | None, attempt_sha: str | None, gpu_s: float | None,
                reason: str | None = None) -> dict:
        req = self.requests().get(request_id)
        if req is None or req["state"] != "running":
            raise LedgerError(f"{request_id}: 실행 중인 요청이 아니다")
        over = gpu_s is not None and gpu_s > req["budget_gpu_s"]
        if over:
            answer, cause = "unknown", "infrastructure"
        if answer not in self.VERDICTS:
            raise LedgerError(f"모르는 답 {answer!r}")
        event = {"event": "receipt", "request_id": request_id, "answer": answer, "cause": cause, "attempt_sha256": attempt_sha,
                 "gpu_s": gpu_s, "over_budget": over}
        if reason is not None:
            event["reason"] = reason
        return self._append(event)


def ledger(case: str) -> RetrainLedger:
    return RetrainLedger(folder(case) / "ledger.jsonl", sha(PROTOCOL, True), case)


# --- 캐시 ---
def cache_dir(case: str, mask: int) -> Path:
    spec = case_spec(case)
    key = f"s{spec['seed']}_" + ("-".join(ids_of(spec["order"], mask)) or "NONE")
    return PRIVATE / sha(PROTOCOL, True)[:16] / "cache" / key


def attempts(case: str, mask: int) -> list[dict]:
    directory = cache_dir(case, mask)
    return [read(p) for p in sorted(directory.glob("attempt*.json"))] if directory.is_dir() else []


def entry(case: str, mask: int) -> dict:
    """부분집합의 실행 기록. final이면 답이 정해졌다(인프라 원인 unknown은 실행 2회째까지 final이 아니다)."""
    rows = attempts(case, mask)
    if not rows:
        return {"attempts": [], "final": False, "answer": None}
    last = rows[-1]
    # 실행 번호는 원장 소비 횟수로 정해지므로(기록 없이 끝난 실행 포함) 파일 수가 아니라 마지막 실행 번호로 센다.
    final = last["answer"] != "unknown" or last["cause"] != "infrastructure" or last["attempt"] >= MAX_EXECUTIONS
    return {"attempts": rows, "final": final, "answer": last["answer"] if final else None, "last": last}


def answers(case: str) -> dict[int, str]:
    spec = case_spec(case)
    k = len(spec["order"])
    out = {}
    for mask in range(1, 1 << k):
        if cache_dir(case, mask).is_dir():
            row = entry(case, mask)
            if row["final"]:
                out[mask] = row["answer"]
    return out


# --- 학습·평가 한 번 ---
def train_command(isaaclab: Path, seed: int, name: str) -> list[str]:
    return [str(Path(isaaclab) / "_isaac_sim" / "python.bat"), str(ROOT / "bench" / "train_with_d3_changes.py"),
            "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(MAX_ITERATIONS),
            "--seed", str(seed), "--headless", "--run_name", name]


def readback_ok(readback, ids) -> bool:
    """[d3_changes] 줄이 그 부분집합의 id이고 ok이며 코드 교체·결합·env 설정이 모두 맞다."""
    if not isinstance(readback, dict) or readback.get("ok") is not True or readback.get("changes") != list(ids):
        return False
    code = {cid: {t["qualname"] for t in D.code_targets(cid)} for cid in ids if D.CHANGES[cid]["kind"] == "code"}
    config = {cid: set(D.CHANGES[cid]["overrides"]) for cid in ids if D.CHANGES[cid]["kind"] == "config"}
    rows, values = readback.get("code") or {}, readback.get("config") or {}
    return ({cid: set(r) for cid, r in rows.items()} == code
            and all(row.get("swapped") is True and row.get("bound") is True for r in rows.values() for row in r.values())
            and {cid: set(v) for cid, v in values.items()} == config
            and all(value is True if key.startswith("env.") else value == "params"
                    for v in values.values() for key, value in v.items()))


def params_ok(differences: dict | None, ids) -> dict[str, bool]:
    if differences is None:
        return {"params_overrides_applied": False, "params_otherwise_match": False}
    expected = D.expected_params(ids)
    return {"params_overrides_applied": all(differences.get(key) == list(pair) for key, pair in expected.items()),
            "params_otherwise_match": set(differences) - set(expected) <= {H.RUN_NAME_ALLOWED_DIFF}}


def params_differences(run: dict, reference: dict, ids) -> dict:
    shown = set(D.expected_params(ids)) | {H.RUN_NAME_ALLOWED_DIFF}
    return {key: [reference.get(key), run.get(key)] if key in shown else "값 생략"
            for key in sorted(set(run) | set(reference)) if run.get(key) != reference.get(key)}


def _admission() -> dict:
    import record_evidence_media as M
    return M.query_gpu(GPU_PROFILE)


def _process_created(pid: int) -> int | None:
    import run_runtime_readback_smoke as P
    return P._process_creation_filetime(pid)


def _admit() -> dict:
    gpu = _admission()
    if gpu["busy"]:
        raise GPUBusy(f"GPU 사용 중: {gpu.get('busy_reasons')}")
    return gpu


def _wait_for_gpu() -> dict:
    started = time.monotonic()
    while True:
        gpu = _admission()
        waited = time.monotonic() - started
        if not gpu["busy"] or waited >= EVAL_GPU_WAIT_S:
            if gpu["busy"]:
                raise RuntimeError(f"고정 평가 전 GPU가 {EVAL_GPU_WAIT_S}초 동안 비지 않았다")
            return {**gpu, "waited_s": round(waited, 1)}
        time.sleep(EVAL_GPU_POLL_S)


def _postflight(executions: dict) -> None:
    after = _admission()
    owned = {pid: created for execution in executions.values() for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and _process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def _run_subset(case: str, mask: int, attempt: int, isaaclab: Path, protocol: dict, commit: str, *,
                request_id: str | None = None, rep: bool = False, evaluate: bool = True, gpu_before: dict | None = None) -> dict:
    """부분집합 하나를 학습·추출·params 대조·고정 평가한다. 예외는 단계와 함께 failure로 남기고 판정은 judge가 한다."""
    spec = case_spec(case)
    ids = ids_of(spec["order"], mask)
    name = run_name(case, mask, attempt, rep)
    work = folder(case) / "exec" / name
    if work.exists() or find_run_dir(isaaclab, name) or (RUNS_DIR / f"{name}.telemetry.json").exists():
        raise FileExistsError(f"{name}: 실행 산출물이 이미 있다. 덮어쓰지 않는다")
    gpu_before = gpu_before or _admit()
    ctx = {"stage": "training", "executions": {}, "walls": {"train_s": None, "telemetry_extract_s": None, "fixed_eval_s": None}}
    evidence = {"run_name": name, "ids": list(ids), "mask": mask, "attempt": attempt, "request_id": request_id,
                "source_commit_sha": commit, "gpu_before": gpu_before, "failure": None}
    run_dir = None
    try:
        os.environ[D.ENV_VAR] = spec_of(ids)
        try:
            execution = _owned(work / "train", train_command(isaaclab, spec["seed"], name), isaaclab, WATCHDOG_S["training"])
        finally:
            os.environ.pop(D.ENV_VAR, None)
        ctx["executions"]["training"] = execution
        ctx["walls"]["train_s"] = round(execution["wall_s"], 1)
        evidence["train_exit_code"] = execution["exit_code"]
        evidence["readback"] = H.parse_readback((work / "train/stdout.log").read_text(encoding="utf-8", errors="replace"))
        run_dirs = find_run_dir(isaaclab, name)
        run_dir = run_dirs[0] if len(run_dirs) == 1 else None
        checkpoint = run_dir / FINAL_CHECKPOINT if run_dir else None
        evidence["checkpoint_sha256"] = sha(checkpoint) if checkpoint and checkpoint.is_file() else None
        evidence["run_dir"] = str(run_dir) if run_dir else None
        if not finished(execution):
            raise RuntimeError("학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")
        if run_dir is None:
            return {**evidence, "walls": ctx["walls"]}
        ctx["stage"] = "telemetry_extract"
        telemetry = RUNS_DIR / f"{name}.telemetry.json"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        extract = _owned(work / "extract", [str(Path(isaaclab) / "_isaac_sim" / "python.bat"), str(ROOT / "bench/extract_tb.py"),
                                            str(run_dir), str(telemetry)], isaaclab, WATCHDOG_S["telemetry_extract"])
        ctx["executions"]["telemetry_extract"] = extract
        ctx["walls"]["telemetry_extract_s"] = round(extract["wall_s"], 1)
        if not finished(extract) or not telemetry.is_file():
            raise RuntimeError("텔레메트리 추출이 끝나지 않았거나 파일이 없다")
        evidence["telemetry"] = {"path": rel(telemetry), "sha256_lf": sha(telemetry, True)}
        ctx["stage"] = "params"
        if mask:
            none = entry(case, 0)["last"]
            evidence["params_differences"] = params_differences(load_params(run_dir), load_params(Path(none["run_dir"])), ids)
        if not evaluate or not (evidence["train_exit_code"] == 0 and evidence["checkpoint_sha256"]
                                and readback_ok(evidence["readback"], ids)
                                and (not mask or all(params_ok(evidence.get("params_differences"), ids).values()))):
            return {**evidence, "walls": ctx["walls"]}
        ctx["stage"] = "pre_evaluation"
        _verify_case(case, isaaclab)
        evidence["gpu_before_fixed_evaluation"] = _wait_for_gpu()
        ctx["stage"] = "fixed_evaluation"
        report_path = RESULTS / case / "runs" / f"{name}.json"
        jobs = work / "eval" / "jobs.json"
        write_new(jobs, {"action_scale": ACTION_SCALE, "jobs": [{"name": name, "checkpoint": str(checkpoint),
                                                                 "output": str(report_path)}]})
        execution = _owned(work / "eval", [str(Path(isaaclab) / "_isaac_sim" / "python.bat"), str(ROOT / "evals/fixed_eval.py"),
                                           "--protocol", str(FIXED_EVAL_PROTOCOL), "--jobs", str(jobs), "--headless"],
                           isaaclab, WATCHDOG_S["fixed_evaluation"])
        ctx["executions"]["fixed_evaluation"] = execution
        ctx["walls"]["fixed_eval_s"] = round(execution["wall_s"], 1)
        if not finished(execution) or execution["exit_code"] != 0 or not report_path.is_file():
            raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
        evidence["report"] = {"path": rel(report_path), "sha256_lf": sha(report_path, True)}
        ctx["stage"] = "postflight"
        _verify_case(case, isaaclab)
        if sha(checkpoint) != evidence["checkpoint_sha256"]:
            raise ValueError("평가 중 체크포인트가 바뀌었다")
        _postflight(ctx["executions"])
    except Exception as error:
        evidence["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
        write_new(work / "failure.json", {**evidence["failure"], "detail": str(error)})
    return {**evidence, "walls": ctx["walls"]}


CHECK_CAUSES = {"pipeline_completed": "infrastructure", "training_exit_zero": "training_process",
                "final_checkpoint_present": "training_process", "injection_readback": "injection",
                "params_overrides_applied": "injection", "params_otherwise_match": "comparability",
                "fixed_report_valid": "evaluation", "runtime_matches_reference": "comparability",
                "reference_walks": "comparability", "verdict_determined": "evaluation"}
TRAINING_CHECKS = ("training_exit_zero", "final_checkpoint_present", "injection_readback", "params_overrides_applied",
                   "params_otherwise_match")


def judge(case: str, mask: int, evidence: dict, bands: dict, *, evaluated: bool = True) -> dict:
    """유효 조건과 답. NONE(mask 0)은 기준이라 답이 reference이고, 그 밖은 같은 사례 NONE 대비 p0c 밴드 판정이다."""
    ids = evidence["ids"]
    report = read(ROOT / evidence["report"]["path"]) if evidence.get("report") else None
    checks = {"pipeline_completed": evidence["failure"] is None, "training_exit_zero": evidence.get("train_exit_code") == 0,
              "final_checkpoint_present": bool(evidence.get("checkpoint_sha256")),
              "injection_readback": readback_ok(evidence.get("readback"), ids),
              **(params_ok(evidence.get("params_differences"), ids) if mask else
                 {"params_overrides_applied": True, "params_otherwise_match": True})}
    result = {"verdict": None, "comparison": None}
    if evaluated:
        none_report = report if mask == 0 else read(ROOT / entry(case, 0)["last"]["report"]["path"])
        checks.update(fixed_report_valid=False, runtime_matches_reference=False, reference_walks=H.walks(none_report)
                      if none_report else False, verdict_determined=None)
        if report is not None and checks["final_checkpoint_present"]:
            try:
                C.validate_fixed_report(report, {"name": evidence["run_name"], "checkpoint_file": FINAL_CHECKPOINT,
                                                 "checkpoint_sha256": evidence["checkpoint_sha256"], "action_scale": ACTION_SCALE,
                                                 "protocol_sha256_lf": sha(FIXED_EVAL_PROTOCOL, True)})
                checks["fixed_report_valid"] = True
                checks["runtime_matches_reference"] = report.get("versions") == none_report.get("versions")
            except (ValueError, KeyError, TypeError):
                pass
        if mask == 0:
            checks["verdict_determined"] = True
        elif all(ok for name, ok in checks.items() if name != "verdict_determined"):
            comparison = RC.compare(C._behavior(report), C._behavior(none_report))
            verdict = RC.verdict(comparison, bands)
            checks["verdict_determined"] = verdict["label"] in ("healthy", "unhealthy")
            result.update(verdict=verdict, comparison=comparison)
    failed = [name for name, ok in checks.items() if ok is False]
    if failed:
        cause = CHECK_CAUSES[failed[0]]
        stage = (evidence["failure"] or {}).get("stage")
        training_failed = [name for name in TRAINING_CHECKS if name in failed]
        if failed[0] == "pipeline_completed" and stage != "training" and training_failed:
            cause = CHECK_CAUSES[training_failed[0]]  # 학습 뒤 단계 실패보다 학습 산출물 검사가 앞선다
        answer = "unknown"
    else:
        cause = None
        answer = "reference" if mask == 0 else result["verdict"]["label"] if evaluated else "trained"
    return {"checks": checks, "failed_checks": failed, "answer": answer, "cause": cause, **result}


def _record_attempt(case: str, mask: int, evidence: dict, outcome: dict) -> dict:
    record = {**{k: v for k, v in evidence.items() if k not in ("gpu_before", "gpu_before_fixed_evaluation")},
              **outcome, "parent_wall_s": round(sum(v for v in evidence["walls"].values() if v), 1)}
    write_new(cache_dir(case, mask) / f"attempt{evidence['attempt']}.json", record)
    return record


# --- 단계 ---
def setup(case: str, arm: str, isaaclab: Path) -> dict:
    """NONE 또는 실패 실행(변경 8개 전부)을 학습·평가한다. 인프라 원인이면 같은 명령으로 한 번 더 할 수 있다."""
    _check_open(case, "begin", *(("setup_none",) if arm == "failure" else ()))
    if step(case, f"setup_{arm}") is not None:
        raise FileExistsError(f"{case} setup {arm}은 이미 끝났다")
    spec = case_spec(case)
    mask = 0 if arm == "none" else MT.full(len(spec["order"]))
    current = entry(case, mask)
    if current["final"]:
        raise ValueError(f"{case} {arm} 실행은 이미 최종 결과가 있다")
    protocol, commit = _verify_case(case, isaaclab)
    evidence = _run_subset(case, mask, len(current["attempts"]) + 1, isaaclab, protocol, commit)
    record = _record_attempt(case, mask, evidence, judge(case, mask, evidence, protocol["bands"]))
    row = entry(case, mask)
    if not row["final"]:
        return {"case": case, "arm": arm, "answer": "unknown", "cause": record["cause"], "next": "같은 setup 명령으로 한 번 더"}
    out = _write_step(case, f"setup_{arm}", {"mask": mask, "answer": row["answer"], "attempts": len(row["attempts"])})
    if arm == "none" and row["answer"] != "reference":
        _end(case, "case_invalid", {"reason": "NONE 실행이 유효하지 않다", "cause": record["cause"]})
    if arm == "failure" and row["answer"] != "unhealthy":
        _end(case, "case_invalid", {"reason": "실패 실행이 NONE 대비 unhealthy가 아니다", "answer": row["answer"],
                                    "cause": record["cause"]})
    return out


def _probe_once(case: str, mask: int, probes: list[str], isaaclab: Path, stem: str) -> dict:
    """probe 프로세스 하나(그 부분집합을 건 환경). 측정값은 확인 줄 ok·체크포인트 일치·종료 0일 때만.

    stem마다 한 번만 잰다. 이미 끝난 stem은 기록을 돌려주고, 기록 없이 멈춘 stem은 인프라 실패로 센다(다시 재지 않는다).
    """
    spec = case_spec(case)
    ids = ids_of(spec["order"], mask)
    source = entry(case, mask)["last"]
    work = folder(case) / "probe_exec" / stem
    output = RESULTS / case / "probes" / f"{stem}.json"
    if (work / "result.json").is_file():
        return read(work / "result.json")
    if work.exists() or output.exists():
        return {"probes": probes, "measurements": dict.fromkeys(probes), "infrastructure_failure": True,
                "readback_ok": None, "parent_wall_s": 0.0, "report": None, "note": "기록 없이 멈춘 측정"}
    gpu = _admit()
    jobs = work / "jobs.json"
    write_new(jobs, {"fault": "NONE", "seed": spec["seed"], "jobs": [
        {"name": source["run_name"], "checkpoint": str(Path(source["run_dir"]) / FINAL_CHECKPOINT), "probes": probes,
         "output": str(output)}]})
    os.environ[D.ENV_VAR] = spec_of(ids)
    try:
        execution = _owned(work / "run", [str(Path(isaaclab) / "_isaac_sim" / "python.bat"), str(ROOT / "bench/d3_isaac.py"),
                                          "--", "--jobs", str(jobs), "--headless"], isaaclab, WATCHDOG_S["probes"])
    finally:
        os.environ.pop(D.ENV_VAR, None)
    readback = H.parse_readback((work / "run/stdout.log").read_text(encoding="utf-8", errors="replace"))
    report = read(output) if output.is_file() else {}
    ok = (finished(execution) and execution.get("exit_code") == 0 and readback_ok(readback, ids)
          and report.get("checkpoint", {}).get("sha256") == source["checkpoint_sha256"])
    rows = report.get("probes", {})
    measurements = {p: rows[p]["measurement"] if ok and rows.get(p, {}).get("exit") == 0 else None for p in probes}
    result = {"probes": probes, "measurements": measurements,
              "infrastructure_failure": not finished(execution) or not output.is_file(),
              "readback_ok": readback_ok(readback, ids), "gpu_before": gpu, "parent_wall_s": round(execution["wall_s"], 1),
              "report": rel(output) if output.is_file() else None}
    write_new(work / "result.json", result)
    return result


def reference(case: str, isaaclab: Path) -> dict:
    """NONE 정책에서 probe 6종(고리 판정 기준). 인프라 실패면 한 번 더 재고, 그래도 빠지면 case_invalid."""
    _check_open(case, "setup_none", "setup_failure")
    if step(case, "reference") is not None:
        raise FileExistsError(f"{case} reference는 이미 끝났다")
    _verify_case(case, isaaclab)
    runs = []
    name = entry(case, 0)["last"]["run_name"]
    for attempt in range(1, MAX_EXECUTIONS + 1):
        runs.append(_probe_once(case, 0, sorted(L.PROBES), isaaclab, name + ("" if attempt == 1 else f"_a{attempt}")))
        if all(v is not None for v in runs[-1]["measurements"].values()):
            break
    record = _write_step(case, "reference", {"runs": runs, "measurements": runs[-1]["measurements"],
                                             "parent_wall_s": round(sum(r["parent_wall_s"] for r in runs), 1)})
    if any(v is None for v in runs[-1]["measurements"].values()):
        _end(case, "case_invalid", {"reason": "기준 probe 6종을 재지 못했다"})
    return record


def loop(case: str, isaaclab: Path) -> dict:
    """실패 정책에 고정 순서 고리. probe 하나의 인프라 실패는 한 번 더 재고, 측정이 없으면 unknown으로 둔다."""
    _check_open(case, "reference")
    if step(case, "loop") is not None:
        raise FileExistsError(f"{case} loop은 이미 끝났다")
    _verify_case(case, isaaclab)
    reference_values = step(case, "reference")["measurements"]
    full_mask = MT.full(len(case_spec(case)["order"]))
    observed, records = {}, {}
    for path in sorted((folder(case) / "loop").glob("*.json")):  # 앞서 멈춘 고리를 이어 간다
        row = read(path)
        observed[row["probe"]], records[row["probe"]] = row["outcome"], row
    name = entry(case, full_mask)["last"]["run_name"]
    while (probe := MT.loop_state(observed)["next"]) is not None:
        runs = []
        for attempt in range(1, MAX_EXECUTIONS + 1):
            runs.append(_probe_once(case, full_mask, [probe], isaaclab,
                                    f"{name}__{probe}" + ("" if attempt == 1 else f"_a{attempt}")))
            if not runs[-1]["infrastructure_failure"]:
                break
        measurement = runs[-1]["measurements"][probe]
        outcome = L.classify(probe, measurement, reference_values.get(probe))
        row = {"probe": probe, "outcome": outcome, "measurement": measurement, "runs": runs,
               "parent_wall_s": round(sum(r["parent_wall_s"] for r in runs), 1)}
        write_new(folder(case) / "loop" / f"{probe}.json", row)
        observed[probe], records[probe] = outcome, row
    state = MT.loop_state(observed)
    return _write_step(case, "loop", {**state, "measurements": {p: r["measurement"] for p, r in records.items()},
                                      "parent_wall_s": round(sum(r["parent_wall_s"] for r in records.values()), 1)})


def _summary_of(attempt: dict) -> dict:
    return read(ROOT / attempt["telemetry"]["path"])["summary"]


def packet(case: str, isaaclab: Path) -> dict:
    """B4·B5 packet. 단서 canary가 0건일 때만 쓰고, WSL 호출 명령을 돌려준다."""
    _check_open(case, "loop")
    if step(case, "packet") is not None:
        raise FileExistsError(f"{case} packet은 이미 있다")
    _verify_case(case, isaaclab)
    spec = case_spec(case)
    full_mask = MT.full(len(spec["order"]))
    facts = AG.telemetry_facts(_summary_of(entry(case, full_mask)["last"]), _summary_of(entry(case, 0)["last"]))
    loop_record = step(case, "loop")
    observations = AG.probe_observations(loop_record["steps"], loop_record["measurements"],
                                         step(case, "reference")["measurements"], L.THRESHOLDS)
    diffs = [R.render(cid, isaaclab) for cid in spec["order"]]
    packets = {"B4": AG.build_packet(diffs, facts, observations), "B5": AG.build_packet(diffs, facts, [])}
    out = {}
    for method, body in packets.items():
        clues = AG.clues(body)
        if clues:
            raise ValueError(f"{method} packet에 단서가 있다: {clues}")
        path = folder(case) / "model" / f"{method}.packet.json"
        write_new(path, body)
        out[method] = {"path": rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    commands = {m: f"python3 {wsl(ROOT / 'evals/d3_model_call.py')} --packet {wsl(ROOT / v['path'])} "
                   f"--output {wsl(folder(case) / 'model' / f'{m}.call1.json')} --model {MODEL}" for m, v in out.items()}
    return {**_write_step(case, "packet", {"packets": out}), "wsl_commands": commands}


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}" if re.fullmatch(r"[A-Za-z]:/.*", p) else p


def rank(case: str, method: str, record_path: Path, isaaclab: Path) -> dict:
    """호출 기록을 packet에 묶어 검증한다. 인프라 오류만 한 번 더 부를 수 있다."""
    if method not in ("B4", "B5"):
        raise ValueError("rank는 B4·B5만")
    _check_open(case, "packet")
    if step(case, f"rank_{method}") is not None:
        raise FileExistsError(f"{case} {method} 순위는 이미 정해졌다")
    _verify_case(case, isaaclab)
    packet_path = ROOT / step(case, "packet")["packets"][method]["path"]
    record = read(record_path)
    reviewed = AG.review(record, packet_path.read_bytes(), MODEL)
    previous = sorted((folder(case) / "model").glob(f"{method}.review*.json"))
    attempt = len(previous) + 1
    if attempt > MAX_EXECUTIONS:
        raise ValueError(f"{method} 호출은 {MAX_EXECUTIONS}회까지다")
    write_new(folder(case) / "model" / f"{method}.review{attempt}.json",
              {**reviewed, "record_sha256": hashlib.sha256(Path(record_path).read_bytes()).hexdigest(),
               "record_path": rel(record_path)})
    if reviewed["status"] == "infra_error" and attempt < MAX_EXECUTIONS:
        return {"case": case, "method": method, "status": "infra_error", "next": f"새 기록 파일로 한 번 더 호출(.call{attempt + 1}.json)"}
    labels = dict(zip(AG.labels(len(case_spec(case)["order"])), range(len(case_spec(case)["order"]))))
    valid = reviewed["status"] == "valid"
    tokens = [read(p).get("total_tokens") for p in [*previous, folder(case) / "model" / f"{method}.review{attempt}.json"]]
    return _write_step(case, f"rank_{method}", {
        "status": "valid" if valid else "model_invalid", "review_status": reviewed["status"], "problems": reviewed["problems"],
        "ranking_labels": reviewed["ranking"], "ranking": [labels[x] for x in reviewed["ranking"]] if valid else None,
        "uncertain": reviewed.get("uncertain"), "attempts": attempt,
        "total_tokens": sum(t for t in tokens if isinstance(t, int)) if any(isinstance(t, int) for t in tokens) else None})


def method_states(case: str) -> dict[str, dict]:
    spec = case_spec(case)
    k, known = len(spec["order"]), answers(case)
    loop_record, states = step(case, "loop"), {}
    for method in MT.METHODS:
        if method == "B3" and loop_record is None:
            states[method] = {"method": method, "status": "waiting", "reason": "loop", "queries": [], "next": None}
            continue
        ranked = step(case, f"rank_{method}") if method in ("B4", "B5") else None
        if method in ("B4", "B5") and ranked is None:
            states[method] = {"method": method, "status": "waiting", "reason": f"rank_{method}", "queries": [], "next": None}
            continue
        states[method] = MT.run_method(method, k, known, categories=spec["categories"],
                                       loop_conclusion=(loop_record or {}).get("conclusion"),
                                       ranking=(ranked or {}).get("ranking"),
                                       ranking_invalid=bool(ranked) and ranked["status"] != "valid")
    return states


def _interaction_check(case: str) -> list[int]:
    spec = case_spec(case)
    found = MT.interaction(answers(case), spec["harmful_index"], len(spec["order"]))
    if found and not terminal(case):
        _end(case, "interaction_observed", {"masks": found})
    return found


def plan(case: str, isaaclab: Path) -> dict:
    """방법들이 다음에 필요한 부분집합을 모아 원장에 제안한다(진행 중·최종 답이 있는 부분집합은 빼고)."""
    _check_open(case, "setup_failure")
    _verify_case(case, isaaclab)
    if _interaction_check(case):
        return {"case": case, "status": "interaction_observed"}
    spec = case_spec(case)
    states = method_states(case)
    needed: dict[int, list[str]] = {}
    for method, state in states.items():
        if state["status"] == "pending":
            needed.setdefault(state["next"], []).append(method)
    book = ledger(case)
    requests = book.requests()
    proposed = []
    for mask, methods in sorted(needed.items()):
        ids = ids_of(spec["order"], mask)
        same = [r for r in requests.values() if r.get("effect_key") == book.effect(ids)]
        if any(r["state"] in ("pending", "approved", "running") for r in same):
            continue
        done = [r["request_id"] for r in same if r["state"] == "done"]
        row = book.propose(mask, ids, methods, repeat_of=done[-1] if done else None)
        reverted = [cid for cid in spec["order"] if cid not in ids]
        proposed.append({"request_id": row["request_id"], "mask": f"{mask:02x}", "reverted": reverted, "requested_by": methods})
    return {"case": case, "proposed": proposed, "methods": {m: s["status"] for m, s in states.items()},
            "pending_requests": [rid for rid, r in book.requests().items() if r["state"] in ("pending", "approved")],
            "approval": APPROVAL}


def approve(case: str, request_ids: list[str], approver: str) -> dict:
    if not approver.strip():
        raise ValueError("승인자 문자열이 필요하다")
    book = ledger(case)
    return {"case": case, "approved": [book.approve(rid, approver)["request_id"] for rid in request_ids]}


def retrain(case: str, request_id: str, isaaclab: Path) -> dict:
    """승인을 한 번 소비하고 그 부분집합을 학습·평가·판정해 캐시에 넣는다."""
    _check_open(case, "setup_failure")
    book = ledger(case)
    req = book.requests().get(request_id)
    if req is None or req["state"] != "approved":
        raise ValueError(f"{request_id}: 승인된 요청이 아니다")
    if entry(case, req["mask"])["final"]:
        raise ValueError(f"{request_id}: 그 부분집합은 이미 최종 답이 있다(캐시)")
    protocol, commit = _verify_case(case, isaaclab)
    executions = sum(1 for r in book.requests().values() if r.get("effect_key") == req["effect_key"]
                     and r["state"] in ("running", "done"))
    gpu = _admit()  # 승인을 소비하기 전에 GPU를 확인한다
    book.consume(request_id, req["mask"])
    evidence = _run_subset(case, req["mask"], executions + 1, isaaclab, protocol, commit, request_id=request_id,
                           gpu_before=gpu)
    record = _record_attempt(case, req["mask"], evidence, judge(case, req["mask"], evidence, protocol["bands"]))
    attempt_sha = sha(cache_dir(case, req["mask"]) / f"attempt{evidence['attempt']}.json")
    book.receipt(request_id, record["answer"], record["cause"], attempt_sha, record["parent_wall_s"])
    _interaction_check(case)
    return {"case": case, "request_id": request_id, "answer": record["answer"], "cause": record["cause"],
            "final": entry(case, req["mask"])["final"]}


def recover(case: str, request_id: str) -> dict:
    """소비했지만 receipt가 없는 요청을 닫는다. 그 요청의 실행 기록이 있으면 그 답으로, 없으면 인프라 원인 unknown."""
    book = ledger(case)
    if request_id not in book.orphans():
        raise ValueError(f"{request_id}: 결과 없이 끝난 요청이 아니다")
    req = book.requests()[request_id]
    found = [(p, read(p)) for p in sorted(cache_dir(case, req["mask"]).glob("attempt*.json"))
             if read(p).get("request_id") == request_id] if cache_dir(case, req["mask"]).is_dir() else []
    if found:
        path, row = found[-1]
        return book.receipt(request_id, row["answer"], row["cause"], sha(path), row["parent_wall_s"], reason="recovered_from_attempt")
    return book.receipt(request_id, "unknown", "infrastructure", None, None, reason="orphan_without_attempt")


def determinism(case: str, isaaclab: Path, rerun: bool = False) -> dict:
    """D0: 해로운 변경만 되돌린 부분집합의 체크포인트를 무해성 smoke s2026과, 필요하면 한 번 더 학습한 것과 대조한다."""
    if case != "D0":
        raise ValueError("결정성 대조는 D0에서만 한다")
    if step(case, "determinism") is not None:
        raise FileExistsError("결정성 대조는 이미 기록됐다")
    _check_open(case, "setup_failure")
    spec = case_spec(case)
    mask = MT.single_revert(len(spec["order"]), spec["harmful_index"])
    row = entry(case, mask)
    if not row["final"] or not row["last"].get("checkpoint_sha256"):
        raise ValueError("해로운 변경만 되돌린 부분집합의 학습 결과가 아직 없다")
    first = row["last"]["checkpoint_sha256"]
    smoke = None
    smoke_final = H._consumed(spec["seed"], "final")
    if smoke_final.is_file():
        smoke_result = read(H.RESULTS / read(smoke_final)["tag"] / "result.json")
        smoke = {"tag": smoke_result["tag"], "checkpoint_sha256": smoke_result["checkpoint_sha256"],
                 "same": smoke_result["checkpoint_sha256"] == first, "changes": smoke_result["changes"]}
        if smoke_result["changes"] != list(ids_of(spec["order"], mask)):
            raise ValueError("smoke 부분집합이 D0 대조 부분집합과 다르다")
    second = None
    if not (smoke and smoke["same"]):
        if not rerun:
            return {"case": case, "status": "rerun_needed", "smoke": smoke,
                    "next": "determinism --case D0 --rerun (GPU 학습 1회, 평가 없음)"}
        protocol, commit = _verify_case(case, isaaclab)
        evidence = _run_subset(case, mask, 1, isaaclab, protocol, commit, rep=True, evaluate=False)
        outcome = judge(case, mask, evidence, protocol["bands"], evaluated=False)
        second = {"run_name": evidence["run_name"], "checkpoint_sha256": evidence.get("checkpoint_sha256"),
                  "answer": outcome["answer"], "cause": outcome["cause"], "parent_wall_s": round(sum(
                      v for v in evidence["walls"].values() if v), 1)}
        write_new(folder(case) / "determinism_rerun.json", {**evidence, **outcome})
    same = bool(smoke and smoke["same"]) or bool(second and second["checkpoint_sha256"] == first)
    source = "smoke" if smoke and smoke["same"] else "rerun"
    return _write_step(case, "determinism", {"mask": mask, "first_checkpoint_sha256": first, "smoke": smoke, "rerun": second,
                                             "same": same, "source": source})


def _gpu_s(case: str, mask: int) -> float:
    return round(sum(a["parent_wall_s"] for a in entry(case, mask)["attempts"]), 1)


def close(case: str) -> dict:
    """방법 결과를 모아 사례 결과를 공개 폴더에 쓴다. 대기·진행 중 방법이 있으면 거부한다."""
    if step(case, "close") is not None:
        raise FileExistsError(f"{case}는 이미 닫혔다")
    if step(case, "begin") is None:
        raise ValueError(f"{case}: begin 전이다")
    spec = case_spec(case)
    end = terminal(case)
    states = method_states(case) if step(case, "setup_failure") else {}
    open_methods = [m for m, s in states.items() if s["status"] in ("pending", "waiting")]
    if end is None and (open_methods or not states):
        raise ValueError(f"{case}: 아직 끝나지 않은 방법 {open_methods or '(setup 전)'}")
    if case == "D0" and end is None and step(case, "determinism") is None:
        raise ValueError("D0는 결정성 대조 뒤에 닫는다")
    probe_s = sum((step(case, n) or {}).get("parent_wall_s", 0) for n in ("reference", "loop"))
    methods = {}
    for method, state in states.items():
        cost = SU.method_cost(state, spec["harmful_index"]) if state["status"] in ("confirmed", "unresolved") else None
        rank_record = step(case, f"rank_{method}") if method in ("B4", "B5") else None
        methods[method] = {
            "status": state["status"], "reason": state.get("reason"),
            "confirmed": None if state.get("confirmed") is None else spec["order"][state["confirmed"]],
            "queries": [{"reverted": [cid for cid in spec["order"] if cid not in ids_of(spec["order"], m)],
                         "answer": answers(case).get(m)} for m in state["queries"]],
            "cost": cost, "retrain_parent_wall_s": round(sum(_gpu_s(case, m) for m in state["queries"]), 1),
            "probe_parent_wall_s": round(probe_s, 1) if method in ("B3", "B4") else 0.0,
            "model": None if rank_record is None else {k: rank_record[k] for k in ("status", "ranking_labels", "uncertain",
                                                                                   "attempts", "total_tokens")}}
    valid = end is None
    result = {"schema": "d3_case_result_v1", "case": case, "seed": spec["seed"], "aggregate": spec["aggregate"],
              "harmful": spec["harmful"], "position": spec["position"], "order": spec["order"], "labels": spec["labels"],
              "valid": valid, "invalid_reason": None if valid else end["outcome"], "terminal": end,
              "methods": methods, "loop": {k: (step(case, "loop") or {}).get(k) for k in ("status", "conclusion", "probes")},
              "determinism": step(case, "determinism") if case == "D0" else None,
              "shared_parent_wall_s": {"none": _gpu_s(case, 0), "failure": _gpu_s(case, MT.full(len(spec["order"])))},
              "source_commit_sha": step(case, "begin")["source_commit_sha"], "protocol_sha256_lf": sha(PROTOCOL, True),
              "approval": APPROVAL, "model_calls": sum((methods.get(m, {}).get("model") or {}).get("attempts") or 0
                                                       for m in ("B4", "B5"))}
    write_new(RESULTS / case / "case_result.json", result)
    _write_step(case, "close", {"result": rel(RESULTS / case / "case_result.json")})
    return result


def summarize(tag: str) -> dict:
    if not isinstance(tag, str) or not re.fullmatch(SUMMARY_TAG_RE, tag):
        raise ValueError(f"tag 형식이 {SUMMARY_TAG_RE}가 아니다")
    rows = []
    for case in CASES:
        path = RESULTS / case / "case_result.json"
        if not path.is_file():
            if CASES[case]["aggregate"]:
                raise ValueError(f"{case}가 아직 닫히지 않았다")
            continue
        result = read(path)
        spec = case_spec(case)
        states = {m: {"status": v["status"], "reason": v["reason"],
                      "confirmed": None if v["confirmed"] is None else spec["order"].index(v["confirmed"]),
                      "queries": v["queries"]} for m, v in result["methods"].items()}
        rows.append({"case": case, "aggregate": result["aggregate"], "valid": result["valid"] and len(states) == 5,
                     "invalid_reason": result["invalid_reason"], "harmful_index": spec["harmful_index"], "methods": states})
    marker = PRIVATE / sha(PROTOCOL, True)[:16] / "summary.json"
    if marker.exists() or (SUMMARIES / tag).exists():
        raise FileExistsError("요약은 프로토콜마다 한 번만 쓴다")
    verdict = SU.verdict(rows)
    write_new(marker, {"tag": tag, "outcome": verdict["outcome"]})
    summary = {"schema": "d3_experiment_summary_v1", "tag": tag, "protocol_sha256_lf": sha(PROTOCOL, True), **verdict,
               "cases": {r["case"]: rel(RESULTS / r["case"] / "case_result.json") for r in rows},
               "forbidden_claims": list(SU.FORBIDDEN_CLAIMS), "approval": APPROVAL}
    write_new(SUMMARIES / tag / "summary.json", summary)
    return {"outcome": verdict["outcome"], "summary": f"evals/results/{tag}/summary.json"}


def status(case: str) -> dict:
    states = method_states(case) if step(case, "setup_failure") else {}
    return {"case": case, "terminal": terminal(case),
            "steps": sorted(p.stem for p in (folder(case) / "steps").glob("*.json")) if (folder(case) / "steps").is_dir() else [],
            "methods": {m: {"status": s["status"], "queried": len(s["queries"]), "next": s.get("next")} for m, s in states.items()},
            "requests": {rid: r["state"] for rid, r in ledger(case).requests().items()} if (folder(case) / "ledger.jsonl").exists()
            else {}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "begin", "setup", "reference", "loop", "packet", "rank", "plan",
                                            "approve", "retrain", "recover", "determinism", "close", "summarize", "status"))
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=("none", "failure"))
    parser.add_argument("--method", choices=("B4", "B5"))
    parser.add_argument("--record", type=Path)
    parser.add_argument("--request", nargs="+")
    parser.add_argument("--approver")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    needs = {"setup": ("case", "arm"), "rank": ("case", "method", "record"), "approve": ("case", "request", "approver"),
             "retrain": ("case", "request"), "recover": ("case", "request"), "summarize": ("tag",)}
    for name in needs.get(args.command, ("case",) if args.command not in ("prepare", "verify") else ()):
        if getattr(args, name) in (None, []):
            parser.error(f"{args.command}에는 --{name}가 필요하다")
    import record_evidence_media as M
    try:
        out = {
            "prepare": lambda: prepare(args.isaaclab),
            "verify": lambda: {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]},
            "begin": lambda: begin(args.case, args.isaaclab),
            "setup": lambda: setup(args.case, args.arm, args.isaaclab),
            "reference": lambda: reference(args.case, args.isaaclab),
            "loop": lambda: loop(args.case, args.isaaclab),
            "packet": lambda: packet(args.case, args.isaaclab),
            "rank": lambda: rank(args.case, args.method, args.record, args.isaaclab),
            "plan": lambda: plan(args.case, args.isaaclab),
            "approve": lambda: approve(args.case, args.request, args.approver),
            "retrain": lambda: retrain(args.case, args.request[0], args.isaaclab),
            "recover": lambda: recover(args.case, args.request[0]),
            "determinism": lambda: determinism(args.case, args.isaaclab, args.rerun),
            "close": lambda: close(args.case),
            "summarize": lambda: summarize(args.tag),
            "status": lambda: status(args.case),
        }[args.command]()
    except GPUBusy as error:
        print(json.dumps({"status": "gpu_busy_no_launch", "detail": str(error)}, ensure_ascii=False))
        return 3
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, LedgerError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
