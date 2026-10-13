"""D3 무해성 smoke: 무해 변경 7개를 모두 적용한 학습이 같은 seed 정상 학습 대비 여유 있게 healthy인지 본다.

D3(코드 수준 다중 변경과 이분 탐색 기준선)는 "무해 부분집합은 healthy"를 가정한다. 무해 변경이 판정 경계 근처에
있으면 이분 탐색 도중 결과가 흔들리므로, 본실험 사전등록 전에 무해 7개(bench/d3_changes.py HARMLESS)를 모두 적용해
이미 본 seed 2026·2027에서 한 번씩 학습·고정 평가하고 p0c 밴드에 여유 기준을 더해 거른다. 기준은 P0-C의 같은 seed
정상 실행(baseline_p0c_s<seed>)이다. 무해성 판정만 하고 진단 성능·방법 비교는 재지 않는다.
지금 판은 v2(bench/protocols/d3_harmless_smoke_v2.json)다. v1은 fail이었고 무해 목록에서 H2를 S4로 바꿨다(FIXED version_history).

  python evals/run_d3_harmless_smoke.py prepare           # 프로토콜 작성(커밋 전)
  python evals/run_d3_harmless_smoke.py verify            # 계약·코드·Isaac Lab 소스·기준 입력·커밋 확인
  python evals/run_d3_harmless_smoke.py execute --seed 2026 --tag d3_harmless_smoke_<날짜>_s2026   # GPU
  python evals/run_d3_harmless_smoke.py execute --seed 2026 --tag <새 tag> --retry-of <앞 tag>       # 시도 2
  python evals/run_d3_harmless_smoke.py resume --tag <새 tag> --from <앞 tag>       # 학습은 두고 추출·평가·판정만
  python evals/run_d3_harmless_smoke.py summarize --tag d3_harmless_summary_<날짜>  # 두 seed 최종 결과 종합

실행 구조(소유 Windows Job·watchdog·GPU 입장·결합 경로 재확인·시도 2·이어 하기·최초 유효 결과 최종)는
evals/run_upstream_smoke.py(U1 smoke)와 같고, 그 모듈의 공용 함수를 그대로 가져다 쓴다. 모델·네트워크는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import d3_changes as D  # noqa: E402
import d3_diff as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
import run_upstream_smoke as S  # noqa: E402

sha, read, write_new = S.sha, S.read, S.write_new
standstill_ratio, walks, load_params, finished = S.standstill_ratio, S.walks, S.load_params, S.finished
find_run_dir, _owned = S.find_run_dir, S._owned  # 실행 폴더 찾기·소유 Job 실행은 U1 smoke와 같다

# v2: v1(bench/protocols/d3_harmless_smoke_v1.json, 결과 a17480d, fail)에서 무해 목록만 H2 → S4로 바꿨다.
# v1 프로토콜·결과는 그대로 두고, v1 실행 코드는 커밋 a17480d에 있다. 이 파일이 바뀌어 v1 verify는 HEAD에서 거부된다.
VERSION = "v2"
PROTOCOL = ROOT / f"bench/protocols/d3_harmless_smoke_{VERSION}.json"
PREVIOUS = {"protocol": "bench/protocols/d3_harmless_smoke_v1.json", "summary": "evals/results/d3_harmless_summary_20261014/summary.json",
            "results": {"2026": "evals/results/d3_harmless_smoke_20261014_s2026/result.json",
                        "2027": "evals/results/d3_harmless_smoke_20261014_s2027/result.json"}}
PRIVATE = ROOT / "bench/private/d3_harmless_smoke"
RESULTS = ROOT / "evals/results"
RUNS_DIR = ROOT / "bench/runs_d3"
P0C = ROOT / "evals/results/p0c_fixed_eval_20261009"
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
SEEDS, NUM_ENVS, MAX_ITERATIONS, ACTION_SCALE = (2026, 2027), 4096, 300, 0.25
CHANGES = D.HARMLESS
SPEC = ",".join(CHANGES)
FINAL_CHECKPOINT = "model_299.pt"
TAG_RE = r"d3_harmless_smoke_[0-9]{8}[a-z0-9_]{0,16}"
SUMMARY_TAG_RE = r"d3_harmless_summary_[0-9]{8}[a-z0-9_]{0,10}"
MARGINS = {"err_xy_ratio_max": 1.05, "err_yaw_ratio_max": 1.05, "fall_frac_max": 0.02}
RUN_NAME_ALLOWED_DIFF = "agent.yaml:run_name"
GPU_PROFILE = "base_com_reset_readback_v2"
WATCHDOG_S = {"training": 2700, "telemetry_extract": 300, "fixed_evaluation": 360}
EVAL_GPU_WAIT_S, EVAL_GPU_POLL_S = 1800, 60
RESUMABLE_STAGES = ("telemetry_extract", "pre_evaluation", "fixed_evaluation", "postflight")
TRAINING_CHECKS = ("training_exit_zero", "final_checkpoint_present", "injection_readback")
PARAMS_CHECKS = ("params_overrides_applied", "params_otherwise_match")
ISAACLAB_FILES = sorted({target["file"] for cid in CHANGES for target in D.code_targets(cid)}
                        | {D.CHANGES[cid]["file"] for cid in CHANGES if D.CHANGES[cid]["kind"] == "config"}
                        | {"scripts/reinforcement_learning/rsl_rl/train.py"})
CODE = ("bench/d3_changes.py", "bench/d3_diff.py", "bench/train_with_d3_changes.py", "evals/run_d3_harmless_smoke.py",
        "bench/upstream_faults.py", "bench/hidden_faults.py", "evals/run_upstream_smoke.py",
        "bench/extract_tb.py", "evals/fixed_eval.py", "bench/protocols/fixed_eval_v1.json",
        "src/rl_triage/recovery.py", "src/rl_triage/grounded_recovery.py", "bench/record_evidence_media.py",
        "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py")


def run_name(seed: int, attempt: int) -> str:
    """프로토콜 판을 넣어 앞뒤 판의 실행 폴더·텔레메트리와 이름이 겹치지 않게 한다."""
    return f"d3_harmless_{VERSION}_s{seed}" + ("" if attempt == 1 else f"_a{attempt}")


def reference_run(seed: int) -> str:
    return f"baseline_p0c_s{seed}"


def catalog_entry(cid: str) -> dict:
    """프로토콜에 남길 변경 정의(함수 객체는 빼고 JSON으로 쓸 수 있는 값만)."""
    change = D.CHANGES[cid]
    entry = {key: change[key] for key in ("role", "kind", "category", "definition", "sources")}
    if change["kind"] == "code":
        entry["targets"] = [{"file": t["file"], "qualname": t["qualname"], "edits": [list(e) for e in t["edits"]]}
                            for t in D.code_targets(cid)]
    else:
        entry.update(overrides={key: list(pair) for key, pair in change["overrides"].items()},
                     file=change["file"], scope=change["scope"], edits=[list(e) for e in change["edits"]],
                     range=change["range"])
    return entry


FIXED = {
    "schema": f"d3_harmless_smoke_protocol_{VERSION}",
    "version_history": {
        "v1": {**PREVIOUS, "commit": "a17480d", "changes": list(D.HARMLESS_V1), "outcome": "fail",
               "observed": {"2026": {"verdict": "healthy", "err_xy_ratio": 1.0623, "err_yaw_ratio": 0.9464, "fall_frac": 0.0},
                            "2027": {"verdict": "healthy", "err_xy_ratio": 1.0896, "err_yaw_ratio": 0.783, "fall_frac": 0.0}},
               "reading": "[관찰] 두 seed 모두 p0c healthy·낙상 0이었지만 선속도 오차 비만 여유 기준 1.05를 넘었고 yaw 오차 비는 1 "
                          "미만이었다. [추론] 선속도 추종 보상 가중치를 낮춘 H2(1.5 → 1.4)가 가장 유력하다(선속도 추종 비중이 줄고 "
                          "상대적으로 회전 비중이 커지는 방향과 일치). 7개를 함께 적용해 원인은 가르지 않았다"},
        "v2_change": {"retired": list(D.RETIRED), "added": ["S4"],
                      "rule": "v1 결과를 보고 고른 교체다. H2를 같은 reward 범주의 수식 보존 예비 S4(action_rate_l2 제곱 → 같은 값의 곱)로 "
                              "바꿨다. 여유 기준(오차 비 1.05·낙상 0.02)·seed(2026·2027)·나머지 6개 변경은 바꾸지 않았다",
                      "caveat": "v1과 같은 seed라 나머지 6개는 이미 이 seed에서 (H2와 함께) 학습된 적이 있다. v2 pass도 H2가 원인이었다는 "
                                "증명이 아니다(H2를 단독으로 재지 않았다)"}},
    "question": "D3 무해 변경 v2 목록 7개를 모두 적용한 Go2 flat 4096 env × 300회 학습이 이미 본 seed 2026·2027에서 같은 seed "
                "정상 학습 대비 p0c 밴드 healthy이고 여유 기준 안인가",
    "scope": "D3 본실험 사전등록 전 무해 목록 확정을 위한 거르기. 변경 하나하나의 무해성·변경 간 상호작용·새 seed에서의 "
             "무해성·진단 성능·방법 비교가 아니다",
    "changes": {cid: catalog_entry(cid) for cid in CHANGES},
    "reserves": {cid: catalog_entry(cid) for cid in D.RESERVE},
    "amendments_from_design_draft": D.AMENDMENTS,
    "apply": {
        "env_var": D.ENV_VAR, "spec": SPEC, "hydra_overrides": D.hydra_overrides(CHANGES),
        "expected_params": {key: list(pair) for key, pair in D.expected_params(CHANGES).items()},
        "allowed_other_params_difference": RUN_NAME_ALLOWED_DIFF,
        "method": "코드 변경은 환경 생성 직전 함수 __code__ 교체(bench/d3_changes.py before_env), 설정 변경은 같은 값의 "
                  "hydra override를 train.py 인자 뒤에 붙인다(bench/train_with_d3_changes.py). 변경 id는 환경 변수로만 "
                  "넘기며 명령줄·실행 이름에 남지 않는다",
        "diff_rendering": "bench/d3_diff.py가 변경마다 Isaac Lab 소스 diff를 만들고, 코드는 실제 교체 본문과 AST가 같은지, "
                          "설정은 바뀐 숫자가 override와 같은지 확인한다. prepare·verify는 무해·예비·해로운 변경 전부가 "
                          "렌더링되고 단서 canary가 0건일 때만 통과한다",
    },
    "task": TASK, "seeds": list(SEEDS), "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS,
    "action_scale": ACTION_SCALE, "final_checkpoint": FINAL_CHECKPOINT,
    "training_entry": "bench/train_with_d3_changes.py(TRIAGE_D3_CHANGES=H1,…,H7), run_case.ps1과 같은 train.py 인자",
    "reference": {
        "runs": {str(seed): reference_run(seed) for seed in SEEDS},
        "fixed_reports": {str(seed): f"evals/results/p0c_fixed_eval_20261009/runs/{reference_run(seed)}.json" for seed in SEEDS},
        "bands_source": "evals/results/p0c_fixed_eval_20261009/summary.json",
        "reuse": "P0-C 정상 실행(같은 task·seed·예산, bench/train_with_fault.py NONE)의 학습·고정 평가를 다시 쓰고 정상 학습은 하지 않는다",
    },
    "seed_choice": {
        "decision": "이미 본 seed 2026·2027. 같은 task·seed·예산의 정상 학습·고정 평가가 있어 학습 2회로 끝난다(설계안 2.3절)",
        "cons": ["이미 본 seed라 미관측 seed 평가가 아니다",
                 "D3 본실험 새 seed(2035~2037)에서의 무해성은 이 smoke가 보장하지 않는다. 본실험은 사례마다 같은 seed 정상 학습을 "
                 "따로 하고, 무해 부분집합이 unhealthy면 interaction_observed로 처리한다(설계안 2.3절)",
                 "기준은 2026-10-09 학습이라 그사이 런타임 변화가 섞일 수 있다 → 평가 런타임 버전 일치와 params 일치를 유효 조건으로 둔다",
                 "기준은 다른 진입점(bench/train_with_fault.py, TRIAGE_FAULT=NONE)으로 학습됐다. 변경이 없을 때 두 진입점이 같은 "
                 "학습을 내는지 실행으로 대조하지 않았다",
                 "7개를 함께 적용하므로 변경끼리 효과가 상쇄되는 경우를 가르지 못한다",
                 "같은 고정 평가 계약의 정상 기준 baseline_p0c_s2026~s2034(2032 없음) 8개 중 s2026은 yaw 오차가 가장 크고 선속도 "
                 "오차가 두 번째로 크며, s2027은 둘 다 네 번째다. 기준 오차가 큰 편이라 같은 1.05 여유 기준이 오차가 작은 기준에서보다 "
                 "느슨하고, 오차가 작은 새 seed 기준에서는 더 엄격하게 작용할 수 있다",
                 "v1과 같은 seed를 다시 쓴다. v1 결과(선속도 오차 비 초과)를 본 뒤 목록을 바꿨으므로 v2 결과는 결과를 보고 고친 목록의 "
                 "확인이다(version_history.v2_change)"],
    },
    "judgment": {
        "bands": "P0-C summary.json bands(p0c 밴드)로 같은 seed 기준 고정 평가와 비교한 healthy/unhealthy",
        "margins": MARGINS,
        "margin_scope": "선속도·yaw 오차 비는 같은 seed 기준 대비 비, 낙상은 이 실행의 고정 평가 낙상 비율(절대값)이다. "
                        "생존 비는 p0c 밴드만 본다",
        "margin_reason": "무해 변경이 판정 경계 근처에 있으면 이분 탐색 도중 무해 부분집합이 unhealthy로 나와 결과가 흔들린다(설계안 2.3절)",
        "seed_outcome": "유효 조건을 모두 채우고 p0c healthy이며 여유 기준 셋을 모두 지키면 harmless_pass, p0c unhealthy이거나 "
                        "여유 기준 하나라도 넘으면 harmless_fail, 유효 조건이 어긋나면 invalid",
    },
    "execution": {
        "owned_process": "bench/run_runtime_readback_smoke.run_owned: 소유 Windows Job, 시간 초과 시 프로세스 트리 종료, stdout·stderr 분리",
        "watchdog_s": WATCHDOG_S,
        "gpu_admission_profile": GPU_PROFILE,
        "gpu_admission": "학습 전 GPU가 사용 중이면 아무것도 쓰지 않고 실행하지 않는다. 고정 평가 전에는 최대 "
                         f"{EVAL_GPU_WAIT_S}초 동안 {EVAL_GPU_POLL_S}초 간격으로 다시 확인한다",
        "fixed_evaluation_entry": "evals/fixed_eval.py --protocol bench/protocols/fixed_eval_v1.json --jobs <jobs.json> --headless(U1 smoke와 같은 호출)",
        "reverify": "고정 평가 직전과 평가 뒤에 프로토콜·결합 코드·Isaac Lab 대상 파일을 다시 확인하고(결합 경로 clean·커밋됨, "
                    "시작 커밋과 현재 HEAD 사이 결합 경로 diff 없음), 평가 뒤 체크포인트 SHA를 다시 계산한다. 결합 밖 파일의 "
                    "커밋은 막지 않고 본 커밋을 모두 기록한다",
        "seed_order": "seed 2026을 먼저, 2027을 다음에 실행한다. 한 seed가 harmless_fail로 확정되면 전체는 fail이므로 남은 seed는 "
                      "실행하지 않아도 된다",
        "skip_rule": "학습 측 검사(종료 코드·체크포인트·교체 확인 줄·params)가 하나라도 어긋나면 판정에 쓰지 않을 고정 평가는 하지 않는다",
        "attempts": {str(seed): {"1": run_name(seed, 1), "2": run_name(seed, 2)} for seed in SEEDS},
        "retry_rule": "seed마다 시도 2는 같은 프로토콜에서 시도 1의 마지막 결과(이어 하기를 했으면 그 결과)가 끝났고 invalid_cause가 "
                      "infrastructure일 때, 그 결과의 tag를 지정해 한 번만 한다. 변경·seed·판정 기준은 같다",
        "resume_rule": "학습 측 검사(종료 코드·체크포인트·교체 확인 줄)를 통과한 뒤 추출·평가 단계의 인프라 원인으로 invalid가 된 결과는 "
                       "새 tag로 같은 학습의 추출·평가·판정만 한 번 다시 한다. 학습은 다시 하지 않고, 다음 시도가 이미 시작됐으면 하지 않는다",
        "final_rule": "seed마다 처음 나온 유효 결과(harmless_pass 또는 harmless_fail)가 최종이다. 그 seed의 시도 2·이어 하기는 모두 거부한다",
    },
    "seed_outcomes": {
        "harmless_pass": "유효 조건을 모두 채우고 p0c healthy이며 여유 기준 안",
        "harmless_fail": "유효 조건을 모두 채우고 p0c unhealthy이거나 여유 기준 밖",
        "invalid": "아래 유효 조건 중 하나라도 어긋남(판정이 undetermined인 경우 포함). 무해 여부를 판정하지 않는다",
    },
    "overall_outcomes": {
        "pass": "두 seed의 최종 결과가 모두 harmless_pass",
        "fail": "어느 seed든 최종 결과가 harmless_fail",
        "invalid": "fail이 없고, 모든 seed가 최종이거나 더 할 단계가 없으며, 어느 seed가 유효 결과 없이 끝남",
        "undecided": "위 셋이 아니면 요약을 쓰지 않는다",
    },
    "validity_checks": {
        "pipeline_completed": "실행기 단계(학습·추출·params 대조·평가 전 재확인·고정 평가·사후 확인)가 예외·시간 초과·소스 변경 없이 끝남",
        "training_exit_zero": "학습 프로세스 종료 코드 0",
        "final_checkpoint_present": "실행 폴더에 model_299.pt",
        "injection_readback": "학습 stdout의 [d3_changes] 줄 하나: 무해 7개 id, 코드 교체 유지·결합, env 설정 값이 모두 맞음",
        "params_overrides_applied": "params에서 override 키마다 기준 실행 값이 등록 기준값이고 이번 실행 값이 등록 새 값",
        "params_otherwise_match": "그 밖의 params 차이는 agent run_name뿐",
        "reference_walks": "기준 정책이 고정 평가에서 걷는다(정지 대비 오차 비 0.5 미만, 낙상 0.05 이하)",
        "fixed_report_valid": "고정 평가 보고서가 결합한 fixed_eval_v1 계약 SHA(기준 보고서와 같음)·체크포인트 SHA·action scale을 통과",
        "runtime_matches_reference": "고정 평가 런타임 버전이 기준 보고서와 같다",
        "verdict_determined": "p0c 밴드 판정이 healthy 또는 unhealthy다",
    },
    "invalid_causes": {
        "rule": "validity_checks 순서에서 처음 어긋난 검사의 원인 하나를 invalid_cause로 적고, 어긋난 검사 전체는 failed_checks에 남긴다. "
                "학습 뒤 단계(추출 이후)에서 실행기가 멈췄어도 학습 3검사와, params를 읽었으면 params 2검사 중 어긋난 것이 있으면 "
                "그 원인을 적는다",
        "by_check": {"pipeline_completed": "infrastructure", "training_exit_zero": "training_process",
                     "final_checkpoint_present": "training_process", "injection_readback": "injection",
                     "params_overrides_applied": "injection", "params_otherwise_match": "comparability",
                     "reference_walks": "comparability", "fixed_report_valid": "evaluation",
                     "runtime_matches_reference": "comparability", "verdict_determined": "evaluation"},
        "change_induced_crash": "변경 때문에 학습 프로세스가 죽거나(종료 코드≠0) 체크포인트가 없어도 training_process로 invalid다. "
                                "harmless_fail로 세지 않지만 목록은 확정되지 않는다. 학습 단계 watchdog 초과는 원인과 관계없이 "
                                "infrastructure로 기록된다",
        "judge_failure": "판정 계산 자체가 예외로 끝나면 실패 단계 judge, invalid_cause runner로 남긴다. 실행기 결함이라 재실행·이어 하기 대상이 아니다",
    },
    "prediction": {"seed_outcome": "harmless_pass", "overall": "pass",
                   "reason": "코드 변경 4개는 같은 수식이거나(H1·S4) 보행 체공 시간보다 훨씬 긴 경우에만 걸리거나(H3) 학습 값과 "
                             "무관하고(H6), 설정 변경 3개는 Isaac Lab 다른 로봇 설정에 문서화된 범위 안의 작은 수치 변화다. v1에서 "
                             "선속도 오차 비를 1.05 위로 민 것으로 보이는 보상 가중치 변경(H2)은 뺐다[추론]"},
    "failure_rule": "fail·invalid가 나와도 이 결과를 보고 여유 기준·변경 수치·seed를 바꾸지 않는다. 7개를 함께 적용한 smoke는 원인 "
                    "변경을 가르지 못하므로, 후속은 그 이유를 적은 새 프로토콜 버전으로 사전등록한다. 시도·최종 표식은 프로토콜 SHA별이다",
    "next_if_fail": {
        "v3": "범주별 분리 smoke를 seed 2026 하나로 사전등록한다. 묶음 A(reward 코드 H1·S4·H3), B(설정 H4·H5·H7), C(로깅 H6)를 "
              "따로 학습해 같은 여유 기준으로 본다(학습 3회, 기준은 baseline_p0c_s2026)",
        "v4": "여유 기준을 넘은 묶음의 변경을 남은 예비(S1·S2·S3)나 새 후보로 바꾼 목록으로 두 seed smoke를 다시 한다",
        "no_single_group": "어느 묶음도 단독으로 넘지 않으면(조합 효과) 무해 변경 수(k)를 줄이는 것은 설계 변경이므로 메인·사용자 결정으로 넘긴다",
        "invalid": "invalid는 실행 규칙(시도 2·이어 하기)대로만 다시 하고, 끝내 유효 결과가 없으면 같은 목록으로 새 판을 만든다"},
    "used_for": "D3 무해 목록 확정 여부만. 진단 성능·교정·방법 비교 수치로 쓰지 않는다",
    "gpu_estimate": {
        "unit": "부모 프로세스 벽시계(초). GPU 커널 시간이나 절감량이 아니다",
        "train_s": [492.6, 873.6],
        "train_basis": "같은 4096 env × 300회 학습의 공개 기록: 무해성 smoke v1 492.6·621.0 s(evals/results/d3_harmless_smoke_20261014_s*), "
                       "종단 시연 교정 재학습 508.8~541.1 s(evals/results/e2e_*_20261013 train_parent_wall_s), U1 smoke 590.8 s. "
                       "상한 873.6 s는 GPU 경합 구간 기록(run_upstream_smoke gpu_estimate의 ref_e01_s2028)",
        "telemetry_extract_s": [1.2, 60],
        "fixed_eval_s": [48.2, 67.5],
        "fixed_eval_basis": "무해성 smoke v1 48.2·52.4 s, 시연 교정 평가 48.4~54.9 s, U1 smoke 67.5 s",
        "per_seed_s": [542, 1001],
        "total_s": [1084, 2002],
    },
    "wording": {
        "pass": "무해 변경 7개를 모두 적용한 학습이 이미 본 seed 2026·2027에서 같은 seed 정상 학습 대비 p0c 밴드와 여유 기준"
                "(선속도·yaw 오차 비 1.05 이하, 낙상 0.02 이하) 안이었다. 7개를 함께 적용한 seed 두 개의 결과이며, 변경 하나하나나 "
                "새 seed에서의 무해성을 보장하지 않는다.",
        "fail": "무해 변경 7개를 모두 적용한 학습이 적어도 한 seed에서 p0c 밴드나 여유 기준을 벗어났다. 어느 변경 때문인지는 이 smoke로 "
                "가르지 않으며, 무해 목록은 확정하지 않는다.",
        "invalid": "smoke가 사전등록한 유효 조건을 채우지 못해 무해 목록 확정 여부를 판정하지 않는다.",
    },
}
DYNAMIC = {"source_files", "isaaclab_sources", "reference_inputs", "rendered_diffs"}


# --- 입력 결합 ---
def isaaclab_sources(isaaclab: Path) -> dict:
    return {relative: sha(Path(isaaclab) / relative, True) for relative in ISAACLAB_FILES}


def rendered_diffs(isaaclab: Path) -> dict:
    """모든 변경(무해·예비·해로운)이 렌더링되고 canary가 0건이어야 한다. diff 원문 대신 SHA만 남긴다."""
    out = {}
    for cid in D.CHANGES:
        diff = R.render(cid, isaaclab)
        clues = R.canary(diff)
        if clues:
            raise ValueError(f"{cid} diff에 단서가 있다: {clues}")
        out[cid] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return out


def reference_paths(isaaclab: Path, seed: int) -> dict[str, Path]:
    """P0-C 기준 실행의 meta·체크포인트·params·고정 평가. meta가 등록 task·seed·예산과 다르면 거부한다."""
    name = reference_run(seed)
    meta_path = ROOT / "bench/private" / f"{name}.meta.json"
    meta = read(meta_path)
    expected = {"task": TASK, "seed": seed, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS,
                "fault": "NONE", "train_exit_code": 0}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"기준 실행 {name} meta가 등록값과 다르다")
    log_root = (Path(isaaclab) / "logs/rsl_rl/unitree_go2_flat").resolve()
    run_dir = Path(meta["run_dir"]).resolve()
    if run_dir.parent != log_root or not run_dir.name.endswith("_" + name):
        raise ValueError(f"기준 실행 {name} 폴더가 등록 로그 폴더 밖이다")
    return {"metadata": meta_path, "run_dir": run_dir, "checkpoint": run_dir / FINAL_CHECKPOINT,
            "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml",
            "fixed_report": P0C / "runs" / f"{name}.json", "summary": P0C / "summary.json"}


def reference_inputs(isaaclab: Path) -> dict:
    out = {}
    for seed in SEEDS:
        paths = reference_paths(isaaclab, seed)
        summary, report = read(paths["summary"]), read(paths["fixed_report"])
        if len([row for row in summary["runs"] if row["run"] == reference_run(seed)]) != 1:
            raise ValueError(f"P0-C 요약에서 {reference_run(seed)} 행을 하나로 찾지 못했다")
        if report["protocol_sha256_lf"] != sha(FIXED_EVAL_PROTOCOL, True):
            raise ValueError("기준 고정 평가의 프로토콜 SHA가 현재 fixed_eval_v1과 다르다")
        params = load_params(paths["run_dir"])
        wrong = {key: params.get(key, "<없음>") for key, (old, _new) in D.expected_params(CHANGES).items()
                 if params.get(key) != old}
        if wrong:
            raise ValueError(f"기준 실행 params의 override 키 값이 카탈로그 기준값과 다르다: {wrong}")
        out[str(seed)] = {
            "run": reference_run(seed), "bands": summary["bands"], "checkpoint_sha256": sha(paths["checkpoint"]),
            "fixed_eval_protocol_sha256_lf": report["protocol_sha256_lf"],
            "hashes": {key: sha(paths[key], key in ("fixed_report", "summary"))
                       for key in ("metadata", "environment", "agent", "fixed_report", "summary")},
            "runtime_versions": report["versions"], "walks": walks(report), "standstill_ratio": standstill_ratio(report)}
    return out


def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": isaaclab_sources(isaaclab), "reference_inputs": reference_inputs(isaaclab),
            "rendered_diffs": rendered_diffs(isaaclab)}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in json.loads(json.dumps(FIXED)).items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab 소스·기준 입력·diff가 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_execute", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    try:
        commit = verify_committed(ROOT, set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix()})
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 프로토콜이 커밋되지 않았다") from error
    return protocol, commit


# --- 판정(GPU 없이 시험한다) ---
def params_differences(run: dict, reference: dict) -> dict[str, list | str]:
    """다른 평면 키 → [기준 값, 이번 값]. 공개 결과에 쓰므로 override 키·run_name 밖의 값은 남기지 않는다."""
    shown = set(D.expected_params(CHANGES)) | {RUN_NAME_ALLOWED_DIFF}
    return {key: [reference.get(key), run.get(key)] if key in shown else "값 생략"
            for key in sorted(set(run) | set(reference)) if run.get(key) != reference.get(key)}


def parse_readback(log_text: str) -> dict | None:
    lines = [line for line in log_text.splitlines() if line.startswith(D.READBACK_PREFIX)]
    if len(lines) != 1:
        return None
    try:
        return json.loads(lines[0][len(D.READBACK_PREFIX):])
    except json.JSONDecodeError:
        return None


def readback_ok(readback: dict | None) -> bool:
    if not isinstance(readback, dict) or readback.get("ok") is not True or readback.get("changes") != list(CHANGES):
        return False
    code = {cid: {t["qualname"] for t in D.code_targets(cid)} for cid in CHANGES if D.CHANGES[cid]["kind"] == "code"}
    config = {cid: set(D.CHANGES[cid]["overrides"]) for cid in CHANGES if D.CHANGES[cid]["kind"] == "config"}
    rows, values = readback.get("code") or {}, readback.get("config") or {}
    return ({cid: set(r) for cid, r in rows.items()} == code
            and all(row.get("swapped") is True and row.get("bound") is True for r in rows.values() for row in r.values())
            and {cid: set(v) for cid, v in values.items()} == config
            and all(value is True if key.startswith("env.") else value == "params"
                    for v in values.values() for key, value in v.items()))


def params_checks(differences: dict | None) -> dict[str, bool]:
    if differences is None:
        return dict.fromkeys(PARAMS_CHECKS, False)
    expected = D.expected_params(CHANGES)
    return {"params_overrides_applied": all(differences.get(key) == list(pair) for key, pair in expected.items()),
            "params_otherwise_match": set(differences) - set(expected) <= {RUN_NAME_ALLOWED_DIFF}}


def training_checks(evidence: dict) -> dict[str, bool]:
    """학습 산출물만으로 정해지는 유효 조건. 하나라도 어긋나면 고정 평가를 하지 않는다."""
    return {"training_exit_zero": evidence.get("train_exit_code") == 0,
            "final_checkpoint_present": bool(evidence.get("checkpoint_sha256")),
            "injection_readback": readback_ok(evidence.get("readback")),
            **params_checks(evidence.get("params_differences"))}


def margins(comparison: dict) -> dict[str, bool]:
    return {"err_xy_ratio": comparison["err_xy_ratio"] <= MARGINS["err_xy_ratio_max"],
            "err_yaw_ratio": comparison["err_yaw_ratio"] <= MARGINS["err_yaw_ratio_max"],
            "fall_frac": comparison["fall_frac"] <= MARGINS["fall_frac_max"]}


def judge(protocol: dict, evidence: dict) -> dict:
    """seed 하나의 사전등록 유효 조건·p0c 밴드·여유 기준 판정. evidence는 실행에서 모은 원자료다."""
    reference = protocol["reference_inputs"][str(evidence["seed"])]
    checks = {"pipeline_completed": evidence.get("failure") is None, **training_checks(evidence),
              "reference_walks": reference["walks"] is True,
              "fixed_report_valid": False, "runtime_matches_reference": False, "verdict_determined": None}
    checks = {name: checks[name] for name in protocol["validity_checks"]}
    report = evidence.get("report")
    if report is not None and checks["final_checkpoint_present"]:
        try:
            C.validate_fixed_report(report, {"name": evidence["run_name"], "checkpoint_file": FINAL_CHECKPOINT,
                                             "checkpoint_sha256": evidence["checkpoint_sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": reference["fixed_eval_protocol_sha256_lf"]})
            checks["fixed_report_valid"] = True
            checks["runtime_matches_reference"] = report.get("versions") == reference["runtime_versions"]
        except (ValueError, KeyError, TypeError):
            pass
    result = {"verdict": None, "behavior": None, "comparison": None, "margins": None}
    if all(ok for name, ok in checks.items() if name != "verdict_determined"):
        behavior = C._behavior(report)
        comparison = RC.compare(behavior, C._behavior(evidence["reference_report"]))
        verdict = RC.verdict(comparison, reference["bands"])
        checks["verdict_determined"] = verdict["label"] in ("healthy", "unhealthy")
        result.update(verdict=verdict, behavior=behavior, comparison=comparison, margins=margins(comparison),
                      standstill_ratio=standstill_ratio(report))
    failed_checks = [name for name, ok in checks.items() if ok is False]
    cause = None
    if failed_checks:
        outcome = "invalid"
        cause = protocol["invalid_causes"]["by_check"][failed_checks[0]]
        stage = (evidence.get("failure") or {}).get("stage")
        priority = [*TRAINING_CHECKS, *(PARAMS_CHECKS if evidence.get("params_differences") is not None else ())]
        failed_priority = [name for name in priority if name in failed_checks]
        if failed_checks[0] == "pipeline_completed" and stage == "judge":
            cause = "runner"
        elif failed_checks[0] == "pipeline_completed" and stage != "training" and failed_priority:
            cause = protocol["invalid_causes"]["by_check"][failed_priority[0]]  # 학습 뒤 단계 실패보다 학습 산출물 검사가 앞선다
    else:
        passed = result["verdict"]["label"] == "healthy" and all(result["margins"].values())
        outcome = "harmless_pass" if passed else "harmless_fail"
    prediction = protocol["prediction"]["seed_outcome"]
    return {"checks": checks, "failed_checks": failed_checks, **result, "outcome": outcome, "invalid_cause": cause,
            "prediction": prediction, "prediction_match": None if outcome == "invalid" else outcome == prediction}


# --- 실행(GPU) ---
def train_command(isaaclab: Path, seed: int, name: str) -> list[str]:
    return [str(Path(isaaclab) / "_isaac_sim" / "python.bat"), str(ROOT / "bench" / "train_with_d3_changes.py"),
            "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(MAX_ITERATIONS),
            "--seed", str(seed), "--headless", "--run_name", name]


def telemetry_path(name: str) -> Path:
    return RUNS_DIR / f"{name}.telemetry.json"


def _admission() -> dict:
    import record_evidence_media as M
    return M.query_gpu(GPU_PROFILE)


def _process_created(pid: int) -> int | None:
    import run_runtime_readback_smoke as P
    return P._process_creation_filetime(pid)


def _wait_for_gpu(private: Path) -> None:
    started = time.monotonic()
    while True:
        gpu = _admission()
        waited = time.monotonic() - started
        if not gpu["busy"] or waited >= EVAL_GPU_WAIT_S:
            write_new(private / "gpu_before_fixed_evaluation.json", {**gpu, "waited_s": round(waited, 1)})
            if gpu["busy"]:
                raise RuntimeError(f"고정 평가 전 GPU가 {EVAL_GPU_WAIT_S}초 동안 비지 않았다: {gpu['busy_reasons']}")
            return
        time.sleep(EVAL_GPU_POLL_S)


def _check_tag(tag, pattern: str = TAG_RE) -> None:
    if not isinstance(tag, str) or not re.fullmatch(pattern, tag):
        raise ValueError(f"tag 형식이 {pattern}가 아니다")


def _consumed(seed: int, kind: str) -> Path:
    return PRIVATE / "consumed" / f"{sha(PROTOCOL, True)[:16]}_s{seed}_{kind}.json"


def _refuse_existing(tag: str, *paths: Path) -> None:
    if (RESULTS / tag).exists() or (PRIVATE / tag).exists() or any(path.exists() for path in paths):
        raise FileExistsError("같은 tag·시도·실행 이름의 산출물이 이미 있다. 덮어쓰지 않는다")


def _context(tag: str, isaaclab: Path, protocol: dict, commit: str, seed: int, attempt: int, **extra) -> dict:
    name = run_name(seed, attempt)
    return {"tag": tag, "isaaclab": isaaclab, "protocol": protocol, "commit": commit, "seed": seed,
            "protocol_sha256_lf": sha(PROTOCOL, True), "attempt": attempt, "run_name": name,
            "out": RESULTS / tag, "private": PRIVATE / tag, "run_dir": None, "stage": "training", "failure": None,
            "executions": {}, "evidence": {"run_name": name, "seed": seed},
            "walls": {"train_s": None, "telemetry_extract_s": None, "fixed_eval_s": None}, "commits_seen": [commit],
            "training_commit": commit, "retry_of": None, "resumed_from": None, **extra}


def bound_sources_same(start: str, now: str) -> bool:
    """두 커밋 사이에 결합 경로(CODE·프로토콜)의 diff가 없는가. git 오류는 다르다고 본다."""
    paths = [*CODE, PROTOCOL.relative_to(ROOT).as_posix()]
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", start, now, "--", *paths],
                          capture_output=True).returncode == 0


def _reverify(ctx: dict) -> None:
    """결합 경로만 본다. 같은 checkout의 다른 작업이 결합 밖 파일을 커밋해도 실행을 무효로 하지 않는다."""
    protocol, commit = verify(ctx["isaaclab"])
    if protocol != ctx["protocol"]:
        raise ValueError("실행 중 프로토콜이나 결합 소스가 바뀌었다")
    if commit != ctx["commit"] and not bound_sources_same(ctx["commit"], commit):
        raise ValueError("시작 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")
    if commit not in ctx["commits_seen"]:
        ctx["commits_seen"].append(commit)


def _train(ctx: dict) -> None:
    isaaclab, evidence = ctx["isaaclab"], ctx["evidence"]
    os.environ[D.ENV_VAR] = SPEC
    try:
        execution = _owned(ctx["private"] / "train", train_command(isaaclab, ctx["seed"], ctx["run_name"]), isaaclab,
                           WATCHDOG_S["training"])
    finally:
        os.environ.pop(D.ENV_VAR, None)
    ctx["executions"]["training"] = execution
    ctx["walls"]["train_s"] = round(execution["wall_s"], 1)
    evidence["train_exit_code"] = execution["exit_code"]
    evidence["readback"] = parse_readback((ctx["private"] / "train/stdout.log").read_text(encoding="utf-8", errors="replace"))
    run_dirs = find_run_dir(isaaclab, ctx["run_name"])
    ctx["run_dir"] = run_dirs[0] if len(run_dirs) == 1 else None
    checkpoint = ctx["run_dir"] / FINAL_CHECKPOINT if ctx["run_dir"] else None
    evidence["checkpoint_sha256"] = sha(checkpoint) if checkpoint and checkpoint.is_file() else None
    if not finished(execution):
        raise RuntimeError("학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")


def _after_training(ctx: dict) -> None:
    """추출·params 대조·평가 전 재확인·고정 평가·사후 확인. ctx["stage"]에 진행 단계를 남기고 예외는 그대로 올린다."""
    isaaclab, evidence, run_dir, private = ctx["isaaclab"], ctx["evidence"], ctx["run_dir"], ctx["private"]
    if run_dir is None:
        return  # 체크포인트 검사에서 invalid
    ctx["stage"] = "telemetry_extract"
    telemetry = telemetry_path(ctx["run_name"])
    if not telemetry.exists():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        execution = _owned(private / "extract", [str(Path(isaaclab) / "_isaac_sim" / "python.bat"),
                                                 str(ROOT / "bench" / "extract_tb.py"), str(run_dir), str(telemetry)],
                           isaaclab, WATCHDOG_S["telemetry_extract"])
        ctx["executions"]["telemetry_extract"] = execution
        ctx["walls"]["telemetry_extract_s"] = round(execution["wall_s"], 1)
        # 추출은 성공해도 0이 아닌 종료 코드를 남길 때가 있어 결과 파일로 판정한다(run_case.ps1과 같다).
        if not finished(execution) or not telemetry.is_file():
            raise RuntimeError("텔레메트리 추출이 끝나지 않았거나 파일이 없다")
    evidence["telemetry"] = read(telemetry)
    evidence["telemetry_sha256"] = sha(telemetry)
    ctx["stage"] = "params"
    evidence["params_differences"] = params_differences(
        load_params(run_dir), load_params(reference_paths(isaaclab, ctx["seed"])["run_dir"]))
    if not all(training_checks(evidence).values()):
        return  # 판정에 쓰지 않을 고정 평가는 하지 않는다
    ctx["stage"] = "pre_evaluation"
    _reverify(ctx)
    _wait_for_gpu(private)
    ctx["stage"] = "fixed_evaluation"
    report_path = ctx["out"] / "runs" / f"{ctx['run_name']}.json"
    jobs = private / "eval" / "jobs.json"
    write_new(jobs, {"action_scale": ACTION_SCALE, "jobs": [{"name": ctx["run_name"],
                                                             "checkpoint": str(run_dir / FINAL_CHECKPOINT),
                                                             "output": str(report_path)}]})
    execution = _owned(private / "eval", [str(Path(isaaclab) / "_isaac_sim" / "python.bat"),
                                          str(ROOT / "evals" / "fixed_eval.py"), "--protocol", str(FIXED_EVAL_PROTOCOL),
                                          "--jobs", str(jobs), "--headless"],
                       isaaclab, WATCHDOG_S["fixed_evaluation"])
    ctx["executions"]["fixed_evaluation"] = execution
    ctx["walls"]["fixed_eval_s"] = round(execution["wall_s"], 1)
    if not finished(execution) or execution["exit_code"] != 0 or not report_path.is_file():
        raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
    evidence["report"] = read(report_path)
    ctx["stage"] = "postflight"
    _reverify(ctx)
    if sha(run_dir / FINAL_CHECKPOINT) != evidence["checkpoint_sha256"]:
        raise ValueError("평가 중 체크포인트가 바뀌었다")
    after = _admission()
    write_new(private / "gpu_after.json", after)
    # PID는 재사용될 수 있어 생성 시각까지 같을 때만 소유 프로세스로 본다.
    owned = {pid: created for execution in ctx["executions"].values()
             for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and _process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def _record_failure(ctx: dict, error: Exception) -> None:
    ctx["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
    write_new(ctx["private"] / "failure.json", {**ctx["failure"], "detail": str(error)})


def _finish(ctx: dict) -> dict:
    """판정하고 공개 result.json과 비공개 execution.json을 한 번씩 쓴다. 판정이 깨져도 invalid로 남긴다."""
    evidence, seed = ctx["evidence"], ctx["seed"]
    ctx["stage"] = "judge"
    try:
        evidence["reference_report"] = read(P0C / "runs" / f"{reference_run(seed)}.json")
        verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
    except Exception as error:
        if ctx["failure"] is None:
            _record_failure(ctx, error)
        else:
            write_new(ctx["private"] / "judge_error.json", {"error_type": type(error).__name__, "detail": str(error)})
        # 실패가 기록되면 판정은 기준 보고서·밴드 계산 없이 invalid로 끝난다.
        try:
            verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
        except Exception:
            verdict = {"checks": None, "failed_checks": ["pipeline_completed"], "verdict": None, "behavior": None,
                       "comparison": None, "margins": None, "outcome": "invalid", "invalid_cause": "runner",
                       "prediction": FIXED["prediction"]["seed_outcome"], "prediction_match": None}
    write_new(ctx["private"] / "execution.json", {
        "run_dir": str(ctx["run_dir"]) if ctx["run_dir"] else None, "seed": seed, "attempt": ctx["attempt"],
        "train_command": train_command(ctx["isaaclab"], seed, ctx["run_name"]), "failure": ctx["failure"],
        "retry_of": ctx["retry_of"], "resumed_from": ctx["resumed_from"], "commits_seen": ctx["commits_seen"]})
    result = {"schema": "d3_harmless_smoke_result_v1", "tag": ctx["tag"], "seed": seed, "changes": list(CHANGES),
              "attempt": ctx["attempt"], "run_name": ctx["run_name"], "retry_of": ctx["retry_of"],
              "resumed_from": ctx["resumed_from"], "source_commit_sha": ctx["commit"],
              "training_source_commit_sha": ctx["training_commit"], "commits_seen": ctx["commits_seen"],
              "protocol_sha256_lf": ctx["protocol_sha256_lf"], "reference_run": reference_run(seed),
              **verdict, "failure": ctx["failure"], "train_exit_code": evidence.get("train_exit_code"),
              "params_differences": evidence.get("params_differences"), "readback": evidence.get("readback"),
              "telemetry_sha256": evidence.get("telemetry_sha256"),
              "telemetry_summary": S.telemetry_summary(evidence.get("telemetry")),
              "checkpoint_sha256": evidence.get("checkpoint_sha256"), "parent_wall_s": ctx["walls"],
              "parent_wall_scope": FIXED["gpu_estimate"]["unit"], "scope": FIXED["scope"], "model_calls": 0}
    if result["outcome"] != "invalid":
        # execution.final_rule. 결과보다 먼저 써서, 그 사이에 멈춰도 다음 시도가 열리지 않게 한다.
        write_new(_consumed(seed, "final"), {"tag": ctx["tag"], "outcome": result["outcome"]})
    write_new(ctx["out"] / "result.json", result)
    return {"seed": seed, "outcome": result["outcome"], "invalid_cause": result["invalid_cause"],
            "failure": result["failure"], "failed_checks": result["failed_checks"],
            "result": f"evals/results/{ctx['tag']}/result.json"}


def _previous(tag: str) -> tuple[dict, dict]:
    _check_tag(tag)
    return read(RESULTS / tag / "result.json"), read(PRIVATE / tag / "execution.json")


def _last_result(seed: int, attempt: int) -> tuple[str | None, dict | None]:
    """시도의 마지막 tag와 결과. 이어 하기를 했으면 그 결과이고, 결과 파일이 아직 없으면(실행 중·중단) None이다."""
    resumed, started = _consumed(seed, f"resume{attempt}"), _consumed(seed, f"attempt{attempt}")
    marker = resumed if resumed.exists() else started if started.exists() else None
    if marker is None:
        return None, None
    tag = read(marker)["tag"]
    path = RESULTS / tag / "result.json"
    return tag, read(path) if path.is_file() else None


def _resumable(result: dict) -> bool:
    failure = result.get("failure") or {}
    return (result.get("protocol_sha256_lf") == sha(PROTOCOL, True) and result.get("outcome") == "invalid"
            and result.get("invalid_cause") == "infrastructure" and failure.get("stage") in RESUMABLE_STAGES
            and result.get("resumed_from") is None
            and all((result.get("checks") or {}).get(name) is True for name in TRAINING_CHECKS))


def execute(tag: str, seed: int, isaaclab: Path, retry_of: str | None = None) -> dict:
    _check_tag(tag)
    if seed not in SEEDS:
        raise ValueError(f"seed는 {SEEDS} 중 하나다")
    protocol, commit = verify(isaaclab)
    attempt = 1
    if retry_of is not None:
        _check_tag(retry_of)
        last_tag, last = _last_result(seed, 1)
        if (retry_of != last_tag or last is None or last.get("outcome") != "invalid"
                or last.get("invalid_cause") != "infrastructure"):
            raise ValueError("시도 2는 그 seed 시도 1의 마지막 결과(이어 하기를 했으면 그 결과)가 인프라 원인 invalid로 끝났을 때 "
                             "그 tag로만 허용된다")
        attempt = 2
    name = run_name(seed, attempt)
    consumed = _consumed(seed, f"attempt{attempt}")
    _refuse_existing(tag, _consumed(seed, "final"), consumed, telemetry_path(name), *find_run_dir(isaaclab, name))
    gpu = _admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    ctx = _context(tag, isaaclab, protocol, commit, seed, attempt, retry_of=retry_of)
    write_new(consumed, {"tag": tag, "seed": seed, "attempt": attempt, "run_name": name})
    write_new(ctx["private"] / "gpu_before_training.json", gpu)
    try:
        _train(ctx)
        _after_training(ctx)
    except Exception as error:
        _record_failure(ctx, error)
    return _finish(ctx)


def resume(tag: str, from_tag: str, isaaclab: Path) -> dict:
    """앞 실행의 학습 산출물을 그대로 두고 추출·평가·판정만 새 tag로 다시 한다(execution.resume_rule)."""
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    previous, execution = _previous(from_tag)
    if not _resumable(previous):
        raise ValueError("이어 하기는 학습 측 검사를 통과한 뒤 추출·평가 단계의 인프라 원인으로 invalid가 된 결과에 한 번만 허용된다")
    seed, attempt = previous["seed"], previous["attempt"]
    consumed = _consumed(seed, f"resume{attempt}")
    later = [_consumed(seed, f"attempt{n}") for n in (1, 2) if n > attempt]  # 다음 시도가 시작됐으면 이어 하지 않는다
    _refuse_existing(tag, _consumed(seed, "final"), consumed, *later)
    ctx = _context(tag, isaaclab, protocol, commit, seed, attempt, resumed_from=from_tag,
                   training_commit=previous["training_source_commit_sha"])
    run_dir = Path(execution["run_dir"])
    if find_run_dir(isaaclab, ctx["run_name"]) != [run_dir] or sha(run_dir / FINAL_CHECKPOINT) != previous["checkpoint_sha256"]:
        raise ValueError("앞 실행의 실행 폴더나 체크포인트가 그대로가 아니다")
    telemetry = telemetry_path(ctx["run_name"])
    if telemetry.exists() and sha(telemetry) != previous.get("telemetry_sha256"):
        raise ValueError("남아 있는 텔레메트리가 앞 결과에 기록된 것과 다르다. 내용을 확인한 뒤 처리한다")
    gpu = _admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    previous_private = PRIVATE / from_tag
    ctx["run_dir"] = run_dir
    ctx["walls"]["train_s"] = previous["parent_wall_s"]["train_s"]
    # 앞 학습의 PID는 이미 다른 프로세스가 쓸 수 있어 GPU 잔존 검사(ctx["executions"])에 넣지 않는다.
    ctx["evidence"].update(
        train_exit_code=read(previous_private / "train/execution.json")["exit_code"],
        checkpoint_sha256=previous["checkpoint_sha256"],
        readback=parse_readback((previous_private / "train/stdout.log").read_text(encoding="utf-8", errors="replace")))
    write_new(consumed, {"tag": tag, "seed": seed, "resumed_from": from_tag, "attempt": attempt})
    write_new(ctx["private"] / "gpu_before_resume.json", gpu)
    try:
        _after_training(ctx)
    except Exception as error:
        _record_failure(ctx, error)
    return _finish(ctx)


# --- 두 seed 종합 ---
def seed_state(seed: int) -> dict:
    """seed의 최종 결과, 없으면 마지막 결과와 남은 단계(retry·resume)."""
    final = _consumed(seed, "final")
    if final.exists():
        tag = read(final)["tag"]
        return {"seed": seed, "final": True, "tag": tag, "result": read(RESULTS / tag / "result.json"), "next": []}
    attempt = 2 if _consumed(seed, "attempt2").exists() else 1
    tag, last = _last_result(seed, attempt)
    if tag is None:
        return {"seed": seed, "final": False, "tag": None, "result": None, "next": ["execute"]}
    if last is None:
        return {"seed": seed, "final": False, "tag": tag, "result": None, "next": ["running_or_stopped"]}
    # 마지막 결과가 이어 하기 결과면 _resumable이 거짓이다. 시도 2가 시작됐으면 attempt가 2다.
    steps = ["resume"] if _resumable(last) else []
    if attempt == 1 and last.get("invalid_cause") == "infrastructure":
        steps.append("retry")
    return {"seed": seed, "final": False, "tag": tag, "result": last, "next": steps}


def overall(states: list[dict]) -> str | None:
    """fail은 한 seed로 정해진다. invalid는 모든 seed가 최종이거나 더 할 단계가 없을 때만이다(남은 seed가 fail일 수 있다)."""
    finals = [s["result"]["outcome"] for s in states if s["final"]]
    if "harmless_fail" in finals:
        return "fail"
    if len(finals) == len(states):
        return "pass"
    if all(s["final"] or (s["result"] is not None and not s["next"]) for s in states):
        return "invalid"
    return None


def summarize(tag: str) -> dict:
    _check_tag(tag, SUMMARY_TAG_RE)
    states = [seed_state(seed) for seed in SEEDS]
    outcome = overall(states)
    if outcome is None:
        raise ValueError(f"아직 결정되지 않았다: { {s['seed']: s['next'] for s in states} }")
    marker = PRIVATE / "consumed" / f"{sha(PROTOCOL, True)[:16]}_summary.json"  # 프로토콜마다 공개 요약은 하나
    _refuse_existing(tag, marker)
    write_new(marker, {"tag": tag, "outcome": outcome})
    seeds = {str(s["seed"]): None if s["result"] is None else {
        "tag": s["tag"], "final": s["final"], "outcome": s["result"]["outcome"],
        "invalid_cause": s["result"]["invalid_cause"], "comparison": s["result"].get("comparison"),
        "margins": s["result"].get("margins"), "verdict": (s["result"].get("verdict") or {}).get("label"),
        "checkpoint_sha256": s["result"].get("checkpoint_sha256"), "parent_wall_s": s["result"]["parent_wall_s"]}
        for s in states}
    summary = {"schema": "d3_harmless_smoke_summary_v1", "tag": tag, "protocol_sha256_lf": sha(PROTOCOL, True),
               "changes": list(CHANGES), "seeds": seeds, "outcome": outcome,
               "prediction": FIXED["prediction"]["overall"], "prediction_match": outcome == FIXED["prediction"]["overall"],
               "wording": FIXED["wording"][outcome], "margins": MARGINS, "scope": FIXED["scope"], "model_calls": 0}
    write_new(RESULTS / tag / "summary.json", summary)
    return {"outcome": outcome, "summary": f"evals/results/{tag}/summary.json"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "execute", "resume", "summarize"))
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--retry-of", dest="retry_of")
    parser.add_argument("--from", dest="from_tag")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command in ("execute", "resume", "summarize") and not args.tag:
        parser.error(f"{args.command}에는 --tag가 필요하다")
    if args.command == "execute" and args.seed is None:
        parser.error("execute에는 --seed가 필요하다")
    if args.command == "resume" and not args.from_tag:
        parser.error("resume에는 --from이 필요하다")
    import record_evidence_media as M
    try:
        if args.command == "prepare":
            out = prepare(args.isaaclab)
        elif args.command == "verify":
            out = {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]}
        elif args.command == "execute":
            out = execute(args.tag, args.seed, args.isaaclab, args.retry_of)
        elif args.command == "resume":
            out = resume(args.tag, args.from_tag, args.isaaclab)
        else:
            out = summarize(args.tag)
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("status") == "gpu_busy_no_launch" else 0


if __name__ == "__main__":
    sys.exit(main())
