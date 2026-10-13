"""N2 반복 일관성(pass^k)과 비용: 이미 본 결정 packet을 단일 호출로 같은 횟수 반복하고, 공개된 NAT v4 반복 기록과
결정적 기준선을 같은 표에 둔다. 사전등록·실행 계획·실행·요약을 한 파일에서 한다.

질문은 둘이다. 같은 입력을 반복하면 모델 결정이 매번 같은가, 그 비용(토큰·벽시계)이 결정적 기준선 대비 얼마인가.
- 단일 호출: evals/active_model_call.py(진단)·evals/correction_model_call.py(교정)를 packet마다 5회 부른다. 새 측정이며
  이 프로토콜을 커밋한 뒤에만 실행한다(WSL, `. ~/.config/nvidia/env`를 source한다. 키 값은 읽거나 출력하지 않는다).
- NAT v4: 새 호출을 하지 않는다. E4b v4가 같은 모델·temperature 0으로 같은 packet을 5회씩 돌린 공개 요약
  (evals/results/e4b_nat_compare_v4/summary.json)의 실행별 결정·비용을 읽는다. 사전등록 전에 본 기록이다.
- 결정적 기준선: 동결 P0-A2 규칙(빈도순·특징 규칙·v1 최근접 사례), 저장된 probe 결과가 있는 사례의 전수 probe,
  교정의 등록 회복 방식 규칙. 같은 입력이면 같은 결과이므로 pass^k는 1이고 비용은 이 기기의 계산 시간이다.
- 입력은 모두 이미 본 packet이다(시연 seed 2030·2031·2033·2034·2040의 진단·교정 packet, E4b 개발 입력 h02_s2027 진단
  packet). 개발 측정이며 정확도·우월성·일반화·절감의 근거가 아니다.

  python evals/n2_pass_k.py prepare|verify|plan|run|summarize
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

CODE_ROOT = ROOT  # 코드 파일 위치. 입력·결과 위치(ROOT)와 따로 둬 테스트가 ROOT만 바꿀 수 있다
PROTOCOL = ROOT / "bench/protocols/n2_pass_k_v1.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/n2_pass_k"
RESULTS = ROOT / "evals/results"
RESULT_TAG = "n2_pass_k_v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
KINDS = ("diagnosis", "correction")
KIND_KO = {"diagnosis": "진단", "correction": "교정"}
RUNS = 5
K_VALUES = (1, 2, 4, 5)
API_CAP = 80  # 단일 호출 요청 상한(인프라 재시도 포함). 계획은 55회
MAX_ATTEMPTS = 2  # 인프라 오류일 때만 같은 run 번호로 한 번 더
CALL_TIMEOUT_S = 420
USE_THRESHOLD = 0.8  # 경로·종류별 평균 pass^4(결정 일치) 하한
SCRIPTS = {"diagnosis": ("evals/active_model_call.py", "--packet"), "correction": ("evals/correction_model_call.py", "--evidence")}
E4B_SUMMARY = "evals/results/e4b_nat_compare_v4/summary.json"
DEV_TELEMETRY = "bench/runs"  # v1 dev(seed 7·42) 최근접 사례 학습 입력
P0C_REFERENCE_PARAMS = "bench/reference/params/p0c_seed2027/env.yaml"  # P0-C 학습 설정(seed 말고 같다)

DEMO = "bench/private/{dir}/model"
INPUT_SETS = {
    "h02_s2027": {"case": "h02_s2027", "seed": 2027, "mechanism": "reward", "kinds": ["diagnosis"], "role": "e4b_v4_development",
                  "packets": {"diagnosis": "bench/private/e4b_v4/inputs/h02_s2027/diagnosis.packet.json"},
                  "recorded": {}, "nat_repeats": {"diagnosis": "h02_s2027"}, "fault_report": None,
                  "probes": "evals/results/p1a_probes_holdout_20261009/probes"},
    "f2_s2031": {"case": "h06_s2031", "seed": 2031, "mechanism": "actuator", "kinds": list(KINDS), "role": "demo_v3_f2",
                 "dir": "e2e_demo_v3/e2e_demo_v3_f2_s2031_20261013",
                 "recorded": {"diagnosis": ["single", "diagnosis.call1.json"], "correction": ["single", "correction.call1.json"]},
                 "nat_repeats": {"diagnosis": "f2_s2031", "correction": "f2_s2031"},
                 "fault_report": "evals/results/e2e_demo_v3_f2_s2031_20261013/runs/h06_s2031.json", "probes": None},
    "r_s2030": {"case": "h02_s2030", "seed": 2030, "mechanism": "reward", "kinds": list(KINDS), "role": "demo_reward_v2",
                "dir": "e2e_reward_demo/e2e_reward_s2030_20261013",
                "recorded": {"diagnosis": ["single", "diagnosis.call1.json"], "correction": ["single", "correction.call1.json"]},
                "nat_repeats": {}, "fault_report": "evals/results/e2e_reward_s2030_20261013/runs/h02_s2030.json", "probes": None},
    "f2_s2033": {"case": "h06_s2033", "seed": 2033, "mechanism": "actuator", "kinds": list(KINDS), "role": "demo_v3_1_f2",
                 "dir": "e2e_demo_v3_1/e2e_demo_v3_1_f2_s2033_20261013",
                 "recorded": {"diagnosis": ["single", "diagnosis.call1.json"], "correction": ["single", "correction.call1.json"]},
                 "nat_repeats": {}, "fault_report": "evals/results/e2e_demo_v3_1_f2_s2033_20261013/runs/h06_s2033.json",
                 "probes": None},
    "f1_s2034": {"case": "h02_s2034", "seed": 2034, "mechanism": "reward", "kinds": list(KINDS), "role": "demo_v3_1_f1_nat",
                 "dir": "e2e_demo_v3_1/e2e_demo_v3_1_f1_s2034_20261013",
                 "recorded": {"diagnosis": ["nat_v4", "diagnosis.nat1.json"], "correction": ["nat_v4", "correction.nat1.json"]},
                 "nat_repeats": {}, "fault_report": "evals/results/e2e_demo_v3_1_f1_s2034_20261013/runs/h02_s2034.json",
                 "probes": None},
    "u1_s2040": {"case": "u1_s2040", "seed": 2040, "mechanism": "reward", "kinds": list(KINDS), "role": "demo_u1_upstream",
                 "dir": "e2e_demo_u1/e2e_demo_u1_s2040_20261013",
                 "recorded": {"diagnosis": ["single", "diagnosis.call1.json"], "correction": ["single", "correction.call1.json"]},
                 "nat_repeats": {}, "fault_report": "evals/results/e2e_demo_u1_s2040_20261013/runs/u1_s2040.json", "probes": None},
}
SET_ORDER = tuple(INPUT_SETS)
CODE = {
    "evals/n2_pass_k.py", "evals/active_model_call.py", "evals/correction_model_call.py", "evals/e2e_reward_demo_v2.py",
    "evals/baselines.py", "evals/replay.py", "src/rl_triage/active_diagnosis.py", "src/rl_triage/rules.py",
    "src/rl_triage/probe_loop.py", "src/rl_triage/correction_registry.py",
    "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
FIXED = {
    "schema": "n2_pass_k_protocol_v1",
    "question": ("이미 본 결정 packet을 같은 모델·temperature 0으로 반복하면 진단·교정 결정이 매번 같은가(pass^k), 그 비용이 "
                 "결정적 기준선 대비 얼마인가"),
    "scope": "repeated_decisions_on_already_seen_packets_development_measurement_not_accuracy_superiority_generalization_or_savings",
    "model": MODEL, "temperature": 0, "reasoning_effort": "none",
    "input_sets": INPUT_SETS, "set_order": list(SET_ORDER), "runs_per_packet": RUNS, "k_values": list(K_VALUES),
    "paths": {
        "single": {"new_calls": True, "scripts": {kind: script for kind, (script, _) in SCRIPTS.items()},
                   "environment": "WSL Ubuntu, `. ~/.config/nvidia/env`를 source한 셸, ~/.venvs/rl-triage 파이썬",
                   "order": ("run 번호 1~5를 바깥 반복으로, 세트 순서(set_order)·종류(diagnosis, correction)를 안쪽 반복으로 차례로 "
                             "한 번씩 부른다(같은 packet 반복을 시간에 걸쳐 흩는다)")},
        "nat_v4": {"new_calls": False, "source": E4B_SUMMARY,
                   "note": ("E4b v4(사전등록 bench/protocols/e4b_nat_compare_v4.json)가 같은 모델·temperature 0으로 돌린 5회 기록. "
                            "이 프로토콜 전에 본 기록이다. 시연 packet의 NAT 반복은 하지 않는다(NAT 1회가 LLM 요청 7~9회라 "
                            "API 상한 안에서 단일 호출과 같은 5회를 맞출 수 없다)")},
    },
    "attempts": "run 번호마다 첫 시도. 인프라 오류(연결·시간 초과·429·5xx·빈 응답·잘림)일 때만 같은 run 번호로 1회 더",
    "api_budget": {"cap_requests": API_CAP, "planned_requests": RUNS * sum(len(s["kinds"]) for s in INPUT_SETS.values()),
                   "counting": "단일 호출 기록 파일 하나가 요청 하나다(인프라 재시도 포함). 상한에 닿으면 남은 실행은 하지 않고 "
                               "budget_exhausted로 기록한다", "nat_new_requests": 0},
    "status_definitions": {
        "valid": "진단은 공개 판단 계약(active_diagnosis.validate_decision), 교정은 packet 등록 표로 검증한 계약(accepted)을 통과",
        "invalid": "응답이 계약을 어김(JSON 파손·필드 오류·거절)",
        "call_error": "인프라가 아닌 호출 오류",
        "infra_error": "두 번째 시도까지 인프라 오류. n에서 빼고 따로 센다",
    },
    "decision_keys": {"diagnosis": "action과 ranking 1위(보조: 제안 probe까지, ranking 전체)",
                      "correction": "decision(propose·HOLD), correction_id, recovery_mode"},
    "pass_k": ("세트·종류마다 n=인프라 오류를 뺀 실행 수, c_g=결정 키가 g인 유효 실행 수일 때 pass^k(결정 일치) = Σ_g C(c_g,k)/C(n,k) "
               "(k개를 비복원으로 뽑았을 때 모두 유효하고 같은 결정일 확률, τ-bench pass^k의 일관성 판). pass^k(유효) = "
               "C(유효 수,k)/C(n,k). n<k면 계산하지 않는다. 경로·종류 요약은 세트 평균이다"),
    "metrics": ["세트·종류별 상태 수와 결정 분포", "pass^k(결정 일치·유효), k=1·2·4·5", "기록된 시연 결정과의 일치",
                "1위가 등록 결함 메커니즘과 같은 유효 실행 수(개발 측정, 정확도 근거 아님)",
                "입력·출력 토큰, 벽시계(호출 1회·유효 결정 1건당)", "결정적 기준선의 결정과 계산 시간·저장 probe GPU 초"],
    "baselines": {
        "rule_prior": "동결 P0-A2 빈도순(입력 없음)",
        "rule_features": "동결 P0-A2 특징 규칙. 사례·같은 seed NONE 텔레메트리 요약과 P0-C 학습 설정(seed 2027 params)",
        "rule_template": "동결 P0-A2 v1 dev(seed 7·42) 최근접 사례",
        "exhaustive_probes": ("저장된 6 probe 결과가 있는 사례만(h02_s2027, P1-A 홀드아웃). probe_loop.simulate('exhaustive'). "
                              "시연 seed는 고리가 실행한 probe만 있어 계산하지 않는다"),
        "correction_rule": ("등록 회복 방식 규칙: 보상은 결함 고정 평가 standstill ratio ≥ 0.9이면 restart, 아니면 resume. 모터는 "
                            "restart(W1 갱신). 교정은 correction_registry 등록 표. packet의 rule_result와 같아야 한다. 이 규칙은 "
                            "교정 packet에 실려 모델이 본다"),
        "cost": ("계산 시간은 같은 계산을 50회 반복한 평균 벽시계 초다(이 기기, 입력 읽기 제외). 전수 probe는 저장 probe의 "
                 "elapsed_s 합(GPU 실행)"),
    },
    "use_rule": {"threshold_pass4": USE_THRESHOLD,
                 "rule": ("경로·종류마다 세트 평균 pass^4(결정 일치)가 0.8 이상이면 그 경로의 그 결정을 다음 시연 결정 단계에 계속 쓸 "
                          "수 있다. 미만이면 쓰기 전에 재검토한다. 이미 끝난 시연 결과는 바꾸지 않는다. NAT 기록은 사전등록 전에 본 "
                          "값이라 이 판정을 참고로만 적는다")},
    "wording": {
        "stable": "{path} {kind}: 같은 packet을 {n}회 반복한 결정이 모든 packet에서 유효했고 서로 같았다.",
        "varied": "{path} {kind}: 같은 packet 반복에서 결정이 흔들렸다. 흔들린 packet과 결정 분포를 그대로 공개한다.",
        "usable": "{path} {kind}: 평균 pass^4가 사전등록 하한 0.8 이상이라 다음 시연 결정 단계에 계속 쓸 수 있다.",
        "reconsider": "{path} {kind}: 평균 pass^4가 사전등록 하한 0.8 미만이라 다음 시연 결정 단계에 쓰기 전에 재검토한다.",
        "insufficient": "{path} {kind}: 인프라 오류·예산 소진으로 pass^4를 계산하지 못한 packet이 있어 판정하지 않는다.",
        "common": ("이미 본 packet(시연·개발 입력)을 같은 모델·temperature 0으로 반복한 개발 측정이다. 반복 일관성은 정확도의 근거가 "
                   "아니고, 결정적 기준선은 같은 입력에서 늘 같으므로 pass^k가 1이다. NAT 반복은 사전등록 전에 본 E4b v4 기록을 "
                   "다시 읽었다. 비용은 토큰·벽시계·계산 시간 기록이며 절감 주장이 아니다."),
        "correction_rule_visible": "교정 packet에는 결정적 규칙의 결과가 실려 있어, 교정 결정과 규칙의 일치를 모델의 독립 판단으로 말하지 않는다.",
    },
    "forbidden_claims": ["LLM이 규칙보다 우월하다", "GPU·토큰·시간 절감", "일반화", "정확도", "최초",
                         "NAT와 단일 호출을 같은 조건으로 비교했다", "이미 본 packet 결과를 보류 평가로 부르기"],
}
DYNAMIC = {"source_files", "inputs"}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


# --- 입력 ---
def packet_path(set_name: str, kind: str) -> Path:
    spec = INPUT_SETS[set_name]
    if kind not in spec["kinds"]:
        raise ValueError(f"등록되지 않은 세트·종류 {set_name} {kind}")
    if "packets" in spec:
        return ROOT / spec["packets"][kind]
    return ROOT / DEMO.format(dir=spec["dir"]) / f"{kind}.packet.json"


def recorded_path(set_name: str, kind: str) -> Path | None:
    spec = INPUT_SETS[set_name]
    if kind not in spec["recorded"]:
        return None
    return ROOT / DEMO.format(dir=spec["dir"]) / spec["recorded"][kind][1]


def public_inputs() -> list[str]:
    """결정적 기준선·NAT 반복·교정 규칙이 읽는 공개 파일(저장소 상대 경로)."""
    files = {E4B_SUMMARY, P0C_REFERENCE_PARAMS, "bench/answer_key.json", "bench/cases.json", "bench/reference/manifest.json",
             "bench/answer_key_p0c.json"}
    for spec in INPUT_SETS.values():
        files |= {f"bench/runs_p0c/{spec['case']}.telemetry.json", f"bench/runs_p0c/baseline_p0c_s{spec['seed']}.telemetry.json"}
        if spec["fault_report"]:
            files.add(spec["fault_report"])
        if spec["probes"]:
            files |= {f"{spec['probes']}/{spec['case']}.json", f"{spec['probes']}/baseline_p0c_s{spec['seed']}.json"}
    manifest = read(ROOT / "bench/reference/manifest.json")
    for seed in ("7", "42"):  # v1 dev 학습 입력(evals/baselines.load_seed가 읽는 파일)
        files |= {f"bench/reference/params/seed{seed}/{name}" for name in manifest["seeds"][seed]["sha256"]}
        files.add(manifest["seeds"][seed]["telemetry"])
        files |= {f"{DEV_TELEMETRY}/{case['case_id']}_s{seed}.telemetry.json" for case in read(ROOT / "bench/cases.json")}
    return sorted(files)


def private_inputs() -> dict:
    """비공개 packet·기록된 시연 결정 파일의 원 바이트 SHA. 커밋하지 않는 파일이라 SHA로만 묶는다."""
    out = {}
    for set_name, spec in INPUT_SETS.items():
        for kind in spec["kinds"]:
            entry = {"packet_sha256": sha_raw(packet_path(set_name, kind))}
            recorded = recorded_path(set_name, kind)
            if recorded is not None:
                entry.update(recorded_path_kind=spec["recorded"][kind][0], recorded_sha256=sha_raw(recorded))
            out[f"{set_name}/{kind}"] = entry
    return out


def dynamic_fields() -> dict:
    return {"source_files": {name: sha_lf(CODE_ROOT / name) for name in sorted(CODE)},
            "inputs": {"public": {name: sha_lf(ROOT / name) for name in public_inputs()}, "private": private_inputs()}}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed pass^k contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources or inputs changed")


def prepare() -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_any_call", "path": rel(PROTOCOL),
            "planned_requests": FIXED["api_budget"]["planned_requests"]}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    try:  # 공개 입력·코드·프로토콜이 커밋돼 있고 바뀌지 않았다. 비공개 packet은 SHA로 묶었다
        return protocol, P.verify_committed(ROOT, CODE | set(public_inputs()) | {rel(PROTOCOL)})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


# --- 실행 ---
def run_path(set_name: str, kind: str, run: int, attempt: int) -> Path:
    packet_path(set_name, kind)  # 등록 확인
    return PRIVATE / "single" / set_name / kind / f"run{run}_a{attempt}.json"


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def call_argv(set_name: str, kind: str, run: int, attempt: int) -> list[str]:
    """WSL에서 자격증명 파일을 source한 셸로 단일 호출 스크립트를 부른다. 키 값은 이 프로세스가 보지 않는다."""
    script, flag = SCRIPTS[kind]
    shell = ('. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
             f"{wsl(CODE_ROOT / script)} {flag} {wsl(packet_path(set_name, kind))} "
             f"--output {wsl(run_path(set_name, kind, run, attempt))} --model {MODEL}")
    return ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", shell]


def schedule() -> list[tuple[str, str, int]]:
    return [(set_name, kind, run) for run in range(1, RUNS + 1) for set_name in SET_ORDER
            for kind in INPUT_SETS[set_name]["kinds"]]


def requests_used() -> int:
    return sum(1 for _ in (PRIVATE / "single").glob("*/*/run*_a*.json")) if (PRIVATE / "single").exists() else 0


def plan() -> dict:
    protocol, commit = verify_protocol()
    return {"source_commit_sha": commit, "protocol_sha256_lf": sha_lf(PROTOCOL), "planned_requests": len(schedule()),
            "cap_requests": API_CAP, "requests_used": requests_used(),
            "first_commands": [" ".join(call_argv(s, k, r, 1)) for s, k, r in schedule()[:3]],
            "retry_rule": FIXED["attempts"]}


def _is_infra(record: dict) -> bool:
    return bool(record.get("infra_error"))


def execute(runner=None) -> dict:
    """등록 순서대로 남은 실행을 부른다. 끝난 실행은 건너뛰고(재개 가능), 상한에 닿으면 멈춘다."""
    protocol, commit = verify_protocol()
    runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, timeout=CALL_TIMEOUT_S).returncode)
    log_path = PRIVATE / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done, stopped = 0, None
    for set_name, kind, run in schedule():
        for attempt in range(1, MAX_ATTEMPTS + 1):
            path = run_path(set_name, kind, run, attempt)
            if path.exists():
                if attempt < MAX_ATTEMPTS and _is_infra(read(path)):
                    continue  # 인프라 오류였던 첫 시도: 두 번째 시도로
                break
            if requests_used() >= API_CAP:
                stopped = "budget_exhausted"
                break
            started = time.perf_counter()
            try:
                code = runner(call_argv(set_name, kind, run, attempt))
            except subprocess.TimeoutExpired:
                code = "timeout"  # 스크립트 자체 제한(300 s)보다 길다. 기록 유무를 보고 멈춘다
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(json.dumps({"set": set_name, "kind": kind, "run": run, "attempt": attempt, "exit": code,
                                      "parent_wall_s": round(time.perf_counter() - started, 2), "commit": commit,
                                      "at": datetime.now(timezone.utc).isoformat()}) + "\n")
            if not path.exists():
                stopped = f"no_record:{set_name}/{kind}/run{run}"  # 스크립트가 기록을 못 남겼다. 원인을 본 뒤 재개한다
                break
            done += 1
            if not _is_infra(read(path)):
                break
        if stopped:
            break
    return {"calls_made": done, "requests_used": requests_used(), "cap_requests": API_CAP, "stopped": stopped,
            "remaining": sum(1 for s, k, r in schedule() if final_record(s, k, r)[0] is None)}


# --- 기록 재검증 ---
def final_record(set_name: str, kind: str, run: int) -> tuple[Path | None, dict | None]:
    """run 번호의 최종 시도. 첫 시도가 인프라 오류이고 두 번째 시도가 없으면 (None, None)."""
    records = [(run_path(set_name, kind, run, a), a) for a in range(1, MAX_ATTEMPTS + 1)]
    present = [(p, a) for p, a in records if p.exists()]
    if [a for _, a in present] != list(range(1, len(present) + 1)):
        raise ValueError(f"{set_name} {kind} run{run}: 시도 번호가 이어지지 않는다")
    if not present:
        return None, None
    path, attempt = present[-1]
    record = read(path)
    if attempt < MAX_ATTEMPTS and _is_infra(record):
        return None, None  # 인프라 재시도가 남았다
    return path, record


def decision_key(kind: str, decision: dict) -> tuple:
    if kind == "diagnosis":
        return (decision["action"], decision["ranking"][0])
    return (decision["decision"], decision["proposal"].get("correction_id"), decision["proposal"].get("recovery_mode"))


def review_single(kind: str, record: dict, packet: Path) -> dict:
    """단일 호출 기록 → {status, decision?}. 시연과 같은 검토 함수를 쓴다."""
    if (record.get("request") or {}).get("model") != MODEL or (record.get("request") or {}).get("temperature") != 0:
        raise ValueError("등록 모델·temperature 0 호출이 아니다")
    if kind == "diagnosis":
        import e2e_reward_demo_v2 as V2
        return V2.review_diagnosis_record(record, packet)
    import correction_model_call as CM
    if record.get("packet_sha256") != sha_raw(packet):
        raise ValueError("교정 호출 기록의 packet이 등록 packet과 다르다")
    if _is_infra(record):
        return {"status": "infra_error", "reason": str(record["infra_error"])[:200]}
    if record.get("error_type") or record.get("exit_code") != 0:
        return {"status": "call_error", "reason": str(record.get("error_type"))[:200]}
    packet_data = read(packet)
    review = CM.review_call_record(record, packet_data, sha_raw(packet), packet_data["registry"])
    if not review["accepted"]:
        return {"status": "invalid", "reason": "; ".join(review["reasons"])[:300]}
    return {"status": "valid", "decision": {"decision": review["decision"], "proposal": review["proposal"],
                                            "agrees_with_rule": review["agrees_with_rule"],
                                            "unmet_checklist": review["unmet_checklist"]}}


def recorded_decision(set_name: str, kind: str) -> dict | None:
    """시연에 기록된 결정(경로·결정 키). 기록이 없거나 유효하지 않으면 None."""
    path = recorded_path(set_name, kind)
    if path is None:
        return None
    route = INPUT_SETS[set_name]["recorded"][kind][0]
    record = read(path)
    if route == "nat_v4":
        if record.get("status") != "valid":
            return {"path": route, "key": None}
        decision = record["decision"]
        if kind == "correction":
            decision = {"decision": decision["decision"], "proposal": decision["proposal"]}
        return {"path": route, "key": list(decision_key(kind, decision))}
    review = review_single(kind, record, packet_path(set_name, kind))
    return {"path": route, "key": list(decision_key(kind, review["decision"])) if review["status"] == "valid" else None}


# --- pass^k ---
def pass_hat_k(keys: list, k: int) -> float | None:
    """keys: 인프라 오류를 뺀 실행마다 결정 키(무효면 None). k개를 비복원으로 뽑아 모두 유효하고 같을 확률."""
    n = len(keys)
    if n < k or k < 1:
        return None
    groups = Counter(tuple(key) for key in keys if key is not None)
    return sum(comb(c, k) for c in groups.values()) / comb(n, k)


def pass_hat_k_valid(keys: list, k: int) -> float | None:
    n = len(keys)
    if n < k or k < 1:
        return None
    return comb(sum(1 for key in keys if key is not None), k) / comb(n, k)


def _stats(values: list) -> dict | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return {"n": len(numbers), "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers),
            "total": sum(numbers)}


def repeat_summary(kind: str, runs: list[dict], mechanism: str, recorded: dict | None) -> dict:
    """실행 목록(status, key, 비용) → 상태 수·분포·pass^k·일치·비용."""
    judged = [r for r in runs if r["status"] in ("valid", "invalid", "call_error")]
    keys = [r["key"] if r["status"] == "valid" else None for r in judged]
    valid = [r for r in judged if r["status"] == "valid"]
    distribution = Counter(" · ".join(map(str, r["key"])) for r in valid)
    counts = Counter(r["status"] for r in runs)
    tokens = [(r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in judged
              if r.get("input_tokens") is not None]
    out = {"counts": {s: counts.get(s, 0) for s in ("valid", "invalid", "call_error", "infra_error", "missing",
                                                     "budget_exhausted")},
           "n": len(judged), "decisions": dict(sorted(distribution.items())),
           "pass_k": {str(k): pass_hat_k(keys, k) for k in K_VALUES},
           "pass_k_valid": {str(k): pass_hat_k_valid(keys, k) for k in K_VALUES},
           "agree_with_recorded": None if not recorded or recorded.get("key") is None else sum(
               1 for r in valid if list(r["key"]) == list(recorded["key"])),
           "recorded": recorded,
           **{name: _stats([r.get(name) for r in judged]) for name in ("input_tokens", "output_tokens", "wall_s", "llm_calls")},
           "tokens_per_valid_decision": (sum(tokens) / len(valid)) if valid and tokens else None,
           "runs": runs}
    if kind == "diagnosis":
        out["top1_is_registered_mechanism"] = sum(1 for r in valid if r["key"][1] == mechanism)
        probes = Counter(f"{r['key'][1]} · {r['probe']}" for r in valid)
        out["top1_probe_decisions"] = dict(sorted(probes.items()))
    return out


def single_runs(set_name: str, kind: str) -> list[dict]:
    packet = packet_path(set_name, kind)
    exhausted = requests_used() >= API_CAP
    runs = []
    for run in range(1, RUNS + 1):
        path, record = final_record(set_name, kind, run)
        if record is None:
            runs.append({"run": run, "status": "budget_exhausted" if exhausted else "missing"})
            continue
        review = review_single(kind, record, packet)
        row = {"run": run, "status": review["status"], "input_tokens": record.get("input_tokens"),
               "output_tokens": record.get("output_tokens"), "wall_s": record.get("execution_wall_s"),
               "llm_calls": 1, "record_sha256": sha_raw(path)}
        if review["status"] == "valid":
            row["key"] = list(decision_key(kind, review["decision"]))
            if kind == "diagnosis":
                row["probe"] = review["decision"].get("probe")
        runs.append(row)
    return runs


def nat_runs(set_name: str, kind: str) -> list[dict] | None:
    """E4b v4 공개 요약의 실행별 기록. 그 세트·종류의 NAT 반복이 없으면 None."""
    name = INPUT_SETS[set_name]["nat_repeats"].get(kind)
    if name is None:
        return None
    rows = []
    for run in read(ROOT / E4B_SUMMARY)["sets"][name]["kinds"][kind]["runs"]:
        row = {"run": run["run"], "status": run["status"], "input_tokens": run.get("input_tokens"),
               "output_tokens": run.get("output_tokens"), "wall_s": run.get("wall_s"), "llm_calls": run.get("llm_calls")}
        if run["status"] == "valid":
            row["key"] = list(decision_key(kind, run["decision"]))
            if kind == "diagnosis":
                row["probe"] = run["decision"].get("probe")
        rows.append(row)
    return rows


# --- 결정적 기준선 ---
def _compute_s(fn, repeats: int = 50) -> float:
    """같은 계산을 반복한 평균 벽시계 초(이 기기). process_time은 Windows 해상도(약 15.6 ms)보다 작은 계산을 0으로 잰다."""
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats


def diagnosis_baselines(set_name: str, dev_train: list) -> dict:
    """동결 P0-A2 규칙·v1 최근접 사례·(저장 probe가 있으면) 전수 probe. 5회 계산해 같은지 확인한다."""
    import baselines as BL
    from rl_triage import probe_loop as L
    from rl_triage import rules as R
    spec = INPUT_SETS[set_name]
    run = read(ROOT / f"bench/runs_p0c/{spec['case']}.telemetry.json")["summary"]
    ref = read(ROOT / f"bench/runs_p0c/baseline_p0c_s{spec['seed']}.telemetry.json")["summary"]
    cfg = R.ref_config(BL._yaml(ROOT / P0C_REFERENCE_PARAMS))
    outputs = {
        "rule_prior": lambda: R.prior_ranking(),
        "rule_features": lambda: R.feature_ranking(R.features(run, ref, cfg))[0],
        "rule_template": lambda: R.template_ranking(R.features(run, ref, cfg), dev_train)[0],
    }
    out = {}
    for name, fn in outputs.items():
        rankings = [fn() for _ in range(RUNS)]
        out[name] = {"top1": rankings[0][0], "ranking": rankings[0], "same_over_runs": all(r == rankings[0] for r in rankings),
                     "compute_s_mean": _compute_s(fn), "top1_is_registered_mechanism": rankings[0][0] == spec["mechanism"]}
    if spec["probes"]:
        case_probes = read(ROOT / spec["probes"] / f"{spec['case']}.json")["probes"]
        ref_probes = read(ROOT / spec["probes"] / f"baseline_p0c_s{spec['seed']}.json")["probes"]
        outcomes = {p: (L.classify(p, case_probes[p]["measurement"], ref_probes[p]["measurement"])
                        if case_probes.get(p, {}).get("exit") == 0 and ref_probes.get(p, {}).get("exit") == 0 else "unknown")
                    for p in L.PROBES}
        result = L.simulate("exhaustive", outcomes)
        out["exhaustive_probes"] = {"status": result["status"], "top1": result["conclusion"],
                                    "probes_used": result["probes_used"],
                                    "gpu_s_saved_probes": round(sum(case_probes[p].get("elapsed_s", 0.0) for p in L.PROBES), 1),
                                    "top1_is_registered_mechanism": result["conclusion"] == spec["mechanism"]}
    else:
        out["exhaustive_probes"] = None
    return out


def correction_baseline(set_name: str) -> dict:
    """등록 회복 방식 규칙. 결함 고정 평가 보고서로 다시 계산하고 packet의 rule_result와 대조한다."""
    from rl_triage import correction_registry as CR
    spec = INPUT_SETS[set_name]
    report = read(ROOT / spec["fault_report"])

    def rule():
        mechanism = spec["mechanism"]
        mode = ("restart" if CR.standstill_ratio(report) >= 0.9 else "resume") if mechanism == "reward" else "restart"
        return {"decision": "propose", "correction_id": CR.REGISTRY[mechanism]["correction_id"], "mode": mode}
    decided = rule()
    packet_rule = read(packet_path(set_name, "correction"))["rule_result"]
    if {"correction_id": decided["correction_id"], "mode": decided["mode"]} != {
            "correction_id": packet_rule["correction_id"], "mode": packet_rule["mode"]}:
        raise ValueError(f"{set_name}: 다시 계산한 교정 규칙이 packet의 rule_result와 다르다")
    return {"key": [decided["decision"], decided["correction_id"], decided["mode"]], "same_over_runs": True,
            "compute_s_mean": _compute_s(rule), "standstill_ratio": CR.standstill_ratio(report)}


def dev_training() -> list:
    """v1 dev seed 7·42의 (특징, 정답 범주) — evals/baselines.run과 같은 학습 입력."""
    import baselines as BL
    from rl_triage import rules as R
    key, cases, manifest = read(ROOT / "bench/answer_key.json"), read(ROOT / "bench/cases.json"), read(
        ROOT / "bench/reference/manifest.json")
    train = []
    for seed in R.DEV_SEEDS:
        data = BL.load_seed(seed, manifest, cases, [], "v1")
        train += [(R.features(data["runs"][c["case_id"]], data["ref"], data["cfg"]), key["cases"][c["case_id"]]["category"])
                  for c in cases]
    return train


# --- 요약 ---
def outcome_wording(path: str, kind: str, sets: dict) -> dict:
    rows = [s for s in sets.values() if s is not None]
    pass4 = [s["pass_k"]["4"] for s in rows]
    if not rows or any(v is None for v in pass4):
        return {"stability": None, "use": "insufficient", "mean_pass4": None,
                "wording": [FIXED["wording"]["insufficient"].format(path=path, kind=kind)]}
    stable = all(s["pass_k"]["5"] == 1 for s in rows)
    mean4 = sum(pass4) / len(pass4)
    use = "usable" if mean4 >= USE_THRESHOLD else "reconsider"
    return {"stability": "stable" if stable else "varied", "use": use, "mean_pass4": mean4,
            "mean_pass_k": {str(k): (sum(s["pass_k"][str(k)] for s in rows) / len(rows)
                                     if all(s["pass_k"][str(k)] is not None for s in rows) else None) for k in K_VALUES},
            "wording": [FIXED["wording"]["stable" if stable else "varied"].format(path=path, kind=kind, n=RUNS),
                        FIXED["wording"][use].format(path=path, kind=kind)]}


def summarize(write: bool = True) -> dict:
    protocol, commit = verify_protocol()
    train = dev_training()
    sets, single_by_kind, nat_by_kind = {}, {k: {} for k in KINDS}, {k: {} for k in KINDS}
    for set_name in SET_ORDER:
        spec = INPUT_SETS[set_name]
        sets[set_name] = {"case": spec["case"], "seed": spec["seed"], "role": spec["role"], "mechanism": spec["mechanism"],
                          "kinds": {}}
        for kind in spec["kinds"]:
            if sha_raw(packet_path(set_name, kind)) != protocol["inputs"]["private"][f"{set_name}/{kind}"]["packet_sha256"]:
                raise ValueError(f"{set_name} {kind}: packet이 사전등록 SHA와 다르다")
            recorded = recorded_decision(set_name, kind)
            single = repeat_summary(kind, single_runs(set_name, kind), spec["mechanism"], recorded)
            nat_rows = nat_runs(set_name, kind)
            nat = None if nat_rows is None else repeat_summary(kind, nat_rows, spec["mechanism"], recorded)
            baseline = diagnosis_baselines(set_name, train) if kind == "diagnosis" else correction_baseline(set_name)
            sets[set_name]["kinds"][kind] = {"single": single, "nat_v4": nat, "deterministic": baseline}
            single_by_kind[kind][set_name] = single
            if nat is not None:
                nat_by_kind[kind][set_name] = nat
    complete = all(s["counts"]["missing"] == 0 and s["counts"]["budget_exhausted"] == 0
                   for kinds in single_by_kind.values() for s in kinds.values())
    exhausted = requests_used() >= API_CAP
    outcomes = {f"single/{kind}": outcome_wording("단일 호출", KIND_KO[kind], rows) for kind, rows in single_by_kind.items()}
    outcomes |= {f"nat_v4/{kind}": {**outcome_wording("NAT v4", KIND_KO[kind], rows), "seen_before_preregistration": True}
                 for kind, rows in nat_by_kind.items() if rows}
    matched = {f"{set_name}/{kind}": {"single_pass_k": sets[set_name]["kinds"][kind]["single"]["pass_k"],
                                      "nat_pass_k": sets[set_name]["kinds"][kind]["nat_v4"]["pass_k"]}
               for set_name in SET_ORDER for kind in INPUT_SETS[set_name]["kinds"]
               if sets[set_name]["kinds"][kind]["nat_v4"] is not None}
    result = {"schema": "n2_pass_k_result_v1", "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit,
              "complete": complete, "budget": {"requests_used": requests_used(), "cap_requests": API_CAP,
                                               "exhausted": exhausted},
              "sets": sets, "matched_packets": matched, "outcomes": outcomes,
              "common_wording": FIXED["wording"]["common"],
              "correction_rule_wording": FIXED["wording"]["correction_rule_visible"]}
    if write and (complete or exhausted):
        write_new(RESULTS / RESULT_TAG / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N2 반복 일관성(pass^k)과 비용")
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
