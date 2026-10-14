"""R1 고리 단계: smoke가 failure_present일 때 같은 seed 정상·R1 정책에 동결 probe 6종을 한 번씩 재고, 같은 관측표 위에서
고리 방식들(fixed·exhaustive·random·모델 순위 가르기)이 무엇을 결론 내는지 기록한다. smoke 결과를 보기 전에 커밋한다.

R1은 rsl_rl #38의 의미(PPO가 설정의 epoch 수를 쓰지 않음)를 옮긴 optimizer 범주의 설계자 밖 결함이다(bench/optimizer_faults.py,
smoke evals/run_upstream_r1_smoke.py). 동결 P_value는 critic 파라미터가 거의 움직이지 않는 결함을 겨누므로, 이 단계는 등록 범주
안의 다른 구현을 고리가 맞게 확정하는지·틀리게 확정하는지·멈추는지를 사전등록한 예측과 대조한다. 진단 정확도를 재지 않는다.

  python evals/run_upstream_r1_loop.py prepare                                  # 프로토콜 작성(커밋 전, smoke 실행 전)
  python evals/run_upstream_r1_loop.py verify
  python evals/run_upstream_r1_loop.py gate --tag upstream_r1_loop_<날짜>       # smoke 최종 결과 확인(failure_absent면 조건 미충족으로 끝)
  python evals/run_upstream_r1_loop.py packet --tag <tag>                       # 진단 packet(텔레메트리 사실만)과 WSL 호출 명령
  python evals/run_upstream_r1_loop.py rank --tag <tag> --record <호출 기록>     # 단일 호출 검토(무효면 model_invalid, 재호출 없음)
  python evals/run_upstream_r1_loop.py probes --tag <tag>                       # GPU: 정상·R1 정책에 probe 6종(한 Isaac 프로세스)
  python evals/run_upstream_r1_loop.py summarize --tag <tag>                    # 관측표·방식별 결론·예측 대조·공개 결과

probe 프로세스에는 결함을 걸지 않는다(jobs fault NONE). R1은 PPO.update만 바꾸는데 동결 probes.py는 체크포인트를 읽고 롤아웃만 하며
update를 부르지 않는다. 모델 호출은 WSL에서 `. ~/.config/nvidia/env`를 source한 셸로만 하고 이 실행기는 키를 읽지 않는다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rl_triage import probe_loop as L  # noqa: E402
import run_upstream_r1_smoke as S  # noqa: E402

U = S.U
PROTOCOL = ROOT / "bench/protocols/upstream_r1_loop_v1.json"
PRIVATE = ROOT / "bench/private/upstream_r1_loop"
RESULTS = ROOT / "evals/results"
TAG_RE = r"upstream_r1_loop_[0-9]{8}[a-z0-9_]{0,10}"
TRUTH = "optimizer"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
LOOP_TOP_K, LOOP_BUDGET, PROBE_EXECUTION_BUDGET_S = 3, 4, 180
MAX_INFRA_ATTEMPTS, PROBE_ATTEMPTS = 2, 2
RANDOM_SEEDS = range(5)
PROBE_ORDER = tuple(sorted(L.PROBES))
WATCHDOG_S = 1200
SPECIFICITY_GATES = {"actuator": ["P_noise", "P_value"]}
WSL_PYTHON = "$HOME/.venvs/rl-triage/bin/python"
CODE = tuple(dict.fromkeys((
    "evals/run_upstream_r1_loop.py", *S.CODE, "evals/probes.py", "bench/hidden_faults.py", "bench/external_faults.py",
    "evals/run_active_diagnosis.py", "evals/active_model_call.py", "evals/e2e_reward_demo_v2.py",
    "src/rl_triage/active_diagnosis.py")))
OUTCOMES = ("condition_not_met", "invalid", "stopped", "wrong_confirmation", "correct_confirmation")
FIXED = {
    "schema": "upstream_optimizer_fault_loop_protocol_v1",
    "question": "R1(rsl_rl #38의 의미를 옮긴 optimizer 결함) 정책에서 동결 probe 고리는 맞게 확정하는가, 틀리게 확정하는가, 멈추는가",
    "scope": "설계자 밖 외부 결함 한 종류·새 seed 하나(2042)의 고리 동작 기록. 진단 정확도·안전성 보장·일반화가 아니다",
    "fault": {"id": "R1", "category": TRUTH, "smoke_protocol": "bench/protocols/upstream_r1_smoke_v1.json",
              "note": "결함 정의·seed·학습은 smoke 프로토콜을 따른다. 이 단계는 smoke가 만든 두 정책만 읽는다"},
    "condition": {
        "rule": "smoke 프로토콜의 최종 결과(최종 표식이 가리키는 result.json)가 failure_present일 때만 실행한다",
        "failure_absent": "실행하지 않고 공개 결과에 condition_not_met(조건 미충족)을 한 번 남긴다. probe·모델 호출 0",
        "pending": "smoke 최종 결과가 없으면(아직 실행 전·invalid만 있음) gate가 거부하고 아무것도 쓰지 않는다",
        "committed_before_smoke": "이 프로토콜은 smoke 실행 전에 커밋한다. smoke 프로토콜 SHA를 결합해 다른 smoke 결과를 쓰지 못하게 한다",
    },
    "observation": {
        "design": "정상(NONE)·R1 정책의 마지막 체크포인트(model_299.pt)에 동결 probe 6종을 한 번씩 잰다. 방식들은 같은 관측표 위에서 돌린다"
                  "(P1-A 7절·T1과 같은 비교). 같은 체크포인트·seed에서 probe 결과는 순서와 무관하다는 동결 simulate의 전제를 따른다",
        "process": "실행마다 Isaac 프로세스 하나(정상 먼저, R1 다음), 프로세스마다 job 하나·probe 6종 이름순. P1-A holdout처럼 실행별 프로세스로 "
                   "재서 같은 프로세스 안의 job 순서 영향을 두지 않는다. evals/probes.py를 그대로 부른다(fault NONE, seed 2042)",
        "no_injection_reason": "R1은 PPO.update만 바꾼다. probes.py는 OnPolicyRunner를 만들고 체크포인트를 load_optimizer=False로 읽은 뒤 "
                               "롤아웃만 하며 update를 부르지 않는다. P_value는 체크포인트 파일(model_0.pt·model_299.pt)만 읽는다",
        "classification": "probe_loop.classify(같은 seed 정상 측정 대비). 동결 임계값을 바꾸지 않는다",
        "approval": "probe 측정은 방식의 선택이 아니라 모든 방식이 함께 쓰는 고정 관측이라 원장 승인 없이 운영자 명령으로 잰다(D3와 같다). "
                    "승인 고리 실행이 아니다",
        "retry": "Isaac 프로세스가 끝나지 않았거나 보고서가 없거나 probe 하나라도 exit≠0이면 인프라 실패로 보고 12개를 새 폴더에 한 번 더 잰다. "
                 "그 뒤에도 남으면 invalid",
        "watchdog_s": WATCHDOG_S,
    },
    "strategies": {
        "fixed": "FIXED_ORDER(P_episode→P_noise→P_reward→P_torque→P_physics→P_value), 여섯 가설 모두, 예산 6",
        "exhaustive": "이름순으로 여섯 개를 모두 보고 결론, 예산 6",
        "random": f"seed {list(RANDOM_SEEDS)}의 무작위 순서, 예산 6. seed마다 따로 센다",
        "discriminate": f"모델 순위 상위 {LOOP_TOP_K}가설, 예산 {LOOP_BUDGET}. 첫 probe는 모델이 고른 probe가 상위 가설을 가르면 그 probe, "
                        "아니면 동결 next_probe(종단 시연 start_loop와 같은 규칙). 이후 동결 next_probe·update·status. 관측 뒤 모델 재호출 없음",
        "simulator": "src/rl_triage/probe_loop.simulate(fixed·exhaustive·random)와 같은 update·status 규칙",
    },
    "model": {
        "id": MODEL, "packet": "evals/run_active_diagnosis.input_packet(R1·정상 텔레메트리 summary의 마지막 20% 평균 사실, 공개 probe "
                                "카탈로그·임계값, max_probes 4, execution_budget_s 180). 실행 이름·결함 이름·정답을 넣지 않는다",
        "call": "evals/active_model_call.py(WSL, temperature 0, max_tokens 16384, 단일 호출). 키는 `. ~/.config/nvidia/env`로만",
        "contract": "e2e_reward_demo_v2.review_diagnosis_record → rl_triage.active_diagnosis.validate_decision",
        "invalid": "계약 위반이면 model_invalid로 기록하고 재호출하지 않는다. discriminate 칸은 model_invalid로 남긴다",
        "infra": f"빈 응답·잘림·연결·429/5xx(infra_error)만 최대 {MAX_INFRA_ATTEMPTS}회 호출. 그 밖 호출 오류는 model_failed",
        "hold": "action이 hold이면 discriminate 칸은 model_hold(probe 0)로 남긴다",
        "basis": "N2 반복 일관성·구조화 출력 결과(계약 위반 6/55, 구조화 arm 37/55)로 단일 호출 경로를 그대로 쓰고 재호출로 무효를 덮지 않는다",
        "approval": "API 호출은 메인 세션이 WSL에서 실행한다(사용자가 위임한 운영자 승인 또는 사람 승인, 승인자 문자열 기록, 신원 인증 아님)",
    },
    "specificity_gate": {"rule": "종단 시연의 특이성 관문을 같은 관측표로 계산만 한다: actuator로 확정한 방식이 있으면 P_noise·P_value가 모두 "
                                 "normal인지 본다. 판정에 쓰지 않는 보조 기록이다", "gates": SPECIFICITY_GATES},
    "validity_checks": {
        "smoke_failure_present": "smoke 최종 결과가 failure_present이고 그 결과의 프로토콜 SHA가 결합한 smoke 프로토콜과 같다",
        "probe_reports_bound": "두 probe 보고서의 실행 이름·seed·체크포인트 SHA가 smoke 결과와 같고 probe 6종이 모두 있다",
        "probes_complete": "12개 측정이 모두 exit 0(재측정 한 번 뒤)",
        "model_resolved": "모델 단계가 valid·model_invalid·model_failed 중 하나로 끝났다(discriminate 칸의 상태가 정해짐)",
    },
    "outcomes": {
        "condition_not_met": "smoke가 failure_absent라 실행하지 않았다",
        "invalid": "유효 조건 중 하나라도 어긋남. 방식별 결론을 판정에 쓰지 않는다",
        "stopped": "결론을 낸 모든 방식이 확정하지 않았다(none_supported·unidentifiable)",
        "wrong_confirmation": "어느 한 방식이라도 optimizer가 아닌 범주를 확정했다",
        "correct_confirmation": "틀린 확정 없이 어느 방식이 optimizer로 확정했다",
    },
    "prediction": {
        "probes": {
            "P_value": {"outcome": "normal", "basis": "critic 변화가 같은 seed 정상의 10% 이상. P0-C 정상 1.43~1.47, 학습률이 하한에 머문 "
                                                      "h05도 0.40~0.60(정상의 27~41%). R1은 갱신 1200회가 남는다"},
            "P_noise": {"outcome": "normal", "basis": "샘플링 경로를 바꾸지 않는다"},
            "P_reward": {"outcome": "normal", "basis": "보상 코드를 바꾸지 않는다"},
            "P_physics": {"outcome": "normal", "basis": "물리 설정을 바꾸지 않는다"},
            "P_episode": {"outcome": "normal", "basis": "P0-C 실패 정책 중 종료 결함 밖에서는 시간 제한 종료 비 1.0",
                          "risk": "정책이 넘어져 에피소드가 일찍 끝나면 abnormal → termination 틀린 확정"},
            "P_torque": {"outcome": "normal", "basis": "기준은 바닥값 0.01의 3배인 0.03. P0-C h01·h02·h03은 0.006 이하",
                         "risk": "덜 학습된 정책은 저속 토크 포화가 커질 수 있다. P0-C h04 1/3(0.038), h05 3/3(0.06~0.81)이 넘었다 → actuator 틀린 확정"},
        },
        "strategies": {"fixed": "none_supported", "exhaustive": "none_supported", "random": "none_supported(5개 seed 모두)",
                       "discriminate": "action이 continue이면 none_supported(예측한 관측표에서는 어떤 상위 3가설·첫 probe로도 none_supported)"},
        "outcome": "stopped",
        "model": "순위 내용·action·계약 통과 여부는 예측하지 않는다(기록만)",
        "specificity_gate": "actuator 확정이 없으면 계산하지 않는다. P_torque 위험 시나리오에서 actuator로 확정되면 P_noise·P_value가 예측대로 "
                            "normal이라 관문을 통과해 틀린 확정을 막지 못할 것으로 본다",
        "scenarios": "P_torque만 abnormal → 네 방식 모두 actuator 확정(틀림, discriminate는 actuator가 상위 3에 있을 때), P_episode만 "
                     "abnormal → termination 확정(틀림), P_value만 abnormal → optimizer 확정(맞음, 예측하지 않음). 동결 simulate 값이다",
    },
    "no_tuning_rule": "관측을 본 뒤 임계값·probe·방식·예산·상위 가설 수를 바꾸지 않는다. 다른 seed나 재학습으로 다시 하지 않는다",
    "wording": {
        "condition_not_met": "R1 smoke가 failure_absent라 고리 단계의 실행 조건을 채우지 못했다. 고리를 실행하지 않았다.",
        "invalid": "고리 단계가 사전등록한 유효 조건을 채우지 못해 방식별 결론을 판정에 쓰지 않는다.",
        "stopped": "rsl_rl #38의 의미를 옮긴 optimizer 결함에서 동결 probe 고리는 비교한 방식 모두 원인을 확정하지 않고 멈췄다(맞는 확정 0, "
                   "틀린 확정 0). 등록 범주 안의 다른 구현을 동결 P_value가 겨누지 못한 새 seed 단일 사례이며, 안전성 보장이나 일반화가 아니다.",
        "wrong_confirmation": "rsl_rl #38의 의미를 옮긴 optimizer 결함에서 동결 probe 고리의 일부 방식이 원인이 아닌 범주를 확정했다(틀린 확정). "
                              "설계 밖 구현의 결함에서 고리가 틀릴 수 있음을 보이는 새 seed 단일 사례다.",
        "correct_confirmation": "rsl_rl #38의 의미를 옮긴 optimizer 결함에서 동결 probe 고리가 optimizer로 확정했고 틀린 확정은 없었다. "
                                "새 seed 단일 사례라 진단 정확도의 근거가 아니다.",
    },
    "gpu_estimate": {"unit": "부모 프로세스 벽시계(초). GPU 커널 시간이나 절감량이 아니다",
                     "probes_s": [200, 400],
                     "basis": "P1-A holdout 실행 하나의 probe 6종 측정 합 61.9~82.3 s × 2 + Isaac 시작·종료 2회(측정 기록 없음, 추정)"},
    "api_estimate": {"calls": [1, MAX_INFRA_ATTEMPTS], "note": "인프라 오류일 때만 2회"},
}
DYNAMIC = {"source_files", "isaaclab_sources", "smoke_protocol_sha256_lf"}


# --- 결합 ---
def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: U.sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": S.isaaclab_sources(isaaclab), "smoke_protocol_sha256_lf": U.sha(S.PROTOCOL, True)}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab·rsl_rl 소스·smoke 프로토콜이 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    U.write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_smoke", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = U.read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    paths = set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix(), S.PROTOCOL.relative_to(S.ROOT).as_posix()}
    try:
        commit = verify_committed(ROOT, paths)
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 프로토콜이 커밋되지 않았다") from error
    return protocol, commit


# --- 관측표 위의 방식(GPU 없이 시험한다) ---
def classify_table(reports: dict) -> dict[str, str]:
    """R1 보고서의 probe마다 같은 seed 정상 보고서 대비 판정. 어느 쪽이든 exit≠0이면 unknown."""
    reference, fault = reports["reference"]["probes"], reports["fault"]["probes"]
    table = {}
    for probe in L.PROBES:
        r, ref = fault.get(probe) or {}, reference.get(probe) or {}
        ok = r.get("exit") == 0 and ref.get("exit") == 0
        table[probe] = L.classify(probe, r.get("measurement"), ref.get("measurement")) if ok else "unknown"
    return table


def discriminate(outcomes: dict[str, str], ranking: list[str], model_probe: str | None) -> dict:
    """종단 시연 고리와 같은 규칙: 상위 3가설, 첫 probe는 모델 probe가 가르면 그것, 이후 동결 next_probe, 예산 4."""
    hyps = list(ranking[:LOOP_TOP_K])
    rule_probe = L.next_probe(hyps, {})
    usable = model_probe in L.PROBES and (L.discriminates(model_probe, hyps) or (
        len(hyps) == 1 and model_probe == L.confirming_probe(hyps[0])))
    probe = model_probe if usable else rule_probe
    observed: dict[str, str] = {}
    steps = []
    while probe is not None and len(observed) < LOOP_BUDGET:
        outcome = outcomes.get(probe, "unknown")
        observed[probe] = outcome
        hyps, dropped = L.update(hyps, probe, outcome)
        steps.append({"probe": probe, "outcome": outcome, "dropped": dropped, "left": list(hyps)})
        if L.status(hyps, observed, LOOP_BUDGET - len(observed)) != "open":
            break
        probe = L.next_probe(hyps, observed)
    final = L.status(hyps, observed, LOOP_BUDGET - len(observed))
    final = "unidentifiable" if final == "open" else final
    return {"strategy": "discriminate", "status": final, "conclusion": hyps[0] if final == "confirmed" else None,
            "probes_used": len(observed), "steps": steps, "hypotheses": list(ranking[:LOOP_TOP_K]),
            "model_probe": model_probe, "model_probe_used": bool(usable), "rule_probe": rule_probe}


def strategy_rows(outcomes: dict[str, str], model: dict) -> list[dict]:
    plans = [("fixed", {}), ("exhaustive", {})] + [(f"random{s}", {"seed": s}) for s in RANDOM_SEEDS]
    rows = []
    for label, kwargs in plans:
        res = L.simulate("random" if label.startswith("random") else label, outcomes, **kwargs)
        rows.append({"strategy": label, "status": res["status"], "conclusion": res["conclusion"],
                     "probes_used": res["probes_used"], "steps": res["steps"]})
    status = model.get("status")
    decision = model.get("decision") or {}
    if status != "valid":
        rows.append({"strategy": "discriminate", "status": status, "conclusion": None, "probes_used": 0, "steps": []})
    elif decision.get("action") != "continue":
        rows.append({"strategy": "discriminate", "status": "model_hold", "conclusion": None, "probes_used": 0, "steps": [],
                     "action": decision.get("action")})
    else:
        rows.append({**discriminate(outcomes, decision["ranking"], decision.get("probe")), "strategy": "discriminate"})
    for row in rows:
        row["correct"] = row["status"] == "confirmed" and row["conclusion"] == TRUTH
        row["wrong"] = row["status"] == "confirmed" and row["conclusion"] != TRUTH
    return rows


def specificity(rows: list[dict], outcomes: dict[str, str]) -> dict:
    out = {}
    for row in rows:
        gate = SPECIFICITY_GATES.get(row["conclusion"]) if row["status"] == "confirmed" else None
        if gate:
            out[row["strategy"]] = {"probes": {p: outcomes.get(p) for p in gate},
                                    "passed": all(outcomes.get(p) == "normal" for p in gate)}
    return out


def overall(rows: list[dict]) -> str:
    if any(row["wrong"] for row in rows):
        return "wrong_confirmation"
    if any(row["correct"] for row in rows):
        return "correct_confirmation"
    return "stopped"


def prediction_check(outcomes: dict[str, str], rows: list[dict], outcome: str) -> dict:
    predicted = {p: spec["outcome"] for p, spec in FIXED["prediction"]["probes"].items()}
    expected = {row["strategy"]: "none_supported" for row in rows
                if not (row["strategy"] == "discriminate" and row["status"] in ("model_invalid", "model_failed", "model_hold"))}
    return {"probes": {p: outcomes.get(p) == predicted[p] for p in L.PROBES},
            "strategies": {s: next(r["status"] for r in rows if r["strategy"] == s) == v for s, v in expected.items()},
            "outcome": outcome == FIXED["prediction"]["outcome"]}


# --- 단계 기록 ---
def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("tag는 upstream_r1_loop_<YYYYMMDD>[접미]")


def folder(tag: str) -> Path:
    _check_tag(tag)
    return PRIVATE / tag


def step(tag: str, name: str) -> dict | None:
    path = folder(tag) / "steps" / f"{name}.json"
    return U.read(path) if path.is_file() else None


def _write_step(tag: str, name: str, record: dict) -> dict:
    U.write_new(folder(tag) / "steps" / f"{name}.json", record)
    return record


def _consumed() -> Path:
    return PRIVATE / "consumed" / f"{U.sha(PROTOCOL, True)[:16]}_gate.json"


def _require_open(tag: str, *needed: str) -> dict:
    if (RESULTS / tag / "result.json").exists():
        raise FileExistsError("이 tag의 공개 결과가 이미 있다")
    gate = step(tag, "gate")
    if gate is None or gate.get("smoke_outcome") != "failure_present":
        raise ValueError("gate가 failure_present로 통과한 tag가 아니다")
    for name in needed:
        if step(tag, name) is None:
            raise ValueError(f"{name} 단계가 먼저다")
    return gate


def smoke_final() -> dict:
    """smoke 프로토콜의 최종 표식과 그 결과. 프로토콜 SHA·결과 범주가 맞아야 한다."""
    protocol_sha = U.sha(S.PROTOCOL, True)
    marker = S.PRIVATE / "consumed" / f"{protocol_sha[:16]}_final.json"
    if not marker.is_file():
        raise ValueError("smoke 최종 결과가 아직 없다(실행 전이거나 invalid만 있다)")
    final = U.read(marker)
    result = U.read(S.RESULTS / final["tag"] / "result.json")
    if (result.get("protocol_sha256_lf") != protocol_sha or result.get("outcome") != final.get("outcome")
            or result["outcome"] not in ("failure_present", "failure_absent")):
        raise ValueError("smoke 최종 표식과 결과가 맞지 않는다")
    return {"tag": final["tag"], "result": result}


def gate(tag: str, isaaclab: Path) -> dict:
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    if folder(tag).exists() or (RESULTS / tag).exists() or _consumed().exists():
        raise FileExistsError("고리 단계는 프로토콜마다 한 번만 연다")
    final = smoke_final()
    if U.sha(S.PROTOCOL, True) != protocol["smoke_protocol_sha256_lf"]:
        raise ValueError("smoke 프로토콜이 결합값과 다르다")
    result = final["result"]
    U.write_new(_consumed(), {"tag": tag, "smoke_tag": final["tag"], "smoke_outcome": result["outcome"]})
    record = {"source_commit_sha": commit, "smoke_tag": final["tag"], "smoke_outcome": result["outcome"],
              "smoke_protocol_sha256_lf": result["protocol_sha256_lf"], "run_names": result["run_names"],
              "checkpoints": {arm: result["runs"][arm]["checkpoint_sha256"] for arm in S.ARMS},
              "telemetry": {arm: result["runs"][arm]["telemetry_sha256"] for arm in S.ARMS}}
    _write_step(tag, "gate", record)
    if result["outcome"] == "failure_absent":
        public = {"schema": "upstream_optimizer_fault_loop_result_v1", "tag": tag, "outcome": "condition_not_met",
                  "wording": FIXED["wording"]["condition_not_met"], "smoke": {"tag": final["tag"], "outcome": result["outcome"]},
                  "protocol_sha256_lf": U.sha(PROTOCOL, True), "source_commit_sha": commit, "probes_measured": 0,
                  "model_calls": 0, "scope": FIXED["scope"]}
        U.write_new(RESULTS / tag / "result.json", public)
        return {"status": "condition_not_met", "result": f"evals/results/{tag}/result.json"}
    return {"status": "gate_passed", "smoke_tag": final["tag"], "next": f"python evals/run_upstream_r1_loop.py packet --tag {tag}"}


def wsl_path(path: Path) -> str:
    text = Path(path).resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", text):
        raise ValueError("Windows 경로가 필요하다")
    return f"/mnt/{text[0].lower()}/{text[3:]}"


def packet(tag: str, isaaclab: Path) -> dict:
    import run_active_diagnosis as RA
    gate_record = _require_open(tag)
    verify(isaaclab)
    if step(tag, "packet") is not None:
        raise FileExistsError("packet은 한 번만 만든다")
    sources = {}
    for arm, key in (("fault", "telemetry"), ("reference", "reference_telemetry")):
        path = S.telemetry_path(gate_record["run_names"][arm])
        if U.sha(path) != gate_record["telemetry"][arm]:
            raise ValueError(f"{arm} 텔레메트리가 smoke 결과에 기록된 것과 다르다")
        sources[key] = {"path": path.relative_to(ROOT).as_posix()}
    built = RA.input_packet({"sources": sources}, {"max_probes": LOOP_BUDGET, "probe_execution_budget_s": PROBE_EXECUTION_BUDGET_S})
    text = json.dumps(built, ensure_ascii=False)
    if any(name in text for name in gate_record["run_names"].values()) or "R1" in text:
        raise ValueError("packet에 실행 이름이나 결함 이름이 들어갔다")
    path = folder(tag) / "model" / "diagnosis.packet.json"
    U.write_new(path, built)
    _write_step(tag, "packet", {"path": path.relative_to(ROOT).as_posix(), "sha256": U.sha(path)})
    output = folder(tag) / "model" / "diagnosis.call1.json"
    command = (f"wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"{WSL_PYTHON}\" "
               f"{wsl_path(ROOT / 'evals/active_model_call.py')} --packet {wsl_path(path)} --output {wsl_path(output)} "
               f"--model {MODEL}'")
    return {"status": "packet_written", "packet": path.relative_to(ROOT).as_posix(), "model_command": command,
            "next": f"python evals/run_upstream_r1_loop.py rank --tag {tag} --record {output.relative_to(ROOT).as_posix()}"}


def rank(tag: str, record_path: Path, isaaclab: Path) -> dict:
    import e2e_reward_demo_v2 as V2
    _require_open(tag, "packet")
    verify(isaaclab)
    if step(tag, "model") is not None:
        raise FileExistsError("모델 단계가 이미 끝났다")
    model_dir = (folder(tag) / "model").resolve()
    attempts = sorted(model_dir.glob("diagnosis.attempt*.json"))
    expected = model_dir / f"diagnosis.call{len(attempts) + 1}.json"
    if Path(record_path).resolve() != expected:
        raise ValueError(f"다음 호출 기록은 {expected.name}이어야 한다(model 폴더, 순서대로)")
    packet_info = step(tag, "packet")
    packet_path = ROOT / packet_info["path"]
    if U.sha(packet_path) != packet_info["sha256"]:
        raise ValueError("packet이 바뀌었다")
    record = U.read(record_path)
    review = V2.review_diagnosis_record(record, packet_path)
    keys = ("response_id", "provider_model", "input_tokens", "output_tokens", "total_tokens", "execution_wall_s", "finish_reason")
    call = {"record": Path(record_path).relative_to(ROOT).as_posix(), "record_sha256": U.sha(record_path),
            **{key: record.get(key) for key in keys}}
    number = len(attempts) + 1
    U.write_new(model_dir / f"diagnosis.attempt{number}.json", {**call, "status": review["status"], "reason": review.get("reason")})
    if review["status"] == "infra_error" and number < MAX_INFRA_ATTEMPTS:
        return {"status": "retry_allowed", "attempt": number, "next_record": f"diagnosis.call{number + 1}.json"}
    status = {"valid": "valid", "invalid": "model_invalid"}.get(review["status"], "model_failed")
    calls = [U.read(path) for path in sorted(model_dir.glob("diagnosis.attempt*.json"))]
    out = _write_step(tag, "model", {"status": status, "reason": review.get("reason"), "decision": review.get("decision"),
                                     "attempts": number, "calls": calls})
    return {"status": status, "decision": out["decision"]}


def _probe_folder(tag: str, attempt: int) -> tuple[Path, Path]:
    return folder(tag) / f"probes_a{attempt}", RESULTS / tag / "probes" / f"a{attempt}"


def _bind_reports(reports: dict, gate_record: dict) -> bool:
    return all(isinstance(report, dict) and report.get("name") == gate_record["run_names"][arm]
               and report.get("seed") == S.SEED
               and (report.get("checkpoint") or {}).get("sha256") == gate_record["checkpoints"][arm]
               and set((report.get("probes") or {})) == set(L.PROBES) for arm, report in reports.items())


def _complete(reports: dict) -> bool:
    return all(record.get("exit") == 0 for report in reports.values() for record in report["probes"].values())


def probes(tag: str, isaaclab: Path) -> dict:
    gate_record = _require_open(tag)
    protocol, commit = verify(isaaclab)
    if step(tag, "probes") is not None:
        raise FileExistsError("probe 단계가 이미 끝났다")
    attempt = 1 + len(sorted(folder(tag).glob("probes_a*")))
    if attempt > PROBE_ATTEMPTS:
        raise ValueError("probe 재측정은 한 번뿐이다")
    gpu = U._admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "launched": False}
    private, public = _probe_folder(tag, attempt)
    jobs = {}
    for arm in S.ARMS:
        run_dirs = U.find_run_dir(isaaclab, gate_record["run_names"][arm])
        if len(run_dirs) != 1:
            raise ValueError(f"{arm} 실행 폴더가 하나가 아니다")
        checkpoint = run_dirs[0] / S.FINAL_CHECKPOINT
        if U.sha(checkpoint) != gate_record["checkpoints"][arm] or not (run_dirs[0] / "model_0.pt").is_file():
            raise ValueError(f"{arm} 체크포인트가 smoke 결과와 다르거나 model_0.pt가 없다")
        jobs[arm] = {"name": gate_record["run_names"][arm], "checkpoint": str(checkpoint), "probes": list(PROBE_ORDER),
                     "output": str(public / f"{gate_record['run_names'][arm]}.json")}
    U.write_new(private / "gpu_before.json", gpu)
    failure, executions = None, {}
    try:
        for arm in S.ARMS:  # 실행마다 프로세스 하나(P1-A holdout과 같다)
            jobs_path = private / f"jobs_{arm}.json"
            U.write_new(jobs_path, {"fault": "NONE", "seed": S.SEED, "jobs": [jobs[arm]]})
            executions[arm] = U._owned(private / f"run_{arm}", [str(isaaclab / "_isaac_sim" / "python.bat"),
                                                                str(ROOT / "evals" / "probes.py"), "--jobs", str(jobs_path),
                                                                "--headless"], isaaclab, WATCHDOG_S)
            if not U.finished(executions[arm]) or executions[arm]["exit_code"] != 0:
                raise RuntimeError(f"{arm} probe 프로세스가 끝나지 않았거나 종료 코드가 0이 아니다")
        after = U._admission()
        U.write_new(private / "gpu_after.json", after)
        owned = {pid: created for execution in executions.values()
                 for pid, created in execution["owned_process_identities"].items()}
        if any(str(row["pid"]) in owned and U._process_created(int(row["pid"])) == owned[str(row["pid"])]
               for row in after["processes"]):
            raise ValueError("소유 프로세스가 GPU에 남아 있다")
        _reverify(isaaclab, protocol, commit)
    except Exception as error:  # 실패해도 시도 기록을 남겨 재측정 규칙을 적용한다
        failure = {"error_type": type(error).__name__, "detail": str(error)[:300]}
    reports = {arm: U.read(public / f"{gate_record['run_names'][arm]}.json")
               if (public / f"{gate_record['run_names'][arm]}.json").is_file() else None for arm in S.ARMS}
    bound = failure is None and _bind_reports(reports, gate_record)
    complete = bound and _complete(reports)
    U.write_new(private / "attempt.json", {"attempt": attempt, "failure": failure, "bound": bound, "complete": complete})
    if complete or attempt == PROBE_ATTEMPTS:
        _write_step(tag, "probes", {"attempt": attempt, "bound": bound, "complete": complete, "failure": failure,
                                    "reports": {arm: (public / f"{gate_record['run_names'][arm]}.json").relative_to(ROOT).as_posix()
                                                for arm in S.ARMS} if bound else None,
                                    "process_wall_s": {arm: round(e["wall_s"], 1) for arm, e in executions.items()}})
        return {"status": "complete" if complete else "final_incomplete", "attempt": attempt}
    return {"status": "retry_allowed", "attempt": attempt, "failure": failure}


def _reverify(isaaclab: Path, protocol: dict, commit: str) -> None:
    now_protocol, now = verify(isaaclab)
    if now_protocol != protocol:
        raise ValueError("실행 중 프로토콜이나 결합 소스가 바뀌었다")
    paths = [*CODE, PROTOCOL.relative_to(ROOT).as_posix()]
    if now != commit and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, now, "--", *paths],
                                        capture_output=True).returncode != 0:
        raise ValueError("시작 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")


def summarize(tag: str, isaaclab: Path) -> dict:
    gate_record = _require_open(tag, "probes", "model")
    _, commit = verify(isaaclab)
    probe_step, model = step(tag, "probes"), step(tag, "model")
    reports = ({arm: U.read(ROOT / path) for arm, path in probe_step["reports"].items()} if probe_step["reports"] else None)
    checks = {"smoke_failure_present": gate_record["smoke_outcome"] == "failure_present",
              "probe_reports_bound": bool(reports) and _bind_reports(reports, gate_record),
              "probes_complete": bool(reports) and _complete(reports),
              "model_resolved": model.get("status") in ("valid", "model_invalid", "model_failed")}
    result = {"schema": "upstream_optimizer_fault_loop_result_v1", "tag": tag, "protocol_sha256_lf": U.sha(PROTOCOL, True),
              "source_commit_sha": commit, "smoke": {"tag": gate_record["smoke_tag"], "outcome": gate_record["smoke_outcome"]},
              "truth_category": TRUTH, "checks": checks, "model": {k: model.get(k) for k in ("status", "reason", "decision", "attempts")},
              "model_calls": model.get("attempts"), "model_tokens": sum((c.get("total_tokens") or 0) for c in model.get("calls") or []),
              "probe_attempts": probe_step["attempt"], "probe_process_wall_s": probe_step.get("process_wall_s"),
              "scope": FIXED["scope"]}
    if not all(checks.values()):
        result.update(outcome="invalid", failed_checks=[k for k, ok in checks.items() if not ok], table=None, strategies=None,
                      wording=FIXED["wording"]["invalid"])
    else:
        table = classify_table(reports)
        rows = strategy_rows(table, model)
        outcome = overall(rows)
        result.update(outcome=outcome, failed_checks=[], table=table,
                      measurements={arm: {p: r.get("measurement") for p, r in reports[arm]["probes"].items()} for arm in S.ARMS},
                      strategies=rows, specificity_gate=specificity(rows, table),
                      counts={"correct": sum(r["correct"] for r in rows), "wrong": sum(r["wrong"] for r in rows),
                              "stopped": sum(r["status"] in ("none_supported", "unidentifiable") for r in rows)},
                      prediction_match=prediction_check(table, rows, outcome), wording=FIXED["wording"][outcome])
    U.write_new(RESULTS / tag / "result.json", result)
    return {"outcome": result["outcome"], "result": f"evals/results/{tag}/result.json"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "gate", "packet", "rank", "probes", "summarize"))
    parser.add_argument("--tag")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command not in ("prepare", "verify") and not args.tag:
        parser.error(f"{args.command}에는 --tag가 필요하다")
    if args.command == "rank" and not args.record:
        parser.error("rank에는 --record가 필요하다")
    import record_evidence_media as M
    try:
        out = {"prepare": lambda: prepare(args.isaaclab),
               "verify": lambda: {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]},
               "gate": lambda: gate(args.tag, args.isaaclab), "packet": lambda: packet(args.tag, args.isaaclab),
               "rank": lambda: rank(args.tag, args.record, args.isaaclab), "probes": lambda: probes(args.tag, args.isaaclab),
               "summarize": lambda: summarize(args.tag, args.isaaclab)}[args.command]()
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("status") == "gpu_busy_no_launch" else 0


if __name__ == "__main__":
    sys.exit(main())
