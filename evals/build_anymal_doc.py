"""두 번째 로봇 task(ANYmal-C flat) 이식 공개 문서(docs/ANYMAL-C-PORT.md)를 커밋된 파일에서 만든다. 문서의 수치는 손으로 적지 않는다.

  python evals/build_anymal_doc.py                                              # 단계 0·1 결과와 단계 2 사전등록까지
  python evals/build_anymal_doc.py --demo evals/results/<tag>/demo_result.json  # 단계 2 시연 결과 절까지 채운다
  python evals/build_anymal_doc.py [--demo …] --check                           # 쓰지 않고 지금 문서와 같은지만 본다

읽는 파일: 단계마다 사전등록 프로토콜과 결과(단계 0 관문, 단계 1 밴드, 단계 2 시연 결과·고정 평가 보고서). 결과가 그 단계
프로토콜의 것이 아니거나 단계 2 프로토콜이 결합한 단계 1 결과와 다르면 거부한다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "docs/ANYMAL-C-PORT.md"
STAGE0 = {"protocol": "bench/protocols/anymal_c_gate_v1.json", "result": "evals/results/anymal_c_gate_20261014/result.json"}
STAGE1 = {"protocol": "bench/protocols/anymal_c_band_v1.json", "result": "evals/results/anymal_c_band_20261014/result.json"}
STAGE2 = {"protocol": "bench/protocols/e2e_demo_anymal_c_f1.json"}
DEMO_SCHEMA = "e2e_demo_result_anymal_c_v1"
CHECK_NAMES = {"survival_ratio": "생존", "fall_frac": "낙상", "err_xy_ratio": "선속도", "err_yaw_ratio": "yaw"}
MEASURE_KEYS = {"P_noise": "noise_ratio", "P_value": "critic_change", "P_reward": "track_lin_vel_xy_exp_rel_error",
                "P_torque": "low_speed_saturation", "P_physics": "mismatch_count", "P_episode": "timeout_ratio"}
PREDICTION_NAMES = {"source_failure": "학습 실패", "alarm": "경고 R1", "diagnosis": "첫 판단", "loop": "검사 고리",
                    "specificity": "특이성 관문", "correction_rule": "회복 방식 규칙", "correction": "교정 결과"}


def read(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def r(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if value != 0 and abs(value) < 10 ** -digits:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def bound(root: Path, stage: dict) -> tuple[dict, dict, str]:
    """단계 프로토콜과 결과. 결과의 protocol_sha256_lf가 프로토콜 LF SHA와 다르면 거부한다."""
    protocol, result = read(root, stage["protocol"]), read(root, stage["result"])
    sha = lf_sha(root / stage["protocol"])
    if result.get("protocol_sha256_lf") != sha:
        raise ValueError(f"{stage['result']}가 {stage['protocol']}의 결과가 아니다")
    return protocol, result, sha


def files_table(root: Path, rows: list[tuple[str, str]]) -> list[str]:
    out = ["| 근거 | 파일 | LF SHA256 |", "|---|---|---|"]
    out += [f"| {label} | `{relative}` | `{lf_sha(root / relative)[:12]}…` |" for label, relative in rows]
    return out + [""]


def intro(stage2: dict) -> list[str]:
    return [
        "# 두 번째 로봇 task 이식: ANYmal-C flat", "",
        "이 문서의 수치는 `evals/build_anymal_doc.py`가 단계별 사전등록 프로토콜과 결과 파일에서 만든다. 손으로 고치지 않는다.", "",
        f"Go2 flat에서 만든 고리를 다른 제조사 로봇 task `{stage2['task']}`에서 같은 코드로 돌렸다. 고리는 동결 probe 6종, "
        "고정 평가 계약, 등록 교정, 추종 경고 R1, 모델 결정 경로를 말한다. 단계마다 실행 전에 프로토콜을 커밋했고, 앞 단계가 "
        "pass일 때만 다음 단계를 사전등록했다.", "",
        "- **단계 0 관문:** 새 seed 정상 학습 1회로 걷기·probe 측정·절대 기준 probe·학습 비용을 확인한다.",
        "- **단계 1 밴드:** 정상 seed 2개를 더한 정상 실행 3개로 판정 밴드를 정하고, 비 기준 probe와 경고 R1이 정상 실행끼리 "
        "오경보를 내지 않는지 본다.",
        "- **단계 2 F1 종단 시연:** 보상 좌표 결함 하나를 시연 v4와 같은 흐름으로 끝까지 연결한다.", "",
        "범위:",
        "- 결함·probe·교정을 같은 설계자가 만들었다. 로봇 하나·seed 몇 개의 기록이다.",
        "- 진단 정확도·교정 성공률·일반화·다른 방법 대비 우위·GPU 절감을 주장하지 않는다. Go2 결과와 수치를 비교하지 않는다"
        "(task·로봇이 다르다).",
        "- 승인은 사용자가 위임한 운영자 승인 또는 사람 승인이다. 승인자 문자열만 기록하며 신원 인증이 아니다.", "",
    ]


def what_changed(stage0: dict, eval_protocol: dict) -> list[str]:
    return [
        "## 바꾼 것과 그대로 둔 것", "",
        "| 부품 | 이 task에서 | Go2와 같은가 |", "|---|---|---|",
        f"| 학습 | `bench/train_with_fault.py`(`TRIAGE_FAULT`, `--task {stage0['task']}`), 등록 학습 설정 그대로(action scale "
        f"{stage0['action_scale']}, push·COM 무작위화 켬) | 진입은 같고 task만 다르다 |",
        f"| 고정 평가 | `{eval_protocol['name']}`: 명령 격자 26개, 지평 {eval_protocol['horizon_steps']} step, 평가 seed "
        f"{eval_protocol['eval_seed']}이 Go2 계약과 같다. 평가 환경에서만 학습 이벤트 "
        f"{'·'.join(eval_protocol['env_events_disabled'])}을 끈다(`evals/fixed_eval_task.py`) | 계약은 같고 평가 환경 이벤트 둘을 끈다 |",
        "| probe 6종 | 동결 `evals/probes.py`의 task 상수만 바꾸는 구동기(`evals/probes_task.py`, 고리 측정은 "
        "`evals/metered_probes_task.py`). 측정 수식·인자·임계값은 그대로 | 같다 |",
        "| 판정 밴드 | 이 task의 정상 실행 3개(단계 1)로 P0-C와 같은 방식으로 정했다 | 방식만 같고 값은 새로 정했다 |",
        "| 경고 R1 | 규칙 상수 그대로, 참조는 이 task의 정상 실행 3개 | 규칙은 같다 |",
        "| 진단 packet | 사실·카탈로그·임계값·예산은 같고 task 문구만 ANYmal-C | 문구 외 같다 |",
        "| 교정·원장·결정 경로 | 시연 v4의 회복 방식 규칙·등록 표·교정 원장·검증 피드백 1회 재요청 | 같다 |", "",
    ]


def stage_table(results: dict) -> list[str]:
    rows = ["## 단계별 결과", "", "| 단계 | 질문 | 결과 | 실험 커밋 | 공개 결과 |", "|---|---|---|---|---|"]
    for name, (question, outcome, commit, path) in results.items():
        rows.append(f"| {name} | {question} | {outcome} | {commit} | `{path}` |")
    return rows + [""]


def stage0_section(result: dict) -> list[str]:
    gate, behavior = result["gate"], result["gate"]["behavior"]
    walls = result["parent_wall_s"]
    ratio = gate["ratio_probe_measurements"]
    return [
        f"## 단계 0: 이식 관문 (seed {result['seed']})", "",
        f"결과 **`{result['outcome']}`**(예측 {result['prediction']}, 적중). 사전등록 결과 문구: \"{result['wording']}\"", "",
        f"- 유효 조건 {sum(result['checks'].values())}/{len(result['checks'])} 참. 학습 프로세스의 결함 readback은 "
        f"`{result['readback'][0]['fault']}`이었다. 평가 환경 readback에서 {'·'.join(result['eval_env'][0]['disabled'])}은 학습 설정에서 "
        "켜져 있다가 평가 환경에서 꺼졌다.",
        f"- 고정 평가: 정지 대비 선속도 오차 비 {r(result['gate']['standstill_ratio'])}, 낙상 비율 {r(behavior['fall_frac'])}, "
        f"생존 {r(behavior['survival_s'], 2)} / {r(behavior['horizon_s'], 1)} s → 걷는다(기준 0.5 미만·0.05 이하).",
        f"- probe 6종이 모두 측정됐다. 절대 기준 4종({'·'.join(gate['absolute_probes'])})이 "
        f"{'모두 normal이다' if set(gate['absolute_probes'].values()) == {'normal'} else gate['absolute_probes']}.",
        f"- 비 기준 probe 측정값(판정은 단계 1): P_value critic 변화 {r(ratio['P_value']['critic_change'])}, P_torque 저속 포화 "
        f"{r(ratio['P_torque']['low_speed_saturation'])}.",
        f"- 비용(부모 프로세스 벽시계, 기동 포함): 학습 {r(walls['train'], 1)} s(반복당 {r(result['train_wall_per_iteration_s'])} s), "
        f"고정 평가 {r(walls['fixed_eval'], 1)} s, probe 6종 {r(walls['probes'], 1)} s. 모델 호출 {result['model_calls']}회.", "",
    ]


def stage1_section(protocol: dict, result: dict) -> list[str]:
    a = result["analysis"]
    bands = a["bands"]
    seeds = [str(s) for s in result["seeds"]]
    rows = ["| seed | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 대비 비 | critic 변화 | 저속 포화 | R1 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for seed in seeds:
        b, m = a["behavior"][seed], a["ratio_measurements"][seed]
        rows.append(f"| {seed} | {r(b['err_xy'])} | {r(b['err_yaw'])} | {r(b['fall_frac'])} | {r(b['survival_s'], 2)} | "
                    f"{r(a['standstill_ratio'][seed])} | {r(m['P_value']['critic_change'])} | "
                    f"{r(m['P_torque']['low_speed_saturation'])} | {'울림' if a['tracking_alarm'][seed]['warned'] else '울리지 않음'} |")
    normal = sum(1 for row in a["ratio_probes"] if row["outcome"] == "normal")
    walls = {s: rec["parent_wall_s"] for s, rec in result["seed_records"].items()}
    return [
        f"## 단계 1: 정상 실행 3개와 판정 밴드 (seed {', '.join(seeds)})", "",
        f"결과 **`{result['outcome']}`**(예측 {result['prediction']}, 적중). 사전등록 결과 문구: \"{result['wording']}\"", "",
        *rows, "",
        f"- 사전등록 기준 {len(a['criteria'])}개 중 {sum(a['criteria'].values())}개가 참이다.",
        *[f"  - {'참' if ok else '거짓'}: {protocol['criteria'][name]}" for name, ok in a['criteria'].items()],
        f"- 비 기준 probe를 순서쌍 {len(a['ratio_probes']) // 2}개마다 동결 classify에 넣은 {len(a['ratio_probes'])}개 중 "
        f"{normal}개가 normal이다. R1이 울린 정상 실행은 {sum(1 for s in seeds if a['tracking_alarm'][s]['warned'])}개다"
        f"(경고 조건을 채운 반복 수 seed별 {', '.join(str(a['tracking_alarm'][s]['qualifying_iteration_count']) for s in seeds)}).",
        f"- 판정 밴드(순서쌍 {len(a['comparisons'])}개, 여유 비 0.10·낙상 0.05): 생존 비 ≥ {r(bands['survival_ratio_min'])}, "
        f"선속도 오차 비 ≤ {r(bands['err_xy_ratio_max'])}, yaw 오차 비 ≤ {r(bands['err_yaw_ratio_max'])}, 낙상 비율 ≤ "
        f"{r(bands['fall_frac_max'])}. 단계 2 판정에 이 값을 쓴다.",
        "- 비용(부모 프로세스 벽시계): " + "; ".join(
            f"seed {s} 학습 {r(w['train'], 1)} s·고정 평가 {r(w['fixed_eval'], 1)} s·probe {r(w['probes'], 1)} s" for s, w in walls.items())
        + f". 모델 호출 {result['model_calls']}회.", "",
    ]


def stage2_prereg(protocol: dict, protocol_sha: str) -> list[str]:
    rule = protocol["recovery_rule"]
    return [
        f"## 단계 2: 보상 좌표 결함(F1) 종단 시연 (seed {protocol['seed']})", "",
        f"- 사전등록 `{STAGE2['protocol']}`(LF SHA256 `{protocol_sha[:8]}…`), 실행기 `evals/e2e_demo_anymal_c.py`, 교정 실행기 "
        "`bench/demo_correction_anymal_c.py`.",
        f"- 결함 `{protocol['fault']}`(`bench/hidden_faults.py`, 동결), 사례 `{protocol['cases']['FAULT']}`·"
        f"`{protocol['cases']['NONE']}`, 교정 재학습 seed {protocol['restart_seed']}.",
        "- 흐름은 시연 v4와 같다: 경고 R1 → 고정 평가 → 새 Nemotron 판단(검증 피드백 재요청 최대 1회) → 승인된 검사 → 확정 → "
        "교정 제안(같은 재요청) → 승인된 교정 → 재학습 → 고정 평가.",
        f"- 판정 밴드와 R1 참조는 단계 1 결과에서 오고 SHA로 묶었다(`{protocol['stage1']['result']['path']}`). 단계 1이 pass가 아니면 "
        "시작하지 않도록 사전등록했다.",
        f"- 회복 방식 규칙: {rule['mechanism']}에서 정지 대비 오차 비 ≥ {rule['threshold']}이면 {rule['at_or_above']}, 아니면 "
        f"{rule['below']}(Go2 시연과 같은 규칙).",
        "- 사전 예측:",
        *[f"  - {PREDICTION_NAMES[k]}: {v}" for k, v in protocol["predictions"].items()], "",
    ]


def _measure(probe: str, measurement: dict | None) -> str:
    key = MEASURE_KEYS.get(probe)
    return f"{key} {r((measurement or {}).get(key))}" if key else "—"


def _call_row(name: str, view: dict | None, call: dict | None) -> str:
    if not view or not call:
        return f"| {name} | — | — | — | — | — | — |"
    yes = {True: "예", False: "아니오"}
    return (f"| {name} | {yes[view['first_valid']]} | {view['retries']}회 | {yes[view['final_valid']]} | {call.get('requests')} | "
            f"{call.get('input_tokens'):,}·{call.get('output_tokens'):,} | {r(call.get('execution_wall_s'), 2)} s |")


def _verdict(verdict: dict | None) -> str:
    if not verdict:
        return "—"
    failed = [CHECK_NAMES[k] for k in verdict["failed"]]
    return f"{verdict['label']}" + (f"({' · '.join(failed)} 실패)" if failed else "(네 검사 모두 통과)")


def stage2_result(root: Path, demo: dict, protocol: dict) -> list[str]:
    out = ["### 결과", ""]
    unmet = demo["unmet_conditions"]
    success = "시연 성공" if demo["demo_success"] else "시연 성공 아님"
    out += [f"결과는 **`{demo['outcome']}`, {success}**이다. 사전등록 결과 문구: \"{demo['wording']}\"", ""]
    if demo.get("terminal"):
        out += [f"- 시연은 `{demo['terminal']['outcome']}`로 끝났다(종료 기록 {demo['terminal']['at']}).", ""]
    feedback, diagnosis, fix = demo["model_feedback"], demo.get("diagnosis"), demo.get("correction")
    out += ["| 결정 단계 | 1차 유효 | 재요청 | 최종 유효 | 요청 수 | 토큰(입력·출력) | 대화 전체 |", "|---|---|---|---|---|---|---|",
            _call_row("첫 판단", feedback["diagnosis"], (diagnosis or {}).get("call")),
            _call_row("교정 제안", feedback["correction"], (fix or {}).get("call")), ""]
    steps = ["| 단계 | 기록 |", "|---|---|"]
    walls = demo["training_parent_wall_s"]
    steps.append(f"| 정상·결함 학습 | 학습 벽시계 NONE {r(walls['NONE'], 1)} s · F1 {r(walls['FAULT'], 1)} s |")
    alarm = demo.get("alarm") or {}
    if alarm:
        fault = alarm.get("FAULT", {})
        steps.append(f"| 경고 R1 | F1 {'경고(첫 경고 iteration ' + str(fault.get('first_alert_iteration')) + ')' if fault.get('warned') else '경고 없음'}, "
                     f"NONE {'경고' if alarm.get('NONE', {}).get('warned') else '경고 없음'} |")
    gate = (demo.get("evaluation") or {}).get("gate")
    if gate:
        steps.append(f"| 고정 평가 | NONE {'보행 통과' if gate['none_walks'] else '보행 실패'}, F1 {_verdict(gate['fault_verdict'])} → "
                     f"`{gate['outcome']}` |")
    if diagnosis:
        decision, started = diagnosis["decision"], diagnosis.get("started") or {}
        claims = ", ".join(f"`{c['id']}` {r(c['value'])}" for c in decision.get("claims", []))
        steps.append(f"| Nemotron 첫 판단 | `{decision['action']}`. 순위 {' > '.join(decision.get('ranking', []))}, probe "
                     f"`{decision.get('probe')}`(동결 규칙 대안 `{started.get('rule_probe')}`). 인용 근거 {claims or '—'} |")
    loop = demo.get("loop")
    if loop:
        for row in loop["requests"]:
            what = (f"`{row['probe']}` {row.get('outcome')}({_measure(row['probe'], row.get('measurement'))}), GPU 벽시계 "
                    f"{r(row.get('gpu_s'), 1)} s" if row["state"] == "done" else f"`{row['probe']}` {row['state']}")
            steps.append(f"| 검사 `{row['request_id']}` | {what} |")
        steps.append(f"| 고리 종료 | `{loop['status']}`, 남은 가설 {', '.join(loop['hypotheses']) or '없음'} "
                     f"(검사 {sum(1 for x in loop['requests'] if x['state'] == 'done')}회, 예산 {protocol['loop']['budget_probes']}) |")
    if demo.get("specificity"):
        steps.append("| 특이성 관문 | 게이트 없음(reward), 통과로 기록 |" if not demo["specificity"]["required"]
                     else f"| 특이성 관문 | {demo['specificity']['outcomes']} |")
    if fix:
        proposal, request, receipt, result = fix["model_proposal"], fix["request"], fix["receipt"], fix.get("result") or {}
        steps.append(f"| Nemotron 교정 제안 | 검증 {'accepted' if proposal['accepted'] else 'rejected'}, 규칙과 "
                     f"{'일치' if proposal.get('agrees_with_rule') else '불일치'}. 요청은 `{request['correction_id']}` + "
                     f"`{request['recovery_mode']}`(예산 {request['budget_gpu_s']} s) |")
        steps.append(f"| 교정 요청 `{fix['request_id']}` | receipt `{receipt.get('verdict')}`, gpu_s {r(receipt.get('gpu_s'), 1)}"
                     f"{', 예산 초과' if receipt.get('over_budget') else ''} |")
        steps.append(f"| 교정 재학습·고정 평가 | {result.get('recovery_mode')}, {_verdict(result.get('verdict'))}, 정지 대비 비 "
                     f"{r(result.get('standstill_ratio'))} |")
    out += [*steps, ""]
    reports = ((demo.get("evaluation") or {}).get("reports")) or {}
    if reports:
        out += ["고정 평가:", "", "| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 대비 비 |",
                "|---|---|---|---|---|---|"]
        for arm, label in (("NONE", f"NONE s{demo['seed']}"), ("FAULT", f"F1 s{demo['seed']}")):
            report = read(root, reports[arm]["path"])
            if lf_sha(root / reports[arm]["path"]) != reports[arm]["sha256_lf"]:
                raise ValueError(f"{reports[arm]['path']}가 시연 결과에 기록된 SHA와 다르다")
            m = report["metrics"]
            out.append(f"| {label} | {r(m['lin_vel_rmse_mps'])} | {r(m['yaw_rate_rmse_radps'])} | {r(m['fall_rate'])} | "
                       f"{r(m['mean_survival_s'], 2)} | {r(m['lin_vel_rmse_mps'] / report['standstill']['lin_vel_rmse_mps'])} |")
        if fix and (fix.get("result") or {}).get("standstill_ratio") is not None:
            out.append(f"| {fix['result']['recovery_mode']} 교정 | — | — | — | — | {r(fix['result']['standstill_ratio'])} |")
        out.append("")
    hits = demo["prediction_results"]
    out += [f"사전 예측 {len(hits)}개 중 적중 {sum(1 for v in hits.values() if v is True)}, 빗나감 "
            f"{sum(1 for v in hits.values() if v is False)}, 그 단계까지 가지 못함 {sum(1 for v in hits.values() if v is None)}:",
            *[f"- {PREDICTION_NAMES[k]}: {'적중' if v is True else '빗나감' if v is False else '미도달'}" for k, v in hits.items()], "",
            f"성공 조건 {len(demo['success_conditions']) - len(unmet)}/{len(demo['success_conditions'])} 참"
            + (f"(거짓: {', '.join(unmet)})." if unmet else "."), ""]
    costs = []
    if loop and loop.get("probe_costs"):
        costs.append("- 고리 검사: " + "; ".join(f"`{c['probe']}` GPU 벽시계 {r(c['gpu_wall_s'], 1)} s·CPU {r(c['cpu_s'], 1)} s·"
                                               f"시뮬레이터 {c['simulator_steps']:,} step" for c in loop["probe_costs"]))
    if demo.get("reference_parent_wall_s") is not None:
        costs.append(f"- 정상 기준 probe 6종 {r(demo['reference_parent_wall_s'], 1)} s")
    fix_costs = ((fix or {}).get("result") or {}).get("costs") or {}
    if fix_costs:
        costs.append(f"- 교정 학습 {r(fix_costs.get('train_parent_wall_s'), 1)} s, 교정 정책 고정 평가 "
                     f"{r(fix_costs.get('eval_parent_wall_s'), 1)} s")
    if costs:
        out += ["비용(부모 프로세스 벽시계):", *costs, ""]
    out += ["해석 경계:", f"- {demo['common_wording']}",
            "- 판정 밴드는 정상 실행 3개로 정했다. 밴드 폭은 이 세 실행의 변동에 기댄다.", ""]
    return out


def build(root: Path, demo_path: str | None) -> str:
    stage0_protocol, stage0, _ = bound(root, STAGE0)
    stage1_protocol, stage1, _ = bound(root, STAGE1)
    stage2 = read(root, STAGE2["protocol"])
    stage2_sha = lf_sha(root / STAGE2["protocol"])
    if stage2["stage1"]["result"]["sha256_lf"] != lf_sha(root / STAGE1["result"]):
        raise ValueError("단계 2 프로토콜이 결합한 단계 1 결과가 지금 파일과 다르다")
    eval_protocol = read(root, "bench/protocols/fixed_eval_v1_anymal_c.json")
    demo = None
    if demo_path is not None:
        demo = read(root, demo_path)
        if demo.get("schema") != DEMO_SCHEMA or demo.get("protocol_sha256_lf") != stage2_sha:
            raise ValueError("시연 결과가 단계 2 프로토콜의 결과가 아니다")
    results = {
        "0 관문": ("정상 학습 1회가 걷고 probe가 재지는가", f"`{stage0['outcome']}`", f"`{stage0['source_commit_sha'][:7]}`",
                 STAGE0["result"]),
        "1 밴드": ("정상 실행 3개로 밴드를 정하고 오경보가 없는가", f"`{stage1['outcome']}`", f"`{stage1['source_commit_sha'][:7]}`",
                 STAGE1["result"]),
        "2 F1 시연": ("결함 하나를 끝까지 연결하는가",
                     "진행 중" if demo is None else f"`{demo['outcome']}`" + (", 시연 성공" if demo["demo_success"] else ""),
                     "—" if demo is None else ", ".join(f"`{c[:7]}`" for c in demo["source_commit_sha"]),
                     "—" if demo is None else demo_path),
    }
    lines = [*intro(stage2), *what_changed(stage0_protocol, eval_protocol), *stage_table(results), *stage0_section(stage0),
             *stage1_section(stage1_protocol, stage1), *stage2_prereg(stage2, stage2_sha)]
    lines += (["### 결과", "", "시연이 아직 끝나지 않았다. 결과가 커밋되면 `python evals/build_anymal_doc.py --demo <결과 경로>`로 이 절을 "
               "채운다.", ""] if demo is None else stage2_result(root, demo, stage2))
    lines += ["## 주장하지 않는 것", "",
              *[f"- {claim}" for claim in stage2["forbidden_claims"]],
              "- 정상 실행 3개(단계 0·1)와 시연 seed 하나(단계 2)로 이 task 전반의 동작을 말하지 않는다.", "",
              "## 근거 파일", "",
              *files_table(root, [("단계 0 프로토콜", STAGE0["protocol"]), ("단계 0 결과", STAGE0["result"]),
                                  ("단계 1 프로토콜", STAGE1["protocol"]), ("단계 1 결과", STAGE1["result"]),
                                  ("단계 2 프로토콜", STAGE2["protocol"]), ("고정 평가 프로토콜", "bench/protocols/fixed_eval_v1_anymal_c.json"),
                                  *([("단계 2 시연 결과", demo_path)] if demo_path else [])]),
              "## 재현", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe evals/build_anymal_doc.py" + (f" --demo {demo_path}" if demo_path else "") + " --check",
              "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        content = build(ROOT, args.demo)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    output = ROOT / args.output
    if args.check:
        same = output.is_file() and output.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if same else "stale"}, ensure_ascii=False))
        return 0 if same else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "path": args.output, "demo": args.demo}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
