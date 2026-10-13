"""D3 공개 문서(docs/D3-MULTI-CHANGE.md)를 커밋된 파일에서 만든다. 문서의 수치는 손으로 적지 않는다.

  python evals/build_d3_doc.py                                   # 요약 전: 사전등록·smoke 절만 채우고 결과 절은 비운다
  python evals/build_d3_doc.py --summary evals/results/<tag>/summary.json   # 결과 절까지 채운다
  python evals/build_d3_doc.py --summary <…> --check              # 쓰지 않고 지금 문서와 같은지만 본다

읽는 파일: 실험 프로토콜(bench/protocols/d3_experiment_v2.json), 무해성 smoke v2 프로토콜과 v1·v2 요약, D3 요약과
요약이 가리키는 사례 결과(case_result.json). 요약·사례 결과가 그 프로토콜의 것이 아니면 거부한다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "docs/D3-MULTI-CHANGE.md"
PROTOCOL = "bench/protocols/d3_experiment_v2.json"
SMOKE_PROTOCOL = "bench/protocols/d3_harmless_smoke_v2.json"
SMOKE_V2_SUMMARY = "evals/results/d3_harmless_summary_20261014_v2/summary.json"
METHODS = ("B1", "B2", "B3", "B4", "B5")
NAMES = {"B1": "이분 탐색", "B2": "하나씩 되돌리기", "B3": "probe 범주 규칙", "B4": "모델 순위", "B5": "모델 순위(probe 없음)"}
OUTCOME_NAMES = {"agent_fewer_retrains": "재학습 감소", "no_advantage": "우위 없음", "agent_worse": "열세", "invalid": "판정 없음"}
CONDITION_KEYS = ("fewer_than_every_deterministic_sum", "never_more_than_b3_in_a_case", "no_wrong_or_unresolved",
                  "three_valid_cases")
APPROVAL = "사용자가 위임한 운영자 승인 또는 사람 승인(승인자 문자열 기록, 신원 인증 아님)"


def read(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def r(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def smoke_rows(summary: dict) -> list[str]:
    out = []
    for seed, row in summary["seeds"].items():
        c = row["comparison"]
        out.append(f"seed {seed}: {row['verdict']}, 선속도 오차 비 {r(c['err_xy_ratio'])}, yaw {r(c['err_yaw_ratio'])}, "
                   f"낙상 {r(c['fall_frac'], 2)}")
    return out


def prereg_section(protocol: dict, smoke_protocol: dict, v1: dict, v2: dict) -> list[str]:
    cases, computed = protocol["cases"], protocol["predictions"]["computed"]
    counted = [c for c, s in cases.items() if s["aggregate"]]
    sums = {m: sum(computed[c][m] for c in counted) for m in ("B1", "B2", "B3")}
    harmful = {cid: protocol["changes"][cid] for cid in dict.fromkeys(s["harmful"] for s in cases.values())}
    harmless = [cid for cid in next(iter(cases.values()))["order"] if protocol["changes"][cid]["role"] != "harmful"]
    history = smoke_protocol["version_history"]
    rule_code, _, rule_text = protocol["position_rule"].partition(". ")
    lines = [
        "## 질문", "",
        "로봇 RL 엔지니어가 설정과 코드를 여러 군데 바꾼 뒤 학습했더니 정책이 무너졌다. 바뀐 것 중 무엇이 원인인지 확정하려면 결국 "
        "그 변경만 되돌리고 다시 학습해 봐야 한다. 이 실험은 그 확인 재학습이 몇 번 필요한지를 방법별로 센다.", "",
        f"- 실패한 학습 하나에 변경 {len(next(iter(cases.values()))['order'])}개가 섞여 있고, 그중 하나만 학습을 망가뜨린다.",
        f"- 같은 재학습 상한({protocol['cap']}회)과 같은 확정 규칙 아래에서 다섯 방법을 비교한다.",
        f"- 범위: {protocol['scope']}.", "",
        "## 사례", "",
        "Go2 flat, 4096 env × 300회 학습이다. 원인 변경의 위치는 결과를 보기 전에 문자열 seed 난수로 정했다"
        f"(`{rule_code}`). {rule_text}.", "",
        "| 사례 | 학습 seed | 원인 변경 | 위치 | 집계 |", "|---|---|---|---|---|",
        *[f"| {c} | {s['seed']} | {s['harmful']} | {s['position']} / {len(s['order'])} | "
          f"{'집계' if s['aggregate'] else '개발 사례(집계하지 않음)'} |" for c, s in cases.items()], "",
        "원인 변경:", "",
        *[f"- {cid}: {c['definition']}" + (f" 출처: {', '.join(c['sources'])}" if cid == "U1" else "") for cid, c in harmful.items()],
        "",
        "무해 변경(원인 변경과 같은 범주의 변경을 일부러 넣었다):", "",
        *[f"- {cid}({protocol['changes'][cid]['category']}): {protocol['changes'][cid]['definition']}" for cid in harmless], "",
        "## 다섯 방법과 확정 규칙", "",
        f"확정 규칙: {protocol['confirmation_rule']}. 상한 {protocol['cap']}회 안에 확정하지 못하면 미확정이다.", "",
        "| 기호 | 방법 | 정의 |", "|---|---|---|",
        *[f"| {m} | {NAMES[m]} | {protocol['methods'][m].split(': ', 1)[-1]} |" for m in METHODS], "",
        f"- 모델: `{protocol['model']['id']}`, {protocol['model']['call']}. {protocol['model']['contract']}.",
        f"- 비용: {protocol['costs']['main']}. 같은 seed·같은 부분집합의 재학습은 한 번만 하고 결과를 방법끼리 같이 쓴다"
        f"({protocol['cache']['reuse']}).",
        f"- 재학습 승인: {APPROVAL}.", "",
        "## 사전 예측과 결정적 기준선", "",
        "결과를 보기 전에 계산한 결정적 방법의 확인 재학습 수다. 변경끼리 상호작용이 없고 검사가 원인 범주를 맞힌다고 가정했다.", "",
        "| 사례 | B1 이분 탐색 | B2 하나씩 | B3 probe 범주 규칙 |", "|---|---|---|---|",
        *[f"| {c} | {computed[c]['B1']} | {computed[c]['B2']} | {computed[c]['B3']} |" for c in counted],
        f"| 합 | {sums['B1']} | {sums['B2']} | {sums['B3']} |", "",
        *[f"- 검사 예측({cid}): {text}" for cid, text in protocol["predictions"]["loop"].items() if cid in harmful], "",
        "## 우위로 말하는 조건", "",
        "모델 순위(B4)가 결정적 기준선보다 재학습을 줄였다고 말하려면 다음을 모두 만족해야 한다.", "",
        *[f"- {text}" for text in protocol["advantage_conditions"]], "",
        "결과는 네 가지 중 하나로만 쓴다.", "",
        "| 결과 | 문구 |", "|---|---|",
        *[f"| {OUTCOME_NAMES[k]}({k}) | {v} |" for k, v in protocol["outcomes"].items()], "",
        "## 무해 변경 목록: 무해성 smoke v1 실패와 v2 교체", "",
        f"본실험 전에 무해 변경을 모두 적용해 이미 본 seed에서 학습하고, p0c 판정에 더해 여유 기준(오차 비 "
        f"{v2['margins']['err_xy_ratio_max']} 이하, 낙상 {v2['margins']['fall_frac_max']} 이하)으로 걸렀다.", "",
        f"- v1(무해 목록 {', '.join(v1['changes'])}): {v1['outcome']}", *[f"  - {row}" for row in smoke_rows(v1)],
        f"- 해석: {history['v1']['reading']}",
        f"- v2(무해 목록 {', '.join(v2['changes'])}): {v2['outcome']}", *[f"  - {row}" for row in smoke_rows(v2)],
        f"- 교체 규칙: {history['v2_change']['rule']}. {history['v2_change']['caveat']}.", "",
    ]
    return lines


def result_section(root: Path, summary: dict, protocol: dict, protocol_sha: str) -> list[str]:
    results = {}
    for case, relative in summary["cases"].items():
        result = read(root, relative)
        if result.get("protocol_sha256_lf") != protocol_sha:
            raise ValueError(f"{case} 사례 결과가 이 프로토콜의 것이 아니다")
        results[case] = result
    lines = ["## 결과", "", f"요약: `{summary['tag']}`", "",
             "| 사례 | " + " | ".join(f"{m} {NAMES[m]}" for m in METHODS) + " | B4 1위 |",
             "|---|" + "---|" * (len(METHODS) + 1)]
    for case, result in results.items():
        costs = summary["costs"].get(case)
        top = ((result["methods"].get("B4") or {}).get("model") or {}).get("ranking_labels") or [None]
        shown = f"{top[0]} = {result['labels'][top[0]]}" if top[0] else "-"
        if costs is None:
            reason = result.get("invalid_reason") or ("개발 사례" if not result["aggregate"] else "-")
            lines.append(f"| {case} | " + " | ".join("-" for _ in METHODS) + f" | 집계 제외: {reason} |")
        else:
            lines.append(f"| {case} | " + " | ".join(str(costs[m]["retrains"]) for m in METHODS) + f" | {shown} |")
    lines += ["", "| 항목 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS),
              "| 확인 재학습 합 | " + " | ".join(str(summary["sums"][m]) for m in METHODS) + " |",
              "| 틀린 확정 | " + " | ".join(str(summary["wrong_confirmations"][m]) for m in METHODS) + " |",
              "| 미확정 | " + " | ".join(str(summary["unresolved"][m]) for m in METHODS) + " |",
              "| 확인 재학습 부모 벽시계 합(초) | " + " | ".join(
                  str(round(sum(r["methods"][m]["retrain_parent_wall_s"] for c, r in results.items() if c in summary["costs"]), 1))
                  for m in METHODS) + " |",
              "| 기준·고리 probe 부모 벽시계 합(초) | " + " | ".join(
                  str(round(sum(r["methods"][m]["probe_parent_wall_s"] for c, r in results.items() if c in summary["costs"]), 1))
                  for m in METHODS) + " |",
              "| 모델 토큰 합 | - | - | - | " + " | ".join(
                  str(sum((r["methods"][m].get("model") or {}).get("total_tokens") or 0 for c, r in results.items()
                          if c in summary["costs"])) for m in ("B4", "B5")) + " |", "",
              "우위 조건:", "",
              *[f"- {text}: {'충족' if summary['conditions'][key] else '미충족'}"
                for key, text in zip(CONDITION_KEYS, protocol["advantage_conditions"])], "",
              f"**결과: {OUTCOME_NAMES[summary['outcome']]}({summary['outcome']}).** {summary['wording']}", "",
              "- 모든 방법이 같이 쓰는 비용(방법 비용에 넣지 않음, NONE · 실패 실행의 학습과 고정 평가 부모 벽시계): " + ", ".join(
                  f"{c} {r['shared_parent_wall_s']['none']} · {r['shared_parent_wall_s']['failure']}초"
                  for c, r in results.items() if c in summary["costs"]) + ".", ""]
    if summary["excluded_cases"]:
        lines += [f"- 집계 제외: {', '.join(f'{c}({why})' for c, why in summary['excluded_cases'].items())}", ""]
    return lines


def determinism_line(root: Path, summary: dict | None) -> str:
    """개발 사례 D0의 결정성 대조 결과. 아직 없으면 할 일로 적는다."""
    relative = (summary or {}).get("cases", {}).get("D0")
    record = read(root, relative).get("determinism") if relative else None
    if not record:
        todo = "개발 사례 D0에서 한 번 대조한다" if summary is None else "D0 결정성 대조 기록이 요약이 가리키는 사례 결과에 없다"
        return f"- 재학습 캐시는 같은 seed·같은 코드의 학습이 체크포인트까지 같다는 가정에 기댄다. {todo}."
    same = "같았다" if record["same"] else "달랐다"
    return (f"- 재학습 캐시는 같은 seed·같은 코드의 학습이 체크포인트까지 같다는 가정에 기댄다. 개발 사례 D0에서 같은 부분집합의 "
            f"두 학습 체크포인트 SHA가 {same}(대조 출처 {record['source']}).")


def build(root: Path, summary_path: str | None) -> str:
    protocol = read(root, PROTOCOL)
    protocol_sha = lf_sha(root / PROTOCOL)
    smoke_protocol = read(root, SMOKE_PROTOCOL)
    v1, v2 = read(root, smoke_protocol["version_history"]["v1"]["summary"]), read(root, SMOKE_V2_SUMMARY)
    if v2["protocol_sha256_lf"] != lf_sha(root / SMOKE_PROTOCOL):
        raise ValueError("smoke v2 요약이 smoke v2 프로토콜의 것이 아니다")
    lines = ["# 변경 여러 개가 섞인 실패 학습에서 원인 변경을 확정하는 데 드는 재학습", "",
             f"이 문서의 수치는 `evals/build_d3_doc.py`가 사전등록 프로토콜(`{PROTOCOL}`), 무해성 smoke 요약, 본실험 요약에서 "
             "만든다. 손으로 고치지 않는다.", "",
             *prereg_section(protocol, smoke_protocol, v1, v2)]
    summary = None
    if summary_path is None:
        lines += ["## 결과", "", "본실험 요약이 아직 없다. 요약이 커밋되면 `python evals/build_d3_doc.py --summary <요약 경로>`로 "
                  "이 절을 채운다.", ""]
    else:
        summary = read(root, summary_path)
        if summary.get("protocol_sha256_lf") != protocol_sha:
            raise ValueError("요약이 이 프로토콜의 것이 아니다")
        lines += result_section(root, summary, protocol, protocol_sha)
    lines += ["## 이 실험이 말하지 않는 것", "",
              "- 결함·무해 변경·검사를 같은 팀이 만들었다. U1만 다른 프레임워크에서 원인이 확정된 결함의 의미를 옮긴 것이다.",
              "- 변경 8개, 사전등록 사례 3건이다. 변경 수가 다르거나 사례가 많아도 같은 결론이 나온다고 말하지 않고 통계 검정도 하지 않는다.",
              "- 무해 변경 목록은 smoke v1 결과를 보고 한 번 바꿨다(위 절).",
              "- 원인 변경의 위치는 난수로 정했다. 위치가 앞이면 하나씩 되돌리기가 유리하다.",
              "- U1·F1의 diff에는 속도를 0으로 바꾸거나 좌표계를 바꾸는 줄이 보인다. 모델이 이런 흔한 버그 형태를 알아본 것인지 학습 "
              "신호를 읽은 것인지 이 실험으로 가를 수 없다.",
              determinism_line(root, summary),
              f"- 하지 않는 주장: {', '.join(protocol['forbidden_claims'])}.", "",
              "## 재현", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe evals/build_d3_doc.py" + (f" --summary {summary_path}" if summary_path else "") + " --check",
              "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        content = build(ROOT, args.summary)
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
    print(json.dumps({"status": "written", "path": args.output, "summary": args.summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
