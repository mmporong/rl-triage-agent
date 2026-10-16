"""ANYmal-C 기존 단계와 F6 smoke 공개 근거로 이식 문서를 만든다(CPU 전용)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import build_anymal_doc as BASE
import export_anymal_termination_smoke_v1 as SMOKE

ROOT = Path(__file__).resolve().parents[1]
DEMO = "evals/results/anymal_c_e2e_f1_s2047_20261014/demo_result.json"
PUBLIC = "evals/results/anymal_f6_smoke_public_s2100_20261017"
OUTPUT = "docs/ANYMAL-C-PORT.md"


def smoke_section(root: Path) -> str:
    folder = root / PUBLIC
    SMOKE.verify_public(folder)
    payload = {name: SMOKE.read(folder / name) for name in SMOKE.PUBLIC_FILES}
    summary = payload["summary.json"]
    runtime = payload["runtime_signature.json"]
    costs = payload["costs.json"]
    lines = [
        "## F6 종료 조건 smoke", "",
        f"새 seed {summary['seed']}에서 정상·F6 정책을 각각 300회 학습했다. F6는 학습의 시간 제한을 "
        f"{runtime['configured_episode_length_s']:g}s에서 {runtime['fault_timeout_s']:g}s로 줄이는 등록 결함이다. "
        "판정에는 기존 ANYmal-C 단계 1 밴드와 26조건 고정 평가를 사용했다.", "",
        f"[사실] 결과는 `{summary['outcome']}`다. 정상 정책은 걷기 기준을 통과했고, F6 정책도 "
        "같은 seed 정상 정책 대비 생존·낙상·선속도·yaw 기준 네 항목을 통과했다. "
        f"사전 예측 `{summary['prediction']}`은 빗나갔다. "
        "등록된 중단 규칙에 따라 후속 종단 시연을 사전등록하거나 실행하지 않았다.", "",
        "| 학습 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 대비 비 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("none", "f6"):
        row = payload[f"evaluations/{arm}.json"]
        m = row["metrics"]
        ratio = m["lin_vel_rmse_mps"] / row["standstill"]["lin_vel_rmse_mps"]
        lines.append(f"| {arm.upper()} | {m['lin_vel_rmse_mps']:.6f} | {m['yaw_rate_rmse_radps']:.6f} | "
                     f"{m['fall_rate']:.6f} | {m['mean_survival_s']:.3f} | {ratio:.6f} |")
    lines += [
        "",
        f"[사실] 학습 끝 {runtime['tail_iterations']}회의 에피소드 길이 중앙값은 NONE "
        f"{runtime['episode_steps_median']['NONE']:.2f} step, F6 "
        f"{runtime['episode_steps_median']['F6']:g} step이었다. F6의 timeout 항도 기록됐다. "
        "종료 조건 변경은 적용됐으며, 고정 평가는 결함을 주입하지 않은 환경에서 학습된 정책을 평가했다.", "",
        "[추론] 이 seed에서는 학습 중 조기 종료가 최종 정책의 등록 행동 실패로 이어지지 않았다. "
        "결함 미적용이나 교정 후 회복을 뜻하지 않으며, 다른 seed·조건의 결과는 이 기록으로 결정할 수 없다.", "",
        "| 측정 구간 | 부모 프로세스 벽시계 (s) |",
        "|---|---:|",
        f"| NONE 학습 | {costs['training']['NONE']['training_wall_s']:.3f} |",
        f"| F6 학습 | {costs['training']['F6']['training_wall_s']:.3f} |",
        f"| 두 정책 고정 평가 | {costs['fixed_eval_wall_s']:.3f} |",
        f"| 학습 + 고정 평가 | {costs['totals_wall_s']['training_and_fixed_evaluation']:.3f} |", "",
        "시간은 부모 실행 시작부터 자연 종료와 소유 프로세스 정리 확인까지다. 위 합계는 CPU 의미 점검·"
        f"텔레메트리 추출·대기 구간을 포함하지 않는다. 모델 호출은 {summary['model_calls']}회다.", "",
        f"근거: [요약](../{PUBLIC}/summary.json), [정상 평가](../{PUBLIC}/evaluations/none.json), "
        f"[F6 평가](../{PUBLIC}/evaluations/f6.json), [주입 확인](../{PUBLIC}/runtime_signature.json), "
        f"[비용](../{PUBLIC}/costs.json), [SHA 결합](../{PUBLIC}/provenance.json), "
        "[사전등록](../bench/protocols/anymal_termination_smoke_v1.json).", "",
        f"실험 커밋 `{summary['source_commit_sha']}`, 프로토콜 LF SHA256 "
        f"`{summary['protocol_sha256_lf']}`.", "",
    ]
    return "\n".join(lines)


def build(root: Path = ROOT) -> str:
    base = BASE.build(root, DEMO)
    base = base.replace("`evals/build_anymal_doc.py`가", "`evals/build_anymal_doc_v2.py`가", 1)
    marker = "## 주장하지 않는 것\n"
    if base.count(marker) != 1:
        raise ValueError("기존 문서 구획이 바뀌었다")
    base = base.replace(marker, smoke_section(root) + "\n" + marker, 1)
    old = ".\\.venv\\Scripts\\python.exe evals/build_anymal_doc.py --demo " + DEMO + " --check"
    if old not in base:
        raise ValueError("기존 재현 명령이 바뀌었다")
    return base.replace(old, ".\\.venv\\Scripts\\python.exe evals/build_anymal_doc_v2.py --check", 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        content = build(ROOT)
    except (ValueError, KeyError, OSError) as exc:
        print(f"거부: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    path = ROOT / args.output
    if args.check:
        same = path.is_file() and path.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if same else "stale"}))
        return 0 if same else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "path": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
