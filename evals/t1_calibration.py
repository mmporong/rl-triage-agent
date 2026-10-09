"""T1 보정 판정(docs/T1-EXTERNAL-FAULTS.md 4절). dev seed 7 고정 평가 결과로 후보마다 holdout 포함 여부를 정한다.

사용: python evals/t1_calibration.py evals/results/<고정 평가 tag> --smoke evals/results/<smoke tag>/summary.json

- 깨짐: 짝 기준 대비 P0-C 보정 규칙(recovery.compare·verdict, 정상 변동 0): 생존 비 < 0.90, 낙상 > 기준 + 0.05,
  선속도·회전 오차 비 > 1.10 중 하나. 크기는 키우지 않는다(상류 기본값에서 깨지지 않으면 제외).
- 짝 기준의 타당성: 기준 실행이 걷는다(선속도 RMSE < 제자리 RMSE × 0.5, 낙상 ≤ 0.05). 아니면 그 후보를 제외한다.
- 짝 기준: X1·X3는 t1cal_<X>_REF_s7(smoke가 X1 Bug 2를 재현하면 X1은 순정 p0ccal_NONE_s7), X2는 p0ccal_NONE_s7.
- 출력: <폴더>/t1_calibration.json. GPU·모델을 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from fixed_eval_summary import behavior_of  # noqa: E402
from replay import _sha256_lf, _show, code_version  # noqa: E402

from rl_triage import recovery as RC  # noqa: E402

STOCK = "p0ccal_NONE_s7"
WALK_FRACTION = 0.5
REF_FALL_MAX = 0.05
CODE_FILES = ("evals/t1_calibration.py", "src/rl_triage/recovery.py", "evals/fixed_eval_summary.py")


def reference_of(fault: str, smoke: dict) -> str:
    if fault == "X2" or (fault == "X1" and smoke["X1"]["decision"] == "stock_reference"):
        return STOCK
    return f"t1cal_{fault}_REF_s7"


def judge(fault: str, smoke: dict, reports: dict) -> dict:
    decision = smoke[fault]["decision"]
    if decision not in ("include", "pair", "stock_reference"):
        return {"include": False, "reason": f"smoke {decision}"}
    run, ref = f"t1cal_{fault}_s7", reference_of(fault, smoke)
    missing = [n for n in (run, ref) if n not in reports]
    if missing:
        return {"include": False, "reason": f"고정 평가 결과 없음 {missing}"}
    r = reports[ref]
    walks = (r["metrics"]["lin_vel_rmse_mps"] < WALK_FRACTION * r["standstill"]["lin_vel_rmse_mps"]
             and r["metrics"]["fall_rate"] <= REF_FALL_MAX)
    b_ref = behavior_of(r)
    bands = RC.derive_bands([RC.compare(b_ref, b_ref)])  # 정상 변동 0 + 여유값(P0-C 보정 규칙)
    c = RC.compare(behavior_of(reports[run]), b_ref)
    v = RC.verdict(c, bands)
    broken = v["label"] == "unhealthy"
    reason = "포함" if walks and broken else "기준이 걷지 않음" if not walks else "상류 크기에서 깨지지 않음"
    return {"include": walks and broken, "reason": reason, "run": run, "reference": ref, "reference_walks": walks,
            "label": v["label"], "failed": v["failed"], "bands": {k: round(x, 4) for k, x in bands.items()},
            "compared": {k: round(x, 4) for k, x in c.items()}, "reference_metrics": r["metrics"],
            "run_metrics": reports[run]["metrics"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T1 dev seed 7 보정 판정")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--smoke", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.folder / "t1_calibration.json"
    if out.exists():
        raise SystemExit(f"{_show(out)}가 이미 있다. 결과는 덮어쓰지 않는다")
    files = sorted((args.folder / "runs").glob("*.json"))
    reports = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files if "__" not in f.stem}
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))["decisions"]
    result = {f: judge(f, smoke, reports) for f in ("X1", "X2", "X3")}
    report = {"kind": "t1_calibration", "smoke": _show(args.smoke), "smoke_sha256_lf": _sha256_lf(args.smoke),
              "include": [f for f, r in result.items() if r["include"]], "candidates": result,
              "inputs": [{"path": _show(f), "sha256_lf": _sha256_lf(f)} for f in files if "__" not in f.stem],
              "code": code_version([_show(args.folder)], CODE_FILES)}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f, r in result.items():
        print(f"{f}: {r['reason']}" + (f" ({r['label']} {r['failed']})" if "label" in r else ""))
    print(f"include={report['include']} saved {_show(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
