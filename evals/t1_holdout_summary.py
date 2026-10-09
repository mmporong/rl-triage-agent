"""T1 holdout 고정 평가를 사전등록한 짝·보행 조건·정상 변동 0으로 판정한다.

python evals/t1_holdout_summary.py evals/results/t1_fixed_eval_<date>
원시 결과는 수정하지 않고 새 holdout_summary.json을 쓴다. probe 결과를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
from build_workspace_t1 import cells  # noqa: E402
from fixed_eval_summary import behavior_of  # noqa: E402
from replay import _sha256_lf, code_version  # noqa: E402
from t1_calibration import REF_FALL_MAX, WALK_FRACTION  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

CODE_FILES = ("evals/t1_holdout_summary.py", "evals/t1_calibration.py", "evals/fixed_eval_summary.py",
              "src/rl_triage/recovery.py", "bench/protocols/fixed_eval_v1.json")


def summarize(folder: Path, key: dict, protocol_sha: str) -> dict:
    plan = cells(key)
    names = {c[k] for c in plan for k in ("run", "reference")}
    files = {n: folder / "runs" / f"{n}.json" for n in names}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise ValueError(f"T1 고정 평가 누락: {sorted(missing)}")
    reports = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in files.items()}
    for name, r in reports.items():
        if (r["name"] != name or r["protocol"] != "fixed_eval_v1" or r["protocol_sha256_lf"] != protocol_sha
                or r["eval_seed"] != 2026 or r["num_envs"] != 1040 or r["horizon_steps"] != 1000
                or r["step_dt"] != 0.02 or r["action_scale"] != 0.25):
            raise ValueError(f"T1 고정 평가 프로토콜 불일치: {name}")
        if not all(math.isfinite(x) for x in r["metrics"].values()):
            raise ValueError(f"T1 비유한 지표: {name}")
    rows = []
    for cell in plan:
        r, fault = reports[cell["reference"]], reports[cell["run"]]
        walks = (r["metrics"]["lin_vel_rmse_mps"] < WALK_FRACTION * r["standstill"]["lin_vel_rmse_mps"]
                 and r["metrics"]["fall_rate"] <= REF_FALL_MAX)
        ref_behavior = behavior_of(r)
        bands = RC.derive_bands([RC.compare(ref_behavior, ref_behavior)])
        compared = RC.compare(behavior_of(fault), ref_behavior)
        verdict = RC.verdict(compared, bands)
        rows.append({"run": cell["run"], "reference": cell["reference"], "seed": cell["seed"],
                     "reference_walks": walks, "included": walks,
                     "exclusion_reason": None if walks else "기준 보행 조건 미충족",
                     "label": verdict["label"], "failure_absent": verdict["label"] == "healthy",
                     "failed": verdict["failed"], "bands": bands, "compared": compared,
                     "metrics": fault["metrics"], "reference_metrics": r["metrics"]})
    valid = [r for r in rows if r["included"]]
    return {"kind": "t1_holdout_summary", "planned": len(rows), "valid": len(valid),
            "unhealthy": sum(r["label"] == "unhealthy" for r in valid),
            "failure_absent": sum(r["failure_absent"] for r in valid), "runs": rows,
            "inputs": [{"name": n, "sha256_lf": _sha256_lf(p)} for n, p in sorted(files.items())]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    out = args.folder / "holdout_summary.json"
    if out.exists():
        raise SystemExit("T1 holdout 판정 파일이 이미 있다. 덮어쓰지 않는다")
    key_path = ROOT / "bench/private/answer_key_t1.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    result = summarize(args.folder, key, _sha256_lf(ROOT / "bench/protocols/fixed_eval_v1.json"))
    result.update(answer_key_sha256_lf=_sha256_lf(key_path), code=code_version([], CODE_FILES))
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"T1 planned={result['planned']} valid={result['valid']} unhealthy={result['unhealthy']} "
          f"failure_absent={result['failure_absent']}")


if __name__ == "__main__":
    main()
