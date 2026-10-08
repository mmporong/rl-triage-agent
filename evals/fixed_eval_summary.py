"""P0-B2: 고정 평가 결과(evals/fixed_eval.py)에 회복 판정을 적용하고 현행·P0-B1 판정과 비교한다.

사용: python evals/fixed_eval_summary.py evals/results/<tag> [--relabel evals/results/p0b1_relabel_20261008/relabel.json]

- 입력: <tag>/runs/<실행 이름>.json(체크포인트 접미사 없는 파일만 실행 판정에 쓴다).
- 판정은 P0-B1과 같은 함수(rl_triage.recovery.compare·derive_bands·verdict)로 한다. 지표만 고정 평가 값으로 바꾼다:
  survival_s=mean_survival_s, fall_frac=fall_rate, err_xy=lin_vel_rmse_mps, err_yaw=yaw_rate_rmse_radps.
  모든 실행이 같은 20초 조건이라 undetermined는 나오지 않는다.
- 정상 실행 비교값: 서로 다른 seed의 기준 실행끼리 + benign_all 대 같은 seed 기준 + 같은 seed 반복 학습
  (baseline_repeat<N>_s<seed>) 대 기준.
- 출력: <tag>/summary.json. 모델·네트워크·GPU를 쓰지 않는다. 종료 코드 0 완료, 2 입력 오류.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from replay import _sha256_lf, _show, code_version  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

CODE_FILES = ("evals/fixed_eval_summary.py", "evals/fixed_eval.py", "src/rl_triage/recovery.py",
              "bench/protocols/fixed_eval_v1.json")
REPEAT = re.compile(r"baseline_repeat\d+_s(\d+)")


class InputError(Exception):
    """입력 오류(종료 코드 2)."""


def behavior_of(report: dict) -> dict:
    m = report["metrics"]
    return {"survival_s": m["mean_survival_s"], "horizon_s": report["horizon_steps"] * report["step_dt"],
            "fall_frac": m["fall_rate"], "err_xy": m["lin_vel_rmse_mps"], "err_yaw": m["yaw_rate_rmse_radps"]}


def summarize(folder: Path, relabel: dict | None, reference_prefix: str = "baseline") -> dict:
    files = sorted((folder / "runs").glob("*.json"))
    reports = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files if "__" not in f.stem}
    if not reports:
        raise InputError(f"{_show(folder)}/runs에 실행 결과가 없다")
    protocols = {(r["protocol"], r["protocol_sha256_lf"]) for r in reports.values()}
    if len(protocols) != 1:
        raise InputError(f"서로 다른 프로토콜 결과가 섞여 있다: {sorted(protocols)}")
    beh = {n: behavior_of(r) for n, r in reports.items()}
    reference = re.compile(rf"{re.escape(reference_prefix)}_s(\d+)")
    refs = {int(m[1]): n for n in beh if (m := reference.fullmatch(n))}
    if len(refs) < 2:
        raise InputError("서로 다른 seed의 기준 실행이 둘 이상 필요하다")
    base = [refs[s] for s in sorted(refs)]
    normal = [{"pair": f"{a} vs {b}", **RC.compare(beh[a], beh[b])} for a in base for b in base if a != b]
    for n in sorted(beh):
        seed = RC.seed_of(n)
        if (n.startswith("benign_all_s") or REPEAT.fullmatch(n)) and seed in refs:
            normal.append({"pair": f"{n} vs {refs[seed]}", **RC.compare(beh[n], beh[refs[seed]])})
    bands = RC.derive_bands(normal)
    previous = {r["run"]: r for r in (relabel or {}).get("runs", [])}
    rows = []
    for n in sorted(beh):
        seed = RC.seed_of(n)
        if seed not in refs:
            raise InputError(f"{n}: seed {seed}의 기준 실행 결과가 없다")
        c = RC.compare(beh[n], beh[refs[seed]])
        v = RC.verdict(c, bands)
        prev = previous.get(n, {})
        rows.append({"run": n, "seed": seed, "reference": refs[seed], "is_reference": n == refs[seed],
                     "fixed_label": v["label"], "fixed_failed": v["failed"],
                     "p0b1_label": prev.get("behavior_label"), "current_recovered": prev.get("current_recovered"),
                     "metrics": reports[n]["metrics"], "checkpoint_sha256": reports[n]["checkpoint"]["sha256"],
                     "compared": {k: round(x, 4) for k, x in c.items()}})
    agree = lambda r: r["p0b1_label"] in (None, r["fixed_label"])  # noqa: E731
    return {"protocol": sorted(protocols)[0][0], "protocol_sha256_lf": sorted(protocols)[0][1],
            "standstill": next(iter(reports.values()))["standstill"],
            "bands": {k: round(x, 4) for k, x in bands.items()},
            "normal": [{k: (round(x, 4) if isinstance(x, float) else x) for k, x in c.items()} for c in normal],
            "runs": rows, "changes_vs_p0b1": [r["run"] for r in rows if not agree(r) and not r["is_reference"]],
            "inputs": [{"path": _show(f), "sha256_lf": _sha256_lf(f)} for f in files if "__" not in f.stem]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="고정 평가 결과에 회복 판정을 적용한다(모델·네트워크·GPU 없음)")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--relabel", type=Path, help="P0-B1 relabel.json(비교용, 선택)")
    ap.add_argument("--reference-prefix", default="baseline",
                    help="기준 실행 이름 접두어(<접두어>_s<seed>). P0-C는 baseline_p0c")
    args = ap.parse_args(argv)
    try:
        out = args.folder / "summary.json"
        if out.exists():
            raise InputError(f"{_show(out)}가 이미 있다. 결과는 덮어쓰지 않는다")
        relabel = json.loads(args.relabel.read_text(encoding="utf-8")) if args.relabel else None
        result = summarize(args.folder, relabel, args.reference_prefix)
    except (InputError, ValueError, FileNotFoundError) as e:
        print(f"fixed eval summary 입력 오류: {e}", file=sys.stderr)
        return 2
    command = ["python", "evals/fixed_eval_summary.py", _show(args.folder)]
    if args.relabel:
        command += ["--relabel", _show(args.relabel)]
    if args.reference_prefix != "baseline":
        command += ["--reference-prefix", args.reference_prefix]
    report = {"kind": "fixed_eval_summary", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "command": " ".join(command), "contract": {"doc": "docs/P0-B2-FIXED-EVAL.md", "margin": RC.MARGIN},
              **result, "code": code_version([_show(args.folder)], CODE_FILES),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "model_calls": 0, "network": "not used", "gpu": "not used"}}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {}
    for r in report["runs"]:
        k = (r["p0b1_label"], r["fixed_label"])
        counts[k] = counts.get(k, 0) + 1
    print(f"runs={len(report['runs'])} changes_vs_p0b1={len(report['changes_vs_p0b1'])}")
    print("bands " + " ".join(f"{k}={v}" for k, v in report["bands"].items()))
    for (p, f), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  p0b1={p!s:12s} fixed={f:10s} {n}")
    for name in report["changes_vs_p0b1"]:
        r = next(x for x in report["runs"] if x["run"] == name)
        print(f"  CHANGE {name:22s} p0b1={r['p0b1_label']} -> fixed={r['fixed_label']} {r['fixed_failed']}")
    print(f"saved {_show(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
