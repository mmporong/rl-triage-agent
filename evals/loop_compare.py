"""P1-A 선택 방식 비교(docs/P1-A-LOOP.md 4절). 실행마다 한 번 잰 probe 결과표 위에서 방식들을 돌린다. GPU·모델 없음.

사용:
  python evals/loop_compare.py --probes evals/results/<probe tag> --tag <새 폴더> \
      [--ranking rules=evals/results/p0c_rules_20261009] [--ranking agent=evals/results/p0c_holdout_20261009] ...

- probe 판정은 같은 seed 기준 실행 측정값 대비(probe_loop.classify)다. 기준은 정답표 사례의 reference
  (T1: ref_<사례> 또는 baseline_p0c), 없으면 baseline_p0c_s<seed>다.
- 방식: exhaustive, fixed, random(seed 0~4 평균), discriminate:<이름>(그 결과 폴더의 진단 순위 상위 3개를 가르기).
  순위 폴더는 replay 형식 jsonl이다. rules 폴더는 rule_features 행, 그 밖은 mode별(agent, control_full)로 읽는다.
- 출력: <폴더>/loop_compare.jsonl(실행×방식 한 줄), <폴더>/loop_compare.json(방식별 표). 정답은 --key로 준 정답표(기본 P0-C).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from replay import _sha256_lf, _show, code_version  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

KEYS = (ROOT / "bench" / "answer_key_p0c.json", ROOT / "bench" / "private" / "answer_key_p0c.json")
CODE_FILES = ("evals/loop_compare.py", "src/rl_triage/probe_loop.py")
RANDOM_SEEDS = range(5)


def load_outcomes(folder: Path, key: dict | None = None) -> tuple[dict, dict]:
    """(실행 이름 → probe → 판정, 실행 이름 → probe → 측정 초)."""
    reports = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((folder / "probes").glob("*.json"))}
    cases = (key or {}).get("cases", {})
    references = {"baseline_p0c"} | {c.get("reference", "baseline_p0c") for c in cases.values()}
    outcomes, seconds = {}, {}
    for name, rep in reports.items():
        case, seed = name.rsplit("_s", 1)
        if case in references:
            continue
        ref_name = f"{cases.get(case, {}).get('reference', 'baseline_p0c')}_s{seed}"
        if ref_name not in reports:
            raise SystemExit(f"{name}: 기준 실행 {ref_name} probe 결과가 없다")
        ref = reports[ref_name]["probes"]
        outcomes[name], seconds[name] = {}, {}
        for p in L.PROBES:
            r = rep["probes"].get(p)
            ok = r is not None and r.get("exit") == 0 and ref.get(p, {}).get("exit") == 0
            outcomes[name][p] = L.classify(p, r["measurement"], ref[p]["measurement"]) if ok else "unknown"
            seconds[name][p] = (r or {}).get("elapsed_s", 0.0)
    return outcomes, seconds


def load_rankings(spec: str) -> dict[str, dict[str, list[str]]]:
    """'이름=폴더' → {방식 이름: {실행 이름: 순위}}. 칸마다 인프라 오류가 아닌 마지막 행(replay 규칙)."""
    name, folder = spec.split("=", 1)
    out: dict[str, dict[str, list[str]]] = {}
    for f in sorted(Path(folder).glob("seed*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("infra_error") or not row.get("ranking"):
                continue
            mode = row["mode"]
            if name == "rules" and mode != "rule_features":
                continue
            label = name if name in ("rules", mode) else f"{name}_{mode}"
            out.setdefault(label, {})[f"{row['case_id']}_s{row['seed']}"] = row["ranking"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="probe 결과표 위에서 실험 선택 방식을 비교한다(GPU·모델 없음)")
    ap.add_argument("--probes", type=Path, required=True)
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--tag", help="evals/results/<tag>/에 쓴다")
    dest.add_argument("--out", type=Path, help="새 폴더에 쓴다(테스트용)")
    ap.add_argument("--ranking", action="append", default=[], help="이름=결과 폴더(진단 순위)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--key", type=Path, help="정답표 경로(기본: 공개본, 없으면 bench/private 사본)")
    args = ap.parse_args(argv)
    out_dir = ROOT / "evals" / "results" / args.tag if args.tag else args.out.resolve()
    if out_dir.exists():
        raise SystemExit(f"{_show(out_dir)}가 이미 있다. 결과는 덮어쓰지 않는다")
    key_path = args.key or next((p for p in KEYS if p.exists()), None)
    if key_path is None:
        raise SystemExit("P0-C 정답표가 없다")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    outcomes, seconds = load_outcomes(args.probes, key)
    rankings = {}
    for spec in args.ranking:
        rankings.update(load_rankings(spec))
    rows = []
    for run in sorted(outcomes):
        case, seed = run.rsplit("_s", 1)
        truth = key["cases"][case]["category"]
        plans = [("exhaustive", {}), ("fixed", {})] + [(f"random{s}", {"seed": s}) for s in RANDOM_SEEDS]
        plans += [(f"discriminate:{label}", {"ranking": r[run], "top_k": args.top_k})
                  for label, r in sorted(rankings.items()) if run in r]
        for label, kw in plans:
            strategy = "random" if label.startswith("random") else label.split(":")[0]
            res = L.simulate(strategy, outcomes[run], **kw)
            used = [s["probe"] for s in res["steps"]]
            rows.append({"run": run, "seed": int(seed), "truth": truth, "strategy": label, "status": res["status"],
                         "conclusion": res["conclusion"], "correct": res["conclusion"] == truth,
                         "probes_used": res["probes_used"], "gpu_s_est": res["gpu_s"],
                         "gpu_s_measured": round(sum(seconds[run].get(p, 0.0) for p in used), 1),
                         "steps": res["steps"]})
    table = {}
    for r in rows:
        name = "random" if r["strategy"].startswith("random") else r["strategy"]
        t = table.setdefault(name, {"cells": 0, "correct": 0, "wrong": 0, "unidentifiable": 0, "none_supported": 0,
                                    "probes_used": 0, "gpu_s_est": 0, "gpu_s_measured": 0.0})
        t["cells"] += 1
        t["correct"] += r["correct"]
        t["wrong"] += r["status"] == "confirmed" and not r["correct"]
        t["unidentifiable"] += r["status"] == "unidentifiable"
        t["none_supported"] += r["status"] == "none_supported"
        t["probes_used"] += r["probes_used"]
        t["gpu_s_est"] += r["gpu_s_est"]
        t["gpu_s_measured"] = round(t["gpu_s_measured"] + r["gpu_s_measured"], 1)
    for t in table.values():
        t["mean_probes"] = round(t["probes_used"] / t["cells"], 2)
    out_dir.mkdir(parents=True)
    (out_dir / "loop_compare.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                encoding="utf-8")
    report = {"kind": "loop_compare", "probes": _show(args.probes), "rankings": [s.split("=", 1)[0] for s in args.ranking],
              "top_k": args.top_k, "random_seeds": list(RANDOM_SEEDS), "answer_key": _show(key_path),
              "answer_key_sha256_lf": _sha256_lf(key_path), "table": table,
              "code": code_version([_show(args.probes)], CODE_FILES)}
    (out_dir / "loop_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, t in table.items():
        print(f"{name:34s} correct {t['correct']}/{t['cells']} wrong {t['wrong']} unident {t['unidentifiable']} "
              f"none {t['none_supported']} mean_probes {t['mean_probes']} gpu_s_est {t['gpu_s_est']}")
    print(f"saved {_show(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
