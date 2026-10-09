"""저장된 probe 고리 결과의 시간·확정률을 비교한다(GPU·모델 없음).

사용: python evals/diagnostic_cost.py --loop evals/results/<loop tag> \
      --training-meta bench/private --tag <새 tag>
공개 재계산: --training-meta 대신 결과 폴더의 --training-times training_times.json을 쓴다.

probe 시간은 measure() 안의 elapsed_s 합이다. 시뮬레이터 시작·환경 생성·reset,
정책 로드·기준 probe·모델 호출·사람 검토 시간은 포함하지 않는다. GPU 장치 사용 시간을
측정한 값이 아니며, 재학습 소거법은 실제 실행한 비교군이 아닌 최대 6회 시나리오다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from replay import _sha256_lf, _show, code_version  # noqa: E402

CODE_FILES = ("evals/diagnostic_cost.py", "evals/replay.py")
STATUSES = ("confirmed", "unidentifiable", "none_supported")


def seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("측정 초는 누락 없는 유한한 비음수여야 한다")
    return float(value)


def summarize(rows: list[dict]) -> dict:
    """무작위 순서 반복은 독립 사례 수를 늘리지 않고 반복당 합계로 환산한다."""
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        label, run = row["strategy"], row["run"]
        name = "random" if re.fullmatch(r"random\d+", label) else label
        bucket = groups[name][label]
        if run in bucket:
            raise ValueError(f"중복 실행×방식: {run}/{label}")
        if row["status"] not in STATUSES:
            raise ValueError(f"알 수 없는 판정: {row['status']}")
        if row["correct"] != (row["conclusion"] == row["truth"]):
            raise ValueError(f"저장된 correct가 정답과 다르다: {run}/{label}")
        seconds(row["gpu_s_measured"])
        bucket[run] = row
    if "exhaustive" not in groups or not groups["exhaustive"]["exhaustive"]:
        raise ValueError("전수 probe 기준 결과가 없다")
    cohort = set(groups["exhaustive"]["exhaustive"])
    table = {}
    for name, variants in sorted(groups.items()):
        run_sets = [set(v) for v in variants.values()]
        if any(s != run_sets[0] for s in run_sets):
            raise ValueError(f"반복마다 사례 집합이 다르다: {name}")
        if not run_sets[0] <= cohort:
            raise ValueError(f"전수 결과에 없는 사례: {name}")
        cells = [r for variant in variants.values() for r in variant.values()]
        repeats, n = len(variants), len(run_sets[0])
        elapsed = math.fsum(seconds(r["gpu_s_measured"]) for r in cells)
        table[name] = {
            "unique_cases": n, "evaluation_cells": len(cells), "order_repeats": repeats,
            "same_cases_as_exhaustive": run_sets[0] == cohort,
            "correct_per_repeat": sum(r["correct"] for r in cells) / repeats,
            "wrong_per_repeat": sum(r["status"] == "confirmed" and not r["correct"] for r in cells) / repeats,
            "unidentifiable_per_repeat": sum(r["status"] == "unidentifiable" for r in cells) / repeats,
            "none_supported_per_repeat": sum(r["status"] == "none_supported" for r in cells) / repeats,
            "correct_rate": sum(r["correct"] for r in cells) / len(cells),
            "probe_measurement_s_per_repeat": round(elapsed / repeats, 3),
            "probe_measurement_s_per_case": round(elapsed / len(cells), 3),
            "mean_probes": sum(r["probes_used"] for r in cells) / len(cells),
            "relative_probe_time_vs_exhaustive": None,
        }
    exhaustive = table["exhaustive"]["probe_measurement_s_per_case"]
    for row in table.values():
        if row["same_cases_as_exhaustive"] and row["probe_measurement_s_per_case"] > 0:
            row["relative_probe_time_vs_exhaustive"] = round(exhaustive / row["probe_measurement_s_per_case"], 4)
    return table


def training_times(meta_dir: Path, runs: set[str]) -> list[dict]:
    """비공개 meta에서 측정 열만 추출한다. 사용자 경로·주입 이름은 저장하지 않는다."""
    names = runs | {f"baseline_p0c_s{name.rsplit('_s', 1)[1]}" for name in runs}
    out = []
    for name in sorted(names):
        meta = json.loads((meta_dir / f"{name}.meta.json").read_text(encoding="utf-8-sig"))
        out.append({"run": name, "num_envs": meta["num_envs"], "max_iterations": meta["max_iterations"],
                    "train_exit_code": meta["train_exit_code"], "wall_time_s": seconds(meta["wall_time_s"])})
    return out


def training_summary(records: list[dict], runs: set[str]) -> dict:
    expected = runs | {f"baseline_p0c_s{name.rsplit('_s', 1)[1]}" for name in runs}
    names = [r["run"] for r in records]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError("학습 시간 표본이 결함 실행과 같은 seed 기준 실행의 집합과 다르다")
    if any((r["num_envs"], r["max_iterations"], r["train_exit_code"]) != (4096, 300, 0) for r in records):
        raise ValueError("학습 시간은 성공한 4096 env × 300회 실행만 비교한다")
    times = [seconds(r["wall_time_s"]) for r in records]
    if not times or min(times) <= 0:
        raise ValueError("학습 시간은 양수여야 한다")
    return {"runs": len(times), "mean_wall_s": statistics.mean(times), "median_wall_s": statistics.median(times),
            "min_wall_s": min(times), "max_wall_s": max(times), "total_wall_s": math.fsum(times)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-meta", type=Path)
    source.add_argument("--training-times", type=Path)
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--tag")
    dest.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise ValueError("tag에는 영문·숫자·._-만 쓴다")
    out = ROOT / "evals" / "results" / args.tag if args.tag else args.out.resolve()
    if out.exists():
        raise ValueError("결과 폴더가 이미 있다. 덮어쓰지 않는다")
    rows_path = args.loop / "loop_compare.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = summarize(rows)
    runs = {r["run"] for r in rows if r["strategy"] == "exhaustive"}
    records = (training_times(args.training_meta, runs) if args.training_meta else
               json.loads(args.training_times.read_text(encoding="utf-8")))
    fields = ("run", "num_envs", "max_iterations", "train_exit_code", "wall_time_s")
    records = [{field: record[field] for field in fields} for record in records]
    training = training_summary(records, runs)
    limitations = [f"{len(runs)}건은 저장된 결함·probe 결과의 재생이다. 결함과 probe를 같은 설계자가 만든 시험이다.",
                   "선택 방식마다 정답 확정률이 다르므로 시간 비율은 같은 성능에서의 절감률이 아니다."]
    if "random" in table:
        random = table["random"]
        limitations.append(f"무작위 순서 {random['order_repeats']}회는 같은 {random['unique_cases']}건의 재생이며 "
                           f"독립 학습 표본 {random['evaluation_cells']}건이 아니다.")
    limitations.extend(f"{name}은 {row['unique_cases']}건만 있어 전체 {len(runs)}건과 시간 비율을 계산하지 않는다."
                       for name, row in table.items() if not row["same_cases_as_exhaustive"])
    report = {
        "kind": "diagnostic_cost", "unique_cases": len(runs), "table": table, "training": training,
        "retraining_elimination_scenario": {
            "kind": "hypothetical_upper_budget", "max_trainings_per_case": 6,
            "wall_s_per_case": 6 * training["mean_wall_s"],
            "wall_s_for_cohort": len(runs) * 6 * training["mean_wall_s"],
            "applicable_to_hidden_config_diff": False, "measured_savings_ratio": None,
            "reason": "숨은 결함에는 설정 diff로 되돌릴 대상이 없다. 원인 범주별 수정 개입도 정의·실행하지 않았다.",
        },
        "measurement_scope": {
            "probe": "evals/probes.py measure() elapsed_s 합(0.1초 반올림); 저장 결과표의 선택 경로 재생",
            "training": "bench/run_case.ps1 프로세스 시작부터 종료까지 wall_time_s",
            "excluded": ["probe 프로세스 시작·환경 생성·reset·정책 로드", "기준 probe", "모델 호출·토큰", "사람 검토", "개발·실패·재시도 전체 비용"],
            "gpu_device_busy_time_measured": False, "end_to_end_savings_measured": False,
        },
        "limitations": limitations,
        "inputs": [{"path": _show(rows_path), "sha256_lf": _sha256_lf(rows_path)}],
        "code": code_version([_show(args.loop)], CODE_FILES),
    }
    out.mkdir(parents=True)
    snapshot = out / "training_times.json"
    snapshot.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report["inputs"].append({"path": "training_times.json", "sha256_lf": _sha256_lf(snapshot)})
    (out / "diagnostic_cost.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy", *next(iter(table.values()))])
        writer.writeheader()
        writer.writerows({"strategy": name, **row} for name, row in table.items())
    for name, row in table.items():
        print(f"{name}: n={row['unique_cases']} correct={row['correct_per_repeat']:g} "
              f"wrong={row['wrong_per_repeat']:g} probe_s={row['probe_measurement_s_per_repeat']:g}")
    print(f"training n={training['runs']} mean={training['mean_wall_s']:.3f}s; saved {_show(out)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
