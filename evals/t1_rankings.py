"""T1 짝별 입력에 동결 P0-C 규칙 또는 에이전트를 실행한다. 새 tag만 허용한다.

python evals/t1_rankings.py --mode rules --tag t1_rules_<date>
TRIAGE_ANALYSIS_SANDBOX=required uv run --no-sync python evals/t1_rankings.py --mode agent --tag t1_agent_<date>
에이전트는 Linux Landlock canary를 통과해야 한다. 원시 stdout/stderr는 비공개로 보존한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
import build_workspace_t1 as W  # noqa: E402
import run_eval as E  # noqa: E402
from baselines import _yaml, freeze_problem  # noqa: E402
from replay import _sha256_lf, code_version  # noqa: E402
from rl_triage import rules as R  # noqa: E402

MODEL = "nvidia/nemotron-3-super-120b-a12b"
CODE_FILES = ("bench/build_workspace_t1.py", "evals/t1_rankings.py", "evals/run_eval.py", "evals/baselines.py",
              "evals/replay.py", "configs/triage_blind.yml", "src/rl_triage/triage_tools.py",
              "src/rl_triage/nat_functions.py", "src/rl_triage/rules.py", "src/rl_triage/leakcheck.py")
PUBLIC_FIELDS = ("elapsed_s", "exit_code", "trace", "ranking", "suspected", "analysis_calls", "analysis_sandboxed")


def rule_result(ws: Path, cid: str) -> dict:
    run = json.loads((ws / "cases" / cid / "telemetry.json").read_text(encoding="utf-8"))
    ref = json.loads((ws / "reference/telemetry.json").read_text(encoding="utf-8"))
    f = R.features(run["summary"], ref["summary"], R.ref_config(_yaml(ws / "reference/params/env.yaml")))
    ranking, fired = R.feature_ranking(f)
    return {"ranking": ranking, "suspected": ranking[0], "evidence": {"features": f, "fired": fired},
            "elapsed_s": 0.0, "exit_code": 0}


def valid_ranking(ranking) -> bool:
    return (isinstance(ranking, list) and len(ranking) == len(R.MECHANISMS)
            and all(isinstance(m, str) for m in ranking) and set(ranking) == set(R.MECHANISMS))


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("rules", "agent"), required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise SystemExit("잘못된 tag")
    out = ROOT / "evals/results" / args.tag
    private = ROOT / "bench/private" / args.tag
    if out.exists() or private.exists():
        raise SystemExit("결과 폴더가 이미 있다. 덮어쓰지 않는다")
    key_path = ROOT / "bench/private/answer_key_t1.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    manifest = W.load_manifest(key)
    inputs = [ROOT / "bench/reference/t1_manifest.json"]
    for cell in manifest["cells"]:
        W.validate_cell(cell, key, workspace=True)
        inputs += [ROOT / cell[k] for k in ("telemetry", "reference_telemetry")]
        inputs += [ROOT / cell["params"] / n for n in cell["params_sha256"]]
    code = code_version([p.relative_to(ROOT).as_posix() for p in inputs], CODE_FILES)
    if problem := freeze_problem(code, key["holdout_seeds"]):
        raise SystemExit(problem)
    if args.mode == "agent" and (os.environ.get("TRIAGE_ANALYSIS_SANDBOX") != "required" or E.MODEL != MODEL):
        raise SystemExit("동결 모델과 TRIAGE_ANALYSIS_SANDBOX=required가 필요하다")
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    attempts = completed = 0
    for cell in manifest["cells"]:
        ws, cid, seed = ROOT / cell["workspace"], cell["case_id"], cell["seed"]
        if args.mode == "agent":
            canary = E.analysis_canary(ws, cid, key_path)
            write_new(private / f"canary_s{seed}_{cid}.json", canary)
            public_canary = {k: canary.get(k) for k in ("runner", "sandbox_mode", "sandbox", "targets", "denied", "exit_code")}
            write_new(out / f"canary_s{seed}_{cid}.json", public_canary)
            if not canary.get("denied") or canary.get("sandbox") != "landlock":
                raise SystemExit("T1 canary 실패: 에이전트를 실행하지 않는다")
        for attempt in range(1, 5):  # P0-C와 같은 최대 4회, 인프라 오류만 재시도
            t0 = time.perf_counter()
            raw = E.run_agent_blind(ws, cid) if args.mode == "agent" else rule_result(ws, cid)
            if args.mode == "rules":
                raw["elapsed_s"] = round(time.perf_counter() - t0, 6)
            infra = raw.get("suspected") is None and E.is_infra_error(raw)
            text = json.dumps(raw, ensure_ascii=False)
            if secret := os.environ.get("NVIDIA_API_KEY"):
                text = text.replace(secret, "<redacted>")
            write_new(private / f"{cid}_s{seed}_attempt{attempt}.json", json.loads(text))
            if args.mode == "agent" and raw.get("analysis_calls", 0) != raw.get("analysis_sandboxed", 0):
                raise SystemExit("샌드박스 없이 실행된 분석 호출: T1 순위 실행 중단")
            row = {k: raw[k] for k in PUBLIC_FIELDS if k in raw}
            if "evidence" in raw:
                row["evidence"] = raw["evidence"]
            valid = valid_ranking(row.get("ranking"))
            if not valid:
                row.update(ranking=None, suspected=None)
            row.update(case_id=cid, seed=seed, mode="agent" if args.mode == "agent" else "rule_features",
                       model=MODEL if args.mode == "agent" else None, bench="t1", task="blind",
                       attempt=attempt, infra_error=infra, valid_ranking=valid,
                       analysis_sandbox="required" if args.mode == "agent" else "not_used")
            with (out / f"seed{seed}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            attempts += 1
            print(f"{cid}_s{seed} {args.mode} attempt={attempt} valid={valid} infra={infra}", flush=True)
            if not infra or attempt == 4:
                completed += valid and not infra
                break
            time.sleep(30 * 2 ** (attempt - 1))
    write_new(out / "rankings.json", {"kind": "t1_rankings", "mode": args.mode, "code": code,
              "model": MODEL if args.mode == "agent" else None, "planned_cells": len(manifest["cells"]),
              "completed_cells": completed, "attempts": attempts, "answer_key_sha256_lf": _sha256_lf(key_path),
              "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256_lf": _sha256_lf(p)}
                         for p in sorted(set(inputs))]})
    return 0 if completed == len(manifest["cells"]) else 2


if __name__ == "__main__":
    sys.exit(main())
