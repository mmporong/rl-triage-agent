"""사전등록한 개발 평가 JSON에 행동 게이트를 적용하고 추가 실행 없이 종료한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from rl_triage import behavior_oracle as O
from rl_triage import experiment_cost as C

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ("model_calls", "probe_executions", "approved", "consumed", "receipts", "interventions")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_source(source):
    path = (ROOT / source["path"]).resolve()
    path.relative_to(ROOT.resolve())
    if sha(path) != source["sha256_lf"]:
        raise ValueError(f"입력 SHA가 다르다: {source['path']}")
    return path


def verified_protocol(path):
    execution = read(path)
    base_path = checked_source(execution["base_protocol"])
    base = read(base_path)
    files = {**base["code_sha256_lf"], **execution["code_sha256_lf"]}
    relatives = [path.resolve().relative_to(ROOT.resolve()).as_posix(), execution["base_protocol"]["path"], *files]
    subprocess.run(["git", "ls-files", "--error-unmatch", *relatives], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *relatives], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for relative, expected in files.items():
        checked_source({"path": relative, "sha256_lf": expected})
    for source in base["sources"].values():
        checked_source(source)
    if execution["schema"] != "t2_gate_execution_prereg_v1" or base["schema"] != "t2_no_intervention_prereg_v1":
        raise ValueError("T2 사전등록 버전이 다르다")
    if [case["case"] for case in base["cases"]] != ["X2", "X3", "NONE"]:
        raise ValueError("사전등록한 3건의 선정과 순서가 다르다")
    return execution, base


def gate_case(spec, base):
    cpu, wall = time.process_time(), time.perf_counter()
    row = {"case": spec["case"], "input_hashes_verified": False, "reference_valid": False,
           "behavior_label": "undetermined", "decision": "stop_undetermined", "error": None,
           "loop_opened": False, **{key: 0 for key in COUNTERS}, "gpu_wall_s": 0,
           "checkpoint_file_rehashed": False, "assessment": None, "contract": None}
    try:
        paths = {key: checked_source(source) for key, source in spec["inputs"].items()}
        reports = {}
        for key, path in paths.items():
            raw = path.read_bytes()
            if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != spec["inputs"][key]["sha256_lf"]:
                raise ValueError("평가 입력이 읽는 동안 달라졌다")
            reports[key] = json.loads(raw)
        row["input_hashes_verified"] = True
        for key, report in reports.items():
            if report["checkpoint"]["sha256"] != spec["checkpoint_sha256_as_recorded"][key]:
                raise ValueError("평가 JSON의 checkpoint SHA가 사전등록과 다르다")
        contract = O.register_contract(reports["reference"], [], base["absolute_gates"])
        if contract["evaluation"] != spec["reference_identity"] or contract["bands"] != spec["relative_bands"]:
            raise ValueError("기준 평가 조건·상대 허용 범위가 사전등록과 다르다")
        row["reference_valid"] = True
        assessment = O.assess(reports["candidate"], reports["reference"], contract)
        row.update(contract=contract, assessment=assessment, behavior_label=assessment["label"])
        if assessment["label"] == "healthy":
            row["decision"] = "do_not_open_loop"
        elif assessment["label"] == "unhealthy":
            row["decision"] = "outside_T2_healthy_scope_stop"
    except (O.OracleError, ValueError, KeyError, TypeError, OSError) as exc:
        row["error"] = str(exc)
    row["gate_cpu_s"] = time.process_time() - cpu
    row["gate_wall_s"] = time.perf_counter() - wall
    return row


def loop_snapshot():
    folder = ROOT / "evals/loops"
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(folder.rglob("*")) if path.is_file()}


def run(prereg_path, tag):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag에는 영문·숫자·밑줄·하이픈만 쓴다")
    out = ROOT / "evals/results" / tag
    if out.exists():
        raise FileExistsError(f"기존 결과 tag는 덮어쓰지 않는다: {tag}")
    cpu, wall = time.process_time(), time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    execution, base = verified_protocol(prereg_path)
    planned = execution.get("planned_tag")
    if not isinstance(planned, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", planned) or tag != planned:
        raise ValueError("실행 tag가 사전등록한 planned_tag와 다르거나 유효하지 않다")
    before = loop_snapshot()
    rows = [gate_case(spec, base) for spec in base["cases"]]
    after = loop_snapshot()
    elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
    counts = {key: sum(row[key] for row in rows) for key in COUNTERS}
    acceptance = {"selected_three_cases_healthy": len(rows) == 3 and all(row["behavior_label"] == "healthy" and row["error"] is None for row in rows),
                  "all_inputs_verified": all(row["input_hashes_verified"] and row["reference_valid"] for row in rows),
                  "no_loop_opened": not any(row["loop_opened"] for row in rows),
                  "no_execution_or_intervention": all(value == 0 for value in counts.values()),
                  "loop_files_unchanged": before == after,
                  "no_isaac_gpu_execution": all(row["gpu_wall_s"] == 0 for row in rows)}
    costs = {key: 0 for key in C.COST_KEYS}
    costs.update(cpu_s=elapsed_cpu, execution_wall_s=elapsed_wall)
    summary = {"schema": "t2_gate_result_v1", "started_at": started_at,
               "execution_prereg_sha256_lf": sha(prereg_path), "base_protocol": execution["base_protocol"],
               "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "cases": len(rows), "healthy": sum(row["behavior_label"] == "healthy" for row in rows),
               "rows": rows, "counts": counts, "loop_snapshots": {"before": before, "after": after},
               "costs": costs, "acceptance": acceptance, "accepted": all(acceptance.values()),
               "scope": "offline_gate_on_previously_seen_development_reports; no_model_probe_repair_training",
               "cost_scope": "gate Python CPU and elapsed from preflight through decisions and loop snapshots; Git child CPU, interpreter/import startup and result writes excluded",
               "claim_limits": [*base["claim_limits"][:-1], "gate-only execution; general loop CLI routing not changed", "checkpoint hashes checked as recorded in reports; checkpoint files not rehashed", "legacy condition metrics; raw sums absent, assessment.metrics_verified remains false"]}
    out.mkdir(parents=True, exist_ok=False)
    with (out / "summary.json").open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    result = run(args.prereg, args.tag)
    print(json.dumps({key: result[key] for key in ("cases", "healthy", "counts", "costs", "accepted")}, ensure_ascii=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
