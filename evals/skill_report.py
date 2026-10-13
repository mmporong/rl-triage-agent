"""한 사례의 기존 기록으로 공식 Isaac Lab 디버깅 스킬 체크리스트 보고서를 만든다(GPU·모델·네트워크 없음).

  python evals/skill_report.py --case h02_s2027 --output-json <report.json> --output-md <report.md> [--isaaclab <경로>]

입력은 공개 결과·고리 기록·공개 설정과 Isaac Lab 로그의 체크포인트다. 공개 설정은 실험 프로토콜에 묶인
설정 SHA와 같을 때만 근거로 쓴다. 출력 파일은 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rl_triage.skill_report import CaseInputs, build_report, render_markdown  # noqa: E402

V3_RESULTS = {"h02": "reward_resume_dev3_20261013", "h06": "motor_resume_dev_20261013",
              "h03": "timeout_resume_dev_20261013"}
FAULT_BY_PREFIX = {"baseline_p0c": "NONE", "h02": "F1", "h06": "F2", "h03": "F6"}
LOG_ROOT = Path("logs/rsl_rl/unitree_go2_flat")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_env_sha(seed: str, fault: str | None) -> str | None:
    """공개 설정이 이 실행의 저장 설정과 같음을 보이는 프로토콜 결합 SHA."""
    if fault is None:
        return None
    if seed == "2027":
        runs = json.loads((ROOT / "bench/protocols/training_correction_v3.json").read_text(encoding="utf-8"))["resume_inputs"]["runs"]
        row = runs.get(fault)
    else:
        seeds = json.loads((ROOT / "bench/protocols/training_correction_v4.json").read_text(encoding="utf-8"))["resume_inputs"]["seeds"]
        row = seeds.get(seed, {}).get("runs", {}).get(fault)
    return row["hashes"]["environment"] if row else None


def _r1_alerts(case: str) -> list[int] | None:
    import build_replay_page as B
    series = {}
    for name in B.CASES:
        for seed in B.SEEDS:
            path = ROOT / f"bench/runs_p0c/{name}_s{seed}.telemetry.json"
            if not path.is_file():
                return None
            series[f"{name}_s{seed}"] = json.loads(path.read_text(encoding="utf-8"))["series"]["Metrics/base_velocity/error_vel_xy"]
    alerts = B.r1_alerts(series, {name: name for name in B.CASES})
    return alerts.get(case, {}).get("alert_iterations")


def inputs_for(case: str, isaaclab: Path | None) -> CaseInputs:
    match = re.fullmatch(r"(baseline_p0c|h0[1-6])_s(202[678])", case)
    if match is None:
        raise ValueError("case must be a P0-C run name such as h02_s2027")
    prefix, seed = match[1], match[2]
    fault = FAULT_BY_PREFIX.get(prefix)
    env_config = f"bench/reference/params/p0c_seed{seed}/env.yaml"
    bound = _bound_env_sha(seed, fault)
    if bound is None or not (ROOT / env_config).is_file() or _sha(ROOT / env_config) != bound:
        env_config = None
    checkpoint = None
    if isaaclab is not None and (isaaclab / LOG_ROOT).is_dir():
        runs = [path for path in (isaaclab / LOG_ROOT).iterdir() if path.is_dir() and path.name.endswith(f"_{case}")]
        if len(runs) == 1 and (runs[0] / "model_299.pt").is_file():
            checkpoint = (LOG_ROOT / runs[0].name / "model_299.pt").as_posix()
    result_tag = V3_RESULTS.get(prefix) if seed == "2027" else (f"reward_v4_s{seed}_20261013" if prefix == "h02" else None)
    return CaseInputs(
        case=case,
        telemetry=f"bench/runs_p0c/{case}.telemetry.json",
        fixed_report=f"evals/results/p0c_fixed_eval_20261009/runs/{case}.json",
        fixed_summary="evals/results/p0c_fixed_eval_20261009/summary.json",
        fixed_protocol="bench/protocols/fixed_eval_v1.json",
        loop_state=f"evals/loops/{case}/state.json",
        loop_ledger=f"evals/loops/{case}/ledger.jsonl",
        correction_ledger=f"evals/loops/{case}/correction_ledger.jsonl",
        probe_report=f"evals/results/p1a_probes_holdout_20261009/probes/{case}.json",
        env_config=env_config,
        run_protocol="bench/protocols/training_correction_v3.json" if seed == "2027" else "bench/protocols/training_correction_v4.json",
        correction_result=f"evals/results/{result_tag}/result.json" if result_tag else None,
        correction_registry="src/rl_triage/correction_registry.py",
        checkpoint=checkpoint,
        r1_alert_iterations=_r1_alerts(case),
    )


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    isaaclab = args.isaaclab.resolve() if args.isaaclab.is_dir() else None
    report = build_report(inputs_for(args.case, isaaclab), ROOT, isaaclab)
    _write_new(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_new(args.output_md, render_markdown(report))
    print(json.dumps({"case": report["case"], "counts": report["counts"], "missing_inputs": report["missing_inputs"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
