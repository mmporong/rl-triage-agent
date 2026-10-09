"""동결된 probes.py를 호출하며 CPU와 성공한 env.step 전이 수만 관측한다.

probe 수식·인자·RNG·환경 reset은 probes.py에 맡긴다. --metrics 출력은
별도 sidecar다. simulator_steps는 measure 안에서 성공한 제어 전이의 환경별
합이며 환경 생성·reset·PhysX 내부 substep은 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import probes  # noqa: E402


def observe_measure(original, records):
    def measured(probe, env, runner, ckpt):
        original_step = env.step
        num_envs = int(env.unwrapped.num_envs)
        row = {"probe": probe, "checkpoint_sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
               "num_envs": num_envs, "successful_step_calls": 0, "simulator_steps": 0, "completed": False}
        records.append(row)

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            row["successful_step_calls"] += 1
            row["simulator_steps"] += num_envs
            return result

        env.step = step
        try:
            result = original(probe, env, runner, ckpt)
            row["completed"] = True
            return result
        finally:
            env.step = original_step
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics", type=Path, required=True)
    opts, remaining = parser.parse_known_args(argv)
    if opts.metrics.exists():
        raise SystemExit("계측 sidecar는 덮어쓰지 않는다")
    records = []
    original, original_run, saved_argv = probes.measure, probes.run, sys.argv
    probes.measure = observe_measure(original, records)
    sys.argv = [str(ROOT / "evals/probes.py"), *remaining]

    def persist(path, completed, phase):
        report = {"schema": "probe_process_metrics_v1", "completed": completed, "measurements": records, "phase": phase,
                  "cpu_s": time.process_time(),
                  "cpu_scope": "isaac_python_lifetime_after_main" if phase == "after_main" else "isaac_python_lifetime_through_env_close_before_app_close",
                  "simulator_steps_scope": "successful_measure_env_step_calls_times_actual_num_envs",
                  "jobs_sha256": hashlib.sha256(Path(remaining[remaining.index("--jobs") + 1]).read_bytes()).hexdigest(),
                  "probes_sha256_lf": hashlib.sha256((ROOT / "evals/probes.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(report, indent=1, allow_nan=False) + "\n")

    def measured_run(args):
        completed = False
        try:
            result = original_run(args)
            completed = True
            return result
        finally:
            persist(opts.metrics, completed, "before_app_close")

    probes.run = measured_run
    completed = False
    try:
        code = probes.main()
        completed = code == 0
        return code
    finally:
        probes.measure, probes.run, sys.argv = original, original_run, saved_argv
        # Isaac shutdown이 Python으로 돌아오지 않아도 run의 종료 시점 계측은 남는다.
        path = opts.metrics.with_suffix(".post_close.json") if opts.metrics.exists() else opts.metrics
        persist(path, completed, "after_main")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
