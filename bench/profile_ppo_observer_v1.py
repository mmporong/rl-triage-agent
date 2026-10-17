"""실제 RSL CPU fixture에서 v4/v5 관찰자의 비용과 결과 동등성을 비교한다."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
import hashlib
import importlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import tempfile
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import isaac_training_guard_v1 as GUARD
import ppo_gradient_cpu_fixture_v2 as F

V4 = "ppo_gradient_training_v4"
V5 = "ppo_gradient_training_v5"
SEED = 17
UPDATES = 3
WARMUPS = 1
MEASURED_PAIRS = 6
TARGETS = ("item", "cpu", "tolist", "grad_snapshot", "summarize", "sink",
           "manifest_fingerprint", "json_serialization", "fsync", "sha256",
           "autograd_grad", "backward")
HELPER_NAMES = {"grad_snapshot": ("_gradient_snapshot",),
                "summarize": ("summarize", "_batch_summaries"),
                "manifest_fingerprint": ("manifest_fingerprint",)}


def _target_names(key: tuple[str, int, str]) -> list[str]:
    filename, _, name = key
    function_name, normalized = name.lower(), filename.replace("\\", "/").lower()
    targets = []
    for target in TARGETS[:7]:
        lowered = target.lower()
        helpers = HELPER_NAMES.get(target, ())
        if ((helpers and any(helper in function_name for helper in helpers))
                or function_name == lowered
                or function_name.startswith(f"<method '{lowered}' ")
                or function_name.startswith(f"<built-in method {lowered}")):
            targets.append(target)
    if "/json/" in normalized and function_name in ("dump", "dumps", "encode", "iterencode"):
        targets.append("json_serialization")
    if "fsync" in function_name:
        targets.append("fsync")
    if "sha256" in function_name:
        targets.append("sha256")
    if function_name == "grad" and "/torch/autograd/" in normalized:
        targets.append("autograd_grad")
    if (function_name == "backward" or "tensor.backward" in function_name) and "/torch/" in normalized:
        targets.append("backward")
    return targets


def _environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1인 CPU 전용 실행만 허용한다")
    if os.environ.get("RSL_CPU_SYNTHETIC_ONLY") != "1":
        raise RuntimeError("RSL_CPU_SYNTHETIC_ONLY=1이 필요하다")


def _safe_filename(filename: str) -> str:
    if filename.startswith("~") or filename.startswith("<"):
        return filename
    path = Path(filename)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _function(key: tuple[str, int, str]) -> dict:
    return {"file": _safe_filename(key[0]), "line": key[1], "name": key[2]}


def _profile_summary(profile: cProfile.Profile) -> dict:
    stats: Any = pstats.Stats(profile)
    matches = []
    counts = {target: 0 for target in TARGETS}
    for key, values in stats.stats.items():
        primitive_calls, total_calls, self_s, cumulative_s, callers = values
        targets = _target_names(key)
        if not targets:
            continue
        for target in targets:
            counts[target] += total_calls
        caller_rows = []
        for caller, caller_values in callers.items():
            if isinstance(caller_values, tuple):
                row = {"function": _function(caller), "calls": int(caller_values[1]),
                       "self_s": float(caller_values[2]), "cumulative_s": float(caller_values[3])}
            else:
                row = {"function": _function(caller), "calls": int(caller_values)}
            caller_rows.append(row)
        matches.append({"function": _function(key), "targets": targets,
                        "primitive_calls": primitive_calls, "total_calls": total_calls,
                        "self_s": self_s, "cumulative_s": cumulative_s,
                        "callers": sorted(caller_rows, key=lambda row: json.dumps(row, sort_keys=True))})
    return {"total_calls": stats.total_calls, "primitive_calls": stats.prim_calls,
            "total_cpu_s": stats.total_tt, "target_call_counts": counts,
            "functions": sorted(matches, key=lambda row: json.dumps(row["function"], sort_keys=True))}


@contextmanager
def _training_module(name: str) -> Iterator[object]:
    previous = F.T
    module = importlib.import_module(name)
    F.T = module
    try:
        yield module
    finally:
        F.T = previous


def _read_run(output: Path, result: dict, elapsed_s: float) -> dict:
    evidence = json.loads((output / "training_evidence.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (output / "records.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    return {"result": result, "evidence": evidence, "update_wall_s": evidence["update_wall_s"],
            "elapsed_s": elapsed_s, "records": records}


def _exercise(root: Path, label: str, module: str, *, arm: str = "NONE",
              observed: bool = True, updates: int = UPDATES, fail: bool = False,
              profiler: cProfile.Profile | None = None) -> dict:
    output = root / label
    started = time.perf_counter()
    with _training_module(module):
        if profiler is None:
            result = F.exercise(output, arm, observed, updates, fail=fail)
        else:
            profiler.enable()
            try:
                result = F.exercise(output, arm, observed, updates, fail=fail)
            finally:
                profiler.disable()
    elapsed_s = time.perf_counter() - started
    if fail:
        return {"result": result, "elapsed_s": elapsed_s,
                "failure": json.loads((output / "training_failure.json").read_text(encoding="utf-8")),
                "evidence_exists": (output / "training_evidence.json").exists()}
    return _read_run(output, result, elapsed_s)


def _canonical(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _equivalent(left: dict, right: dict) -> bool:
    keys = ("state", "losses", "updates", "completion_before_close")
    left_evidence = {key: value for key, value in left["evidence"].items()
                     if key != "update_wall_s"}
    right_evidence = {key: value for key, value in right["evidence"].items()
                      if key != "update_wall_s"}
    return all(left["result"][key] == right["result"][key] for key in keys) \
        and left_evidence == right_evidence and left["records"] == right["records"]


def _measurement(row: dict, version: str, pair: int, order: int) -> dict:
    return {"version": version, "pair": pair, "order": order,
            "elapsed_s": row["elapsed_s"], "update_wall_s": row["update_wall_s"],
            "state_sha256": _canonical(row["result"]["state"]),
            "losses_sha256": _canonical(row["result"]["losses"]),
            "records_sha256": _canonical(row["records"])}


def _median(measurements: list[dict], version: str, field: str) -> float:
    rows = [row for row in measurements if row["version"] == version]
    values = [row[field] if field == "elapsed_s" else sum(row[field]) for row in rows]
    return statistics.median(values)


def _profile(root: Path, module: str, label: str) -> dict:
    profiler = cProfile.Profile()
    run = _exercise(root, label, module, profiler=profiler)
    return {"version": "v4" if module == V4 else "v5", "profile": _profile_summary(profiler),
            "update_wall_s": run["update_wall_s"], "elapsed_s": run["elapsed_s"]}


def _validate_v5(root: Path) -> dict:
    off = _exercise(root, "validate-off", V5, observed=False)
    on = _exercise(root, "validate-on", V5)
    inverse = _exercise(root, "validate-inverse", V5, arm="INVERSE_RATIO")
    restored = _exercise(root, "validate-restored", V5, arm="RESTORED")
    dry = _exercise(root, "validate-dry", V5, updates=1)
    failure = _exercise(root, "validate-failure", V5, fail=True)
    off_on_exact = (off["result"]["state"] == on["result"]["state"]
                    and off["result"]["losses"] == on["result"]["losses"])
    restored_exact = (restored["result"]["state"] == off["result"]["state"]
                      and restored["result"]["losses"] == off["result"]["losses"])
    checks = {"off_on_exact": off_on_exact, "restored_exact": restored_exact,
              "normal_status": on["result"]["inspection"]["status"],
              "inverse_status": inverse["result"]["inspection"]["status"],
              "restored_status": restored["result"]["inspection"]["status"],
              "one_update_completed": dry["result"]["updates"] == 1,
              "failure_preserved": (failure["result"].get("failure_preserved") is True
                                    and failure["evidence_exists"] is False)}
    if checks != {"off_on_exact": True, "restored_exact": True, "normal_status": "normal",
                  "inverse_status": "abnormal", "restored_status": "normal",
                  "one_update_completed": True, "failure_preserved": True}:
        raise ValueError(f"v5 CPU 회귀가 다르다: {checks}")
    return checks


def run(mode: str) -> dict:
    _environment()
    if mode not in ("baseline", "compare"):
        raise ValueError("mode는 baseline 또는 compare여야 한다")
    if mode == "compare" and find_spec(V5) is None:
        raise FileNotFoundError("v5 학습 모듈이 아직 없다")
    with tempfile.TemporaryDirectory(prefix="ppo-observer-profile-") as temporary:
        root = Path(temporary)
        versions = [V4] if mode == "baseline" else [V4, V5]
        for version in versions:
            for warmup in range(WARMUPS):
                _exercise(root, f"warmup-{version}-{warmup}", version)
        profiles = [_profile(root, version, f"profile-{version}") for version in versions]
        measurements = []
        equivalent_pairs = []
        for pair in range(MEASURED_PAIRS):
            order = versions if mode == "baseline" or pair % 2 == 0 else list(reversed(versions))
            pair_rows = {}
            for position, version in enumerate(order):
                row = _exercise(root, f"measure-{pair}-{position}-{version}", version)
                pair_rows[version] = row
                measurements.append(_measurement(row, "v4" if version == V4 else "v5", pair, position))
            if mode == "compare":
                exact = _equivalent(pair_rows[V4], pair_rows[V5])
                equivalent_pairs.append(exact)
                if not exact:
                    raise ValueError(f"v4/v5 ON 결과가 pair {pair}에서 다르다")
        medians = {"v4_elapsed_s": _median(measurements, "v4", "elapsed_s"),
                   "v4_update_wall_s": _median(measurements, "v4", "update_wall_s")}
        validation = None
        if mode == "compare":
            medians.update(v5_elapsed_s=_median(measurements, "v5", "elapsed_s"),
                           v5_update_wall_s=_median(measurements, "v5", "update_wall_s"))
            validation = _validate_v5(root)
    return {"schema": "ppo_observer_profile_v1", "status": "complete", "mode": mode,
            "device": "cpu", "gpu_executions": 0, "actual_rsl_runner": True,
            "rsl_cpu_synthetic_only": True, "python_version": list(sys.version_info[:3]),
            "synthetic_cpu_seed": SEED, "warmups_per_version": WARMUPS,
            "measured_pairs": MEASURED_PAIRS, "alternating_order": mode == "compare",
            "profiled_separately_from_wall_measurements": True, "profiles": profiles,
            "measurements": measurements, "medians": medians,
            "v4_v5_exact_pairs": equivalent_pairs, "v5_validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.mode)
    GUARD.durable_new(arguments.output, result)
    print(json.dumps({"status": result["status"], "mode": result["mode"],
                      "measured_pairs": result["measured_pairs"],
                      "gpu_executions": result["gpu_executions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
