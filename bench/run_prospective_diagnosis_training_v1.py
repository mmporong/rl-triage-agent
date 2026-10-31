"""새 seed 진단 학습 protocol을 준비·검증하고 등록된 GPU batch를 순차 실행한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import core_seed_guard_v1 as SEEDS
import lineage_compound_first_recovery_v1 as FIRST
import lineage_lease_context_v2 as LEASE
import prospective_diagnosis_inputs_v1 as INPUT
import prospective_diagnosis_training_v1 as TRAIN
import run_runtime_readback_smoke as OWNED

PREFLIGHT_SCHEMA = "prospective_diagnosis_preflight_v1"
BATCH_SCHEMA = "prospective_diagnosis_batch_result_v1"
OWNER = "rl-triage-lineage-sources-v1"
MAX_WALL_S = 14400


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def prepare_protocol(seeds_path: Path, categories_path: Path, runtime_pins_path: Path,
                     out: Path) -> dict[str, Any]:
    """JSON 입력 세 개에서 공개 protocol 후보를 write-once로 만든다."""
    seeds = json.loads(Path(seeds_path).read_text(encoding="utf-8"))
    categories = json.loads(Path(categories_path).read_text(encoding="utf-8"))
    runtime = _read(runtime_pins_path)
    _require(isinstance(seeds, list) and isinstance(categories, list),
             "seed/category 입력은 JSON 배열이어야 한다")
    value = TRAIN.prepare_protocol(seeds, categories, runtime)
    _write(Path(out).resolve(), value)
    return value


def _manifests(protocol_path: Path, protocol: dict[str, Any], anchor: str,
               root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in protocol["seeds"]:
        for category in protocol["categories"]:
            rows.append(TRAIN.build_manifest(
                protocol_path, protocol, anchor, stage=root / f"s{seed}-{category.lower()}",
                seed=seed, category=category, run_name=f"pd_{category.lower()}_s{seed}"))
    return rows


def preflight(protocol_path: Path, anchor: str, lab: Path, isaac_python: Path,
              out: Path) -> dict[str, Any]:
    """실행 protocol·runtime pin·seed freshness·모든 manifest를 CPU에서 소비한다."""
    out, lab, isaac_python = Path(out).resolve(), Path(lab).resolve(), Path(isaac_python).resolve()
    _require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    _require(isaac_python.is_file(), "Isaac Python이 없다")
    protocol = TRAIN.verify_protocol(Path(protocol_path), anchor, lab)
    freshness = SEEDS.assert_fresh_seeds(protocol["seeds"], lab)
    manifests = _manifests(Path(protocol_path), protocol, anchor, out / "planned")
    _require(len(manifests) == len(protocol["seeds"]) * len(protocol["categories"]),
             "등록된 seed×category manifest 수가 다르다")
    _require(1 + len(manifests) <= 9, "smoke 포함 등록 GPU job 수가 batch 상한을 넘는다")
    value = {
        "schema": PREFLIGHT_SCHEMA, "status": "ready", "protocol": _ref(protocol_path),
        "anchor": anchor, "lab": str(lab), "isaac_python": str(isaac_python),
        "freshness": freshness,
        "planned": [{"seed": row["seed"], "category": category,
                     "runtime_arm": row["arm"], "manifest": row}
                    for row, category in zip(
                        manifests, protocol["categories"] * len(protocol["seeds"]), strict=True)],
        "bounds": {"training_jobs": 1 + len(manifests),
                   "fixed_eval_jobs": len(manifests),
                   "owned_sdk_jobs": 1 + 2 * len(manifests),
                   "max_training_jobs": 9, "max_owned_sdk_jobs": 17,
                   "wall_s": MAX_WALL_S,
                   "automatic_retry": False, "model_requests": 0, "repairs": 0},
    }
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "preflight.json", value)
    return value


def execute(
    protocol_path: Path, anchor: str, lab: Path, isaac_python: Path, out: Path,
    smoke_checkpoint_sha256: str, *, runner: Callable = OWNED.run_owned,
    wait: Callable = FIRST._lease_wait,
) -> dict[str, Any]:
    """외부 lease 아래 smoke 1회와 seed×category 학습을 순차 실행한다."""
    out, lab, isaac_python = Path(out).resolve(), Path(lab).resolve(), Path(isaac_python).resolve()
    _require(not out.exists(), "batch out은 fresh 경로여야 한다")
    _require(len(smoke_checkpoint_sha256) == 64
             and all(char in "0123456789abcdef" for char in smoke_checkpoint_sha256),
             "smoke checkpoint SHA 형식이 다르다")
    protocol = TRAIN.verify_protocol(Path(protocol_path), anchor, lab)
    _require(1 + len(protocol["seeds"]) * len(protocol["categories"]) <= 9,
             "smoke 포함 등록 GPU job 수가 batch 상한을 넘는다")
    SEEDS.assert_fresh_seeds(protocol["seeds"], lab)
    out.mkdir(parents=True, exist_ok=False)
    lease = LEASE.lease_context(out / "lease-initial")
    _require(lease.get("owner") == OWNER, "공용 Use-GpuLease 소유자가 다르다")
    start = time.monotonic()
    stages: list[dict[str, Any]] = []
    summary = {"schema": BATCH_SCHEMA, "status": "running", "protocol": _ref(protocol_path),
               "anchor": anchor, "seeds": protocol["seeds"],
               "categories": protocol["categories"], "stages": stages,
               "automatic_retry": False, "model_requests": 0, "repairs": 0}

    def budget() -> None:
        elapsed = time.monotonic() - start
        _require(math.isfinite(elapsed) and 0 <= elapsed < MAX_WALL_S,
                 "prospective batch wall 상한이 끝났다")
        TRAIN.verify_protocol(Path(protocol_path), anchor, lab)
        current = LEASE.lease_context()
        _require(current.get("owner") == OWNER and current.get("lease_id") == lease.get("lease_id"),
                 "실행 중 공용 lease identity가 바뀌었다")

    try:
        budget()
        smoke_seed = protocol["seeds"][0]
        smoke_fixed = TRAIN.build_manifest(
            Path(protocol_path), protocol, anchor, stage=out / "smoke/train",
            seed=smoke_seed, category="NONE", run_name=f"pd_smoke_s{smoke_seed}",
            mode="inspection", source_checkpoint_sha256=smoke_checkpoint_sha256)
        smoke = TRAIN.RUNTIME.run_training(
            out / "smoke/train", smoke_fixed, lab=lab, isaac_python=isaac_python,
            runner=runner, wait=wait)
        smoke_path = out / "smoke/result.json"
        _write(smoke_path, {"schema": "prospective_diagnosis_smoke_v1", "status": "complete",
                            "seed": smoke_seed, "source_checkpoint_sha256": smoke_checkpoint_sha256,
                            "checkpoint_loaded": False, "training": smoke})
        stages.append({"stage": "smoke", "result": _ref(smoke_path)})
        budget()
        for seed in protocol["seeds"]:
            for category in protocol["categories"]:
                budget()
                label = f"s{seed}-{category.lower()}"
                fixed = TRAIN.build_manifest(
                    Path(protocol_path), protocol, anchor, stage=out / label / "train",
                    seed=seed, category=category, run_name=f"pd_{category.lower()}_s{seed}")
                result = TRAIN.run_case(
                    out / label, fixed, category=category, lab=lab, isaac_python=isaac_python,
                    runner=runner, wait=wait)
                budget()
                result_path = out / label / "result.json"
                collected = INPUT.collect(
                    out / label / "training-result.json", category=category,
                    out=out / label / "input")
                collected_path = out / label / "input" / INPUT.COLLECTED
                stages.append({"stage": label, "result": _ref(result_path),
                               "checkpoint": result["checkpoint"],
                               "input": _ref(collected_path),
                               "input_binding": collected["source_binding"]})
        summary.update(status="complete", wall_s=time.monotonic() - start)
        _write(out / "result.json", summary)
        return summary
    except BaseException as error:
        _write(out / "failure.json", {**summary, "status": "invalid",
               "error_type": type(error).__name__, "error": str(error),
               "wall_s": time.monotonic() - start})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("seeds", "categories", "runtime-pins", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("preflight")
    for name in ("protocol", "lab", "isaac-python", "out"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--anchor", required=True)
    run = sub.add_parser("execute")
    for name in ("protocol", "lab", "isaac-python", "out"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    run.add_argument("--smoke-checkpoint-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        prepare_protocol(args.seeds, args.categories, args.runtime_pins, args.out)
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.lab, args.isaac_python, args.out)
    else:
        execute(args.protocol, args.anchor, args.lab, args.isaac_python, args.out,
                args.smoke_checkpoint_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
