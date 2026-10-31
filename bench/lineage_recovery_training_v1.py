"""기존 residual 교정 학습을 terminal close 증거 경계로 감싸는 child wrapper."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import isaac_training_guard_v1 as GUARD
import residual_training_v1 as RESIDUAL

SCHEMA = RESIDUAL.SCHEMA
EVIDENCE_SCHEMA = "lineage_recovery_training_evidence_v1"
COMPLETION_SCHEMA = "lineage_recovery_training_completion_v1"
BOUND_SOURCES = frozenset({
    *RESIDUAL.MINIMUM_BOUND_SOURCES,
    "bench/lineage_recovery_training_v1.py",
    "tests/test_lineage_recovery_training_v1.py",
    "bench/isaac_training_guard_v1.py",
    "bench/residual_training_v1.py",
})


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """기존 폐쇄형 manifest를 유지하면서 wrapper source 결합만 강화한다."""
    data = RESIDUAL.validate_manifest(value)
    missing = BOUND_SOURCES - set(data["source_sha256_lf"])
    if missing:
        raise ValueError(f"lineage recovery 필수 source binding이 없다: {sorted(missing)}")
    return data


def manifest_fingerprint(value: Mapping[str, Any]) -> str:
    data = validate_manifest(value)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_binding(value: Mapping[str, Any], *, lab: Path) -> None:
    """residual protocol/commit/runtime과 wrapper source closure를 함께 검증한다."""
    data = validate_manifest(value)
    RESIDUAL.verify_source_binding(data, runtime_root=Path(lab).resolve())


def _failure(out: Path, exc: BaseException, stage: str) -> None:
    path = Path(out) / "failure.json"
    if path.exists():
        return
    try:
        GUARD.durable_new(path, {
            "status": "invalid", "stage": stage, "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })
    except FileExistsError:
        pass


def _checkpoint(runner: object, expected_iteration: int) -> dict[str, Any]:
    log_dir = getattr(runner, "log_dir", None)
    if not isinstance(log_dir, (str, os.PathLike)):
        raise TypeError("runner.log_dir checkpoint 위치가 없다")
    directory = Path(log_dir).resolve()
    path = directory / f"model_{expected_iteration}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"최종 checkpoint model_{expected_iteration}.pt가 없다")
    return {"path": str(path), "log_dir": str(directory), "iteration": expected_iteration,
            "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _validate_residual_outputs(data: Mapping[str, Any]) -> dict[str, Any]:
    records_path, readback_path = Path(data["records_path"]), Path(data["readback_path"])
    if not records_path.is_file() or not readback_path.is_file():
        raise FileNotFoundError("residual observer records/readback이 없다")
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    report = RESIDUAL.OBSERVE.assemble_inspection(rows, RESIDUAL.observer_manifest(data))
    remaining = list(RESIDUAL.FAULTS.remaining_faults(
        data["original_faults"], data["applied_corrections"]))
    expected_status = "abnormal" if "R1" in remaining else "normal"
    if report.get("records") != data["runtime_updates_to_observe"]:
        raise ValueError("residual observer 첫 3 update가 완전하지 않다")
    if report.get("status") != expected_status:
        raise ValueError("residual observer 판정이 남은 R1 의미와 다르다")
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    if (not isinstance(readback, dict)
            or readback.get("schema") != "residual_training_readback_v1"
            or readback.get("original_faults") != data["original_faults"]
            or readback.get("applied_corrections") != data["applied_corrections"]
            or readback.get("remaining_faults") != remaining):
        raise ValueError("residual readback이 manifest의 원본/누적 교정 의미와 다르다")
    return {"remaining_faults": remaining, "report": report,
            "records": {"path": str(records_path.resolve()), "sha256": _sha(records_path)},
            "readback": {"path": str(readback_path.resolve()), "sha256": _sha(readback_path)}}


def execute(manifest_path: Path, lab: Path, out: Path) -> dict[str, Any]:
    """residual 학습을 실행하고 실제 learn 반환과 terminal close 전 봉인을 검증한다."""
    manifest_path, lab, out = Path(manifest_path).resolve(), Path(lab).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("out은 존재하지 않는 fresh 경로여야 한다")
    out.mkdir(parents=True)
    stage = "manifest"
    try:
        data = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest_sha = manifest_fingerprint(data)
        stage = "binding"
        verify_binding(data, lab=lab)
        upstream = (lab / "scripts/reinforcement_learning/rsl_rl/train.py").resolve()
        if not upstream.is_file():
            raise FileNotFoundError("Isaac Lab 상류 train.py가 없다")

        state: dict[str, Any] = {
            "ppo": [], "learn_called": False, "learn_returned": False, "load_count": 0,
            "environment_exception": None, "completion": None, "scope": None,
        }

        def before_close(_app: object) -> None:
            verify_binding(data, lab=lab)
            if state["environment_exception"] is not None:
                raise RuntimeError("환경 생성 예외를 상류가 삼켰다: " + state["environment_exception"])
            if not state["learn_returned"] or not isinstance(state["completion"], dict):
                raise RuntimeError("runner.learn 반환 완료 증거 없이 앱을 닫을 수 없다")
            if state["load_count"] != 0:
                raise RuntimeError("Runner.load 호출이 감지됐다")
            residual = _validate_residual_outputs(data)
            completion = dict(state["completion"])
            completion["learning_completed"] = True
            completion["recorded_before_app_close"] = True
            completion["remaining_faults"] = residual["remaining_faults"]
            completion["residual_inspection"] = residual["report"]
            records_path = out / GUARD.RECORDS
            with records_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(completion, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            evidence = {
                "schema": EVIDENCE_SCHEMA, "manifest_sha256": manifest_sha,
                "updates": completion["updates"], "update_count": completion["update_count"],
                "learning_completed": True, "learn_returned": True,
                "recorded_before_app_close": True, "arm": data["arm"], "seed": data["seed"],
                "original_faults": data["original_faults"],
                "applied_corrections": data["applied_corrections"],
                "remaining_faults": residual["remaining_faults"],
                "source_checkpoint_sha256": data["source_checkpoint_sha256"],
                "fresh_initialization": True, "checkpoint_loaded": False,
                "resume_requested": False, "log_dir": completion["checkpoint"]["log_dir"],
                "checkpoint": completion["checkpoint"],
                "ppo_readback": completion["ppo_readback"],
                "residual_records": residual["records"],
                "residual_readback": residual["readback"],
            }
            GUARD.durable_new(out / GUARD.EVIDENCE, evidence)
            GUARD.durable_new(out / GUARD.CLOSE_READY, {
                "schema": GUARD.CLOSE_SCHEMA, "manifest_sha256": manifest_sha,
                "updates": completion["updates"],
                "evidence_sha256_lf": _sha_lf(out / GUARD.EVIDENCE),
                "records_sha256_lf": _sha_lf(records_path),
            })

        def adapter(path_name: str | os.PathLike[str], init_globals=None, run_name=None):
            if Path(path_name).resolve() != upstream or init_globals is not None or run_name != "__main__":
                raise ValueError("residual wrapper는 등록된 Lab train.py의 __main__ 실행만 허용한다")
            try:
                gymnasium = importlib.import_module("gymnasium")
                ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
                runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
                original_ppo_init, original_learn = ppo_class.__init__, runner_class.learn
                original_load = getattr(runner_class, "load", None)
                if not callable(original_load):
                    raise TypeError("OnPolicyRunner.load가 호출 가능해야 한다")
                original_make = gymnasium.make
                state["scope"] = (gymnasium, original_make, ppo_class, original_ppo_init,
                                  runner_class, original_learn, original_load)

                def ppo_init(this, *args, **kwargs):
                    original_ppo_init(this, *args, **kwargs)
                    if state["ppo"]:
                        raise ValueError("fresh 학습에는 PPO 인스턴스가 하나여야 한다")
                    configured = {"learning_epochs": this.num_learning_epochs,
                                  "mini_batches": this.num_mini_batches}
                    if configured != {"learning_epochs": 5, "mini_batches": 4}:
                        raise ValueError("PPO 초기 configured epochs/minibatches가 등록값과 다르다")
                    original_update = getattr(this, "update", None)
                    if not callable(original_update):
                        raise TypeError("PPO.update가 호출 가능해야 한다")
                    row = {"instance": this, "configured": configured,
                           "original_update": original_update, "update_count": 0,
                           "had_instance_update": "update" in this.__dict__,
                           "instance_update": this.__dict__.get("update")}

                    def counted_update(_this, *update_args, **update_kwargs):
                        result = original_update(*update_args, **update_kwargs)
                        row["update_count"] += 1
                        if row["update_count"] > data["max_iterations"]:
                            raise ValueError("실제 PPO.update 성공 반환이 등록 예산을 초과했다")
                        return result

                    this.update = MethodType(counted_update, this)
                    state["ppo"].append(row)

                def load(_this, *_args, **_kwargs):
                    state["load_count"] += 1
                    raise RuntimeError("source checkpoint SHA는 provenance이며 Runner.load는 금지된다")

                def learn(this, *args, **kwargs):
                    if state["learn_called"]:
                        raise ValueError("OnPolicyRunner.learn은 한 번만 호출해야 한다")
                    iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
                    random_ep = kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None)
                    if iterations != data["max_iterations"] or random_ep is not True:
                        raise ValueError("learn update 수 또는 fresh episode initialization 인자가 다르다")
                    state["learn_called"] = True
                    result = original_learn(this, *args, **kwargs)
                    if len(state["ppo"]) != 1 or getattr(this, "alg", None) is not state["ppo"][0]["instance"]:
                        raise ValueError("runner와 PPO instance 결합이 다르다")
                    row = state["ppo"][0]
                    if row["update_count"] != data["max_iterations"]:
                        raise ValueError("실제 PPO.update 성공 반환 횟수가 등록 예산과 다르다")
                    if state["load_count"]:
                        raise RuntimeError("Runner.load 호출이 감지됐다")
                    effective = {"learning_epochs": row["instance"].num_learning_epochs,
                                 "mini_batches": row["instance"].num_mini_batches}
                    remaining = RESIDUAL.FAULTS.remaining_faults(
                        data["original_faults"], data["applied_corrections"])
                    expected = {"learning_epochs": 1 if "R1" in remaining else 5,
                                "mini_batches": 4}
                    if effective != expected:
                        raise ValueError("PPO effective readback이 남은 R1 의미와 다르다")
                    checkpoint = _checkpoint(this, data["max_iterations"] - 1)
                    state["learn_returned"] = True
                    state["completion"] = {
                        "schema": COMPLETION_SCHEMA, "arm": data["arm"], "seed": data["seed"],
                        "updates": row["update_count"], "update_count": row["update_count"],
                        "learn_called": True, "learn_returned": True,
                        "fresh_initialization": True, "checkpoint_loaded": False,
                        "resume_requested": False, "source_checkpoint_sha256": data["source_checkpoint_sha256"],
                        "original_faults": data["original_faults"],
                        "applied_corrections": data["applied_corrections"],
                        "ppo_readback": {"configured": row["configured"], "effective": effective},
                        "checkpoint": checkpoint,
                    }
                    return result

                def make(*args, **kwargs):
                    try:
                        return original_make(*args, **kwargs)
                    except BaseException as exc:
                        state["environment_exception"] = f"{type(exc).__name__}: {exc}"
                        raise

                ppo_class.__init__, runner_class.learn, runner_class.load = ppo_init, learn, load
                gymnasium.make = make
                return GUARD.run_upstream(upstream, out, before_close)
            except BaseException as exc:
                state["adapter_failure"] = f"{type(exc).__name__}: {exc}"
                raise

        original_runpy, original_cwd = RESIDUAL.runpy, Path.cwd()
        stage = "upstream"
        try:
            RESIDUAL.runpy = SimpleNamespace(run_path=adapter)
            os.chdir(lab)
            RESIDUAL.execute(manifest_path)
        finally:
            os.chdir(original_cwd)
            RESIDUAL.runpy = original_runpy
            scope = state["scope"]
            if scope is not None:
                (_gymnasium, _residual_make, ppo_class, original_ppo_init,
                 runner_class, original_learn, original_load) = scope
                for row in state["ppo"]:
                    if row["had_instance_update"]:
                        row["instance"].update = row["instance_update"]
                    else:
                        row["instance"].__dict__.pop("update", None)
                ppo_class.__init__, runner_class.learn, runner_class.load = (
                    original_ppo_init, original_learn, original_load)
        GUARD.validate_completion(out, manifest_sha)
        return {"schema": SCHEMA + "_lineage_recovery_result_v1", "status": "complete",
                "manifest_sha256": manifest_sha, "updates": data["max_iterations"]}
    except BaseException as exc:
        _failure(out, exc, stage)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        if args.validate_only:
            print(json.dumps({"status": "valid", "schema": value["schema"]}))
            return 0
        execute(args.manifest, args.lab, args.out)
        return 0
    except BaseException as exc:  # noqa: BLE001
        if not args.validate_only and not args.out.exists():
            args.out.mkdir(parents=True, exist_ok=True)
            _failure(args.out, exc, "entrypoint")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
