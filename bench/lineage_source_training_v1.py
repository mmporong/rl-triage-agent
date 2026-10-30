"""새 source 계보 비교의 고정 300-update Isaac Lab 학습 child."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import isaac_training_guard_v1 as GUARD

SCHEMA = "lineage_source_training_manifest_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ARMS = ("NONE", "F1", "F2", "F3", "F5", "COMPOUND")
SOURCE_FILES = (
    "bench/lineage_source_training_v1.py",
    "tests/test_lineage_source_training_v1.py",
    "bench/hidden_faults.py",
    "bench/external_faults.py",
    "bench/isaac_training_guard_v1.py",
)
JOB_FIELDS = ("job_id", "case_id", "arm", "seed", "run_name", "task", "num_envs", "max_iterations")
REQUIRED = {
    "schema", "purpose", *JOB_FIELDS, "source_commit", "protocol_relative", "protocol_sha256_lf",
    "source_sha256_lf", "runtime_source_sha256_lf",
}
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _relative(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not Path(value).is_absolute() and ".." not in Path(value).parts)


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def manifest_fingerprint(value: Mapping[str, Any]) -> str:
    """검증된 manifest의 canonical JSON SHA-256을 반환한다."""
    data = validate_manifest(value)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_map(value: Any, name: str) -> dict[str, str]:
    if (not isinstance(value, dict) or not value
            or any(not _relative(path) or not _hex(digest, 64) for path, digest in value.items())):
        raise ValueError(f"{name} 형식이 잘못됐다")
    return dict(value)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """heavy runtime import 없이 manifest의 폐쇄형 계약을 검증한다."""
    if not isinstance(value, Mapping):
        raise TypeError("manifest는 JSON 객체여야 한다")
    data = dict(value)
    if set(data) != REQUIRED:
        raise ValueError(f"manifest 필드가 고정 계약과 다르다: {sorted(set(data) ^ REQUIRED)}")
    if data["schema"] != SCHEMA or data["purpose"] not in {"smoke", "source"} or data["arm"] not in ARMS:
        raise ValueError("manifest schema, purpose 또는 arm이 잘못됐다")
    if any(not isinstance(data[key], str) or _ID.fullmatch(data[key]) is None for key in ("job_id", "case_id")):
        raise ValueError("job_id와 case_id 형식이 잘못됐다")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    if not isinstance(data["run_name"], str) or _RUN_NAME.fullmatch(data["run_name"]) is None:
        raise ValueError("run_name 형식 또는 80자 제한이 잘못됐다")
    expected_iterations = 1 if data["purpose"] == "smoke" else 300
    if (data["task"] != TASK or data["num_envs"] != 4096
            or data["max_iterations"] != expected_iterations
            or (data["purpose"] == "smoke" and data["arm"] != "NONE")):
        raise ValueError("task·환경 수·update 예산이 고정 source 학습 계약과 다르다")
    if not _hex(data["source_commit"], 40) or not _hex(data["protocol_sha256_lf"], 64):
        raise ValueError("source commit 또는 protocol SHA 형식이 잘못됐다")
    if not _relative(data["protocol_relative"]):
        raise ValueError("protocol_relative는 안전한 저장소 상대경로여야 한다")
    source = _hash_map(data["source_sha256_lf"], "source_sha256_lf")
    runtime = _hash_map(data["runtime_source_sha256_lf"], "runtime_source_sha256_lf")
    if not set(SOURCE_FILES) <= set(source):
        raise ValueError("필수 source binding이 없다")
    data["source_sha256_lf"], data["runtime_source_sha256_lf"] = source, runtime
    return data


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True)


def _protocol_job(protocol: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise TypeError("protocol은 JSON 객체여야 한다")
    if data["purpose"] == "smoke":
        row = protocol.get("smoke_job")
        matches = [row] if isinstance(row, dict) and row.get("job_id") == data["job_id"] else []
    else:
        rows = protocol.get("training_jobs")
        if not isinstance(rows, list) or len(rows) != 18:
            raise ValueError("protocol.training_jobs는 18개 배열이어야 한다")
        matches = [row for row in rows if isinstance(row, dict) and row.get("job_id") == data["job_id"]]
    if len(matches) != 1:
        raise ValueError("protocol에서 purpose에 맞는 유일한 job_id를 찾지 못했다")
    expected = {field: data[field] for field in JOB_FIELDS}
    actual = {field: matches[0].get(field) for field in JOB_FIELDS}
    if actual != expected:
        raise ValueError("protocol job 실행 인자가 manifest와 다르다")
    return matches[0]


def verify_binding(value: Mapping[str, Any], *, repo: Path, lab: Path) -> None:
    """source commit/blob, 현재 source/runtime 및 protocol job 결합을 검증한다."""
    data = validate_manifest(value)
    repo, lab = Path(repo).resolve(), Path(lab).resolve()
    commit = data["source_commit"]
    _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    ancestor = _git(repo, "merge-base", "--is-ancestor", commit, "HEAD", check=False)
    if ancestor.returncode != 0:
        raise ValueError("source commit이 현재 HEAD의 조상이 아니다")
    commitments = {data["protocol_relative"]: data["protocol_sha256_lf"], **data["source_sha256_lf"]}
    for relative, expected in commitments.items():
        path = repo / relative
        if not path.is_file() or _sha_lf(path) != expected:
            raise ValueError(f"현재 source가 binding과 다르다: {relative}")
        blob = _git(repo, "show", f"{commit}:{relative}").stdout
        digest = hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        if digest != expected:
            raise ValueError(f"source commit blob이 binding과 다르다: {relative}")
    for relative, expected in data["runtime_source_sha256_lf"].items():
        path = lab / relative
        if not path.is_file() or _sha_lf(path) != expected:
            raise ValueError(f"Isaac Lab runtime source가 binding과 다르다: {relative}")
    protocol = json.loads((repo / data["protocol_relative"]).read_text(encoding="utf-8"))
    _protocol_job(protocol, data)


def _new_json(path: Path, value: object) -> None:
    GUARD.durable_new(path, value)


def _failure(out: Path, exc: BaseException, stage: str) -> None:
    path = Path(out) / "failure.json"
    if path.exists():
        return
    try:
        _new_json(path, {"status": "invalid", "stage": stage, "type": type(exc).__name__,
                         "message": str(exc),
                         "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
    except FileExistsError:
        pass


def _checkpoint(runner: object, expected_iteration: int) -> dict[str, Any]:
    log_dir = getattr(runner, "log_dir", None)
    if not isinstance(log_dir, (str, os.PathLike)):
        raise TypeError("runner.log_dir checkpoint 위치가 없다")
    path = Path(log_dir) / f"model_{expected_iteration}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"학습 완료 checkpoint model_{expected_iteration}.pt가 없다")
    return {"path": str(path.resolve()), "log_dir": str(Path(log_dir).resolve()),
            "iteration": expected_iteration, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def execute(manifest_path: Path, lab: Path, out: Path) -> dict[str, Any]:
    """fresh output에서 상류 학습을 실행하고 terminal close 전에 완료 증거를 봉인한다."""
    manifest_path, lab, out = Path(manifest_path).resolve(), Path(lab).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("output은 존재하지 않는 fresh 경로여야 한다")
    out.mkdir(parents=True)
    stage = "manifest"
    try:
        data = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest_sha = manifest_fingerprint(data)
        stage = "binding"
        verify_binding(data, repo=ROOT, lab=lab)
        upstream = lab / "scripts/reinforcement_learning/rsl_rl/train.py"
        if not upstream.is_file():
            raise FileNotFoundError("Isaac Lab 상류 train.py가 없다")

        stage = "runtime_import"
        import gymnasium
        hidden = importlib.import_module("hidden_faults")
        ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
        runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
        original_ppo_init, original_learn = ppo_class.__init__, runner_class.learn
        original_load = getattr(runner_class, "load", None)
        if not callable(original_load):
            raise TypeError("OnPolicyRunner.load가 호출 가능해야 한다")
        original_make, original_argv, original_path = gymnasium.make, sys.argv[:], sys.path[:]
        state: dict[str, Any] = {"ppo": [], "learn_called": False, "learn_returned": False,
                                 "load_called": False, "environment_exception": None,
                                 "completion": None}

        def ppo_init(this, *args, **kwargs):
            original_ppo_init(this, *args, **kwargs)
            if state["ppo"]:
                raise ValueError("fresh 학습에는 PPO 인스턴스가 하나여야 한다")
            before = {"learning_epochs": this.num_learning_epochs,
                      "mini_batches": this.num_mini_batches}
            if before != {"learning_epochs": 5, "mini_batches": 4}:
                raise ValueError("PPO 초기 configured epochs/minibatches가 등록값과 다르다")
            if data["arm"] == "COMPOUND":
                this.num_learning_epochs = 1
            after = {"learning_epochs": this.num_learning_epochs,
                     "mini_batches": this.num_mini_batches}
            expected = {"learning_epochs": 1 if data["arm"] == "COMPOUND" else 5,
                        "mini_batches": 4}
            if after != expected:
                raise ValueError("PPO epoch runtime readback이 arm과 다르다")
            original_update = getattr(this, "update", None)
            if not callable(original_update):
                raise TypeError("PPO.update가 호출 가능해야 한다")
            row = {"instance": this, "before": before, "after": after,
                   "original_update": original_update, "update_count": 0,
                   "had_instance_update": "update" in this.__dict__,
                   "instance_update": this.__dict__.get("update")}

            def counted_update(_this, *update_args, **update_kwargs):
                result = original_update(*update_args, **update_kwargs)
                row["update_count"] += 1
                if row["update_count"] > data["max_iterations"]:
                    raise ValueError("실제 PPO update가 등록 예산을 초과했다")
                return result

            this.update = MethodType(counted_update, this)
            state["ppo"].append(row)

        def load(this, *args, **kwargs):
            state["load_called"] = True
            raise RuntimeError("fresh source 학습에서 Runner.load/resume/checkpoint input은 금지된다")

        def learn(this, *args, **kwargs):
            if state["learn_called"]:
                raise ValueError("runner.learn은 한 번만 호출해야 한다")
            iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
            random_ep = kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None)
            if iterations != data["max_iterations"] or random_ep is not True:
                raise ValueError("learn update 수 또는 fresh episode initialization 인자가 다르다")
            state["learn_called"] = True
            result = original_learn(this, *args, **kwargs)
            if len(state["ppo"]) != 1 or getattr(this, "alg", None) is not state["ppo"][0]["instance"]:
                raise ValueError("runner와 PPO instance 결합이 다르다")
            actual_updates = state["ppo"][0]["update_count"]
            if actual_updates != data["max_iterations"]:
                raise ValueError("실제 PPO update 성공 반환 횟수가 등록 예산과 다르다")
            if state["load_called"]:
                raise RuntimeError("Runner.load 호출이 감지됐다")
            state["learn_returned"] = True
            state["completion"] = {
                "schema": "lineage_source_training_completion_v1", "job_id": data["job_id"],
                "case_id": data["case_id"], "arm": data["arm"], "purpose": data["purpose"],
                "updates": actual_updates, "update_count": actual_updates,
                "learn_called": True, "learn_returned": True, "fresh_initialization": True,
                "resume_requested": False, "checkpoint_input": False,
                "ppo_readback": {"configured": state["ppo"][0]["before"],
                                 "effective": state["ppo"][0]["after"]},
                "patch_readback": {"configured_epochs": state["ppo"][0]["before"]["learning_epochs"],
                                   "effective_epochs": state["ppo"][0]["after"]["learning_epochs"],
                                   "r1_applied": data["arm"] == "COMPOUND"},
                "checkpoint": _checkpoint(this, data["max_iterations"] - 1),
                "telemetry": {"task": data["task"], "seed": data["seed"],
                              "num_envs": data["num_envs"], "max_iterations": data["max_iterations"],
                              "run_name": data["run_name"], "fault": "F1" if data["arm"] == "COMPOUND" else data["arm"]},
            }
            return result

        ppo_class.__init__, runner_class.learn, runner_class.load = ppo_init, learn, load
        hidden.arm("F1" if data["arm"] == "COMPOUND" else data["arm"])
        armed_make = gymnasium.make

        def make(*args, **kwargs):
            try:
                return armed_make(*args, **kwargs)
            except BaseException as exc:
                state["environment_exception"] = f"{type(exc).__name__}: {exc}"
                raise

        gymnasium.make = make
        argv = [str(upstream), "--task", data["task"], "--seed", str(data["seed"]),
                "--num_envs", "4096", "--max_iterations", str(data["max_iterations"]), "--headless",
                "--run_name", data["run_name"]]

        def before_close(_app: object) -> None:
            # 소비 시점에도 source/protocol과 등록 job을 다시 확인한다.
            verify_binding(data, repo=ROOT, lab=lab)
            if state["environment_exception"] is not None:
                raise RuntimeError("환경 생성 예외를 상류가 삼켰다: " + state["environment_exception"])
            if not state["learn_returned"] or not isinstance(state["completion"], dict):
                raise RuntimeError("runner.learn 반환 완료 증거 없이 앱을 닫을 수 없다")
            record = state["completion"]
            records_path = out / GUARD.RECORDS
            with records_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            evidence = {"schema": "lineage_source_training_evidence_v1", "manifest_sha256": manifest_sha,
                        "updates": data["max_iterations"], "update_count": data["max_iterations"],
                        "learning_completed": True, "learn_returned": True,
                        "recorded_before_app_close": True, "arm": data["arm"], "seed": data["seed"],
                        "log_dir": record["checkpoint"]["log_dir"],
                        "checkpoint_sha256": record["checkpoint"]["sha256"],
                        "checkpoint": record["checkpoint"], "patch_readback": record["patch_readback"],
                        "ppo_readback": record["ppo_readback"]}
            _new_json(out / GUARD.EVIDENCE, evidence)
            _new_json(out / GUARD.CLOSE_READY, {
                "schema": GUARD.CLOSE_SCHEMA, "manifest_sha256": manifest_sha,
                "updates": data["max_iterations"],
                "evidence_sha256_lf": _sha_lf(out / GUARD.EVIDENCE),
                "records_sha256_lf": _sha_lf(records_path),
            })

        stage = "upstream"
        try:
            sys.argv = argv
            sys.path.insert(0, str(upstream.parent))
            result = GUARD.run_upstream(upstream, out, before_close)
        finally:
            gymnasium.make = original_make
            for row in state["ppo"]:
                if row["had_instance_update"]:
                    row["instance"].update = row["instance_update"]
                else:
                    row["instance"].__dict__.pop("update", None)
            ppo_class.__init__, runner_class.learn, runner_class.load = (
                original_ppo_init, original_learn, original_load,
            )
            sys.argv, sys.path[:] = original_argv, original_path
        GUARD.validate_completion(out, manifest_sha)
        return {"schema": SCHEMA + "_result", "status": "complete", "job_id": data["job_id"],
                "manifest_sha256": manifest_sha, **result}
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
