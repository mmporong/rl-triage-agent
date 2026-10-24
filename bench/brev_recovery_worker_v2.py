"""Linux Brev 호스트에서 동결 학습·검사·평가 진입점을 실행한다.

이 파일은 child 실행만 담당한다. 시간 제한, 프로세스 그룹 종료, GPU 점유
검사는 ``evals/brev_recovery_backend_v1.py``가 맡는다. 성공 receipt는 학습의
``learn`` 반환 또는 검사/평가 산출물 검증 뒤에만 기록한다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS = 4096
TRAIN_ITERATIONS = (20, 300)
ARMS = {
    "NONE": ([], []),
    "F1": (["F1"], []),
    "CORRECTED": (["F1"], ["restore_body_frame_reward"]),
}
LEGACY_PROBES = ("P_reward", "P_value", "P_torque", "P_physics", "P_episode", "P_noise")
OPTIMIZER_PROBE = "P_optimizer_updates"
COMPUTE = ROOT / "bench/brev_compute_probe_v1.py"
SLUG = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,78}[a-z0-9])?\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def completed(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("completed") is True


def _inside(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(), root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("경로가 허용된 실행 폴더를 벗어났다")
    return resolved


def _inside_any(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    if not any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots):
        raise ValueError("입력 경로가 허용된 workspace를 벗어났다")
    return resolved


def _relative_map(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}가 비었다")
    out: dict[str, str] = {}
    for relative, digest in value.items():
        path = Path(relative) if isinstance(relative, str) else Path("/")
        if (not isinstance(relative, str) or not relative or path.is_absolute() or ".." in path.parts
                or not isinstance(digest, str) or HEX64.fullmatch(digest) is None):
            raise ValueError(f"{name} 형식이 잘못됐다")
        out[relative] = digest
    return out


def validate_job(value: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(value)
    required = {"schema", "mode", "stage", "lab", "repo", "protocol_path", "protocol_sha256_lf",
                "source_commit_sha", "source_sha256_lf", "runtime_source_sha256_lf",
                "python_package_source_sha256_lf"}
    if not required <= set(job) or job.get("schema") != "brev_recovery_worker_job_v1":
        raise ValueError("worker job 공통 계약이 잘못됐다")
    if job.get("mode") not in {"train", "probe", "evaluate"}:
        raise ValueError("worker mode가 잘못됐다")
    for key in ("stage", "lab", "repo", "protocol_path"):
        if not isinstance(job.get(key), str) or not Path(job[key]).is_absolute():
            raise ValueError(f"{key}는 절대경로여야 한다")
    stage = Path(job["stage"]).resolve()
    for key in ("protocol_path",):
        _inside(Path(job[key]), Path(job["repo"]))
    for name in ("source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf"):
        job[name] = _relative_map(job[name], name)
    if not isinstance(job.get("protocol_sha256_lf"), str) or HEX64.fullmatch(job["protocol_sha256_lf"]) is None:
        raise ValueError("protocol SHA 형식이 잘못됐다")
    if not isinstance(job.get("source_commit_sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", job["source_commit_sha"]):
        raise ValueError("source commit 형식이 잘못됐다")
    if job["mode"] == "train":
        expected = required | {"arm", "seed", "run_name", "iterations", "manifest_path", "records_path",
                               "readback_path", "telemetry_path", "source_checkpoint_sha256"}
        if set(job) != expected:
            raise ValueError("train job 필드가 고정 계약과 다르다")
        if job["arm"] not in ARMS or type(job["seed"]) is not int or job["seed"] < 0:
            raise ValueError("train arm/seed가 잘못됐다")
        if type(job["iterations"]) is not int or job["iterations"] not in TRAIN_ITERATIONS:
            raise ValueError("train iterations는 20 또는 300이어야 한다")
        if not isinstance(job["run_name"], str) or SLUG.fullmatch(job["run_name"]) is None:
            raise ValueError("run_name 형식이 잘못됐다")
        source_checkpoint = job["source_checkpoint_sha256"]
        if ((job["arm"] == "CORRECTED" and
             (not isinstance(source_checkpoint, str) or HEX64.fullmatch(source_checkpoint) is None))
                or (job["arm"] != "CORRECTED" and source_checkpoint is not None)):
            raise ValueError("교정 restart의 source checkpoint SHA provenance가 다르다")
        for key in ("manifest_path", "records_path", "readback_path", "telemetry_path"):
            _inside(Path(job[key]), stage)
    elif job["mode"] == "probe":
        expected = required | {"arm", "seed", "run_name", "probe", "checkpoint_path",
                               "checkpoint_sha256", "training_manifest_path", "report_path",
                               "inspection_manifest_path", "records_path", "readback_path"}
        if set(job) != expected or job["arm"] not in ARMS:
            raise ValueError("probe job 필드가 고정 계약과 다르다")
        if job["probe"] not in (*LEGACY_PROBES, OPTIMIZER_PROBE):
            raise ValueError("등록되지 않은 probe")
        for key in ("report_path", "inspection_manifest_path", "records_path", "readback_path"):
            _inside(Path(job[key]), stage)
        roots = (Path(job["repo"]), Path(job["lab"]), stage.parent)
        for key in ("checkpoint_path", "training_manifest_path"):
            if not isinstance(job.get(key), str) or not Path(job[key]).is_absolute():
                raise ValueError(f"{key}는 절대경로여야 한다")
            _inside_any(Path(job[key]), roots)
        if sha256(Path(job["checkpoint_path"])) != job["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA가 다르다")
    else:
        expected = required | {"artifacts", "fixed_eval_protocol", "reports_dir", "jobs_path"}
        if set(job) != expected or not isinstance(job["artifacts"], dict) or not job["artifacts"]:
            raise ValueError("evaluate job 필드가 고정 계약과 다르다")
        _inside(Path(job["reports_dir"]), stage)
        _inside(Path(job["jobs_path"]), stage)
        _inside(Path(job["fixed_eval_protocol"]), Path(job["repo"]))
    return job


def _latest_checkpoint(log_dir: Path) -> Path:
    candidates = []
    for path in log_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("학습 checkpoint가 없다")
    return max(candidates, key=lambda item: item[0])[1]


def _telemetry(run_dir: Path, output: Path) -> dict[str, Any]:
    extract = importlib.import_module("extract_tb")
    extract.main(str(run_dir), str(output))
    body = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(body.get("summary"), dict) or not isinstance(body.get("series"), dict):
        raise TypeError("telemetry 형식이 잘못됐다")
    return body


def _residual_manifest(job: Mapping[str, Any], *, mode: str, manifest_path: Path,
                       records_path: Path, readback_path: Path) -> dict[str, Any]:
    import residual_training_v1 as training

    faults, corrections = ARMS[job["arm"]]
    arm = job["arm"] if job["arm"] != "CORRECTED" else "FIRST"
    return training.validate_manifest({
        "schema": training.SCHEMA,
        "mode": mode,
        "arm": arm,
        "seed": job["seed"],
        "run_name": job["run_name"],
        "task": TASK,
        "num_envs": NUM_ENVS,
        "max_iterations": 300 if mode == "train" else 3,
        "configured_learning_epochs": 5,
        "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
        "source_commit_sha": job["source_commit_sha"],
        "protocol_path": Path(job["protocol_path"]).resolve().relative_to(Path(job["repo"]).resolve()).as_posix(),
        "protocol_sha256_lf": job["protocol_sha256_lf"],
        "source_sha256_lf": job["source_sha256_lf"],
        "runtime_source_sha256_lf": job["runtime_source_sha256_lf"],
        "records_path": str(records_path),
        "readback_path": str(readback_path),
        "source_checkpoint_sha256": (job.get("checkpoint_sha256") if mode == "inspection"
                                      else job.get("source_checkpoint_sha256")),
        "original_faults": faults,
        "applied_corrections": corrections,
    })


def run_train(job: Mapping[str, Any], output: Path,
              execute: Callable[[Path], None] | None = None) -> None:
    if job["iterations"] == 20:
        if job["arm"] != "NONE":
            raise ValueError("20-update dry-run은 NONE arm만 허용한다")
        dry_receipt = Path(job["stage"]) / "dryrun.train.json"
        _run_checked([
            sys.executable, str(COMPUTE), "--mode", "train", "--output", str(dry_receipt),
            "--isaaclab", job["lab"], "--seed", str(job["seed"]), "--num-envs", str(NUM_ENVS),
            "--iterations", "20", "--run-name", job["run_name"], "--timeout-s", "1800",
        ])
        dry = json.loads(dry_receipt.read_text(encoding="utf-8"))
        if dry.get("completed") is not True or dry.get("observed_iterations") != 20:
            raise ValueError("20-update dry-run receipt가 유효하지 않다")
        checkpoint = Path(dry["checkpoint_path"])
        run_dir = checkpoint.parent
        telemetry_path = Path(job["telemetry_path"])
        telemetry = _telemetry(run_dir, telemetry_path)
        manifest_path = Path(job["manifest_path"])
        write_new(manifest_path, {"schema": "brev_recovery_dryrun_manifest_v1", "arm": "NONE",
                                  "seed": job["seed"], "iterations": 20})
        readback = {"schema": "brev_recovery_dryrun_readback_v1", "remaining_faults": []}
        write_new(Path(job["readback_path"]), readback)
        with Path(job["records_path"]).open("x", encoding="utf-8", newline="\n"):
            pass
        write_new(output, {
            "schema": "brev_recovery_train_worker_v1", "completed": True,
            "arm": "NONE", "seed": job["seed"], "iterations": 20,
            "learn_wall_s": dry["learn_wall_s"], "wall_s": dry["wall_s"],
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "telemetry": {"path": str(telemetry_path), "sha256": sha256(telemetry_path),
                          "summary": telemetry["summary"]},
            "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "records": {"path": job["records_path"], "sha256": sha256(Path(job["records_path"]))},
            "readback": readback, "readback_sha256": sha256(Path(job["readback_path"])),
        })
        return

    import residual_training_v1 as training
    manifest_path = Path(job["manifest_path"])
    records_path, readback_path = Path(job["records_path"]), Path(job["readback_path"])
    manifest = _residual_manifest(job, mode="train", manifest_path=manifest_path,
                                  records_path=records_path, readback_path=readback_path)
    write_new(manifest_path, manifest)
    runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
    original_learn = runner_class.learn
    started = time.perf_counter()
    state: dict[str, Any] = {"calls": 0}

    def observed_learn(self: Any, *args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        requested = kwargs.get("num_learning_iterations", args[0] if args else None)
        if state["calls"] != 1 or requested != 300:
            raise ValueError("learn 호출 또는 300-update 예산이 다르다")
        initial = getattr(self, "current_learning_iteration", None)
        learn_started = time.perf_counter()
        result = original_learn(self, *args, **kwargs)
        learn_wall = time.perf_counter() - learn_started
        final = getattr(self, "current_learning_iteration", None)
        observed = final - initial + 1 if isinstance(initial, int) and isinstance(final, int) else None
        if observed != 300:
            raise ValueError("실제 learn 반환 update 수가 300이 아니다")
        run_dir = Path(self.log_dir)
        checkpoint = _latest_checkpoint(run_dir)
        telemetry_path = Path(job["telemetry_path"])
        telemetry = _telemetry(run_dir, telemetry_path)
        readback = json.loads(readback_path.read_text(encoding="utf-8"))
        expected_remaining = ["F1"] if job["arm"] == "F1" else []
        if readback.get("remaining_faults") != expected_remaining:
            raise ValueError("결함/교정 readback이 arm과 다르다")
        write_new(output, {
            "schema": "brev_recovery_train_worker_v1", "completed": True,
            "arm": job["arm"], "seed": job["seed"], "iterations": 300,
            "learn_wall_s": learn_wall, "wall_s": time.perf_counter() - started,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "telemetry": {"path": str(telemetry_path), "sha256": sha256(telemetry_path),
                          "summary": telemetry["summary"]},
            "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "records": {"path": str(records_path), "sha256": sha256(records_path)},
            "readback": readback,
            "readback_sha256": sha256(readback_path),
        })
        return result

    runner_class.learn = observed_learn
    old_cwd = Path.cwd()
    try:
        os.chdir(Path(job["lab"]))
        (execute or training.execute)(manifest_path)
        if not output.is_file():
            raise RuntimeError("learn 반환 전 완료 receipt가 기록되지 않았다")
    finally:
        runner_class.learn = original_learn
        os.chdir(old_cwd)


def _run_checked(argv: list[str]) -> None:
    result = subprocess.run(argv, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv)


def run_probe(job: Mapping[str, Any], output: Path) -> None:
    started = time.perf_counter()
    report_path = Path(job["report_path"])
    inspection_descriptor = None
    if job["probe"] in LEGACY_PROBES:
        remaining = "F1" if job["arm"] == "F1" else "NONE"
        jobs_path = report_path.with_name(report_path.stem + ".jobs.json")
        write_new(jobs_path, {"fault": remaining, "seed": job["seed"], "task": TASK, "jobs": [{
            "name": job["run_name"], "checkpoint": job["checkpoint_path"],
            "probes": [job["probe"]], "output": str(report_path),
        }]})
        _run_checked([sys.executable, str(ROOT / "bench/residual_probe_v1.py"), "legacy",
                      "--manifest", job["training_manifest_path"], "--", "--jobs", str(jobs_path),
                      "--headless"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = report.get("probes", {}).get(job["probe"], {})
        if row.get("exit") != 0 or not isinstance(row.get("measurement"), dict):
            raise ValueError("동결 probe가 유효한 measurement를 남기지 않았다")
        measurement = row["measurement"]
    else:
        inspection = _residual_manifest(job, mode="inspection",
                                        manifest_path=Path(job["inspection_manifest_path"]),
                                        records_path=Path(job["records_path"]),
                                        readback_path=Path(job["readback_path"]))
        write_new(Path(job["inspection_manifest_path"]), inspection)
        inspection_descriptor = {"path": job["inspection_manifest_path"],
                                 "sha256": sha256(Path(job["inspection_manifest_path"]))}
        _run_checked([sys.executable, str(ROOT / "bench/residual_probe_v1.py"), "optimizer",
                      "--manifest", job["inspection_manifest_path"]])
        rows = [json.loads(line) for line in Path(job["records_path"]).read_text(encoding="utf-8").splitlines()]
        import residual_training_v1 as training
        measurement = training.OBSERVE.assemble_inspection(rows, training.observer_manifest(inspection))
        write_new(report_path, measurement)
    write_new(output, {"schema": "brev_recovery_probe_worker_v1", "completed": True,
                       "probe": job["probe"], "measurement": measurement,
                       "report": {"path": str(report_path), "sha256": sha256(report_path)},
                       "inspection_manifest": inspection_descriptor,
                       "wall_s": time.perf_counter() - started})


def run_evaluate(job: Mapping[str, Any], output: Path) -> None:
    from rl_triage import grounded_recovery as grounded

    reports_dir = Path(job["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for name, artifact in sorted(job["artifacts"].items()):
        checkpoint = artifact.get("checkpoint", {})
        path = Path(checkpoint.get("path", ""))
        _inside_any(path, (Path(job["repo"]), Path(job["lab"]), Path(job["stage"]).parent))
        if not path.is_file() or sha256(path) != checkpoint.get("sha256"):
            raise ValueError("평가 checkpoint binding이 다르다")
        jobs.append({"name": name, "checkpoint": str(path), "output": str(reports_dir / f"{name}.json")})
    write_new(Path(job["jobs_path"]), {"action_scale": 0.25, "jobs": jobs})
    started = time.perf_counter()
    _run_checked([sys.executable, str(ROOT / "evals/fixed_eval.py"), "--protocol",
                  job["fixed_eval_protocol"], "--jobs", job["jobs_path"], "--headless"])
    reports = {}
    for row in jobs:
        path = Path(row["output"])
        body = json.loads(path.read_text(encoding="utf-8"))
        source = job["artifacts"][row["name"]]["checkpoint"]
        checked = grounded.validate_fixed_report(body, {
            "name": row["name"], "checkpoint_file": Path(source["path"]).name,
            "checkpoint_sha256": source["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": sha256(Path(job["fixed_eval_protocol"]), lf=True),
        })
        reports[row["name"]] = {"path": str(path), "sha256": sha256(path), "metrics": checked["metrics"]}
    write_new(output, {"schema": "brev_recovery_evaluate_worker_v1", "completed": True,
                       "reports": reports, "wall_s": time.perf_counter() - started})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("worker output은 덮어쓰지 않는다")
        job = validate_job(json.loads(args.job.read_text(encoding="utf-8")))
        if job["mode"] == "train":
            run_train(job, args.output)
        elif job["mode"] == "probe":
            run_probe(job, args.output)
        else:
            run_evaluate(job, args.output)
    except BaseException as error:  # noqa: BLE001 - Isaac app의 SystemExit false-zero를 receipt와 함께 판정한다.
        if isinstance(error, SystemExit) and error.code in (0, None) and completed(args.output):
            return 0
        print(json.dumps({"completed": False, "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
