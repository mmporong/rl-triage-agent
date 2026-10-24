"""Brev Linux 인스턴스 내부의 회복 고리 GPU 실행 backend.

호출자는 모델 대화와 상태 전이를 소유한다. 이 backend는 고정된 학습·검사·평가
child를 새 POSIX 세션에서 실행하고, 종료 코드·시간 제한·process group 정리·GPU
잔여 PID·source binding이 모두 유효할 때만 산출물을 반환한다.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import optimizer_observability as OPTIMIZER

from rl_triage import grounded_recovery as GROUNDED

RESIDUAL = importlib.import_module("residual_training_v1")

WORKER = ROOT / "bench/brev_recovery_worker_v1.py"
LEGACY_PROBES = ("P_reward", "P_value", "P_torque", "P_physics", "P_episode", "P_noise")
OPTIMIZER_PROBE = "P_optimizer_updates"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
TIMEOUTS = {"train": 1800, "probe": 900, "evaluate": 900}
ARMS = {
    "NONE": ([], []),
    "F1": (["F1"], []),
    "CORRECTED": (["F1"], ["restore_body_frame_reward"]),
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class BackendError(RuntimeError):
    def __init__(self, kind: str, receipt: Mapping[str, Any] | None = None):
        self.kind = kind
        self.receipt = dict(receipt or {})
        super().__init__(kind)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Brev Python 3.10


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


def _relative_hashes(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}가 비었다")
    out = {}
    for relative, digest in value.items():
        path = Path(relative) if isinstance(relative, str) else Path("/")
        if (not isinstance(relative, str) or not relative or path.is_absolute() or ".." in path.parts
                or not isinstance(digest, str) or HEX64.fullmatch(digest) is None):
            raise ValueError(f"{name} 형식이 잘못됐다")
        out[relative] = digest
    return out


def _inside(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(), root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("경로가 실행 stage를 벗어났다")
    return resolved


def gpu_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="strict",
    )
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()})


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True,
                          text=True, encoding="utf-8", errors="strict")


def _package_root() -> Path:
    distribution = metadata.distribution("rsl-rl-lib")
    return Path(str(distribution.locate_file("")))


def verify_protocol_binding(path: Path, expected_sha: str, anchor: str) -> tuple[str, str]:
    """현재 protocol과 사전등록 commit blob이 같은 LF SHA인지 확인한다."""
    resolved = _inside(path, ROOT)
    relative = resolved.relative_to(ROOT).as_posix()
    if not resolved.is_file() or sha256(resolved, lf=True) != expected_sha:
        raise ValueError("protocol 파일 LF SHA가 사전등록값과 다르다")
    blob = _git(ROOT, "show", f"{anchor}:{relative}").stdout.encode("utf-8").replace(b"\r\n", b"\n")
    if hashlib.sha256(blob).hexdigest() != expected_sha:
        raise ValueError("protocol anchor blob LF SHA가 사전등록값과 다르다")
    return relative, expected_sha


def validate_probe_result(artifact: Mapping[str, Any], probe: str, run_name: str,
                          result: Mapping[str, Any]) -> dict[str, Any]:
    """raw probe 보고서와 반환 measurement의 provenance를 순수 검증한다."""
    if probe not in (*LEGACY_PROBES, OPTIMIZER_PROBE) or not isinstance(run_name, str) or not run_name:
        raise ValueError("probe 또는 run_name이 등록 계약과 다르다")
    checkpoint = artifact.get("checkpoint")
    report, measurement = result.get("report"), result.get("measurement")
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("sha256"), str)
            or not isinstance(report, dict) or not isinstance(measurement, dict)):
        raise TypeError("probe artifact/result 결합값이 없다")
    if probe in LEGACY_PROBES:
        if (report.get("name") != run_name or report.get("task") != TASK
                or report.get("seed") != artifact.get("seed")):
            raise ValueError("legacy probe 실행 identity가 artifact와 다르다")
        report_checkpoint = report.get("checkpoint")
        if (not isinstance(report_checkpoint, dict)
                or report_checkpoint.get("sha256") != checkpoint["sha256"]):
            raise ValueError("legacy probe checkpoint SHA가 artifact와 다르다")
        probes = report.get("probes")
        if not isinstance(probes, dict) or set(probes) != {probe}:
            raise ValueError("legacy probe 보고서가 요청한 한 probe만 포함하지 않는다")
        row = probes[probe]
        if (not isinstance(row, dict) or row.get("exit") != 0
                or not isinstance(row.get("measurement"), dict)
                or row["measurement"] != measurement):
            raise ValueError("legacy raw measurement와 반환 measurement가 다르다")
        return dict(row["measurement"])

    if report != measurement:
        raise ValueError("optimizer raw report와 반환 measurement가 다르다")
    descriptor = result.get("inspection_manifest")
    training_manifest = artifact.get("manifest")
    if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "body"}
            or not isinstance(descriptor.get("path"), str)
            or Path(descriptor["path"]).name != "inspection.manifest.json"
            or not isinstance(descriptor.get("sha256"), str)
            or HEX64.fullmatch(descriptor["sha256"]) is None
            or not isinstance(descriptor.get("body"), dict)
            or not isinstance(training_manifest, dict)):
        raise ValueError("optimizer residual inspection manifest 결합값이 없다")
    encoded = (json.dumps(descriptor["body"], ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != descriptor["sha256"]:
        raise ValueError("optimizer residual inspection manifest 본문 SHA가 다르다")
    inspection = RESIDUAL.validate_manifest(descriptor["body"])
    source_keys = ("source_commit_sha", "protocol_path", "protocol_sha256_lf",
                   "source_sha256_lf", "runtime_source_sha256_lf")
    expected_arm = "FIRST" if artifact.get("arm") == "CORRECTED" else artifact.get("arm")
    if (inspection["mode"] != "inspection"
            or inspection["source_checkpoint_sha256"] != checkpoint["sha256"]
            or inspection["seed"] != artifact.get("seed") or inspection["arm"] != expected_arm
            or inspection["run_name"] != run_name or inspection["task"] != TASK
            or inspection["max_iterations"] != 3
            or any(inspection[key] != training_manifest.get(key) for key in source_keys)):
        raise ValueError("optimizer inspection manifest가 학습 artifact와 다르다")
    if report.get("schema") != "optimizer_observability_inspection_v1":
        raise ValueError("optimizer inspection schema가 다르다")
    manifest = OPTIMIZER.validate_manifest(report.get("manifest", {}))
    if manifest != RESIDUAL.observer_manifest(inspection):
        raise ValueError("optimizer 첫 3 update observer와 inspection manifest가 다르다")
    observer_source_keys = ("source_commit_sha", "protocol_path", "protocol_sha256_lf", "source_sha256_lf")
    if (manifest.get("runtime_updates_to_observe") != 3 or manifest.get("max_iterations") != 3
            or manifest.get("task") != TASK or manifest.get("seed") != artifact.get("seed")
            or manifest.get("arm") != expected_arm
            or manifest.get("inspection_primitives") != list(OPTIMIZER.PRIMITIVES)
            or any(manifest[key] != inspection[key] for key in observer_source_keys)):
        raise ValueError("optimizer 첫 3 update 고정 manifest가 artifact와 다르다")
    if report.get("records") != 3:
        raise ValueError("optimizer inspection record 수가 3이 아니다")
    primitives = report.get("primitives")
    if not isinstance(primitives, dict) or set(primitives) != set(OPTIMIZER.PRIMITIVES):
        raise ValueError("optimizer primitive 집합이 고정 계약과 다르다")
    decisive = [primitives[name].get("status") for name in ("optimizer_step_count", "minibatch_count")
                if isinstance(primitives.get(name), dict)]
    expected_status = ("abnormal" if "abnormal" in decisive else
                       "normal" if len(decisive) == 2 and all(value == "normal" for value in decisive)
                       else "unknown")
    if report.get("status") != expected_status:
        raise ValueError("optimizer inspection status가 primitive와 다르다")
    return dict(measurement)


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {"source_sha256_lf", "runtime_source_sha256_lf",
                "python_package_source_sha256_lf", "environment", "protocol_path",
                "protocol_sha256_lf"}
    if not required <= set(protocol) or not isinstance(protocol.get("environment"), dict):
        raise ValueError("Brev recovery protocol 공통 필드가 없다")
    out = dict(protocol)
    protocol_path = Path(out["protocol_path"]) if isinstance(out["protocol_path"], str) else Path("/")
    if not protocol_path.is_absolute() or protocol_path.resolve() != protocol_path:
        raise ValueError("protocol_path는 정규화된 절대경로여야 한다")
    if not isinstance(out["protocol_sha256_lf"], str) or HEX64.fullmatch(out["protocol_sha256_lf"]) is None:
        raise ValueError("protocol_sha256_lf 형식이 잘못됐다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf"):
        out[key] = _relative_hashes(out[key], key)
    expected_package = {
        "rsl_rl/algorithms/ppo.py", "rsl_rl/storage/rollout_storage.py",
        "rsl_rl/runners/on_policy_runner.py",
    }
    if set(out["python_package_source_sha256_lf"]) != expected_package:
        raise ValueError("RSL-RL distribution source binding이 고정 3파일과 다르다")
    expected_runtime = {
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "source/isaaclab/isaaclab/envs/mdp/rewards.py",
    }
    if set(out["runtime_source_sha256_lf"]) != expected_runtime:
        raise ValueError("Isaac Lab runtime source binding이 고정 2파일과 다르다")
    return out


def environment_snapshot(expected: Mapping[str, Any], anchor: str) -> dict[str, Any]:
    package_keys = ("isaacsim", "isaaclab", "isaaclab_rl", "torch", "rsl-rl-lib", "gymnasium")
    actual: dict[str, Any] = {"python": platform.python_version(), "isaaclab_sha": anchor}
    for name in package_keys:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual[name] = None
    try:
        visible = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="strict",
        ).stdout.splitlines()
        actual["cuda_device_count"] = len([line for line in visible if line.strip().isdigit()])
        actual["cuda_available"] = actual["cuda_device_count"] > 0
    except (OSError, subprocess.SubprocessError):
        actual["cuda_available"] = False
        actual["cuda_device_count"] = 0
    for key in ("python", "isaacsim", "isaaclab_sha", "isaaclab", "isaaclab_rl", "torch",
                "rsl-rl-lib", "gymnasium"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"runtime environment가 protocol과 다르다: {key}")
    if actual["cuda_available"] is not True or actual["cuda_device_count"] != 1:
        raise ValueError("Brev worker에는 CUDA GPU 한 개가 보여야 한다")
    return actual


def cleanup_group(pid: int, process: subprocess.Popen[Any]) -> bool:
    if os.name != "posix":
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        return process.poll() is not None

    def alive() -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    if process.poll() is None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    if alive():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if alive():
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while alive() and time.monotonic() < deadline:
                time.sleep(0.05)
    return not alive()


class Backend:
    def __init__(self, lab: Path, python: Path, protocol: dict, anchor: str,
                 *, read_gpu: Callable[[], list[int]] = gpu_pids,
                 popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen):
        self.lab = Path(lab).resolve()
        self.python = Path(python).resolve()
        self.protocol = validate_protocol(protocol)
        self.anchor = anchor
        self.read_gpu = read_gpu
        self.popen = popen
        if not re.fullmatch(r"[0-9a-f]{40}", anchor):
            raise ValueError("실험 source anchor SHA 형식이 잘못됐다")
        if not re.fullmatch(r"[0-9a-f]{40}", str(self.protocol["environment"].get("isaaclab_sha", ""))):
            raise ValueError("protocol environment의 Isaac Lab SHA가 잘못됐다")
        if not self.python.is_file() or not self.lab.is_dir():
            raise FileNotFoundError("Brev runtime 경로가 없다")

    def _source_snapshot(self) -> dict[str, Any]:
        head = _git(self.lab, "rev-parse", "HEAD").stdout.strip()
        dirty = _git(self.lab, "status", "--porcelain", "--untracked-files=no").stdout
        lab_anchor = self.protocol["environment"]["isaaclab_sha"]
        if head != lab_anchor or dirty:
            raise ValueError("Isaac Lab HEAD 또는 tracked cleanliness가 고정 계약과 다르다")
        _git(ROOT, "cat-file", "-e", f"{self.anchor}^{{commit}}")
        if _git(ROOT, "merge-base", "--is-ancestor", self.anchor, "HEAD", check=False).returncode:
            raise ValueError("실험 source anchor가 현재 HEAD의 조상이 아니다")
        protocol_relative, protocol_sha = verify_protocol_binding(
            Path(self.protocol["protocol_path"]), self.protocol["protocol_sha256_lf"], self.anchor,
        )
        for relative, digest in self.protocol["runtime_source_sha256_lf"].items():
            path = self.lab / relative
            if not path.is_file() or sha256(path, lf=True) != digest:
                raise ValueError("Isaac Lab runtime source hash가 다르다")
        for relative, digest in self.protocol["source_sha256_lf"].items():
            path = ROOT / relative
            if not path.is_file() or sha256(path, lf=True) != digest:
                raise ValueError("repository source hash가 다르다")
            committed = _git(ROOT, "show", f"{self.anchor}:{relative}").stdout.encode("utf-8").replace(b"\r\n", b"\n")
            if hashlib.sha256(committed).hexdigest() != digest:
                raise ValueError("실험 source anchor blob hash가 다르다")
        package_root = _package_root()
        for relative, digest in self.protocol["python_package_source_sha256_lf"].items():
            path = package_root / relative
            if not path.is_file() or sha256(path, lf=True) != digest:
                raise ValueError("RSL-RL distribution source hash가 다르다")
        return {"lab_head": head, "lab_tracked_clean": True,
                "source_anchor": self.anchor,
                "protocol": {"path": protocol_relative, "sha256_lf": protocol_sha},
                "environment": environment_snapshot(self.protocol["environment"], lab_anchor),
                "source_sha256_lf": self.protocol["source_sha256_lf"],
                "runtime_source_sha256_lf": self.protocol["runtime_source_sha256_lf"],
                "python_package_source_sha256_lf": self.protocol["python_package_source_sha256_lf"]}

    def _base_job(self, stage: Path, mode: str) -> dict[str, Any]:
        protocol_path = Path(self.protocol["protocol_path"])
        if not protocol_path.is_file():
            raise FileNotFoundError("protocol 파일이 없다")
        expected_protocol = self.protocol["protocol_sha256_lf"]
        if sha256(protocol_path, lf=True) != expected_protocol:
            raise ValueError("protocol source hash가 다르다")
        return {
            "schema": "brev_recovery_worker_job_v1", "mode": mode,
            "stage": str(stage), "lab": str(self.lab), "repo": str(ROOT),
            "protocol_path": str(protocol_path.resolve()), "protocol_sha256_lf": expected_protocol,
            "source_commit_sha": self.anchor,
            "source_sha256_lf": self.protocol["source_sha256_lf"],
            "runtime_source_sha256_lf": self.protocol["runtime_source_sha256_lf"],
            "python_package_source_sha256_lf": self.protocol["python_package_source_sha256_lf"],
        }

    def _execute(self, stage: Path, mode: str, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        stage = Path(stage).resolve()
        if stage.exists():
            if not stage.is_dir() or any(stage.iterdir()):
                raise FileExistsError("stage는 새 빈 폴더여야 한다")
        else:
            stage.mkdir(parents=True)
        job_path, worker_output = stage / "job.json", stage / "worker.json"
        parent_output = stage / "execution.json"
        write_new(job_path, job)
        preflight_started = time.perf_counter()
        receipt: dict[str, Any] = {
            "schema": "brev_recovery_posix_execution_v1", "mode": mode,
            "completed": False, "started_at": now(), "exit_code": None,
            "timed_out": False, "cleanup_verified": True,
            "launch": {"session_id": None}, "gpu_before": None, "gpu_after": None,
            "wall_s": 0.0, "source_before": None, "source_after": None,
        }
        try:
            receipt["source_before"] = self._source_snapshot()
            receipt["gpu_before"] = self.read_gpu()
        except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
            receipt.update(finished_at=now(), wall_s=time.perf_counter() - preflight_started,
                           error_type=type(error).__name__)
            try:
                receipt["gpu_after"] = self.read_gpu()
            except (OSError, subprocess.SubprocessError, TypeError, ValueError):
                receipt["gpu_after"] = None
            write_new(parent_output, receipt)
            raise BackendError(type(error).__name__, receipt) from error
        if receipt["gpu_before"]:
            receipt.update(finished_at=now(), wall_s=time.perf_counter() - preflight_started,
                           gpu_after=receipt["gpu_before"], error_type="ForeignGPUBusy")
            write_new(parent_output, receipt)
            raise BackendError("ForeignGPUBusy", receipt)
        started, process = time.perf_counter(), None
        stdout_path, stderr_path = stage / "stdout.log", stage / "stderr.log"
        try:
            with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                process = self.popen([str(self.python), "-B", str(WORKER), "--job", str(job_path),
                                      "--output", str(worker_output)], cwd=str(self.lab),
                                     env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1",
                                          "CUDA_VISIBLE_DEVICES": "0"}, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
                receipt["launch"]["session_id"] = process.pid
                try:
                    receipt["exit_code"] = process.wait(timeout=TIMEOUTS[mode])
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                finally:
                    receipt["cleanup_verified"] = cleanup_group(process.pid, process)
                    receipt["exit_code"] = -signal.SIGTERM if receipt["timed_out"] else process.returncode
            receipt["gpu_after"] = self.read_gpu()
            receipt["source_after"] = self._source_snapshot()
            if worker_output.is_file():
                worker = json.loads(worker_output.read_text(encoding="utf-8"))
            else:
                worker = {}
            receipt["completed"] = bool(
                receipt["exit_code"] == 0 and not receipt["timed_out"]
                and receipt["cleanup_verified"] and receipt["gpu_after"] == []
                and worker.get("completed") is True
                and receipt["source_before"] == receipt["source_after"]
            )
            if not receipt["completed"]:
                receipt["error_type"] = "InvalidExecutionEvidence"
        except BaseException as error:  # noqa: BLE001 - child/SystemExit도 cleanup receipt로 닫는다.
            receipt["error_type"] = type(error).__name__
            if process is not None:
                try:
                    receipt["cleanup_verified"] = cleanup_group(process.pid, process)
                    receipt["exit_code"] = process.returncode
                except BaseException:  # noqa: BLE001 - cleanup은 원래 오류를 가리지 않는다.
                    receipt["cleanup_verified"] = False
            try:
                receipt["gpu_after"] = self.read_gpu()
            except BaseException:  # noqa: BLE001 - GPU 조회 실패 자체를 receipt에 보존한다.
                receipt["gpu_after"] = None
            worker = {}
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            receipt["finished_at"] = now()
            write_new(parent_output, receipt)
        if not receipt["completed"]:
            raise BackendError(str(receipt.get("error_type", "InvalidExecutionEvidence")), receipt)
        return worker, receipt

    def train(self, stage: Path, arm: str, seed: int, run_name: str,
              corrections: list[str] | None = None, *, iterations: int = 300,
              source_checkpoint_sha256: str | None = None) -> dict[str, Any]:
        corrections = list(corrections or [])
        if arm not in ARMS or corrections != ARMS[arm][1]:
            raise ValueError("arm과 등록 교정 목록이 다르다")
        if iterations not in (20, 300):
            raise ValueError("학습 반복은 20 dry-run 또는 300 본 실행만 허용한다")
        if ((arm == "CORRECTED" and
             (not isinstance(source_checkpoint_sha256, str) or HEX64.fullmatch(source_checkpoint_sha256) is None))
                or (arm != "CORRECTED" and source_checkpoint_sha256 is not None)):
            raise ValueError("교정 restart의 source checkpoint SHA provenance가 다르다")
        stage = Path(stage).resolve()
        base = self._base_job(stage, "train")
        job = {**base, "arm": arm, "seed": seed, "run_name": run_name, "iterations": iterations,
               "source_checkpoint_sha256": source_checkpoint_sha256,
               "manifest_path": str(stage / "training.manifest.json"),
               "records_path": str(stage / "optimizer.records.jsonl"),
               "readback_path": str(stage / "training.readback.json"),
               "telemetry_path": str(stage / "telemetry.json")}
        worker, execution = self._execute(stage, "train", job)
        manifest_path = Path(worker["manifest_path"])
        if sha256(manifest_path) != worker["manifest_sha256"]:
            raise BackendError("TrainingManifestHashMismatch", execution)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if iterations == 300:
            manifest = RESIDUAL.validate_manifest(manifest)
        faults, applied = ARMS[arm]
        artifact = {
            "checkpoint": worker["checkpoint"], "telemetry": worker["telemetry"],
            "seed": seed, "arm": arm, "original_faults": faults,
            "applied_corrections": applied, "readback": worker["readback"],
            "execution": execution, "manifest_path": worker["manifest_path"],
            "manifest_sha256": worker["manifest_sha256"],
            "manifest": manifest,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        }
        return artifact

    def probe(self, stage: Path, artifact: dict[str, Any], probe: str,
              run_name: str) -> dict[str, Any]:
        if probe not in (*LEGACY_PROBES, OPTIMIZER_PROBE):
            raise ValueError("등록되지 않은 probe")
        stage = Path(stage).resolve()
        checkpoint = artifact.get("checkpoint", {})
        if not isinstance(checkpoint, dict) or not Path(checkpoint.get("path", "")).is_file():
            raise ValueError("artifact checkpoint가 없다")
        base = self._base_job(stage, "probe")
        job = {**base, "arm": artifact["arm"], "seed": artifact["seed"], "run_name": run_name,
               "probe": probe, "checkpoint_path": checkpoint["path"],
               "checkpoint_sha256": checkpoint["sha256"],
               "training_manifest_path": artifact["manifest_path"],
               "report_path": str(stage / "report.json"),
               "inspection_manifest_path": str(stage / "inspection.manifest.json"),
               "records_path": str(stage / "optimizer.records.jsonl"),
               "readback_path": str(stage / "inspection.readback.json")}
        worker, execution = self._execute(stage, "probe", job)
        report_path = Path(worker["report"]["path"])
        if sha256(report_path) != worker["report"]["sha256"]:
            raise BackendError("ProbeReportHashMismatch", execution)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = {"measurement": worker["measurement"], "execution": execution,
                  "report": report}
        if probe == OPTIMIZER_PROBE:
            descriptor = worker.get("inspection_manifest")
            expected_path = stage / "inspection.manifest.json"
            if (not isinstance(descriptor, dict) or descriptor.get("path") != str(expected_path)
                    or not expected_path.is_file() or sha256(expected_path) != descriptor.get("sha256")):
                raise BackendError("InspectionManifestBindingMismatch", execution)
            result["inspection_manifest"] = {
                **descriptor, "body": json.loads(expected_path.read_text(encoding="utf-8")),
            }
        result["measurement"] = validate_probe_result(artifact, probe, run_name, result)
        return result

    def evaluate(self, stage: Path, artifacts: dict[str, dict[str, Any]],
                 run_name: str) -> dict[str, Any]:
        if not run_name or not artifacts:
            raise ValueError("evaluate run_name/artifacts가 비었다")
        stage = Path(stage).resolve()
        base = self._base_job(stage, "evaluate")
        fixed = Path(self.protocol.get("fixed_eval_protocol", ROOT / "bench/protocols/fixed_eval_v1.json"))
        if not fixed.is_absolute():
            fixed = ROOT / fixed
        job = {**base, "artifacts": artifacts, "fixed_eval_protocol": str(fixed.resolve()),
               "reports_dir": str(stage / "reports"), "jobs_path": str(stage / "fixed_eval.jobs.json")}
        worker, execution = self._execute(stage, "evaluate", job)
        reports = {}
        for name, item in worker["reports"].items():
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise BackendError("FixedEvaluationReportHashMismatch", execution)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("name") != name:
                raise BackendError("FixedEvaluationReportBindingMismatch", execution)
            artifact = artifacts.get(name)
            if not isinstance(artifact, dict) or not isinstance(artifact.get("checkpoint"), dict):
                raise BackendError("FixedEvaluationArtifactMissing", execution)
            checkpoint = artifact["checkpoint"]
            reports[name] = GROUNDED.validate_fixed_report(report, {
                "name": name, "checkpoint_file": Path(checkpoint["path"]).name,
                "checkpoint_sha256": checkpoint["sha256"], "action_scale": 0.25,
                "protocol_sha256_lf": sha256(fixed, lf=True),
            })
        return {"reports": reports, "execution": execution}
