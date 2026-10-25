"""동결 Brev runner의 record/render-smoke 실행을 attestation으로 봉인한다."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import brev_recovery_backend_v4 as B
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import vision_brev_backend_v1 as V1  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]
import vision_render_attestation_v1 as ATTEST  # pyright: ignore[reportMissingImports]
import vision_render_context_v1 as CONTEXT  # pyright: ignore[reportMissingImports]
import vision_selection_sources_v1 as SOURCES  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
VISION_WORKER = ROOT / "bench/vision_play_worker_v1.py"
ACTION_TIMEOUT_S = 900
OWNED_WORKER_GATE = (
    "import os,sys\n"
    "command=sys.argv[1:]\n"
    "if sys.stdin.buffer.readline()!=b'GO\\n': raise SystemExit(97)\n"
    "os.execv(command[0],command)\n"
)


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("render action 시간 한도가 끝났다")
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Brev Python 3.10


def _cleanup_waiting_gate(process: subprocess.Popen[Any]) -> bool:
    """GO 전에는 하위 process가 없는 gate PID만 종료하고 회수한다."""
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return process.poll() is not None
    except (OSError, subprocess.SubprocessError):
        return False


class AttestedVisionBrevBackend(V1.VisionBrevBackend):
    """동결된 두 렌더 mode를 source/context/execute 순서 안에서 실행한다."""

    def record(self, job: Mapping[str, Any]) -> dict[str, Any]:
        return self._capture(job, R.RECORD_MODE)

    def smoke(self, job: Mapping[str, Any]) -> dict[str, Any]:
        return self._capture(job, R.RENDER_SYNC_SMOKE_MODE)

    def _capture(self, job: Mapping[str, Any], mode: str) -> dict[str, Any]:
        if not isinstance(job, Mapping):
            raise TypeError("render job은 mapping이어야 한다")
        job_id = job.get("job_id")
        if not isinstance(job_id, str):
            raise TypeError("render job_id가 없다")
        if V1.SLUG.fullmatch(job_id) is None:
            raise ValueError("render job_id가 잘못됐다")
        if job.get("mode") != mode:
            raise ValueError(f"{mode} 메서드에는 같은 mode의 job만 허용한다")
        stage = V1._owned(self.runtime_root / "record" / job_id, self.runtime_root, "render stage")
        cached = self._cached_capture(stage, job, mode)
        if cached is not None:
            return cached
        with self.action_admission():
            cached = self._cached_capture(stage, job, mode)
            if cached is not None:
                return cached
            return self._record_new(dict(job), stage)

    def _source_protocol(self) -> dict[str, Any]:
        protocol = dict(self.recovery.protocol)
        digest = protocol.get("protocol_sha256_lf")
        if not isinstance(digest, str) or V1.HEX64.fullmatch(digest) is None:
            raise ValueError("source protocol LF SHA 결합이 없다")
        protocol["_verified_sha256_lf"] = digest
        return protocol

    def _action_deadline(self, job: Mapping[str, Any]) -> float:
        protocol = self._source_protocol()
        resources = protocol.get("resources")
        if not isinstance(resources, Mapping):
            raise TypeError("source protocol resource 계약이 없다")
        limits = resources.get("action_limits")
        limit = limits.get("record") if isinstance(limits, Mapping) else None
        quote = resources.get("compute_quote_usd_per_h")
        if (
            not isinstance(limit, Mapping)
            or set(limit) != {"max_wall_s", "max_compute_usd"}
            or limit.get("max_wall_s") != ACTION_TIMEOUT_S
            or isinstance(quote, bool)
            or not isinstance(quote, (int, float))
            or isinstance(limit.get("max_compute_usd"), bool)
            or not isinstance(limit.get("max_compute_usd"), (int, float))
            or abs(float(limit["max_compute_usd"]) - float(quote) * ACTION_TIMEOUT_S / 3600.0) > 1e-9
        ):
            raise ValueError("record 900초 wall/compute 비용 상한이 고정 계약과 다르다")
        remote = job.get("remote_gpu_admission")
        if not isinstance(remote, Mapping) or not isinstance(remote.get("path"), str):
            raise TypeError("remote GPU admission 경로가 없다")
        admission = V1._read(V1._single_regular(Path(remote["path"]), "render admission"))
        now = _utc_now()
        wall_deadlines = (
            CONTEXT._utc_stamp(admission.get("expires_at_utc"), "admission expires_at_utc"),
            CONTEXT._utc_stamp(resources.get("deadline_utc"), "source deadline_utc"),
        )
        seconds = min((stamp - now).total_seconds() for stamp in wall_deadlines)
        if seconds <= 0:
            raise TimeoutError("admission 또는 source deadline이 끝났다")
        return time.monotonic() + min(float(ACTION_TIMEOUT_S), seconds)

    def _capture_context(
        self,
        job_path: Path,
        snapshot: dict[str, Any],
        python: Path,
        output: Path,
        deadline: float,
    ) -> Path:
        def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            kwargs["timeout"] = min(30.0, _remaining(deadline))
            return subprocess.run(argv, **kwargs)  # noqa: PLW1510 - caller fixes check=False

        return CONTEXT.capture_context(job_path, snapshot, python, output, runner=runner)

    def _cached_capture(
        self, stage: Path, job: Mapping[str, Any], mode: str,
    ) -> dict[str, Any] | None:
        if not stage.exists():
            return None
        if not stage.is_dir() or stage.is_symlink():
            raise ValueError("render stage가 symlink 아닌 directory가 아니다")
        job_path = stage / "job.json"
        if not job_path.is_file() or job_path.is_symlink() or V1._read(job_path) != dict(job):
            raise ValueError("같은 render stage에 다른 job을 재사용할 수 없다")
        execution_path = stage / "execution.json"
        if not execution_path.is_file():
            return {"status": "HOLD", "reason": "render_pending", "job_id": job.get("job_id")}
        execution = V1._read(V1._single_regular(execution_path, "render execution"))
        if execution.get("completed") is not True:
            raise B.BackendError(str(execution.get("error_type", "InvalidRenderEvidence")), execution)
        attestation = stage / "attestation" / ATTEST.ATTESTATION_FILE
        if not attestation.is_file():
            raise ValueError("완료 render execution에 attestation이 없다")
        return self._verified_payload(stage, dict(job), mode)

    def _record_new(self, job: dict[str, Any], stage: Path) -> dict[str, Any]:
        deadline = self._action_deadline(job)
        stage.mkdir(parents=True, exist_ok=False)
        job_path = stage / "job.json"
        B.write_new(job_path, job)

        def execute() -> dict[str, dict[str, str]]:
            return self._execute_worker(job_path, stage, deadline)

        ATTEST.attest_capture(
            job_path,
            self.recovery._source_snapshot,
            self.recovery.python,
            (stage / "attestation").resolve(),
            execute,
            context_capture=lambda path, snapshot, python, output: self._capture_context(
                path, snapshot, python, output, deadline,
            ),
        )
        _remaining(deadline)
        return self._verified_payload(stage, job, str(job["mode"]))

    def _execute_worker(
        self, job_path: Path, stage: Path, deadline: float,
    ) -> dict[str, dict[str, str]]:
        receipt: dict[str, Any] = {
            "schema": "vision_brev_record_execution_v1",
            "completed": False,
            "job_sha256": B.sha256(job_path),
            "exit_code": None,
            "timed_out": False,
            "cleanup_verified": True,
            "gpu_before": None,
            "gpu_after": None,
            "gpu_before_observation": None,
            "gpu_after_observation": None,
            "source_before": None,
            "source_after": None,
            "result_sha256": None,
            "launch": None,
        }
        process: subprocess.Popen[Any] | None = None
        started = time.perf_counter()
        error: BaseException | None = None
        result_path = stage / "output" / "result.json"
        try:
            validated = R.validate_job(job_path)
            if validated["output_directory"] != (stage / "output").resolve():
                raise ValueError("render output_directory는 소유 stage/output이어야 한다")
            receipt["source_before"] = self.recovery._source_snapshot()
            receipt["gpu_before"] = self.recovery.read_gpu()
            receipt["gpu_before_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
            if receipt["gpu_before"]:
                raise B.BackendError("ForeignGPUBusy", receipt)
            _remaining(deadline)
            with (stage / "stdout.log").open("x", encoding="utf-8") as stdout, (
                stage / "stderr.log"
            ).open("x", encoding="utf-8") as stderr:
                process = self.recovery.popen(
                    [
                        str(self.recovery.python), "-I", "-S", "-c", OWNED_WORKER_GATE,
                        str(self.recovery.python), "-B", str(VISION_WORKER), "--job", str(job_path),
                    ],
                    cwd=str(self.recovery.lab),
                    env={
                        **os.environ,
                        "OMNI_KIT_ACCEPT_EULA": "YES",
                        "PYTHONUNBUFFERED": "1",
                        "CUDA_VISIBLE_DEVICES": "0",
                    },
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    receipt["launch"] = V1._process_identity(process.pid)
                except BaseException:  # identity 없는 gate는 PID로만 회수한다.
                    receipt["cleanup_verified"] = _cleanup_waiting_gate(process)
                    process = None
                    raise
                if process.stdin is None:
                    raise RuntimeError("owned worker gate stdin이 없다")
                process.stdin.write(b"GO\n")
                process.stdin.flush()
                process.stdin.close()
                try:
                    receipt["exit_code"] = process.wait(timeout=_remaining(deadline))
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                finally:
                    receipt["cleanup_verified"] = V1._safe_cleanup(receipt["launch"], process)
                    if receipt["timed_out"]:
                        receipt["exit_code"] = -signal.SIGTERM
            receipt["gpu_after"] = self.recovery.read_gpu()
            receipt["gpu_after_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
            receipt["source_after"] = self.recovery._source_snapshot()
            if isinstance(self.recovery.read_gpu, REMOTE.StrictRemoteGpuReader):
                self.verify_execution_gpu_observations(receipt)
            result_path = V1._owned(result_path, self.runtime_root, "render result")
            result = self._verify_mode_result(result_path, validated["job"], receipt["job_sha256"])
            receipt["result_sha256"] = B.sha256(result_path)
            auxiliary = self._execution_auxiliary(result_path, validated["job"])
            if auxiliary is not None:
                receipt["schema"] = "vision_brev_record_execution_v2"
                receipt["auxiliary"] = auxiliary
            _remaining(deadline)
            receipt["completed"] = bool(
                receipt["exit_code"] == 0
                and receipt["timed_out"] is False
                and receipt["cleanup_verified"] is True
                and receipt["gpu_after"] == []
                and receipt["source_before"] == receipt["source_after"]
                and result.get("completed") is True
            )
            if not receipt["completed"]:
                raise ValueError("render execution 완료 증거가 유효하지 않다")
        except BaseException as caught:  # noqa: BLE001 - 실패 증거도 write-once로 닫는다.
            error = caught
            receipt["error_type"] = caught.kind if isinstance(caught, B.BackendError) else type(caught).__name__
            receipt["error_message"] = str(caught)
            if process is not None:
                try:
                    launch = receipt.get("launch")
                    receipt["cleanup_verified"] = bool(
                        isinstance(launch, Mapping) and V1._safe_cleanup(launch, process)
                    )
                    receipt["exit_code"] = process.returncode
                except BaseException:  # noqa: BLE001
                    receipt["cleanup_verified"] = False
            if receipt["source_after"] is None:
                try:
                    receipt["source_after"] = self.recovery._source_snapshot()
                except BaseException:  # noqa: BLE001
                    receipt["source_after"] = None
            if receipt["gpu_after"] is None:
                try:
                    receipt["gpu_after"] = self.recovery.read_gpu()
                    receipt["gpu_after_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
                except BaseException:  # noqa: BLE001
                    receipt["gpu_after"] = None
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            B.write_new(stage / "execution.json", receipt)
        if error is not None or receipt["completed"] is not True:
            raise B.BackendError(str(receipt.get("error_type", "InvalidRenderEvidence")), receipt)
        return {
            "execution": {"path": str((stage / "execution.json").resolve()), "sha256": B.sha256(stage / "execution.json")},
            "result": {"path": str(result_path.resolve()), "sha256": B.sha256(result_path)},
        }

    def _execution_auxiliary(
        self, result_path: Path, job: Mapping[str, Any],
    ) -> dict[str, str] | None:
        """기존 worker는 보조 결합 없이 v1 실행 영수증을 유지한다."""
        return None

    def _verify_mode_result(
        self, result_path: Path, job: Mapping[str, Any], job_sha: str,
    ) -> dict[str, Any]:
        if job.get("mode") == R.RECORD_MODE:
            return V1._verify_record_result(result_path, job, job_sha)
        result = V1._read(V1._single_regular(result_path, "render smoke result"))
        if set(result) != V1.RESULT_KEYS:
            raise ValueError("render smoke result 필드가 계약과 다르다")
        remote = job.get("remote_gpu_admission")
        resource = job.get("resource")
        if not isinstance(remote, Mapping) or not isinstance(resource, Mapping):
            raise TypeError("render smoke resource/admission 결합이 없다")
        admission_path = V1._single_regular(Path(str(remote.get("path"))), "render smoke admission")
        admission = V1._read(admission_path)
        expected = {
            "schema": R.RESULT_SCHEMA,
            "completed": True,
            "job_sha256": job_sha,
            "protocol_sha256": job["protocol"]["sha256"],
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "request_binding": R.request_binding(job),
            "resource": dict(resource),
            "resource_parent_sha256": remote["resource_parent_sha256"],
            "live_lease_proof_sha256": remote["live_lease_proof_sha256"],
            "admission_sha256": remote["sha256"],
            "admission_issued_at_utc": admission.get("issued_at_utc"),
            "admission_expires_at_utc": admission.get("expires_at_utc"),
            "profile_sha256": R.digest(R.fixed_profile()),
            "mode": R.RENDER_SYNC_SMOKE_MODE,
            "video_sha256": None,
            "fixed_evaluation_used_as_behavior_ground_truth": False,
            "independent_visual_evidence_source_count": 1,
            "zero_update_pixel_lag_confirmed": False,
            "gpu_smoke_required_before_model_use": False,
        }
        if (
            any(result.get(key) != value for key, value in expected.items())
            or set(result.get("artifacts", {})) != SOURCES._SMOKE_ARTIFACTS
        ):
            raise ValueError("render smoke result provenance/artifact가 잘못됐다")
        for failure_name in ("failure.json", "shutdown_failure.json"):
            if (result_path.parent / failure_name).exists():
                raise ValueError("render smoke 완료 폴더에 실패 증거가 함께 있다")
        artifacts = result["artifacts"]
        names: set[str] = set()
        paths: dict[str, Path] = {}
        for logical_id, descriptor in artifacts.items():
            if not isinstance(descriptor, dict) or set(descriptor) != {"file", "sha256", "bytes"}:
                raise ValueError("render smoke artifact descriptor가 잘못됐다")
            name = descriptor["file"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in names
                or not isinstance(descriptor["sha256"], str)
                or V1.HEX64.fullmatch(descriptor["sha256"]) is None
                or type(descriptor["bytes"]) is not int
                or descriptor["bytes"] <= 0
            ):
                raise ValueError("render smoke artifact 이름·SHA·크기가 잘못됐다")
            matches = [
                path for path in result_path.parent.rglob(name)
                if path.is_file() and not path.is_symlink()
            ]
            if len(matches) != 1:
                raise ValueError("render smoke artifact가 없거나 같은 이름이 중복됐다")
            artifact = V1._single_regular(matches[0], f"render smoke artifact {logical_id}")
            if artifact.stat().st_size != descriptor["bytes"] or B.sha256(artifact) != descriptor["sha256"]:
                raise ValueError("render smoke artifact 크기 또는 SHA가 다르다")
            names.add(name)
            paths[logical_id] = artifact
        report = paths["render_sync_smoke"]
        if result["primary_artifact_sha256"] != B.sha256(report):
            raise ValueError("render smoke primary artifact가 report와 다르다")
        return result

    def capture_references(self, job: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        if not isinstance(job, Mapping):
            raise TypeError("capture reference job은 mapping이어야 한다")
        mode = job.get("mode")
        if not isinstance(mode, str) or mode not in {R.RECORD_MODE, R.RENDER_SYNC_SMOKE_MODE}:
            raise ValueError("capture reference mode가 잘못됐다")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or V1.SLUG.fullmatch(job_id) is None:
            raise ValueError("capture reference job_id가 잘못됐다")
        stage = V1._owned(self.runtime_root / "record" / job_id, self.runtime_root, "render stage")
        cached = self._cached_capture(stage, job, mode)
        if cached is None or cached.get("status") == "HOLD":
            raise ValueError("완료·검증된 capture만 reference를 제공한다")
        return self._build_capture_references(job)

    def _build_capture_references(self, job: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or V1.SLUG.fullmatch(job_id) is None:
            raise ValueError("capture reference job_id가 잘못됐다")
        stage = V1._owned(self.runtime_root / "record" / job_id, self.runtime_root, "render stage")
        execution = V1._read(stage / "execution.json")
        result = V1._read(stage / "output" / "result.json")
        files: dict[str, Path] = {
            "job": stage / "job.json",
            "protocol": Path(str(job["protocol"]["path"])),
            "admission": Path(str(job["remote_gpu_admission"]["path"])),
            "result": stage / "output" / "result.json",
            "execution": stage / "execution.json",
            "context": stage / "attestation" / "context" / CONTEXT.RECEIPT_FILE,
            "attestation": stage / "attestation" / ATTEST.ATTESTATION_FILE,
            "gpu_observation_before": Path(str(execution["gpu_before_observation"]["path"])),
            "gpu_observation_after": Path(str(execution["gpu_after_observation"]["path"])),
        }
        auxiliary = ATTEST.execution_evidence_path(
            stage / "execution.json", execution, job, stage / "output" / "result.json",
        )
        if auxiliary is not None:
            files["execution_evidence"] = auxiliary
        output = stage / "output"
        if job.get("mode") == R.RECORD_MODE:
            files["video"] = output / "same_play.mp4"
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise TypeError("capture result artifact 목록이 없다")
        for logical_id, descriptor in artifacts.items():
            if not isinstance(logical_id, str) or not isinstance(descriptor, Mapping):
                raise TypeError("capture artifact descriptor가 잘못됐다")
            name = descriptor.get("file")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("capture artifact 이름이 잘못됐다")
            matches = [path for path in output.rglob(name) if path.is_file() and not path.is_symlink()]
            if len(matches) != 1:
                raise ValueError("capture artifact 원문이 없거나 중복됐다")
            files[logical_id] = matches[0]
        return {key: self._repo_reference(path, key) for key, path in files.items()}

    def _repo_reference(self, path: Path, anchor: str) -> dict[str, str]:
        root = ROOT.resolve()
        original = Path(path)
        if not original.is_absolute() or original.resolve() != original:
            raise ValueError(f"{anchor} 경로는 canonical absolute path여야 한다")
        current = original
        while current != root:
            if current.is_symlink() or current.parent == current:
                raise ValueError(f"{anchor} 경로에 symlink가 있거나 저장소를 벗어났다")
            current = current.parent
        actual = V1._single_regular(original, anchor)
        try:
            relative = actual.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{anchor}는 저장소 내부 원본이어야 한다") from error
        name = relative.as_posix()
        if not name or name.startswith("../"):
            raise ValueError(f"{anchor} 저장소 상대경로가 잘못됐다")
        return {"path": name, "sha256": B.sha256(actual)}

    def _verified_payload(
        self, stage: Path, job: dict[str, Any], mode: str,
    ) -> dict[str, Any]:
        result_path = stage / "output" / "result.json"
        job_sha = B.sha256(stage / "job.json")
        result = self._verify_mode_result(result_path, job, job_sha)
        refs = self._build_capture_references(job)
        source_protocol = self._source_protocol()
        remote = job["remote_gpu_admission"]
        resource = getattr(self.recovery, "vision_resource_proof", None)
        if not isinstance(resource, dict):
            raise TypeError("capture 전체 검증에 resource proof가 필요하다")
        capture = SOURCES._verify_capture(
            ROOT.resolve(),
            refs,
            mode,
            source_protocol,
            str(remote["resource_parent_sha256"]),
            str(remote["live_lease_proof_sha256"]),
            resource,
        )
        if mode == R.RENDER_SYNC_SMOKE_MODE:
            SOURCES._verify_smoke(capture)
        execution = V1._read(stage / "execution.json")
        status = (
            "capture_complete_gpu_smoke_pending"
            if mode == R.RECORD_MODE
            else "render_sync_smoke_complete"
        )
        return {"status": status, "model_use_ready": False, "result": result, "execution": execution}
