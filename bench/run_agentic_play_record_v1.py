"""Go2 F2·정상 쌍의 same-Play 녹화 실행기. RUN5 부모·자식 흐름을 그대로 쓰고 새 JOB·새 entry·
상태 sidecar·F2 playback을 결합한다.

- 부모: 새 JOB으로 먼저 검증(F1·v3 job, 미등록 checkpoint, 입장 원문 누락은 child 실행 전 거부)한 뒤
  RUN5 ``run_job``을 새 계약 안에서 호출한다. AABA·400초 child·600초 policy·200초 post reserve·
  native 625 frame 50fps·원문/PTS 검증·gpu_reader·cleanup·실패 봉인·자원 입장은 RUN5 그대로다.
- 자식 명령: RUN5 진입점만 이 파일로 바꾼다. F1 transferred·sidecar 기본 진입점은 실행하지 않는다.
- 녹화 자식: ``SIDECAR.child_main(transferred=이 모듈)``으로 sidecar worker를 붙이고, inner worker의
  ``run_real``에 메인 소유 ``agentic_fault_playback_v1.make_run_real(등록값)``을 넣는다.
- 종료 뒤: native 성공만으로 끝내지 않고 sidecar 봉인·trace 대조·playback receipt·실제 모터 함수
  출처를 함께 확인해 ``agentic-play-final.json``을 쓴다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_fault_playback_v1 as PLAYBACK
import agentic_play_job_v1 as JOB
import record_agentic_state_sidecar_v1 as SIDECAR
import run_vision_pull_record_v5 as PREVIOUS
import vision_pull_record_worker_v3 as WORKER

POSTFLIGHT_RESERVE_S = float(JOB.LIMITS["postflight_reserve_s_per_policy"])
FINAL_SCHEMA = "agentic_play_final_v1"
FINAL_NAME = "agentic-play-final.json"
CHILD_FLAGS = ("--aaba-child", "--sdk-child")
# RUN5 대신 실행되면 안 되는 기존 진입점. sidecar 모듈의 기본 child는 F1 v3로 위임한다.
FOREIGN_ENTRIES = frozenset({
    "run_vision_transferred_pull_record_v1.py",
    "run_vision_transferred_pull_record_v2.py",
    "run_vision_transferred_pull_record_v3.py",
    "record_agentic_state_sidecar_v1.py",
    "run_vision_pull_record_v1.py",
})
NONE_CLIP_FUNCTION = "isaaclab.actuators.actuator_pd.DCMotor._clip_effort"
F2_CLIP_MODULE_PREFIX = "hidden_faults."
# 운영 고정 callable. 하나라도 다른 값이 들어오면 그 실행은 live evidence가 아니다.
PRODUCTION = {
    "popen": subprocess.Popen,
    "identify": PREVIOUS.OWNED._process_identity,
    "cleanup": PREVIOUS.OWNED._safe_cleanup,
}


def live_arguments(
    popen: Callable[..., Any], gpu_reader: Any, identify: Callable[..., Any], cleanup: Callable[..., Any]
) -> bool:
    return (
        popen is PRODUCTION["popen"]
        and gpu_reader is None
        and identify is PRODUCTION["identify"]
        and cleanup is PRODUCTION["cleanup"]
    )


@contextmanager
def _job_contract() -> Iterator[None]:
    previous = (
        PREVIOUS.JOB,
        PREVIOUS.WORKER,
        PREVIOUS.REPORT_NAME,
        PREVIOUS.POSTFLIGHT_RESERVE_S,
        PREVIOUS.V1.WORKER,
        PREVIOUS.V1.REPORT_NAME,
    )
    PREVIOUS.JOB = JOB
    PREVIOUS.WORKER = WORKER
    PREVIOUS.REPORT_NAME = WORKER.REPORT_NAME
    PREVIOUS.POSTFLIGHT_RESERVE_S = POSTFLIGHT_RESERVE_S
    PREVIOUS.V1.WORKER = WORKER
    PREVIOUS.V1.REPORT_NAME = WORKER.REPORT_NAME
    try:
        yield
    finally:
        (
            PREVIOUS.JOB,
            PREVIOUS.WORKER,
            PREVIOUS.REPORT_NAME,
            PREVIOUS.POSTFLIGHT_RESERVE_S,
            PREVIOUS.V1.WORKER,
            PREVIOUS.V1.REPORT_NAME,
        ) = previous


def aaba_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.aaba_child(job_path, **kwargs)


def sdk_child(job_path: Path, **kwargs: Any) -> int:
    with _job_contract():
        return PREVIOUS.sdk_child(job_path, **kwargs)


def _child_launcher(
    popen: Callable[..., Any], job_path: Path, *, before_launch: Callable[[], Any] | None = None
) -> Callable[..., Any]:
    old = str(Path(PREVIOUS.__file__).resolve())
    current = str(Path(__file__).resolve())
    job = str(Path(job_path).absolute())

    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        values = list(command)
        if values.count(old) != 1:
            raise ValueError("새 entry로 바꿀 RUN5 진입점이 정확히 하나가 아니다")
        if any(value.endswith(name) for value in values for name in FOREIGN_ENTRIES):
            raise ValueError("F1 transferred·sidecar 기본 진입점은 실행하지 않는다")
        index = values.index(old)
        tail = values[index + 1:]
        if len(tail) != 3 or tail[0] not in CHILD_FLAGS or tail[1:] != ["--job", job]:
            raise ValueError("child 인자가 등록 job과 다르다")
        if before_launch is not None:
            before_launch()  # AABA·record child 직전마다 부모 lease 활성 확인
        values[index] = current
        return popen(values, **kwargs)

    return launch


def _lease_checker(validated: Mapping[str, Any]) -> Callable[[], dict[str, Any]]:
    reference = validated["job"]["lease"]

    def check() -> dict[str, Any]:
        return JOB.verify_lease(reference, protocol=validated["protocol"], protocol_sha=validated["protocol_sha256"])

    return check


def _leased_reader(reader: Callable[[], list[int]], check: Callable[[], Any]) -> Callable[[], list[int]]:
    def read() -> list[int]:
        check()  # GPU reader 호출 전마다 lease 활성 확인
        return reader()

    return read


def sidecar_environ(job: Mapping[str, Any]) -> dict[str, str]:
    row = JOB.COMMON._exact(job["sidecar"], JOB.SIDECAR_KEYS, "job sidecar")
    return {
        SIDECAR.ENV_SIDECAR_DIR: str(row["directory"]),
        SIDECAR.ENV_SOURCE_ID: str(row["source_id"]),
        SIDECAR.ENV_EXPECTED_CHECKPOINT: str(job["checkpoint"]["sha256"]),
        SIDECAR.ENV_ERROR_POLICY: str(row["error_policy"]),
    }


def playback_worker(configured: Mapping[str, Any]) -> Callable[..., dict[str, Any]]:
    """sidecar worker 안쪽 worker. 등록 playback의 run_real을 기존 native worker에 넣는다."""
    expected = dict(configured)

    def worker(validated: Mapping[str, Any], app: Any, **kwargs: Any) -> dict[str, Any]:
        if "run_real" in kwargs:
            raise TypeError("run_real은 등록 playback만 결합한다")
        if validated["job"]["sidecar"] != expected:
            raise ValueError("검증된 job sidecar가 child 설정과 다르다")
        identity = validated["job"]["checkpoint"]["logical_id"]
        registration = validated["protocol"]["playback_registrations"][identity]
        return WORKER.run_policy_record(
            validated, app, run_real=PLAYBACK.make_run_real(registration), **kwargs
        )

    return worker


def child_main(flag: str, job_path: Path) -> int:
    if flag == "--aaba-child":
        return aaba_child(job_path)
    if flag != "--sdk-child":
        raise ValueError("child flag가 등록 밖이다")
    path = JOB._regular(Path(job_path).absolute(), "agentic play job")
    job = JOB._read(path)
    if job.get("schema") != JOB.JOB_SCHEMA:
        raise ValueError("새 agentic play job이 아니다")
    configured = JOB.COMMON._exact(job["sidecar"], JOB.SIDECAR_KEYS, "job sidecar")
    return SIDECAR.child_main(
        "--sdk-child",
        path,
        environ=sidecar_environ(job),
        transferred=sys.modules[__name__],
        run_worker=playback_worker(configured),
        expected_frames=int(configured["expected_frames"]),
    )


def _descriptor(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"file": path.name, "sha256": JOB._sha(path), "bytes": path.stat().st_size}


def _clip_consistent(manifest: Mapping[str, Any] | None, fault_id: str) -> bool:
    actuators = None if manifest is None else manifest.get("actuators")
    if not isinstance(actuators, Mapping) or not actuators:
        return False
    for group in actuators.values():
        function = group.get("clip_effort_function") if isinstance(group, Mapping) else None
        if not isinstance(function, str):
            return False
        if fault_id == "NONE" and function != NONE_CLIP_FUNCTION:
            return False
        if fault_id == "F2" and not function.startswith(F2_CLIP_MODULE_PREFIX):
            return False
    return True


def finalize(
    validated: Mapping[str, Any], receipt: Mapping[str, Any], *, live_evidence: bool
) -> dict[str, Any]:
    """native 성공·sidecar 봉인·trace 대조·playback receipt·모터 함수 출처를 함께 확인한다."""
    stage = Path(validated["job_path"]).parent
    output = Path(validated["output_directory"])
    sidecar_dir = Path(validated["job"]["sidecar"]["directory"])
    registration = validated["playback_registration"]
    native = receipt.get("completed") is True and receipt.get("technical_recording_complete") is True
    manifest_path = sidecar_dir / SIDECAR.MANIFEST_NAME
    failure_path = sidecar_dir / SIDECAR.FAILURE_NAME
    status = "sealed" if manifest_path.is_file() else "failed" if failure_path.is_file() else "missing"
    manifest: dict[str, Any] | None = None
    join: dict[str, Any] | None = None
    join_error: str | None = None
    if status == "sealed":
        try:
            manifest, _records = SIDECAR.load_sidecar(sidecar_dir)
            if native:
                join = SIDECAR.join_with_trace(
                    sidecar_dir, output / "evidence" / "raw_trace.json", output / "native_capture_manifest.json"
                )
        except Exception as error:  # noqa: BLE001 - 대조 실패도 최종 기록에 남긴다.
            join_error = f"{type(error).__name__}: {error}"
    playback_path = stage / "playback-receipt.json"
    playback = JOB._read(playback_path) if playback_path.is_file() else None
    playback_ok = (
        isinstance(playback, dict)
        and playback.get("status") == "completed"
        and playback.get("registration") == registration
        and playback.get("installation", {}).get("fault_id") == registration["fault_id"]
    )
    clip_ok = _clip_consistent(manifest, registration["fault_id"])
    lease: dict[str, Any] | None = None
    lease_error: str | None = None
    try:
        lease = _lease_checker(validated)()
    except Exception as error:  # noqa: BLE001 - 종료 시점 lease 상태도 최종 기록에 남긴다.
        lease_error = f"{type(error).__name__}: {error}"
    complete = bool(native and status == "sealed" and join is not None and playback_ok and clip_ok
                    and lease is not None)
    live_lock = lease is not None and lease["live_lock_proof"] is True
    body = {
        "schema": FINAL_SCHEMA,
        "job_sha256": validated["job_sha256"],
        "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "fault_id": registration["fault_id"],
        "pair": validated["pair"],
        "execution": _descriptor(stage / "execution.json"),
        "native_technical_complete": native,
        "native_error": receipt.get("error_message"),
        "sidecar": {
            "status": status,
            "directory": str(sidecar_dir),
            "manifest": _descriptor(manifest_path),
            "failure": _descriptor(failure_path),
            "error_policy": validated["job"]["sidecar"]["error_policy"],
        },
        "join": join,
        "join_error": join_error,
        "playback_receipt": _descriptor(playback_path),
        "playback_completed": playback_ok,
        "actuator_clip_consistent_with_registration": clip_ok,
        "lease": {
            "reference": dict(validated["job"]["lease"]),
            "active_at_finalize": lease is not None,
            "lease_id": None if lease is None else lease["lease_id"],
            "lock_proof": None if lease is None else lease["lock_proof"],
            "live_lock_proof": live_lock,
            "error": lease_error,
        },
        "final_technical_complete": complete,
        "live_evidence": live_evidence,
        "acceptable_as_live_evidence": complete and live_evidence and live_lock,
        "interpretation": JOB.INTERPRETATION,
        "behavior_label": None,
        "used_for_decision": False,
    }
    JOB.write_new(stage / FINAL_NAME, body)
    return body


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = PREVIOUS.OWNED._process_identity,
    cleanup: Callable[..., bool] = PREVIOUS.OWNED._safe_cleanup,
) -> dict[str, Any]:
    live = live_arguments(popen, gpu_reader, identify, cleanup)  # 내부 wrapper 전 최초 인자로 판정
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("agentic play 실제 실행은 소유 Linux group에서 수행한다")
    validated = JOB.validate_job(job_path)
    stage = Path(validated["job_path"]).parent
    if (stage / FINAL_NAME).exists():
        raise FileExistsError("agentic play job을 반복 실행하지 않는다")
    check = _lease_checker(validated)
    reader = gpu_reader
    if reader is None:  # RUN5 기본 reader와 같은 구성. lease 확인을 앞에 두려고 여기서 만든다
        resource = validated["resource"]
        deadline = time.monotonic() + JOB.LIMITS["max_wall_s_per_policy"]
        reader = PREVIOUS.REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in PREVIOUS.REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=PREVIOUS.BASE._gpu_query_run(deadline),
        )
    with _job_contract():
        receipt = PREVIOUS.run_job(
            job_path,
            popen=_child_launcher(popen, job_path, before_launch=check),
            gpu_reader=_leased_reader(reader, check),
            identify=identify,
            cleanup=cleanup,
        )
    return finalize(validated, receipt, live_evidence=live)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--aaba-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id")
    args = parser.parse_args(argv)
    if args.sdk_child or args.aaba_child:
        if args.job is None or (args.sdk_child and args.aaba_child):
            parser.error("child 실행에는 flag 하나와 --job이 필요하다")
        return child_main("--aaba-child" if args.aaba_child else "--sdk-child", args.job)
    if args.job is None:
        required = (args.protocol, args.resource_proof, args.resource_sha256, args.isaaclab,
                    args.python, args.stage, args.checkpoint_logical_id)
        if any(value is None for value in required):
            parser.error("job 생성 인자가 모두 필요하다")
        args.job = JOB.create_job(
            args.protocol, args.resource_proof, args.resource_sha256,
            args.isaaclab, args.python, args.stage, args.checkpoint_logical_id,
        )
    result = run_job(args.job)
    return 0 if result.get("acceptable_as_live_evidence") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
