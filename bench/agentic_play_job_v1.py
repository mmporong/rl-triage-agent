"""Go2 F2·정상 checkpoint 쌍을 same-Play 녹화와 상태 sidecar로 묶는 새 등록 계약.

F1 transferred 계약(historical catalog·protocol·기록)은 읽지도 바꾸지도 않는다. checkpoint는 새
protocol의 generic catalog(논리 id·checkpoint SHA·Isaac Lab 상대 경로)로 고정하고, 실행마다 최대
2 policy(F2 한 건과 같은 seed 정상 한 건)를 녹화한다. 검증 helper는 기존 COMMON·RECORD_JOB의
checkpoint·runtime pin·source ancestor/blob·고유 파일 검사를 그대로 쓴다.

새 F2 영상은 학습 fault가 적용된 Play이고 정상 대조는 NONE checkpoint의 같은 Play다. 정상 env 고정
평가의 행동 정답으로 부르지 않으며, F2는 한쪽 다리 결함이 아니라 12관절 DCMotor 토크-속도 함수 변경이다.

자원 입장은 실제 임대(lease)를 요구한다. ``resource_lease``가 자원별 고정 경로의 OS advisory lock
(``vision_brev_backend_v1._gpu_lock`` 재사용)을 잡고, 불변 snapshot과 독립 thread heartbeat를 남긴다.
resource proof·job은 snapshot 참조를 결합하고, 부모·child 검증은 lease를 다시 잡지 않고 소유 프로세스의
생존·lock 보유·heartbeat를 확인한다.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import agentic_fault_playback_v1 as PLAYBACK
import record_agentic_state_sidecar_v1 as SIDECAR
import vision_brev_backend_v1 as OWNED  # pyright: ignore[reportMissingImports]
import vision_pull_record_job_v5 as BASE
import vision_transferred_pull_job_v1 as TRANSFERRED
import vision_transferred_pull_job_v3 as TRANSFERRED_V3

COMMON = BASE.COMMON
RECORD = BASE.RECORD
RECORD_JOB = BASE.RECORD_JOB
REMOTE = TRANSFERRED.REMOTE
ROOT = BASE.ROOT
MODE = BASE.MODE
PROTOCOL_SCHEMA = "agentic_play_protocol_v1"
JOB_SCHEMA = "agentic_play_job_v1"
RESOURCE_SCHEMA = "agentic_play_resource_v1"
RESULT_SCHEMA = BASE.RESULT_SCHEMA
FAILURE_SCHEMA = BASE.FAILURE_SCHEMA
TECHNICAL_LIMITS = TRANSFERRED.TECHNICAL_LIMITS
LIMITS = TECHNICAL_LIMITS
FIXED_FLAGS = BASE.FIXED_FLAGS
FRESH_CAPTURE_S = TRANSFERRED.FRESH_CAPTURE_S
FAULT_IDS = ("NONE", "F2")
SIDECAR_SUFFIX = "-sidecar"
OWN_FILES = frozenset({
    "bench/agentic_play_job_v1.py",
    "bench/run_agentic_play_record_v1.py",
    "tests/test_agentic_play_job_v1.py",
    "tests/test_run_agentic_play_record_v1.py",
})
# 이 계약이 검증·임대에 쓰는 실행 helper(COMMON, RECORD_JOB, REMOTE, OWNED).
HELPER_SOURCE_FILES = frozenset({
    "bench/vision_semantic_fingerprint_job_v1.py",
    "bench/vision_record_job_v1.py",
    "evals/vision_remote_gpu_v1.py",
    "evals/vision_brev_backend_v1.py",
})
REQUIRED_SOURCE_FILES = frozenset(
    set(BASE.REQUIRED_SOURCE_FILES)
    | set(TRANSFERRED_V3.REQUIRED_SOURCE_FILES)
    | set(SIDECAR.SOURCE_FILES)
    | set(PLAYBACK.SOURCE_FILES)
    | OWN_FILES
    | HELPER_SOURCE_FILES
    | {"bench/record_vision_evidence_v1.py", "bench/vision_play_worker_v1.py", "bench/vision_play_worker_v3.py"}
)
# F2 playback이 교체하는 DCMotor와 Go2 actuator 설정을 실제 Isaac Lab 설치본에서 고정한다.
REQUIRED_RUNTIME_FILES = frozenset({
    "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
})
INTERPRETATION = {
    "f2_scope": "all_12_go2_joints_dcmotor_torque_speed_function",
    "single_leg_fault": False,
    "normal_control": "same_seed_none_checkpoint_same_play",
    "playback_scope": "same_play_state_not_pristine_fixed_evaluation",
    "fixed_eval_relation": "separate_pristine_env_execution_not_behavior_ground_truth",
    "p_torque_relation": "separate_fault_reinjected_probe",
    "behavior_labels_created": False,
    "thresholds_created": False,
}
PROTOCOL_KEYS = {
    "schema", "protocol_id", "registered_before_measurement", "registered_at_utc",
    "source_commit", "source_sha256_lf", "runtime_source_sha256_lf", "recording_target",
    "resources", "checkpoints", "playback_registrations", "pairs", "sidecar",
    "record_profile", "limits", "interpretation", "preregistration", "gpu_validation_status",
}
JOB_KEYS = set(RECORD_JOB.JOB_KEYS) | {"sidecar", "lease"}
SIDECAR_KEYS = {"directory", "source_id", "error_policy", "expected_frames"}
RESOURCE_KEYS = {
    "schema", "status", "observed_at_utc", "expires_at_utc",
    "recording_protocol_sha256_lf", "recording_target", "capture", "lease",
}
PAIR_KEYS = {"pair_id", "seed", "case", "reference"}
RESOURCE_SCOPE = "new_recording_instance_elapsed_wall_excludes_storage_v1"
LEASE_SCHEMA = "agentic_play_lease_v1"
HEARTBEAT_SCHEMA = "agentic_play_lease_heartbeat_v1"
# 자원별 고정 위치. run마다 다른 lock으로 배타성을 우회하지 않도록 경로는 resource ID로만 정한다.
LEASE_ROOT = Path.home() / ".cache" / "rl-triage" / "cloud-gpu-leases"
LEASE_TTL_S = 3600
HEARTBEAT_INTERVAL_S = 2.0
HEARTBEAT_MAX_AGE_S = 15.0
CLOCK_SKEW_S = 5.0
LEASE_KEYS = {
    "schema", "lease_id", "resource_id", "protocol_sha256_lf", "owner",
    "acquired_at_utc", "expires_at_utc", "lock_path", "lock_identity",
}
HEARTBEAT_KEYS = {"schema", "lease_id", "resource_id", "owner", "active", "sequence", "updated_at_utc"}
POSIX_LOCK_PROOF = "posix_proc_locks_flock_owner"
FIXTURE_LOCK_PROOF = "non_posix_second_handle_probe_fixture_only"

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new
_utc = TRANSFERRED._utc
_positive_decimal = TRANSFERRED._positive_decimal
_target = TRANSFERRED._target


class LeaseError(RuntimeError):
    """실제 자원 임대가 없거나 활성 상태가 아니다."""


def lease_paths(resource_id: str) -> dict[str, Path]:
    if not isinstance(resource_id, str) or COMMON.SLUG.fullmatch(resource_id) is None:
        raise LeaseError("lease resource ID 형식이 잘못됐다")
    root = Path(LEASE_ROOT).absolute()
    return {
        "lock": root / f"{resource_id}.lock",
        "heartbeat": root / f"{resource_id}.heartbeat.json",
        "snapshots": root / "snapshots",
    }


def _windows_start_identity(pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyright: ignore[reportAttributeAccessIssue]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise ProcessLookupError(pid)
    try:
        code = wintypes.DWORD()
        times = [wintypes.FILETIME() for _ in range(4)]
        if (not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259
                or not kernel32.GetProcessTimes(handle, *(ctypes.byref(item) for item in times))):
            raise ProcessLookupError(pid)
        return f"windows_creation_filetime:{(times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}"
    finally:
        kernel32.CloseHandle(handle)


def process_start_identity(pid: int) -> str:
    """살아 있는 프로세스의 시작 식별. POSIX는 기존 OWNED._start_ticks(/proc stat)를 쓴다."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LeaseError("lease owner PID가 잘못됐다")
    if os.name == "posix":
        try:
            return f"proc_start_ticks:{OWNED._start_ticks(pid)}"
        except (OSError, ValueError, IndexError) as error:
            raise ProcessLookupError(pid) from error
    return _windows_start_identity(pid)


def proc_lock_owners(text: str) -> set[tuple[int, int, int, int]]:
    """/proc/locks의 활성 FLOCK WRITE 행을 (pid, major, minor, inode)로 읽는다(대기 행 제외)."""
    owners: set[tuple[int, int, int, int]] = set()
    for line in text.splitlines():
        parts = line.split()
        if "->" in parts or "FLOCK" not in parts:
            continue
        index = parts.index("FLOCK")
        try:
            if parts[index + 2] != "WRITE":
                continue
            major, minor, inode = parts[index + 4].split(":")
            owners.add((int(parts[index + 3]), int(major, 16), int(minor, 16), int(inode)))
        except (IndexError, ValueError):
            continue
    return owners


def _lock_held(lock: Path, identity: Mapping[str, Any], pid: int) -> str:
    info = lock.stat()
    if {"device": info.st_dev, "inode": info.st_ino} != dict(identity):
        raise LeaseError("lease lock 파일이 snapshot 이후 바뀌었다")
    if os.name == "posix":
        owners = proc_lock_owners(Path("/proc/locks").read_text(encoding="utf-8"))
        if (pid, os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) not in owners:  # pyright: ignore[reportAttributeAccessIssue]
            raise LeaseError("lease owner가 자원 lock을 보유하지 않는다")
        return POSIX_LOCK_PROOF
    import msvcrt

    with lock.open("r+b") as handle:  # 다른 handle로 잡히면 아직 보유 중이다(fixture 판정만)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError:
            return FIXTURE_LOCK_PROOF
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
    raise LeaseError("lease 자원 lock이 잡혀 있지 않다")


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()  # noqa: UP017 - Isaac Python 3.10


@contextmanager
def resource_lease(
    protocol_path: Path, *, repo: Path = ROOT, ttl_s: float = LEASE_TTL_S, now: datetime | None = None
) -> Iterator[dict[str, str]]:
    """등록 recording 자원의 실제 임대. 같은 자원 경쟁자는 즉시 거부되고 다른 자원은 병렬로 잡힌다.

    yield 값은 불변 snapshot 참조 ``{"path", "sha256"}``다. heartbeat는 이 context 안의 thread가 갱신하고,
    종료 때 thread를 멈춘 뒤 inactive를 기록하고 lock을 푼다. 프로세스가 죽으면 OS가 lock을 푼다.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)  # noqa: UP017
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path, repo=repo, now=current)
    protocol_sha = _sha(protocol_path, lf=True)
    resource_id = protocol["_recording_target"]["resource_id"]
    paths = lease_paths(resource_id)
    COMMON._check_ancestors(paths["lock"].parent, "lease root")
    deadline = _utc(protocol["resources"]["deadline_utc"], "deadline")
    expires = min(deadline, current + timedelta(seconds=float(ttl_s)))
    if expires <= current:
        raise LeaseError("lease 만료가 현재 시각 뒤가 아니다")
    try:
        lock_context = OWNED._gpu_lock(paths["lock"])
        lock_context.__enter__()
    except OWNED.B.BackendError as error:
        raise LeaseError("같은 recording 자원 lease가 이미 잡혀 있다") from error
    stop = threading.Event()
    thread: threading.Thread | None = None
    primary: BaseException | None = None
    try:
        info = paths["lock"].stat()
        owner = {"pid": os.getpid(), "start_identity": process_start_identity(os.getpid())}
        lease_id = uuid.uuid4().hex
        snapshot = {
            "schema": LEASE_SCHEMA, "lease_id": lease_id, "resource_id": resource_id,
            "protocol_sha256_lf": protocol_sha, "owner": owner,
            "acquired_at_utc": _iso(current), "expires_at_utc": _iso(expires),
            "lock_path": str(paths["lock"]), "lock_identity": {"device": info.st_dev, "inode": info.st_ino},
        }
        snapshot_path = paths["snapshots"] / f"{resource_id}-{lease_id}.json"
        write_new(snapshot_path, snapshot)
        sequence = [0]

        def beat(active: bool) -> None:
            _replace_json(paths["heartbeat"], {
                "schema": HEARTBEAT_SCHEMA, "lease_id": lease_id, "resource_id": resource_id,
                "owner": owner, "active": active, "sequence": sequence[0],
                "updated_at_utc": _iso(datetime.now(timezone.utc)),  # noqa: UP017
            })
            sequence[0] += 1

        def loop() -> None:
            while not stop.wait(HEARTBEAT_INTERVAL_S):
                try:
                    beat(True)
                except OSError:
                    continue  # 실패가 계속되면 heartbeat가 오래되어 검증이 닫힌다

        beat(True)
        thread = threading.Thread(target=loop, name=f"agentic-play-lease-{lease_id}", daemon=True)
        thread.start()
        yield {"path": str(snapshot_path), "sha256": _sha(snapshot_path)}
    except BaseException as error:
        primary = error
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=HEARTBEAT_INTERVAL_S * 5)
            try:
                beat(False)  # pyright: ignore[reportPossiblyUnbound]
            except OSError:
                pass
        lock_context.__exit__(None, None, None)
        if thread is not None and thread.is_alive() and primary is None:
            raise LeaseError("lease heartbeat thread가 멈추지 않았다")


def verify_lease(
    reference: Any, *, protocol: Mapping[str, Any], protocol_sha: str, now: datetime | None = None
) -> dict[str, Any]:
    """lease를 다시 잡지 않고 소유 프로세스 생존·lock 보유·heartbeat 활성을 확인한다."""
    ref = COMMON._exact(reference, {"path", "sha256"}, "lease reference")
    try:
        path = _regular(Path(ref["path"]), "lease snapshot")
    except ValueError as error:
        raise LeaseError("lease snapshot이 없다") from error
    if _sha(path) != COMMON._hash(ref["sha256"], "lease SHA"):
        raise LeaseError("lease snapshot SHA가 다르다")
    snapshot = COMMON._exact(_read(path), LEASE_KEYS, "lease snapshot")
    resource_id = snapshot["resource_id"]
    paths = lease_paths(resource_id)
    if (snapshot["schema"] != LEASE_SCHEMA or resource_id != protocol["_recording_target"]["resource_id"]
            or snapshot["protocol_sha256_lf"] != protocol_sha
            or snapshot["lock_path"] != str(paths["lock"]) or path.parent != paths["snapshots"].resolve()):
        raise LeaseError("lease가 등록 자원·protocol·고정 lock 경로와 다르다")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)  # noqa: UP017
    acquired = _utc(snapshot["acquired_at_utc"], "lease acquired")
    expires = _utc(snapshot["expires_at_utc"], "lease expires")
    if (acquired > current + timedelta(seconds=CLOCK_SKEW_S) or current >= expires
            or expires > _utc(protocol["resources"]["deadline_utc"], "deadline")):
        raise LeaseError("lease가 유효 기간 밖이다")
    owner = COMMON._exact(snapshot["owner"], {"pid", "start_identity"}, "lease owner")
    try:
        alive = process_start_identity(owner["pid"]) == owner["start_identity"]
    except ProcessLookupError:
        alive = False
    if not alive:
        raise LeaseError("lease owner 프로세스가 살아 있지 않다")
    try:
        heartbeat = COMMON._exact(_read(paths["heartbeat"]), HEARTBEAT_KEYS, "lease heartbeat")
    except (OSError, ValueError, TypeError) as error:
        raise LeaseError("lease heartbeat가 없다") from error
    age = (current - _utc(heartbeat["updated_at_utc"], "heartbeat")).total_seconds()
    if (heartbeat["schema"] != HEARTBEAT_SCHEMA or heartbeat["lease_id"] != snapshot["lease_id"]
            or heartbeat["resource_id"] != resource_id or heartbeat["owner"] != owner
            or heartbeat["active"] is not True or age > HEARTBEAT_MAX_AGE_S or age < -CLOCK_SKEW_S):
        raise LeaseError("lease heartbeat가 활성 상태가 아니다")
    proof = _lock_held(paths["lock"], snapshot["lock_identity"], owner["pid"])
    return {"lease_id": snapshot["lease_id"], "resource_id": resource_id, "owner": owner,
            "lock_proof": proof, "live_lock_proof": proof == POSIX_LOCK_PROOF,
            "heartbeat_sequence": heartbeat["sequence"], "reference": dict(ref)}


def _registrations(
    value: Any, checkpoints: Sequence[Mapping[str, Any]], sources: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    ids = {row["logical_id"] for row in checkpoints}
    if not isinstance(value, dict) or set(value) != ids:
        raise ValueError("playback registration이 checkpoint catalog와 1:1이 아니다")
    hidden = sources.get("bench/hidden_faults.py")
    for row in checkpoints:
        registered = value[row["logical_id"]]
        if not isinstance(registered, dict) or registered.get("fault_id") not in FAULT_IDS:
            raise ValueError("playback registration fault가 NONE·F2 밖이다")
        expected = PLAYBACK.registration(row["logical_id"], row["sha256"], registered["fault_id"])
        if (registered != expected or registered["hidden_faults_sha256_lf"] != hidden
                or expected["interpretation_scope"] != INTERPRETATION["playback_scope"]):
            raise ValueError("playback registration이 checkpoint·hidden_faults source와 다르다")
    return value


def _pairs(value: Any, registrations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("F2·정상 pair가 비었다")
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    result = []
    for raw in value:
        row = COMMON._exact(raw, PAIR_KEYS, "pair")
        pair_id, seed = row["pair_id"], row["seed"]
        if not isinstance(pair_id, str) or COMMON.SLUG.fullmatch(pair_id) is None or pair_id in seen_pairs:
            raise ValueError("pair_id가 잘못됐거나 중복됐다")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("pair seed는 0 이상의 정수여야 한다")
        case, reference = row["case"], row["reference"]
        if case not in registrations or reference not in registrations or case == reference:
            raise ValueError("pair checkpoint가 catalog 밖이거나 같다")
        if registrations[case]["fault_id"] != "F2" or registrations[reference]["fault_id"] != "NONE":
            raise ValueError("pair는 F2 case 하나와 NONE reference 하나여야 한다")
        if {case, reference} & seen_ids:
            raise ValueError("checkpoint가 두 pair에 들어갔다")
        seen_pairs.add(pair_id)
        seen_ids |= {case, reference}
        result.append(dict(row))
    if seen_ids != set(registrations):
        raise ValueError("pair에 들어가지 않은 checkpoint가 있다")
    return result


def _sidecar(value: Any) -> dict[str, Any]:
    row = COMMON._exact(value, {"error_policy", "required", "source_files"}, "sidecar")
    if (row["error_policy"] not in SIDECAR.ERROR_POLICIES or row["required"] is not True
            or row["source_files"] != list(SIDECAR.SOURCE_FILES)):
        raise ValueError("sidecar 등록이 필수·정책·source 계약과 다르다")
    return row


def _runtime(value: Any) -> dict[str, str]:
    runtime = COMMON._source_map(value)
    if not REQUIRED_RUNTIME_FILES <= runtime.keys():
        raise ValueError("Isaac Lab runtime pin에 DCMotor·Go2 actuator 설정이 없다")
    return runtime


def _check_body(
    protocol: Mapping[str, Any], *, now: datetime, require_current: bool
) -> dict[str, Any]:
    """git 밖의 구조·의미 계약. build와 validate가 함께 쓴다."""
    if (protocol["schema"] != PROTOCOL_SCHEMA
            or protocol["registered_before_measurement"] is not True
            or not isinstance(protocol["protocol_id"], str)
            or COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
            or protocol["gpu_validation_status"] != "pending_live_gpu_execution"):
        raise ValueError("agentic play protocol 기본 계약이 다르다")
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("agentic play source closure가 부족하다")
    runtime = _runtime(protocol["runtime_source_sha256_lf"])
    target = _target(protocol["recording_target"], "recording target")
    if "l40s" not in target["instance_type"].lower():
        raise ValueError("recording target이 L40S 계열이 아니다")
    resources = COMMON._exact(protocol["resources"], {
        "compute_quote_usd_per_h", "batch_budget_usd", "deadline_utc", "max_owned_instances", "cost_scope",
    }, "recording resources")
    if resources["max_owned_instances"] != 1 or resources["cost_scope"] != RESOURCE_SCOPE:
        raise ValueError("recording resource 상한이 다르다")
    quote = _positive_decimal(resources["compute_quote_usd_per_h"], "compute quote")
    budget = _positive_decimal(resources["batch_budget_usd"], "batch budget")
    required = (quote * Decimal(TECHNICAL_LIMITS["max_policies"])
                * Decimal(TECHNICAL_LIMITS["max_wall_s_per_policy"]) / Decimal(3600))
    if budget < required:
        raise ValueError("batch budget이 2 policy wall 상한보다 작다")
    registered = _utc(protocol["registered_at_utc"], "registered")
    deadline = _utc(resources["deadline_utc"], "deadline")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    current = now.astimezone(timezone.utc)  # noqa: UP017 - Isaac Python 3.10
    if registered > current or registered >= deadline or (require_current and current >= deadline):
        raise ValueError("protocol 등록·deadline이 유효하지 않다")
    checkpoints = RECORD_JOB._checkpoints(protocol["checkpoints"])
    registrations = _registrations(protocol["playback_registrations"], checkpoints, sources)
    pairs = _pairs(protocol["pairs"], registrations)
    _sidecar(protocol["sidecar"])
    profile = COMMON._exact(protocol["record_profile"], {"value", "sha256"}, "record profile")
    if profile["value"] != RECORD.fixed_profile() or profile["sha256"] != RECORD.digest(RECORD.fixed_profile()):
        raise ValueError("same Play record profile이 다르다")
    if protocol["limits"] != TECHNICAL_LIMITS:
        raise ValueError("625 frame/50 fps/2 policy 기술 상한이 다르다")
    if protocol["interpretation"] != INTERPRETATION:
        raise ValueError("F2·정상 영상 해석 범위가 등록값과 다르다")
    prereg = COMMON._exact(protocol["preregistration"], {"prediction", "outcomes", "fixed_flags"},
                           "preregistration")
    if (not isinstance(prereg["prediction"], str) or not prereg["prediction"].strip()
            or not isinstance(prereg["outcomes"], dict) or set(prereg["outcomes"]) != {
                "recording_complete", "capture_hold", "infrastructure_failure"}
            or prereg["fixed_flags"] != FIXED_FLAGS):
        raise ValueError("agentic play 사전등록이 다르다")
    return {"sources": sources, "runtime": runtime, "target": target, "pairs": pairs}


def build_protocol(
    *,
    protocol_id: str,
    registered_at_utc: str,
    source_commit: str,
    source_sha256_lf: Mapping[str, str],
    runtime_source_sha256_lf: Mapping[str, str],
    recording_target: Mapping[str, str],
    compute_quote_usd_per_h: str,
    batch_budget_usd: str,
    deadline_utc: str,
    checkpoints: Sequence[Mapping[str, Any]],
    faults: Mapping[str, str],
    pairs: Sequence[Mapping[str, Any]],
    sidecar_error_policy: str = SIDECAR.ABORT,
) -> dict[str, Any]:
    """측정 전 protocol 본문을 만든다. source commit 결합은 메인이 커밋 뒤 validate_protocol로 확인한다."""
    rows = [dict(row) for row in checkpoints]
    if set(faults) != {row.get("logical_id") for row in rows}:
        raise ValueError("fault 지정이 checkpoint catalog와 1:1이 아니다")
    value = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": protocol_id,
        "registered_before_measurement": True,
        "registered_at_utc": registered_at_utc,
        "source_commit": source_commit,
        "source_sha256_lf": dict(source_sha256_lf),
        "runtime_source_sha256_lf": dict(runtime_source_sha256_lf),
        "recording_target": dict(recording_target),
        "resources": {
            "compute_quote_usd_per_h": str(compute_quote_usd_per_h),
            "batch_budget_usd": str(batch_budget_usd),
            "deadline_utc": deadline_utc,
            "max_owned_instances": 1,
            "cost_scope": RESOURCE_SCOPE,
        },
        "checkpoints": rows,
        "playback_registrations": {
            row["logical_id"]: PLAYBACK.registration(row["logical_id"], row["sha256"], faults[row["logical_id"]])
            for row in rows
        },
        "pairs": [dict(row) for row in pairs],
        "sidecar": {"error_policy": sidecar_error_policy, "required": True,
                    "source_files": list(SIDECAR.SOURCE_FILES)},
        "record_profile": {"value": RECORD.fixed_profile(), "sha256": RECORD.digest(RECORD.fixed_profile())},
        "limits": dict(TECHNICAL_LIMITS),
        "interpretation": dict(INTERPRETATION),
        "preregistration": {
            "prediction": "등록한 F2·정상 checkpoint 쌍을 새 idle GPU에서 같은 Play 설정과 상태 sidecar로 재생한다.",
            "outcomes": {
                "recording_complete": "두 정책의 native 녹화·sidecar 봉인·trace 대조가 완료된다.",
                "capture_hold": "AABA·terminal·sidecar·playback 조건이 어긋나면 HOLD한다.",
                "infrastructure_failure": "자원·복원·실행 계층 실패를 분리 기록한다.",
            },
            "fixed_flags": dict(FIXED_FLAGS),
        },
        "gpu_validation_status": "pending_live_gpu_execution",
    }
    _check_body(value, now=_utc(registered_at_utc, "registered"), require_current=False)
    return value


def _verify_protocol_committed(repo: Path, path: Path) -> None:
    """측정 전 등록은 커밋된 protocol만 인정한다. repo 밖·미추적·커밋 뒤 변경은 거부한다."""
    try:
        relative = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
        head = COMMON._git(repo, "rev-parse", "HEAD").stdout.strip()
        COMMON._verify_source_blobs(repo, head, {relative: _sha(path, lf=True)})
    except Exception as error:  # 하위 실패 형태와 관계없이 미커밋으로 닫는다
        raise ValueError("agentic play protocol이 HEAD에 커밋된 파일이 아니다") from error


def validate_protocol(
    path: Path, *, repo: Path | None = None, now: datetime | None = None, require_current: bool = True
) -> dict[str, Any]:
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "agentic play protocol")
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "agentic play protocol")
    commit = protocol["source_commit"]
    if (not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None
            or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode):
        raise ValueError("agentic play source commit이 HEAD 조상이 아니다")
    checked = _check_body(protocol, now=now or datetime.now(timezone.utc),  # noqa: UP017
                          require_current=require_current)
    COMMON._verify_source_blobs(repo, commit, checked["sources"])
    _verify_protocol_committed(repo, path)
    protocol["_recording_target"] = checked["target"]
    protocol["_runtime_parent"] = {"runtime_source_sha256_lf": checked["runtime"]}
    return protocol


def validate_resource_proof(
    proof_path: Path,
    proof_sha: str,
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    now: datetime,
    require_fresh: bool,
) -> dict[str, Any]:
    """새 recording 자원 입장 원문. 파일이 없거나 다른 자원·만료·오래된 관찰이면 거부한다."""
    path = _regular(Path(proof_path).absolute(), "agentic play resource proof")
    if _sha(path) != COMMON._hash(proof_sha, "resource proof SHA"):
        raise ValueError("agentic play resource proof SHA가 다르다")
    proof = COMMON._exact(_read(path), RESOURCE_KEYS, "agentic play resource proof")
    target = _target(proof["recording_target"], "proof recording target")
    if (proof["schema"] != RESOURCE_SCHEMA or proof["status"] != "granted"
            or proof["recording_protocol_sha256_lf"] != protocol_sha
            or target != protocol["_recording_target"]):
        raise ValueError("resource proof가 등록 protocol·recording target과 다르다")
    observed = _utc(proof["observed_at_utc"], "resource observed")
    expires = _utc(proof["expires_at_utc"], "resource expires")
    deadline = _utc(protocol["resources"]["deadline_utc"], "protocol deadline")
    current = now.astimezone(timezone.utc)  # noqa: UP017
    age = (current - observed).total_seconds()
    if (observed > current or expires > deadline or observed >= expires
            or (require_fresh and (age > FRESH_CAPTURE_S or current >= expires))):
        raise ValueError("recording resource proof가 fresh하지 않다")
    capture = COMMON._exact(proof["capture"], REMOTE.CAPTURE_KEYS, "resource capture")
    if capture["resource_id"] != target["resource_id"]:
        raise ValueError("capture resource ID가 recording target과 다르다")
    REMOTE.verify_capture_proof(capture)
    provider = REMOTE.verify_provider_inventory(
        Path(capture["provider_inventory_path"]), capture["provider_inventory_sha256"], target["resource_id"],
    )
    if provider["name"] != target["resource_name"] or provider["instance_type"] != target["instance_type"]:
        raise ValueError("provider target 이름·instance type이 등록값과 다르다")
    # 원문 30초 기한(require_fresh)과 무관하게 lease 활성은 항상 확인한다.
    lease = verify_lease(proof["lease"], protocol=protocol, protocol_sha=protocol_sha, now=current)
    return {**proof, **capture, "_path": path, "_lease": lease}


def build_resource_proof(
    protocol_path: Path,
    *,
    provider_inventory_path: Path,
    nvidia_gpu_csv_path: Path,
    nvidia_process_csv_path: Path,
    observed_at_utc: str,
    expires_at_utc: str,
    lease_reference: Mapping[str, str],
    repo: Path = ROOT,
) -> dict[str, Any]:
    """활성 lease 안에서 새 VM의 provider·GPU·idle 원문으로 write 전 resource proof 본문을 만든다."""
    observed = _utc(observed_at_utc, "resource observed")
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = validate_protocol(protocol_path, repo=repo, now=observed)
    lease = verify_lease(lease_reference, protocol=protocol, protocol_sha=_sha(protocol_path, lf=True), now=observed)
    target = protocol["_recording_target"]
    capture = {
        "resource_id": target["resource_id"],
        "gpu_uuid": Path(nvidia_gpu_csv_path).read_text(encoding="utf-8").strip(),
        "provider_inventory_path": str(Path(provider_inventory_path).resolve(strict=True)),
        "provider_inventory_sha256": _sha(provider_inventory_path),
        "nvidia_gpu_csv_path": str(Path(nvidia_gpu_csv_path).resolve(strict=True)),
        "nvidia_gpu_csv_sha256": _sha(nvidia_gpu_csv_path),
        "nvidia_process_csv_path": str(Path(nvidia_process_csv_path).resolve(strict=True)),
        "nvidia_process_csv_sha256": _sha(nvidia_process_csv_path),
    }
    REMOTE.verify_capture_proof(capture)
    provider = REMOTE.verify_provider_inventory(
        Path(capture["provider_inventory_path"]), capture["provider_inventory_sha256"], target["resource_id"],
    )
    if provider["name"] != target["resource_name"] or provider["instance_type"] != target["instance_type"]:
        raise ValueError("provider target 이름·instance type이 등록값과 다르다")
    expires = _utc(expires_at_utc, "resource expires")
    if observed >= expires or expires > _utc(protocol["resources"]["deadline_utc"], "deadline"):
        raise ValueError("resource proof 만료가 protocol deadline 밖이다")
    return {
        "schema": RESOURCE_SCHEMA, "status": "granted",
        "observed_at_utc": observed_at_utc, "expires_at_utc": expires_at_utc,
        "recording_protocol_sha256_lf": _sha(protocol_path, lf=True),
        "recording_target": target, "capture": capture, "lease": dict(lease["reference"]),
    }


def sidecar_directory_for(stage: Path) -> Path:
    """job stage 옆(stage 밖) 고정 경로. native stage·output 검증과 겹치지 않는다."""
    stage = Path(stage).absolute()
    return stage.parent / f"{stage.name}{SIDECAR_SUFFIX}"


def _pair_for(protocol: Mapping[str, Any], identity: str) -> dict[str, Any]:
    rows = [row for row in protocol["pairs"] if identity in (row["case"], row["reference"])]
    if len(rows) != 1:
        raise ValueError("checkpoint pair가 유일하지 않다")
    return dict(rows[0])


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "agentic play job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "agentic play job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("agentic play job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    protocol_path = _regular(Path(ref["path"]), "protocol path")
    protocol_sha = COMMON._hash(ref["sha256_lf"], "protocol SHA")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("agentic play protocol SHA가 다르다")
    current = now or datetime.now(timezone.utc)  # noqa: UP017
    protocol = validate_protocol(protocol_path, repo=repo, now=current, require_current=require_fresh)
    if source != {"repo_root": str(repo), "commit": protocol["source_commit"],
                  "files_lf": protocol["source_sha256_lf"]}:
        raise ValueError("agentic play job source가 protocol과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("agentic play job same Play profile이 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint")
    identity = checkpoint["logical_id"]
    if identity not in protocol["playback_registrations"]:
        raise ValueError("checkpoint가 generic catalog 등록 밖이다")
    row, checkpoint_path = RECORD_JOB._checkpoint(protocol, identity, lab)
    if checkpoint != {"path": str(checkpoint_path), "sha256": row["sha256"], "logical_id": identity}:
        raise ValueError("복원 checkpoint 경로·SHA가 catalog와 다르다")
    job_id = f"{protocol['protocol_id']}-{identity}"
    if job["job_id"] != job_id:
        raise ValueError("agentic play job ID가 다르다")
    resource_ref = COMMON._exact(job["resource_proof"], {"path", "sha256"}, "resource ref")
    resource = validate_resource_proof(
        Path(resource_ref["path"]), resource_ref["sha256"], protocol=protocol,
        protocol_sha=protocol_sha, now=current, require_fresh=require_fresh,
    )
    if COMMON._exact(job["lease"], {"path", "sha256"}, "job lease") != resource["lease"]:
        raise LeaseError("job lease가 resource proof의 lease와 다르다")
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_runtime_parent"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("agentic play output 경계가 다르다")
    COMMON._check_ancestors(output.parent, "agentic play output")
    sidecar = COMMON._exact(job["sidecar"], SIDECAR_KEYS, "job sidecar")
    expected_sidecar = {
        "directory": str(sidecar_directory_for(job_path.parent)),
        "source_id": job_id,
        "error_policy": protocol["sidecar"]["error_policy"],
        "expected_frames": TECHNICAL_LIMITS["frames_per_policy"],
    }
    if sidecar != expected_sidecar:
        raise ValueError("job sidecar 경로·식별·정책이 protocol 규칙과 다르다")
    sidecar_directory = Path(sidecar["directory"])
    COMMON._check_ancestors(sidecar_directory.parent, "agentic play sidecar")
    present = output.exists() or output.is_symlink()
    if require_fresh:
        if present:
            raise FileExistsError(output)
        SIDECAR.check_destination(sidecar_directory, output, job_path.parent)
    elif present:
        output = COMMON._directory(output, "agentic play output")
    return {"job_path": job_path, "job_sha256": _sha(job_path), "job": job,
            "protocol": protocol, "protocol_path": protocol_path, "protocol_sha256": protocol_sha,
            "output_directory": output, "repo": repo, "resource": resource,
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": row["sha256"],
            "playback_registration": protocol["playback_registrations"][identity],
            "pair": _pair_for(protocol, identity), "sidecar_directory": sidecar_directory,
            "lease": resource["_lease"]}


def create_job(
    protocol_path: Path, resource_path: Path, resource_sha: str, isaaclab: Path,
    python: Path, stage: Path, checkpoint_logical_id: str, *, repo: Path = ROOT,
) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path, repo=repo)
    if checkpoint_logical_id not in protocol["playback_registrations"]:
        raise ValueError("checkpoint가 generic catalog 등록값이 아니다")
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(protocol, checkpoint_logical_id, lab)
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "agentic play stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    job_id = f"{protocol['protocol_id']}-{checkpoint_logical_id}"
    lease = _read(_regular(Path(resource_path).absolute(), "agentic play resource proof")).get("lease")
    stage.mkdir(parents=True)
    job_path = stage / "job.json"
    write_new(job_path, {
        "schema": JOB_SCHEMA, "mode": MODE, "job_id": job_id,
        "source": {"repo_root": str(Path(repo).resolve()), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(Path(resource_path).absolute()), "sha256": resource_sha},
        "isaaclab": str(lab), "python": str(Path(python).absolute()),
        "output_directory": str(stage / "output"), "profile": RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": row["sha256"], "logical_id": row["logical_id"]},
        "sidecar": {
            "directory": str(sidecar_directory_for(stage)), "source_id": job_id,
            "error_policy": protocol["sidecar"]["error_policy"],
            "expected_frames": TECHNICAL_LIMITS["frames_per_policy"],
        },
        "lease": lease,
    })
    validate_job(job_path)
    return job_path


def _relative_to_lab(path: str, isaaclab_root: str) -> str:
    windows = ":" in isaaclab_root or "\\" in isaaclab_root
    kind = PureWindowsPath if windows else PurePosixPath
    return kind(path).relative_to(kind(isaaclab_root)).as_posix()


def motor_catalog(motor_sources: Mapping[str, Any], *, isaaclab_root: str) -> dict[str, Any]:
    """audit07 motor_sources.json에서 catalog·fault·pair 초안을 만든다(checkpoint 파일은 읽지 않음)."""
    if motor_sources.get("schema") != "ri_agentic_motor_sources_audit_v1":
        raise ValueError("motor sources audit schema가 다르다")
    rows: list[dict[str, Any]] = []
    faults: dict[str, str] = {}
    pairs: list[dict[str, Any]] = []
    for candidate in motor_sources["candidates"]:
        if candidate.get("fault") != "F2" or candidate.get("robot") != "Go2":
            raise ValueError("motor candidate가 Go2 F2가 아니다")
        seed = candidate["seed"]
        ids = {"case": f"go2-s{seed}-f2", "reference": f"go2-s{seed}-none"}
        for role, fault in (("case", "F2"), ("reference", "NONE")):
            checkpoint = candidate["policies"][role]["checkpoint"]
            rows.append({
                "logical_id": ids[role],
                "isaaclab_relative": _relative_to_lab(checkpoint["path"], isaaclab_root),
                "sha256": checkpoint["sha256"],
            })
            faults[ids[role]] = fault
        pairs.append({"pair_id": f"go2-s{seed}", "seed": seed, **ids})
    RECORD_JOB._checkpoints(rows)
    return {"checkpoints": rows, "faults": faults, "pairs": pairs}


def batch_layout(protocol: Mapping[str, Any], pair_id: str, remote_root: str) -> dict[str, Any]:
    """메인이 원격 host에 만들 stage·job·output·sidecar 경로(POSIX). 한 batch는 pair 하나(2 policy)."""
    root = PurePosixPath(remote_root)
    if not root.is_absolute():
        raise ValueError("remote_root는 POSIX 절대 경로여야 한다")
    rows = [row for row in protocol["pairs"] if row["pair_id"] == pair_id]
    if len(rows) != 1:
        raise ValueError("pair_id가 등록되지 않았다")
    policies = []
    for role in ("case", "reference"):
        identity = rows[0][role]
        stage = root / str(protocol["protocol_id"]) / identity
        policies.append({
            "role": role,
            "logical_id": identity,
            "fault_id": protocol["playback_registrations"][identity]["fault_id"],
            "stage": str(stage),
            "job": str(stage / "job.json"),
            "output": str(stage / "output"),
            "sidecar": str(stage.parent / f"{stage.name}{SIDECAR_SUFFIX}"),
        })
    return {"pair_id": pair_id, "max_policies": TECHNICAL_LIMITS["max_policies"], "policies": policies}
