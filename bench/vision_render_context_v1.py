"""Isaac 렌더러 실행의 GPU·Torch 컨텍스트를 write-once 원문으로 결합한다."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "vision_render_runtime_context_v1"
FINGERPRINT_SCHEMA = "vision_render_runtime_fingerprint_v1"
JOB_SCHEMA = "vision_play_record_job_v1"
JOB_METHOD = "same_play_raw_rgb_state_lockstep_v1"
JOB_MODES = frozenset({"record", "render_sync_smoke"})
WRITER_RELATIVE = "bench/vision_render_context_v1.py"
GPU_COMMAND = [
    "nvidia-smi",
    "--query-gpu=uuid,name,driver_version",
    "--format=csv,noheader,nounits",
]
TORCH_SCRIPT = (
    "import json\n"
    "import torch\n"
    "capability=torch.cuda.get_device_capability(0)\n"
    "print(json.dumps({"
    "'version':torch.__version__,"
    "'built_cuda_version':torch.version.cuda,"
    "'device_name':torch.cuda.get_device_name(0),"
    "'compute_capability':[capability[0],capability[1]],"
    "'device_count':torch.cuda.device_count(),"
    "'cuda_available':torch.cuda.is_available()"
    "},sort_keys=True,separators=(',',':')))\n"
)
GPU_FILE = "gpu.csv"
TORCH_FILE = "torch.json"
RECEIPT_FILE = "context.receipt.json"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CUDA_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}\Z")
PATH_TEXT = re.compile(r"[A-Za-z0-9_.@+/-]+\Z")
PACKAGE_KEYS = frozenset({
    "isaacsim", "isaaclab", "isaaclab_rl", "torch", "rsl-rl-lib", "gymnasium",
})
SNAPSHOT_KEYS = frozenset({
    "lab_head", "lab_tracked_clean", "source_anchor", "protocol", "environment",
    "selected_interpreter", "source_sha256_lf", "runtime_source_sha256_lf",
    "python_package_source_sha256_lf",
})
ENVIRONMENT_KEYS = frozenset({
    "python", "isaaclab_sha", *PACKAGE_KEYS, "cuda_available", "cuda_device_count",
})
INTERPRETER_KEYS = frozenset({
    "launcher", "python", "executable", "prefix", "base_prefix", "packages",
})
RAW_KEYS = frozenset({"argv", "stdout_file", "stdout_sha256", "exit_code", "stderr"})
RECEIPT_KEYS = frozenset({
    "schema", "job_sha256", "source_protocol_sha256_lf", "source_snapshot_sha256",
    "source_snapshot", "captured_at_utc", "finished_at_utc", "gpu_raw", "torch_raw",
})
FRACTIONAL_STAMP = re.compile(
    r"(?P<head>^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?P<fraction>\.\d+)?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})\Z"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Brev Python 3.10


def _utc_stamp(value: Any, anchor: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{anchor}는 UTC ISO timestamp여야 한다")
    match = FRACTIONAL_STAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"{anchor}는 UTC ISO timestamp여야 한다")
    fraction = match.group("fraction") or ""
    if len(fraction) > 7:
        fraction = fraction[:7]
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    try:
        stamp = datetime.fromisoformat(match.group("head") + fraction + zone)
    except ValueError as error:
        raise ValueError(f"{anchor} UTC ISO timestamp가 잘못됐다") from error
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError(f"{anchor}는 UTC timezone이어야 한다")
    return stamp.astimezone(timezone.utc)  # noqa: UP017 - Brev Python 3.10


def _sha256(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _read_object(path: Path, anchor: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{anchor}에 비유한 JSON 수 {value}가 있다")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{anchor}에 중복 JSON key {key}가 있다")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{anchor} JSON이 잘못됐다") from error
    if not isinstance(value, dict):
        raise TypeError(f"{anchor}는 JSON object여야 한다")
    return value


def _reject_symlink_ancestors(path: Path, anchor: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{anchor} 경로에 symlink가 있다")
        if current.parent == current:
            return
        current = current.parent


def _canonical_regular(path: Path, anchor: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
    _reject_symlink_ancestors(path, anchor)
    if path.resolve() != path:
        raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"{anchor}는 단일 link regular file이어야 한다")
    return path


def _canonical_directory(path: Path, anchor: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
    _reject_symlink_ancestors(path, anchor)
    if path.resolve() != path:
        raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
    if not path.is_dir():
        raise ValueError(f"{anchor}는 directory여야 한다")
    return path


def _portable_absolute_path(value: Any, anchor: str) -> PurePosixPath | PureWindowsPath:
    text = _nonempty(value, anchor)
    posix = PurePosixPath(text)
    if posix.is_absolute() and posix.as_posix() == text and ".." not in posix.parts:
        return posix
    windows = PureWindowsPath(text)
    if windows.is_absolute() and str(windows) == text and ".." not in windows.parts:
        return windows
    raise ValueError(f"{anchor}가 canonical absolute path가 아니다")


def _selected_python_file(path: Path, anchor: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.absolute() != path:
        raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
    _reject_symlink_ancestors(path.parent, anchor)
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"{anchor}는 단일 link regular file 또는 그 symlink여야 한다")
    return path


def _exact(value: Any, keys: frozenset[str] | set[str], anchor: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{anchor} 필드가 계약과 다르다")
    return value


def _hash(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{anchor} SHA-256이 잘못됐다")
    return value


def _nonempty(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{anchor} 문자열이 비었거나 정규형이 아니다")
    if value.casefold() in {"unknown", "n/a", "none"}:
        raise ValueError(f"{anchor}가 미확정 값이다")
    return value


def _relative_hashes(value: Any, anchor: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{anchor}가 비었다")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if not isinstance(relative, str) or PATH_TEXT.fullmatch(relative) is None:
            raise ValueError(f"{anchor} 상대경로가 잘못됐다")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative != path.as_posix():
            raise ValueError(f"{anchor} 상대경로가 정규형이 아니다")
        result[relative] = _hash(digest, f"{anchor}[{relative}]")
    return result


def _validate_snapshot(
    value: Mapping[str, Any], python: Path | None = None, *, portable: bool,
) -> dict[str, Any]:
    snapshot = _exact(dict(value), SNAPSHOT_KEYS, "source snapshot")
    if HEX40.fullmatch(str(snapshot["lab_head"])) is None:
        raise ValueError("source snapshot lab_head가 잘못됐다")
    if snapshot["lab_tracked_clean"] is not True:
        raise ValueError("source snapshot tracked cleanliness가 true가 아니다")
    if HEX40.fullmatch(str(snapshot["source_anchor"])) is None:
        raise ValueError("source snapshot source_anchor가 잘못됐다")

    protocol = _exact(snapshot["protocol"], {"path", "sha256_lf"}, "source snapshot protocol")
    protocol_path = protocol["path"]
    if (
        not isinstance(protocol_path, str)
        or PATH_TEXT.fullmatch(protocol_path) is None
        or Path(protocol_path).is_absolute()
        or ".." in Path(protocol_path).parts
        or protocol_path != Path(protocol_path).as_posix()
    ):
        raise ValueError("source snapshot protocol path가 잘못됐다")
    _hash(protocol["sha256_lf"], "source snapshot protocol")

    environment = _exact(snapshot["environment"], ENVIRONMENT_KEYS, "source snapshot environment")
    for key in ("python", *sorted(PACKAGE_KEYS)):
        _nonempty(environment[key], f"source snapshot environment.{key}")
    if environment["isaaclab_sha"] != snapshot["lab_head"]:
        raise ValueError("source snapshot Isaac Lab SHA가 lab_head와 다르다")
    if environment["cuda_available"] is not True or environment["cuda_device_count"] != 1:
        raise ValueError("source snapshot에는 CUDA GPU 한 개가 필요하다")

    selected = _exact(
        snapshot["selected_interpreter"], INTERPRETER_KEYS, "source snapshot selected_interpreter",
    )
    packages = _exact(selected["packages"], PACKAGE_KEYS, "selected_interpreter packages")
    if selected["python"] != environment["python"]:
        raise ValueError("selected_interpreter Python 버전이 environment와 다르다")
    for name in PACKAGE_KEYS:
        if packages[name] != environment[name]:
            raise ValueError(f"selected_interpreter package가 environment와 다르다: {name}")
    launcher_text = _portable_absolute_path(selected["launcher"], "selected_interpreter launcher")
    executable_text = _portable_absolute_path(selected["executable"], "selected_interpreter executable")
    prefix = _portable_absolute_path(selected["prefix"], "selected_interpreter prefix")
    base_prefix = _portable_absolute_path(selected["base_prefix"], "selected_interpreter base_prefix")
    if type(launcher_text) is not type(prefix) or type(prefix) is not type(base_prefix):
        raise ValueError("selected_interpreter path 형식이 서로 다르다")
    if launcher_text != executable_text or prefix == base_prefix or launcher_text.parent.parent != prefix:
        raise ValueError("selected_interpreter 실행 경로 결합이 잘못됐다")
    if not portable:
        if python is None:
            raise ValueError("capture source snapshot 검증에는 선택 Python이 필요하다")
        launcher = _selected_python_file(Path(str(launcher_text)), "selected_interpreter launcher")
        if _selected_python_file(python, "선택 Python") != launcher:
            raise ValueError("capture Python이 source snapshot launcher와 다르다")

    source_hashes = _relative_hashes(snapshot["source_sha256_lf"], "source_sha256_lf")
    runtime_hashes = _relative_hashes(
        snapshot["runtime_source_sha256_lf"], "runtime_source_sha256_lf",
    )
    package_hashes = _relative_hashes(
        snapshot["python_package_source_sha256_lf"], "python_package_source_sha256_lf",
    )
    if WRITER_RELATIVE not in source_hashes:
        raise ValueError("render context writer가 source hash map에 없다")
    return {
        **snapshot,
        "protocol": dict(protocol),
        "environment": dict(environment),
        "selected_interpreter": {**selected, "packages": dict(packages)},
        "source_sha256_lf": source_hashes,
        "runtime_source_sha256_lf": runtime_hashes,
        "python_package_source_sha256_lf": package_hashes,
    }


def _validate_job(
    job: Mapping[str, Any], source_snapshot: Mapping[str, Any], *, verify_files: bool,
) -> dict[str, Any]:
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("method") != JOB_METHOD
        or job.get("mode") not in JOB_MODES
    ):
        raise ValueError("record/smoke job identity가 계약과 다르다")
    source = _exact(job.get("source"), {"repo_root", "commit", "files_lf"}, "job source")
    resource = _exact(job.get("resource"), {"resource_id", "gpu_uuid"}, "job resource")
    repo_text = _portable_absolute_path(source["repo_root"], "job source repo")
    if source["commit"] != source_snapshot["source_anchor"]:
        raise ValueError("job source commit이 source snapshot과 다르다")
    files = _relative_hashes(source["files_lf"], "job source files_lf")
    source_hashes = source_snapshot["source_sha256_lf"]
    if any(source_hashes.get(relative) != digest for relative, digest in files.items()):
        raise ValueError("job source files가 source snapshot에 포함되지 않는다")
    writer_sha = _sha256(Path(__file__).resolve(), lf=True)
    if files.get(WRITER_RELATIVE) != writer_sha or source_hashes[WRITER_RELATIVE] != writer_sha:
        raise ValueError("job render context writer LF SHA가 현재 writer와 다르다")
    if verify_files:
        repo = _canonical_directory(Path(str(repo_text)), "job source repo")
        protocol_path = _canonical_regular(
            repo / source_snapshot["protocol"]["path"], "source protocol",
        )
        if _sha256(protocol_path, lf=True) != source_snapshot["protocol"]["sha256_lf"]:
            raise ValueError("source protocol LF SHA가 source snapshot과 다르다")
        for relative, digest in files.items():
            path = _canonical_regular(repo / relative, f"job source file {relative}")
            if _sha256(path, lf=True) != digest:
                raise ValueError(f"job source file LF SHA가 다르다: {relative}")
    _nonempty(resource["resource_id"], "job resource_id")
    gpu_uuid = _nonempty(resource["gpu_uuid"], "job gpu_uuid")
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("job gpu_uuid가 NVIDIA GPU UUID 형식이 아니다")
    return {**job, "source": {**source, "files_lf": files}, "resource": dict(resource)}


def _write_bytes_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_new(path, _canonical(dict(value)) + b"\n")


def _call(
    runner: Callable[..., subprocess.CompletedProcess[str]], argv: list[str], anchor: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            argv, capture_output=True, text=True, encoding="utf-8", errors="strict",
            check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{anchor} 실행에 실패했다") from error
    if type(result.returncode) is not int or not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
        raise TypeError(f"{anchor} subprocess 결과 형식이 잘못됐다")
    return result


def _parse_gpu_csv(text: str, expected_uuid: str) -> dict[str, str]:
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError("GPU context CSV가 잘못됐다") from error
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("GPU context CSV는 한 GPU의 세 열이어야 한다")
    fields = [field.strip() for field in rows[0]]
    if any(not field or field.casefold() in {"unknown", "n/a", "none"} for field in fields):
        raise ValueError("GPU context CSV에 비어 있거나 미확정인 값이 있다")
    canonical = f"{fields[0]}, {fields[1]}, {fields[2]}\n"
    if text != canonical:
        raise ValueError("GPU context CSV가 canonical 형식이 아니다")
    if fields[0] != expected_uuid:
        raise ValueError("GPU context UUID가 job resource와 다르다")
    return {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}


def _normalized_gpu_name(value: Any, anchor: str) -> str:
    name = _nonempty(value, anchor)
    collapsed = " ".join(name.split()).removeprefix("NVIDIA ")
    if not collapsed:
        raise ValueError(f"{anchor} 정규화 결과가 비었다")
    return collapsed


def _parse_torch_json(text: str, source_snapshot: Mapping[str, Any], gpu_name: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Torch context JSON이 잘못됐다") from error
    torch = _exact(
        value,
        {"version", "built_cuda_version", "device_name", "compute_capability",
         "device_count", "cuda_available"},
        "Torch context",
    )
    if torch["version"] != source_snapshot["environment"]["torch"]:
        raise ValueError("Torch runtime 버전이 source snapshot과 다르다")
    if not isinstance(torch["built_cuda_version"], str) or CUDA_VERSION.fullmatch(
        torch["built_cuda_version"],
    ) is None:
        raise ValueError("Torch built CUDA 버전이 잘못됐다")
    if torch["cuda_available"] is not True or type(torch["device_count"]) is not int or torch["device_count"] != 1:
        raise ValueError("Torch runtime에는 CUDA GPU 한 개가 보여야 한다")
    capability = torch["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
    ):
        raise ValueError("Torch compute capability가 잘못됐다")
    device_name = _nonempty(torch["device_name"], "Torch device name")
    normalized_device = _normalized_gpu_name(device_name, "Torch device name")
    normalized_gpu = _normalized_gpu_name(gpu_name, "GPU CSV name")
    if normalized_device != normalized_gpu:
        raise ValueError("Torch device와 nvidia-smi GPU 이름이 다르다")
    return {
        "version": torch["version"],
        "built_cuda_version": torch["built_cuda_version"],
        "device_name": device_name,
        "normalized_device_name": normalized_device,
        "compute_capability": list(capability),
        "device_count": 1,
        "cuda_available": True,
    }


def _raw_reference(
    value: Any, expected_argv: list[str], expected_file: str, context_dir: Path, anchor: str,
) -> tuple[dict[str, Any], str]:
    raw = _exact(value, RAW_KEYS, anchor)
    if (
        raw["argv"] != expected_argv
        or raw["stdout_file"] != expected_file
        or Path(str(raw["stdout_file"])).name != raw["stdout_file"]
        or raw["exit_code"] != 0
        or raw["stderr"] != ""
    ):
        raise ValueError(f"{anchor} 실행 계약이 다르다")
    expected_sha = _hash(raw["stdout_sha256"], f"{anchor} stdout")
    path = _canonical_regular(context_dir / expected_file, f"{anchor} stdout")
    if _sha256(path) != expected_sha:
        raise ValueError(f"{anchor} stdout SHA가 다르다")
    try:
        return raw, path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{anchor} stdout UTF-8이 잘못됐다") from error


def capture_context(
    job_path: Path,
    source_snapshot: dict[str, Any],
    python: Path,
    output_dir: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """GPU/Torch 원문을 새 폴더에 기록하고 검증된 context receipt를 반환한다."""
    job_path = _canonical_regular(Path(job_path), "job")
    python = _selected_python_file(Path(python), "선택 Python")
    snapshot = _validate_snapshot(source_snapshot, python, portable=False)
    job = _validate_job(_read_object(job_path, "job"), snapshot, verify_files=True)
    job_sha = _sha256(job_path)

    output_dir = Path(output_dir)
    if not output_dir.is_absolute() or output_dir.resolve() != output_dir:
        raise ValueError("context output 경로는 정규화된 절대경로여야 한다")
    _canonical_directory(output_dir.parent, "context output parent")
    output_dir.mkdir(exist_ok=False)
    _reject_symlink_ancestors(output_dir, "context output")

    captured_at = _utc_now()
    if captured_at.tzinfo is None or captured_at.utcoffset() != timedelta(0):
        raise ValueError("capture 시작 시각은 UTC timezone이어야 한다")
    gpu = _call(runner, GPU_COMMAND, "nvidia-smi context query")
    gpu_path = output_dir / GPU_FILE
    _write_bytes_new(gpu_path, gpu.stdout.encode("utf-8"))
    if gpu.returncode != 0 or gpu.stderr != "":
        raise ValueError("nvidia-smi context query가 깨끗하게 끝나지 않았다")
    gpu_value = _parse_gpu_csv(gpu.stdout, job["resource"]["gpu_uuid"])

    torch_command = [str(python), "-I", "-c", TORCH_SCRIPT]
    torch_result = _call(runner, torch_command, "Torch context query")
    torch_path = output_dir / TORCH_FILE
    _write_bytes_new(torch_path, torch_result.stdout.encode("utf-8"))
    if torch_result.returncode != 0 or torch_result.stderr != "":
        raise ValueError("Torch context query가 깨끗하게 끝나지 않았다")
    _parse_torch_json(torch_result.stdout, snapshot, gpu_value["name"])
    finished_at = _utc_now()
    if finished_at.tzinfo is None or finished_at.utcoffset() != timedelta(0) or finished_at < captured_at:
        raise ValueError("capture 종료 시각이 UTC이거나 시작 이후가 아니다")

    receipt = {
        "schema": SCHEMA,
        "job_sha256": job_sha,
        "source_protocol_sha256_lf": snapshot["protocol"]["sha256_lf"],
        "source_snapshot_sha256": _digest(snapshot),
        "source_snapshot": snapshot,
        "captured_at_utc": captured_at.astimezone(timezone.utc).isoformat(),  # noqa: UP017
        "finished_at_utc": finished_at.astimezone(timezone.utc).isoformat(),  # noqa: UP017
        "gpu_raw": {
            "argv": GPU_COMMAND,
            "stdout_file": GPU_FILE,
            "stdout_sha256": _sha256(gpu_path),
            "exit_code": gpu.returncode,
            "stderr": gpu.stderr,
        },
        "torch_raw": {
            "argv": torch_command,
            "stdout_file": TORCH_FILE,
            "stdout_sha256": _sha256(torch_path),
            "exit_code": torch_result.returncode,
            "stderr": torch_result.stderr,
        },
    }
    receipt_path = output_dir / RECEIPT_FILE
    _write_json_new(receipt_path, receipt)
    read_context(receipt_path, job, job_sha, snapshot)
    return receipt_path


def read_context(
    context_path: Path,
    job: dict[str, Any],
    job_sha256: str,
    source_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Context receipt와 두 원문을 재검증해 record/smoke 비교 fingerprint를 만든다."""
    context_path = _canonical_regular(Path(context_path), "render context receipt")
    if context_path.name != RECEIPT_FILE:
        raise ValueError("render context receipt 파일명이 다르다")
    snapshot = _validate_snapshot(source_snapshot, portable=True)
    checked_job = _validate_job(job, snapshot, verify_files=False)
    expected_job_sha = _hash(job_sha256, "job")
    receipt = _exact(_read_object(context_path, "render context receipt"), RECEIPT_KEYS, "render context receipt")
    if (
        receipt["schema"] != SCHEMA
        or receipt["job_sha256"] != expected_job_sha
        or receipt["source_protocol_sha256_lf"] != snapshot["protocol"]["sha256_lf"]
        or receipt["source_snapshot_sha256"] != _digest(snapshot)
        or _canonical(receipt["source_snapshot"]) != _canonical(snapshot)
    ):
        raise ValueError("render context receipt 결합이 다르다")
    captured_at = _utc_stamp(receipt["captured_at_utc"], "captured_at_utc")
    finished_at = _utc_stamp(receipt["finished_at_utc"], "finished_at_utc")
    if finished_at < captured_at:
        raise ValueError("render context 종료 시각이 시작 시각보다 빠르다")
    context_dir = context_path.parent
    gpu_raw, gpu_text = _raw_reference(
        receipt["gpu_raw"], GPU_COMMAND, GPU_FILE, context_dir, "GPU raw",
    )
    torch_command = [snapshot["selected_interpreter"]["launcher"], "-I", "-c", TORCH_SCRIPT]
    torch_raw, torch_text = _raw_reference(
        receipt["torch_raw"], torch_command, TORCH_FILE, context_dir, "Torch raw",
    )
    gpu = _parse_gpu_csv(gpu_text, checked_job["resource"]["gpu_uuid"])
    torch = _parse_torch_json(torch_text, snapshot, gpu["name"])
    return {
        "schema": FINGERPRINT_SCHEMA,
        "source_snapshot_sha256": receipt["source_snapshot_sha256"],
        "source_protocol_sha256_lf": receipt["source_protocol_sha256_lf"],
        "gpu": {
            **gpu,
            "normalized_name": _normalized_gpu_name(gpu["name"], "GPU CSV name"),
        },
        "torch": torch,
        "selected_interpreter": {
            "launcher": snapshot["selected_interpreter"]["launcher"],
            "python": snapshot["selected_interpreter"]["python"],
        },
        "raw_sha256": {
            "gpu": gpu_raw["stdout_sha256"],
            "torch": torch_raw["stdout_sha256"],
        },
    }
