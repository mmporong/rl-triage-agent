"""기존 semantic 진단 실행에 callback whole-payload 지문을 결합한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import run_vision_semantic_fingerprint_base_v1 as BASE
import vision_semantic_fingerprint_sdk_v1 as SDK

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_NAME = "fingerprint-execution.json"
EXECUTION_SCHEMA = "vision_semantic_fingerprint_execution_v1"
REQUIRED_SOURCE_FILES = frozenset(
    {
        "bench/vision_callback_fingerprint_v1.py",
        "bench/vision_semantic_fingerprint_sdk_v1.py",
        "bench/run_vision_semantic_fingerprint_v1.py",
        "bench/vision_semantic_fingerprint_job_v1.py",
        "bench/run_vision_semantic_fingerprint_base_v1.py",
        "bench/run_vision_semantic_diagnostic_v1.py",
        "bench/vision_semantic_sdk_v1.py",
        "bench/vision_semantic_diagnostic_v1.py",
        "tests/test_vision_callback_fingerprint_v1.py",
        "tests/test_vision_semantic_fingerprint_runtime_v1.py",
    }
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
OBSERVATION_PHRASE_KEYS = {
    "fingerprint_equal",
    "fingerprint_unequal",
    "fingerprint_insufficient",
    "capture_not_approved",
}


def _source_contract(validated: Mapping[str, Any]) -> None:
    job = validated.get("job")
    if not isinstance(job, Mapping):
        raise TypeError("fingerprint job이 mapping이 아니다")
    source = job.get("source")
    files = source.get("files_lf") if isinstance(source, Mapping) else None
    if not isinstance(files, Mapping) or not REQUIRED_SOURCE_FILES.issubset(files):
        raise ValueError("fingerprint source closure가 새 실행 파일을 포함하지 않는다")
    for relative in REQUIRED_SOURCE_FILES:
        digest = files.get(relative)
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise ValueError("fingerprint source closure SHA-256이 잘못됐다")
    protocol = validated.get("protocol")
    preregistration = (
        protocol.get("preregistration") if isinstance(protocol, Mapping) else None
    )
    phrases = (
        preregistration.get("observation_phrases")
        if isinstance(preregistration, Mapping)
        else None
    )
    if not isinstance(phrases, Mapping) or not OBSERVATION_PHRASE_KEYS.issubset(phrases):
        raise ValueError("fingerprint 관찰 문구가 사전등록되지 않았다")
    selected_phrases = [phrases[key] for key in sorted(OBSERVATION_PHRASE_KEYS)]
    if (
        not all(isinstance(value, str) and value for value in selected_phrases)
        or len(set(selected_phrases)) != len(selected_phrases)
    ):
        raise ValueError("fingerprint 관찰 문구 tag가 유일하지 않다")
    if Path(BASE.__file__).resolve() != (
        ROOT / "bench/run_vision_semantic_fingerprint_base_v1.py"
    ):
        raise ValueError("fingerprint semantic parent entry 결합이 다르다")
    if Path(__file__).resolve() != ROOT / "bench/run_vision_semantic_fingerprint_v1.py":
        raise ValueError("fingerprint child entry 결합이 다르다")


class _ChildEntryAdapter:
    """기존 부모 명령에서 등록된 child entry 한 항목만 교체한다."""

    def __init__(self, launch: Callable[..., Any]) -> None:
        self._launch = launch
        self.actual_command: list[str] | None = None

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        expected = str(Path(BASE.__file__).resolve())
        replacement = str(Path(__file__).resolve())
        if (
            len(command) != 11
            or command[0] != command[5]
            or command[1:4] != ["-I", "-S", "-c"]
            or command[4] != BASE.GATE.OWNED_WORKER_GATE
            or command[6:10] != ["-B", expected, "--sdk-child", "--job"]
            or not Path(command[10]).is_absolute()
            or command.count(expected) != 1
        ):
            raise ValueError("기존 semantic child entry 명령 계약이 다르다")
        actual = list(command)
        actual[7] = replacement
        for index, value in enumerate(command):
            if index != 7 and actual[index] != value:
                raise AssertionError("fingerprint child adapter가 다른 인자를 바꿨다")
        self.actual_command = actual
        return self._launch(actual, **kwargs)


def _adapter_evidence(stage: Path, adapter: _ChildEntryAdapter) -> dict[str, Any]:
    launch_path = stage / "launch.json"
    launch = BASE._read(launch_path)
    registered = launch.get("command")
    actual = adapter.actual_command
    if not isinstance(registered, list) or not all(
        isinstance(value, str) for value in registered
    ):
        raise TypeError("기존 semantic launch command가 문자열 목록이 아니다")
    if not isinstance(actual, list) or len(actual) != len(registered):
        raise ValueError("fingerprint 실제 child command 증거가 없다")
    changed = [
        index
        for index, (before, after) in enumerate(zip(registered, actual, strict=True))
        if before != after
    ]
    if changed != [7]:
        raise ValueError("fingerprint child entry 외 인자가 바뀌었다")
    if (
        registered[7] != str(Path(BASE.__file__).resolve())
        or actual[7] != str(Path(__file__).resolve())
        or registered[8:] != actual[8:]
        or registered[:7] != actual[:7]
    ):
        raise ValueError("fingerprint child entry 경로 결합이 다르다")
    return {
        "base": "bench/run_vision_semantic_fingerprint_base_v1.py",
        "replacement": "bench/run_vision_semantic_fingerprint_v1.py",
        "changed_argument_index": 7,
        "only_entry_replaced": True,
        "base_launch": BASE._descriptor(launch_path),
    }


def _read_fingerprint_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    base = BASE.verify_result(validated)
    sidecar_path = output / SDK.SIDECAR_NAME
    sidecar = BASE._read(sidecar_path)
    SDK._validate_counterparts(sidecar, base["sdk"])
    if (
        sidecar.get("schema") != SDK.SIDECAR_SCHEMA
        or sidecar.get("job_sha256") != validated["job_sha256"]
        or sidecar.get("callback_limit") != SDK.MAX_RAW_CALLBACKS
        or sidecar.get("used_for_decision") is not False
        or sidecar.get("sealed_before_base_report") is not True
        or sidecar.get("no_capture_approved") is not True
        or sidecar.get("comparison") not in {"equal", "unequal", "insufficient"}
    ):
        raise ValueError("fingerprint sidecar 결합이 다르다")
    return {
        "base_result": BASE._descriptor(output / "result.json"),
        "fingerprint_sidecar": BASE._descriptor(sidecar_path),
        "comparison": sidecar["comparison"],
        "callbacks_observed": sidecar["callbacks_observed"],
        "valid_fingerprints": sum(
            row["fingerprint"] is not None for row in sidecar["callbacks"]
        ),
        "used_for_decision": False,
    }


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = BASE.OWNED._process_identity,
    cleanup: Callable[..., bool] = BASE.OWNED._safe_cleanup,
) -> dict[str, Any]:
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("실제 fingerprint GPU 실행은 소유 Linux process group에서 수행한다")
    started = time.monotonic()
    deadline = started + float(BASE.JOB.LIMITS["max_wall_s"])
    validated = BASE.JOB.validate_job(job_path)
    _source_contract(validated)
    stage = Path(validated["job_path"]).parent
    target = stage / EXECUTION_NAME
    if target.exists() or target.is_symlink():
        raise FileExistsError("fingerprint 실행 증거를 덮어쓰지 않는다")
    adapter = _ChildEntryAdapter(popen)
    base_receipt = BASE.run_job(
        job_path,
        popen=adapter,
        gpu_reader=gpu_reader,
        identify=identify,
        cleanup=cleanup,
    )
    result: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "job_sha256": validated["job_sha256"],
        "base_execution": None,
        "child_entry_adapter": None,
        "diagnostic": None,
        "partial_fingerprint_sidecar": None,
        "error_type": None,
        "error_message": None,
        "no_capture_approved": True,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
        "wall_s": None,
    }
    try:
        BASE._remaining(deadline)
        result["base_execution"] = BASE._descriptor(stage / "execution.json")
        result["child_entry_adapter"] = _adapter_evidence(stage, adapter)
        if base_receipt.get("completed") is not True:
            raise RuntimeError("기존 semantic 실행이 완료되지 않았다")
        result["diagnostic"] = _read_fingerprint_result(validated)
        BASE._remaining(deadline)
        result["completed"] = True
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        sidecar_path = Path(validated["output_directory"]) / SDK.SIDECAR_NAME
        if sidecar_path.is_file():
            result["partial_fingerprint_sidecar"] = BASE._descriptor(sidecar_path)
    result["wall_s"] = time.monotonic() - started
    if result["wall_s"] >= float(BASE.JOB.LIMITS["max_wall_s"]):
        result["completed"] = False
        result["error_type"] = "TimeoutError"
        result["error_message"] = "fingerprint 전체 시간 상한이 끝났다"
    BASE.JOB.write_new(target, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args(argv)
    if args.sdk_child:
        if args.job is None:
            parser.error("--sdk-child에는 --job이 필요하다")
        validated = BASE.JOB.validate_job(args.job)
        _source_contract(validated)
        return BASE.sdk_child(args.job, run_sdk=SDK.run_one_view)
    if args.job is None:
        if any(
            value is None
            for value in (
                args.protocol,
                args.resource_proof,
                args.resource_sha256,
                args.isaaclab,
                args.python,
                args.stage,
            )
        ):
            parser.error(
                "새 실행에는 protocol/resource/isaaclab/python/stage가 필요하다"
            )
        args.job = BASE.prepare_job(
            args.protocol,
            args.resource_proof,
            args.resource_sha256,
            args.isaaclab,
            args.python,
            args.stage,
        )
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["completed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
