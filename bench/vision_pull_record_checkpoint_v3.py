"""정책 녹화 전 AABA 원시 산출물을 독립 출력 경계에 봉인한다."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import vision_annotator_pull_checkpoint_v2 as V2

MODE = V2.MODE
VARIANT = V2.VARIANT
SCHEMA = "vision_pull_record_checkpoint_v3"


class PullRecordCheckpointFailure(RuntimeError):
    """AABA 최초 오류와 직렬화 보조 오류를 분리해 전달한다."""

    def __init__(
        self,
        report: dict[str, Any],
        primary: BaseException,
        *,
        serialization_error: BaseException | None = None,
    ) -> None:
        super().__init__(f"pull record AABA failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.envelope = report
        self.primary = primary
        self.serialization_error = serialization_error


def _json_problem(value: Any, path: str = "$") -> str | None:
    if isinstance(value, (np.ndarray, np.generic)):
        return path
    if isinstance(value, Mapping):
        for key, item in value.items():
            problem = _json_problem(item, f"{path}.{key}")
            if problem is not None:
                return problem
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            problem = _json_problem(item, f"{path}[{index}]")
            if problem is not None:
                return problem
    return None


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(value))
    problem = _json_problem(copied)
    if problem is not None:
        raise TypeError(f"AABA report에 봉인되지 않은 NumPy 값이 있다: {problem}")
    json.dumps(copied, ensure_ascii=False, allow_nan=False)
    return copied


def _failure_report(
    validated: Mapping[str, Any],
    primary: BaseException,
    serialization_error: BaseException | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "diagnostic_complete": False,
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "primary_error": {
            "error_type": f"{type(primary).__module__}.{type(primary).__qualname__}",
            "error_message": str(primary),
        },
        "serialization_error": (
            None
            if serialization_error is None
            else {
                "error_type": (
                    f"{type(serialization_error).__module__}."
                    f"{type(serialization_error).__qualname__}"
                ),
                "error_message": str(serialization_error),
            }
        ),
        "capture_approved": False,
        "used_for_decision": False,
    }


def _verify_checkpoint(validated: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    worker = report.get("worker")
    probe = worker.get("policy_probe") if isinstance(worker, Mapping) else None
    if (
        not isinstance(probe, Mapping)
        or probe.get("checkpoint_sha256") != validated.get("checkpoint_sha256")
        or probe.get("environment_steps") != 1
    ):
        raise ValueError("AABA report의 checkpoint/policy step 결합이 다르다")


def run_annotator_pull_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_worker: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """출력을 먼저 소유하고, 외부 봉인이 최초 오류를 덮지 못하게 한다."""
    output = Path(validated["output_directory"])
    job_path = Path(validated["job_path"])
    if output != job_path.parent / "aaba-output" or not output.is_absolute():
        raise ValueError("AABA output은 stage/aaba-output이어야 한다")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=False)

    emitted: dict[str, Any] | None = None

    def hold_report(value: Mapping[str, Any]) -> None:
        nonlocal emitted
        emitted = deepcopy(dict(value))

    selected = run_worker
    try:
        kwargs: dict[str, Any] = {
            "seal_report": hold_report,
            "dependencies": dependencies,
        }
        if selected is not None:
            kwargs["run_worker"] = selected
        result = V2.run_annotator_pull_diagnostic(validated, app, **kwargs)
        report = _json_safe(emitted or result)
        _verify_checkpoint(validated, report)
        seal_report(report)
        return report
    except V2.AnnotatorPullDiagnosticFailure as error:
        primary = error.primary
        serialization_error: BaseException | None = None
        try:
            report = _json_safe(emitted or error.report)
        except BaseException as problem:  # noqa: BLE001 - 최초 오류와 보조 봉인 오류를 분리한다.
            serialization_error = problem
            report = _failure_report(validated, primary, problem)
        try:
            seal_report(report)
        except BaseException as seal_error:  # noqa: BLE001 - 최초 worker 오류가 우선이다.
            serialization_error = serialization_error or seal_error
        raise PullRecordCheckpointFailure(
            report, primary, serialization_error=serialization_error
        ) from error
    except BaseException as error:
        primary = getattr(error, "primary", error)
        report = _failure_report(validated, primary, None)
        raise PullRecordCheckpointFailure(report, primary) from error
