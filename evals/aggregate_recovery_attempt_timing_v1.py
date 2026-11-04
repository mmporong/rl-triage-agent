"""여러 recovery controller 시도의 기록 시간을 중복 없이 합산한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evals import registered_recovery_timing_v1 as recovery_timing

SCHEMA = "recovery_attempt_timing_aggregate_v1"
CASES = tuple(f"case-{number:02d}" for number in range(5, 9))
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
MANIFEST_SCHEMA = "registered_recovery_comparison_result_v1_manifest"
ANCHOR_SCHEMA = "registered_recovery_comparison_result_v1_anchor"


class AttemptTimingError(ValueError):
    """시도 시간 원문의 결합·중복·범위 계약이 잘못됐다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttemptTimingError("JSON 원문을 읽을 수 없다") from error
    if not isinstance(value, dict):
        raise AttemptTimingError("JSON 원문 최상위가 object가 아니다")
    return value


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise AttemptTimingError("UTC timestamp가 문자열이 아니다")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AttemptTimingError("UTC timestamp 형식이 잘못됐다") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise AttemptTimingError("aware UTC timestamp가 아니다")
    return parsed


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttemptTimingError(f"{label}이 숫자가 아니다")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AttemptTimingError(f"{label}이 유효한 비음수 값이 아니다")
    return number


def _verify_prepared(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    folder = root.resolve(strict=True)
    if folder.is_symlink() or not folder.is_dir():
        raise AttemptTimingError("prepared root가 일반 디렉터리가 아니다")
    manifest_path = folder / "manifest.json"
    anchor_path = folder / "anchor.json"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, anchor_path)):
        raise AttemptTimingError("manifest 또는 anchor가 일반 파일이 아니다")
    manifest = _json(manifest_path)
    anchor = _json(anchor_path)
    manifest_sha = _sha256(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AttemptTimingError("manifest schema가 다르다")
    if anchor != {"schema": ANCHOR_SCHEMA, "manifest_sha256": manifest_sha}:
        raise AttemptTimingError("anchor가 manifest SHA와 다르다")
    plan_ref = manifest.get("plan")
    if not isinstance(plan_ref, Mapping) or set(plan_ref) != {"path", "sha256"}:
        raise AttemptTimingError("manifest plan reference 형식이 다르다")
    plan_path = Path(str(plan_ref["path"])).resolve(strict=True)
    if plan_path.is_symlink() or not plan_path.is_file() or _sha256(plan_path) != plan_ref["sha256"]:
        raise AttemptTimingError("manifest plan reference가 바뀌었다")
    timing = recovery_timing.read_timing(folder)
    if timing.get("schema") != recovery_timing.SUMMARY_SCHEMA:
        raise AttemptTimingError("timing summary schema가 다르다")
    return timing, {
        "manifest_sha256": manifest_sha,
        "anchor_sha256": _sha256(anchor_path),
        "plan_sha256": str(plan_ref["sha256"]),
    }


def aggregate(prepared: Sequence[Path]) -> dict[str, Any]:
    if not prepared:
        raise AttemptTimingError("prepared 입력이 없다")
    roots: set[Path] = set()
    source_shas: set[str] = set()
    anchor_shas: set[str] = set()
    transition_ids: set[str] = set()
    timing_ref_shas: set[str] = set()
    attempts: list[dict[str, Any]] = []
    arm_rows: dict[tuple[str, str], dict[str, Any]] = {
        (case, method): {
            "case_id": case,
            "method": method,
            "transition_count": 0,
            "unknown_transition_count": 0,
            "known_active_command_wall_s": 0.0,
        }
        for case in CASES
        for method in METHODS
    }
    first_begin: datetime | None = None
    last_end: datetime | None = None
    first_known_begin: datetime | None = None
    last_known_end: datetime | None = None
    total_transitions = 0
    total_unknown = 0
    total_known_active = 0.0

    for index, candidate in enumerate(prepared, 1):
        root = candidate.resolve(strict=True)
        if root in roots:
            raise AttemptTimingError("같은 prepared root가 중복됐다")
        roots.add(root)
        timing, binding = _verify_prepared(root)
        if binding["manifest_sha256"] in source_shas:
            raise AttemptTimingError("같은 manifest source가 중복됐다")
        if binding["anchor_sha256"] in anchor_shas:
            raise AttemptTimingError("같은 anchor source가 중복됐다")
        source_shas.add(binding["manifest_sha256"])
        anchor_shas.add(binding["anchor_sha256"])
        transition_count = timing.get("transition_count")
        unknown_count = timing.get("unknown_transition_count")
        transitions = timing.get("transitions")
        if type(transition_count) is not int or type(unknown_count) is not int or not isinstance(
            transitions, list
        ):
            raise AttemptTimingError("timing summary count 형식이 다르다")
        if transition_count != len(transitions):
            raise AttemptTimingError("timing transition count가 행 수와 다르다")
        total_transitions += transition_count
        total_unknown += unknown_count
        attempt_active = timing.get("known_active_command_wall_s")
        if attempt_active is not None:
            total_known_active += _finite(attempt_active, "known active command wall")

        for row in transitions:
            if not isinstance(row, Mapping):
                raise AttemptTimingError("timing transition 행이 object가 아니다")
            case_id, method = row.get("case_id"), row.get("method")
            if not isinstance(case_id, str) or not isinstance(method, str):
                raise AttemptTimingError("timing transition case/method 형식이 다르다")
            key = (case_id, method)
            if key not in arm_rows:
                raise AttemptTimingError(f"timing transition이 16-arm 범위 밖이다: {key}")
            transition_id = row.get("transition_id")
            if not isinstance(transition_id, str) or transition_id in transition_ids:
                raise AttemptTimingError("transition ID가 없거나 중복됐다")
            transition_ids.add(transition_id)
            sequence = row.get("sequence")
            if type(sequence) is not int or sequence < 1:
                raise AttemptTimingError("transition sequence가 잘못됐다")
            begin_path = root / "timing" / "transitions" / f"{sequence:06d}-begin.json"
            begin = _json(begin_path)
            begin_ref = row.get("begin")
            if (
                not isinstance(begin_ref, Mapping)
                or begin_ref.get("sha256") != _sha256(begin_path)
                or begin_ref["sha256"] in timing_ref_shas
            ):
                raise AttemptTimingError("begin reference가 바뀌었거나 중복됐다")
            timing_ref_shas.add(str(begin_ref["sha256"]))
            started = _utc(begin.get("started_utc"))
            first_begin = started if first_begin is None else min(first_begin, started)
            arm = arm_rows[key]
            arm["transition_count"] += 1
            active = row.get("active_wall_s")
            if active is None:
                arm["unknown_transition_count"] += 1
            else:
                active_value = _finite(active, "transition active wall")
                arm["known_active_command_wall_s"] += active_value

            end_ref = row.get("end")
            if end_ref is None:
                continue
            end_path = root / "timing" / "transitions" / f"{sequence:06d}-end.json"
            end = _json(end_path)
            if (
                not isinstance(end_ref, Mapping)
                or end_ref.get("sha256") != _sha256(end_path)
                or end_ref["sha256"] in timing_ref_shas
            ):
                raise AttemptTimingError("end reference가 바뀌었거나 중복됐다")
            timing_ref_shas.add(str(end_ref["sha256"]))
            ended = _utc(end.get("ended_utc"))
            last_end = ended if last_end is None else max(last_end, ended)
            if active is not None:
                first_known_begin = (
                    started if first_known_begin is None else min(first_known_begin, started)
                )
                last_known_end = ended if last_known_end is None else max(last_known_end, ended)

        attempts.append(
            {
                "logical_id": f"attempt-{index:02d}",
                **binding,
                "transition_count": transition_count,
                "unknown_transition_count": unknown_count,
                "known_active_command_wall_s": timing.get("known_active_command_wall_s"),
                "active_command_wall_s": timing.get("active_command_wall_s"),
                "known_window_span_s": timing.get("known_window_span_s"),
                "window_span_s": timing.get("window_span_s"),
            }
        )

    per_arm: list[dict[str, Any]] = []
    for case in CASES:
        for method in METHODS:
            row = arm_rows[(case, method)]
            observed = row["transition_count"] > 0
            unknown = row["unknown_transition_count"]
            per_arm.append(
                {
                    **row,
                    "known_active_command_wall_s": (
                        row["known_active_command_wall_s"] if observed else 0.0
                    ),
                    "active_command_wall_s": (
                        0.0
                        if not observed
                        else row["known_active_command_wall_s"]
                        if unknown == 0
                        else None
                    ),
                    "observation_status": (
                        "not_observed" if not observed else "incomplete" if unknown else "observed"
                    ),
                }
            )
    known_span = (
        (last_known_end - first_known_begin).total_seconds()
        if first_known_begin is not None and last_known_end is not None
        else None
    )
    complete_span = (
        (last_end - first_begin).total_seconds()
        if total_unknown == 0 and first_begin is not None and last_end is not None
        else None
    )
    return {
        "schema": SCHEMA,
        "status": "complete" if total_unknown == 0 else "incomplete",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "transition_count": total_transitions,
        "unknown_transition_count": total_unknown,
        "known_active_command_wall_s": total_known_active if total_transitions else None,
        "active_command_wall_s": total_known_active if total_transitions and total_unknown == 0 else None,
        "known_completed_transition_span_s": known_span,
        "all_attempt_wall_span_s": complete_span,
        "all_attempt_wall_span_scope": (
            "utc_first_transition_begin_to_last_transition_end_including_inter_attempt_gaps"
        ),
        "all_attempt_wall_span_is_algorithm_cost": False,
        "all_attempt_wall_span_is_gpu_owned_time": False,
        "per_arm": per_arm,
        "unknown_values_are_not_zero": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.prepared)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
