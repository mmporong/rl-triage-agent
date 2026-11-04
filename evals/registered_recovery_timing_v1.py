"""Write-once transition timing evidence for registered recovery comparisons."""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from rl_triage import investigation_session as io

BEGIN_SCHEMA = "registered_recovery_transition_begin_v1"
END_SCHEMA = "registered_recovery_transition_end_v1"
SUMMARY_SCHEMA = "registered_recovery_timing_v1"
CLOCK_TOLERANCE_S = 2.0
_CASE = re.compile(r"case-[0-9]{2}\Z")
_METHOD = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TimingError(ValueError):
    """Timing evidence is malformed, changed, duplicated, or out of order."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise TimingError("aware UTC timestamp required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TimingError("UTC timestamp differs") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise TimingError("aware UTC timestamp required")
    return parsed


def _folder(folder: Path) -> Path:
    root = Path(folder).resolve()
    if root.is_symlink():
        raise TimingError("timing root cannot be a symlink")
    return root / "timing" / "transitions"


def _files(folder: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    root = _folder(folder)
    if not root.exists():
        return {}, {}
    begins, ends = {}, {}
    for path in root.iterdir():
        match = re.fullmatch(r"([0-9]{6})-(begin|end)\.json", path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise TimingError("unexpected timing artifact")
        sequence = int(match.group(1))
        target = begins if match.group(2) == "begin" else ends
        if sequence in target:
            raise TimingError("duplicate timing sequence")
        target[sequence] = path
    return begins, ends


@contextmanager
def transition(folder: Path, *, case_id: str, method: str) -> Iterator[dict]:
    """Record one controller transition; Python exceptions still receive an end receipt."""
    if _CASE.fullmatch(case_id) is None or _METHOD.fullmatch(method) is None:
        raise TimingError("case/method binding differs")
    begins, ends = _files(folder)
    if set(ends) - set(begins) or any(sequence not in ends for sequence in begins):
        raise TimingError("an earlier transition is incomplete")
    sequence = len(begins) + 1
    if set(begins) != set(range(1, sequence)):
        raise TimingError("timing sequence is not contiguous")
    started_utc, started_perf = _utc_now(), time.perf_counter_ns()
    transition_id = io.digest({"sequence": sequence, "case_id": case_id, "method": method,
                               "started_utc": started_utc, "started_perf_ns": started_perf})
    begin = {"schema": BEGIN_SCHEMA, "sequence": sequence, "case_id": case_id, "method": method,
             "transition_id": transition_id, "started_utc": started_utc,
             "started_perf_ns": started_perf}
    begin_path = _folder(folder) / f"{sequence:06d}-begin.json"
    io.write_new(begin_path, begin)
    try:
        yield {"sequence": sequence, "transition_id": transition_id,
               "begin_ref": io.reference(begin_path)}
    finally:
        end = {"schema": END_SCHEMA, "sequence": sequence, "case_id": case_id, "method": method,
               "transition_id": transition_id, "begin_sha256": io.file_sha(begin_path),
               "ended_utc": _utc_now(), "ended_perf_ns": time.perf_counter_ns()}
        io.write_new(_folder(folder) / f"{sequence:06d}-end.json", end)


def read_timing(folder: Path) -> dict:
    """Reverify timing evidence without creating or modifying files."""
    begins, ends = _files(folder)
    sequences = sorted(begins)
    if sequences != list(range(1, len(sequences) + 1)) or set(ends) - set(begins):
        raise TimingError("timing sequence is not contiguous")
    rows, private_refs = [], []
    known_active = 0.0
    known_count = 0
    unknown = 0
    first_start = None
    last_end = None
    seen_ids = set()
    per_arm: dict[str, dict] = {}
    for sequence in sequences:
        begin_path = begins[sequence]
        begin = io.read_ref(io.reference(begin_path))
        if (set(begin) != {"schema", "sequence", "case_id", "method", "transition_id",
                          "started_utc", "started_perf_ns"}
                or begin["schema"] != BEGIN_SCHEMA or begin["sequence"] != sequence
                or _CASE.fullmatch(str(begin["case_id"])) is None
                or _METHOD.fullmatch(str(begin["method"])) is None
                or _SHA.fullmatch(str(begin["transition_id"])) is None
                or type(begin["started_perf_ns"]) is not int or begin["started_perf_ns"] < 0):
            raise TimingError("transition begin binding differs")
        started = _parse_utc(begin["started_utc"])
        expected_id = io.digest({"sequence": sequence, "case_id": begin["case_id"],
                                 "method": begin["method"], "started_utc": begin["started_utc"],
                                 "started_perf_ns": begin["started_perf_ns"]})
        if begin["transition_id"] != expected_id:
            raise TimingError("transition identity differs from begin evidence")
        if begin["transition_id"] in seen_ids:
            raise TimingError("duplicate transition identity")
        seen_ids.add(begin["transition_id"])
        first_start = started if first_start is None else min(first_start, started)
        begin_ref = io.reference(begin_path)
        private_refs.append({"kind": "begin", **begin_ref})
        arm_key = f"{begin['case_id']}:{begin['method']}"
        aggregate = per_arm.setdefault(arm_key, {"case_id": begin["case_id"], "method": begin["method"],
                                                 "completed_transitions": 0,
                                                 "unknown_transitions": 0, "known_active_wall_s": 0.0})
        end_path = ends.get(sequence)
        if end_path is None:
            unknown += 1
            aggregate["unknown_transitions"] += 1
            rows.append({"sequence": sequence, "case_id": begin["case_id"], "method": begin["method"],
                         "transition_id": begin["transition_id"], "status": "unknown_crash",
                         "active_wall_s": None,
                         "begin": {"logical_id": f"transition:{sequence}:begin",
                                   "sha256": begin_ref["sha256"]}, "end": None})
            continue
        end = io.read_ref(io.reference(end_path))
        if (set(end) != {"schema", "sequence", "case_id", "method", "transition_id",
                        "begin_sha256", "ended_utc", "ended_perf_ns"}
                or end["schema"] != END_SCHEMA or end["sequence"] != sequence
                or end["case_id"] != begin["case_id"] or end["method"] != begin["method"]
                or end["transition_id"] != begin["transition_id"]
                or _SHA.fullmatch(str(end["begin_sha256"])) is None
                or end["begin_sha256"] != begin_ref["sha256"]
                or type(end["ended_perf_ns"]) is not int
                or end["ended_perf_ns"] < begin["started_perf_ns"]):
            raise TimingError("transition end binding differs")
        ended = _parse_utc(end["ended_utc"])
        if ended < started:
            raise TimingError("transition UTC order differs")
        last_end = ended if last_end is None else max(last_end, ended)
        perf_s = (end["ended_perf_ns"] - begin["started_perf_ns"]) / 1_000_000_000
        utc_s = (ended - started).total_seconds()
        duration = perf_s if abs(perf_s - utc_s) <= CLOCK_TOLERANCE_S else None
        if duration is None:
            unknown += 1
            aggregate["unknown_transitions"] += 1
        else:
            known_active += duration
            known_count += 1
            aggregate["completed_transitions"] += 1
            aggregate["known_active_wall_s"] += duration
        end_ref = io.reference(end_path)
        private_refs.append({"kind": "end", **end_ref})
        rows.append({"sequence": sequence, "case_id": begin["case_id"], "method": begin["method"],
                     "transition_id": begin["transition_id"],
                     "status": "complete" if duration is not None else "unknown_clock_mismatch",
                     "active_wall_s": duration,
                     "begin": {"logical_id": f"transition:{sequence}:begin", "sha256": begin_ref["sha256"]},
                     "end": {"logical_id": f"transition:{sequence}:end", "sha256": end_ref["sha256"]}})
    known_span = ((last_end - first_start).total_seconds()
                  if first_start is not None and last_end is not None else None)
    per_arm_rows = []
    for row in per_arm.values():
        total = row["completed_transitions"] + row["unknown_transitions"]
        known_arm = row["known_active_wall_s"] if row["completed_transitions"] else None
        per_arm_rows.append({**row, "known_active_wall_s": known_arm,
                             "active_wall_s": (known_arm if total and row["unknown_transitions"] == 0 else None)})
    return {"schema": SUMMARY_SCHEMA, "transition_count": len(sequences),
            "unknown_transition_count": unknown,
            "known_active_command_wall_s": known_active if known_count else None,
            "active_command_wall_s": known_active if sequences and unknown == 0 else None,
            "known_window_span_s": known_span,
            "window_span_s": known_span if sequences and unknown == 0 else None,
            "per_arm": per_arm_rows, "transitions": rows,
            "private_refs": private_refs}
