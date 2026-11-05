"""Preserve admission observations without changing the registered predicate."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import registered_desktop_gpu_retry_v1 as gpu

ROOT = Path(__file__).absolute().parents[1]
PRIVATE = ROOT / "bench" / "private"
FORBIDDEN = "ri_codex2_tokenfactory_live_160"
UTC = timezone.utc  # noqa: UP017 - Keep the SDK Python 3.10 import surface compatible.
_LOCK = threading.RLock()
_LOCAL = threading.local()


class AdmissionJournalError(RuntimeError):
    """The observation cannot be preserved under its registered boundary."""


def _folder(value: Path) -> Path:
    path = Path(value)
    parts = str(path).replace("\\", "/").split("/")
    if any(part.casefold() == FORBIDDEN for part in parts) or ".." in parts:
        raise AdmissionJournalError("journal path is outside the allowed boundary")
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(PRIVATE)
    except ValueError as error:
        raise AdmissionJournalError("journal requires a private run directory") from error
    if not relative.parts:
        raise AdmissionJournalError("journal requires a fresh child directory")
    current = ROOT
    for part in absolute.relative_to(ROOT).parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise AdmissionJournalError("journal path contains a link or reparse point")
    return absolute


def _write(path: Path, body: dict[str, Any]) -> None:
    raw = (json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)


def _chain(error: BaseException) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None and len(rows) < 8:
        rows.append({"error_type": type(current).__name__, "message": str(current)})
        current = current.__cause__
    return rows


@contextmanager
def admission_journal_scope(out_dir: Path) -> Iterator[Path]:
    """Wrap the existing query; retain result identity and original exceptions."""
    folder = _folder(out_dir)
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            if folder != _LOCAL.folder:
                raise AdmissionJournalError("nested journal directory differs")
            _LOCAL.depth += 1
            try:
                yield folder
            finally:
                _LOCAL.depth -= 1
            return
        original = gpu.query_registered_desktop_gpu_retry
        if getattr(original, "_admission_journal_wrapper", False):
            raise AdmissionJournalError("admission query is already journaled")
        folder.mkdir(parents=True, exist_ok=False)
        owner = threading.get_ident()
        sequence = 0
        source_sha = hashlib.sha256(Path(gpu.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

        def recorded() -> dict[str, Any]:
            nonlocal sequence
            if threading.get_ident() != owner:
                raise AdmissionJournalError("admission query was called from another thread")
            sequence += 1
            prefix = folder / f"{sequence:06d}"
            started, cpu = time.perf_counter(), time.process_time()
            intent = {"schema": "registered_gpu_admission_journal_v1", "sequence": sequence,
                      "started_utc": datetime.now(UTC).isoformat(),
                      "query_source_sha256_lf": source_sha,
                      "role": "observation_journal_not_execution_authority",
                      "SDK": 0, "API": 0}
            _write(prefix.with_suffix(".intent.json"), {**intent, "status": "intent"})
            try:
                result = original()
            except BaseException as error:
                try:
                    _write(prefix.with_suffix(".result.json"), {
                        **intent, "status": "raised", "exception_chain": _chain(error),
                        "error_receipt": getattr(error, "receipt", None),
                        "finished_utc": datetime.now(UTC).isoformat(),
                        "wall_s": time.perf_counter() - started,
                        "parent_process_cpu_s": time.process_time() - cpu,
                    })
                except Exception as journal_error:
                    raise AdmissionJournalError("admission error could not be preserved") from journal_error
                raise
            _write(prefix.with_suffix(".result.json"), {
                **intent, "status": "returned", "snapshot": result,
                "finished_utc": datetime.now(UTC).isoformat(),
                "wall_s": time.perf_counter() - started,
                "parent_process_cpu_s": time.process_time() - cpu,
            })
            return result

        recorded.__dict__["_admission_journal_wrapper"] = True
        gpu.query_registered_desktop_gpu_retry = recorded
        _LOCAL.depth, _LOCAL.folder = 1, folder
        try:
            yield folder
        finally:
            replaced = gpu.query_registered_desktop_gpu_retry is not recorded
            gpu.query_registered_desktop_gpu_retry = original
            _LOCAL.depth = 0
            del _LOCAL.folder
            if replaced:
                raise AdmissionJournalError("admission query changed inside the journal scope")
