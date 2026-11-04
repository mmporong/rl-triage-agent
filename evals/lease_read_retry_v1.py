"""Bounded rereading of a Windows lease metadata sharing failure; no cached lease."""
from __future__ import annotations

import errno
import os
import time
from collections.abc import Callable
from pathlib import Path

DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.16)


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return os.path.normcase(os.path.abspath(os.fsdecode(value))) == os.path.normcase(
        os.path.abspath(expected))


def read_live_lease(reader: Callable[[], dict], *, lease_path: Path | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> dict:
    """Repeat only the original reader on EACCES for the exact metadata path.

    The original reader still validates caller ancestry, owner, heartbeat and
    exclusive lock. Invalid JSON, missing metadata and failed authority checks
    propagate immediately. A persistent permission failure is also rejected.
    """
    expected = lease_path if lease_path is not None else Path.home() / "CodexSignals/gpu-lease.json"
    for attempt in range(len(DELAYS_S) + 1):
        try:
            return reader()
        except PermissionError as error:
            if (error.errno != errno.EACCES or not _same_path(error.filename, expected)
                    or attempt == len(DELAYS_S)):
                raise
            sleep(DELAYS_S[attempt])
    raise RuntimeError("unreachable lease read state")
