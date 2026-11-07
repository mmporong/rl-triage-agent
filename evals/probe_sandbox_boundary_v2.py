"""Operator-run boundary probes, not model-generated prompt-injection attempts."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path


def error_errno(error: BaseException) -> int | None:
    current: object = error
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        number = getattr(current, "errno", None)
        if type(number) is int:
            return number
        current = getattr(current, "reason", None)
    return None


def observe(operation) -> dict:
    try:
        operation()
        return {"outcome": "allowed", "errno": None, "policy_eacces": False}
    except OSError as error:
        number = error_errno(error)
        return {"outcome": "denied_or_failed", "error_type": type(error).__name__,
                "errno": number, "policy_eacces": number == errno.EACCES}


def probe() -> dict:
    app = Path("/sandbox/app/src/rl_triage/triage_tools.py")
    before = hashlib.sha256(app.read_bytes()).hexdigest()

    def writable_app():
        # Test permission to open, never append data even in the control container.
        with app.open("ab"):
            pass

    def forbidden_host():
        with urllib.request.urlopen("https://huggingface.co", timeout=10) as response:
            response.read(1)

    def metadata():
        with socket.create_connection(("169.254.169.254", 80), timeout=5):
            pass

    checks = {"outside_read": observe(lambda: os.listdir("/home")),
              "unregistered_host": observe(forbidden_host),
              "cloud_metadata_tcp": observe(metadata), "application_write_open": observe(writable_app)}
    after = hashlib.sha256(app.read_bytes()).hexdigest()
    return {"schema": "sandbox_boundary_probe_v2", "checks": checks,
            "application_unchanged": before == after, "application_sha256": after,
            "application_mode": oct(app.stat().st_mode & 0o777), "uid": os.getuid(),
            "model_calls": 0, "model_prompt_injection_measured": False,
            "gpu_runs": 0, "data_sent_to_unregistered_host": False,
            "probe_scope": "operator syscall controls; EACCES distinguished from outage or refused connection"}


if __name__ == "__main__":
    print(json.dumps(probe(), sort_keys=True))
