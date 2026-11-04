"""커밋된 post 결과 조회를 추가한 조사 CLI v3."""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

from rl_triage.investigation_session import canonical
from rl_triage.published_post_status_v1 import published_post_status

v1 = import_module("rl_triage.investigate")
v2 = import_module("rl_triage.investigate_v2")


def run(command: str, session_path: Path, payload: dict, *, backend_name: str,
        fixture: bool) -> dict:
    if command != "post-status":
        return v2.run(
            command, session_path, payload, backend_name=backend_name, fixture=fixture,
        )
    if payload:
        raise ValueError("post-status input must be empty")
    if fixture or backend_name != v1.REGISTERED_BACKEND:
        raise ValueError("post-status requires the registered read-only backend")
    return published_post_status(session_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "next", "approve", "dispatch", "settle", "status", "export",
            *v1.RECOVERY_COMMANDS, "post-status",
        ),
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--backend", default=v1.REGISTERED_BACKEND)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.command,
            args.session,
            v1._payload(args.input),
            backend_name=args.backend,
            fixture=args.fixture,
        )
    except Exception as error:  # noqa: BLE001 - v2와 같은 JSON 오류 경계를 유지한다.
        result = {
            "status": "rejected",
            "error_type": type(error).__name__,
            "reason": str(error),
            "completed": False,
        }
    sys.stdout.write(canonical(result).decode() + "\n")
    return 0 if result.get("status") != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
