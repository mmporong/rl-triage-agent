"""terminal-aware status를 제공하는 조사 CLI v2."""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

from evals.diagnostic_terminal_status_v1 import terminal_aware_status
from rl_triage.investigation_session import InvestigationSession, canonical

v1 = import_module("rl_triage.investigate")


def run(command: str, session_path: Path, payload: dict, *, backend_name: str,
        fixture: bool) -> dict:
    """status만 보강하고 나머지는 v1 계약에 그대로 위임한다."""
    if command != "status":
        return v1.run(
            command,
            session_path,
            payload,
            backend_name=backend_name,
            fixture=fixture,
        )

    result = terminal_aware_status(InvestigationSession(session_path))
    tool_ledger_status = "open" if result["status"] == "model_hold" else result["status"]
    return {**result, "tool_ledger_status": tool_ledger_status}


def main(argv: list[str] | None = None) -> int:
    """v1과 같은 인자, 오류 JSON, 종료 코드 계약을 유지한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "next", "approve", "dispatch", "settle", "status", "export",
            *v1.RECOVERY_COMMANDS,
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
    except Exception as error:  # noqa: BLE001 - v1 CLI와 같은 안전한 오류 경계다.
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
