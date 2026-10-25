from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src", ROOT / "tests"):
    sys.path.insert(0, str(folder))

RUN = importlib.import_module("run_vision_selection_e2e_v1")
FIXTURE = importlib.import_module("test_vision_selection_e2e_caller_v1")


def argv(caller: Any, command: str) -> list[str]:
    context = caller.context
    return [
        command,
        "--repo", str(caller.root),
        "--plan", str(caller.plan_path),
        "--execution-protocol", str(context["protocol_path"]),
        "--execution-protocol-sha256", context["protocol_sha256"],
        "--lease", str(context["lease_path"]),
        "--lease-sha256", context["lease_sha256"],
        "--runtime", str(caller.runtime),
    ]


def test_cli_verify_remaining_and_summarize_use_actual_registered_inputs(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    caller, *_ = FIXTURE.setup_caller(tmp_path)
    assert RUN.main(argv(caller, "verify")) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == {"status": "verified", "execution_id": "caller-fixture", "api_cap": 32}

    assert RUN.main(argv(caller, "remaining_commands")) == 0
    remaining = json.loads(capsys.readouterr().out)
    assert remaining[0]["kind"] == "initial_selection"

    assert RUN.main(argv(caller, "summarize")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["remaining_commands"] == remaining
    assert summary["verified_compute_usd"] == 0.0


def test_cli_backend_factory_loads_only_registered_concrete_class(tmp_path: Path) -> None:
    caller, *_ = FIXTURE.setup_caller(tmp_path)
    backend = RUN._backend(caller, {"lab": ".", "python": sys.executable})
    assert type(backend).__name__ == "FixtureService"
    module_file = sys.modules[type(backend).__module__].__file__
    assert module_file is not None
    assert Path(module_file).resolve() == (
        tmp_path / "fixture_service.py"
    ).resolve()
    record_backend = RUN._backend(
        caller, {"lab": ".", "python": sys.executable}, binding_name="record_verifier",
    )
    assert type(record_backend).__name__ == "FixtureService"
    with pytest.raises(ValueError, match="지원하지 않는"):
        RUN._backend(caller, {"lab": ".", "python": sys.executable}, binding_name="foreign")


def test_cli_advance_without_runnable_action_or_explicit_approval_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    caller, *_ = FIXTURE.setup_caller(tmp_path)
    monkeypatch.setattr(RUN, "load_caller", lambda _args: caller)
    monkeypatch.setattr(caller, "remaining_commands", list)
    assert RUN.main(argv(caller, "advance")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "HOLD", "reason": "no_runnable_command",
    }


def test_cli_expired_lease_allows_completed_summary_but_blocks_new_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    caller, _prepared, contract, binding, snapshot = FIXTURE.setup_caller(tmp_path)

    class ExpiredDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2028, 1, 1, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(RUN.ADAPTER, "datetime", ExpiredDateTime)
    assert RUN.main(argv(caller, "summarize")) == 0
    assert json.loads(capsys.readouterr().out)["selection_count"] == 0
    payload = FIXTURE.write(tmp_path / "expired-input.json", {
        "live_snapshot": snapshot, "live_contract": str(contract),
        "source_binding": binding,
    })
    with pytest.raises(ValueError, match="만료"):
        RUN.main([*argv(caller, "advance"), "--input", str(payload)])
    assert caller.api_requests() == 0
