from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import agentic_brev_probe_transport_remote_v6 as R  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_actual_start_argv_selects_parent4(tmp_path: Path) -> None:
    folder = tmp_path / "transport"
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    prepared = {"transport_folder": folder, "python": Path(sys.executable)}
    calls = []

    class Process:
        pid = 123

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()

    identity = {"pid": 123, "pgid": 123, "sid": 123, "start_ticks": 1,
                "posix_group_verified": True}
    with R.parent4_scope():
        value = R.V1._start(prepared, request, "a" * 64, popen, lambda _pid: identity)
    assert value["status"] == "running"
    assert calls[0][0][2] == str((ROOT / R.PARENT.SOURCE).resolve())
    assert calls[0][1]["start_new_session"] is True


def test_dead_parent_without_outcome_becomes_hold_with_log_refs(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "transport"
    folder.mkdir()
    (folder / "parent.stdout").write_bytes(b"")
    (folder / "parent.stderr").write_text("trace", encoding="utf-8")
    intent = folder / "launch-intent.json"
    intent.write_text(json.dumps({"identity": {"pid": 10}}), encoding="utf-8")
    monkeypatch.setattr(R.V1.OWNED, "_identity_is_current", lambda _identity: False)
    value = R._dead_parent_status(
        {"transport_folder": folder}, {"status": "pending", "pair": {"status": "pending"}},
    )
    assert value["status"] == "hold"
    assert value["terminal_source"] == "dead_parent_without_outcome"
    assert value["exit_code_known"] is False
    assert set(value["parent_logs"]) == {"parent.stdout", "parent.stderr"}


def test_live_parent_and_existing_terminal_state_are_preserved(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(R.V1, "_launch_status", lambda _folder: {"owned_current": True})
    pending = {"status": "pending", "pair": {"status": "pending"}}
    assert R._dead_parent_status({"transport_folder": tmp_path}, pending) is pending
    complete = {"status": "completed", "pair": {"status": "complete"}}
    assert R._dead_parent_status({"transport_folder": tmp_path}, complete) is complete
