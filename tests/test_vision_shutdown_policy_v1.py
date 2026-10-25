from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import vision_shutdown_policy_v1 as S  # pyright: ignore[reportMissingImports]


class Settings:
    def __init__(self, value: bool = False, *, refuse: bool = False) -> None:
        self.value, self.refuse = value, refuse
        self.events: list[str] = []
    def get_as_bool(self, _name: str) -> bool:
        self.events.append(f"read:{self.value}")
        return self.value
    def set_bool(self, _name: str, value: bool) -> None:
        self.events.append(f"set:{value}")
        if not self.refuse:
            self.value = value


def _fixture(tmp_path: Path, terminal: str = "result.json") -> tuple[Path, dict[str, Any], Path]:
    job = tmp_path / "job.json"; job.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"; output.mkdir()
    evidence = output / terminal; evidence.write_text('{"terminal":true}\n', encoding="utf-8")
    validated = {"job_path": job, "job_sha256": R.sha256_file(job), "output_directory": output}
    return job, validated, evidence


def test_policy_reads_false_transitions_true_and_binds_terminal(tmp_path: Path) -> None:
    job, validated, terminal = _fixture(tmp_path)
    settings = Settings()
    value = S.apply(validated, settings, startup_fast_shutdown=False)
    assert settings.events == ["read:False", "set:True", "read:True"]
    assert value["terminal_evidence"] == {"file": "result.json", "sha256": R.sha256_file(terminal)}
    assert value["phase"] == "after_terminal_evidence_before_sdk_close"
    assert value["wait_for_replicator"] is False
    assert S.verify(job, validated, terminal) == value


def test_proxy_writes_policy_before_close_and_passes_wait_false(tmp_path: Path, monkeypatch) -> None:
    _, validated, _ = _fixture(tmp_path)
    events: list[str] = []
    original = S.R.write_new
    monkeypatch.setattr(S.R, "write_new", lambda path, value: (
        events.append("policy"), original(path, value))[1])
    class App:
        def close(self, **kwargs: Any) -> None:
            assert kwargs == {"wait_for_replicator": False}
            assert (tmp_path / S.SIDECAR).is_file()
            events.append("close")
    proxy = S.AppProxy(App(), validated, Settings())
    proxy.close()
    assert events == ["policy", "close"]


@pytest.mark.parametrize("case", ["missing", "tamper", "terminal_sha", "hardlink"])
def test_verify_rejects_missing_tampered_or_linked_policy(tmp_path: Path, case: str) -> None:
    job, validated, terminal = _fixture(tmp_path)
    if case != "missing":
        S.apply(validated, Settings(), startup_fast_shutdown=False)
    sidecar = tmp_path / S.SIDECAR
    if case == "tamper":
        value = json.loads(sidecar.read_text(encoding="utf-8")); value["after"] = False
        sidecar.write_text(json.dumps(value), encoding="utf-8")
    elif case == "terminal_sha":
        terminal.write_text("changed\n", encoding="utf-8")
    elif case == "hardlink":
        try:
            os.link(sidecar, tmp_path / "policy-link.json")
        except OSError as error:
            pytest.skip(f"hardlink unavailable: {error}")
    with pytest.raises((ValueError, FileNotFoundError)):
        S.verify(job, validated, terminal)


def test_policy_failure_still_closes_and_raises(tmp_path: Path) -> None:
    _, validated, _ = _fixture(tmp_path)
    closed: list[bool] = []
    class App:
        def close(self, **_kwargs: Any) -> None: closed.append(True)
    proxy = S.AppProxy(App(), validated, Settings(refuse=True))
    with pytest.raises(RuntimeError, match="readback"):
        proxy.close()
    assert closed == [True] and not (tmp_path / S.SIDECAR).exists()


def test_terminal_evidence_must_exist_exactly_once(tmp_path: Path) -> None:
    _, validated, _ = _fixture(tmp_path)
    (validated["output_directory"] / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="정확히 하나"):
        S.apply(validated, Settings(), startup_fast_shutdown=False)


def test_shutdown_modules_do_not_force_exit_or_kill() -> None:
    assert S.__file__ is not None
    sources = (Path(S.__file__).read_text(encoding="utf-8"),
               Path(ROOT / "bench/vision_play_worker_v3.py").read_text(encoding="utf-8"))
    assert all("os._exit" not in source and ".kill(" not in source for source in sources)
