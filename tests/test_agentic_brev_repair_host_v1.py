from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import agentic_brev_repair_evidence_v1 as E
import agentic_brev_repair_host_v1 as M
import pytest


def _runtime(tmp_path: Path, protocol: Path, protocol_ref: dict, bands: dict):
    return SimpleNamespace(
        protocol={"sources_lf": {name: "0" * 64 for name in M.SOURCES}},
        protocol_ref=protocol_ref,
        repo=tmp_path,
        anchor="1" * 40,
        reg={"bands": bands},
    )


def test_windows_and_posix_aliases_resolve_to_same_runtime(tmp_path: Path) -> None:
    protocol = tmp_path / "runtime.json"
    protocol.write_text("{}\n", encoding="utf-8")
    sha = E.J.file_sha(protocol)
    rows = [
        {"remote": r"C:\registered", "local": str(tmp_path)},
        {"remote": "/provider", "local": str(tmp_path)},
    ]
    resolver = E.RootMap(rows)
    runtime = _runtime(tmp_path, protocol, {"path": r"C:\registered\runtime.json", "sha256": sha}, {"x": 1})
    registered = {
        "resolver": resolver,
        "raw": {"runtime": {"protocol": {"path": "/provider/runtime.json", "sha256": sha},
                              "repo": "/provider", "anchor": "1" * 40}},
        "protocol": runtime.protocol,
        "contract": {"bands": {"x": 1}},
    }
    M._registered_runtime_binding(registered, runtime)


def test_aliases_cannot_hide_different_runtime_bytes(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    (left / "runtime.json").write_text("{}\n", encoding="utf-8")
    (right / "runtime.json").write_text("{}\n", encoding="utf-8")
    sha = E.J.file_sha(left / "runtime.json")
    resolver = E.RootMap([
        {"remote": r"C:\registered", "local": str(left)},
        {"remote": "/provider", "local": str(right)},
    ])
    runtime = _runtime(right, right / "runtime.json", {"path": "/provider/runtime.json", "sha256": sha}, {"x": 1})
    registered = {
        "resolver": resolver,
        "raw": {"runtime": {"protocol": {"path": r"C:\registered\runtime.json", "sha256": sha},
                              "repo": r"C:\registered", "anchor": "1" * 40}},
        "protocol": runtime.protocol,
        "contract": {"bands": {"x": 1}},
    }
    with pytest.raises(ValueError, match="same Runtime bytes"):
        M._registered_runtime_binding(registered, runtime)


def test_repair_host_requires_exact_host5(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M.BrevRepairRuntimeHost, "_sources", lambda self: None)
    runtime = object.__new__(M.R.Runtime)
    runtime.reg = {"allow_fixture": True}
    runtime.folder = Path.cwd()
    exact = object.__new__(M.H5.BrevRuntimeHost)
    exact.runtime = runtime
    host = M.BrevRepairRuntimeHost(
        runtime, exact, object(), {}, [], "approver", probe_seals=[], probe_records=[],
        admitter=lambda *_: None, allow_fixture=True,
    )
    assert host.probe_host is exact

    class Derived(M.H5.BrevRuntimeHost):
        pass

    derived = object.__new__(Derived)
    derived.runtime = runtime
    with pytest.raises(ValueError, match="exact Host5"):
        M.BrevRepairRuntimeHost(
            runtime, derived, object(), {}, [], "approver", probe_seals=[], probe_records=[],
            admitter=lambda *_: None, allow_fixture=True,
        )
