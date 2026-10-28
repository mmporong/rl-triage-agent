"""합성 CPU transport로 Parent2 actual argv·closure·scope 복원을 검증한다."""
from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]
import agentic_brev_probe_transport_remote_v3 as T


def _prepared(tmp_path: Path) -> tuple[Path, dict]:
    request = tmp_path / "request.json"
    request.write_bytes(b"{}")
    sources = {name: T.PARENT.C.digest_lf_file(ROOT / name) for name in T.SOURCES}
    prepared = {
        "status": "prepared", "repo": tmp_path, "python": Path("/venv/bin/python"),
        "protocol": {"sources_lf": sources, "transport_root": {"base": "repo", "relative": "transport"}},
        "execution": {"runtime_host_binding": {
            "evidence_roots": {arm: [{"remote": f"/{arm.lower()}", "local": str(tmp_path / arm)}] for arm in T.ARMS},
            "metadata_roots": [{"remote": str(tmp_path), "local": str(tmp_path / "host")}],
        }},
        "operations": {arm: {"op": {"operation_id": arm.lower()}} for arm in T.ARMS},
        "fixture": True,
    }
    return request, prepared


def test_start_argv_uses_parent2_entry_and_scopes_restore(tmp_path) -> None:
    request, prepared = _prepared(tmp_path)
    calls = []

    class Process:
        pid = 4321

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()

    result = T.run_transport(
        request=request, request_sha256=sha256(request.read_bytes()).hexdigest(), mode="start",
        allow_fixture=True, preparer=lambda *_args, **_kwargs: prepared, popen_factory=popen,
        identity_fn=lambda pid: {"pid": pid, "posix_group_verified": True},
    )
    assert result["status"] == "running" and result["schema"] == T.SCHEMA
    assert Path(calls[0][0][2]).name == "run_agentic_brev_probe_parent_v2.py"
    assert T.V1.P.__name__ == "run_agentic_brev_probe_parent_v1" and T.V2.P is T.V1.P
    assert not T.PARENT.scope_active()


def test_frozen_v1_prepare_accepts_only_default_parent2_actual_identity(tmp_path, monkeypatch) -> None:
    request, prepared = _prepared(tmp_path)
    observed = []

    def parent_prepare(*args, **kwargs):
        observed.append((T.V1.P, T.V1.PRODUCTION, args, kwargs))
        return prepared

    monkeypatch.setattr(T.PARENT.V1, "prepare", parent_prepare)
    monkeypatch.setattr(T.V1, "os", SimpleNamespace(name="posix"))
    result = T._prepare(
        request, sha256(request.read_bytes()).hexdigest(), allow_fixture=False,
        home=None, now=None, preparer=T.PRODUCTION[0],
    )
    assert result["status"] == "prepared" and observed
    assert observed[0][0] is T.PARENT and observed[0][1] is T.PRODUCTION
    assert T.V1.PRODUCTION is T._BASE_V1_PRODUCTION and T.V2.PRODUCTION is T._BASE_V2_PRODUCTION


def test_v3_closure_missing_refused_before_spawn(tmp_path) -> None:
    request, prepared = _prepared(tmp_path)
    prepared["protocol"]["sources_lf"].pop(T.SOURCE)
    called = []
    with pytest.raises(ValueError, match="v3 Parent2"):
        T.run_transport(
            request=request, request_sha256=sha256(request.read_bytes()).hexdigest(), mode="start",
            allow_fixture=True, preparer=lambda *_args, **_kwargs: prepared,
            popen_factory=lambda *_args, **_kwargs: called.append(True),
        )
    assert called == []


def test_same_name_foreign_parent_module_is_refused(monkeypatch) -> None:
    foreign = SimpleNamespace(__name__=T._BASE_PARENT.__name__)
    monkeypatch.setattr(T.V1, "P", foreign)
    monkeypatch.setattr(T.V2, "P", foreign)
    with pytest.raises(RuntimeError, match="이미 교체"), T.parent2_scope():
        pass


@pytest.mark.parametrize("mode", T.MODES)
@pytest.mark.parametrize("dependency", ["preparer", "popen_factory", "identity_fn"])
def test_actual_dependency_injection_remains_refused(tmp_path, mode, dependency) -> None:
    called = []
    kwargs: dict[str, Any] = {dependency: lambda *_args, **_kwargs: called.append(True)}
    with pytest.raises(ValueError, match="dependency injection"):
        T.run_transport(request=tmp_path / "absent", request_sha256="0" * 64, mode=mode, **kwargs)
    assert called == []

