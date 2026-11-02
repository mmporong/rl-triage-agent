# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import recovery_legacy_validation_isolation_v1 as M
import test_local_pair_probe_host_v1 as HOST_FIXTURE


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def runtime_protocol(tmp_path: Path) -> Path:
    return save(tmp_path / "runtime-view.json", M.RV.prepare_protocol())


def test_runtime_scope_reproduces_legacy_snapshot_error_and_isolation_fixes_it(
        tmp_path: Path) -> None:
    snapshot = HOST_FIXTURE.snapshot(tmp_path)
    protocol = runtime_protocol(tmp_path)
    legacy = M._view_state()
    with M.RV.view_scope(protocol, None, fixture=True):
        runtime = M._view_state()
        assert M._is_registered_runtime(runtime)
        with pytest.raises(ValueError, match="runtime view protocol SHA"):
            M._VIEW.checked(snapshot)
        with M.legacy_validation_scope():
            assert M._is_legacy(M._view_state())
            assert M._VIEW.checked(snapshot) == snapshot
        assert M._view_state() == runtime
        with pytest.raises(ValueError, match="runtime view protocol SHA"):
            M._VIEW.checked(snapshot)
    assert M._view_state() == legacy


def test_legacy_scope_rejects_unknown_view_and_restores_runtime_after_mutation(
        tmp_path: Path) -> None:
    protocol = runtime_protocol(tmp_path)
    legacy = M._view_state()
    unknown = tmp_path / "unknown.json"
    unknown.write_text("{}\n", encoding="utf-8")
    M._VIEW.PROTOCOL = unknown
    try:
        with pytest.raises(ValueError, match="알 수 없는"), M.legacy_validation_scope():
            pass
    finally:
        M._VIEW.BASE_FACT_IDS, M._VIEW.PROTOCOL, M._VIEW.verify_protocol = M._LEGACY
    assert M._view_state() == legacy

    with M.RV.view_scope(protocol, None, fixture=True):
        runtime = M._view_state()
        with pytest.raises(RuntimeError, match="외부"), M.legacy_validation_scope():
            M._VIEW.PROTOCOL = unknown
        assert M._view_state() == runtime
    assert M._view_state() == legacy


def test_source_validation_wraps_only_two_functions_and_restores_on_exception(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = runtime_protocol(tmp_path)
    calls: list[tuple[str, object]] = []

    def checked(source: dict, **_kwargs: object) -> dict:
        assert M._is_legacy(M._view_state())
        calls.append(("checked", source))
        if source.get("raise"):
            raise RuntimeError("source failure")
        return dict(source)

    def probe(path: Path, _lab: Path, role: str) -> tuple[dict, dict, dict]:
        assert M._is_legacy(M._view_state())
        calls.append(("probe", role))
        return {"path": str(path)}, {}, {"role": role}

    monkeypatch.setattr(M, "_ORIGINAL_CHECKED_SOURCE", checked)
    monkeypatch.setattr(M, "_ORIGINAL_PROBE_SOURCE", probe)
    monkeypatch.setattr(M.SECOND, "checked_source", checked)
    monkeypatch.setattr(M.SECOND._PROBE, "_source", probe)
    originals = M.SECOND.checked_source, M.SECOND._PROBE._source
    with M.RV.view_scope(protocol, None, fixture=True):
        runtime = M._view_state()
        with M.source_validation_scope():
            assert M.SECOND.checked_source({"ok": True}) == {"ok": True}
            assert M.SECOND._PROBE._source(tmp_path / "input", tmp_path, "case")[2] == {
                "role": "case"}
            with pytest.raises(RuntimeError, match="source failure"):
                M.SECOND.checked_source({"raise": True})
            assert M._view_state() == runtime
        assert M.SECOND.checked_source is originals[0]
        assert M.SECOND._PROBE._source is originals[1]
        assert M._view_state() == runtime
    assert calls == [("checked", {"ok": True}), ("probe", "case"),
                     ("checked", {"raise": True})]


def test_source_validation_detects_external_wrapper_change_and_restores(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def checked(source: dict, **_kwargs: object) -> dict:
        return source

    def probe(_path: Path, _lab: Path, _role: str) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    monkeypatch.setattr(M, "_ORIGINAL_CHECKED_SOURCE", checked)
    monkeypatch.setattr(M, "_ORIGINAL_PROBE_SOURCE", probe)
    monkeypatch.setattr(M.SECOND, "checked_source", checked)
    monkeypatch.setattr(M.SECOND._PROBE, "_source", probe)
    scope = M.source_validation_scope()
    scope.__enter__()
    M.SECOND.__dict__["checked_source"] = lambda source, **_kwargs: source
    with pytest.raises(RuntimeError, match="외부"):
        scope.__exit__(None, None, None)
    assert M.SECOND.checked_source is checked
    assert M.SECOND._PROBE._source is probe
