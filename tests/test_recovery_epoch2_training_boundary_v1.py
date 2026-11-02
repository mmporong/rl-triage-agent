# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import recovery_epoch2_training_boundary_v1 as B


def test_prepare_derives_guarded_and_runtime_closures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loop = tmp_path / "loop.json"
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"runtime_source_sha256_lf": {str(i): "a" * 64 for i in range(9)}}), encoding="utf-8")
    loop.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(B, "ROOT", tmp_path)
    monkeypatch.setattr(B.SECOND.TRAINING.TRAIN, "BOUND_SOURCES", frozenset({"guard.py"}))
    (tmp_path / "guard.py").write_text("guard\n", encoding="utf-8")
    for name in B.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    protocol = {
        "first_correction_component": {"path": "first.json", "sha256_lf": B._sha(first, lf=True)},
        "sources_lf": {"guard.py": B._sha(tmp_path / "guard.py", lf=True)},
    }
    monkeypatch.setattr(B.PROBE_BOUNDARY, "verify_loop_protocol", lambda *_a, **_k: protocol)
    value = B.prepare_protocol(loop, "1" * 40)
    assert value["runtime_source_sha256_lf"] == json.loads(first.read_text())["runtime_source_sha256_lf"]
    assert set(value["source_sha256_lf"]) == {"guard.py", *B.SOURCE_FILES}


def test_scope_substitutes_only_exact_loop_and_restores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loop, component = tmp_path / "loop.json", tmp_path / "component.json"
    loop.write_text("{}", encoding="utf-8")
    component.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(B, "verify_protocol", lambda *_a, **_k: {"component": True})
    monkeypatch.setattr(B.PROBE_BOUNDARY, "verify_loop_protocol", lambda *_a, **_k: {})
    calls: list[tuple[Path, str]] = []
    verified: list[bool] = []

    def original(path: Path, _value: object, anchor: str, **_kwargs: object) -> dict:
        calls.append((Path(path), anchor))
        return {"ok": True}

    monkeypatch.setattr(B, "_ORIGINAL_BUILD", original)
    base_export = lambda *_a, **_k: {"export": True}
    monkeypatch.setattr(B, "_ORIGINAL_EXPORT_TRAINING", base_export)
    monkeypatch.setattr(B.EXPORT, "_training", B._ORIGINAL_EXPORT_TRAINING)
    monkeypatch.setattr(B, "verify_receipt_binding",
                        lambda *_a, **_k: verified.append(True) or {})
    monkeypatch.setattr(B.SECOND.TRAINING, "build_manifest", original)
    with B.training_boundary_scope(loop_protocol=loop, loop_anchor="1" * 40,
                                   training_protocol=component, training_anchor="2" * 40,
                                   lab=tmp_path):
        assert B.SECOND.TRAINING.build_manifest(loop, {}, "1" * 40) == {"ok": True}
        first_receipt = {"protocol": {"path": "first.json", "sha256": "f" * 64}}
        assert B.EXPORT._training(first_receipt, {}, second=False) == {"export": True}
        assert verified == []
        assert B.EXPORT._training({}, {}, second=True) == {"export": True}
        with pytest.raises(ValueError, match="등록 loop"):
            B.SECOND.TRAINING.build_manifest(tmp_path / "other", {}, "1" * 40)
    assert calls == [(component.resolve(), "2" * 40)]
    assert verified == [True]
    assert B.SECOND.TRAINING.build_manifest is original
    assert B.EXPORT._training is base_export


def test_export_scope_rejects_second_with_wrong_component_but_not_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    loop, component = tmp_path / "loop.json", tmp_path / "component.json"
    loop.write_text("{}", encoding="utf-8")
    component.write_text("{}", encoding="utf-8")
    first_receipt = {"protocol": {"path": "first.json", "sha256": "1" * 64}}
    base_export = lambda receipt, _candidate, **_kwargs: {"protocol": receipt["protocol"]}
    monkeypatch.setattr(B, "verify_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(B, "_ORIGINAL_EXPORT_TRAINING", base_export)
    monkeypatch.setattr(B.EXPORT, "_training", base_export)
    monkeypatch.setattr(B, "verify_receipt_binding",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("wrong component")))
    with B.training_boundary_scope(loop_protocol=loop, loop_anchor="1" * 40,
                                   training_protocol=component, training_anchor="2" * 40,
                                   lab=tmp_path):
        assert B.EXPORT._training(first_receipt, {}, second=False) == {
            "protocol": first_receipt["protocol"]}
        with pytest.raises(ValueError, match="wrong component"):
            B.EXPORT._training({"protocol": {}}, {}, second=True)


def test_scope_restores_after_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loop, component = tmp_path / "loop.json", tmp_path / "component.json"
    loop.write_text("{}", encoding="utf-8")
    component.write_text("{}", encoding="utf-8")
    old_build, old_export = B.SECOND.TRAINING.build_manifest, B.EXPORT._training
    monkeypatch.setattr(B, "verify_protocol", lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="boom"), B.training_boundary_scope(
        loop_protocol=loop, loop_anchor="1" * 40,
        training_protocol=component, training_anchor="2" * 40, lab=tmp_path,
    ):
        raise RuntimeError("boom")
    assert B.SECOND.TRAINING.build_manifest is old_build
    assert B.EXPORT._training is old_export


def test_checked_loop_uses_typed_verifier_without_global_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    protocol = tmp_path / "typed.json"
    protocol.write_text("{}", encoding="utf-8")
    calls: list[tuple[Path, str, Path | None]] = []

    def typed(path: Path, anchor: str, *, lab: Path | None = None) -> dict:
        calls.append((Path(path), anchor, lab))
        return {"typed": True}

    monkeypatch.setattr(B.PROBE_BOUNDARY, "verify_loop_protocol", typed)
    monkeypatch.setattr(
        B.SECOND, "verify_loop_protocol",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("legacy verifier called")),
    )
    assert B._checked_loop(protocol, "1" * 40, lab=tmp_path) == {"typed": True}
    assert calls == [(protocol, "1" * 40, tmp_path)]


def test_receipt_binding_rejects_wrong_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loop, component, manifest = (tmp_path / name for name in ("loop.json", "component.json", "manifest.json"))
    for path in (loop, component):
        path.write_text("{}", encoding="utf-8")
    manifest.write_text(json.dumps({"protocol_path": "wrong", "protocol_sha256_lf": "0" * 64,
                                    "source_commit_sha": "2" * 40}), encoding="utf-8")
    candidate = {"inputs": {"loop_protocol": B.SECOND.ref(loop)}}
    receipt = {"protocol": B.SECOND.ref(loop), "training": {"manifest": B.SECOND.ref(manifest)}}
    monkeypatch.setattr(B, "ROOT", tmp_path)
    monkeypatch.setattr(B.SECOND.TRAINING.TRAIN, "validate_manifest", lambda value: value)
    with pytest.raises(ValueError, match="component 결합"):
        B.verify_receipt_binding(receipt, candidate, loop_protocol=loop,
                                 training_protocol=component, training_anchor="2" * 40, lab=tmp_path)
