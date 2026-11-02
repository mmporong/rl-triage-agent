# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_recovery_same_play_capture_v2 as C


def test_target_scope_restores_and_reuses_prevalidated_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    design, config = tmp_path / "design.json", {"schema": C.CONFIG_SCHEMA}
    design.write_text("{}", encoding="utf-8")
    targets = [{"key": "case-07:logs_only", "status": "registered_recovered"}]
    original = C.V1.validated_targets
    original_schema = C.V1.CONFIG_SCHEMA
    monkeypatch.setattr(C, "validated_targets", lambda *_a, **_k: targets)
    with C.target_scope(design, config) as rows:
        assert rows == targets
        assert C.V1.validated_targets(design, config) == targets
        assert C.V1.CONFIG_SCHEMA == C.CONFIG_SCHEMA
        with pytest.raises(ValueError, match="다른 media config"):
            C.V1.validated_targets(tmp_path / "other", config)
    assert C.V1.validated_targets is original
    assert C.V1.CONFIG_SCHEMA == original_schema


def test_target_scope_restores_after_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    design = tmp_path / "design.json"
    design.write_text("{}", encoding="utf-8")
    original = C.V1.validated_targets
    monkeypatch.setattr(C, "validated_targets", lambda *_a, **_k: [])
    with pytest.raises(RuntimeError, match="boom"), C.target_scope(design, {}):
        raise RuntimeError("boom")
    assert C.V1.validated_targets is original


def test_validated_targets_requires_single_registered_case07_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    first, second, export = tmp_path / "first.json", tmp_path / "second.json", tmp_path / "export"
    first.write_text('{"schema":"five_policy_recovery_first_continuation_result_v2","status":"complete"}', encoding="utf-8")
    second.write_text('{"schema":"five_policy_recovery_parent_second_result_v1","status":"complete"}', encoding="utf-8")
    export.mkdir()
    config = {"first_result": C.V1.B._ref(first), "second_result": C.V1.B._ref(second),
              "lab": str(tmp_path), "v9_protocol": C.V1.B._ref(first),
              "v9_private_plan": C.V1.B._ref(second), "v9_anchor": "a" * 40,
              "recovery_export": str(export)}
    monkeypatch.setattr(C.EXPORT2, "_continuation", lambda *_a, **_k: ())
    monkeypatch.setattr(C.V9, "verify_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(C, "_export_binding", lambda *_a, **_k: None)
    monkeypatch.setattr(C, "_verify_selected_second",
                        lambda *_a, **_k: {"fixed_conditions": 26, "fresh_initialization": True})
    clone = type("Clone", (), {"validated_targets": staticmethod(lambda *_a, **_k: [
        {"key": "case-03:logs_only", "status": "HOLD"},
        {"key": "case-07:logs_only", "status": "registered_recovered",
         "registered_seed": 5557, "final_epoch": "second"},
    ])})
    monkeypatch.setattr(C, "_isolated_v1", lambda: clone)
    rows = C.validated_targets(tmp_path / "design.json", config)
    assert [row["status"] for row in rows] == ["HOLD", "registered_recovered"]


def test_export_binding_rejects_v1_first_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first, second, folder = tmp_path / "first.json", tmp_path / "second.json", tmp_path / "export"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    folder.mkdir()
    for name, value in {
        "summary.json": {}, "cases.json": {},
        "provenance.json": {"first_result_schema": "five_policy_recovery_parent_result_v1",
                            "first_result": {"sha256": C.sha(first)},
                            "second_result": {"sha256": C.sha(second)}}}.items():
        (folder / name).write_text(__import__("json").dumps(value), encoding="utf-8")
    files = {name: {"sha256": C.sha(folder / name), "size_bytes": (folder / name).stat().st_size}
             for name in ("summary.json", "cases.json", "provenance.json")}
    (folder / "manifest.json").write_text(__import__("json").dumps(
        {"schema": C.EXPORT2.MANIFEST_SCHEMA, "status": "complete", "files": files}), encoding="utf-8")
    with pytest.raises(ValueError, match="public/raw"):
        C._export_binding(folder, first, second)


def test_prepare_cli_forwards_all_registered_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = [tmp_path / name for name in (
        "adapter.json", "design.json", "first.json", "second.json", "export",
        "batch.json", "index.json", "lab", "python", "v9.json", "plan.json", "config.json")]
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(C, "prepare_config", lambda *args: calls.append(args) or {})
    argv = ["prepare"]
    names = ("adapter-protocol", "design", "first-result", "second-result", "recovery-export",
             "input-capture-batch", "input-source-index", "lab", "isaac-python", "v9-protocol",
             "v9-private-plan", "out")
    for name, path in zip(names, paths, strict=True):
        argv.extend((f"--{name}", str(path)))
    argv.extend(("--adapter-anchor", "a" * 40, "--v9-anchor", "b" * 40))
    assert C.main(argv) == 0
    assert calls and calls[0][0] == paths[0] and calls[0][-1] == paths[-1]


def test_protocol_value_validates_parent_source_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    design = tmp_path / "design.json"
    design.write_text("{}", encoding="utf-8")
    calls: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C.V1.CAPTURE, "validate_design", lambda *_a, **_k: {})
    monkeypatch.setattr(C, "_git", lambda *args: b"b" * 40 + b"\n")
    monkeypatch.setattr(
        C.V1.CAPTURE, "validate_source_anchor",
        lambda path, source, execution: calls.append((Path(path), source, execution)),
    )
    monkeypatch.setattr(C, "SOURCE_FILES", ())
    C.protocol_value(design, "a" * 40)
    assert calls == [(design, "a" * 40, "b" * 40)]


def test_protocol_value_propagates_parent_anchor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    design = tmp_path / "design.json"
    design.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C.V1.CAPTURE, "validate_design", lambda *_a, **_k: {})
    monkeypatch.setattr(C, "_git", lambda *args: b"b" * 40 + b"\n")
    monkeypatch.setattr(
        C.V1.CAPTURE, "validate_source_anchor",
        lambda *_a: (_ for _ in ()).throw(ValueError("parent blob mismatch")),
    )
    with pytest.raises(ValueError, match="parent blob mismatch"):
        C.protocol_value(design, "a" * 40)


def test_selected_second_uses_outcome_and_training_scopes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def save(name: str, value: dict) -> Path:
        path = tmp_path / name
        path.write_text(__import__("json").dumps(value), encoding="utf-8")
        return path

    candidate = save("candidate.json", {"inputs": {}})
    receipt = save("receipt.json", {"candidate": C.V1.B._ref(candidate)})
    result = save("result.json", {"receipt": C.V1.B._ref(receipt)})
    second = save("second.json", {"records": [{"case_id": "case-07", "policy": "logs_only",
                                                "outcome": "recovered", "result": C.V1.B._ref(result),
                                                "candidate": C.V1.B._ref(candidate)}]})
    loop = save("loop.json", {})
    second_plan = save("second-plan.json", {"loop_protocol": C.V1.B._ref(loop), "anchor": "l" * 40})
    parent = save("parent.json", {})
    parent_private = save("parent-private.json", {})
    training = save("training.json", {})
    active: set[str] = set()

    @contextmanager
    def outcome_scope():
        active.add("outcome")
        try:
            yield
        finally:
            active.remove("outcome")

    @contextmanager
    def training_scope(**_kwargs: object):
        active.add("training")
        try:
            yield
        finally:
            active.remove("training")

    monkeypatch.setattr(C.EXPORT2.OUTCOME, "recovery_outcome_scope", outcome_scope)
    monkeypatch.setattr(C.TRAIN_BOUNDARY, "training_boundary_scope", training_scope)
    monkeypatch.setattr(C.V9.V8, "verify_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(C.V9.V8, "_parents", lambda *_a: ({}, {}, {}, second_plan, []))
    monkeypatch.setattr(
        C.EXPORT2, "_training",
        lambda *_a, **_k: ({"fixed_conditions": 26, "fresh_initialization": True}
                            if active == {"outcome", "training"}
                            else pytest.fail("required scopes are not active")),
    )
    value = C._verify_selected_second(
        second,
        v9_plan={"parent_protocol": C.V1.B._ref(parent), "parent_private_plan": C.V1.B._ref(parent_private),
                 "parent_anchor": "p" * 40, "training_protocol": C.V1.B._ref(training),
                 "training_anchor": "t" * 40},
        lab=tmp_path,
    )
    assert value["fixed_conditions"] == 26 and active == set()
