from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_second_recovery_v2 as M


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": M._sha(path)}


def test_prepare_protocol_is_thin_parent_adapter_only() -> None:
    inner = ROOT / "bench/protocols/lineage_compound_second_recovery_20261031_v1.json"
    if not inner.is_file():
        pytest.skip("main has not registered the S1 protocol yet")
    protocol = M.prepare_protocol(inner)
    assert protocol["schema"] == M.SCHEMA
    assert protocol["adapter_contract"]["inner_manifest_schema_unchanged"] == M.S1.MANIFEST_SCHEMA
    assert protocol["adapter_contract"]["inner_result_schema_unchanged"] == M.S1.RESULT_SCHEMA
    assert protocol["adapter_contract"]["training_path_unchanged"] == "S1_BOTH_fresh300_fixed26"
    assert protocol["adapter_contract"]["arbitrary_manifest_rejected"] is True
    assert set(M.SOURCE_FILES) == {
        "evals/lineage_post_parent_binding_v1.py",
        "tests/test_lineage_post_parent_binding_v1.py",
        "evals/lineage_compound_second_recovery_v2.py",
        "tests/test_lineage_compound_second_recovery_v2.py",
    }


def test_prepare_adapter_receipt_binds_manifest_epoch_and_parent_context(tmp_path: Path) -> None:
    protocol = write(tmp_path / "wrapper.json", {})
    manifest = write(tmp_path / "manifest.json", {"schema": M.S1.MANIFEST_SCHEMA})
    epoch = write(tmp_path / "epoch.json", {"schema": M.S1.E2.SCHEMA})
    context = {"run_id": "post_registry", "policy": "registry_next",
               "parent_binding": {"packet_digest": "a" * 64}}
    value = M.prepare_adapter_receipt(
        protocol, "b" * 40, manifest, epoch, context, tmp_path / M.PREPARE_RECEIPT)
    assert value["inner_manifest"] == ref(manifest)
    assert value["epoch2_result"] == ref(epoch)
    assert value["context_sha256"] == M._digest(context)
    assert value["candidate_revalidated_in_parent_scope"] is True


def test_prepare_inputs_runs_s1_and_full_candidate_inside_exact_parent_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inner = ROOT / "bench/protocols/lineage_compound_first_recovery_20261031_v1.json"
    post = write(tmp_path / "post.json", {})
    reference = write(tmp_path / "reference.json", {})
    epoch = write(tmp_path / "epoch.json", {"schema": M.S1.E2.SCHEMA,
        "epoch": 2, "run_id": "post_registry", "policy": "registry_next"})
    wrapper = write(tmp_path / "wrapper.json", {})
    out = tmp_path / "inputs"
    calls: list[str] = []
    base_context = {"post_input": ref(post), "reference_input": ref(reference),
                    "epoch2_result": ref(epoch), "run_id": "post_registry",
                    "policy": "registry_next"}
    parent = {"packet_digest": "a" * 64, "run_id": "post_registry",
              "policy": "registry_next"}

    @contextmanager
    def parent_scope(*args):
        assert args == (post, reference, "post_registry", "registry_next")
        calls.append("scope-enter")
        yield parent
        calls.append("scope-exit")

    def s1_prepare(output: Path, *_args, **_kwargs):
        calls.append("s1-prepare")
        value = {"schema": M.S1.MANIFEST_SCHEMA, "row": {"status": "candidate"}}
        write(output / "manifest.json", value)
        return value

    monkeypatch.setattr(M, "verify_protocol", lambda *_a, **_k: {
        "component_protocols": {"second_recovery_s1": {
            "path": M._relative(inner), "sha256_lf": M._sha(inner, lf=True)}}})
    monkeypatch.setattr(M, "_context", lambda *_a: base_context)
    monkeypatch.setattr(M.PARENT, "parent_scope", parent_scope)
    monkeypatch.setattr(M.S1, "prepare_inputs", s1_prepare)
    monkeypatch.setattr(M.S1, "_checked_candidate",
                        lambda *_a: calls.append("candidate-revalidated"))
    manifest = M.prepare_inputs(
        wrapper_protocol=wrapper, wrapper_anchor="b" * 40, out=out,
        protocol_path=inner, post_input=post, epoch2_result_path=epoch,
        reference_input=reference, first_manifest_path=tmp_path / "first.json",
        first_smoke_root=tmp_path / "smoke", gate_path=tmp_path / "gate.json",
        gate_sha256="c" * 64, lab=tmp_path, isaac_python=Path(sys.executable),
        execution_anchor="d" * 40, diagnosis_anchor="e" * 40)
    assert manifest["schema"] == M.S1.MANIFEST_SCHEMA
    assert calls == ["scope-enter", "s1-prepare", "candidate-revalidated", "scope-exit"]
    receipt = M._read(out / M.PREPARE_RECEIPT)
    assert receipt["context"]["parent_binding"] == parent
    assert receipt["context_sha256"] == M._digest(receipt["context"])


def test_prepare_inputs_hold_still_requires_parent_scope_but_skips_candidate_rebuild(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inner = ROOT / "bench/protocols/lineage_compound_first_recovery_20261031_v1.json"
    post, reference = write(tmp_path / "post.json", {}), write(tmp_path / "reference.json", {})
    epoch = write(tmp_path / "epoch.json", {})
    wrapper = write(tmp_path / "wrapper.json", {})
    base = {"post_input": ref(post), "reference_input": ref(reference),
            "epoch2_result": ref(epoch), "run_id": "r", "policy": "registry_next"}

    @contextmanager
    def scope(*_args):
        yield {"run_id": "r", "policy": "registry_next"}

    monkeypatch.setattr(M, "verify_protocol", lambda *_a, **_k: {
        "component_protocols": {"second_recovery_s1": {
            "path": M._relative(inner), "sha256_lf": M._sha(inner, lf=True)}}})
    monkeypatch.setattr(M, "_context", lambda *_a: base)
    monkeypatch.setattr(M.PARENT, "parent_scope", scope)

    def prepare(output: Path, *_args, **_kwargs):
        value = {"schema": M.S1.MANIFEST_SCHEMA, "row": {"status": "HOLD"}}
        write(output / "manifest.json", value)
        return value

    monkeypatch.setattr(M.S1, "prepare_inputs", prepare)
    monkeypatch.setattr(M.S1, "_checked_candidate",
                        lambda *_a: pytest.fail("HOLD must not rebuild candidate"))
    M.prepare_inputs(
        wrapper_protocol=wrapper, wrapper_anchor="b" * 40, out=tmp_path / "out",
        protocol_path=inner, post_input=post, epoch2_result_path=epoch,
        reference_input=reference, first_manifest_path=tmp_path / "first.json",
        first_smoke_root=tmp_path / "smoke", gate_path=tmp_path / "gate.json",
        gate_sha256="c" * 64, lab=tmp_path, isaac_python=Path(sys.executable),
        execution_anchor="d" * 40, diagnosis_anchor="e" * 40)


def test_execute_rejects_manifest_without_prepare_sidecar(tmp_path: Path) -> None:
    manifest = write(tmp_path / "manifest.json", {"schema": M.S1.MANIFEST_SCHEMA})
    with pytest.raises(FileNotFoundError):
        M.execute(manifest, tmp_path / "approval.json", tmp_path / "out")


def test_execute_wraps_s1_in_same_parent_and_writes_execution_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write(tmp_path / "manifest.json", {"schema": M.S1.MANIFEST_SCHEMA})
    approval = write(tmp_path / "approval.json", {})
    wrapper = write(tmp_path / "wrapper.json", {})
    prepare_receipt = write(tmp_path / M.PREPARE_RECEIPT, {})
    context = {"post_input": {"path": str(tmp_path / "post.json")},
               "reference_input": {"path": str(tmp_path / "reference.json")},
               "run_id": "post_registry", "policy": "registry_next",
               "parent_binding": {"packet_digest": "a" * 64}}
    receipt = {"wrapper_protocol": ref(wrapper), "wrapper_anchor": "b" * 40,
               "context_sha256": M._digest(context)}
    calls: list[str] = []

    @contextmanager
    def scope(*_args):
        calls.append("enter")
        yield context["parent_binding"]
        calls.append("exit")

    def execute_s1(_manifest: Path, _approval: Path, out: Path, **_kwargs):
        calls.append("s1")
        value = {"schema": M.S1.RESULT_SCHEMA, "status": "complete"}
        write(out / "result.json", value)
        return value

    monkeypatch.setattr(M, "_checked_prepare_receipt", lambda _path: (receipt, context))
    monkeypatch.setattr(M.PARENT, "parent_scope", scope)
    monkeypatch.setattr(M.S1, "execute", execute_s1)
    result = M.execute(manifest, approval, tmp_path / "execution")
    assert calls == ["enter", "s1", "exit"]
    assert result["schema"] == M.S1.RESULT_SCHEMA
    adapter = M._read(tmp_path / "execution" / M.EXECUTION_RECEIPT)
    assert adapter["inner_result_schema"] == M.S1.RESULT_SCHEMA
    assert adapter["context_sha256"] == receipt["context_sha256"]
    assert adapter["prepare_adapter_receipt"] == ref(prepare_receipt)


def test_parent_mismatch_blocks_s1_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write(tmp_path / "manifest.json", {})
    approval = write(tmp_path / "approval.json", {})
    context = {"post_input": {"path": str(tmp_path / "post.json")},
               "reference_input": {"path": str(tmp_path / "reference.json")},
               "run_id": "r", "policy": "registry_next",
               "parent_binding": {"packet_digest": "a" * 64}}
    monkeypatch.setattr(M, "_checked_prepare_receipt", lambda _path: ({}, context))

    @contextmanager
    def scope(*_args):
        yield {"packet_digest": "b" * 64}

    monkeypatch.setattr(M.PARENT, "parent_scope", scope)
    monkeypatch.setattr(M.S1, "execute", lambda *_a, **_k: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="parent binding"):
        M.execute(manifest, approval, tmp_path / "out")


def test_cli_keeps_inner_s1_arguments_and_adds_wrapper_anchor() -> None:
    source = (ROOT / "evals/lineage_compound_second_recovery_v2.py").read_text(encoding="utf-8")
    for command in ("prepare-protocol", "prepare-inputs", "execute"):
        assert f'add_parser("{command}")' in source
    for item in ("wrapper-protocol", "wrapper-anchor", "diagnosis-anchor",
                 "first-smoke-root", "parent-s1-protocol"):
        assert item in source
