from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_epoch_v2 as V2


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_prepare_protocol_binds_parent_without_changing_parent(tmp_path, monkeypatch):
    parent = write(tmp_path / "parent.json", {"schema": V2.E2.PROTOCOL_SCHEMA})
    monkeypatch.setattr(V2, "_relative", lambda _path: "bench/protocols/parent.json")
    value = V2.prepare_protocol(parent)
    assert value["schema"] == V2.PROTOCOL_SCHEMA
    assert value["scope"]["change"] == "preflight_evidence_sibling_path_only"
    assert value["scope"]["parent_result_schema_unchanged"] is True
    assert value["source_sha256_lf"]["evals/lineage_post_diagnosis_epoch_v1.py"]


def test_v1_internal_evidence_path_reproduces_fresh_out_collision(tmp_path):
    out = tmp_path / "run"

    @contextmanager
    def old_scope(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        yield

    with old_scope(out / "lease-read-evidence.json"):
        pass
    with pytest.raises(FileExistsError):
        out.mkdir(exist_ok=False)


def test_v2_redirect_keeps_out_fresh_until_parent_runner(tmp_path, monkeypatch):
    out = tmp_path / "run"
    observed = []

    @contextmanager
    def old_scope(path: Path):
        observed.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        yield

    monkeypatch.setattr(V2.E2.RETRY, "lease_read_scope", old_scope)
    with V2._preflight_redirect(out) as sibling:
        with V2.E2.RETRY.lease_read_scope(out / "lease-read-evidence.json"):
            pass
        assert not out.exists()
        out.mkdir(exist_ok=False)
    assert observed == [sibling]
    assert sibling == tmp_path / "run.lease-read-preflight.json"


def test_redirect_rejects_any_unregistered_requested_path(tmp_path, monkeypatch):
    @contextmanager
    def old_scope(_path: Path):
        yield

    out = tmp_path / "run"
    monkeypatch.setattr(V2.E2.RETRY, "lease_read_scope", old_scope)
    with (V2._preflight_redirect(out),
          pytest.raises(ValueError, match="등록 내부 경로"),
          V2.E2.RETRY.lease_read_scope(tmp_path / "other.json")):
        pass


def test_redirect_restores_parent_scope_after_exception(tmp_path, monkeypatch):
    @contextmanager
    def old_scope(_path: Path):
        yield

    monkeypatch.setattr(V2.E2.RETRY, "lease_read_scope", old_scope)
    with pytest.raises(RuntimeError, match="stop"), V2._preflight_redirect(tmp_path / "run"):
        raise RuntimeError("stop")
    assert V2.E2.RETRY.lease_read_scope is old_scope


def test_run_epoch_uses_default_parent_pipeline_and_writes_wrapper_receipt(
        tmp_path, monkeypatch):
    out = tmp_path / "run"
    wrapper = tmp_path / "wrapper.json"
    parent = tmp_path / "parent.json"
    post, reference = tmp_path / "post.json", tmp_path / "reference.json"
    for path in (wrapper, parent, post, reference):
        write(path, {})
    protocol = {"component_parent_protocol": {"path": "parent.json", "sha256_lf": "a" * 64}}
    monkeypatch.setattr(V2, "ROOT", tmp_path)
    monkeypatch.setattr(V2, "verify_protocol", lambda *_args, **_kwargs: protocol)

    @contextmanager
    def old_scope(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        yield

    monkeypatch.setattr(V2.E2.RETRY, "lease_read_scope", old_scope)
    calls = []

    def parent_run(actual_out, *_args, **kwargs):
        calls.append(kwargs)
        with V2.E2.RETRY.lease_read_scope(actual_out / "lease-read-evidence.json"):
            pass
        actual_out.mkdir(exist_ok=False)
        backend_evidence = actual_out / "backend-stage/lease-reads/1.json"
        with V2.E2.RETRY.lease_read_scope(backend_evidence):
            pass
        assert backend_evidence.is_file()
        result = {"schema": V2.E2.SCHEMA, "epoch": 2, "evidence_scope": "actual_backend"}
        write(actual_out / "result.json", result)
        return result

    monkeypatch.setattr(V2.E2, "run_epoch", parent_run)
    result = V2.run_epoch(
        out, post, reference, "registry_next", wrapper_protocol=wrapper,
        wrapper_anchor="a" * 40, epoch_protocol=parent, anchor="a" * 40, lab=tmp_path)
    assert result["schema"] == V2.E2.SCHEMA and len(calls) == 1
    receipt = V2._read(out / "v2-wrapper-receipt.json")
    assert receipt["preflight_inside_output"] is False
    assert receipt["diagnosis_replayed"] is receipt["training_replayed"] is False
    assert Path(receipt["preflight_evidence"]["path"]).parent == tmp_path


def test_run_rejects_parent_protocol_or_anchor_mismatch(tmp_path, monkeypatch):
    wrapper, parent = tmp_path / "wrapper.json", tmp_path / "parent.json"
    write(wrapper, {})
    write(parent, {})
    with pytest.raises(ValueError, match="anchor"):
        V2.run_epoch(tmp_path / "out", tmp_path / "post", tmp_path / "reference",
                     "registry_next", wrapper_protocol=wrapper, wrapper_anchor="a" * 40,
                     epoch_protocol=parent, anchor="b" * 40)
    monkeypatch.setattr(V2, "verify_protocol", lambda *_args, **_kwargs: {
        "component_parent_protocol": {"path": "other.json", "sha256_lf": "a" * 64}})
    monkeypatch.setattr(V2, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="parent protocol"):
        V2.run_epoch(tmp_path / "out2", tmp_path / "post", tmp_path / "reference",
                     "registry_next", wrapper_protocol=wrapper, wrapper_anchor="a" * 40,
                     epoch_protocol=parent, anchor="a" * 40, lab=tmp_path)
