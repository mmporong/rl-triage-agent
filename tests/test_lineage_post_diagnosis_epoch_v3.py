from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_epoch_v3 as V3


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_prepare_protocol_binds_v2_and_parent_scope_sources(tmp_path, monkeypatch):
    parent = write(tmp_path / "v2.json", {"schema": V3.V2.PROTOCOL_SCHEMA})
    monkeypatch.setattr(V3, "_relative", lambda _path: "bench/protocols/v2.json")
    value = V3.prepare_protocol(parent)
    assert value["schema"] == V3.PROTOCOL_SCHEMA
    assert value["scope"]["change"] == "epoch2_exact_parent_binding_only"
    assert set(value["source_sha256_lf"]) == set(V3.SOURCE_FILES)


def test_v3_wraps_v2_in_parent_scope_and_keeps_result_schema(tmp_path, monkeypatch):
    out = tmp_path / "run"
    v3, v2, parent, post, reference = [tmp_path / name for name in (
        "v3.json", "v2.json", "parent.json", "post.json", "reference.json")]
    for path in (v3, v2, parent, post, reference):
        write(path, {})
    monkeypatch.setattr(V3, "ROOT", tmp_path)
    monkeypatch.setattr(V3, "verify_protocol", lambda *_args, **_kwargs: {
        "component_v2_protocol": {"path": "v2.json", "sha256_lf": "1" * 64}})
    context = {"packet_digest": "2" * 64, "run_id": "epoch2_registry",
               "policy": "registry_next", "parent": {"epoch": 1}}
    entered = []

    @contextmanager
    def scope(*args):
        entered.append(args)
        yield context

    monkeypatch.setattr(V3.PARENT, "parent_scope", scope)

    def v2_run(actual_out, *_args, **kwargs):
        assert entered and kwargs["run_id"] == "epoch2_registry"
        actual_out.mkdir()
        result = {"schema": V3.V2.E2.SCHEMA, "epoch": 2}
        write(actual_out / "result.json", result)
        return result

    monkeypatch.setattr(V3.V2, "run_epoch", v2_run)
    result = V3.run_epoch(
        out, post, reference, "registry_next", v3_protocol=v3, v3_anchor="a" * 40,
        wrapper_protocol=v2, wrapper_anchor="a" * 40, epoch_protocol=parent,
        anchor="a" * 40, run_id="epoch2_registry", lab=tmp_path)
    assert result["schema"] == V3.V2.E2.SCHEMA
    receipt = V3._read(out / "v3-adapter-receipt.json")
    assert receipt["parent_context"] == context
    assert receipt["post_observation_sha256"] == V3._sha(post)


def test_v3_rejects_anchor_or_v2_component_mismatch(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="anchor"):
        V3.run_epoch(
            tmp_path / "out", tmp_path / "post", tmp_path / "reference", "registry_next",
            v3_protocol=tmp_path / "v3", v3_anchor="a" * 40,
            wrapper_protocol=tmp_path / "v2", wrapper_anchor="b" * 40,
            epoch_protocol=tmp_path / "parent", anchor="a" * 40, run_id="epoch2")
    v3, v2, other = tmp_path / "v3", tmp_path / "v2", tmp_path / "other"
    for path in (v3, v2, other):
        write(path, {})
    monkeypatch.setattr(V3, "ROOT", tmp_path)
    monkeypatch.setattr(V3, "verify_protocol", lambda *_args, **_kwargs: {
        "component_v2_protocol": {"path": "other", "sha256_lf": "1" * 64}})
    with pytest.raises(ValueError, match="v2 protocol"):
        V3.run_epoch(
            tmp_path / "out2", tmp_path / "post", tmp_path / "reference", "registry_next",
            v3_protocol=v3, v3_anchor="a" * 40, wrapper_protocol=v2,
            wrapper_anchor="a" * 40, epoch_protocol=other, anchor="a" * 40,
            run_id="epoch2", lab=tmp_path)
