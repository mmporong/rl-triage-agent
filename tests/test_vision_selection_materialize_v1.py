from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import test_vision_selection_sources_v1 as SOURCE_FIXTURE

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("vision_selection_materialize_v1")
full_case = SOURCE_FIXTURE.full_case


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n") if lf else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(repo).as_posix(), "sha256": _sha(path)}


def _install_fakes(monkeypatch: pytest.MonkeyPatch, repo: Path, proofs: dict[str, dict[str, Any]]) -> None:
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    monkeypatch.setattr(M.VIEW, "checked", lambda value: value)
    monkeypatch.setattr(M.INPUTS, "camera_pose_audit", lambda path: {
        "schema": M.INPUTS.CAMERA_SCHEMA, "trace_sha256": _sha(path),
    })

    def prepare(spec_path: Path, output: Path) -> Path:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        checkpoint = Path(spec["checkpoint"]["path"])
        video = Path(spec["video"]["path"])
        measurement = Path(spec["measurement"]["path"])
        contract = output / "contract.json"
        _write(contract, {"provenance": {
            "source_lineage_id": f"lineage-{_sha(checkpoint)[:12]}",
            "parent_source_lineage_id": None,
            "checkpoint": {"sha256": _sha(checkpoint)},
            "video": {"sha256": _sha(video)},
            "measurement": {"sha256": _sha(measurement)},
        }, "frames": [{"sha256": str(index) * 64} for index in range(1, 7)]})
        return contract.resolve()

    monkeypatch.setattr(M.VISION, "prepare", prepare)
    monkeypatch.setattr(M.VISION, "verify", lambda path: json.loads(path.read_text(encoding="utf-8")))

    def freeze(snapshot: dict, contract: Path, binding: dict, view: str, folder: Path) -> dict[str, str]:
        del contract, binding
        messages = [{"role": "user", "content": [{
            "type": "text", "text": json.dumps(
                snapshot.get("leak", {"probe_expectation": "normal", "view": view}), sort_keys=True,
            ),
        }]}]
        _write(folder / "bundle.json", {"view_kind": view})
        _write(folder / "messages.json", {"messages": messages})
        methods = {name: _sha(repo / name) for name in M.CALL._METHODS}
        _write(folder / "capsule.json", {
            "schema": M.CALL.CAPSULE_SCHEMA,
            "bundle_sha256": _sha(folder / "bundle.json"),
            "messages_sha256": _sha(folder / "messages.json"),
            "method_sha256": methods,
        })
        capsule = (folder / "capsule.json").resolve()
        return {"path": str(capsule), "sha256": _sha(capsule), "status": "frozen"}

    def prepared(reference: dict) -> dict[str, Any]:
        capsule = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
        messages = json.loads((Path(reference["path"]).parent / "messages.json").read_text(encoding="utf-8"))
        bundle = json.loads((Path(reference["path"]).parent / "bundle.json").read_text(encoding="utf-8"))
        return {"capsule": capsule, "messages": messages["messages"], "bundle": bundle}

    monkeypatch.setattr(M.CALL, "freeze", freeze)
    monkeypatch.setattr(M.CALL, "prepared", prepared)
    monkeypatch.setattr(M, "_fixture_proofs", proofs, raising=False)


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir(parents=True)
    for relative in {*M.WORKER.REQUIRED_SOURCES, M.SELF, *M.CALL._METHODS}:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    method_path = repo / "bench/protocols/method.json"
    method_sources = {name: _sha(repo / name, lf=True) for name in M.CALL._METHODS}
    _write(method_path, {
        "schema": "vision_selection_comparison_method_v1",
        "source_sha256_lf": method_sources,
        "evidence_sha256_lf": {},
        "inputs": {"slots": M.COMPARISON.design()},
    })
    source_path = repo / "bench/protocols/source.json"
    _write(source_path, {
        "schema": "vision_brev_source_protocol_v1", "source_sha256_lf": {},
        "recording": {"source_files_lf": {}},
    })
    policies = []
    none_bindings: dict[str, dict[str, str]] = {}
    proofs: dict[str, dict[str, Any]] = {}
    for index, (group, role) in enumerate(M.POLICY_ORDER):
        folder = repo / f"actual/{group}-{role}"
        checkpoint = folder / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{group}-{role}".encode())
        telemetry_sha = hashlib.sha256(f"telemetry-{group}-{role}".encode()).hexdigest()
        training_artifact = folder / "training_artifact.json"
        _write(training_artifact, {
            "checkpoint": {"sha256": _sha(checkpoint)},
            "telemetry": {"sha256": telemetry_sha},
            "seed": 5101 if group == "source-01" else 5102,
            "arm": "NONE" if role == "normal" else "F1",
        })
        video = folder / "same_play.mp4"
        video.write_bytes(f"video-{group}-{role}".encode())
        trace = folder / "trace.json"
        _write(trace, {"trace": f"{group}-{role}"})
        measurements, specs = {}, {}
        for window in (1, 2):
            measurement = folder / f"measurement_{window}.json"
            _write(measurement, {"window": window})
            measurements[window] = measurement
            spec = folder / f"spec_{window}.json"
            _write(spec, {
                "video": {"path": "/remote/video"},
                "checkpoint": {"path": "/remote/checkpoint"},
                "measurement": {"path": "/remote/measurement"},
            })
            specs[window] = spec
        readiness = folder / "readiness.json"
        _write(readiness, {
            "source_training_artifact": _ref(repo, training_artifact),
            "record": {
                "trace": _ref(repo, trace), "video": _ref(repo, video),
                "measurement_1": _ref(repo, measurements[1]), "measurement_2": _ref(repo, measurements[2]),
                "spec_1": _ref(repo, specs[1]), "spec_2": _ref(repo, specs[2]),
            },
        })
        case_binding = {"checkpoint_sha256": _sha(checkpoint), "telemetry_sha256": telemetry_sha}
        if role == "normal":
            none_bindings[group] = case_binding
        snapshot = folder / "snapshot.json"
        _write(snapshot, {"inputs": {"packet": {"input_bindings": {
            "case": case_binding, "reference": none_bindings[group],
        }}}})
        seed = 5101 if group == "source-01" else 5102
        proofs[str(readiness.resolve())] = {
            "source_role": role, "source_seed": seed,
            "checkpoint_sha256": _sha(checkpoint), "video_sha256": _sha(video),
            "trace_sha256": _sha(trace),
            "measurement_sha256": {
                "window-01": _sha(measurements[1]), "window-02": _sha(measurements[2]),
            },
        }
        policies.append({
            "seed_group": group, "source_role": role,
            "model_readiness": _ref(repo, readiness), "checkpoint": _ref(repo, checkpoint),
            "snapshot": _ref(repo, snapshot),
        })
    manifest_path = repo / "private/actual.json"
    manifest = {
        "schema": M.MANIFEST_SCHEMA, "protocol_id": "selection-materialized-v1",
        "execution_id": "selection-materialized-v1",
        "method_protocol": {"path": method_path.relative_to(repo).as_posix(),
                            "sha256_lf": _sha(method_path, lf=True)},
        "source_protocol": {"path": source_path.relative_to(repo).as_posix(),
                            "sha256_lf": _sha(source_path, lf=True)},
        "policies": policies,
    }
    _write(manifest_path, manifest)
    _install_fakes(monkeypatch, repo, proofs)
    return {"repo": repo, "manifest_path": manifest_path, "manifest": manifest,
            "proofs": proofs, "output": repo / "private/inputs", "protocol": repo / "bench/protocols/run.json"}


def _run(case: dict[str, Any]) -> dict[str, Any]:
    def readiness(repo: Path, path: Path, source: dict) -> dict:
        del repo, source
        return case["proofs"][str(path.resolve())]

    def verify(repo: Path, protocol: dict, locations: Path) -> dict[str, Any]:
        del repo, locations
        checked = M.COMPARISON.validate_inputs(
            protocol["rows"],
            method_protocol_sha256=protocol["method_protocol"]["sha256_lf"],
            source_protocol_sha256=protocol["source_protocol"]["sha256_lf"],
        )
        return {"schema": M.INPUTS.VERIFIED_SCHEMA, "rows": checked}

    return M.materialize(
        case["repo"], case["manifest_path"], case["output"], case["protocol"],
        readiness_verifier=readiness, input_verifier=verify,
    )


def _production_manifest(case: dict[str, Any]) -> Path:
    repo = case["repo"]
    for relative in {*M.WORKER.REQUIRED_SOURCES, M.SELF}:
        target = repo / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
    method_path = repo / "bench/protocols/materialized_method.json"
    _write(method_path, {
        "schema": "vision_selection_comparison_method_v1",
        "source_sha256_lf": {name: _sha(repo / name, lf=True) for name in M.CALL._METHODS},
        "evidence_sha256_lf": {},
        "inputs": {"slots": M.COMPARISON.design()},
    })
    source_ref = case["protocol"]["source_protocol"]
    source_path = repo / source_ref["path"]
    policies = []
    none_bindings: dict[str, dict[str, str]] = {}
    for group, role in M.POLICY_ORDER:
        source = case["role_data"][(group, role)]
        readiness = json.loads(source["readiness"].read_text(encoding="utf-8"))
        artifact_path = repo / readiness["source_training_artifact"]["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        case_binding = {
            "checkpoint_sha256": artifact["checkpoint"]["sha256"],
            "telemetry_sha256": artifact["telemetry"]["sha256"],
        }
        if role == "normal":
            none_bindings[group] = case_binding
        template_path = repo / f"inputs/input-{len(policies) * 2 + 1:02d}/snapshot.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        packet = template["inputs"]["packet"]
        packet["input_bindings"] = {"case": case_binding, "reference": none_bindings[group]}
        snapshot = M.VIEW.build_snapshot(
            packet,
            f"materialized_{group.replace('-', '_')}_{role.replace('-', '_')}",
            "model_adaptive",
            SOURCE_FIXTURE.SELECTION_FIXTURE.VIEW_SHA,
            repo / f"materialized/snapshots/{group}-{role}/probes.ledger.jsonl",
        )
        snapshot_path = repo / f"materialized/snapshots/{group}-{role}/snapshot.json"
        _write(snapshot_path, snapshot)
        policies.append({
            "seed_group": group, "source_role": role,
            "model_readiness": _ref(repo, source["readiness"]),
            "checkpoint": _ref(repo, source["checkpoint"]),
            "snapshot": _ref(repo, snapshot_path),
        })
    manifest_path = repo / "private/materialized_actual.json"
    _write(manifest_path, {
        "schema": M.MANIFEST_SCHEMA,
        "protocol_id": "selection-materialized-production-v1",
        "execution_id": "selection-materialized-production-v1",
        "method_protocol": {"path": method_path.relative_to(repo).as_posix(),
                            "sha256_lf": _sha(method_path, lf=True)},
        "source_protocol": {"path": source_path.relative_to(repo).as_posix(),
                            "sha256_lf": _sha(source_path, lf=True)},
        "policies": policies,
    })
    return manifest_path


def test_materializes_fixed_24_order_without_label_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)
    result = _run(case)
    protocol = json.loads(case["protocol"].read_text(encoding="utf-8"))

    assert result["rows"] == 24 and len(protocol["rows"]) == 24
    assert [row["call_id"] for row in protocol["rows"]] == [f"choice-{index:02d}" for index in range(1, 25)]
    assert [row["view_kind"] for row in protocol["rows"][:3]] == M.COMPARISON.design()[0]["view_order"]
    assert {row["source_seed"] for row in protocol["rows"]} == {5101, 5102}


def test_rejects_mutated_readiness_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)
    readiness = case["repo"] / case["manifest"]["policies"][0]["model_readiness"]["path"]
    readiness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model readiness SHA"):
        _run(case)


def test_rejects_checkpoint_different_from_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)
    readiness = case["repo"] / case["manifest"]["policies"][0]["model_readiness"]["path"]
    case["proofs"][str(readiness.resolve())]["checkpoint_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="role/checkpoint"):
        _run(case)


def test_rejects_model_visible_label_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)
    policy = case["manifest"]["policies"][0]
    snapshot = case["repo"] / policy["snapshot"]["path"]
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    value["leak"] = {"source_role": "normal"}
    _write(snapshot, value)
    policy["snapshot"]["sha256"] = _sha(snapshot)
    _write(case["manifest_path"], case["manifest"])

    with pytest.raises(ValueError, match="노출"):
        _run(case)


@pytest.mark.parametrize("leak_kind", ["source_seed", "reference_checkpoint", "reference_telemetry"])
def test_rejects_seed_key_or_reference_artifact_value_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leak_kind: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    policy = case["manifest"]["policies"][1]
    snapshot = case["repo"] / policy["snapshot"]["path"]
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    reference = value["inputs"]["packet"]["input_bindings"]["reference"]
    leaks: dict[str, object] = {
        "source_seed": {"source_seed": 5101},
        "reference_checkpoint": {"observation": reference["checkpoint_sha256"]},
        "reference_telemetry": {"observation": reference["telemetry_sha256"]},
    }
    value["leak"] = leaks[leak_kind]
    _write(snapshot, value)
    policy["snapshot"]["sha256"] = _sha(snapshot)
    _write(case["manifest_path"], case["manifest"])

    with pytest.raises(ValueError, match="노출"):
        _run(case)


def test_rejects_telemetry_or_none_reference_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    policy = case["manifest"]["policies"][1]
    snapshot = case["repo"] / policy["snapshot"]["path"]
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    bindings = value["inputs"]["packet"]["input_bindings"]
    bindings["reference"] = bindings["case"]
    _write(snapshot, value)
    policy["snapshot"]["sha256"] = _sha(snapshot)
    _write(case["manifest_path"], case["manifest"])

    with pytest.raises(ValueError, match="case/reference"):
        _run(case)


def test_write_once_and_changed_policy_order_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _run(case)
    with pytest.raises(FileExistsError):
        _run(case)

    second = _case(tmp_path / "second", monkeypatch)
    second["manifest"]["policies"][0], second["manifest"]["policies"][1] = (
        second["manifest"]["policies"][1], second["manifest"]["policies"][0],
    )
    _write(second["manifest_path"], second["manifest"])
    with pytest.raises(ValueError, match="순서"):
        _run(second)


def test_full_v04_fixture_materializes_and_passes_production_verifier(full_case: dict[str, Any]) -> None:
    repo = full_case["repo"]
    manifest = _production_manifest(full_case)
    result = M.materialize(
        repo, manifest, repo / "materialized/inputs", repo / "bench/protocols/materialized_run.json",
    )
    protocol = json.loads((repo / "bench/protocols/materialized_run.json").read_text(encoding="utf-8"))
    locations = repo / result["input_locations"]["path"]
    verified = M.INPUTS.verify_inputs(repo, protocol, locations)

    assert result["rows"] == 24
    assert len(verified["rows"]) == 24
    assert [row["call_id"] for row in verified["rows"]] == [f"choice-{index:02d}" for index in range(1, 25)]

    actual = json.loads(manifest.read_text(encoding="utf-8"))
    f1 = actual["policies"][1]
    snapshot_path = repo / f1["snapshot"]["path"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    inputs = snapshot["inputs"]
    inputs["packet"]["input_bindings"]["reference"] = inputs["packet"]["input_bindings"]["case"]
    _write(snapshot_path, M.VIEW.build_snapshot(**inputs))
    f1["snapshot"]["sha256"] = _sha(snapshot_path)
    _write(manifest, actual)
    with pytest.raises(ValueError, match="case/reference"):
        M.materialize(
            repo, manifest, repo / "materialized/tampered-inputs",
            repo / "bench/protocols/materialized_tampered.json",
        )
