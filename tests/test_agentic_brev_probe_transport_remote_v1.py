from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import agentic_brev_probe_transport_remote_v1 as T  # pyright: ignore[reportMissingImports]


def save(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def row(resource: str = "vm-1", *, status: str = "RUNNING") -> dict:
    return {"build_status": "COMPLETED", "gpu": "L40S", "health_status": "HEALTHY", "id": resource,
            "instance_kind": "gpu", "instance_type": "gpu-l40s", "name": "probe-vm",
            "shell_status": "READY", "status": status}


@pytest.fixture
def fixture(tmp_path: Path):
    repo, control, parent, stage, source = (tmp_path / name for name in ("repo", "control", "parent", "stage", "source"))
    for path in (repo, control, parent, stage, source):
        path.mkdir()
    request_ref = save(tmp_path / "request.json", {"schema": "fixture"})
    op_id = "1" * 64
    operation = {"operation_id": op_id, "provider_packet_path": "/control/provider.json"}
    host_roots = [{"remote": remote, "local": str(tmp_path / "host" / name)} for remote, name in (
        ("/control", "control"), ("/parent", "parent"), ("/stage", "stage"), ("/source", "source"))]
    sources = {name: "fixture" for name in T.P.SOURCES}
    sources.update({name: T.P.C.digest_lf_file(T.ROOT / name) for name in (T.SOURCE, T.TEST_SOURCE)})
    prepared = {
        "status": "prepared", "repo": repo, "python": repo / "venv/bin/python", "folder": parent,
        "protocol": {"transport_root": {"base": "repo", "relative": "runtime/transport"},
                     "sources_lf": sources,
                     "provider_target": {"resource_id": "vm-1", "resource_name": "probe-vm",
                                         "instance_type": "gpu-l40s", "gpu_model": "L40S"}},
        "execution": {"pair_id": "2" * 64, "windows": {"admission_max_age_s": 30},
                      "runtime_host_binding": {"evidence_roots": {"NONE": host_roots, "F1": host_roots}}},
        "operations": {arm: {"op": {**operation, "arm": arm}, "unit": parent, "stage": stage}
                       for arm in T.ARMS},
        "resolver": T.P.E.Resolver([{"remote": "/control", "local": str(control)}],
                                    T.P.E.digest([{"remote": "/control", "local": str(control)}])),
    }
    prepared["python"].parent.mkdir(parents=True)
    prepared["python"].write_text("python")

    def preparer(ref, **kwargs):
        assert kwargs["allow_fixture"] is True
        assert ref == request_ref
        return prepared

    return SimpleNamespace(tmp=tmp_path, repo=repo, control=control, parent=parent, stage=stage,
                           source=source, request=Path(request_ref["path"]), request_ref=request_ref,
                           op_id=op_id, prepared=prepared, preparer=preparer)


def run(fixture, **kwargs):
    return T.run_transport(request=fixture.request, request_sha256=fixture.request_ref["sha256"],
                           allow_fixture=True, preparer=fixture.preparer, **kwargs)


def provider_payload(fixture, observed: datetime, *, inventory_raw: bytes | None = None,
                     operation: str | None = None) -> str:
    inventory_raw = inventory_raw or json.dumps({"workspaces": [row()]}, separators=(",", ":")).encode()
    packet = {"schema": T.PROVIDER_SCHEMA, "operation_id": operation or fixture.op_id,
              "observed_at_utc": observed.isoformat(),
              "inventory": {"path": "/control/provider.json.inventory.json",
                            "sha256": hashlib.sha256(inventory_raw).hexdigest()}}
    envelope = {"packet": packet, "inventory_base64": base64.b64encode(inventory_raw).decode()}
    return base64.b64encode(json.dumps(envelope, separators=(",", ":")).encode()).decode()


def test_prepare_is_read_only_and_actual_platform_check_is_stable(fixture, monkeypatch):
    value = run(fixture, mode="prepare")
    assert value["status"] == "prepared"
    assert not (fixture.repo / "runtime").exists()
    monkeypatch.setattr(T, "os", SimpleNamespace(name="nt"))
    with pytest.raises(ValueError, match="POSIX"):
        T.run_transport(request=fixture.request, request_sha256=fixture.request_ref["sha256"],
                        preparer=T.PRODUCTION[0])
    fixture.request.write_text('{"schema":"changed"}')
    with pytest.raises(ValueError, match="request raw SHA changed"):
        run(fixture, mode="prepare")


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_actual_fake_preparer_is_rejected_before_call_on_all_platforms(fixture, monkeypatch, platform):
    called = []

    def fake(*args, **kwargs):
        called.append((args, kwargs))
        return fixture.prepared

    monkeypatch.setattr(T, "os", SimpleNamespace(name=platform))
    with pytest.raises(ValueError, match="actual transport dependency injection forbidden"):
        T.run_transport(request=fixture.request, request_sha256=fixture.request_ref["sha256"], preparer=fake)
    assert called == []


@pytest.mark.parametrize("mode", T.MODES)
@pytest.mark.parametrize("injection", ["popen_factory", "identity_fn", "home", "now", "before_add"])
def test_every_actual_mode_rejects_nonproduction_injection(fixture, monkeypatch, mode, injection):
    monkeypatch.setattr(T, "os", SimpleNamespace(name="posix"))
    kwargs = {injection: {
        "popen_factory": lambda *args, **kw: None,
        "identity_fn": lambda pid: {"pid": pid},
        "home": fixture.tmp,
        "now": datetime(2026, 10, 28, tzinfo=timezone.utc),  # noqa: UP017 - Python 3.10
        "before_add": lambda path: None,
    }[injection]}
    with pytest.raises(ValueError, match="actual transport dependency injection forbidden"):
        T.run_transport(request=fixture.request, request_sha256=fixture.request_ref["sha256"],
                        mode=mode, **kwargs)


def test_prepare_requires_source_and_test_closure(fixture):
    fixture.prepared["protocol"]["sources_lf"].pop(T.TEST_SOURCE)
    with pytest.raises(ValueError, match="source/test closure"):
        run(fixture, mode="prepare")


def test_start_uses_exact_parent_command_and_duplicate_does_not_spawn(fixture, monkeypatch):
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=321)

    identity = {"pid": 321, "pgid": 321, "sid": 321, "start_ticks": 99, "posix_group_verified": True}
    monkeypatch.setattr(T.OWNED, "_identity_is_current", lambda value: value == identity)
    first = run(fixture, mode="start", popen_factory=popen, identity_fn=lambda pid: identity)
    second = run(fixture, mode="start", popen_factory=popen, identity_fn=lambda pid: identity)
    assert first["spawned"] is True and second["spawned"] is False and len(calls) == 1
    argv, options = calls[0]
    assert argv == [str(fixture.prepared["python"]), "-B", str(Path(T.P.__file__).resolve()), "--request",
                    str(fixture.request), "--request-sha256", fixture.request_ref["sha256"], "--execute"]
    assert options["start_new_session"] is True and options["stdin"] == T.subprocess.DEVNULL
    assert all(T.API_ENV.search(key) is None for key in options["env"])


def test_failed_identity_leaves_claim_and_never_relaunches(fixture):
    calls = []

    def popen(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(pid=55)

    with pytest.raises(ValueError, match="owned POSIX"):
        run(fixture, mode="start", popen_factory=popen,
            identity_fn=lambda pid: {"pid": pid, "posix_group_verified": False})
    value = run(fixture, mode="start", popen_factory=popen,
                identity_fn=lambda pid: {"pid": pid, "posix_group_verified": True})
    assert value["status"] == "unknown" and len(calls) == 1


def test_provider_write_once_validates_raw_inventory_and_freshness(fixture):
    now = datetime(2026, 10, 28, 1, 2, 3, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
    payload = provider_payload(fixture, now)
    first = run(fixture, mode="provider", arm="NONE", payload_base64=payload, now=now)
    second = run(fixture, mode="provider", arm="NONE", payload_base64=payload, now=now)
    assert first["status"] == "stored" and second["status"] == "existing"
    assert Path(first["inventory"]["path"]).read_bytes() == base64.b64decode(
        json.loads(base64.b64decode(payload))["inventory_base64"])
    changed = provider_payload(fixture, now, inventory_raw=json.dumps({"workspaces": [row(status="STOPPED")]}).encode())
    with pytest.raises(ValueError):
        run(fixture, mode="provider", arm="NONE", payload_base64=changed, now=now)
    stale = provider_payload(fixture, now - timedelta(seconds=31))
    with pytest.raises(ValueError, match="stale"):
        run(fixture, mode="provider", arm="F1", payload_base64=stale, now=now)


def make_completion(fixture):
    raw = fixture.stage / "records.jsonl"
    raw.write_bytes(b'{"update":1}\n')
    proof = save(fixture.parent / "proof.json", {"schema": "proof"})
    refs = {name: save(fixture.parent / f"{name}.json", {"schema": name}) for name in T.REQUIRED_PLAN_REFS - {"proof"}}
    roots = [{"remote": "/control", "local": str(fixture.control)},
             {"remote": "/parent", "local": str(fixture.parent)},
             {"remote": "/stage", "local": str(fixture.stage)},
             {"remote": "/source", "local": str(fixture.source)}]
    plan = {"operation_id": fixture.op_id, "roots": roots, "proof": proof, **refs}
    plan_ref = save(fixture.parent / "plan.json", plan)
    evidence_ref = save(fixture.parent / "evidence.json", {"raw_files_sha256": {"/stage/records.jsonl": T.sha_bytes(raw.read_bytes())}})
    outcome = {"status": "complete", "operation_id": fixture.op_id, "plan": plan_ref, "evidence": evidence_ref}
    outcome_ref = save(fixture.parent / "outcome.json", outcome)
    return outcome_ref, raw, plan


def test_status_reverifies_completion_without_writes(fixture, monkeypatch):
    outcome, _, _ = make_completion(fixture)
    seen = []
    monkeypatch.setattr(T.P, "_verify_completion", lambda prepared, value: seen.append(value))
    before = {path: path.read_bytes() for path in fixture.parent.iterdir() if path.is_file()}
    value = run(fixture, mode="status", arm="NONE")
    after = {path: path.read_bytes() for path in fixture.parent.iterdir() if path.is_file()}
    assert value["status"] == "completed" and value["operation_id"] == fixture.op_id and value["result"] == outcome
    assert len(seen) == 1 and before == after


def test_export_contains_only_manifest_and_bound_raw_bytes(fixture, monkeypatch):
    outcome, raw, plan = make_completion(fixture)
    monkeypatch.setattr(T.P, "_verify_completion", lambda prepared, value: None)
    output = fixture.tmp / "evidence.tar.gz"
    value = run(fixture, mode="export", arm="NONE", output=output)
    assert value["status"] == "exported" and value["authority_issued"] is False
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        assert names[0] == "manifest.json"
        assert all(name == "manifest.json" or name.startswith("files/") for name in names)
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.load(manifest_file)
        assert manifest["outcome"] == outcome
        assert manifest["producer_roots_sha256"] == T.P.E.digest(plan["roots"])
        record = next(item for item in manifest["files"] if item["logical_path"] == "/stage/records.jsonl")
        record_file = archive.extractfile(record["archive_path"])
        assert record_file is not None and record_file.read() == raw.read_bytes()


def test_logical_alias_wins_over_native_alias_and_overlap_is_rejected(fixture):
    raw = fixture.source / "artifact.json"
    raw.write_bytes(b'{"value":1}')
    reference = {"path": "/original/artifact.json", "sha256": T.sha_bytes(raw.read_bytes())}
    rows = [(3, T.P.E._logical("/original"), fixture.source),
            (7, T.P.E._logical("/archive"), fixture.source)]
    index, relative, found, logical = T._locate(reference, rows)
    assert (index, relative, found, logical) == (3, "artifact.json", raw, "/original/artifact.json")
    overlapping = rows + [(9, T.P.E._logical("/original/artifact.json"), fixture.source)]
    with pytest.raises(ValueError, match="ambiguous in logical"):
        T._locate(reference, overlapping)


def test_export_rejects_outside_required_ref_link_and_mid_export_change(fixture, monkeypatch):
    _, raw, _ = make_completion(fixture)
    monkeypatch.setattr(T.P, "_verify_completion", lambda prepared, value: None)
    plan_path = fixture.parent / "plan.json"
    plan = json.loads(plan_path.read_text())
    outside = save(fixture.tmp / "outside.json", {"bad": True})
    plan["proof"] = outside
    plan_ref = save(fixture.parent / "plan2.json", plan)
    outcome = json.loads((fixture.parent / "outcome.json").read_text())
    outcome["plan"] = plan_ref
    save(fixture.parent / "outcome.json", outcome)
    with pytest.raises(ValueError, match="outside"):
        run(fixture, mode="export", arm="NONE", output=fixture.tmp / "outside.tar.gz")

    # Restore a valid completion and reject source links before archive creation.
    for path in fixture.parent.iterdir():
        if path.is_file():
            path.unlink()
    _, raw, _ = make_completion(fixture)
    link = fixture.stage / "link.json"
    os.link(raw, link)
    assert link.stat().st_nlink > 1
    evidence = json.loads((fixture.parent / "evidence.json").read_text())
    evidence["raw_files_sha256"]["/stage/link.json"] = T.sha_bytes(raw.read_bytes())
    save(fixture.parent / "evidence.json", evidence)
    outcome = json.loads((fixture.parent / "outcome.json").read_text())
    outcome["evidence"] = T._ref(fixture.parent / "evidence.json")
    save(fixture.parent / "outcome.json", outcome)
    with pytest.raises(ValueError, match="single regular"):
        run(fixture, mode="export", arm="NONE", output=fixture.tmp / "link.tar.gz")

    link.unlink()
    for path in fixture.parent.iterdir():
        if path.is_file():
            path.unlink()
    _, raw, _ = make_completion(fixture)
    changed = False

    def mutate(path: Path):
        nonlocal changed
        if not changed and path == raw:
            path.write_bytes(b"changed")
            changed = True

    output = fixture.tmp / "changed.tar.gz"
    with pytest.raises(ValueError, match="changed"):
        run(fixture, mode="export", arm="NONE", output=output, before_add=mutate)
    assert output.exists() and Path(str(output) + ".error.json").is_file()


def test_export_race_preserves_other_file(fixture, monkeypatch):
    make_completion(fixture)
    monkeypatch.setattr(T.P, "_verify_completion", lambda prepared, value: None)
    output = fixture.tmp / "race.tar.gz"
    created = False

    def race(_path: Path):
        nonlocal created
        if not created:
            output.write_bytes(b"other-owner")
            created = True

    with pytest.raises(FileExistsError):
        run(fixture, mode="export", arm="NONE", output=output, before_add=race)
    assert output.read_bytes() == b"other-owner"
    error = json.loads(Path(str(output) + ".error.json").read_text())
    assert error["partial_output_preserved"] is True and error["error_type"] == "FileExistsError"


def test_export_rejects_noncanonical_archive_source_path(fixture, monkeypatch):
    make_completion(fixture)
    monkeypatch.setattr(T.P, "_verify_completion", lambda prepared, value: None)
    evidence = json.loads((fixture.parent / "evidence.json").read_text())
    evidence["raw_files_sha256"] = {"/stage/../records.jsonl": T.sha_bytes((fixture.stage / "records.jsonl").read_bytes())}
    save(fixture.parent / "evidence.json", evidence)
    outcome = json.loads((fixture.parent / "outcome.json").read_text())
    outcome["evidence"] = T._ref(fixture.parent / "evidence.json")
    save(fixture.parent / "outcome.json", outcome)
    with pytest.raises(ValueError, match="canonical POSIX"):
        run(fixture, mode="export", arm="NONE", output=fixture.tmp / "traversal.tar.gz")
