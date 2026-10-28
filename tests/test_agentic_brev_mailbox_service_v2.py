"""분리된 metadata archive 회수·동결 lifecycle·재전송 거부를 CPU로 검증한다."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src", "tests")]
import agentic_brev_mailbox_service_v2 as S
from test_agentic_brev_probe_transport_remote_v2 import exported, pair, save, unpack  # noqa: F401 - pytest fixture
from test_agentic_investigation_runtime_v1 import commit_objects


def rewrite(source, target, change):
    with tarfile.open(source, "r:gz") as archive:
        raw = {}
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            assert stream is not None
            raw[member.name] = stream.read()
    manifest = json.loads(raw["manifest.json"])
    change(manifest, raw)
    raw["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    with tarfile.open(target, "w:gz") as archive:
        for name, body in raw.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))


@pytest.mark.parametrize("fault", ["arm", "operation", "metadata_digest", "evidence_digest", "binding", "sha", "duplicate", "escape", "layer"])
def test_archive_faults_refused_before_copy(pair, tmp_path, fault):  # noqa: F811 - pytest fixture
    source, changed = tmp_path / "good.tar.gz", tmp_path / "changed.tar.gz"
    exported(pair, source)

    def change(manifest, _raw):
        row = manifest["files"][0]
        if fault == "arm":
            manifest["arm"] = "F1"
        elif fault == "operation":
            manifest["operation_id"] = "f" * 64
        elif fault.endswith("digest"):
            manifest["metadata_roots_sha256" if fault.startswith("metadata") else "evidence_roots_sha256"] = "f" * 64
        elif fault == "binding":
            manifest["binding_sha256"] = "f" * 64
        elif fault == "sha":
            row["sha256"] = "f" * 64
        elif fault == "duplicate":
            manifest["files"].append(dict(row))
        elif fault == "escape":
            row["relative"] = "../outside"
        else:
            row["layer"] = "unregistered"

    rewrite(source, changed, change)
    with pytest.raises(ValueError):
        unpack(pair, changed)
    assert list(pair["host"].iterdir()) == []


def test_same_archive_replay_and_existing_metadata_are_refused(pair, tmp_path):  # noqa: F811 - pytest fixture
    path = tmp_path / "archive.tar.gz"
    exported(pair, path)
    reference = unpack(pair, path)
    original = Path(reference["path"]).read_bytes()
    with pytest.raises(ValueError, match="not replayable"):
        unpack(pair, path)
    assert Path(reference["path"]).read_bytes() == original


def test_unregistered_copy_permission_and_source_difference_refused(pair, tmp_path):  # noqa: F811 - pytest fixture
    archive = tmp_path / "archive.tar.gz"
    exported(pair, archive)
    prepared = pair["prepared"]
    unit = prepared["operations"]["NONE"]["unit"]
    target = pair["host"] / "producer" / unit.relative_to(pair["w"]["root"]) / "outcome.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old original")
    with pytest.raises(ValueError, match="existing raw file differs"):
        unpack(pair, archive)
    assert target.read_bytes() == b"old original"


def test_interrupted_unpack_preserves_claim_and_forbids_replay(pair, tmp_path, monkeypatch):  # noqa: F811 - pytest fixture
    archive = tmp_path / "archive.tar.gz"
    exported(pair, archive)
    target = pair["host"] / "producer" / pair["prepared"]["operations"]["NONE"]["stage"].relative_to(pair["w"]["root"]) / "execution.json"
    original = Path.open

    def interrupted(path, mode="r", *args, **kwargs):
        if path == target and mode == "xb":
            raise OSError("CPU archive copy interrupted")
        return original(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            unpack(pair, archive)
    assert list(pair["host"].rglob("outcome.json.transport-v2-claim.json"))
    assert not list(pair["host"].rglob("outcome.json.transport-v2-completed.json"))
    before = {path: path.read_bytes() for path in pair["host"].rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="no replay"):
        unpack(pair, archive)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_allowed_copy_roots_and_order_are_still_exact(pair, tmp_path):  # noqa: F811 - pytest fixture
    archive = tmp_path / "archive.tar.gz"
    exported(pair, archive)
    prepared = pair["prepared"]
    common = {"metadata_roots": pair["metadata_roots"], "arm": "NONE",
        "operation_id": prepared["operations"]["NONE"]["op"]["operation_id"],
        "parent_request_sha256": pair["request_ref"]["sha256"],
        "parent_protocol_sha256": prepared["request"]["protocol"]["sha256"], "binding_sha256": prepared["request"]["binding"]["sha256"]}
    with pytest.raises(ValueError, match="unique allowed"):
        S.unpack_archive(archive, pair["evidence_roots"]["NONE"], [], **common)
    with pytest.raises(ValueError, match="ordered maps"):
        S.unpack_archive(archive, list(reversed(pair["evidence_roots"]["NONE"])), [], **common)
    assert list(pair["host"].iterdir()) == []


def test_archive_symlink_and_hardlink_refused(pair, tmp_path):  # noqa: F811 - pytest fixture
    source, linked = tmp_path / "good.tar.gz", tmp_path / "linked.tar.gz"
    exported(pair, source)
    with tarfile.open(source, "r:gz") as original, tarfile.open(linked, "w:gz") as changed:
        replaced = False
        for item in original.getmembers():
            stream = original.extractfile(item)
            assert stream is not None
            raw = stream.read()
            if item.name != "manifest.json" and not replaced:
                item.type, item.linkname, item.size = tarfile.SYMTYPE, "../outside", 0
                changed.addfile(item)
                replaced = True
            else:
                changed.addfile(item, io.BytesIO(raw))
    with pytest.raises(ValueError, match="regular unique bounded"):
        unpack(pair, linked)
    os.link(source, tmp_path / "hardlink.tar.gz")
    with pytest.raises(ValueError, match="single-link"):
        unpack(pair, source)
    assert list(pair["host"].iterdir()) == []


def test_service_rpc_calls_only_v2_helper_and_budget_stays_v1(tmp_path, monkeypatch):
    assert S.SCHEMA == S.V1.SCHEMA
    assert S.BrevMailboxService._budget is S.V1.BrevMailboxService._budget
    assert S.BrevMailboxService._call is S.V1.BrevMailboxService._call
    assert S.BrevMailboxService._validate_host_intent is S.V1.BrevMailboxService._validate_host_intent
    service = object.__new__(S.BrevMailboxService)
    service.contract = {"target": {"resource_name": "cpu-vm"},
        "host_mount_projection": {"host": str(tmp_path), "wsl": "/cpu"}}
    monkeypatch.setattr(service, "_inventory", lambda _folder: None)
    calls = []

    def call(folder, _label, args):
        calls.append(args)
        text = next(folder.glob("rpc-prepare-*.sh")).read_text(encoding="utf-8")
        encoded = re.search(r"base64.b64decode\('([^']+)'\)", text)
        assert encoded is not None
        argv = json.loads(base64.b64decode(encoded.group(1)))
        assert argv[2] == "/repo/bench/agentic_brev_probe_transport_remote_v2.py"
        result = {"schema": S.REMOTE.SCHEMA, "status": "prepared", "authority_issued": False}
        envelope = {"exit_code": 0, "stdout_base64": base64.b64encode(json.dumps(result).encode()).decode(), "stderr_base64": ""}
        raw = S.V1.MARKER + base64.b64encode(json.dumps(envelope).encode()).decode() + "\ncpu-vm\n"
        output = folder / "rpc.stdout"
        output.write_text(raw, encoding="utf-8")
        return {"stdout": str(output)}

    monkeypatch.setattr(service, "_call", call)
    result = service._rpc(tmp_path, {"remote_repo": "/repo", "remote_python": "/venv/bin/python", "remote_request": "/request.json"},
                          {"parent_request": {"sha256": "a" * 64}}, "prepare")
    assert result["schema"] == S.REMOTE.SCHEMA and len(calls) == 1


def test_service_constructor_checks_own_v2_hash_and_registered_closure(pair, tmp_path):  # noqa: F811 - pytest fixture
    w = pair["w"]
    folder = tmp_path / "runtime"
    folder.mkdir()
    (folder / "transport-mailbox").mkdir()
    service_sha = S.R.source_hash(ROOT / "evals/agentic_brev_mailbox_service_v2.py")
    contract_ref = save(tmp_path / "contract.json", {"schema": S.T.SCHEMA + "_contract", "mailbox": str(folder / "transport-mailbox"),
        "service_source_sha256_lf": service_sha})
    sources = {name: (ROOT / name).read_bytes() for name in S.SOURCES}
    protocol = {"schema": S.R.SCHEMA + "_protocol", "sources_lf": {name: S.R.source_hash(ROOT / name) for name in S.SOURCES},
        "host_transport_contract_sha256": contract_ref["sha256"], "host_transport_service_sha256_lf": service_sha}
    sources["bench/protocols/v2-runtime.json"] = S.E.J._bytes(protocol)
    anchor = commit_objects(w["repo"], sources, pair["prepared"]["request"]["source_anchor"])
    runtime = SimpleNamespace(folder=folder, reg={"allow_fixture": False}, protocol=protocol, anchor=anchor, repo=w["repo"],
        protocol_ref=S.T.read_ref(save(tmp_path / "protocol-ref.json", S.REMOTE.V1._ref(w["repo"] / "bench/protocols/v2-runtime.json"))))
    service = S.BrevMailboxService(runtime, {}, contract_ref)
    assert service.root == folder / "transport-mailbox"
    runtime.protocol["host_transport_service_sha256_lf"] = S.R.source_hash(ROOT / "evals/agentic_brev_mailbox_service_v1.py")
    with pytest.raises(ValueError, match="v2 service source"):
        S.BrevMailboxService(runtime, {}, contract_ref)


def test_partial_service_intent_and_completed_response_never_resubmit(tmp_path, monkeypatch):
    service = object.__new__(S.BrevMailboxService)
    service.root = tmp_path / "mailbox"
    operation = "a" * 64
    registration = {"path": str(tmp_path / "registration.json"), "sha256": "b" * 64}
    service.bindings, service.contract_ref = {operation: registration}, {"sha256": "c" * 64}
    request = {"schema": S.T.SCHEMA + "_request", "operation_id": operation, "phase": "reference",
        "registration": registration, "contract_sha256": "c" * 64}
    path = service.root / operation / "reference/request.json"
    save(path, request)
    (path.parent / "service").mkdir()
    monkeypatch.setattr(service, "_registered", lambda *_args: pytest.fail("no new validation/IO after unknown service intent"))
    assert service.serve_once(path)["status"] == "unknown"
    response = {"schema": S.T.SCHEMA + "_response", "request_sha256": S.E.J.file_sha(path),
        "operation_id": operation, "phase": "reference", "status": "unknown"}
    save(path.with_name("response.json"), response)
    assert service.serve_once(path) == response
