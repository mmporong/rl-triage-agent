"""운반 파일 검증과 RPC envelope의 CPU 회귀. 실제 Brev나 GPU는 호출하지 않는다."""
import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import agentic_brev_mailbox_service_v1 as S


def archive(tmp_path, *, change=None, link=False):
    raw = b'{"cpu_fixture_only":true}\n'
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {"arm": "NONE", "operation_id": "1" * 64,
                "authority_issued": False, "evidence_accepted": False,
                "outcome": {"path": "/producer/op/outcome.json", "sha256": digest},
                "files": [{"root_index": 0, "relative": "op/outcome.json", "logical_path": "/producer/op/outcome.json",
                           "archive_path": "files/0/op/outcome.json", "sha256": digest, "bytes": len(raw)}]}
    if change:
        change(manifest)
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as out:
        data = json.dumps(manifest).encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(data)
        out.addfile(member, io.BytesIO(data))
        member = tarfile.TarInfo("files/0/op/outcome.json")
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            out.addfile(member)
        else:
            member.size = len(raw)
            out.addfile(member, io.BytesIO(raw))
    local = tmp_path / "result"
    local.mkdir()
    return path, [{"remote": "/producer", "local": str(local)}], [{"path": str(local), "writable": True}]


def extract(args):
    return S.unpack_archive(*args, arm="NONE", operation_id="1" * 64)


def test_archive_validates_raw_then_copies_and_reuses_exact(tmp_path):
    args = archive(tmp_path)
    ref = extract(args)
    path = Path(ref["path"])
    original = path.read_bytes(), path.stat().st_mtime_ns
    assert S.T.read_ref(ref) == {"cpu_fixture_only": True}
    assert extract(args) == ref
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    path.write_bytes(b"old original")
    with pytest.raises(ValueError, match="existing raw"):
        extract(args)
    assert path.read_bytes() == b"old original"


@pytest.mark.parametrize("fault", ["sha", "size", "relative", "logical", "root", "arm", "operation", "authority"])
def test_archive_fault_refuses_before_copy(tmp_path, fault):
    def change(manifest):
        row = manifest["files"][0]
        if fault == "sha":
            row["sha256"] = "0" * 64
        elif fault == "size":
            row["bytes"] += 1
        elif fault == "relative":
            row["relative"] = "../escape"
        elif fault == "logical":
            row["logical_path"] = "/elsewhere/outcome.json"
        elif fault == "root":
            row["root_index"] = 1
        elif fault == "arm":
            manifest["arm"] = "F1"
        elif fault == "operation":
            manifest["operation_id"] = "2" * 64
        else:
            manifest["authority_issued"] = True
    args = archive(tmp_path, change=change)
    with pytest.raises(ValueError):
        extract(args)
    assert list(Path(args[1][0]["local"]).iterdir()) == []


def test_archive_symlink_and_readonly_source_missing_refused(tmp_path):
    args = archive(tmp_path, link=True)
    with pytest.raises(ValueError, match="regular unique"):
        extract(args)
    args[0].unlink()
    Path(args[1][0]["local"]).rmdir()
    args = archive(tmp_path)
    args[2][0]["writable"] = False
    with pytest.raises(ValueError, match="read-only"):
        extract(args)
    assert list(Path(args[1][0]["local"]).iterdir()) == []


@pytest.mark.parametrize("exit_code", [0, 7])
def test_generated_rpc_preserves_code_and_strips_keys(tmp_path, monkeypatch, exit_code):
    monkeypatch.setenv("NVIDIA_API_KEY", "cpu-fixture-not-a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "cpu-fixture-not-a-key")
    body = S.remote_script([sys.executable, "-c",
                            "import os;print('NVIDIA_API_KEY' in os.environ, 'OPENAI_API_KEY' in os.environ);"
                            "raise SystemExit(" + str(exit_code) + ")"])
    code = body.split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    script = tmp_path / "rpc.py"
    script.write_text(code, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=30, check=False)
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith(S.MARKER)
    response = json.loads(base64.b64decode(lines[0][len(S.MARKER):]))
    assert response["exit_code"] == exit_code
    assert base64.b64decode(response["stdout_base64"]).strip() == b"False False"


def test_stop_intent_blocks_even_inventory_before_subprocess(tmp_path):
    service = object.__new__(S.BrevMailboxService)
    stop = tmp_path / "stop-intent.json"
    stop.write_text("{}", encoding="utf-8")
    life = tmp_path / "lifecycle.json"
    life.write_text("{}", encoding="utf-8")
    service.contract = {"lifecycle": {"path": str(life), "sha256": hashlib.sha256(life.read_bytes()).hexdigest()},
                        "lifecycle_stop_intent_path": str(stop)}
    with pytest.raises(ValueError, match="Stop intent"):
        service._call(tmp_path / "out", "inventory", ["ls", "--json"])
    assert not (tmp_path / "out").exists()


def pending_service(tmp_path):
    service = object.__new__(S.BrevMailboxService)
    service.root = tmp_path / "mailbox"
    op = "1" * 64
    registration = {"path": str(tmp_path / "registration-not-read.json"), "sha256": "b" * 64}
    service.bindings = {op: registration}
    service.contract_ref = {"path": str(tmp_path / "contract-not-read.json"), "sha256": "a" * 64}
    path = service.root / op / "reference/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": S.T.SCHEMA + "_request", "operation_id": op, "phase": "reference",
                                "contract_sha256": "a" * 64, "registration": registration}), encoding="utf-8")
    return service, path


def test_interrupted_service_intent_never_starts_or_sends_provider(tmp_path, monkeypatch):
    service, path = pending_service(tmp_path)
    (path.parent / "service").mkdir()
    monkeypatch.setattr(service, "_rpc", lambda *args: pytest.fail("중단 뒤 원격 실행 금지"))
    result = service.serve_once(path)
    assert result["status"] == "unknown" and result["reason"] == "existing_service_intent_lookup_only"
    assert not path.with_name("response.json").exists()


def test_existing_service_response_requires_same_request(tmp_path):
    service, path = pending_service(tmp_path)
    response = {"schema": S.T.SCHEMA + "_response", "operation_id": "1" * 64, "phase": "reference",
                "request_sha256": "0" * 64, "status": "completed"}
    path.with_name("response.json").write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="existing service response"):
        service.serve_once(path)
    response["request_sha256"] = S.E.J.file_sha(path)
    path.with_name("response.json").write_text(json.dumps(response), encoding="utf-8")
    assert service.serve_once(path) == response


@pytest.mark.parametrize("fault", [None, "future", "cost", "forged_phase", "foreign_path", "reset_clock"])
def test_budget_uses_committed_registration_clock_not_dynamic_start(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace

    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    start = now + timedelta(hours=1) if fault == "future" else now - timedelta(seconds=30)
    if fault == "reset_clock":
        start = now - timedelta(hours=2)
    phase = {"schema": S.SCHEMA + "_lifecycle_protocol", "resource_id": "cpu-fixture-vm",
             "resource_name": "cpu-fixture", "instance_type": "cpu-fixture-type",
             "quoted_usd_per_h": "2", "compute_cap_usd": "0.001" if fault == "cost" else "1",
             "stop_reserve_s": 60, "max_lifecycle_s": 7200,
             "registered_at_utc": start.isoformat(), "deadline_utc": (now + timedelta(hours=2)).isoformat(),
             "budget_origin": "immutable_registered_at_utc", "source_commit": "a" * 40, "sources_lf": {}}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    phase_ref = save(tmp_path / "bench/protocols/budget.json", phase)
    service = object.__new__(S.BrevMailboxService)
    service.runtime = SimpleNamespace(repo=tmp_path, anchor="b" * 40,
                                     protocol={"lifecycle_protocol_sha256": phase_ref["sha256"]})
    service.contract = {"lifecycle_protocol": phase_ref, "lifecycle_stop_intent_path": str(tmp_path / "stop.json"),
                        "deadline_utc": phase["deadline_utc"], "target": {key: phase[key] for key in
                         ("resource_id", "resource_name", "instance_type")}}
    checks = []
    monkeypatch.setattr(S.R, "verify_sources", lambda *args, **kwargs: checks.append((args, kwargs)))
    if fault == "forged_phase":
        service.contract["lifecycle_protocol"] = save(tmp_path / "bench/protocols/forged.json",
                                                      {**phase, "registered_at_utc": now.isoformat()})
    elif fault == "foreign_path":
        service.contract["lifecycle_protocol"] = save(tmp_path / "forged/phase.json", phase)
    elif fault == "reset_clock":
        service.contract["lifecycle"] = save(tmp_path / "forged/lifecycle.json", {
            "started_at_utc": now.isoformat(), "compute_cap_usd": "1000", "command": ["brev", "start", "cpu-fixture"]})
    if fault is None:
        service._budget(120)
        assert len(checks) == 1 and checks[0][1]["protocol_path"] == Path(phase_ref["path"])
    else:
        with pytest.raises(ValueError):
            service._budget(120)
