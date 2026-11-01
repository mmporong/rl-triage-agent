# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import agentic_investigation_call_v2 as CALL
import five_policy_recovery_nim_transport_v1 as T
import lineage_five_policy_nim_transport_v1 as BASE72
import local_pair_nim_transport_v1 as BASE
import test_selection_comparison_controller_v1 as FIX


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    for name in T.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    protocol = repo / "bench/protocols/recovery.json"
    protocol.parent.mkdir(parents=True)
    value = {
        "schema": T.PROTOCOL_SCHEMA, "purpose": "R4", "api_cap": 32,
        "model": CALL.PARAMS["model"],
        "accepted_provider_models": sorted(T.MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(CALL.PARAMS),
        "sources_lf": {name: _sha(repo / name) for name in T.SOURCE_FILES},
        "backend0prefix": 0, "authority": deepcopy(T.AUTHORITY), "automatic_retries": 0,
    }
    protocol.write_text(json.dumps(value), encoding="utf-8")
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "fixture@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "Fixture", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "register fixture", cwd=repo)
    anchor = _run("git", "rev-parse", "HEAD", cwd=repo)
    monkeypatch.setattr(T, "ROOT", repo)
    monkeypatch.setattr(T._ENGINE, "ROOT", repo)
    transport = tmp_path / "transport"
    T.initialize(transport, protocol_path=protocol, execution_anchor=anchor, api_cap=32)
    return transport


def _request(tmp_path: Path, number: int) -> dict:
    context = CALL.A.prepare(FIX.snapshot(tmp_path / f"snapshot-{number}", "case", "model_adaptive"),
                             goal="R4 회복 경로 선택")
    messages = CALL.A.messages(context)
    return {"schema": T.REQUEST_SCHEMA, "request_id": f"r{number}_logs_only_1",
            "run_id": f"r{number}", "arm": "logs_only", "step": 1,
            "context_digest": "a" * 64, "messages": messages,
            "params": CALL.params_for_messages(messages),
            "source_binding": {"pair_sha256": "b" * 64,
                               "case_checkpoint_sha256": "c" * 64,
                               "reference_checkpoint_sha256": "d" * 64}}


def _raw(*, model: str | None = None) -> dict:
    return {"live": False, "http_status": 200, "wall_s": 0.01,
            "provider_response": {"id": "fixture", "model": model or T.MODEL_CALL.MODEL,
                                  "choices": [{"message": {"content": '{"action":"hold"}'},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 2, "completion_tokens": 1,
                                            "total_tokens": 3}}}


def test_r4_transport_has_isolated_schemas_authority_and_cap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = (BASE.AUTHORITY, BASE72.AUTHORITY, BASE72.API_CAP)
    root = _registered(tmp_path, monkeypatch)
    manifest = T._read(root / "manifest.json")
    assert manifest["schema"] == T.MANIFEST_SCHEMA
    assert T.AUTHORITY == {"api": True, "gpu": True, "probe": True,
                           "correction": True, "retraining": True}
    assert T.status(root)["remaining"] == 32
    assert (BASE.AUTHORITY, BASE72.AUTHORITY, BASE72.API_CAP) == shared


def test_receipt_reuse_and_unknown_charge_without_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _registered(tmp_path, monkeypatch)
    sends = []
    first = T.complete(root, _request(tmp_path, 1), send=lambda *args: (sends.append(args), _raw())[1])
    second = T.complete(root, _request(tmp_path, 1), send=lambda *_: pytest.fail("duplicate send"))
    unknown = T.complete(root, _request(tmp_path, 2), send=lambda *_: _raw(model="wrong/model"))
    assert first == second and first["schema"] == T.VERIFIED_SCHEMA and len(sends) == 1
    assert unknown["status"] == "unknown"
    assert T.status(root)["remaining"] == 30


def test_pending_is_charged_and_not_resent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _registered(tmp_path, monkeypatch)
    manifest = T._ENGINE._manifest(root)
    call, created = T._ENGINE._allocate(root, _request(tmp_path, 1), manifest)
    assert created and call.is_dir()
    with pytest.raises(RuntimeError, match="미완료"):
        T.complete(root, _request(tmp_path, 1), send=lambda *_: pytest.fail("send"))
    status = T.status(root)
    assert status["reservations_charged"] == status["pending_no_retry"] == 1


def test_cap_allows_31_and_32_then_rejects_33(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _registered(tmp_path, monkeypatch)
    manifest = T._ENGINE._manifest(root)
    monkeypatch.setattr(T._ENGINE, "_manifest", lambda _root: deepcopy(manifest))
    for number in range(1, 33):
        T.complete(root, _request(tmp_path, number), send=lambda *_: _raw())
        if number == 31:
            assert T.status(root)["remaining"] == 1
    assert T.status(root)["remaining"] == 0
    with pytest.raises(RuntimeError, match="cap"):
        T.complete(root, _request(tmp_path, 33), send=lambda *_: _raw())


def test_wrong_authority_or_requested_cap_is_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _registered(tmp_path, monkeypatch)
    manifest = T._read(root / "manifest.json")
    assert manifest["api_cap"] == 32
    protocol = Path(T._ENGINE.ROOT) / manifest["protocol"]["relative"]
    with pytest.raises(ValueError, match="32"):
        T.initialize(tmp_path / "wrong-cap", protocol_path=protocol,
                     execution_anchor=manifest["execution_anchor"], api_cap=33)
    value = T._read(protocol)
    value["authority"]["correction"] = False
    protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="결합"):
        T.status(root)
