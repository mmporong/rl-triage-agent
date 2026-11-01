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
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "tests"))
import agentic_investigation_call_v2 as CALL
import lineage_five_policy_nim_transport_v1 as T
import local_pair_nim_transport_v1 as BASE
import test_selection_comparison_controller_v1 as FIX


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


def _sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    for name in T.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    outer = repo / "evals/outer_driver.py"
    outer.write_text("VALUE = 'five-policy fixture'\n", encoding="utf-8")
    protocol = repo / "bench/protocols/five-policy.json"
    protocol.parent.mkdir(parents=True)
    value = {
        "schema": T.PROTOCOL_SCHEMA, "api_cap": T.API_CAP,
        "model": T.CALL.PARAMS["model"],
        "accepted_provider_models": sorted(T.MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(T.CALL.PARAMS),
        "sources_lf": {**{name: _sha(repo / name) for name in T.SOURCE_FILES},
                       "evals/outer_driver.py": _sha(outer)},
        "backend0prefix": 0, "authority": deepcopy(T.AUTHORITY),
        "automatic_retries": 0,
        "policies": ["registry_next", "fixed_eligible", "numeric", "numeric_frames",
                     "optimizer_first_registered_rule"],
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
    T.initialize(transport, protocol_path=protocol, execution_anchor=anchor, api_cap=72)
    return transport, protocol, anchor


def _request(tmp_path: Path, number: int = 1, *, arm: str = "logs_only") -> dict:
    context = CALL.A.prepare(
        FIX.snapshot(tmp_path / f"snapshot-{number}", "case", "model_adaptive"),
        goal="새 lineage 검사 선택")
    messages = CALL.A.messages(context)
    run_id = f"run{number}"
    return {"schema": T.REQUEST_SCHEMA, "request_id": f"{run_id}_{arm}_1",
            "run_id": run_id, "arm": arm, "step": 1, "context_digest": "a" * 64,
            "messages": messages, "params": CALL.params_for_messages(messages),
            "source_binding": {"pair_sha256": "b" * 64,
                               "case_checkpoint_sha256": "c" * 64,
                               "reference_checkpoint_sha256": "d" * 64}}


def _raw(*, content: str | None = '{"action":"hold"}', model: str | None = None,
         finish: str = "stop") -> dict:
    return {"live": False, "http_status": 200, "wall_s": 0.01,
            "provider_response": {"id": "fixture", "model": model or T.MODEL_CALL.MODEL,
                                  "choices": [{"message": {"content": content},
                                               "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 2, "completion_tokens": 1,
                                            "total_tokens": 3}}}


def test_fresh_transport_has_new_schemas_and_does_not_mutate_shared_v1(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = (BASE.PROTOCOL_SCHEMA, BASE.MANIFEST_SCHEMA, BASE.SOURCE_FILES)
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    manifest = T._read(root / "manifest.json")
    assert manifest["schema"] == T.MANIFEST_SCHEMA
    assert manifest["api_cap"] == 72 and manifest["backend0prefix"] == 0
    assert T.status(root)["remaining"] == 72
    assert (BASE.PROTOCOL_SCHEMA, BASE.MANIFEST_SCHEMA, BASE.SOURCE_FILES) == original


def test_submit_preserves_request_and_reuses_receipt_without_second_send(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    request, sends = _request(tmp_path), []

    def send(body, call):
        sends.append((body, call))
        return _raw()

    first = T.submit(root, request, send=send)
    second = T.complete(root, request, send=lambda *_: pytest.fail("중복 send"))
    call = Path(first["call_ref"]["path"])
    assert first == second == T.verify(call)
    assert first["schema"] == T.VERIFIED_SCHEMA
    assert T._read(call / "intent.json")["schema"] == T.INTENT_SCHEMA
    assert T._read(call / "response.json")["schema"] == T.RESULT_SCHEMA
    assert T._read(call / "receipt.json")["schema"] == T.RECEIPT_SCHEMA
    assert len(sends) == 1
    status = T.status(root)
    assert status["reservations_charged"] == status["completed"] == 1
    assert status["remaining"] == 71 and status["outcomes"]["semantic_valid"] == 1


def test_semantic_invalid_and_infra_unknown_consume_reservations(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    invalid = T.complete(root, _request(tmp_path, 1), send=lambda *_: _raw(model="unexpected/model"))
    infra = T.complete(root, _request(tmp_path, 2), send=lambda *_: _raw(content=None))
    assert invalid["status"] == "unknown" and infra["status"] == "infra_error"
    status = T.status(root)
    assert status["reservations_charged"] == 2 and status["remaining"] == 70
    assert status["outcomes"] == {"semantic_valid": 0, "semantic_invalid": 1,
                                  "infrastructure_unknown": 1}


def test_pending_request_is_charged_and_never_resent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    request = _request(tmp_path)
    call, created = T._ENGINE._allocate(root, request, T._ENGINE._manifest(root))
    assert created and call.is_dir()
    with pytest.raises(RuntimeError, match="미완료"):
        T.complete(root, request, send=lambda *_: pytest.fail("send 금지"))
    status = T.status(root)
    assert status["reservations_charged"] == status["pending_no_retry"] == 1
    assert status["dispatched"] == 0 and status["remaining"] == 71


def test_protocol_authority_cap_and_source_tamper_are_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, protocol, anchor = _registered(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        T.initialize(root, protocol_path=protocol, execution_anchor=anchor, api_cap=72)
    value = T._read(protocol)
    value["authority"]["correction"] = True
    protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="결합"):
        T.status(root)


def test_global_cap_is_seventy_two_and_ordinal_is_atomic(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    # Scratch Git 결합은 initialize에서 이미 확인했다. 이 반복은 원장 cap만 격리 검증한다.
    manifest = T._ENGINE._manifest(root)
    monkeypatch.setattr(T._ENGINE, "_manifest", lambda _root: deepcopy(manifest))
    for number in range(1, 73):
        T.complete(root, _request(tmp_path, number), send=lambda *_: _raw())
    assert T.status(root)["remaining"] == 0
    with pytest.raises(RuntimeError, match="cap"):
        T.complete(root, _request(tmp_path, 73), send=lambda *_: _raw())
