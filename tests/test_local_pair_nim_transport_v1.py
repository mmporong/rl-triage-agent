# pyright: reportMissingImports=false
"""네트워크 없이 local pair NIM transport의 원장·상한·복구 경계를 검증한다."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import agentic_investigation_call_v2 as CALL
import local_pair_nim_transport_v1 as T
import test_selection_comparison_controller_v1 as FIX


def _run(*args: str, cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _registered(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    for name in T.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    outer = repo / "evals/outer_fixture.py"
    outer.write_text("VALUE = 'outer source'\n", encoding="utf-8")
    protocol = repo / "bench/protocols/local-pair.json"
    protocol.parent.mkdir(parents=True)
    value = {
        "schema": T.PROTOCOL_SCHEMA,
        "api_cap": 12,
        "model": T.CALL.PARAMS["model"],
        "accepted_provider_models": sorted(T.MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(T.CALL.PARAMS),
        "sources_lf": {**{name: _sha(repo / name) for name in T.SOURCE_FILES},
                       "evals/outer_fixture.py": _sha(outer)},
        "backend0prefix": 0,
        "authority": deepcopy(T.AUTHORITY),
        "automatic_retries": 0,
        "experiment": {"arms": ["logs_only", "logs_plus_frames"]},
    }
    protocol.write_text(json.dumps(value), encoding="utf-8")
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "fixture@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "Fixture", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "register fixture", cwd=repo)
    anchor = _run("git", "rev-parse", "HEAD", cwd=repo)
    monkeypatch.setattr(T, "ROOT", repo)
    transport = tmp_path / "transport"
    T.initialize(transport, protocol_path=protocol, execution_anchor=anchor, api_cap=12)
    return transport, protocol, anchor


def _request(tmp_path: Path, number: int = 1, *, arm: str = "logs_only") -> dict:
    context = CALL.A.prepare(FIX.snapshot(tmp_path / f"snapshot-{number}", "case", "model_adaptive"),
                             goal="새 local pair 검사 선택")
    messages = CALL.A.messages(context)
    run_id = f"run{number}"
    return {
        "schema": T.REQUEST_SCHEMA,
        "request_id": f"{run_id}_{arm}_1",
        "run_id": run_id,
        "arm": arm,
        "step": 1,
        "context_digest": "a" * 64,
        "messages": messages,
        "params": CALL.params_for_messages(messages),
        "source_binding": {"pair_sha256": "b" * 64,
                           "case_checkpoint_sha256": "c" * 64,
                           "reference_checkpoint_sha256": "d" * 64},
    }


def _raw(content: str | None = '{"action":"hold"}', *, model: str | None = None,
         finish: str = "stop") -> dict:
    return {"live": False, "http_status": 200, "wall_s": 0.01,
            "provider_response": {
                "id": "cpu-fixture", "model": model or T.MODEL_CALL.MODEL,
                "choices": [{"message": {"content": content, "reasoning_content": "cpu"},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}}


def test_completed_call_preserves_request_usage_and_reuses_without_second_send(tmp_path, monkeypatch):
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    request = _request(tmp_path)
    calls = []

    def send(body, call_dir):
        calls.append((deepcopy(body), call_dir))
        return _raw()

    first = T.complete(root, request, send=send)
    second = T.complete(root, request, send=lambda *_: pytest.fail("중복 HTTP 호출"))
    assert first == second
    assert first["status"] == "completed" and first["live"] is False
    assert first["request"] == request and first["usage"]["total_tokens"] == 15
    assert first["call_ref"]["sha256"] == T._sha(Path(first["call_ref"]["path"]) / "receipt.json")
    assert len(calls) == 1 and calls[0][0] == T.wire_body(request)
    assert "extra_body" not in calls[0][0]
    assert calls[0][0]["reasoning_effort"] == "none"
    assert calls[0][0]["response_format"] == request["params"]["response_format"]


def test_atomic_ordinals_and_global_cap_are_enforced(tmp_path, monkeypatch):
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)

    def consume(number):
        return T.complete(root, _request(tmp_path, number), send=lambda *_: _raw())

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = list(pool.map(consume, range(1, 5)))
    rest = [consume(number) for number in range(5, 13)]
    assert sorted(row["ordinal"] for row in [*first, *rest]) == list(range(1, 13))
    assert len(list((root / "calls").glob("*/intent.json"))) == 12
    with pytest.raises(RuntimeError, match="cap"):
        consume(13)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [(_raw(content=None), "infra_error"),
     (_raw(finish="length"), "infra_error"),
     (_raw(model="unexpected/model"), "unknown")],
)
def test_nullable_truncation_and_semantic_invalid_are_separate(tmp_path, monkeypatch, provider, expected):
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    result = T.complete(root, _request(tmp_path), send=lambda *_: provider)
    assert result["status"] == expected
    assert result["outcome"] == ("semantic_invalid" if expected == "unknown"
                                  else "infrastructure_unknown")


def test_timeout_is_sealed_unknown_and_never_retried(tmp_path, monkeypatch):
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    count = 0

    def timeout(*_):
        nonlocal count
        count += 1
        raise TimeoutError("cpu timeout")

    request = _request(tmp_path)
    result = T.complete(root, request, send=timeout)
    assert result["status"] == "infra_error" and result["content"] is None
    assert T.complete(root, request, send=timeout) == result and count == 1


def test_tampered_response_request_and_protocol_are_rejected(tmp_path, monkeypatch):
    root, protocol, _anchor = _registered(tmp_path, monkeypatch)
    result = T.complete(root, _request(tmp_path), send=lambda *_: _raw())
    call = Path(result["call_ref"]["path"])
    response = T._read(call / "response.json")
    response["result"]["usage"] = {"total_tokens": 999}
    (call / "response.json").write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        T.verify_call(call)
    second = T.complete(root, _request(tmp_path, 2), send=lambda *_: _raw())
    second_call = Path(second["call_ref"]["path"])
    intent = T._read(second_call / "intent.json")
    intent["request"]["context_digest"] = "e" * 64
    (second_call / "intent.json").write_text(json.dumps(intent), encoding="utf-8")
    with pytest.raises(ValueError):
        T.verify_call(second_call)
    protocol.write_text(protocol.read_text(encoding="utf-8").replace('"api_cap": 12', '"api_cap": 11'),
                        encoding="utf-8")
    with pytest.raises(ValueError):
        T._manifest(root)


def test_live_flag_requires_worker_claim_and_matching_completion(tmp_path, monkeypatch):
    root, _protocol, _anchor = _registered(tmp_path, monkeypatch)
    result = T.complete(root, _request(tmp_path), send=lambda *_: {**_raw(), "live": True})
    assert result["status"] == "infra_error" and result["live"] is False


def test_worker_claim_is_single_send_and_live_evidence_is_verified(tmp_path, monkeypatch):
    root, protocol, anchor = _registered(tmp_path, monkeypatch)
    request = _request(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "cpu-fixture-secret")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        @staticmethod
        def read():
            return json.dumps(_raw()["provider_response"]).encode()

    wire = []

    def urlopen(outgoing, **_kwargs):
        wire.append(json.loads(outgoing.data))
        return Response()

    monkeypatch.setattr(T.urllib.request, "urlopen", urlopen)
    identity = {"pid": 4321, "proc_start_ticks": "987654", "wsl": True}
    monkeypatch.setattr(T, "_actual_wsl_ready", lambda: None)
    monkeypatch.setattr(T, "_process_identity", lambda *, actual: {**identity, "wsl": actual})

    def send(_body, call_dir):
        assert T._run_worker_cli(root, call_dir, protocol, anchor) == 0
        with pytest.raises(FileExistsError):
            T._run_worker_cli(root, call_dir, protocol, anchor)
        return T._read(call_dir / "worker-response.json")

    result = T.complete(root, request, send=send)
    assert result["status"] == "completed" and result["live"] is True
    call = Path(result["call_ref"]["path"])
    assert T._read(call / "sent-claim.json")["endpoint"] == T.ENDPOINT
    assert result["worker_evidence"] == {
        "process_sha256": T._sha(call / "worker-process.json"),
        "exit_sha256": T._sha(call / "worker-exit.json"),
    }
    assert wire == [T.wire_body(request)]
    assert "extra_body" not in wire[0] and wire[0]["reasoning_effort"] == "none"
    assert wire[0]["response_format"] == request["params"]["response_format"]
    assert "cpu-fixture-secret" not in json.dumps([T._read(path) for path in call.glob("*.json")])
    process = T._read(call / "worker-process.json")
    process["proc_start_ticks"] = "123"
    (call / "worker-process.json").write_text(json.dumps(process), encoding="utf-8")
    with pytest.raises(ValueError, match="process/exit identity"):
        T.verify_call(call)


def test_direct_worker_fixture_is_not_marked_live(tmp_path, monkeypatch):
    root, protocol, anchor = _registered(tmp_path, monkeypatch)
    call, created = T._allocate(root, _request(tmp_path), T._manifest(root))
    assert created is True
    monkeypatch.setenv("NVIDIA_API_KEY", "cpu-fixture-secret")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        @staticmethod
        def read():
            return json.dumps(_raw()["provider_response"]).encode()

    monkeypatch.setattr(T.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert T._worker(root, call, protocol, anchor) == 0
    assert T._read(call / "worker-response.json")["live"] is False
    assert not (call / "worker-process.json").exists()


def test_wire_body_rejects_extra_body_collision(tmp_path):
    request = _request(tmp_path)
    request["params"]["extra_body"]["model"] = "collision"
    with pytest.raises(ValueError, match="충돌"):
        T._flatten_wire_body(request["params"], request["messages"])
