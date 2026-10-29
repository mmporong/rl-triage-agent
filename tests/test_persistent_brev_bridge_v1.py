from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from bench import persistent_brev_bridge_v1 as B
from evals import persistent_provider_complete_v1 as PROVIDER


def test_provider_cold_import_without_pythonpath(tmp_path: Path) -> None:
    source = Path(PROVIDER.__file__).resolve()
    code = (
        "import importlib.util,sys; "
        "spec=importlib.util.spec_from_file_location('isolated_provider',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token.txt"
    path.write_text("A" * 48 + "\n", encoding="utf-8")
    return path


def fixture_config(tmp_path: Path, port: int) -> dict[str, Any]:
    return {"schema": B.SCHEMA, "host": "127.0.0.1", "port": port,
            "token_file": str(token_file(tmp_path)), "executable": sys.executable,
            "allowed_commands": ["-c"], "max_timeout_s": 2,
            "receipt_root": str(tmp_path / "server-receipts"), "provider": None}


def start_server(config: dict[str, Any]) -> tuple[Any, threading.Thread]:
    state = B.BridgeState(config, production=False)
    server = B.ThreadingHTTPServer((state.host, state.port), B.handler_type(state))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return server, thread


def client_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("schema", "host", "port", "token_file", "executable",
                                          "allowed_commands", "max_timeout_s")} | {
        "original_prefix": ["wsl", "-d", "Ubuntu", "--exec", config["executable"]]}


def post(port: int, value: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(value).encode()
    try:
        connection.request("POST", "/v1/call", body=raw, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_actual_loopback_roundtrip_timeout_token_replay_and_no_shell(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, free_port())
    server, thread = start_server(config)
    try:
        client = B.BridgeClient(client_config(config))
        literal = ";echo shell-was-not-used"
        result = client.call(["-c", f"print({literal!r})"], 2)
        assert B.base64.b64decode(result["stdout_b64"]).decode().strip() == literal
        assert result["exit_code"] == 0 and result["timed_out"] is False

        timeout = client.call(["-c", "import time; time.sleep(2)"], 1)
        assert timeout["timed_out"] is True and timeout["exit_code"] is None

        request = {"schema": B.REQUEST_SCHEMA, "token": "wrong" * 8,
                   "nonce": "1" * 32, "argv": [sys.executable, "-c", "print(1)"], "timeout_s": 1}
        assert post(config["port"], request)[0] == 403
        request["token"] = "A" * 48
        assert post(config["port"], request)[0] == 200
        assert post(config["port"], request)[0] == 403

        with pytest.raises(ValueError, match="registered executable/subcommand"):
            B.validate_argv([sys.executable, "--version"], executable=sys.executable,
                            allowed_commands=frozenset({"-c"}))
        receipt_text = "\n".join(path.read_text(encoding="utf-8")
                                  for path in (tmp_path / "server-receipts").glob("*.json"))
        assert "A" * 48 not in receipt_text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_transport_preserves_original_argv_and_receipt_shape(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, free_port())
    server, thread = start_server(config)
    try:
        client = client_config(config)
        root = tmp_path / "transport"
        transport = B.PersistentBrevTransport(root, client["original_prefix"], object(), config=client)
        receipt = transport.call("probe", ["-c", "print('ok')"], 2)
        intent = json.loads(Path(receipt["intent"]["path"]).read_text(encoding="utf-8"))
        assert intent["command"] == [*client["original_prefix"], "-c", "print('ok')"]
        assert intent["shell"] is False and Path(receipt["stdout_path"]).read_bytes() == b"ok\r\n"
        assert set(receipt) >= {"exit_code", "timed_out", "stdout_path", "stderr_path", "intent",
                                "bridge_nonce", "bridge_argv_sha256"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_production_config_rejects_non_brev_executable(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, free_port())
    with pytest.raises(ValueError, match="command allowlist|required"):
        B.BridgeState(config)


def test_provider_endpoint_fixed_worker_hash_model_cap_and_timeout(tmp_path: Path) -> None:
    worker = tmp_path / "provider_worker.py"
    worker.write_text(
        "import json,sys,time\n"
        "p=json.load(sys.stdin)\n"
        "time.sleep(2 if p['messages']==['timeout'] else 0)\n"
        "print(json.dumps({'schema':'agentic_provider_once_v2','messages_sha256':p['messages_sha256'],"
        "'params_sha256':p['params_sha256'],'result':{'kind':'ok'}}))\n",
        encoding="utf-8",
    )
    config = fixture_config(tmp_path, free_port())
    config["provider"] = {"python": sys.executable, "source": str(worker),
                          "source_sha256": B.sha(worker), "cwd": str(tmp_path),
                          "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                          "max_calls": 2, "timeout_s": 1}
    server, thread = start_server(config)
    provider_config = {"schema": B.SCHEMA, "host": "127.0.0.1", "port": config["port"],
                       "token_file": config["token_file"], "model": config["provider"]["model"],
                       "timeout_s": 1, "source_sha256": B.sha(worker)}
    try:
        client = B.ProviderClient(provider_config)
        params = {"model": config["provider"]["model"]}
        provider_config_path = tmp_path / "provider-client.json"
        provider_config_path.write_text(json.dumps(provider_config), encoding="utf-8")
        helper = PROVIDER
        assert helper.CONFIG_ENV not in helper.os.environ
        with helper.config_scope(provider_config_path):
            assert helper.bridge_complete(["ok"], params) == {"kind": "ok"}
        assert helper.CONFIG_ENV not in helper.os.environ and helper.CONFIG_SHA_ENV not in helper.os.environ
        with pytest.raises(B.BridgeError, match="pending"):
            client.complete(["timeout"], params)
        with pytest.raises(B.BridgeError, match="pending"):
            client.complete(["third"], params)

        bad = {"schema": B.PROVIDER_REQUEST_SCHEMA, "token": "A" * 48, "nonce": "9" * 32,
               "payload": {"messages": ["x"], "params": {"model": "other"},
                           "messages_sha256": B.digest(["x"]),
                           "params_sha256": B.digest({"model": "other"})}, "timeout_s": 1}
        assert post(config["port"], bad)[0] == 400
        assert B.sha(worker) == config["provider"]["source_sha256"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_provider_source_change_is_rejected_before_worker(tmp_path: Path) -> None:
    worker = tmp_path / "provider_worker.py"
    worker.write_text("raise SystemExit('must not run')\n", encoding="utf-8")
    config = fixture_config(tmp_path, free_port())
    original_sha = B.sha(worker)
    config["provider"] = {"python": sys.executable, "source": str(worker),
                          "source_sha256": original_sha, "cwd": str(tmp_path),
                          "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                          "max_calls": 2, "timeout_s": 1}
    server, thread = start_server(config)
    try:
        worker.write_text("raise SystemExit('changed')\n", encoding="utf-8")
        provider = B.ProviderClient({"schema": B.SCHEMA, "host": "127.0.0.1", "port": config["port"],
                                     "token_file": config["token_file"], "model": config["provider"]["model"],
                                     "timeout_s": 1, "source_sha256": original_sha})
        with pytest.raises(B.BridgeError, match="pending"):
            provider.complete(["x"], {"model": config["provider"]["model"]})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
