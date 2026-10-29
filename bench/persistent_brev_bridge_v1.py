"""기존 Linux 세션의 Brev CLI를 제한된 loopback RPC로 호출한다."""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import json
import re
import secrets
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCHEMA = "ri97_persistent_brev_bridge_v1"
REQUEST_SCHEMA = SCHEMA + "_request"
RESPONSE_SCHEMA = SCHEMA + "_response"
PROVIDER_REQUEST_SCHEMA = SCHEMA + "_provider_request"
PROVIDER_RESPONSE_SCHEMA = SCHEMA + "_provider_response"
TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
NONCE = re.compile(r"^[0-9a-f]{32}$")
LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ALLOWED_BREV_COMMANDS = frozenset({"ls", "exec", "copy", "start", "stop"})
MAX_BODY = 1 << 20
MAX_ARGV = 64
MAX_ARG = 16384


class BridgeError(RuntimeError):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def read_token(path: Path) -> str:
    target = Path(path).resolve(strict=True)
    if not target.is_file() or target.is_symlink():
        raise ValueError("regular token file required")
    token = target.read_text(encoding="utf-8").strip()
    if TOKEN.fullmatch(token) is None:
        raise ValueError("token format")
    return token


def validate_argv(argv: Any, *, executable: str, allowed_commands: frozenset[str]) -> list[str]:
    if (not isinstance(argv, list) or not 2 <= len(argv) <= MAX_ARGV
            or any(not isinstance(arg, str) or not arg or len(arg) > MAX_ARG
                   or "\x00" in arg or "\n" in arg or "\r" in arg for arg in argv)):
        raise ValueError("bounded argv required")
    if argv[0] != executable or argv[1] not in allowed_commands:
        raise ValueError("registered executable/subcommand required")
    return list(argv)


class BridgeState:
    def __init__(self, config: Mapping[str, Any], *, production: bool = True):
        required = {"schema", "host", "port", "token_file", "executable", "allowed_commands",
                    "max_timeout_s", "receipt_root", "provider"}
        if set(config) != required or config.get("schema") != SCHEMA:
            raise ValueError("bridge config fields")
        if config["host"] != "127.0.0.1" or type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
            raise ValueError("loopback endpoint required")
        commands = config["allowed_commands"]
        if (not isinstance(commands, list) or not commands or any(not isinstance(item, str) or not item for item in commands)
                or (production and frozenset(commands) != ALLOWED_BREV_COMMANDS)):
            raise ValueError("registered command allowlist required")
        timeout = config["max_timeout_s"]
        if type(timeout) is not int or not 1 <= timeout <= 3600:
            raise ValueError("bounded server timeout")
        executable = config["executable"]
        if (not isinstance(executable, str) or not executable
                or (production and (not executable.startswith("/") or executable.split("/")[-1] != "brev"))):
            raise ValueError("absolute Brev executable required")
        root = Path(config["receipt_root"]).resolve()
        root.mkdir(parents=True, exist_ok=False)
        self.host, self.port = config["host"], config["port"]
        self.token = read_token(Path(config["token_file"]))
        self.executable = executable
        self.allowed = frozenset(commands)
        self.max_timeout = timeout
        self.receipt_root = root
        self.seen: set[str] = set()
        self.lock = threading.Lock()
        self.provider_calls = 0
        self.provider = self._provider(config["provider"], production=production)

    @staticmethod
    def _provider(value: Any, *, production: bool) -> dict[str, Any] | None:
        if value is None:
            return None
        fields = {"python", "source", "source_sha256", "cwd", "model", "max_calls", "timeout_s"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("provider config fields")
        python, source, cwd = (str(value[key]) for key in ("python", "source", "cwd"))
        if ((production and not all(path.startswith("/") and "\\" not in path
                                    for path in (python, source, cwd)))
                or (not production and not all(Path(path).is_absolute() for path in (python, source, cwd)))
                or (production and python != str(Path.home() / ".venvs/rl-triage/bin/python"))):
            raise ValueError("fixed Linux provider paths")
        source_path = Path(source).resolve(strict=True)
        if (not source_path.is_file() or source_path.is_symlink()
                or sha(source_path) != value["source_sha256"]):
            raise ValueError("provider source pin")
        if (value["model"] != "nvidia/nemotron-3.5-super-vl-120b-a12b"
                or value["max_calls"] != 2 or type(value["timeout_s"]) is not int
                or not 1 <= value["timeout_s"] <= 240 or (production and value["timeout_s"] != 240)):
            raise ValueError("provider model/budget contract")
        return dict(value)

    def execute(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        fields = {"schema", "token", "nonce", "argv", "timeout_s"}
        if set(request) != fields or request.get("schema") != REQUEST_SCHEMA:
            raise ValueError("request fields")
        token, nonce = request["token"], request["nonce"]
        if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
            raise PermissionError("invalid token")
        if not isinstance(nonce, str) or NONCE.fullmatch(nonce) is None:
            raise ValueError("nonce format")
        with self.lock:
            if nonce in self.seen:
                raise PermissionError("replayed nonce")
            self.seen.add(nonce)
        timeout = request["timeout_s"]
        if type(timeout) is not int or not 1 <= timeout <= self.max_timeout:
            raise ValueError("bounded timeout")
        argv = validate_argv(request["argv"], executable=self.executable, allowed_commands=self.allowed)
        before = utc()
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                                    check=False, shell=False)
            code, out, err, timed_out = result.returncode, result.stdout, result.stderr, False
        except subprocess.TimeoutExpired as error:
            code, out, err, timed_out = None, error.stdout or b"", error.stderr or b"", True
        response = {
            "schema": RESPONSE_SCHEMA,
            "nonce": nonce,
            "argv_sha256": hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "exit_code": code,
            "timed_out": timed_out,
            "stdout_b64": base64.b64encode(out).decode("ascii"),
            "stderr_b64": base64.b64encode(err).decode("ascii"),
            "request_before_utc": before,
            "response_after_utc": utc(),
        }
        write_new(self.receipt_root / f"{nonce}.json", {key: value for key, value in response.items()
                                                        if key not in {"stdout_b64", "stderr_b64"}})
        return 200, response

    def execute_provider(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.provider is None:
            raise PermissionError("provider endpoint disabled")
        fields = {"schema", "token", "nonce", "payload", "timeout_s"}
        if set(request) != fields or request.get("schema") != PROVIDER_REQUEST_SCHEMA:
            raise ValueError("provider request fields")
        token, nonce, payload = request["token"], request["nonce"], request["payload"]
        if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
            raise PermissionError("invalid token")
        if not isinstance(nonce, str) or NONCE.fullmatch(nonce) is None:
            raise ValueError("nonce format")
        if (not isinstance(payload, dict)
                or set(payload) != {"messages", "params", "messages_sha256", "params_sha256"}
                or digest(payload["messages"]) != payload["messages_sha256"]
                or digest(payload["params"]) != payload["params_sha256"]
                or payload["params"].get("model") != self.provider["model"]):
            raise ValueError("provider payload hash/model")
        timeout = request["timeout_s"]
        if timeout != self.provider["timeout_s"]:
            raise ValueError("provider timeout contract")
        with self.lock:
            if nonce in self.seen:
                raise PermissionError("replayed nonce")
            if self.provider_calls >= self.provider["max_calls"]:
                raise PermissionError("provider call cap exhausted")
            self.seen.add(nonce)
            self.provider_calls += 1
            call_number = self.provider_calls
        source = Path(self.provider["source"]).resolve(strict=True)
        if sha(source) != self.provider["source_sha256"]:
            raise ValueError("provider source changed")
        command = [self.provider["python"], str(source), "provider-once"]
        before = utc()
        try:
            result = subprocess.run(command, cwd=self.provider["cwd"], input=json.dumps(payload), text=True,
                                    encoding="utf-8", errors="strict", stdin=None,
                                    capture_output=True, timeout=timeout, check=False, shell=False)
            code, out, timed_out = result.returncode, result.stdout, False
        except subprocess.TimeoutExpired:
            code, out, timed_out = None, "", True
        worker = None
        if code == 0:
            try:
                worker = json.loads(out)
            except json.JSONDecodeError:
                worker = None
        valid = (isinstance(worker, dict) and worker.get("schema") == "agentic_provider_once_v2"
                 and worker.get("messages_sha256") == payload["messages_sha256"]
                 and worker.get("params_sha256") == payload["params_sha256"])
        response = {"schema": PROVIDER_RESPONSE_SCHEMA, "nonce": nonce, "call_number": call_number,
                    "exit_code": code, "timed_out": timed_out, "valid_worker_result": valid,
                    "source_sha256": self.provider["source_sha256"],
                    "messages_sha256": payload["messages_sha256"],
                    "params_sha256": payload["params_sha256"],
                    "result": worker.get("result") if valid else None,
                    "request_before_utc": before, "response_after_utc": utc()}
        write_new(self.receipt_root / f"provider-{nonce}.json",
                  {key: value for key, value in response.items() if key != "result"})
        return 200, response


def handler_type(state: BridgeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ri97"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback
            status, body = 400, {"schema": RESPONSE_SCHEMA, "error": "invalid request"}
            try:
                if self.path not in {"/v1/call", "/v1/provider-once"}:
                    raise ValueError("endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("request size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("request object")
                status, body = (state.execute_provider(value) if self.path == "/v1/provider-once"
                                else state.execute(value))
            except PermissionError as error:
                status, body = 403, {"schema": RESPONSE_SCHEMA, "error": str(error)}
            except (ValueError, json.JSONDecodeError) as error:
                status, body = 400, {"schema": RESPONSE_SCHEMA, "error": str(error)}
            raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


class BridgeClient:
    def __init__(self, config: Mapping[str, Any]):
        required = {"schema", "host", "port", "token_file", "executable", "allowed_commands",
                    "max_timeout_s", "original_prefix"}
        if set(config) != required or config.get("schema") != SCHEMA or config["host"] != "127.0.0.1":
            raise ValueError("client config fields")
        if config["original_prefix"] != ["wsl", "-d", "Ubuntu", "--exec", config["executable"]]:
            raise ValueError("exact original WSL argv prefix")
        self.config = dict(config)
        self.token = read_token(Path(config["token_file"]))

    def call(self, args: Sequence[str], timeout: int) -> dict[str, Any]:
        nonce = secrets.token_hex(16)
        argv = validate_argv([self.config["executable"], *args], executable=self.config["executable"],
                             allowed_commands=frozenset(self.config["allowed_commands"]))
        request = {"schema": REQUEST_SCHEMA, "token": self.token, "nonce": nonce,
                   "argv": argv, "timeout_s": timeout}
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        connection = http.client.HTTPConnection(self.config["host"], self.config["port"], timeout=timeout + 5)
        try:
            connection.request("POST", "/v1/call", body=raw, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = json.loads(response.read())
        finally:
            connection.close()
        if response.status != 200 or body.get("schema") != RESPONSE_SCHEMA or body.get("nonce") != nonce:
            raise BridgeError(f"bridge request rejected: HTTP {response.status}")
        return body


class ProviderClient:
    def __init__(self, config: Mapping[str, Any]):
        fields = {"schema", "host", "port", "token_file", "model", "timeout_s", "source_sha256"}
        if (set(config) != fields or config.get("schema") != SCHEMA or config["host"] != "127.0.0.1"
                or config["model"] != "nvidia/nemotron-3.5-super-vl-120b-a12b"
                or type(config["timeout_s"]) is not int or not 1 <= config["timeout_s"] <= 240):
            raise ValueError("provider client config")
        self.config, self.token = dict(config), read_token(Path(config["token_file"]))

    def complete(self, messages: list[Any], params: dict[str, Any]) -> dict[str, Any]:
        payload = {"messages": messages, "params": params, "messages_sha256": digest(messages),
                   "params_sha256": digest(params)}
        request = {"schema": PROVIDER_REQUEST_SCHEMA, "token": self.token,
                   "nonce": secrets.token_hex(16), "payload": payload,
                   "timeout_s": self.config["timeout_s"]}
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        connection = http.client.HTTPConnection(self.config["host"], self.config["port"], timeout=245)
        try:
            connection.request("POST", "/v1/provider-once", body=raw,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = json.loads(response.read())
        finally:
            connection.close()
        if (response.status != 200 or body.get("schema") != PROVIDER_RESPONSE_SCHEMA
                or body.get("messages_sha256") != payload["messages_sha256"]
                or body.get("params_sha256") != payload["params_sha256"]
                or body.get("source_sha256") != self.config["source_sha256"]
                or body.get("valid_worker_result") is not True):
            raise BridgeError("provider completion pending; no automatic resend")
        result = body.get("result")
        if not isinstance(result, dict):
            raise BridgeError("provider result missing")
        return result


class PersistentBrevTransport:
    """Frozen lifecycle `_RawTransport`와 같은 receipt 인터페이스를 유지한다."""

    def __init__(self, root: Path, prefix: list[str], services: Any, *, config: Mapping[str, Any]):
        del services
        if prefix != config["original_prefix"]:
            raise ValueError("registered original argv changed")
        self.root, self.prefix, self.sequence = root, list(prefix), 0
        (root / "calls").mkdir(parents=True, exist_ok=False)
        self.client = BridgeClient(config)

    def call(self, label: str, args: list[str], timeout: int) -> dict[str, Any]:
        if LABEL.fullmatch(label) is None:
            raise ValueError("transport label")
        self.sequence += 1
        folder = self.root / "calls" / f"{self.sequence:04d}-{label}"
        folder.mkdir(parents=True, exist_ok=False)
        original = [*self.prefix, *args]
        before = utc()
        write_new(folder / "intent.json", {"command": original, "request_before_utc": before,
                                            "shell": False, "timeout_s": timeout,
                                            "bridge": {"host": "127.0.0.1", "port": self.client.config["port"]}})
        response = self.client.call(args, timeout)
        stdout = base64.b64decode(response["stdout_b64"], validate=True)
        stderr = base64.b64decode(response["stderr_b64"], validate=True)
        stdout_path, stderr_path = folder / "stdout.bin", folder / "stderr.bin"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        receipt = {"exit_code": response["exit_code"], "timed_out": response["timed_out"],
                   "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                   "request_before_utc": before,
                   "intent": {"path": str((folder / "intent.json").resolve()),
                              "sha256": sha(folder / "intent.json")},
                   "bridge_nonce": response["nonce"], "bridge_argv_sha256": response["argv_sha256"]}
        write_new(folder / "receipt.json", receipt)
        return receipt


@contextmanager
def transport_scope(lifecycle_module: Any, config: Mapping[str, Any]) -> Iterator[None]:
    original = lifecycle_module._RawTransport

    class BoundTransport(PersistentBrevTransport):
        def __init__(self, root: Path, prefix: list[str], services: Any):
            super().__init__(root, prefix, services, config=config)

    lifecycle_module._RawTransport = BoundTransport
    try:
        yield
    finally:
        lifecycle_module._RawTransport = original


def digest(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def serve(config_path: Path) -> None:
    state = BridgeState(read_object(config_path))
    server = ThreadingHTTPServer((state.host, state.port), handler_type(state))
    server.serve_forever(poll_interval=0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.serve != "serve":
        raise SystemExit("serve subcommand required")
    serve(args.config.resolve(strict=True))


if __name__ == "__main__":
    main()
