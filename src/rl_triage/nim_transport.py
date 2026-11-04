"""고정 NVIDIA endpoint의 단일 요청 전송. 예약과 재사용은 기존 journal이 소유한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
PARAMS = {"model": MODEL, "temperature": 0, "max_tokens": 16384,
          "stream": False, "reasoning_effort": "none"}
HTTP_TIMEOUT_S = 300
BRIDGE_TIMEOUT_S = 360


def _request_body(messages: list, params: dict) -> bytes:
    if params != PARAMS or not isinstance(messages, list) or not messages:
        raise ValueError("registered NVIDIA request parameters required")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("text-only registered messages required")
    return json.dumps({**params, "messages": messages}, ensure_ascii=False,
                      sort_keys=True, allow_nan=False).encode("utf-8")


def complete_local(messages: list, params: dict, *, environ=None, opener=urlopen) -> dict:
    """키는 메모리 Header에만 사용하며 오류 본문·Header를 반환하지 않는다."""
    body = _request_body(messages, params)
    environment = os.environ if environ is None else environ
    key = environment.get("NVIDIA_API_KEY")
    if not isinstance(key, str) or not key.strip():
        return {"status": "error", "error_type": "MissingCredential", "http_status": None}
    request = Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with opener(request, timeout=HTTP_TIMEOUT_S) as response:
            value = json.loads(response.read().decode("utf-8"))
            status_code = response.status
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("one provider choice required")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("text content unavailable")  # noqa: TRY004 - JSON 응답 계약 오류
        usage = value.get("usage", {})
        tokens = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            token_count = usage.get(name) if isinstance(usage, dict) else None
            if token_count is not None and (type(token_count) is not int or token_count < 0):
                raise ValueError("invalid provider token usage")
            tokens[name] = token_count
        return {"status": "ok", "status_code": status_code, "content": content,
                "finish_reason": choice.get("finish_reason"), "model_returned": value.get("model"),
                "usage": tokens, "wall_s": time.perf_counter() - started,
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "automatic_retries": 0, "actual_billing_usd": None, "provider_gpu_seconds": None}
    except HTTPError as error:
        return {"status": "error", "error_type": "HTTPError", "status_code": error.code,
                "wall_s": time.perf_counter() - started, "automatic_retries": 0}
    except (TimeoutError, URLError, OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "error", "error_type": type(error).__name__,
                "wall_s": time.perf_counter() - started, "automatic_retries": 0}


def _mounted(path: Path) -> str:
    pure = PureWindowsPath(str(path))
    if len(pure.drive) != 2 or pure.drive[1] != ":" or not pure.is_absolute():
        raise ValueError("local Windows drive required for WSL request bridge")
    return "/mnt/" + pure.drive[0].lower() + "/" + "/".join(pure.parts[1:])


def complete(messages: list, params: dict, *, runner=subprocess.run) -> dict:
    """Windows에서는 기존 WSL 자격증명 환경을 사용한다. 설치·재시작은 하지 않는다."""
    _request_body(messages, params)
    if os.name != "nt" or os.environ.get("NVIDIA_API_KEY"):
        return complete_local(messages, params)
    with tempfile.TemporaryDirectory(prefix="rl-triage-nim-") as temporary:
        input_path = Path(temporary) / "request.json"
        input_path.write_text(json.dumps({"messages": messages, "params": params},
                                         ensure_ascii=False, allow_nan=False), encoding="utf-8")
        script = _mounted(Path(__file__).resolve())
        request = _mounted(input_path.resolve())
        command = ('set -e; . "$HOME/.config/nvidia/env"; '
                   'exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.quote(script)
                   + " --request " + shlex.quote(request))
        try:
            result = runner(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=BRIDGE_TIMEOUT_S, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode != 0:
                return {"status": "error", "error_type": "CredentialBridgeError",
                        "exit_code": result.returncode, "automatic_retries": 0}
            value = json.loads(result.stdout)
            if not isinstance(value, dict) or value.get("status") not in {"ok", "error"}:
                raise ValueError("invalid bridge response")
            return value
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return {"status": "error", "error_type": type(error).__name__, "automatic_retries": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    arguments = parser.parse_args(argv)
    value = json.loads(arguments.request.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"messages", "params"}:
        raise ValueError("request envelope fields")
    result = complete_local(value["messages"], value["params"])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
