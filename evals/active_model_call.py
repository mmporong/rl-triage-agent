"""WSL의 기존 NVIDIA 자격증명 환경에서 새 inference 한 번을 기록한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from typing import cast, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.active_diagnosis import decision_messages  # noqa: E402


def call(packet_path: Path, output: Path, model: str) -> int:
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
    from openai.types.chat import ChatCompletionMessageParam

    data = packet_path.read_bytes()
    packet = json.loads(data)
    messages = decision_messages(packet)
    started = datetime.now(timezone.utc).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    params: dict[str, Any] = {"model": model, "temperature": 0, "max_tokens": 16384,
              "stream": False, "extra_body": {"reasoning_effort": "none"}}
    record = {"schema": "active_model_call_v1", "packet_sha256": hashlib.sha256(data).hexdigest(),
              "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False,
                                                            separators=(",", ":")).encode()).hexdigest(),
              "started_at": started, "request": params, "client_cache_used": False,
              "saved_response_cached": False,
              "response_id": None, "provider_model": None, "finish_reason": None, "reasoning_content": None,
              "input_tokens": None, "output_tokens": None, "total_tokens": None}
    exit_code = 0
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                        api_key=os.environ["NVIDIA_API_KEY"], timeout=300, max_retries=0)
        response = client.chat.completions.create(messages=cast(list[ChatCompletionMessageParam], messages), **params)
        record["response_id"] = response.id
        record["provider_model"] = response.model
        if response.choices:
            choice = response.choices[0]
            record.update(content=choice.message.content, finish_reason=choice.finish_reason,
                          reasoning_content=getattr(choice.message, "reasoning_content", None))
        else:
            record["content"] = ""
        if response.usage is not None:
            record.update(input_tokens=response.usage.prompt_tokens,
                          output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens)
        if not (record.get("content") or "").strip() or record["finish_reason"] == "length":
            record["infra_error"] = "empty_or_truncated_response"
            record["retryable"] = True
            exit_code = 2
    except Exception as exc:
        # 예외 문자열에는 요청 헤더·자격증명이 섞일 수 있으므로 종류만 보존한다.
        retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        record["error_type"] = type(exc).__name__
        record["retryable"] = retryable
        if retryable:
            record["infra_error"] = type(exc).__name__
        exit_code = 2
    record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                  execution_wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu,
                  exit_code=exit_code)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    pid_path = args.output.with_suffix(".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with pid_path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        code = call(args.packet, args.output, args.model)
    finally:
        pid_path.unlink(missing_ok=True)
    raise SystemExit(code)
