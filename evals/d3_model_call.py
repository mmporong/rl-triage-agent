"""D3 B4·B5 단일 모델 호출 한 번을 기록한다. WSL의 기존 NVIDIA 자격증명 환경에서 메인이 실행한다.

  python3 evals/d3_model_call.py --packet <B4|B5 packet.json> --output <기록.json> --model <모델 id>
evals/active_model_call.py와 같은 요청 인자(temperature 0, max_tokens 16384, reasoning_effort none)·기록 필드를 쓰고,
메시지만 evals/d3_agent.messages로 만든다. 인프라 오류만 retryable로 표시하며 재시도는 새 --output으로 한 번 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_agent import messages  # noqa: E402

PARAMS = {"temperature": 0, "max_tokens": 16384, "stream": False, "extra_body": {"reasoning_effort": "none"}}


def call(packet_path: Path, output: Path, model: str) -> int:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

    data = packet_path.read_bytes()
    sent = messages(json.loads(data))
    params: dict[str, Any] = {"model": model, **PARAMS}
    record = {"schema": "d3_model_call_v1", "packet_sha256": hashlib.sha256(data).hexdigest(),
              "messages_sha256": hashlib.sha256(json.dumps(sent, sort_keys=True, ensure_ascii=False,
                                                           separators=(",", ":")).encode()).hexdigest(),
              "started_at": datetime.now(timezone.utc).isoformat(), "request": params, "client_cache_used": False,
              "response_id": None, "provider_model": None, "finish_reason": None, "content": None,
              "input_tokens": None, "output_tokens": None, "total_tokens": None}
    wall, cpu = time.perf_counter(), time.process_time()
    exit_code = 0
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"],
                        timeout=300, max_retries=0)
        response = client.chat.completions.create(messages=sent, **params)
        record.update(response_id=response.id, provider_model=response.model)
        if response.choices:
            record.update(content=response.choices[0].message.content, finish_reason=response.choices[0].finish_reason)
        if response.usage is not None:
            record.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens)
        if not (record.get("content") or "").strip() or record["finish_reason"] == "length":
            record.update(infra_error="empty_or_truncated_response", retryable=True)
            exit_code = 2
    except Exception as exc:
        # 예외 문자열에는 요청 헤더·자격증명이 섞일 수 있으므로 종류만 보존한다.
        retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        record.update(error_type=type(exc).__name__, retryable=retryable)
        if retryable:
            record["infra_error"] = type(exc).__name__
        exit_code = 2
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), execution_wall_s=time.perf_counter() - wall,
                  cpu_s=time.process_time() - cpu, exit_code=exit_code)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raise SystemExit(call(args.packet, args.output, args.model))
