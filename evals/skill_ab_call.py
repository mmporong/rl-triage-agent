"""스킬 유무 A/B(skill_ab_v1)의 프롬프트 하나를 WSL의 기존 NVIDIA 자격증명 환경에서 한 번 호출해 기록한다.

키는 환경변수로만 받고 파일을 읽지 않는다. 프롬프트는 skill_ab_compare.py plan이 만든 prompts.json에서 읽는다.
WSL venv에 저장소 의존성을 요구하지 않도록 openai와 표준 라이브러리만 쓴다. 재시도는 하지 않는다(plan이 판정).

  python evals/skill_ab_call.py --prompts <prompts.json> --prompt-id S1:with_skill --repeat 1 --attempt 1 --output <record.json>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

RECORD_SCHEMA = "skill_ab_call_v1"
PROMPTS_SCHEMA = "skill_ab_prompts_v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
# 기존 단일 호출(active_model_call.py)과 같은 설정. top_p는 보내지 않는다(제공자 기본값).
REQUEST: dict[str, Any] = {"model": MODEL, "temperature": 0, "max_tokens": 16384, "stream": False,
                           "extra_body": {"reasoning_effort": "none"}}
MAX_ATTEMPTS = 2
INFRA_EXCEPTIONS = ("APIConnectionError", "APITimeoutError")


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(value: Any) -> str:
    return text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def is_infra_exception(error: BaseException) -> bool:
    """연결·시간 초과·429·5xx만 인프라 오류로 본다. 예외 문자열은 쓰지 않는다(헤더가 섞일 수 있다)."""
    if any(cls.__name__ in INFRA_EXCEPTIONS for cls in type(error).__mro__):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def default_client() -> Any:
    from openai import OpenAI
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"],
                  timeout=300, max_retries=0)


def call(prompts_path: Path, prompt_id: str, repeat: int, attempt: int, output: Path, *,
         client_factory: Callable[[], Any] = default_client, require_posix: bool = True) -> int:
    # 실행 환경 전제가 틀리면 기록을 남기지 않고 멈춘다(호출 기록으로 세지 않는다).
    if require_posix and os.name != "posix":
        raise RuntimeError("skill A/B worker는 기존 WSL 자격증명 환경에서 실행한다")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("자격증명 환경변수가 없다")
    if repeat < 1 or attempt not in range(1, MAX_ATTEMPTS + 1):
        raise ValueError("반복 번호나 시도 번호가 등록 범위 밖이다")
    if output.exists():
        raise FileExistsError("기존 호출 기록은 덮어쓰지 않는다")
    data = prompts_path.read_bytes()
    prompts = json.loads(data)
    rows = [row for row in prompts.get("prompts", []) if row.get("prompt_id") == prompt_id]
    if prompts.get("schema") != PROMPTS_SCHEMA or len(rows) != 1:
        raise ValueError("prompts.json 형식이 다르거나 프롬프트 ID가 하나로 정해지지 않는다")
    row = rows[0]
    messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
    record: dict[str, Any] = {
        "schema": RECORD_SCHEMA, "prompt_id": prompt_id, "repeat": repeat, "attempt": attempt,
        "prompts_sha256": hashlib.sha256(data).hexdigest(), "system_sha256": text_sha(row["system"]),
        "user_sha256": text_sha(row["user"]), "messages_sha256": canonical_sha(messages),
        "request": deepcopy(REQUEST), "started_at": datetime.now(timezone.utc).isoformat(),
        "status": None, "infra_error": None, "error_type": None, "retryable": False,
        "response_id": None, "provider_model": None, "content": None, "finish_reason": None,
        "reasoning_content": None, "input_tokens": None, "output_tokens": None, "total_tokens": None,
    }
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        response = client_factory().chat.completions.create(messages=messages, **REQUEST)
        record["response_id"] = response.id
        record["provider_model"] = response.model
        if response.choices:
            choice = response.choices[0]
            record.update(content=choice.message.content, finish_reason=choice.finish_reason,
                          reasoning_content=getattr(choice.message, "reasoning_content", None))
        if response.usage is not None:
            record.update(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens)
        if not (record["content"] or "").strip() or record["finish_reason"] == "length":
            record.update(status="infra_error", infra_error="empty_or_truncated_response", retryable=True)
        else:
            record["status"] = "answered"
    except Exception as error:  # noqa: BLE001 - 종류만 남기고 인프라 여부로 나눈다
        retryable = is_infra_exception(error)
        record.update(error_type=type(error).__name__, retryable=retryable,
                      status="infra_error" if retryable else "call_error",
                      infra_error=type(error).__name__ if retryable else None)
    record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                  execution_wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if record["status"] == "answered" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return call(args.prompts, args.prompt_id, args.repeat, args.attempt, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
