"""E4b: E4 진단·교정 packet 하나로 NeMo Agent Toolkit tool_calling_agent 결정을 한 번 돌리고 기록한다.

WSL의 기존 NVIDIA 자격증명 환경에서 실행한다(키는 환경변수로만 받고 파일을 읽지 않는다). 시스템 프롬프트는
기존 판단 계약(active_diagnosis.SYSTEM_PROMPT, correction_model_call.SYSTEM)에 설정 파일의 도구 규칙을 붙인 것이다.
결정은 제출 도구가 쓴 파일에서만 읽고, 같은 계약 검증기로 다시 검증한다. 중간 단계 사건 전부(LLM·TOOL·WORKFLOW
시작/끝, 시각, 토큰, 입출력 앞부분)와 ATIF 궤적을 남긴다. 기존 기록은 덮어쓰지 않는다.

  python evals/nat_decision_call.py --kind diagnosis|correction --packet <packet.json> --output <run.json>
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
import time
from traceback import extract_tb

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

MODEL = "nvidia/nemotron-3-super-120b-a12b"
RECORD_SCHEMA = "e4b_nat_decision_v1"
EVENTS_SCHEMA = "e4b_nat_events_v1"
NAT_VERSION = "1.9.0"
WORKFLOW_TIMEOUT_S = 900
CONFIGS = {"diagnosis": ROOT / "configs/nat_demo_diagnosis.yml", "correction": ROOT / "configs/nat_demo_correction.yml"}
PACKET_SCHEMAS = {"diagnosis": "active_decision_packet_v2", "correction": "correction_model_packet_v1"}
TOOLS_MODULE = ROOT / "src/rl_triage/nat_demo_tools.py"
# 제공자·전송 계층 실패만 인프라 오류로 본다(같은 packet으로 사전등록 횟수만큼 재시도). 그 밖은 의미 오류.
INFRA_ERROR_TYPES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                     "ServiceUnavailableError", "TimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
                     "RemoteProtocolError"}
# NAT 1.9 tool_calling_agent는 빈 모델 응답(내용·도구 호출 없음)을 예외로 올린다. 단일 호출 경로처럼 인프라 오류로 본다.
EMPTY_RESPONSE_MARKERS = ("LLM returned an empty response",)


def lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def contract_prompt(kind: str) -> str:
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        return D.SYSTEM_PROMPT
    import correction_model_call as CM
    return CM.SYSTEM


def system_prompt(kind: str, tool_rules: str) -> str:
    return contract_prompt(kind).rstrip() + "\n\n" + tool_rules.strip() + "\n"


def user_message(kind: str, packet: dict) -> str:
    """사실·카탈로그·등록 표는 도구로 조회하게 하고, 나머지 공개 문맥만 넣는다."""
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        public = D._public_packet(packet)
        for key in ("evidence_facts", "probe_catalog", "thresholds"):
            public.pop(key)
        public["evidence_fact_count"] = len(packet["facts"])
        return (f"Decision phase: {packet.get('phase', 'initial')}. Sampling temperature: 0.\n"
                + json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return json.dumps({"confirmed_mechanism": packet["mechanism"], "fact_count": len(packet["facts"]),
                       "task": "Propose one registered correction and recovery mode, or HOLD, citing facts and "
                               "mapping every skill checklist item."},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_submission(kind: str, packet: dict, body: dict) -> dict:
    """제출 파일을 같은 계약으로 다시 검증한다. 결정(또는 제안과 검증 결과)을 돌려준다."""
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        if body.get("kind") != "diagnosis":
            raise ValueError("제출 종류가 진단이 아니다")
        return D.validate_decision(body["decision"], packet)
    import correction_model_call as CM
    if body.get("kind") != "correction":
        raise ValueError("제출 종류가 교정이 아니다")
    verdict = CM.validate_model_proposal(body["proposal"], packet)
    if not verdict["accepted"] or verdict != body["validation"]:
        raise ValueError("교정 제출이 다시 검증되지 않는다")
    return {"proposal": body["proposal"], **verdict}


def _head(value, limit: int = 600):
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


def event_row(step) -> dict:
    """NAT IntermediateStep 하나의 기록 행."""
    payload = step.payload
    usage = payload.usage_info.token_usage.model_dump() if payload.usage_info is not None else None
    data = payload.data
    return {"type": str(payload.event_type), "name": payload.name, "uuid": str(payload.UUID),
            "t": payload.event_timestamp, "span_t": payload.span_event_timestamp,
            "usage": None if usage is None else {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                                           "total_tokens")},
            "input_head": None if data is None else _head(data.input),
            "output_head": None if data is None else _head(data.output)}


def _is_tool(row: dict, phase: str) -> bool:
    # NAT 1.9는 등록 함수(도구) 호출을 FUNCTION_*로, 워크플로 자체를 이름 "<workflow>"의 FUNCTION_*로 남긴다.
    return (row["type"] in (f"TOOL_{phase}", f"FUNCTION_{phase}")
            and isinstance(row.get("name"), str) and not row["name"].startswith("<"))


def summarize_events(rows: list[dict]) -> dict:
    """LLM 호출 수·토큰, 도구 호출 순서·지연. 시작·끝 짝이 맞지 않으면 coverage가 incomplete다."""
    llm_start = [r for r in rows if r["type"] == "LLM_START"]
    llm_end = {r["uuid"]: r for r in rows if r["type"] == "LLM_END"}
    tool_start = [r for r in rows if _is_tool(r, "START")]
    tool_end = {r["uuid"]: r for r in rows if _is_tool(r, "END")}
    paired = {r["uuid"] for r in llm_start} == set(llm_end) and len(llm_start) == len(llm_end)
    usages = [llm_end[r["uuid"]]["usage"] for r in llm_start if r["uuid"] in llm_end]
    complete = paired and bool(usages) and all(
        isinstance(u, dict) and all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens"))
        for u in usages)
    latencies = [{"tool": r["name"], "latency_s": round(tool_end[r["uuid"]]["t"] - r["t"], 4)
                  if r["uuid"] in tool_end else None} for r in tool_start]
    return {"llm_calls": len(llm_start), "llm_call_coverage": "complete" if paired else "incomplete",
            "input_tokens": sum(u["prompt_tokens"] for u in usages) if complete else None,
            "output_tokens": sum(u["completion_tokens"] for u in usages) if complete else None,
            "tool_sequence": [r["name"] for r in tool_start], "tool_latencies": latencies,
            "tool_calls_completed": sum(1 for r in tool_start if r["uuid"] in tool_end)}


def classify_error(error: BaseException) -> str:
    """예외 사슬에 제공자·전송 계층 실패가 있으면 infra_error, 아니면 invalid."""
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in INFRA_ERROR_TYPES or isinstance(current, asyncio.TimeoutError):
            return "infra_error"
        if any(marker in str(current) for marker in EMPTY_RESPONSE_MARKERS):
            return "infra_error"
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return "infra_error"
        current = current.__cause__ or current.__context__
    return "invalid"


async def run_workflow(config_path: Path, prompt: str, message: str, steps: list) -> str:
    """실제 NAT 워크플로 한 번. steps에 IntermediateStep 객체를 순서대로 모은다."""
    from nat.builder.context import Context
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
    from nat.runtime.loader import load_config
    from nat.runtime.session import SessionManager
    from rl_triage import nat_demo_tools  # noqa: F401  (e4b_* 함수 등록)

    config = load_config(config_path)
    config.workflow.system_prompt = prompt
    config.llms["nemotron"].callbacks = [LangchainProfilerHandler()]
    async with WorkflowBuilder.from_config(config=config) as builder:
        manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            async with manager.run(message) as runner:
                subscription = Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
                try:
                    return await asyncio.wait_for(runner.result(to_type=str), timeout=WORKFLOW_TIMEOUT_S)
                finally:
                    subscription.unsubscribe()
        finally:
            await manager.shutdown()


def write_atif(steps: list, path: Path, kind: str) -> str:
    from nat.utils.atif_converter import IntermediateStepToATIFConverter
    trajectory = IntermediateStepToATIFConverter().convert(steps, session_id=path.stem, agent_name=f"e4b_{kind}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(trajectory.model_dump_json(indent=2) + "\n")
    return raw_sha(path)


def call(kind: str, packet_path: Path, output: Path, *, runner=run_workflow, require_posix: bool = True) -> int:
    if kind not in CONFIGS:
        raise ValueError(f"모르는 결정 종류 {kind}")
    if require_posix and os.name != "posix":
        raise RuntimeError("NAT worker는 기존 WSL 자격증명 환경에서 실행한다")
    paths = {suffix: output.with_suffix(suffix) for suffix in
             (".events.json", ".atif.json", ".tools.jsonl", ".submission.json", ".submission.rejected.json")}
    if output.exists() or any(p.exists() for p in paths.values()):
        raise FileExistsError("기존 E4b 기록은 덮어쓰지 않는다")
    packet = json.loads(packet_path.read_bytes())
    import yaml
    tool_rules = yaml.safe_load(CONFIGS[kind].read_text(encoding="utf-8"))["workflow"]["system_prompt"]
    prompt = system_prompt(kind, tool_rules)
    record = {"schema": RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0,
              "packet_sha256": raw_sha(packet_path), "config_sha256_lf": lf_sha(CONFIGS[kind]),
              "tools_module_sha256_lf": lf_sha(TOOLS_MODULE), "system_prompt_sha256": text_sha(prompt),
              "user_message_sha256": None, "nat_version": None,
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "invalid", "error_type": None,
              "error_stage": None, "error_frames": [], "decision": None, "submission_sha256": None,
              "final_text_sha256": None, "semantic_retries": 0, "provider_cache_status": "not_attested"}
    steps, stage, run_error, final = [], "preflight", None, None
    wall, cpu = time.perf_counter(), time.process_time()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths[".tools.jsonl"].open("x", encoding="utf-8").close()
    submission, rejected = paths[".submission.json"], paths[".submission.rejected.json"]
    try:
        if packet.get("schema") != PACKET_SCHEMAS[kind]:
            raise ValueError("packet 형식이 결정 종류와 다르다")
        if kind == "diagnosis" and (packet.get("phase", "initial") != "initial" or packet.get("observed")):
            raise ValueError("E4b 진단은 관측 전 초기 packet만 쓴다")
        message = user_message(kind, packet)
        record["user_message_sha256"] = text_sha(message)
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("자격증명 환경변수가 없다")
        record["nat_version"] = version("nvidia-nat")
        if record["nat_version"] != NAT_VERSION:
            raise ValueError("NAT 버전이 등록 값과 다르다")
        os.environ.update(E4B_KIND=kind, E4B_PACKET=str(packet_path.resolve()), E4B_PACKET_SHA256=record["packet_sha256"],
                          E4B_SUBMISSION=str(submission), E4B_TRACE=str(paths[".tools.jsonl"]))
        stage = "workflow"
        final = asyncio.run(runner(CONFIGS[kind], prompt, message, steps))
        record["final_text_sha256"] = text_sha(final or "")
    except Exception as error:  # noqa: BLE001 - 아래에서 제출 상태와 함께 분류한다
        run_error = error
    # 판정 순서: 거절 기록 > 제출 파일 > 실행 오류 > 빈 응답 > 미제출(사전등록 status_definitions)
    try:
        if rejected.exists():
            stage = "submission"
            raise ValueError("제출이 계약 검증에서 거절됐거나 두 번째 제출이 있었다")
        if submission.exists():
            stage = "submission"
            body = json.loads(submission.read_bytes())
            record.update(decision=validate_submission(kind, packet, body), submission_sha256=raw_sha(submission),
                          status="valid")
            if run_error is not None:  # 제출 뒤 마지막 응답 단계의 오류는 결정을 버리지 않고 따로 남긴다
                record["post_submission_error"] = {"type": type(run_error).__name__,
                                                   "class": classify_error(run_error)}
        elif run_error is not None:
            raise run_error
        elif not (final or "").strip():
            record.update(status="infra_error", error_type="EmptyFinalWithoutSubmission", error_stage="submission")
        else:
            stage = "submission"
            raise ValueError("결정을 제출하지 않고 끝났다")
    except Exception as error:  # noqa: BLE001 - 상태를 분류해 기록으로 남긴다
        record.update(status="invalid" if rejected.exists() else classify_error(error),
                      error_type=type(error).__name__, error_stage=stage, decision=None, submission_sha256=None,
                      error_frames=[{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                    for f in extract_tb(error.__traceback__)[-8:]])
    rows = [event_row(step) for step in steps]
    with paths[".events.json"].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"schema": EVENTS_SCHEMA, "packet_sha256": record["packet_sha256"], "events": rows}, handle,
                  ensure_ascii=False, indent=1, allow_nan=False, default=str)
        handle.write("\n")
    record.update(events_sha256=raw_sha(paths[".events.json"]), tools_trace_sha256=raw_sha(paths[".tools.jsonl"]),
                  **summarize_events(rows))
    try:
        record.update(atif_sha256=write_atif(steps, paths[".atif.json"], kind), atif_error=None)
    except Exception as error:  # noqa: BLE001 - 궤적 변환 실패는 기록하고 결정 판정은 바꾸지 않는다
        record.update(atif_sha256=None, atif_error=f"{type(error).__name__}: {error}"[:300])
    if paths[".submission.rejected.json"].exists():
        record["rejection_sha256"] = raw_sha(paths[".submission.rejected.json"])
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter() - wall,
                  cpu_s=time.process_time() - cpu)
    code = {"valid": 0, "invalid": 1, "infra_error": 2}[record["status"]]
    record["exit_code"] = code
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    return call(args.kind, args.packet, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
