"""E4b v2: 고친 NAT 도구(nat_demo_tools_v2)와 v2 설정으로 E4 진단·교정 packet 하나의 결정을 한 번 돌리고 기록한다.

v1(evals/nat_decision_call.py)은 v1 사전등록 결과의 소스 결합 때문에 바꾸지 않고, 사건 기록·요약·오류 분류·ATIF·
계약 재검증·프롬프트 구성 함수를 그대로 재사용한다. call()은 설정·도구 모듈·기록 스키마가 모듈 상수에 묶여 있어
복제했고, v1과 달리 기록에 제출 전 계약 점검 호출 수와 상한, v1·v2 도구 모듈 SHA를 남긴다. 사용자 메시지와
판정 순서(거절 기록 > 제출 파일 > 실행 오류 > 빈 응답 > 미제출)는 v1과 같다.

  python evals/nat_decision_call_v2.py --kind diagnosis|correction --packet <packet.json> --output <run.json>
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
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

import nat_decision_call as W1  # noqa: E402

RECORD_SCHEMA = "e4b_nat_decision_v2"
CONFIGS = {"diagnosis": ROOT / "configs/nat_demo_v2_diagnosis.yml",
           "correction": ROOT / "configs/nat_demo_v2_correction.yml"}
TOOLS_MODULE = ROOT / "src/rl_triage/nat_demo_tools_v2.py"
BASE_TOOLS_MODULE = W1.TOOLS_MODULE


def tool_rules(kind: str) -> str:
    import yaml
    return yaml.safe_load(CONFIGS[kind].read_text(encoding="utf-8"))["workflow"]["system_prompt"]


def system_prompt(kind: str) -> str:
    return W1.system_prompt(kind, tool_rules(kind))


def precheck_calls(trace_path: Path) -> int:
    from rl_triage import nat_demo_tools_v2 as T
    return T.precheck_calls(str(trace_path))


async def run_workflow(config_path: Path, prompt: str, message: str, steps: list) -> str:
    """실제 NAT 워크플로 한 번(v2 도구 등록). steps에 IntermediateStep 객체를 순서대로 모은다."""
    from nat.builder.context import Context
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
    from nat.runtime.loader import load_config
    from nat.runtime.session import SessionManager
    from rl_triage import nat_demo_tools_v2  # noqa: F401  (e4b2_* 함수 등록)

    config = load_config(config_path)
    config.workflow.system_prompt = prompt
    config.llms["nemotron"].callbacks = [LangchainProfilerHandler()]
    async with WorkflowBuilder.from_config(config=config) as builder:
        manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            async with manager.run(message) as runner:
                subscription = Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
                try:
                    return await asyncio.wait_for(runner.result(to_type=str), timeout=W1.WORKFLOW_TIMEOUT_S)
                finally:
                    subscription.unsubscribe()
        finally:
            await manager.shutdown()


def call(kind: str, packet_path: Path, output: Path, *, runner=run_workflow, require_posix: bool = True) -> int:
    from rl_triage import nat_demo_tools_v2 as T
    if kind not in CONFIGS:
        raise ValueError(f"모르는 결정 종류 {kind}")
    if require_posix and os.name != "posix":
        raise RuntimeError("NAT worker는 기존 WSL 자격증명 환경에서 실행한다")
    paths = {suffix: output.with_suffix(suffix) for suffix in
             (".events.json", ".atif.json", ".tools.jsonl", ".submission.json", ".submission.rejected.json")}
    if output.exists() or any(p.exists() for p in paths.values()):
        raise FileExistsError("기존 E4b 기록은 덮어쓰지 않는다")
    packet = json.loads(packet_path.read_bytes())
    prompt = system_prompt(kind)
    record = {"schema": RECORD_SCHEMA, "kind": kind, "model": W1.MODEL, "temperature": 0,
              "packet_sha256": W1.raw_sha(packet_path), "config_sha256_lf": W1.lf_sha(CONFIGS[kind]),
              "tools_module_sha256_lf": W1.lf_sha(TOOLS_MODULE),
              "base_tools_module_sha256_lf": W1.lf_sha(BASE_TOOLS_MODULE),
              "system_prompt_sha256": W1.text_sha(prompt), "user_message_sha256": None, "nat_version": None,
              "precheck_limit": T.PRECHECK_LIMIT, "precheck_calls": None,
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "invalid", "error_type": None,
              "error_stage": None, "error_frames": [], "decision": None, "submission_sha256": None,
              "final_text_sha256": None, "semantic_retries": 0, "provider_cache_status": "not_attested"}
    steps, stage, run_error, final = [], "preflight", None, None
    wall, cpu = time.perf_counter(), time.process_time()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths[".tools.jsonl"].open("x", encoding="utf-8").close()
    submission, rejected = paths[".submission.json"], paths[".submission.rejected.json"]
    try:
        if packet.get("schema") != W1.PACKET_SCHEMAS[kind]:
            raise ValueError("packet 형식이 결정 종류와 다르다")
        if kind == "diagnosis" and (packet.get("phase", "initial") != "initial" or packet.get("observed")):
            raise ValueError("E4b 진단은 관측 전 초기 packet만 쓴다")
        message = W1.user_message(kind, packet)
        record["user_message_sha256"] = W1.text_sha(message)
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("자격증명 환경변수가 없다")
        record["nat_version"] = version("nvidia-nat")
        if record["nat_version"] != W1.NAT_VERSION:
            raise ValueError("NAT 버전이 등록 값과 다르다")
        os.environ.update(E4B_KIND=kind, E4B_PACKET=str(packet_path.resolve()), E4B_PACKET_SHA256=record["packet_sha256"],
                          E4B_SUBMISSION=str(submission), E4B_TRACE=str(paths[".tools.jsonl"]))
        stage = "workflow"
        final = asyncio.run(runner(CONFIGS[kind], prompt, message, steps))
        record["final_text_sha256"] = W1.text_sha(final or "")
    except Exception as error:  # noqa: BLE001 - 아래에서 제출 상태와 함께 분류한다
        run_error = error
    try:
        if rejected.exists():
            stage = "submission"
            raise ValueError("제출이 계약 검증에서 거절됐거나 두 번째 제출이 있었다")
        if submission.exists():
            stage = "submission"
            body = json.loads(submission.read_bytes())
            record.update(decision=W1.validate_submission(kind, packet, body), submission_sha256=W1.raw_sha(submission),
                          status="valid")
            if run_error is not None:  # 제출 뒤 마지막 응답 단계의 오류는 결정을 버리지 않고 따로 남긴다
                record["post_submission_error"] = {"type": type(run_error).__name__,
                                                   "class": W1.classify_error(run_error)}
        elif run_error is not None:
            raise run_error
        elif not (final or "").strip():
            record.update(status="infra_error", error_type="EmptyFinalWithoutSubmission", error_stage="submission")
        else:
            stage = "submission"
            raise ValueError("결정을 제출하지 않고 끝났다")
    except Exception as error:  # noqa: BLE001 - 상태를 분류해 기록으로 남긴다
        record.update(status="invalid" if rejected.exists() else W1.classify_error(error),
                      error_type=type(error).__name__, error_stage=stage, decision=None, submission_sha256=None,
                      error_frames=[{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                    for f in extract_tb(error.__traceback__)[-8:]])
    rows = [W1.event_row(step) for step in steps]
    with paths[".events.json"].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"schema": W1.EVENTS_SCHEMA, "packet_sha256": record["packet_sha256"], "events": rows}, handle,
                  ensure_ascii=False, indent=1, allow_nan=False, default=str)
        handle.write("\n")
    record.update(events_sha256=W1.raw_sha(paths[".events.json"]), tools_trace_sha256=W1.raw_sha(paths[".tools.jsonl"]),
                  precheck_calls=precheck_calls(paths[".tools.jsonl"]), **W1.summarize_events(rows))
    try:
        record.update(atif_sha256=W1.write_atif(steps, paths[".atif.json"], kind), atif_error=None)
    except Exception as error:  # noqa: BLE001 - 궤적 변환 실패는 기록하고 결정 판정은 바꾸지 않는다
        record.update(atif_sha256=None, atif_error=f"{type(error).__name__}: {error}"[:300])
    if rejected.exists():
        record["rejection_sha256"] = W1.raw_sha(rejected)
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
