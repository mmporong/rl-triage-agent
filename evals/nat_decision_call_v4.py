"""E4b v4: 진단은 실수 표기 고정·표기 일치 치환, 교정은 null 없는 스키마·packet 등록 표 검증으로 결정을 한 번 돌린다.

v1~v3 worker는 사전등록 결과의 소스 결합 때문에 바꾸지 않고 사건 기록·요약·오류 분류·ATIF·프롬프트 구성과 v2 워크플로
실행 함수를 재사용한다. run_call()은 설정·도구 모듈·기록 스키마·점검 집계·제출 재검증을 인자로 받는 일반판이다(v3
call()과 같은 판정 순서: 거절 기록 > 제출 파일 > 실행 오류 > 빈 응답 > 미제출). 교정 제출 재검증은 v1과 달리 packet에
실린 등록 표로 한다(도구 검증과 같은 표).

  python evals/nat_decision_call_v4.py --kind diagnosis|correction --packet <packet.json> --output <run.json>
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
import nat_decision_call_v2 as W2  # noqa: E402
import nat_decision_call_v3 as W3  # noqa: E402

RECORD_SCHEMA = "e4b_nat_decision_v4"
CONFIGS = {"diagnosis": ROOT / "configs/nat_demo_v4_diagnosis.yml", "correction": ROOT / "configs/nat_demo_v4_correction.yml"}
TOOLS_MODULES = {"v4": ROOT / "src/rl_triage/nat_demo_tools_v4.py", **W3.TOOLS_MODULES}


def system_prompt(kind: str) -> str:
    import yaml
    return W1.system_prompt(kind, yaml.safe_load(CONFIGS[kind].read_text(encoding="utf-8"))["workflow"]["system_prompt"])


def precheck_calls(kind: str, trace_path: Path) -> int:
    from rl_triage import nat_demo_tools_v4 as T4
    return T4.precheck_calls(kind, str(trace_path))


def validate_submission(kind: str, packet: dict, body: dict) -> dict:
    """제출 파일을 같은 계약으로 다시 검증한다. 교정은 packet 등록 표로 검증한다."""
    if kind == "diagnosis":
        return W1.validate_submission(kind, packet, body)
    import correction_model_call as CM
    if body.get("kind") != "correction":
        raise ValueError("제출 종류가 교정이 아니다")
    verdict = CM.validate_model_proposal(body["proposal"], packet, packet["registry"])
    if not verdict["accepted"] or verdict != body["validation"]:
        raise ValueError("교정 제출이 다시 검증되지 않는다")
    return {"proposal": body["proposal"], **verdict}


async def run_workflow(config_path: Path, prompt: str, message: str, steps: list) -> str:
    from rl_triage import nat_demo_tools_v4  # noqa: F401  (e4b4_* 등록, v1~v3 도구도 함께 등록된다)
    return await W2.run_workflow(config_path, prompt, message, steps)


def run_call(kind: str, packet_path: Path, output: Path, *, configs: dict, tools_modules: dict, record_schema: str,
             prompt: str, precheck, validate, runner, require_posix: bool = True) -> int:
    """NAT 결정 한 번과 기록. 기존 기록은 덮어쓰지 않는다."""
    if kind not in configs:
        raise ValueError(f"모르는 결정 종류 {kind}")
    if require_posix and os.name != "posix":
        raise RuntimeError("NAT worker는 기존 WSL 자격증명 환경에서 실행한다")
    paths = {suffix: output.with_suffix(suffix) for suffix in
             (".events.json", ".atif.json", ".tools.jsonl", ".submission.json", ".submission.rejected.json")}
    if output.exists() or any(p.exists() for p in paths.values()):
        raise FileExistsError("기존 E4b 기록은 덮어쓰지 않는다")
    packet = json.loads(packet_path.read_bytes())
    from rl_triage import nat_demo_tools_v2 as T2
    record = {"schema": record_schema, "kind": kind, "model": W1.MODEL, "temperature": 0,
              "packet_sha256": W1.raw_sha(packet_path), "config_sha256_lf": W1.lf_sha(configs[kind]),
              "tools_modules_sha256_lf": {name: W1.lf_sha(path) for name, path in tools_modules.items()},
              "system_prompt_sha256": W1.text_sha(prompt), "user_message_sha256": None, "nat_version": None,
              "precheck_limit": T2.PRECHECK_LIMIT, "precheck_calls": None,
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
        final = asyncio.run(runner(configs[kind], prompt, message, steps))
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
            record.update(decision=validate(kind, packet, body), submission_sha256=W1.raw_sha(submission), status="valid")
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
                  precheck_calls=precheck(kind, paths[".tools.jsonl"]), **W1.summarize_events(rows))
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


def call(kind: str, packet_path: Path, output: Path, *, runner=run_workflow, require_posix: bool = True) -> int:
    return run_call(kind, packet_path, output, configs=CONFIGS, tools_modules=TOOLS_MODULES, record_schema=RECORD_SCHEMA,
                    prompt=system_prompt(kind) if kind in CONFIGS else "", precheck=precheck_calls,
                    validate=validate_submission, runner=runner, require_posix=require_posix)


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
