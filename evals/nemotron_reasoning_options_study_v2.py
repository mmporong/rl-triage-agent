"""Reasoning mode 비교 v2: 메시지 파일 SHA와 canonical digest를 분리한다.

v1은 pretty JSON의 raw SHA를 canonical message digest처럼 비교해 API 발송 전에
중단됐다. v2는 그 결과를 보존하고 동일 corpus·모드·ABBA·판정 규칙을 새 schema로
등록한다. 실제 provider 호출·승인·교정 권한은 이 모듈 자체에 없다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import nemotron_reasoning_options_study_v1 as V1

PLAN_SCHEMA = "nemotron_reasoning_options_study_plan_v2"
CALL_SCHEMA = "nemotron_reasoning_options_call_v2"
RESULT_SCHEMA = "nemotron_reasoning_options_result_v2"
V1_PROTOCOL = "bench/protocols/nemotron_reasoning_options_development_20261101_v1.json"
SOURCE_FILES = (
    "evals/nemotron_reasoning_options_study_v2.py",
    "tests/test_nemotron_reasoning_options_study_v2.py",
)


def prepare_plan(corpus_ref: Path) -> dict[str, Any]:
    """v1의 고정 설계를 유지하며 raw SHA와 canonical digest를 명명해 분리한다."""
    base = V1.prepare_plan(corpus_ref)
    _corpus, cases = V1._corpus_ref(corpus_ref, load_payloads=False)
    by_id = {row["case_id"]: row for row in cases}
    calls = []
    for old in base["calls"]:
        row = deepcopy(old)
        raw_sha = row.pop("messages_sha256")
        source = by_id[row["case_id"]]
        V1.require(raw_sha == source["messages_sha256"], "v1/raw messages SHA 결합 불일치")
        row.update(messages_file_sha256=raw_sha,
                   messages_digest=source["messages_digest"])
        calls.append(row)
    closure = deepcopy(base["source_sha256_lf"])
    closure.update({name: V1.lf_sha(ROOT / name) for name in SOURCE_FILES})
    closure[V1_PROTOCOL] = V1.lf_sha(ROOT / V1_PROTOCOL)
    return {
        **{key: deepcopy(value) for key, value in base.items()
           if key not in {"schema", "calls", "source_sha256_lf"}},
        "schema": PLAN_SCHEMA,
        "prior_invalid": {
            "protocol": {"path": V1_PROTOCOL, "sha256_lf": V1.lf_sha(ROOT / V1_PROTOCOL)},
            "result_schema": V1.RESULT_SCHEMA,
            "status": "invalid", "api_requests": 0, "intent_files": 0,
            "failure_stage": "before_first_dispatch",
            "cause": "raw_messages_file_sha_was_compared_to_canonical_messages_digest",
            "semantic_retry": False,
        },
        "calls": calls,
        "source_sha256_lf": closure,
    }


def verify_plan(plan_path: Path, corpus_path: Path) -> dict[str, Any]:
    plan = V1._read(plan_path)
    V1.require(plan == prepare_plan(corpus_path), "v2 plan이 현재 corpus/source와 다르다")
    V1._corpus_ref(corpus_path, load_payloads=True)
    V1.require(len(plan["calls"]) == V1.API_CAP
               and [row["call_index"] for row in plan["calls"]]
               == list(range(1, V1.API_CAP + 1)), "v2 32회 계획 불일치")
    return plan


def verify_registered(plan_path: Path, corpus_path: Path, anchor: str) -> dict[str, Any]:
    V1.require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    plan_path = Path(plan_path).resolve()
    V1.require(plan_path.is_relative_to(ROOT / "bench" / "protocols")
               and plan_path.suffix == ".json", "v2 plan은 공개 protocol 경로여야 한다")
    plan = verify_plan(plan_path, corpus_path)
    V1._git("cat-file", "-e", f"{anchor}^{{commit}}")
    V1._git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = plan_path.relative_to(ROOT).as_posix()
    V1.require(V1._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
               == plan_path.read_bytes().replace(b"\r\n", b"\n"),
               "anchor v2 plan blob 불일치")
    for source, expected in plan["source_sha256_lf"].items():
        blob = V1._git("show", f"{anchor}:{source}").replace(b"\r\n", b"\n")
        V1.require(V1.hashlib.sha256(blob).hexdigest() == expected,
                   f"anchor v2 source blob 불일치: {source}")
    return plan


def execute(plan_path: Path, corpus_path: Path, anchor: str, out: Path, *,
            complete: Callable[[dict[str, Any]], dict[str, Any]] = V1.openai_complete,
            registration_check: Callable[[Path, Path, str], dict[str, Any]] = verify_registered,
            ) -> dict[str, Any]:
    """등록된 v2 32회 호출. 테스트의 registration_check 주입은 실제 등록 주장이 아니다."""
    plan = registration_check(plan_path, corpus_path, anchor)
    _corpus, checked = V1._corpus_ref(corpus_path, load_payloads=True)
    cases = {row["case_id"]: row for row in checked}
    out = Path(out).resolve()
    V1.require(not out.exists(), "v2 실행 out은 fresh 경로여야 한다")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        for planned in plan["calls"]:
            V1.require(verify_plan(plan_path, corpus_path) == plan,
                       "실행 중 v2 plan/corpus/source 변경")
            V1.require(len(rows) < V1.API_CAP, "v2 API cap 초과")
            case = cases[planned["case_id"]]
            V1.require(case["messages_sha256"] == planned["messages_file_sha256"]
                       and case["messages_digest"] == planned["messages_digest"],
                       "v2 raw/canonical messages 결합 변경")
            request = V1.build_request(case["messages"], planned["mode"])
            V1.require(V1.O2.O1._digest(request["messages"]) == planned["messages_digest"],
                       "호출 직전 canonical messages 변경")
            call_root = out / "calls" / f"{planned['call_index']:02d}"
            intent = {"schema": "nemotron_reasoning_options_intent_v2",
                      "planned": deepcopy(planned), "request": deepcopy(request),
                      "messages_file_sha256": planned["messages_file_sha256"],
                      "messages_digest": planned["messages_digest"],
                      "request_sha256": V1.digest(request),
                      "created_at": datetime.now(timezone.utc).isoformat()}  # noqa: UP017
            V1._write_new(call_root / "intent.json", intent)
            result = complete(deepcopy(request))
            V1.require(isinstance(result, dict), "provider 결과는 dict여야 한다")
            result = deepcopy(result)
            result.setdefault("reasoning_tokens", None)
            result.setdefault("reasoning_tokens_status", "unknown")
            wall = result.get("wall_s")
            V1.require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0,
                       "provider wall_s 불일치")
            review, outcome = V1._review(result, case["state"])
            record = {"schema": CALL_SCHEMA, "planned": deepcopy(planned),
                      "request_sha256": V1.digest(request), "response": result,
                      "response_field_names": sorted(result), "review": review,
                      "model_action": outcome,
                      "normal_model_probe_proposed": (planned["evaluation_role"] == "normal_model_only"
                                                       and outcome == "run_probe"),
                      "actual_system_gate_executed": False,
                      "approval": False, "repair": False, "recovery": False}
            V1._write_new(call_root / "private.json", record)
            V1._write_new(call_root / "public.json", V1._public_call(record))
            rows.append({"call_index": planned["call_index"], "case_id": planned["case_id"],
                         "mode": planned["mode"], "replicate": planned["replicate"],
                         "evaluation_role": planned["evaluation_role"],
                         "valid": review.get("status") == "valid", "action": outcome,
                         "finish_reason": result.get("finish_reason"),
                         "provider_model": result.get("provider_model"),
                         "input_tokens": result.get("input_tokens"),
                         "output_tokens": result.get("output_tokens"),
                         "total_tokens": result.get("total_tokens"),
                         "reasoning_tokens": result.get("reasoning_tokens"),
                         "reasoning_tokens_status": result.get("reasoning_tokens_status"),
                         "wall_s": wall,
                         "private_sha256": V1.file_sha(call_root / "private.json"),
                         "public_sha256": V1.file_sha(call_root / "public.json")})
            if planned["mode"] == "enabled_low" and result.get("error_type"):
                raise RuntimeError("enabled_low provider 지원 확인 실패; fallback 없이 중단")
            if result.get("error_type"):
                raise RuntimeError("provider 호출 오류; 재시도 없이 중단")
        V1.require(len(rows) == V1.API_CAP, "v2 32회 호출이 완결되지 않았다")
        V1.require(verify_plan(plan_path, corpus_path) == plan,
                   "종료 전 v2 plan/corpus/source 변경")
        final = {"schema": RESULT_SCHEMA, "status": "complete",
                 "plan": {"sha256": V1.file_sha(plan_path)},
                 "corpus": {"sha256": V1.file_sha(corpus_path)}, "calls": rows,
                 "request_count": len(rows), "api_cap": V1.API_CAP,
                 "parent_wall_s": time.perf_counter() - started,
                 "prior_invalid_api_requests": 0,
                 "claims": {"model_or_reasoning_advantage": False,
                            "new_seed_effect": False, "default_adoption": False,
                            "approval": False, "repair": False, "recovery": False}}
        V1._write_new(out / "result.json", final)
        return final
    except BaseException as error:
        V1._write_new(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
                      "error_type": type(error).__name__, "error": str(error),
                      "completed_calls": len(rows), "api_cap": V1.API_CAP,
                      "automatic_retry": False, "fallback": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-plan")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-plan")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--corpus", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-plan":
        V1._write_new(args.out, prepare_plan(args.corpus))
    elif args.command == "verify-plan":
        verify_plan(args.plan, args.corpus)
    else:
        V1.require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")),
                   "실제 API 실행은 등록 뒤 WSL 단일 process에서만 허용한다")
        execute(args.plan, args.corpus, args.anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CALL_SCHEMA", "PLAN_SCHEMA", "RESULT_SCHEMA", "execute", "main",
           "prepare_plan", "verify_plan", "verify_registered"]
