"""Nemotron 3.5 Super VL의 reasoning 요청 모드를 같은 메시지로 비교한다.

이 모듈은 기존 ``reasoning_effort=none`` 판을 수정하지 않는다. 공개 사전등록
manifest와 비공개 corpus를 결합한 뒤, case별 ABBA 순서로 단일 호출만 수행한다.
모델 응답은 검사 선택 제안의 개발 측정일 뿐 실행·승인·교정 권한이 없다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import registry_action_options_v2 as O2

CALL = O2.CALL
MODEL = CALL.MODEL
MODES = ("legacy_none", "enabled_low")
ORDER = (("legacy_none", 1), ("enabled_low", 1),
         ("enabled_low", 2), ("legacy_none", 2))
API_CAP = 32
PLAN_SCHEMA = "nemotron_reasoning_options_study_plan_v1"
CORPUS_SCHEMA = "nemotron_reasoning_options_corpus_v1"
CALL_SCHEMA = "nemotron_reasoning_options_call_v1"
RESULT_SCHEMA = "nemotron_reasoning_options_result_v1"
PARENT_PROTOCOL = "bench/protocols/diagnostic_option_study_20261101_v1.json"
SOURCE_FILES = (
    "evals/nemotron_reasoning_options_study_v1.py",
    "tests/test_nemotron_reasoning_options_study_v1.py",
    "evals/registry_action_options_v2.py",
    "evals/registry_action_options_v1.py",
    "evals/evidence_guided_selector_v1.py",
    "evals/selection_comparison_call_v2.py",
    "evals/feedback_model_call.py",
)
FORBIDDEN_STATE_KEYS = frozenset({
    "answer_key", "fault", "fault_id", "fault_label", "fault_type",
    "original_faults", "expected_diagnosis", "ground_truth",
})


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체가 아니다: {path}")
    return value


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def _relative(parent: Path, ref: dict[str, Any]) -> Path:
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["path"], str) and bool(ref["path"])
            and "\\" not in ref["path"] and ":" not in ref["path"]
            and all(part not in {"", ".", ".."} for part in ref["path"].split("/"))
            and _valid_sha(ref["sha256"]), "corpus 상대 ref 불일치")
    path = (parent / ref["path"]).resolve()
    require(path.is_relative_to(parent.resolve()) and path.is_file()
            and file_sha(path) == ref["sha256"], "corpus ref 원문 SHA 불일치")
    return path


def _role(host_gate: object) -> str:
    require(isinstance(host_gate, dict), "host_gate 형식 불일치")
    gate = cast(dict[str, Any], host_gate)
    if gate.get("run") is False:
        require(gate.get("reason") == "healthy_gate", "비실행 host gate 이유 불일치")
        return "normal_model_only"
    require(gate.get("run") is True, "fault 개발 행 host gate 불일치")
    return "fault_development"


def _corpus_ref(path: Path, *, load_payloads: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    value = _read(path)
    require(set(value) == {"schema", "source_anchor", "entries", "source_batch_sha256",
                           "source_protocol_sha256_lf", "same_source_measurements",
                           "post_exposure_development", "approval_granted", "GPU_executed"}
            and value["schema"] == CORPUS_SCHEMA
            and isinstance(value["source_anchor"], str)
            and re.fullmatch(r"[0-9a-f]{40}", value["source_anchor"]) is not None
            and _valid_sha(value["source_batch_sha256"])
            and _valid_sha(value["source_protocol_sha256_lf"])
            and value["same_source_measurements"] is True
            and value["post_exposure_development"] is True
            and value["approval_granted"] is False and value["GPU_executed"] is False,
            "corpus 정본 형식 불일치")
    entries = value["entries"]
    require(isinstance(entries, list) and len(entries) == 8
            and [row.get("case_id") for row in entries] == [f"c{i:02d}" for i in range(1, 9)],
            "corpus opaque case 8개 순서 불일치")
    checked = []
    for row in entries:
        require(isinstance(row, dict)
                and set(row) == {"case_id", "host_gate", "state", "messages", "messages_digest"}
                and _valid_sha(row["messages_digest"]), "corpus entry 필드 불일치")
        state_path = _relative(path.parent, row["state"])
        messages_path = _relative(path.parent, row["messages"])
        item = {"case_id": row["case_id"], "evaluation_role": _role(row["host_gate"]),
                "state_sha256": row["state"]["sha256"],
                "messages_sha256": row["messages"]["sha256"],
                "messages_digest": row["messages_digest"]}
        if load_payloads:
            state, messages = _read(state_path), json.loads(messages_path.read_text(encoding="utf-8"))
            require(isinstance(messages, list) and not (_walk_keys(state) & FORBIDDEN_STATE_KEYS),
                    "state/messages 형식 또는 결함 정답 노출")
            require(O2.messages(state) == messages and O2.O1._digest(messages) == row["messages_digest"],
                    "messages가 O2 state에서 재생성되지 않는다")
            item.update(state=state, messages=messages)
        checked.append(item)
    require(sum(row["evaluation_role"] == "normal_model_only" for row in checked) == 2,
            "normal model-only case는 2개여야 한다")
    return value, checked


def build_request(messages: list[dict[str, str]], mode: str) -> dict[str, Any]:
    """메시지는 바꾸지 않고 reasoning 관련 요청 필드만 모드별로 만든다."""
    require(mode in MODES, "등록되지 않은 reasoning mode")
    require(isinstance(messages, list) and bool(messages), "messages가 비어 있다")
    request = {"model": MODEL, "messages": deepcopy(messages), "temperature": 0,
               "max_tokens": 16384, "stream": False}
    if mode == "legacy_none":
        request["extra_body"] = {"reasoning_effort": "none"}
    else:
        request["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "low",
        }
    return request


def prepare_plan(corpus_ref: Path) -> dict[str, Any]:
    """corpus 정본 ref를 받아 결과 전 고정할 32회 ABBA 계획을 만든다."""
    corpus, cases = _corpus_ref(corpus_ref, load_payloads=False)
    parent = _read(ROOT / PARENT_PROTOCOL)
    require(parent.get("schema") == "diagnostic_option_study_protocol_v1",
            "parent diagnostic option protocol schema 불일치")
    closure = {**parent["inherited_source_sha256_lf"], **parent["source_sha256_lf"]}
    for name, expected in closure.items():
        require(lf_sha(ROOT / name) == expected, f"parent source closure 불일치: {name}")
    closure.update({name: lf_sha(ROOT / name) for name in SOURCE_FILES})
    closure[PARENT_PROTOCOL] = lf_sha(ROOT / PARENT_PROTOCOL)
    calls = []
    for case in cases:
        for position, (mode, replicate) in enumerate(ORDER, 1):
            calls.append({"call_index": len(calls) + 1, "case_id": case["case_id"],
                          "evaluation_role": case["evaluation_role"], "abba_position": position,
                          "mode": mode, "replicate": replicate,
                          "messages_sha256": case["messages_sha256"]})
    return {
        "schema": PLAN_SCHEMA,
        "status": "registered_only_after_commit",
        "model": MODEL,
        "accepted_provider_models": sorted(CALL.ACCEPTED_PROVIDER_MODELS),
        "parent_protocol": {"path": PARENT_PROTOCOL, "sha256_lf": lf_sha(ROOT / PARENT_PROTOCOL)},
        "corpus": {"sha256": file_sha(corpus_ref), "source_anchor": corpus["source_anchor"],
                   "entries_digest": digest(cases)},
        "modes": {
            mode: {key: value for key, value in build_request(
                [{"role": "user", "content": "<registered-messages>"}], mode).items()
                   if key not in {"messages"}}
            for mode in MODES
        },
        "design": {"cases": 8, "fault_development_cases": 6,
                   "normal_model_only_cases": 2, "order_per_case": [row[0] for row in ORDER],
                   "single_call_no_retry": True, "feedback_requests": 0,
                   "api_cap": API_CAP, "gpu_runs": 0, "approval": False,
                   "repair": False, "recovery": False,
                   "same_messages_between_modes": True,
                   "fault_labels_exposed": False,
                   "normal_calls_are_model_only_proposal_evaluation": True,
                   "actual_system_health_gate_unchanged": True},
        "analysis": {"transport_token_first": True,
                     "validity_before_action_comparison": True,
                     "paired_action_comparison_requires_both_valid": True,
                     "reasoning_surface_is_auxiliary_telemetry": True,
                     "missing_reasoning_tokens": "unknown_not_zero",
                     "unsupported_enabled_low": "stop_without_fallback",
                     "incomplete_or_length": "HOLD",
                     "development_inputs_already_observed": True,
                     "model_or_reasoning_advantage_claim": False,
                     "prediction": "requested low reasoning may change investigation actions; benefit and hosted execution of the mode are unconfirmed",
                     "adoption_rule": {
                         "scope": "development_action_candidate_only_not_default",
                         "complete_calls": 32,
                         "enabled_low_fault_valid_at_least": "10/12",
                         "enabled_low_fault_valid_not_below_legacy": True,
                         "enabled_low_fault_read_or_probe_above_legacy": True,
                         "distinct_fault_cases_with_valid_read_or_probe_at_least": 2,
                         "normal_valid_probe_proposals_not_above_legacy": True,
                         "enabled_low_incomplete_not_above_legacy": True,
                         "reasoning_surface_or_positive_reported_reasoning_tokens_required": True,
                         "actual_probe_confirmation_before_diagnosis_benefit": True,
                     },
                     "result_wording": {
                         "unsupported_or_incomplete": "indeterminate_preserve_calls_no_fallback",
                         "candidate_rule_pass": "post_exposure_action_candidate_not_diagnosis_or_recovery_improvement",
                         "same_or_worse": "not_improved_do_not_adopt",
                         "normal_proposals_increased": "safety_tradeoff_do_not_adopt",
                         "reasoning_unexposed": "requested_mode_recorded_effective_reasoning_unverified",
                     }},
        "calls": calls,
        "source_sha256_lf": closure,
    }


def verify_plan(plan_path: Path, corpus_path: Path) -> dict[str, Any]:
    plan = _read(plan_path)
    require(plan == prepare_plan(corpus_path), "plan이 현재 corpus/source와 다르다")
    _corpus_ref(corpus_path, load_payloads=True)
    require(len(plan["calls"]) == API_CAP
            and [row["call_index"] for row in plan["calls"]] == list(range(1, API_CAP + 1)),
            "32회 호출 계획 불일치")
    return plan


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    require(run.returncode == 0, f"git {' '.join(args)} 실패")
    return run.stdout


def verify_registered(plan_path: Path, corpus_path: Path, anchor: str) -> dict[str, Any]:
    """공개 plan과 source closure가 실행 anchor에 등록되었는지 확인한다."""
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    plan_path = Path(plan_path).resolve()
    require(plan_path.is_relative_to(ROOT / "bench" / "protocols")
            and plan_path.suffix == ".json", "plan은 공개 protocol 경로여야 한다")
    plan = verify_plan(plan_path, corpus_path)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = plan_path.relative_to(ROOT).as_posix()
    require(_git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
            == plan_path.read_bytes().replace(b"\r\n", b"\n"), "anchor plan blob 불일치")
    for source, expected in plan["source_sha256_lf"].items():
        blob = _git("show", f"{anchor}:{source}").replace(b"\r\n", b"\n")
        require(hashlib.sha256(blob).hexdigest() == expected,
                f"anchor source blob 불일치: {source}")
    return plan


def _usage_details(value: object) -> dict[str, Any]:
    if value is None:
        return {"reasoning_tokens": None, "reasoning_tokens_status": "unknown"}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    if not isinstance(value, dict):
        return {"reasoning_tokens": None, "reasoning_tokens_status": "unknown"}
    token = value.get("reasoning_tokens")
    if type(token) is int and token >= 0:
        return {"reasoning_tokens": token, "reasoning_tokens_status": "reported"}
    return {"reasoning_tokens": None, "reasoning_tokens_status": "unknown"}


def openai_complete(request: dict[str, Any]) -> dict[str, Any]:
    """WSL 실행 전용 단일 호출. 키 값이나 예외 문자열은 결과에 넣지 않는다."""
    from openai import OpenAI

    secret = os.environ.get("NVIDIA_API_KEY")
    require(isinstance(secret, str) and bool(secret), "NVIDIA_API_KEY가 없다")
    started = time.perf_counter()
    output: dict[str, Any] = {}
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=secret,
                        timeout=300, max_retries=0)
        response = client.chat.completions.create(**request)
        output.update(response_id=response.id, provider_model=response.model,
                      response_field_names=sorted(response.model_fields_set))
        if response.choices:
            choice = response.choices[0]
            message = choice.message
            output.update(content=message.content, finish_reason=choice.finish_reason,
                          message_field_names=sorted(message.model_fields_set),
                          reasoning_content=getattr(message, "reasoning_content", None),
                          reasoning=getattr(message, "reasoning", None))
        if response.usage is not None:
            details = _usage_details(getattr(response.usage, "completion_tokens_details", None))
            output.update(input_tokens=response.usage.prompt_tokens,
                          output_tokens=response.usage.completion_tokens,
                          total_tokens=response.usage.total_tokens, **details)
    except Exception as error:  # noqa: BLE001 - provider 예외 본문을 버리고 형식만 보존한다.
        output.update(error_type=type(error).__name__,
                      status_code=getattr(error, "status_code", None))
    output["wall_s"] = time.perf_counter() - started
    return output


def _review(result: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    provider_ok = result.get("provider_model") in CALL.ACCEPTED_PROVIDER_MODELS
    content = result.get("content")
    if (result.get("error_type") or not provider_ok or result.get("finish_reason") != "stop"
            or not isinstance(content, str) or not content.strip()):
        reason = ("length" if result.get("finish_reason") == "length" else
                  "provider_or_content_incomplete")
        return ({"status": "invalid", "codes": [reason], "response_text": content}, "HOLD")
    review = O2.assess(content, state)
    action = O2.select_model(review, state)
    outcome = action["action"] if review["status"] == "valid" else "HOLD"
    return review, outcome


def _public_call(record: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(record)
    result = public["response"]
    for key in ("reasoning_content", "reasoning"):
        value = result.pop(key, None)
        result[f"{key}_presence"] = ("present" if isinstance(value, str) and bool(value)
                                      else "absent_or_unexposed")
    public["public_projection"] = "reasoning_trace_removed"
    return public


def execute(plan_path: Path, corpus_path: Path, anchor: str, out: Path, *,
            complete: Callable[[dict[str, Any]], dict[str, Any]] = openai_complete) -> dict[str, Any]:
    """등록된 32회 호출을 한 번씩 수행한다. 오류·unsupported에 재시도하지 않는다."""
    plan = verify_registered(plan_path, corpus_path, anchor)
    _corpus, checked = _corpus_ref(corpus_path, load_payloads=True)
    cases = {row["case_id"]: row for row in checked}
    out = Path(out).resolve()
    require(not out.exists(), "실행 out은 fresh 경로여야 한다")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        for planned in plan["calls"]:
            require(verify_plan(plan_path, corpus_path) == plan,
                    "실행 중 plan/corpus/source 변경")
            require(len(rows) < API_CAP, "API cap 초과")
            case = cases[planned["case_id"]]
            request = build_request(case["messages"], planned["mode"])
            require(digest(request["messages"]) == planned["messages_sha256"],
                    "호출 직전 messages 변경")
            call_root = out / "calls" / f"{planned['call_index']:02d}"
            intent = {"schema": "nemotron_reasoning_options_intent_v1",
                      "planned": deepcopy(planned), "request": deepcopy(request),
                      "request_sha256": digest(request),
                      "created_at": datetime.now(timezone.utc).isoformat()}  # noqa: UP017 - Python 3.10 호환
            _write_new(call_root / "intent.json", intent)
            result = complete(deepcopy(request))
            require(isinstance(result, dict), "provider 결과는 dict여야 한다")
            result = deepcopy(result)
            result.setdefault("reasoning_tokens", None)
            result.setdefault("reasoning_tokens_status", "unknown")
            wall = result.get("wall_s")
            require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0,
                    "provider wall_s 불일치")
            review, outcome = _review(result, case["state"])
            record = {"schema": CALL_SCHEMA, "planned": deepcopy(planned),
                      "request_sha256": digest(request), "response": result,
                      "response_field_names": sorted(result), "review": review,
                      "model_action": outcome,
                      "normal_model_probe_proposed": (planned["evaluation_role"] == "normal_model_only"
                                                       and outcome == "run_probe"),
                      "actual_system_gate_executed": False,
                      "approval": False, "repair": False, "recovery": False}
            _write_new(call_root / "private.json", record)
            _write_new(call_root / "public.json", _public_call(record))
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
                         "private_sha256": file_sha(call_root / "private.json"),
                         "public_sha256": file_sha(call_root / "public.json")})
            if planned["mode"] == "enabled_low" and result.get("error_type"):
                raise RuntimeError("enabled_low provider 지원 확인 실패; fallback 없이 중단")
            if result.get("error_type"):
                raise RuntimeError("provider 호출 오류; 재시도 없이 중단")
        require(len(rows) == API_CAP, "32회 호출이 완결되지 않았다")
        require(verify_plan(plan_path, corpus_path) == plan,
                "종료 전 plan/corpus/source 변경")
        result = {"schema": RESULT_SCHEMA, "status": "complete",
                  "plan": {"sha256": file_sha(plan_path)},
                  "corpus": {"sha256": file_sha(corpus_path)}, "calls": rows,
                  "request_count": len(rows), "api_cap": API_CAP,
                  "parent_wall_s": time.perf_counter() - started,
                  "claims": {"model_or_reasoning_advantage": False,
                             "new_seed_effect": False, "default_adoption": False,
                             "approval": False, "repair": False, "recovery": False}}
        _write_new(out / "result.json", result)
        return result
    except BaseException as error:
        _write_new(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
                   "error_type": type(error).__name__, "error": str(error),
                   "completed_calls": len(rows), "api_cap": API_CAP,
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
        _write_new(args.out, prepare_plan(args.corpus))
    elif args.command == "verify-plan":
        verify_plan(args.plan, args.corpus)
    else:
        require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")),
                "실제 API 실행은 등록 뒤 WSL 단일 process에서만 허용한다")
        execute(args.plan, args.corpus, args.anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["API_CAP", "CORPUS_SCHEMA", "MODES", "ORDER",
           "build_request", "execute", "main", "prepare_plan", "verify_plan",
           "verify_registered"]
