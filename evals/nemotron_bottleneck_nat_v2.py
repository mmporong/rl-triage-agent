"""B3: 기존 NAT 도구 계약에서 모델만 바꾸고 HTTP 요청을 공통 원장에 결합한다."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import nat_decision_call as W1
import nat_decision_call_v2 as W2
import nemotron_bottleneck_ledger_v2 as L

SOURCE_FILES = {
    "evals/nemotron_bottleneck_nat_v2.py", "tests/test_nemotron_bottleneck_nat_v2.py",
    "evals/nat_decision_call.py", "evals/nat_decision_call_v2.py",
    "configs/nat_demo_v2_correction.yml", "configs/nat_demo_v2_diagnosis.yml",
    "src/rl_triage/nat_demo_tools.py", "src/rl_triage/nat_demo_tools_v2.py",
    "src/rl_triage/active_diagnosis.py", "evals/correction_model_call.py",
}

PRECHECK_TOOLS = {"e4b2_check_decision", "e4b2_check_proposal",
                  "e4b3_check_decision", "e4b3_check_proposal"}


def build_cases():
    import yaml
    specs = [
        ("e4b_v2_s2030_correction", "correction", W2.CONFIGS["correction"],
         ROOT / "bench/private/e2e_reward_demo/e2e_reward_s2030_20261013/model/correction.packet.json"),
        ("e4b_v3_h02_s2027_diagnosis", "diagnosis", W2.CONFIGS["diagnosis"],
         ROOT / "bench/private/e4b_v3/inputs/h02_s2027/diagnosis.packet.json"),
    ]
    out = []
    for name, kind, config, path in specs:
        packet = json.loads(path.read_bytes())
        loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
        system = W1.system_prompt(kind, loaded["workflow"]["system_prompt"])
        user = W1.user_message(kind, packet)
        out.append({"id": name, "bottleneck": "B3", "kind": kind, "packet": packet,
                    "packet_path": path, "config_path": config, "system_prompt": system, "user_message": user,
                    "binding": {"packet_sha256_raw": W1.raw_sha(path), "config_sha256_lf": W1.lf_sha(config),
                                "system_prompt_sha256": W1.text_sha(system), "user_message_sha256": W1.text_sha(user),
                                "max_iterations": loaded["workflow"]["max_iterations"]}})
    return out


def public_case(case):
    return {key: case[key] for key in ("id", "bottleneck", "kind", "binding")}


def decode_wire(body: bytes, status_code: int):
    """JSON 또는 SSE에서 내용·도구 인자를 복원한다. HTTP 오류 본문은 기록하지 않는다."""
    if status_code >= 400:
        return {"error_type": "HTTPStatusError", "status_code": status_code,
                "infra_error": status_code == 429 or status_code >= 500}
    text = body.decode("utf-8")
    if text.lstrip().startswith("{"):
        chunks = [json.loads(text)]
    else:
        chunks = [json.loads(line[5:].strip()) for line in text.splitlines()
                  if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")]
    result = {"content": "", "tool_calls": [], "provider_model": None, "finish_reason": None}
    tools = {}
    for chunk in chunks:
        if chunk.get("model"):
            result["provider_model"] = chunk["model"]
        if chunk.get("usage"):
            usage = chunk["usage"]
            result.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                          total_tokens=usage.get("total_tokens"))
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
            message = choice.get("message", choice.get("delta", {}))
            result["content"] += message.get("content") or ""
            for offset, tool in enumerate(message.get("tool_calls") or []):
                target = tools.setdefault(tool.get("index", offset), {"name": "", "arguments": ""})
                function = tool.get("function", {})
                target["name"] += function.get("name") or ""
                target["arguments"] += function.get("arguments") or ""
    result["tool_calls"] = [tools[index] for index in sorted(tools)]
    if result["finish_reason"] == "length":
        result.update(error_type="LengthFinish", infra_error=True)
    elif not result["content"].strip() and not result["tool_calls"]:
        result.update(error_type="EmptyResponse", infra_error=True)
    return result


def classify_tool_trace(rows):
    """점검 결과와 도구 오류를 원문 구조·해시로 분류한다.

    ``result_head``는 최대 400자라 일반 도구 결과에는 불완전할 수 있다. 점검 결과는
    짧은 JSON이어야 하므로 전체 결과 SHA와 일치할 때만 판정 근거로 쓴다.
    """
    prechecks = {"valid": 0, "contract_invalid": 0, "limit_exhausted": 0,
                 "malformed": 0, "tool_error": 0}
    tool_errors = {"precheck": 0, "submission": 0, "other": 0}
    for row in rows:
        tool = row.get("tool")
        if not isinstance(tool, str):
            continue
        if row.get("error"):
            category = ("precheck" if tool in PRECHECK_TOOLS else
                        "submission" if "submit_" in tool else "other")
            tool_errors[category] += 1
            if tool in PRECHECK_TOOLS:
                prechecks["tool_error"] += 1
            continue
        if tool not in PRECHECK_TOOLS:
            continue
        head, expected_sha = row.get("result_head"), row.get("result_sha256")
        if not isinstance(head, str) or W1.text_sha(head) != expected_sha:
            prechecks["malformed"] += 1
            continue
        try:
            value = json.loads(head)
        except (json.JSONDecodeError, TypeError):
            prechecks["malformed"] += 1
            continue
        state = (value.get("checked"), value.get("ok")) if isinstance(value, dict) else None
        if state == (True, True):
            prechecks["valid"] += 1
        elif state == (True, False):
            prechecks["contract_invalid"] += 1
        elif state == (False, False):
            prechecks["limit_exhausted"] += 1
        else:
            prechecks["malformed"] += 1
    return {"prechecks": prechecks, "tool_errors": tool_errors}


def nat_transport_status(raw, model):
    """NAT 도구 호출의 빈 content는 tool_calls가 있으면 정상 응답이다."""
    code = raw.get("status_code")
    if raw.get("error_type") or isinstance(code, int) and code >= 400:
        return ("infra_error" if code == 429 or isinstance(code, int) and code >= 500
                or raw.get("infra_error") else "call_error")
    if raw.get("provider_model") != model:
        return "call_error"
    if raw.get("finish_reason") == "length" or not ((raw.get("content") or "").strip()
                                                      or raw.get("tool_calls")):
        return "infra_error"
    return "ok"


def rejection_is_validation_error(case, rejected):
    """거절 산출물의 제출값이 동결 계약을 실제로 어겼을 때만 참이다."""
    if not rejected.exists():
        return False
    try:
        submitted = L.read(rejected)["submitted"]
        if case["kind"] == "diagnosis":
            from rl_triage import active_diagnosis as diagnosis
            try:
                diagnosis.validate_decision(submitted, case["packet"])
            except (ValueError, TypeError, KeyError):
                return True
            return False
        import correction_model_call as correction
        return correction.validate_model_proposal(submitted, case["packet"])["accepted"] is False
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def tool_argument_metrics(checks, errors, submission_valid, tool_reached=True,
                          submission_rejection_verified=False):
    """확인된 계약 오류와 원인을 확정할 수 없는 도구 오류를 분리한다."""
    confirmed_submission = int(submission_rejection_verified)
    count = checks["contract_invalid"] + confirmed_submission
    uncertain = errors["precheck"] + errors["other"] + max(0, errors["submission"] - confirmed_submission)
    if not tool_reached:
        status = "not_reached"
    elif count:
        status = "rejected_then_repaired" if submission_valid else "rejected_unrepaired"
    else:
        status = "pass"
    return {"tool_arg_issue_count": count, "tool_argument_status": status,
            "uncertain_tool_error_count": uncertain}


@contextmanager
def metered_http(book, case_id, model):
    """NAT의 비동기 HTTP 경계에서 모든 생성 요청을 센다. SDK 재시도는 설정상 0이다."""
    import httpx
    original_async, original_sync = httpx.AsyncClient.send, httpx.Client.send
    counter = 0
    first_request = {"request_count": 0, "transport_complete": True}

    async def send(client, request, *args, **kwargs):
        nonlocal counter
        if str(request.url) != "https://integrate.api.nvidia.com/v1/chat/completions" or request.method != "POST":
            raise ValueError("등록한 NAT 생성 endpoint 외 요청")
        payload = json.loads(request.content)
        if payload.get("model") != model:
            raise ValueError("NAT 전송 모델이 등록 모델과 다르다")
        if payload.get("temperature") != 0 or payload.get("reasoning_effort") != "none":
            raise ValueError("NAT 요청 설정이 기존 계약과 다르다")
        if counter == 0:
            first_request.update({"request_without_model_sha256": L.digest({k: v for k, v in payload.items() if k != "model"}),
                                  "tools_sha256": L.digest(payload.get("tools")), "stream": payload.get("stream", False)})
        counter += 1
        async def complete():
            response = await original_async(client, request, *args, **kwargs)
            raw = await response.aread()
            decoded = decode_wire(raw, response.status_code)
            if response.status_code < 400 and decoded.get("provider_model") != model:
                decoded.update(error_type="ProviderModelMismatch", infra_error=False)
            return decoded, response
        result, response = await book.async_call("B3", f"{case_id}_{model.split('/')[-1]}_http{counter}", payload, complete)
        first_request["request_count"] = counter
        if nat_transport_status(result, model) != "ok":
            first_request["transport_complete"] = False
        if result.get("error_type") == "ProviderModelMismatch":
            raise ValueError("NAT 응답 모델이 등록 모델과 다르다")
        return response

    def reject_sync(*args, **kwargs):
        raise RuntimeError("계측하지 않은 동기 NAT 전송은 허용하지 않는다")

    httpx.AsyncClient.send, httpx.Client.send = send, reject_sync
    try:
        yield first_request
    finally:
        httpx.AsyncClient.send, httpx.Client.send = original_async, original_sync


async def workflow(case, model, steps):
    from nat.builder.context import Context
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
    from nat.runtime.loader import load_config
    from nat.runtime.session import SessionManager
    # 두 등록 설정을 같은 프로세스에서 사용한다. v2가 v1을 import하는 구현 세부에
    # 기대지 않고 각 설정의 도구 등록 모듈을 명시한다.
    from rl_triage import nat_demo_tools, nat_demo_tools_v2  # noqa: F401

    config = load_config(case["config_path"])
    config.workflow.system_prompt = case["system_prompt"]
    config.llms["nemotron"].model_name = model
    config.llms["nemotron"].callbacks = [LangchainProfilerHandler()]
    if config.llms["nemotron"].max_retries != 0:
        raise ValueError("SDK 재시도는 0이어야 한다")
    async with WorkflowBuilder.from_config(config=config) as builder:
        manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            async with manager.run(case["user_message"]) as runner:
                subscription = Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
                try:
                    return await asyncio.wait_for(runner.result(to_type=str), timeout=W1.WORKFLOW_TIMEOUT_S)
                finally:
                    subscription.unsubscribe()
        finally:
            await manager.shutdown()


def run_case(case, model, output, book):
    if os.name != "posix" or version("nvidia-nat") != "1.9.0":
        raise RuntimeError("WSL의 등록 NAT 1.9.0 환경이 필요하다")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    trace, submission = output / "tools.jsonl", output / "submission.json"
    trace.touch(exist_ok=False)
    env = {"E4B_KIND": case["kind"], "E4B_PACKET": str(case["packet_path"]),
           "E4B_PACKET_SHA256": case["binding"]["packet_sha256_raw"],
           "E4B_SUBMISSION": str(submission), "E4B_TRACE": str(trace)}
    old_env = {key: os.environ.get(key) for key in env}
    steps, error_type, error_class, first_request = [], None, None, {}
    os.environ.update(env)
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with metered_http(book, case["id"], model) as first_request:
            asyncio.run(workflow(case, model, steps))
    except Exception as error:
        error_type = type(error).__name__
        error_class = W1.classify_error(error)
    finally:
        logging.disable(previous_logging)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    rejected = submission.with_suffix(".rejected.json")
    submission_valid, decision = False, None
    if submission.exists() and not rejected.exists():
        try:
            decision = W1.validate_submission(case["kind"], case["packet"], L.read(submission))
            submission_valid = True
        except (ValueError, TypeError, KeyError):
            error_type = "SubmissionRevalidationError"
    events = [W1.event_row(step) for step in steps]
    L.write_new(output / "events.json", events)
    tool_rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    trace_summary = classify_tool_trace(tool_rows)
    checks = trace_summary["prechecks"]
    errors = trace_summary["tool_errors"]
    rejected_validation = rejection_is_validation_error(case, rejected)
    argument_metrics = tool_argument_metrics(
        checks, errors, submission_valid, bool(tool_rows), rejected_validation)
    event_summary = W1.summarize_events(events)
    metered_complete = (first_request.get("request_count") == event_summary["llm_calls"]
                        and event_summary["llm_call_coverage"] == "complete")
    trace_complete = (checks["malformed"] == 0 and metered_complete
                      and argument_metrics["uncertain_tool_error_count"] == 0)
    transport_complete = first_request.get("transport_complete") is True
    final_valid = submission_valid and trace_complete and transport_complete
    if not final_valid and error_type is None:
        error_type = ("TraceIncomplete" if not trace_complete else
                      "TransportIncomplete" if not transport_complete else "SubmissionMissing")
        error_class = "invalid"
    result = {"id": case["id"], "model": model, "binding": case["binding"],
              "valid": final_valid, "submission_valid": submission_valid,
              "decision": decision, "error_type": error_type, "error_class": error_class,
              "first_request": first_request, "submission_rejected": rejected.exists(),
              "tool_error_count": sum(errors.values()), "tool_error_types": errors,
              "precheck_invalid_count": checks["contract_invalid"],
              "precheck_valid_count": checks["valid"],
              "precheck_limit_exhausted_count": checks["limit_exhausted"],
              "precheck_malformed_count": checks["malformed"],
              **argument_metrics,
              "trace_complete": trace_complete,
              "transport_complete": transport_complete,
              "metered_http_count": first_request.get("request_count"),
              "metered_complete": metered_complete,
              "events_summary": event_summary,
              "artifact_sha256": {p.name: W1.raw_sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    L.write_new(output / "result.json", result)
    return result


def review_case(case, model, output, book):
    """저장 결과를 산출물·원시 HTTP 원장으로 다시 만들고 양성 판정을 fail-closed로 대조한다."""
    output = Path(output)
    stored = L.read(output / "result.json")
    if stored.get("id") != case["id"] or stored.get("model") != model or stored.get("binding") != case["binding"]:
        raise ValueError("NAT 결과 입력 결합이 다르다")

    expected_artifacts = stored.get("artifact_sha256")
    actual_files = {p.name for p in output.iterdir() if p.is_file() and p.name != "result.json"}
    if not isinstance(expected_artifacts, dict) or set(expected_artifacts) != actual_files:
        raise ValueError("NAT 산출물 목록이 다르다")
    actual_artifacts = {name: W1.raw_sha(output / name) for name in sorted(actual_files)}
    if actual_artifacts != expected_artifacts:
        raise ValueError("NAT 산출물 SHA가 다르다")

    events = L.read(output / "events.json")
    event_summary = W1.summarize_events(events)
    rows = [json.loads(line) for line in (output / "tools.jsonl").read_text(encoding="utf-8").splitlines()
            if line]
    trace_summary = classify_tool_trace(rows)
    checks, errors = trace_summary["prechecks"], trace_summary["tool_errors"]
    trace_complete = checks["malformed"] == 0

    submission, rejected = output / "submission.json", output / "submission.rejected.json"
    if submission.exists() and rejected.exists():
        raise ValueError("NAT 제출과 거절 기록이 함께 있다")
    decision, submission_valid = None, False
    if submission.exists():
        try:
            decision = W1.validate_submission(case["kind"], case["packet"], L.read(submission))
            submission_valid = True
        except (ValueError, TypeError, KeyError):
            submission_valid = False
    rejected_validation = rejection_is_validation_error(case, rejected)
    argument_metrics = tool_argument_metrics(checks, errors, submission_valid, bool(rows), rejected_validation)
    trace_complete = trace_complete and argument_metrics["uncertain_tool_error_count"] == 0

    prefix = f"{case['id']}_{model.split('/')[-1]}_http"
    calls = [row for row in book.audit()
             if row["reservation"]["stage"] == "B3" and row["reservation"]["label"].startswith(prefix)]
    if not calls or [row["reservation"]["label"] for row in calls] != [
            f"{prefix}{index}" for index in range(1, len(calls) + 1)]:
        raise ValueError("NAT HTTP 원장 순서가 다르다")
    for row in calls:
        reservation, response = row["reservation"], row["response"]
        if reservation.get("model") != model or response is None:
            raise ValueError("NAT HTTP 예약 또는 응답이 불완전하다")
    raw_results = [row["response"]["result"] for row in calls]
    transport_states = [nat_transport_status(raw, model) for raw in raw_results]
    metered_complete = (len(calls) == event_summary["llm_calls"]
                        and event_summary["llm_call_coverage"] == "complete")
    if not metered_complete:
        trace_complete = False
    first_hash = calls[0]["reservation"].get("request_without_model_sha256")
    if not isinstance(first_hash, str) or len(first_hash) != 64:
        raise ValueError("NAT 첫 요청의 모델 제외 결합이 없다")
    first_request = stored.get("first_request")
    if not isinstance(first_request, dict) or first_request.get("request_without_model_sha256") != first_hash:
        raise ValueError("NAT 첫 요청 결합이 저장 결과와 다르다")
    transport_complete = all(state == "ok" for state in transport_states)
    if (first_request.get("request_count") != len(calls)
            or first_request.get("transport_complete") is not transport_complete):
        raise ValueError("NAT HTTP 계측 요약이 원장과 다르다")
    if not isinstance(first_request.get("tools_sha256"), str) or len(first_request["tools_sha256"]) != 64:
        raise ValueError("NAT 첫 요청 도구 결합이 없다")
    valid = submission_valid and trace_complete and transport_complete

    recomputed = {
        "valid": valid, "submission_valid": submission_valid, "decision": decision,
        "submission_rejected": rejected.exists(), "tool_error_count": sum(errors.values()),
        "tool_error_types": errors, "precheck_invalid_count": checks["contract_invalid"],
        "precheck_valid_count": checks["valid"],
        "precheck_limit_exhausted_count": checks["limit_exhausted"],
        "precheck_malformed_count": checks["malformed"],
        **argument_metrics,
        "trace_complete": trace_complete, "transport_complete": transport_complete,
        "metered_http_count": len(calls), "metered_complete": metered_complete,
        "events_summary": event_summary,
        "artifact_sha256": actual_artifacts,
    }
    for key, value in recomputed.items():
        if stored.get(key) != value:
            raise ValueError(f"NAT 저장 결과 재검증 불일치: {key}")
    if valid and (stored.get("error_type") is not None or stored.get("error_class") is not None):
        raise ValueError("유효 NAT 결과에 실행 오류가 기록됐다")
    if not valid and stored.get("error_type") is None:
        raise ValueError("불완전 NAT 결과에 오류 종류가 없다")
    return stored
