"""동일 수치/영상 짝의 네 선택 방식을 실제 검사 영수증으로 이어가는 개발 실행기."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]
import agentic_investigation_call_v2 as CALL
import agentic_investigation_contract_v2 as A
import local_same_play_pair_provider_v1 as P
import selection_comparison_controller_v1 as C

SCHEMA = "local_pair_four_arm_protocol_v1"
ARMS = ("registry_next", "fixed_eligible", "logs_only", "logs_plus_frames")
MODEL_ARMS = frozenset(ARMS[2:])
SOURCE_FILES = tuple(sorted({
    *P.SOURCE_FILES,
    "evals/local_pair_four_arm_v1.py", "tests/test_local_pair_four_arm_v1.py",
    "evals/local_pair_probe_host_v1.py", "tests/test_local_pair_probe_host_v1.py",
    "evals/local_pair_nim_transport_v1.py", "tests/test_local_pair_nim_transport_v1.py",
}))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(value: object) -> str:
    return hashlib.sha256(C.V.canonical(value)).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=A.V._unique_object)
    require(isinstance(value, dict), "JSON 객체가 필요하다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(C.V.canonical(value) + b"\n")


def verify_protocol(path: Path, anchor: str) -> dict:
    value = read(path)
    require(value.get("schema") == SCHEMA
            and value.get("scope") == "development_seen_sources"
            and value.get("arms") == list(ARMS)
            and value.get("automatic_retries") == 0
            and value.get("max_probes_per_arm") == 3
            and value.get("api_cap") == 12
            and value.get("model") == CALL.PARAMS["model"]
            and value.get("authority") == {"api": True, "gpu": True, "probe": True,
                                           "correction": False, "retraining": False},
            "등록된 네 방식 개발 비교 계약이 아니다")
    require(isinstance(anchor, str) and bool(re.fullmatch(r"[0-9a-f]{40}", anchor)), "실행 anchor")
    pins = value.get("sources_lf", {})
    require(set(pins) >= set(SOURCE_FILES), "네 방식 source closure 누락")
    for name, expected in pins.items():
        item = Path(name)
        require(not item.is_absolute() and ".." not in item.parts
                and item.parts[0] in {"evals", "bench", "tests", "src"}, "source 경로 범위")
        require(sha(ROOT / name, lf=True) == expected, "현재 source SHA 변경: " + name)
        blob = subprocess.run(["git", "show", f"{anchor}:{name}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        require(hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == expected,
                "커밋 source blob 변경: " + name)
    subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"],
                   cwd=ROOT, check=True, capture_output=True)
    relative = path.resolve().relative_to(ROOT).as_posix()
    blob = subprocess.run(["git", "show", f"{anchor}:{relative}"], cwd=ROOT,
                          capture_output=True, check=True).stdout
    require(hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == sha(path, lf=True),
            "protocol의 실행 commit blob 불일치")
    return value


def _bound_file(reference: dict) -> Path:
    require(isinstance(reference, dict) and set(reference) >= {"path", "sha256"}, "파일 참조")
    path = Path(reference["path"]).resolve(strict=True)
    require(sha(path) == reference["sha256"], "원문 SHA 변경: " + path.name)
    return path


def load_pair(result_path: Path) -> dict:
    """기존 준비 산출물의 권한 false는 유지하고 새 실행의 출처로만 읽는다."""
    result = read(result_path)
    require(result.get("schema") == P.PREPARED_SCHEMA
            and result.get("status") == "prepared_cpu_inputs_api_not_authorized"
            and all(result.get(key) is False for key in ("api_authorized", "gpu_authorized",
                    "probe_authorized", "correction_authorized", "heldout", "recovery", "video_benefit")),
            "입력 준비 결과의 기존 권한 경계")
    pair_path = _bound_file(result["pair"])
    pair = read(pair_path)
    require(pair["binding_sha256"] == result["pair_binding_sha256"], "pair binding")
    arms = {}
    for arm in MODEL_ARMS:
        provider = read(_bound_file(pair["arms"][arm]["provider_input"]))
        context = A.checked(provider["context"])
        outgoing = CALL._messages(context, provider["messages"])
        require(provider["params"] == CALL.params_for_messages(outgoing), "provider params")
        require(provider.get("api_authorized") is False and provider.get("approval_granted") is False,
                "입력 준비는 실행 승인이 아니다")
        arms[arm] = provider
    only, plus = arms["logs_only"], arms["logs_plus_frames"]
    require(P.PAIRED._context_common(only["context"]) == P.PAIRED._context_common(plus["context"]),
            "수치/영상 공통 context가 다르다")
    require(only["context"]["goal"] == plus["context"]["goal"] == P.PAIR_GOAL,
            "공통 frame-role goal")
    base = only["context"]["snapshot"]["inputs"]["packet"]
    require(len(base["facts"]) == 48
            and set(base["facts"]) <= C.V.BASE_FACT_IDS, "공통 training 48 facts")
    blocks = plus["messages"][1]["content"]
    require(isinstance(blocks, list) and sum(x.get("type") == "image_url" for x in blocks) == 6
            and isinstance(only["messages"][1]["content"], str), "이미지 0/6 입력")
    pending = read(_bound_file(result["pending"]))
    state = read(_bound_file(result["state"]))
    require(pending["source"]["training_facts"] == base["facts"], "training facts 원문 결합")
    require(state["initial_packet"]["input_bindings"] == base["input_bindings"], "backend source CP")
    return {"prepared": ref(result_path), "pair": ref(pair_path), "pair_binding": pair["binding_sha256"],
            "packet": deepcopy(base), "backend_packet": deepcopy(state["initial_packet"]),
            "providers": arms, "pending": pending}


def _context(source: dict, snapshot: dict, arm: str, remaining: int) -> tuple[dict, list]:
    provider_arm = "logs_plus_frames" if arm == "logs_plus_frames" else "logs_only"
    original = source["providers"][provider_arm]
    context = A.prepare(snapshot, goal=P.PAIR_GOAL,
                        extra_facts=original["context"]["extra_facts"], decisions_left=remaining)
    messages = A.messages(context)
    if arm == "logs_plus_frames":
        messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
                                   *deepcopy(original["messages"][1]["content"][1:])]
    return context, CALL._messages(context, messages)


def _request(source: dict, context: dict, messages: list, run_id: str, arm: str, step: int) -> dict:
    bindings = source["packet"]["input_bindings"]
    return {"schema": "local_pair_nim_request_v1", "request_id": f"{run_id}_{arm}_{step}",
            "run_id": run_id, "arm": arm, "step": step,
            "context_digest": context["context_digest"], "messages": messages,
            "params": CALL.params_for_messages(messages), "source_binding": {
                "pair_sha256": source["pair"]["sha256"],
                "case_checkpoint_sha256": bindings["case"]["checkpoint_sha256"],
                "reference_checkpoint_sha256": bindings["reference"]["checkpoint_sha256"]}}


def _verify_transport(response: dict, request: dict, verifier: Callable) -> dict:
    reference = response["call_ref"]
    call = Path(reference["path"]).resolve(strict=True)
    require(call.is_dir() and sha(call / "receipt.json") == reference["sha256"],
            "transport call 폴더와 receipt SHA 결합")
    checked = verifier(call)
    require(checked.get("request") == request and checked.get("live") is True,
            "원문 live transport와 model request 결합")
    require(all(checked.get(key) == response.get(key)
                for key in ("status", "content", "usage", "wall_s", "call_ref")),
            "transport 결과 바꿔치기")
    return reference


def run_arm(source: dict, *, arm: str, run_id: str, output: Path,
            protocol_path: Path, execution_anchor: str,
            complete: Callable | None, execute_probe: Callable, verify_probe: Callable,
            verify_call: Callable | None = None, allow_fixture: bool = False) -> dict:
    """승인·검사·원장 재생 뒤 다음 공개 상태를 만든다. 모델 무효는 HOLD다."""
    require(arm in ARMS and bool(re.fullmatch(r"[a-z0-9_]{1,60}", run_id)), "arm/run_id")
    if allow_fixture:
        protocol = read(protocol_path)
        require(protocol.get("fixture") is True, "fixture 계약이 필요하다")
    else:
        protocol = verify_protocol(protocol_path, execution_anchor)
        matches = [row for row in protocol.get("pairs", [])
                   if row.get("prepared_result_sha256") == source["prepared"]["sha256"]
                   and row.get("pair_sha256") == source["pair"]["sha256"]]
        require(len(matches) == 1, "사전등록 pair/source에 없다")
        require(callable(verify_probe) and (arm not in MODEL_ARMS or callable(verify_call)),
                "실제 probe/transport verifier가 필요하다")
    output = output.resolve()
    require(not output.exists(), "실행 출력은 새 경로여야 한다")
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "probes.ledger.jsonl"
    packet = deepcopy(source["packet"])
    packet["budget_left"] = protocol["max_probes_per_arm"]
    policy = arm if arm not in MODEL_ARMS else "model_adaptive"
    result: dict[str, Any] = {"schema": "local_pair_four_arm_result_v1", "arm": arm,
        "run_id": run_id, "status": "complete", "scope": "cpu_fixture" if allow_fixture else "actual_backend",
        "source": source["prepared"], "pair": source["pair"], "steps": [], "model_calls": 0,
        "hold_reason": None, "repair": False, "recovery": False, "heldout": False,
        "semantic_invalid": 0, "infra_unknown": 0, "provider_unknown": 0,
        "frame_fact_citations": 0}
    for step in range(1, protocol["max_probes_per_arm"] + 1):
        snapshot = C.V.build_snapshot(packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger)
        if snapshot["view"]["status"] != "open":
            break
        context, messages = _context(source, snapshot, arm, protocol["max_probes_per_arm"] - step + 1)
        decision_dir = output / f"step{step:02d}"
        write(decision_dir / "context.json", context)
        model_ref = None
        if arm in MODEL_ARMS:
            if complete is None:
                raise ValueError("모델 callback 부재")
            request = _request(source, context, messages, run_id, arm, step)
            write(decision_dir / "request.json", request)
            response = complete(deepcopy(request))
            result["model_calls"] += 1
            if not allow_fixture:
                if verify_call is None:
                    raise ValueError("transport verifier 부재")
                model_ref = _verify_transport(response, request, verify_call)
            if response.get("status") != "completed":
                field = "provider_unknown" if response.get("status") == "unknown" else "infra_unknown"
                result[field] += 1
                result["hold_reason"] = "model_provider_unknown" if field == "provider_unknown" else "model_transport_unknown"
                write(decision_dir / "model-result.json", response)
                break
            review = A.assess(response.get("content"), context)
            write(decision_dir / "model-review.json", {"transport": model_ref, "review": review})
            if review["status"] != "valid":
                result["semantic_invalid"] += 1
                result["hold_reason"] = "model_contract_invalid"
                break
            if review["decision"]["action"] != "run_probe":
                result["hold_reason"] = "model_hold"
                break
            choice = A.probe_choice(context, review)
            result["frame_fact_citations"] += sum(x.startswith("vision:frame:") for x in choice["fact_ids"])
        else:
            choice = C.choose(snapshot, arm)
            require(choice.get("probe") == C.V.rule_probe(context["model_view"], arm),
                    "동일 공개 자료에서 규칙 선택 불일치")
        if choice["action"] != "continue":
            result["hold_reason"] = choice["reason"]
            break
        spec = C.request_spec(snapshot, choice)
        write(decision_dir / "choice.json", choice)
        write(decision_dir / "request-spec.json", spec)
        dispatched = execute_probe(snapshot, choice, out=decision_dir / "probe")
        receipt_ref = dispatched["result_ref"]
        verified = verify_probe(_bound_file(receipt_ref), snapshot, choice)
        require(verified.get("probe") == spec["probe"] and verified.get("outcome") in
                {"normal", "abnormal", "unknown"}, "검증된 probe 결과가 없다")
        fresh = C.V.build_snapshot(packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger)
        before_rows, after_rows = snapshot["view"]["observations"], fresh["view"]["observations"]
        require(after_rows == before_rows + [{"probe": spec["probe"], "outcome": verified["outcome"]}],
                "검사 영수증을 원장에 반영하지 못했다")
        if not allow_fixture:
            require(verified.get("evidence_kind") == "actual_runtime_observation", "fixture 결과 승격 거부")
        result["steps"].append({"step": step, "selected_by": choice["selected_by"],
            "probe": spec["probe"], "fact_ids": choice["fact_ids"], "outcome": verified["outcome"],
            "probe_result": receipt_ref, "model_call": model_ref,
            "before_view_digest": snapshot["view_digest"], "after_view_digest": fresh["view_digest"],
            "remaining_before": snapshot["view"]["remaining_candidates"],
            "remaining_after": fresh["view"]["remaining_candidates"],
            "owned_child_wall_s": verified.get("owned_child_wall_s")})
        if verified["outcome"] == "unknown":
            result["hold_reason"] = "probe_outcome_unknown"
            break
    final = C.V.build_snapshot(packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger)
    result["remaining_candidates"] = final["view"]["remaining_candidates"]
    result["final_view_digest"] = final["view_digest"]
    result["confirmed_subtype"] = (result["remaining_candidates"][0]
                                   if final["view"]["status"] == "confirmed" else None)
    result["end_state"] = "confirmed" if result["confirmed_subtype"] is not None else "HOLD"
    if result["end_state"] == "HOLD" and result["hold_reason"] is None:
        result["hold_reason"] = "probe_budget_or_no_informative_probe"
    write(output / "result.json", result)
    return result
