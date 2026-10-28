"""등록된 개발 입력의 자료 조회와 실제 모델 제안을 재생 가능한 원장에 연결한다.

검사·교정은 실행하지 않는다. 다음 probe 요청 또는 HOLD에서 끝내며 실제 API는
Windows 원장 예약 뒤 WSL의 기존 provider 경계에서 한 번 호출한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import agentic_investigation_call_v1 as CALL
import agentic_investigation_evidence_v1 as B

E, A, J = B.E, B.A, CALL.J
SCHEMA = "agentic_development_plan_v1"


class ProviderPending(BaseException):
    """자식 응답이 불명확하면 예약을 보존하고 자동 재전송하지 않는다."""


def _new_or_same(path: Path, value: dict) -> None:
    if path.exists():
        if J._read(path) != value:
            raise ValueError("완료 기록이 현재 재생과 다르다")
    else:
        J._write_new(path, value)


def _lf_sha(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def verify_sources(protocol: dict, anchor: str) -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT,
                      capture_output=True, check=False).returncode:
        raise ValueError("등록 소스 커밋이 현재 HEAD의 조상이 아니다")
    for name, expected in protocol["sources_lf"].items():
        path = ROOT / name
        if path.resolve() != path.absolute() or ROOT not in path.resolve().parents:
            raise ValueError("등록 소스 경로가 저장소를 벗어났다")
        bound = subprocess.run(["git", "show", f"{anchor}:{name}"], cwd=ROOT,
                               capture_output=True, check=True).stdout
        if _lf_sha(bound) != expected or _lf_sha(path.read_bytes()) != expected:
            raise ValueError("등록 소스 blob 불일치: " + name)


def wsl_complete(messages: list, params: dict) -> dict:
    """호출 본문만 stdin으로 전달한다. 자격증명은 WSL 환경에만 남는다."""
    script = Path(__file__).resolve()
    if len(script.drive) != 2 or script.drive[1] != ":":
        raise ValueError("Windows 호스트에서만 WSL 전송을 시작한다")
    remote = "/mnt/" + script.drive[0].lower() + "/" + script.as_posix()[3:]
    command = ['wsl.exe', '-d', 'Ubuntu', '--exec', '/bin/bash', '-lc',
        'set -e; . "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" "$1" provider-once',
        'agentic-provider', remote]
    try:
        result = subprocess.run(command, input=json.dumps({"messages": messages, "params": params}),
            capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=240, check=False)
        if result.returncode != 0:
            raise ProviderPending("WSL provider가 결합된 응답 없이 종료했다")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise TypeError("provider 결과 형식")
        return CALL.O._provider_result(value)
    except (subprocess.TimeoutExpired, OSError, UnicodeError, ValueError, TypeError) as error:
        raise ProviderPending(type(error).__name__ + ": 완료 여부 미확정; 자동 재전송 금지") from None


def run_case(plan: dict, case: dict, complete) -> dict:
    """처음부터 완료 결정을 재생하고 미완료 API만 기존 journal로 처리한다."""
    roots = {key: Path(value) for key, value in plan["roots"].items()}
    index = J._read(Path(plan["index"]["path"]))
    if J.file_sha(Path(plan["index"]["path"])) != plan["index"]["sha256"]:
        raise ValueError("등록 색인 원문 SHA 변경")
    entry = next(row for row in index["spec"]["entries"] if row["evidence_id"] == case["evidence_id"])
    original = J._read(E._ref(entry["snapshot"], roots))
    output = Path(plan["output"]) / case["id"]
    output.mkdir(parents=True, exist_ok=True)
    old = original["inputs"]["packet"]
    facts = old["facts"] if case["initial_facts"] == "all_training" else {
        "config:action_scale": old["facts"]["config:action_scale"]}
    packet = A.C.S.build_packet(epoch=old["epoch"], applied_corrections=old["applied_corrections"],
        prior_receipt=old["prior_receipt"], facts=facts, input_bindings=old["input_bindings"],
        budget_left=old["budget_left"], execution_budget_s=old["execution_budget_s"])
    snapshot = A.C.V.build_snapshot(packet, case["id"], "model_adaptive", original["inputs"]["protocol_sha256"],
                                    output / "unexecuted-probes.jsonl")
    aliases = {"evidence:training-01": {"kind": "training", "evidence_id": case["evidence_id"], "frame_ids": []},
               "evidence:clip-01": {"kind": "frames", "evidence_id": case["evidence_id"],
                                    "frame_ids": case["frame_ids"]}}
    registration = B.register(index, roots, snapshot, expected_index_sha256=plan["index_canonical_sha256"],
        source_lineage_id=entry["binding"]["source_lineage_id"], aliases=aliases)
    _new_or_same(output / "registration.json", registration)
    bridge = B.EvidenceAdapter(index, roots, registration, expected_registration_sha256=E.index_sha256(registration))
    loop = bridge.start(snapshot, goal=plan["goal"], max_decisions=plan["max_decisions"])
    visual, decisions = None, []
    journal = {}
    for number in range(1, plan["max_decisions"] + 1):
        request = bridge.messages(loop, visual_context=visual)
        folder = output / f"decision-{number:02d}"
        reference_path = output / f"decision-{number:02d}.reference.json"
        if reference_path.exists():
            reference = J._read(reference_path)
            _, capsule = CALL.prepared(reference)
            if capsule["messages"] != request["messages"]:
                raise ValueError("재개된 판단 입력이 이전 원문과 다르다")
        else:
            reference = CALL.freeze(request["context"], folder, source_anchor=plan["source_anchor"],
                protocol_sha256=plan["protocol"]["sha256"], messages=request["messages"])
            J._write_new(reference_path, reference)
        answer = CALL.advance(reference, Path(plan["output"]) / "api", complete, api_cap=plan["api_cap"])
        journal = answer["journal"]
        if journal["status"] not in {"choice_ready", "terminal_hold"}:
            return {"case": case["id"], "status": journal["status"], "completed_decisions": decisions,
                    "api": journal["api"], "approval_granted": False}
        engine = journal["engine_result"]
        content = engine.get("final_content")
        review: dict = loop.propose(content) if isinstance(content, str) else {"status": "unavailable"}
        decision = {"number": number, "model_status": engine.get("model_status", "unavailable"), "review": review,
                    "image_count": request["image_count"], "frame_refs": request["frame_refs"],
                    "messages_sha256": request["messages_sha256"], "capsule": reference}
        decisions.append(decision)
        _new_or_same(output / f"decision-{number:02d}.result.json", decision)
        if review["status"] != "valid" or loop.status == "hold":
            status = "hold"
            break
        choice = review["decision"]
        if choice["action"] == "read_evidence":
            receipt = bridge.read_evidence(loop, choice["target"], visual_context=visual)
        elif choice["action"] == "inspect_frames":
            receipt = bridge.inspect_frames(loop, choice["target"], visual_context=visual)
        elif choice["action"] == "run_probe":
            _new_or_same(output / "probe-request.json", A.C.request_spec(loop.snapshot, loop.probe_choice()))
            status = "probe_proposed_unexecuted"
            break
        else:
            raise ValueError("검사 receipt 전 교정은 개발 경로에 없다")
        _new_or_same(output / f"decision-{number:02d}.evidence.json", receipt)
        visual = receipt["visual_context"]
    else:
        status = "decision_budget_exhausted"
    summary = {"schema": "agentic_development_case_result_v1", "case": case["id"], "status": status,
        "decisions": decisions, "api": journal["api"], "loop": loop.audit(), "split": "development",
        "already_seen": True, "actual_probe_executed": False, "approval_granted": False,
        "visual_effect_measured": False}
    stored = output / "result.json"
    if stored.exists():
        previous = J._read(stored)
        if {k: v for k, v in previous.items() if k != "api"} != {k: v for k, v in summary.items() if k != "api"}:
            raise ValueError("개발 결과 재생이 이전 판단·행동과 다르다")
        return previous  # 전역 API 소비가 뒤에서 늘어도 원래 완료 시점 기록은 보존한다.
    J._write_new(stored, summary)
    return summary


def execute(plan_path: Path, complete=wsl_complete) -> dict:
    plan = J._read(plan_path)
    if plan["schema"] != SCHEMA or plan["api_cap"] != 12 or plan["max_decisions"] != 4:
        raise ValueError("등록된 개발 호출 상한과 다르다")
    protocol_path = Path(plan["protocol"]["path"])
    if J.file_sha(protocol_path) != plan["protocol"]["sha256"]:
        raise ValueError("사전등록 SHA 불일치")
    protocol = J._read(protocol_path)
    if (plan["cases"] != protocol["cases"] or plan["goal"] != protocol["goal"]
            or plan["index"]["sha256"] != protocol["input_index_raw_sha256"]
            or plan["index_canonical_sha256"] != protocol["input_index_canonical_sha256"]):
        raise ValueError("등록 개발 사례·목표 변경")
    budget = plan["budget"]
    audit = Path(budget["prior_audit"]["path"])
    if (J.file_sha(audit) != budget["prior_audit"]["sha256"]
            or budget["original_core_allocation_cap"] != 120 or budget["new_phase_cap"] != 12
            or budget["combined_allocation_cap"] != 132 or J._read(audit)["known_pending"] != 0
            or protocol["budget"]["combined_allocation_cap"] != 132):
        raise ValueError("기존 감사·새 단계 배정 결합 불일치")
    verify_sources(protocol, plan["source_anchor"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with J._lease(output / ".driver-active", {"scope": "agentic-development-driver",
                                            "manifest_sha256": J.file_sha(plan_path)}):
        results = []
        for case in plan["cases"]:
            value = run_case(plan, case, complete)
            results.append(value)
            if value["status"] in {"paused_infra", "blocked_pending"}:
                break
        return {"schema": "agentic_development_batch_result_v1", "results": results,
                "new_gpu_jobs": 0, "used_for_recovery_decision": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    sub.add_parser("provider-once")
    args = parser.parse_args()
    if args.command == "provider-once":
        payload = json.load(sys.stdin)
        if set(payload) != {"messages", "params"} or payload["params"] != CALL.PARAMS:
            raise ValueError("provider-once 요청 형식")
        result = CALL.O._safe_complete(CALL.O.default_complete, payload["messages"])
    else:
        result = execute(args.plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
