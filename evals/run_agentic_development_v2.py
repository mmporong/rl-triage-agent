"""기존 개발 API12 원장을 이어 쓰는 v2 runner. Provider/실제 등록은 메인이 소유한다."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path, PurePosixPath

import agentic_investigation_call_v2 as CALL
import run_agentic_development_v1 as V1

B, E, A, J = V1.B, V1.E, CALL.A, CALL.J
ROOT = V1.ROOT
SCHEMA = "agentic_development_plan_v2"
PROTOCOL_SCHEMA = "agentic_investigation_development_protocol_v2"
ProviderPending = V1.ProviderPending
_new_or_same = V1._new_or_same
NEW_SOURCES = ("evals/agentic_investigation_contract_v2.py", "evals/agentic_investigation_call_v2.py",
               "evals/run_agentic_development_v2.py", "evals/vision_transferred_e2e_v2.py")
NEW_TESTS = ("tests/test_agentic_investigation_contract_v2.py", "tests/test_agentic_investigation_call_v2.py",
             "tests/test_agentic_development_v2.py")


def provider_once(payload: dict, complete=CALL.O.default_complete) -> dict:
    """WSL의 단일 provider 경계. 판단별 params를 보존하며 고정 O._safe_complete는 쓰지 않는다."""
    _require(set(payload) == {"messages", "params", "messages_sha256", "params_sha256"}, "provider payload 필드")
    messages, params = payload["messages"], payload["params"]
    _require(J.digest(messages) == payload["messages_sha256"]
             and J.digest(params) == payload["params_sha256"]
             and params == CALL.params_for_messages(messages), "provider messages/params hash/schema 불일치")
    started = time.perf_counter()
    try:
        raw = complete(deepcopy(messages), deepcopy(params))
    except J.JournalError:
        raise
    except Exception as error:  # noqa: BLE001 - 자격증명·예외 원문은 반환하지 않는다.
        raw = {"error_type": type(error).__name__, "wall_s": time.perf_counter() - started}
    return {"schema": "agentic_provider_once_v2", "messages_sha256": J.digest(messages),
            "params_sha256": J.digest(params), "result": CALL.O._provider_result(raw)}


def wsl_complete(messages: list, params: dict, *, run_process=subprocess.run) -> dict:
    """예약된 v2 body와 hash만 전달한다. 키는 WSL 환경에서 기존 SDK 경계가 읽는다."""
    payload = {"messages": messages, "params": params, "messages_sha256": J.digest(messages), "params_sha256": J.digest(params)}
    _require(params == CALL.params_for_messages(messages), "전송 전 params/schema 불일치")
    script = Path(__file__).resolve()
    _require(len(script.drive) == 2 and script.drive[1] == ":", "WSL 전송은 Windows 호스트 전용")
    remote = "/mnt/" + script.drive[0].lower() + "/" + script.as_posix()[3:]
    command = ["wsl.exe", "-d", "Ubuntu", "--exec", "/bin/bash", "-lc",
        'set -e; . "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" "$1" provider-once',
        "agentic-v2-provider", remote]
    try:
        result = run_process(command, input=json.dumps(payload), capture_output=True, text=True,
                             encoding="utf-8", errors="strict", timeout=240, check=False)
        if result.returncode != 0:
            raise ProviderPending("WSL provider가 결합된 응답 없이 종료했다")
        value = json.loads(result.stdout)
        if (value.get("schema") != "agentic_provider_once_v2"
                or value.get("messages_sha256") != payload["messages_sha256"]
                or value.get("params_sha256") != payload["params_sha256"]):
            raise ProviderPending("WSL provider request hash 불일치; 자동 재전송 금지")
        return CALL.O._provider_result(value["result"])
    except (subprocess.TimeoutExpired, OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise ProviderPending(type(error).__name__ + ": 완료 여부 미확정; 자동 재전송 금지") from None


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    J._read(path)
    return {"path": str(path), "sha256": J.file_sha(path)}


def _read_ref(value: dict) -> dict:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "원문 ref 형식")
    path = Path(value["path"])
    _require(path == path.resolve() and J.file_sha(path) == value["sha256"], "원문 ref SHA/경로 변경")
    return J._read(path)


def capture_legacy(legacy_plan_path: Path) -> dict:
    """v1 원문은 읽기만 하며 처음 세 예약·응답·receipt를 불변 prefix로 고정한다."""
    legacy = J._read(legacy_plan_path)
    _require(legacy["schema"] == V1.SCHEMA and legacy["api_cap"] == 12 and legacy["max_decisions"] == 4,
             "v1 개발 범위 불일치")
    protocol = _read_ref(legacy["protocol"])
    _require(len(legacy["cases"]) == 3 and protocol["cases"] == legacy["cases"]
             and protocol["goal"] == legacy["goal"] and protocol["model"] == CALL.PARAMS
             and protocol["input_index_raw_sha256"] == legacy["index"]["sha256"]
             and protocol["input_index_canonical_sha256"] == legacy["index_canonical_sha256"],
             "v1 사전등록/3입력 결합 불일치")
    api = (Path(legacy["output"]) / "api").resolve()
    _require(J._read(api / "manifest.json") == {"schema": J.API_SCHEMA, "api_cap": 12}, "공유 API12 manifest 불일치")
    names = sorted(p.name for p in (api / "reservations").glob("*.json"))
    _require(names == ["0001.json", "0002.json", "0003.json"], "초기 v1 소비는 정확히3이어야 한다")
    prefix = {"manifest.json": _ref(api / "manifest.json")}
    previous = None
    for index, name in enumerate(names, 1):
        reservation, response, receipt = (J._read(api / kind / name) for kind in ("reservations", "responses", "receipts"))
        _require(reservation["index"] == response["index"] == receipt["index"] == index
                 and response["status"] == "completed"
                 and response["reservation_sha256"] == receipt["reservation_sha256"] == J.file_sha(api / "reservations" / name)
                 and receipt["response_sha256"] == J.file_sha(api / "responses" / name)
                 and receipt["previous_receipt_sha256"] == previous, "v1 API prefix 체인 불일치")
        previous = J.file_sha(api / "receipts" / name)
        for kind in ("reservations", "responses", "receipts"):
            prefix[f"{kind}/{name}"] = _ref(api / kind / name)
    for case in legacy["cases"]:
        result = J._read(Path(legacy["output"]) / case["id"] / "decision-01.result.json")
        _require(result["model_status"] == "invalid" and result["review"]["status"] == "invalid", "v1 invalid/HOLD 기록 불일치")
    return {"plan": _ref(legacy_plan_path), "protocol": deepcopy(legacy["protocol"]),
            "api_root": str(api), "baseline_reserved": 3, "prefix": prefix}


def verify_legacy(seal: dict) -> dict:
    legacy = _read_ref(seal["plan"])
    _require(seal["baseline_reserved"] == 3 and legacy["api_cap"] == 12
             and Path(seal["api_root"]) == (Path(legacy["output"]) / "api").resolve()
             and seal["protocol"] == legacy["protocol"], "v1 공유 API 결합 변경")
    _read_ref(seal["protocol"])
    expected = {"manifest.json"} | {f"{kind}/{index:04d}.json" for kind in ("reservations", "responses", "receipts") for index in range(1, 4)}
    _require(set(seal["prefix"]) == expected, "v1 prefix 목록 변경")
    for name, value in seal["prefix"].items():
        _require(Path(value["path"]) == Path(seal["api_root"]) / name, "v1 prefix 경로 변경")
        _read_ref(value)
    _require(J._read(Path(seal["api_root"]) / "manifest.json") == {"schema": J.API_SCHEMA, "api_cap": 12}, "공유 API12 상한 변경")
    return legacy


def build_protocol(legacy_plan_path: Path, *, sources_lf: dict) -> dict:
    """사전등록 후보만 반환한다. private 절대경로는 protocol에 넣지 않는다."""
    seal = capture_legacy(legacy_plan_path)
    legacy = _read_ref(seal["plan"])
    protocol = deepcopy(_read_ref(legacy["protocol"]))
    _require(set(NEW_SOURCES) <= set(sources_lf) and set(protocol["sources_lf"]) <= set(sources_lf), "v1/new source closure 부족")
    protocol.update(schema=PROTOCOL_SCHEMA, sources_lf=deepcopy(sources_lf),
        v2_system_sha256=J.digest(A.SYSTEM), legacy_plan_sha256=seal["plan"]["sha256"],
        legacy_api_prefix_sha256=J.digest(seal["prefix"]), baseline_reserved=3,
        remaining_additional_cap=9, structured_output_policy=deepcopy(A.STRUCTURED_OUTPUT_POLICY),
        protocol_change="prompt_and_registered_per_decision_json_schema_package")
    return protocol


def source_hashes(legacy_plan_path: Path) -> dict:
    """v1 closure·새 구현/회귀·기존 object reader의 현재 LF SHA를 생성한다."""
    legacy = J._read(legacy_plan_path)
    original = _read_ref(legacy["protocol"])
    names = set(original["sources_lf"]) | set(NEW_SOURCES) | set(NEW_TESTS)
    return {name: V1._lf_sha(E._inside(ROOT, name).read_bytes()) for name in sorted(names)}


def prepare_plan(legacy_plan_path: Path, protocol_path: Path, *, source_anchor: str, output: Path) -> dict:
    seal = capture_legacy(legacy_plan_path)
    legacy = _read_ref(seal["plan"])
    plan = deepcopy(legacy)
    plan.update(schema=SCHEMA, source_anchor=source_anchor, protocol=_ref(protocol_path),
                output=str(output.resolve()), api_root=seal["api_root"], legacy=seal)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> dict:
    _require(plan["schema"] == SCHEMA and plan["api_cap"] == 12 and plan["max_decisions"] == 4,
             "v2 API12/판단4 상한 불일치")
    _require(re.fullmatch(r"[0-9a-f]{40}", plan["source_anchor"]) is not None, "source anchor 형식")
    legacy = verify_legacy(plan["legacy"])
    old_protocol = _read_ref(legacy["protocol"])
    protocol = _read_ref(plan["protocol"])
    for field in ("cases", "goal", "index", "index_canonical_sha256", "roots", "budget", "max_decisions", "api_cap"):
        _require(plan[field] == legacy[field], "v1 고정 입력/목표/상한/budget 변경: " + field)
    _require(plan["api_root"] == plan["legacy"]["api_root"], "새 API 원장으로 상한 리셋 금지")
    output, old_output = Path(plan["output"]).resolve(), Path(legacy["output"]).resolve()
    _require(output != old_output and old_output not in output.parents and output not in old_output.parents,
             "v2 결과는 v1과 분리된 새 폴더여야 한다")
    _require(protocol["schema"] == PROTOCOL_SCHEMA and protocol["cases"] == plan["cases"]
             and protocol["goal"] == plan["goal"] and protocol["model"] == CALL.PARAMS
             and protocol["api_cap"] == 12 and protocol["max_decisions_per_case"] == 4
             and protocol["input_index_raw_sha256"] == plan["index"]["sha256"]
             and protocol["input_index_canonical_sha256"] == plan["index_canonical_sha256"]
             and protocol["v2_system_sha256"] == J.digest(A.SYSTEM)
             and protocol["structured_output_policy"] == A.STRUCTURED_OUTPUT_POLICY
             and protocol["legacy_plan_sha256"] == plan["legacy"]["plan"]["sha256"]
             and protocol["legacy_api_prefix_sha256"] == J.digest(plan["legacy"]["prefix"])
             and protocol["baseline_reserved"] == 3 and protocol["remaining_additional_cap"] == 9,
             "v2 등록 입력/system/공유 API prefix 변경")
    audit = _read_ref(plan["budget"]["prior_audit"])
    _require(audit["known_pending"] == 0 and plan["budget"]["original_core_allocation_cap"] == 120
             and plan["budget"]["new_phase_cap"] == 12 and plan["budget"]["combined_allocation_cap"] == 132
             and protocol["budget"] == old_protocol["budget"], "기존 budget 감사·배정 변경")
    _require(set(NEW_SOURCES) <= set(protocol["sources_lf"])
             and set(old_protocol["sources_lf"]) <= set(protocol["sources_lf"]), "등록 source closure 부족")
    return protocol


def verify_sources(protocol: dict, anchor: str, *, repo: Path = ROOT, protocol_path: Path | None = None) -> None:
    """기존 readonly object reader로 조상·blob을 검증한다. Git 프로세스는 실행하지 않는다."""
    from vision_transferred_e2e_v2 import ObjectEvidence

    repo = repo.resolve(strict=True)
    metadata = repo / ".git"
    if metadata.is_file():
        metadata = (repo / metadata.read_text().strip().removeprefix("gitdir: ")).resolve()
    common = metadata / "commondir"
    common_dir = (metadata / common.read_text().strip()).resolve() if common.exists() else metadata
    head = (metadata / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        name = head[5:]
        path = common_dir / name
        head = path.read_text().strip() if path.exists() else next(
            row.split(" ", 1)[0] for row in (common_dir / "packed-refs").read_text().splitlines()
            if row.endswith(" " + name))
    store = ObjectEvidence(repo)
    todo, seen = [head], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == anchor:
            store.commit(current)
            break
        todo.extend(store.commit(current)["parents"])
    _require(anchor in seen, "등록 source anchor가 HEAD의 조상이 아니다")
    rows = dict(protocol["sources_lf"])
    if protocol_path is not None:
        relative = protocol_path.resolve(strict=True).relative_to(repo).as_posix()
        rows[relative] = V1._lf_sha(protocol_path.read_bytes())
    for relative, expected in rows.items():
        _require(isinstance(relative, str) and "\\" not in relative and ":" not in relative
                 and not PurePosixPath(relative).is_absolute()
                 and all(part not in {"", ".", ".."} for part in relative.split("/")), "등록 source 경로 탈출")
        path = E._inside(repo, relative)
        _, blob = store.file(anchor, relative)
        _require(V1._lf_sha(blob) == expected == V1._lf_sha(path.read_bytes()), "등록 source/blob 불일치: " + relative)


def run_case(plan: dict, case: dict, complete) -> dict:
    validate_plan(plan)
    _require(case in plan["cases"], "미등록 개발 사례")
    roots = {key: Path(value) for key, value in plan["roots"].items()}
    index = _read_ref(plan["index"])
    entry = next(row for row in index["spec"]["entries"] if row["evidence_id"] == case["evidence_id"])
    original = J._read(E._ref(entry["snapshot"], roots))
    output = Path(plan["output"]) / case["id"]
    output.mkdir(parents=True, exist_ok=True)
    old = original["inputs"]["packet"]
    facts = old["facts"] if case["initial_facts"] == "all_training" else {"config:action_scale": old["facts"]["config:action_scale"]}
    packet = A.C.S.build_packet(epoch=old["epoch"], applied_corrections=old["applied_corrections"],
        prior_receipt=old["prior_receipt"], facts=facts, input_bindings=old["input_bindings"],
        budget_left=old["budget_left"], execution_budget_s=old["execution_budget_s"])
    snapshot = A.C.V.build_snapshot(packet, case["id"], "model_adaptive", original["inputs"]["protocol_sha256"],
                                    output / "unexecuted-probes.jsonl")
    aliases = {"evidence:training-01": {"kind": "training", "evidence_id": case["evidence_id"], "frame_ids": []},
               "evidence:clip-01": {"kind": "frames", "evidence_id": case["evidence_id"], "frame_ids": case["frame_ids"]}}
    registration = B.register(index, roots, snapshot, expected_index_sha256=plan["index_canonical_sha256"],
        source_lineage_id=entry["binding"]["source_lineage_id"], aliases=aliases)
    _new_or_same(output / "registration.json", registration)
    bridge = B.EvidenceAdapter(index, roots, registration, expected_registration_sha256=E.index_sha256(registration))
    loop = bridge.start(snapshot, goal=plan["goal"], max_decisions=4)
    visual, decisions, journal = None, [], {}
    for number in range(1, 5):
        request = bridge.messages(loop, visual_context=visual)
        messages = CALL.from_v1(request["context"], request["messages"])
        ref_path = output / f"decision-{number:02d}.reference.json"
        if ref_path.exists():
            reference = J._read(ref_path)
            _, capsule = CALL.prepared(reference)
            _require(capsule["messages"] == messages
                     and capsule["source_anchor"] == plan["source_anchor"]
                     and capsule["protocol_sha256"] == plan["protocol"]["sha256"]
                     and capsule["context"]["snapshot"]["binding"] == request["context"]["snapshot"]["binding"],
                     "재개한 v2 source/checkpoint/판단 입력 변경")
        else:
            reference = CALL.freeze(request["context"], output / f"decision-{number:02d}",
                source_anchor=plan["source_anchor"], protocol_sha256=plan["protocol"]["sha256"], messages=messages)
            J._write_new(ref_path, reference)
        answer = CALL.advance(reference, Path(plan["api_root"]), complete, api_cap=12)
        journal = answer["journal"]
        if journal["status"] not in {"choice_ready", "terminal_hold"}:
            return {"case": case["id"], "status": journal["status"], "completed_decisions": decisions,
                    "api": journal["api"], "approval_granted": False}
        engine = journal["engine_result"]
        content = engine.get("final_content")
        review: dict = loop.propose(content) if isinstance(content, str) else {"status": "unavailable"}
        decisions.append({"number": number, "model_status": engine.get("model_status", "unavailable"), "review": review,
                          "image_count": request["image_count"], "frame_refs": request["frame_refs"],
                          "messages_sha256": J.digest(messages), "params_sha256": answer["params_sha256"], "capsule": reference})
        _new_or_same(output / f"decision-{number:02d}.result.json", decisions[-1])
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
            raise ValueError("개발 범위는 검사 제안 또는 HOLD까지다")
        _new_or_same(output / f"decision-{number:02d}.evidence.json", receipt)
        visual = receipt["visual_context"]
    else:
        status = "decision_budget_exhausted"
    summary = {"schema": "agentic_development_case_result_v2", "case": case["id"], "status": status,
        "decisions": decisions, "api": journal["api"], "loop": loop.audit(), "split": "development",
        "already_seen": True, "actual_probe_executed": False, "approval_granted": False, "visual_effect_measured": False}
    stored = output / "result.json"
    if stored.exists():
        previous = J._read(stored)
        _require({k: v for k, v in previous.items() if k != "api"} == {k: v for k, v in summary.items() if k != "api"}, "v2 완료 재생 변경")
        return previous
    J._write_new(stored, summary)
    return summary


def execute(plan_path: Path, complete=wsl_complete) -> dict:
    plan = J._read(plan_path)
    protocol = validate_plan(plan)
    verify_sources(protocol, plan["source_anchor"], protocol_path=Path(plan["protocol"]["path"]))
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with J._lease(output / ".driver-active", {"scope": "agentic-development-v2", "manifest_sha256": J.file_sha(plan_path)}):
        results = []
        for case in plan["cases"]:
            value = run_case(plan, case, complete)
            results.append(value)
            if value["status"] in {"paused_infra", "blocked_pending"}:
                break
        return {"schema": "agentic_development_batch_result_v2", "results": results,
                "baseline_requests": 3, "shared_api_cap": 12, "max_additional_requests": 9,
                "new_gpu_jobs": 0, "used_for_recovery_decision": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources")
    sources.add_argument("--legacy-plan", type=Path, required=True)
    sources.add_argument("--output", type=Path, required=True)
    protocol = sub.add_parser("protocol-candidate")
    protocol.add_argument("--legacy-plan", type=Path, required=True)
    protocol.add_argument("--sources", type=Path, required=True)
    protocol.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--legacy-plan", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--anchor", required=True)
    prepare.add_argument("--run-output", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    sub.add_parser("provider-once")
    args = parser.parse_args()
    if args.command == "sources":
        result = source_hashes(args.legacy_plan)
        J._write_new(args.output, result)
    elif args.command == "protocol-candidate":
        result = build_protocol(args.legacy_plan, sources_lf=J._read(args.sources))
        J._write_new(args.output, result)
    elif args.command == "prepare":
        result = prepare_plan(args.legacy_plan, args.protocol, source_anchor=args.anchor, output=args.run_output)
        J._write_new(args.output, result)
    elif args.command == "provider-once":
        result = provider_once(json.load(sys.stdin))
    else:
        result = execute(args.plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
