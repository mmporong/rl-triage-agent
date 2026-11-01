"""완료된 local pair 네 방식 개발 비교를 경로 없는 공개 결과로 투영한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import local_pair_four_arm_v1 as M
import local_pair_nim_transport_v1 as NIM
import local_pair_probe_host_v1 as HOST

SCHEMA = "local_pair_four_arm_public_v1"
SOURCE_FILES = ("evals/export_local_pair_four_arm_v1.py",
                "tests/test_export_local_pair_four_arm_v1.py")
PRIVATE_PATTERN = re.compile(r"(?i)(bench/private|[A-Z]:[/\\]|/home/|/mnt/[a-z]/|Users[/\\])")
_PREFIX_LOCK = threading.RLock()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _bound(reference: Any) -> Path:
    if not isinstance(reference, dict) or set(reference) < {"path", "sha256"}:
        raise ValueError("원문 참조가 불완전하다")
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_file() or _sha(path) != reference["sha256"]:
        raise ValueError("원문 참조 SHA가 다르다")
    return path


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson 분율 입력이 잘못됐다")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total
                         + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "total": total, "fraction": proportion,
            "wilson95_low": max(0.0, center - half), "wilson95_high": min(1.0, center + half)}


def _exact_p(discordant_logs: int, discordant_frames: int) -> dict[str, int | float | None]:
    total = discordant_logs + discordant_frames
    if total == 0:
        return {"logs_only_better": 0, "logs_plus_frames_better": 0,
                "discordant": 0, "exact_two_sided_p": None}
    tail = sum(math.comb(total, k) for k in range(min(discordant_logs, discordant_frames) + 1))
    return {"logs_only_better": discordant_logs,
            "logs_plus_frames_better": discordant_frames, "discordant": total,
            "exact_two_sided_p": min(1.0, 2 * tail / (2 ** total))}


def _callbacks(allow_fixture: bool, verifiers: dict[str, Any] | None) -> dict[str, Any]:
    if not allow_fixture:
        if verifiers is not None:
            raise ValueError("actual export는 verifier 교체를 허용하지 않는다")
        return {"protocol": M.verify_protocol, "context": M.A.checked,
                "assess": M.A.assess, "choice": M.A.probe_choice,
                "pair": HOST.verify_pair, "call": NIM.verify_call, "fixture": False}
    if not isinstance(verifiers, dict):
        raise TypeError("cpu_fixture verifier가 필요하다")
    required = {"protocol", "context", "assess", "choice", "pair", "call"}
    if set(verifiers) != required or not all(callable(verifiers[key]) for key in required):
        raise ValueError("cpu_fixture verifier 구성이 다르다")
    return {**verifiers, "fixture": True}


def _prefix_bytes(snapshot: dict) -> bytes:
    inputs = snapshot.get("inputs")
    view = snapshot.get("view")
    if not isinstance(inputs, dict) or not isinstance(view, dict):
        raise TypeError("snapshot ledger 입력이 없다")
    ledger_value = inputs.get("ledger_path")
    if not isinstance(ledger_value, str):
        raise TypeError("snapshot ledger 경로가 없다")
    path = Path(ledger_value).resolve(strict=True)
    observations = view.get("observations")
    if not isinstance(observations, list):
        raise TypeError("snapshot 관측 행이 없다")
    count = len(observations)
    raw = path.read_bytes()
    if count == 0:
        return b""
    seen = 0
    offset = 0
    for line in raw.splitlines(keepends=True):
        offset += len(line)
        event = json.loads(line)
        if not isinstance(event, dict):
            raise TypeError("ledger event는 object여야 한다")
        if event.get("event") == "receipt":
            seen += 1
            if seen == count:
                return raw[:offset]
    raise ValueError("snapshot 관측 수에 맞는 ledger prefix가 없다")


@contextmanager
def _history_prefix_scope(snapshot: dict):
    """저장 snapshot 검증 동안 해당 ledger read만 과거 prefix로 바꾼다."""
    original_path = Path(snapshot["inputs"]["ledger_path"]).resolve(strict=True)
    prefix = _prefix_bytes(snapshot)
    original = M.C.V.read_history
    with _PREFIX_LOCK, tempfile.TemporaryDirectory(prefix="rl-triage-prefix-") as temporary:
        replay = Path(temporary) / "probes.ledger.jsonl"
        replay.write_bytes(prefix)

        def scoped(path: Path, binding: dict, maximum: float):
            candidate = Path(path).resolve()
            return original(replay if candidate == original_path else path, binding, maximum)

        M.C.V.read_history = scoped  # type: ignore[attr-defined]
        try:
            yield prefix
        finally:
            M.C.V.read_history = original  # type: ignore[attr-defined]


def _model_evidence(step_dir: Path, context: dict, raw: dict, callbacks: dict[str, Any],
                    registered: dict, step_number: int) -> tuple[dict, dict | None]:
    request_path = step_dir / "request.json"
    if not request_path.is_file():
        raise ValueError("모델 request 원문이 없다")
    request = _read(request_path)
    expected_binding = {"pair_sha256": registered["pair_sha256"],
                        "case_checkpoint_sha256": registered["case_checkpoint_sha256"],
                        "reference_checkpoint_sha256": registered["reference_checkpoint_sha256"]}
    if (request.get("run_id") != raw.get("run_id") or request.get("arm") != raw.get("arm")
            or request.get("step") != step_number
            or request.get("request_id") != f"{raw.get('run_id')}_{raw.get('arm')}_{step_number}"
            or request.get("context_digest") != context.get("context_digest")
            or request.get("source_binding") != expected_binding):
        raise ValueError("모델 request의 source/context/step 결합이 다르다")
    review_path, result_path = step_dir / "model-review.json", step_dir / "model-result.json"
    if review_path.is_file() == result_path.is_file():
        raise ValueError("모델 review/result 분기가 불완전하다")
    if review_path.is_file():
        wrapper = _read(review_path)
        reference = wrapper.get("transport")
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise ValueError("모델 transport ref가 불완전하다")
        call = callbacks["call"](Path(reference["path"]))
        if (call.get("request") != request or call.get("call_ref") != reference
                or call.get("status") != "completed" or call.get("live") is not True):
            raise ValueError("모델 request/transport 결합이 다르다")
        review = callbacks["assess"](call.get("content"), context)
        if wrapper.get("review") != review:
            raise ValueError("모델 review 재계산이 다르다")
        choice = callbacks["choice"](context, review) if review.get("status") == "valid" \
            and review.get("decision", {}).get("action") == "run_probe" else None
        return call, choice
    response = _read(result_path)
    reference = response.get("call_ref")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise TypeError("모델 미완료 call ref가 불완전하다")
    call = callbacks["call"](Path(reference["path"]))
    for key in ("status", "content", "call_ref", "live", "usage", "wall_s"):
        if response.get(key) != call.get(key):
            raise ValueError("모델 미완료 원문과 transport 결과가 다르다")
    if call.get("request") != request or call.get("status") == "completed":
        raise ValueError("모델 미완료 request/status가 다르다")
    return call, None


def _raw_owned_executions(host_result_path: Path) -> list[dict[str, Any]]:
    """HOST 검증 뒤 raw role receipt의 실행 시간과 owned cleanup을 별도 재검산한다."""
    host = _read(host_result_path)
    rows = []
    for role in HOST.ROLES:
        receipt_ref = host.get("receipts", {}).get(role)
        plan_ref = host.get("plans", {}).get(role)
        if not isinstance(receipt_ref, dict) or not isinstance(plan_ref, dict):
            raise TypeError("actual host role receipt/plan 참조가 없다")
        receipt_path, plan_path = _bound(receipt_ref), _bound(plan_ref)
        receipt = _read(receipt_path)
        if receipt.get("plan") != plan_ref or plan_path.parent != receipt_path.parent:
            raise ValueError("raw role receipt와 plan parent가 다르다")
        owned_ref = receipt.get("raw_refs", {}).get("owned_result")
        if not isinstance(owned_ref, dict):
            raise TypeError("raw role owned-result 참조가 없다")
        owned_path = _bound(owned_ref)
        if owned_path != plan_path.parent / "owned-result.json":
            raise ValueError("owned-result가 role plan parent 밖이다")
        owned = _read(owned_path)
        execution = receipt.get("execution")
        if execution != owned:
            raise ValueError("raw receipt execution과 owned-result가 다르다")
        identities = owned.get("owned_process_identities")
        wall = owned.get("wall_s")
        if (owned.get("natural_exit") is not True or owned.get("timed_out") is not False
                or owned.get("error_type") is not None or owned.get("exit_code") != 0
                or owned.get("cleanup_verified") is not True
                or owned.get("wall_scope") != "parent_launch_to_exit_and_owned_job_empty_verification"
                or not isinstance(identities, dict) or not identities
                or isinstance(wall, bool) or not isinstance(wall, (int, float))
                or not math.isfinite(float(wall)) or wall < 0):
            raise ValueError("raw owned child 실행 증거가 완결되지 않았다")
        identity_digest = _digest(identities)
        rows.append({"role": role, "receipt_sha256": receipt_ref["sha256"],
                     "owned_result_sha256": owned_ref["sha256"],
                     "identity_digest": identity_digest, "wall_s": float(wall),
                     "wall_scope": owned["wall_scope"], "exit_code": 0,
                     "natural_exit": True, "cleanup_verified": True})
    return rows


def _verify_row(row: dict, batch_root: Path, registered: dict, callbacks: dict[str, Any]) -> dict[str, Any]:
    source_id, arm = row.get("id"), row.get("arm")
    if (not isinstance(source_id, str) or not isinstance(arm, str)
            or source_id != registered.get("id") or arm not in M.ARMS):
        raise ValueError("batch row source/arm이 등록과 다르다")
    result_path = _bound(row.get("result"))
    if result_path != (batch_root / source_id / arm / "result.json").resolve():
        raise ValueError("batch row 결과 위치가 다르다")
    raw = _read(result_path)
    if (raw.get("schema") != "local_pair_four_arm_result_v1"
            or raw.get("scope") != "actual_backend"
            or raw.get("arm") != arm or raw.get("run_id") != f"{source_id}_{arm}"
            or raw.get("source", {}).get("sha256") != registered["prepared_result_sha256"]
            or raw.get("pair", {}).get("sha256") != registered["pair_sha256"]
            or raw.get("repair") is not False or raw.get("recovery") is not False
            or raw.get("heldout") is not False):
        raise ValueError("arm 결과의 등록·scope·claim 경계가 다르다")
    if any(row.get(key) != raw.get(key) for key in (
            "end_state", "confirmed_subtype", "hold_reason", "model_calls")):
        raise ValueError("batch row 요약이 arm 원문과 다르다")
    if row.get("paired_probes") != len(raw.get("steps", [])):
        raise ValueError("batch paired probe 수가 다르다")
    provider_wall = 0.0
    provider_calls = 0
    transport_calls: list[str] = []
    probe_order: list[str] = []
    owned_known = 0.0
    owned_unknown = 0
    raw_owned_wall = 0.0
    raw_role_executions: list[dict[str, Any]] = []
    last_inputs = None
    for index, step in enumerate(raw.get("steps", []), 1):
        decision = result_path.parent / f"step{index:02d}"
        stored_context = _read(decision / "context.json")
        call: dict[str, Any] | None = None
        if callbacks["fixture"]:
            context = callbacks["context"](stored_context)
            choice = _read(decision / "choice.json")
            model_call_ref = None
            if arm in M.MODEL_ARMS:
                call, expected_choice = _model_evidence(
                    decision, context, raw, callbacks, registered, index)
                model_call_ref = call.get("call_ref")
                if expected_choice != choice:
                    raise ValueError("모델 choice 재계산이 다르다")
        else:
            with _history_prefix_scope(stored_context["snapshot"]):
                context = callbacks["context"](stored_context)
                choice = _read(decision / "choice.json")
                model_call_ref = None
                if arm in M.MODEL_ARMS:
                    call, expected_choice = _model_evidence(
                        decision, context, raw, callbacks, registered, index)
                    model_call_ref = call.get("call_ref")
                    if expected_choice != choice:
                        raise ValueError("모델 choice 재계산이 다르다")
                elif M.C.choose(context["snapshot"], arm) != choice:
                    raise ValueError("등록 규칙 choice 재계산이 다르다")
        last_inputs = context["snapshot"].get("inputs")
        if arm in M.MODEL_ARMS:
            if call is None:
                raise AssertionError("모델 arm call 검증 누락")
            provider_wall += float(call["wall_s"])
            provider_calls += 1
            transport_calls.append(call["call_ref"]["sha256"])
        host_result_path = _bound(step.get("probe_result"))
        verified = callbacks["pair"](host_result_path, context["snapshot"], choice)
        if (step.get("probe") != verified.get("probe")
                or step.get("outcome") != verified.get("outcome")
                or step.get("probe") != choice.get("probe")
                or step.get("selected_by") != choice.get("selected_by")
                or step.get("fact_ids") != choice.get("fact_ids")
                or step.get("before_view_digest") != context["snapshot"].get("view_digest")
                or step.get("model_call") != model_call_ref):
            raise ValueError("paired probe 영수증 재검산이 다르다")
        if not callbacks["fixture"]:
            public = verified.get("public_ledger", {})
            prefix = public.get("prefix", {}) if isinstance(public, dict) else {}
            current_prefix = _prefix_bytes({**context["snapshot"], "view": {
                **context["snapshot"]["view"],
                "observations": [*context["snapshot"]["view"]["observations"],
                                 {"probe": step["probe"], "outcome": step["outcome"]}]}})
            if (prefix.get("byte_length") != len(current_prefix)
                    or prefix.get("sha256") != hashlib.sha256(current_prefix).hexdigest()):
                raise ValueError("host public ledger append prefix가 다르다")
            with _history_prefix_scope({**context["snapshot"], "view": {
                    **context["snapshot"]["view"],
                    "observations": [*context["snapshot"]["view"]["observations"],
                                     {"probe": step["probe"], "outcome": step["outcome"]}]}}):
                after = M.C.V.build_snapshot(**context["snapshot"]["inputs"])
            if (step.get("after_view_digest") != after["view_digest"]
                    or step.get("remaining_after") != after["view"]["remaining_candidates"]):
                raise ValueError("검사 뒤 remaining/view 상태가 다르다")
        probe_order.append(step["probe"])
        wall = verified.get("owned_child_wall_s")
        if isinstance(wall, bool) or wall is None:
            owned_unknown += 1
        elif isinstance(wall, (int, float)) and math.isfinite(float(wall)) and wall >= 0:
            owned_known += float(wall)
        else:
            raise ValueError("owned child wall 형식이 다르다")
        if not callbacks["fixture"]:
            role_rows = _raw_owned_executions(host_result_path)
            raw_role_executions.extend(role_rows)
            raw_owned_wall += sum(row["wall_s"] for row in role_rows)
    # 모델이 검사 전에 HOLD/invalid/unknown이면 step 디렉터리도 검증한다.
    for decision in sorted(result_path.parent.glob("step[0-9][0-9]")):
        step_number = int(decision.name[4:])
        if step_number <= len(raw.get("steps", [])):
            continue
        stored_context = _read(decision / "context.json")
        call = None
        if callbacks["fixture"]:
            context = callbacks["context"](stored_context)
            if arm in M.MODEL_ARMS:
                call, choice = _model_evidence(
                    decision, context, raw, callbacks, registered, step_number)
            else:
                choice = None
        else:
            with _history_prefix_scope(stored_context["snapshot"]):
                context = callbacks["context"](stored_context)
                if arm in M.MODEL_ARMS:
                    call, choice = _model_evidence(
                        decision, context, raw, callbacks, registered, step_number)
                else:
                    expected_rule = M.C.choose(context["snapshot"], arm)
                    if expected_rule.get("action") == "continue":
                        raise ValueError("규칙 continue 결과에 검사 step이 없다")
                    choice = None
        last_inputs = context["snapshot"].get("inputs")
        if arm not in M.MODEL_ARMS:
            if (decision / "choice.json").exists():
                raise ValueError("규칙 HOLD 분기에 choice 원문이 있다")
            continue
        choice_path = decision / "choice.json"
        if choice is not None and choice.get("action") == "continue":
            if not choice_path.is_file() or _read(choice_path) != choice:
                raise ValueError("후속 model choice가 다르다")
        elif choice_path.exists():
            raise ValueError("HOLD/invalid model 분기에 choice 원문이 있다")
        if call is None:
            raise AssertionError("모델 HOLD call 검증 누락")
        provider_wall += float(call["wall_s"])
        provider_calls += 1
        transport_calls.append(call["call_ref"]["sha256"])
    if provider_calls != raw.get("model_calls"):
        raise ValueError("model call 수가 원문과 다르다")
    if not callbacks["fixture"]:
        if not isinstance(last_inputs, dict):
            raise ValueError("최종 상태를 재생할 snapshot 입력이 없다")
        final = M.C.V.build_snapshot(**last_inputs)
        expected_state = "confirmed" if len(final["view"]["remaining_candidates"]) == 1 \
            and final["view"]["status"] == "confirmed" else "HOLD"
        if (raw.get("final_view_digest") != final["view_digest"]
                or raw.get("remaining_candidates") != final["view"]["remaining_candidates"]
                or raw.get("end_state") != expected_state
                or raw.get("confirmed_subtype") != (
                    final["view"]["remaining_candidates"][0] if expected_state == "confirmed" else None)):
            raise ValueError("최종 remaining/confirmed 상태가 다르다")
    correct = raw.get("confirmed_subtype") in registered["truth_subtypes"]
    wrong = raw.get("confirmed_subtype") is not None and not correct
    if row.get("correct_confirmation") is not correct:
        raise ValueError("batch correct confirmation 요약이 다르다")
    return {"source_id": source_id, "seed": registered["seed"], "arm": arm,
            "end_state": raw["end_state"], "confirmed_subtype": raw.get("confirmed_subtype"),
            "correct_confirmation": correct, "wrong_confirmation": wrong,
            "hold": raw["end_state"] == "HOLD", "hold_reason": raw.get("hold_reason"),
            "semantic_invalid": raw.get("semantic_invalid"),
            "infra_unknown": raw.get("infra_unknown"),
            "provider_unknown": raw.get("provider_unknown"), "probe_order": probe_order,
            "paired_probe_count": len(raw["steps"]), "provider_calls": provider_calls,
            "transport_call_digests": transport_calls,
            "provider_wall_s": provider_wall, "owned_child_known_wall_s": owned_known,
            "owned_child_unknown_count": owned_unknown,
            "raw_owned_child_wall_s": raw_owned_wall,
            "raw_role_executions": raw_role_executions,
            "frame_fact_citation_count": raw.get("frame_fact_citations"),
            "result_digest": _sha(result_path)}


def _safe_public(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if PRIVATE_PATTERN.search(encoded) or "reasoning_content" in encoded or "NVIDIA_API_KEY" in encoded:
        raise ValueError("공개 결과에 경로·reasoning·key 표식이 있다")


def export(batch_result: Path, protocol_path: Path, anchor: str, out: Path, *,
           allow_fixture: bool = False, verifiers: dict[str, Any] | None = None) -> dict[str, Any]:
    """완료 원문을 검증하고 새 공개 폴더에 네 파일만 쓴다."""
    output = Path(out).resolve()
    if output.exists():
        raise FileExistsError("공개 출력은 fresh 경로여야 한다")
    batch_path = Path(batch_result).resolve(strict=True)
    batch = _read(batch_path)
    if batch.get("schema") != "local_pair_four_arm_batch_v1" or batch.get("status") != "complete":
        raise ValueError("완료된 네 방식 batch만 공개할 수 있다")
    callbacks = _callbacks(allow_fixture, verifiers)
    protocol = callbacks["protocol"](Path(protocol_path), anchor)
    if allow_fixture and protocol.get("fixture") is not True:
        raise ValueError("cpu_fixture protocol 표식이 없다")
    if (batch.get("scope") != "development_seen_sources"
            or batch.get("independent_source_seeds") != 2
            or any(batch.get(key) is not False for key in (
                "heldout", "repair", "recovery", "llm_superiority", "video_advantage_confirmed"))):
        raise ValueError("batch 개발 scope/claim 경계가 다르다")
    registered = {row["id"]: row for row in protocol.get("pairs", [])}
    rows = batch.get("rows")
    if (not isinstance(rows, list) or len(rows) != 8
            or {(row.get("id"), row.get("arm")) for row in rows}
            != {(source, arm) for source in registered for arm in M.ARMS}):
        raise ValueError("fault row 8개 구성이 다르다")
    cases = [_verify_row(row, batch_path.parent, registered[row["id"]], callbacks) for row in rows]
    controls = batch.get("normal_host_controls")
    expected_controls = {(case, arm) for case in protocol.get("normal_controls", []) for arm in M.ARMS}
    if (not isinstance(controls, list) or len(controls) != 8
            or {(row.get("id"), row.get("arm")) for row in controls} != expected_controls
            or any(row.get("end_state") != "HOLD" or row.get("reason") != "healthy_host_gate"
                   or row.get("model_calls") != 0 or row.get("paired_probes") != 0
                   or row.get("model_normal_recognition") is not False for row in controls)):
        raise ValueError("정상 host gate 8행이 다르다")
    if batch.get("model_calls") != sum(row["provider_calls"] for row in cases):
        raise ValueError("batch API call 수가 재검산과 다르다")
    call_digests = [digest for row in cases for digest in row["transport_call_digests"]]
    if len(call_digests) != len(set(call_digests)) or len(call_digests) != batch["model_calls"]:
        raise ValueError("transport call이 행 사이에서 중복되거나 누락됐다")
    if batch.get("paired_intents") != sum(row["paired_probe_count"] for row in cases):
        raise ValueError("batch paired intent 수가 재검산과 다르다")
    role_keys = [(item["receipt_sha256"], item["identity_digest"])
                 for row in cases for item in row["raw_role_executions"]]
    if not allow_fixture and (len(role_keys) != 2 * batch["paired_intents"]
                              or len(role_keys) != len(set(role_keys)) or len(role_keys) != 32):
        raise ValueError("actual SDK role 실행 32개가 고유 receipt/identity로 완결되지 않았다")
    by_arm = {}
    for arm in M.ARMS:
        arm_rows = [row for row in cases if row["arm"] == arm]
        successes = sum(row["correct_confirmation"] for row in arm_rows)
        by_arm[arm] = {"correct_confirmation": _wilson(successes, len(arm_rows)),
                       "wrong_confirmation": sum(row["wrong_confirmation"] for row in arm_rows),
                       "hold": sum(row["hold"] for row in arm_rows),
                       "semantic_invalid": sum(int(row["semantic_invalid"]) for row in arm_rows),
                       "infra_unknown": sum(int(row["infra_unknown"]) for row in arm_rows),
                       "provider_unknown": sum(int(row["provider_unknown"]) for row in arm_rows),
                       "paired_probe_count": sum(row["paired_probe_count"] for row in arm_rows),
                       "provider_calls": sum(row["provider_calls"] for row in arm_rows),
                       "provider_wall_s": sum(row["provider_wall_s"] for row in arm_rows),
                       "owned_child_known_wall_s": sum(row["owned_child_known_wall_s"] for row in arm_rows),
                       "owned_child_unknown_count": sum(row["owned_child_unknown_count"] for row in arm_rows),
                       "raw_owned_child_wall_s": sum(row["raw_owned_child_wall_s"] for row in arm_rows),
                       "raw_role_execution_count": sum(len(row["raw_role_executions"])
                                                       for row in arm_rows),
                       "frame_fact_citation_count": sum(int(row["frame_fact_citation_count"])
                                                        for row in arm_rows)}
    paired = []
    logs_better = frames_better = 0
    for source in registered:
        logs = next(row for row in cases if row["source_id"] == source and row["arm"] == "logs_only")
        frames = next(row for row in cases if row["source_id"] == source and row["arm"] == "logs_plus_frames")
        logs_better += int(logs["correct_confirmation"] and not frames["correct_confirmation"])
        frames_better += int(frames["correct_confirmation"] and not logs["correct_confirmation"])
        paired.append({"source_id": source, "seed": registered[source]["seed"],
                       "logs_only_correct": logs["correct_confirmation"],
                       "logs_plus_frames_correct": frames["correct_confirmation"]})
    summary = {"schema": SCHEMA + "_summary", "status": "complete",
               "scope": "cpu_fixture" if allow_fixture else "development_seen_sources",
               "fault_rows": 8, "healthy_host_controls": 8, "independent_source_seeds": 2,
               "arms": by_arm, "paired_model_contrast": _exact_p(logs_better, frames_better),
               "seed_paired_contrasts": paired, "actual_api_calls": batch["model_calls"],
               "actual_sdk_role_executions": len(role_keys) if not allow_fixture else None,
               "sdk_role_executions_upper_bound": protocol.get("sdk_role_executions_upper_bound"),
               "claims": {"heldout": False, "llm_superiority": False,
                          "video_advantage_confirmed": False, "recovery": False,
                          "repair": False, "default_mode_adopted": False},
               "uncertainty": "독립 source seed는 2개이며 8 fault row를 독립 seed 8개로 해석하지 않는다."
                              " paired exact p와 Wilson 구간은 기술값이며 우월성 주장이 아니다.",
               "normal_control_semantics": "정상 8행은 host healthy gate이며 모델의 정상 식별이 아니다."}
    public_cases = {"schema": SCHEMA + "_cases", "fault_rows": cases,
                    "healthy_host_controls": controls}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    provenance = {"schema": SCHEMA + "_provenance", "protocol_sha256_lf": _sha(protocol_path, lf=True),
                  "execution_anchor": anchor, "batch_sha256": _sha(batch_path),
                  "analysis_head": head,
                  "exporter_sources_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
                  "protocol_sources_digest": _digest(protocol.get("sources_lf")),
                  "logical_sources": [{"id": key, "seed": value["seed"],
                                       "pair_sha256": value["pair_sha256"],
                                       "case_checkpoint_sha256": value["case_checkpoint_sha256"],
                                       "reference_checkpoint_sha256": value["reference_checkpoint_sha256"]}
                                      for key, value in registered.items()],
                  "raw_model_responses_copied": False, "raw_images_or_npz_copied": False,
                  "api_keys_copied": False}
    for value in (summary, public_cases, provenance):
        _safe_public(value)
    output.mkdir(parents=True)
    _write_new(output / "summary.json", summary)
    _write_new(output / "cases.json", public_cases)
    _write_new(output / "provenance.json", provenance)
    manifest = {"schema": SCHEMA + "_manifest", "files": {
        name: _sha(output / name) for name in ("summary.json", "cases.json", "provenance.json")}}
    _safe_public(manifest)
    _write_new(output / "manifest.json", manifest)
    return deepcopy(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.batch_result, args.protocol, args.anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

