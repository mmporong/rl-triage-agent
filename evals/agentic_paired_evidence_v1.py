"""동일한 개발 snapshot에서 logs-only와 logs+frames 모델 입력을 짝으로 준비한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import agentic_investigation_call_v2 as CALL
import agentic_investigation_contract_v2 as A2

SCHEMA = "agentic_paired_evidence_v1"
ARMS = ("logs_only", "logs_plus_frames")
FRAME_PREFIX = "vision:frame:"
PLAY_PREFIX = "vision:play:"
PAIRED_GOAL = (
    "case와 reference의 같은 Play 수치와 학습 기록을 비교해 다음 등록 검사 하나를 선택한다. "
    "근거가 부족하거나 출력이 무효면 hold한다."
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def _without_response_format(params: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(params)
    value.pop("response_format", None)
    return value


def _redacted_message_shape(messages: list[dict[str, Any]]) -> dict[str, Any]:
    content = messages[1]["content"]
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    return {
        "roles": [row["role"] for row in messages],
        "block_types": [row.get("type") for row in blocks],
        "text_blocks": sum(row.get("type") == "text" for row in blocks),
        "image_blocks": sum(row.get("type") == "image_url" for row in blocks),
        "messages_sha256": digest(messages),
    }


def _context_common(context: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(context)
    value.pop("context_digest", None)
    extra = value.pop("extra_facts")
    value["extra_facts"] = {key: item for key, item in extra.items() if not key.startswith(FRAME_PREFIX)}
    view = value["model_view"]
    facts = view.pop("evidence_facts")
    view["evidence_facts"] = {key: item for key, item in facts.items() if not key.startswith(FRAME_PREFIX)}
    return value


def _arm(name: str, context: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    checked = A2.checked(context)
    outgoing = CALL._messages(checked, messages)
    params = CALL.params_for(checked)
    require(params == CALL.params_for_messages(outgoing), f"{name} provider messages/params contract")
    return {"name": name, "context": checked, "messages": outgoing, "params": params}


def prepare(prepared_path: Path, output: Path) -> dict[str, Any]:
    prepared_path = prepared_path.resolve(strict=True)
    source = read(prepared_path)
    require(source.get("schema") == "fresh_frame_decision_prepared_v1"
            and source.get("status") == "ready_for_registered_calls"
            and source.get("approval_granted") is False, "fresh frame prepared input")
    source_context = A2.checked(source["context"])
    source_messages = CALL._messages(source_context, source["messages"])
    extra = source_context["extra_facts"]
    frame_facts = {key: value for key, value in extra.items() if key.startswith(FRAME_PREFIX)}
    common_extra = {key: value for key, value in extra.items() if not key.startswith(FRAME_PREFIX)}
    require(len(frame_facts) == 6 and all(value.get("evidence_type") == "raw_frame"
            for value in frame_facts.values()), "six registered raw-frame facts required")
    require(sum(key.startswith(PLAY_PREFIX) for key in common_extra) == 54,
            "same-Play numeric facts must remain common")
    only_context = A2.prepare(source_context["snapshot"], goal=PAIRED_GOAL,
                              evidence=source_context["evidence"], revealed=source_context["revealed"],
                              extra_facts=common_extra, decisions_left=source_context["decisions_left"])
    plus_context = A2.prepare(source_context["snapshot"], goal=PAIRED_GOAL,
                              evidence=source_context["evidence"], revealed=source_context["revealed"],
                              extra_facts=extra, decisions_left=source_context["decisions_left"])
    only_messages = A2.messages(only_context)
    source_blocks = source_messages[1]["content"]
    require(isinstance(source_blocks, list) and len(source_blocks) == 13,
            "fresh-frame message blocks")
    plus_messages = A2.messages(plus_context)
    plus_messages[1]["content"] = [
        {"type": "text", "text": plus_messages[1]["content"]}, *deepcopy(source_blocks[1:])]
    only = _arm("logs_only", only_context, only_messages)
    plus = _arm("logs_plus_frames", plus_context, plus_messages)
    require(_context_common(only["context"]) == _context_common(plus["context"]),
            "paired contexts differ outside goal/raw-frame exposure")
    only_view, plus_view = only["context"]["model_view"], plus["context"]["model_view"]
    require(only_view["eligible_probes"] == plus_view["eligible_probes"]
            and only_view["available_actions"] == plus_view["available_actions"]
            and only["context"]["snapshot"] == plus["context"]["snapshot"],
            "paired snapshot/probes/actions differ")
    require(_without_response_format(only["params"]) == _without_response_format(plus["params"]),
            "paired base model params differ")
    require(not any(key.startswith(FRAME_PREFIX) for key in only_view["evidence_facts"])
            and set(plus_view["evidence_facts"]) - set(only_view["evidence_facts"]) == set(frame_facts),
            "raw-frame fact exposure differs from registered six")
    output.mkdir(parents=True, exist_ok=False)
    provider_refs = {}
    for arm in (only, plus):
        provider_refs[arm["name"]] = write_new(output / f"{arm['name']}.provider.json", {
            "schema": SCHEMA + "_provider_input", "arm": arm["name"],
            "context": arm["context"], "messages": arm["messages"], "params": arm["params"],
            "approval_granted": False,
        })
    differences = {
        "allowed_only": [
            "context.context_digest", "context.extra_facts[vision:frame:*]",
            "context.model_view.evidence_facts[vision:frame:*]",
            "messages.user.image_blocks", "messages.user.frame_label_blocks", "params.response_format.fact_ids",
        ],
        "removed_raw_frame_fact_ids": sorted(frame_facts),
        "common_telemetry_fact_count": sum(key.startswith("telemetry:") for key in common_extra),
        "common_same_play_numeric_fact_count": sum(key.startswith(PLAY_PREFIX) for key in common_extra),
        "snapshot_same": True,
        "goal_same": True,
        "eligible_probes_same": True,
        "available_actions_same": True,
        "base_model_params_same": True,
    }
    pair = {
        "schema": SCHEMA + "_pair", "status": "prepared_cpu_fixture_not_model_result",
        "source_prepared": {"sha256": file_sha(prepared_path)},
        "arms": {
            only["name"]: {"provider_input": provider_refs[only["name"]],
                            "shape": _redacted_message_shape(only["messages"]),
                            "context_sha256": digest(only["context"]), "params_sha256": digest(only["params"])},
            plus["name"]: {"provider_input": provider_refs[plus["name"]],
                            "shape": _redacted_message_shape(plus["messages"]),
                            "context_sha256": digest(plus["context"]), "params_sha256": digest(plus["params"])},
        },
        "differences": differences,
        "binding_sha256": digest({"source": file_sha(prepared_path), "arms": provider_refs,
                                   "differences": differences}),
        "labels_exposed": False,
        "approval_granted": False,
        "model_calls": 0,
    }
    write_new(output / "pair.json", pair)
    return pair


def _receipt(value: dict[str, str] | None, arm: str, pair: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    require(set(value) == {"path", "sha256"}, f"{arm} receipt ref")
    path = Path(value["path"]).resolve(strict=True)
    require(file_sha(path) == value["sha256"], f"{arm} receipt SHA")
    receipt = read(path)
    require(receipt.get("schema") == SCHEMA + "_arm_receipt" and receipt.get("arm") == arm,
            f"{arm} receipt schema/arm")
    require(receipt.get("pair_binding_sha256") == pair["binding_sha256"]
            and receipt.get("provider_input_sha256") == pair["arms"][arm]["provider_input"]["sha256"],
            f"{arm} receipt pair/provider binding")
    return receipt


def compare(pair_path: Path, output: Path, logs_only_receipt: dict[str, str] | None = None,
            logs_plus_frames_receipt: dict[str, str] | None = None) -> dict[str, Any]:
    pair = read(pair_path.resolve(strict=True))
    require(pair.get("schema") == SCHEMA + "_pair", "paired evidence manifest")
    receipts = {
        "logs_only": _receipt(logs_only_receipt, "logs_only", pair),
        "logs_plus_frames": _receipt(logs_plus_frames_receipt, "logs_plus_frames", pair),
    }
    complete = all(row is not None and row.get("status") == "complete" for row in receipts.values())
    required = {"model_valid", "hold", "selected_probe", "frame_fact_ids", "confirmed_cause",
                "false_confirmation", "normal_intervention", "probe_count", "cost"}
    if complete:
        require(all(required <= set(row["outcomes"]) for row in receipts.values() if row is not None),
                "arm receipt outcomes incomplete")
    result: dict[str, Any] = {
        "schema": SCHEMA + "_comparison", "pair_binding_sha256": pair["binding_sha256"],
        "status": "inconclusive", "contribution": "inconclusive",
        "actual_outcomes_verified": False,
        "reason": ("operational_model_probe_and_fixed_evaluation_verifier_not_connected" if complete
                   else "both_arm_bound_receipts_required_before_operational_verification"),
        "model_selection_change_is_success": False,
        "frame_citation_is_success": False,
        "receipt_presence": {name: row is not None for name, row in receipts.items()},
    }
    if complete:
        result["self_reported_arm_outcomes"] = {
            name: row["outcomes"] for name, row in receipts.items() if row is not None}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--pair", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--logs-only-receipt", type=Path)
    c.add_argument("--logs-only-receipt-sha256")
    c.add_argument("--logs-plus-frames-receipt", type=Path)
    c.add_argument("--logs-plus-frames-receipt-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.prepared, args.output)
    else:
        only = None if args.logs_only_receipt is None else {
            "path": str(args.logs_only_receipt.resolve()), "sha256": args.logs_only_receipt_sha256}
        plus = None if args.logs_plus_frames_receipt is None else {
            "path": str(args.logs_plus_frames_receipt.resolve()),
            "sha256": args.logs_plus_frames_receipt_sha256}
        result = compare(args.pair, args.output, only, plus)
    print(json.dumps({key: value for key, value in result.items() if key not in {"messages", "params"}},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
