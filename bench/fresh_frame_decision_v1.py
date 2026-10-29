from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _source_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "evals/agentic_investigation_contract_v2.py").is_file():
            return parent
    raise ValueError("repo root를 찾지 못함")


def resolve(repo: Path, ref: dict) -> Path:
    path = (repo / ref["path"]).resolve(strict=True)
    require(path.is_relative_to(repo.resolve()), "repo 밖 입력")
    require(sha(path) == ref["sha256"], "입력 SHA 불일치")
    return path


def verify_budget(repo: Path, protocol: dict) -> dict:
    budget = protocol["api_budget"]
    baseline_path = resolve(repo, budget["baseline"])
    baseline = read(baseline_path)
    require(baseline["schema"] == "agentic_api_preserved_baseline_declaration_v1", "baseline schema")
    require(baseline["api_cap"] == 12 and baseline["baseline_completed"] == 7, "baseline 7/12")
    registration = read(resolve(repo, budget["registration"]))
    root = Path(registration["api"]["api_root"]).resolve(strict=True)
    for name, expected in baseline["preserved_raw_files"].items():
        require(sha(root / name) == expected, f"baseline raw mismatch: {name}")
    indices = []
    for journal_ref in budget["additional_journals"]:
        manifest_path = resolve(repo, journal_ref["manifest"])
        terminal_path = resolve(repo, journal_ref["terminal"])
        event_path = resolve(repo, journal_ref["event"])
        manifest, terminal, event = read(manifest_path), read(terminal_path), read(event_path)
        require(manifest["api_cap"] == 12 and manifest["params"]["model"] == MODEL, "journal model/cap")
        require(terminal["schema"] == "model_decision_terminal_v1"
                and terminal["completed_calls"] == 1 and terminal["status"] == "choice_ready", "journal unsettled")
        require(event["manifest_sha256"] == sha(manifest_path)
                and terminal["manifest_sha256"] == sha(manifest_path), "journal chain")
        indices.append(event["global_index"])
    require(indices == [8, 9], "additional call indices")
    return {"api_cap": 12, "completed": 9, "pending": 0, "remaining": 3,
            "registered_new_call_cap": 2, "unallocated_after_run": 1}


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _local_path(value: str) -> Path:
    if value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def prepare(repo: Path, protocol_path: Path, output: Path, public_protocol: Path | None = None) -> dict:
    protocol = read(protocol_path)
    require(protocol["schema"] == "fresh_frame_decision_preregistration_v1", "protocol schema")
    public_protocol = (public_protocol or protocol_path.with_name("fresh_frame_decision_public_protocol_v1.json")).resolve(strict=True)
    require(sha(public_protocol) == protocol["public_preregistration_sha256"], "public preregistration SHA")
    budget = verify_budget(repo, protocol)
    inputs = protocol["inputs"]
    receipt = read(resolve(repo, inputs["receipt"]))
    frame_refs = read(resolve(repo, inputs["frame_refs"]))
    facts = read(resolve(repo, inputs["same_play_facts"]))
    index = read(resolve(repo, inputs["index_spec"]))
    capture = read(resolve(repo, inputs["capture_receipt"]))
    training = read(resolve(repo, inputs["training_receipt"]))
    delivery = read(resolve(repo, inputs["root_delivery"]))
    require(receipt["cpu_preparation"] == "PASS" and receipt["mode"] == "production", "RI111 production receipt")
    require(receipt["collection"]["all_match"] is True and receipt["collection"]["files_verified"] == 3958,
            "collection binding")
    require(len(frame_refs["frames"]) == 24, "24 fresh frames")
    require(facts["thresholds_created"] is False and facts["behavior_labels_created"] is False, "labels created")
    require(len(index["entries"]) == 4 and capture["cpu_preparation"] == "PASS"
            and training["cpu_preparation"] == "PASS",
            "capture/training receipts")
    require(delivery["resource_status"] == "STOPPED", "H delivery not stopped")
    for policy in receipt["agentic_stages"].values():
        require(policy["old_renderer_overlap"] is False, "old renderer overlap")

    sys.path[:0] = [str(repo / "evals"), str(repo / "src")]
    import agentic_investigation_call_v2 as caller
    import agentic_investigation_contract_v2 as contract
    snapshot_path = resolve(repo, inputs["case_snapshot"])
    evidence_snapshot = read(snapshot_path)
    registration = read(resolve(repo, protocol["api_budget"]["registration"]))
    snapshot = registration["snapshot"]
    require(snapshot["inputs"]["packet"]["input_bindings"]
            == evidence_snapshot["inputs"]["packet"]["input_bindings"], "snapshot input bindings")
    # F1/NONE 이름은 모델 입력에 넣지 않고 case/reference 역할만 제공한다.
    chosen = []
    extra = {key: value for key, value in evidence_snapshot["view"]["evidence_facts"].items()
             if key != "config:action_scale"}
    bundles = {"case": read(resolve(repo, inputs["case_bundle"])),
               "reference": read(resolve(repo, inputs["reference_bundle"]))}
    for alias, logical in (("case", "source-s2221-f1"), ("reference", "source-s2221-none")):
        rows = [row for row in frame_refs["frames"] if row["logical_id"] == logical
                and row["window"] == "window-01" and row["frame_id"] in {"f01", "f03", "f06"}]
        require(len(rows) == 3, "fresh frame selection")
        for offset, row in enumerate(rows, 1):
            image = _local_path(row["jpeg"]["path"])
            require(image.is_file() and sha(image) == row["jpeg"]["sha256"], "fresh frame bytes")
            frame_id = f"f{offset if alias == 'case' else offset + 3}"
            extra[f"vision:frame:{frame_id}"] = {"evidence_type": "raw_frame",
                "timestamp_s": row["timestamp_s"], "source_group": "visual_source_1"}
            for key, value in bundles[alias]["selection_view"]["evidence_facts"].items():
                if key.startswith(f"vision:play:{row['frame_id']}:"):
                    extra[f"vision:play:{frame_id}:" + key.split(":", 3)[3]] = value
            chosen.append({"alias": alias, "frame_id": frame_id, "timestamp_s": row["timestamp_s"],
                           "sha256": row["jpeg"]["sha256"],
                           "image_url": {"url": _data_url(image), "detail": "low"}})
    context = contract.prepare(snapshot, goal=protocol["goal"], extra_facts=extra, decisions_left=2)
    view = context["model_view"]
    require(view["status"] == "open" and len(view["eligible_probes"]) == 7, "registered probes")
    public = {"schema": "fresh_frame_decision_packet_v1", "split": "development", "seed": 2221,
              "case_aliases": ["case", "reference"],
              "training_bindings": snapshot["inputs"]["packet"]["input_bindings"],
              "frame_bindings": [{k: row[k] for k in ("alias", "frame_id", "timestamp_s", "sha256")} for row in chosen],
              "api_budget": budget, "labels_available_to_model": False,
              "old_renderer_prefix_reused": False, "used_for_decision": True}
    messages = contract.messages(context)
    blocks = [{"type": "text", "text": messages[1]["content"]}]
    for row in chosen:
        blocks += [{"type": "text", "text": json.dumps({"frame_id": row["frame_id"],
                                                           "timestamp_s": row["timestamp_s"]}, sort_keys=True)},
                   {"type": "image_url", "image_url": row["image_url"]}]
    messages[1]["content"] = blocks
    messages = caller._messages(context, messages)
    params = caller.params_for(context)
    require(params == caller.params_for_messages(messages), "provider params/messages contract")
    result = {"schema": "fresh_frame_decision_prepared_v1", "status": "ready_for_registered_calls",
              "protocol_sha256": sha(public_protocol), "budget": budget, "context": context,
              "public_packet": public,
              "messages": messages, "params": params,
              "call_sequence": ["choose_one_registered_probe", "follow_up_after_valid_probe_receipt"],
              "invalid_decision_policy": "HOLD", "approval_granted": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "prepared.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                                      separators=(",", ":")), encoding="utf-8")
    return result


def assess(prepared_path: Path, content: str) -> dict:
    prepared = read(prepared_path)
    repo = _source_repo(); sys.path[:0] = [str(repo / "evals"), str(repo / "src")]
    import agentic_investigation_contract_v2 as contract
    review = contract.assess(content, prepared["context"])
    if review["status"] != "valid":
        return {"status": "HOLD", "model_status": "invalid", "review": review, "approval_granted": False}
    choice = contract.probe_choice(prepared["context"], review) if review["decision"]["action"] == "run_probe" else None
    return {"status": "choice_ready", "model_status": "valid", "review": review,
            "controller_choice": choice, "approval_granted": False}


def call(prepared_path: Path, journal_root: Path, api_root: Path, source_anchor: str) -> dict:
    prepared = read(prepared_path)
    require(prepared["budget"]["completed"] == 9 and prepared["budget"]["pending"] == 0, "prepared budget")
    repo = _source_repo()
    sys.path[:0] = [str(repo / "evals"), str(repo / "src")]
    import grounded_vision_observer_v1 as observer
    import model_decision_journal_v1 as journal
    import persistent_provider_complete_v1 as provider
    public = prepared["public_packet"]
    binding = {"run_id": "fresh-frame-s2221-" + prepared["protocol_sha256"][:16],
               "source_anchor": source_anchor, "protocol_sha256": prepared["protocol_sha256"],
               "checkpoint_sha256": public["training_bindings"]["case"]["checkpoint_sha256"],
               "snapshot_sha256": journal.digest(prepared["context"]["snapshot"]),
               "initial_messages_sha256": journal.digest(prepared["messages"])}
    decision = journal.DecisionJournal(journal_root, api_root, binding, prepared["params"], api_cap=12)

    def engine(complete):
        safe = complete(prepared["messages"], prepared["params"])
        review = assess(prepared_path, safe.get("content", ""))
        return {"selection": review["review"].get("decision") or {"action": "hold"},
                "review": review, "model_status": review["model_status"], "approval_granted": False}

    return decision.advance(engine,
        lambda messages, params: observer._provider_result(provider.bridge_complete(messages, params)),
        lambda: datetime.now(timezone.utc))  # noqa: UP017 - Python 3.10


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--repo", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--public-protocol", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("assess")
    a.add_argument("--prepared", type=Path, required=True); a.add_argument("--content", required=True)
    c = sub.add_parser("call")
    c.add_argument("--prepared", type=Path, required=True); c.add_argument("--journal", type=Path, required=True)
    c.add_argument("--api-root", type=Path, required=True); c.add_argument("--source-anchor", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.repo.resolve(strict=True), args.protocol.resolve(strict=True), args.out.resolve(),
                        args.public_protocol.resolve(strict=True) if args.public_protocol else None)
    elif args.command == "assess":
        value = assess(args.prepared.resolve(strict=True), args.content)
    else:
        value = call(args.prepared.resolve(strict=True), args.journal.resolve(),
                     args.api_root.resolve(strict=True), args.source_anchor)
    print(json.dumps(value if args.command != "prepare" else {"status": value["status"], "budget": value["budget"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
