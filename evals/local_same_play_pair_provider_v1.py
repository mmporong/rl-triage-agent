"""승인된 local same-Play 원문을 API 권한 없는 paired provider 입력으로 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_evidence_index_v1 as E
import agentic_investigation_call_v2 as CALL
import agentic_investigation_contract_v2 as A2
import agentic_paired_evidence_v1 as PAIRED
import local_same_play_pair_inputs_v1 as INPUT
import selection_comparison_controller_v1 as C
import vision_evidence_contract_v1 as V01

SCHEMA = "local_same_play_pair_provider_protocol_v1"
PREPARED_SCHEMA = "local_same_play_pair_provider_prepared_v1"
PAIR_GOAL = (PAIRED.PAIRED_GOAL
             + " Frame role mapping: case=f1,f2,f3; reference=f4,f5,f6.")
FRAME_SELECTION = {"window": 1, "source_indices_per_role": [0, 2, 5],
                   "case": ["f1", "f2", "f3"], "reference": ["f4", "f5", "f6"]}
PROJECTION = {"training_facts": 48, "same_play_numeric_facts": 54,
              "runtime_facts_excluded_equally": 16, "image_counts": [0, 6]}
AUTHORITY = {"cpu_input_preparation": True, "api": False, "gpu": False,
             "probe": False, "correction": False}
SOURCE_FILES = (
    "evals/local_same_play_pair_provider_v1.py",
    "tests/test_local_same_play_pair_provider_v1.py",
    "evals/local_same_play_pair_inputs_v1.py",
    "evals/agentic_state_facts_v1.py",
    "evals/vision_evidence_contract_v1.py",
    "evals/agentic_evidence_index_v1.py",
    "evals/agentic_investigation_contract_v1.py",
    "evals/agentic_investigation_contract_v2.py",
    "evals/agentic_investigation_call_v1.py",
    "evals/agentic_investigation_call_v2.py",
    "evals/agentic_paired_evidence_v1.py",
    "evals/selection_comparison_controller_v1.py",
    "evals/selection_comparison_view_v1.py",
    "evals/residual_selection_v2.py",
    "evals/vision_transferred_input_acceptance_v2.py",
    "evals/vision_selection_sources_v1.py",
    "bench/record_vision_evidence_v1.py",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")))
    require(isinstance(value, dict), "JSON object required")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")
                          .replace(b"\r", b"\n")).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    checked = Path(path).resolve(strict=True)
    require(checked.is_file() and not checked.is_symlink(), "unsafe ref")
    return {"path": str(checked), "sha256": sha(checked)}


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_protocol(input_protocol: Path, input_execution_anchor: str, output: Path,
                     *, source_anchor: str | None = None) -> dict[str, Any]:
    require(len(input_execution_anchor) == 40, "input execution anchor")
    value = {
        "schema": SCHEMA,
        "status": "registered_before_provider_materialization" if source_anchor else "candidate_not_registered",
        "source_anchor": source_anchor,
        "sources_lf": {name: lf_sha(ROOT / name) for name in SOURCE_FILES},
        "input_protocol": {"path": "bench/protocols/local_same_play_pair_inputs_s2430_s2431_20261101_v1.json",
                           "sha256": sha(input_protocol), "execution_anchor": input_execution_anchor},
        "frame_selection": FRAME_SELECTION,
        "projection": PROJECTION,
        "authority": AUTHORITY,
    }
    if source_anchor is not None:
        require(len(source_anchor) == 40 and all(c in "0123456789abcdef" for c in source_anchor),
                "source anchor")
    write_new(output, value)
    return value


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=check)


def verify_protocol(path: Path, input_protocol: Path, execution_anchor: str,
                    *, repo: Path = ROOT) -> dict[str, Any]:
    value = read(path)
    source_anchor = value.get("source_anchor")
    require(value.get("schema") == SCHEMA and value.get("status") == "registered_before_provider_materialization"
            and isinstance(source_anchor, str) and len(source_anchor) == 40
            and len(execution_anchor) == 40, "registered provider protocol")
    checked_source_anchor = str(source_anchor)
    require(value.get("input_protocol") == {
        "path": "bench/protocols/local_same_play_pair_inputs_s2430_s2431_20261101_v1.json",
        "sha256": sha(input_protocol), "execution_anchor": value["input_protocol"]["execution_anchor"]},
        "input protocol binding")
    require(value.get("frame_selection") == FRAME_SELECTION
            and value.get("projection") == PROJECTION
            and value.get("authority") == AUTHORITY, "provider registered policy")
    INPUT.verify_protocol(
        input_protocol,
        repo / "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json",
        value["input_protocol"]["execution_anchor"],
        repo=repo,
    )
    current = {name: lf_sha(repo / name) for name in SOURCE_FILES}
    require(value.get("sources_lf") == current, "provider source closure")
    require(_git(repo, "merge-base", "--is-ancestor", checked_source_anchor, execution_anchor,
                 check=False).returncode == 0
            and _git(repo, "merge-base", "--is-ancestor", execution_anchor, "HEAD",
                     check=False).returncode == 0, "provider anchor ancestry")
    for name, expected in current.items():
        blob = _git(repo, "show", f"{checked_source_anchor}:{name}").stdout
        require(hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
                == expected, f"provider source blob: {name}")
    relative = Path(path).resolve().relative_to(repo.resolve()).as_posix()
    require(_git(repo, "show", f"{execution_anchor}:{relative}").stdout.replace(b"\r\n", b"\n")
            == Path(path).read_bytes().replace(b"\r\n", b"\n"), "provider protocol blob")
    return value


def _approval(path: Path, pending_path: Path, pending: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == "local_pair_provider_preparation_approval_v1"
            and value.get("scope") == "CPU_provider_input_materialization_only"
            and value.get("pending") == ref(pending_path)
            and value.get("pending_binding_sha256") == pending["binding_sha256"]
            and value.get("input_execution_anchor") == protocol["input_protocol"]["execution_anchor"]
            and value.get("expected_image_counts") == [0, 6]
            and value.get("window") == 1 and value.get("source_indices_per_role") == [0, 2, 5]
            and value.get("frame_role_mapping") == {"case": ["f1", "f2", "f3"],
                                                     "reference": ["f4", "f5", "f6"]}
            and value.get("input_preparation_authorized") is True
            and all(value.get(key) is False for key in ("api_authorized", "probe_approval",
                                                        "gpu_authorized", "correction_authorized",
                                                        "recovery_claim", "capture_flags_mutated")),
            "provider preparation approval")
    return value


def _packet(state_path: Path, pending: dict[str, Any]) -> dict[str, Any]:
    state = read(state_path)
    packet = C.V.S.validate_packet(state["initial_packet"])
    runtime = {key: value for key, value in packet["facts"].items() if key.startswith("runtime:")}
    common = {key: value for key, value in packet["facts"].items() if not key.startswith("runtime:")}
    require(len(runtime) == 16 and len(common) == 48
            and common == pending["source"]["training_facts"], "common training fact projection")
    expected = {role: {"checkpoint_sha256": pending["source"]["training"][role]["checkpoint"]["sha256"],
                       "telemetry_sha256": pending["source"]["training"][role]["telemetry"]["sha256"]}
                for role in ("case", "reference")}
    require(packet["input_bindings"] == expected, "packet/pending CP telemetry binding")
    projected = deepcopy(packet)
    projected["facts"] = common
    return C.V.S.validate_packet(projected)


def _evidence_index(contract_path: Path, raw_trace: Path, checkpoint_sha: str) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = V01.verify(contract_path)
    roots = {"contract": contract_path.parent, "trace": raw_trace.parent}
    spec = {"schema": E.SPEC_SCHEMA, "entries": [{
        "evidence_id": "evidence-01", "split": "development", "already_seen": True,
        "binding": {"source_lineage_id": contract["provenance"]["source_lineage_id"],
                    "checkpoint_sha256": checkpoint_sha},
        "contract": {"root": "contract", "path": contract_path.name, "sha256": sha(contract_path)},
        "snapshot": None, "training": None,
        "same_play_trace": {"root": "trace", "path": raw_trace.name, "sha256": sha(raw_trace)},
        "window": "window-01",
    }]}
    return E.build_index(spec, roots), roots


def _materialize_frames(pending: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    evidence_refs: dict[str, Any] = {}
    for role, offset in (("case", 0), ("reference", 3)):
        capture = pending["source"]["captures"][role]
        spec_path = Path(capture["v01_specs"][0]["path"])
        require(sha(spec_path) == capture["v01_specs"][0]["sha256"], "V01 spec binding")
        role_root = output / "evidence" / role
        contract_path = V01.prepare(spec_path, role_root)
        raw_ref = capture["proofs"]["raw_trace"]
        raw_trace = Path(raw_ref["path"]).resolve(strict=True)
        require(sha(raw_trace) == raw_ref["sha256"], "raw trace binding")
        checkpoint_sha = pending["source"]["training"][role]["checkpoint"]["sha256"]
        index, roots = _evidence_index(contract_path, raw_trace, checkpoint_sha)
        index_sha = E.index_sha256(index)
        contract = V01.verify(contract_path)
        for selected, source_index in enumerate((0, 2, 5), 1):
            source_id = contract["frames"][source_index]["frame_id"]
            view = E.frame_view(index, "evidence-01", source_id, roots, expected_sha256=index_sha)
            blocks = view["messages"][0]["content"]
            require(isinstance(blocks, list) and len(blocks) == 2
                    and blocks[1].get("type") == "image_url", "frame view blocks")
            frame_id = f"f{offset + selected}"
            timestamp = contract["frames"][source_index]["timestamp_s"]
            chosen.append({"role": role, "frame_id": frame_id, "source_frame_id": source_id,
                           "timestamp_s": timestamp, "label": {"type": "text", "text": json.dumps(
                               {"frame_id": frame_id, "timestamp_s": timestamp},
                               sort_keys=True)}, "image": blocks[1]})
        evidence_refs[role] = {"contract": ref(contract_path), "index_sha256": index_sha,
                               "raw_trace": ref(raw_trace)}
    require([row["frame_id"] for row in chosen] == [f"f{i}" for i in range(1, 7)]
            and [row["role"] for row in chosen] == ["case"] * 3 + ["reference"] * 3,
            "role/frame mapping")
    return chosen, evidence_refs


def _prepare_pair(prepared_path: Path, output: Path) -> dict[str, Any]:
    source = read(prepared_path.resolve(strict=True))
    require(source.get("schema") == "fresh_frame_decision_prepared_v1"
            and source.get("status") == "ready_for_cpu_provider_inputs_api_not_authorized"
            and source.get("approval_granted") is False
            and source.get("api_authorized") is False, "local prepared input")
    source_context = A2.checked(source["context"])
    require(source_context["goal"] == PAIR_GOAL, "explicit common frame-role goal")
    source_messages = CALL._messages(source_context, source["messages"])
    extra = source_context["extra_facts"]
    frame_facts = {key: value for key, value in extra.items() if key.startswith(PAIRED.FRAME_PREFIX)}
    common_extra = {key: value for key, value in extra.items() if not key.startswith(PAIRED.FRAME_PREFIX)}
    require(sorted(frame_facts) == [f"vision:frame:f{i}" for i in range(1, 7)]
            and sum(key.startswith(PAIRED.PLAY_PREFIX) for key in common_extra) == 54,
            "registered six frames/common facts")
    only_context = A2.prepare(source_context["snapshot"], goal=PAIR_GOAL,
                              evidence=source_context["evidence"], revealed=source_context["revealed"],
                              extra_facts=common_extra, decisions_left=source_context["decisions_left"])
    plus_context = A2.prepare(source_context["snapshot"], goal=PAIR_GOAL,
                              evidence=source_context["evidence"], revealed=source_context["revealed"],
                              extra_facts=extra, decisions_left=source_context["decisions_left"])
    only_messages = A2.messages(only_context)
    blocks = source_messages[1]["content"]
    require(isinstance(blocks, list) and len(blocks) == 13, "six-frame message blocks")
    plus_messages = A2.messages(plus_context)
    plus_messages[1]["content"] = [{"type": "text", "text": plus_messages[1]["content"]},
                                   *deepcopy(blocks[1:])]
    only = PAIRED._arm("logs_only", only_context, only_messages)
    plus = PAIRED._arm("logs_plus_frames", plus_context, plus_messages)
    require(PAIRED._context_common(only["context"]) == PAIRED._context_common(plus["context"]),
            "paired contexts differ outside raw-frame exposure")
    require(only["context"]["goal"] == plus["context"]["goal"] == PAIR_GOAL,
            "role mapping differs across modes")
    require(only["context"]["snapshot"] == plus["context"]["snapshot"]
            and only["context"]["model_view"]["eligible_probes"]
            == plus["context"]["model_view"]["eligible_probes"], "paired snapshot/probes differ")
    require(PAIRED._without_response_format(only["params"])
            == PAIRED._without_response_format(plus["params"]), "paired base params differ")
    output.mkdir(parents=True, exist_ok=False)
    provider_refs = {}
    for arm in (only, plus):
        provider_refs[arm["name"]] = PAIRED.write_new(output / f"{arm['name']}.provider.json", {
            "schema": PAIRED.SCHEMA + "_provider_input", "arm": arm["name"],
            "context": arm["context"], "messages": arm["messages"], "params": arm["params"],
            "approval_granted": False, "api_authorized": False})
    differences = {
        "allowed_only": ["context.context_digest", "context.extra_facts[vision:frame:*]",
                         "context.model_view.evidence_facts[vision:frame:*]",
                         "messages.user.image_blocks", "messages.user.frame_label_blocks",
                         "params.response_format.fact_ids"],
        "removed_raw_frame_fact_ids": sorted(frame_facts),
        "common_telemetry_fact_count": 0,
        "common_same_play_numeric_fact_count": 54,
        "snapshot_same": True, "goal_same": True, "explicit_frame_role_mapping_same": True,
        "eligible_probes_same": True, "available_actions_same": True,
        "base_model_params_same": True,
    }
    pair = {"schema": PAIRED.SCHEMA + "_pair",
            "status": "prepared_registered_cpu_inputs_not_model_result",
            "source_prepared": {"sha256": sha(prepared_path)},
            "arms": {arm["name"]: {"provider_input": provider_refs[arm["name"]],
                                     "shape": PAIRED._redacted_message_shape(arm["messages"]),
                                     "context_sha256": digest(arm["context"]),
                                     "params_sha256": digest(arm["params"])} for arm in (only, plus)},
            "differences": differences,
            "frame_role_mapping": {"case": ["f1", "f2", "f3"],
                                   "reference": ["f4", "f5", "f6"]},
            "binding_sha256": digest({"source": sha(prepared_path), "arms": provider_refs,
                                      "differences": differences, "goal": PAIR_GOAL}),
            "labels_exposed": False, "approval_granted": False, "api_authorized": False,
            "model_calls": 0, "compare_compatibility": "PAIRED.compare_operationally_inconclusive"}
    PAIRED.write_new(output / "pair.json", pair)
    return pair


def prepare(pending_path: Path, approval_path: Path, input_spec: Path, lab: Path,
            state_path: Path, protocol_path: Path, execution_anchor: str, output: Path,
            *, allow_fixture: bool = False, materializer: Any = None,
            input_inspector: Any = None) -> dict[str, Any]:
    pending_path = pending_path.resolve(strict=True)
    pending = read(pending_path)
    require(pending.get("schema") == INPUT.PENDING_SCHEMA
            and pending.get("status") == "local_prepared_pending_main_approval"
            and pending.get("training_fact_count") == 48
            and pending.get("same_play_numeric_fact_count") == 54
            and pending.get("planned_image_count") == 6
            and all(pending.get(key) is False for key in ("ready_for_registered_calls",
                                                          "model_use_authorized", "capture_approved",
                                                          "heldout", "recovery", "video_benefit")),
            "pending input boundary")
    input_protocol_path = Path(pending["source"]["input_protocol"]["path"]).resolve(strict=True)
    require(sha(input_protocol_path) == pending["source"]["input_protocol"]["sha256"],
            "pending input protocol")
    protocol = read(protocol_path) if allow_fixture else verify_protocol(
        protocol_path, input_protocol_path, execution_anchor)
    require(protocol.get("schema") == SCHEMA, "provider protocol")
    require(protocol.get("frame_selection") == FRAME_SELECTION
            and protocol.get("projection") == PROJECTION
            and protocol.get("authority") == AUTHORITY, "provider protocol policy")
    _approval(approval_path.resolve(strict=True), pending_path, pending, protocol)
    inspector = input_inspector or INPUT.inspect
    recomputed = inspector(input_spec.resolve(strict=True), lab.resolve(strict=True),
                           execution_anchor=protocol["input_protocol"]["execution_anchor"],
                           allow_fixture=allow_fixture)
    require(recomputed == pending and pending["binding_sha256"] == INPUT.digest(pending["source"]),
            "pending/current input evidence mismatch")
    packet = _packet(state_path.resolve(strict=True), pending)
    output = output.resolve()
    ledger_path = output / "history.jsonl"
    snapshot = C.V.build_snapshot(packet, f"local_pair_s{packet['epoch']}", "model_adaptive",
                                  C.V.file_digest(C.V.PROTOCOL), ledger_path)
    output.mkdir(parents=True, exist_ok=False)
    ledger_path.write_bytes(b"")
    chosen, evidence_refs = (materializer or _materialize_frames)(pending, output)
    require([row.get("frame_id") for row in chosen] == [f"f{i}" for i in range(1, 7)]
            and [row.get("role") for row in chosen] == ["case"] * 3 + ["reference"] * 3,
            "role/frame mapping")
    play = pending["source"]["same_play_facts"]
    require(len(play) == 54 and all(key.startswith("vision:play:") for key in play),
            "same-Play common facts")
    frame_facts = {f"vision:frame:{row['frame_id']}": {
        "evidence_type": "raw_frame", "timestamp_s": row["timestamp_s"],
        "source_group": "visual_source_1"} for row in chosen}
    context = A2.prepare(snapshot, goal=PAIR_GOAL,
                         extra_facts={**play, **frame_facts}, decisions_left=3)
    messages = A2.messages(context)
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
                               *[block for row in chosen for block in (row["label"], row["image"])] ]
    messages = CALL._messages(context, messages)
    params = CALL.params_for(context)
    require(params == CALL.params_for_messages(messages), "provider params/messages")
    fresh = {"schema": "fresh_frame_decision_prepared_v1",
             "status": "ready_for_cpu_provider_inputs_api_not_authorized",
             "context": context, "messages": messages, "params": params,
             "approval_granted": False, "api_authorized": False, "gpu_authorized": False,
             "source_preparation_approval": ref(approval_path), "source_pending": ref(pending_path)}
    write_new(output / "fresh-prepared.json", fresh)
    pair = _prepare_pair(output / "fresh-prepared.json", output / "paired")
    result = {"schema": PREPARED_SCHEMA, "status": "prepared_cpu_inputs_api_not_authorized",
              "pending": ref(pending_path), "approval": ref(approval_path),
              "protocol": ref(protocol_path), "state": ref(state_path),
              "evidence": evidence_refs, "pair": ref(output / "paired/pair.json"),
              "pair_binding_sha256": pair["binding_sha256"], "training_fact_count": 48,
              "same_play_numeric_fact_count": 54, "runtime_facts_excluded_equally": 16,
              "image_counts": {"logs_only": 0, "logs_plus_frames": 6},
              "model_calls": 0, "api_authorized": False, "gpu_authorized": False,
              "probe_authorized": False, "correction_authorized": False,
              "heldout": False, "recovery": False, "video_benefit": False}
    write_new(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-protocol")
    p.add_argument("--input-protocol", type=Path, required=True)
    p.add_argument("--input-execution-anchor", required=True)
    p.add_argument("--source-anchor")
    p.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("prepare")
    for name in ("pending", "approval", "input-spec", "lab", "state", "protocol", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--execution-anchor", required=True)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        result = prepare_protocol(args.input_protocol, args.input_execution_anchor, args.output,
                                  source_anchor=args.source_anchor)
    else:
        result = prepare(args.pending, args.approval, args.input_spec, args.lab, args.state, args.protocol,
                         args.execution_anchor, args.output)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
