"""실제 V01/V03·조사 고리를 사용하는 CPU 접속 회귀. 정책·모델 응답은 구조 fixture다."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "tests")]
import agentic_investigation_evidence_v1 as B
from test_vision_selection_contract_v1 import setup_case

E, A = B.E, B.A


def ref(root: Path, path: Path) -> dict:
    return {"root": "fixture", "path": path.relative_to(root).as_posix(), "sha256": E.V01.sha256(path)}


def write(path: Path, value: dict) -> None:
    path.write_bytes(E.V01.canonical(value))


def response(action: str, target: str | None, facts: list[str] | None = None) -> str:
    return json.dumps({"action": action, "target": target, "fact_ids": facts or ["config:action_scale"]})


@pytest.fixture
def case(tmp_path: Path) -> dict:
    contract, binding, old = setup_case(tmp_path)
    roots = {"fixture": tmp_path}
    case_cp = contract.parent / "artifacts/checkpoint.bin"
    reference_cp = tmp_path / "reference.pt"
    reference_cp.write_bytes(b"cpu-reference-checkpoint")
    facts, bindings, training = {"config:action_scale": 0.25}, {}, {}
    for role, checkpoint in (("case", case_cp), ("reference", reference_cp)):
        summary = {metric: {"n": 3, "mean_last_20pct": 0.5} for metric in A.C.V.TELEMETRY_FIELDS}
        telemetry = tmp_path / f"{role}-telemetry.json"
        write(telemetry, {"summary": summary, "series": {metric: [0.5] * 3 for metric in summary}})
        artifact = tmp_path / f"{role}-artifact.json"
        write(artifact, {"arm": "F1", "fixed_eval_label": "healthy", "checkpoint": {"sha256": E.V01.sha256(checkpoint)},
                         "telemetry": {"sha256": E.V01.sha256(telemetry), "summary": summary}})
        training[role] = {"artifact": ref(tmp_path, artifact), "telemetry": ref(tmp_path, telemetry),
                          "checkpoint": ref(tmp_path, checkpoint)}
        bindings[role] = {"checkpoint_sha256": E.V01.sha256(checkpoint),
                          "telemetry_sha256": E.V01.sha256(telemetry)}
        facts.update({f"telemetry:{role}:{metric}": 0.5 for metric in summary})

    def snapshot(facts: dict, run_id: str) -> dict:
        packet = A.C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None, facts=facts,
                                   input_bindings=bindings, execution_budget_s=1200)
        return A.C.V.build_snapshot(packet, run_id, "model_adaptive", old["inputs"]["protocol_sha256"],
                                    tmp_path / f"{run_id}.jsonl")

    full = snapshot(facts, "index_context")
    source = snapshot({"config:action_scale": 0.25}, "investigation_context")
    snapshot_path = tmp_path / "index-snapshot.json"
    write(snapshot_path, full)
    entry = {"evidence_id": "evidence-01", "split": "development", "already_seen": True,
             "binding": {"source_lineage_id": binding["source_lineage_id"],
                         "checkpoint_sha256": bindings["case"]["checkpoint_sha256"]},
             "contract": ref(tmp_path, contract), "snapshot": ref(tmp_path, snapshot_path),
             "training": training, "same_play_trace": None, "window": None}
    index = E.build_index({"schema": E.SPEC_SCHEMA, "entries": [entry]}, roots)
    aliases = {"evidence:training-01": {"kind": "training", "evidence_id": "evidence-01", "frame_ids": []},
               "evidence:clip-01": {"kind": "frames", "evidence_id": "evidence-01", "frame_ids": ["f01", "f02"]}}
    registration = B.register(index, roots, source, expected_index_sha256=E.index_sha256(index),
                              source_lineage_id=binding["source_lineage_id"], aliases=aliases)
    return {"root": tmp_path, "roots": roots, "index": index, "snapshot": source, "full_snapshot": full,
            "registration": registration, "entry": entry}


def adapter(case: dict) -> B.EvidenceAdapter:
    return B.EvidenceAdapter(case["index"], case["roots"], case["registration"],
                             expected_registration_sha256=E.index_sha256(case["registration"]))


def test_real_contract_lookup_images_and_next_probe_keep_one_action_system(case: dict) -> None:
    bridge = adapter(case)
    loop = bridge.start(case["snapshot"], goal="명령 추종 실패 원인 조사", allow_fixture=True)
    initial = bridge.messages(loop)
    assert initial["image_count"] == 0
    assert initial["messages"][0] == {"role": "system", "content": A.SYSTEM}
    assert initial["messages"][1]["content"] == [{"type": "text", "text": A.messages(initial["context"])[1]["content"]}]
    loop.propose(response("read_evidence", "evidence:training-01"))
    read = bridge.read_evidence(loop, "evidence:training-01")
    assert len(read["facts"]) == 48 and read["image_count"] == 0
    trained = bridge.messages(loop, visual_context=read["visual_context"])
    assert trained["image_count"] == 0
    loop.propose(response("inspect_frames", "evidence:clip-01"))
    inspected = bridge.inspect_frames(loop, "evidence:clip-01", visual_context=read["visual_context"])
    assert inspected["image_count"] == 2 and inspected["approval_granted"] is False
    assert all("f03" not in key for key in inspected["facts"])
    subsequent = bridge.messages(loop, visual_context=inspected["visual_context"])
    assert subsequent["image_count"] == 2
    assert [frame["frame_id"] for frame in subsequent["frame_refs"]] == ["f01", "f02"]
    assert subsequent["messages_sha256"] == E.index_sha256({"messages": subsequent["messages"]})
    assert E.SELECT.SYSTEM not in [message["content"] for message in subsequent["messages"] if message["role"] == "system"]
    text = E.TRANSFER.MATERIAL._model_text(subsequent["messages"])
    assert all(hidden not in text for hidden in ["healthy", "recovered", "lineage-contract-1", "source_role", "checkpoint_sha256", str(case["root"]).lower()])
    content = response("run_probe", "P_reward", ["vision:frame:f01"])
    assert A.assess(content, subsequent["context"])["status"] == "valid"
    assert loop.propose(content)["status"] == "valid"
    request = A.C.request_spec(loop.snapshot, loop.probe_choice())
    assert request["probe"] == "P_reward" and request["approval_granted"] is False


def test_no_new_training_facts_are_not_advertised(case: dict) -> None:
    registration = B.register(case["index"], case["roots"], case["full_snapshot"],
        expected_index_sha256=E.index_sha256(case["index"]),
        source_lineage_id=case["registration"]["session"]["source_lineage_id"], aliases=case["registration"]["aliases"])
    bridge = B.EvidenceAdapter(case["index"], case["roots"], registration,
                               expected_registration_sha256=E.index_sha256(registration))
    loop = bridge.start(case["full_snapshot"], goal="조사")
    message = bridge.messages(loop)
    assert message["context"]["model_view"]["available_actions"]["read_evidence"] == []
    assert message["image_count"] == 0


@pytest.mark.parametrize("action,target", [("read_evidence", "evidence:training-01"),
                                            ("inspect_frames", "evidence:clip-01")])
def test_response_requires_exact_pending_action_and_target(case: dict, action: str, target: str) -> None:
    bridge = adapter(case)
    loop = bridge.start(case["snapshot"], goal="조사")
    call = bridge.read_evidence if action == "read_evidence" else bridge.inspect_frames
    with pytest.raises(ValueError, match="pending"):
        call(loop, target)
    loop.propose(response(action, target))
    before = loop.audit()
    with pytest.raises(ValueError, match="pending action/target"):
        call(loop, "evidence:clip-99")
    wrong = bridge.inspect_frames if action == "read_evidence" else bridge.read_evidence
    with pytest.raises(ValueError, match="pending action/target"):
        wrong(loop, target)
    assert loop.audit() == before


def test_registered_alias_payload_and_cross_checkpoint_telemetry_lineage_are_checked(case: dict) -> None:
    registration = deepcopy(case["registration"])
    registration["aliases"]["evidence:clip-01"]["frame_ids"] = ["f03"]
    with pytest.raises(ValueError, match="alias SHA"):
        B.EvidenceAdapter(case["index"], case["roots"], registration,
                          expected_registration_sha256=E.index_sha256(case["registration"]))
    for role, field in [("case", "checkpoint_sha256"), ("reference", "checkpoint_sha256"),
                         ("case", "telemetry_sha256"), ("reference", "telemetry_sha256")]:
        registration = deepcopy(case["registration"])
        registration["session"]["input_bindings"][role][field] = "1" * 64
        with pytest.raises(ValueError, match="mismatch"):
            B.EvidenceAdapter(case["index"], case["roots"], registration,
                              expected_registration_sha256=E.index_sha256(registration))
    registration = deepcopy(case["registration"])
    registration["session"]["source_lineage_id"] = "another-source"
    with pytest.raises(ValueError, match="lineage mismatch"):
        B.EvidenceAdapter(case["index"], case["roots"], registration,
                          expected_registration_sha256=E.index_sha256(registration))


def test_stale_source_missing_and_path_escape_fail_without_consuming_pending(case: dict) -> None:
    bridge = adapter(case)
    loop = bridge.start(case["snapshot"], goal="조사")
    loop.propose(response("inspect_frames", "evidence:clip-01"))
    original = loop.audit()
    path = E._ref(case["entry"]["training"]["case"]["telemetry"], case["roots"])
    raw = path.read_bytes()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA"):
        bridge.inspect_frames(loop, "evidence:clip-01")
    assert loop.audit() == original
    path.write_bytes(raw)
    image = case["root"] / "contract/artifacts/frames/frame_01.jpg"
    image.unlink()
    with pytest.raises(FileNotFoundError):
        bridge.inspect_frames(loop, "evidence:clip-01")
    assert loop.audit() == original
    index = deepcopy(case["index"])
    index["spec"]["entries"][0]["contract"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="registered index SHA"):
        E.verify_index(index, case["roots"], expected_sha256=E.index_sha256(case["index"]))


def test_revealed_context_and_frame_digest_tampering_are_rejected(case: dict) -> None:
    bridge = adapter(case)
    loop = bridge.start(case["snapshot"], goal="조사")
    loop.propose(response("inspect_frames", "evidence:clip-01"))
    receipt = bridge.inspect_frames(loop, "evidence:clip-01")
    before = loop.audit()
    with pytest.raises(ValueError, match="context"):
        bridge.messages(loop)
    for change in ("facts", "frame_refs", "payload_sha256"):
        visual = deepcopy(receipt["visual_context"])
        if change == "facts":
            visual["records"][0]["facts"]["vision:frame:f01"]["timestamp_s"] += 1
        elif change == "frame_refs":
            visual["records"][0]["frame_refs"][0]["content_sha256"] = "0" * 64
        else:
            visual["records"][0][change] = "0" * 64
        with pytest.raises(ValueError):
            bridge.messages(loop, visual_context=visual)
    assert loop.audit() == before
    assert bridge.messages(loop, visual_context=receipt["visual_context"])["image_count"] == 2
    assert loop.propose(response("inspect_frames", "evidence:clip-01"))["status"] == "invalid"


def test_duplicate_alias_with_no_new_facts_disappears_and_old_context_cannot_cross_session(case: dict) -> None:
    registration = deepcopy(case["registration"])
    registration["aliases"]["evidence:training-02"] = deepcopy(registration["aliases"]["evidence:training-01"])
    bridge = B.EvidenceAdapter(case["index"], case["roots"], registration,
                               expected_registration_sha256=E.index_sha256(registration))
    loop = bridge.start(case["snapshot"], goal="조사")
    loop.propose(response("read_evidence", "evidence:training-01"))
    receipt = bridge.read_evidence(loop, "evidence:training-01")
    after = bridge.messages(loop, visual_context=receipt["visual_context"])
    assert after["context"]["model_view"]["available_actions"]["read_evidence"] == []
    old = deepcopy(loop.snapshot)
    loop.snapshot = case["full_snapshot"]
    with pytest.raises(ValueError, match="session/epoch/checkpoint"):
        bridge.messages(loop, visual_context=receipt["visual_context"])
    loop.snapshot = old


def test_existing_fact_mismatch_and_invalid_frame_registration_are_rejected(case: dict) -> None:
    registration = deepcopy(case["registration"])
    registration["aliases"]["evidence:clip-01"]["frame_ids"] = ["f99"]
    with pytest.raises(ValueError, match="registered frame"):
        B.EvidenceAdapter(case["index"], case["roots"], registration,
                          expected_registration_sha256=E.index_sha256(registration))
    bridge = adapter(case)
    loop = bridge.start(case["snapshot"], goal="조사")
    loop.extra_facts["telemetry:case:Train/mean_reward"] = 1.5
    with pytest.raises(ValueError, match="loop facts"):
        bridge.messages(loop)
