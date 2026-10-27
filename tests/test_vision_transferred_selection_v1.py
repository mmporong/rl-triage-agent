from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as METHOD
import vision_selection_sources_v1 as OLD_INPUTS
import vision_transferred_selection_v1 as M


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _fixture(base: Path) -> tuple[Path, dict[int, dict]]:
    pairs = []
    by_seed = {}
    roles = (("NONE", "none"), ("F1", "f1"))
    for seed in (2220, 2221):
        protocol = _write(base / f"protocol-{seed}.json", {"seed": seed})
        training = _write(base / f"training-{seed}.json", {"seed": seed})
        capture_policies = []
        policies = []
        for source_role, suffix in roles:
            logical = f"source-s{seed}-{suffix}"
            checkpoint = hashlib.sha256(f"checkpoint-{logical}".encode()).hexdigest()
            telemetry = hashlib.sha256(f"telemetry-{logical}".encode()).hexdigest()
            trace = _write(base / logical / "trace.json", {"logical_id": logical})
            video = _write(base / logical / "same_play.mp4", f"video-{logical}".encode())
            capture_policies.append({"logical_id": logical, "source_inputs": {
                "raw_trace": _absref(trace), "video": _absref(video)}})
            snapshot = _write(base / logical / "snapshot.json", {"logical_id": logical})
            payloads = []
            lineages = []
            lineage = f"lineage-{logical}"
            for window in (1, 2):
                lineages.append({"window": window, "source_lineage_id": lineage,
                                 "parent_lineage_id": f"parent-{logical}"})
                for view in sorted(METHOD.VIEWS):
                    bundle = {
                        "schema": "vision_selection_bundle_v1", "view_kind": view,
                        "operational_base": {"case_checkpoint_sha256": checkpoint},
                        "source": {"contract_sha256": hashlib.sha256(
                            f"contract-{logical}-{window}".encode()).hexdigest(),
                            "checkpoint_sha256": checkpoint, "video_sha256": _sha(video)},
                    }
                    bundle_path = _write(base / logical / f"w{window}-{view}-bundle.json", bundle)
                    image_count = 0 if view == "text_only" else 6
                    content = [{"type": "text", "text": f"payload {logical} {window} {view}"}]
                    content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}}
                                for i in range(image_count)]
                    messages = [{"role": "system", "content": "select"},
                                {"role": "user", "content": content}]
                    messages_path = _write(base / logical / f"w{window}-{view}-messages.json", messages)
                    payloads.append({"window": window, "view": view, "bundle": _absref(bundle_path),
                                     "messages": _absref(messages_path), "image_count": image_count})
            policies.append({"logical_id": logical, "source_role": source_role,
                             "checkpoint_sha256": checkpoint, "telemetry_sha256": telemetry,
                             "snapshot": _absref(snapshot), "training_sources": {},
                             "nested_lineage": lineages, "payloads": payloads})
        capture = _write(base / f"capture-{seed}.json", {"policies": capture_policies})
        pair = {"seed": seed, "protocol": _absref(protocol), "capture_receipt": _absref(capture),
                "training_receipt": _absref(training), "policies": policies}
        pairs.append(pair)
        by_seed[seed] = pair
    receipt = {
        "schema": "vision_transferred_input_acceptance_v3", "status": "HOLD",
        "cpu_preparation": "PASS", "pairs": pairs, "policy_count": 4, "payload_count": 24,
        "expected_complete_payload_count": 24, "complete_four_policy_set": True,
        "use_scope_attestation": {"scopes": ["registered_native_pull_combination_only",
            "first_probe_comparison_only"], "mechanically_verified": True,
            "behavior_labels_created": False, "behavior_ground_truth_assigned": False,
            "model_use_authorized": False, "used_for_decision": False,
            "final_approval": "leader_pending"},
        "model_api_requests": 0, "gpu_executions": 0, "new_training_actions": 0,
    }
    return _write(base / "acceptance-v3.json", receipt), by_seed


def _acceptor(by_seed: dict[int, dict]):
    def accept(protocol_path, capture_path, training_path, *, seed, verify_raw_rgb=True):
        pair = by_seed[seed]
        assert protocol_path == Path(pair["protocol"]["path"])
        assert capture_path == Path(pair["capture_receipt"]["path"])
        assert training_path == Path(pair["training_receipt"]["path"])
        assert isinstance(verify_raw_rgb, bool)
        return pair
    return accept


def test_materialize_preserves_v05_order_caps_and_unapproved_boundary(tmp_path, monkeypatch):
    owned = ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025/vision_goal/v05_transferred_selection_cpu"
    work = owned / f"pytest-{tmp_path.name}"
    receipt, pairs = _fixture(work / "sources")
    monkeypatch.setattr(OLD_INPUTS, "camera_pose_audit", lambda _path: {
        "schema": OLD_INPUTS.CAMERA_SCHEMA, "camera": "fixture"})
    output = work / "inputs"
    protocol = work / "execution.json"
    try:
        result = M.materialize(ROOT, receipt, output, protocol,
                               execution_id="transferred-selection-cpu-v1",
                               protocol_id="transferred-selection-cpu-v1",
                               accept_pair=_acceptor(pairs), verify_raw_rgb=False)
        assert result["rows"] == 24
        assert result["model_use_authorized"] is False
        body = json.loads(protocol.read_text(encoding="utf-8"))
        assert body["api"] == M.API
        assert [(r["slot_id"], r["view_kind"]) for r in body["rows"]] == [
            (slot["slot_id"], view) for slot in METHOD.design() for view in slot["view_order"]]
        assert [r["source_seed"] for r in body["rows"][:12]] == [2220] * 12
        assert [r["source_seed"] for r in body["rows"][12:]] == [2221] * 12
        locations = json.loads((output / "input_locations.json").read_text(encoding="utf-8"))
        assert locations["use_attestation"] == {
            "model_use_authorized": False, "approval_id": None, "used_for_decision": False}
        checked = M.verify_input_verifier(ROOT, body, output / "input_locations.json",
                                          verify_raw_rgb=False, accept_pair=_acceptor(pairs))
        assert len(checked["rows"]) == 24
        assert [row["call_id"] for row in checked["rows"]] == [f"choice-{i:02d}" for i in range(1, 25)]
        assert all(CALL.prepared(row["capsule"])["reference"]["status"] == "frozen"
                   for row in checked["rows"])
        verified_protocol = M.WORKER.verify_protocol(protocol, M._sha(protocol, lf=True), root=ROOT)
        assert verified_protocol["api"] == M.API
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_prepare_injects_acceptance_verifier(monkeypatch, tmp_path):
    seen = {}

    def fake_prepare(*args, **kwargs):
        seen.update(kwargs)
        return {"plan": {"path": "planned", "sha256": "0" * 64, "status": "planned"}}

    monkeypatch.setattr(M.WORKER, "prepare", fake_prepare)
    result = M.prepare(tmp_path / "p", "a" * 64, "b" * 40, tmp_path / "i",
                       root=tmp_path, verify_raw_rgb=False)
    assert result["plan"]["status"] == "planned"
    assert callable(seen["input_verifier"])


def test_advance_holds_without_new_use_attestation(monkeypatch, tmp_path):
    locations = _write(tmp_path / "locations.json", {
        "use_attestation": {"model_use_authorized": False, "approval_id": None,
                            "used_for_decision": False}})
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider worker must not run")

    monkeypatch.setattr(M.WORKER, "advance", forbidden)
    result, code = M.advance(tmp_path / "p", "a" * 64, "b" * 40, locations, "choice-01",
                             root=tmp_path, verify_raw_rgb=False)
    assert result == {"status": "terminal_hold", "reason": "model_use_approval_pending",
                      "call_id": "choice-01", "provider_calls": 0}
    assert code == M.WORKER.EXIT_CODES["terminal_hold"]
    assert called is False


def test_advance_injects_verifier_after_explicit_use_attestation(monkeypatch, tmp_path):
    locations = _write(tmp_path / "locations.json", {
        "use_attestation": {"model_use_authorized": True, "approval_id": "leader-approval-1",
                            "used_for_decision": False}})
    seen = {}

    def fake_advance(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "choice_ready"}, 0

    monkeypatch.setattr(M.WORKER, "advance", fake_advance)
    result, code = M.advance(tmp_path / "p", "a" * 64, "b" * 40, locations, "choice-01",
                             root=tmp_path, verify_raw_rgb=False)
    assert (result, code) == ({"status": "choice_ready"}, 0)
    assert callable(seen["input_verifier"])


def test_inspect_maps_unknown_pending_to_hold(monkeypatch, tmp_path):
    _write(tmp_path / "p", {})
    _write(tmp_path / "i", {})
    monkeypatch.setattr(M, "prepare", lambda *args, **kwargs: {"plan": {"path": "x"}})
    monkeypatch.setattr(M.CALL, "_plan", lambda _plan: (tmp_path, tmp_path / "stage", {"rows": []}))
    monkeypatch.setattr(M.EXECUTION, "inspect_run", lambda _context: {
        "status": "blocked_pending", "call_id": "choice-03"})
    result = M.inspect(tmp_path / "p", "a" * 64, "b" * 40, tmp_path / "i", root=tmp_path)
    assert result == {"status": "blocked_pending", "call_id": "choice-03", "decision": "HOLD"}


def test_host_attestation_rejects_changed_pair(tmp_path):
    receipt, pairs = _fixture(tmp_path)
    bad = json.loads(json.dumps(pairs[2221]))
    bad["policies"][0]["checkpoint_sha256"] = "f" * 64
    pairs[2221] = bad
    with pytest.raises(ValueError, match="host 재검증"):
        M._verified_acceptance(receipt, verify_raw_rgb=False, accept_pair=_acceptor(pairs))
