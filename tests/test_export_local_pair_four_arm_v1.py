# pyright: reportMissingImports=false
"""local pair 네 방식 공개 exporter의 재계산과 변조 거부를 검증한다."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import export_local_pair_four_arm_v1 as E
import test_local_pair_four_arm_v1 as DRIVER


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path, str, dict]:
    anchor = "a" * 40
    protocol_path = tmp_path / "protocol.json"
    pairs = [
        {"id": "source2430", "seed": 2430, "pair_sha256": "1" * 64,
         "prepared_result_sha256": "2" * 64, "case_checkpoint_sha256": "3" * 64,
         "reference_checkpoint_sha256": "4" * 64, "truth_subtypes": ["reward"]},
        {"id": "source2431", "seed": 2431, "pair_sha256": "5" * 64,
         "prepared_result_sha256": "6" * 64, "case_checkpoint_sha256": "7" * 64,
         "reference_checkpoint_sha256": "8" * 64, "truth_subtypes": ["reward"]},
    ]
    protocol = {"schema": "local_pair_four_arm_protocol_v1", "fixture": True,
                "pairs": pairs, "normal_controls": ["c01", "c05"], "sources_lf": {},
                "sdk_role_executions_upper_bound": 48}
    _write(protocol_path, protocol)
    batch_root = tmp_path / "batch"
    rows = []
    correct = {("source2430", "registry_next"), ("source2431", "registry_next"),
               ("source2430", "logs_only"), ("source2431", "logs_plus_frames")}
    wrong = {(source, "fixed_eligible") for source in ("source2430", "source2431")}
    calls: dict[str, dict] = {}
    for pair in pairs:
        for arm in E.M.ARMS:
            result_dir = batch_root / pair["id"] / arm
            decision = result_dir / "step01"
            snapshot = {"source": pair["id"], "arm": arm, "view_digest": "a" * 64}
            context = {"snapshot": snapshot, "context_digest": "9" * 64}
            choice = {"action": "continue", "probe": "P_reward", "selected_by": arm,
                      "fact_ids": ["vision:frame:x"] if arm == "logs_plus_frames" else []}
            _write(decision / "context.json", context)
            _write(decision / "choice.json", choice)
            host = decision / "probe/result.json"
            _write(host, {"probe": "P_reward", "outcome": "abnormal",
                          "owned_child_wall_s": 1.25})
            model_calls = 0
            reference = None
            if arm in E.M.MODEL_ARMS:
                model_calls = 1
                request = {"request_id": f"{pair['id']}_{arm}_{arm}_1",
                           "run_id": f"{pair['id']}_{arm}",
                           "arm": arm, "step": 1, "context_digest": "9" * 64,
                           "source_binding": {"pair_sha256": pair["pair_sha256"],
                              "case_checkpoint_sha256": pair["case_checkpoint_sha256"],
                              "reference_checkpoint_sha256": pair["reference_checkpoint_sha256"]}}
                _write(decision / "request.json", request)
                call = tmp_path / "calls" / f"{pair['id']}-{arm}"
                _write(call / "receipt.json", {"done": True, "request_id": request["request_id"]})
                reference = _ref(call / "receipt.json")
                reference["path"] = str(call.resolve())
                review = {"status": "valid", "decision": {"action": "run_probe"},
                          "content": "ok"}
                _write(decision / "model-review.json", {"transport": reference, "review": review})
                verified = {"request": request, "call_ref": reference, "status": "completed",
                            "live": True, "content": "ok", "usage": {"total_tokens": 3},
                            "wall_s": 0.5}
                _write(call / "verified.json", verified)
                calls[str(call.resolve())] = verified
            if (pair["id"], arm) in correct:
                end_state, subtype = "confirmed", "reward"
            elif (pair["id"], arm) in wrong:
                end_state, subtype = "confirmed", "physics"
            else:
                end_state, subtype = "HOLD", None
            raw = {"schema": "local_pair_four_arm_result_v1", "scope": "actual_backend",
                   "arm": arm, "run_id": f"{pair['id']}_{arm}",
                   "source": {"sha256": pair["prepared_result_sha256"]},
                   "pair": {"sha256": pair["pair_sha256"]}, "repair": False,
                   "recovery": False, "heldout": False, "end_state": end_state,
                   "confirmed_subtype": subtype, "hold_reason": "budget" if subtype is None else None,
                   "model_calls": model_calls, "semantic_invalid": 0, "infra_unknown": 0,
                   "provider_unknown": 0, "frame_fact_citations": len(choice["fact_ids"]),
                   "steps": [{"probe": "P_reward", "outcome": "abnormal",
                              "probe_result": _ref(host), "selected_by": arm,
                              "fact_ids": choice["fact_ids"], "before_view_digest": "a" * 64,
                              "model_call": reference}]}
            _write(result_dir / "result.json", raw)
            rows.append({"id": pair["id"], "arm": arm, "end_state": end_state,
                         "confirmed_subtype": subtype, "correct_confirmation": subtype == "reward",
                         "hold_reason": raw["hold_reason"], "paired_probes": 1,
                         "model_calls": model_calls, "result": _ref(result_dir / "result.json")})
    controls = [{"id": case, "arm": arm, "end_state": "HOLD",
                 "reason": "healthy_host_gate", "model_calls": 0, "paired_probes": 0,
                 "model_normal_recognition": False}
                for case in ("c01", "c05") for arm in E.M.ARMS]
    batch = {"schema": "local_pair_four_arm_batch_v1", "status": "complete", "rows": rows,
             "normal_host_controls": controls, "paired_intents": 8, "model_calls": 4,
             "scope": "development_seen_sources", "independent_source_seeds": 2,
             "heldout": False, "repair": False, "recovery": False,
             "llm_superiority": False, "video_advantage_confirmed": False}
    _write(batch_root / "result.json", batch)

    def verify_protocol(_path, _anchor):
        assert _path == protocol_path and _anchor == anchor
        return protocol

    def verify_call(path):
        value = dict(calls[str(path.resolve())])
        receipt = path / "receipt.json"
        if _sha(receipt) != value["call_ref"]["sha256"]:
            raise ValueError("receipt SHA")
        return value

    verifiers = {"protocol": verify_protocol, "context": lambda value: value,
                 "assess": lambda content, _context: {"status": "valid",
                     "decision": {"action": "run_probe"}, "content": content},
                 "choice": lambda context, _review: {"action": "continue", "probe": "P_reward",
                     "selected_by": context["snapshot"]["arm"],
                     "fact_ids": ["vision:frame:x"]
                     if context["snapshot"]["arm"] == "logs_plus_frames" else []},
                 "pair": lambda path, _snapshot, _choice: _read_pair(path),
                 "call": verify_call}
    return batch_root / "result.json", protocol_path, anchor, verifiers


def _read_pair(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exports_four_public_files_with_two_seed_paired_statistics(tmp_path):
    batch, protocol, anchor, verifiers = _fixture(tmp_path)
    out = tmp_path / "public"
    result = E.export(batch, protocol, anchor, out, allow_fixture=True, verifiers=verifiers)
    assert sorted(path.name for path in out.iterdir()) == [
        "cases.json", "manifest.json", "provenance.json", "summary.json"]
    assert result["fault_rows"] == 8 and result["independent_source_seeds"] == 2
    assert result["arms"]["logs_only"]["correct_confirmation"]["successes"] == 1
    assert result["arms"]["logs_plus_frames"]["correct_confirmation"]["successes"] == 1
    assert result["paired_model_contrast"] == {
        "logs_only_better": 1, "logs_plus_frames_better": 1,
        "discordant": 2, "exact_two_sided_p": 1.0}
    assert result["claims"] == {"heldout": False, "llm_superiority": False,
                                 "video_advantage_confirmed": False, "recovery": False,
                                 "repair": False, "default_mode_adopted": False}
    assert not E.PRIVATE_PATTERN.search("\n".join(path.read_text(encoding="utf-8")
                                                   for path in out.iterdir()))


def test_rejects_call_receipt_tamper_and_does_not_overwrite_output(tmp_path):
    batch, protocol, anchor, verifiers = _fixture(tmp_path)
    first = tmp_path / "first"
    E.export(batch, protocol, anchor, first, allow_fixture=True, verifiers=verifiers)
    with pytest.raises(FileExistsError):
        E.export(batch, protocol, anchor, first, allow_fixture=True, verifiers=verifiers)
    call = next((tmp_path / "calls").iterdir())
    (call / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA"):
        E.export(batch, protocol, anchor, tmp_path / "tampered", allow_fixture=True,
                 verifiers=verifiers)


def test_rejects_incomplete_batch_before_following_result_refs(tmp_path):
    batch, protocol, anchor, verifiers = _fixture(tmp_path)
    value = json.loads(batch.read_text(encoding="utf-8"))
    value["status"] = "running"
    _write(batch, value)
    with pytest.raises(ValueError, match="완료된"):
        E.export(batch, protocol, anchor, tmp_path / "public", allow_fixture=True,
                 verifiers=verifiers)


def test_history_prefix_replays_two_step_driver_contexts_without_mutating_full_ledger(tmp_path):
    data = DRIVER.source(tmp_path)
    observed = []

    def model(request):
        content = request["messages"][1]["content"]
        view = json.loads(content)
        observed.append(view["observations"])
        target = "P_episode" if not view["observations"] else "P_reward"
        return {"status": "completed", "content": json.dumps({
            "action": "run_probe", "target": target,
            "fact_ids": [next(iter(view["evidence_facts"]))]})}

    result = DRIVER.run(tmp_path, data, "logs_only", complete=model)
    assert [row["probe"] for row in result["steps"]] == ["P_episode", "P_reward"]
    ledger = Path(json.loads((tmp_path / "logs_only/step01/context.json").read_text(
        encoding="utf-8"))["snapshot"]["inputs"]["ledger_path"])
    full_before = ledger.read_bytes()
    for number in (1, 2):
        decision = tmp_path / f"logs_only/step{number:02d}"
        context = E._read(decision / "context.json")
        with pytest.raises(ValueError, match="오래된 snapshot"):
            E.M.A.checked(context)
        with E._history_prefix_scope(context["snapshot"]):
            checked = E.M.A.checked(context)
            wrapper = E._read(decision / "model-review.json")
            review = E.M.A.assess(wrapper["review"]["raw"], checked) \
                if "raw" in wrapper["review"] else wrapper["review"]
            if review is wrapper["review"]:
                # 실제 driver review는 이미 A.assess의 정규형이며 choice 재계산이 핵심이다.
                review = wrapper["review"]
            assert E.M.A.probe_choice(checked, review) == E._read(decision / "choice.json")
        assert ledger.read_bytes() == full_before


def test_raw_owned_role_wall_requires_receipt_plan_and_owned_result_identity(tmp_path):
    host = {"receipts": {}, "plans": {}}
    expected_wall = 0.0
    for index, role in enumerate(E.HOST.ROLES, 1):
        stage = tmp_path / role
        plan = stage / "plan.json"
        owned = stage / "owned-result.json"
        receipt = stage / "probe_receipt.json"
        _write(plan, {"role": role})
        execution = {"job_assigned_before_release": True, "natural_exit": True,
                     "timed_out": False, "error_type": None, "exit_code": 0,
                     "cleanup_verified": True,
                     "owned_process_identities": {str(100 + index): 200 + index},
                     "wall_s": float(index),
                     "wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification"}
        _write(owned, execution)
        _write(receipt, {"plan": _ref(plan), "execution": execution,
                         "raw_refs": {"owned_result": _ref(owned)}})
        host["plans"][role] = _ref(plan)
        host["receipts"][role] = _ref(receipt)
        expected_wall += index
    host_path = tmp_path / "host-result.json"
    _write(host_path, host)
    rows = E._raw_owned_executions(host_path)
    assert len(rows) == 2 and sum(row["wall_s"] for row in rows) == expected_wall
    changed = json.loads((tmp_path / "case/owned-result.json").read_text(encoding="utf-8"))
    changed["cleanup_verified"] = False
    _write(tmp_path / "case/owned-result-changed.json", changed)
    (tmp_path / "case/owned-result.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        E._raw_owned_executions(host_path)

