"""D4의 공유 source·참조와 정책별 최대 2 epoch 실행 흐름."""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import residual_demo_v2 as RESIDUAL
import selection_comparison_controller_v1 as CONTROL
import selection_comparison_demo_v1 as DEMO
import selection_comparison_execution_v1 as EXECUTION

SEED = 2170
SOURCE_ARM = "COMPOUND"
GPU_CONTRACT = EXECUTION.GPU_CONTRACT
VIEW_PROTOCOL_SHA = CONTROL.V.file_digest(CONTROL.V.PROTOCOL)


def _wall(raw: Mapping[str, Any]) -> float:
    values = []
    for key in ("execution", "checkpoint_execution", "extract_execution"):
        receipt = raw.get(key)
        if not isinstance(receipt, Mapping):
            continue
        found = [float(receipt[name]) for name in ("wall_s", "parent_wall_s", "elapsed_s")
                 if isinstance(receipt.get(name), (int, float)) and not isinstance(receipt.get(name), bool)]
        if found:
            values.append(max(found))
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("stage의 확인 가능한 owned 벽시계 소비가 없다")
    return total


def _verified(stage: Path, kind: str, raw: dict) -> dict:
    seal = raw.get("stage_seal")
    if not isinstance(seal, dict):
        raise ValueError("backend stage seal이 없다")
    EXECUTION.verify_stage_seal(stage, kind, raw, seal)
    return raw


def _backend_raw(artifact: dict) -> dict:
    return artifact["backend"]["artifact"]


def _snapshot(folder: Path, tag: str, policy: str, epoch: int, current: dict, reference: dict,
              applied: list[str], prior: dict | None) -> dict:
    packet = RESIDUAL.diagnosis_packet(
        _backend_raw(current), _backend_raw(reference), epoch, applied, prior)
    if epoch == 1:
        parent = None
    else:
        if not isinstance(prior, dict):
            raise TypeError("epoch2의 검증된 이전 receipt가 없다")
        parent = {"run_id": f"{tag}_{policy}", "policy": policy, "epoch": 1, **deepcopy(prior)}
    return CONTROL.V.build_snapshot(
        packet, f"{tag}_{policy}", policy, VIEW_PROTOCOL_SHA,
        folder / "arms" / policy / f"epoch{epoch}" / "probes.ledger.jsonl", parent,
    )


def _call_dialogue(caller: Any, snapshot: dict, condition: str, policy: str,
                   epoch: int, ordinal: int) -> dict:
    if caller is None:
        raise RuntimeError("모델 정책에 대화 caller가 없다")
    return caller(snapshot, condition, policy, epoch, ordinal)


def _finish_epoch(caller: Any, policy: str, epoch: int, snapshot: dict | None, reason: str) -> None:
    function = getattr(caller, "finish_epoch", None)
    terminals = getattr(caller, "terminals", None)
    if isinstance(terminals, dict) and (policy, epoch) in terminals:
        return
    if callable(function):
        function(policy, epoch, snapshot, reason)


def _finalize(caller: Any, status: str, arms: dict) -> Any:
    function = getattr(caller, "finalize", None)
    return function(status=status, arms=deepcopy(arms)) if callable(function) else None


def _reference_plan(path: Path, checkpoint_sha256: str) -> dict:
    evidence = DEMO.read(path)
    if evidence.get("backend_kind") == "legacy":
        run_name = evidence.get("expected_run_name")
        seed = evidence.get("artifact", {}).get("backend", {}).get("seed")
    elif evidence.get("backend_kind") == "optimizer":
        run_name = evidence.get("raw", {}).get("run_name")
        seed = evidence.get("seed")
    else:
        raise ValueError("참조 probe evidence backend 종류가 다르다")
    return {"evidence_path": str(path.resolve()), "evidence_sha256": CONTROL.digest_file(path),
            "checkpoint_sha256": checkpoint_sha256,
            "stage": evidence["stage"], "run_name": run_name, "seed": seed,
            "gpu_contract": evidence["gpu_contract"]}


def _ledger_audit(folder: Path) -> dict:
    pending = []
    inventories = {"probe": 0, "correction": 0}
    for path in sorted(folder.glob("arms/*/epoch*/probes.ledger.jsonl")):
        inventories["probe"] += 1
        ledger = CONTROL.S.M.RegistryLedger(path, CONTROL.RC.registry())
        pending.extend({"kind": "probe", "path": str(path), "request_id": request_id,
                        "state": row["state"]}
                       for request_id, row in ledger.requests().items()
                       if row["state"] in {"pending", "approved", "running"})
    for path in sorted(folder.glob("arms/*/corrections.ledger.jsonl")):
        inventories["correction"] += 1
        ledger = CONTROL.D1.CorrectionLedger(path, CONTROL.RC.correction_registry())
        pending.extend({"kind": "correction", "path": str(path), "request_id": request_id,
                        "state": row["state"]}
                       for request_id, row in ledger.requests().items()
                       if row["state"] in {"pending", "approved", "running"})
    return {"inventories": inventories, "pending": pending, "pending_count": len(pending)}


def _close_incomplete_ledgers(folder: Path) -> None:
    for path in sorted(folder.glob("arms/*/epoch*/probes.ledger.jsonl")):
        ledger = CONTROL.S.M.RegistryLedger(path, CONTROL.RC.registry())
        for request_id, row in list(ledger.requests().items()):
            if row["state"] == "pending":
                ledger.reject(request_id, "flow-fail-closed", "batch_invalid")
            elif row["state"] == "running":
                ledger._append({  # noqa: SLF001 - null 비용의 종결 receipt를 보존하는 flow adapter
                    "event": "receipt", "request_id": request_id, "outcome": "unknown",
                    "measurement": {}, "gpu_s": None, "exit_code": None, "over_budget": False,
                    "registry_digest": ledger.registry_digest, "args_digest": row["args_digest"],
                    "effect_key": row["effect_key"], "reason": "batch_invalid",
                })
    for path in sorted(folder.glob("arms/*/corrections.ledger.jsonl")):
        ledger = CONTROL.D1.CorrectionLedger(path, CONTROL.RC.correction_registry())
        for request_id, row in list(ledger.requests().items()):
            if row["state"] == "pending":
                ledger.reject(request_id, "flow-fail-closed", "batch_invalid")
            elif row["state"] == "approved":
                ledger.consume(request_id, row["loop_state_digest"])
                ledger.receipt(request_id, "controller_actual_backend_v1", "unknown", None,
                               None, None, reason="batch_invalid")
            elif row["state"] == "running":
                ledger.receipt(request_id, "controller_actual_backend_v1", "unknown", None,
                               None, None, reason="batch_invalid")


def _budget_audit(backend: Any) -> dict | None:
    budget = getattr(backend, "budget", None)
    if not isinstance(budget, EXECUTION.BudgetedOwned):
        return None
    return {"consumed_owned_parent_wall_s": budget.consumed_owned_parent_wall_s,
            "remaining": budget.remaining()}


def _reports(raw: dict, artifacts: Mapping[str, dict]) -> dict[str, dict]:
    paths = raw.get("report_paths", {})
    if set(paths) != set(artifacts):
        raise ValueError("고정 평가 report arm이 다르다")
    return {arm: CONTROL.fixed_report(Path(paths[arm]), artifact) for arm, artifact in artifacts.items()}


def _source(folder: Path, tag: str, isaaclab: Path, protocol: dict, anchor: str,
            backend: Any) -> tuple[dict, dict, dict, dict, dict, dict]:
    raw = {}
    bindings = {}
    for arm, faults in (("NONE", []), (SOURCE_ARM, ["F1", "R1"])):
        DEMO.verify(isaaclab, anchor)
        bindings[arm] = DEMO.binding(protocol, anchor, faults=faults)
        stage = folder / "shared" / "training" / arm.lower()
        raw[arm] = _verified(stage, "train", backend.train(
            stage, isaaclab, arm, SEED, f"{tag}_s{SEED}_{arm.lower()}", bindings[arm]))
    reference = CONTROL.adapt_actual_artifact(
        raw["NONE"], bindings["NONE"], backend_arm="NONE", seed=SEED,
        arm="NONE", policy="shared_reference", epoch=0, corrections=[], source_checkpoint_sha256=None)
    compound = CONTROL.adapt_actual_artifact(
        raw[SOURCE_ARM], bindings[SOURCE_ARM], backend_arm=SOURCE_ARM, seed=SEED,
        arm="shared_compound", policy=CONTROL.POLICIES[0], epoch=0, corrections=[],
        source_checkpoint_sha256=None)
    artifact_paths = {"NONE": folder / "shared" / "artifact_none.controller.json",
                      SOURCE_ARM: folder / "shared" / "artifact_compound_evaluation.controller.json"}
    DEMO.write_new(artifact_paths["NONE"], reference)
    DEMO.write_new(artifact_paths[SOURCE_ARM], compound)
    DEMO.verify(isaaclab, anchor)
    stage = folder / "shared" / "source_evaluation"
    evaluated = _verified(stage, "fixed_evaluate", backend.fixed_evaluate(
        stage, isaaclab, raw, f"{tag}_s{SEED}_source", DEMO.binding(protocol, anchor, faults=["F1", "R1"])))
    reports = _reports(evaluated, {"NONE": reference, SOURCE_ARM: compound})
    bands = DEMO.read(DEMO.ROOT / RESIDUAL.BANDS)["bands"]
    gate = {**RESIDUAL.recovery(reports[SOURCE_ARM], reports["NONE"], bands),
            "none_walks": RESIDUAL.D1.V.R1.U.walks(reports["NONE"])}
    costs = {"training": {arm: _wall(raw[arm]) for arm in ("NONE", SOURCE_ARM)},
             "evaluation": _wall(evaluated)}
    source_info = {
        "artifacts": {arm: {"path": str(path), "sha256": CONTROL.digest_file(path)}
                      for arm, path in artifact_paths.items()},
        "reports": {arm: {"path": evaluated["report_paths"][arm],
                          "sha256": CONTROL.digest_file(Path(evaluated["report_paths"][arm]))}
                    for arm in ("NONE", SOURCE_ARM)},
    }
    return raw, reference, reports, gate, costs, source_info


def _shared_references(folder: Path, tag: str, isaaclab: Path, protocol: dict, anchor: str,
                       backend: Any, raw_none: dict, reference: dict) -> tuple[dict[str, Path], dict]:
    results: dict[str, Path] = {}
    binding = DEMO.binding(protocol, anchor, faults=[])
    legacy_stage = folder / "shared" / "reference_legacy"
    expected = sorted(CONTROL.L.PROBES)
    DEMO.verify(isaaclab, anchor)
    legacy = _verified(legacy_stage, "legacy_probe", backend.legacy_probe(
        legacy_stage, isaaclab, {"NONE": raw_none}, f"{tag}_s{SEED}_reference_legacy", binding))
    if legacy.get("probes") != expected:
        raise ValueError("공유 NONE legacy probe가 등록 정렬 6종과 다르다")
    for probe in expected:
        value = CONTROL.actual_legacy_probe_evidence(
            legacy_stage, legacy, reference, probe, f"{tag}_s{SEED}_reference_legacy",
            expected_probes=expected, gpu_contract=GPU_CONTRACT)
        path = folder / "shared" / "evidence" / f"{probe}.json"
        DEMO.write_new(path, value)
        results[probe] = path
    optimizer_stage = folder / "shared" / "reference_optimizer"
    inspect_binding = DEMO.binding(protocol, anchor, faults=[], checkpoint=reference["checkpoint"]["sha256"], updates=3)
    DEMO.verify(isaaclab, anchor)
    inspected = _verified(optimizer_stage, "inspect", backend.inspect(
        optimizer_stage, isaaclab, "NONE", SEED, f"{tag}_s{SEED}_reference_optimizer", inspect_binding))
    value = CONTROL.actual_optimizer_probe_evidence(
        optimizer_stage, inspected, inspect_binding, backend_arm="NONE", seed=SEED,
        source_checkpoint_sha256=reference["checkpoint"]["sha256"], gpu_contract=GPU_CONTRACT)
    path = folder / "shared" / "evidence" / f"{CONTROL.RC.NEW_PROBE}.json"
    DEMO.write_new(path, value)
    results[CONTROL.RC.NEW_PROBE] = path
    return results, {"legacy_six": _wall(legacy), "optimizer_three_updates": _wall(inspected)}


def _case_probe(folder: Path, tag: str, isaaclab: Path, protocol: dict, anchor: str,
                backend: Any, artifact: dict, policy: str, epoch: int, snapshot: dict,
                choice: dict, approver: str, references: Mapping[str, Path], ordinal: int) -> dict:
    probe = choice["probe"]
    stage = (folder / "arms" / policy / f"epoch{epoch}" / "probe_runs" /
             f"{ordinal:02d}_{probe.lower()}").resolve()
    run_name = f"{tag}_{policy}_e{epoch}_p{ordinal}_{probe.lower()}"
    spec = CONTROL.request_spec(snapshot, choice)
    plan = {"stage": str(stage), "run_name": run_name,
            "checkpoint_sha256": artifact["checkpoint"]["sha256"], "seed": SEED,
            "args": spec["args"], "gpu_contract": GPU_CONTRACT,
            "reference": _reference_plan(
                references[probe], snapshot["inputs"]["packet"]["input_bindings"]
                ["reference"]["checkpoint_sha256"])}
    auth = CONTROL.authorize_probe(folder / "arms", policy, epoch, snapshot, choice, approver,
                                   evidence_kind=CONTROL.ACTUAL_EVIDENCE, execution_plan=plan)
    CONTROL.validate_probe_authorization(auth, snapshot)
    try:
        DEMO.verify(isaaclab, anchor)
        raw_artifact = _backend_raw(artifact)
        backend_arm = artifact["backend"]["arm"]
        if probe == CONTROL.RC.NEW_PROBE:
            binding = DEMO.binding(protocol, anchor, applied=artifact["corrections"],
                                   faults=["F1", "R1"], checkpoint=artifact["checkpoint"]["sha256"], updates=3)
            raw = _verified(stage, "inspect", backend.inspect(
                stage, isaaclab, backend_arm, SEED, run_name, binding))
            evidence = CONTROL.actual_optimizer_probe_evidence(
                stage, raw, binding, backend_arm=backend_arm, seed=SEED,
                source_checkpoint_sha256=artifact["checkpoint"]["sha256"], gpu_contract=GPU_CONTRACT)
        else:
            binding = DEMO.binding(protocol, anchor, applied=artifact["corrections"], faults=["F1", "R1"],
                                   checkpoint=artifact["checkpoint"]["sha256"])
            raw = _verified(stage, "legacy_probe", backend.legacy_probe(
                stage, isaaclab, {backend_arm: raw_artifact}, run_name, binding, probes=[probe]))
            evidence = CONTROL.actual_legacy_probe_evidence(
                stage, raw, artifact, probe, run_name, expected_probes=[probe], gpu_contract=GPU_CONTRACT)
        case_path = stage / "controller_evidence.json"
        DEMO.write_new(case_path, evidence)
        sealed = CONTROL.seal_probe(auth, snapshot, case_path, references[probe], _wall(raw), 0)
        CONTROL.verify_probe(Path(sealed["path"]), snapshot)
        sealed["owned_wall_s"] = _wall(raw)
        return sealed
    except Exception as error:
        ledger_path = Path(auth["path"]).parent / "probes.ledger.jsonl"
        request = CONTROL.S.M.RegistryLedger(ledger_path, CONTROL.RC.registry()).requests().get(
            auth["request_id"], {})
        if request.get("state") == "running":
            CONTROL.fail_probe(auth, snapshot, gpu_s=None, exit_code=None, reason=type(error).__name__)
        raise


def _repair(folder: Path, tag: str, isaaclab: Path, protocol: dict, anchor: str, backend: Any,
            policy: str, epoch: int, applied: list[str], current: dict, reference: dict,
            reference_report: Path, snapshot: dict, seals: list[dict], approver: str,
            bands: dict, parent: Path | None) -> tuple[Path, dict, dict]:
    ledger = folder / "arms" / policy / "corrections.ledger.jsonl"
    auth = CONTROL.authorize_repair(ledger, policy, snapshot, applied, seals, approver, 3600)
    after = [*applied, auth["correction_id"]]
    backend_arm = "FIRST" if epoch == 1 else "BOTH"
    binding = DEMO.binding(protocol, anchor, applied=after, faults=["F1", "R1"],
                           checkpoint=current["checkpoint"]["sha256"])
    stage = folder / "arms" / policy / f"epoch{epoch}" / "repair_training"
    try:
        DEMO.verify(isaaclab, anchor)
        raw = _verified(stage, "train", backend.train(
            stage, isaaclab, backend_arm, SEED, f"{tag}_{policy}_e{epoch}_repair", binding))
        output = CONTROL.adapt_actual_artifact(
            raw, binding, backend_arm=backend_arm, seed=SEED, arm=policy, policy=policy, epoch=epoch,
            corrections=after, source_checkpoint_sha256=current["checkpoint"]["sha256"])
        eval_stage = folder / "arms" / policy / f"epoch{epoch}" / "repair_evaluation"
        DEMO.verify(isaaclab, anchor)
        evaluation = _verified(eval_stage, "fixed_evaluate", backend.fixed_evaluate(
            eval_stage, isaaclab, {backend_arm: raw}, f"{tag}_{policy}_e{epoch}_evaluation", binding))
        output_report = Path(evaluation["report_paths"][backend_arm])
        CONTROL.fixed_report(output_report, output)
        repair_path = folder / "arms" / policy / f"epoch{epoch}" / "repair.json"
        CONTROL.seal_repair(
            repair_path, ledger_path=ledger, authorization=auth, arm=policy, policy=policy, epoch=epoch,
            applied_before=applied, input_artifact=current, output_artifact=output,
            output_report_path=output_report, reference_artifact=reference,
            reference_report_path=reference_report, bands=bands,
            gpu_s=_wall(raw) + _wall(evaluation), parent_path=parent)
        verified = CONTROL.verify_repair(repair_path, ledger_path=ledger, bands=bands)
        verified["owned_wall_s"] = _wall(raw) + _wall(evaluation)
        return repair_path, output, verified
    except Exception:
        correction = CONTROL.D1.CorrectionLedger(ledger, CONTROL.RC.correction_registry())
        request = correction.requests().get(auth["request_id"], {})
        if request.get("state") == "running":
            correction.receipt(auth["request_id"], "controller_actual_backend_v1", "unknown", None,
                               None, None, reason="repair_execution_failed")
        raise


def run(folder: Path, tag: str, isaaclab: Path, approver: str, protocol: dict, anchor: str, *,
        backend: Any, dialogue_caller: Any = None) -> dict:
    """사전 검증·seed guard 후의 D4 실행. CPU test는 backend·controller를 대체한다."""
    result: dict[str, Any] = {"schema": "selection_comparison_result_v1", "tag": tag,
                              "seed": SEED, "status": "running", "source_gate": None,
                              "shared": {}, "arms": {}, "costs": {"shared": {}, "arms": {}}}
    status = "invalid"
    bands = DEMO.read(DEMO.ROOT / RESIDUAL.BANDS)["bands"]
    try:
        raw, reference, _source_reports, gate, source_costs, source_info = _source(
            folder, tag, isaaclab, protocol, anchor, backend)
        result["source_gate"] = gate
        result["shared"]["source"] = source_info
        result["costs"]["shared"]["source"] = source_costs
        if not gate["none_walks"]:
            status = "baseline_invalid"
            result["status"] = status
            return result
        if gate["outcome"] == "recovered":
            status = "source_failure_absent"
            result["status"] = status
            return result
        if gate["outcome"] != "not_recovered":
            status = "HOLD"
            result.update(status=status, reason="source_evaluation_unknown")
            return result
        references, reference_costs = _shared_references(
            folder, tag, isaaclab, protocol, anchor, backend, raw["NONE"], reference)
        result["shared"]["reference_evidence"] = {key: str(path) for key, path in references.items()}
        result["costs"]["shared"]["reference"] = reference_costs
        reference_report = Path(source_info["reports"]["NONE"]["path"])
        for policy in CONTROL.POLICIES:
            current = CONTROL.adapt_actual_artifact(
                raw[SOURCE_ARM], DEMO.binding(protocol, anchor, faults=["F1", "R1"]),
                backend_arm=SOURCE_ARM, seed=SEED, arm=policy, policy=policy, epoch=0,
                corrections=[], source_checkpoint_sha256=None)
            DEMO.write_new(folder / "arms" / policy / "epoch0_artifact.json", current)
            applied: list[str] = []
            prior = None
            parent = None
            arm_result = {"status": "running", "epochs": [], "probe_order": [],
                          "costs": {"probes": [], "repairs": []}}
            result["arms"][policy] = arm_result
            result["costs"]["arms"][policy] = arm_result["costs"]
            for epoch in (1, 2):
                if epoch == 2 and not CONTROL.next_epoch(
                        parent, ledger_path=folder / "arms" / policy / "corrections.ledger.jsonl",
                        bands=bands):
                    _finish_epoch(dialogue_caller, policy, epoch, None, "absent")
                    break
                seals: list[dict] = []
                ordinal = 0
                exposed = False
                terminal_snapshot = None
                while True:
                    snapshot = _snapshot(folder, tag, policy, epoch, current, reference, applied, prior)
                    terminal_snapshot = snapshot
                    if snapshot["view"]["status"] != "open":
                        break
                    dialogue = None
                    if CONTROL.dialogue_input(snapshot)["needs_model"]:
                        dialogue = _call_dialogue(dialogue_caller, snapshot, "production", policy, epoch, ordinal)
                    choice = CONTROL.choose(snapshot, policy, dialogue)
                    if choice["action"] != "continue":
                        arm_result["status"] = "HOLD"
                        arm_result["reason"] = choice["reason"]
                        break
                    ordinal += 1
                    sealed = _case_probe(folder, tag, isaaclab, protocol, anchor, backend, current,
                                         policy, epoch, snapshot, choice, approver, references, ordinal)
                    seals.append(sealed)
                    arm_result["probe_order"].append(choice["probe"])
                    arm_result["costs"]["probes"].append(sealed["owned_wall_s"])
                    if policy == "model_adaptive" and not exposed and sealed["outcome"] in {"normal", "abnormal"}:
                        observed = _snapshot(folder, tag, policy, epoch, current, reference, applied, prior)
                        _call_dialogue(dialogue_caller, observed, "exposure_full", policy, epoch, ordinal)
                        _call_dialogue(dialogue_caller, observed, "exposure_masked", policy, epoch, ordinal)
                        exposed = True
                if arm_result["status"] == "HOLD":
                    reason = arm_result.get("reason", "model_invalid")
                    terminal_reason = reason if reason in {"model_hold", "model_invalid", "model_infra", "budget"} else "model_invalid"
                    _finish_epoch(dialogue_caller, policy, epoch, terminal_snapshot, terminal_reason)
                    if epoch == 1:
                        _finish_epoch(dialogue_caller, policy, 2, None, "absent")
                    break
                snapshot = _snapshot(folder, tag, policy, epoch, current, reference, applied, prior)
                if snapshot["view"]["status"] != "confirmed":
                    arm_result.update(status="HOLD", reason="diagnosis_" + snapshot["view"]["status"])
                    _finish_epoch(dialogue_caller, policy, epoch, snapshot, "view_terminal")
                    if epoch == 1:
                        _finish_epoch(dialogue_caller, policy, 2, None, "absent")
                    break
                repair_path, output, verified = _repair(
                    folder, tag, isaaclab, protocol, anchor, backend, policy, epoch, applied,
                    current, reference, reference_report, snapshot, seals, approver, bands, parent)
                arm_result["epochs"].append({"epoch": epoch, "outcome": verified["outcome"],
                                              "correction_id": verified["applied_after"][-1],
                                              "repair_path": str(repair_path)})
                arm_result["costs"]["repairs"].append(verified["owned_wall_s"])
                terminal_snapshot = _snapshot(
                    folder, tag, policy, epoch, current, reference, applied, prior)
                _finish_epoch(dialogue_caller, policy, epoch, terminal_snapshot, "view_terminal")
                applied = verified["applied_after"]
                current = output
                parent = repair_path
                if verified["outcome"] == "recovered":
                    arm_result["status"] = "recovered"
                    if epoch == 1:
                        _finish_epoch(dialogue_caller, policy, 2, None, "absent")
                    break
                if verified["outcome"] != "not_recovered" or epoch == 2:
                    arm_result["status"] = verified["outcome"]
                    break
                prior = {"verdict": "not_recovered", "correction_id": applied[-1],
                         "result_sha256": verified["result_sha256"],
                         "checkpoint_sha256": output["checkpoint"]["sha256"]}
            if arm_result["status"] == "running":
                arm_result["status"] = "HOLD"
        status = "complete"
        result["status"] = status
        return result
    except Exception as error:
        result.update(status="invalid", failure={"type": type(error).__name__})
        for policy in CONTROL.POLICIES:
            existing = result["arms"].get(policy, {})
            epoch = min(2, len(existing.get("epochs", [])) + 1)
            _finish_epoch(dialogue_caller, policy, epoch, None, "batch_invalid")
        return result
    finally:
        if result.get("status") == "invalid":
            _close_incomplete_ledgers(folder)
        result["ledger_audit"] = _ledger_audit(folder)
        budget_audit = _budget_audit(backend)
        if budget_audit is not None:
            result["owned_budget"] = budget_audit
        if result["ledger_audit"]["pending_count"]:
            result.update(status="invalid", failure={"type": "PendingLedgerRequest"})
            _close_incomplete_ledgers(folder)
            result["ledger_audit"] = _ledger_audit(folder)
        final_status = result.get("status", status)
        try:
            audit = _finalize(dialogue_caller, final_status, result["arms"])
            if audit is not None:
                result["model_audit"] = audit
                if isinstance(audit, dict) and audit.get("status") != "complete":
                    result.update(status="invalid", failure={"type": "ModelAuditInvalid"})
        except Exception as error:
            result.update(status="invalid", failure={"type": type(error).__name__})


def execute(tag: str, isaaclab: Path, approver: str, approve: bool, *, backend: Any = None,
            dialogue_caller: Any = None,
            seed_guard: Callable[..., dict] = RESIDUAL.assert_fresh_seeds) -> dict:
    if not approve:
        return {"status": "approval_required", "tag": tag}
    if not DEMO.TAG.fullmatch(tag):
        raise ValueError("D4 tag 형식이 다르다")
    protocol, anchor = DEMO.verify(isaaclab)
    freshness = seed_guard([SEED], isaaclab, repo=DEMO.ROOT)
    folder = DEMO.PRIVATE / tag
    if folder.exists():
        raise FileExistsError("동일 tag 실행 폴더가 이미 있다")
    folder.mkdir(parents=True, exist_ok=False)
    DEMO.write_new(folder / "begin.json", {"tag": tag, "seed": SEED, "anchor": anchor,
                                             "protocol_sha256_lf": DEMO.sha(DEMO.PROTOCOL, True)})
    DEMO.write_new(folder / "seed_freshness.json", freshness)
    try:
        if backend is None:
            budget = EXECUTION.BudgetedOwned(folder / "owned_audit")
            backend = EXECUTION.SelectionComparisonBackend(budget)
            if dialogue_caller is None:
                import selection_comparison_worker_v1
                dialogue_caller = selection_comparison_worker_v1.ModelCaller(
                    folder / "model", protocol, anchor, budget)
        result = run(folder, tag, isaaclab, approver, protocol, anchor,
                     backend=backend, dialogue_caller=dialogue_caller)
    except Exception as error:
        result = {"schema": "selection_comparison_result_v1", "tag": tag, "seed": SEED,
                  "status": "invalid", "source_gate": None, "shared": {}, "arms": {},
                  "costs": {"shared": {}, "arms": {}},
                  "failure": {"type": type(error).__name__}}
    result["preflight"] = {"seed_freshness_path": str(folder / "seed_freshness.json"),
                           "seed_freshness_sha256": CONTROL.digest_file(folder / "seed_freshness.json")}
    DEMO.write_new(folder / "result.json", result)
    return result
