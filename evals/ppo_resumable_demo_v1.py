"""PPO 계산 오류 고리: 완료 단계 재사용, 판단 장애 일시정지, 승인 한 번 소비.

모델은 첫 검사와 근거 ID를 고른다. 이후 검사는 기존 등록 규칙, 교정은 확정된
surrogate 범주의 등록 교정이다. 기존 terminal 실행은 이 판으로 인수하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "evals", ROOT / "src", ROOT / "bench"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import model_decision_journal_v1 as J
import ppo_ratio_repair_demo_v4 as OLD
import ppo_resumable_transport_v1 as TRANSPORT
import ppo_selection_adapter_v1 as SELECT
from preregistered_seed_guard import assert_fresh_seeds

C, E, PILOT, G, REC = OLD.C, OLD.E, OLD.PILOT, OLD.G, OLD.REC
SEED, RESTART_SEED = 2200, 3200
FAULT = OLD.FAULT
PROTOCOL = ROOT / "bench/protocols/ppo_resumable_demo_v1.json"
PRIVATE = ROOT / "bench/private/codex_main_20261025/ppo_resumable_v1"
NEW_CODE = {
    "evals/model_decision_journal_v1.py", "tests/test_model_decision_journal_v1.py",
    "evals/ppo_selection_adapter_v1.py", "tests/test_ppo_selection_adapter_v1.py",
    "evals/ppo_resumable_model_v1.py", "tests/test_ppo_resumable_model_v1.py",
    "evals/ppo_resumable_transport_v1.py", "tests/test_ppo_resumable_transport_v1.py",
    "evals/ppo_resumable_demo_v1.py", "tests/test_ppo_resumable_demo_v1.py",
    "bench/run_ppo_resumable_demo_v1.ps1",
}
CODE = OLD.CODE | NEW_CODE
TAG = re.compile(r"ppo_resumable_s2200_[0-9]{8}[a-z0-9_]{0,16}")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_protocol() -> dict:
    return {
        "schema": "ppo_resumable_demo_protocol_v1", "seed": SEED, "restart_seed": RESTART_SEED,
        "task": OLD.TASK, "num_envs": 4096, "training_updates": 300,
        "inspection_updates": 3, "configured_epochs": 5, "configured_minibatches": 4,
        "model": SELECT.MODEL, "provider_aliases": sorted(SELECT.ACCEPTED_PROVIDER_MODELS),
        "decision_contract": ["action", "probe", "fact_ids"],
        "selection_authority": {"first_probe": "model", "later_probes": "registered_rule",
                                "correction": "registered_rule"},
        "api": {"total_cap": 20, "dialogue_cap": 3, "infra_retry": 1,
                "semantic_feedback": 1, "retry_delay_s": 30, "sdk_retries": 0},
        "resume": {
            "completed_stage": "결합·저장 파일 SHA 일치 확인 후 재사용",
            "completed_infra_response": "paused_infra 뒤 30초 이후 같은 판단 원장에서 한 번 재시도",
            "unknown_pending": "blocked_pending: 응답 미저장 예약은 재전송하지 않음",
            "gpu_stage_interrupted": "blocked_stage_unknown: 부분 실행을 자동 반복하지 않음",
            "old_terminal": "기존 HOLD·invalid·완료 실행 인수 금지",
            "semantic_invalid": "고정 피드백 한 번 뒤에도 무효이면 terminal HOLD; 규칙 전환 없음",
        },
        "confirmation": "기존 8개 검사·subtype 표·C.advance·correction_allowed 그대로",
        "correction": {"id": C.CORRECTION, "recovery_mode": "restart", "updates": 300,
                       "requires": "optimizer_surrogate 단일 확정과 P_ppo_gradient abnormal receipt"},
        "fixed_evaluation": {"conditions": 26, "bands_evidence": PILOT.P1.BANDS_EVIDENCE,
                             "reference": "같은 source seed의 정상 정책"},
        "prediction": "횟수 검사는 normal이고 gradient 검사가 abnormal이면 surrogate 범주를 가를 수 있다",
        "success": ["source_failure_present", "valid_model_first_probe", "surrogate_confirmed",
                    "approved_probe_receipts", "approved_correction_once", "restored_gradient",
                    "corrected_fixed_eval_healthy"],
        "wording": {
            "recovered": "새 seed에서 등록 검사·교정 뒤 고정 평가 회복을 확인했다",
            "not_recovered": "계산 교정 뒤에도 고정 평가 회복은 확인하지 못했다",
            "HOLD": "모델 판단 또는 원인 확정 조건을 충족하지 못해 교정하지 않았다",
            "paused_infra": "완료된 인프라 오류를 기록하고 같은 판단의 재개를 대기한다",
            "blocked_pending": "응답 저장 여부가 불명확해 자동 재전송을 하지 않았다",
        },
        "design_coupling": {
            "development": True,
            "seen": ["PPO inverse-ratio 검사 개발 결과", "seed 2165의 이전 계약 무효",
                     "Brev seed 2196의 두 번째 판단 HTTP 500 HOLD"],
            "interpretation": "이미 관찰한 결함에서 계약·실행 지속성을 개발한 판이며 새 결함 대응 성능 주장이 아니다",
            "old_results": "이전 결과·소비·terminal 상태를 보존하고 새 seed·새 tag로 실행한다",
        },
    }


def dynamic(isaaclab: Path) -> dict:
    return {
        "source_sha256_lf": {name: E.sha(ROOT / name, True) for name in sorted(CODE)},
        "evidence_sha256_lf": {name: E.sha(ROOT / name, True) for name in OLD.EVIDENCE},
        "runtime_source_sha256_lf": {
            row["path"]: row["sha256_lf"] for row in PILOT.R1S.isaaclab_sources(isaaclab).values()
        },
    }


def verify(isaaclab: Path, anchor: str | None = None, *, committed: bool = True) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    if protocol != {**fixed_protocol(), **dynamic(isaaclab)}:
        raise ValueError("사전등록·소스·입력·런타임 결합 불일치")
    head = OLD.git("rev-parse", "HEAD")
    anchor = anchor or head
    if committed:
        paths = sorted(CODE | set(OLD.EVIDENCE) | {PROTOCOL.relative_to(ROOT).as_posix()})
        OLD.git("merge-base", "--is-ancestor", anchor, head)
        OLD.git("ls-files", "--error-unmatch", "--", *paths)
        if OLD.git("status", "--porcelain", "--", *paths):
            raise ValueError("결합 파일은 실행 전에 커밋되어야 한다")
        for name in paths:
            if OLD.git("rev-parse", f"{anchor}:{name}") != OLD.git("rev-parse", f"{head}:{name}"):
                raise ValueError("실험 commit 뒤 결합 blob이 변경됐다")
    return protocol, anchor


def binding(protocol: dict, anchor: str, checkpoint: str | None = None, updates: int = 300) -> dict:
    value = OLD.V3.binding(protocol, anchor, checkpoint, updates)
    value.update(protocol_path=PROTOCOL.relative_to(ROOT).as_posix(), protocol_sha256_lf=E.sha(PROTOCOL, True))
    return value


@contextmanager
def exclusive(folder: Path, run_binding: dict):
    with J._lease(folder / ".controller_active", {
        "scope": "controller", "manifest_sha256": J.digest(run_binding),
    }):
        yield


class StageStore:
    """단계 시작·완료와 산출물 SHA를 이어 붙인다. 미완료 시작은 자동 재실행하지 않는다."""

    def __init__(self, folder: Path, run_binding: dict):
        self.folder = folder
        self.manifest = {"schema": "ppo_resumable_run_v1", "binding": run_binding}
        path = folder / "manifest.json"
        if path.exists():
            if read(path) != self.manifest:
                raise ValueError("기존 run 결합 불일치")
        else:
            if any(path.name != ".controller_active" for path in folder.iterdir()):
                raise ValueError("manifest 없는 기존 실행을 인수하지 않는다")
            E.write_new(path, self.manifest)
        self.events: list[dict] = []
        previous = None
        paths = sorted((folder / "events").glob("*.json"))
        for index, event_path in enumerate(paths, 1):
            if event_path.name != f"{index:04d}.json":
                raise ValueError("단계 원장 순서 불일치")
            event = read(event_path)
            if event["previous_sha256"] != previous or event["manifest_sha256"] != J.digest(self.manifest):
                raise ValueError("단계 원장 chain 불일치")
            for file, sha in event.get("inventory", {}).items():
                if not Path(file).is_file() or E.sha(Path(file)) != sha:
                    raise ValueError("완료 단계 산출물 SHA 불일치")
            for name, prefix in event.get("ledger_prefixes", {}).items():
                path = folder / name
                if not path.is_file():
                    raise ValueError("승인 원장이 사라졌다")
                raw = path.read_bytes()
                if len(raw) < prefix["size"] or hashlib.sha256(raw[:prefix["size"]]).hexdigest() != prefix["sha256"]:
                    raise ValueError("승인 원장 이전 기록이 바뀌었다")
            self.events.append(event)
            previous = J.digest(event)

    def append(self, event: str, stage: str, **extra) -> None:
        prefixes = {}
        for name in ("probes.ledger.jsonl", "correction.ledger.jsonl"):
            path = self.folder / name
            if path.is_file():
                raw = path.read_bytes()
                prefixes[name] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        row = {"event": event, "stage": stage, "previous_sha256": J.digest(self.events[-1]) if self.events else None,
               "manifest_sha256": J.digest(self.manifest), "ledger_prefixes": prefixes, **extra}
        E.write_new(self.folder / "events" / f"{len(self.events) + 1:04d}.json", row)
        self.events.append(row)

    def once(self, name: str, operation) -> dict:
        rows = [row for row in self.events if row["stage"] == name]
        if rows:
            if len(rows) != 2 or rows[0]["event"] != "started" or rows[1]["event"] != "done":
                raise RuntimeError("blocked_stage_unknown:" + name)
            return rows[1]["value"]
        self.append("started", name)
        value = operation(self.folder / name)
        files = {path for path in (self.folder / name).rglob("*") if path.is_file()}

        def collect(item):
            if isinstance(item, dict):
                if isinstance(item.get("path"), str) and "sha256" in item:
                    path = Path(item["path"])
                    if not path.is_file() or E.sha(path) != item["sha256"]:
                        raise ValueError("체크포인트·텔레메트리 SHA 불일치")
                    files.add(path)
                for child in item.values():
                    collect(child)
            elif isinstance(item, list):
                for child in item:
                    collect(child)

        collect(value)
        self.append("done", name, value=value, inventory={str(p.resolve()): E.sha(p) for p in sorted(files)})
        return value


def model_decision(folder: Path, packet: dict, decision_binding: dict) -> dict:
    return TRANSPORT.model_decision(folder, packet, decision_binding)


def resume_model(store: StageStore, packet: dict, decision_binding: dict, caller) -> dict:
    rows = [row for row in store.events if row["stage"] == "model"]
    if rows and rows[-1]["decision_binding"] != decision_binding:
        raise ValueError("모델 단계 결합 불일치")
    if rows and rows[-1]["event"] == "model_done":
        return rows[-1]["value"]
    if rows and rows[-1]["event"] == "model_advance_started":
        raise RuntimeError("blocked_pending:model_advance_without_controller_receipt")
    folder = store.folder / "model"
    if not rows and folder.exists():
        raise ValueError("controller 기록 없는 모델 원장을 인수하지 않는다")
    store.append("model_advance_started", "model", decision_binding=decision_binding)
    value = caller(folder, packet, decision_binding)
    inventory = {str(path.resolve()): E.sha(path) for path in sorted(folder.rglob("*")) if path.is_file()}
    if not inventory:
        raise ValueError("모델 판단 원장 증거가 없다")
    event = "model_progress" if value["status"] in ("paused_infra", "blocked_pending") else "model_done"
    store.append(event, "model", value=value, decision_binding=decision_binding, inventory=inventory)
    return value


def execute(tag: str, isaaclab: Path, approver: str, approved: bool, *, backend: Any = E,
            caller=model_decision, seed_guard=assert_fresh_seeds) -> dict:
    if not TAG.fullmatch(tag) or not approver or not approved:
        raise ValueError("등록 tag·승인자·등록 행동 승인 플래그가 필요하다")
    folder = PRIVATE / tag
    existing = read(folder / "manifest.json")["binding"] if (folder / "manifest.json").is_file() else None
    protocol, anchor = verify(isaaclab, existing["source_anchor"] if existing else None)
    run_binding = {"run_id": tag, "source_anchor": anchor, "protocol_sha256": E.sha(PROTOCOL, True),
                   "seed": SEED, "restart_seed": RESTART_SEED, "approver": approver}
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder, run_binding):
        store = StageStore(folder, run_binding)
        if (folder / "result.json").exists():
            terminal = next(row for row in store.events if row["stage"] == "terminal" and row["event"] == "terminal")
            if J.digest(read(folder / "result.json")) != terminal["result_sha256"]:
                raise ValueError("terminal 결과 불일치")
            return read(folder / "result.json")
        if any(row["event"] == "terminal" for row in store.events):
            raise RuntimeError("blocked_stage_unknown:terminal_write")
        store.once("preflight", lambda _: {"paths": backend.preflight(isaaclab),
                   "seeds": seed_guard([SEED, RESTART_SEED], isaaclab, repo=ROOT)})
        artifacts = {}
        for arm in ("NONE", FAULT):
            artifacts[arm] = store.once("train_" + arm.lower(), lambda stage, arm=arm: backend.train(
                stage, isaaclab, arm, SEED, tag + "_" + arm.lower(), binding(protocol, anchor)))
        source = artifacts[FAULT]["checkpoint"]["sha256"]
        evaluation = store.once("source_evaluation", lambda stage: backend.fixed_evaluate(
            stage, isaaclab, artifacts, tag, binding(protocol, anchor)))
        reports = PILOT.validated_reports(evaluation, artifacts)
        bands = read(ROOT / PILOT.P1.BANDS_EVIDENCE)["bands"]
        gate = OLD.source_gate(reports, bands)
        result = {"schema": "ppo_resumable_demo_result_v1", **run_binding, "source_gate": gate,
                  "conditions": {"source_failure_present": gate["outcome"] == "failure_present"},
                  "probe_steps": [], "selection_authority": protocol["selection_authority"]}

        def finish(outcome: str, reason: str) -> dict:
            result.update(status="complete", outcome=outcome, reason=reason)
            result["demo_success"] = outcome == "recovered" and all(
                result["conditions"].get(name) is True for name in protocol["success"])
            result["api_requests"] = result.get("model", {}).get("api", {}).get("reserved", 0)
            result["pending_api_requests"] = result.get("model", {}).get("api", {}).get("pending", 0)
            inventory = {str(path.resolve()): E.sha(path) for path in (
                folder / "probes.ledger.jsonl", folder / "correction.ledger.jsonl") if path.is_file()}
            store.append("terminal", "terminal", result_sha256=J.digest(result), inventory=inventory)
            E.write_new(folder / "result.json", result)
            return result

        if gate["outcome"] != "failure_present":
            return finish(gate["outcome"], "source_gate")
        reference = store.once("reference", lambda stage: backend.legacy_probe(
            stage, isaaclab, {"NONE": artifacts["NONE"]}, tag + "_ref", binding(protocol, anchor)))
        packet = OLD.diagnosis_packet(artifacts)
        snapshot = SELECT.build_snapshot(packet)
        decision_binding = {key: run_binding[key] for key in ("run_id", "source_anchor", "protocol_sha256")}
        decision_binding.update(checkpoint_sha256=source, snapshot_sha256=J.digest(snapshot),
                                initial_messages_sha256=J.digest(SELECT.messages(snapshot)))
        verify(isaaclab, anchor)
        model = resume_model(store, packet, decision_binding, caller)
        result["model"] = model
        if model["status"] in ("paused_infra", "blocked_pending"):
            return {**result, "status": model["status"], "demo_success": False}
        if model["status"] != "choice_ready":
            return finish("HOLD", "model_terminal_hold")
        engine = model["engine_result"]
        decision = C.validate_diagnosis(engine["host_decision"], packet)
        result["conditions"]["valid_model_first_probe"] = True
        state = C.initial_state(decision)
        ledger = C.M.RegistryLedger(folder / "probes.ledger.jsonl", C.registry())
        prereg = {"checkpoint_sha256": source, "seed": SEED, "protocol_sha256_lf": E.sha(PROTOCOL, True),
                  "experiment_commit": anchor, "registry_digest": C.M.registry_digest(C.registry())}
        while state["status"] == "open":
            probe = state["next_probe"]

            def perform(stage, probe=probe, first=not state["observed"]):
                verify(isaaclab, anchor)
                request = ledger.propose(prereg, probe, OLD.PROBE_BUDGET_S)
                rid = request["request_id"]
                ledger.approve(rid, approver)
                ledger.consume(rid, prereg, probe)
                if probe in (C.COUNT_PROBE, C.GRADIENT_PROBE):
                    controls = {arm: backend.inspect(stage / arm.lower(), isaaclab, arm, SEED,
                        f"{tag}_{probe.lower()}_{arm.lower()}",
                        binding(protocol, anchor, artifacts[arm]["checkpoint"]["sha256"], 3))
                        for arm in ("NONE", FAULT)}
                    E.write_new(stage / "evidence.json", controls)
                    outcome, measurement = OLD._inspection_receipt(probe, controls, source, E.sha(stage / "evidence.json"))
                    seconds = sum(OLD._duration(row) for row in controls.values())
                else:
                    raw = backend.legacy_probe(stage, isaaclab, {FAULT: artifacts[FAULT]},
                        tag + "_" + probe.lower(), binding(protocol, anchor), probes=[probe])
                    item = raw["report"]["probes"][probe]
                    ref = reference["report"]["probes"][probe]
                    measurement = item["measurement"] or {}
                    outcome = C.L.classify(probe, measurement if item["exit"] == 0 else None,
                                           ref["measurement"] if ref["exit"] == 0 else None)
                    seconds = OLD._duration(raw)
                receipt = ledger.receipt(rid, outcome, measurement, seconds, 0)
                receipt.update(probe=probe, selected_by="model" if first else "registered_rule")
                return receipt

            receipt = store.once("probe_" + probe.lower(), perform)
            result["probe_steps"].append(receipt)
            state = C.advance(state, probe, receipt["outcome"])
        result["loop"] = state
        result["conditions"]["approved_probe_receipts"] = all(
            row["state"] == "done" for row in ledger.requests().values())
        if not C.correction_allowed(state):
            return finish("HOLD", "cause_not_confirmed_or_scope_insufficient")
        result["conditions"]["surrogate_confirmed"] = True
        correction_ledger = OLD.CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())

        def correct(stage):
            verify(isaaclab, anchor)
            request = correction_ledger.propose(C.L.digest(state), "optimizer", C.L.digest(result["probe_steps"]),
                C.CORRECTION, "restart", {"case": "optimizer_surrogate_ratio", "arm": "corrected"},
                OLD.CORRECTION_BUDGET_S, evidence={"checkpoint_sha256": source,
                    "protocol_sha256_lf": E.sha(PROTOCOL, True), "selected_by": "registered_rule"})
            rid = request["request_id"]
            correction_ledger.approve(rid, approver)
            correction_ledger.consume(rid, C.L.digest(state))
            trained = backend.train(stage, isaaclab, "CORRECTED", RESTART_SEED, tag + "_corrected",
                                    binding(protocol, anchor, source))
            return {"artifact": trained, "request_id": rid}

        correction = store.once("correction_training", correct)
        corrected = correction["artifact"]
        observed = corrected["inspection"]
        result["conditions"]["restored_gradient"] = (
            observed.get("count_status") == observed.get("gradient_status") == "normal"
            and observed.get("optimizer_steps") == observed.get("minibatches") == [20, 20, 20])
        if not result["conditions"]["restored_gradient"]:
            store.once("correction_validation_receipt", lambda stage: correction_ledger.receipt(
                correction["request_id"], tag, "unknown", None, OLD._duration(corrected), 1,
                reason="gradient_not_restored"))
            return finish("correction_unknown", "gradient_not_restored")
        repaired_eval = store.once("correction_evaluation", lambda stage: backend.fixed_evaluate(
            stage, isaaclab, {"CORRECTED": corrected}, tag + "_fixed", binding(protocol, anchor, source)))
        repaired = PILOT.validated_reports(repaired_eval, {"CORRECTED": corrected})["CORRECTED"]
        if repaired["versions"] != reports["NONE"]["versions"]:
            raise ValueError("정상·교정 평가 런타임이 다르다")
        verdict = REC.verdict(REC.compare(G._behavior(repaired), G._behavior(reports["NONE"])), bands)
        outcome = "recovered" if verdict["label"] == "healthy" else "not_recovered"
        result["recovery"] = {"verdict": verdict, "corrected_metrics": repaired["metrics"],
                              "reference_metrics": reports["NONE"]["metrics"]}

        def close_receipt(stage):
            E.write_new(stage / "recovery.json", result["recovery"])
            return correction_ledger.receipt(correction["request_id"], tag, outcome,
                E.sha(stage / "recovery.json"), OLD._duration(corrected) + OLD._duration(repaired_eval), 0)

        receipt = store.once("correction_receipt", close_receipt)
        result["conditions"]["approved_correction_once"] = receipt["verdict"] != "unknown"
        result["conditions"]["corrected_fixed_eval_healthy"] = outcome == "recovered"
        verify(isaaclab, anchor)
        if receipt["verdict"] == "unknown":
            return finish("correction_unknown", "correction_receipt_unknown")
        return finish(outcome, "fixed_eval")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "PPO 재개 고리").splitlines()[0])
    parser.add_argument("command", choices=("prepare", "check", "verify", "execute", "fresh-seeds", "gpu-gate"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default="ppo_resumable_s2200_20261025")
    parser.add_argument("--approver", default="codex-main-user-delegated")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        E.write_new(PROTOCOL, {**fixed_protocol(), **dynamic(args.isaaclab)})
        value = {"status": "prepared_commit_required"}
    elif args.command in ("verify", "check"):
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "ok", "anchor": anchor}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
    elif args.command == "gpu-gate":
        value = OLD.gpu_gate()
    else:
        value = execute(args.tag, args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
