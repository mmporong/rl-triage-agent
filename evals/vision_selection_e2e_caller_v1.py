"""V05 E2E의 canonical API·probe-pair·교정 원장 작성자."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import grounded_vision_observer_v1 as OBSERVER
import model_decision_journal_v1 as JOURNAL
import vision_selection_brev_adapter_v1 as ADAPTER
import vision_selection_call_v1 as CALL
import vision_selection_e2e_v1 as CONTRACT

from rl_triage.correction_ledger import CorrectionLedger

UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 compatibility
PAIR_LEDGER_SCHEMA = "vision_selection_probe_pair_ledger_event_v1"
STATE_SCHEMA = "vision_selection_e2e_state_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}은 JSON object여야 한다")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    body = dict(value)
    if path.exists():
        if _read(path) != body:
            raise FileExistsError(f"기존 {path.name}과 새 값이 다르다")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def _at() -> str:
    return datetime.now(UTC_TZ).isoformat()


class PairLedger:
    """NONE/F1 두 실행 전에 동일한 승인 prefix를 고정하는 append-only 원장."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        states: dict[str, str] = {}
        allowed = {None: "proposed", "proposed": "approved", "approved": "consumed",
                   "consumed": "receipt"}
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("probe pair 원장 행이 object가 아니다")
            request_id = row.get("request_id")
            event = row.get("event")
            if (row.get("schema") != PAIR_LEDGER_SCHEMA
                    or not isinstance(request_id, str) or event != allowed.get(states.get(request_id))):
                raise ValueError("probe pair 원장 순서 또는 schema가 다르다")
            states[request_id] = str(event)
        return rows

    def _append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema": PAIR_LEDGER_SCHEMA, "at": _at(), **dict(row)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        self.events()
        return body

    def propose(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("schema") != ADAPTER.PAIR_SCHEMA:
            raise ValueError("probe pair request schema가 다르다")
        existing = [row for row in self.events() if row.get("event") == "proposed"]
        if len(existing) >= 2:
            raise ValueError("probe pair 상한 2를 넘었다")
        request_id = digest({"request": request, "ordinal": len(existing) + 1})[:16]
        self._append({"event": "proposed", "request_id": request_id,
                      "request": dict(request)})
        return {"status": "approval_required", "request_id": request_id,
                "request_sha256": digest(request)}

    def approve_and_consume(self, request_id: str, approver: str) -> dict[str, Any]:
        if not isinstance(approver, str) or not approver.strip():
            raise ValueError("probe pair approver가 비어 있다")
        rows = [row for row in self.events() if row.get("request_id") == request_id]
        if len(rows) != 1 or rows[0].get("event") != "proposed":
            raise ValueError("probe pair 승인은 proposed 상태에서만 가능하다")
        request_sha = digest(rows[0]["request"])
        approved = self._append({"event": "approved", "request_id": request_id,
                                 "request_sha256": request_sha, "approver": approver.strip()})
        consumed = self._append({"event": "consumed", "request_id": request_id,
                                 "request_sha256": request_sha})
        prefix = [rows[0], approved, consumed]
        return {"path": str(self.path.resolve()), "sha256": ADAPTER.file_sha(self.path),
                "request_id": request_id, "prefix_sha256": ADAPTER.digest(prefix)}

    def authorize(self, request: Mapping[str, Any], approver: str) -> dict[str, Any]:
        proposed = self.propose(request)
        return self.approve_and_consume(str(proposed["request_id"]), approver)

    def consumed_reference(self, request_id: str) -> dict[str, Any]:
        rows = [row for row in self.events() if row.get("request_id") == request_id]
        if [row["event"] for row in rows[:3]] != ["proposed", "approved", "consumed"]:
            raise ValueError("probe pair consumed prefix가 다르다")
        raw_lines = self.path.read_bytes().splitlines(keepends=True)
        consumed_index = None
        for index, line in enumerate(raw_lines):
            raw_event = json.loads(line)
            if raw_event.get("request_id") == request_id and raw_event.get("event") == "consumed":
                consumed_index = index
                break
        if consumed_index is None:
            raise ValueError("probe pair consumed 원문이 없다")
        prefix_bytes = b"".join(raw_lines[:consumed_index + 1])
        return {"path": str(self.path.resolve()),
                "sha256": hashlib.sha256(prefix_bytes).hexdigest(),
                "request_id": request_id, "prefix_sha256": ADAPTER.digest(rows[:3])}

    def receipt(self, request_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        rows = [row for row in self.events() if row.get("request_id") == request_id]
        if [row["event"] for row in rows] != ["proposed", "approved", "consumed"]:
            raise ValueError("probe pair receipt 전 prefix가 다르다")
        return self._append({"event": "receipt", "request_id": request_id,
                             "result_sha256": digest(result), "status": result.get("status")})

    def completed_reference(self, request_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        rows = [row for row in self.events() if row.get("request_id") == request_id]
        if ([row["event"] for row in rows] != ["proposed", "approved", "consumed", "receipt"]
                or rows[-1].get("result_sha256") != digest(result)
                or rows[-1].get("status") != result.get("status")):
            raise ValueError("probe pair 완료 receipt가 결과와 다르다")
        return self.consumed_reference(request_id)


class E2ECaller:
    def __init__(self, root: Path, plan_path: Path, adapter_context: Mapping[str, Any], runtime: Path):
        self.root = Path(root).resolve(strict=True)
        self.plan_path = Path(plan_path).resolve(strict=True)
        self.plan = _read(self.plan_path)
        if (self.plan.get("schema") != CONTRACT.SCHEMA or self.plan.get("api_cap") != CONTRACT.API_CAP
                or len(self.plan.get("rows", [])) != 2):
            raise ValueError("E2E plan이 두 target/API32 계약과 다르다")
        self.context = dict(adapter_context)
        groups = self.context.get("protocol", {}).get("groups", [])
        call_repository, _, call_manifest = CALL._plan(self.plan["selection_plan"])
        selection_manifest_path = Path(self.plan["selection_plan"]["path"])
        plan_refs = self.context["protocol"]["plan_references"]
        registered_source = ADAPTER._file_ref(self.root, plan_refs["source"], "source E2E plan")
        registered_selection = ADAPTER._file_ref(
            self.root, plan_refs["selection"], "selection plan",
        )
        if (call_repository != self.root or call_manifest.get("api_cap") != CONTRACT.API_CAP
                or self.plan.get("source_anchor") != self.context["protocol"].get("source_anchor")
                or registered_source != self.plan_path or registered_selection != selection_manifest_path
                or ADAPTER.file_sha(self.plan_path) != self.context["protocol"]["source_plan_sha256"]
                or ADAPTER.file_sha(selection_manifest_path)
                != self.context["protocol"]["selection_plan_sha256"]):
            raise ValueError("selection plan이 현재 repository/API32와 다르다")
        for row in self.plan["rows"]:
            identities = {row.get("source_job_id"), row.get("record_job_id"),
                          row.get("source_lineage_id"), row.get("corrected_job_id")}
            registered = [item for item in groups
                          if isinstance(item, Mapping)
                          and item.get("group") == row.get("target", {}).get("seed_group")]
            _, call_row, frozen = CALL._row(self.root, call_manifest, row.get("call_id"))
            if (len(identities) != 4 or None in identities or len(registered) != 1
                    or call_row.get("capsule_sha256") != row.get("capsule_sha256")
                    or frozen.get("checkpoint_sha256") != row.get("checkpoint_sha256")
                    or registered[0].get("seed") != row.get("source_seed")
                    or registered[0].get("checkpoints") != {
                        "NONE": row.get("reference_checkpoint_sha256"),
                        "F1": row.get("checkpoint_sha256"),
                    }
                    or registered[0].get("source_protocol_sha256_lf")
                    != row.get("source_protocol_sha256_lf")
                    or registered[0].get("corrected_job_id") != row.get("corrected_job_id")):
                raise ValueError("E2E plan의 job/lineage/checkpoint가 execution protocol과 다르다")
        self.runtime = Path(runtime).resolve()
        expected = self.root / ".triage-runtime/vision-selection-e2e" / self.plan["execution_id"]
        if self.runtime != expected.resolve():
            raise ValueError("E2E caller runtime이 canonical 경로가 아니다")
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.pairs = PairLedger(self.runtime / "probe_pairs.ledger.jsonl")
        self.corrections = CorrectionLedger(self.runtime / "corrections.ledger.jsonl")

    def _row(self, group: str) -> dict[str, Any]:
        return CONTRACT._row(self.plan, group)

    def _selection_path(self, group: str) -> Path:
        return self.runtime / "selections" / f"{group}.json"

    def _validated_selection(self, group: str) -> dict[str, Any]:
        path = self._selection_path(group)
        if path.is_symlink() or not path.is_file():
            raise ValueError("selection cache 원문이 없다")
        cached = ADAPTER._read(path)
        row = self._row(group)
        replay = CALL.run(
            self.plan["selection_plan"], row["call_id"],
            lambda *_: (_ for _ in ()).throw(AssertionError("terminal cache provider 호출")),
        )["journal"]
        if not isinstance(replay, Mapping) or not isinstance(replay.get("engine_result"), Mapping):
            raise TypeError("cached selection terminal engine이 없다")
        engine = replay["engine_result"]
        selection = engine.get("selection")
        _, stage, _ = CALL._plan(self.plan["selection_plan"])
        terminal = stage / "decisions" / row["call_id"] / "terminal.json"
        bridge = self.runtime / "bridges" / f"{group}.json"
        if (terminal.is_symlink() or not terminal.is_file() or bridge.is_symlink()
                or not bridge.is_file() or not isinstance(selection, Mapping)
                or cached.get("model_status") != engine.get("model_status")
                or any(cached.get(key) != value for key, value in selection.items())
                or cached.get("terminal_sha256") != ADAPTER.file_sha(terminal)
                or cached.get("activation_bridge_sha256") != ADAPTER.file_sha(bridge)
                or cached.get("calls") != len(engine.get("calls", []))):
            raise ValueError("cached selection이 terminal/activation 원문과 다르다")
        return cached

    def _expected_probe_source(self) -> dict[str, Any]:
        return ADAPTER.EXECUTION._probe_execution_source(self.root, {
            "source_anchor": self.plan["source_anchor"],
            "protocol": self.context["protocol"]["probe_execution"],
        })

    def _cached_correction(self, group: str) -> dict[str, Any] | None:
        path = self.runtime / "corrections" / f"{group}.json"
        if not path.is_file():
            return None
        result = ADAPTER.validate_correction_result(self.context, group, _read(path))
        row = self._row(group)
        requests = [request for request in self.corrections.requests().values()
                    if request.get("evidence", {}).get("source_job_id") == row["source_job_id"]]
        if len(requests) != 1 or requests[0].get("state") != "done":
            raise ValueError("cached correction의 완료 원장이 유일하지 않다")
        receipt = requests[0].get("receipt")
        recovered = (result["fixed_eval"]["none"] == "healthy"
                     and result["fixed_eval"]["corrected"] == "healthy")
        if (not isinstance(receipt, Mapping) or receipt.get("tag") != group
                or receipt.get("result_sha") != digest(result)
                or receipt.get("gpu_s") != result["gpu_s"] or receipt.get("exit_code") != 0
                or receipt.get("verdict") != ("recovered" if recovered else "not_recovered")):
            raise ValueError("cached correction receipt가 결과와 다르다")
        return result

    def _record_job(self, group: str, correction: Mapping[str, Any]) -> dict[str, Any]:
        plan_ref = self.context["protocol"]["execution_bindings"]["corrected_render_plan"]
        _, render_plan = ADAPTER._ref(self.root, plan_ref, "corrected render plan")
        jobs = [row for row in render_plan.get("groups", [])
                if isinstance(row, Mapping) and row.get("group") == group]
        if len(jobs) != 1 or not isinstance(jobs[0].get("job"), Mapping):
            raise ValueError("corrected render plan에 group job이 유일하지 않다")
        record_job = dict(jobs[0]["job"])
        checkpoint = record_job.get("checkpoint")
        if (not isinstance(checkpoint, Mapping)
                or checkpoint.get("sha256") != correction["checkpoint_sha256"]
                or record_job.get("source_job_id") != self._row(group)["corrected_job_id"]):
            raise ValueError("record job checkpoint/lineage가 correction과 다르다")
        return record_job

    def run_initial(self, group: str, complete: Callable[[list[dict], dict], dict], *,
                    live_snapshot: dict, live_contract: Path, source_binding: dict) -> dict[str, Any]:
        ADAPTER.refresh_context(self.context)
        selection_path = self._selection_path(group)
        if selection_path.is_file():
            return self._validated_selection(group)
        row = self._row(group)
        result = CALL.run(self.plan["selection_plan"], row["call_id"], complete)
        journal = result.get("journal")
        if not isinstance(journal, dict) or journal.get("status") not in {"choice_ready", "terminal_hold"}:
            return {"status": "pending", "journal": journal}
        activated = CALL.activate(
            self.plan["selection_plan"], row["call_id"], live_snapshot, live_contract,
            source_binding, self.runtime / "bridges" / f"{group}.json",
        )
        engine = journal.get("engine_result")
        if not isinstance(engine, Mapping):
            raise TypeError("terminal engine 결과가 없다")
        selection = engine.get("selection")
        if not isinstance(selection, Mapping):
            raise TypeError("terminal selection이 없다")
        body = {"group": group, "model_status": engine.get("model_status"), **dict(selection),
                "terminal_sha256": activated["terminal_sha256"],
                "activation_bridge_sha256": ADAPTER.file_sha(Path(activated["bridge_reference"]["path"])),
                "calls": len(engine.get("calls", []))}
        _write_once(selection_path, body)
        return body

    def _pair_request(self, group: str, selection: Mapping[str, Any],
                      commands: Mapping[str, Any]) -> dict[str, Any]:
        row = self._row(group)
        registered = [item for item in self.context["protocol"]["groups"] if item.get("group") == group]
        if len(registered) != 1:
            raise ValueError("execution protocol group이 유일하지 않다")
        entry = registered[0]
        if (selection.get("model_status") != "valid" or selection.get("action") != "continue"
                or selection.get("probe") not in row["eligible_probes"]):
            raise ValueError("유효한 모델 선택만 pair 승인할 수 있다")
        return {"schema": ADAPTER.PAIR_SCHEMA, "group": group, "probe": selection["probe"],
                "seed": row["source_seed"], "arms": ["NONE", "F1"],
                "checkpoints": {"NONE": row["reference_checkpoint_sha256"],
                                "F1": row["checkpoint_sha256"]},
                "selection": {"terminal_sha256": selection["terminal_sha256"],
                              "activation_bridge_sha256": selection["activation_bridge_sha256"]},
                "protocols": {"source": row["source_protocol_sha256_lf"],
                              "probe": self.context["protocol"]["probe_execution"]},
                "execution": entry["execution"], "commands": dict(commands),
                "commands_sha256": digest(commands),
                "budget_gpu_s": entry["probe_budget_gpu_s"],
                "resource_sha256": self.context["lease"]["resource"]["sha256"]}

    def run_probe_pair(self, group: str, commands: Mapping[str, Mapping[str, Any]],
                       expected_source: Mapping[str, Any], *, approver: str | None = None,
                       runner: Callable[..., Any] = ADAPTER.subprocess.run) -> dict[str, Any]:
        ADAPTER.refresh_context(self.context)
        if dict(expected_source) != self._expected_probe_source():
            raise ValueError("probe expected source가 등록 plan/protocol과 다르다")
        selection = self._validated_selection(group)
        state_path = self.runtime / "probe_pairs" / f"{group}.json"
        if state_path.is_file():
            cached_state = _read(state_path)
            proposed = [row for row in self.pairs.events()
                        if row.get("event") == "proposed"
                        and row.get("request", {}).get("group") == group]
            if len(proposed) != 1:
                raise ValueError("cached probe pair 승인 request가 유일하지 않다")
            request = proposed[0]["request"]
            approved_commands = request.get("commands")
            if not isinstance(approved_commands, Mapping):
                raise ValueError("cached probe pair 승인 command가 없다")
            expected = self._pair_request(group, selection, approved_commands)
            if request != expected:
                raise ValueError("cached probe pair request가 현재 selection과 다르다")
            if cached_state.get("request_sha256") != digest(request):
                raise ValueError("저장 probe pair 상태가 현재 선택·checkpoint와 다르다")
            reference = self.pairs.completed_reference(proposed[0]["request_id"], cached_state)
            return ADAPTER.validate_pair_result(
                self.context, request, cached_state, expected_source, reference["sha256"],
            )
        cached_path = self.runtime / "probes" / group / "result.json"
        if cached_path.is_file():
            cached = _read(cached_path)
            return {"schema": ADAPTER.PAIR_RESULT_SCHEMA, "status": "HOLD",
                    "reason": "caller_receipt_missing",
                    "request_sha256": cached.get("request_sha256")}
        stage = self.runtime / "probes" / group
        if (stage.exists() and any(stage.iterdir())
                and not (stage / "partial.json").is_file()):
            return {"schema": ADAPTER.PAIR_RESULT_SCHEMA, "status": "HOLD",
                    "reason": "partial_stage"}
        if set(commands) != {"NONE", "F1"}:
            raise ValueError("probe commands는 NONE/F1 exact pair여야 한다")
        normalized: dict[str, dict[str, Any]] = {}
        for arm in ("NONE", "F1"):
            spec = commands[arm]
            if (not isinstance(spec, Mapping) or set(spec) != {"argv", "descriptor"}
                    or not isinstance(spec["argv"], list)
                    or any(not isinstance(part, str) or not part for part in spec["argv"])):
                raise ValueError("probe command 형식이 다르다")
            descriptor = Path(str(spec["descriptor"])).resolve()
            descriptor.relative_to(stage.resolve())
            normalized[arm] = {"argv": list(spec["argv"]), "descriptor": str(descriptor)}
        request = self._pair_request(group, selection, normalized)
        proposed = [row for row in self.pairs.events()
                    if row.get("event") == "proposed"
                    and row.get("request", {}).get("group") == group]
        if not proposed:
            pending = self.pairs.propose(request)
            return {"schema": ADAPTER.PAIR_RESULT_SCHEMA, "status": "HOLD",
                    "reason": "approval_required", **pending}
        if len(proposed) != 1 or proposed[0].get("request") != request:
            raise ValueError("대기 probe pair command가 현재 등록 command와 다르다")
        pair_rows = [row for row in self.pairs.events()
                     if row.get("request_id") == proposed[0]["request_id"]]
        if [row["event"] for row in pair_rows] == ["proposed"]:
            reference = self.pairs.approve_and_consume(proposed[0]["request_id"],
                                                        str(approver or ""))
        elif [row["event"] for row in pair_rows] == ["proposed", "approved", "consumed"]:
            reference = self.pairs.consumed_reference(proposed[0]["request_id"])
            return ADAPTER.run_probe_pair(
                self.context, reference, normalized, expected_source, stage, runner=runner,
            )
        else:
            raise ValueError("probe pair 원장과 완료 state 경계가 다르다")
        result = ADAPTER.run_probe_pair(
            self.context, reference, normalized, expected_source,
            stage, runner=runner,
        )
        if result.get("status") != "complete":
            return result
        self.pairs.receipt(reference["request_id"], result)
        _write_once(state_path, result)
        return result

    def propose_correction(self, group: str, pair: Mapping[str, Any]) -> dict[str, Any] | None:
        selection = _read(self._selection_path(group))
        contract_pair = {key: pair.get(key) for key in ("probe", "none", "f1")}
        candidate = CONTRACT.approval_candidate(self.plan, group, selection, contract_pair)
        if pair.get("status") != "complete" or candidate is None:
            return None
        proposed = [event for event in self.corrections.events() if event.get("event") == "proposed"]
        if len(proposed) >= 2:
            raise ValueError("CORRECTED 상한 2를 넘었다")
        evidence = {**candidate, "pair_result_sha256": digest(pair)}
        loop_state = digest({"group": group, "pair": pair, "selection": selection})
        return self.corrections.propose(
            loop_state, "reward", digest(pair), "restore_body_frame_reward", "restart",
            {"case": "case_01", "arm": "corrected"}, 3600, evidence=evidence,
        )

    def approve_correction(self, request_id: str, approver: str) -> dict[str, Any]:
        """교정 승인을 실행과 분리해 remaining_commands가 소비 전 상태를 재검산하게 한다."""
        if not isinstance(approver, str) or not approver.strip():
            raise ValueError("correction approver가 비어 있다")
        return self.corrections.approve(request_id, approver)

    def run_approved_correction(self, group: str, request_id: str, *,
                                backend_reference: Mapping[str, Any],
                                train: Callable[..., dict[str, Any]],
                                evaluate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        ADAPTER.refresh_context(self.context)
        cached_path = self.runtime / "corrections" / f"{group}.json"
        cached = self._cached_correction(group)
        if cached is not None:
            return cached
        request = self.corrections.requests().get(request_id)
        if not isinstance(request, dict) or request.get("state") not in {"approved", "running"}:
            raise ValueError("승인되거나 실행 중인 교정 요청이 아니다")
        registered_backend = self.context["protocol"]["execution_bindings"]["correction_backend"]
        if dict(backend_reference) != registered_backend:
            raise ValueError("주입된 correction backend가 execution protocol과 다르다")
        service = getattr(train, "__self__", None)
        if service is None or getattr(evaluate, "__self__", None) is not service:
            raise ValueError("train/evaluate가 같은 correction service가 아니다")
        ADAPTER.verify_correction_service(self.context, service)
        if service.preflight() is not True:
            return {"schema": "vision_selection_correction_hold_v1", "status": "HOLD",
                    "reason": "source_none_cache_missing", "group": group,
                    "job_id": self._row(group)["corrected_job_id"], "gpu_s": 0.0}
        entry = next(row for row in self.context["protocol"]["groups"]
                     if row["group"] == group)
        ADAPTER.require_action_window(self.context, entry, "correction")
        if request["state"] == "approved":
            self.corrections.consume(request_id, request["loop_state_digest"])
        reference = {"path": str(self.corrections.path.resolve()),
                     "sha256": ADAPTER.file_sha(self.corrections.path),
                     "request_id": request_id, "loop_state_digest": request["loop_state_digest"]}
        result = ADAPTER.run_correction(
            self.context, group, reference, self._row(group)["corrected_job_id"], train, evaluate,
        )
        if result.get("status") != "complete":
            return result
        recovered = (result["fixed_eval"]["none"] == "healthy"
                     and result["fixed_eval"]["corrected"] == "healthy")
        self.corrections.receipt(request_id, group, "recovered" if recovered else "not_recovered",
                                 digest(result), result["gpu_s"], result["exit_code"])
        _write_once(cached_path, result)
        return result

    def approve_and_run_correction(self, group: str, request_id: str, *, approver: str,
                                   backend_reference: Mapping[str, Any],
                                   train: Callable[..., dict[str, Any]],
                                   evaluate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        ADAPTER.refresh_context(self.context)
        cached = self._cached_correction(group)
        if cached is not None:
            return cached
        request = self.corrections.requests().get(request_id)
        if not isinstance(request, dict) or request.get("state") != "pending":
            raise ValueError("대기 중인 교정 요청이 아니다")
        registered_backend = self.context["protocol"]["execution_bindings"]["correction_backend"]
        if dict(backend_reference) != registered_backend:
            raise ValueError("주입된 correction backend가 execution protocol과 다르다")
        self.approve_correction(request_id, approver)
        return self.run_approved_correction(
            group, request_id, backend_reference=backend_reference, train=train, evaluate=evaluate,
        )

    def run_post_observation(self, group: str, *,
                             record: Callable[[Mapping[str, Any]], dict[str, Any]],
                             record_plan_reference: Mapping[str, Any],
                             observation_input: Callable[[dict[str, Any]], tuple[Path, dict[str, Any]]],
                             complete: Callable[[list[dict], dict], dict],
                             verifier_reference: Mapping[str, Any],
                             ) -> dict[str, Any]:
        ADAPTER.refresh_context(self.context)
        cached_path = self.runtime / "observations" / f"{group}.json"
        registered_verifier = self.context["protocol"]["execution_bindings"]["record_verifier"]
        if dict(verifier_reference) != registered_verifier:
            raise ValueError("주입된 record verifier가 execution protocol과 다르다")
        registered_plan = self.context["protocol"]["execution_bindings"]["corrected_render_plan"]
        if dict(record_plan_reference) != registered_plan:
            raise ValueError("주입된 corrected render plan이 execution protocol과 다르다")
        ADAPTER.verify_record_service(self.context, record)
        correction = self._cached_correction(group)
        if correction is None:
            raise ValueError("후속 관찰 전에 완료·검증된 correction이 없다")
        materialized: dict[str, Any] = {}
        def observe(checked: dict[str, Any]) -> dict[str, Any]:
            contract_path, binding = observation_input(checked)
            expected_path = self.runtime / "post-observation" / group / "evidence" / "contract.json"
            if Path(contract_path).resolve(strict=True) != expected_path.resolve(strict=True):
                raise ValueError("후속 V01 계약이 canonical runtime 경로가 아니다")
            expected_binding = OBSERVER.binding_template(
                contract_path, run_id=f"{self.plan['execution_id']}-post-{group}",
                source_anchor=self.plan["source_anchor"],
                protocol_sha256=self.context["protocol_sha256"],
            )
            if binding != expected_binding:
                raise ValueError("후속 관찰 binding이 record V01에서 결정적으로 생성되지 않았다")
            materialized.update({
                "contract": {"path": Path(contract_path).relative_to(self.root).as_posix(),
                             "sha256": ADAPTER.file_sha(contract_path)},
                "binding": dict(binding),
            })
            verified = OBSERVER.checked_binding(contract_path, binding)
            journal = JOURNAL.DecisionJournal(
                self.runtime / "post-observation" / group / "decision",
                CALL._plan(self.plan["selection_plan"])[1] / "api",
                verified, OBSERVER.REQUEST_PARAMS, retry_delay_s=30, api_cap=CONTRACT.API_CAP,
            )
            result = journal.advance(
                lambda completion: OBSERVER.engine(contract_path, completion), complete,
                lambda: datetime.now(UTC_TZ),
            )
            if result.get("engine_result") is None:
                return {"status": result["status"], "used_for_decision": False,
                        "api": result["api"]}
            engine = result["engine_result"]
            return {"status": "observation_ready" if engine.get("status") == "observation_ready"
                    else result["status"], "engine_result": engine, "api": result["api"],
                    "used_for_decision": False}
        record_job = self._record_job(group, correction)
        decision_root = self.runtime / "post-observation" / group / "decision"
        if cached_path.is_file():
            return self._validated_observation(group, correction)
        result = ADAPTER.run_post_evidence(
            self.context, correction, record_job, record,
            observe,
        )
        terminal = decision_root / "terminal.json"
        if result.get("status") == "complete":
            if not terminal.is_file():
                raise ValueError("후속 관찰 terminal 원문이 없다")
            if set(materialized) != {"contract", "binding"}:
                raise ValueError("후속 관찰 V01/binding 원문이 없다")
            result = {**result, "record_job_sha256": digest(record_job),
                      "post_terminal_sha256": ADAPTER.file_sha(terminal),
                      "observation_contract": materialized["contract"],
                      "observation_binding": materialized["binding"]}
        _write_once(cached_path, result)
        return result

    def _validated_observation(self, group: str,
                               correction: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cached = ADAPTER._read(self.runtime / "observations" / f"{group}.json")
        required = {"status", "record", "observation", "used_for_decision",
                    "record_job_sha256", "post_terminal_sha256",
                    "observation_contract", "observation_binding"}
        if (set(cached) != required or cached.get("status") != "complete"
                or cached.get("used_for_decision") is not False):
            raise ValueError("cached observation schema가 다르다")
        verified_correction = correction or self._cached_correction(group)
        if verified_correction is None:
            raise ValueError("관찰 cache 앞에 검증된 correction이 없다")
        record_job = self._record_job(group, verified_correction)
        terminal = self.runtime / "post-observation" / group / "decision" / "terminal.json"
        if (terminal.is_symlink() or not terminal.is_file()
                or cached.get("record_job_sha256") != digest(record_job)
                or cached.get("post_terminal_sha256") != ADAPTER.file_sha(terminal)):
            raise ValueError("cached observation job/terminal 결합이 다르다")
        checked_record = ADAPTER.validate_record_result(
            self.context, cached.get("record"), record_job=record_job,
        )
        del checked_record
        contract_path = ADAPTER._file_ref(
            self.root, cached.get("observation_contract"), "cached observation contract",
        )
        expected_path = self.runtime / "post-observation" / group / "evidence" / "contract.json"
        if contract_path != expected_path.resolve(strict=True):
            raise ValueError("cached observation contract 경로가 canonical하지 않다")
        binding = cached.get("observation_binding")
        expected_binding = OBSERVER.binding_template(
            contract_path, run_id=f"{self.plan['execution_id']}-post-{group}",
            source_anchor=self.plan["source_anchor"],
            protocol_sha256=self.context["protocol_sha256"],
        )
        if binding != expected_binding:
            raise ValueError("cached observation binding이 V01과 다르다")
        verified = OBSERVER.checked_binding(contract_path, expected_binding)
        journal = JOURNAL.DecisionJournal(
            self.runtime / "post-observation" / group / "decision",
            CALL._plan(self.plan["selection_plan"])[1] / "api",
            verified, OBSERVER.REQUEST_PARAMS, retry_delay_s=30, api_cap=CONTRACT.API_CAP,
        )
        replay = journal.advance(
            lambda completion: OBSERVER.engine(contract_path, completion),
            lambda *_: (_ for _ in ()).throw(AssertionError("cached VL provider 호출")),
            lambda: datetime.now(UTC_TZ),
        )
        if replay.get("engine_result") is None:
            replayed = {"status": replay["status"], "used_for_decision": False,
                        "api": replay["api"]}
        else:
            engine = replay["engine_result"]
            replayed = {
                "status": "observation_ready" if engine.get("status") == "observation_ready"
                else replay["status"],
                "engine_result": engine, "api": replay["api"], "used_for_decision": False,
            }
        if cached.get("observation") != replayed:
            raise ValueError("cached observation이 shared API journal replay와 다르다")
        return cached

    def api_requests(self) -> int:
        _, stage, manifest = CALL._plan(self.plan["selection_plan"])
        if manifest.get("api_cap") != CONTRACT.API_CAP:
            raise ValueError("E2E API manifest가 cap 32와 다르다")
        rows = list(manifest["rows"])
        if not rows:
            return 0
        _, row, frozen = CALL._row(self.root, manifest, rows[0]["call_id"])
        return int(CALL._journal(stage, manifest, row, frozen)._audit_api()["reserved"])

    def inspect(self) -> dict[str, Any]:
        """원문 파일과 append-only 원장에서 현재 순차 실행 상태를 다시 만든다."""
        state: dict[str, Any] = {"api_used": self.api_requests(), "selections": {}, "probes": {},
                                 "approvals": {}, "corrections": {}, "observations": {}}
        requests = self.corrections.requests()
        for row in self.plan["rows"]:
            group = row["target"]["seed_group"]
            paths = {
                "selections": self._selection_path(group),
                "probes": self.runtime / "probe_pairs" / f"{group}.json",
                "corrections": self.runtime / "corrections" / f"{group}.json",
                "observations": self.runtime / "observations" / f"{group}.json",
            }
            for key, path in paths.items():
                if path.is_file():
                    value = _read(path)
                    if key == "selections":
                        value = self._validated_selection(group)
                    elif key == "probes":
                        selection = _read(self._selection_path(group))
                        proposed = [event for event in self.pairs.events()
                                    if event.get("event") == "proposed"
                                    and event.get("request", {}).get("group") == group]
                        if len(proposed) != 1:
                            raise ValueError("cached probe pair 승인 request가 유일하지 않다")
                        request = proposed[0]["request"]
                        approved_commands = request.get("commands")
                        if (not isinstance(approved_commands, Mapping)
                                or request != self._pair_request(
                                    group, selection, approved_commands)):
                            raise ValueError("cached probe pair request가 현재 selection과 다르다")
                        reference = self.pairs.completed_reference(proposed[0]["request_id"], value)
                        checked = ADAPTER.validate_pair_result(
                            self.context, request, value, self._expected_probe_source(),
                            reference["sha256"],
                        )
                        value = {name: checked[name]
                                 for name in ("probe", "none", "f1", "gpu_s")}
                    elif key == "corrections":
                        value = self._cached_correction(group)
                    elif key == "observations":
                        value = self._validated_observation(group)
                    state[key][group] = value
            if group not in state["probes"]:
                proposed = [event for event in self.pairs.events()
                            if event.get("event") == "proposed"
                            and event.get("request", {}).get("group") == group]
                if len(proposed) == 1:
                    request_id = proposed[0]["request_id"]
                    pair_rows = [event for event in self.pairs.events()
                                 if event.get("request_id") == request_id]
                    if [event["event"] for event in pair_rows] == [
                            "proposed", "approved", "consumed"]:
                        partial_path = self.runtime / "probes" / group / "partial.json"
                        if not partial_path.is_file():
                            raise ValueError("consumed partial probe evidence가 없다")
                        reference = self.pairs.consumed_reference(request_id)
                        state["probes"][group] = ADAPTER.validate_partial_pair_result(
                            self.context, proposed[0]["request"], _read(partial_path),
                            self._expected_probe_source(), reference["sha256"],
                        )
            matches = [request for request in requests.values()
                       if request.get("evidence", {}).get("source_job_id") == row["source_job_id"]]
            if len(matches) > 1:
                raise ValueError("seed group에 교정 요청이 둘 이상 결합됐다")
            if matches:
                request = matches[0]
                approval_state = {"pending": "pending", "approved": "approved",
                                  "running": "consumed", "done": "consumed"}[request["state"]]
                evidence = {field: request["evidence"][field]
                            for field in CONTRACT.APPROVAL_FIELDS}
                state["approvals"][group] = {"request_id": request["request_id"],
                                              "state": approval_state, "evidence": evidence}
        return state

    def remaining_commands(self) -> list[dict[str, Any]]:
        return CONTRACT.commands(self.plan, self.inspect())

    def advance(self, command: Mapping[str, Any], services: Mapping[str, Any]) -> dict[str, Any] | None:
        """pure contract가 낸 현재 명령 하나만 canonical writer 경계로 dispatch한다."""
        requested = dict(command)
        if requested not in self.remaining_commands():
            raise ValueError("현재 상태에서 허용되지 않은 E2E 명령이다")
        kind, group = requested["kind"], requested["group"]
        if kind == "initial_selection":
            return self.run_initial(
                group, services["complete"], live_snapshot=services["live_snapshot"],
                live_contract=services["live_contract"], source_binding=services["source_binding"],
            )
        if kind == "probe_pair":
            return self.run_probe_pair(
                group, services["commands"], services["expected_source"],
                approver=services["approver"], runner=services.get("runner", ADAPTER.subprocess.run),
            )
        if kind == "propose_correction":
            pair = _read(self.runtime / "probe_pairs" / f"{group}.json")
            return self.propose_correction(group, pair)
        if kind == "consume_and_run_correction":
            approval = self.inspect()["approvals"][group]
            return self.run_approved_correction(
                group, approval["request_id"], backend_reference=services["backend_reference"],
                train=services["train"], evaluate=services["evaluate"],
            )
        if kind == "post_observation":
            return self.run_post_observation(
                group, record=services["record"],
                record_plan_reference=services["record_plan_reference"],
                observation_input=services["observation_input"], complete=services["complete"],
                verifier_reference=services["verifier_reference"],
            )
        raise ValueError(f"지원하지 않는 E2E 명령 {kind!r}")

    def summarize(self) -> dict[str, Any]:
        state = self.inspect()
        quote = float(self.context["protocol"]["resources"]["compute_quote_usd_per_h"])
        compute_usd = (
            sum(float(row.get("compute_usd", 0.0)) for row in state["corrections"].values())
            + sum(float(row.get("gpu_s", 0.0)) * quote / 3600.0
                  for row in state["probes"].values())
            + sum(float(row.get("record", {}).get("compute_usd", 0.0))
                  for row in state["observations"].values())
        )
        if compute_usd > float(self.context["protocol"]["resources"]["whole_batch_budget_usd"]):
            raise ValueError("검증된 E2E 실행 비용이 전체 등록 예산을 넘었다")
        return {
            "schema": STATE_SCHEMA, "execution_id": self.plan["execution_id"],
            "api_used": state["api_used"],
            "selection_count": len(state["selections"]),
            "probe_pair_count": len(state["probes"]),
            "correction_count": len(state["corrections"]),
            "observation_count": len(state["observations"]),
            "verified_compute_usd": compute_usd,
            "remaining_commands": self.remaining_commands(),
        }
