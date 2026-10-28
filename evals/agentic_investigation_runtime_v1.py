"""불변 CPU 조사 기록과 명시적 host callback을 연결한다. 기본 provider/승인은 없다."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import agentic_investigation_call_v2 as CALL
import agentic_investigation_evidence_v1 as B
import agentic_investigation_loop_v1 as L
from run_agentic_development_v2 import verify_sources

A, C, J = CALL.A, CALL.A.C, CALL.J
SCHEMA = "agentic_investigation_runtime_v1"
ROOT = Path(__file__).resolve().parents[1]


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    J._read(path)
    return {"path": str(path), "sha256": J.file_sha(path)}


def read_ref(value: dict) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "reference fields")
    path = Path(value["path"])
    require(path == path.resolve() and J.file_sha(path) == value["sha256"], "reference SHA/path")
    return J._read(path)


def same_or_new(path: Path, value: dict) -> None:
    if path.exists():
        require(J._read(path) == value, "immutable record changed: " + path.name)
    else:
        J._write_new(path, value)


def capture_api(api_root: Path, *, baseline: int = 7) -> dict:
    """사전등록용 기존 API12 prefix를 읽는다. 새 API root를 만들지 않는다."""
    api = api_root.resolve(strict=True)
    require(type(baseline) is int and 0 <= baseline <= 12, "API baseline")
    require(J._read(api / "manifest.json") == {"schema": J.API_SCHEMA, "api_cap": 12}, "API cap12")
    names = sorted(p.name for p in (api / "reservations").glob("*.json"))
    require(names == [f"{i:04d}.json" for i in range(1, baseline + 1)], "API baseline count")
    rows = {"manifest.json": ref(api / "manifest.json")}
    previous = None
    for name in names:
        reservation, response, receipt = (J._read(api / kind / name)
                                          for kind in ("reservations", "responses", "receipts"))
        require(reservation["index"] == response["index"] == receipt["index"] == int(name[:4])
                and response["reservation_sha256"] == receipt["reservation_sha256"]
                == J.file_sha(api / "reservations" / name)
                and receipt["response_sha256"] == J.file_sha(api / "responses" / name)
                and receipt["previous_receipt_sha256"] == previous, "API prefix chain")
        previous = J.file_sha(api / "receipts" / name)
        for kind in ("reservations", "responses", "receipts"):
            rows[f"{kind}/{name}"] = ref(api / kind / name)
    require(sorted(p.name for p in (api / "responses").glob("*.json")) == names
            == sorted(p.name for p in (api / "receipts").glob("*.json")), "API pending baseline")
    return {"api_root": str(api), "api_cap": 12, "baseline_reserved": baseline, "prefix": rows}


def verify_api(seal: dict) -> Path:
    api = Path(seal["api_root"])
    n = seal["baseline_reserved"]
    require(api == api.resolve() and seal["api_cap"] == 12 and type(n) is int and 0 <= n <= 12,
            "registered API12 binding")
    expected = {"manifest.json"} | {f"{kind}/{i:04d}.json"
        for kind in ("reservations", "responses", "receipts") for i in range(1, n + 1)}
    require(set(seal["prefix"]) == expected, "API prefix list")
    for name, value in seal["prefix"].items():
        require(Path(value["path"]) == api / name, "API root reset/move")
        read_ref(value)
    require(J._read(api / "manifest.json") == {"schema": J.API_SCHEMA, "api_cap": 12}, "API cap reset")
    return api


def completed(reference: dict, api_root: Path) -> dict:
    """기존 journal의 terminal을 원장 응답으로 재계산한다. 파일 작성/API 호출은 없다."""
    folder, capsule = CALL.prepared(reference)
    manifest = J._read(folder / "journal/manifest.json")
    binding = {"run_id": "agentic-v2-" + reference["sha256"][:24],
        "source_anchor": capsule["source_anchor"], "protocol_sha256": capsule["protocol_sha256"],
        "checkpoint_sha256": capsule["context"]["snapshot"]["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"],
        "snapshot_sha256": reference["sha256"], "initial_messages_sha256": J.digest(capsule["messages"])}
    require(manifest == {"schema": J.SCHEMA, "binding": binding, "binding_sha256": J.digest(binding),
        "params": capsule["params"], "params_sha256": capsule["params_sha256"],
        "retry_delay_s": 30.0, "api_cap": 12}, "completed journal manifest")
    # 읽기 전용 검사에 constructor의 디렉터리/manifest 생성 경로를 사용하지 않는다.
    journal = object.__new__(J.DecisionJournal)
    journal.root, journal.api_root = folder / "journal", api_root
    journal.binding, journal.params = binding, capsule["params"]
    journal.api_cap, journal.retry_delay_s = 12, 30.0
    journal.manifest_path = journal.root / "manifest.json"
    journal.api_manifest_path = api_root / "manifest.json"
    journal.manifest_sha256 = J.file_sha(journal.manifest_path)
    journal.binding_sha256, journal.params_sha256 = J.digest(binding), capsule["params_sha256"]
    journal._verify_prepared()
    audit = journal._audit_api()
    rows = audit["own"]
    require(all(row["response"] is not None for row in rows), "completed prefix pending")
    if not rows:
        terminal = J._read(folder / "journal/terminal.json")
        engine = {"stop_reason": "budget_exhausted", "selection": {"action": "hold"}}
        require(audit["remaining"] == 0 and terminal == journal._terminal(engine, "terminal_hold", []),
                "zero-call terminal requires exhausted shared budget")
        return engine
    used: list[dict] = []

    def replay(messages: list, params: dict) -> dict:
        require(len(used) < len(rows), "prefix requests another API")
        row = rows[len(used)]
        require(row["reservation"]["messages_sha256"] == J.digest(messages)
                and row["reservation"]["params_sha256"] == J.digest(params), "prefix exact request")
        used.append(deepcopy(row["response"]["result"]))
        return used[-1]

    engine = CALL._engine(capsule, replay)
    terminal = J._read(folder / "journal/terminal.json")
    require(len(used) == len(rows) and terminal == journal._terminal(
        engine, journal._result_status(engine), used), "prefix terminal changed")
    return engine


def evidence_spec(adapter: B.EvidenceAdapter) -> dict:
    return {"index": deepcopy(adapter.index), "roots": {k: str(v) for k, v in adapter.roots.items()},
            "registration": deepcopy(adapter.registration),
            "registration_sha256": adapter.registration_sha256}


def _adapter(spec: dict | None) -> B.EvidenceAdapter | None:
    if spec is None:
        return None
    return B.EvidenceAdapter(spec["index"], {k: Path(v) for k, v in spec["roots"].items()},
        spec["registration"], expected_registration_sha256=spec["registration_sha256"])


def _equivalent(left: dict, right: dict) -> bool:
    one, two = deepcopy(left), deepcopy(right)
    one["inputs"].pop("ledger_path")
    two["inputs"].pop("ledger_path")
    return one == two


def _archive(snapshot: dict, folder: Path) -> dict:
    """변경 가능한 실행 원장과 분리된 snapshot을 보존한다."""
    C.V.checked(snapshot)
    path = folder / "history.jsonl"
    source = Path(snapshot["inputs"]["ledger_path"])
    raw = source.read_bytes() if source.exists() else b""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, "archived history changed")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    inputs = deepcopy(snapshot["inputs"])
    inputs["ledger_path"] = path
    frozen = C.V.build_snapshot(**inputs)
    require(_equivalent(snapshot, frozen), "history archive changed view")
    return frozen


def registration(snapshot: dict, *, goal: str, api: dict, max_decisions: int,
                 evidence: dict | None = None, prefix: list[dict] | None = None,
                 allow_fixture: bool = False, bands: dict | None = None) -> dict:
    require(type(allow_fixture) is bool, "explicit fixture bool")
    C.V.checked(snapshot)
    A.prepare(snapshot, goal=goal, decisions_left=max_decisions)
    verify_api(api)
    return {"schema": SCHEMA, "snapshot": deepcopy(snapshot), "goal": goal,
        "max_decisions": max_decisions, "api": deepcopy(api), "evidence": deepcopy(evidence),
        "prefix": deepcopy(prefix or []), "allow_fixture": allow_fixture,
        "bands": deepcopy(C.read(ROOT / C.D1.BANDS)["bands"] if bands is None else bands)}


def verify_prefix(item: dict, request: dict, api_root: Path, *, repo: Path = ROOT) -> tuple[dict, dict]:
    """원래 committed protocol과 완료 journal을 현재 공개 입력에 결합한다. 쓰기 동작은 없다."""
    protocol = read_ref(item["protocol"])
    _, capsule = CALL.prepared(item["capsule"])
    require(capsule["protocol_sha256"] == item["protocol"]["sha256"], "prefix original protocol")
    verify_sources(protocol, capsule["source_anchor"], repo=repo,
                   protocol_path=Path(item["protocol"]["path"]))
    old, new = capsule["context"], request["context"]
    require(_equivalent(old["snapshot"], new["snapshot"])
            and all(old[k] == new[k] for k in ("goal", "evidence", "revealed", "extra_facts", "decisions_left"))
            and capsule["messages"] == request["messages"]
            and capsule["params"] == CALL.params_for(new), "prefix policy/checkpoint/budget/public view mismatch")
    return capsule, completed(item["capsule"], api_root)


def validate_prefix(reg: dict, *, repo: Path = ROOT) -> dict:
    """main이 새 등록 전에 실행할 수 있는 read-only prefix 사전 검사다."""
    api = verify_api(reg["api"])
    bridge = _adapter(reg["evidence"])
    snapshot = reg["snapshot"]
    loop = (bridge.start(snapshot, goal=reg["goal"], max_decisions=reg["max_decisions"],
                         allow_fixture=reg["allow_fixture"]) if bridge else
            L.Investigation(snapshot, goal=reg["goal"], max_decisions=reg["max_decisions"],
                            allow_fixture=reg["allow_fixture"]))
    visual, decisions = None, []
    require(len(reg["prefix"]) <= reg["max_decisions"], "prefix decision budget")
    for ordinal, item in enumerate(reg["prefix"], 1):
        request = Runtime._request(loop, bridge, visual)
        _, engine = verify_prefix(item, request, api, repo=repo)
        content = engine.get("final_content")
        review = loop.propose(content if isinstance(content, str) else "")
        require(review["status"] == "valid", "prefix decision invalid")
        decision = review["decision"]
        decisions.append(decision)
        action, target = decision["action"], decision["target"]
        if action in {"read_evidence", "inspect_frames"}:
            if bridge is None:
                raise ValueError("prefix evidence adapter required")
            method = bridge.read_evidence if action == "read_evidence" else bridge.inspect_frames
            visual = method(loop, target, visual_context=visual)["visual_context"]
        else:
            require(ordinal == len(reg["prefix"]), "prefix cannot continue past host action/HOLD")
    return {"schema": SCHEMA + "_prefix_validation", "decisions": decisions,
            "decisions_left": loop.remaining, "pending": loop.pending,
            "new_api_calls": 0, "host_dispatches": 0, "original_files_written": 0}


@dataclass(frozen=True)
class HostCallbacks:
    """각 callback은 (operation_id, intent)를 받으며 dispatch만 authorization 인자가 추가된다."""

    approve: Callable[[str, dict], dict]
    dispatch: Callable[[str, dict, dict], dict]
    lookup: Callable[[str, dict], dict]


def _host_history(folder: Path, operation_id: str) -> tuple[list[Path], dict | None]:
    paths = sorted((folder / "host-events").glob("*.json"))
    require([p.name for p in paths] == [f"{i:04d}.json" for i in range(1, len(paths) + 1)], "host event order")
    previous = None
    response = None
    for path in paths:
        row = J._read(path)
        require(row["operation_id"] == operation_id and row["previous_sha256"] == previous, "host event chain")
        previous = J.file_sha(path)
        response = row.get("response")
    return paths, response


def _host_call(folder: Path, phase: str, callback: Callable, operation_id: str, *args) -> dict:
    paths, _ = _host_history(folder, operation_id)
    previous = J.file_sha(paths[-1]) if paths else None
    event = {"operation_id": operation_id, "phase": phase, "previous_sha256": previous}
    try:
        response = callback(operation_id, *deepcopy(args))
        require(isinstance(response, dict), "host callback must return object")
    except BaseException as error:
        J._write_new(folder / "host-events" / f"{len(paths) + 1:04d}.json",
                     {**event, "interrupted": type(error).__name__})
        raise
    J._write_new(folder / "host-events" / f"{len(paths) + 1:04d}.json", {**event, "response": response})
    return response


class Runtime:
    """advance 한 번은 판단 하나와 그 도구 결과까지 처리한다. pending은 lookup으로만 재개한다."""

    def __init__(self, folder: Path, registration_ref: dict, protocol_ref: dict,
                 source_anchor: str, *, repo: Path = ROOT):
        self.folder = folder.resolve()
        self.reg = read_ref(registration_ref)
        self.protocol = read_ref(protocol_ref)
        require(self.reg["schema"] == SCHEMA and self.protocol["schema"] == SCHEMA + "_protocol",
                "runtime registration/protocol schema")
        require(self.protocol["registration_sha256"] == registration_ref["sha256"]
                and self.protocol["api_cap"] == 12
                and {"evals/agentic_investigation_runtime_v1.py",
                     "tests/test_agentic_investigation_runtime_v1.py"} <= self.protocol["sources_lf"].keys(),
                "runtime public registration binding")
        require(type(self.reg["allow_fixture"]) is bool
                and (self.reg["allow_fixture"] or self.reg["api"]["baseline_reserved"] == 7)
                and len(self.reg["prefix"]) <= self.reg["max_decisions"], "runtime baseline/prefix budget")
        verify_sources(self.protocol, source_anchor, repo=repo, protocol_path=Path(protocol_ref["path"]))
        self.api = verify_api(self.reg["api"])
        self.anchor, self.protocol_ref, self.repo = source_anchor, protocol_ref, repo
        self.manifest = {"schema": SCHEMA + "_manifest", "registration": registration_ref,
                         "protocol": protocol_ref, "source_anchor": source_anchor}
        same_or_new(self.folder / "manifest.json", self.manifest)
        initial_path = self.folder / "initial.json"
        if not initial_path.exists():
            frozen = _archive(self.reg["snapshot"], self.folder / "initial")
            J._write_new(initial_path, {"snapshot": frozen, "live_snapshot": self.reg["snapshot"]})
        self.initial = J._read(initial_path)
        C.V.checked(self.initial["snapshot"])
        require(_equivalent(self.initial["snapshot"], self.reg["snapshot"])
                and self.initial["live_snapshot"] == self.reg["snapshot"], "initial snapshot changed")

    @staticmethod
    def _request(loop: L.Investigation, bridge: B.EvidenceAdapter | None, visual: dict | None) -> dict:
        if bridge is None:
            context = loop.context()
            return {"context": context, "messages": A.messages(context)}
        request = bridge.messages(loop, visual_context=visual)
        return {"context": request["context"], "messages": CALL.from_v1(request["context"], request["messages"])}

    def _prefix(self, item: dict, request: dict) -> tuple[dict, dict]:
        return verify_prefix(item, request, self.api, repo=self.repo)

    def _authorization(self, intent: dict, authorization: dict, *, done: bool) -> None:
        snapshot = intent["snapshot"]
        if intent["action"] == "run_probe":
            value = C._authorization(authorization, snapshot, "done" if done else "running", require_digest=True)
            require(value["choice"] == intent["choice"] and value["spec"] == intent["request_spec"],
                    "probe approval choice mismatch")
            kind = value["evidence_kind"]
        else:
            require(set(authorization) == {"authorization", "ledger_path"}, "repair approval fields")
            auth = authorization["authorization"]
            component, target = C.registered_correction(snapshot, snapshot["inputs"]["packet"]["applied_corrections"])
            ledger = C.D1.CorrectionLedger(Path(authorization["ledger_path"]), C.RC.correction_registry())
            row = ledger.requests().get(auth["request_id"], {})
            events = [e["event"] for e in ledger.events() if e.get("request_id") == auth["request_id"]]
            require(events == ["proposed", "approved", "consumed"] + (["receipt"] if done else [])
                    and row.get("state") == ("done" if done else "running")
                    and row.get("loop_state_digest") == C.L.digest(snapshot)
                    and row.get("correction_id") == auth["correction_id"] == target == intent["target"]
                    and auth["component"] == component and row.get("evidence") == auth["evidence"],
                    "repair approval/one-time consumption")
            kind = auth["evidence_kind"]
            require(kind == auth["evidence"]["evidence_kind"], "repair evidence kind")
            seals = auth["evidence"]["probe_seals"]
            require(bool(seals), "repair requires probe receipts")
            verified = []
            for value in seals:
                raw = read_ref(value)
                require(raw["snapshot"]["binding"] == snapshot["binding"], "repair probe lineage")
                verified.append(C.verify_probe(Path(value["path"]), raw["snapshot"]))
            probe = "P_reward" if component == "reward" else C.RC.NEW_PROBE
            require(any(v["probe"] == probe and v["outcome"] == "abnormal" for v in verified)
                    and all(v["evidence_kind"] == kind for v in verified), "repair confirmed receipt")
        require(self.reg["allow_fixture"] or kind == C.ACTUAL_EVIDENCE, "actual evidence required; fixture denied")

    def _operation(self, folder: Path, intent: dict, host: HostCallbacks | None) -> dict:
        path = folder / "intent.json"
        same_or_new(path, intent)  # approve도 부수효과이므로 그 전에 intent를 보존한다.
        operation_id = J.digest({"manifest": self.manifest, "intent": intent})
        same_or_new(folder / "operation.json", {"operation_id": operation_id, "intent_sha256": J.file_sha(path)})
        _, last_response = _host_history(folder, operation_id)
        result_path, approval_path = folder / "host-result.json", folder / "approval.json"
        if result_path.exists():
            response = J._read(result_path)
            require(last_response == response, "stored host event/result changed")
            if response.get("status") == "completed":
                require(J._read(approval_path) == response["authorization"], "stored approval changed")
            return response
        if host is None:
            return {"status": "blocked_host", "operation_id": operation_id}
        if approval_path.exists():
            authorization = J._read(approval_path)
        else:
            marker = folder / "approval-intent.json"
            if marker.exists():
                same_or_new(marker, {"operation_id": operation_id})
                response = _host_call(folder, "lookup_approval", host.lookup, operation_id, intent)
            else:
                J._write_new(marker, {"operation_id": operation_id})
                response = _host_call(folder, "approve", host.approve, operation_id, intent)
            if response.get("status") == "hold":
                J._write_new(result_path, response)
                return response
            if response.get("status") not in {"approved", "completed"}:
                return {"status": "blocked_host", "operation_id": operation_id, "host_status": response.get("status")}
            authorization = response["authorization"]
            self._authorization(intent, authorization, done=response["status"] == "completed")
            J._write_new(approval_path, authorization)
            if response["status"] == "completed":
                J._write_new(result_path, response)
                return response
        marker = folder / "dispatch-intent.json"
        dispatch_intent = {"operation_id": operation_id, "approval_sha256": J.file_sha(approval_path)}
        if marker.exists():
            same_or_new(marker, dispatch_intent)
            response = _host_call(folder, "lookup_dispatch", host.lookup, operation_id, intent)
        else:
            self._authorization(intent, authorization, done=False)
            J._write_new(marker, dispatch_intent)
            response = _host_call(folder, "dispatch", host.dispatch, operation_id, intent, authorization)
        if response.get("status") == "completed":
            require(response.get("authorization") == authorization, "host completed approval mismatch")
            self._authorization(intent, authorization, done=True)
            J._write_new(result_path, response)
            return response
        require(response.get("status") in {"pending", "unknown", "not_found", "hold"}, "host result status")
        if response["status"] == "hold":
            same_or_new(result_path, response)
            return response
        return {"status": "blocked_host", "operation_id": operation_id, "host_status": response["status"]}

    def _tool(self, loop: L.Investigation, intent: dict, envelope: dict, folder: Path) -> dict:
        require(envelope["status"] == "completed", "tool completion required")
        authorization, result = envelope["authorization"], envelope["result"]
        self._authorization(intent, authorization, done=True)
        before = intent["snapshot"]
        require(_equivalent(before, loop.snapshot), "tool before snapshot changed")
        assert loop.pending is not None
        loop.snapshot = deepcopy(before)
        loop.pending["context"]["snapshot"] = deepcopy(before)
        checkpoint = folder / "tool-checkpoint.json"
        historical = checkpoint.exists()
        if historical:
            record = J._read(checkpoint)
            require(record["host_result_sha256"] == J.file_sha(folder / "host-result.json"), "tool result changed")
            after = record["snapshot"]
            if after is not None:
                C.V.checked(after)
                require(_equivalent(after, result["next_snapshot"]), "tool archived snapshot changed")
        else:
            after = result["next_snapshot"]
        if intent["action"] == "run_probe":
            require(set(result) == {"seal", "next_snapshot"}, "probe result fields")
            if not isinstance(after, dict):
                raise ValueError("probe next snapshot required")
            raw = read_ref(result["seal"])
            require(raw["authorization_path"] == authorization["path"]
                    and raw["authorization_sha256"] == authorization["sha256"], "probe seal approval mismatch")
            if historical:
                receipt = C.verify_probe(Path(result["seal"]["path"]), before)
                require(after["binding"] == before["binding"]
                        and {k: v for k, v in after["inputs"].items() if k != "ledger_path"}
                        == {k: v for k, v in before["inputs"].items() if k != "ledger_path"}
                        and after["view"]["observations"] == [*before["view"]["observations"],
                            {"probe": intent["target"], "outcome": receipt["outcome"]}], "probe next lineage/observations")
                loop.snapshot, loop.pending = after, None
                loop.events.append({"event": "probe", "path": result["seal"]["path"],
                    "sha256": result["seal"]["sha256"], "probe": receipt["probe"], "outcome": receipt["outcome"]})
                if receipt["outcome"] == "unknown":
                    loop.status = "hold"
            else:
                loop.complete_probe(result["seal"], after)
        else:
            require(set(result) == {"seal", "ledger_path", "bands", "next_snapshot", "evidence"}, "repair result fields")
            require(result["bands"] == self.reg["bands"], "registered recovery bands changed")
            raw = read_ref(result["seal"])
            require(raw["authorization"] == authorization["authorization"]
                    and result["ledger_path"] == authorization["ledger_path"], "repair seal approval mismatch")
            loop.complete_repair(Path(result["seal"]["path"]), ledger_path=Path(result["ledger_path"]),
                                 bands=result["bands"], next_snapshot=after)
        if not historical:
            archived = _archive(after, folder / "after") if after is not None else None
            J._write_new(checkpoint, {"host_result_sha256": J.file_sha(folder / "host-result.json"), "snapshot": archived})
            if archived is not None:
                loop.snapshot = archived
        return result

    def advance(self, *, complete: Callable[[list, dict], dict] | None = None,
                host: HostCallbacks | None = None, now_utc: Callable | None = None) -> dict:
        """완료 이력은 원문으로 검증하고 한 새 transition만 처리한다. API retry는 CALL/J가 소유한다."""
        require(read_ref(self.manifest["registration"]) == self.reg
                and read_ref(self.protocol_ref) == self.protocol
                and J._read(self.folder / "manifest.json") == self.manifest
                and J._read(self.folder / "initial.json") == self.initial, "immutable runtime binding changed")
        verify_sources(self.protocol, self.anchor, repo=self.repo, protocol_path=Path(self.protocol_ref["path"]))
        verify_api(self.reg["api"])
        bridge = _adapter(self.reg["evidence"])
        initial = self.initial["snapshot"]
        loop = (bridge.start(initial, goal=self.reg["goal"], max_decisions=self.reg["max_decisions"],
                             allow_fixture=self.reg["allow_fixture"]) if bridge else
                L.Investigation(initial, goal=self.reg["goal"], max_decisions=self.reg["max_decisions"],
                                allow_fixture=self.reg["allow_fixture"]))
        visual, live, events = None, self.initial["live_snapshot"], []
        prefix = self.reg["prefix"]
        paths = sorted((self.folder / "steps").glob("*"))
        require([p.name for p in paths] == [f"{i:04d}" for i in range(1, len(paths) + 1)], "step order")
        for ordinal in range(1, self.reg["max_decisions"] + 1):
            if loop.status != "active":
                return self._summary(loop, events)
            folder = self.folder / "steps" / f"{ordinal:04d}"
            request = self._request(loop, bridge, visual)
            record = {**request, "live_snapshot": live}
            same_or_new(folder / "request.json", record)
            from_prefix = ordinal <= len(prefix)
            if from_prefix:
                capsule, engine = self._prefix(prefix[ordinal - 1], request)
                capsule_ref = prefix[ordinal - 1]["capsule"]
            else:
                ref_path = folder / "capsule-reference.json"
                if not ref_path.exists():
                    capsule_path = folder / "call/capsule.json"
                    reference = ref(capsule_path) if capsule_path.exists() else CALL.freeze(request["context"],
                        folder / "call", source_anchor=self.anchor, protocol_sha256=self.protocol_ref["sha256"],
                        messages=request["messages"])
                    same_or_new(ref_path, reference)
                capsule_ref = J._read(ref_path)
                _, capsule = CALL.prepared(capsule_ref)
                require(capsule["messages"] == request["messages"]
                        and capsule["source_anchor"] == self.anchor
                        and capsule["protocol_sha256"] == self.protocol_ref["sha256"], "runtime capsule binding")
                if (folder / "call/journal/terminal.json").exists():
                    engine = completed(capsule_ref, self.api)
                else:
                    if complete is None:
                        return self._summary(loop, events, "blocked_model")
                    outcome = CALL.advance(capsule_ref, self.api, complete, api_cap=12, now_utc=now_utc)
                    if outcome["journal"]["status"] not in {"choice_ready", "terminal_hold"}:
                        return self._summary(loop, events, outcome["journal"]["status"])
                    engine = outcome["journal"]["engine_result"]
            content = engine.get("final_content")
            review = loop.propose(content if isinstance(content, str) else "")
            decision = review.get("decision") or {}
            receipt_facts = sorted(k for k in decision.get("fact_ids", []) if k.startswith("observation:"))
            observations = loop.snapshot["view"]["observations"]
            followup = None
            if observations:
                latest = observations[-1]
                fact_prefix = f"observation:{len(observations) - 1}:{latest['probe']}:"
                cited = [key for key in receipt_facts if key.startswith(fact_prefix)]
                sealed = next((e["host_judgment"] for e in reversed(events)
                               if (e["host_judgment"] or {}).get("event") == "probe"), None)
                followup = {"seal": {"path": sealed["path"], "sha256": sealed["sha256"]} if sealed else None,
                    "probe": latest["probe"], "outcome": latest["outcome"],
                    "fact_ids_cited": cited, "cited": bool(cited)}
            event = {"ordinal": ordinal, "capsule": capsule_ref, "imported_prefix": from_prefix,
                "model_status": engine.get("model_status", "unavailable"), "review": review,
                "selector": "model", "receipt_fact_ids_cited": receipt_facts,
                "latest_probe_followup": followup,
                "causal_effect_measured": False, "host_judgment": None}
            if loop.pending is None:
                same_or_new(folder / "event.json", event)
                events.append(event)
                return self._summary(loop, events)
            action, target = decision["action"], decision["target"]
            completed_before = (folder / "event.json").exists()
            if action in {"read_evidence", "inspect_frames"}:
                if bridge is None:
                    raise ValueError("evidence adapter required")
                method = bridge.read_evidence if action == "read_evidence" else bridge.inspect_frames
                receipt = method(loop, target, visual_context=visual)
                same_or_new(folder / "evidence.json", receipt)
                visual = receipt["visual_context"]
            else:
                require(action in {"run_probe", "propose_correction"}, "unsupported host action")
                require(_equivalent(live, loop.snapshot), "live execution snapshot mismatch")
                intent = {"action": action, "target": target, "snapshot": live,
                    "review": review, "source_anchor": self.anchor, "protocol": self.protocol_ref,
                    "selector": "model", "approval_granted": False}
                if action == "run_probe":
                    # request_spec는 원장 변경 전 검증된 immutable snapshot에서 계산한다.
                    choice = loop.probe_choice()
                    intent.update(choice=choice, request_spec=C.request_spec(loop.snapshot, choice))
                envelope = self._operation(folder, intent, host)
                if envelope["status"] == "blocked_host":
                    return self._summary(loop, events, "blocked_host", envelope)
                if envelope["status"] == "hold":
                    event["host_judgment"] = "hold"
                    same_or_new(folder / "event.json", event)
                    return self._summary(loop, [*events, event], "host_hold")
                result = self._tool(loop, intent, envelope, folder)
                event["host_judgment"] = loop.events[-1]
                live = result["next_snapshot"] if result["next_snapshot"] is not None else live
                if action == "propose_correction" and loop.status == "active":
                    # 새 checkpoint의 조회 이력은 새 등록과 새 epoch 객체에서 시작한다.
                    bridge = _adapter(result["evidence"])
                    snapshot, remaining = loop.snapshot, loop.remaining
                    loop = (bridge.start(snapshot, goal=self.reg["goal"], max_decisions=remaining,
                                         allow_fixture=self.reg["allow_fixture"]) if bridge else
                            L.Investigation(snapshot, goal=self.reg["goal"], max_decisions=remaining,
                                            allow_fixture=self.reg["allow_fixture"]))
                    visual = None
                elif action == "propose_correction":
                    require(result["evidence"] is None, "terminal repair cannot register next evidence")
            same_or_new(folder / "event.json", event)
            events.append(event)
            if not completed_before and not from_prefix:
                return self._summary(loop, events)
        return self._summary(loop, events, "decision_budget_exhausted" if loop.status == "active" else None)

    @staticmethod
    def _summary(loop: L.Investigation, events: list, status: str | None = None, pending: dict | None = None) -> dict:
        settled = "decision_budget_exhausted" if loop.status == "active" and loop.remaining == 0 else loop.status
        return {"schema": SCHEMA + "_result", "status": status or settled,
            "decisions_left": loop.remaining, "events": events, "pending": pending,
            "approval_granted_by_runtime": False, "causal_effect_measured": False,
            "model_validity_is_host_outcome": False}


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
