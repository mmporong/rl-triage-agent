"""등록된 검사와 승인 원장을 사용하는 재개 가능한 조사 세션."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_triage import mechanism_registry as M

SCHEMA = "triage_investigation_session_v1"
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
MODEL_METHODS = METHODS[2:]
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_HIDDEN = {
    "fault", "fault_id", "faults", "fault_label", "fault_labels", "hidden_label", "hidden_labels",
    "original_faults", "remaining_faults", "scoring", "benchmark_score", "benchmark_scoring",
    "answer_key",
}
_EXCLUDED = "ri_codex2_tokenfactory_live_160"


def require(condition: object, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    require(_EXCLUDED not in str(path).casefold(), "excluded input scope")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_ref(reference: Mapping[str, Any]) -> dict:
    require(set(reference) == {"path", "sha256"}, "artifact reference fields")
    path = Path(reference["path"])
    require(_EXCLUDED not in str(path).casefold(), "excluded input scope")
    require(path.is_file() and not path.is_symlink() and file_sha(path) == reference["sha256"],
            "artifact SHA changed")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    require(isinstance(value, dict), "artifact must be JSON object")
    canonical(value)
    return value


def reference(path: Path) -> dict:
    require(_EXCLUDED not in str(path).casefold(), "excluded input scope")
    return {"path": str(path.resolve(strict=True)), "sha256": file_sha(path)}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_once_or_match(path: Path, value: dict) -> None:
    """Crash 뒤 같은 증거로 재개할 수 있지만 다른 값으로 덮어쓰지는 않는다."""
    if path.exists():
        require(read_ref({"path": str(path.resolve(strict=True)), "sha256": file_sha(path)}) == value,
                f"existing {path.name} differs")
        return
    write_new(path, value)


def _public(value: Any) -> None:
    if isinstance(value, Mapping):
        require(not (_HIDDEN & {str(key).casefold() for key in value}),
                "hidden truth is not a decision input")
        for item in value.values():
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)
    canonical(value)


def _journal_module():
    folder = Path(__file__).resolve().parents[2] / "evals"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("model_decision_journal_v1")


@contextmanager
def _single_writer(folder: Path) -> Iterator[None]:
    lock = folder / ".session-writer"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def _initial_state(source: dict, method: str, max_checks: int, verification: dict) -> dict:
    return {
        "schema": SCHEMA + "_status", "case_id": source["case_id"], "method": method,
        "status": "open", "hypotheses": [], "observed": {}, "checks_remaining": max_checks,
        "pending": None, "confirmed_mechanism": None,
        "source_verification": deepcopy(verification), "approval_granted_by_session": False,
        "policy_recovery_verified": False,
    }


class InvestigationSession:
    """모델은 probe ID만 고르고 권한과 실제 실행은 등록 host callback이 소유한다."""

    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve(strict=True)
        require(not self.folder.is_symlink(), "session folder symlink")
        anchor_path = self.folder / "anchor.json"
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        require(set(anchor) == {"schema", "manifest_sha256"}
                and anchor["schema"] == SCHEMA + "_anchor"
                and _HEX.fullmatch(str(anchor["manifest_sha256"])) is not None, "session anchor")
        self._manifest_sha256 = anchor["manifest_sha256"]
        self._anchor_sha256 = file_sha(anchor_path)
        self._load_verified()
        self.ledger = M.RegistryLedger(self.folder / "execution.jsonl", self.registry)

    @classmethod
    def prepare(
        cls, source_path: Path, folder: Path, *, registry: dict, method: str,
        max_checks: int, gpu_budget_per_check_s: int,
        source_verifier: Callable[[Path], dict], model_binding: dict | None = None,
        execution_class: str = "actual",
    ) -> InvestigationSession:
        require(method in METHODS, "method")
        require(type(max_checks) is int and max_checks > 0, "positive check budget")
        require(type(gpu_budget_per_check_s) is int and gpu_budget_per_check_s >= 0, "GPU budget")
        require(execution_class in {"actual", "recorded", "fixture"}, "execution class")
        source_ref = reference(source_path)
        source = read_ref(source_ref)
        require(set(source) == {"schema", "case_id", "checkpoint_sha256", "facts", "binding"},
                "source fields")
        require(isinstance(source["case_id"], str) and bool(source["case_id"]), "case ID")
        require(_HEX.fullmatch(str(source["checkpoint_sha256"])) is not None, "checkpoint SHA")
        _public(source["facts"])
        verification = source_verifier(Path(source_ref["path"]))
        cls._validate_source_verification(verification, source_ref, source, execution_class)
        normalized = M.normalize(registry)
        initial = _initial_state(source, method, max_checks, verification)
        initial["hypotheses"] = list(normalized["mechanisms"])
        if method in MODEL_METHODS:
            if not isinstance(model_binding, dict) or set(model_binding) not in (
                {"source_anchor", "protocol_sha256", "params", "api_cap"},
                {"source_anchor", "protocol_sha256", "params", "api_cap", "api_root"},
            ):
                raise ValueError("model binding")
            require(type(model_binding["api_cap"]) is int and model_binding["api_cap"] >= 1,
                    "shared API budget")
            require(re.fullmatch(r"[0-9a-f]{40}", str(model_binding["source_anchor"])) is not None,
                    "model source anchor")
            require(_HEX.fullmatch(str(model_binding["protocol_sha256"])) is not None,
                    "model protocol SHA")
            canonical(model_binding["params"])
            if "api_root" in model_binding:
                api_text = model_binding["api_root"]
                require(isinstance(api_text, str) and bool(api_text)
                        and _EXCLUDED not in api_text.casefold()
                        and "://" not in api_text and not api_text.startswith("\\\\"),
                        "shared API root scope")
                api_path = Path(api_text)
                require(api_path.is_absolute() and (not api_path.exists() or not api_path.is_symlink()),
                        "shared API root must be an absolute local path")
                model_binding = {**model_binding, "api_root": str(api_path.resolve(strict=False))}
        else:
            if model_binding is not None:
                raise ValueError("non-model method has no provider binding")
        folder = Path(folder)
        require(not folder.exists(), "fresh session required")
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema": SCHEMA, "source": source_ref, "source_verification": verification,
            "registry": normalized, "registry_sha256": M.registry_digest(normalized),
            "method": method, "max_checks": max_checks,
            "gpu_budget_per_check_s": gpu_budget_per_check_s,
            "model_binding": deepcopy(model_binding),
            "model_initial_state": initial if method == "one_shot_llm" else None,
            "execution_class": execution_class, "approval_granted_by_session": False,
        }
        write_new(folder / "manifest.json", manifest)
        write_new(folder / "anchor.json", {
            "schema": SCHEMA + "_anchor", "manifest_sha256": file_sha(folder / "manifest.json"),
        })
        return cls(folder)

    @staticmethod
    def _validate_source_verification(
        verification: object, source_ref: Mapping[str, Any], source: Mapping[str, Any],
        execution_class: str,
    ) -> None:
        require(isinstance(verification, dict)
                and verification.get("source_sha256") == source_ref["sha256"]
                and verification.get("checkpoint_sha256") == source["checkpoint_sha256"]
                and verification.get("status") == "verified"
                and verification.get("execution_class") == execution_class
                and isinstance(verification.get("verifier_id"), str)
                and bool(verification["verifier_id"]),
                "registered source verifier did not bind this input")
        canonical(verification)

    def _load_verified(self) -> None:
        anchor_path = self.folder / "anchor.json"
        require(file_sha(anchor_path) == self._anchor_sha256, "session anchor changed")
        manifest_path = self.folder / "manifest.json"
        require(file_sha(manifest_path) == self._manifest_sha256, "session manifest changed")
        manifest = read_ref({"path": str(manifest_path), "sha256": self._manifest_sha256})
        required = {
            "schema", "source", "source_verification", "registry", "registry_sha256", "method",
            "max_checks", "gpu_budget_per_check_s", "model_binding", "model_initial_state",
            "execution_class", "approval_granted_by_session",
        }
        require(set(manifest) == required and manifest["schema"] == SCHEMA, "session schema")
        source = read_ref(manifest["source"])
        registry = M.normalize(manifest["registry"])
        require(M.registry_digest(registry) == manifest["registry_sha256"], "registry changed")
        require(source["case_id"] == manifest["source_verification"].get("case_id", source["case_id"]),
                "source case binding changed")
        require(source["checkpoint_sha256"] == manifest["source_verification"]["checkpoint_sha256"],
                "source checkpoint binding changed")
        self.manifest: dict[str, Any] = manifest
        self.source: dict[str, Any] = source
        self.registry: dict[str, Any] = registry

    def _verify_operation(self, source_verifier: Callable[[Path], dict] | None = None) -> None:
        self._load_verified()
        if source_verifier is not None:
            current = source_verifier(Path(self.manifest["source"]["path"]))
            self._validate_source_verification(
                current, self.manifest["source"], self.source, self.manifest["execution_class"],
            )
            require(current == self.manifest["source_verification"],
                    "source verifier result changed after prepare")

    def _prereg(self) -> dict:
        return {
            "case_id": self.source["case_id"], "source_sha256": self.manifest["source"]["sha256"],
            "checkpoint_sha256": self.source["checkpoint_sha256"],
            "session_sha256": self._manifest_sha256,
            "registry_sha256": self.manifest["registry_sha256"],
            "execution_class": self.manifest["execution_class"],
        }

    def _state(self) -> dict:
        requests = self.ledger.requests()
        completed = [row for row in requests.values() if row["state"] == "done"]
        observed = {row["probe"]: row["receipt"]["outcome"] for row in completed}
        require(len(observed) == len(completed), "duplicate completed probe")
        hypotheses = list(self.registry["mechanisms"])
        for probe, outcome in observed.items():
            hypotheses, _ = M.update(self.registry, hypotheses, probe, outcome)
        remaining = max(0, self.manifest["max_checks"] - len(requests))
        live = [row for row in requests.values() if row["state"] in {"pending", "approved", "running"}]
        require(len(live) <= 1, "one operation at a time")
        status = M._status(self.registry, hypotheses, observed, remaining)
        if any(outcome == "unknown" for outcome in observed.values()):
            status = "hold_unknown"
        gate_path = self.folder / "host_gate.json"
        if gate_path.exists():
            gate = self._gate(json.loads(gate_path.read_text(encoding="utf-8")))
            require(gate["decision"] in {"host_no_intervention", "host_hold"}, "host gate changed")
            status = ("host_no_intervention" if gate["decision"] == "host_no_intervention"
                      else "hold_observation_unqualified")
        if live:
            status = {"pending": "approval_pending", "approved": "ready_to_dispatch",
                      "running": "pending_result"}[live[0]["state"]]
        return {
            "schema": SCHEMA + "_status", "case_id": self.source["case_id"],
            "method": self.manifest["method"], "status": status, "hypotheses": hypotheses,
            "observed": observed, "checks_remaining": remaining,
            "pending": deepcopy(live[0]) if live else None,
            "confirmed_mechanism": hypotheses[0] if status == "confirmed" else None,
            "source_verification": deepcopy(self.manifest["source_verification"]),
            "approval_granted_by_session": False, "policy_recovery_verified": False,
        }

    def status(self) -> dict:
        self._verify_operation()
        return self._state()

    def _facts(self, state: dict) -> dict:
        facts = deepcopy(self.source["facts"])
        for ordinal, (probe, outcome) in enumerate(state["observed"].items()):
            facts[f"observation:{ordinal}:{probe}:outcome"] = outcome
        return facts

    def _model_choice(self, state: dict, complete: Callable | None) -> dict:
        method = self.manifest["method"]
        ordinal = 0 if method == "one_shot_llm" else len(state["observed"])
        folder = self.folder / "model" / f"{ordinal:04d}"
        initial = (deepcopy(self.manifest["model_initial_state"])
                   if method == "one_shot_llm" else deepcopy(state))
        facts = self._facts(initial)
        field_contract: dict[str, Any] = {
            "required_fields": ["action", "probe", "probes", "fact_ids"],
            "fact_ids": {"minItems": 1, "allowed_ids": sorted(facts)},
        }
        prompt = ("Select a fixed ordered plan of at most three distinct registered probe IDs, or hold. "
                  if method == "one_shot_llm" else "Select one next registered probe ID, or hold. ")
        messages = [{
            "role": "system",
            "content": prompt + "Return one JSON object with exactly action, probe, probes, fact_ids. "
            "For every action, including hold, fact_ids must contain at least one supplied fact ID; "
            "an empty array is invalid. Follow field_contract in the input. "
            "Use probe=null and probes=[] for hold; use probe=null for plan; "
            "use probes=[] for run_probe. "
            "Do not create arguments, permissions, or causes.",
        }, {
            "role": "user",
            "content": canonical({"facts": facts, "field_contract": field_contract,
                                  "catalog": self.registry["probes"],
                                  "observed": initial["observed"],
                                  "budget": self.manifest["max_checks"]}).decode(),
        }]
        J = _journal_module()
        binding = self.manifest["model_binding"]
        if not isinstance(binding, dict):
            raise TypeError("model binding missing")
        api_root = Path(binding.get("api_root", self.folder / "api"))
        journal = J.DecisionJournal(folder, api_root, {
            "run_id": f"{self.source['case_id']}:{method}:{ordinal}",
            "source_anchor": binding["source_anchor"], "protocol_sha256": binding["protocol_sha256"],
            "checkpoint_sha256": self.source["checkpoint_sha256"],
            "snapshot_sha256": digest(initial), "initial_messages_sha256": J.digest(messages),
        }, binding["params"], api_cap=binding["api_cap"])
        if complete is None and not (folder / "terminal.json").exists():
            return {"status": "blocked_backend", "probe": None, "model_status": "not_called"}

        def engine(cached_complete: Callable) -> dict:
            response = cached_complete(messages, binding["params"])
            if J.DecisionJournal._retryable(response) or "error_type" in response or response.get("status") == "error":
                return {"selection": {"action": "hold"}, "model_status": "infrastructure_error",
                        "error_type": response.get("error_type"), "stop_reason": "infra_error"}
            try:
                content = response.get("content", response.get("final_content"))
                choice = json.loads(content) if isinstance(content, str) else None
                if not isinstance(choice, dict):
                    raise TypeError("model output fields")
                require(set(choice) == set(field_contract["required_fields"]),
                        "model output fields")
                require(isinstance(choice["fact_ids"], list)
                        and len(choice["fact_ids"]) >= field_contract["fact_ids"]["minItems"]
                        and all(item in field_contract["fact_ids"]["allowed_ids"]
                                for item in choice["fact_ids"]), "model fact citations")
                action = choice["action"]
                if method == "one_shot_llm" and action == "plan":
                    plan = choice["probes"]
                    require(isinstance(plan, list) and 0 < len(plan) <= self.manifest["max_checks"]
                            and len(plan) <= 3 and len(set(plan)) == len(plan)
                            and all(item in self.registry["probes"] for item in plan)
                            and choice["probe"] is None, "registered fixed model plan")
                elif action == "run_probe" and method == "iterative_agent":
                    require(choice["probe"] in self.registry["probes"]
                            and choice["probe"] not in initial["observed"]
                            and choice["probes"] == [], "registered unused probe")
                else:
                    require(action == "hold" and choice["probe"] is None and choice["probes"] == [],
                            "hold shape")
                return {"selection": choice, "model_status": "valid"}
            except (ValueError, TypeError, KeyError):
                return {"selection": {"action": "hold"}, "model_status": "contract_invalid"}

        result = journal.advance(
            engine, complete or (lambda *_: (_ for _ in ()).throw(ValueError("no new HTTP"))),
            lambda: datetime.now(UTC),
        )
        if result["status"] in {"blocked_pending", "paused_infra"}:
            return {"status": "hold_" + result["status"], "probe": None,
                    "model_status": "not_successful", "api": result["api"]}
        output = result["engine_result"]
        if not isinstance(output, dict):
            raise TypeError("model engine result missing")
        choice = output["selection"]
        if not isinstance(choice, dict):
            raise TypeError("model selection missing")
        if choice["action"] == "plan":
            probe = next((item for item in choice["probes"] if item not in state["observed"]), None)
        else:
            probe = choice.get("probe") if choice["action"] == "run_probe" else None
        return {"status": "choice_ready" if probe else "model_hold", "probe": probe,
                "model_status": output.get("model_status", "not_successful"), "api": result["api"]}

    @staticmethod
    def _gate(gate: object) -> dict:
        if not isinstance(gate, dict):
            raise TypeError("observation gate result")
        require(set(gate) == {"decision", "gate_id", "reason"}
                and gate["decision"] in {"allow", "host_no_intervention", "host_hold"}
                and isinstance(gate["gate_id"], str) and bool(gate["gate_id"])
                and isinstance(gate["reason"], str), "observation gate result")
        return deepcopy(gate)

    def next(
        self, *, complete: Callable | None = None,
        source_verifier: Callable[[Path], dict] | None = None,
        observation_gate: Callable[[dict], dict] | None = None,
        metrics_collector: Callable[[dict], None] | None = None,
    ) -> dict:
        with _single_writer(self.folder):
            self._verify_operation(source_verifier)
            state = self._state()
            if state["status"] != "open":
                return state
            if observation_gate is not None:
                gate = self._gate(observation_gate({
                    "state": deepcopy(state), "source": deepcopy(self.source),
                    "source_verification": deepcopy(self.manifest["source_verification"]),
                }))
                if gate["decision"] in {"host_no_intervention", "host_hold"}:
                    write_new(self.folder / "host_gate.json", gate)
                    result = self._state()
                    if metrics_collector:
                        metrics_collector({"event": gate["decision"], "model_called": False})
                    return result
            method = self.manifest["method"]
            model = None
            if method == "fixed_sequence":
                probe = next((item for item in self.registry["fixed_order"]
                              if item not in state["observed"]), None)
            elif method == "static_rule":
                probe = M._next(self.registry, state["hypotheses"], state["observed"])
            else:
                model = self._model_choice(state, complete)
                probe = model["probe"]
                if probe is None:
                    result = {**state, "status": model["status"], "model": model}
                    if metrics_collector:
                        metrics_collector({"event": "model_hold", "model": deepcopy(model)})
                    return result
            if probe is None:
                return {**state, "status": "hold_no_registered_probe"}
            self.ledger.propose(self._prereg(), probe, self.manifest["gpu_budget_per_check_s"])
            result = self._state()
            if model is not None:
                result["model"] = model
            if metrics_collector:
                metrics_collector({"event": "probe_proposed", "probe": probe,
                                   "model_status": model and model["model_status"]})
            return result

    def approve(
        self, request_id: str, *, actor: str, authorization: dict,
        authorize: Callable[[dict, str, dict], dict],
        source_verifier: Callable[[Path], dict] | None = None,
    ) -> dict:
        require(isinstance(actor, str) and actor.strip(), "named host actor required")
        require(isinstance(authorization, dict), "authorization input")
        with _single_writer(self.folder):
            self._verify_operation(source_verifier)
            request = self.ledger.requests().get(request_id)
            if request is None or request["state"] != "pending":
                raise ValueError("pending operation required")
            proof = authorize(deepcopy(request), actor, deepcopy(authorization))
            if not isinstance(proof, dict):
                raise TypeError("registered host did not authorize this request")
            require(proof.get("status") == "authorized"
                    and proof.get("actor") == actor and proof.get("request_id") == request_id
                    and proof.get("effect_key") == request["effect_key"]
                    and isinstance(proof.get("authority_id"), str) and bool(proof["authority_id"]),
                    "registered host did not authorize this request")
            canonical(proof)
            _write_once_or_match(self.folder / "approvals" / f"{request_id}.json", {
                "schema": SCHEMA + "_approval", "proof": proof, "request_sha256": digest(request),
            })
            self.ledger.approve(request_id, actor)
            return self._state()

    def _intent(self, request: dict, consume: dict) -> dict:
        return {
            "request": deepcopy(request), "consumption": deepcopy(consume),
            "preregistration": self._prereg(), "source": deepcopy(self.source),
            "source_verification": deepcopy(self.manifest["source_verification"]),
            "registry_sha256": self.manifest["registry_sha256"],
            "execution_class": self.manifest["execution_class"],
            "host_receipt_path": str((self.folder / "host_receipts" /
                                      f"{request['request_id']}.json").resolve()),
        }

    def _normalize_receipt(
        self, raw: dict, intent: dict, verify_receipt: Callable[[dict, dict], dict],
    ) -> dict:
        normalized = verify_receipt(deepcopy(raw), deepcopy(intent))
        fields = {
            "outcome", "measurement", "gpu_s", "execution_wall_s", "exit_code", "receipt_ref",
            "source_sha256", "checkpoint_sha256", "case_id", "probe", "request_id", "effect_key",
            "registry_sha256", "execution_class",
        }
        require(isinstance(normalized, dict) and set(normalized) == fields,
                "normalized host receipt fields")
        request = intent["request"]
        bindings = {
            "source_sha256": self.manifest["source"]["sha256"],
            "checkpoint_sha256": self.source["checkpoint_sha256"], "case_id": self.source["case_id"],
            "probe": request["probe"], "request_id": request["request_id"],
            "effect_key": request["effect_key"], "registry_sha256": self.manifest["registry_sha256"],
            "execution_class": self.manifest["execution_class"],
        }
        require(all(normalized.get(key) == value for key, value in bindings.items()),
                "host receipt binding mismatch")
        artifact = read_ref(normalized["receipt_ref"])
        require(all(artifact.get(key) == value for key, value in bindings.items()),
                "host receipt artifact binding mismatch")
        for field in ("gpu_s", "execution_wall_s"):
            value = normalized[field]
            require(type(value) in {int, float} and math.isfinite(value) and value >= 0,
                    f"{field} scope")
        require(normalized["outcome"] in self.registry["outcomes"], "receipt outcome")
        require(isinstance(normalized["measurement"], dict), "receipt measurement")
        require(normalized["exit_code"] is None or type(normalized["exit_code"]) is int,
                "receipt exit code")
        canonical(normalized)
        return normalized

    def _finish(self, request_id: str, raw: dict, intent: dict,
                verify_receipt: Callable[[dict, dict], dict]) -> dict:
        normalized = self._normalize_receipt(raw, intent, verify_receipt)
        _write_once_or_match(self.folder / "receipts" / f"{request_id}.json", {
            "schema": SCHEMA + "_receipt", "request_id": request_id,
            "normalized": normalized, "raw_sha256": digest(raw),
        })
        self.ledger.receipt(request_id, normalized["outcome"], normalized["measurement"],
                            normalized["gpu_s"], normalized["exit_code"])
        return self._state()

    def dispatch(
        self, request_id: str, *, execute: Callable[[dict], dict],
        verify_receipt: Callable[[dict, dict], dict], source_verifier: Callable[[Path], dict],
        metrics_collector: Callable[[dict], None] | None = None,
    ) -> dict:
        with _single_writer(self.folder):
            self._verify_operation(source_verifier)
            request = self.ledger.requests().get(request_id)
            if request is None or request["state"] != "approved":
                raise ValueError("approved unused operation required")
            consume = self.ledger.consume(request_id, self._prereg(), request["probe"])
            intent = self._intent(request, consume)
            try:
                raw = execute(deepcopy(intent))
                if not isinstance(raw, dict):
                    raise TypeError("host execute result")
                result = self._finish(request_id, raw, intent, verify_receipt)
            except Exception as error:  # noqa: BLE001 - 실행 여부가 불명확해 같은 요청을 재실행하지 않는다.
                error_path = self.folder / "dispatch_errors" / f"{request_id}.json"
                if not error_path.exists():
                    write_new(error_path, {"schema": SCHEMA + "_dispatch_error",
                                           "request_id": request_id,
                                           "error_type": type(error).__name__})
                result = self._state()
                result["dispatch_error"] = {"error_type": type(error).__name__,
                                            "reexecute_allowed": False}
            if metrics_collector:
                metrics_collector({"event": "dispatch_returned", "request_id": request_id,
                                   "state": result["status"]})
            return result

    def settle(
        self, request_id: str, *, host_receipt: dict | None,
        verify_receipt: Callable[[dict, dict], dict], source_verifier: Callable[[Path], dict],
        metrics_collector: Callable[[dict], None] | None = None,
    ) -> dict:
        """running 요청은 재실행하지 않고 host receipt 검증 또는 unknown으로만 닫는다."""
        with _single_writer(self.folder):
            self._verify_operation(source_verifier)
            request = self.ledger.requests().get(request_id)
            if request is None or request["state"] != "running":
                raise ValueError("running operation required")
            intent = self._intent(request, {"event": "recovered_without_reexecution"})
            if host_receipt is None:
                self.ledger.recover(request_id, None)
                result = self._state()
            else:
                require(isinstance(host_receipt, dict), "host receipt object")
                result = self._finish(request_id, host_receipt, intent, verify_receipt)
            if metrics_collector:
                metrics_collector({"event": "settled_without_reexecution", "request_id": request_id,
                                   "receipt_present": host_receipt is not None})
            return result

    def export(self, path: Path) -> dict:
        self._verify_operation()
        result = {
            "schema": SCHEMA + "_export", "manifest": reference(self.folder / "manifest.json"),
            "source": deepcopy(self.manifest["source"]), "state": self._state(),
            "events": self.ledger.events(), "actual_billing_usd": None,
            "human_review_seconds": None, "provider_gpu_seconds": None,
        }
        write_new(path, result)
        return result
