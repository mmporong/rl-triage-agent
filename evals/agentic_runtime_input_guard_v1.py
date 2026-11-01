"""기존 agentic runtime에 입력 불가·GPU busy 경계만 추가한다.

모델 호출, 승인, 실행, lease 조회는 이 모듈이 소유하지 않는다. 등록 evidence
입력의 명확한 불가 상태는 write-once receipt로 닫고, GPU busy 관측은 주입된
read-only callback을 승인 전에 확인한다.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import agentic_investigation_runtime_v1 as BASE

J = BASE.J
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "agentic_runtime_input_guard_v1"
REQUIRED_SOURCES = {
    "evals/agentic_runtime_input_guard_v1.py",
    "tests/test_agentic_runtime_input_guard_v1.py",
}
GPU_ACTIONS = {"run_probe", "propose_correction"}
UNAVAILABLE_ERRORS = (FileNotFoundError, PermissionError)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _binding(runtime: Runtime) -> dict:
    return {
        "source_anchor": runtime.anchor,
        "registration": deepcopy(runtime.manifest["registration"]),
        "protocol": deepcopy(runtime.manifest["protocol"]),
        "runtime_manifest_sha256": J.file_sha(runtime.folder / "manifest.json"),
        "initial_sha256": J.file_sha(runtime.folder / "initial.json"),
    }


def _registered_location(runtime: Runtime, error: OSError) -> dict | None:
    """예외 경로가 등록 evidence root 안에 있을 때만 안전한 상대 식별자를 반환한다."""
    evidence = runtime.reg.get("evidence")
    filename = getattr(error, "filename", None)
    if not isinstance(evidence, dict) or not filename:
        return None
    candidate = Path(filename).resolve()
    for name, raw_root in evidence.get("roots", {}).items():
        if not isinstance(name, str) or not isinstance(raw_root, str):
            continue
        root = Path(raw_root).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return {"root": name, "relative_path": relative.as_posix()}
    return None


def _unavailable_result(runtime: Runtime, receipt: dict) -> dict:
    _require(receipt.get("schema") == SCHEMA + "_unavailable_receipt", "guard receipt schema")
    _require(receipt.get("binding") == _binding(runtime), "guard receipt runtime binding")
    _require(receipt.get("decision") == "HOLD" and receipt.get("outcome") == "unknown"
             and receipt.get("automatic_retry") is False, "guard receipt refusal contract")
    path = runtime.folder / "input-unavailable.json"
    return {
        "schema": SCHEMA + "_result",
        "status": "input_unavailable_hold",
        "decision": "HOLD",
        "outcome": "unknown",
        "automatic_retry": False,
        "receipt": {"path": str(path), "sha256": J.file_sha(path)},
        "approval_granted_by_guard": False,
    }


def guarded_host(host: BASE.HostCallbacks, busy_reader: Callable[[str, dict], dict]) -> BASE.HostCallbacks:
    """GPU host action만 주입형 busy 관측으로 승인 전에 막는다."""
    _require(isinstance(host, BASE.HostCallbacks), "host callbacks required")
    _require(callable(busy_reader), "busy reader must be callable")

    def approve(operation_id: str, intent: dict) -> dict:
        if intent.get("action") not in GPU_ACTIONS:
            return host.approve(operation_id, deepcopy(intent))
        observation = busy_reader(operation_id, deepcopy(intent))
        _require(isinstance(observation, dict)
                 and set(observation) == {"status", "observation_id"}
                 and observation.get("status") in {"available", "busy"}
                 and isinstance(observation.get("observation_id"), str)
                 and bool(observation["observation_id"]), "busy observation contract")
        if observation["status"] == "busy":
            return {"status": "hold", "reason": "gpu_resource_busy",
                    "busy_observation": deepcopy(observation)}
        return host.approve(operation_id, deepcopy(intent))

    return BASE.HostCallbacks(approve=approve, dispatch=host.dispatch, lookup=host.lookup)


class Runtime(BASE.Runtime):
    """기존 Runtime 서명·상태 전이를 유지하는 입력 guard 적용판."""

    def __init__(self, folder: Path, registration_ref: dict, protocol_ref: dict,
                 source_anchor: str, *, repo: Path = ROOT):
        protocol = BASE.read_ref(protocol_ref)
        _require(isinstance(protocol.get("sources_lf"), dict)
                 and REQUIRED_SOURCES <= protocol["sources_lf"].keys(),
                 "guard/test sources are not registered")
        super().__init__(folder, registration_ref, protocol_ref, source_anchor, repo=repo)

    def _preflight_inputs(self) -> None:
        bridge = BASE._adapter(self.reg["evidence"])
        if bridge is not None:
            bridge.start(self.initial["snapshot"], goal=self.reg["goal"],
                         max_decisions=self.reg["max_decisions"],
                         allow_fixture=self.reg["allow_fixture"])

    def _verify_current(self) -> None:
        _require(BASE.read_ref(self.manifest["registration"]) == self.reg
                 and BASE.read_ref(self.protocol_ref) == self.protocol
                 and J._read(self.folder / "manifest.json") == self.manifest
                 and J._read(self.folder / "initial.json") == self.initial,
                 "immutable guard runtime binding changed")
        BASE.verify_sources(self.protocol, self.anchor, repo=self.repo,
                            protocol_path=Path(self.protocol_ref["path"]))
        BASE.verify_api(self.reg["api"])

    def advance(self, *, complete: Callable[[list, dict], dict] | None = None,
                host: BASE.HostCallbacks | None = None, now_utc: Callable | None = None) -> dict:
        self._verify_current()
        receipt_path = self.folder / "input-unavailable.json"
        if receipt_path.exists():
            return _unavailable_result(self, J._read(receipt_path))
        try:
            self._preflight_inputs()
        except UNAVAILABLE_ERRORS as error:
            location = _registered_location(self, error)
            if location is None:
                raise
            receipt = {
                "schema": SCHEMA + "_unavailable_receipt",
                "binding": _binding(self),
                "decision": "HOLD",
                "outcome": "unknown",
                "reason": "registered_input_unavailable",
                "error_type": type(error).__name__,
                "artifact": location,
                "automatic_retry": False,
                "approval_granted": False,
            }
            J._write_new(receipt_path, receipt)
            return _unavailable_result(self, receipt)
        return super().advance(complete=complete, host=host, now_utc=now_utc)


def source_hash(path: Path) -> str:
    return BASE.source_hash(path)
