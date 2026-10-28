"""등록된 호스트 IO 서비스에 검사 요청을 전달한다. 이 모듈은 VM과 승인을 만들지 않는다."""
from __future__ import annotations

import hashlib
import json
import stat
import time
from pathlib import Path

import agentic_brev_probe_evidence_v1 as E
import selection_comparison_controller_v1 as C

SCHEMA = "agentic_brev_mailbox_transport_v1"


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def plain_path(path: Path) -> None:
    require(path.is_absolute() and ".." not in path.parts, "canonical native absolute path required")
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not item.is_symlink() and not getattr(info, "st_file_attributes", 0) & 0x400,
                    "symlink/reparse path refused")


def regular(path: Path) -> None:
    plain_path(path)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "regular single-link file required")


def read_ref(reference: dict) -> dict:
    require(set(reference) == {"path", "sha256"}, "immutable reference fields")
    path = Path(reference["path"])
    regular(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "reference SHA changed")
    return E.loads(raw.decode("utf-8"))


class BrevMailboxTransport:
    """실제 서비스의 완료 원문을 전달하며, 실제 증거 판정은 기존 E13/C host가 수행한다."""

    EVIDENCE_KIND = E.ACTUAL

    def __init__(self, runtime, contract_ref: dict):
        self.runtime = runtime
        self.contract_ref = dict(contract_ref)
        contract = read_ref(contract_ref)
        require(contract.get("schema") == SCHEMA + "_contract"
                and runtime.protocol["host_transport_contract_sha256"] == contract_ref["sha256"],
                "registered mailbox contract required")
        self.contract = contract
        self.root = Path(contract["mailbox"])
        plain_path(self.root)
        require(self.root.is_absolute() and self.root == runtime.folder / "transport-mailbox"
                and self.root.is_dir(), "registered runtime mailbox directory required")
        self.results = Path(contract["result_root"])
        plain_path(self.results)
        require(self.results.is_absolute() and self.results.is_dir(), "registered result root required")
        self.wait_s = contract["response_wait_s"]
        require(type(self.wait_s) in (int, float) and 0 < self.wait_s <= 1800,
                "bounded response wait required")
        require(contract["service_source_sha256_lf"] == runtime.protocol["host_transport_service_sha256_lf"],
                "host service source registration required")

    def _base(self, operation: str, intent: dict, registration_ref: dict) -> dict:
        require(read_ref(self.contract_ref) == self.contract, "mailbox contract changed")
        require(isinstance(operation, str) and len(operation) == 64
                and all(char in "0123456789abcdef" for char in operation), "operation id")
        require(intent.get("action") == "run_probe", "repair transport unsupported")
        registration = read_ref(registration_ref)
        require(registration.get("schema") == "agentic_brev_runtime_host_v1_registration",
                "parent registration required")
        return {"schema": SCHEMA + "_request", "operation_id": operation,
                "intent_sha256": E.digest(intent), "registration": registration_ref,
                "contract_sha256": self.contract_ref["sha256"]}

    def _lookup(self, phase: str, operation: str, intent: dict, registration_ref: dict) -> dict:
        base = self._base(operation, intent, registration_ref)
        folder = self.root / operation / phase
        plain_path(folder)
        request_path, response_path = folder / "request.json", folder / "response.json"
        if not request_path.exists():
            require(not response_path.exists(), "response without request")
            return {"status": "not_found", "operation_id": operation}
        regular(request_path)
        request = E.loads(request_path.read_text(encoding="utf-8"))
        require(all(request.get(key) == value for key, value in base.items())
                and request.get("phase") == phase
                and set(request) == set(base) | {"phase"} | ({"authorization"} if phase == "case" else set()),
                "existing mailbox intent differs")
        if not response_path.exists():
            return {"status": "pending", "operation_id": operation}
        regular(response_path)
        response = E.loads(response_path.read_text(encoding="utf-8"))
        require(response.get("schema") == SCHEMA + "_response"
                and response.get("request_sha256") == E.J.file_sha(request_path)
                and response.get("operation_id") == operation and response.get("phase") == phase
                and response.get("status") in {"completed", "pending", "unknown", "hold"},
                "mailbox response/request binding")
        result = {"status": response["status"], "operation_id": operation}
        if response["status"] == "completed":
            reference = response["result"]
            path = Path(reference["path"])
            plain_path(path)
            require(self.results.resolve(strict=True) in path.resolve(strict=True).parents,
                    "raw result outside registered root")
            read_ref(reference)
            result["result"] = reference
        return result

    def _submit(self, phase: str, operation: str, intent: dict, registration_ref: dict,
                authorization: dict | None = None) -> dict:
        previous = self._lookup(phase, operation, intent, registration_ref)
        if previous["status"] != "not_found":
            return previous
        base = self._base(operation, intent, registration_ref)
        request = {**base, "phase": phase}
        if phase == "case":
            if authorization is None:
                raise ValueError("case requires consumed C authorization")
            C.validate_probe_authorization(authorization, intent["snapshot"])
            request["authorization"] = authorization
        folder = self.root / operation / phase
        folder.mkdir(parents=True, exist_ok=False)
        # 요청의 배타 생성이 전송 의도다. 부분 파일·응답 불명은 같은 요청을 재전송하지 않는다.
        with (folder / "request.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(request, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        deadline = time.monotonic() + self.wait_s
        while True:
            result = self._lookup(phase, operation, intent, registration_ref)
            if result["status"] != "pending" or time.monotonic() >= deadline:
                return result
            time.sleep(0.1)

    def prepare_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        self._base(operation_id, intent, registration_ref)
        return {"status": "prepared", "operation_id": operation_id}

    def start_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return self._submit("reference", operation_id, intent, registration_ref)

    def lookup_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return self._lookup("reference", operation_id, intent, registration_ref)

    def dispatch_case(self, operation_id: str, intent: dict, registration_ref: dict, authorization: dict) -> dict:
        return self._submit("case", operation_id, intent, registration_ref, authorization)

    def lookup_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return self._lookup("case", operation_id, intent, registration_ref)
