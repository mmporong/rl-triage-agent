"""v1 lifecycle와 T1을 유지하며 등록 metadata layer로 parent 원문을 회수한다."""
from __future__ import annotations

import base64
import hashlib
import os
import sys
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import agentic_brev_mailbox_service_v1 as V1
import agentic_brev_probe_transport_remote_v2 as REMOTE
import agentic_brev_runtime_host_v1 as H

T, E, R = V1.T, V1.E, V1.R
SCHEMA = V1.SCHEMA  # MainLifecycle 공개 schema·필드·시계는 그대로 쓴다.
REMOTE_SOURCE = REMOTE.SOURCE
SOURCES = REMOTE.SOURCES | H.SOURCES
write_new, logical = V1.write_new, V1.logical


def _target(reference, rows):
    value = REMOTE.locate(reference, rows, existing=False)
    path = value["_path"]
    T.plain_path(path)
    return path


def unpack_archive(archive, roots, allowed, *, metadata_roots, arm, operation_id,
                   parent_request_sha256, parent_protocol_sha256, binding_sha256):
    T.regular(archive)
    rows = REMOTE.maps_for_host(roots, metadata_roots)
    expected = {"parent_request_sha256": parent_request_sha256,
        "parent_protocol_sha256": parent_protocol_sha256, "binding_sha256": binding_sha256,
        "evidence_roots_sha256": E.digest(roots), "metadata_roots_sha256": E.digest(metadata_roots)}
    with V1.tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [row.name for row in members]
        T.require(len(members) <= 257 and sum(row.size for row in members) <= 65 * 1024 * 1024
            and all(0 <= row.size <= 64 * 1024 * 1024 and row.isfile() and not row.issym() and not row.islnk() for row in members)
            and len(set(names)) == len(names) and "manifest.json" in names, "regular unique bounded archive files required")
        T.require(bundle.getmember("manifest.json").size <= 1024 * 1024, "archive manifest cap")
        manifest = E.loads(V1.member_bytes(bundle, "manifest.json").decode("utf-8"))
        T.require(manifest.get("schema") == REMOTE.ARCHIVE_SCHEMA and manifest["arm"] == arm
            and manifest["operation_id"] == operation_id and manifest["authority_issued"] is False
            and manifest["evidence_accepted"] is False and all(manifest.get(key) == value for key, value in expected.items()),
            "archive identity/ordered maps/registration binding")
        files = manifest["files"]
        T.require(len(files) <= 256 and sum(row["bytes"] for row in files) <= 64 * 1024 * 1024
            and set(names) == {"manifest.json", *(row["archive_path"] for row in files)}
            and len({row["archive_path"] for row in files}) == len(files), "archive inventory differs")
        staged = {}
        for row in files:
            layer, index = row["layer"], row["root_index"]
            T.require(layer in REMOTE.LAYERS and type(index) is int, "archive registered layer/index")
            layer_roots = roots if layer == "evidence" else metadata_roots
            T.require(0 <= index < len(layer_roots), "archive root index")
            text = row["relative"]
            rel = PurePosixPath(text)
            T.require(isinstance(text, str) and bool(text) and not rel.is_absolute()
                and all(part not in {"", ".", ".."} for part in text.split("/"))
                and "\\" not in text and ":" not in text, "archive canonical relative path")
            root = layer_roots[index]
            T.require(row["archive_path"] == f"files/{layer}/{index}/{text}"
                and row["logical_path"] == str(REMOTE.producer_path(root["remote"]).joinpath(*rel.parts)), "archive layer/logical path binding")
            target = Path(root["local"]).joinpath(*rel.parts)
            T.plain_path(target)
            permissions = [entry for entry in allowed if Path(entry["path"]) in target.parents]
            T.require(len(permissions) == 1 and type(permissions[0]["writable"]) is bool, "archive destination outside unique allowed root")
            raw = V1.member_bytes(bundle, row["archive_path"])
            T.require(type(row["bytes"]) is int and len(raw) == row["bytes"]
                and hashlib.sha256(raw).hexdigest() == row["sha256"], "archive raw SHA/size changed")
            T.require(target not in staged, "duplicate native destination")
            if target.exists():
                T.regular(target)
                T.require(target.read_bytes() == raw, "existing raw file differs")
            else:
                T.require(permissions[0]["writable"] is True, "read-only source copy missing")
            staged[target] = raw
    def body(key):
        reference = manifest[key]
        target = _target(reference, rows)
        T.require(target in staged and hashlib.sha256(staged[target]).hexdigest() == reference["sha256"], "required parent metadata missing/SHA")
        return target, E.loads(staged[target].decode("utf-8"))

    outcome_path, outcome = body("outcome")
    plan_path, plan = body("plan")
    evidence_path, evidence = body("evidence")
    request_path, request = body("request")
    T.require(outcome_path.name == "outcome.json" and evidence_path == outcome_path.with_name("evidence.json")
        and request_path == outcome_path.with_name("request.json")
        and plan_path.name == "plan.json" and outcome["status"] == "complete" and outcome["operation_id"] == operation_id
        and outcome["plan"] == manifest["plan"] and outcome["evidence"] == manifest["evidence"]
        and plan["operation_id"] == operation_id and plan["identity"]["arm"] == arm
        and plan["parent_request"]["sha256"] == parent_request_sha256
        and E.digest(plan["roots"]) == manifest["producer_evidence_roots_sha256"] == evidence["root_map_sha256"]
        and request == evidence["request"] and E.digest(request) == outcome["request_sha256"], "parent metadata identity/content binding")
    for key in REMOTE.V1.REQUIRED_PLAN_REFS:
        ref = plan[key]
        target = _target(ref, rows)
        T.require(target in staged and hashlib.sha256(staged[target]).hexdigest() == ref["sha256"], "required plan metadata absent")
    T.require(not any(path.exists() for path in (outcome_path, evidence_path, request_path)), "existing/partial unit metadata is not replayable")
    claim = outcome_path.with_name("outcome.json.transport-v2-claim.json")
    T.plain_path(claim)
    T.require(not claim.exists(), "existing unpack intent; no replay")
    claim.parent.mkdir(parents=True, exist_ok=True)
    T.plain_path(claim)
    write_new(claim, {"archive_sha256": E.J.file_sha(archive), "operation_id": operation_id, "arm": arm,
        **expected, "authority_issued": False, "evidence_accepted": False})
    for target, raw in staged.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            T.plain_path(target)
            with target.open("xb") as stream:
                stream.write(raw)
        T.regular(target)
        T.require(target.read_bytes() == raw, "copied raw file changed")
    result = {"path": str(outcome_path), "sha256": manifest["outcome"]["sha256"]}
    T.read_ref(result)
    write_new(claim.with_name("outcome.json.transport-v2-completed.json"), {"outcome": result, "authority_issued": False, "evidence_accepted": False})
    return result


class BrevMailboxService(V1.BrevMailboxService):
    def __init__(self, runtime, bindings, contract_ref):
        self.runtime, self.bindings, self.contract_ref = runtime, bindings, contract_ref
        self.contract = T.read_ref(contract_ref)
        T.require(runtime.reg["allow_fixture"] is False and self.contract["schema"] == T.SCHEMA + "_contract"
            and runtime.protocol["host_transport_contract_sha256"] == contract_ref["sha256"], "registered service contract")
        actual = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        T.require(runtime.protocol["host_transport_service_sha256_lf"] == actual == self.contract["service_source_sha256_lf"]
            and SOURCES <= runtime.protocol["sources_lf"].keys(), "v2 service source closure changed")
        R.verify_sources(runtime.protocol, runtime.anchor, repo=runtime.repo, protocol_path=Path(runtime.protocol_ref["path"]))
        self.root = Path(self.contract["mailbox"])
        T.require(self.root == runtime.folder / "transport-mailbox", "service mailbox binding")
        T.plain_path(self.root)

    def _rpc(self, folder, plan, registration, mode, *extra):
        self._inventory(folder)
        repo, python = str(logical(plan["remote_repo"])), str(logical(plan["remote_python"]))
        request = str(logical(plan["remote_request"]))
        argv = [python, "-B", repo + "/" + REMOTE_SOURCE, "--request", request,
                "--request-sha256", registration["parent_request"]["sha256"], "--mode", mode, *extra]
        script = folder / f"rpc-{mode}-{time.time_ns()}.sh"
        with script.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(V1.remote_script(argv))
        native = str(script)
        if os.name == "nt":
            projection = self.contract["host_mount_projection"]
            T.require(native.startswith(projection["host"]), "registered WSL script mount")
            native = projection["wsl"] + native[len(projection["host"]):].replace("\\", "/")
        receipt = self._call(folder, "rpc-" + mode, ["exec", self.contract["target"]["resource_name"], "@" + native])
        lines = [line for line in Path(receipt["stdout"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines and lines[-1] == self.contract["target"]["resource_name"]:
            lines.pop()
        T.require(len(lines) == 1 and lines[0].startswith(V1.MARKER), "single remote RPC response required")
        response = E.loads(base64.b64decode(lines[0][len(V1.MARKER):], validate=True).decode())
        write_new(folder / f"rpc-{mode}-{time.time_ns()}.json", response)
        T.require(response["exit_code"] == 0, "remote helper failed; raw preserved")
        result = E.loads(base64.b64decode(response["stdout_base64"], validate=True).decode("utf-8"))
        T.require(result.get("status") != "HOLD" and result.get("schema") == REMOTE.SCHEMA, "v2 remote helper HOLD/schema")
        return result

    def _registered(self, request):
        registration = T.read_ref(request["registration"])
        protocol, binding, plan = (T.read_ref(registration[key]) for key in ("parent_protocol", "parent_binding", "service_plan"))
        parent_request = T.read_ref(registration["parent_request"])
        T.require(protocol["host_transport_service_plan_sha256"] == registration["service_plan"]["sha256"]
            and protocol["binding_sha256"] == registration["parent_binding"]["sha256"]
            and SOURCES <= protocol["sources_lf"].keys()
            and parent_request["protocol"]["sha256"] == registration["parent_protocol"]["sha256"]
            and parent_request["binding"]["sha256"] == registration["parent_binding"]["sha256"]
            and parent_request["source_anchor"] == registration["parent_anchor"], "registered exact parent/plan/raw refs")
        H._ancestor(self.runtime.repo, self.runtime.anchor, registration["parent_anchor"])
        R.verify_sources(protocol, registration["parent_anchor"], repo=self.runtime.repo, protocol_path=Path(registration["parent_protocol"]["path"]))
        execution = binding["expected_request"]["execution"]
        host = execution["runtime_host_binding"]
        T.require(host["operation_id"] == request["operation_id"]
            and host["runtime_source_anchor"] == self.runtime.anchor
            and host["runtime_protocol_sha256"] == self.runtime.protocol_ref["sha256"], "registered Runtime R/operation")
        for arm in REMOTE.ARMS:
            REMOTE.validate_maps(host["evidence_roots"][arm], host["metadata_roots"], host_native=True)
        self._validate_host_intent(request)
        return registration, plan, execution

    def serve_once(self, request_path):
        T.regular(request_path)
        T.require(self.root in request_path.parents and request_path.name == "request.json", "mailbox request path")
        request = E.loads(request_path.read_text(encoding="utf-8"))
        operation, phase = request["operation_id"], request["phase"]
        T.require(request_path == self.root / operation / phase / "request.json" and phase in {"reference", "case"}
            and request["schema"] == T.SCHEMA + "_request" and request["contract_sha256"] == self.contract_ref["sha256"]
            and self.bindings[operation] == request["registration"], "registered mailbox operation")
        response_path = request_path.with_name("response.json")
        if response_path.exists():
            previous = T.read_ref({"path": str(response_path), "sha256": E.J.file_sha(response_path)})
            T.require(previous.get("schema") == T.SCHEMA + "_response"
                and previous["request_sha256"] == E.J.file_sha(request_path)
                and previous["operation_id"] == operation and previous["phase"] == phase, "existing service response differs")
            return previous
        folder = request_path.parent / "service"
        if folder.exists():
            return {"status": "unknown", "reason": "existing_service_intent_lookup_only"}
        folder.mkdir()
        write_new(folder / "intent.json", {"request_sha256": E.J.file_sha(request_path)})
        registration, plan, execution = self._registered(request)
        arm = "NONE" if phase == "reference" else "F1"
        response = {"schema": T.SCHEMA + "_response", "request_sha256": E.J.file_sha(request_path),
            "operation_id": operation, "phase": phase, "status": "unknown"}
        try:
            if phase == "reference":
                self._rpc(folder, plan, registration, "prepare")
                self._rpc(folder, plan, registration, "start")
            else:
                T.require("authorization" in request, "case C authorization missing")
            self._provider(folder, plan, registration, execution, arm)
            deadline = time.monotonic() + plan["arm_wait_s"]
            while True:
                status = self._rpc(folder, plan, registration, "status", "--arm", arm)
                if status["status"] == "completed":
                    break
                T.require(status["status"] == "pending" and time.monotonic() < deadline, "parent incomplete/unknown; no redispatch")
                time.sleep(plan["poll_s"])
            remote = str(logical(plan["export_root"]) / f"{operation}-{arm}.tar.gz")
            packet = self._rpc(folder, plan, registration, "export", "--arm", arm, "--output", remote)
            archive = folder / "raw.tar.gz"
            self._inventory(folder)
            target = str(archive)
            if os.name == "nt":
                projection = self.contract["host_mount_projection"]
                T.require(target.startswith(projection["host"]), "archive mount projection")
                target = projection["wsl"] + target[len(projection["host"]):].replace("\\", "/")
            self._call(folder, "download", ["copy", self.contract["target"]["resource_name"] + ":" + remote, target], timeout=300)
            T.require(E.J.file_sha(archive) == packet["archive"]["sha256"], "download archive SHA")
            host = execution["runtime_host_binding"]
            result = unpack_archive(archive, host["evidence_roots"][arm], self.contract["allowed_copy_roots"],
                metadata_roots=host["metadata_roots"], arm=arm, operation_id=execution["operations"][arm]["operation_id"],
                parent_request_sha256=registration["parent_request"]["sha256"],
                parent_protocol_sha256=registration["parent_protocol"]["sha256"], binding_sha256=registration["parent_binding"]["sha256"])
            response.update(status="completed", result=result)
        except Exception as error:  # noqa: BLE001 - 첫 실패와 부분 출력 보존, 재전송0.
            response.update(status="unknown", error_type=type(error).__name__, reason=str(error))
        write_new(response_path, response)
        return response
