"""검사 운반 원장 한 건을 실제 Brev CLI와 원격 helper로 처리한다. VM 수명주기는 메인 소유다."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath

import agentic_brev_mailbox_transport_v1 as T
import agentic_brev_probe_evidence_v1 as E
import agentic_investigation_runtime_v1 as R

SCHEMA = "agentic_brev_mailbox_service_v1"
REMOTE = "bench/agentic_brev_probe_transport_remote_v1.py"
MARKER = "AGENTIC_TRANSPORT_RPC="


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def logical(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    T.require(path.is_absolute() and ".." not in path.parts and "\\" not in value,
              "registered remote POSIX path required")
    return path


def remote_script(argv: list[str]) -> str:
    """helper의 최초 stdout·stderr·종료 코드를 보존한다. Brev의 재실행은 helper의 배타 intent가 거부한다."""
    encoded = base64.b64encode(json.dumps(argv).encode()).decode()
    return """#!/usr/bin/env bash
set -u
python3 - <<'PY'
import base64,json,os,subprocess
args=json.loads(base64.b64decode('@@ARGV@@'))
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
for key in ('NVIDIA_API_KEY','OPENAI_API_KEY'):env.pop(key,None)
try:
 result=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,env=env,timeout=120)
 value={'exit_code':result.returncode,'stdout_base64':base64.b64encode(result.stdout).decode(),'stderr_base64':base64.b64encode(result.stderr).decode()}
except subprocess.TimeoutExpired as error:
 value={'exit_code':None,'timed_out':True,'stdout_base64':base64.b64encode(error.stdout or b'').decode(),'stderr_base64':base64.b64encode(error.stderr or b'').decode()}
print('@@MARKER@@'+base64.b64encode(json.dumps(value).encode()).decode(),flush=True)
PY
exit 0
""".replace("@@ARGV@@", encoded).replace("@@MARKER@@", MARKER)


def member_bytes(source: tarfile.TarFile, name: str) -> bytes:
    stream = source.extractfile(name)
    if stream is None:
        raise ValueError("regular archive member required")
    with stream:
        return stream.read()


def unpack_archive(archive: Path, roots: list[dict], allowed: list[dict], *, arm: str,
                   operation_id: str) -> dict:
    """등록 root에 raw 파일을 보존한다. 기존 사본은 SHA가 같은 경우에만 재사용한다."""
    T.regular(archive)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        T.require(len(members) <= 257 and sum(row.size for row in members) <= 65 * 1024 * 1024
                  and all(0 <= row.size <= 64 * 1024 * 1024 for row in members), "archive member cap")
        names = [row.name for row in members]
        T.require(len(names) == len(set(names)) and "manifest.json" in names
                  and all(row.isfile() and not row.issym() and not row.islnk() for row in members),
                  "regular unique archive files required")
        T.require(source.getmember("manifest.json").size <= 1024 * 1024, "archive manifest cap")
        manifest = E.loads(member_bytes(source, "manifest.json").decode("utf-8"))
        T.require(manifest["arm"] == arm and manifest["operation_id"] == operation_id
                  and manifest["authority_issued"] is False and manifest["evidence_accepted"] is False,
                  "archive identity/authority binding")
        rows = manifest["files"]
        T.require(len(rows) <= 256 and sum(row["bytes"] for row in rows) <= 64 * 1024 * 1024,
                  "archive size/file cap")
        T.require(set(names) == {"manifest.json", *(row["archive_path"] for row in rows)}
                  and len({row["archive_path"] for row in rows}) == len(rows), "archive inventory differs")
        staged = []
        for row in rows:
            index = row["root_index"]
            T.require(type(index) is int and 0 <= index < len(roots), "archive root index")
            relative = PurePosixPath(row["relative"])
            T.require(not relative.is_absolute() and bool(relative.parts) and ".." not in relative.parts
                      and "\\" not in row["relative"] and ":" not in row["relative"], "archive relative path")
            T.require(row["archive_path"] == f"files/{index}/{relative.as_posix()}"
                      and str(logical(roots[index]["remote"]) / relative) == row["logical_path"],
                      "archive root/logical path binding")
            target = Path(roots[index]["local"]).joinpath(*relative.parts)
            T.plain_path(target)
            permissions = [entry for entry in allowed if Path(entry["path"]) in target.parents]
            T.require(len(permissions) == 1, "archive destination outside unique allowed root")
            raw = member_bytes(source, row["archive_path"])
            T.require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                      "archive raw SHA/size changed")
            if target.exists():
                T.regular(target)
                T.require(target.read_bytes() == raw, "existing raw file differs")
            else:
                T.require(permissions[0]["writable"] is True, "read-only source copy missing")
            staged.append((target, raw))
        # 모든 항목을 확인한 뒤 새 파일만 만든다. 기존 원문은 덮어쓰지 않는다.
        for target, raw in staged:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                T.plain_path(target)
                with target.open("xb") as stream:
                    stream.write(raw)
            T.regular(target)
            T.require(target.read_bytes() == raw, "copied raw file changed")
        reference = manifest["outcome"]
        candidates = [(logical(row["remote"]), Path(row["local"])) for row in roots
                      if logical(row["remote"]) in logical(reference["path"]).parents]
        T.require(len(candidates) == 1, "outcome root mapping")
        remote, local = candidates[0]
        path = local.joinpath(*logical(reference["path"]).relative_to(remote).parts)
        result = {"path": str(path), "sha256": reference["sha256"]}
        T.read_ref(result)
        return result


class BrevMailboxService:
    """등록된 소유 VM에서 요청 한 건을 처리한다. Start/Stop은 발급하지 않는다."""

    def __init__(self, runtime, bindings: dict, contract_ref: dict):
        self.runtime, self.bindings, self.contract_ref = runtime, bindings, contract_ref
        self.contract = T.read_ref(contract_ref)
        T.require(runtime.reg["allow_fixture"] is False and self.contract["schema"] == T.SCHEMA + "_contract"
                  and runtime.protocol["host_transport_contract_sha256"] == contract_ref["sha256"],
                  "registered service contract")
        expected = runtime.protocol["host_transport_service_sha256_lf"]
        actual = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        T.require(expected == actual == self.contract["service_source_sha256_lf"], "service source changed")
        self.root = Path(self.contract["mailbox"])
        T.require(self.root == runtime.folder / "transport-mailbox", "service mailbox binding")
        T.plain_path(self.root)

    def _budget(self, seconds: int = 0) -> None:
        stop = Path(self.contract["lifecycle_stop_intent_path"])
        T.plain_path(stop)
        T.require(not stop.exists(), "Stop intent forbids further transport")
        # 비용 입장은 변경 가능한 Start 기록을 신뢰하지 않는다. 더 이른 사전등록 시각부터
        # 전체 경과를 세므로 정지/재기동·새 receipt로 시계를 초기화할 수 없다.
        reference = self.contract["lifecycle_protocol"]
        T.require(reference["sha256"] == self.runtime.protocol["lifecycle_protocol_sha256"],
                  "registered immutable budget phase SHA")
        path = Path(reference["path"])
        T.regular(path)
        relative = path.resolve(strict=True).relative_to(self.runtime.repo.resolve()).as_posix()
        T.require(relative.startswith("bench/protocols/"), "public committed budget phase required")
        phase = T.read_ref(reference)
        fields = {"schema", "registered_at_utc", "deadline_utc", "resource_id", "resource_name",
                  "instance_type", "quoted_usd_per_h", "compute_cap_usd", "stop_reserve_s",
                  "max_lifecycle_s", "budget_origin", "source_commit", "sources_lf"}
        T.require(set(phase) == fields and phase["schema"] == SCHEMA + "_lifecycle_protocol"
                  and phase["budget_origin"] == "immutable_registered_at_utc", "immutable budget phase schema")
        R.verify_sources(phase, self.runtime.anchor, repo=self.runtime.repo, protocol_path=path)
        T.require(all(phase[key] == self.contract["target"][key]
                      for key in ("resource_id", "resource_name", "instance_type")), "budget VM identity")
        now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
        start = E._stamp(phase["registered_at_utc"])
        elapsed = Decimal(str((now - start).total_seconds()))
        reserve, rate, cap = (Decimal(str(phase[key])) for key in
                              ("stop_reserve_s", "quoted_usd_per_h", "compute_cap_usd"))
        T.require(all(value.is_finite() and value > 0 for value in (reserve, rate, cap))
                  and elapsed >= 0 and elapsed + seconds + reserve <= Decimal(str(phase["max_lifecycle_s"]))
                  and now < E._stamp(self.contract["deadline_utc"])
                  and now < E._stamp(phase["deadline_utc"])
                  and (elapsed + seconds + reserve) * rate / 3600 <= cap,
                  "immutable phase deadline/budget/Stop boundary")

    def _call(self, folder: Path, label: str, args: list[str], *, timeout: int = 120) -> dict:
        self._budget(timeout)
        path = folder / (str(time.time_ns()) + "-" + label)
        path.mkdir(parents=True, exist_ok=False)
        prefix = self.contract["cli_argv_prefix"]
        T.require(isinstance(prefix, list) and bool(prefix) and all(isinstance(arg, str) for arg in prefix), "CLI argv prefix")
        if os.name == "nt":
            T.require(prefix[:4] == ["wsl", "-d", "Ubuntu", "--exec"] and len(prefix) == 5
                      and prefix[-1].endswith("/.local/bin/brev"), "registered WSL Brev CLI only")
        else:
            T.require(len(prefix) == 1 and Path(prefix[0]).name == "brev", "registered native Brev CLI only")
        command = [*prefix, *args]
        write_new(path / "intent.json", {"command": command, "request_before_utc": datetime.now(timezone.utc).isoformat()})  # noqa: UP017 - Python 3.10
        env = dict(os.environ)
        for key in ("NVIDIA_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        try:
            completed = subprocess.run(command, capture_output=True, timeout=timeout, env=env, check=False)
            code, stdout, stderr, timed_out = completed.returncode, completed.stdout, completed.stderr, False
        except subprocess.TimeoutExpired as error:
            code, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
        for name, raw in (("stdout", stdout), ("stderr", stderr)):
            (path / (name + ".bin")).write_bytes(raw)
        receipt = {"exit_code": code, "timed_out": timed_out, "stdout": str(path / "stdout.bin"),
                   "stderr": str(path / "stderr.bin"), "request_before_utc": T.read_ref({
                       "path": str(path / "intent.json"), "sha256": E.J.file_sha(path / "intent.json")})["request_before_utc"]}
        write_new(path / "receipt.json", receipt)
        T.require(code == 0, "Brev transport failed; no automatic resubmit")
        return receipt

    def _inventory(self, folder: Path) -> tuple[dict, bytes]:
        receipt = self._call(folder, "inventory", ["ls", "--json"], timeout=60)
        raw = Path(receipt["stdout"]).read_bytes()
        rows = E.loads(raw.decode("utf-8"))["workspaces"]
        target = self.contract["target"]
        matches = [row for row in rows if row.get("id") == target["resource_id"]]
        T.require(len(matches) == 1 and sum(row.get("name") == target["resource_name"] for row in rows) == 1
                  and matches[0]["name"] == target["resource_name"]
                  and matches[0]["instance_type"] == target["instance_type"]
                  and matches[0]["status"] == "RUNNING", "registered owned VM identity/RUNNING required")
        return receipt, raw

    def _rpc(self, folder: Path, plan: dict, registration: dict, mode: str, *extra: str) -> dict:
        self._inventory(folder)
        repo, python = str(logical(plan["remote_repo"])), str(logical(plan["remote_python"]))
        request = str(logical(plan["remote_request"]))
        argv = [python, "-B", repo + "/" + REMOTE, "--request", request,
                "--request-sha256", registration["parent_request"]["sha256"], "--mode", mode, *extra]
        script = folder / f"rpc-{mode}-{time.time_ns()}.sh"
        with script.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(remote_script(argv))
        # WSL CLI가 읽을 수 있는 등록 mount prefix만 사용한다.
        native = str(script)
        if os.name == "nt":
            prefix = self.contract["host_mount_projection"]
            T.require(native.startswith(prefix["host"]), "registered WSL script mount")
            native = prefix["wsl"] + native[len(prefix["host"]):].replace("\\", "/")
        receipt = self._call(folder, "rpc-" + mode, ["exec", self.contract["target"]["resource_name"], "@" + native])
        text = Path(receipt["stdout"]).read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and lines[-1] == self.contract["target"]["resource_name"]:
            lines.pop()
        T.require(len(lines) == 1 and lines[0].startswith(MARKER), "single remote RPC response required")
        response = E.loads(base64.b64decode(lines[0][len(MARKER):], validate=True).decode())
        write_new(folder / f"rpc-{mode}-{time.time_ns()}.json", response)
        T.require(response["exit_code"] == 0, "remote helper failed; raw preserved")
        result = E.loads(base64.b64decode(response["stdout_base64"], validate=True).decode("utf-8"))
        T.require(result.get("status") != "HOLD", "remote helper HOLD; no automatic retry")
        return result

    def _provider(self, folder: Path, plan: dict, registration: dict, execution: dict, arm: str) -> None:
        receipt, raw = self._inventory(folder)
        operation = execution["operations"][arm]
        packet = {"schema": "agentic_brev_probe_pair_parent_v1_provider_packet", "operation_id": operation["operation_id"],
                  "observed_at_utc": receipt["request_before_utc"], "inventory": {
                      "path": operation["provider_packet_path"] + ".inventory.json", "sha256": hashlib.sha256(raw).hexdigest()}}
        payload = base64.b64encode(json.dumps({"packet": packet, "inventory_base64": base64.b64encode(raw).decode()}).encode()).decode()
        self._rpc(folder, plan, registration, "provider", "--arm", arm, "--payload-base64", payload)

    def _validate_host_intent(self, request: dict) -> None:
        import agentic_brev_runtime_host_v1 as H

        operation = request["operation_id"]
        path = self.runtime.folder / "brev-host" / operation / "intent.json"
        T.regular(path)
        value = E.loads(path.read_text(encoding="utf-8"))
        T.require(value["registration"] == request["registration"]
                  and E.digest(value["intent"]) == request["intent_sha256"], "immutable host intent changed")
        transport = T.BrevMailboxTransport(self.runtime, self.contract_ref)
        with H.BrevRuntimeHost(self.runtime, self.bindings, transport, "registered-main-io-service") as host:
            bound = host._binding(operation, value["intent"])
            if request["phase"] == "case":
                existing = host._authorization(bound)
                T.require(existing is not None and existing[1] is False
                          and existing[0] == request.get("authorization"), "case authorization not consumed/exact")

    def serve_once(self, request_path: Path) -> dict:
        T.regular(request_path)
        T.require(self.root in request_path.parents and request_path.name == "request.json", "mailbox request path")
        request = E.loads(request_path.read_text(encoding="utf-8"))
        operation, phase = request["operation_id"], request["phase"]
        T.require(request_path == self.root / operation / phase / "request.json" and phase in {"reference", "case"}
                  and request["schema"] == T.SCHEMA + "_request"
                  and request["contract_sha256"] == self.contract_ref["sha256"]
                  and self.bindings[operation] == request["registration"], "registered mailbox operation")
        response_path = request_path.with_name("response.json")
        if response_path.exists():
            T.regular(response_path)
            previous = E.loads(response_path.read_text(encoding="utf-8"))
            T.require(previous.get("schema") == T.SCHEMA + "_response"
                      and previous.get("request_sha256") == E.J.file_sha(request_path)
                      and previous.get("operation_id") == operation and previous.get("phase") == phase,
                      "existing service response differs")
            return previous
        folder = request_path.parent / "service"
        if folder.exists():
            return {"status": "unknown", "reason": "existing_service_intent_lookup_only"}
        folder.mkdir()
        write_new(folder / "intent.json", {"request_sha256": E.J.file_sha(request_path)})
        registration = T.read_ref(request["registration"])
        protocol = T.read_ref(registration["parent_protocol"])
        plan = T.read_ref(registration["service_plan"])
        T.require(protocol["host_transport_service_plan_sha256"] == registration["service_plan"]["sha256"],
                  "operation service plan registration")
        R.verify_sources(protocol, registration["parent_anchor"], repo=self.runtime.repo,
                         protocol_path=Path(registration["parent_protocol"]["path"]))
        binding = T.read_ref(registration["parent_binding"])
        execution = binding["expected_request"]["execution"]
        T.require(execution["runtime_host_binding"]["operation_id"] == operation, "runtime operation/parent binding")
        self._validate_host_intent(request)
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
                T.require(status["status"] == "pending" and time.monotonic() < deadline,
                          "parent incomplete/unknown; no redispatch")
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
            result = unpack_archive(archive, execution["runtime_host_binding"]["evidence_roots"][arm],
                                    self.contract["allowed_copy_roots"], arm=arm,
                                    operation_id=execution["operations"][arm]["operation_id"])
            response.update(status="completed", result=result)
        except Exception as error:  # noqa: BLE001 - 최초 운반 실패를 unknown으로 보존하고 재전송하지 않는다.
            response.update(status="unknown", error_type=type(error).__name__, reason=str(error))
        write_new(response_path, response)
        return response
