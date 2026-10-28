"""Registered Brev CLI bridge for a single owned repair lifetime."""
from __future__ import annotations

import base64
import inspect
import json
import os
import re
import time
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import agentic_brev_mailbox_service_v1 as OLD
import agentic_brev_repair_transport_remote_v1 as REMOTE
import agentic_brev_repair_transport_v1 as T

SCHEMA = T.SCHEMA + "_service"
SOURCES = T.SOURCES
MARKER = "AGENTIC_REPAIR_RPC="
PRODUCTION_RUN = OLD.subprocess.run


def remote_script(argv, payload, *, helper_sha256, timeout):
    T.require(isinstance(argv, list) and all(isinstance(x, str) for x in argv)
              and len(argv) == 6 and argv[1:3] == ["-I", "-B"] and argv[-2] == "--mode"
              and argv[-1] in {"admit", "execute", "lookup"} and type(timeout) is int and timeout > 0
              and re.fullmatch(r"[0-9a-f]{64}", helper_sha256) is not None, "fixed remote helper command")
    encoded_argv = base64.b64encode(json.dumps(argv).encode()).decode()
    encoded_payload = base64.b64encode(T.E.J._bytes(payload)).decode()
    return r"""#!/usr/bin/env bash
set -u
python3 - <<'PY'
import base64,hashlib,json,os,re,subprocess
args=json.loads(base64.b64decode('@@ARGV@@'))
payload=base64.b64decode('@@PAYLOAD@@')
with open(args[3],'rb') as stream: source=stream.read().replace(b'\r\n',b'\n')
if hashlib.sha256(source).hexdigest()!='@@SHA@@':raise RuntimeError('remote helper source SHA changed')
env={key:value for key,value in os.environ.items() if re.search(r'(?:^|_)(?:API|TOKEN|SECRET|KEY)(?:_|$)',key,re.I) is None and key not in ('PYTHONPATH','PYTHONHOME')}
try:
 result=subprocess.run(args,input=payload,capture_output=True,env=env,timeout=@@TIMEOUT@@,check=False)
 reply={'exit_code':result.returncode,'stdout_base64':base64.b64encode(result.stdout).decode(),'stderr_base64':base64.b64encode(result.stderr).decode()}
except subprocess.TimeoutExpired as error:
 reply={'exit_code':None,'timed_out':True,'stdout_base64':base64.b64encode(error.stdout or b'').decode(),'stderr_base64':base64.b64encode(error.stderr or b'').decode()}
print('@@MARKER@@'+base64.b64encode(json.dumps(reply).encode()).decode(),flush=True)
PY
exit 0
""".replace("@@ARGV@@", encoded_argv).replace("@@PAYLOAD@@", encoded_payload).replace("@@SHA@@", helper_sha256).replace(
        "@@TIMEOUT@@", str(timeout)).replace("@@MARKER@@", MARKER)


class BrevRepairMailboxService(OLD.BrevMailboxService):
    """Main owns VM Start/Stop; this service only uses registered ls/exec."""

    def __init__(self, registration_ref, root_map, contract_ref, *, allow_fixture=False, fixture_rpc: Any = None):
        self.registration_ref, self.root_map, self.contract_ref = deepcopy(registration_ref), deepcopy(root_map), deepcopy(contract_ref)
        self.allow_fixture, self.fixture_rpc = allow_fixture, fixture_rpc
        reg, self.contract = T.registered(registration_ref, root_map, contract_ref, allow_fixture=allow_fixture)
        self.runtime = SimpleNamespace(protocol=reg["protocol"], anchor=reg["raw"]["runtime"]["anchor"], repo=reg["repo"])
        self.production_identity()

    def production_identity(self):
        T.require(type(self) is BrevRepairMailboxService, "exact repair service class required")
        for name in ("admitter", "_admission", "execute", "lookup", "_registered", "_rpc", "_budget", "_clock"):
            method = getattr(self, name, None)
            T.require(inspect.ismethod(method) and method.__self__ is self
                      and method.__func__ is getattr(BrevRepairMailboxService, name), "repair service method identity")
        if not self.allow_fixture:
            T.require(self.fixture_rpc is None and OLD.subprocess.run is PRODUCTION_RUN
                      and self._call.__func__ is OLD.BrevMailboxService._call
                      and self._inventory.__func__ is OLD.BrevMailboxService._inventory,
                      "production CLI identity/injection guard")
        else:
            T.require(self.fixture_rpc is not None and self.fixture_rpc.synthetic_cpu is True, "explicit synthetic RPC substrate")

    def _registered(self):
        self.production_identity()
        reg, config = T.registered(self.registration_ref, self.root_map, self.contract_ref, allow_fixture=self.allow_fixture)
        T.require(config == self.contract and T.E.C.digest_lf_file(Path(__file__)) == config["service_source_sha256_lf"], "service raw/source changed")
        return reg, config

    def _clock(self):
        return self.fixture_rpc.clock() if self.allow_fixture else T.clock()

    def _budget(self, seconds=0):
        reg, config = self._registered()
        T.require(not T.safe(Path(config["lifecycle_stop_intent_path"])).exists(), "main Stop intent forbids transport")
        phase_path = reg["resolver"].file(config["lifecycle_protocol"])
        phase = reg["resolver"].read(config["lifecycle_protocol"])
        T.require(phase["schema"] == OLD.SCHEMA + "_lifecycle_protocol"
                  and phase["budget_origin"] == "immutable_registered_at_utc"
                  and all(phase[k] == config["target"][k] for k in ("resource_id", "resource_name", "instance_type")), "immutable phase identity")
        T.CORE.R.verify_sources(phase, reg["raw"]["runtime"]["anchor"], repo=reg["repo"], protocol_path=phase_path)
        now = self._clock()
        elapsed = Decimal(str((now - T.E.E._stamp(phase["registered_at_utc"])).total_seconds()))
        reserve, rate, cap = (Decimal(str(phase[k])) for k in ("stop_reserve_s", "quoted_usd_per_h", "compute_cap_usd"))
        T.require(all(v.is_finite() and v > 0 for v in (reserve, rate, cap)) and elapsed >= 0
                  and elapsed + seconds + reserve <= Decimal(str(phase["max_lifecycle_s"]))
                  and (elapsed + seconds + reserve) * rate / 3600 <= cap
                  and now + timedelta(seconds=float(Decimal(str(seconds)) + reserve)) <= T.E.E._stamp(config["deadline_utc"])
                  and now + timedelta(seconds=float(Decimal(str(seconds)) + reserve)) <= T.E.E._stamp(phase["deadline_utc"]), "phase time/money/deadline reservation")

    def _folder(self, operation):
        T.require(len(operation) == 64 and all(c in "0123456789abcdef" for c in operation), "operation id")
        folder = Path(self.contract["host_control_root"]) / operation
        T.safe(folder)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _rpc(self, operation, request_ref, authorization_ref, admission_ref, mode):
        reg, config = self._registered()
        folder = self._folder(operation)
        self._budget(config["limits"]["rpc_timeout_s"])
        calls = list(folder.glob("rpc-*/intent.json"))
        T.require(len(calls) < config["limits"]["max_rpc_calls"], "registered RPC call cap exhausted")
        rpc = folder / ("rpc-" + str(time.time_ns()))
        rpc.mkdir()
        if self.allow_fixture:
            inventory_ref, observed = self.fixture_rpc.inventory(rpc, config["target"])
        else:
            receipt, inventory_raw = self._inventory(rpc)
            inventory_ref = T.new_bytes(rpc / "provider.json", inventory_raw)
            observed = receipt["request_before_utc"]
        T.save(rpc / "intent.json", {"mode": mode, "operation_id": operation, "request_ref": request_ref,
            "authorization_ref": authorization_ref, "admission_ref": admission_ref,
            "registration_ref": self.registration_ref, "contract_ref": self.contract_ref, "provider": inventory_ref,
            "provider_observed_at_utc": observed, "authority_issued": False})
        roots = reg["layout"]["root_maps"]["provider"]
        seeds = [request_ref, authorization_ref, self.registration_ref, self.contract_ref, inventory_ref]
        if admission_ref is not None:
            seeds.append(admission_ref)
        upload = T.packet(seeds, reg, config, "host", operation=operation, kind="host-upload")
        body = {"operation_id": operation, "registration_ref": self.registration_ref, "contract_ref": self.contract_ref,
            "root_map": roots, "request_ref": request_ref, "authorization_ref": authorization_ref,
            "admission_ref": admission_ref, "inventory_ref": inventory_ref, "inventory_observed_at_utc": observed, "upload": upload}
        if self.allow_fixture:
            response = self.fixture_rpc(mode, body)
            T.save(rpc / "fixture-response.json", response)
        else:
            argv = [config["remote_python"], "-I", "-B", config["remote_repo"].rstrip("/") + "/" + T.REMOTE_SOURCE, "--mode", mode]
            script = rpc / "invoke.sh"
            T.new_bytes(script, remote_script(argv, body, helper_sha256=config["remote_source_sha256_lf"],
                                              timeout=config["limits"]["rpc_timeout_s"]).encode())
            native = str(script)
            if os.name == "nt":
                mapping = config["host_mount_projection"]
                T.require(native.startswith(mapping["host"]), "registered WSL mount projection")
                native = mapping["wsl"] + native[len(mapping["host"]):].replace("\\", "/")
            receipt = self._call(rpc, "repair-" + mode, ["exec", config["target"]["resource_name"], "@" + native],
                                 timeout=config["limits"]["rpc_timeout_s"] + 10)
            lines = Path(receipt["stdout"]).read_text(encoding="utf-8").splitlines()
            if lines and lines[-1] == config["target"]["resource_name"]:
                lines.pop()
            T.require(len(lines) == 1 and lines[0].startswith(MARKER), "single repair RPC envelope required")
            envelope = T.E.E.loads(base64.b64decode(lines[0][len(MARKER):], validate=True).decode())
            T.save(rpc / "envelope.json", envelope)
            if envelope.get("timed_out") or envelope.get("exit_code") is None:
                raise T.RepairPending("RPC interrupted; preserve owner and lookup only")
            T.require(envelope["exit_code"] == 0, "remote repair helper failed; raw stdout/stderr preserved")
            response = T.E.E.loads(base64.b64decode(envelope["stdout_base64"], validate=True).decode())
        T.require(response["schema"] == REMOTE.SCHEMA and response["operation_id"] == operation
                  and response["request_ref"] == request_ref and response["mode"] == mode
                  and response["authority_issued"] is False and response["evidence_accepted"] is False, "remote reply identity")
        if response["packet"] is not None:
            T.save(rpc / "archive-envelope.json", response["packet"])
            T.unpack(response["packet"], config, "host", operation=operation, kind="remote-" + mode,
                     archive_root=rpc / "raw-archive")
        self._budget()
        return response

    def admitter(self, operation_id, request_ref, authorization_ref, registration_ref):
        T.require(registration_ref == self.registration_ref, "admitter registration ref")
        folder = self._folder(operation_id)
        claim = folder / "admit-intent.json"
        body = {"operation_id": operation_id, "request_ref": request_ref, "authorization_ref": authorization_ref,
                "registration_ref": registration_ref}
        if claim.exists():
            T.require(T.E.E.loads(claim.read_text(encoding="utf-8")) == body, "admission request changed")
            hold = folder / "admission-hold.json"
            if hold.exists():
                saved = T.E.E.loads(T.safe(hold, exists=True).read_text(encoding="utf-8"))
                T.require(saved["intent"] == body, "admission HOLD identity changed")
                raise ValueError("admission recovery HOLD: " + saved["reason"])
            known = folder / "admission-ref.json"
            reference = T.E.E.loads(T.safe(known, exists=True).read_text(encoding="utf-8")) if known.exists() else None
            response = self._rpc(operation_id, request_ref, authorization_ref, reference, "lookup")
        else:
            T.save(claim, body)
            response = self._rpc(operation_id, request_ref, authorization_ref, None, "admit")
        return self._admission(response, body, folder)

    def _admission(self, response, intent, folder):
        state = response["state"]
        if state["status"] == "hold" or state["owner_ended"] is True:
            reason = state.get("reason", "owner_ended_or_held")
            T.save(folder / "admission-hold.json", {"schema": SCHEMA + "_admission_hold", "intent": intent,
                "reason": reason, "state": state, "authority_issued": False, "evidence_accepted": False})
            raise ValueError("admission recovery HOLD: " + reason)
        if state["owner_current"] is not True or state.get("admission") is None:
            raise T.RepairPending("existing owner admission pending; registered lookup only")
        reg, _ = self._registered()
        admission = state["admission"]
        try:
            T.require(reg["resolver"].read(state["admission_pointer"]) == admission, "original owner admission pointer differs")
            value = REMOTE.UNIT.admission_body(admission, intent["operation_id"], reg, self._clock(), entry=True)
            T.require(value["registration"] == intent["registration_ref"], "admission registration changed")
        except (ValueError, KeyError, TypeError, OSError) as error:
            T.save(folder / "admission-hold.json", {"schema": SCHEMA + "_admission_hold", "intent": intent,
                "reason": "invalid_or_stale_admission", "error": str(error), "state": state,
                "authority_issued": False, "evidence_accepted": False})
            raise ValueError("admission recovery HOLD: invalid or stale admission") from error
        T.save(folder / "admission-ref.json", admission)
        return admission

    def execute(self, operation_id, request_ref, authorization_ref, admission_ref):
        folder = self._folder(operation_id)
        expected = {"request_ref": request_ref, "authorization_ref": authorization_ref, "admission_ref": admission_ref}
        claim = folder / "execute-intent.json"
        if claim.exists():
            T.require(T.E.E.loads(claim.read_text(encoding="utf-8")) == expected, "execute request changed")
            return self.lookup(operation_id, request_ref)
        T.save(claim, expected)
        response = self._rpc(operation_id, request_ref, authorization_ref, admission_ref, "execute")
        result = self._outcome(response)
        if result is not None:
            T.save(folder / "completed-response.json", response)
        return result

    def _outcome(self, response):
        state = response["state"]
        if state["status"] == "hold":
            raise ValueError("remote owner HOLD; raw failure/cleanup retained, main Stop may be required")
        if state["status"] != "completed":
            return None
        T.require(state["owner_ended"] is True and state["owner_current"] is False and state["release"] is not None,
                  "owned parent/IO not terminated")
        reg, _ = self._registered()
        release = reg["resolver"].read(state["release"])
        T.require(release["operation_id"] == response["operation_id"] and release["request_ref"] == response["request_ref"]
                  and release["owned_children_closed"] is True and release["flock_released"] is True
                  and release["heartbeat_stopped"] is True and release["synthetic_cpu"] is self.allow_fixture,
                  "verified ownership/lease termination required")
        return state["outcome"]

    def lookup(self, operation_id, request_ref):
        folder = self._folder(operation_id)
        path = folder / "execute-intent.json"
        if not path.exists():
            return None
        intent = T.E.E.loads(path.read_text(encoding="utf-8"))
        T.require(intent["request_ref"] == request_ref, "lookup request changed")
        completed = folder / "completed-response.json"
        if completed.exists():
            response = T.E.E.loads(T.safe(completed, exists=True).read_text(encoding="utf-8"))
            T.require(response["request_ref"] == request_ref and response["operation_id"] == operation_id, "cached repair completion identity")
            return self._outcome(response)
        response = self._rpc(operation_id, request_ref, intent["authorization_ref"], intent["admission_ref"], "lookup")
        result = self._outcome(response)
        if result is not None:
            T.save(completed, response)
        return result
