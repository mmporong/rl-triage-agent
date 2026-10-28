"""Probe Host1 delegation plus one explicitly registered Brev reward repair."""
from __future__ import annotations

import inspect
import sys
import threading
from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "src")]

import agentic_brev_repair_evidence_v1 as E
import agentic_brev_runtime_host_v1 as H1
import agentic_investigation_runtime_v1 as R
import run_agentic_brev_repair_unit_v1 as UNIT

C, J = E.C, E.J
SOURCES = UNIT.SOURCES | H1.SOURCES


class BrevRuntimeHost(AbstractContextManager):
    """Keep this context around Runtime.advance, including later receipt replay."""

    def __init__(self, runtime, probe_host, provider, registration_ref, root_map, approver, *,
                 probe_seals, probe_records, admitter, allow_fixture=False):
        E.require(isinstance(runtime, R.Runtime) and runtime.reg["allow_fixture"] is allow_fixture
                  and type(allow_fixture) is bool and isinstance(approver, str) and bool(approver.strip()), "host mode/approver")
        E.require(allow_fixture or (type(probe_host) is H1.BrevRuntimeHost and probe_host.runtime is runtime),
                  "actual probe callbacks require frozen Host1")
        self.runtime, self.provider = runtime, provider
        self.probe_host: Any = probe_host
        self.registration_ref, self.root_map = deepcopy(registration_ref), deepcopy(root_map)
        self.probe_seals, self.probe_records = deepcopy(probe_seals), deepcopy(probe_records)
        self.approver, self.admitter, self.allow_fixture = approver.strip(), admitter, allow_fixture
        self.folder = runtime.folder / "brev-repair-host"
        self.owner: int | None = None
        self.artifacts: E.RepairArtifactScope | None = None
        self._sources()

    def _sources(self):
        registered = E.registration(self.registration_ref, self.root_map, allow_fixture=self.allow_fixture)
        E.require(registered["protocol"] == self.runtime.protocol
                  and registered["raw"]["runtime"]["protocol"] == self.runtime.protocol_ref
                  and registered["raw"]["runtime"]["anchor"] == self.runtime.anchor
                  and SOURCES <= registered["protocol"]["sources_lf"].keys()
                  and self.runtime.reg["bands"] == registered["contract"]["bands"], "host/runtime/repair source binding")
        provider = registered["layout"]["provider"]
        methods = [getattr(self.provider, name, None) for name in ("execute", "lookup")]
        for method in methods:
            if not inspect.ismethod(method):
                raise TypeError("registered provider bound methods required")
            E.require(method.__self__ is self.provider and method.__func__ is getattr(type(self.provider), method.__name__),
                      "registered provider bound methods")
            source = inspect.getsourcefile(method.__func__)
            E.require(source is not None and C.digest_lf_file(Path(source)) == provider["sha256_lf"]
                      and registered["protocol"]["sources_lf"].get(provider["path"]) == provider["sha256_lf"]
                      and type(self.provider).__name__ == provider["class"], "provider service source/class binding")
        source = inspect.getsourcefile(self.admitter)
        expected = registered["layout"]["admission_service"]
        E.require(source is not None and C.digest_lf_file(Path(source)) == expected["sha256_lf"]
                  and registered["protocol"]["sources_lf"].get(expected["path"]) == expected["sha256_lf"],
                  "admission service source binding")
        E.require(getattr(self.provider, "allow_fixture", None) is self.allow_fixture
                  and getattr(self.provider, "registration_ref", None) == self.registration_ref,
                  "provider fixture/registration boundary")
        return registered

    def __enter__(self):
        E.require(self.owner is None, "nested Host2")
        registered = self._sources()
        self.probe_host.__enter__()
        try:
            scope = self.probe_host.scope
            for row in self.probe_records:
                if E.J.digest(E.E.loads(Path(row["reference"]["path"]).read_text(encoding="utf-8"))) not in scope.entries:
                    scope.register(**row)
            self.artifacts = E.RepairArtifactScope({"reference": self.registration_ref, "root_map": self.root_map,
                "allow_fixture": self.allow_fixture}, list(registered["layout"]["sources"].values()))
            self.artifacts.__enter__()
            self.owner = threading.get_ident()
            return self
        except BaseException:
            self.probe_host.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *error):
        self._owner()
        if self.artifacts is None:
            raise ValueError("repair scope missing at exit")
        try:
            self.artifacts.__exit__(*error)
        finally:
            self.owner = None
            self.probe_host.__exit__(*error)

    def _owner(self):
        E.require(self.owner == threading.get_ident(), "Host2 scope/thread ownership")

    def callbacks(self):
        return R.HostCallbacks(self.approve, self.dispatch, self.lookup)

    def _bind(self, operation, intent):
        self._owner()
        reg = self._sources()
        E.require(operation == J.digest({"manifest": self.runtime.manifest, "intent": intent}), "Runtime operation identity")
        rows = []
        for path in (self.runtime.folder / "steps").glob("*/intent.json"):
            if E.E.loads(path.read_text(encoding="utf-8")) == intent:
                rows.append(path.parent)
        E.require(len(rows) == 1, "one original Runtime intent required")
        step = rows[0]
        E.require(E.E.loads((step / "operation.json").read_text(encoding="utf-8")) == {
            "operation_id": operation, "intent_sha256": J.file_sha(step / "intent.json")}, "Runtime operation/intent raw binding")
        context = E.E.loads((step / "request.json").read_text(encoding="utf-8"))["context"]
        request = {"schema": E.SCHEMA + "_request", "operation_id": operation,
            "registration": self.registration_ref, "intent": deepcopy(intent), "context": context,
            "probe_seals": self.probe_seals, "probe_records": self.probe_records}
        E.verify_model(request, reg)
        return self.folder / operation, request, reg

    def approve(self, operation_id, intent) -> dict:
        self._owner()
        if intent["action"] == "run_probe":
            return self.probe_host.approve(operation_id, intent)
        folder, request, registered = self._bind(operation_id, intent)
        if (folder / "approval.json").exists():
            return self.lookup(operation_id, intent)
        claim = self.folder / "operation-claim.json"
        expected = {"operation_id": operation_id, "intent_digest": J.digest(intent), "registration": self.registration_ref}
        if claim.exists():
            E.require(E.E.loads(claim.read_text(encoding="utf-8")) == expected, "one repair operation per registration")
            return {"status": "hold", "reason": "approval_interrupted_requires_reconciliation"}
        E.write(claim, expected)
        E.write(folder / "intent.json", intent)
        # C owns approval and one-time consumption. Real probe verification is retained.
        seals = [{**s, "snapshot": registered["resolver"].read(s)["snapshot"]} for s in self.probe_seals]
        ledger = folder / "corrections.ledger.jsonl"
        auth = {"authorization": C.authorize_repair(ledger, "F1", intent["snapshot"], [], seals,
            self.approver, registered["contract"]["budget"]["gpu_s"]), "ledger_path": str(ledger.resolve())}
        self.runtime._authorization(intent, auth, done=False)
        auth_ref = E.write(folder / "approval.json", auth)
        prefix = folder / "approval-prefix.jsonl"
        with prefix.open("xb") as stream:
            stream.write(ledger.read_bytes())
        request.update(authorization=auth_ref, approval_ledger=E.ref(ledger), approval_prefix=E.ref(prefix))
        E.verify_authorization(request, auth_ref, registered)
        E.write(folder / "request.json", request)
        return {"status": "approved", "authorization": auth}

    def _prepared(self, operation, intent, authorization):
        folder, expected, reg = self._bind(operation, intent)
        request_ref = E.ref(folder / "request.json")
        request = reg["resolver"].read(request_ref)
        E.require({k: request[k] for k in expected} == expected
                  and reg["resolver"].read(request["authorization"]) == authorization, "stored repair request/approval changed")
        return folder, request_ref, request, reg

    def dispatch(self, operation_id, intent, authorization) -> dict:
        self._owner()
        if intent["action"] == "run_probe":
            return self.probe_host.dispatch(operation_id, intent, authorization)
        folder, request_ref, request, _ = self._prepared(operation_id, intent, authorization)
        if (folder / "response.json").exists() or (folder / "dispatch-intent.json").exists():
            return self.lookup(operation_id, intent)
        self.runtime._authorization(intent, authorization, done=False)
        admission_ref = self.admitter(operation_id, request_ref, request["authorization"], self.registration_ref)
        E.write(folder / "dispatch-intent.json", {"request": request_ref,
            "authorization": request["authorization"], "admission": admission_ref})
        result = self.provider.execute(operation_id, request_ref, request["authorization"], admission_ref)
        return self._settle(folder, intent, authorization, request_ref, result)

    def _settle(self, folder, intent, authorization, request_ref, result_ref) -> dict:
        reg = self._sources()
        raw = reg["resolver"].read(result_ref)
        marker = E.E.loads((folder / "dispatch-intent.json").read_text(encoding="utf-8"))
        E.require(raw.get("admission") == marker["admission"] and raw.get("authorization") == marker["authorization"],
                  "provider admission/authorization differs from dispatch")
        if raw.get("status") != "completed":
            return {"status": "hold", "reason": "repair_provider_incomplete", "raw": result_ref}
        checked = E.verify_repair_evidence(result_ref, request_ref=request_ref,
            registration_ref=self.registration_ref, root_map=self.root_map)
        output_ref = raw["training"]
        if self.artifacts is None:
            raise ValueError("repair artifact scope required")
        self.artifacts.register_output(result_ref, request_ref)
        policy = intent["snapshot"]["inputs"]["policy"]
        current = E.artifact(reg["layout"]["sources"]["case"], "case", reg, policy=policy)
        reference = E.artifact(reg["layout"]["sources"]["reference"], "reference", reg, policy="shared_reference")
        output = E.artifact(output_ref, "output", reg, policy=policy)
        seal = folder / "repair-seal.json"
        if not seal.exists():
            self.runtime._authorization(intent, authorization, done=False)
            C.seal_repair(seal, ledger_path=Path(authorization["ledger_path"]), authorization=authorization["authorization"],
                arm="F1", policy=policy, epoch=1, applied_before=[], input_artifact=current, output_artifact=output,
                output_report_path=reg["resolver"].file(checked["evaluation"]["CORRECTED"]),
                reference_artifact=reference, reference_report_path=reg["resolver"].file(checked["evaluation"]["NONE"]),
                bands=reg["contract"]["bands"], gpu_s=checked["gpu_s"])
        self.runtime._authorization(intent, authorization, done=True)
        verified = C.verify_repair(seal, ledger_path=Path(authorization["ledger_path"]), bands=reg["contract"]["bands"])
        sealed = E.E.loads(seal.read_text(encoding="utf-8"))
        E.require(sealed["output_artifact"] == output and sealed["input_artifact"] == current
                  and sealed["reference_artifact"] == reference and sealed["authorization"] == authorization["authorization"],
                  "repair seal belongs to another raw result/approval")
        after = None
        if C.next_epoch(seal, ledger_path=Path(authorization["ledger_path"]), bands=reg["contract"]["bands"]):
            old = intent["snapshot"]["inputs"]
            prior = {"verdict": verified["outcome"], "correction_id": E.CORRECTION,
                     "result_sha256": verified["result_sha256"], "checkpoint_sha256": output["checkpoint"]["sha256"]}
            bindings = deepcopy(old["packet"]["input_bindings"])
            bindings["case"] = {"checkpoint_sha256": prior["checkpoint_sha256"],
                                "telemetry_sha256": checked["output"]["bundle"]["telemetry"]["sha256"]}
            packet = C.S.build_packet(epoch=2, applied_corrections=[E.CORRECTION], prior_receipt=prior,
                facts={"config:action_scale": 0.25}, input_bindings=bindings,
                execution_budget_s=old["packet"]["execution_budget_s"])
            after = C.V.build_snapshot(packet, old["run_id"], policy, old["protocol_sha256"],
                folder / "epoch2/probes.ledger.jsonl", parent={"run_id": old["run_id"], "policy": policy, "epoch": 1, **prior})
        response = {"status": "completed", "authorization": authorization,
            "result": {"seal": E.ref(seal), "ledger_path": authorization["ledger_path"],
                       "bands": reg["contract"]["bands"], "next_snapshot": after, "evidence": None}}
        for name, value in (("outcome-ref.json", result_ref), ("response.json", response)):
            path = folder / name
            if path.exists():
                E.require(E.E.loads(path.read_text(encoding="utf-8")) == value, "host " + name + " tampered")
            else:
                E.write(path, value)
        return response

    def lookup(self, operation_id, intent) -> dict:
        self._owner()
        if intent["action"] == "run_probe":
            return self.probe_host.lookup(operation_id, intent)
        folder, _, _ = self._bind(operation_id, intent)
        if not (folder / "approval.json").exists():
            return {"status": "hold", "reason": "approval_interrupted_requires_reconciliation"} if folder.exists() else {"status": "not_found"}
        authorization = E.E.loads((folder / "approval.json").read_text(encoding="utf-8"))
        folder, request_ref, _, _ = self._prepared(operation_id, intent, authorization)
        if (folder / "outcome-ref.json").exists():
            return self._settle(folder, intent, authorization, request_ref,
                                E.E.loads((folder / "outcome-ref.json").read_text(encoding="utf-8")))
        self.runtime._authorization(intent, authorization, done=(folder / "repair-seal.json").exists())
        if not (folder / "dispatch-intent.json").exists():
            return {"status": "approved", "authorization": authorization}
        result = self.provider.lookup(operation_id, request_ref)
        if result is None:
            return {"status": "pending", "authorization": authorization}
        return self._settle(folder, intent, authorization, request_ref, result)
