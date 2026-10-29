"""fresh selection_source를 Runtime actual_result와 대조하는 probe Host 판."""
from __future__ import annotations

from pathlib import Path

import agentic_brev_runtime_host_v1 as V1
import agentic_investigation_runtime_v2 as R2
import run_agentic_brev_probe_parent_v5 as P2

SCHEMA = "agentic_brev_runtime_host_v3"
SOURCES = V1.SOURCES | P2.SOURCES | {
    "evals/agentic_brev_runtime_host_v3.py",
    "tests/test_agentic_brev_runtime_host_v3.py",
}


class BrevRuntimeHost(V1.BrevRuntimeHost):
    def _binding(self, operation, intent, registration_ref=None):
        self._owner()
        self._sources()
        V1.require(operation == V1.J.digest({"manifest": self.runtime.manifest, "intent": intent}),
                   "Runtime operation/intent changed")
        V1.require(intent["action"] == "run_probe", "probe host action required")
        folder = self.root / operation
        P2.V1._directory(folder)
        saved_path = folder / "intent.json"
        if registration_ref is None:
            registration_ref = (V1.J._read(saved_path)["registration"] if saved_path.exists()
                                else self.bindings[operation])
        registration = R2.read_ref(registration_ref)
        V1.require(registration.get("schema") == SCHEMA + "_registration",
                   "host parent registration schema")
        protocol = R2.read_ref(registration["parent_protocol"])
        binding = R2.read_ref(registration["parent_binding"])
        request = R2.read_ref(registration["parent_request"])
        V1.require(protocol.get("schema") == P2.SCHEMA + "_protocol"
                   and protocol.get("registered_before_execution") is True
                   and SOURCES <= protocol["sources_lf"].keys()
                   and protocol["binding_sha256"] == registration["parent_binding"]["sha256"]
                   and request.get("schema") == P2.SCHEMA + "_request"
                   and request["action"] == "probe"
                   and request["protocol"]["sha256"] == registration["parent_protocol"]["sha256"]
                   and request["binding"]["sha256"] == registration["parent_binding"]["sha256"]
                   and request["source_anchor"] == registration["parent_anchor"],
                   "parent raw registration binding")
        P2.V1._public(protocol)
        R2.verify_sources(protocol, registration["parent_anchor"], repo=self.runtime.repo,
                          protocol_path=Path(registration["parent_protocol"]["path"]))
        V1._ancestor(self.runtime.repo, self.runtime.anchor, registration["parent_anchor"])
        pair, execution = binding["expected_request"], binding["expected_request"]["execution"]
        host = execution["runtime_host_binding"]
        V1.require(host.get("schema") == SCHEMA + "_runtime_binding"
                   and host["operation_id"] == operation
                   and host["intent_sha256"] == V1.J.digest(intent)
                   and host["runtime_manifest_sha256"] == V1.J.digest(self.runtime.manifest)
                   and host["runtime_source_anchor"] == self.runtime.anchor
                   and host["runtime_protocol_sha256"] == self.runtime.protocol_ref["sha256"],
                   "registered Runtime R binding")
        kind = V1.E.FIXTURE if self.allow_fixture else V1.E.ACTUAL
        V1.require(binding["evidence_kind"] == execution["evidence_kind"] == kind
                   and pair["arms"] == list(P2.V1.ORDER) and pair["seed"] == 2221
                   and execution["order"] == list(P2.V1.ORDER)
                   and execution["gpu_binding"] == P2.V1.GPU_BINDING
                   and pair["probe"] == P2.V1.registered_probe(protocol, pair)
                   == intent["target"] == intent["request_spec"]["probe"]
                   and pair["commands_sha256"] == P2.V1.PAIR.digest(pair["commands"])
                   and execution["source_closure_sha256"] == V1.E.digest(protocol["sources_lf"]),
                   "pair identity/source/kind")
        resolver = V1.E.Resolver(host["host_input_roots"], V1.E.digest(host["host_input_roots"]))
        snapshot, spec, _transport = P2.fresh_selection(execution, resolver)
        V1.require(R2._equivalent(snapshot, intent["snapshot"]) and spec == intent["request_spec"],
                   "parent model/input snapshot mismatch")
        steps = [p.parent for p in (self.runtime.folder / "steps").glob("*/operation.json")
                 if V1.J._read(p)["operation_id"] == operation]
        V1.require(len(steps) == 1, "Runtime immutable operation record missing")
        step = steps[0]
        ordinal = int(step.name)
        V1.require(ordinal <= len(self.runtime.reg["prefix"]), "fresh prefix ordinal")
        runtime_source = self.runtime.reg["prefix"][ordinal - 1]["actual_result"]
        selection_source = execution["selection_source"]
        V1.require(selection_source["actual_result"]["sha256"] == runtime_source["sha256"]
                   and R2.read_ref(runtime_source) == resolver.read(selection_source["actual_result"]),
                   "different Runtime fresh actual_result")
        V1.require(2 * execution["windows"]["max_wall_s"] <= pair["budget_gpu_s"]
                   <= spec["budget_gpu_s"], "pair/C budget mismatch")
        metadata = V1.MetadataMap(host["metadata_roots"], self.allow_fixture)
        for arm, side in (("NONE", "reference"), ("F1", "case")):
            op = execution["operations"][arm]
            source = intent["snapshot"]["inputs"]["packet"]["input_bindings"][side]
            V1.require(op["arm"] == arm and op["side"] == side and op["seed"] == 2221
                       and op["probe"] == pair["probe"] and op["args"] == spec["args"]
                       and pair["commands"][arm]["probe"] == op["probe"]
                       and pair["commands"][arm]["args"] == spec["args"]
                       and pair["checkpoints"][arm] == op["source"]["checkpoint"]["sha256"]
                       == source["checkpoint_sha256"]
                       and op["source"]["telemetry"]["sha256"] == source["telemetry_sha256"],
                       "pair CP/telemetry/probe/args/seed changed")
            if not self.allow_fixture:
                V1.require(all(op["source"][key]["sha256"]
                               == P2.V1.U.INITIAL[side][key + "_sha256"]
                               for key in ("checkpoint", "telemetry")),
                           "actual initial source mismatch")
            V1.require(Path(op["host_stage"]).is_absolute()
                       and pair["commands"][arm]["stage"] == op["host_stage"]
                       and pair["commands"][arm]["operation_id"] == op["operation_id"]
                       and pair["commands"][arm]["remote_stage"] == op["remote_stage"]
                       and pair["commands"][arm]["run_name"] == op["run_name"],
                       "registered host stage/operation/command")
            argv = pair["commands"][arm]["argv"]
            V1.require(isinstance(argv, list) and len(argv) == 7
                       and argv[1] == "-B" and argv[3] == "--job" and argv[5] == "--output"
                       and V1._path(argv[2]).name == "brev_recovery_worker_v1.py"
                       and V1._path(argv[4]).name == "job.json"
                       and V1._path(argv[6]).name == "worker.json"
                       and V1._path(argv[4]).parent == V1._path(argv[6]).parent
                       and (self.allow_fixture
                            or str(V1._path(argv[4]).parent) == op["remote_stage"]),
                       "registered worker argv/stage")
        state = {"intent": intent, "registration": registration_ref}
        if saved_path.exists():
            V1.require(V1.J._read(saved_path) == state,
                       "immutable host intent/parent binding changed")
        return {"folder": folder, "registration_ref": registration_ref,
                "registration": registration, "protocol": protocol, "binding": binding,
                "request": request, "execution": execution, "host": host,
                "resolver": resolver, "metadata": metadata, "intent": intent,
                "operation": operation, "step": step,
                "authority": P2.verify_authority({"request": request, "binding": binding,
                                                  "execution": execution, "resolver": resolver})}
