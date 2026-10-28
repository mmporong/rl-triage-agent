"""진짜 NONE 원문 뒤 C case 승인을 발급하고 등록 transport의 provider 해제를 잇는다."""
from __future__ import annotations

import inspect
import sys
import threading
from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import agentic_brev_controller_scope_v1 as S
import agentic_investigation_runtime_v1 as R
import run_agentic_brev_probe_parent_v1 as P
from vision_transferred_e2e_v2 import ObjectEvidence

C, E, J = S.C, S.E, R.J
SCHEMA = "agentic_brev_runtime_host_v1"
SOURCES = P.SOURCES | set(C.SOURCE_FILES) | {
    "evals/agentic_brev_runtime_host_v1.py", "tests/test_agentic_brev_runtime_host_v1.py",
    "evals/agentic_investigation_runtime_v1.py", "tests/test_agentic_investigation_runtime_v1.py",
    "evals/agentic_investigation_loop_v1.py",
}
METHODS = ("prepare_pair", "start_reference", "lookup_reference", "dispatch_case", "lookup_pair")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _path(value):
    require(isinstance(value, str) and bool(value), "registered producer path")
    path = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts, "absolute producer path required")
    return path


class MetadataMap:
    """원본 producer의 POSIX/fixture Windows ref만 등록 native 사본으로 해석한다."""
    def __init__(self, rows, fixture):
        self.rows = []
        for row in rows:
            require(set(row) == {"remote", "local"}, "metadata root fields")
            remote, local = _path(row["remote"]), Path(row["local"])
            require(fixture or isinstance(remote, PurePosixPath), "actual producer must preserve POSIX refs")
            require(local.is_absolute() and local == local.resolve(strict=True) and local.is_dir()
                    and not any(p.is_symlink() for p in (local, *local.parents)), "metadata native root")
            require(all(type(remote) is not type(old) or (remote != old and remote not in old.parents and old not in remote.parents)
                        for old, _ in self.rows), "overlapping metadata root")
            self.rows.append((remote, local))
        require(bool(self.rows), "registered metadata roots required")

    def file(self, ref):
        require(set(ref) == {"path", "sha256"}, "producer reference fields")
        source = _path(ref["path"])
        matches = [(remote, local) for remote, local in self.rows
                   if type(remote) is type(source) and (remote == source or remote in source.parents)]
        require(len(matches) == 1, "producer ref outside registered metadata map")
        remote, local = matches[0]
        path = local.joinpath(*source.relative_to(remote).parts)
        require(P.raw_ref(path)["sha256"] == ref["sha256"], "producer raw SHA changed")
        return path

    def read(self, ref):
        return E.loads(self.file(ref).read_text(encoding="utf-8"))


def _ancestor(repo, older, newer):
    store, todo, seen = ObjectEvidence(repo), [newer], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == older:
            return
        todo.extend(store.commit(current)["parents"])
    raise ValueError("parent P must descend from fixed Runtime R")


class BrevRuntimeHost(AbstractContextManager):
    def __init__(self, runtime, bindings, transport, approver, *, allow_fixture=False):
        require(isinstance(runtime, R.Runtime) and type(allow_fixture) is bool
                and allow_fixture == runtime.reg["allow_fixture"], "host fixture/runtime mode mismatch")
        require(allow_fixture or runtime.repo.resolve() == ROOT, "actual host must execute registered checkout")
        require(isinstance(approver, str) and bool(approver.strip()), "explicit host approver required")
        self.runtime, self.bindings, self.transport = runtime, deepcopy(bindings), transport
        self.approver, self.allow_fixture = approver.strip(), allow_fixture
        self.root = runtime.folder / "brev-host"
        P._directory(self.root)
        self.registration = {"protocol": runtime.protocol_ref, "anchor": runtime.anchor, "repo": str(runtime.repo)}
        self.scope = None
        self.owner = None
        self.registered = {}
        self._sources()

    def _sources(self):
        protocol = R.read_ref(self.runtime.protocol_ref)
        require(protocol == self.runtime.protocol and SOURCES <= protocol["sources_lf"].keys(), "host/runtime source closure")
        S.validate_source_registration(self.registration)
        kind = E.FIXTURE if self.allow_fixture else E.ACTUAL
        require(getattr(type(self.transport), "EVIDENCE_KIND", None) == kind, "fixture transport cannot become actual")
        source = inspect.getsourcefile(type(self.transport))
        if source is None:
            raise ValueError("registered transport class source missing")
        path = Path(source).resolve(strict=True)
        relative = path.relative_to(self.runtime.repo.resolve()).as_posix() if not self.allow_fixture else None
        sha = C.digest_lf_file(path)
        require(relative is None or protocol["sources_lf"].get(relative) == sha, "unregistered actual transport source")
        if self.allow_fixture:
            require(sha in protocol["sources_lf"].values(), "fixture transport source closure")
        contract_ref = getattr(self.transport, "contract_ref", None)
        if not isinstance(contract_ref, dict):
            raise TypeError("registered private IO contract required")
        require(contract_ref.get("sha256") == protocol.get("host_transport_contract_sha256"), "registered private IO contract SHA")
        contract = R.read_ref(contract_ref)
        require(contract == getattr(self.transport, "contract", None)
                and contract.get("service_source_sha256_lf") == protocol.get("host_transport_service_sha256_lf")
                and contract["service_source_sha256_lf"] in protocol["sources_lf"].values(), "registered IO service source required")
        for name in METHODS:
            method = getattr(self.transport, name, None)
            require(inspect.ismethod(method) and method.__self__ is self.transport
                    and method.__func__ is getattr(type(self.transport), name)
                    and inspect.getsourcefile(method.__func__) == source, "transport must use registered bound methods")

    def __enter__(self):
        require(self.owner is None, "nested host context")
        self._sources()
        self.scope = S.BrevProbeScope(allow_fixture=self.allow_fixture, source_registration=self.registration)
        self.scope.__enter__()
        self.owner, self.registered = threading.get_ident(), {}
        try:
            for path in sorted(self.root.glob("*/intent.json")):
                saved = R.read_ref(R.ref(path))
                binding = self._binding(path.parent.name, saved["intent"], saved["registration"])
                for arm in P.ORDER:
                    if (path.parent / f"{arm}.projection.json").exists():
                        self._evidence(binding, arm)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, kind, value, traceback):
        require(self.owner == threading.get_ident() and self.scope is not None, "host context owner")
        assert self.scope is not None
        try:
            self.scope.__exit__(kind, value, traceback)
        finally:
            self.owner, self.scope, self.registered = None, None, {}

    def callbacks(self):
        return R.HostCallbacks(self.approve, self.dispatch, self.lookup)

    def advance(self, **kwargs):
        require("host" not in kwargs, "host callback override forbidden")
        with self:
            return self.runtime.advance(host=self.callbacks(), **kwargs)

    def _owner(self):
        require(self.owner == threading.get_ident() and self.scope is not None, "use host context around Runtime.advance")

    def _binding(self, operation, intent, registration_ref=None):
        self._owner()
        self._sources()
        require(operation == J.digest({"manifest": self.runtime.manifest, "intent": intent}), "Runtime operation/intent changed")
        require(intent["action"] == "run_probe", "probe host action required")
        folder = self.root / operation
        P._directory(folder)
        saved_path = folder / "intent.json"
        if registration_ref is None:
            registration_ref = (J._read(saved_path)["registration"] if saved_path.exists() else self.bindings[operation])
        registration = R.read_ref(registration_ref)
        require(registration.get("schema") == SCHEMA + "_registration", "host parent registration schema")
        protocol = R.read_ref(registration["parent_protocol"])
        binding = R.read_ref(registration["parent_binding"])
        request = R.read_ref(registration["parent_request"])
        require(protocol.get("schema") == P.SCHEMA + "_protocol" and protocol.get("registered_before_execution") is True
                and SOURCES <= protocol["sources_lf"].keys()
                and protocol["binding_sha256"] == registration["parent_binding"]["sha256"]
                and request.get("schema") == P.SCHEMA + "_request" and request["action"] == "probe"
                and request["protocol"]["sha256"] == registration["parent_protocol"]["sha256"]
                and request["binding"]["sha256"] == registration["parent_binding"]["sha256"]
                and request["source_anchor"] == registration["parent_anchor"], "parent raw registration binding")
        P._public(protocol)
        R.verify_sources(protocol, registration["parent_anchor"], repo=self.runtime.repo,
                         protocol_path=Path(registration["parent_protocol"]["path"]))
        _ancestor(self.runtime.repo, self.runtime.anchor, registration["parent_anchor"])
        pair, execution = binding["expected_request"], binding["expected_request"]["execution"]
        host = execution["runtime_host_binding"]
        require(host.get("schema") == SCHEMA + "_runtime_binding" and host["operation_id"] == operation
                and host["intent_sha256"] == J.digest(intent)
                and host["runtime_manifest_sha256"] == J.digest(self.runtime.manifest)
                and host["runtime_source_anchor"] == self.runtime.anchor
                and host["runtime_protocol_sha256"] == self.runtime.protocol_ref["sha256"], "registered Runtime R binding")
        kind = E.FIXTURE if self.allow_fixture else E.ACTUAL
        require(binding["evidence_kind"] == execution["evidence_kind"] == kind
                and pair["arms"] == list(P.ORDER) and pair["seed"] == 2221
                and execution["order"] == list(P.ORDER) and execution["gpu_binding"] == P.GPU_BINDING
                and pair["probe"] == P.registered_probe(protocol, pair) == intent["target"] == intent["request_spec"]["probe"]
                and pair["commands_sha256"] == P.PAIR.digest(pair["commands"])
                and execution["source_closure_sha256"] == E.digest(protocol["sources_lf"]), "pair identity/source/kind")
        resolver = E.Resolver(host["host_input_roots"], E.digest(host["host_input_roots"]))
        snapshot, spec, _transport = P._selection(execution, resolver)
        require(R._equivalent(snapshot, intent["snapshot"]) and spec == intent["request_spec"], "parent model/input snapshot mismatch")
        steps = [p.parent for p in (self.runtime.folder / "steps").glob("*/operation.json")
                 if J._read(p)["operation_id"] == operation]
        require(len(steps) == 1, "Runtime immutable operation record missing")
        step = steps[0]
        ordinal = int(step.name)
        capsule_ref = (self.runtime.reg["prefix"][ordinal - 1]["capsule"] if ordinal <= len(self.runtime.reg["prefix"])
                       else J._read(step / "capsule-reference.json"))
        require(execution["selection_capsule"]["sha256"] == capsule_ref["sha256"]
                and R.read_ref(capsule_ref) == resolver.read(execution["selection_capsule"]), "different Runtime model capsule")
        require(2 * execution["windows"]["max_wall_s"] <= pair["budget_gpu_s"] <= spec["budget_gpu_s"], "pair/C budget mismatch")
        metadata = MetadataMap(host["metadata_roots"], self.allow_fixture)
        for arm, side in (("NONE", "reference"), ("F1", "case")):
            op = execution["operations"][arm]
            source = intent["snapshot"]["inputs"]["packet"]["input_bindings"][side]
            require(op["arm"] == arm and op["side"] == side and op["seed"] == 2221 and op["probe"] == pair["probe"]
                    and op["args"] == spec["args"] and pair["commands"][arm]["probe"] == op["probe"]
                    and pair["commands"][arm]["args"] == spec["args"]
                    and pair["checkpoints"][arm] == op["source"]["checkpoint"]["sha256"] == source["checkpoint_sha256"]
                    and op["source"]["telemetry"]["sha256"] == source["telemetry_sha256"], "pair CP/telemetry/probe/args/seed changed")
            if not self.allow_fixture:
                require(all(op["source"][key]["sha256"] == P.U.INITIAL[side][key + "_sha256"] for key in ("checkpoint", "telemetry")),
                        "actual initial source mismatch")
            require(Path(op["host_stage"]).is_absolute() and pair["commands"][arm]["stage"] == op["host_stage"]
                    and pair["commands"][arm]["operation_id"] == op["operation_id"]
                    and pair["commands"][arm]["remote_stage"] == op["remote_stage"]
                    and pair["commands"][arm]["run_name"] == op["run_name"], "registered host stage/operation/command")
            argv = pair["commands"][arm]["argv"]
            require(isinstance(argv, list) and len(argv) == 7 and argv[1] == "-B" and argv[3] == "--job"
                    and argv[5] == "--output" and _path(argv[2]).name == "brev_recovery_worker_v1.py"
                    and _path(argv[4]).name == "job.json" and _path(argv[6]).name == "worker.json"
                    and _path(argv[4]).parent == _path(argv[6]).parent
                    and (self.allow_fixture or str(_path(argv[4]).parent) == op["remote_stage"]), "registered worker argv/stage")
        state = {"intent": intent, "registration": registration_ref}
        if saved_path.exists():
            require(J._read(saved_path) == state, "immutable host intent/parent binding changed")
        return {"folder": folder, "registration_ref": registration_ref, "registration": registration,
            "protocol": protocol, "binding": binding, "request": request, "execution": execution, "host": host,
            "resolver": resolver, "metadata": metadata, "intent": intent, "operation": operation,
            "step": step, "authority": P.verify_authority({"request": request, "binding": binding, "execution": execution, "resolver": resolver})}

    def _evidence(self, bound, arm, result_ref=None):
        folder, op = bound["folder"], bound["execution"]["operations"][arm]
        record = folder / f"{arm}.projection.json"
        if result_ref is None:
            result_ref = J._read(record)["producer_outcome"]
        outcome = R.read_ref(result_ref)
        stage = Path(op["host_stage"])
        require(Path(result_ref["path"]) == stage.parent / "outcome.json"
                and outcome.get("status") == "complete" and outcome["operation_id"] == op["operation_id"], "producer completion location/operation")
        metadata = bound["metadata"]
        plan, original = metadata.read(outcome["plan"]), metadata.read(outcome["evidence"])
        require(metadata.file(outcome["evidence"]) == stage.parent / "evidence.json", "raw evidence archive location")
        identity = {key: op[key] for key in ("side", "arm", "seed", "probe", "run_name")}
        require(plan.get("schema") == P.SCHEMA + "_plan" and plan["identity"] == identity
                and plan["operation_id"] == op["operation_id"] and plan["logical_stage"] == op["remote_stage"]
                and plan["source"] == op["source"] and plan["expected_source"] == bound["execution"]["backend"]["expected_source"]
                and plan["evidence_kind"] == original["evidence_kind"] == bound["binding"]["evidence_kind"], "parent plan/receipt differs from registered source")
        require(plan["parent_request"]["sha256"] == bound["registration"]["parent_request"]["sha256"]
                and plan["authority"]["request_sha256"] == bound["authority"]["request_sha256"]
                and plan["authority"]["host_reference"] == bound["authority"]["host_reference"], "parent plan authority changed")
        proof = metadata.read(plan["proof"])
        require(proof["operation_id"] == op["operation_id"] and proof["pair_id"] == bound["execution"]["pair_id"]
                and proof["protocol_sha256"] == bound["registration"]["parent_protocol"]["sha256"]
                and proof["authority_sha256"] == bound["authority"]["local_copy"]["sha256"]
                and proof["authority_request_sha256"] == bound["authority"]["request_sha256"]
                and proof["evidence_kind"] == plan["evidence_kind"] and proof["gpu_binding"] == P.GPU_BINDING,
                "parent proof authority/protocol/kind")
        for name, key in (("lease_snapshot", "lease_snapshot_sha256"), ("capture_parent", "capture_parent_sha256"),
                          ("boot", "boot_sha256"), ("model_observation", "gpu_model_observation_sha256"),
                          ("admission_times", "admission_times_sha256")):
            require(metadata.file(plan[name]) is not None and plan[name]["sha256"] == proof[key], "parent raw metadata changed")
        require(plan["lease_basis"] == bound["execution"]["lease_basis"]
                and proof["lease_basis_sha256"] == plan["lease_basis"]["sha256"]
                and original["root_map_sha256"] == outcome["roots_sha256"] == E.digest(plan["roots"]), "producer root-map/lease binding")
        roots = bound["host"]["evidence_roots"][arm]
        resolver = E.Resolver(roots, E.digest(roots))
        require(resolver.file(plan["resource"]["proof"]) == metadata.file(plan["proof"]), "resource proof projection differs")
        request = P.U._request({"plan": plan, "stage": stage,
            "protocol": {"max_wall_s": bound["execution"]["windows"]["max_wall_s"]}})
        require(request == original["request"] and E.digest(request) == original["request_sha256"] == outcome["request_sha256"],
                "original E13 request/raw inventory changed")
        derived = E.build(request, roots, expected_request_sha256=E.digest(request), expected_roots_sha256=E.digest(roots), allow_fixture=self.allow_fixture)
        expected_original = {**derived, "root_map_sha256": original["root_map_sha256"]}
        require(original == expected_original, "host projection changed E13 fields beyond root-map digest")
        E.verify(derived, roots, expected_request_sha256=E.digest(request), expected_roots_sha256=E.digest(roots), allow_fixture=self.allow_fixture)
        derived_path = folder / f"{arm}.evidence.json"
        R.same_or_new(derived_path, derived)
        ref = R.ref(derived_path)
        projection = {"schema": SCHEMA + "_projection", "producer_outcome": result_ref,
            "original_evidence": P.raw_ref(metadata.file(outcome["evidence"])), "original_evidence_sha256": outcome["evidence"]["sha256"],
            "original_root_map_sha256": original["root_map_sha256"], "host_evidence": ref, "host_roots_sha256": E.digest(roots)}
        R.same_or_new(record, projection)
        expected = {key: op[key] for key in S.EXPECTED_KEYS - {"checkpoint_sha256", "stage"}}
        expected.update(checkpoint_sha256=op["source"]["checkpoint"]["sha256"], stage=op["remote_stage"])
        entry = {"expected": expected, "roots": roots, "request_sha256": E.digest(request), "roots_sha256": E.digest(roots)}
        key = E.digest(derived)
        if key in self.registered:
            require(self.registered[key] == entry, "duplicate evidence intent differs")
        else:
            assert self.scope is not None
            self.scope.register(ref, **entry)
            self.registered[key] = entry
        return {"ref": ref, "evidence": derived, "proof": proof, "plan": plan, "stage": stage}

    def _response(self, value, operation):
        require(isinstance(value, dict) and value.get("operation_id") == operation
                and value.get("status") in {"prepared", "completed", "pending", "unknown", "not_found", "hold"}, "transport response schema/operation")
        if value["status"] != "completed":
            require("result" not in value, "future output must not masquerade as completed ref")
        return value

    def approve(self, operation, intent):
        self._owner()
        if intent.get("action") != "run_probe":
            return {"status": "hold", "reason": "repair_unsupported"}
        bound = self._binding(operation, intent)
        folder = bound["folder"]
        R.same_or_new(folder / "intent.json", {"intent": intent, "registration": bound["registration_ref"]})
        marker = folder / "approve-intent.json"
        if marker.exists():
            return self.lookup(operation, intent)
        ledger = C.S.M.RegistryLedger(Path(intent["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        try:
            J._write_new(marker, {"operation_id": operation, "ledger_digest": C.L.digest(ledger.events())})
        except FileExistsError:
            return self.lookup(operation, intent)
        ready = self._response(self.transport.prepare_pair(operation, deepcopy(intent), bound["registration_ref"]), operation)
        require(ready["status"] == "prepared", "transport pair not prepared")
        result = self._response(self.transport.start_reference(operation, deepcopy(intent), bound["registration_ref"]), operation)
        if result["status"] != "completed":
            return {"status": "unknown", "reason": "reference_pending_requires_explicit_settle"}
        reference = self._evidence(bound, "NONE", result["result"])
        return self._authorize(bound, reference)

    def _authorize(self, bound, reference):
        intent, folder = bound["intent"], bound["folder"]
        inputs = intent["snapshot"]["inputs"]
        ledger_path = Path(inputs["ledger_path"])
        epoch = inputs["packet"]["epoch"]
        require(ledger_path.parent.name == f"epoch{epoch}", "C ledger epoch path")
        arm, root = ledger_path.parent.parent.name, ledger_path.parent.parent.parent
        ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
        require(C.L.digest(ledger.events()) == J._read(folder / "approve-intent.json")["ledger_digest"],
                "C ledger changed while reference was running; explicit reconciliation required")
        future = ledger_path.parent / f"authorization_{len(ledger.requests()) + 1:02d}.json"
        require(not (folder / "case-approval-intent.json").exists(), "C approval already attempted; lookup only")
        J._write_new(folder / "case-approval-intent.json", {"future_authorization_path": str(future),
            "ledger_digest": C.L.digest(ledger.events()), "reference": reference["ref"]})
        op = bound["execution"]["operations"]["F1"]
        require(not (Path(op["host_stage"]) / "job.json").exists()
                and not (Path(op["host_stage"]).parent / "outcome.json").exists(), "case already executed before C approval")
        plan = {"stage": op["host_stage"], "run_name": op["run_name"], "seed": op["seed"],
            "checkpoint_sha256": op["source"]["checkpoint"]["sha256"], "args": op["args"], "gpu_contract": E.GPU_CONTRACT,
            "reference": {"evidence_path": reference["ref"]["path"], "evidence_sha256": reference["ref"]["sha256"],
                "checkpoint_sha256": reference["evidence"]["checkpoint_sha256"], "stage": str(reference["stage"]),
                "run_name": reference["evidence"]["run_name"], "seed": reference["evidence"]["seed"], "gpu_contract": E.GPU_CONTRACT}}
        require(C.L.digest(ledger.events()) == J._read(folder / "case-approval-intent.json")["ledger_digest"], "C ledger changed before approval")
        authorization = C.authorize_probe(root, arm, epoch, intent["snapshot"], intent["choice"], self.approver,
            evidence_kind=C.FIXTURE_EVIDENCE if self.allow_fixture else C.ACTUAL_EVIDENCE, execution_plan=plan)
        require(authorization["path"] == str(future.resolve()), "C approval path changed during approval")
        R.same_or_new(folder / "authorization.json", authorization)
        return {"status": "approved", "authorization": authorization}

    def _authorization(self, bound):
        folder, intent = bound["folder"], bound["intent"]
        if (folder / "authorization.json").exists():
            auth = J._read(folder / "authorization.json")
        elif (folder / "case-approval-intent.json").exists():
            future = Path(J._read(folder / "case-approval-intent.json")["future_authorization_path"])
            if not future.exists():
                return None
            raw = R.read_ref(R.ref(future))
            auth = {**R.ref(future), "request_id": raw["request_id"], "snapshot": deepcopy(intent["snapshot"])}
        else:
            return None
        value = R.read_ref({key: auth[key] for key in ("path", "sha256")})
        ledger = C.S.M.RegistryLedger(Path(auth["path"]).parent / "probes.ledger.jsonl", C.RC.registry())
        done = ledger.requests().get(value["request_id"], {}).get("state") == "done"
        checked = C._authorization(auth, intent["snapshot"], "done" if done else "running", require_digest=True)
        require(checked["choice"] == intent["choice"] and checked["spec"] == intent["request_spec"], "C approval/intent changed")
        plan, op = checked["execution_plan"], bound["execution"]["operations"]["F1"]
        reference = self._evidence(bound, "NONE")
        require(plan["stage"] == op["host_stage"] and plan["run_name"] == op["run_name"] and plan["seed"] == op["seed"]
                and plan["checkpoint_sha256"] == op["source"]["checkpoint"]["sha256"]
                and plan["reference"]["evidence_path"] == reference["ref"]["path"]
                and plan["reference"]["evidence_sha256"] == reference["ref"]["sha256"], "C approval source/reference projection changed")
        R.same_or_new(folder / "authorization.json", auth)
        return auth, done

    def dispatch(self, operation, intent, authorization):
        self._owner()
        if intent.get("action") != "run_probe":
            return {"status": "hold", "reason": "repair_unsupported"}
        bound = self._binding(operation, intent)
        existing = self._authorization(bound)
        require(existing is not None and existing[0] == authorization and not existing[1], "case dispatch requires running immutable C approval")
        C.validate_probe_authorization(authorization, intent["snapshot"])
        marker = bound["folder"] / "case-dispatch-intent.json"
        if marker.exists():
            return self.lookup(operation, intent)
        try:
            J._write_new(marker, {"operation_id": operation, "authorization": authorization})
        except FileExistsError:
            return self.lookup(operation, intent)
        result = self._response(self.transport.dispatch_case(operation, deepcopy(intent), bound["registration_ref"], deepcopy(authorization)), operation)
        return self._finish(bound, result, authorization)

    def _finish(self, bound, response, authorization):
        if response["status"] != "completed":
            return {"status": "pending" if response["status"] == "pending" else "unknown"}
        reference, case = self._evidence(bound, "NONE"), self._evidence(bound, "F1", response["result"])
        require(all(reference["proof"][key] == case["proof"][key] for key in ("gpu_uuid", "boot_sha256", "lease_snapshot_sha256")),
                "pair crossed UUID/boot/physical lease")
        if not self.allow_fixture:
            executions = []
            for arm, item in (("NONE", reference), ("F1", case)):
                roots = bound["host"]["evidence_roots"][arm]
                executions.append(E.Resolver(roots, E.digest(roots)).read(item["evidence"]["request"]["files"]["execution"]))
            events = C.S.M.RegistryLedger(Path(authorization["path"]).parent / "probes.ledger.jsonl", C.RC.registry()).events()
            consumed = next(row["at"] for row in events if row.get("request_id") == authorization["request_id"] and row["event"] == "consumed")
            require(E._stamp(executions[0]["finished_at"]) <= E._stamp(consumed) <= E._stamp(executions[1]["started_at"]),
                    "actual case ran before C consumption or reference completion")
        seal_path = C._probe_seal_path(Path(authorization["path"]))
        if seal_path.exists():
            C.verify_probe(seal_path, bound["intent"]["snapshot"])
        else:
            state = C.S.M.RegistryLedger(Path(authorization["path"]).parent / "probes.ledger.jsonl", C.RC.registry()).requests()
            if state[authorization["request_id"]]["state"] != "running":
                return {"status": "unknown", "reason": "C_done_without_seal_requires_explicit_settle"}
            gpu_s = reference["evidence"]["wall_s"] + case["evidence"]["wall_s"]
            require(gpu_s <= bound["binding"]["expected_request"]["budget_gpu_s"], "completed pair over approved budget")
            C.seal_probe(authorization, bound["intent"]["snapshot"], Path(case["ref"]["path"]), Path(reference["ref"]["path"]), gpu_s, 0)
        seal_ref = R.ref(seal_path)
        C.verify_probe(seal_path, bound["intent"]["snapshot"])
        result_path = bound["folder"] / "completed.json"
        if result_path.exists():
            result = J._read(result_path)
            require(result["authorization"] == authorization and result["result"]["seal"] == seal_ref, "completed host result changed")
            return result
        after = C.V.build_snapshot(**bound["intent"]["snapshot"]["inputs"])
        result = {"status": "completed", "authorization": authorization, "result": {"seal": seal_ref, "next_snapshot": after}}
        R.same_or_new(result_path, result)
        return result

    def lookup(self, operation, intent):
        self._owner()
        if intent.get("action") != "run_probe":
            return {"status": "hold", "reason": "repair_unsupported"}
        bound = self._binding(operation, intent)
        if not (bound["folder"] / "intent.json").exists():
            return {"status": "unknown", "reason": "host_intent_missing"}
        authorization = self._authorization(bound)
        if authorization is None:
            self._response(self.transport.lookup_reference(operation, deepcopy(intent), bound["registration_ref"]), operation)
            return {"status": "unknown", "reason": "approval_incomplete_requires_explicit_settle"}
        auth, done = authorization
        if not (bound["folder"] / "case-dispatch-intent.json").exists():
            require(not done, "done C approval without host case intent")
            if (bound["step"] / "dispatch-intent.json").exists():
                return {"status": "unknown", "reason": "runtime_dispatch_without_host_intent_requires_settle"}
            return {"status": "approved", "authorization": auth}
        response = self._response(self.transport.lookup_pair(operation, deepcopy(intent), bound["registration_ref"]), operation)
        return self._finish(bound, response, auth)

    def settle_approval(self, operation, intent):
        """명시 호출만 허용한다. C가 부분 작성된 원장은 새 승인으로 덮지 않는다."""
        self._owner()
        if intent.get("action") != "run_probe":
            return {"status": "hold", "reason": "repair_unsupported"}
        bound = self._binding(operation, intent)
        require((bound["folder"] / "approve-intent.json").exists(), "settle requires existing approve intent")
        if (bound["folder"] / "case-approval-intent.json").exists():
            return self.lookup(operation, intent)
        response = self._response(self.transport.lookup_reference(operation, deepcopy(intent), bound["registration_ref"]), operation)
        if response["status"] != "completed":
            return {"status": "unknown", "reason": "reference_not_completed"}
        return self._authorize(bound, self._evidence(bound, "NONE", response["result"]))
