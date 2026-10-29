"""Parent5의 fresh 선택을 기존 Parent prepare 전체 경계에 적용한다."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

import run_agentic_brev_probe_parent_v5 as V5

V1 = V5.V1
SCHEMA = "agentic_brev_probe_pair_parent_v6"
SOURCES = V5.SOURCES | {
    "bench/run_agentic_brev_probe_parent_v6.py",
    "tests/test_run_agentic_brev_probe_parent_v6.py",
}

def _prepare_impl(request_ref: dict, *, allow_fixture: bool = False, home: Path | None = None,
                  now=None) -> dict:
    request = V1.U.read_ref(request_ref)
    V1.require(request.get("schema") == SCHEMA + "_request", "parent request schema")
    if request.get("action") != "probe":
        return {"status": "HOLD", "reason": "repair_unsupported", "dispatched": False}
    protocol, binding = V1.U.read_ref(request["protocol"]), V1.U.read_ref(request["binding"])
    fixture = binding.get("evidence_kind") == V1.E.FIXTURE
    V1.require(type(allow_fixture) is bool and binding.get("evidence_kind") in {V1.E.FIXTURE, V1.E.ACTUAL}
               and (not fixture or allow_fixture), "fixture mode must be explicit")
    V1.require(fixture or (home is None and now is None), "actual HOME/clock injection forbidden")
    V1._public(protocol)
    V1.require(protocol.get("schema") == SCHEMA + "_protocol" and protocol.get("registered_before_execution") is True
               and SOURCES <= protocol.get("sources_lf", {}).keys()
               and protocol["binding_sha256"] == request["binding"]["sha256"]
               and protocol["gpu_binding"] == V1.GPU_BINDING and "gpu_uuid" not in protocol,
               "registered pair source closure/dynamic GPU binding")
    layout = protocol["layout"]
    V1.require(set(layout) == {"repo", "isaaclab", "python"}, "HOME layout fields")
    base = (home or Path.home()).absolute()
    locations = {key: base.joinpath(*V1.U._relative(value)).absolute() for key, value in layout.items()}
    repo, lab, python = (locations[key] for key in ("repo", "isaaclab", "python"))
    V1.require(repo.is_dir() and lab.is_dir() and python.is_file(), "registered repo/lab/Python unavailable")
    V1.require(fixture or repo.resolve() == V1.ROOT.resolve(), "actual code must be registered repository")
    V1.U.verify_sources(protocol, request["source_anchor"], repo=repo, protocol_path=Path(request["protocol"]["path"]))
    V1.require(binding.get("schema") == SCHEMA + "_binding", "pair binding schema")
    pair = binding["expected_request"]
    probe = V1.registered_probe(protocol, pair)
    V1.require(set(pair) == V1.PAIR.PAIR_KEYS and pair.get("schema") == V1.PAIR.PAIR_SCHEMA
               and pair["arms"] == list(V1.ORDER) and pair["seed"] == 2221,
               "registered NONE/F1 s2221 pair only")
    execution = pair["execution"]
    V1.require(execution.get("schema") == SCHEMA + "_execution"
               and execution["order"] == list(V1.ORDER) and execution["gpu_binding"] == V1.GPU_BINDING
               and execution["evidence_kind"] == binding["evidence_kind"]
               and execution["unverified_child_policy"] == V1.UNVERIFIED_CHILD_POLICY
               and execution["source_closure_sha256"] == V1.E.digest(protocol["sources_lf"])
               and V1.E.J.HEX64.fullmatch(str(execution["pair_id"])) is not None,
               "pair execution/source/order binding")
    root = repo.joinpath(*V1.U._relative(protocol["execution_root"]))
    folder = root / "pairs" / execution["pair_id"]
    V1._directory(folder)
    current = now or V1.utc()
    V1.require(folder.exists() or current < V1.E._stamp(protocol["deadline_utc"]), "pair registration expired")
    resolver = V1.E.Resolver(binding["roots"], V1.E.digest(binding["roots"]))
    snapshot, spec, model_transport = V5.fresh_selection(execution, resolver)
    V1.require(pair["probe"] == spec["probe"], "registered pair/model target/frozen request probe mismatch")
    V1.require(pair["selection"] == {"schema": "fresh_selection_pair_binding_v1",
               "api_global_index": execution["selection_source"]["api_global_index"],
               "actual_result_sha256": execution["selection_source"]["actual_result"]["sha256"],
               "journal_terminal_sha256": execution["selection_source"]["terminal"]["sha256"]}
               and pair["commands_sha256"] == V1.PAIR.digest(pair["commands"])
               and set(pair["commands"]) == set(V1.ORDER)
               and pair["checkpoints"] == {"NONE": spec["reference_checkpoint_sha256"],
                                            "F1": spec["checkpoint_sha256"]},
               "pair model/commands/checkpoints binding")
    windows = execution["windows"]
    V1.require(set(windows) == {"lease_ttl_s", "execution_window_s", "max_wall_s", "admission_max_age_s",
                                "admission_valid_s", "provider_wait_s"}
               and all(V1.E._positive(value) for value in windows.values())
               and windows["admission_max_age_s"] == V1.JOB2.FRESH_CAPTURE_S
               and windows["admission_valid_s"] == 120
               and windows["max_wall_s"] <= windows["execution_window_s"] <= windows["lease_ttl_s"]
               and 2 * windows["max_wall_s"] <= pair["budget_gpu_s"] <= spec["budget_gpu_s"],
               "registered pair time/budget windows")
    resource = resolver.read(execution["host_resource"])
    V1.require(set(protocol["provider_target"]) == {"resource_id", "resource_name", "instance_type", "gpu_model"},
               "fixed provider identity/model fields")
    V1.require(resource.get("schema") == "agentic_brev_probe_pair_parent_v1_host_resource"
               and resource.get("status") == "granted" and resource["gpu_binding"] == V1.GPU_BINDING
               and "gpu_uuid" not in resource and pair["resource_sha256"] == execution["host_resource"]["sha256"]
               and resource["target"] == protocol["provider_target"]
               and protocol["provider_target"]["gpu_model"] == "L40S", "host resource provider binding")
    if not folder.exists():
        V1.require(V1.E._stamp(resource["issued_at_utc"]) <= current < V1.E._stamp(resource["expires_at_utc"]),
                   "host resource expired")
    basis_path, backend_path = resolver.file(execution["lease_basis"]), resolver.file(execution["backend"]["protocol"])
    basis, backend = resolver.read(execution["lease_basis"]), resolver.read(execution["backend"]["protocol"])
    V1.require(pair["protocols"] == {"source": execution["source_protocol_sha256_lf"],
               "probe": V1.C.digest_lf_file(backend_path)}
               and V1.JOB2.REQUIRED_SOURCE_FILES <= basis.get("source_sha256_lf", {}).keys(),
               "lease/backend source closure")
    V1.U.verify_sources({"sources_lf": basis["source_sha256_lf"]}, request["source_anchor"], repo=repo,
                        protocol_path=basis_path)
    if not fixture:
        V1.U.verify_sources({"sources_lf": backend["source_sha256_lf"]}, execution["backend"]["source_anchor"], repo=repo)
        V1.U.verify_sources({"sources_lf": {}}, request["source_anchor"], repo=repo, protocol_path=backend_path)
    operations = {}
    V1.require(set(execution["operations"]) == set(V1.ORDER), "exactly two registered operations")
    for arm, side in (("NONE", "reference"), ("F1", "case")):
        op = execution["operations"][arm]
        V1.require(op["side"] == side and op["arm"] == arm and type(op["seed"]) is int and op["seed"] == 2221
                   and op["probe"] == probe and V1.E.J.HEX64.fullmatch(str(op["operation_id"])) is not None
                   and op["args"] == spec["args"] and op["source"]["checkpoint"]["sha256"] == pair["checkpoints"][arm],
                   "operation/probe/CP/args identity")
        V1.C.BACKEND.V1._name(op["run_name"])
        unit, stage = root / "operations" / op["operation_id"], root / "operations" / op["operation_id"] / "raw"
        V1._directory(stage)
        V1.E._logical(op["remote_stage"])
        V1.require(fixture or stage.as_posix() == op["remote_stage"], "exact registered POSIX stage projection")
        V1.require(PureWindowsPath(op["host_stage"]).is_absolute() or PurePosixPath(op["host_stage"]).is_absolute(),
                   "original host stage required")
        if not fixture:
            V1.require(all(op["source"][key]["sha256"] == V1.U.INITIAL[side][key + "_sha256"]
                           for key in ("checkpoint", "telemetry")), "initial checkpoint/telemetry source only")
        source_resolver = V1.E.Resolver(op["source_roots"], V1.E.digest(op["source_roots"]))
        combined = V1.E.Resolver(op["source_roots"] + binding["roots"], V1.E.digest(op["source_roots"] + binding["roots"]))
        packet = V1.E._logical(op["provider_packet_path"])
        V1.require(sum(remote == packet or remote in packet.parents for remote, _ in resolver.rows) == 1,
                   "provider packet outside registered transport root")
        stage_logical = V1.E._logical(op["remote_stage"])
        V1.require(all(remote != stage_logical and remote not in stage_logical.parents and stage_logical not in remote.parents
                       for remote, _ in combined.rows), "stage overlaps source/transport root")
        checkpoint = source_resolver.file(op["source"]["checkpoint"])
        source_manifest = source_resolver.file(op["source"]["manifest"])
        V1.E._source({"source": op["source"], **op}, source_resolver, {"seed": 2221, "arm": arm,
            "checkpoint_path": op["source"]["checkpoint"]["path"] if fixture else str(checkpoint),
            "checkpoint_sha256": pair["checkpoints"][arm],
            "training_manifest_path": op["source"]["manifest"]["path"] if fixture else str(source_manifest)})
        expected = execution["backend"]["expected_source"]
        V1.require(expected["source_anchor"] == execution["backend"]["source_anchor"]
                   and expected["protocol"]["sha256_lf"] == V1.C.digest_lf_file(backend_path)
                   and all(expected[key] == backend[key] for key in V1.E.HASHES), "backend source snapshot")
        command = [str(python), "-B", str(repo / "bench/brev_recovery_worker_v1.py"),
                   "--job", str(stage / "job.json"), "--output", str(stage / "worker.json")]
        V1.require(pair["commands"][arm] == {"operation_id": op["operation_id"], "stage": op["host_stage"],
                   "remote_stage": op["remote_stage"], "run_name": op["run_name"], "args": op["args"],
                   "argv": command, "probe": probe}, "exact approved command descriptor/argv")
        operations[arm] = {"op": op, "unit": unit, "stage": stage, "checkpoint": checkpoint,
                           "manifest": source_manifest}
    V1.require(operations["NONE"]["unit"] != operations["F1"]["unit"], "distinct operation/stage required")
    prepared = {"status": "prepared", "request": request, "request_ref": request_ref, "protocol": protocol,
        "binding": binding, "execution": execution, "repo": repo, "lab": lab, "python": python,
        "folder": folder, "resolver": resolver, "operations": operations, "basis_path": basis_path,
        "basis": basis, "basis_sha": V1.C.digest_lf_file(basis_path), "backend_path": backend_path,
        "backend": backend, "resource": resource, "snapshot": snapshot, "fixture": fixture, "now": now,
        "model_transport": model_transport}
    prepared["authority"] = V1.verify_authority(prepared)
    return prepared


@contextmanager
def fresh_parent_scope():
    with V5.BASE.registered_worker_scope():
        original = V1._selection, V1.SCHEMA, V1.prepare
        V1._selection, V1.SCHEMA, V1.prepare = V5.fresh_selection, SCHEMA, _prepare_impl
        try:
            yield
        finally:
            V1._selection, V1.SCHEMA, V1.prepare = original


def prepare(*args, **kwargs):
    with fresh_parent_scope():
        return _prepare_impl(*args, **kwargs)


def lookup(*args, **kwargs):
    with fresh_parent_scope():
        return V1.lookup(*args, **kwargs)


def run(*args, **kwargs):
    with fresh_parent_scope():
        return V1.run(*args, **kwargs)


def verify_authority(*args, **kwargs):
    with fresh_parent_scope():
        return V1.verify_authority(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    with fresh_parent_scope():
        return V1._verify_completion(*args, **kwargs)


def main() -> None:
    with fresh_parent_scope():
        V1.main()


def __getattr__(name: str):
    return getattr(V1, name)


if __name__ == "__main__":
    main()
