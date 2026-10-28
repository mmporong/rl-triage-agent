"""Owned POSIX repair parent: admission and execution share one physical lease."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_brev_repair_transport_v1 as T
import agentic_play_job_v4 as JOB
import run_agentic_brev_repair_unit_v1 as UNIT
import vision_brev_backend_v1 as OWNED

SCHEMA = T.SCHEMA + "_remote"
SOURCE = T.REMOTE_SOURCE
SOURCES = T.SOURCES
PRODUCTION = (UNIT.RepairProvider, subprocess.Popen, OWNED._process_identity, subprocess.run)
SECRET_ENV = re.compile(r"(?:^|_)(?:API|TOKEN|SECRET|KEY)(?:_|$)", re.IGNORECASE)


def _os_function(name) -> Any:
    value = getattr(os, name, None)
    if not callable(value):
        raise TypeError("registered POSIX OS function unavailable: " + name)
    return value


def provider(registration_ref, root_map, *, allow_fixture=False, factory: Any = PRODUCTION[0], fixture_services=None):
    if not allow_fixture:
        T.require(factory is PRODUCTION[0] is UNIT.RepairProvider and fixture_services is None,
                  "production repair provider identity/injection guard")
        # Do not supply services=None through a substitute constructor; use the real default.
        return UNIT.RepairProvider(registration_ref, root_map)
    T.require(fixture_services is not None and fixture_services.synthetic_cpu is True, "explicit synthetic provider services")
    return factory(registration_ref, root_map, services=fixture_services, allow_fixture=True)


def prepare(body, *, allow_fixture=False):
    T.require(set(body) == {"operation_id", "registration_ref", "contract_ref", "root_map", "request_ref",
                            "authorization_ref", "admission_ref", "inventory_ref", "inventory_observed_at_utc", "upload"},
              "remote invocation fields")
    reg, config = T.registered(body["registration_ref"], body["root_map"], body["contract_ref"], allow_fixture=allow_fixture)
    T.require(body["root_map"] == reg["layout"]["root_maps"]["provider"], "producer core root map")
    if not allow_fixture:
        T.require(os.name == "posix" and ROOT.resolve() == Path(config["remote_repo"]).resolve()
                  and Path(sys.executable).resolve() == Path(config["remote_python"]).resolve(), "registered remote install/interpreter")
    request = reg["resolver"].read(body["request_ref"])
    T.require(request["operation_id"] == body["operation_id"]
              and request["authorization"] == body["authorization_ref"]
              and request["registration"] == body["registration_ref"], "remote exact request/approval")
    T.E.verify_model(request, reg)
    T.E.verify_authorization(request, body["authorization_ref"], reg)
    folder = Path(config["remote_control_root"]) / body["operation_id"]
    T.safe(folder)
    return reg, config, request, folder


def _current(identity):
    return OWNED._identity_is_current(identity)


def _status(folder, body):
    launch = folder / "launch.json"
    if not launch.exists():
        return {"status": "unknown", "owner_current": False, "owner_ended": False}
    saved = T.E.E.loads(T.safe(launch, exists=True).read_text(encoding="utf-8"))
    T.require(saved["request_ref"] == body["request_ref"] and saved["operation_id"] == body["operation_id"], "owned launch request")
    identity = saved["identity"]
    current = _current(identity)
    release = folder / "release.json"
    result = folder / "result-ref.json"
    failure = folder / "failure.json"
    return {"status": "completed" if result.exists() and release.exists() and not current else
                       "hold" if failure.exists() and not current else "pending",
            "owner_current": current, "owner_ended": not current, "launch": T.E.ref(launch),
            "release": T.E.ref(release) if release.exists() else None,
            "outcome": T.E.E.loads(result.read_text(encoding="utf-8")) if result.exists() else None,
            "failure": T.E.ref(failure) if failure.exists() else None}


def start(body, reg, config, folder, *, popen: Any = PRODUCTION[1], identity: Any = PRODUCTION[2]):
    T.require(popen is PRODUCTION[1] is subprocess.Popen and identity is PRODUCTION[2] is OWNED._process_identity,
              "production parent launch identity/injection guard")
    if (folder / "launch-claim.json").exists():
        return _status(folder, body)
    folder.mkdir(parents=True, exist_ok=True)
    T.save(folder / "launch-claim.json", {"operation_id": body["operation_id"], "request_ref": body["request_ref"],
        "registration_ref": body["registration_ref"], "authorization_ref": body["authorization_ref"]})
    invocation = T.save(folder / "parent-input.json", body)
    argv = [config["remote_python"], "-I", "-B", str(ROOT / SOURCE), "--owner", invocation["path"],
            "--owner-sha256", invocation["sha256"]]
    env = {key: value for key, value in os.environ.items() if SECRET_ENV.search(key) is None
           and key not in {"PYTHONPATH", "PYTHONHOME"}}
    with (folder / "owner.stdout").open("xb") as out, (folder / "owner.stderr").open("xb") as err:
        process = popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env,
                        start_new_session=True, close_fds=True)
        owned = identity(process.pid)
        T.require(owned.get("pid") == process.pid and owned.get("posix_group_verified") is True,
                  "owned parent POSIX identity")
        T.save(folder / "launch.json", {"operation_id": body["operation_id"], "request_ref": body["request_ref"],
            "identity": owned, "argv": argv, "environment_secrets_removed": True})
    return _status(folder, body)


def fresh_admission(body, reg, config, folder, lease):
    current = T.clock()
    observed = T.E.E._stamp(body["inventory_observed_at_utc"])
    T.require(0 <= (current - observed).total_seconds() <= 30, "provider inventory stale/future before capture")
    resolver = reg["resolver"]
    inventory = resolver.file(body["inventory_ref"])
    target = config["target"]
    row = T.E.GPU.verify_provider_inventory(inventory, body["inventory_ref"]["sha256"], target["resource_id"])
    T.require(row["name"] == target["resource_name"] and row["instance_type"] == target["instance_type"], "provider inventory target")
    basis_path = resolver.file(reg["layout"]["lease_basis"])
    basis = JOB.validate_protocol(basis_path, repo=reg["repo"], now=current)

    def check():
        JOB.verify_lease(lease, protocol=basis, protocol_sha=T.E.C.digest_lf_file(basis_path), now=T.clock())

    capture_dir = folder / "admission"
    capture_dir.mkdir(exist_ok=False)
    queries = []
    for name, command in (("gpu", T.E.GPU.GPU_COMMAND), ("process", T.E.GPU.PROCESS_COMMAND),
                          ("model", ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"])):
        check()
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=False)
        check()
        T.new_bytes(capture_dir / (name + ".csv"), result.stdout)
        T.new_bytes(capture_dir / (name + ".stderr"), result.stderr)
        stamp = T.clock().isoformat()
        receipt = T.save(capture_dir / (name + ".receipt.json"), {"command": command, "exit_code": result.returncode,
            "query_at_utc": stamp, "stdout_sha256": T.E.J.file_sha(capture_dir / (name + ".csv"))})
        queries.append({"receipt": receipt, "stderr": T.E.ref(capture_dir / (name + ".stderr")), "query_at_utc": stamp})
        T.require(result.returncode == 0 and result.stderr == b"", "fresh GPU query failed")
    admitted = T.clock()
    T.require(0 <= (admitted - observed).total_seconds() <= 30, "provider inventory stale/future after capture")
    lease_body = T.E.E.loads(Path(lease["path"]).read_text(encoding="utf-8"))
    T.require(T.E.E._stamp(lease_body["acquired_at_utc"]) <= admitted < T.E.E._stamp(lease_body["expires_at_utc"]), "current admission outside lease")
    proof = JOB.build_resource_proof(basis_path, provider_inventory_path=inventory,
        nvidia_gpu_csv_path=capture_dir / "gpu.csv", nvidia_process_csv_path=capture_dir / "process.csv",
        observed_at_utc=admitted.isoformat(), expires_at_utc=min(admitted + timedelta(seconds=120),
            T.E.E._stamp(lease_body["expires_at_utc"])).isoformat(), lease_reference=lease, repo=reg["repo"])
    proof_ref = T.save(capture_dir / "resource.json", proof)
    boot = T.new_bytes(capture_dir / "boot.txt", Path("/proc/sys/kernel/random/boot_id").read_bytes())
    admission = T.save(capture_dir / "admission.json", {"schema": T.E.SCHEMA + "_admission",
        "operation_id": body["operation_id"], "registration": body["registration_ref"], "resource": proof_ref,
        "boot": boot, "gpu_model": T.E.ref(capture_dir / "model.csv"),
        "provider_observed_at_utc": body["inventory_observed_at_utc"], "query_at_utc": queries[-1]["query_at_utc"],
        "admission_at_utc": admitted.isoformat(), "lease_acquired_at_utc": lease_body["acquired_at_utc"],
        "lease_expires_at_utc": lease_body["expires_at_utc"]})
    UNIT.admission_body(admission, body["operation_id"], reg, T.clock(), entry=True)
    T.save(capture_dir / "queries.json", {"queries": queries, "admission": admission})
    T.save(folder / "admission-ref.json", admission)
    return admission, basis


def _cleanup_execution_groups(reg, operation):
    """Only identities emitted by this operation's immutable execution records."""
    output = reg["resolver"].path(reg["layout"]["output_root"]) / operation
    for mode in ("train", "evaluate"):
        path = output / mode / "execution.json"
        if not path.exists():
            continue
        raw = T.E.E.loads(T.safe(path, exists=True).read_text(encoding="utf-8"))
        identity = raw.get("launch")
        if identity is None:
            continue
        members = OWNED._group_members(identity["pid"])
        if members:
            T.require(OWNED._identity_is_current(identity), "unverified child group requires main Stop")
            _os_function("killpg")(identity["pgid"], signal.SIGTERM)
            until = time.monotonic() + 5
            while OWNED._group_members(identity["pid"]) and time.monotonic() < until:
                time.sleep(0.05)
            if OWNED._group_members(identity["pid"]):
                T.require(OWNED._identity_is_current(identity), "child identity changed during cleanup")
                kill = getattr(signal, "SIGKILL", None)
                if kill is None:
                    raise RuntimeError("POSIX SIGKILL unavailable")
                _os_function("killpg")(identity["pgid"], kill)
                until = time.monotonic() + 5
                while OWNED._group_members(identity["pid"]) and time.monotonic() < until:
                    time.sleep(0.05)
        T.require(not OWNED._group_members(identity["pid"]), "owned GPU children remain; Stop required")


def _record_release(body, config, folder, lease_ref, *, children_closed):
    """Capture facts after the lease context exits, including failed executions."""
    snapshot = T.E.E.loads(Path(lease_ref["path"]).read_text(encoding="utf-8"))
    locks = T.new_bytes(folder / "locks-after.txt", Path("/proc/locks").read_bytes())
    held = JOB.proc_lock_owners(Path(locks["path"]).read_text(encoding="utf-8"))
    identity = snapshot["lock_identity"]
    expected = (snapshot["owner"]["pid"], _os_function("major")(identity["device"]),
                _os_function("minor")(identity["device"]), identity["inode"])
    paths = JOB.lease_paths(config["target"]["resource_id"])
    heartbeat = T.new_bytes(folder / "heartbeat-after.json", paths["heartbeat"].read_bytes())
    value = T.E.E.loads(Path(heartbeat["path"]).read_text(encoding="utf-8"))
    facts = {"schema": SCHEMA + "_release", "operation_id": body["operation_id"],
        "request_ref": body["request_ref"], "lease": lease_ref, "locks_after": locks,
        "heartbeat_after": heartbeat, "owned_children_closed": children_closed,
        "flock_released": expected not in held,
        "heartbeat_stopped": value["active"] is False and value["lease_id"] == snapshot["lease_id"],
        "synthetic_cpu": False}
    T.save(folder / "cleanup-observation.json", facts)
    T.require(all(facts[key] is True for key in ("owned_children_closed", "flock_released", "heartbeat_stopped")),
              "owner cleanup incomplete; main Stop required")
    T.save(folder / "release.json", facts)


def owner(body):
    reg, config, _request, folder = prepare(body)
    output_root = T.safe(reg["resolver"].path(reg["layout"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    stop_path = Path(config["remote_stop_intent_path"])
    deadline = time.monotonic() + config["limits"]["owner_ttl_s"]
    T.require(T.clock() + timedelta(seconds=config["limits"]["owner_ttl_s"]) <= T.E.E._stamp(config["deadline_utc"]), "owner TTL exceeds registered deadline")
    basis_path = reg["resolver"].file(reg["layout"]["lease_basis"])
    lease_ref, children_closed = None, False
    try:
        try:
            # JOB4 is the registered extension; core JOB2 identity is unchanged.
            with JOB.aws_f1_play_scope(), JOB.resource_lease(basis_path, repo=reg["repo"],
                    ttl_s=config["limits"]["owner_ttl_s"]) as lease:
                lease_ref = lease
                try:
                    admission, basis = fresh_admission(body, reg, config, folder, lease)
                    command = folder / "execute.json"
                    wait_end = time.monotonic() + config["limits"]["command_wait_s"]
                    while not command.exists():
                        T.require(time.monotonic() < min(deadline, wait_end) and not stop_path.exists(), "owner TTL/Stop before execute")
                        JOB.verify_lease(lease, protocol=basis, protocol_sha=T.E.C.digest_lf_file(basis_path), now=T.clock())
                        time.sleep(0.1)
                    value = T.E.E.loads(T.safe(command, exists=True).read_text(encoding="utf-8"))
                    T.require(value == {"operation_id": body["operation_id"], "request_ref": body["request_ref"],
                        "authorization_ref": body["authorization_ref"], "admission_ref": admission}, "one exact remote execute command")
                    T.save(folder / "execute-claim.json", value)
                    T.require(not stop_path.exists() and time.monotonic() < deadline, "owner Stop/deadline")
                    worker = provider(body["registration_ref"], body["root_map"])
                    outcome = worker.execute(body["operation_id"], body["request_ref"], body["authorization_ref"], admission)
                    T.save(folder / "result-ref.json", outcome)
                finally:
                    _cleanup_execution_groups(reg, body["operation_id"])
                    children_closed = True
        finally:
            if lease_ref is not None:
                _record_release(body, config, folder, lease_ref, children_closed=children_closed)
    except BaseException as error:
        T.save(folder / "failure.json", {"schema": SCHEMA + "_failure", "operation_id": body["operation_id"],
            "request_ref": body["request_ref"], "error_type": type(error).__name__, "error": str(error),
            "stop_required_if_unreleased": not (folder / "release.json").exists()})
        raise


def _wait(folder, body, seconds, *, admission=False):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        state = _status(folder, body)
        if admission and (folder / "admission-ref.json").exists() and state["owner_current"]:
            return {**state, "admission": T.E.E.loads((folder / "admission-ref.json").read_text(encoding="utf-8"))}
        if not admission and state["status"] in {"completed", "hold"}:
            return state
        time.sleep(0.1)
    return _status(folder, body)


def _lookup_admission(folder, body, reg, state, current):
    """Recover an existing owner's raw admission; never start or renew it."""
    if state["status"] in {"completed", "hold"}:
        return state
    if state["owner_ended"] is True:
        return {**state, "status": "hold", "reason": "admission_owner_ended"}
    pointer = folder / "admission-ref.json"
    if state["owner_current"] is not True or not pointer.exists():
        return state
    try:
        reference = T.E.E.loads(T.safe(pointer, exists=True).read_text(encoding="utf-8"))
        T.require(body["admission_ref"] is None or body["admission_ref"] == reference, "original admission ref changed")
        value = UNIT.admission_body(reference, body["operation_id"], reg, current, entry=True)
        T.require(value["registration"] == body["registration_ref"], "admission registration changed")
        return {**state, "admission": reference, "admission_pointer": T.E.ref(pointer)}
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {**state, "status": "hold", "reason": "invalid_or_stale_admission", "error": str(error)}


def _response(mode, body, reg, config, folder, state):
    seeds = []
    if state["status"] == "completed":
        T.require(state["owner_ended"] and not state["owner_current"], "owner IO process not terminated")
        seeds = [state["outcome"], state["release"], state["launch"]]
    elif mode in {"admit", "lookup"} and state["status"] != "hold" and state.get("admission") is not None:
        pointer = T.E.ref(folder / "admission-ref.json")
        T.require(reg["resolver"].read(pointer) == state["admission"], "owner raw admission pointer changed")
        state = {**state, "admission_pointer": pointer}
        seeds = [state["admission"], pointer, state["launch"]]
        queries = folder / "admission/queries.json"
        if queries.exists():
            seeds.append(T.E.ref(queries))
    elif state["status"] == "hold":
        seeds = [state.get(key) for key in ("failure", "release", "launch")]
    seeds = [reference for reference in seeds if reference is not None]
    return {"schema": SCHEMA, "mode": mode, "operation_id": body["operation_id"],
        "request_ref": body["request_ref"], "authority_issued": False, "evidence_accepted": False,
        "state": state, "packet": T.packet(seeds, reg, config, "remote", operation=body["operation_id"], kind="remote-" + mode) if seeds else None}


def run_rpc(mode, body, *, allow_fixture=False, fixture_engine: Any = None):
    T.require(mode in {"admit", "execute", "lookup"}, "registered repair remote mode")
    T.require(type(allow_fixture) is bool and (allow_fixture or fixture_engine is None), "fixture remote injection guard")
    if allow_fixture:
        if fixture_engine is None:
            raise ValueError("explicit synthetic remote engine required")
        T.require(fixture_engine.synthetic_cpu is True, "explicit synthetic remote engine")
        return fixture_engine(mode, body)
    # Raw host upload is validated against the already mirrored private contract before parsing authority.
    reg, config = T.registered(body["registration_ref"], body["root_map"], body["contract_ref"])
    if body["upload"] is not None:
        T.require(len(body["operation_id"]) == 64 and all(c in "0123456789abcdef" for c in body["operation_id"]), "upload operation id")
        T.unpack(body["upload"], config, "remote", operation=body["operation_id"], kind="host-upload",
                 archive_root=Path(config["remote_control_root"]) / body["operation_id"] / ("upload-" + str(time.time_ns())))
    reg, config, _request, folder = prepare(body)
    T.require(not Path(config["remote_stop_intent_path"]).exists() and T.clock() < T.E.E._stamp(config["deadline_utc"]), "remote Stop/deadline")
    if mode == "admit":
        T.require(body["admission_ref"] is None, "admit cannot accept old admission")
        start(body, reg, config, folder)
        state = _wait(folder, body, config["limits"]["admission_wait_s"], admission=True)
    elif mode == "execute":
        state = _status(folder, body)
        T.require(state["owner_current"] and (folder / "admission-ref.json").exists(), "same owner lease must be alive at execute")
        admission = T.E.E.loads((folder / "admission-ref.json").read_text(encoding="utf-8"))
        T.require(body["admission_ref"] == admission, "same owner fresh admission ref")
        T.save(folder / "execute.json", {"operation_id": body["operation_id"], "request_ref": body["request_ref"],
            "authorization_ref": body["authorization_ref"], "admission_ref": admission})
        state = _wait(folder, body, config["limits"]["owner_ttl_s"])
    else:
        state = _lookup_admission(folder, body, reg, _status(folder, body), T.clock())
    return _response(mode, body, reg, config, folder, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("admit", "execute", "lookup"))
    parser.add_argument("--owner", type=Path)
    parser.add_argument("--owner-sha256")
    args = parser.parse_args(argv)
    if args.owner is not None:
        T.require(args.mode is None and T.E.ref(args.owner)["sha256"] == args.owner_sha256, "owned invocation raw SHA")
        owner(T.E.E.loads(args.owner.read_text(encoding="utf-8")))
    else:
        T.require(args.mode is not None, "remote mode required")
        body = T.E.E.loads(sys.stdin.buffer.read().decode("utf-8"))
        print(json.dumps(run_rpc(args.mode, body), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
