"""명시 승인된 POSIX probe 한 건만 실행한다. 기본은 prepare/lookup이며 권한을 발급하지 않는다."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import agentic_brev_controller_scope_v1 as S
import agentic_play_job_v1 as JOB
import vision_brev_backend_v1 as OWNED
from run_agentic_development_v2 import verify_sources

E, C = S.E, S.C
SCHEMA = "agentic_brev_probe_unit_v1"
SOURCES = S.SOURCE_FILES | set(C.SOURCE_FILES) | {
    "bench/run_agentic_brev_probe_unit_v1.py", "tests/test_agentic_brev_probe_unit_v1.py",
    "bench/agentic_play_job_v1.py", "bench/run_agentic_capture_unit_v1.py",
    "evals/vision_brev_backend_v1.py", "bench/brev_recovery_worker_v1.py",
}
INITIAL = {
    "case": {"arm": "F1", "checkpoint_sha256": "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731",
             "telemetry_sha256": "07ad9b4b3981eafd3a4ed84aba4a3ef04f54b94ddbc7be5b0b189d55b2e8ff6e"},
    "reference": {"arm": "NONE", "checkpoint_sha256": "b346a0c3e3ad6410ce93dcf73e6165ed69e2d6ab95b2c5398943b3a8874cf54d",
                  "telemetry_sha256": "dccf7b0fab6060dee8c7b209451d6f15591f2ac3b630eb5178cf1e16b3f3998b"},
}


def require(ok: bool, reason: str) -> None:
    E.require(ok, reason)


def ref(path: Path) -> dict:
    path = path.absolute()
    E.J._read(path)
    return {"path": str(path), "sha256": E.J.file_sha(path)}


def read_ref(value: dict) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "immutable reference fields")
    path = Path(value["path"])
    require(path.is_absolute() and E.J.file_sha(path) == value["sha256"], "immutable reference SHA")
    E.J._read(path)
    return E.loads(path.read_text(encoding="utf-8"))


def _relative(value: str) -> tuple[str, ...]:
    require(isinstance(value, str) and bool(value) and "\\" not in value and ":" not in value
            and not PurePosixPath(value).is_absolute()
            and all(p not in {"", ".", ".."} for p in value.split("/")), "registered relative layout")
    return PurePosixPath(value).parts


def _now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10


def _validate_authorization(plan: dict, authorization_ref: dict, authorization: dict,
                            repo: Path, unit: Path, *, fixture: bool) -> dict:
    """승인·소비 원문 prefix와 controller 상태를 읽는다. 승인이나 consume은 수행하지 않는다."""
    registration = plan["source_registration"]
    require(Path(registration["repo"]).resolve() == repo.resolve()
            and registration["anchor"] == plan["source_anchor"], "authorization source registration differs")
    registered = read_ref(registration["protocol"])
    require(SOURCES <= registered.get("sources_lf", {}).keys(), "authorization source closure")
    path = Path(authorization_ref["path"])
    ledger_path = path.parent / "probes.ledger.jsonl"
    require(ledger_path.is_file() and not ledger_path.is_symlink() and ledger_path.stat().st_nlink == 1,
            "host authorization ledger missing or nonregular")
    before = E.J.file_sha(ledger_path)
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    rows = [row for row in ledger.events() if row.get("request_id") == authorization["request_id"]]
    require(len(rows) >= 3 and [row.get("event") for row in rows[:3]] == ["proposed", "approved", "consumed"]
            and E.digest(rows[:3]) == plan["authorization_prefix_sha256"], "host authorization prefix differs")
    state = ledger.requests().get(authorization["request_id"], {}).get("state")
    reference = {**authorization_ref, "request_id": authorization["request_id"]}
    with S.BrevProbeScope(allow_fixture=fixture, source_registration=registration):
        if state == "done":
            require((unit / "outcome.json").is_file() and (unit / "intent.json").is_file(),
                    "done authorization cannot dispatch a new unit")
            outcome, intent = E.J._read(unit / "outcome.json"), E.J._read(unit / "intent.json")
            require(outcome.get("status") == "complete" and intent.get("authorization_ref") == authorization_ref
                    and intent.get("plan_ref") == plan["_plan_ref"], "done authorization requires exact completed unit")
            checked = C._authorization(reference, authorization["snapshot"], "done", require_digest=True)
        else:
            checked = C.validate_probe_authorization(reference, authorization["snapshot"])
    # RegistryLedger의 기존 effect-key/registry 결합도 검증한다.
    ledger._bound_request(authorization["request_id"], str(state))
    require(E.J.file_sha(ledger_path) == before, "authorization ledger changed during validation")
    return checked


def prepare(plan_ref: dict, authorization_ref: dict | None = None, *, allow_fixture: bool = False,
            home: Path | None = None, now: datetime | None = None) -> dict:
    """원문 읽기만 한다. 실제 lease 재검증·GPU query·dispatch는 execute 경계 안에 있다."""
    plan = read_ref(plan_ref)
    require(plan.get("schema") == SCHEMA + "_plan", "unit plan schema")
    if plan.get("action") != "probe":
        return {"status": "HOLD", "reason": "repair_unsupported", "execution_permitted": False}
    require(type(allow_fixture) is bool and plan["evidence_kind"] in {E.ACTUAL, E.FIXTURE}
            and (plan["evidence_kind"] == E.ACTUAL or allow_fixture), "fixture mode must be explicit")
    fixture_mode = plan["evidence_kind"] == E.FIXTURE
    require(home is None or fixture_mode, "HOME override is fixture-only")
    require(now is None or fixture_mode, "clock override is fixture-only")
    protocol = read_ref(plan["protocol"])
    require(protocol.get("schema") == SCHEMA + "_protocol" and protocol.get("registered_before_execution") is True
            and SOURCES <= protocol["sources_lf"].keys(), "unit source/protocol closure")
    layout = protocol["layout"]
    require(set(layout) == {"repo", "isaaclab", "python"}, "HOME layout fields")
    base = (home or Path.home()).absolute()
    locations = {key: base.joinpath(*_relative(value)).absolute() for key, value in layout.items()}
    repo, lab, python = (locations[key] for key in ("repo", "isaaclab", "python"))
    require(repo.is_dir() and lab.is_dir() and python.is_file(), "registered repo/lab/Python unavailable")
    require(fixture_mode or repo.resolve() == ROOT.resolve(), "unit code is outside registered repo")
    verify_sources(protocol, plan["source_anchor"], repo=repo, protocol_path=Path(plan["protocol"]["path"]))
    current = now or _now()
    require(E.J.HEX64.fullmatch(str(plan["operation_id"])) is not None, "stable operation ID")
    identity = plan["identity"]
    require(set(identity) == {"side", "arm", "seed", "probe", "run_name"}
            and identity["side"] in INITIAL and identity["arm"] == INITIAL[identity["side"]]["arm"]
            and type(identity["seed"]) is int and identity["seed"] == 2221
            and identity["probe"] in protocol["allowed_probes"] and identity["probe"] in E.PROBES,
            "registered single probe identity")
    E.B._relative_hashes(protocol["sources_lf"], "unit sources")
    unit = repo.joinpath(*_relative(protocol["execution_root"]), plan["operation_id"])
    stage = unit / "raw"
    require(stage.resolve().is_relative_to(repo.resolve())
            and not any(p.is_symlink() for p in (stage, *stage.parents)), "stage root/symlink boundary")
    require(unit.exists() or current < E._stamp(protocol["deadline_utc"]), "unit protocol expired")
    require(Path(plan["stage"]).absolute() == stage, "registered stage/operation binding")
    require(fixture_mode or plan["logical_stage"] == stage.as_posix(), "actual stage must preserve native POSIX path")
    roots = deepcopy(plan["roots"])
    future = [row for row in roots if row["remote"] == plan["logical_stage"]]
    require(len(future) == 1 and Path(future[0]["local"]).absolute() == stage, "registered stage root-map")
    source_roots = [row for row in roots if row is not future[0]]
    resolver = E.Resolver(source_roots, E.digest(source_roots))
    source = plan["source"]
    allowed = protocol["inputs"][identity["side"]]
    observed = {"arm": identity["arm"], "seed": identity["seed"],
        "checkpoint_sha256": source["checkpoint"]["sha256"], "telemetry_sha256": source["telemetry"]["sha256"],
        "lineage_id": source["lineage_id"]}
    require(observed == allowed, "protocol source CP/telemetry/lineage differs")
    if not fixture_mode:
        require(all(observed[k] == v for k, v in INITIAL[identity["side"]].items()), "initial s2221 source CP only")
    checkpoint, manifest = resolver.file(source["checkpoint"]), resolver.file(source["manifest"])
    job_view = {"seed": identity["seed"], "arm": identity["arm"], "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": source["checkpoint"]["sha256"], "training_manifest_path": str(manifest)}
    if fixture_mode:
        # CPU에서는 원래 POSIX alias를 유지하고 실제 backend 대신 명시 fakeprocess를 주입한다.
        job_view.update(checkpoint_path=source["checkpoint"]["path"], training_manifest_path=source["manifest"]["path"])
    E._source({"source": source, **identity}, resolver, job_view)
    backend_path = repo.joinpath(*_relative(protocol["backend"]["protocol"]["path"]))
    backend_ref = {"path": str(backend_path), "sha256": protocol["backend"]["protocol"]["sha256"]}
    backend_protocol = read_ref(backend_ref)
    expected = plan["expected_source"]
    require(expected["source_anchor"] == protocol["backend"]["source_anchor"]
            and expected["protocol"] == {"path": protocol["backend"]["protocol"]["path"],
                "sha256_lf": C.digest_lf_file(backend_path)}
            and all(expected[key] == backend_protocol[key] for key in E.HASHES), "backend runtime source binding")
    require(plan["lease_basis"] == protocol["lease_basis"]
            and protocol["lease_usage"] == "physical_mutex_only_not_probe_authority", "lease basis not registered")
    resource_proof = resolver.read(plan["resource"]["proof"])
    resource = plan["resource"]
    require(resource_proof.get("schema") == "brev_resource_proof_v1"
            and resource_proof.get("status") == "granted"
            and all(resource_proof.get(k) == resource[k] for k in (
                "resource_id", "gpu_uuid", "lease_id", "protocol_sha256"))
            and all(resource_proof.get(k) == v for k, v in resource["capture"].items()),
            "host resource grant/capture mismatch")
    if not unit.exists():
        require(E._stamp(resource_proof["issued_at_utc"]) <= current < E._stamp(resource_proof["expires_at_utc"])
                <= E._stamp(protocol["deadline_utc"]), "probe resource proof expired")
    require(plan["resource"]["protocol"]["sha256"] == plan["protocol"]["sha256"]
            and resource_proof["protocol_sha256"] == plan["protocol"]["sha256"]
            and resource_proof.get("lease_basis_sha256") == plan["lease_basis"]["sha256"]
            and resource_proof.get("lease_snapshot_sha256") == plan["lease_snapshot"]["sha256"]
            and resource_proof.get("capture_parent_sha256") == plan["capture_parent"]["sha256"]
            and resource_proof.get("operation_id") == plan["operation_id"], "host resource proof parent binding required")
    result = {"status": "approval_required", "plan": plan, "plan_ref": plan_ref, "protocol": protocol,
        "repo": repo, "lab": lab, "python": python, "unit": unit, "stage": stage,
        "projection": {"original_source": deepcopy(source), "resolved_checkpoint": str(checkpoint),
                       "resolved_manifest": str(manifest), "root_map_sha256": E.digest(roots)},
        "backend_protocol": {**backend_protocol, "protocol_path": str(backend_path),
                             "protocol_sha256_lf": C.digest_lf_file(backend_path)}}
    if authorization_ref is None:
        return result
    authorization = read_ref(authorization_ref)
    expected_kind = C.FIXTURE_EVIDENCE if plan["evidence_kind"] == E.FIXTURE else C.ACTUAL_EVIDENCE
    require(authorization.get("schema") == "selection_probe_authorization_v1"
            and authorization.get("evidence_kind") == expected_kind and bool(authorization.get("request_id")),
            "host immutable probe AUTH required")
    spec = C._frozen_request_spec(authorization["snapshot"], authorization["choice"])
    execution = authorization["execution_plan"]
    require(authorization["snapshot_digest"] == C.L.digest(authorization["snapshot"])
            and authorization["prereg"] == spec["prereg"] and spec["probe"] == identity["probe"]
            and spec["checkpoint_sha256"] == source["checkpoint"]["sha256"]
            and execution["checkpoint_sha256"] == source["checkpoint"]["sha256"]
            and execution["seed"] == identity["seed"] and execution["args"] == spec["args"]
            and execution["run_name"] == identity["run_name"] and Path(execution["stage"]).absolute() == stage
            and execution["gpu_contract"] == E.GPU_CONTRACT
            and E._positive(protocol["max_wall_s"]) and protocol["max_wall_s"] <= spec["budget_gpu_s"],
            "AUTH plan/probe/CP/stage mismatch")
    checked = _validate_authorization({**plan, "_plan_ref": plan_ref}, authorization_ref,
        authorization, repo, unit, fixture=fixture_mode)
    result.update(status="prepared", authorization_ref=authorization_ref, authorization=authorization,
                  authorization_state=checked["request"]["state"])
    return result


@contextmanager
def parent_resource(prepared: dict):
    """main이 보유한 실제 lease와 발급한 proof를 읽는다. 자식이 probe granted를 만들지 않는다."""
    plan, protocol = prepared["plan"], prepared["protocol"]
    basis_path = prepared["repo"].joinpath(*_relative(plan["lease_basis"]["path"]))
    require(E.J.file_sha(basis_path) == plan["lease_basis"]["sha256"], "lease basis SHA")
    basis = JOB.validate_protocol(basis_path, repo=prepared["repo"])
    basis_sha = C.digest_lf_file(basis_path)
    capture = JOB.validate_resource_proof(Path(plan["capture_parent"]["path"]), plan["capture_parent"]["sha256"],
        protocol=basis, protocol_sha=basis_sha, now=_now(), require_fresh=True)
    require(capture["lease"] == plan["lease_snapshot"] and capture["_lease"]["live_lock_proof"] is True,
            "actual POSIX parent lock proof required")
    resource = plan["resource"]
    require(resource["lease_id"] == capture["_lease"]["lease_id"]
            and resource["capture"] == capture["capture"]
            and resource["resource_id"] == protocol["resource_id"] == capture["resource_id"]
            and resource["gpu_uuid"] == protocol["gpu_uuid"] == capture["gpu_uuid"], "parent lease/resource identity")

    def check() -> None:
        live = JOB.verify_lease(plan["lease_snapshot"], protocol=basis, protocol_sha=basis_sha, now=_now())
        require(live["live_lock_proof"] is True and live["lease_id"] == resource["lease_id"], "parent lease lost")
        snapshots = prepared["unit"] / "lease-observations"
        ordinal = len(list(snapshots.glob("*"))) + 1
        folder = snapshots / f"{ordinal:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        owner = live["owner"]["pid"]
        paths = JOB.lease_paths(resource["resource_id"])
        raw = {"proc-locks.txt": Path("/proc/locks"), "owner-stat.txt": Path(f"/proc/{owner}/stat"),
               "heartbeat.json": paths["heartbeat"], "lease-snapshot.json": Path(plan["lease_snapshot"]["path"])}
        for name, source_path in raw.items():
            with (folder / name).open("xb") as stream:
                stream.write(source_path.read_bytes())
        confirmed = JOB.verify_lease(plan["lease_snapshot"], protocol=basis, protocol_sha=basis_sha, now=_now())
        require(confirmed["owner"] == live["owner"] and confirmed["lease_id"] == live["lease_id"]
                and confirmed["live_lock_proof"] is True, "parent owner changed during observation")
        E.write_new(folder / "proof.json", {"schema": SCHEMA + "_mutex_observation", "observed_at_utc": _now().isoformat(),
            "physical_mutex_only": True, "probe_authority_issued": False, "lease_reference": plan["lease_snapshot"],
            "live_check": confirmed, "raw_sha256": {name: E.J.file_sha(folder / name) for name in raw}})

    reader = E.GPU.StrictRemoteGpuReader(resource["capture"], prepared["unit"] / "gpu-observations")

    class GuardedReader:
        @property
        def last_receipt(self):
            return reader.last_receipt

        def __call__(self):
            check()
            return reader()

    check()
    yield {"check": check, "reader": GuardedReader()}
    check()


def _backend(prepared: dict, live: dict):
    return OWNED.SafeRecoveryBackend(prepared["lab"], prepared["python"], prepared["backend_protocol"],
        prepared["protocol"]["backend"]["source_anchor"], read_gpu=live["reader"], popen=_owned_popen)


def _owned_popen(args, **kwargs):
    environment = dict(kwargs["env"])
    for key in ("NVIDIA_API_KEY", "OPENAI_API_KEY"):
        environment.pop(key, None)
    return subprocess.Popen(args, **{**kwargs, "env": environment})


def _request(prepared: dict) -> dict:
    plan, stage = prepared["plan"], prepared["stage"]
    names = {"job": "job.json", "worker": "worker.json", "report": "report.json", "execution": "execution.json",
             "backend_result": "backend_result.json", "report_jobs": "report.jobs.json"}
    if plan["identity"]["probe"] == E.B.OPTIMIZER_PROBE:
        names.pop("report_jobs")
        names.update(inspection="inspection.manifest.json", records="optimizer.records.jsonl", readback="inspection.readback.json")
    files = {key: {"path": plan["logical_stage"] + "/" + name, "sha256": E.J.file_sha(stage / name)}
             for key, name in names.items()}
    return {"schema": E.REQUEST_SCHEMA, "action": "probe", "operation_id": plan["operation_id"],
        "evidence_kind": plan["evidence_kind"], **plan["identity"], "stage": plan["logical_stage"],
        "source": plan["source"], "expected_source": plan["expected_source"], "resource": plan["resource"],
        "max_wall_s": prepared["protocol"]["max_wall_s"], "files": files}


def lookup(prepared: dict, *, allow_fixture: bool = False) -> dict:
    unit, plan = prepared["unit"], prepared["plan"]
    if not (unit / "intent.json").exists():
        return {"status": "pending_unknown", "operation_id": plan["operation_id"], "redispatch": False,
                "reason": "incomplete_existing_unit_intent"}
    intent = E.J._read(unit / "intent.json")
    require("authorization_state" in prepared, "existing unit lookup requires controller AUTH validation")
    require(intent["plan_ref"] == prepared["plan_ref"]
            and intent["operation_id"] == plan["operation_id"], "existing unit differs from plan")
    if "authorization_ref" in prepared:
        require(intent["authorization_ref"] == prepared["authorization_ref"], "existing AUTH changed")
    result_path = unit / "outcome.json"
    if not result_path.exists():
        return {"status": "pending_unknown", "operation_id": plan["operation_id"], "redispatch": False}
    result = E.J._read(result_path)
    if result["status"] == "complete":
        evidence = read_ref(result["evidence"])
        E.verify(evidence, plan["roots"], expected_request_sha256=result["request_sha256"],
                 expected_roots_sha256=E.digest(plan["roots"]), allow_fixture=allow_fixture)
    return result


def run(plan_ref: dict, authorization_ref: dict | None = None, *, execute: bool = False,
        allow_fixture: bool = False, home: Path | None = None, now: datetime | None = None,
        backend_factory=None, resource_service=None) -> dict:
    prepared = prepare(plan_ref, authorization_ref, allow_fixture=allow_fixture, home=home, now=now)
    if prepared["status"] == "HOLD":
        return prepared
    unit, plan = prepared["unit"], prepared["plan"]
    if unit.exists():
        if authorization_ref is None and (unit / "intent.json").is_file():
            intent = E.J._read(unit / "intent.json")
            prepared = prepare(plan_ref, intent["authorization_ref"], allow_fixture=allow_fixture, home=home, now=now)
        return lookup(prepared, allow_fixture=allow_fixture)
    if not execute or prepared["status"] != "prepared":
        return {"status": prepared["status"], "operation_id": plan["operation_id"], "dispatched": False}
    fixture_mode = plan["evidence_kind"] == E.FIXTURE
    require(fixture_mode or os.name == "posix", "actual execution requires owned POSIX VM")
    require(fixture_mode or (backend_factory is None and resource_service is None), "injection is fixture-only")
    require(not fixture_mode or (backend_factory is not None and resource_service is not None),
            "fixture execution requires explicit fakeprocess/resource service")
    unit.mkdir(parents=True, exist_ok=False)
    E.write_new(unit / "intent.json", {"schema": SCHEMA + "_intent", "operation_id": plan["operation_id"],
        "plan_ref": plan_ref, "authorization_ref": authorization_ref, "protocol": plan["protocol"]})
    E.write_new(unit / "projection.json", prepared["projection"])
    dispatched = False
    try:
        with (resource_service or parent_resource)(prepared) as live:
            live["check"]()
            _validate_authorization({**plan, "_plan_ref": plan_ref}, prepared["authorization_ref"],
                prepared["authorization"], prepared["repo"], unit, fixture=fixture_mode)
            backend = (backend_factory or _backend)(prepared, live)
            projected = {"arm": plan["identity"]["arm"], "seed": plan["identity"]["seed"],
                "checkpoint": {"path": prepared["projection"]["resolved_checkpoint"],
                               "sha256": plan["source"]["checkpoint"]["sha256"]},
                "manifest_path": prepared["projection"]["resolved_manifest"]}
            E.write_new(unit / "dispatch-intent.json", {"operation_id": plan["operation_id"],
                                                       "intent_sha256": E.J.file_sha(unit / "intent.json")})
            dispatched = True
            raw = backend.probe(prepared["stage"], projected, plan["identity"]["probe"], plan["identity"]["run_name"])
            E.write_new(prepared["stage"] / "backend_result.json", raw)
            live["check"]()
            request = _request(prepared)
            request_sha = E.digest(request)
            evidence = E.build(request, plan["roots"], expected_request_sha256=request_sha,
                expected_roots_sha256=E.digest(plan["roots"]), allow_fixture=allow_fixture)
            E.verify(evidence, plan["roots"], expected_request_sha256=request_sha,
                expected_roots_sha256=E.digest(plan["roots"]), allow_fixture=allow_fixture)
            E.write_new(unit / "request.json", request)
            E.write_new(unit / "evidence.json", evidence)
        result = {"schema": SCHEMA + "_result", "status": "complete", "operation_id": plan["operation_id"],
            "evidence": ref(unit / "evidence.json"), "request_sha256": request_sha,
            "root_map_sha256": E.digest(plan["roots"]), "approval_granted_by_unit": False,
            "lease_observations": [ref(path) for path in sorted((unit / "lease-observations").glob("*/proof.json"))]}
    except BaseException as error:  # noqa: BLE001 - 중단도 원문을 보존하고 재dispatch를 차단한다.
        result = {"schema": SCHEMA + "_result", "status": "pending_unknown" if not isinstance(error, Exception) else "HOLD",
            "operation_id": plan["operation_id"], "dispatched": dispatched,
            "error_type": type(error).__name__, "reason": str(error), "redispatch": False,
            "raw_preserved": True, "approval_granted_by_unit": False}
    E.write_new(unit / "outcome.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    authorization = None if args.authorization is None else {
        "path": str(args.authorization.absolute()), "sha256": args.authorization_sha256}
    try:
        result = run({"path": str(args.plan.absolute()), "sha256": args.plan_sha256}, authorization, execute=args.execute)
    except Exception as error:  # noqa: BLE001 - CLI는 실패를 명시 HOLD JSON으로 보존한다.
        result = {"status": "HOLD", "error_type": type(error).__name__, "reason": str(error), "dispatched": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
