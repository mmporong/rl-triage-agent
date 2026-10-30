"""Remote10/Parent10/Service11 실행 계층을 GPU Start 전에 CPU에서 읽기 전용으로 점검한다.

판정은 생산 validator·Resolver·worker 검증이 내린다. 이 모듈은 등록 입력을 그 함수들에 같은 순서로
넘기고 첫 실패 단계를 보고할 뿐이며, 디렉터리를 만들거나 실행·GPU Start 권한을 발급하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / name).is_dir() for name in ("bench", "evals", "src")):
            return candidate
    raise RuntimeError("repository root unavailable")


ROOT = _repo_root()
for location in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import agentic_brev_mailbox_service_v6 as SERVICE6  # pyright: ignore[reportMissingImports]
import agentic_brev_mailbox_service_v9 as SERVICE9  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v10 as REMOTE  # pyright: ignore[reportMissingImports]
import brev_recovery_worker_v1 as WORKER  # pyright: ignore[reportMissingImports]
import run_agentic_brev_probe_parent_v10 as PARENT  # pyright: ignore[reportMissingImports]

V1 = PARENT.V1
E, U, OWNED = V1.E, V1.U, V1.U.OWNED
SCHEMA = "agentic_pipeline_preflight_v1"
SOURCE = "evals/agentic_pipeline_preflight_v1.py"
TEST_SOURCE = "tests/test_agentic_pipeline_preflight_v1.py"
STAGES = ("registration", "root_map", "prepare_layout", "backend_args", "lifecycle", "budget")
NEXT = {
    "registration": "Remote10 closure를 포함한 새 Parent 판을 등록하고 같은 anchor의 blob 일치를 다시 확인한다",
    "root_map": "operation별 gpu-observations root를 registration root 아래에 정확히 한 번 등록한다",
    "prepare_layout": "준비 시점 root는 미리 두고, 관측 root는 독점 unit 밖에 두며, 새 pair/operation unit은 비워 둔다",
    "backend_args": "source local을 repo·lab·unit 안의 등록 배치로 맞추고 backend protocol을 repo 안에서 해석한다",
    "lifecycle": "IO contract의 lifecycle_protocol을 anchor blob이 있는 저장소 상대 경로와 raw SHA로 결합한다",
    "budget": "lifecycle 기산점·기한·Stop reserve와 contract 기한을 같은 판으로 맞춘다",
}


class Unsupported(Exception):
    """입력이 없거나 모르는 판이라 판정하지 않은 단계."""


class _JobCaptured(Exception):
    def __init__(self, job: dict[str, Any]) -> None:
        super().__init__("probe job captured before child launch")
        self.job = job


def _binding(ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if ctx.get("binding") is None:
        raise Unsupported("parent request/binding reference required")
    return ctx["protocol"], ctx["binding"]


def layout(protocol: dict[str, Any], binding: dict[str, Any], home: Path | None) -> dict[str, Any]:
    """Parent6 prepare와 Parent1 run이 쓰는 HOME 배치·독점 claim 경로를 계산한다. 파일은 만들지 않는다."""
    base = (home or Path.home()).absolute()
    places = {key: base.joinpath(*U._relative(value)).absolute() for key, value in protocol["layout"].items()}
    V1.require(set(places) == {"repo", "isaaclab", "python"}, "HOME layout fields")
    execution = binding["expected_request"]["execution"]
    root = places["repo"].joinpath(*U._relative(protocol["execution_root"]))
    units = {arm: root / "operations" / execution["operations"][arm]["operation_id"] for arm in V1.ORDER}
    return {"repo": places["repo"], "lab": places["isaaclab"], "python": places["python"],
            "folder": root / "pairs" / execution["pair_id"], "units": units,
            "stages": {arm: unit / "raw" for arm, unit in units.items()}}


def _registration(ctx: dict[str, Any]) -> dict[str, Any]:
    request = None
    if ctx["request_ref"] is not None:
        request = U.read_ref(ctx["request_ref"])
        if request.get("schema") != PARENT.SCHEMA + "_request":
            raise Unsupported("unknown parent request schema: " + str(request.get("schema")))
        if request.get("action") != "probe":
            raise Unsupported("repair_unsupported")
        protocol_ref, anchor = request["protocol"], request["source_anchor"]
        binding = U.read_ref(request["binding"])
    elif ctx["protocol_ref"] is not None:
        protocol_ref, anchor, binding = ctx["protocol_ref"], ctx["source_anchor"], None
    else:
        raise Unsupported("parent request or protocol reference required")
    protocol = U.read_ref(protocol_ref)
    if protocol.get("schema") != PARENT.SCHEMA + "_protocol":
        raise Unsupported("unknown parent protocol schema: " + str(protocol.get("schema")))
    fixture = binding is not None and binding.get("evidence_kind") == E.FIXTURE
    V1.require(binding is None or (binding.get("schema") == PARENT.SCHEMA + "_binding"
               and binding.get("evidence_kind") in {E.FIXTURE, E.ACTUAL}
               and (not fixture or ctx["allow_fixture"])), "fixture mode must be explicit")
    V1._public(protocol)
    V1.require(protocol.get("registered_before_execution") is True
               and (request is None or protocol.get("binding_sha256") == request["binding"]["sha256"]),
               "registered protocol/binding SHA")
    # Remote8 _prepare가 Remote10 판에서 요구하는 closure다. 빠지면 원격 prepare가 GPU 입장 뒤에 거부한다.
    missing = sorted(set(REMOTE.SOURCES) - protocol.get("sources_lf", {}).keys())
    V1.require(not missing, f"registered source closure lacks {len(missing)} Remote10 entry sources: "
               + ", ".join(missing[:3]))
    if anchor is None:
        # protocol의 source_commit은 protocol 파일보다 앞선 커밋이다. 등록 커밋 anchor는 request가 준다.
        raise Unsupported("registration source anchor required with protocol-only input")
    protocol_path = Path(protocol_ref["path"])
    inside = ctx["repo"].resolve() in protocol_path.resolve().parents
    V1.require(fixture or inside, "registered protocol outside source repository")
    U.verify_sources(protocol, anchor, repo=ctx["repo"], protocol_path=protocol_path if inside else None)
    ctx.update(protocol=protocol, binding=binding, fixture=fixture)
    return {"source_anchor": anchor, "sources_lf": len(protocol["sources_lf"]), "protocol_blob": inside}


def _root_map(ctx: dict[str, Any]) -> dict[str, Any]:
    protocol, binding = _binding(ctx)
    ctx["observations"] = PARENT.registered_observation_roots(protocol, binding, home=ctx["home"])
    return {"binding_roots": len(binding["roots"]), "observation_roots": len(ctx["observations"])}


def _prepare_layout(ctx: dict[str, Any]) -> dict[str, Any]:
    protocol, binding = _binding(ctx)
    places = layout(protocol, binding, ctx["home"])
    V1.require(places["repo"].is_dir() and places["lab"].is_dir() and places["python"].is_file(),
               "registered repo/lab/Python unavailable")
    V1.require(ctx["fixture"] or places["repo"].resolve() == V1.ROOT.resolve(),
               "actual code must be registered repository")
    observed = {value["remote"] for value in ctx["observations"].values()}
    present = [row for row in binding["roots"] if row["remote"] not in observed]
    operations = binding["expected_request"]["execution"]["operations"]
    sources = {arm: operations[arm]["source_roots"] for arm in V1.ORDER}
    claims = [places["folder"], *places["units"].values()]
    # Parent10은 prepare 전에 관측 root를 만들고 Parent1 run은 그 뒤 pair/unit을 exist_ok=False로 claim한다.
    early = [Path(row["local"]) for row in present] + [Path(value["local"]) for value in ctx["observations"].values()]
    early += [Path(row["local"]) for rows in sources.values() for row in rows]
    for path in early:
        V1.require(not any(path == claim or claim in path.parents for claim in claims),
                   "prepare-time root inside exclusive claim unit: " + str(path))
    V1.require(not any(claim.exists() for claim in claims),
               "exclusive pair/operation unit already exists; exist_ok=False claim would fail")
    for value in ctx["observations"].values():
        local = Path(value["local"])
        V1.require(not local.exists() or (local.is_dir() and not any(local.iterdir())),
                   "stale GPU observation root before pair start")
    # 준비 시점 Resolver는 실제 local root가 지금 있어야 한다. 미래 stage·unit은 여기에 넣지 않는다.
    ctx["resolver"] = E.Resolver(present, E.digest(present))
    ctx["source_resolvers"] = {arm: E.Resolver(rows, E.digest(rows)) for arm, rows in sources.items()}
    ctx["layout"] = places
    return {"prepare_roots": len(present) + sum(len(rows) for rows in sources.values()),
            "created_by_parent10": len(observed), "claim_targets": len(claims)}


def probe_job(protocol: dict[str, Any], anchor: str, places: dict[str, Any], arm: str, operation: dict[str, Any],
              checkpoint: Path, manifest: Path) -> dict[str, Any]:
    """등록 SafeRecoveryBackend.probe가 child 실행 직전에 쓰는 job을 stage 생성·실행 없이 얻는다."""
    backend = object.__new__(OWNED.SafeRecoveryBackend)
    backend.lab, backend.python = places["lab"].resolve(), places["python"].resolve()
    backend.protocol, backend.anchor = protocol, anchor

    def capture(_stage: Path, _mode: str, job: dict[str, Any]):
        raise _JobCaptured(job)

    backend._execute = capture
    # Parent1 _dispatch가 넘기는 artifact 투영과 같다.
    artifact = {"arm": arm, "seed": operation["seed"], "manifest_path": str(manifest), "checkpoint": {
        "path": str(checkpoint), "sha256": operation["source"]["checkpoint"]["sha256"]}}
    try:
        backend.probe(places["stages"][arm], artifact, operation["probe"], operation["run_name"])
    except _JobCaptured as captured:
        return captured.job
    raise RuntimeError("backend probe returned without building a job")


def _backend_args(ctx: dict[str, Any]) -> dict[str, Any]:
    _, binding = _binding(ctx)
    if "resolver" not in ctx:
        raise Unsupported("prepare-time resolver unavailable")
    execution = binding["expected_request"]["execution"]
    reference = execution["backend"]["protocol"]
    path = ctx["resolver"].file(reference)
    # Parent1 _dispatch가 SafeRecoveryBackend에 넘기는 protocol 사본과 같은 결합이다.
    protocol = OWNED.B.validate_protocol({**ctx["resolver"].read(reference), "protocol_path": str(path),
                                          "protocol_sha256_lf": V1.C.digest_lf_file(path)})
    checked = []
    for arm in V1.ORDER:
        operation = execution["operations"][arm]
        source = ctx["source_resolvers"][arm]
        checkpoint, manifest = source.file(operation["source"]["checkpoint"]), source.file(operation["source"]["manifest"])
        WORKER.validate_job(probe_job(protocol, execution["backend"]["source_anchor"], ctx["layout"], arm, operation,
                                      checkpoint, manifest))
        checked.append(arm)
    return {"validated_jobs": checked, "worker": "bench/brev_recovery_worker_v1.py"}


def _lifecycle(ctx: dict[str, Any]) -> dict[str, Any]:
    if ctx["contract_ref"] is None:
        raise Unsupported("IO transport contract reference required")
    if ctx["runtime_anchor"] is None:
        raise Unsupported("runtime source anchor required")
    contract = U.read_ref(ctx["contract_ref"])
    reference = contract.get("lifecycle_protocol")
    if not isinstance(reference, dict):
        raise TypeError("registered lifecycle reference")
    runtime = SimpleNamespace(repo=ctx["repo"], anchor=ctx["runtime_anchor"])
    native = SERVICE6._native_lifecycle_reference(runtime, reference)
    ctx.update(contract=contract, phase=SERVICE9.V8.T.read_ref(native))
    return {"lifecycle": reference["path"], "blob_anchor": ctx["runtime_anchor"]}


def _budget(ctx: dict[str, Any]) -> dict[str, Any]:
    if "phase" not in ctx:
        raise Unsupported("verified lifecycle phase required")
    seconds = ctx["work_seconds"]
    if seconds is None:
        binding = ctx.get("binding")
        seconds = binding["expected_request"]["budget_gpu_s"] if binding is not None else 0
    SERVICE9._validate_budget_origin(ctx["phase"], ctx["contract"], seconds=seconds, now=ctx["now"])
    return {"work_seconds": seconds, "stop_reserve_s": ctx["phase"]["stop_reserve_s"],
            "deadline_utc": ctx["phase"]["deadline_utc"]}


STEPS = dict(zip(STAGES, (_registration, _root_map, _prepare_layout, _backend_args, _lifecycle, _budget),
                 strict=True))


def check(*, request_ref: dict | None = None, protocol_ref: dict | None = None, contract_ref: dict | None = None,
          source_anchor: str | None = None, runtime_anchor: str | None = None, home: Path | None = None,
          repo: Path = ROOT, work_seconds: int | None = None, allow_fixture: bool = False,
          now: datetime | None = None) -> dict:
    ctx: dict[str, Any] = {"request_ref": request_ref, "protocol_ref": protocol_ref, "contract_ref": contract_ref,
        "source_anchor": source_anchor, "runtime_anchor": runtime_anchor, "home": home, "repo": Path(repo),
        "work_seconds": work_seconds,
        "allow_fixture": allow_fixture, "fixture": False,
        "now": now or datetime.now(timezone.utc)}  # noqa: UP017 - Python 3.10
    scope: dict[str, str] = {}
    checked: dict[str, Any] = {}
    unsupported: dict[str, str] = {}
    failure: dict[str, str] | None = None
    for name, step in STEPS.items():
        if failure is not None:
            scope[name] = "not_run"
            continue
        try:
            checked[name] = step(ctx)
            scope[name] = "verified"
        except Unsupported as reason:
            scope[name], unsupported[name] = "unsupported", str(reason)
        except Exception as error:  # noqa: BLE001 - 생산 검증의 거부를 실패 단계로 그대로 보고한다.
            scope[name] = "failed"
            failure = {"failed_stage": name, "error_type": type(error).__name__, "reason": str(error)}
    status = "FAIL" if failure else "PASS" if not unsupported else "PARTIAL" if checked else "UNSUPPORTED"
    if failure:
        follow = NEXT[failure["failed_stage"]]
    elif unsupported:
        follow = "미판정 단계의 입력을 제공한다: " + ", ".join(unsupported)
    else:
        follow = "원래 PC가 같은 입력으로 POSIX Remote10 prepare를 실행해 실제 소비를 확인한다"
    binding = ctx.get("binding")
    kind = binding.get("evidence_kind") if binding is not None else None
    return {"schema": SCHEMA, "status": status, **(failure or {"failed_stage": None}),
        "inputs": {name: reference["path"] for name, reference in (("request", request_ref),
                   ("protocol", protocol_ref), ("contract", contract_ref)) if reference is not None},
        "scope": scope, "checked": checked, "unsupported": unsupported, "evidence_kind": kind,
        "fixture_scope": "synthetic_cpu" if kind == E.FIXTURE else None, "next": follow,
        "execution_permitted": False, "gpu_start_authorized": False}


def _reference(path: Path | None, sha256: str | None, name: str) -> dict | None:
    if path is None:
        return None
    if sha256 is None:
        raise ValueError(name + " SHA required")
    return {"path": str(path.absolute()), "sha256": sha256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--source-anchor")
    parser.add_argument("--runtime-anchor")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--work-seconds", type=int)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check(request_ref=_reference(args.request, args.request_sha256, "request"),
                       protocol_ref=_reference(args.protocol, args.protocol_sha256, "protocol"),
                       contract_ref=_reference(args.contract, args.contract_sha256, "contract"),
                       source_anchor=args.source_anchor, runtime_anchor=args.runtime_anchor,
                       home=args.home, repo=args.repo,
                       work_seconds=args.work_seconds, allow_fixture=args.allow_fixture)
    except Exception as error:  # noqa: BLE001 - CLI 입력 오류도 실패 JSON으로 표시한다.
        result = {"schema": SCHEMA, "status": "FAIL", "failed_stage": "input", "error_type": type(error).__name__,
                  "reason": str(error), "execution_permitted": False, "gpu_start_authorized": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return {"PASS": 0, "FAIL": 1}.get(result["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
