"""POSIX Brev VM에서 parent probe를 시작·조회·반출하는 최소 transport helper."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_agentic_brev_probe_parent_v1 as P
import vision_brev_backend_v1 as OWNED  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]

SCHEMA = "agentic_brev_probe_transport_remote_v1"
SOURCE = "bench/agentic_brev_probe_transport_remote_v1.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v1.py"
SOURCES = P.SOURCES | {SOURCE, TEST_SOURCE}
PROVIDER_SCHEMA = P.SCHEMA + "_provider_packet"
MODES = ("prepare", "start", "status", "provider", "export")
ARMS = ("NONE", "F1")
MAX_PROVIDER_BYTES = 1024 * 1024
MAX_EXPORT_FILES = 256
MAX_EXPORT_BYTES = 64 * 1024 * 1024
API_ENV = re.compile(r"(?:^|_)(?:API|TOKEN|SECRET|KEY)(?:_|$)", re.IGNORECASE)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PRODUCTION = (P.prepare, subprocess.Popen, OWNED._process_identity)
REQUIRED_PLAN_REFS = {
    "proof", "lease_snapshot", "capture_parent", "boot", "model_observation",
    "provider_packet_binding", "admission_times",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _read_json(raw: bytes, anchor: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{anchor} JSON이 잘못됐다") from error
    require(isinstance(value, dict), f"{anchor} JSON object가 필요하다")
    return value


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def _single(path: Path, *, must_exist: bool = True) -> Path:
    path = path.absolute()
    require(not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents),
            "symlink path forbidden")
    if must_exist:
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1,
                "single regular file required")
    return path


def _new_bytes(path: Path, raw: bytes) -> bool:
    path = _single(path, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    except FileExistsError:
        require(_single(path).read_bytes() == raw, "write-once file differs")
        return False


def _ref(path: Path) -> dict[str, str]:
    path = _single(path)
    return {"path": str(path), "sha256": sha_bytes(path.read_bytes())}


def _error_receipt(path: Path, receipt: Mapping[str, Any]) -> bool:
    """원래 예외를 가리지 않으면서 별도 write-once 실패 영수증을 남긴다."""
    try:
        return _new_bytes(path, canonical(receipt))
    except Exception:  # noqa: BLE001 - 원래 export 실패가 우선이며 기존 영수증은 보존한다.
        return False


def _relative(value: str) -> Path:
    require(isinstance(value, str) and bool(value) and "\\" not in value, "canonical relative path")
    logical = PurePosixPath(value)
    require(not logical.is_absolute() and all(part not in {"", ".", ".."} for part in logical.parts)
            and "bench/private" not in value, "public repo-relative transport_root")
    return Path(*logical.parts)


def _transport(prepared: Mapping[str, Any], request_sha: str) -> tuple[Path, dict[str, str]]:
    registration = prepared["protocol"].get("transport_root")
    require(isinstance(registration, dict) and registration.get("base") == "repo"
            and set(registration) == {"base", "relative"}, "registered transport_root required")
    relative = _relative(registration["relative"])
    root = Path(prepared["repo"]).resolve() / relative
    require(root.resolve(strict=False).is_relative_to(Path(prepared["repo"]).resolve()),
            "transport_root outside repo")
    return root / request_sha, dict(registration)


def _resolve_future(resolver: Any, logical_value: str) -> Path:
    logical = P.E._logical(logical_value)
    matches = [(remote, local) for remote, local in resolver.rows
               if remote == logical or remote in logical.parents]
    require(len(matches) == 1, "path outside registered root-map")
    remote, local = matches[0]
    result = local.joinpath(*logical.relative_to(remote).parts)
    require(result.resolve(strict=False).is_relative_to(local.resolve()), "resolved path escaped root")
    return result


def _prepare(request: Path, request_sha256: str, *, allow_fixture: bool,
             home: Path | None, now: datetime | None, preparer: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    require(HEX64.fullmatch(str(request_sha256)) is not None, "request raw SHA")
    request = _single(request)
    require(sha_bytes(request.read_bytes()) == request_sha256, "request raw SHA changed")
    if not allow_fixture:
        require(preparer is PRODUCTION[0] and home is None and now is None,
                "actual transport dependency injection forbidden")
        require(os.name == "posix", "actual transport requires POSIX")
    prepared = preparer({"path": str(request), "sha256": request_sha256},
                        allow_fixture=allow_fixture, home=home, now=now)
    require(prepared.get("status") == "prepared", "parent request is not prepared")
    registered = prepared["protocol"].get("sources_lf", {})
    require(SOURCES <= registered.keys()
            and registered.get(SOURCE) == P.C.digest_lf_file(ROOT / SOURCE)
            and registered.get(TEST_SOURCE) == P.C.digest_lf_file(ROOT / TEST_SOURCE),
            "remote transport source/test closure is not registered")
    folder, registration = _transport(prepared, request_sha256)
    return {**prepared, "transport_folder": folder, "transport_registration": registration}


def _public_prepare(prepared: Mapping[str, Any], request: Path, request_sha: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": "prepared", "mode": "prepare",
            "request": {"path": str(request), "sha256": request_sha},
            "transport": prepared["transport_registration"],
            "operations": {arm: prepared["operations"][arm]["op"]["operation_id"] for arm in ARMS},
            "pair": prepared["execution"]["pair_id"], "authority_issued": False}


def _launch_status(folder: Path) -> dict[str, Any] | None:
    intent = folder / "launch-intent.json"
    if not intent.exists():
        claim = folder / "launch-claim.json"
        if claim.exists():
            return {"schema": SCHEMA, "status": "unknown", "mode": "start", "spawned": False,
                    "claim": _ref(claim), "reason": "launch_claim_without_owned_identity",
                    "stop_required_if_unreleased": True, "authority_issued": False}
        return None
    value = _read_json(_single(intent).read_bytes(), "launch intent")
    identity = value["identity"]
    current = OWNED._identity_is_current(identity)
    return {"schema": SCHEMA, "status": "running" if current else "unknown",
            "mode": "start", "spawned": False, "launch": _ref(intent),
            "identity": identity, "owned_current": current,
            "stop_required_if_unreleased": not current, "authority_issued": False}


def _start(prepared: Mapping[str, Any], request: Path, request_sha: str,
           popen_factory: Callable[..., Any], identity_fn: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
    folder = Path(prepared["transport_folder"])
    existing = _launch_status(folder)
    if existing is not None:
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    claim_path = folder / "launch-claim.json"
    claim = {"schema": SCHEMA + "_launch_claim", "request_sha256": request_sha,
             "cleanup_policy": P.UNVERIFIED_CHILD_POLICY}
    try:
        created = _new_bytes(claim_path, canonical(claim))
    except ValueError:
        existing = _launch_status(folder)
        require(existing is not None, "launch claim conflict")
        assert existing is not None
        return existing
    if not created:
        existing = _launch_status(folder)
        require(existing is not None, "launch claim without status")
        assert existing is not None
        return existing
    stdout_path, stderr_path = folder / "parent.stdout", folder / "parent.stderr"
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    argv = [str(prepared["python"]), "-B", str(Path(P.__file__).resolve()), "--request", str(request),
            "--request-sha256", request_sha, "--execute"]
    env = {key: value for key, value in os.environ.items() if API_ENV.search(key) is None}
    try:
        process = popen_factory(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                env=env, start_new_session=True, close_fds=True)
        identity = identity_fn(process.pid)
        require(identity.get("posix_group_verified") is True and identity.get("pid") == process.pid,
                "owned POSIX child identity required")
        body = {"schema": SCHEMA + "_launch_intent", "argv": argv, "request_sha256": request_sha,
                "identity": identity, "stdout": str(stdout_path), "stderr": str(stderr_path),
                "api_environment_removed": True, "start_new_session": True,
                "cleanup_policy": P.UNVERIFIED_CHILD_POLICY}
        require(_new_bytes(folder / "launch-intent.json", canonical(body)), "launch intent already exists")
    finally:
        stdout.close()
        stderr.close()
    return {"schema": SCHEMA, "status": "running", "mode": "start", "spawned": True,
            "launch": _ref(folder / "launch-intent.json"), "identity": identity,
            "owned_current": True, "authority_issued": False}


def _provider(prepared: Mapping[str, Any], arm: str, payload_base64: str,
              current: datetime) -> dict[str, Any]:
    require(arm in ARMS, "provider arm required")
    try:
        raw = base64.b64decode(payload_base64, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("provider envelope base64") from error
    require(len(raw) <= MAX_PROVIDER_BYTES, "provider envelope too large")
    envelope = _read_json(raw, "provider envelope")
    require(set(envelope) == {"packet", "inventory_base64"} and isinstance(envelope["packet"], dict),
            "provider envelope fields")
    try:
        inventory_raw = base64.b64decode(envelope["inventory_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("provider inventory base64") from error
    require(len(inventory_raw) <= MAX_PROVIDER_BYTES, "provider inventory too large")
    op = prepared["operations"][arm]["op"]
    packet_path = _resolve_future(prepared["resolver"], op["provider_packet_path"])
    inventory_path = Path(str(packet_path) + ".inventory.json")
    inventory_logical = op["provider_packet_path"] + ".inventory.json"
    packet = envelope["packet"]
    require(packet.get("schema") == PROVIDER_SCHEMA and packet.get("operation_id") == op["operation_id"]
            and set(packet) == {"schema", "operation_id", "observed_at_utc", "inventory"},
            "provider packet schema/operation")
    require(packet["inventory"] == {"path": inventory_logical, "sha256": sha_bytes(inventory_raw)},
            "provider inventory raw reference")
    observed = P.E._stamp(packet["observed_at_utc"])
    age = current.astimezone(timezone.utc) - observed.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10
    require(0 <= age.total_seconds() <= prepared["execution"]["windows"]["admission_max_age_s"],
            "stale provider packet")
    inventory = _read_json(inventory_raw, "provider inventory")
    require(set(inventory) == {"workspaces"} and isinstance(inventory["workspaces"], list),
            "provider inventory schema")
    target = prepared["protocol"]["provider_target"]
    matches = [row for row in inventory["workspaces"] if isinstance(row, dict) and row.get("id") == target["resource_id"]]
    require(len(matches) == 1 and matches[0].get("status") == "RUNNING"
            and matches[0].get("name") == target["resource_name"]
            and matches[0].get("instance_type") == target["instance_type"],
            "registered Brev target is not the unique RUNNING row")
    require(set(matches[0]) == REMOTE.PROVIDER_ROW_KEYS, "provider workspace schema")
    inventory_created = _new_bytes(inventory_path, inventory_raw)
    try:
        packet_created = _new_bytes(packet_path, canonical(packet))
    except Exception:
        if inventory_created:
            # inventory remains as an immutable partial receipt; retry must provide identical bytes.
            pass
        raise
    return {"schema": SCHEMA, "status": "stored" if packet_created else "existing", "mode": "provider",
            "arm": arm, "operation_id": op["operation_id"], "packet": _ref(packet_path),
            "inventory": _ref(inventory_path), "resource_grant_issued": False, "authority_issued": False}


def _status(prepared: Mapping[str, Any], arm: str | None) -> dict[str, Any]:
    require(arm is None or arm in ARMS, "status arm")
    selected = ARMS if arm is None else (arm,)
    operations: dict[str, Any] = {}
    for name in selected:
        op = prepared["operations"][name]["op"]
        path = prepared["operations"][name]["unit"] / "outcome.json"
        if not path.exists():
            operations[name] = {"status": "pending", "operation_id": op["operation_id"]}
            continue
        outcome = P.U.read_ref(_ref(path))
        if outcome.get("status") == "complete":
            P._verify_completion(prepared, outcome)
            state = "completed"
        else:
            state = str(outcome.get("status", "unknown")).lower()
        operations[name] = {"status": state, "operation_id": op["operation_id"], "result": _ref(path)}
    pair_path = prepared["folder"] / "outcome.json"
    pair = {"status": "pending"}
    if pair_path.exists():
        pair = {"status": str(P.U.read_ref(_ref(pair_path)).get("status", "unknown")).lower(),
                "result": _ref(pair_path)}
    if arm is not None:
        selected_status = operations[arm]
        return {"schema": SCHEMA, "status": selected_status["status"], "mode": "status", "arm": arm,
                "operation_id": selected_status["operation_id"],
                **({"result": selected_status["result"]} if "result" in selected_status else {}),
                "pair": pair, "authority_issued": False}
    return {"schema": SCHEMA, "status": "observed", "mode": "status",
            "operations": operations, "pair": pair, "authority_issued": False}


def _root_rows(plan: Mapping[str, Any], prepared: Mapping[str, Any], arm: str) -> list[tuple[int, PurePosixPath, Path]]:
    rows: list[tuple[int, PurePosixPath, Path]] = []
    whitelist = prepared["execution"]["runtime_host_binding"]["evidence_roots"][arm]
    allowed = [P.E._logical(row["remote"]) for row in whitelist]
    for row in plan["roots"]:
        remote, local = P.E._logical(row["remote"]), Path(row["local"])
        require(local.is_absolute(), "producer native root must be absolute")
        matches = [index for index, root in enumerate(allowed) if root == remote]
        require(len(matches) == 1, "producer root outside registered host evidence roots")
        rows.append((matches[0], remote, local))
    return rows


def _locate(reference: Mapping[str, Any], rows: list[tuple[int, PurePosixPath, Path]]) -> tuple[int, str, Path, str]:
    require(set(reference) == {"path", "sha256"} and HEX64.fullmatch(str(reference["sha256"])) is not None,
            "raw reference fields")
    value = str(reference["path"])
    native = Path(value)
    matches: list[tuple[int, PurePosixPath, Path, Path]] = []
    if value.startswith("/") and "\\" not in value:
        candidate = P.E._logical(value)
        for index, remote, local in rows:
            if remote == candidate or remote in candidate.parents:
                matches.append((index, remote, local, local.joinpath(*candidate.relative_to(remote).parts)))
        require(len(matches) <= 1, "reference ambiguous in logical producer roots")
    if not matches and native.is_absolute():
        for index, remote, local in rows:
            try:
                native.relative_to(local)
            except ValueError:
                continue
            matches.append((index, remote, local, native))
    unique = {(index, path): (index, remote, local, path) for index, remote, local, path in matches}
    require(len(unique) == 1, "reference outside or ambiguous in producer roots")
    index, remote, local, path = next(iter(unique.values()))
    path = _single(path)
    require(path.resolve().is_relative_to(local.resolve()), "reference escaped producer root")
    relative = path.relative_to(local).as_posix()
    logical = remote.joinpath(*PurePosixPath(relative).parts)
    require(sha_bytes(path.read_bytes()) == reference["sha256"], "raw reference SHA changed")
    return index, relative, path, str(logical)


def _refs(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"} and isinstance(value.get("path"), str):
            found.append({"path": value["path"], "sha256": value["sha256"]})
        else:
            for item in value.values():
                found.extend(_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_refs(item))
    return found


def _export(prepared: Mapping[str, Any], arm: str, output: Path,
            before_add: Callable[[Path], None] | None = None) -> dict[str, Any]:
    require(arm in ARMS, "export arm required")
    require(output.is_absolute() and output.name.endswith(".tar.gz") and output.parent.is_dir()
            and not any(path.is_symlink() for path in (output.parent, *output.parent.parents))
            and not output.exists(), "new absolute export path required")
    op = prepared["operations"][arm]["op"]
    outcome_path = prepared["operations"][arm]["unit"] / "outcome.json"
    outcome_ref = _ref(outcome_path)
    outcome = P.U.read_ref(outcome_ref)
    require(outcome.get("status") == "complete" and outcome.get("operation_id") == op["operation_id"],
            "completed arm required")
    P._verify_completion(prepared, outcome)
    plan = P.U.read_ref(outcome["plan"])
    evidence = P.U.read_ref(outcome["evidence"])
    require(REQUIRED_PLAN_REFS <= plan.keys(), "required parent metadata refs missing")
    rows = _root_rows(plan, prepared, arm)
    queue = [outcome_ref, outcome["plan"], outcome["evidence"], *[plan[key] for key in REQUIRED_PLAN_REFS]]
    raw_references = [{"path": logical, "sha256": raw_sha}
                      for logical, raw_sha in evidence["raw_files_sha256"].items()]
    queue.extend(raw_references)
    files: dict[tuple[int, str], dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    total = 0
    while queue:
        reference = queue.pop(0)
        try:
            index, relative, path, logical = _locate(reference, rows)
        except ValueError:
            required = reference in [plan[key] for key in REQUIRED_PLAN_REFS] or reference in {
                "outcome": outcome_ref, "plan": outcome["plan"], "evidence": outcome["evidence"]}.values() \
                or reference in raw_references
            if required:
                raise
            skipped.append({"path": str(reference.get("path")), "sha256": str(reference.get("sha256")),
                            "reason": "outside_registered_evidence_roots"})
            continue
        key = (index, relative)
        if key in files:
            require(files[key]["sha256"] == reference["sha256"], "same file has conflicting SHA refs")
            continue
        raw = path.read_bytes()
        total += len(raw)
        require(len(files) + 1 <= MAX_EXPORT_FILES and total <= MAX_EXPORT_BYTES, "export cap exceeded")
        archive_path = f"files/{index}/{relative}"
        parts = PurePosixPath(archive_path).parts
        require(not PurePosixPath(archive_path).is_absolute() and ".." not in parts, "unsafe archive path")
        row = {"root_index": index, "relative": relative, "logical_path": logical,
               "sha256": sha_bytes(raw), "bytes": len(raw), "archive_path": archive_path, "_path": path}
        files[key] = row
        if path.suffix == ".json":
            body = _read_json(raw, "referenced metadata")
            queue.extend(_refs(body))
    ordered = [files[key] for key in sorted(files)]
    manifest_files = [{key: value for key, value in row.items() if key != "_path"} for row in ordered]
    manifest = {"schema": SCHEMA + "_archive_v1", "arm": arm, "operation_id": op["operation_id"],
                "outcome": outcome_ref, "producer_roots_sha256": P.E.digest(plan["roots"]),
                "files": manifest_files, "skipped_references": skipped,
                "authority_issued": False, "evidence_accepted": False}
    if before_add is not None:
        for row in ordered:
            before_add(row["_path"])
    snapshots = {row["_path"]: row["sha256"] for row in ordered}
    try:
        with output.open("xb") as raw_stream, tarfile.open(
                fileobj=raw_stream, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            manifest_raw = canonical(manifest)
            info = tarfile.TarInfo("manifest.json")
            info.size, info.mode, info.mtime = len(manifest_raw), 0o600, 0
            archive.addfile(info, io.BytesIO(manifest_raw))
            for row in ordered:
                raw = _single(row["_path"]).read_bytes()
                require(sha_bytes(raw) == row["sha256"], "raw changed during export")
                info = tarfile.TarInfo(row["archive_path"])
                info.size, info.mode, info.mtime = len(raw), 0o600, 0
                archive.addfile(info, io.BytesIO(raw))
        for path, expected in snapshots.items():
            require(sha_bytes(_single(path).read_bytes()) == expected, "raw changed after export")
    except Exception as error:
        receipt = {"schema": SCHEMA + "_export_error_v1", "arm": arm,
                   "operation_id": op["operation_id"], "output": str(output),
                   "error_type": type(error).__name__, "reason": str(error),
                   "partial_output_preserved": output.exists(), "authority_issued": False,
                   "evidence_accepted": False}
        _error_receipt(Path(str(output) + ".error.json"), receipt)
        raise
    return {"schema": SCHEMA, "status": "exported", "mode": "export", "arm": arm,
            "operation_id": op["operation_id"], "archive": _ref(output),
            "manifest_sha256": sha_bytes(canonical(manifest)), "files": len(ordered), "bytes": total,
            "authority_issued": False, "evidence_accepted": False}


def run_transport(*, request: Path, request_sha256: str, mode: str = "prepare", arm: str | None = None,
                  payload_base64: str | None = None, output: Path | None = None,
                  allow_fixture: bool = False, home: Path | None = None, now: datetime | None = None,
                  preparer: Callable[..., dict[str, Any]] = PRODUCTION[0],
                  popen_factory: Callable[..., Any] = PRODUCTION[1],
                  identity_fn: Callable[[int], dict[str, Any]] = PRODUCTION[2],
                  before_add: Callable[[Path], None] | None = None) -> dict[str, Any]:
    require(mode in MODES and type(allow_fixture) is bool, "transport mode/fixture flag")
    if not allow_fixture:
        require(preparer is PRODUCTION[0] and popen_factory is PRODUCTION[1]
                and identity_fn is PRODUCTION[2] and home is None and now is None and before_add is None,
                "actual transport dependency injection forbidden")
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    prepared = _prepare(request, request_sha256, allow_fixture=allow_fixture, home=home, now=now,
                        preparer=preparer)
    if mode == "prepare":
        return _public_prepare(prepared, request, request_sha256)
    if mode == "start":
        require(arm is None and payload_base64 is None and output is None, "start arguments")
        return _start(prepared, request, request_sha256, popen_factory, identity_fn)
    if mode == "status":
        require(payload_base64 is None and output is None, "status arguments")
        return _status(prepared, arm)
    if mode == "provider":
        require(payload_base64 is not None and output is None, "provider arguments")
        assert payload_base64 is not None
        return _provider(prepared, str(arm), payload_base64, current)
    require(output is not None and payload_base64 is None, "export arguments")
    assert output is not None
    return _export(prepared, str(arm), output, before_add)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, default="prepare")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--payload-base64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_transport(request=args.request, request_sha256=args.request_sha256, mode=args.mode,
                               arm=args.arm, payload_base64=args.payload_base64, output=args.output)
    except Exception as error:  # noqa: BLE001 - CLI는 실패를 구조화해 상위 broker에 전달한다.
        result = {"schema": SCHEMA, "status": "HOLD", "dispatched": False,
                  "error_type": type(error).__name__, "reason": str(error), "authority_issued": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
