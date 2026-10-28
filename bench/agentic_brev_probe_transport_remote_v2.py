"""등록 evidence/metadata layer를 보존해 parent 원문을 반출한다. 실행 권한은 만들지 않는다."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import agentic_brev_probe_transport_remote_v1 as V1

P, ROOT = V1.P, V1.ROOT
SCHEMA = "agentic_brev_probe_transport_remote_v2"
ARCHIVE_SCHEMA = SCHEMA + "_archive_v2"
SOURCE = "bench/agentic_brev_probe_transport_remote_v2.py"
TEST_SOURCE = "tests/test_agentic_brev_probe_transport_remote_v2.py"
SOURCES = V1.SOURCES | {SOURCE, TEST_SOURCE,
    "evals/agentic_brev_mailbox_service_v2.py", "tests/test_agentic_brev_mailbox_service_v2.py",
    "evals/agentic_brev_mailbox_service_v1.py", "tests/test_agentic_brev_mailbox_service_v1.py",
    "evals/agentic_brev_mailbox_transport_v1.py", "tests/test_agentic_brev_mailbox_transport_v1.py"}
PRODUCTION, MODES, ARMS = V1.PRODUCTION, V1.MODES, V1.ARMS
require, canonical, sha_bytes = V1.require, V1.canonical, V1.sha_bytes
LAYERS = ("evidence", "metadata")


class OutsideRoots(ValueError):
    pass


def producer_path(value):
    require(isinstance(value, str) and bool(value), "producer path string")
    if PureWindowsPath(value).drive:
        path = PureWindowsPath(value)
        require(path.is_absolute() and ".." not in path.parts, "absolute producer metadata path")
        return path
    return P.E._logical(value)


def validate_maps(evidence_roots, metadata_roots, *, fixture=False, host_native=False):
    for layer, rows in (("evidence", evidence_roots), ("metadata", metadata_roots)):
        require(isinstance(rows, list) and bool(rows), "registered ordered root maps required")
        seen = []
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"remote", "local"}, "registered root fields")
            remote = producer_path(row["remote"])
            require((layer == "metadata" and fixture) or isinstance(remote, PurePosixPath), "actual roots preserve POSIX refs")
            # producer에서는 Windows host 위치를 POSIX Path로 stat/resolve하지 않는다.
            producer_path(row["local"])
            if host_native:
                local = Path(row["local"])
                require(local.is_absolute() and ".." not in local.parts and local == local.resolve(strict=False)
                        and not any(path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400
                                    for path in (local, *local.parents) if path.exists() or path.is_symlink()),
                        "canonical link-free native host root required")
            require(all(type(remote) is not type(old) or
                        (remote != old and remote not in old.parents and old not in remote.parents) for old in seen),
                    "overlapping roots inside one layer")
            seen.append(remote)


def maps_for_export(plan, prepared, arm):
    host = prepared["execution"]["runtime_host_binding"]
    evidence, metadata = host["evidence_roots"][arm], host["metadata_roots"]
    validate_maps(evidence, metadata, fixture=prepared["fixture"])
    rows = []
    for index, remote, local in V1._root_rows(plan, prepared, arm):
        rows.append({"layer": "evidence", "index": index, "remote": remote, "local": local,
                     "host": producer_path(evidence[index]["local"])})
    require(len(rows) == len(evidence) and {row["index"] for row in rows} == set(range(len(evidence))),
            "exact ordered evidence map membership")
    for index, row in enumerate(metadata):
        rows.append({"layer": "metadata", "index": index, "remote": producer_path(row["remote"]),
                     "local": Path(row["remote"]), "host": producer_path(row["local"])})
    return rows


def maps_for_host(evidence, metadata):
    validate_maps(evidence, metadata, fixture=True, host_native=True)
    return [{"layer": layer, "index": index, "remote": producer_path(row["remote"]),
             "local": Path(row["local"]), "host": Path(row["local"])}
            for layer, roots in (("evidence", evidence), ("metadata", metadata)) for index, row in enumerate(roots)]


def locate(reference, rows, *, existing=True):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}
            and V1.HEX64.fullmatch(str(reference["sha256"])) is not None, "raw reference fields")
    value = producer_path(reference["path"])
    matches = []
    for row in rows:
        remote = row["remote"]
        if type(remote) is type(value) and (remote == value or remote in value.parents):
            parts = value.relative_to(remote).parts
            matches.append((row, row["local"].joinpath(*parts), row["host"].joinpath(*parts), parts))
    # 등록 logical alias가 있으면 native fallback을 섞지 않는다.
    if not matches and Path(reference["path"]).is_absolute():
        native = Path(reference["path"])
        for row in rows:
            if row["local"] == native or row["local"] in native.parents:
                parts = native.relative_to(row["local"]).parts
                matches.append((row, native, row["host"].joinpath(*parts), parts))
    if not matches:
        raise OutsideRoots("reference outside registered evidence/metadata roots")
    require(len({(path.absolute(), host) for _row, path, host, _parts in matches}) == 1,
            "ambiguous reference resolves to different registered files")
    row, path, host, parts = min(matches, key=lambda item: (LAYERS.index(item[0]["layer"]), -len(item[0]["remote"].parts), item[0]["index"]))
    require(bool(parts), "root itself is not a file reference")
    if existing:
        V1._single(path)
        require(path.resolve().is_relative_to(row["local"].resolve()) and sha_bytes(path.read_bytes()) == reference["sha256"],
                "raw reference SHA/location changed")
    return {"layer": row["layer"], "root_index": row["index"], "relative": PurePosixPath(*parts).as_posix(),
            "logical_path": str(row["remote"].joinpath(*parts)), "_path": path, "_host": host}


def _prepare(request, request_sha256, *, allow_fixture, home, now, preparer):
    prepared = V1._prepare(request, request_sha256, allow_fixture=allow_fixture, home=home, now=now, preparer=preparer)
    registered = prepared["protocol"]["sources_lf"]
    require(SOURCES <= registered.keys() and all(registered[name] == P.C.digest_lf_file(ROOT / name)
            for name in SOURCES - P.SOURCES), "v2 transport source/test closure")
    host = prepared["execution"]["runtime_host_binding"]
    for arm in ARMS:
        validate_maps(host["evidence_roots"][arm], host["metadata_roots"], fixture=prepared["fixture"])
    return prepared


def export(prepared, arm, output, before_add=None):
    require(prepared["fixture"] or before_add is None, "actual export callback injection forbidden")
    require(arm in ARMS, "export arm required")
    require(output.is_absolute() and output.name.endswith(".tar.gz") and output.parent.is_dir()
            and not any(path.is_symlink() for path in (output.parent, *output.parent.parents))
            and not output.exists(), "new absolute export path required")
    op = prepared["operations"][arm]["op"]
    unit = prepared["operations"][arm]["unit"]
    outcome_ref, request_ref = V1._ref(unit / "outcome.json"), V1._ref(unit / "request.json")
    outcome = P.U.read_ref(outcome_ref)
    require(outcome.get("status") == "complete" and outcome.get("operation_id") == op["operation_id"], "completed arm required")
    P._verify_completion(prepared, outcome)
    plan, evidence, request = (P.U.read_ref(ref) for ref in (outcome["plan"], outcome["evidence"], request_ref))
    require(V1.REQUIRED_PLAN_REFS <= plan.keys() and plan["operation_id"] == op["operation_id"]
            and plan["identity"]["arm"] == arm and request == evidence["request"]
            and P.E.digest(request) == outcome["request_sha256"], "required exact parent metadata")
    rows = maps_for_export(plan, prepared, arm)
    required = [outcome_ref, request_ref, outcome["plan"], outcome["evidence"], *[plan[key] for key in V1.REQUIRED_PLAN_REFS]]
    required += [{"path": path, "sha256": sha} for path, sha in evidence["raw_files_sha256"].items()]
    queue, files, skipped, total = list(required), {}, [], 0
    while queue:
        reference = queue.pop(0)
        try:
            located = locate(reference, rows)
        except OutsideRoots:
            if reference in required:
                raise
            skipped.append({**reference, "reason": "outside_registered_layers"})
            continue
        path = located["_path"]
        key = str(path.absolute())
        raw = path.read_bytes()
        if key in files:
            old = files[key]
            require(old["sha256"] == reference["sha256"] and old["_host"] == located["_host"], "same raw file maps to conflicting host copies")
            if LAYERS.index(located["layer"]) < LAYERS.index(old["layer"]):
                files[key] = {**located, "sha256": old["sha256"], "bytes": old["bytes"]}
            continue
        total += len(raw)
        require(len(files) + 1 <= V1.MAX_EXPORT_FILES and total <= V1.MAX_EXPORT_BYTES, "export cap exceeded")
        files[key] = {**located, "sha256": reference["sha256"], "bytes": len(raw)}
        # parent request는 두 arm의 binding을 가리키는 등록 문서다. raw copy는 보존하되
        # 다른 arm의 동일 logical namespace를 현재 arm resolver로 재해석하지 않는다.
        if path.suffix == ".json" and reference["sha256"] != prepared["request_ref"]["sha256"]:
            queue.extend(V1._refs(V1._read_json(raw, "referenced metadata")))
    ordered = sorted(files.values(), key=lambda row: (LAYERS.index(row["layer"]), row["root_index"], row["relative"]))
    for row in ordered:
        row["archive_path"] = f"files/{row['layer']}/{row['root_index']}/{row['relative']}"
    host = prepared["execution"]["runtime_host_binding"]
    manifest = {"schema": ARCHIVE_SCHEMA, "arm": arm, "operation_id": op["operation_id"],
        "outcome": outcome_ref, "plan": outcome["plan"], "evidence": outcome["evidence"], "request": request_ref,
        "parent_request_sha256": prepared["request_ref"]["sha256"],
        "parent_protocol_sha256": prepared["request"]["protocol"]["sha256"], "binding_sha256": prepared["request"]["binding"]["sha256"],
        "producer_evidence_roots_sha256": P.E.digest(plan["roots"]),
        "evidence_roots_sha256": P.E.digest(host["evidence_roots"][arm]), "metadata_roots_sha256": P.E.digest(host["metadata_roots"]),
        "files": [{k: v for k, v in row.items() if not k.startswith("_")} for row in ordered],
        "skipped_references": skipped, "authority_issued": False, "evidence_accepted": False}
    if before_add is not None:
        for row in ordered:
            before_add(row["_path"])
    try:
        with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            body = canonical(manifest)
            member = tarfile.TarInfo("manifest.json")
            member.size, member.mode, member.mtime = len(body), 0o600, 0
            archive.addfile(member, io.BytesIO(body))
            for row in ordered:
                raw = V1._single(row["_path"]).read_bytes()
                require(sha_bytes(raw) == row["sha256"], "raw changed during export")
                member = tarfile.TarInfo(row["archive_path"])
                member.size, member.mode, member.mtime = len(raw), 0o600, 0
                archive.addfile(member, io.BytesIO(raw))
        require(all(sha_bytes(V1._single(row["_path"]).read_bytes()) == row["sha256"] for row in ordered), "raw changed after export")
    except Exception as error:
        V1._error_receipt(Path(str(output) + ".error.json"), {"schema": SCHEMA + "_export_error", "error_type": type(error).__name__,
            "reason": str(error), "partial_output_preserved": output.exists(), "authority_issued": False, "evidence_accepted": False})
        raise
    return {"schema": SCHEMA, "status": "exported", "mode": "export", "arm": arm, "operation_id": op["operation_id"],
        "archive": V1._ref(output), "manifest_sha256": sha_bytes(canonical(manifest)), "files": len(ordered), "bytes": total,
        "authority_issued": False, "evidence_accepted": False}


def run_transport(*, request, request_sha256, mode="prepare", arm=None, payload_base64=None, output=None,
                  allow_fixture=False, home=None, now=None, preparer=PRODUCTION[0], popen_factory=PRODUCTION[1],
                  identity_fn=PRODUCTION[2], before_add=None):
    require(mode in MODES and type(allow_fixture) is bool, "transport mode/fixture flag")
    if not allow_fixture:
        require(preparer is PRODUCTION[0] and popen_factory is PRODUCTION[1] and identity_fn is PRODUCTION[2]
                and home is None and now is None and before_add is None, "actual transport dependency injection forbidden")
        require(os.name == "posix", "actual transport requires POSIX")
    prepared = _prepare(request, request_sha256, allow_fixture=allow_fixture, home=home, now=now, preparer=preparer)
    if mode == "prepare":
        value = V1._public_prepare(prepared, request, request_sha256)
    elif mode == "start":
        require(arm is None and payload_base64 is None and output is None, "start arguments")
        value = V1._start(prepared, request, request_sha256, popen_factory, identity_fn)
    elif mode == "status":
        require(payload_base64 is None and output is None, "status arguments")
        value = V1._status(prepared, arm)
    elif mode == "provider":
        require(payload_base64 is not None and output is None, "provider arguments")
        assert payload_base64 is not None
        value = V1._provider(prepared, str(arm), payload_base64, now or datetime.now(timezone.utc))  # noqa: UP017 - Python 3.10
    else:
        require(output is not None and payload_base64 is None, "export arguments")
        value = export(prepared, str(arm), output, before_add)
    return {**value, "schema": SCHEMA}


def main():
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
    except Exception as error:  # noqa: BLE001 - 구조화된 HOLD와 원문을 보존한다.
        result = {"schema": SCHEMA, "status": "HOLD", "dispatched": False,
                  "error_type": type(error).__name__, "reason": str(error), "authority_issued": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
