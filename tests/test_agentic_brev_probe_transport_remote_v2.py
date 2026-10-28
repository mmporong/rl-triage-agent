"""실제 Parent의 unit/raw layout과 별도 metadata layer를 CPU로 검증한다."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]
import agentic_brev_mailbox_service_v2 as S
import agentic_brev_probe_transport_remote_v2 as T
from test_agentic_brev_probe_parent_v1 import NOW, Services, world
from test_agentic_investigation_runtime_v1 import commit_objects

P, E, J = T.P, T.P.E, T.P.E.J


def save(path, value):
    J._write_new(path, value)
    return T.V1._ref(path)


@pytest.fixture
def pair(tmp_path):
    w = world(tmp_path / "producer")
    host = tmp_path / "host"
    host.mkdir()
    execution = w["binding"]["expected_request"]["execution"]
    evidence_roots = {}

    def projected(value):
        return str(host / "producer" / Path(value).relative_to(w["root"]))

    for arm in P.ORDER:
        op = execution["operations"][arm]
        raw = w["repo"] / "parents/operations" / op["operation_id"] / "raw"
        original = op["source_roots"] + w["binding"]["roots"] + [
            {"remote": "/parent/" + op["operation_id"], "local": str(w["folder"] / op["operation_id"])},
            {"remote": op["remote_stage"], "local": str(raw)}]
        evidence_roots[arm] = [{"remote": row["remote"], "local": projected(row["local"])} for row in original]
    execution["runtime_host_binding"] = {"evidence_roots": evidence_roots,
        "metadata_roots": [{"remote": str(w["root"]), "local": str(host / "producer")}]}
    sources = {name: (ROOT / name).read_bytes() for name in S.SOURCES}
    hashes = {name: P.C.digest_lf_file(ROOT / name) for name in S.SOURCES}
    execution["source_closure_sha256"] = E.digest(hashes)
    execution["ledger_projection"]["remote_path"] = "/transport/v2-consumed.jsonl"
    request_pair = w["binding"]["expected_request"]
    prefix = P.PairLedger(w["root"] / "v2-cpu-pairs.jsonl").authorize(request_pair, "explicit_cpu_fixture")
    (w["root"] / "transport/v2-consumed.jsonl").write_bytes(Path(prefix["path"]).read_bytes())
    binding_ref = save(w["root"] / "v2-binding.json", w["binding"])
    protocol = {**w["protocol"], "sources_lf": hashes, "binding_sha256": binding_ref["sha256"],
        "transport_root": {"base": "repo", "relative": "v2-transport"}}
    for name in ("bench/protocols/probe.json", "bench/protocols/basis.json"):
        sources[name] = (w["repo"] / name).read_bytes()
    sources["bench/protocols/v2-parent.json"] = J._bytes(protocol)
    anchor = commit_objects(w["repo"], sources, w["request"]["source_anchor"])
    request = {**w["request"], "protocol": T.V1._ref(w["repo"] / "bench/protocols/v2-parent.json"),
        "binding": binding_ref, "source_anchor": anchor, "authority": {"schema": P.SCHEMA + "_authority",
            "host_reference": {**prefix, "path": execution["ledger_projection"]["host_path"]},
            "projection": execution["ledger_projection"]}}
    request_ref = save(w["root"] / "v2-request.json", request)
    services = Services(w)
    result = P.run(request_ref, execute=True, allow_fixture=True, home=w["home"], now=NOW, services=services)
    assert result["status"] == "complete", result
    prepared = P.prepare(request_ref, allow_fixture=True, home=w["home"], now=NOW)
    return {"w": w, "host": host, "prepared": prepared, "request_ref": request_ref,
        "evidence_roots": evidence_roots, "metadata_roots": execution["runtime_host_binding"]["metadata_roots"]}


def exported(pair, path, arm="NONE"):
    value = T.export(pair["prepared"], arm, path)
    return value


def unpack(pair, archive, arm="NONE"):
    prepared = pair["prepared"]
    return S.unpack_archive(archive, pair["evidence_roots"][arm], [{"path": str(pair["host"]), "writable": True}],
        metadata_roots=pair["metadata_roots"], arm=arm, operation_id=prepared["operations"][arm]["op"]["operation_id"],
        parent_request_sha256=pair["request_ref"]["sha256"], parent_protocol_sha256=prepared["request"]["protocol"]["sha256"],
        binding_sha256=prepared["request"]["binding"]["sha256"])


def test_actual_parent_unit_raw_v1_failure_then_both_arms_v2_host_lookup(pair, tmp_path):
    prepared = pair["prepared"]
    for arm in P.ORDER:
        unit = prepared["operations"][arm]["unit"]
        assert prepared["operations"][arm]["stage"] == unit / "raw"
        assert all((unit / name).is_file() for name in ("outcome.json", "evidence.json", "request.json"))
        before = {path: path.read_bytes() for path in unit.rglob("*") if path.is_file()}
        with pytest.raises(ValueError, match="outside or ambiguous"):
            T.V1._export(prepared, arm, tmp_path / f"v1-{arm}.tar.gz")
        output = tmp_path / f"v2-{arm}.tar.gz"
        result = exported(pair, output, arm)
        assert result["authority_issued"] is result["evidence_accepted"] is False
        ref = unpack(pair, output, arm)
        metadata = S.H.MetadataMap(pair["metadata_roots"], True)
        outcome = S.T.read_ref(ref)
        original = metadata.read(outcome["evidence"])
        plan = metadata.read(outcome["plan"])
        request_path = metadata.file(T.V1._ref(unit / "request.json"))
        assert request_path.read_bytes() == (unit / "request.json").read_bytes()
        for name in T.V1.REQUIRED_PLAN_REFS:
            assert metadata.file(plan[name]).read_bytes() == Path(plan[name]["path"]).read_bytes()
        roots = pair["evidence_roots"][arm]
        derived = E.build(original["request"], roots, expected_request_sha256=original["request_sha256"],
            expected_roots_sha256=E.digest(roots), allow_fixture=True)
        assert {**derived, "root_map_sha256": original["root_map_sha256"]} == original
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert not (tmp_path / f"v1-{arm}.tar.gz").exists()


@pytest.mark.parametrize("mode", T.MODES)
def test_actual_injection_refused_before_any_preparer_or_spawn(tmp_path, mode):
    called = []
    with pytest.raises(ValueError, match="dependency injection"):
        T.run_transport(request=tmp_path / "absent.json", request_sha256="0" * 64, mode=mode,
                        preparer=lambda *_args, **_kwargs: called.append(True))
    assert called == []


def test_actual_export_callback_refused_before_parent_or_file_access(tmp_path):
    with pytest.raises(ValueError, match="callback injection"):
        T.export({"fixture": False}, "NONE", tmp_path / "not-created.tar.gz", lambda _path: None)
    assert not (tmp_path / "not-created.tar.gz").exists()


def test_v2_prepare_checks_source_and_map_closure(pair):
    prepared = pair["prepared"]

    def preparer(*_args, **_kwargs):
        return prepared

    result = T.run_transport(request=Path(pair["request_ref"]["path"]), request_sha256=pair["request_ref"]["sha256"],
        allow_fixture=True, preparer=preparer)
    assert result["schema"] == T.SCHEMA and result["status"] == "prepared"
    prepared["protocol"]["sources_lf"].pop(T.SOURCE)
    with pytest.raises(ValueError, match="v2 transport source"):
        T.run_transport(request=Path(pair["request_ref"]["path"]), request_sha256=pair["request_ref"]["sha256"],
            allow_fixture=True, preparer=preparer)


def test_logical_alias_priority_and_conflicting_layer_projection(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    raw = root / "raw.json"
    raw.write_bytes(b"{}")
    remote = E._logical("/original")
    rows = [{"layer": "evidence", "index": 0, "remote": remote, "local": root, "host": tmp_path / "host"},
            {"layer": "metadata", "index": 0, "remote": remote, "local": root, "host": tmp_path / "host"}]
    reference = {"path": "/original/raw.json", "sha256": T.sha_bytes(raw.read_bytes())}
    assert T.locate(reference, rows)["layer"] == "evidence"
    rows[1]["host"] = tmp_path / "different"
    with pytest.raises(ValueError, match="different registered"):
        T.locate(reference, rows)


def test_producer_accepts_registered_foreign_host_paths_without_filesystem_lookup(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("producer must not resolve the host Windows filesystem")

    monkeypatch.setattr(Path, "resolve", forbidden)
    T.validate_maps([{"remote": "/producer/ops/id/raw", "local": "Z:\\archive\\ops\\id\\raw"}],
                    [{"remote": "/producer", "local": "Z:\\archive"}])


def test_required_metadata_outside_and_partial_export_preserved(pair, tmp_path):
    prepared = pair["prepared"]
    changed = deepcopy(prepared)
    changed["execution"]["runtime_host_binding"]["metadata_roots"][0]["remote"] = str(tmp_path / "unregistered")
    with pytest.raises(T.OutsideRoots):
        T.export(changed, "NONE", tmp_path / "outside.tar.gz")
    target = prepared["operations"]["NONE"]["stage"] / "execution.json"
    original = target.read_bytes()

    def mutate(path):
        if path == target:
            path.write_bytes(original + b" ")

    output = tmp_path / "partial.tar.gz"
    with pytest.raises(ValueError, match="changed during export"):
        T.export(prepared, "NONE", output, mutate)
    assert output.exists() and Path(str(output) + ".error.json").exists()
    with pytest.raises(ValueError, match="new absolute export"):
        T.export(prepared, "NONE", output)
