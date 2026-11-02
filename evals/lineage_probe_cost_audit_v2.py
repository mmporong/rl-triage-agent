# pyright: reportMissingImports=false
"""완료 시점 ledger prefix로 R3 paired probe 비용을 재검증한다."""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_probe_cost_audit_v1 as V1

SCHEMA = "lineage_probe_cost_audit_v2"
SOURCE_FILES = (
    "evals/lineage_probe_cost_audit_v2.py",
    "tests/test_lineage_probe_cost_audit_v2.py",
    "evals/lineage_probe_cost_audit_v1.py",
)
_SCOPE_LOCK = threading.RLock()
HOST_IMPL = V1.HOST._HOST


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _lf_sha(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _event_digest(rows: list[dict[str, Any]]) -> str:
    return V1.digest(rows)


@contextmanager
def _historical_done_prefix_scope(temp_root: Path):
    """HOST의 ledger 완료 검사를 기록된 prefix bytes에만 한정한다."""
    with _SCOPE_LOCK:
        original = HOST_IMPL._done
        evidence: list[dict[str, Any]] = []
        root = Path(temp_root).resolve()
        root.mkdir(parents=True, exist_ok=True)

        def as_of_done(path: Path, binding: dict[str, Any], packet: dict[str, Any],
                       request_id: str, spec: dict[str, Any], *, prior: list[dict[str, Any]],
                       outcome: str, allow_suffix: bool,
                       prefix_proof: dict[str, Any] | None = None) -> dict[str, Any]:
            source = Path(path).resolve(strict=True)
            require(isinstance(prefix_proof, dict), "recorded done prefix proof가 없다")
            raw = source.read_bytes()
            length = prefix_proof.get("byte_length")
            require(type(length) is int and 0 <= length <= len(raw), "recorded prefix length")
            prefix = raw[:length]
            require(hashlib.sha256(prefix).hexdigest() == prefix_proof.get("sha256"),
                    "recorded prefix SHA")
            require(not prefix or prefix.endswith(b"\n"), "recorded prefix line boundary")

            # full ledger는 그대로 canonical parser를 통과해야 한다. pending은 보존한다.
            _rows, _ledger_digest, pending = HOST_IMPL.C.V.read_history(
                source, binding, packet["execution_budget_s"])
            full_ledger = HOST_IMPL.C.S.M.RegistryLedger(
                source, HOST_IMPL.C.RC.registry())
            full_events = full_ledger.events()

            temporary = root / (
                hashlib.sha256((str(source) + request_id).encode("utf-8")).hexdigest()
                + ".ledger.jsonl")
            require(not temporary.exists(), "historical prefix temp collision")
            temporary.write_bytes(prefix)
            prefix_ledger = HOST_IMPL.C.S.M.RegistryLedger(
                temporary, HOST_IMPL.C.RC.registry())
            prefix_events = prefix_ledger.events()
            require(full_events[:len(prefix_events)] == prefix_events,
                    "full ledger/prefix event mismatch")
            suffix = full_events[len(prefix_events):]
            require(all(row.get("request_id") != request_id for row in suffix),
                    "completed request has illegal suffix events")
            require(allow_suffix or not suffix, "backend ledger has illegal suffix")

            checked = original(
                temporary, binding, packet, request_id, spec, prior=prior,
                outcome=outcome, allow_suffix=allow_suffix, prefix_proof=prefix_proof)
            prefix_request = prefix_ledger.requests().get(request_id, {})
            full_request = full_ledger.requests().get(request_id, {})
            require(prefix_request == full_request and prefix_request.get("state") == "done",
                    "completed request state changed after prefix")
            checked["path"] = str(source)
            evidence.append({
                "ledger_kind": "public" if allow_suffix else "backend",
                "request_id_sha256": hashlib.sha256(request_id.encode("utf-8")).hexdigest(),
                "prefix": deepcopy(prefix_proof),
                "prefix_event_count": len(prefix_events),
                "prefix_event_digest": _event_digest(prefix_events),
                "suffix_event_count": len(suffix),
                "suffix_event_digest": _event_digest(suffix),
                "full_event_count": len(full_events),
                "full_pending_request_sha256": (
                    None if pending is None else
                    hashlib.sha256(str(pending).encode("utf-8")).hexdigest()),
                "same_request_suffix_events": 0,
            })
            return checked

        HOST_IMPL._done = as_of_done
        try:
            yield evidence
        finally:
            HOST_IMPL._done = original


def inspect_step(detail_path: Path, source: dict[str, Any], step_index: int,
                 temp_root: Path, *, host_verify: Callable = V1.HOST.verify_pair) -> dict[str, Any]:
    """v1 결합을 유지하고 HOST done 검사만 recorded prefix 시점으로 제한한다."""
    detail = V1.read(Path(detail_path).resolve(strict=True))
    steps = detail.get("steps")
    require(isinstance(steps, list) and 0 <= step_index < len(steps), "step index가 다르다")
    step = cast(dict[str, Any], steps[step_index])
    pair_path = V1._bound(step.get("probe_result"), "step pair result")
    pair = V1.read(pair_path)
    snapshot = V1._historical_snapshot(detail, source, step_index, pair, temp_root)
    choice = pair.get("public_choice")
    require(isinstance(choice, dict), "pair public choice가 없다")
    case_plan = V1._bound(pair["plans"]["case"], "case plan")
    with (
        V1._historical_protocol_scope(case_plan),
        _historical_done_prefix_scope(Path(temp_root) / "as-of-prefix") as prefix_evidence,
    ):
        verified = host_verify(pair_path, snapshot, choice)
    require(len(prefix_evidence) == 2
            and {row["ledger_kind"] for row in prefix_evidence} == {"public", "backend"},
            "public/backend historical prefix evidence")
    inspected = V1.inspect_pair(pair_path)
    require(verified.get("result_ref", {}).get("sha256") == inspected["pair_result_sha256"],
            "HOST verifier/raw pair SHA가 다르다")
    require(inspected["probe"] == step.get("probe")
            and inspected["outcome"] == step.get("outcome")
            and inspected["reported"]["owned_child_wall_s"] == step.get("owned_child_wall_s"),
            "driver step/pair result 결합이 다르다")
    return {
        "step": step.get("step"), "probe": step.get("probe"),
        "outcome": step.get("outcome"), "pair": inspected,
        "historical_prefix": prefix_evidence,
    }


@contextmanager
def _v1_inspect_scope():
    with _SCOPE_LOCK:
        original = V1.inspect_step
        V1.inspect_step = inspect_step
        try:
            yield
        finally:
            V1.inspect_step = original


def audit_batch(batch_path: Path, protocol_path: Path, config_path: Path,
                anchor: str) -> dict[str, Any]:
    with _v1_inspect_scope():
        return V1.audit_batch(batch_path, protocol_path, config_path, anchor)


def _prefix_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [entry for row in rows for step in row["steps"]
                for entry in step.get("historical_prefix", [])]
    require(len(evidence) == sum(len(row["steps"]) for row in rows) * 2,
            "historical prefix evidence count")
    public = [row for row in evidence if row["ledger_kind"] == "public"]
    return {
        "validator_schema": SCHEMA,
        "mode": "recorded_done_prefix_with_validated_full_ledger",
        "completed_pair_steps": len(public),
        "steps_with_future_suffix": sum(row["suffix_event_count"] > 0 for row in public),
        "same_request_suffix_events": sum(row["same_request_suffix_events"] for row in public),
        "evidence_digest": V1.digest(evidence),
        "validator_sources_lf": {name: _lf_sha(ROOT / name) for name in SOURCE_FILES},
    }


def export(batch_path: Path, protocol_path: Path, config_path: Path, anchor: str,
           out: Path) -> dict[str, Any]:
    """v1 4-file schema를 유지해 fresh 공개 tag를 만든다."""
    output = Path(out).resolve()
    V1.require(not output.exists(), "공개 output은 fresh 경로여야 한다")
    V1.require(output.parent == (ROOT / "evals/results").resolve(),
               "공개 output은 evals/results/<fresh-tag>여야 한다")
    paths = tuple(Path(path).resolve(strict=True)
                  for path in (batch_path, protocol_path, config_path))
    before = {path: V1.sha(path) for path in paths}
    audited = audit_batch(*paths, anchor)
    totals = audited["totals"]
    summary = {
        "schema": V1.SUMMARY_SCHEMA, "status": "complete",
        "scope": "actual_backend_probe_cost_receipt_audit",
        "timing_source": "receipt_reconstruction", "wall_scope": V1.WALL_SCOPE,
        "totals": totals, "by_policy": audited["by_policy"], "by_case": audited["by_case"],
        "intent_accounting": audited["intent_accounting"],
        "duplicate_receipt_references": audited["duplicates"],
        "interpretation_limits": {
            "sdk_wall_is_gpu_kernel_time": False,
            "sdk_wall_is_gpu_busy_time": False,
            "diagnostic_efficacy_measured": False,
            "model_comparison_claimed": False,
            "new_gpu_savings_claimed": False,
            "missing_wall_filled_with_zero": False,
        },
    }
    cases = {"schema": V1.CASES_SCHEMA, "status": "complete", "rows": audited["rows"]}
    config = audited["config"]
    source_index = V1.SUMMARY._bound_file(config["source_index"], "source index")
    provenance = {
        "schema": V1.PROVENANCE_SCHEMA, "status": "complete",
        "execution_anchor": anchor,
        "public_protocol": {"path": V1._public_relative(paths[1], "protocol"),
                            "sha256": V1.sha(paths[1])},
        "sha256_only": {"batch": V1.sha(paths[0]), "execution_config": V1.sha(paths[2]),
                        "source_index": V1.sha(source_index)},
        "historical_prefix_validation": _prefix_provenance(audited["rows"]),
        "raw_receipts_included": False, "raw_metadata_included": False,
        "private_paths_included": False, "credentials_included": False,
        "pid_values_included": False,
    }
    V1._assert_public(summary, cases, provenance)
    output.mkdir(parents=True, exist_ok=False)
    V1.write_new(output / "summary.json", summary)
    V1.write_new(output / "cases.json", cases)
    V1.write_new(output / "provenance.json", provenance)
    inventory = {name: {"sha256": V1.sha(output / name),
                        "size_bytes": (output / name).stat().st_size}
                 for name in V1.PUBLIC_FILES if name != "manifest.json"}
    manifest = {"schema": V1.MANIFEST_SCHEMA, "status": "complete",
                "public_files": list(V1.PUBLIC_FILES), "files": inventory,
                "overwrite_allowed": False}
    V1._assert_public(manifest)
    V1.write_new(output / "manifest.json", manifest)
    V1.require(before == {path: V1.sha(path) for path in paths},
               "audit 중 등록 원문이 바뀌었다")
    verify_export(output)
    return deepcopy(manifest)


def verify_export(out: Path) -> dict[str, Any]:
    manifest = V1.verify_export(out)
    root = Path(out).resolve(strict=True)
    provenance = V1.read(root / "provenance.json")
    cases = V1.read(root / "cases.json")
    prefix = provenance.get("historical_prefix_validation")
    require(isinstance(cases.get("rows"), list)
            and prefix == _prefix_provenance(cases["rows"])
            and prefix.get("validator_schema") == SCHEMA
            and prefix.get("mode") == "recorded_done_prefix_with_validated_full_ledger"
            and prefix.get("same_request_suffix_events") == 0,
            "v2 historical prefix provenance")
    return deepcopy(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--batch", type=Path, required=True)
    export_parser.add_argument("--protocol", type=Path, required=True)
    export_parser.add_argument("--config", type=Path, required=True)
    export_parser.add_argument("--anchor", required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.batch, args.protocol, args.config, args.anchor, args.out)
    else:
        verify_export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
