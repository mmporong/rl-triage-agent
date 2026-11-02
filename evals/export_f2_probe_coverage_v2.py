# pyright: reportMissingImports=false
"""완료 ledger와 분리해 RI410의 봉인 snapshot을 복원하는 F2 exporter v2."""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_f2_probe_coverage_v1 as V1

SCHEMA = "f2_probe_coverage_public_export_v2"
SOURCE_FILES = (
    "evals/export_f2_probe_coverage_v2.py",
    "tests/test_export_f2_probe_coverage_v2.py",
    "evals/export_f2_probe_coverage_v1.py",
)
_SCOPE_LOCK = threading.RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _lf_sha(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _frozen_snapshot(pair_path: Path, source: dict[str, Any], case_id: str,
                     temp_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """choice를 empty-ledger 시점에서 검사한 뒤 원 ledger path를 정확히 복원한다."""
    pair_path = Path(pair_path).resolve(strict=True)
    pair = V1.read(pair_path)
    ledger = pair.get("public_ledger")
    require(isinstance(ledger, dict) and isinstance(ledger.get("path"), str),
            "fresh pair public ledger")
    ledger_path = Path(ledger["path"]).resolve(strict=True)
    expected_ledger = pair_path.parents[2] / case_id / "oracle.ledger.jsonl"
    require(case_id in V1.TARGETS
            and ledger_path == expected_ledger.resolve(strict=True),
            "fresh pair storage contract")
    temp_root = Path(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    empty = temp_root / f"{case_id}.ledger.jsonl"
    empty.write_bytes(b"")
    historical = V1.RUN.DRIVER.C.V.build_snapshot(
        source["packet"], f"f2_oracle_{case_id.replace('-', '_')}",
        V1.RUN.HOST_STORAGE_POLICY,
        V1.RUN.DRIVER.C.V.file_digest(V1.RUN.DRIVER.C.V.PROTOCOL), empty)
    choice = pair.get("public_choice")
    require(isinstance(choice, dict)
            and choice == V1.RUN.oracle_choice(historical),
            "fresh pair historical choice")
    snapshot = deepcopy(historical)
    snapshot["inputs"]["ledger_path"] = str(ledger_path)
    require(V1.RUN.B.digest(snapshot) == pair.get("public_snapshot_sha256"),
            "fresh pair retargeted snapshot digest")
    return snapshot, deepcopy(choice)


@contextmanager
def _v1_snapshot_scope():
    with _SCOPE_LOCK:
        original = V1._frozen_snapshot
        V1._frozen_snapshot = _frozen_snapshot
        try:
            yield
        finally:
            V1._frozen_snapshot = original


def _provenance(registration_anchor: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "historical_reconstruction": "validate_choice_before_exact_ledger_retarget",
        "v2_source_sha256_lf": _lf_sha(ROOT / SOURCE_FILES[0]),
        "frozen_v1_verifier_sha256_lf": _lf_sha(ROOT / SOURCE_FILES[2]),
        "frozen_v1_schema": V1.SCHEMA,
        "frozen_v1_source_anchor": registration_anchor,
        "frozen_v1_source_closure_preserved": True,
    }


def build_export(result_path: Path, protocol_path: Path, registration_anchor: str,
                 plan_path: Path, *, lab: Path,
                 validator: Callable = V1.RUN.verify_protocol,
                 inspect_step: Callable = V1.AUDIT.inspect_step,
                 inspect_pair: Callable = V1.AUDIT.inspect_pair,
                 pair_verifier: Callable = V1.RUN.HOST.verify_pair,
                 source_verifier: Callable = V1.verify_source_anchor) -> dict[str, Any]:
    with _v1_snapshot_scope():
        value = V1.build_export(
            result_path, protocol_path, registration_anchor, plan_path, lab=lab,
            validator=validator, inspect_step=inspect_step, inspect_pair=inspect_pair,
            pair_verifier=pair_verifier, source_verifier=source_verifier)
    value = deepcopy(value)
    value["validator"] = _provenance(registration_anchor)
    V1._reject_paths(value)
    return value


def export(result_path: Path, protocol_path: Path, registration_anchor: str,
           plan_path: Path, *, lab: Path, out: Path) -> dict[str, Any]:
    value = build_export(result_path, protocol_path, registration_anchor, plan_path, lab=lab)
    V1.write_new(out, value)
    return deepcopy(value)


def verify(export_path: Path, result_path: Path, protocol_path: Path,
           registration_anchor: str, plan_path: Path, *, lab: Path) -> dict[str, Any]:
    value = V1.read(Path(export_path).resolve(strict=True))
    expected = build_export(result_path, protocol_path, registration_anchor, plan_path, lab=lab)
    require(value == expected, "public F2 coverage v2 export changed")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify"):
        child = sub.add_parser(name)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--registration-anchor", required=True)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--lab", type=Path, required=True)
        child.add_argument("--out" if name == "export" else "--export",
                           type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.result, args.protocol, args.registration_anchor, args.plan,
               lab=args.lab, out=args.out)
    else:
        verify(args.export, args.result, args.protocol, args.registration_anchor,
               args.plan, lab=args.lab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
