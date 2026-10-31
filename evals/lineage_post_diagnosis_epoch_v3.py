"""epoch 2 runner에 검증한 selection parent scope를 추가하는 v3 wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_epoch_v2 as V2
import lineage_post_parent_binding_v1 as PARENT

PROTOCOL_SCHEMA = "lineage_post_diagnosis_epoch_protocol_v3"
WRAPPER_RECEIPT_SCHEMA = "lineage_post_diagnosis_epoch_wrapper_receipt_v3"
SOURCE_FILES = (
    "evals/lineage_post_diagnosis_epoch_v3.py",
    "tests/test_lineage_post_diagnosis_epoch_v3.py",
    "evals/lineage_post_parent_binding_v1.py",
    "tests/test_lineage_post_parent_binding_v1.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("v2 parent protocol은 저장소 파일이어야 한다") from exc


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(parent_v2_protocol: Path) -> dict:
    parent = Path(parent_v2_protocol).resolve()
    value = _read(parent)
    _require(value.get("schema") == V2.PROTOCOL_SCHEMA, "v2 parent protocol schema가 다르다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "component_v2_protocol": {"path": _relative(parent), "sha256_lf": _sha(parent, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "scope": {"change": "epoch2_exact_parent_binding_only",
                  "parent_result_schema_unchanged": True,
                  "budgets_probes_prompts_unchanged": True},
    }


def verify_protocol(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "v3 anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    parent = value.get("component_v2_protocol")
    if not isinstance(parent, dict) or set(parent) != {"path", "sha256_lf"}:
        raise ValueError("v2 parent protocol 결합이 없다")
    parent_path = ROOT / str(parent["path"])
    _require(value == prepare_protocol(parent_path), "v3 protocol 내용 또는 SHA가 다르다")
    V2.verify_protocol(parent_path, lab=Path(lab), anchor=anchor)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), parent["path"]: parent["sha256_lf"],
             **value["source_sha256_lf"]}
    for relative, expected in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 v3 source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected,
                 f"v3 anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def run_epoch(out: Path, post_input: Path, reference_input: Path, policy: str, *,
              v3_protocol: Path, v3_anchor: str,
              wrapper_protocol: Path, wrapper_anchor: str,
              epoch_protocol: Path, anchor: str, run_id: str, **kwargs: Any) -> dict:
    out = Path(out).resolve()
    _require(v3_anchor == wrapper_anchor == anchor, "v3/v2/parent anchor가 다르다")
    protocol = verify_protocol(Path(v3_protocol), lab=Path(str(kwargs.get("lab", ""))),
                               anchor=v3_anchor)
    component = (ROOT / protocol["component_v2_protocol"]["path"]).resolve()
    _require(Path(wrapper_protocol).resolve() == component,
             "실행 v2 protocol이 v3 component와 다르다")
    with PARENT.parent_scope(Path(post_input), Path(reference_input), run_id, policy) as context:
        result = V2.run_epoch(
            out, Path(post_input), Path(reference_input), policy,
            wrapper_protocol=component, wrapper_anchor=wrapper_anchor,
            epoch_protocol=Path(epoch_protocol), anchor=anchor, run_id=run_id, **kwargs)
    _require(result.get("schema") == V2.E2.SCHEMA, "v3가 parent result schema를 바꿨다")
    result_path = out / "result.json"
    _require(result_path.is_file(), "v3 parent result.json이 없다")
    post_sha = _sha(Path(post_input))
    receipt = {
        "schema": WRAPPER_RECEIPT_SCHEMA, "status": "complete",
        "v3_protocol": {"path": str(Path(v3_protocol).resolve()),
                        "sha256": _sha(Path(v3_protocol))},
        "v2_protocol": {"path": str(component), "sha256_lf": _sha(component, lf=True)},
        "anchor": anchor, "post_observation_sha256": post_sha,
        "parent_context": {"packet_digest": context["packet_digest"],
                           "run_id": context["run_id"], "policy": context["policy"],
                           "parent": context["parent"]},
        "parent_result": {"path": str(result_path), "sha256": _sha(result_path)},
        "parent_result_schema": result["schema"],
    }
    _write_new(out / "v3-adapter-receipt.json", receipt)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    prepare.add_argument("--parent-v2-protocol", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--post-input", type=Path, required=True)
    run.add_argument("--reference-input", type=Path, required=True)
    run.add_argument("--policy", choices=V2.E2.E0.POLICIES, required=True)
    run.add_argument("--v3-protocol", type=Path, required=True)
    run.add_argument("--v3-anchor", required=True)
    run.add_argument("--wrapper-protocol", type=Path, required=True)
    run.add_argument("--wrapper-anchor", required=True)
    run.add_argument("--epoch-protocol", type=Path, required=True)
    run.add_argument("--probe-protocol", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    run.add_argument("--lab", type=Path, required=True)
    run.add_argument("--gate", type=Path, required=True)
    run.add_argument("--gate-sha256", required=True)
    run.add_argument("--case-id", default="case-07")
    run.add_argument("--transport-root", type=Path)
    run.add_argument("--slot-id", action="append", default=[])
    run.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.parent_v2_protocol))
        return 0
    run_epoch(
        args.out, args.post_input, args.reference_input, args.policy,
        v3_protocol=args.v3_protocol, v3_anchor=args.v3_anchor,
        wrapper_protocol=args.wrapper_protocol, wrapper_anchor=args.wrapper_anchor,
        epoch_protocol=args.epoch_protocol, anchor=args.anchor, run_id=args.run_id,
        lab=args.lab, probe_protocol=args.probe_protocol, gate_path=args.gate,
        gate_sha256=args.gate_sha256, case_id=args.case_id,
        transport_root=args.transport_root, slot_ids=tuple(args.slot_id),
        max_decisions=3, execution_budget_s=1200,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
