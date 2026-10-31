"""epoch 2 preflight evidence를 fresh 실행 폴더 밖에 봉인하는 v2 wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_epoch_v1 as E2

PROTOCOL_SCHEMA = "lineage_post_diagnosis_epoch_protocol_v2"
WRAPPER_RECEIPT_SCHEMA = "lineage_post_diagnosis_epoch_wrapper_receipt_v2"
SOURCE_FILES = (
    "evals/lineage_post_diagnosis_epoch_v2.py",
    "tests/test_lineage_post_diagnosis_epoch_v2.py",
    "evals/lineage_post_diagnosis_epoch_v1.py",
    "tests/test_lineage_post_diagnosis_epoch_v1.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_LOCK = RLock()
_LOCAL = threading.local()


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("parent protocol은 저장소 파일이어야 한다") from exc


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(parent_protocol: Path) -> dict:
    """동결 v1 protocol을 바꾸지 않고 fresh-out 수정 closure만 추가한다."""
    parent = Path(parent_protocol).resolve()
    value = _read(parent)
    _require(value.get("schema") == E2.PROTOCOL_SCHEMA, "parent epoch2 protocol schema가 다르다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "component_parent_protocol": {"path": _relative(parent), "sha256_lf": _sha(parent, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "scope": {
            "change": "preflight_evidence_sibling_path_only",
            "parent_result_schema_unchanged": True,
            "model_queue_reuse": False,
            "diagnosis_or_training_replay": False,
        },
    }


def verify_protocol(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "wrapper anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    parent = value.get("component_parent_protocol")
    if not isinstance(parent, dict) or set(parent) != {"path", "sha256_lf"}:
        raise ValueError("parent epoch2 protocol 결합이 없다")
    parent_path = ROOT / str(parent["path"])
    _require(value == prepare_protocol(parent_path), "v2 wrapper protocol 내용 또는 SHA가 다르다")
    E2.verify_protocol(parent_path, lab=Path(lab), anchor=anchor)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), parent["path"]: parent["sha256_lf"],
             **value["source_sha256_lf"]}
    for relative, expected in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 v2 wrapper source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected,
                 f"wrapper anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def preflight_evidence_path(out: Path) -> Path:
    out = Path(out).resolve()
    return out.parent / f"{out.name}.lease-read-preflight.json"


@contextmanager
def _preflight_redirect(out: Path) -> Iterator[Path]:
    """v1이 요청한 내부 경로 하나만 형제 write-once 파일로 바꾼다."""
    out = Path(out).resolve()
    requested = out / "lease-read-evidence.json"
    sibling = preflight_evidence_path(out)
    _require(not out.exists(), "epoch2 output은 실행 전 fresh 경로여야 한다")
    _require(not sibling.exists(), "v2 preflight evidence가 이미 존재한다")
    original = E2.RETRY.lease_read_scope
    called = False

    def restore() -> None:
        if E2.RETRY.lease_read_scope is redirected:
            E2.RETRY.lease_read_scope = original

    @contextmanager
    def redirected(path: Path) -> Iterator[None]:
        nonlocal called
        actual = Path(path).resolve()
        _require(not called and actual == requested,
                 "v1 preflight가 등록 내부 경로 외 위치를 요청했거나 두 번 호출됐다")
        called = True
        _require(not out.exists(), "preflight가 epoch2 output 폴더를 먼저 만들었다")
        try:
            with original(sibling):
                yield
        finally:
            restore()
        _require(not out.exists(), "preflight 종료 전에 epoch2 output 폴더가 생겼다")

    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        _require(depth == 0 and E2.RETRY.lease_read_scope is original,
                 "v2 preflight redirect가 이미 활성화됐거나 원본이 바뀌었다")
        E2.RETRY.lease_read_scope = redirected
        _LOCAL.depth = 1
        try:
            yield sibling
        finally:
            _LOCAL.depth = 0
            restore()


def run_epoch(out: Path, post_input: Path, reference_input: Path, policy: str, *,
              wrapper_protocol: Path, wrapper_anchor: str,
              epoch_protocol: Path, anchor: str, **kwargs: Any) -> dict:
    """v1 실행 의미를 유지하고 preflight 파일 위치 충돌만 제거한다."""
    out = Path(out).resolve()
    _require(wrapper_anchor == anchor, "wrapper와 parent 실행 anchor가 다르다")
    wrapper = verify_protocol(Path(wrapper_protocol), lab=Path(str(kwargs.get("lab", ""))),
                              anchor=wrapper_anchor)
    parent_path = (ROOT / wrapper["component_parent_protocol"]["path"]).resolve()
    _require(Path(epoch_protocol).resolve() == parent_path,
             "실행 parent protocol이 wrapper component와 다르다")
    with _preflight_redirect(out) as evidence:
        result = E2.run_epoch(
            out, Path(post_input), Path(reference_input), policy,
            epoch_protocol=parent_path, anchor=anchor, **kwargs)
    _require(result.get("schema") == E2.SCHEMA, "v2가 parent result schema를 바꿨다")
    result_path = out / "result.json"
    _require(result_path.is_file(), "parent epoch2 result.json이 없다")
    receipt = {
        "schema": WRAPPER_RECEIPT_SCHEMA, "status": "complete",
        "wrapper_protocol": {"path": str(Path(wrapper_protocol).resolve()),
                             "sha256": _sha(Path(wrapper_protocol))},
        "parent_protocol": {"path": str(parent_path), "sha256_lf": _sha(parent_path, lf=True)},
        "anchor": anchor,
        "preflight_evidence": {"path": str(evidence), "sha256": _sha(evidence)},
        "parent_result": {"path": str(result_path), "sha256": _sha(result_path)},
        "parent_result_schema": result["schema"],
        "preflight_inside_output": False,
        "diagnosis_replayed": False, "training_replayed": False,
    }
    _write_new(out / "v2-wrapper-receipt.json", receipt)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    prepare.add_argument("--parent-protocol", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--post-input", type=Path, required=True)
    run.add_argument("--reference-input", type=Path, required=True)
    run.add_argument("--policy", choices=E2.E0.POLICIES, required=True)
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
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.parent_protocol))
        return 0
    run_epoch(
        args.out, args.post_input, args.reference_input, args.policy,
        wrapper_protocol=args.wrapper_protocol, wrapper_anchor=args.wrapper_anchor,
        epoch_protocol=args.epoch_protocol, anchor=args.anchor, lab=args.lab,
        probe_protocol=args.probe_protocol, run_id=f"post_{args.policy}",
        gate_path=args.gate, gate_sha256=args.gate_sha256, case_id=args.case_id,
        transport_root=args.transport_root, slot_ids=tuple(args.slot_id),
        max_decisions=3, execution_budget_s=1200,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
