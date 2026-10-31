"""parent-bound snapshot scope 안에서 second recovery V1을 그대로 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_second_recovery_v1 as S1
import lineage_post_parent_binding_v1 as PARENT

SCHEMA = "lineage_compound_second_recovery_protocol_v2"
PREPARE_RECEIPT_SCHEMA = "lineage_compound_second_recovery_parent_adapter_v1"
EXECUTION_RECEIPT_SCHEMA = "lineage_compound_second_recovery_parent_execution_adapter_v1"
PREPARE_RECEIPT = "parent-adapter-receipt.json"
EXECUTION_RECEIPT = "parent-adapter-execution-receipt.json"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = tuple(sorted({
    "evals/lineage_post_parent_binding_v1.py",
    "tests/test_lineage_post_parent_binding_v1.py",
    "evals/lineage_compound_second_recovery_v2.py",
    "tests/test_lineage_compound_second_recovery_v2.py",
}))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    _require(isinstance(value, dict), f"JSON 객체가 아니다: {path}")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("공개 component는 저장소 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(parent_s1_protocol: Path) -> dict[str, Any]:
    """동결 S1 protocol을 parent-binding helper와 새 wrapper source에 결합한다."""
    inner_path = Path(parent_s1_protocol).resolve()
    inner = _read(inner_path)
    _require(inner.get("schema") == S1.SCHEMA, "second recovery S1 protocol schema가 다르다")
    return {
        "schema": SCHEMA,
        "phase": "second_recovery_parent_bound_adapter_only",
        "component_protocols": {
            "second_recovery_s1": {"path": _relative(inner_path),
                                   "sha256_lf": _sha(inner_path, lf=True)},
        },
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(inner["runtime_source_sha256_lf"]),
        "adapter_contract": {
            "inner_manifest_schema_unchanged": S1.MANIFEST_SCHEMA,
            "inner_result_schema_unchanged": S1.RESULT_SCHEMA,
            "parent_scope_required_for_prepare_and_execute": True,
            "prepare_receipt": PREPARE_RECEIPT,
            "execution_receipt": EXECUTION_RECEIPT,
            "arbitrary_manifest_rejected": True,
            "training_path_unchanged": "S1_BOTH_fresh300_fixed26",
        },
        "claims": {"api_requests": 0, "gpu_jobs": 0, "recovery_effect": False},
    }


def verify_protocol(protocol_path: Path, *, wrapper_anchor: str, lab: Path) -> dict[str, Any]:
    _require(isinstance(wrapper_anchor, str) and HEX40.fullmatch(wrapper_anchor) is not None,
             "wrapper anchor 형식이 다르다")
    path, lab = Path(protocol_path).resolve(), Path(lab).resolve()
    value = _read(path)
    components = value.get("component_protocols")
    if not isinstance(components, dict):
        raise TypeError("wrapper component protocol 표가 없다")
    inner_path = ROOT / components["second_recovery_s1"]["path"]
    _require(value == prepare_protocol(inner_path), "wrapper protocol 내용 또는 SHA가 다르다")
    _git("cat-file", "-e", f"{wrapper_anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", wrapper_anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 wrapper source가 다르다: {relative}")
        committed = _git("show", f"{wrapper_anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"wrapper anchor source blob이 다르다: {relative}")
    for relative, expected_sha in value["runtime_source_sha256_lf"].items():
        runtime = lab / relative
        _require(runtime.is_file() and _sha(runtime, lf=True) == expected_sha,
                 f"wrapper runtime source가 다르다: {relative}")
    return deepcopy(value)


def _context(post_input: Path, reference_input: Path, epoch2_result: Path) -> dict[str, Any]:
    post_input = Path(post_input).resolve()
    reference_input = Path(reference_input).resolve()
    epoch2_result = Path(epoch2_result).resolve()
    result = _read(epoch2_result)
    run_id, policy = result.get("run_id"), result.get("policy")
    _require(result.get("schema") == S1.E2.SCHEMA and result.get("epoch") == 2
             and isinstance(run_id, str) and bool(run_id)
             and policy in S1.E2.E0.POLICIES,
             "epoch2 result의 run_id/policy parent context가 다르다")
    return {"post_input": _ref(post_input), "reference_input": _ref(reference_input),
            "epoch2_result": _ref(epoch2_result), "run_id": run_id, "policy": policy}


def prepare_adapter_receipt(wrapper_protocol: Path, wrapper_anchor: str,
                            inner_manifest_path: Path, epoch2_result: Path,
                            context: dict[str, Any], out: Path) -> dict[str, Any]:
    """parent scope로 만들어진 S1 manifest와 exact context를 write-once 봉인한다."""
    value = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "wrapper_protocol": _ref(Path(wrapper_protocol)), "wrapper_anchor": wrapper_anchor,
        "inner_manifest": _ref(Path(inner_manifest_path)),
        "epoch2_result": _ref(Path(epoch2_result)),
        "context": deepcopy(context), "context_sha256": _digest(context),
        "inner_manifest_schema": S1.MANIFEST_SCHEMA, "candidate_revalidated_in_parent_scope": True,
    }
    _write_new(Path(out), value)
    return deepcopy(value)


def _checked_prepare_receipt(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    receipt_path = manifest_path.parent / PREPARE_RECEIPT
    receipt = _read(receipt_path)
    _require(receipt.get("schema") == PREPARE_RECEIPT_SCHEMA
             and receipt.get("inner_manifest") == _ref(manifest_path)
             and receipt.get("inner_manifest_schema") == S1.MANIFEST_SCHEMA
             and receipt.get("candidate_revalidated_in_parent_scope") is True,
             "parent adapter prepare receipt가 manifest와 다르다")
    protocol_ref = receipt.get("wrapper_protocol")
    if not isinstance(protocol_ref, dict):
        raise TypeError("wrapper protocol ref가 없다")
    manifest = _read(manifest_path)
    protocol = verify_protocol(
        Path(protocol_ref["path"]), wrapper_anchor=receipt["wrapper_anchor"],
        lab=Path(manifest["lab"]),
    )
    row = manifest.get("row")
    if not isinstance(row, dict):
        raise TypeError("S1 manifest row가 없다")
    base_context = _context(
        Path(row["post_input"]["path"]), Path(row["reference_input"]["path"]),
        Path(row["epoch_result"]["path"]),
    )
    with PARENT.parent_scope(
            Path(base_context["post_input"]["path"]),
            Path(base_context["reference_input"]["path"]),
            base_context["run_id"], base_context["policy"]) as parent_context:
        context = {**base_context, "parent_binding": parent_context}
    _require(receipt.get("context") == context
             and receipt.get("context_sha256") == _digest(context)
             and receipt.get("epoch2_result") == context["epoch2_result"],
             "prepare adapter parent context가 바뀌었다")
    inner = protocol["component_protocols"]["second_recovery_s1"]
    _require(manifest["protocol"]["path"] == str((ROOT / inner["path"]).resolve())
             and _sha(Path(manifest["protocol"]["path"]), lf=True) == inner["sha256_lf"],
             "임의 S1 protocol/manifest는 wrapper가 소비하지 않는다")
    return receipt, context


def prepare_inputs(*, wrapper_protocol: Path, wrapper_anchor: str,
                   out: Path, protocol_path: Path, post_input: Path,
                   epoch2_result_path: Path, reference_input: Path,
                   first_manifest_path: Path, first_smoke_root: Path, gate_path: Path,
                   gate_sha256: str, lab: Path, isaac_python: Path,
                   execution_anchor: str, diagnosis_anchor: str) -> dict[str, Any]:
    """exact parent scope에서 S1 prepare-inputs와 full candidate 재검증을 수행한다."""
    outer = verify_protocol(wrapper_protocol, wrapper_anchor=wrapper_anchor, lab=lab)
    inner = outer["component_protocols"]["second_recovery_s1"]
    _require(Path(protocol_path).resolve() == (ROOT / inner["path"]).resolve()
             and _sha(Path(protocol_path), lf=True) == inner["sha256_lf"],
             "wrapper가 등록하지 않은 S1 protocol이다")
    base_context = _context(post_input, reference_input, epoch2_result_path)
    with PARENT.parent_scope(
            Path(post_input), Path(reference_input), base_context["run_id"],
            base_context["policy"]) as parent_context:
        manifest = S1.prepare_inputs(
            out, protocol_path, post_input, epoch2_result_path, reference_input,
            first_manifest_path, first_smoke_root, gate_path=gate_path,
            gate_sha256=gate_sha256, lab=lab, isaac_python=isaac_python,
            execution_anchor=execution_anchor, diagnosis_anchor=diagnosis_anchor,
        )
        if manifest["row"]["status"] == "candidate":
            S1._checked_candidate(manifest["row"], manifest)
    context = {**base_context, "parent_binding": parent_context}
    prepare_adapter_receipt(
        wrapper_protocol, wrapper_anchor, Path(out) / "manifest.json", epoch2_result_path,
        context, Path(out) / PREPARE_RECEIPT,
    )
    return deepcopy(manifest)


def execute(manifest_path: Path, approval_path: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    """adapter receipt와 exact parent scope를 검증한 뒤 S1 execute를 그대로 위임한다."""
    receipt, context = _checked_prepare_receipt(manifest_path)
    out = Path(out).resolve()
    _require(not out.exists(), "wrapper execute out은 fresh 경로여야 한다")
    with PARENT.parent_scope(
            Path(context["post_input"]["path"]), Path(context["reference_input"]["path"]),
            context["run_id"], context["policy"]) as parent_context:
        _require(parent_context == context["parent_binding"],
                 "execute parent binding이 prepare receipt와 다르다")
        result = S1.execute(manifest_path, approval_path, out, **kwargs)
    inner_result_path = out / "result.json"
    _require(inner_result_path.is_file() and _read(inner_result_path) == result,
             "S1 result 파일과 반환값이 다르다")
    execution_receipt = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "prepare_adapter_receipt": _ref(Path(manifest_path).parent / PREPARE_RECEIPT),
        "wrapper_protocol": deepcopy(receipt["wrapper_protocol"]),
        "wrapper_anchor": receipt["wrapper_anchor"],
        "inner_result": _ref(inner_result_path),
        "inner_result_schema": S1.RESULT_SCHEMA,
        "context_sha256": receipt["context_sha256"],
        "training_path": "S1_BOTH_fresh300_fixed26",
    }
    _write_new(out / EXECUTION_RECEIPT, execution_receipt)
    return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    public = commands.add_parser("prepare-protocol")
    public.add_argument("--parent-s1-protocol", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare-inputs")
    for name in ("wrapper-protocol", "out", "protocol", "post-input", "epoch2-result",
                 "reference-input", "first-manifest", "first-smoke-root", "gate", "lab",
                 "isaac-python"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("wrapper-anchor", "gate-sha256", "execution-anchor", "diagnosis-anchor"):
        prep.add_argument("--" + name, required=True)
    run = commands.add_parser("execute")
    for name in ("manifest", "approval", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.parent_s1_protocol))
    elif args.command == "prepare-inputs":
        prepare_inputs(
            wrapper_protocol=args.wrapper_protocol, wrapper_anchor=args.wrapper_anchor,
            out=args.out, protocol_path=args.protocol, post_input=args.post_input,
            epoch2_result_path=args.epoch2_result, reference_input=args.reference_input,
            first_manifest_path=args.first_manifest, first_smoke_root=args.first_smoke_root,
            gate_path=args.gate, gate_sha256=args.gate_sha256, lab=args.lab,
            isaac_python=args.isaac_python, execution_anchor=args.execution_anchor,
            diagnosis_anchor=args.diagnosis_anchor,
        )
    else:
        import run_runtime_readback_smoke as OWNED  # pyright: ignore[reportMissingImports]
        execute(args.manifest, args.approval, args.out, runner=OWNED.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
