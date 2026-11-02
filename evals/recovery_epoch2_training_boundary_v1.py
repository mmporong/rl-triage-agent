"""R4 epoch-2 loop와 guarded training protocol을 분리해 결합한다."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v2 as EXPORT
import five_policy_recovery_epoch2_v1 as SECOND
import recovery_epoch2_probe_boundary_v1 as PROBE_BOUNDARY

SCHEMA = "recovery_epoch2_training_boundary_protocol_v1"
SOURCE_FILES = (
    "evals/recovery_epoch2_training_boundary_v1.py",
    "tests/test_recovery_epoch2_training_boundary_v1.py",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LOCK = RLock()
_ORIGINAL_BUILD = SECOND.TRAINING.build_manifest
_ORIGINAL_EXPORT_TRAINING = EXPORT._training


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object가 아니다")
    return cast(dict[str, Any], value)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("training component는 repo 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    _require(result.returncode == 0, "Git training component 결합을 확인하지 못했다")
    return result.stdout


def _ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    return {"path": _relative(target), "sha256_lf": _sha(target, lf=True)}


def _checked_loop(path: Path, anchor: str, *, lab: Path | None = None) -> dict[str, Any]:
    """전역 scope와 무관하게 typed loop와 그 원 parent 검증을 모두 실행한다."""
    return PROBE_BOUNDARY.verify_loop_protocol(Path(path), anchor, lab=lab)


def prepare_protocol(loop_protocol: Path, loop_anchor: str) -> dict[str, Any]:
    """등록 loop와 첫 교정 component에서 guarded/runtime closure를 파생한다."""
    loop_path = Path(loop_protocol).resolve(strict=True)
    loop = _checked_loop(loop_path, loop_anchor)
    first_ref = cast(dict[str, str], loop["first_correction_component"])
    first_path = ROOT / first_ref["path"]
    _require(_ref(first_path) == first_ref, "첫 교정 component ref가 바뀌었다")
    first = _read(first_path)
    sources = cast(dict[str, str], loop["sources_lf"])
    guarded = set(SECOND.TRAINING.TRAIN.BOUND_SOURCES)
    _require(guarded <= set(sources), "loop에 guarded training closure가 없다")
    runtime = first.get("runtime_source_sha256_lf")
    _require(isinstance(runtime, dict) and len(runtime) == 9, "첫 component runtime9가 없다")
    return {
        "schema": SCHEMA,
        "status": "registered",
        "parent_loop_protocol": {**_ref(loop_path), "anchor": loop_anchor},
        "first_correction_component": deepcopy(first_ref),
        "source_sha256_lf": {
            **{name: sources[name] for name in sorted(guarded)},
            **{name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        },
        "runtime_source_sha256_lf": deepcopy(runtime),
        "training": {"arm": "BOTH", "max_iterations": 300, "fixed_conditions": 26},
    }


def verify_protocol(path: Path, anchor: str, *, loop_protocol: Path,
                    loop_anchor: str, lab: Path | None = None) -> dict[str, Any]:
    """새 component, parent loop, current/blob source를 함께 검증한다."""
    target = Path(path).resolve(strict=True)
    _require(_HEX40.fullmatch(anchor) is not None, "training anchor 형식")
    value = _read(target)
    expected = prepare_protocol(loop_protocol, loop_anchor)
    _require(value == expected, "training component가 loop/first component에서 파생되지 않았다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bindings = {_relative(target): _sha(target, lf=True), **value["source_sha256_lf"]}
    for name, digest in bindings.items():
        _require(isinstance(digest, str) and _HEX64.fullmatch(digest) is not None,
                 f"training source SHA 형식: {name}")
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == digest,
                 f"현재 training source가 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"training anchor blob이 다르다: {name}")
    if lab is not None:
        _checked_loop(Path(loop_protocol), loop_anchor, lab=Path(lab).resolve())
    return deepcopy(value)


def verify_receipt_binding(receipt: Mapping[str, Any], candidate: Mapping[str, Any], *,
                           loop_protocol: Path, training_protocol: Path,
                           training_anchor: str, lab: Path) -> dict[str, Any]:
    """candidate→loop receipt→training manifest의 두 protocol 연결을 재검증한다."""
    loop_path = Path(loop_protocol).resolve(strict=True)
    component_path = Path(training_protocol).resolve(strict=True)
    inputs = candidate.get("inputs")
    _require(isinstance(inputs, Mapping), "second candidate inputs가 없다")
    inputs = cast(Mapping[str, Any], inputs)
    _require(inputs.get("loop_protocol") == SECOND.ref(loop_path), "candidate loop ref가 다르다")
    _require(receipt.get("protocol") == SECOND.ref(loop_path), "receipt loop ref가 다르다")
    training = receipt.get("training")
    _require(isinstance(training, Mapping), "receipt training이 없다")
    training = cast(Mapping[str, Any], training)
    manifest_path = SECOND.bound(training.get("manifest"), "training manifest")
    manifest = SECOND.TRAINING.TRAIN.validate_manifest(_read(manifest_path))
    _require(manifest["protocol_path"] == _relative(component_path)
             and manifest["protocol_sha256_lf"] == _sha(component_path, lf=True)
             and manifest["source_commit_sha"] == training_anchor,
             "training manifest component 결합이 다르다")
    SECOND.TRAINING.TRAIN.verify_binding(manifest, lab=Path(lab).resolve())
    return deepcopy(manifest)


@contextmanager
def training_boundary_scope(*, loop_protocol: Path, loop_anchor: str,
                            training_protocol: Path, training_anchor: str,
                            lab: Path) -> Iterator[dict[str, Any]]:
    """원 build/export 검증을 등록 training component로만 좁게 전환한다."""
    with _LOCK:
        loop_path = Path(loop_protocol).resolve(strict=True)
        component_path = Path(training_protocol).resolve(strict=True)
        component = verify_protocol(component_path, training_anchor,
                                    loop_protocol=loop_path, loop_anchor=loop_anchor, lab=lab)
        original_build = SECOND.TRAINING.build_manifest
        original_export = EXPORT._training
        _require(original_build is _ORIGINAL_BUILD and original_export is _ORIGINAL_EXPORT_TRAINING,
                 "training boundary 대상이 이미 바뀌었다")

        def build(protocol_path: Path, protocol: Mapping[str, Any], anchor: str, **kwargs: Any) -> dict[str, Any]:
            _require(Path(protocol_path).resolve() == loop_path and anchor == loop_anchor,
                     "등록 loop가 아닌 training 요청")
            _require(dict(protocol) == _checked_loop(loop_path, loop_anchor, lab=lab),
                     "training 호출의 loop 값이 다르다")
            return original_build(component_path, component, training_anchor, **kwargs)

        def export_training(receipt: Mapping[str, Any], candidate: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
            result = original_export(receipt, candidate, **kwargs)
            second = kwargs.get("second")
            _require(type(second) is bool, "export training의 second 구분이 없다")
            if second:
                verify_receipt_binding(receipt, candidate, loop_protocol=loop_path,
                                       training_protocol=component_path,
                                       training_anchor=training_anchor, lab=lab)
            return result

        SECOND.TRAINING.__dict__["build_manifest"] = build
        EXPORT.__dict__["_training"] = export_training
        try:
            yield deepcopy(component)
        finally:
            current = (SECOND.TRAINING.build_manifest, EXPORT._training)
            SECOND.TRAINING.__dict__["build_manifest"] = original_build
            EXPORT.__dict__["_training"] = original_export
            if current != (build, export_training):
                raise RuntimeError("training boundary가 scope 중 외부에서 바뀌었다")
