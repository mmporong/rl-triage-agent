"""부모 실행 anchor와 continuation 등록 anchor를 분리한 v3 어댑터."""
from __future__ import annotations

import hashlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_continuation_v3 as V3

V2 = V3.V2
V1 = V3.V1
SCHEMA = "lineage_recovery_continuation_protocol_v4"
MANIFEST_SCHEMA = "lineage_recovery_continuation_manifest_v4"
RESULT_SCHEMA = "lineage_recovery_continuation_result_v4"
SOURCE_FILES = (*V3.SOURCE_FILES,
                "evals/lineage_recovery_continuation_v4.py",
                "tests/test_lineage_recovery_continuation_v4.py")
_ORIGINAL = (V3.SCHEMA, V3.MANIFEST_SCHEMA, V3.RESULT_SCHEMA, V3.SOURCE_FILES,
             V3._validate_sources)
_LOCK = RLock()
_LOCAL = threading.local()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _anchor_blob(anchor: str, relative: str) -> str:
    raw = V3._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _validate_sources(protocol_path: Path, manifest_path: Path, approval_path: Path,
                      smoke_root: Path, failed_execution: Path, *, anchor: str,
                      lab: Path) -> tuple[dict, dict, dict, dict]:
    """부모는 기존 anchor에서, 새 wrapper는 등록 anchor에서 각각 검증한다."""
    protocol = V3.verify_protocol(protocol_path, anchor=anchor, lab=lab)
    with V2._v1_scope():
        manifest, parent = V1._load_manifest(manifest_path)
        parent_anchor = manifest["anchor"]
        _require(parent_anchor != anchor, "부모 실행 anchor와 continuation 등록 anchor가 같아서는 안 된다")
        V3._git("merge-base", "--is-ancestor", parent_anchor, anchor)
        _require(Path(manifest["lab"]).resolve() == Path(lab).resolve(),
                 "parent manifest lab이 continuation과 다르다")
        parent_component = protocol["parent_protocol"]
        parent_path = Path(manifest["protocol"]["path"]).resolve()
        registered_parent = (V3.ROOT / parent_component["path"]).resolve()
        _require(parent_path == registered_parent
                 and V3._sha(parent_path, lf=True) == parent_component["sha256_lf"]
                 and _anchor_blob(parent_anchor, parent_component["path"])
                 == parent_component["sha256_lf"]
                 and _anchor_blob(anchor, parent_component["path"])
                 == parent_component["sha256_lf"],
                 "부모 protocol path/SHA/blob이 두 anchor 사이에서 바뀌었다")
        smoke = V1.verify_smoke(smoke_root, manifest, parent)
        approval = V1._approval(approval_path, manifest["rows"])
        failure, _wait = V3._failure_boundary(Path(failed_execution).resolve())
        intent = V3._read(Path(failed_execution).resolve() / "intent.json")
        _require(intent.get("manifest") == V3._ref(manifest_path)
                 and intent.get("approval") == V3._ref(approval_path)
                 and intent.get("smoke") == smoke
                 and intent.get("automatic_retry") is False
                 and intent.get("model_calls") == 0,
                 "이전 execution intent의 manifest/approval/smoke 결합이 다르다")
        training = V3._completed_training(manifest, parent, Path(failed_execution).resolve())
    return manifest, approval, {"path": smoke["path"], "result_sha256": smoke["result_sha256"]}, {
        **training, "failure": failure,
    }


@contextmanager
def _v3_scope():
    """V3 구현을 v4 상수와 anchor 분리 검증기로만 감싸고 복원한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        expected = (SCHEMA, MANIFEST_SCHEMA, RESULT_SCHEMA, SOURCE_FILES, _validate_sources)
        if depth:
            _require((V3.SCHEMA, V3.MANIFEST_SCHEMA, V3.RESULT_SCHEMA, V3.SOURCE_FILES,
                      V3._validate_sources) == expected,
                     "continuation v4 재진입 중 V3 결합이 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require((V3.SCHEMA, V3.MANIFEST_SCHEMA, V3.RESULT_SCHEMA, V3.SOURCE_FILES,
                  V3._validate_sources) == _ORIGINAL,
                 "continuation V3가 등록 원본이 아니다")
        V3.SCHEMA, V3.MANIFEST_SCHEMA, V3.RESULT_SCHEMA = SCHEMA, MANIFEST_SCHEMA, RESULT_SCHEMA
        V3.SOURCE_FILES, V3._validate_sources = SOURCE_FILES, _validate_sources
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            (V3.SCHEMA, V3.MANIFEST_SCHEMA, V3.RESULT_SCHEMA, V3.SOURCE_FILES,
             V3._validate_sources) = _ORIGINAL


def prepare_protocol(parent_protocol: Path) -> dict:
    with _v3_scope():
        return V3.prepare_protocol(parent_protocol)


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict:
    with _v3_scope():
        return V3.verify_protocol(protocol_path, anchor=anchor, lab=lab)


def prepare_inputs(protocol_path: Path, manifest_path: Path, approval_path: Path,
                   smoke_root: Path, failed_execution: Path, out: Path, *, anchor: str,
                   lab: Path) -> dict:
    with _v3_scope():
        return V3.prepare_inputs(protocol_path, manifest_path, approval_path, smoke_root,
                                 failed_execution, out, anchor=anchor, lab=lab)


def execute(manifest_path: Path, out: Path, **kwargs) -> dict:
    with _v3_scope():
        return V3.execute(manifest_path, out, **kwargs)


def main(argv: list[str] | None = None) -> int:
    with _v3_scope():
        return V3.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
